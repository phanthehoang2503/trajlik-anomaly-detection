import argparse
import copy
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

project_root = Path(__file__).resolve().parents[1]
baseline_root = project_root / "baseline" / "InversionAD"
for path in (project_root, baseline_root):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.adeval.eval_utils import calculate_img_metrics, calculate_px_metrics
from src.backbones import get_backbone, get_backbone_feature_shape
from src.datasets import build_dataset
from src.denoiser import get_denoiser
from scripts.cache_trajectories import load_checkpoint
from scripts.train_trajlik import load_trajlik_checkpoint
from trajlik.cache_identity import (
    checkpoint_identity,
    checkpoint_identity_errors,
    normalized_timestep_map,
)
from trajlik.inversion_ad_module import InversionADModule
from trajlik.model import TrajLikAD
from trajlik.reproducibility import compare_checkpoint_config


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TrajLik-AD without fitting on test anomalies",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--invad_checkpoint", required=True)
    parser.add_argument("--trajlik_checkpoint", required=True)
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def config_fingerprint(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_runtime_contract(
    config,
    invad_checkpoint,
    head_checkpoint,
    feature_channels,
    runtime_timestep_map=None,
):
    errors = []
    invad_checkpoint = Path(invad_checkpoint)
    companion_config = invad_checkpoint.parent / "config.yaml"
    if not companion_config.is_file():
        errors.append(f"InvAD checkpoint companion config is missing: {companion_config}")
    else:
        with companion_config.open(encoding="utf-8") as file:
            checkpoint_config = yaml.safe_load(file)
        errors.extend(
            "InvAD checkpoint mismatch " + mismatch
            for mismatch in compare_checkpoint_config(config, checkpoint_config)
        )

    metadata = head_checkpoint.get("cache_metadata", {})
    recorded_identity = metadata.get("invad_checkpoint")
    if recorded_identity is None:
        errors.append("TrajLik cache metadata is missing invad_checkpoint SHA-256")
    else:
        identity_errors = checkpoint_identity_errors(recorded_identity)
        errors.extend(
            "TrajLik cache " + identity_error
            for identity_error in identity_errors
        )
        if not identity_errors:
            try:
                runtime_identity = checkpoint_identity(invad_checkpoint)
            except OSError as error:
                errors.append(f"Cannot hash InvAD checkpoint: {error}")
            else:
                if recorded_identity["sha256"].lower() != runtime_identity["sha256"]:
                    errors.append(
                        "InvAD checkpoint SHA-256 does not match the checkpoint "
                        "used to create the TrajLik cache"
                    )
                if (
                    int(recorded_identity["size_bytes"])
                    != runtime_identity["size_bytes"]
                ):
                    errors.append(
                        "InvAD checkpoint size does not match the checkpoint used "
                        "to create the TrajLik cache"
                    )

    recorded_timestep_map = metadata.get("timestep_map")
    if recorded_timestep_map is None:
        errors.append("TrajLik cache metadata is missing timestep_map")
    elif runtime_timestep_map is None:
        errors.append("Evaluator did not provide a runtime timestep_map")
    else:
        try:
            recorded_timestep_map = normalized_timestep_map(recorded_timestep_map)
            runtime_timestep_map = normalized_timestep_map(runtime_timestep_map)
        except (TypeError, ValueError) as error:
            errors.append(f"Invalid timestep_map identity: {error}")
        else:
            if recorded_timestep_map != runtime_timestep_map:
                errors.append(
                    "InvAD timestep_map does not match the schedule used to create "
                    "the TrajLik cache"
                )

    expected_fingerprint = metadata.get("config_sha256")
    if expected_fingerprint and expected_fingerprint != config_fingerprint(config):
        logger.warning(
            "TrajLik head cache fingerprint differs from the evaluation config; "
            "continuing because runtime-critical fields are validated separately"
        )
    if metadata.get("normal_only") is not True:
        errors.append("TrajLik head was not trained from a declared normal-only cache")
    if int(metadata.get("num_steps", -1)) != 3:
        errors.append("TrajLik main protocol requires a three-step head cache")
    if metadata.get("projection") != "none":
        errors.append(
            "Online evaluation currently requires projection=none; applying a "
            "projected-cache head to raw Module 0 tensors would be invalid"
        )
    if int(metadata.get("output_channels", -1)) != feature_channels:
        errors.append(
            "TrajLik head input channels do not match the InvAD backbone output"
        )
    if metadata.get("backbone") != config["backbone"]["model_type"]:
        errors.append("TrajLik head backbone metadata does not match the config")
    if metadata.get("dataset") not in (None, config["data"]["dataset_name"]):
        errors.append("TrajLik head dataset metadata does not match the config")
    if metadata.get("transform_type") != config["data"]["transform_type"]:
        errors.append("TrajLik head preprocessing metadata does not match the config")
    if int(metadata.get("img_size", -1)) != int(config["data"]["img_size"]):
        errors.append("TrajLik head image-size metadata does not match the config")

    if errors:
        raise ValueError("Invalid TrajLik runtime contract:\n- " + "\n- ".join(errors))


def build_test_datasets(config, category=None):
    dataset_config = copy.deepcopy(config["data"])
    dataset_config.update(train=False, normal_only=False, anom_only=False)
    if category is not None:
        dataset_config["category"] = category
        dataset_config["dataset_name"] = dataset_config["dataset_name"].removesuffix(
            "_all"
        )
    dataset = build_dataset(**dataset_config)
    return dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]


@torch.no_grad()
def evaluate_loader(model, loader, device):
    labels = []
    masks = []
    image_scores = []
    pixel_maps = []
    total_seconds = 0.0
    total_images = 0

    for batch in loader:
        images = batch["samples"].to(device, non_blocking=True)
        class_labels = batch["clslabels"].to(device, non_blocking=True)
        if images.is_cuda:
            torch.cuda.synchronize(images.device)
        start = time.perf_counter()
        output = model(images, class_labels)
        if images.is_cuda:
            torch.cuda.synchronize(images.device)
        total_seconds += time.perf_counter() - start
        total_images += images.shape[0]

        labels.append(batch["labels"].cpu())
        masks.append(batch["masks"].cpu())
        image_scores.append(output["image_score"].cpu())
        pixel_maps.append(output["pixel_map"].cpu())

    if total_images == 0:
        raise ValueError("Test loader is empty")
    return {
        "labels": torch.cat(labels).numpy(),
        "masks": torch.cat(masks).squeeze(1).numpy(),
        "image_scores": torch.cat(image_scores).numpy(),
        "pixel_maps": torch.cat(pixel_maps).numpy(),
        "latency_ms": 1000.0 * total_seconds / total_images,
    }


def calculate_trajlik_metrics(predictions, device="cpu"):
    labels = np.asarray(predictions["labels"])
    masks = (np.asarray(predictions["masks"]) > 0).astype(np.uint8)
    image_scores = np.asarray(predictions["image_scores"])
    pixel_maps = np.asarray(predictions["pixel_maps"])

    image_metrics = calculate_img_metrics(
        gt_labels=labels,
        pred_scores=image_scores,
        metrics=["img_auroc", "img_aupr", "img_f1max", "img_ap"],
        device=device,
    )
    pixel_metrics = calculate_px_metrics(
        gt_masks=masks,
        pred_scores=pixel_maps,
        metrics=["px_auroc", "px_aupr", "px_f1max", "px_ap", "px_aupro"],
        device=device,
    )
    metrics = {
        "I-AUROC": image_metrics["img_auroc"],
        "I-AP": image_metrics["img_ap"],
        "I-F1Max": image_metrics["img_f1max"],
        "P-AUROC": pixel_metrics["px_auroc"],
        "P-AP": pixel_metrics["px_ap"],
        "P-F1Max": pixel_metrics["px_f1max"],
        "PRO": pixel_metrics["px_aupro"],
        "latency_ms": predictions["latency_ms"],
        "NFE": 3,
    }
    metrics["mAD"] = float(
        np.mean(
            [
                metrics["I-AUROC"],
                metrics["I-AP"],
                metrics["I-F1Max"],
                metrics["P-AUROC"],
                metrics["P-AP"],
                metrics["P-F1Max"],
                metrics["PRO"],
            ]
        )
    )
    return metrics


def evaluate(args):
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    device = torch.device(args.device)
    feature_shape = get_backbone_feature_shape(config["backbone"]["model_type"])

    head, calibrator, head_checkpoint = load_trajlik_checkpoint(
        args.trajlik_checkpoint,
        device=device,
    )

    diffusion_config = copy.deepcopy(config["diffusion"])
    diffusion_config["num_sampling_steps"] = "3"
    denoiser = get_denoiser(
        **diffusion_config,
        input_shape=feature_shape,
    ).to(device).eval()
    runtime_timestep_map = [
        int(timestep) for timestep in denoiser.sample_diffusion.timesteps_map
    ]
    validate_runtime_contract(
        config,
        args.invad_checkpoint,
        head_checkpoint,
        feature_shape[0],
        runtime_timestep_map,
    )
    load_checkpoint(
        denoiser,
        save_dir=None,
        use_ema_model=False,
        checkpoint_path=args.invad_checkpoint,
    )
    backbone = get_backbone(**config["backbone"]).to(device).eval()
    module0 = InversionADModule(backbone, denoiser).to(device).eval()
    model = TrajLikAD(module0, head, calibrator).to(device).eval()

    metrics_by_category = {}
    for dataset in build_test_datasets(config, args.category):
        category = getattr(dataset, "category", "unknown")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        metrics_by_category[category] = calculate_trajlik_metrics(
            evaluate_loader(model, loader, device),
            device=device,
        )
        logger.info("[%s] %s", category, metrics_by_category[category])

    if len(metrics_by_category) > 1:
        metric_names = next(iter(metrics_by_category.values())).keys()
        metrics_by_category["average"] = {
            name: float(
                np.mean(
                    [metrics[name] for metrics in metrics_by_category.values()]
                )
            )
            for name in metric_names
        }

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics_by_category, indent=2),
            encoding="utf-8",
        )
    return metrics_by_category


def main():
    logging.basicConfig(level=logging.INFO)
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
