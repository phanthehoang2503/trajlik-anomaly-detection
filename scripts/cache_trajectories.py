import argparse
import copy
import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
baseline_root = project_root / "baseline" / "InversionAD"
for path in (project_root, baseline_root):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from src.backbones import get_backbone, get_backbone_feature_shape
from src.datasets import build_dataset
from src.denoiser import get_denoiser
from trajlik.cache_identity import checkpoint_identity
from trajlik.cache_layout import sanitize_category
from trajlik.trajectory_projector import TrajectoryProjector

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--save_dir",
        help="Directory containing model_latest.pth or model_ema_latest.pth",
    )
    checkpoint_group.add_argument(
        "--checkpoint_path",
        help="Explicit checkpoint file, for example an official model_best.pth",
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--num_inversion_steps", type=int, default=3)
    parser.add_argument("--proj_dim", type=int, default=68)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Cache only the first N images",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_ema_model", action="store_true")
    parser.add_argument(
        "--autocast_dtype",
        choices=["auto", "float16", "bfloat16", "none"],
        default="auto",
        help="CUDA autocast type; auto uses FP16 or BF16 depend on the GPU",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--projection",
        choices=["none", "linear"],
        default="none",
        help="none: keep original channel; linear: project down to proj_dim",
    )
    parser.add_argument(
        "--storage_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Storage dtype for cached .pt tensors",
    )
    return parser.parse_args()


def prepare_cache_dir(cache_dir: Path, force: bool):
    if cache_dir.exists():
        if not force:
            raise FileExistsError(
                f"Cache directory already exists: {cache_dir}. "
                "Use --force to overwrite."
            )
        shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True)


def resolve_checkpoint_path(
    save_dir,
    use_ema_model: bool,
    checkpoint_path,
) -> Path:
    if checkpoint_path is not None:
        if use_ema_model:
            raise ValueError(
                "--use_ema_model cannot be combined with --checkpoint_path; "
                "the explicit path already selects the checkpoint"
            )
        return Path(checkpoint_path)

    checkpoint_name = (
        "model_ema_latest.pth" if use_ema_model else "model_latest.pth"
    )
    return Path(save_dir) / checkpoint_name


def load_checkpoint(
    model,
    save_dir,
    use_ema_model: bool,
    checkpoint_path=None,
):
    checkpoint_path = resolve_checkpoint_path(
        save_dir,
        use_ema_model,
        checkpoint_path,
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    return checkpoint_path


def resolve_autocast(device: torch.device, name: str):
    if device.type != "cuda" or name == "none":
        return False, torch.float32, "none"

    if name == "auto":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[name]

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "This CUDA device does not support bfloat16; use "
            "--autocast_dtype float16 (required for NVIDIA T4)"
        )

    dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
    return True, dtype, dtype_name


@torch.no_grad()
def cache_trajectories(config: dict, args):
    device_name = args.device or config["meta"]["device"]
    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    if args.num_inversion_steps <= 0:
        raise ValueError("--num_inversion_steps must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max_images must be positive")
    if args.projection == "linear" and args.proj_dim <= 0:
        raise ValueError("--proj_dim must be positive")

    dataset_config = copy.deepcopy(config["data"])
    dataset_config.update(
        train=True,
        normal_only=True,
        anom_only=False,
    )

    dataset = build_dataset(**dataset_config)
    if args.max_images is not None:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("The selected training dataset is empty")

    batch_size = args.batch_size or dataset_config["batch_size"]
    num_workers = (
        args.num_workers if args.num_workers is not None else dataset_config["num_workers"]
    )
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if num_workers < 0:
        raise ValueError("--num_workers cannot be negative")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=dataset_config.get("pin_memory", True),
    )

    diffusion_config = copy.deepcopy(config["diffusion"])
    diffusion_config["num_sampling_steps"] = str(args.num_inversion_steps)

    feature_shape = get_backbone_feature_shape(
        model_type=config["backbone"]["model_type"]
    )
    in_channels = feature_shape[0]

    denoiser = get_denoiser(
        **diffusion_config,
        input_shape=feature_shape,
    ).to(device).eval()

    feature_extractor = get_backbone(**config["backbone"]).to(device).eval()

    checkpoint_path = load_checkpoint(
        denoiser,
        args.save_dir,
        args.use_ema_model,
        args.checkpoint_path,
    )
    invad_checkpoint_identity = checkpoint_identity(checkpoint_path)
    logger.info(
        "Cache bound to InvAD checkpoint SHA-256: %s",
        invad_checkpoint_identity["sha256"],
    )

    autocast_enabled, autocast_dtype, autocast_dtype_name = resolve_autocast(
        device,
        args.autocast_dtype,
    )
    logger.info(
        "Autocast: %s",
        autocast_dtype_name if autocast_enabled else "disabled",
    )

    storage_dtype = resolve_storage_dtype(args.storage_dtype)

    projector = TrajectoryProjector(
        in_channels=in_channels,
        proj_dim=args.proj_dim,
        projection=args.projection,
        storage_dtype=storage_dtype,
    ).to(device).eval()

    # Do not remove an existing cache until the dataset, models, and checkpoint
    # have all been validated successfully.
    cache_dir = Path(args.cache_dir)
    prepare_cache_dir(cache_dir, args.force)

    num_cached = 0
    output_paths = set()
    cache_index = []

    for batch in tqdm(loader, desc="Caching trajectories"):
        images = batch["samples"].to(device, non_blocking=True)
        labels = batch["clslabels"].to(device, non_blocking=True)

        z_0, _ = feature_extractor(images)

        start_t = torch.zeros(
            z_0.shape[0],
            dtype=torch.long,
            device=device,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            _, z_seq, eps_seq, delta_z_seq = (
                denoiser.ddim_reverse_sample(
                    z_0,
                    start_t,
                    labels,
                    eta=0.0,
                    return_intermediates=True,
                )
            )

            projected = projector.project_and_compress(
                z_0,
                z_seq,
                eps_seq,
                delta_z_seq,
            )
            projected["a_end_coarse"] = torch.linalg.vector_norm(
                z_seq[-1].float(),
                ord=2,
                dim=1,
            ).to(storage_dtype)

        for index, source_path in enumerate(batch["filenames"]):
            category = batch["clsnames"][index]
            filename = sanitize_filename(source_path, category)
            category_dir = cache_dir / sanitize_category(category)
            category_dir.mkdir(parents=True, exist_ok=True)
            output_path = category_dir / f"{filename}.pt"

            if output_path in output_paths:
                raise RuntimeError(
                    f"Duplicate cache filename for source: {source_path}"
                )
            output_paths.add(output_path)

            output = {
                key: value[index].detach().cpu()
                for key, value in projected.items()
            }

            output["source_path"] = source_path
            output["category"] = category
            output["split"] = "train"
            output["is_normal"] = True

            torch.save(output, output_path)
            cache_index.append(
                {
                    "file": output_path.relative_to(cache_dir).as_posix(),
                    "source_path": source_path,
                    "category": category,
                    "split": "train",
                    "is_normal": True,
                }
            )
            num_cached += 1

    if args.projection == "linear":
        torch.save(
            projector.state_dict(),
            cache_dir / "projector.pt",
        )

    effective_channels = (
        args.proj_dim
        if args.projection == "linear"
        else in_channels
    )

    metadata = {
        "num_images": num_cached,
        "num_steps": args.num_inversion_steps,
        "in_channels": in_channels,
        "output_channels": effective_channels,
        "projection": args.projection,
        "proj_dim": (
            args.proj_dim
            if args.projection == "linear"
            else None
        ),
        "storage_dtype": args.storage_dtype,
        "autocast_dtype": autocast_dtype_name,
        "checkpoint_path": str(checkpoint_path),
        "invad_checkpoint": invad_checkpoint_identity,
        "max_images": args.max_images,
        "timestep_map": [
            int(timestep)
            for timestep in denoiser.sample_diffusion.timesteps_map
        ],
        "backbone": config["backbone"]["model_type"],
        "dataset": dataset_config["dataset_name"],
        "category": dataset_config.get("category"),
        "normal_only": True,
        "img_size": dataset_config["img_size"],
        "transform_type": dataset_config["transform_type"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }

    with open(cache_dir / "cache_meta.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    with open(cache_dir / "cache_index.json", "w", encoding="utf-8") as file:
        json.dump(cache_index, file, indent=2)

    logger.info("Cached %d images into %s", num_cached, cache_dir)


def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    cache_trajectories(config, args)


def sanitize_filename(path: str, category: str) -> str:
    path = Path(path)

    # Ex:
    # .../bottle/train/good/000.png
    # -> bottle_good_000
    stem = f"{category}_{path.parent.name}_{path.stem}"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)


def resolve_storage_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
    }[name]


if __name__ == "__main__":
    main()
