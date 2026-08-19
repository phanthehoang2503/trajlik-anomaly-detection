import argparse
import importlib.metadata
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dcte import DCTE, MSMLoss
from ectf import EndpointConditionedTrajectoryFlow
from trajlik.model import TrajLikHead
from trajlik.normal_tail import EmpiricalTailCalibrator
from trajlik.trajectory_cache_dataset import (
    TrajectoryCacheDataset,
    balanced_category_sampler,
    stratified_normal_split,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DCTE + ECTF on a normal-only trajectory cache",
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--calibration_fraction", type=float, default=0.05)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--trajectory_dim", type=int, default=64)
    parser.add_argument("--dcte_layers", type=int, default=2)
    parser.add_argument("--dcte_heads", type=int, default=4)
    parser.add_argument("--flow_blocks", type=int, default=4)
    parser.add_argument("--flow_bins", type=int, default=8)
    parser.add_argument("--lambda_msm", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_head(input_dim, args):
    dcte = DCTE(
        input_dim=input_dim,
        projection_dim=args.projection_dim,
        token_dim=args.token_dim,
        trajectory_dim=args.trajectory_dim,
        num_steps=3,
        num_heads=args.dcte_heads,
        num_layers=args.dcte_layers,
        feedforward_dim=4 * args.token_dim,
        dropout=0.0,
    )
    ectf = EndpointConditionedTrajectoryFlow(
        trajectory_dim=args.trajectory_dim,
        z0_dim=input_dim,
        condition_dim=64,
        global_dim=32,
        position_dim=16,
        conditioner_hidden_dim=128,
        coupling_hidden_dim=128,
        num_blocks=args.flow_blocks,
        num_bins=args.flow_bins,
    )
    msm_loss = MSMLoss(
        input_dim=input_dim,
        projection_dim=args.projection_dim,
        token_dim=args.token_dim,
        trajectory_dim=args.trajectory_dim,
    )
    return TrajLikHead(
        dcte=dcte,
        ectf=ectf,
        msm_loss=msm_loss,
        lambda_msm=args.lambda_msm,
    )


def package_versions():
    names = ("torch", "torchvision", "numpy", "scipy", "scikit-learn")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def split_normal_indices(
    categories,
    *,
    validation_fraction,
    calibration_fraction,
    seed,
):
    """Keep calibration untouched while splitting the head pool for validation."""

    if validation_fraction <= 0.0 or calibration_fraction <= 0.0:
        raise ValueError("validation and calibration fractions must be positive")
    if validation_fraction + calibration_fraction >= 1.0:
        raise ValueError("validation and calibration fractions must sum to less than one")

    head_indices, calibration_indices = stratified_normal_split(
        categories,
        calibration_fraction=calibration_fraction,
        seed=seed,
    )
    head_categories = [categories[index] for index in head_indices.tolist()]
    relative_validation_fraction = validation_fraction / (
        1.0 - calibration_fraction
    )
    relative_training_indices, relative_validation_indices = (
        stratified_normal_split(
            head_categories,
            calibration_fraction=relative_validation_fraction,
            seed=seed + 1,
        )
    )
    training_indices = head_indices[relative_training_indices]
    validation_indices = head_indices[relative_validation_indices]
    return training_indices, validation_indices, calibration_indices


def output_metrics(output):
    base_nll = 0.5 * (
        output["base_latents"].square() + math.log(2.0 * math.pi)
    ).sum(dim=-1).mean()
    return {
        "loss": float(output["loss"].detach().item()),
        "nll_loss": float(output["nll_loss"].detach().item()),
        "msm_loss": float(output["msm_loss"].detach().item()),
        "base_nll": float(base_nll.detach().item()),
        "log_det": float(output["log_det"].detach().mean().item()),
    }


def average_metrics(totals, total_weight):
    return {
        key: value / max(total_weight, 1)
        for key, value in totals.items()
    }


@torch.no_grad()
def evaluate_head(head, loader, device, *, seed):
    """Evaluate full-trajectory NLL and repeatable masked-step MSM."""

    head.eval()
    totals = {
        "loss": 0.0,
        "nll_loss": 0.0,
        "msm_loss": 0.0,
        "base_nll": 0.0,
        "log_det": 0.0,
    }
    total_weight = 0
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            full_output = head(batch, mask=False)
            masked_output = head.training_loss(batch)

            nll_loss = full_output["path_nll"].mean()
            msm_loss = masked_output["msm_loss"]
            base_nll = 0.5 * (
                full_output["base_latents"].square()
                + math.log(2.0 * math.pi)
            ).sum(dim=-1).mean()
            metrics = {
                "loss": float(
                    (nll_loss + head.lambda_msm * msm_loss).detach().item()
                ),
                "nll_loss": float(nll_loss.detach().item()),
                "msm_loss": float(msm_loss.detach().item()),
                "base_nll": float(base_nll.detach().item()),
                "log_det": float(full_output["log_det"].mean().detach().item()),
            }
            weight = full_output["path_nll"].numel()
            for key, value in metrics.items():
                totals[key] += value * weight
            total_weight += weight
    return average_metrics(totals, total_weight)


def checkpoint_path(output_path: Path, label: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{label}{output_path.suffix}")


@torch.no_grad()
def fit_calibrator(head, loader, device):
    head.eval()
    endpoint_scores = []
    path_scores = []
    progress = tqdm(
        loader,
        desc="Fitting calibration",
        unit="batch",
        leave=False,
        disable=not logger.isEnabledFor(logging.INFO),
    )
    for batch in progress:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = head(batch, mask=False)
        endpoint_scores.append(output["a_end_coarse"].cpu())
        path_scores.append(output["path_nll"].cpu())
    return EmpiricalTailCalibrator().fit(
        torch.cat(endpoint_scores, dim=0),
        torch.cat(path_scores, dim=0),
    )


def load_trajlik_checkpoint(checkpoint_path, device="cpu"):
    """Rebuild a head and calibrator from a self-describing checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("checkpoint_type") == "trajlik_training":
        raise ValueError(
            "Training snapshots do not contain normal calibration; "
            "load the final output or *_best.pth checkpoint instead"
        )
    args = SimpleNamespace(**checkpoint["training_args"])
    input_dim = int(checkpoint["cache_metadata"]["output_channels"])
    head = build_head(input_dim, args).to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    calibrator = EmpiricalTailCalibrator.from_state_dict(
        checkpoint["calibrator_state_dict"]
    ).to(device)
    return head, calibrator, checkpoint


def train(args):
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.grad_clip <= 0
        or args.patience <= 0
        or args.min_delta < 0
    ):
        raise ValueError(
            "epochs, batch_size, grad_clip, and patience must be positive; "
            "min_delta must be non-negative"
        )
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = TrajectoryCacheDataset(args.cache_dir)
    if int(dataset.metadata["num_steps"]) != 3:
        raise ValueError("Main TrajLik protocol requires exactly three inversion steps")
    if dataset.metadata.get("projection") != "none":
        raise ValueError(
            "Online-equivalent head training currently requires projection=none. "
            "Projected caches need their projector reapplied during online "
            "inference and must not be used silently."
        )

    training_indices, validation_indices, calibration_indices = (
        split_normal_indices(
            dataset.categories,
            validation_fraction=args.validation_fraction,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
        )
    )
    training_subset = Subset(dataset, training_indices.tolist())
    validation_subset = Subset(dataset, validation_indices.tolist())
    calibration_subset = Subset(dataset, calibration_indices.tolist())
    sampler = balanced_category_sampler(
        dataset.categories,
        training_indices.tolist(),
        seed=args.seed,
    )
    training_loader = DataLoader(
        training_subset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    calibration_loader = DataLoader(
        calibration_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    input_dim = int(dataset.metadata["output_channels"])
    head = build_head(input_dim, args).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in head.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_path(output_path, "best")
    latest_path = checkpoint_path(output_path, "latest")

    trainable_parameters = sum(
        parameter.numel() for parameter in head.parameters() if parameter.requires_grad
    )
    logger.info(
        "Training TrajLik | device=%s | cache=%s | train=%d | validation=%d "
        "| calibration=%d "
        "| categories=%d | channels=%d | lambda_msm=%.4g "
        "| trainable_params=%d",
        device,
        args.cache_dir,
        len(training_subset),
        len(validation_subset),
        len(calibration_subset),
        len(set(dataset.categories)),
        input_dim,
        args.lambda_msm,
        trainable_parameters,
    )

    training_history = []
    best_validation_nll = float("inf")
    best_epoch = 0
    best_head_state = None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        head.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        epoch_totals = {
            "loss": 0.0,
            "nll_loss": 0.0,
            "msm_loss": 0.0,
            "base_nll": 0.0,
            "log_det": 0.0,
        }
        total_weight = 0
        progress = tqdm(
            training_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
            disable=not logger.isEnabledFor(logging.INFO),
        )
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = head.training_loss(batch)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            batch_metrics = output_metrics(output)
            weight = output["path_nll"].numel()
            for key, value in batch_metrics.items():
                epoch_totals[key] += value * weight
            total_weight += weight
            progress.set_postfix(
                loss=f"{batch_metrics['loss']:.4f}",
                nll=f"{batch_metrics['nll_loss']:.4f}",
                msm=f"{batch_metrics['msm_loss']:.4f}",
            )

        epoch_seconds = time.perf_counter() - epoch_started
        train_metrics = average_metrics(epoch_totals, total_weight)
        validation_metrics = evaluate_head(
            head,
            validation_loader,
            device,
            seed=args.seed,
        )
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        validation_nll = validation_metrics["nll_loss"]
        improved = validation_nll < best_validation_nll - args.min_delta
        if improved:
            best_validation_nll = validation_nll
            best_epoch = epoch + 1
            best_head_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_nll": train_metrics["nll_loss"],
            "train_nll_per_dim": train_metrics["nll_loss"] / args.trajectory_dim,
            "train_msm": train_metrics["msm_loss"],
            "train_base_nll": train_metrics["base_nll"],
            "train_log_det": train_metrics["log_det"],
            "validation_loss": validation_metrics["loss"],
            "validation_nll": validation_nll,
            "validation_nll_per_dim": validation_nll / args.trajectory_dim,
            "validation_msm": validation_metrics["msm_loss"],
            "validation_base_nll": validation_metrics["base_nll"],
            "validation_log_det": validation_metrics["log_det"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": epoch_seconds,
            "peak_memory_mb": peak_memory_mb,
            "is_best": improved,
        }
        training_history.append(epoch_record)

        training_snapshot = {
            "checkpoint_type": "trajlik_training",
            "epoch": epoch + 1,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_args": vars(args),
            "cache_metadata": dataset.metadata,
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "calibration_indices": calibration_indices,
            "training_history": training_history,
            "best_epoch": best_epoch,
            "best_validation_nll": best_validation_nll,
        }
        torch.save(training_snapshot, latest_path)
        if improved:
            torch.save(training_snapshot, best_path)

        logger.info(
            "Epoch %d/%d | train_nll=%.6f (%.6f/dim) | train_msm=%.6f "
            "| val_nll=%.6f (%.6f/dim) | val_msm=%.6f "
            "| val_base_nll=%.6f | val_logdet=%.6f | lr=%.2e "
            "| time=%.1fs | peak_mem=%.1f MB%s",
            epoch + 1,
            args.epochs,
            train_metrics["nll_loss"],
            train_metrics["nll_loss"] / args.trajectory_dim,
            train_metrics["msm_loss"],
            validation_nll,
            validation_nll / args.trajectory_dim,
            validation_metrics["msm_loss"],
            validation_metrics["base_nll"],
            validation_metrics["log_det"],
            optimizer.param_groups[0]["lr"],
            epoch_seconds,
            peak_memory_mb,
            " | best" if improved else "",
        )

        if epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping at epoch %d; best normal validation NLL "
                "%.6f was at epoch %d",
                epoch + 1,
                best_validation_nll,
                best_epoch,
            )
            break

    if best_head_state is None:
        raise RuntimeError("Training finished without a valid best checkpoint")
    head.load_state_dict(best_head_state, strict=True)
    calibrator = fit_calibrator(head, calibration_loader, device)
    logger.info(
        "Fitted normal calibration on %d held-out images",
        len(calibration_subset),
    )
    checkpoint = {
        "checkpoint_type": "trajlik_final",
        "head_state_dict": head.state_dict(),
        "calibrator_state_dict": calibrator.state_dict(),
        "training_args": vars(args),
        "cache_metadata": dataset.metadata,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "calibration_indices": calibration_indices,
        "training_history": training_history,
        "best_epoch": best_epoch,
        "best_validation_nll": best_validation_nll,
        "package_versions": package_versions(),
    }
    torch.save(checkpoint, output_path)
    torch.save(checkpoint, best_path)
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training_args": vars(args),
                "cache_metadata": dataset.metadata,
                "package_versions": checkpoint["package_versions"],
                "num_training_images": training_indices.numel(),
                "num_validation_images": validation_indices.numel(),
                "num_calibration_images": calibration_indices.numel(),
                "best_epoch": best_epoch,
                "best_validation_nll": best_validation_nll,
                "best_checkpoint": str(best_path),
                "latest_checkpoint": str(latest_path),
                "training_history": training_history,
            },
            file,
            indent=2,
        )
    logger.info(
        "Saved best epoch %d with normal calibration to %s",
        best_epoch,
        output_path,
    )
    return checkpoint


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    train(parse_args())


if __name__ == "__main__":
    main()
