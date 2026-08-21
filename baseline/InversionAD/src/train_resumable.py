import argparse
import copy
import os
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.backbones import get_backbone, get_backbone_feature_shape
from src.checkpointing import (
    atomic_torch_save,
    load_initial_weights,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.datasets import build_dataset
from src.denoiser import Denoiser, get_denoiser
from src.evaluate import evaluate_inv
from src.utils import get_lr_scheduler, get_optimizer

try:
    import wandb
except ImportError:
    wandb = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable InvAD training")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Full training_latest.pth checkpoint to resume",
    )
    checkpoint_group.add_argument(
        "--init_weights",
        type=str,
        default=None,
        help="Weights-only checkpoint used to start a fresh run",
    )
    return parser.parse_args()


def _init_wandb(config):
    if load_dotenv is not None:
        load_dotenv()
    if wandb is None or os.getenv("WANDB_API_KEY") is None:
        return False

    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")
    if project is None or entity is None:
        raise ValueError("WANDB_PROJECT and WANDB_ENTITY must be set when W&B is enabled")

    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project=project, entity=entity, config=config)
    return True


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config, *, resume=None, init_weights=None):
    pprint(config)
    use_wandb = _init_wandb(config)
    _seed_everything(config["meta"]["seed"])

    dataset_config = copy.deepcopy(config["data"])
    device = torch.device(config["meta"]["device"])
    batch_size = config["data"]["batch_size"]

    train_config = copy.deepcopy(config["data"])
    train_config.update(train=True, normal_only=True, anom_only=False)
    train_dataset = build_dataset(**train_config)

    dataset_config.update(train=False, anom_only=True)
    anom_dataset = build_dataset(**dataset_config)
    dataset_config.update(anom_only=False, normal_only=True)
    normal_dataset = build_dataset(**dataset_config)

    anom_loader = [
        DataLoader(anom_dataset, batch_size=1, shuffle=False, num_workers=1)
    ]
    normal_loader = [
        DataLoader(normal_dataset, batch_size=1, shuffle=False, num_workers=1)
    ]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=config["data"]["pin_memory"],
        num_workers=config["data"]["num_workers"],
        drop_last=True,
    )

    diff_in_sh = get_backbone_feature_shape(
        model_type=config["backbone"]["model_type"]
    )
    model: Denoiser = get_denoiser(**config["diffusion"], input_shape=diff_in_sh)
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)
    for parameter in model_ema.parameters():
        parameter.requires_grad = False

    if init_weights is not None:
        load_initial_weights(init_weights, model=model, model_ema=model_ema)
        print(f"Initialized model weights from {init_weights}")

    backbone_kwargs = config["backbone"]
    print(
        "Using feature space reconstruction with "
        f"{backbone_kwargs['model_type']} backbone"
    )
    feature_extractor = get_backbone(**backbone_kwargs)
    feature_extractor.to(device).eval()

    optimizer = get_optimizer([model], **config["optimizer"])
    scheduler = None
    if config["optimizer"]["scheduler_type"] != "none":
        scheduler = get_lr_scheduler(
            optimizer,
            **config["optimizer"],
            iter_per_epoch=len(train_loader),
        )

    start_epoch = 0
    global_step = 0
    best_metric = None
    if resume is not None:
        progress = load_training_checkpoint(
            resume,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
        )
        start_epoch = progress["next_epoch"]
        global_step = progress["global_step"]
        best_metric = progress["best_metric"]
        print(
            f"Resumed {resume} after epoch {progress['completed_epoch']} "
            f"at global step {global_step}"
        )

    num_epochs = config["optimizer"]["num_epochs"]
    if start_epoch > num_epochs:
        raise ValueError(
            f"Checkpoint next_epoch={start_epoch} exceeds num_epochs={num_epochs}"
        )

    save_dir = Path(config["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "training_latest.pth"
    with open(save_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    print(f"Steps per epoch: {len(train_loader)}")

    ema_decay = config["diffusion"]["ema_decay"]
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for iteration, data in enumerate(train_loader):
            images = data["samples"].to(device)
            labels = data["clslabels"].to(device)

            with torch.no_grad():
                features, _ = feature_extractor(images)
            loss = model(features, labels)

            optimizer.zero_grad()
            loss.backward()
            if config["optimizer"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["optimizer"]["grad_clip"]
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                for ema_parameter, model_parameter in zip(
                    model_ema.parameters(), model.parameters()
                ):
                    ema_parameter.mul_(ema_decay).add_(
                        model_parameter, alpha=1.0 - ema_decay
                    )

            global_step += 1
            if iteration % config["logging"]["log_interval"] == 0:
                learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch}, Iter {iteration}, Step {global_step}, "
                    f"Loss {loss.item()}, LR {learning_rate}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "Loss": loss.item(),
                            "LR": learning_rate,
                            "global_step": global_step,
                        }
                    )

        try:
            if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
                metrics = evaluate_inv(
                    model,
                    feature_extractor,
                    anom_loader,
                    normal_loader,
                    config,
                    diff_in_sh,
                    epoch + 1,
                    config["evaluation"]["eval_step"],
                    device,
                )
                current_metric = float(
                    metrics[config["data"]["category"]]["mAD"]
                )
                if best_metric is None or current_metric > best_metric:
                    best_metric = current_metric
                print(f"mAD: {current_metric} at epoch {epoch + 1}")
                if use_wandb:
                    wandb.log({"mAD": current_metric, "global_step": global_step})
        finally:
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config,
            )
            print(f"Training checkpoint saved at {checkpoint_path}")

    atomic_torch_save(model.state_dict(), save_dir / "model_latest.pth")
    atomic_torch_save(model_ema.state_dict(), save_dir / "model_ema_latest.pth")
    print(f"Training is done. Evaluation weights are saved at {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, encoding="utf-8") as file:
        runtime_config = yaml.safe_load(file)
    main(runtime_config, resume=args.resume, init_weights=args.init_weights)
