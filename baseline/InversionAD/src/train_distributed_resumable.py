import copy
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

import src.evaluate as evaluate
from src.backbones import get_backbone, get_backbone_feature_shape
from src.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    load_initial_weights,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.datasets import build_dataset
from src.denoiser import Denoiser, get_denoiser
from src.train_distributed import DistributedEvalSampler
from src.utils import get_lr_scheduler, get_optimizer, init_distributed

try:
    import wandb
except ImportError:
    wandb = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_wandb(config, rank):
    if rank != 0:
        return False
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
    torch.cuda.manual_seed_all(seed)


def _save_distributed_checkpoint(
    path,
    *,
    model,
    model_ema,
    optimizer,
    scheduler,
    completed_epoch,
    global_step,
    best_metric,
    config,
    world_size,
    rank,
):
    local_rng = capture_rng_state()
    rng_states = [None] * world_size if rank == 0 else None
    dist.gather_object(local_rng, rng_states, dst=0)
    if rank == 0:
        save_training_checkpoint(
            path,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epoch=completed_epoch,
            global_step=global_step,
            best_metric=best_metric,
            config=config,
            rng_states=rng_states,
        )
        logger.info("Training checkpoint saved at %s", path)
    dist.barrier()


def main(config, *, resume=None, init_weights=None):
    world_size, rank = init_distributed()
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized")
    if rank != 0:
        logger.setLevel(logging.ERROR)

    use_wandb = _init_wandb(config, rank)
    _seed_everything(config["meta"]["seed"] + rank)
    device = torch.device("cuda:0")
    batch_size = config["data"]["batch_size"]

    dataset_config = copy.deepcopy(config["data"])
    train_config = copy.deepcopy(config["data"])
    train_config.update(train=True, normal_only=True, anom_only=False)
    train_dataset = build_dataset(**train_config)
    dataset_config.update(train=False, anom_only=True)
    anom_dataset = build_dataset(**dataset_config)
    dataset_config.update(anom_only=False, normal_only=True)
    normal_dataset = build_dataset(**dataset_config)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config["meta"]["seed"] + rank)
    anom_samplers = [
        DistributedEvalSampler(dataset, world_size, rank)
        for dataset in anom_dataset.datasets
    ]
    normal_samplers = [
        DistributedEvalSampler(dataset, world_size, rank)
        for dataset in normal_dataset.datasets
    ]
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        pin_memory=config["data"].get("pin_memory", True),
        num_workers=config["data"].get("num_workers", 4),
        persistent_workers=config["data"].get("num_workers", 4) > 0,
        drop_last=True,
        generator=loader_generator,
    )
    eval_batch_size = max(1, batch_size // world_size)
    anom_loaders = [
        torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            pin_memory=True,
            num_workers=2,
        )
        for dataset, sampler in zip(anom_dataset.datasets, anom_samplers)
    ]
    normal_loaders = [
        torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=eval_batch_size,
            pin_memory=True,
            num_workers=2,
        )
        for dataset, sampler in zip(normal_dataset.datasets, normal_samplers)
    ]

    diff_in_sh = get_backbone_feature_shape(
        model_type=config["backbone"]["model_type"]
    )
    model: Denoiser = get_denoiser(**config["diffusion"], input_shape=diff_in_sh)
    model_ema = copy.deepcopy(model)
    model.to(device)
    model_ema.to(device)

    if init_weights is not None:
        load_initial_weights(init_weights, model=model, model_ema=model_ema)
        if rank == 0:
            logger.info("Initialized model weights from %s", init_weights)

    model = torch.nn.parallel.DistributedDataParallel(model, static_graph=True)
    model_ema = torch.nn.parallel.DistributedDataParallel(
        model_ema, static_graph=True
    )
    for parameter in model_ema.parameters():
        parameter.requires_grad = False

    feature_extractor = get_backbone(**config["backbone"])
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
            rank=rank,
            world_size=world_size,
        )
        start_epoch = progress["next_epoch"]
        global_step = progress["global_step"]
        best_metric = progress["best_metric"]
        if rank == 0:
            logger.info(
                "Resumed %s after epoch %s at global step %s",
                resume,
                progress["completed_epoch"],
                global_step,
            )
    dist.barrier()

    num_epochs = config["optimizer"]["num_epochs"]
    if start_epoch > num_epochs:
        raise ValueError(
            f"Checkpoint next_epoch={start_epoch} exceeds num_epochs={num_epochs}"
        )

    save_dir = Path(config["logging"]["save_dir"])
    checkpoint_path = save_dir / "training_latest.pth"
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "config.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file)
    dist.barrier()

    logger.info("Steps per epoch: %s", len(train_loader))
    ema_decay = config["diffusion"]["ema_decay"]
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for iteration, data in enumerate(train_loader):
            started_at = time.time()
            images = data["samples"].to(device)
            labels = data["clslabels"].to(device)
            loaded_at = time.time()

            with torch.no_grad():
                features, _ = feature_extractor(images)
            forwarded_at = time.time()

            loss = model(features, labels)
            optimizer.zero_grad()
            loss.backward()
            if config["optimizer"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["optimizer"]["grad_clip"]
                )
            optimizer.step()
            optimized_at = time.time()
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
            if iteration % config["logging"]["log_interval"] == 0 and rank == 0:
                learning_rate = optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %s, Iter %s, Step %s, Loss %.4f, LR %.6f",
                    epoch,
                    iteration,
                    global_step,
                    loss.item(),
                    learning_rate,
                )
                if use_wandb:
                    wandb.log(
                        {
                            "Loss": loss.item(),
                            "LR": learning_rate,
                            "global_step": global_step,
                            "Time/Data [ms]": (loaded_at - started_at) * 1000,
                            "Time/Forward [ms]": (forwarded_at - loaded_at) * 1000,
                            "Time/Backward [ms]": (optimized_at - forwarded_at) * 1000,
                        }
                    )

        _save_distributed_checkpoint(
            checkpoint_path,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            scheduler=scheduler,
            completed_epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            config=config,
            world_size=world_size,
            rank=rank,
        )

        if (epoch + 1) % config["evaluation"]["eval_interval"] == 0:
            all_results = {}
            for anom_loader, normal_loader in zip(anom_loaders, normal_loaders):
                metrics = evaluate.evaluate_dist(
                    model,
                    feature_extractor,
                    anom_loader,
                    normal_loader,
                    config,
                    diff_in_sh,
                    epoch + 1,
                    config["evaluation"]["eval_step"],
                    device,
                    world_size=world_size,
                    rank=rank,
                )
                if rank == 0:
                    all_results.update(metrics)
                evaluate.distributed_barrier()

            if rank == 0:
                current_metric = float(
                    np.mean([result["mAD"] for result in all_results.values()])
                )
                if best_metric is None or current_metric > best_metric:
                    best_metric = current_metric
                logger.info("Average mAD: %s at epoch %s", current_metric, epoch + 1)
                if use_wandb:
                    wandb.log(
                        {"mAD": current_metric, "global_step": global_step}
                    )
            best_metric_container = [best_metric]
            dist.broadcast_object_list(best_metric_container, src=0)
            best_metric = best_metric_container[0]
            _save_distributed_checkpoint(
                checkpoint_path,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config,
                world_size=world_size,
                rank=rank,
            )

    if rank == 0:
        atomic_torch_save(model.module.state_dict(), save_dir / "model_latest.pth")
        atomic_torch_save(
            model_ema.module.state_dict(), save_dir / "model_ema_latest.pth"
        )
        logger.info("Training is done. Evaluation weights are saved at %s", save_dir)
    dist.barrier()
