import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


CHECKPOINT_TYPE = "invad_training_v1"


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _strip_module_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data_config = copy.deepcopy(config["data"])
    for key in ("data_root", "num_workers", "pin_memory"):
        data_config.pop(key, None)

    return {
        "meta": {"seed": config["meta"]["seed"]},
        "data": data_config,
        "backbone": copy.deepcopy(config["backbone"]),
        "diffusion": copy.deepcopy(config["diffusion"]),
        "optimizer": copy.deepcopy(config["optimizer"]),
    }


def config_fingerprint(config: Dict[str, Any]) -> str:
    serialized = json.dumps(
        _training_config(config), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state, but CUDA is unavailable")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the checkpoint: "
                f"checkpoint={len(cuda_state)}, runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _scheduler_state(scheduler) -> Optional[Dict[str, Any]]:
    if scheduler is None:
        return None
    return {
        "step_num": scheduler.step_num,
        "last_lr": scheduler.last_lr,
    }


def _restore_scheduler_state(scheduler, state: Optional[Dict[str, Any]]) -> None:
    if scheduler is None and state is None:
        return
    if scheduler is None or state is None:
        raise ValueError("Scheduler presence differs between config and checkpoint")
    scheduler.step_num = int(state["step_num"])
    scheduler.last_lr = float(state["last_lr"])


def atomic_torch_save(payload: Any, path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    path,
    *,
    model,
    model_ema,
    optimizer,
    scheduler,
    completed_epoch: int,
    global_step: int,
    best_metric: Optional[float],
    config: Dict[str, Any],
    rng_states=None,
) -> None:
    if rng_states is None:
        rng_states = [capture_rng_state()]
    payload = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "model": _unwrap_model(model).state_dict(),
        "model_ema": _unwrap_model(model_ema).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": _scheduler_state(scheduler),
        "progress": {
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "global_step": int(global_step),
        },
        "best_metric": best_metric,
        "rng_by_rank": rng_states,
        "world_size": len(rng_states),
        "config": copy.deepcopy(config),
        "training_config": _training_config(config),
        "config_fingerprint": config_fingerprint(config),
    }
    atomic_torch_save(payload, path)


def _load_checkpoint_file(path):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_training_checkpoint(
    path,
    *,
    model,
    model_ema,
    optimizer,
    scheduler,
    config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    checkpoint = _load_checkpoint_file(path)
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError(
            f"--resume requires a {CHECKPOINT_TYPE} checkpoint; "
            "use --init_weights for a weights-only checkpoint"
        )

    expected_fingerprint = config_fingerprint(config)
    actual_fingerprint = checkpoint.get("config_fingerprint")
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Resume config is incompatible with the checkpoint "
            f"(checkpoint={actual_fingerprint}, runtime={expected_fingerprint})"
        )

    checkpoint_world_size = int(checkpoint.get("world_size", 1))
    if checkpoint_world_size != world_size:
        raise ValueError(
            "Resume world size differs from the checkpoint "
            f"(checkpoint={checkpoint_world_size}, runtime={world_size})"
        )
    if not 0 <= rank < checkpoint_world_size:
        raise ValueError(f"Invalid rank {rank} for world size {checkpoint_world_size}")

    _unwrap_model(model).load_state_dict(
        _strip_module_prefix(checkpoint["model"]), strict=True
    )
    _unwrap_model(model_ema).load_state_dict(
        _strip_module_prefix(checkpoint["model_ema"]), strict=True
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_scheduler_state(scheduler, checkpoint["scheduler"])
    _restore_rng_state(checkpoint["rng_by_rank"][rank])

    progress = checkpoint["progress"]
    return {
        "completed_epoch": int(progress["completed_epoch"]),
        "next_epoch": int(progress["next_epoch"]),
        "global_step": int(progress["global_step"]),
        "best_metric": checkpoint.get("best_metric"),
    }


def load_initial_weights(path, *, model, model_ema) -> None:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict):
        raise ValueError("Weights checkpoint must contain a state_dict mapping")

    state_dict = _strip_module_prefix(state_dict)
    _unwrap_model(model).load_state_dict(state_dict, strict=True)
    _unwrap_model(model_ema).load_state_dict(state_dict, strict=True)
