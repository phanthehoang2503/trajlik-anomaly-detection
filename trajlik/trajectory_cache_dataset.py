import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from trajlik.cache_identity import (
    checkpoint_identity_errors,
    normalized_timestep_map,
)


class TrajectoryCacheDataset(Dataset):
    """Read a cache whose index explicitly proves normal-training provenance."""

    TENSOR_KEYS = (
        "z_0",
        "z_seq",
        "eps_seq",
        "delta_z_seq",
        "a_end_coarse",
    )

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "cache_meta.json").open(encoding="utf-8") as file:
            self.metadata = json.load(file)
        with (self.cache_dir / "cache_index.json").open(encoding="utf-8") as file:
            self.index = json.load(file)

        identity_errors = checkpoint_identity_errors(
            self.metadata.get("invad_checkpoint")
        )
        if identity_errors:
            raise ValueError("Invalid cache identity: " + "; ".join(identity_errors))
        try:
            timestep_map = normalized_timestep_map(self.metadata["timestep_map"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid cache timestep_map") from error
        if len(timestep_map) != int(self.metadata["num_steps"]):
            raise ValueError("Cache timestep_map length does not match num_steps")

        if self.metadata.get("normal_only") is not True:
            raise ValueError("Trajectory head training requires a normal-only cache")
        if int(self.metadata["num_images"]) != len(self.index):
            raise ValueError("Cache metadata and index lengths do not match")
        if not self.index:
            raise ValueError("Trajectory cache is empty")

        self.categories = []
        for entry in self.index:
            if entry.get("split") != "train" or entry.get("is_normal") is not True:
                raise ValueError("Cache index contains non-normal training data")
            path = self.cache_dir / entry["file"]
            if not path.is_file():
                raise FileNotFoundError(f"Indexed cache sample is missing: {path}")
            self.categories.append(entry["category"])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        entry = self.index[index]
        sample = torch.load(
            self.cache_dir / entry["file"],
            map_location="cpu",
            weights_only=False,
        )
        if sample.get("split") != "train" or sample.get("is_normal") is not True:
            raise ValueError("Cache sample failed normal-training provenance check")
        missing = set(self.TENSOR_KEYS) - sample.keys()
        if missing:
            raise ValueError(f"Cache sample is missing tensors: {sorted(missing)}")
        return {key: sample[key] for key in self.TENSOR_KEYS}


def stratified_normal_split(
    categories,
    calibration_fraction=0.05,
    seed=42,
):
    """Split normal images while retaining each sufficiently large category."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    if len(categories) < 2:
        raise ValueError("At least two normal images are required")

    grouped = defaultdict(list)
    for index, category in enumerate(categories):
        grouped[category].append(index)
    generator = torch.Generator().manual_seed(seed)
    training_indices = []
    calibration_indices = []
    for category in sorted(grouped):
        indices = torch.tensor(grouped[category], dtype=torch.long)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        if indices.numel() == 1:
            training_indices.extend(indices.tolist())
            continue
        calibration_size = max(1, round(indices.numel() * calibration_fraction))
        calibration_size = min(calibration_size, indices.numel() - 1)
        calibration_indices.extend(indices[:calibration_size].tolist())
        training_indices.extend(indices[calibration_size:].tolist())

    if not calibration_indices:
        # This occurs only when every category has one image. Preserve a
        # non-empty training set and select one global calibration image.
        calibration_indices.append(training_indices.pop())
    return (
        torch.tensor(sorted(training_indices), dtype=torch.long),
        torch.tensor(sorted(calibration_indices), dtype=torch.long),
    )


def balanced_category_sampler(categories, indices, seed=42):
    selected_categories = [categories[index] for index in indices]
    counts = Counter(selected_categories)
    weights = torch.tensor(
        [1.0 / counts[category] for category in selected_categories],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(selected_categories),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
