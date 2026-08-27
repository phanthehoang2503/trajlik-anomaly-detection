import json
import tempfile
import unittest
from pathlib import Path

import torch

from trajlik.trajectory_cache_dataset import (
    TrajectoryCacheDataset,
    balanced_category_sampler,
    stratified_normal_split,
)


class TrajectoryCacheDatasetTest(unittest.TestCase):
    def _write_cache(self, root, is_normal=True):
        index = []
        categories = ["a", "a", "a", "a", "b", "b", "b", "b"]
        for sample_index, category in enumerate(categories):
            filename = f"{sample_index}.pt"
            sample = {
                "z_0": torch.randn(8, 2, 2),
                "z_seq": torch.randn(3, 8, 2, 2),
                "eps_seq": torch.randn(3, 8, 2, 2),
                "delta_z_seq": torch.randn(3, 8, 2, 2),
                "a_end_coarse": torch.rand(2, 2),
                "split": "train",
                "is_normal": is_normal,
            }
            torch.save(sample, root / filename)
            index.append(
                {
                    "file": filename,
                    "category": category,
                    "split": "train",
                    "is_normal": is_normal,
                }
            )
        (root / "cache_meta.json").write_text(
            json.dumps(
                {
                    "num_images": 8,
                    "num_steps": 3,
                    "timestep_map": [0, 499, 999],
                    "invad_checkpoint": {
                        "filename": "model.pth",
                        "size_bytes": 123,
                        "sha256": "a" * 64,
                    },
                    "output_channels": 8,
                    "normal_only": True,
                }
            ),
            encoding="utf-8",
        )
        (root / "cache_index.json").write_text(json.dumps(index), encoding="utf-8")

    def test_dataset_and_stratified_split_use_only_normal_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            dataset = TrajectoryCacheDataset(root)

            training, calibration = stratified_normal_split(
                dataset.categories,
                calibration_fraction=0.25,
            )
            sample = dataset[0]
            sampler = balanced_category_sampler(dataset.categories, training.tolist())

            self.assertEqual(len(dataset), 8)
            self.assertEqual(set(sample), set(TrajectoryCacheDataset.TENSOR_KEYS))
            self.assertEqual(training.numel(), 6)
            self.assertEqual(calibration.numel(), 2)
            self.assertEqual(len(list(iter(sampler))), 6)

    def test_non_normal_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, is_normal=False)

            with self.assertRaisesRegex(ValueError, "non-normal"):
                TrajectoryCacheDataset(root)

if __name__ == "__main__":
    unittest.main()
