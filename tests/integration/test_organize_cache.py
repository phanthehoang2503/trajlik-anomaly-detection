import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.cache_trajectories_check import validate_cache
from trajlik.cache_layout import organize_cache
from trajlik.trajectory_cache_dataset import TrajectoryCacheDataset


class OrganizeCacheTest(unittest.TestCase):
    def test_flat_cache_is_grouped_and_remains_loadable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            categories = ("bottle", "metal_nut")
            index = []
            for category in categories:
                filename = f"{category}_good_000.pt"
                sample = {
                    "z_0": torch.zeros(4, 2, 2),
                    "z_seq": torch.zeros(3, 4, 2, 2),
                    "eps_seq": torch.zeros(3, 4, 2, 2),
                    "delta_z_seq": torch.zeros(3, 4, 2, 2),
                    "a_end_coarse": torch.zeros(2, 2),
                    "source_path": f"{category}/train/good/000.png",
                    "category": category,
                    "split": "train",
                    "is_normal": True,
                }
                torch.save(sample, root / filename)
                index.append(
                    {
                        "file": filename,
                        "source_path": sample["source_path"],
                        "category": category,
                        "split": "train",
                        "is_normal": True,
                    }
                )

            (root / "cache_meta.json").write_text(
                json.dumps(
                    {
                        "num_images": 2,
                        "num_steps": 3,
                        "invad_checkpoint": {
                            "filename": "model.pth",
                            "size_bytes": 123,
                            "sha256": "a" * 64,
                        },
                        "output_channels": 4,
                        "storage_dtype": "float32",
                        "projection": "none",
                        "normal_only": True,
                        "timestep_map": [0, 500, 999],
                    }
                ),
                encoding="utf-8",
            )
            (root / "cache_index.json").write_text(
                json.dumps(index),
                encoding="utf-8",
            )

            plan = organize_cache(root)
            self.assertEqual(len(plan), 2)
            self.assertTrue((root / "bottle_good_000.pt").is_file())

            organize_cache(root, apply=True)

            self.assertTrue((root / "bottle" / "bottle_good_000.pt").is_file())
            self.assertTrue(
                (root / "metal_nut" / "metal_nut_good_000.pt").is_file()
            )
            updated_index = json.loads(
                (root / "cache_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {entry["file"] for entry in updated_index},
                {
                    "bottle/bottle_good_000.pt",
                    "metal_nut/metal_nut_good_000.pt",
                },
            )
            self.assertEqual(len(TrajectoryCacheDataset(root)), 2)
            validate_cache(root, expected_categories=list(categories))


if __name__ == "__main__":
    unittest.main()
