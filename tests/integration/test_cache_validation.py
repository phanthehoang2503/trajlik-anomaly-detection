import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch



SCRIPT_PATH = Path(__file__).parents[1] / "cache_trajectories_check.py"
SPEC = importlib.util.spec_from_file_location("cache_check", SCRIPT_PATH)
CACHE_CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE_CHECK)


class CacheValidationTest(unittest.TestCase):
    def _write_cache(self, root: Path, **sample_updates):
        metadata = {
            "num_images": 1,
            "num_steps": 3,
            "output_channels": 8,
            "storage_dtype": "float32",
            "projection": "none",
            "normal_only": True,
            "timestep_map": [0, 1, 2],
            "invad_checkpoint": {
                "filename": "model.pth",
                "size_bytes": 123,
                "sha256": "a" * 64,
            },
        }
        (root / "cache_meta.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        (root / "cache_index.json").write_text(
            json.dumps(
                [
                    {
                        "file": "candle_000.pt",
                        "category": "candle",
                        "split": "train",
                        "is_normal": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        sample = {
            "z_0": torch.randn(8, 2, 2),
            "z_seq": torch.randn(3, 8, 2, 2),
            "eps_seq": torch.randn(3, 8, 2, 2),
            "delta_z_seq": torch.randn(3, 8, 2, 2),
            "a_end_coarse": torch.rand(2, 2),
            "category": "candle",
            # A VisA path does not contain MVTec's train/good convention.
            "source_path": "visa/candle/Data/Images/Normal/000.JPG",
            "split": "train",
            "is_normal": True,
        }
        sample.update(sample_updates)
        torch.save(sample, root / "candle_000.pt")

    def test_visa_normal_training_cache_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)

            CACHE_CHECK.validate_cache(root, ["candle"])

    def test_missing_normal_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, is_normal=False)

            with self.assertRaisesRegex(AssertionError, "marked normal"):
                CACHE_CHECK.validate_cache(root)

    def test_checkpoint_identity_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            metadata_path = root / "cache_meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("invad_checkpoint")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "invad_checkpoint"):
                CACHE_CHECK.validate_cache(root)


if __name__ == "__main__":
    unittest.main()
