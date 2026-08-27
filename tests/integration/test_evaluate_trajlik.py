import copy
import sys
import unittest
from pathlib import Path

import torch

baseline_root = str(Path(__file__).resolve().parents[2] / "baseline" / "InversionAD")
if baseline_root not in sys.path:
    sys.path.insert(0, baseline_root)

from scripts.evaluate_trajlik import (
    config_fingerprint,
    evaluate_loader,
    resolve_evaluation_autocast,
    validate_runtime_contract,
)
from trajlik.cache_identity import checkpoint_identity


def runtime_config():
    return {
        "data": {
            "img_size": 256,
            "transform_type": "imagenet",
            "dataset_name": "mvtec_ad_all",
            "data_root": "data/mvtec_ad",
        },
        "backbone": {
            "model_type": "efficientnet-b4",
            "outblocks": [1, 5, 9, 21],
            "outstrides": [2, 4, 8, 16],
            "stride": 16,
        },
        "diffusion": {
            "model_type": "dit",
            "num_classes": 15,
            "z_channels": 768,
            "depth": 8,
            "width": 1024,
            "patch_size": 1,
            "learn_sigma": False,
        },
        "evaluation": {"eval_step": 3},
    }


class EvaluateTrajLikTest(unittest.TestCase):
    def _head_checkpoint(self, config, checkpoint):
        return {
            "cache_metadata": {
                "config_sha256": config_fingerprint(config),
                "normal_only": True,
                "num_steps": 3,
                "timestep_map": [0, 1, 2],
                "invad_checkpoint": checkpoint_identity(checkpoint),
                "projection": "none",
                "output_channels": 272,
                "backbone": "efficientnet-b4",
                "transform_type": "imagenet",
                "img_size": 256,
                "autocast_dtype": "none",
            }
        }

    def test_matching_runtime_contract_is_accepted(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
            (directory / "config.yaml").write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            validate_runtime_contract(
                config,
                checkpoint,
                self._head_checkpoint(config, checkpoint),
                feature_channels=272,
                runtime_timestep_map=[0, 1, 2],
                runtime_autocast_dtype="none",
            )

    def test_official_config_and_relocated_dataset_are_accepted(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        config["data"]["transform_type"] = "default"
        config["data"]["data_root"] = "/kaggle/input/datasets/ipythonx/mvtec-ad"
        config["evaluation"]["eval_step"] = 4
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
            head_checkpoint = self._head_checkpoint(config, checkpoint)
            head_checkpoint["cache_metadata"]["config_sha256"] = (
                "original-cache-hash"
            )
            head_checkpoint["cache_metadata"]["transform_type"] = "default"
            (directory / "config.yaml").write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            with self.assertLogs("scripts.evaluate_trajlik", level="WARNING"):
                validate_runtime_contract(
                    config,
                    checkpoint,
                    head_checkpoint,
                    feature_channels=272,
                    runtime_timestep_map=[0, 1, 2],
                    runtime_autocast_dtype="none",
                )

    def test_projected_cache_and_config_mismatch_are_rejected(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        checkpoint_config = copy.deepcopy(config)
        checkpoint_config["diffusion"]["depth"] = 16
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
            head_checkpoint = self._head_checkpoint(config, checkpoint)
            head_checkpoint["cache_metadata"]["projection"] = "linear"
            (directory / "config.yaml").write_text(
                yaml.safe_dump(checkpoint_config),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "projection=none") as context:
                validate_runtime_contract(
                    config,
                    checkpoint,
                    head_checkpoint,
                    feature_channels=272,
                    runtime_timestep_map=[0, 1, 2],
                    runtime_autocast_dtype="none",
                )
            self.assertIn("diffusion.depth", str(context.exception))

    def test_checkpoint_sha256_mismatch_is_rejected(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.write_bytes(b"checkpoint-a")
            head_checkpoint = self._head_checkpoint(config, checkpoint)
            checkpoint.write_bytes(b"checkpoint-b")
            (directory / "config.yaml").write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_runtime_contract(
                    config,
                    checkpoint,
                    head_checkpoint,
                    feature_channels=272,
                    runtime_timestep_map=[0, 1, 2],
                    runtime_autocast_dtype="none",
                )

    def test_timestep_map_mismatch_is_rejected(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
            (directory / "config.yaml").write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "timestep_map"):
                validate_runtime_contract(
                    config,
                    checkpoint,
                    self._head_checkpoint(config, checkpoint),
                    feature_channels=272,
                    runtime_timestep_map=[0, 2, 3],
                    runtime_autocast_dtype="none",
                )

    def test_missing_cache_identity_is_rejected(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
            (directory / "config.yaml").write_text(
                yaml.safe_dump(config),
                encoding="utf-8",
            )
            head_checkpoint = self._head_checkpoint(config, checkpoint)
            for key in ("invad_checkpoint", "timestep_map"):
                head_checkpoint["cache_metadata"].pop(key)

            with self.assertRaisesRegex(ValueError, "missing invad_checkpoint"):
                validate_runtime_contract(
                    config,
                    checkpoint,
                    head_checkpoint,
                    feature_channels=272,
                    runtime_timestep_map=[0, 1, 2],
                    runtime_autocast_dtype="none",
                )

    def test_cache_autocast_is_used_and_mismatch_is_rejected(self):
        metadata = {"autocast_dtype": "none"}
        enabled, dtype, name = resolve_evaluation_autocast(
            "cpu",
            "cache",
            metadata,
        )
        self.assertFalse(enabled)
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(name, "none")

        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_evaluation_autocast("cpu", "float16", metadata)

    def test_evaluate_loader_exports_all_score_branches(self):
        class FakeModel:
            def __call__(self, images, labels, lambda_path=1.0):
                batch_size = images.shape[0]
                endpoint_tail = torch.tensor(
                    [[[0.0, 1.0], [2.0, 3.0]]],
                ).repeat(batch_size, 1, 1)
                path_tail = torch.tensor(
                    [[[3.0, 2.0], [1.0, 0.0]]],
                ).repeat(batch_size, 1, 1)
                coarse_map = endpoint_tail + lambda_path * path_tail
                return {
                    "endpoint_tail": endpoint_tail,
                    "path_tail": path_tail,
                    "image_score": (
                        coarse_map.amax(dim=(-2, -1))
                        - coarse_map.amin(dim=(-2, -1))
                    ),
                    "pixel_map": torch.nn.functional.interpolate(
                        coarse_map.unsqueeze(1),
                        size=images.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1),
                    "a_end": torch.ones(
                        batch_size,
                        *images.shape[-2:],
                    ),
                    "a_end_coarse": torch.tensor(
                        [[[0.0, 1.0], [2.0, 3.0]]],
                    ).repeat(batch_size, 1, 1),
                    "final_latent": torch.arange(
                        batch_size * 4.0,
                    ).reshape(batch_size, 1, 2, 2),
                }

        loader = [
            {
                "samples": torch.zeros(2, 3, 8, 8),
                "clslabels": torch.zeros(2, dtype=torch.long),
                "labels": torch.tensor([0, 1]),
                "masks": torch.zeros(2, 1, 8, 8),
            }
        ]
        predictions = evaluate_loader(
            FakeModel(),
            loader,
            torch.device("cpu"),
            lambda_path=0.5,
        )

        self.assertEqual(
            set(predictions["branches"]),
            {
                "invad_formula_control",
                "endpoint_only",
                "path_only",
                "fused",
            },
        )
        for scores in predictions["branches"].values():
            self.assertEqual(scores["image_scores"].shape, (2,))
            self.assertEqual(scores["pixel_maps"].shape, (2, 8, 8))
        torch.testing.assert_close(
            torch.from_numpy(
                predictions["branches"]["invad_formula_control"][
                    "image_scores"
                ]
            ),
            torch.tensor([0.0, 1.0]),
        )

if __name__ == "__main__":
    unittest.main()
