import copy
import sys
import unittest
from pathlib import Path

baseline_root = str(Path(__file__).resolve().parents[2] / "baseline" / "InversionAD")
if baseline_root not in sys.path:
    sys.path.insert(0, baseline_root)

from scripts.evaluate_trajlik import config_fingerprint, validate_runtime_contract


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
    def _head_checkpoint(self, config):
        return {
            "cache_metadata": {
                "config_sha256": config_fingerprint(config),
                "normal_only": True,
                "num_steps": 3,
                "projection": "none",
                "output_channels": 272,
                "backbone": "efficientnet-b4",
                "transform_type": "imagenet",
                "img_size": 256,
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
                self._head_checkpoint(config),
                feature_channels=272,
            )

    def test_official_config_and_relocated_dataset_are_accepted(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        config["data"]["transform_type"] = "default"
        config["data"]["data_root"] = "/kaggle/input/datasets/ipythonx/mvtec-ad"
        config["evaluation"]["eval_step"] = 4
        head_checkpoint = self._head_checkpoint(config)
        head_checkpoint["cache_metadata"]["config_sha256"] = "original-cache-hash"
        head_checkpoint["cache_metadata"]["transform_type"] = "default"

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
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
                )

    def test_projected_cache_and_config_mismatch_are_rejected(self):
        import tempfile
        from pathlib import Path
        import yaml

        config = runtime_config()
        checkpoint_config = copy.deepcopy(config)
        checkpoint_config["diffusion"]["depth"] = 16
        head_checkpoint = self._head_checkpoint(config)
        head_checkpoint["cache_metadata"]["projection"] = "linear"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "model.pth"
            checkpoint.touch()
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
                )
            self.assertIn("diffusion.depth", str(context.exception))


if __name__ == "__main__":
    unittest.main()
