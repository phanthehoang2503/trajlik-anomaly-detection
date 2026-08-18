import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from scripts.train_trajlik import evaluate_head, load_trajlik_checkpoint, train


class TrainTrajLikSmokeTest(unittest.TestCase):
    def test_validation_uses_full_nll_and_a_separate_masked_msm_pass(self):
        class TrackingHead:
            lambda_msm = 2.0

            def __init__(self):
                self.masks = []
                self.masked_passes = 0

            def eval(self):
                return self

            def __call__(self, batch, *, mask):
                self.masks.append(mask)
                return {
                    "path_nll": torch.tensor([[5.0]]),
                    "base_latents": torch.zeros(1, 1, 2),
                    "log_det": torch.tensor([[-3.1621229336]]),
                }

            def training_loss(self, batch):
                self.masked_passes += 1
                return {"msm_loss": torch.tensor(0.25)}

        head = TrackingHead()
        metrics = evaluate_head(
            head,
            [{"sample": torch.zeros(1)}],
            torch.device("cpu"),
            seed=42,
        )

        self.assertEqual(head.masks, [False])
        self.assertEqual(head.masked_passes, 1)
        self.assertAlmostEqual(metrics["nll_loss"], 5.0)
        self.assertAlmostEqual(metrics["msm_loss"], 0.25)
        self.assertAlmostEqual(metrics["loss"], 5.5)

    def _write_cache(self, root):
        index = []
        for sample_index in range(4):
            filename = f"{sample_index}.pt"
            states = torch.randn(4, 8, 2, 2)
            torch.save(
                {
                    "z_0": states[0],
                    "z_seq": states[1:],
                    "eps_seq": torch.randn(3, 8, 2, 2),
                    "delta_z_seq": states[1:] - states[:-1],
                    "a_end_coarse": torch.linalg.vector_norm(states[-1], dim=0),
                    "split": "train",
                    "is_normal": True,
                },
                root / filename,
            )
            index.append(
                {
                    "file": filename,
                    "category": "a",
                    "split": "train",
                    "is_normal": True,
                }
            )
        (root / "cache_meta.json").write_text(
            json.dumps(
                {
                    "num_images": 4,
                    "num_steps": 3,
                    "output_channels": 8,
                    "normal_only": True,
                    "projection": "none",
                    "config_sha256": "test",
                }
            ),
            encoding="utf-8",
        )
        (root / "cache_index.json").write_text(json.dumps(index), encoding="utf-8")

    def test_best_checkpoint_can_be_loaded_after_early_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            self._write_cache(cache_dir)
            checkpoint_path = root / "head.pth"
            args = Namespace(
                cache_dir=str(cache_dir),
                output_path=str(checkpoint_path),
                device="cpu",
                epochs=3,
                batch_size=2,
                num_workers=0,
                learning_rate=1e-4,
                weight_decay=1e-4,
                grad_clip=1.0,
                calibration_fraction=0.25,
                validation_fraction=0.25,
                patience=1,
                min_delta=1e9,
                seed=42,
                projection_dim=4,
                token_dim=8,
                trajectory_dim=4,
                dcte_layers=1,
                dcte_heads=2,
                flow_blocks=1,
                flow_bins=4,
                flow_dequantization_std=0.1,
                lambda_msm=1.0,
            )

            train(args)
            head, calibrator, checkpoint = load_trajlik_checkpoint(
                checkpoint_path
            )

            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(calibrator.fitted)
            self.assertFalse(head.training)
            self.assertEqual(checkpoint["calibration_indices"].numel(), 1)
            self.assertEqual(checkpoint["validation_indices"].numel(), 1)
            self.assertEqual(checkpoint["training_indices"].numel(), 2)
            all_indices = torch.cat(
                (
                    checkpoint["training_indices"],
                    checkpoint["validation_indices"],
                    checkpoint["calibration_indices"],
                )
            )
            self.assertEqual(set(all_indices.tolist()), {0, 1, 2, 3})
            self.assertEqual(len(checkpoint["training_history"]), 2)
            self.assertEqual(checkpoint["best_epoch"], 1)
            self.assertIn("validation_nll", checkpoint["training_history"][0])
            self.assertIn("validation_msm", checkpoint["training_history"][0])
            self.assertIn("validation_log_det", checkpoint["training_history"][0])
            self.assertEqual(
                checkpoint["training_args"]["flow_dequantization_std"],
                0.1,
            )
            first_epoch = checkpoint["training_history"][0]
            self.assertAlmostEqual(
                first_epoch["validation_nll"],
                first_epoch["validation_base_nll"]
                - first_epoch["validation_log_det"],
                places=5,
            )
            self.assertTrue((root / "head_best.pth").is_file())
            self.assertTrue((root / "head_latest.pth").is_file())
            best_checkpoint = torch.load(
                root / "head_best.pth",
                map_location="cpu",
                weights_only=False,
            )
            latest_checkpoint = torch.load(
                root / "head_latest.pth",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(best_checkpoint["checkpoint_type"], "trajlik_final")
            self.assertEqual(latest_checkpoint["checkpoint_type"], "trajlik_training")
            with self.assertRaisesRegex(ValueError, "do not contain normal calibration"):
                load_trajlik_checkpoint(root / "head_latest.pth")


if __name__ == "__main__":
    unittest.main()
