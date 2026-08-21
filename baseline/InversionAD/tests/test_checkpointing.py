import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.checkpointing import (
    CHECKPOINT_TYPE,
    atomic_torch_save,
    config_fingerprint,
    load_initial_weights,
    load_training_checkpoint,
    save_training_checkpoint,
)


class DummyScheduler:
    def __init__(self):
        self.step_num = 0
        self.last_lr = 0.1


def make_config():
    return {
        "meta": {"seed": 42, "device": "cpu"},
        "data": {
            "dataset_name": "test",
            "data_root": "first/path",
            "batch_size": 2,
            "img_size": 16,
            "transform_type": "imagenet",
            "num_workers": 0,
            "pin_memory": False,
        },
        "backbone": {"model_type": "tiny"},
        "diffusion": {"model_type": "tiny", "ema_decay": 0.9},
        "optimizer": {
            "optimizer_name": "adamw",
            "scheduler_type": "warmup_cosine",
            "num_epochs": 4,
        },
        "logging": {"save_dir": "ignored"},
    }


class CheckpointingTest(unittest.TestCase):
    def test_round_trip_and_atomic_overwrite(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2)
        model_ema = torch.nn.Linear(3, 2)
        model_ema.load_state_dict(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = DummyScheduler()

        loss = model(torch.ones(1, 3)).sum()
        loss.backward()
        optimizer.step()
        optimizer.param_groups[0]["lr"] = 0.05
        scheduler.step_num = 1
        scheduler.last_lr = 0.05
        expected_weights = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "training_latest.pth"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epoch=0,
                global_step=3,
                best_metric=0.4,
                config=make_config(),
            )
            expected_random_value = torch.rand(1)

            restored_model = torch.nn.Linear(3, 2)
            restored_ema = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.1)
            restored_scheduler = DummyScheduler()
            progress = load_training_checkpoint(
                checkpoint_path,
                model=restored_model,
                model_ema=restored_ema,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                config=make_config(),
            )

            self.assertEqual(progress["next_epoch"], 1)
            self.assertEqual(progress["global_step"], 3)
            self.assertEqual(restored_scheduler.step_num, 1)
            self.assertEqual(restored_scheduler.last_lr, 0.05)
            self.assertTrue(torch.equal(torch.rand(1), expected_random_value))
            for key, value in restored_model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[key]))

            optimizer.zero_grad()
            restored_optimizer.zero_grad()
            model(torch.full((1, 3), 2.0)).sum().backward()
            restored_model(torch.full((1, 3), 2.0)).sum().backward()
            optimizer.step()
            restored_optimizer.step()
            for key, value in restored_model.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))

            save_training_checkpoint(
                checkpoint_path,
                model=restored_model,
                model_ema=restored_ema,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                completed_epoch=1,
                global_step=6,
                best_metric=0.5,
                config=make_config(),
            )
            payload = torch.load(checkpoint_path, weights_only=False)
            self.assertEqual(payload["checkpoint_type"], CHECKPOINT_TYPE)
            self.assertEqual(payload["progress"]["next_epoch"], 2)
            self.assertFalse(Path(f"{checkpoint_path}.tmp").exists())

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "training_latest.pth"
            atomic_torch_save({"epoch": 1}, checkpoint_path)

            def fail_after_partial_write(payload, path):
                Path(path).write_bytes(b"partial")
                raise OSError("simulated interrupted write")

            with mock.patch(
                "src.checkpointing.torch.save", side_effect=fail_after_partial_write
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    atomic_torch_save({"epoch": 2}, checkpoint_path)

            self.assertEqual(
                torch.load(checkpoint_path, weights_only=False)["epoch"], 1
            )
            self.assertFalse(Path(f"{checkpoint_path}.tmp").exists())

    def test_rejects_incompatible_resume_config(self):
        model = torch.nn.Linear(3, 2)
        model_ema = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = DummyScheduler()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "training_latest.pth"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                model_ema=model_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epoch=0,
                global_step=1,
                best_metric=None,
                config=make_config(),
            )
            incompatible = make_config()
            incompatible["data"]["transform_type"] = "default"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    model_ema=model_ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=incompatible,
                )
            with self.assertRaisesRegex(ValueError, "world size"):
                load_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    model_ema=model_ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=make_config(),
                    world_size=2,
                )

    def test_allows_relocated_data_and_loads_legacy_weights(self):
        first_config = make_config()
        relocated_config = make_config()
        relocated_config["data"]["data_root"] = "second/path"
        relocated_config["data"]["num_workers"] = 8
        self.assertEqual(
            config_fingerprint(first_config), config_fingerprint(relocated_config)
        )

        source_model = torch.nn.Linear(3, 2)
        target_model = torch.nn.Linear(3, 2)
        target_ema = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "model_latest.pth"
            torch.save(source_model.state_dict(), weights_path)
            load_initial_weights(
                weights_path, model=target_model, model_ema=target_ema
            )
            for key, value in source_model.state_dict().items():
                self.assertTrue(torch.equal(value, target_model.state_dict()[key]))
                self.assertTrue(torch.equal(value, target_ema.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
