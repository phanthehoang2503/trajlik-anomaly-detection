import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(name):
    return (ROOT / "src" / name).read_text(encoding="utf-8")


class ResumableMetricLoggingParityTest(unittest.TestCase):
    def test_single_gpu_training_logging_matches_regular_training(self):
        resumable = read_source("train_resumable.py")
        training_logging = resumable[
            resumable.index('if iteration % config["logging"]["log_interval"]'):
            resumable.index('if (epoch + 1) % config["logging"]["save_interval"]')
        ]
        self.assertIn(
            'print(f"Epoch {epoch}, Iter {iteration}, Loss {loss.item()}")',
            training_logging,
        )
        self.assertIn(
            'wandb.log({"Loss": loss.item(), "LR": learning_rate})',
            training_logging,
        )
        self.assertNotIn("Step", training_logging)
        self.assertNotIn('"global_step"', training_logging)

    def test_single_gpu_eval_logging_matches_regular_training(self):
        regular = read_source("train.py")
        resumable = read_source("train_resumable.py")
        expected_markers = (
            'metrics_dict = evaluate_inv(',
            'wandb.log({"mAD": current_mad})',
            'print(f"mAD: {current_mad} at epoch {epoch}")',
        )
        for marker in expected_markers:
            self.assertIn(marker, regular)
            self.assertIn(marker, resumable)

    def test_distributed_eval_logging_matches_regular_training(self):
        regular = read_source("train_distributed.py")
        resumable = read_source("train_distributed_resumable.py")
        expected_markers = (
            "Evaluating on",
            "Average results:",
            'current_auc = avg_results["I-AUROC"]',
            "AUC:",
            '"I-AUROC"',
            '"I-AP"',
            '"I-F1Max"',
            '"P-AUROC"',
            '"P-AP"',
            '"P-F1Max"',
            '"PRO"',
            '"mAD"',
        )
        for marker in expected_markers:
            self.assertIn(marker, regular)
            self.assertIn(marker, resumable)

        eval_logging = resumable[
            resumable.index('if (epoch + 1) % config["evaluation"]'):
            resumable.index("best_metric_container")
        ]
        self.assertNotIn('"global_step"', eval_logging)

    def test_distributed_training_logging_matches_regular_training(self):
        resumable = read_source("train_distributed_resumable.py")
        training_logging = resumable[
            resumable.index('if iteration % config["logging"]["log_interval"]'):
            resumable.index('if (epoch + 1) % config["logging"]["save_interval"]')
        ]
        for marker in (
            "Epoch %s, Iter %s, Loss %.4f, LR %.6f",
            '"Loss": loss.item()',
            '"LR": learning_rate',
            '"Time/Data [ms]"',
            '"Time/Forward [ms]"',
            '"Time/Backward [ms]"',
            '"Time/Total [ms]"',
        ):
            self.assertIn(marker, training_logging)
        self.assertNotIn("Step %s", training_logging)
        self.assertNotIn('"global_step"', training_logging)


if __name__ == "__main__":
    unittest.main()
