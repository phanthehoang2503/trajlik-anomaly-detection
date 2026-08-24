import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_source(name):
    return (ROOT / "src" / name).read_text(encoding="utf-8")


class ResumableMetricLoggingParityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
