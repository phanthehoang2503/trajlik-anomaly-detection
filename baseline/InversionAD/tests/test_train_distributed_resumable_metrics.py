import unittest

from src.train_distributed_resumable import METRIC_KEYS, aggregate_metrics


class DistributedMetricAggregationTest(unittest.TestCase):
    def test_aggregates_every_metric_logged_by_regular_training(self):
        first = {key: float(index) for index, key in enumerate(METRIC_KEYS)}
        second = {
            key: float(index + 2) for index, key in enumerate(METRIC_KEYS)
        }

        averages = aggregate_metrics({"first": first, "second": second})

        self.assertEqual(set(averages), set(METRIC_KEYS))
        for index, key in enumerate(METRIC_KEYS):
            self.assertEqual(averages[key], float(index + 1))

    def test_rejects_empty_results(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            aggregate_metrics({})


if __name__ == "__main__":
    unittest.main()
