import unittest

import numpy as np

from evaluation.edge_metrics import evaluate_edge_predictions


class EdgeMetricsTest(unittest.TestCase):
    def test_perfect_prediction_scores_are_one(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[2:6, 4] = 1.0
        pred = gt.copy()

        results = evaluate_edge_predictions(
            predictions={"00001": pred},
            ground_truths={"00001": gt[np.newaxis, ...]},
            thresholds=[1.0, 0.5],
        )

        self.assertAlmostEqual(results["odsF"], 1.0, places=6)
        self.assertAlmostEqual(results["oisF"], 1.0, places=6)
        self.assertAlmostEqual(results["ap"], 1.0, places=6)

    def test_empty_prediction_has_zero_f_score(self):
        gt = np.zeros((8, 8), dtype=np.float32)
        gt[1:7, 3] = 1.0
        pred = np.zeros_like(gt)

        results = evaluate_edge_predictions(
            predictions={"00001": pred},
            ground_truths={"00001": gt[np.newaxis, ...]},
            thresholds=[0.5],
        )

        self.assertAlmostEqual(results["odsF"], 0.0, places=6)
        self.assertAlmostEqual(results["oisF"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
