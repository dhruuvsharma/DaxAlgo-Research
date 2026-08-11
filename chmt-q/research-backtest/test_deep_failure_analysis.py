from __future__ import annotations

import unittest

import numpy as np

from deep_failure_analysis import (
    average_ranks,
    binary_sign_metrics,
    matthews_correlation,
    safe_pearson,
    safe_spearman,
    validate_evidence_class,
    within_session_lag_correlation,
)


class FailureAnalysisTests(unittest.TestCase):
    def test_average_ranks_uses_midranks_for_ties(self) -> None:
        actual = average_ranks(np.asarray([20.0, 10.0, 20.0, 40.0]))
        np.testing.assert_array_equal(actual, np.asarray([2.5, 1.0, 2.5, 4.0]))

    def test_correlations_are_exact_for_monotone_examples(self) -> None:
        left = np.asarray([1.0, 2.0, 3.0, 4.0])
        right = np.asarray([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(safe_pearson(left, right), 1.0)
        self.assertAlmostEqual(safe_spearman(left, right), 1.0)

    def test_mcc_and_sign_metrics(self) -> None:
        self.assertAlmostEqual(matthews_correlation(2, 2, 0, 0), 1.0)
        metrics = binary_sign_metrics(
            np.asarray([-2.0, -1.0, 1.0, 2.0]),
            np.asarray([-1.0, 1.0, 1.0, -1.0]),
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["majority_accuracy"], 0.5)
        self.assertEqual(metrics["mcc"], 0.0)

    def test_lag_correlation_does_not_cross_session_boundaries(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        sessions = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
        self.assertAlmostEqual(
            within_session_lag_correlation(values, sessions, lag=1), 1.0
        )

    def test_holdout_or_unknown_evidence_is_rejected(self) -> None:
        validate_evidence_class("untouched_development_forward")
        with self.assertRaises(ValueError):
            validate_evidence_class("sealed_outer_holdout")


if __name__ == "__main__":
    unittest.main()
