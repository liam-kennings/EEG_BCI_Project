"""Known-answer tests for participant-level intent metrics."""

from __future__ import annotations

import unittest

import numpy as np

from bci_pipeline.translational_metrics import (
    balanced_accuracy_chance_threshold,
    paired_participant_difference,
    participant_level_summary,
    score_intent_operating_point,
)


class ChanceThresholdTests(unittest.TestCase):
    def test_fifteen_windows_per_class(self) -> None:
        # 0.5 + 1.645 * 0.5 * sqrt(0.25/15 + 0.25/15)
        self.assertAlmostEqual(balanced_accuracy_chance_threshold(15, 15), 0.6502, places=3)

    def test_threshold_shrinks_with_more_windows(self) -> None:
        self.assertLess(
            balanced_accuracy_chance_threshold(300, 300),
            balanced_accuracy_chance_threshold(20, 20),
        )

    def test_rejects_single_class(self) -> None:
        with self.assertRaises(ValueError):
            balanced_accuracy_chance_threshold(0, 10)


class ParticipantSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        # Participant "big" has 100 windows and is perfect; "small" has 4
        # windows and is at chance. Pooling is dominated by "big".
        big_labels = np.r_[np.zeros(50), np.ones(50)]
        big_probabilities = big_labels.copy()
        small_labels = np.asarray([0, 0, 1, 1])
        small_probabilities = np.asarray([1.0, 0.0, 1.0, 0.0])
        self.labels = np.r_[big_labels, small_labels].astype(int)
        self.probabilities = np.r_[big_probabilities, small_probabilities]
        self.subjects = np.asarray(["big"] * 100 + ["small"] * 4)
        self.datasets = np.asarray(["a"] * 100 + ["b"] * 4)
        self.available = np.ones(len(self.labels), dtype=bool)

    def test_participants_weighted_equally(self) -> None:
        summary = participant_level_summary(
            self.labels, self.probabilities, self.available, 0.5,
            self.subjects, self.datasets, iterations=50,
        )
        self.assertAlmostEqual(summary["mean_balanced_accuracy"], 0.75)
        pooled = score_intent_operating_point(
            self.labels, self.probabilities, self.available, 0.5
        )
        self.assertGreater(pooled["intent_balanced_accuracy"], 0.97)
        self.assertEqual(summary["participants_above_chance"], 1)
        self.assertAlmostEqual(summary["dataset_equal_balanced_accuracy"], 0.75)

    def test_abstained_windows_count_as_rest(self) -> None:
        available = self.available.copy()
        available[50:100] = False  # all of big's intent windows abstain
        summary = participant_level_summary(
            self.labels, self.probabilities, available, 0.5,
            self.subjects, self.datasets, iterations=50,
        )
        big = next(r for r in summary["records"] if r["subject"] == "big")
        self.assertEqual(big["sensitivity"], 0.0)
        self.assertEqual(big["specificity"], 1.0)

    def test_single_class_participant_is_skipped(self) -> None:
        labels = np.r_[self.labels, [1, 1]]
        summary = participant_level_summary(
            labels,
            np.r_[self.probabilities, [1.0, 1.0]],
            np.r_[self.available, [True, True]],
            0.5,
            np.r_[self.subjects, ["only_intent", "only_intent"]],
            np.r_[self.datasets, ["b", "b"]],
            iterations=50,
        )
        self.assertEqual(summary["skipped_single_class_participants"], ["only_intent"])
        self.assertEqual(summary["participants"], 2)

    def test_paired_difference(self) -> None:
        baseline = {"records": [
            {"subject": "x", "balanced_accuracy": 0.6},
            {"subject": "y", "balanced_accuracy": 0.7},
        ]}
        candidate = {"records": [
            {"subject": "x", "balanced_accuracy": 0.7},
            {"subject": "y", "balanced_accuracy": 0.7},
        ]}
        result = paired_participant_difference(baseline, candidate, iterations=50)
        self.assertAlmostEqual(result["mean_balanced_accuracy_change"], 0.05)
        self.assertEqual(result["participants_improved_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
