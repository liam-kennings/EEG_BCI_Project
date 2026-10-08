"""Regression tests for the quality availability gate.

Version 1 floored a zero MAD at 1e-6, so a binary indicator that differed from
the reference (no auxiliary channels, no CSD) scored ~1e6 sigma and rejected
every external window (HS-Stroke, Cho2017). Version 2 must not.
"""

from __future__ import annotations

import pickle
import unittest

import numpy as np

from bci_pipeline.translational_decoder import QualityAvailabilityGate


def _reference(rng: np.random.Generator, rows: int = 400) -> np.ndarray:
    amplitude = rng.lognormal(mean=3.0, sigma=0.25, size=(rows, 3))
    bad_fraction = np.where(rng.random(rows) < 0.9, 0.0, rng.uniform(0.03, 0.2, rows))
    auxiliary_available = np.ones(rows)
    csd_available = np.ones(rows)
    return np.column_stack(
        (amplitude, bad_fraction, auxiliary_available, csd_available)
    )


class QualityGateVersionTwoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)
        self.reference = _reference(self.rng)
        self.domains = np.full(len(self.reference), "train")

    def test_missing_indicator_alone_does_not_reject(self) -> None:
        gate = QualityAvailabilityGate(0.99).fit(self.reference, self.domains)
        external = _reference(np.random.default_rng(8), rows=200)
        external[:, 4] = 0.0  # no auxiliary channels
        external[:, 5] = 0.0  # no CSD
        available = gate.available(external, np.full(len(external), "external"))
        self.assertGreater(available.mean(), 0.9)
        self.assertLess(gate.threshold_, 100.0)

    def test_legacy_gate_reproduces_indicator_rejection(self) -> None:
        gate = QualityAvailabilityGate(0.99, legacy=True).fit(
            self.reference, self.domains
        )
        external = self.reference[:50].copy()
        external[:, 4] = 0.0
        self.assertFalse(gate.available(external, np.full(50, "external")).any())

    def test_gate_pickled_before_version_two_keeps_legacy_scoring(self) -> None:
        gate = QualityAvailabilityGate(0.99, legacy=True).fit(
            self.reference, self.domains
        )
        del gate.scoring_version_
        del gate.legacy
        restored = pickle.loads(pickle.dumps(gate))
        self.assertEqual(restored.scoring_version, 1)
        external = self.reference[:50].copy()
        external[:, 5] = 0.0
        self.assertFalse(restored.available(external, np.full(50, "external")).any())

    def test_gross_amplitude_outlier_is_still_rejected(self) -> None:
        gate = QualityAvailabilityGate(0.99).fit(self.reference, self.domains)
        artifact = self.reference[:5].copy()
        artifact[:, :3] *= 50.0
        self.assertFalse(gate.available(artifact, np.full(5, "train")).any())

    def test_mostly_constant_feature_uses_fallback_scale(self) -> None:
        gate = QualityAvailabilityGate(0.99).fit(self.reference, self.domains)
        _, scale, _ = gate.statistics_["train"]
        self.assertGreater(scale[3], 0.0)  # bad-channel fraction has zero MAD
        self.assertEqual(scale[4], 0.0)  # indicator excluded
        self.assertEqual(scale[5], 0.0)

    def test_compatibility_report_names_mismatched_indicators(self) -> None:
        gate = QualityAvailabilityGate(0.99).fit(self.reference, self.domains)
        external = self.reference[:20].copy()
        external[:, 4] = 0.0
        report = gate.compatibility_report(
            external,
            ["a", "b", "c", "bad_fraction", "auxiliary_available", "csd_available"],
        )
        self.assertEqual(report["excluded_features"]["auxiliary_available"]["mismatch_fraction"], 1.0)
        self.assertEqual(report["excluded_features"]["csd_available"]["mismatch_fraction"], 0.0)


class QualityGateWithinRecordingTests(unittest.TestCase):
    def test_rescaled_recording_is_available_but_its_artifact_is_not(self) -> None:
        rng = np.random.default_rng(11)
        reference = _reference(rng, rows=600)
        groups = np.repeat(np.arange(6), 100).astype(str)
        gate = QualityAvailabilityGate(0.99).fit(
            reference,
            np.full(len(reference), "train"),
            recording_groups=groups,
        )
        external = _reference(np.random.default_rng(12), rows=100)
        external[:, :3] *= 20.0  # a new device with a different amplitude scale
        external[0, :3] *= 40.0  # one artifact window
        available = gate.available(
            external,
            np.full(len(external), "external"),
            recording_groups=np.full(len(external), "external_recording"),
        )
        self.assertFalse(available[0])
        self.assertGreater(available[1:].mean(), 0.9)

    def test_within_recording_gate_requires_groups_at_inference(self) -> None:
        reference = _reference(np.random.default_rng(13), rows=100)
        gate = QualityAvailabilityGate(0.99).fit(
            reference, recording_groups=np.zeros(len(reference))
        )
        with self.assertRaises(ValueError):
            gate.available(reference)

    def test_legacy_gate_rejects_within_recording_mode(self) -> None:
        reference = _reference(np.random.default_rng(14), rows=100)
        with self.assertRaises(ValueError):
            QualityAvailabilityGate(0.99, legacy=True).fit(
                reference, recording_groups=np.zeros(len(reference))
            )


if __name__ == "__main__":
    unittest.main()
