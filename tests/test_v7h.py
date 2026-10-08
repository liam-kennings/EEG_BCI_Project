from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bci_pipeline.gui_backend import build_v7h_mind_step
from bci_pipeline.mind_dataset import (
    MINDHybridFeatureDataset,
    inspect_mind_dataset,
    synthetic_mind_feature_dataset,
)
from bci_pipeline.mind_download import (
    parse_sciencedb_manifest,
    select_mind_records,
)
from bci_pipeline.v7h import (
    V7H_MODES,
    calibration_and_evaluation_masks,
    evaluate_v7h,
    split_subjects,
)


class MINDDownloadTests(unittest.TestCase):
    def test_manifest_profiles_preserve_root_metadata_and_subject_filter(self) -> None:
        text = "\n".join(
            (
                "https://download.scidb.cn/download?fileId=root&path=/V3/MIND_BIDS/dataset_description.json&fileName=dataset_description.json",
                "https://download.scidb.cn/download?fileId=d2&path=/V3/MIND_BIDS/derivatives/preproc-v1/sub-02/eeg/sub-02_task-mi2d_desc-rest_eeg.mat&fileName=rest.mat",
                "https://download.scidb.cn/download?fileId=d3&path=/V3/MIND_BIDS/derivatives/preproc-v1/sub-03/eeg/sub-03_task-mi2d_desc-rest_eeg.mat&fileName=rest.mat",
                "https://download.scidb.cn/download?fileId=b2&path=/V3/MIND_BIDS/sub-02/eeg/sub-02_task-mi2d_run-01_eeg.vhdr&fileName=eeg.vhdr",
                "https://download.scidb.cn/download?fileId=s2&path=/V3/MIND_BIDS/sourcedata/sub-02/eeg/raw.dat&fileName=raw.dat",
            )
        )
        records = parse_sciencedb_manifest(text)
        derivatives = select_mind_records(
            records,
            profile="derivatives",
            subjects={2},
        )
        self.assertEqual(
            [record.relative_path for record in derivatives],
            [
                "dataset_description.json",
                "derivatives/preproc-v1/sub-02/eeg/sub-02_task-mi2d_desc-rest_eeg.mat",
            ],
        )
        analysis = select_mind_records(records, profile="analysis", subjects={2})
        self.assertEqual(len(analysis), 3)
        self.assertFalse(any("sourcedata" in record.relative_path for record in analysis))

    def test_rejects_unsafe_release_path(self) -> None:
        text = (
            "https://download.scidb.cn/download?fileId=x&"
            "path=/V3/MIND_BIDS/../escape&fileName=escape"
        )
        with self.assertRaises(ValueError):
            parse_sciencedb_manifest(text)


class MINDAdapterTests(unittest.TestCase):
    def test_inspection_requires_all_four_derivative_types_and_excludes_sub13(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "MIND_BIDS" / "derivatives" / "preproc-v1"
            for subject in ("sub-02", "sub-13"):
                eeg = root / subject / "eeg"
                nirs = root / subject / "nirs"
                eeg.mkdir(parents=True)
                nirs.mkdir(parents=True)
                (eeg / f"{subject}_task-mi2d_desc-motorimagery_eeg.mat").touch()
                (eeg / f"{subject}_task-mi2d_desc-rest_eeg.mat").touch()
                (nirs / f"{subject}_task-mi2d_desc-hbo_timeseries.csv").touch()
                (nirs / f"{subject}_task-mi2d_desc-hbr_timeseries.csv").touch()
            incomplete = root / "sub-01" / "nirs"
            incomplete.mkdir(parents=True)
            (incomplete / "sub-01_task-mi2d_desc-hbo_timeseries.csv").touch()
            summary = inspect_mind_dataset(Path(temporary))
            self.assertEqual(summary.paired_subjects, ("sub-02",))
            self.assertEqual(summary.excluded_subjects, ("sub-01", "sub-13"))
            self.assertTrue(summary.adapter_ready)

    def test_feature_cache_round_trip(self) -> None:
        dataset = synthetic_mind_feature_dataset(subjects=9, seed=7)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.npz"
            dataset.save(path)
            restored = MINDHybridFeatureDataset.load(path)
        np.testing.assert_allclose(restored.eeg_features, dataset.eeg_features)
        np.testing.assert_allclose(restored.fnirs_features, dataset.fnirs_features)
        np.testing.assert_array_equal(restored.subjects, dataset.subjects)


class V7HModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = synthetic_mind_feature_dataset(subjects=12, seed=11)

    def test_subject_roles_and_calibration_are_isolated(self) -> None:
        roles = split_subjects(self.dataset.subjects, seed=42)
        role_sets = [set(values) for values in roles.values()]
        self.assertFalse(role_sets[0] & role_sets[1])
        self.assertFalse(role_sets[0] & role_sets[2])
        self.assertFalse(role_sets[1] & role_sets[2])
        calibration, evaluation = calibration_and_evaluation_masks(
            self.dataset,
            roles["locked_internal_test"],
        )
        self.assertFalse(np.any(calibration & evaluation))
        self.assertTrue(np.all(self.dataset.sessions[calibration] == 0))
        self.assertTrue(np.all(self.dataset.sessions[evaluation] > 0))
        for subject in roles["locked_internal_test"]:
            selected = calibration & (self.dataset.subjects == subject)
            self.assertLessEqual(np.sum(selected & (self.dataset.labels == 0)), 10)
            self.assertLessEqual(np.sum(selected & (self.dataset.labels == 1)), 20)

    def test_v7h_runs_all_modes_without_test_subject_training(self) -> None:
        model, report, arrays = evaluate_v7h(
            self.dataset,
            seed=3,
            target_specificity=0.85,
        )
        self.assertEqual(set(report["metrics"]), set(V7H_MODES))
        self.assertEqual(report["endpoint"], "binary_rest_vs_intent")
        self.assertEqual(len(arrays["labels"]), report["dataset"]["test_rows"])
        self.assertFalse(
            set(model.training_subjects_)
            & set(report["roles"]["locked_internal_test"])
        )
        self.assertGreater(
            report["metrics"]["late_fusion_patient_adapted"]["auroc"],
            0.5,
        )

    def test_gui_step_targets_separate_v7h_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = build_v7h_mind_step(
                root / "MIND",
                root / "outputs",
                python_executable="python-test",
                rebuild=True,
            )
        command = " ".join(step.command)
        self.assertIn("51_v7h_mind_benchmark.py", command)
        self.assertIn("--target-specificity 0.93", command)
        self.assertIn("--rebuild-cache", command)
        self.assertTrue(str(step.expected_output).endswith("V7H_MIND_REPORT.md"))


if __name__ == "__main__":
    unittest.main()
