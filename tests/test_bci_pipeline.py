from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bci_pipeline.dataset import (
    ALL_TASK_RUNS,
    BILATERAL_LABEL_NAMES,
    UNILATERAL_LABEL_NAMES,
    map_imagery_annotation,
    map_task_annotation,
    parse_int_list,
)
from bci_pipeline.dataset_registry import dataset_by_key, dataset_catalog
from bci_pipeline.clinical_transfer import (
    CLINICAL_EPOCH_SAMPLES,
    combine_hand_intent_datasets,
    from_harmonized_healthy,
)
from bci_pipeline.classical_eeg_features import (
    FAMILY_ORDER,
    ClassicalIntentDecoder,
    build_classical_feature_cache,
    calibration_robust_alignment,
)
from bci_pipeline.extended_features import (
    derive_asymmetry_features,
    derive_morphology_features,
    extract_phase_features,
    extract_ssd_features,
)
from bci_pipeline.eeg_ssl import (
    BranchScales,
    EEGIntentNetwork,
    MaskedEEGAutoencoder,
    corrupt_eeg_batch,
    load_intent_model_payload,
    model_payload,
)
from bci_pipeline.eeg_bakeoff_models import (
    MODEL_FACTORIES,
    build_bakeoff_model,
    trainable_parameter_count,
)
from bci_pipeline.neural_integration import (
    predict_bakeoff_ensemble,
    predict_mrcp_fbcnet_ensemble,
)
from bci_pipeline.torch_device import resolve_torch_device, torch_device_metadata
from bci_pipeline.torch_layers import temporal_adaptive_average_pool2d
from bci_pipeline.hs_stroke import discover_hs_source_records, resolve_hs_stroke_root
from bci_pipeline.prospective_controller import (
    CueConstrainedController,
    CueControllerConfig,
    FullMontageContaminationVeto,
    controller_metrics,
)
from bci_pipeline.seed_stability import summarize_way_ssl_seeds
from bci_pipeline.feature_atlas import FeatureAtlasDataset
from bci_pipeline.graph_features import (
    MOTOR_GRAPH_CHANNELS,
    fixed_electrode_graph,
    fixed_graph_features,
)
from bci_pipeline.gui_backend import (
    append_hard_negative_manifest,
    audit_recording,
    build_decoder_bakeoff_step,
    build_emg_supervised_transfer_steps,
    build_multisource_privileged_ensemble_step,
    build_openbmi_adapter_step,
    build_pipeline_steps,
    build_translational_v3_step,
    build_translational_v4_neural_step,
    build_translational_v5_patient_adaptive_step,
    build_translational_v6_intent_only_step,
    build_translational_v7_auxiliary_direction_step,
    build_way_eeg_ssl_step,
    build_way_eeg_ssl_seed_stability_step,
    build_way_eeg_gal_transfer_steps,
    inspect_eegmmidb_folder,
    inspect_openbmi_folder,
    load_recording,
)
from bci_pipeline.harmonized_motor_imagery import (
    HARMONIZED_CHANNELS,
    HarmonizedMotorImageryDataset,
    combine_harmonized_datasets,
)
from bci_pipeline.live_sources import FileReplaySource, raw_from_live_chunks
from bci_pipeline.liu_stroke import (
    EXPECTED_TRIALS_PER_SUBJECT,
    _rename_and_type_channels,
    trigger_events,
)
from bci_pipeline.openbmi import (
    OPENBMI_OCCIPITAL_CHANNELS,
    OPENBMI_PAPER_MOTOR_CHANNELS,
    OpenBMIRecording,
    available_openbmi_subjects,
    build_recording_trials,
    mark_openbmi_channel_types,
)
from bci_pipeline.openbmi_benchmark import (
    matrix_exp,
    matrix_invsqrt,
    matrix_log,
    summarize_signal_window,
)
from bci_pipeline.openbmi_controlled import (
    StageProbabilities,
    evaluate_policy,
    select_safety_policy,
    tangent_features,
    tangent_features_from_reference,
)
from bci_pipeline.continuous import (
    ContinuousStream,
    EvidencePolicy,
    command_metrics,
    simulate_stream,
)
from bci_pipeline.metrics import command_policy_metrics, hierarchical_metrics, temporal_metrics
from bci_pipeline.multisource_transfer import (
    DomainFusionRule,
    fit_cross_fitted_stacker,
    logit_component_features,
)
from bci_pipeline.models.eegnet import (
    EEGNetTrainingConfig,
    HierarchicalEEGNet,
    HierarchicalEEGNetModule,
)
from bci_pipeline.models.fbcsp import HierarchicalFBCSP
from bci_pipeline.preprocessing import PreprocessingProfile, is_valid_epoch, preprocess_run
from bci_pipeline.privileged_transfer import (
    EMGConfirmedEEGTransfer,
    PrivilegedSourceDataset,
    emg_window_features,
    robust_domain_align,
)
from bci_pipeline.quality_preprocessing import (
    BranchPreprocessingProfile,
    RecordingQualityError,
    extract_branch_window,
    preprocess_native_branches,
)
from bci_pipeline.research_contract import (
    DEVELOPMENT_OBSERVED_ROLE,
    LOCKED_EXTERNAL_ROLE,
    validate_split_roles,
    write_research_contract,
)
from bci_pipeline.splits import source_target_split, three_way_subject_split
from bci_pipeline.time_frequency_features import (
    MORLET_BANDS,
    MORLET_FREQUENCIES,
    build_morlet_feature_cache,
    morlet_band_power,
    morlet_kernel,
)
from bci_pipeline.translational_dataset import (
    TranslationalWindowDataset,
    summarize_recording_qc,
)
from bci_pipeline.translational_decoder import (
    AuxiliaryIntentBranchEnsemble,
    BranchEnsemble,
    DomainIntentProbabilityCalibrator,
    DomainAwareShrinkageLDAIntentDecoder,
    DomainProbabilityCalibrator,
    IntentBranchEnsemble,
    IntentSpecificFusion,
    QualityAvailabilityGate,
    RapidIntentPatientAdapter,
    RapidAuxiliaryIntentPatientAdapter,
    RapidPatientAdapter,
    StageSpecificFusion,
    patient_adapted_intent_probabilities,
    patient_adapted_auxiliary_intent_probabilities,
    population_intent_features,
    population_stage_features,
)
from bci_pipeline.translational_features import (
    build_external_translational_features,
    build_translational_features,
    mrcp_phase_features,
)
from bci_pipeline.translational_metrics import (
    continuous_false_activations,
    domain_threshold_vector,
    expected_calibration_error,
    score_intent_operating_point,
    score_operating_point,
    select_sensitivity_threshold,
    select_window_threshold,
    subject_bootstrap_confidence_intervals,
    subject_bootstrap_intent_confidence_intervals,
)
from bci_pipeline.way_eeg_gal import (
    TRANSFER_CHANNELS as WAY_TRANSFER_CHANNELS,
    WAY_CHANNEL_MAP,
    _select_transfer_channels,
)
from bci_pipeline.way_eeg_ssl import _window_starts

import mne
import torch


class TorchDeviceTests(unittest.TestCase):
    def test_explicit_cpu_resolution_and_metadata(self) -> None:
        device = resolve_torch_device("cpu")
        metadata = torch_device_metadata("cpu", device)
        self.assertEqual(str(device), "cpu")
        self.assertEqual(metadata["requested"], "cpu")
        self.assertEqual(metadata["resolved"], "cpu")

    def test_auto_always_resolves_to_supported_device(self) -> None:
        self.assertIn(str(resolve_torch_device("auto")), {"cpu", "mps", "cuda"})

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown PyTorch device"):
            resolve_torch_device("quantum")

    def test_mps_safe_temporal_pool_matches_pytorch_reference(self) -> None:
        generator = torch.Generator().manual_seed(19)
        for samples in (8, 12, 36, 40):
            signal = torch.randn((3, 5, 1, samples), generator=generator)
            expected = torch.nn.functional.adaptive_avg_pool2d(signal, (1, 8))
            actual = temporal_adaptive_average_pool2d(signal, 8)
            torch.testing.assert_close(actual, expected)


class DatasetTests(unittest.TestCase):
    def test_parse_int_list_supports_ranges_and_deduplicates(self) -> None:
        self.assertEqual(parse_int_list("1-3, 3, 7"), [1, 2, 3, 7])
        self.assertIsNone(parse_int_list(None))
        with self.assertRaises(ValueError):
            parse_int_list("5-2")

    def test_imagery_annotation_mapping_is_strict(self) -> None:
        self.assertEqual(map_imagery_annotation(" t1 "), 1)
        self.assertIsNone(map_imagery_annotation("left_fist"))

    def test_task_family_resolves_context_dependent_event_names(self) -> None:
        self.assertEqual(map_task_annotation("T1"), 1)
        self.assertEqual(UNILATERAL_LABEL_NAMES[1], "left_fist")
        self.assertEqual(BILATERAL_LABEL_NAMES[1], "both_fists")
        self.assertEqual(UNILATERAL_LABEL_NAMES[2], "right_fist")
        self.assertEqual(BILATERAL_LABEL_NAMES[2], "both_feet")
        self.assertEqual(ALL_TASK_RUNS, tuple(range(3, 15)))


class ExternalDatasetRegistryTests(unittest.TestCase):
    def test_catalog_keys_are_unique_and_clinical_sources_are_present(self) -> None:
        catalog = dataset_catalog()
        self.assertEqual(len({dataset.key for dataset in catalog}), len(catalog))
        self.assertEqual(dataset_by_key("liu2024").cohort, "Acute stroke")
        self.assertIn("23,382", dataset_by_key("hs_stroke_2026").caveat)
        self.assertIn("EMG", dataset_by_key("openbmi").modalities)
        self.assertEqual(dataset_by_key("mrcp_eeg_emg_2025").subjects, 40)
        self.assertIn("EMG", dataset_by_key("stroke_attempt_eeg_emg_2025").modalities)

    def test_liu_adapter_types_reference_eog_and_trigger_channels(self) -> None:
        info = mne.create_info(
            ["C3", "CPz", "HEOL", "HEOR", ""],
            500.0,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(np.zeros((5, 100)), info, verbose=False)
        adjusted = _rename_and_type_channels(raw)
        self.assertEqual(adjusted.ch_names[-1], "STIM")
        self.assertEqual(
            adjusted.get_channel_types(),
            ["eeg", "misc", "eog", "eog", "stim"],
        )

    def test_liu_trigger_parser_preserves_subject_specific_timing(self) -> None:
        samples = EXPECTED_TRIALS_PER_SUBJECT * 8 * 500
        values = np.zeros((1, samples))
        expected_onsets = []
        for trial in range(EXPECTED_TRIALS_PER_SUBJECT):
            for offset, level in ((0.0, 1), (2.0, 2), (6.0, 3)):
                sample = int(round((trial * 8.0 + offset) * 500))
                values[0, sample] = level * 1e-6
                expected_onsets.append(sample / 500.0)
        info = mne.create_info(["STIM"], 500.0, ch_types="stim")
        raw = mne.io.RawArray(values, info, verbose=False)
        events = trigger_events(raw)
        self.assertEqual(len(events), EXPECTED_TRIALS_PER_SUBJECT * 3)
        np.testing.assert_allclose(
            [onset for onset, _ in events],
            expected_onsets,
        )
        self.assertEqual([stage for _, stage in events[:3]], [1, 2, 3])

    def test_openbmi_controlled_montages_and_summary_are_finite(self) -> None:
        self.assertEqual(len(OPENBMI_PAPER_MOTOR_CHANNELS), 20)
        self.assertEqual(len(OPENBMI_OCCIPITAL_CHANNELS), 9)
        signal = np.random.default_rng(41).normal(
            scale=1e-6,
            size=(3, 250),
        )
        features = summarize_signal_window(signal, 100.0)
        self.assertEqual(features.shape, (45,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_openbmi_alignment_uses_calibration_groups(self) -> None:
        rng = np.random.default_rng(43)
        subjects = np.repeat(np.asarray([1, 2]), 12)
        sessions = np.tile(np.repeat(np.asarray([1, 2]), 6), 2)
        recordings = np.tile(np.repeat(np.asarray([0, 1]), 3), 4)
        labels = np.tile(np.asarray([0, 1, 2]), 8)
        matrices = rng.normal(size=(len(labels), 2, 4, 4))
        covariances = matrices @ np.swapaxes(matrices, -1, -2)
        covariances += np.eye(4)[None, None]
        training = (subjects == 1) & (recordings == 0)
        features, metadata = tangent_features(
            covariances,
            training,
            subjects,
            sessions,
            recordings,
            alignment="euclidean",
        )
        self.assertEqual(features.shape, (len(labels), 20))
        self.assertEqual(metadata["alignment_groups"], 4)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_locked_tangent_reference_matches_combined_training_transform(self) -> None:
        rng = np.random.default_rng(44)
        reference_matrix = rng.normal(size=(8, 3, 4, 4))
        external_matrix = rng.normal(size=(5, 3, 4, 4))
        reference = reference_matrix @ np.swapaxes(reference_matrix, -1, -2)
        external = external_matrix @ np.swapaxes(external_matrix, -1, -2)
        reference += np.eye(4)[None, None]
        external += np.eye(4)[None, None]
        combined = np.concatenate((reference, external))
        training = np.arange(len(combined)) < len(reference)
        expected, _ = tangent_features(
            combined,
            training,
            np.arange(len(combined)),
            np.ones(len(combined)),
            np.ones(len(combined)),
            alignment="none",
        )
        actual, metadata = tangent_features_from_reference(external, reference)
        np.testing.assert_allclose(actual, expected[len(reference):], rtol=1e-5, atol=1e-6)
        self.assertEqual(metadata["reference_rows"], len(reference))

    def test_hs_stroke_discovery_accepts_release_or_sourcedata_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            release = Path(temporary) / "30747806"
            eeg = release / "sourcedata" / "sub-01" / "ses-01" / "eeg"
            eeg.mkdir(parents=True)
            path = eeg / "sub-01_ses-01_task-motorimagery_run-01_eeg.mat"
            path.touch()
            self.assertEqual(
                resolve_hs_stroke_root(release),
                (release / "sourcedata").resolve(),
            )
            groups = discover_hs_source_records(release, subjects=("sub-01",))
            self.assertEqual(groups[("sub-01", 1)], (path.resolve(),))

    def test_openbmi_safety_policy_respects_validation_limit(self) -> None:
        labels = np.tile(np.asarray([0, 1, 2]), 20)
        intent = np.linspace(0.0, 1.0, len(labels))
        direction = np.tile(np.asarray([[0.7, 0.3], [0.4, 0.6]]), (30, 1))
        probabilities = StageProbabilities(intent=intent, direction=direction)
        policy = select_safety_policy(labels, probabilities, 0.10)
        metrics = evaluate_policy(labels, probabilities, policy)
        self.assertLessEqual(metrics["false_activation_rate"], 0.10)

    def test_openbmi_matrix_geometry_round_trip(self) -> None:
        rng = np.random.default_rng(47)
        matrix = rng.normal(size=(5, 5))
        matrix = matrix @ matrix.T + np.eye(5)
        whitening = matrix_invsqrt(matrix)
        np.testing.assert_allclose(
            whitening @ matrix @ whitening,
            np.eye(5),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            matrix_exp(matrix_log(matrix)),
            matrix,
            atol=1e-10,
        )

    def test_harmonized_dataset_combination_offsets_openbmi_subjects(self) -> None:
        def synthetic(name: str, subject: int) -> HarmonizedMotorImageryDataset:
            return HarmonizedMotorImageryDataset(
                dataset_name=name,
                epochs=np.zeros(
                    (3, len(HARMONIZED_CHANNELS), 250),
                    dtype=np.float32,
                ),
                labels=np.asarray([0, 1, 2]),
                subjects=np.full(3, subject, dtype=np.int64),
                sessions=np.ones(3, dtype=np.int8),
                recordings=np.asarray([0, 0, 1], dtype=np.int8),
                blocks=np.asarray([4, 4, 8]),
                trial_indices=np.arange(3),
                channel_names=list(HARMONIZED_CHANNELS),
                sfreq=100.0,
                fingerprint=name,
            )

        combined = combine_harmonized_datasets(
            synthetic("physionet", 1),
            synthetic("openbmi", 1),
        )
        combined.validate()
        np.testing.assert_array_equal(
            combined.subjects,
            np.asarray([1, 1, 1, 1001, 1001, 1001]),
        )

    def test_clinical_transfer_crops_and_offsets_healthy_datasets(self) -> None:
        def synthetic(name: str, subject: int) -> HarmonizedMotorImageryDataset:
            return HarmonizedMotorImageryDataset(
                dataset_name=name,
                epochs=np.zeros(
                    (3, len(HARMONIZED_CHANNELS), 250),
                    dtype=np.float32,
                ),
                labels=np.asarray([0, 1, 2]),
                subjects=np.full(3, subject, dtype=np.int64),
                sessions=np.ones(3, dtype=np.int8),
                recordings=np.asarray([0, 0, 1], dtype=np.int8),
                blocks=np.asarray([4, 4, 8]),
                trial_indices=np.arange(3),
                channel_names=list(HARMONIZED_CHANNELS),
                sfreq=100.0,
                fingerprint=name,
            )

        physionet = from_harmonized_healthy(synthetic("physionet", 1))
        openbmi = from_harmonized_healthy(synthetic("openbmi", 1))
        self.assertEqual(physionet.epochs.shape[-1], CLINICAL_EPOCH_SAMPLES)
        combined = combine_hand_intent_datasets(
            (physionet, openbmi),
            subject_offsets=(0, 1000),
        )
        combined.validate()
        np.testing.assert_array_equal(
            combined.subjects,
            np.asarray([1, 1, 1, 1001, 1001, 1001]),
        )
        self.assertEqual(set(combined.dataset_ids), {"physionet", "openbmi"})

    def test_classical_eeg_features_are_finite_and_family_aligned(self) -> None:
        rng = np.random.default_rng(53)
        epochs = rng.normal(scale=1e-6, size=(6, 7, 200)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            features = build_classical_feature_cache(
                epochs,
                100.0,
                ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"),
                "synthetic",
                Path(temporary) / "features.npz",
            )
        features.validate(len(epochs))
        self.assertEqual(tuple(features.family_slices), FAMILY_ORDER)
        self.assertEqual(features.values.shape, (6, 310))
        self.assertTrue(np.all(np.isfinite(features.values)))

    def test_classical_feature_alignment_uses_recording_zero_only(self) -> None:
        rng = np.random.default_rng(59)
        features = rng.normal(size=(24, 5)).astype(np.float32)
        subjects = np.repeat(np.asarray([1, 2]), 12)
        sessions = np.ones(24, dtype=np.int8)
        recordings = np.tile(np.repeat(np.asarray([0, 1]), 6), 2)
        aligned, metadata = calibration_robust_alignment(
            features,
            subjects,
            sessions,
            recordings,
        )
        self.assertEqual(aligned.shape, features.shape)
        self.assertEqual(metadata["groups"], 2)
        self.assertTrue(np.all(np.isfinite(aligned)))

    def test_classical_intent_decoder_returns_finite_probabilities(self) -> None:
        rng = np.random.default_rng(61)
        features = rng.normal(size=(30, 8))
        labels = np.tile(np.asarray([0, 1, 2]), 10)
        decoder = ClassicalIntentDecoder("logistic", 61).fit(
            features,
            labels,
            np.ones(len(labels)),
        )
        probabilities = decoder.predict_intent_probability(features)
        self.assertEqual(probabilities.shape, (len(labels),))
        self.assertTrue(np.all(np.isfinite(probabilities)))

    def test_openbmi_adapter_excludes_emg_from_eeg(self) -> None:
        info = mne.create_info(
            ["C3", "Cz", "C4", "EMG1", "EMG2", "EMG3", "EMG4"],
            1000.0,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(np.zeros((7, 100)), info, verbose=False)
        adjusted = mark_openbmi_channel_types(raw)
        self.assertEqual(
            adjusted.get_channel_types(),
            ["eeg", "eeg", "eeg", "emg", "emg", "emg", "emg"],
        )

    def test_openbmi_uses_trial_type_and_defines_safe_rest_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "sample_events.tsv"
            events.write_text(
                "onset\tduration\ttrial_type\tvalue\tsample\n"
                + "".join(
                    f"{20 + index * 11}.0\t4.0\t"
                    f"{'left_hand' if index % 2 == 0 else 'right_hand'}\t"
                    f"{'99' if index % 2 == 0 else '42'}\t"
                    f"{(20 + index * 11) * 1000}\n"
                    for index in range(100)
                ),
                encoding="ascii",
            )
            recording = OpenBMIRecording(
                subject=1,
                session=1,
                recording="train",
                native_run=1,
                pipeline_run=4,
                bdf_path=root / "sample_eeg.bdf",
                events_path=events,
                channels_path=root / "sample_channels.tsv",
            )
            trials = build_recording_trials(recording)
            self.assertEqual(trials[0].canonical_label, "left_hand")
            self.assertEqual(trials[0].native_value, "99")
            self.assertEqual(trials[0].rest_start_s, 25.0)
            self.assertEqual(trials[0].rest_end_s, 29.0)


class PrivilegedTransferTests(unittest.TestCase):
    def test_emg_confirmed_student_predicts_from_eeg_only(self) -> None:
        rng = np.random.default_rng(17)
        source_labels = np.tile((0, 1), 40)
        source_groups = np.repeat(np.arange(8), 10)
        source_eeg = rng.normal(size=(80, 4))
        source_eeg[:, 0] += 0.8 * source_labels
        source_emg = rng.normal(scale=0.25, size=(80, 3))
        source_emg[:, 0] += 2.5 * source_labels
        target_labels = np.tile((0, 1), 12)
        target_eeg = rng.normal(loc=1.5, scale=1.2, size=(24, 4))
        target_eeg[:, 0] += 0.6 * target_labels
        feature_names = tuple(f"eeg_{index}" for index in range(4))
        source = PrivilegedSourceDataset(
            eeg_features=source_eeg,
            emg_features=source_emg,
            labels=source_labels,
            subject_keys=source_groups,
            premovement_end_seconds=np.where(source_labels == 1, 0.0, -0.5),
            eeg_feature_names=feature_names,
            emg_feature_names=("emg_0", "emg_1", "emg_2"),
            dataset_id="synthetic",
        )
        model = EMGConfirmedEEGTransfer(confidence_floor=0.55).fit(
            source,
            target_eeg,
            target_labels,
            feature_names,
            target_sample_weight=np.linspace(0.5, 1.5, len(target_labels)),
        )
        probabilities = model.predict_intent_probability(target_eeg[:5])
        self.assertEqual(probabilities.shape, (5,))
        self.assertGreater(model.summary_.emg_cross_validated_auroc, 0.8)

    def test_target_training_weights_must_be_positive_and_aligned(self) -> None:
        rng = np.random.default_rng(19)
        labels = np.tile((0, 1), 8)
        names = ("eeg_0", "eeg_1")
        source = PrivilegedSourceDataset(
            eeg_features=rng.normal(size=(16, 2)),
            emg_features=rng.normal(size=(16, 2)),
            labels=labels,
            subject_keys=np.repeat(np.arange(4), 4),
            premovement_end_seconds=np.where(labels == 1, 0.0, -0.5),
            eeg_feature_names=names,
            emg_feature_names=("emg_0", "emg_1"),
            dataset_id="synthetic",
        )
        with self.assertRaisesRegex(ValueError, "finite, positive, and row-aligned"):
            EMGConfirmedEEGTransfer(confidence_floor=0.5).fit(
                source,
                rng.normal(size=(16, 2)),
                labels,
                names,
                target_sample_weight=np.zeros(16),
            )

    def test_source_positive_windows_cannot_extend_past_emg_onset(self) -> None:
        source = PrivilegedSourceDataset(
            eeg_features=np.ones((4, 2)),
            emg_features=np.ones((4, 2)),
            labels=np.asarray((0, 1, 0, 1)),
            subject_keys=np.asarray(("a", "a", "b", "b")),
            premovement_end_seconds=np.asarray((-0.5, 0.1, -0.5, 0.0)),
            eeg_feature_names=("a", "b"),
            emg_feature_names=("c", "d"),
            dataset_id="synthetic",
        )
        with self.assertRaisesRegex(ValueError, "at or before confirmed movement onset"):
            source.validate()

    def test_emg_features_and_domain_alignment_are_finite(self) -> None:
        rng = np.random.default_rng(9)
        values, names = emg_window_features(
            rng.normal(size=(6, 2, 64)),
            ("flexor", "extensor"),
        )
        self.assertEqual(values.shape, (6, 12))
        self.assertEqual(len(names), 12)
        source = rng.normal(loc=-4.0, scale=2.0, size=(30, 3))
        target = rng.normal(loc=3.0, scale=0.5, size=(20, 3))
        aligned = robust_domain_align(source, target)
        np.testing.assert_allclose(
            np.median(aligned, axis=0),
            np.median(target, axis=0),
            atol=1e-8,
        )

    def test_way_spatial_map_reconstructs_shared_transfer_montage(self) -> None:
        names = tuple(sorted({name for values in WAY_CHANNEL_MAP.values() for name in values}))
        signal = np.arange(40 * len(names), dtype=np.float64).reshape(40, len(names))
        selected = _select_transfer_channels(signal, names)
        car = signal * 1e-6 - np.mean(signal * 1e-6, axis=1, keepdims=True)
        indexed = {name: index for index, name in enumerate(names)}
        expected_fc3 = np.mean(car[:, [indexed["FC1"], indexed["FC5"]]], axis=1)
        self.assertEqual(selected.shape, (len(WAY_TRANSFER_CHANNELS), 40))
        np.testing.assert_allclose(selected[0], expected_fc3)
        self.assertTrue(np.all(np.isfinite(selected)))

    def test_multisource_stacker_uses_finite_group_held_probabilities(self) -> None:
        rng = np.random.default_rng(81)
        labels = np.tile(np.asarray((0, 1)), 30)
        groups = np.repeat(np.arange(10), 6).astype(str)
        base = np.clip(0.25 + 0.5 * labels + rng.normal(0, 0.12, len(labels)), 0.01, 0.99)
        components = np.column_stack(
            (
                base,
                np.clip(base + rng.normal(0, 0.04, len(labels)), 0.01, 0.99),
                np.clip(base + rng.normal(0, 0.04, len(labels)), 0.01, 0.99),
            )
        )
        stacker, held_out = fit_cross_fitted_stacker(
            components,
            labels,
            groups,
            seed=81,
        )
        self.assertEqual(logit_component_features(components).shape, components.shape)
        self.assertEqual(held_out.shape, (len(labels),))
        self.assertTrue(np.all(np.isfinite(held_out)))
        self.assertEqual(
            stacker.predict_proba(logit_component_features(components)).shape,
            (len(labels), 2),
        )
        with self.assertRaisesRegex(ValueError, "Unknown fusion method"):
            DomainFusionRule("invalid", 0.5, 0.5, 0.5, 0.5).validate()

    def test_eeg_only_ssl_shapes_masking_and_payload_round_trip(self) -> None:
        torch.manual_seed(83)
        signal = torch.randn(4, 14, 200)
        autoencoder = MaskedEEGAutoencoder()
        corrupted, mask = corrupt_eeg_batch(signal)
        self.assertEqual(tuple(autoencoder(corrupted).shape), (4, 14, 200))
        self.assertEqual(tuple(mask.shape), (4, 14, 200))
        self.assertTrue(torch.any(mask))
        model = EEGIntentNetwork(autoencoder.backbone)
        model.eval()
        self.assertEqual(tuple(model(signal).shape), (4, 2))
        scales = BranchScales(1e-6, 2e-6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "decoder.pt"
            torch.save(
                model_payload(model, scales, {"physionet": 0.6}, ("C3", "C4")),
                path,
            )
            restored, restored_scales, thresholds, channels = load_intent_model_payload(path)
        np.testing.assert_allclose(
            torch.softmax(restored(signal), dim=1).detach().numpy(),
            torch.softmax(model(signal), dim=1).detach().numpy(),
        )
        self.assertEqual(restored_scales, scales)
        self.assertEqual(thresholds, {"physionet": 0.6})
        self.assertEqual(channels, ("C3", "C4"))

    def test_way_ssl_fixed_grid_respects_guard_and_series_cap(self) -> None:
        starts = _window_starts(
            20_000,
            stride_seconds=2.0,
            guard_seconds=3.0,
            max_windows=12,
        )
        self.assertEqual(len(starts), 12)
        self.assertGreaterEqual(int(starts[0]), 300)
        self.assertLessEqual(int(starts[-1]) + 200, 20_000 - 300)


class GuiBackendTests(unittest.TestCase):
    def test_emg_supervised_gui_workflow_prepares_then_reuses_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            steps = build_emg_supervised_transfer_steps(
                root / "source",
                root / "output",
                python_executable="python-test",
            )
            self.assertEqual(len(steps), 2)
            self.assertIn("33_prepare_mrcp_eeg_emg_transfer.py", " ".join(steps[0].command))
            output = root / "output" / "emg_supervised_transfer"
            output.mkdir(parents=True)
            (output / "mrcp_eeg_emg_source_features.npz").touch()
            (output / "pure_eeg_target_features.npz").touch()
            cached = build_emg_supervised_transfer_steps(
                root / "source",
                root / "output",
                python_executable="python-test",
            )
            self.assertEqual(len(cached), 1)
            self.assertIn("32_emg_supervised_eeg_transfer.py", " ".join(cached[0].command))

    def test_way_gui_workflow_prepares_then_trains_eeg_only_student(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            steps = build_way_eeg_gal_transfer_steps(
                root / "way",
                root / "output",
                python_executable="python-test",
            )
            self.assertEqual(len(steps), 2)
            self.assertIn("35_prepare_way_eeg_gal_transfer.py", " ".join(steps[0].command))
            self.assertIn("32_emg_supervised_eeg_transfer.py", " ".join(steps[1].command))

    def test_multisource_gui_workflow_uses_both_standardized_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            step = build_multisource_privileged_ensemble_step(
                Path(temporary),
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("36_multisource_privileged_ensemble.py", command)
            self.assertIn("mrcp_eeg_emg_source_features.npz", command)
            self.assertIn("way_eeg_gal_source_features.npz", command)

    def test_way_ssl_gui_workflow_uses_unlabeled_cache_and_raw_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            step = build_way_eeg_ssl_step(
                Path(temporary) / "way",
                Path(temporary) / "output",
                rebuild=True,
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("37_way_eeg_only_ssl_transfer.py", command)
            self.assertIn("way_unlabeled_eeg_windows.npz", command)
            self.assertIn("translational_windows_v3.npz", command)
            self.assertIn("--rebuild-way-cache", command)
            self.assertIn("--device auto", command)

    def test_way_ssl_seed_stability_gui_workflow_uses_five_fixed_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            step = build_way_eeg_ssl_seed_stability_step(
                Path(temporary) / "way",
                Path(temporary) / "output",
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("38_way_eeg_ssl_seed_stability.py", command)
            self.assertIn("--seeds 42 43 44 45 46", command)
            self.assertIn("--device auto", command)
            self.assertTrue(
                str(step.expected_output).endswith("WAY_EEG_SSL_SEED_STABILITY_REPORT.md")
            )
            forced = build_way_eeg_ssl_seed_stability_step(
                Path(temporary) / "way",
                Path(temporary) / "output",
                device="mps",
                rebuild=True,
                python_executable="python-test",
            )
            self.assertIn("--device mps", " ".join(forced.command))
            self.assertIn("--force", forced.command)

    def test_decoder_bakeoff_gui_workflow_uses_five_fixed_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            step = build_decoder_bakeoff_step(
                Path(temporary) / "output",
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("39_eeg_decoder_model_bakeoff.py", command)
            self.assertIn("--seeds 42 43 44 45 46", command)
            self.assertIn("--device auto", command)
            self.assertTrue(
                str(step.expected_output).endswith("EEG_DECODER_MODEL_BAKEOFF_REPORT.md")
            )
            forced = build_decoder_bakeoff_step(
                Path(temporary) / "output",
                device="mps",
                rebuild=True,
                python_executable="python-test",
            )
            self.assertIn("--device mps", " ".join(forced.command))
            self.assertIn("--force", forced.command)

    def test_translational_v4_gui_workflow_integrates_neural_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = build_translational_v4_neural_step(
                root / "physionet",
                root / "openbmi",
                root / "liu",
                root / "output",
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("--neural-bakeoff-dir", command)
            self.assertIn("--neural-seeds 42 43 44 45 46", command)
            self.assertIn("--neural-device auto", command)
            self.assertTrue(
                str(step.expected_output).endswith("TRANSLATIONAL_V4_NEURAL_REPORT.md")
            )

    def test_translational_v5_uses_gpu_bakeoff_and_dual_neural_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = build_translational_v5_patient_adaptive_step(
                root / "physionet",
                root / "openbmi",
                root / "liu",
                root / "output",
                neural_device="mps",
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("output/eeg_decoder_model_bakeoff", command)
            self.assertIn(
                "--neural-architectures mrcp_fbcnet shallow_convnet",
                command,
            )
            self.assertIn("--neural-device mps", command)
            self.assertTrue(
                str(step.expected_output).endswith(
                    "TRANSLATIONAL_V5_PATIENT_ADAPTIVE_REPORT.md"
                )
            )

    def test_translational_v6_is_binary_and_uses_three_confirmations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = build_translational_v6_intent_only_step(
                root / "physionet",
                root / "openbmi",
                root / "liu",
                root / "output",
                neural_device="mps",
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("--intent-only", command)
            self.assertIn("--target-intent-sensitivity 0.7", command)
            self.assertIn("--temporal-confirmations 3", command)
            self.assertIn("--neural-device mps", command)
            self.assertTrue(
                str(step.expected_output).endswith(
                    "TRANSLATIONAL_V6_INTENT_ONLY_REPORT.md"
                )
            )

    def test_translational_v7_uses_auxiliary_direction_with_binary_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = build_translational_v7_auxiliary_direction_step(
                root / "physionet",
                root / "openbmi",
                root / "liu",
                root / "output",
                neural_device="mps",
                python_executable="python-test",
            )
            command = " ".join(step.command)
            self.assertIn("--auxiliary-direction", command)
            self.assertNotIn("--intent-only", command)
            self.assertIn("--target-intent-sensitivity 0.7", command)
            self.assertIn("--temporal-confirmations 3", command)
            self.assertNotIn("--disable-v7-graph-lda", command)
            self.assertIn("graph/LDA-enhanced", step.name)
            self.assertIn(
                "translational_v7_graph_lda",
                str(step.expected_output),
            )
            self.assertTrue(
                str(step.expected_output).endswith(
                    "TRANSLATIONAL_V7_AUXILIARY_DIRECTION_REPORT.md"
                )
            )

    def test_way_ssl_seed_summary_preserves_paired_control(self) -> None:
        payloads = []
        for index in range(5):
            delta = 0.01 * (index + 1)
            random_result = {
                "balanced_accuracy": 0.50,
                "sensitivity": 0.20,
                "specificity": 0.90,
                "auroc": 0.60,
                "auprc": 0.55,
            }
            ssl_result = {
                "balanced_accuracy": 0.50 + delta,
                "sensitivity": 0.20 + delta,
                "specificity": 0.91,
                "auroc": 0.60 + delta,
                "auprc": 0.55 + delta,
            }
            participant_records = [
                {
                    "dataset_id": "openbmi",
                    "subject_key": "subject-1",
                    "balanced_accuracy_delta": delta,
                },
                {
                    "dataset_id": "physionet",
                    "subject_key": "subject-2",
                    "balanced_accuracy_delta": delta / 2.0,
                },
            ]
            payloads.append(
                {
                    "inputs": {"target": "fixed"},
                    "source_audit": {"windows": 100, "accepted_series": 10},
                    "source_branch_scales": {"mrcp": 1.0, "smr": 2.0},
                    "training": {
                        "development_rows": 60,
                        "controller_rows": 20,
                        "development_observed_rows": 20,
                        "pretrain_epochs": 8,
                        "fine_tune_epochs": 10,
                        "batch_size": 32,
                    },
                    "leakage_boundary": {
                        "way_labels_used": False,
                        "way_emg_used": False,
                    },
                    "evaluation": {
                        "classical_target_only": random_result,
                        "random_cnn": random_result,
                        "way_ssl": ssl_result,
                        "way_privileged": random_result,
                    },
                    "evaluation_by_dataset": {
                        domain: {
                            "random_cnn": random_result,
                            "way_ssl": ssl_result,
                        }
                        for domain in ("openbmi", "physionet")
                    },
                    "participant_analysis": {
                        "mean_balanced_accuracy_delta": 0.75 * delta,
                        "records": participant_records,
                    },
                }
            )
        summary = summarize_way_ssl_seeds(
            payloads,
            [42, 43, 44, 45, 46],
            participant_bootstrap_iterations=100,
        )
        self.assertAlmostEqual(
            summary["paired_ssl_minus_random"]["balanced_accuracy"]["mean"],
            0.03,
        )
        self.assertTrue(
            summary["predeclared_criteria"]["all_predeclared_criteria_met"]
        )
        self.assertEqual(
            len(
                summary["domain_model_summary"]["openbmi"]["random_cnn"][
                    "balanced_accuracy"
                ]["per_seed"]
            ),
            5,
        )

    def test_csv_time_column_infers_sampling_and_voltage_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recording.csv"
            path.write_text(
                "time,C3,C4\n"
                "0.000,10,-10\n"
                "0.005,20,-20\n"
                "0.010,30,-30\n",
                encoding="ascii",
            )
            raw = load_recording(path, sfreq=None, unit="uV")
            self.assertEqual(raw.ch_names, ["C3", "C4"])
            self.assertAlmostEqual(float(raw.info["sfreq"]), 200.0)
            np.testing.assert_allclose(
                raw.get_data()[:, 0],
                np.asarray([10e-6, -10e-6]),
            )

    def test_file_replay_and_live_chunk_conversion_preserve_data(self) -> None:
        info = mne.create_info(["C3", "C4"], 100.0, ch_types="eeg")
        data = np.arange(40, dtype=float).reshape(2, 20) * 1e-6
        raw = mne.io.RawArray(data, info, verbose=False)
        replay = FileReplaySource(raw, chunk_seconds=0.05)
        chunks = [replay.pull_chunk().data for _ in range(4)]
        reconstructed = raw_from_live_chunks(chunks, 100.0, ["C3", "C4"])
        np.testing.assert_allclose(reconstructed.get_data(), data)
        self.assertTrue(replay.finished)

    def test_pipeline_builder_uses_selected_data_and_output_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            output = root / "output"
            data.mkdir()
            steps = build_pipeline_steps(data, output, rebuild=True)
            self.assertEqual([step.key for step in steps], ["atlas", "extended", "ablation"])
            self.assertIn(str(data.resolve()), steps[0].command)
            self.assertTrue(
                str(steps[-1].expected_output).startswith(str(output.resolve()))
            )
            self.assertIn("--rebuild-cache", steps[0].command)

    def test_dataset_folder_inspection_detects_subject_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = root / "S001"
            subject.mkdir()
            for run in range(1, 4):
                (subject / f"S001R{run:02d}.edf").touch()
            summary = inspect_eegmmidb_folder(root)
            self.assertTrue(summary.looks_like_eegmmidb)
            self.assertEqual(summary.subjects, 1)
            self.assertEqual(summary.edf_files, 3)

    def test_openbmi_folder_inspection_and_pipeline_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "nm000338"
            root.mkdir()
            (root / "dataset_description.json").write_text(
                '{"DatasetDOI": "10.82901/nemar.nm000338"}',
                encoding="ascii",
            )
            (root / "participants.tsv").write_text(
                "participant_id\nsub-1\n",
                encoding="ascii",
            )
            for session in (1, 2):
                folder = root / "sub-1" / f"ses-{session}" / "eeg"
                folder.mkdir(parents=True)
                for recording, run in (("train", 1), ("test", 4)):
                    name = (
                        f"sub-1_ses-{session}_task-imagery_run-{run}_"
                        f"recording-{recording}_eeg.bdf"
                    )
                    (folder / name).touch()
            self.assertEqual(available_openbmi_subjects(root), [1])
            summary = inspect_openbmi_folder(root)
            self.assertTrue(summary.adapter_ready)
            self.assertEqual(summary.bdf_files, 4)
            step = build_openbmi_adapter_step(root, Path(temporary) / "out")
            self.assertIn("--build-epoch-cache", step.command)

    def test_recording_audit_writes_figure_and_json(self) -> None:
        sfreq = 160.0
        time = np.arange(640) / sfreq
        data = np.vstack(
            (
                10e-6 * np.sin(2.0 * np.pi * 10.0 * time),
                8e-6 * np.sin(2.0 * np.pi * 12.0 * time + 0.2),
                5e-6 * np.sin(2.0 * np.pi * 20.0 * time),
            )
        )
        info = mne.create_info(["C3", "Cz", "C4"], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        with tempfile.TemporaryDirectory() as temporary:
            audit = audit_recording(
                raw,
                temporary,
                source_path="synthetic.edf",
                plot_seconds=2.0,
            )
            self.assertTrue(audit.figure_path.is_file())
            self.assertTrue(audit.report_path.is_file())
            self.assertGreater(audit.band_metrics["mu_fraction_4_35"], 0.0)

    def test_v3_builder_and_no_control_manifest_preserve_selected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            physionet = root / "physionet"
            openbmi = root / "openbmi"
            liu = root / "liu"
            output = root / "output"
            for path in (physionet, openbmi, liu):
                path.mkdir()
            recording = root / "idle.edf"
            recording.touch()
            manifest = append_hard_negative_manifest(
                root / "hard_negatives.json",
                recording,
                subject_key="healthy:001",
                session=2,
            )
            payload = __import__("json").loads(manifest.read_text(encoding="ascii"))
            self.assertEqual(len(payload["recordings"]), 1)
            step = build_translational_v3_step(
                physionet,
                openbmi,
                liu,
                output,
                hard_negative_manifest=manifest,
                rebuild=True,
            )
            self.assertEqual(step.key, "translational_v3_calibrated")
            self.assertIn(str(manifest.resolve()), step.command)
            self.assertIn("--rebuild-cache", step.command)
            self.assertIn("--target-intent-sensitivity", step.command)
            self.assertIn("--temporal-confirmations", step.command)
            self.assertEqual(
                step.expected_output.parent,
                (output / "translational_v3_calibrated").resolve(),
            )

            legacy_cache = (
                output
                / "translational_v3"
                / "translational_windows_v3.npz"
            )
            legacy_cache.parent.mkdir(parents=True)
            legacy_cache.touch()
            cached_step = build_translational_v3_step(
                physionet,
                openbmi,
                liu,
                output,
            )
            cache_argument = cached_step.command.index("--cache-path") + 1
            self.assertEqual(
                cached_step.command[cache_argument],
                str(legacy_cache.resolve()),
            )


class SplitTests(unittest.TestCase):
    def test_source_target_split_has_no_subject_overlap(self) -> None:
        subjects = np.repeat(np.arange(1, 11), 3)
        labels = np.tile(np.arange(3), 10)
        source, target = source_target_split(labels, subjects, 0.2, 42)
        self.assertFalse(set(source) & set(target))
        self.assertEqual(set(source) | set(target), set(range(1, 11)))

    def test_three_way_split_is_reproducible_and_disjoint(self) -> None:
        first = three_way_subject_split(range(1, 31), random_seed=11)
        second = three_way_subject_split(range(1, 31), random_seed=11)
        self.assertEqual(first, second)
        first.validate()
        combined = first.development + first.controller_validation + first.final_target
        self.assertEqual(set(combined), set(range(1, 31)))


class MetricTests(unittest.TestCase):
    def test_hierarchical_false_activation_and_miss_rates(self) -> None:
        labels = np.asarray([0, 0, 1, 2])
        intent = np.asarray([0.2, 0.8, 0.7, 0.4])
        direction = np.asarray([[0.5, 0.5], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]])
        metrics = hierarchical_metrics(labels, intent, direction, 0.5)
        self.assertEqual(metrics["false_activation_rate"], 0.5)
        self.assertEqual(metrics["missed_intent_rate"], 0.5)

    def test_temporal_false_activations_per_minute_uses_duration(self) -> None:
        metrics = temporal_metrics(
            np.asarray([0, 0, 1]),
            np.asarray([1, 0, 1]),
            [1.0, None, 2.0],
            np.zeros(3),
            trial_seconds=3.0,
        )
        self.assertEqual(metrics["false_activation_rate"], 0.5)
        self.assertEqual(metrics["false_activations_per_minute"], 10.0)

    def test_command_policy_rejects_ambiguous_direction(self) -> None:
        labels = np.asarray([0, 1, 2])
        intent = np.asarray([0.9, 0.9, 0.9])
        direction = np.asarray([[0.51, 0.49], [0.8, 0.2], [0.45, 0.55]])
        metrics = command_policy_metrics(labels, intent, direction, 0.5, 0.2)
        self.assertEqual(metrics["false_activation_rate"], 0.0)
        self.assertEqual(metrics["missed_intent_rate"], 0.5)

    def test_continuous_controller_counts_command_events_per_rest_minute(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 0])
        values = np.asarray([0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1])
        stream = ContinuousStream(
            subject=1,
            runs=np.full(len(labels), 12),
            times=np.arange(len(labels), dtype=float),
            labels=labels,
            segments=np.asarray([0, 0, 0, 1, 1, 1, 2, 2]),
            segment_onsets=np.asarray([0, 0, 0, 3, 3, 3, 6, 6], dtype=float),
            erd_global_intent=values,
            erd_local_intent=values,
            riemann_global_intent=values,
            riemann_local_intent=values,
            erd_global_direction=np.full(len(labels), 0.8),
            erd_local_direction=np.full(len(labels), 0.8),
            riemann_global_direction=np.full(len(labels), 0.8),
            riemann_local_direction=np.full(len(labels), 0.8),
            artifact_rejected=np.zeros(len(labels), dtype=bool),
            calibration_mask=np.asarray([True, False, False, False, False, False, False, False]),
            step_seconds=1.0,
        )
        policy = EvidencePolicy(0.5, 0.5, 0.8, 0.0, 0.5, 0.0, 1.0, 0.0)
        simulation = simulate_stream(stream, policy)
        metrics = command_metrics(stream, simulation)
        self.assertGreaterEqual(metrics["false_activation_events"], 1)
        self.assertEqual(metrics["detected_intent_segments"], 1)
        self.assertGreater(metrics["false_activations_per_minute"], 0.0)

    def test_artifact_abstention_prevents_high_probability_command(self) -> None:
        labels = np.asarray([0, 0])
        values = np.asarray([0.1, 0.99])
        stream = ContinuousStream(
            subject=1,
            runs=np.asarray([12, 12]),
            times=np.asarray([1.0, 2.0]),
            labels=labels,
            segments=np.asarray([0, 0]),
            segment_onsets=np.asarray([0.0, 0.0]),
            erd_global_intent=values,
            erd_local_intent=values,
            riemann_global_intent=values,
            riemann_local_intent=values,
            erd_global_direction=np.full(2, 0.8),
            erd_local_direction=np.full(2, 0.8),
            riemann_global_direction=np.full(2, 0.8),
            riemann_local_direction=np.full(2, 0.8),
            artifact_rejected=np.asarray([False, True]),
            calibration_mask=np.asarray([True, False]),
            step_seconds=1.0,
        )
        policy = EvidencePolicy(0.5, 0.5, 0.8, 0.0, 0.5, 0.0, 1.0, 0.0)
        self.assertEqual(len(simulate_stream(stream, policy, True)["events"]), 0)
        self.assertEqual(len(simulate_stream(stream, policy, False)["events"]), 1)


class ProspectiveControllerTests(unittest.TestCase):
    def test_cue_arms_but_eeg_alone_supplies_confirmed_direction(self) -> None:
        controller = CueConstrainedController(
            CueControllerConfig(
                intent_threshold=0.60,
                confirmation_windows=2,
                arming_seconds=4.0,
                refractory_seconds=1.0,
            )
        )
        controller.step(0.0, 0.90, (0.80, 0.20))
        self.assertFalse(
            any(event.kind == "confirmed_command" for event in controller.events)
        )
        controller.arm("left_001", 1.0, expected_label=1)
        controller.step(1.5, 0.85, (0.80, 0.20))
        controller.step(
            2.0,
            0.95,
            (0.80, 0.20),
            contamination_veto=True,
        )
        controller.step(2.5, 0.86, (0.75, 0.25))
        controller.step(3.0, 0.88, (0.78, 0.22))
        commands = [
            event for event in controller.events if event.kind == "confirmed_command"
        ]
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].decoded_label, 1)
        metrics = controller_metrics(controller.events, session_duration_seconds=10.0)
        self.assertEqual(metrics["motor_command_sensitivity"], 1.0)
        self.assertEqual(metrics["contamination_vetoes"], 1)
        self.assertEqual(metrics["latent_unarmed_candidates"], 1)

    def test_expired_armed_window_is_recorded_as_no_decision(self) -> None:
        controller = CueConstrainedController(
            CueControllerConfig(arming_seconds=1.0)
        )
        controller.arm("right_missed", 0.0, expected_label=2)
        controller.step(1.1, 0.20, (0.40, 0.60))
        self.assertEqual(
            sum(event.kind == "no_decision" for event in controller.events),
            1,
        )

    def test_full_montage_veto_detects_large_burst(self) -> None:
        rng = np.random.default_rng(101)
        clean = rng.normal(scale=5e-6, size=(24, 8, 200))
        channel_names = ("Fp1", "Fp2", "F3", "F4", "C3", "C4", "T7", "T8")
        veto = FullMontageContaminationVeto().fit(
            clean,
            100.0,
            channel_names,
        )
        test = clean[:2].copy()
        test[1, 0, 90:110] += 1e-3
        predictions = veto.predict(test, 100.0, channel_names)
        self.assertFalse(predictions[0])
        self.assertTrue(predictions[1])

    def test_mrcp_phase_features_are_finite_and_named(self) -> None:
        rng = np.random.default_rng(103)
        epochs = rng.normal(size=(4, 7, 200)).astype(np.float32)
        names = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
        features, feature_names = mrcp_phase_features(epochs, 100.0, names)
        self.assertEqual(features.shape[0], len(epochs))
        self.assertEqual(features.shape[1], len(feature_names))
        self.assertTrue(np.all(np.isfinite(features)))


class PreprocessingTests(unittest.TestCase):
    def test_epoch_validation_rejects_nonfinite_and_large_amplitude(self) -> None:
        clean = np.zeros((2, 100))
        self.assertTrue(is_valid_epoch(clean, peak_to_peak_uv=250.0))
        nonfinite = clean.copy()
        nonfinite[0, 0] = np.nan
        self.assertFalse(is_valid_epoch(nonfinite, peak_to_peak_uv=250.0))
        large = clean.copy()
        large[0, 0] = 300e-6
        self.assertFalse(is_valid_epoch(large, peak_to_peak_uv=250.0))

    def test_preprocessing_preserves_requested_channel_order(self) -> None:
        sfreq = 160.0
        rng = np.random.default_rng(4)
        info = mne.create_info(["C3.", "Cz.", "C4."], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(rng.normal(scale=5e-6, size=(3, 800)), info, verbose=False)
        profile = PreprocessingProfile(sfreq=sfreq, l_freq=4.0, h_freq=35.0)
        processed = preprocess_run(raw, profile, ["C4", "C3"])
        self.assertEqual(processed.ch_names, ["C4.", "C3."])
        self.assertEqual(processed.get_data().shape, (2, 800))

    def test_native_branch_preprocessing_repairs_before_reference(self) -> None:
        sfreq = 160.0
        channels = (
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "Fz",
            "FC3",
            "FC4",
            "C3",
            "Cz",
            "C4",
            "CP3",
            "CP4",
            "P3",
            "P4",
            "O1",
            "O2",
        )
        rng = np.random.default_rng(71)
        common = rng.normal(scale=5e-6, size=(1, 1600))
        data = common + rng.normal(scale=2e-6, size=(len(channels), 1600))
        data[0] = 0.0
        info = mne.create_info(channels, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        branches = preprocess_native_branches(
            raw,
            BranchPreprocessingProfile(compute_csd=True),
        )
        self.assertTrue(branches.recording_qc["qc_before_reference"])
        self.assertIn("Fp1", branches.recording_qc["bad_channels"])
        self.assertEqual(branches.mrcp.get_data().shape, (7, 1000))
        self.assertEqual(branches.smr_car.get_data().shape, (7, 1000))
        window = extract_branch_window(branches, 1.0, 2.0)
        self.assertEqual(window["mrcp"].shape, (7, 200))

    def test_native_branch_rejection_preserves_qc_details(self) -> None:
        sfreq = 160.0
        channels = (
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "Fz",
            "FC3",
            "FC4",
            "C3",
            "Cz",
            "C4",
            "CP3",
            "CP4",
            "P3",
            "P4",
            "O1",
            "O2",
        )
        rng = np.random.default_rng(72)
        data = rng.normal(scale=5e-6, size=(len(channels), 1600))
        data[:5] = 0.0
        info = mne.create_info(channels, sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        with self.assertRaises(RecordingQualityError) as context:
            preprocess_native_branches(
                raw,
                BranchPreprocessingProfile(compute_csd=False),
            )
        self.assertGreater(
            context.exception.recording_qc["bad_channel_fraction"],
            0.25,
        )


class TranslationalV3Tests(unittest.TestCase):
    def test_fixed_electrode_graph_is_symmetric_connected_and_order_safe(self) -> None:
        graph = fixed_electrode_graph(MOTOR_GRAPH_CHANNELS)
        np.testing.assert_allclose(graph.adjacency, graph.adjacency.T)
        np.testing.assert_allclose(np.sum(graph.laplacian, axis=1), 0.0, atol=1e-12)
        self.assertEqual(int(np.sum(graph.eigenvalues < 1e-10)), 1)
        self.assertTrue(np.all(np.diff(graph.eigenvalues) >= -1e-12))

        rng = np.random.default_rng(201)
        mrcp = rng.normal(size=(8, 7, 200))
        smr = rng.normal(size=(8, 7, 200))
        filter_bank = (("mu", 8.0, 13.0), ("beta", 13.0, 30.0))
        first, first_names, metadata = fixed_graph_features(
            mrcp,
            smr,
            100.0,
            MOTOR_GRAPH_CHANNELS,
            filter_bank,
            mrcp_time_edges=(0.0, 0.5, 1.0, 1.5, 2.0),
            smr_time_edges=(0.0, 1.0, 2.0),
        )
        permutation = np.asarray([4, 0, 6, 2, 1, 5, 3])
        permuted_names = tuple(MOTOR_GRAPH_CHANNELS[index] for index in permutation)
        second, second_names, _ = fixed_graph_features(
            mrcp[:, permutation],
            smr[:, permutation],
            100.0,
            permuted_names,
            filter_bank,
            mrcp_time_edges=(0.0, 0.5, 1.0, 1.5, 2.0),
            smr_time_edges=(0.0, 1.0, 2.0),
        )
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
        self.assertEqual(first_names, second_names)
        self.assertFalse(metadata["edge_parameters_learned_from_eeg"])
        self.assertTrue(np.all(np.isfinite(first)))

    def test_domain_aware_shrinkage_lda_returns_finite_intent_probabilities(self) -> None:
        rng = np.random.default_rng(202)
        domains = np.repeat(np.asarray(["a", "b", "c"]), 60)
        labels = np.tile(np.repeat(np.asarray([0, 1]), 30), 3)
        features = rng.normal(size=(len(labels), 12))
        features[:, :4] += (labels != 0)[:, None] * 0.8
        features[domains == "b", 4:8] += 0.4
        model = DomainAwareShrinkageLDAIntentDecoder(domain_weight=0.5).fit(
            features,
            labels,
            domains,
        )
        probabilities = model.predict_intent_probabilities(features, domains)
        self.assertEqual(probabilities.shape, (len(labels),))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertGreater(np.mean(probabilities[labels == 1]), np.mean(probabilities[labels == 0]))
        self.assertGreaterEqual(model.global_.shrinkage_, 0.0)
        self.assertLessEqual(model.global_.shrinkage_, 1.0)

    @staticmethod
    def _dataset() -> TranslationalWindowDataset:
        rng = np.random.default_rng(73)
        labels = np.tile(np.asarray([0, 1, 2]), 12)
        dataset_ids = np.repeat(np.asarray(["a", "b"]), 18)
        subject_keys = np.repeat(
            np.asarray([f"S{index:02d}" for index in range(12)]),
            3,
        )
        sessions = np.ones(len(labels), dtype=np.int16)
        recordings = np.tile(np.asarray([0, 1, 1]), 12)
        time = np.arange(200) / 100.0
        smr = rng.normal(scale=2e-6, size=(len(labels), 7, 200))
        smr[labels != 0] += (
            2e-6
            * np.sin(2 * np.pi * 10 * time)[None, None, :]
        )
        mrcp = rng.normal(scale=1e-6, size=(len(labels), 7, 200))
        mrcp[labels != 0] -= np.linspace(0, 3e-6, 200)[None, None, :]
        quality = rng.normal(size=(len(labels), 11)).astype(np.float32)
        dataset = TranslationalWindowDataset(
            mrcp_epochs=mrcp.astype(np.float32),
            smr_epochs=smr.astype(np.float32),
            quality_features=quality,
            labels=labels,
            dataset_ids=dataset_ids,
            subject_keys=subject_keys,
            sessions=sessions,
            recordings=recordings,
            trial_indices=np.arange(len(labels)),
            negative_kinds=np.where(
                labels == 0,
                "protocol_rest",
                "not_applicable",
            ),
            cue_locked=np.ones(len(labels), dtype=bool),
            spatial_methods=np.full(len(labels), "car"),
            window_onsets_seconds=np.arange(len(labels), dtype=float),
            paralysis_sides=np.full(len(labels), "not_applicable"),
            nihss=np.full(len(labels), np.nan, dtype=np.float32),
            channel_names=("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"),
            quality_feature_names=tuple(f"q{index}" for index in range(11)),
            sfreq=100.0,
            fingerprint="synthetic",
            recording_qc=(),
        )
        dataset.validate()
        return dataset

    def test_contract_marks_observed_and_requires_disjoint_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            json_path, markdown_path = write_research_contract(temporary)
            self.assertTrue(json_path.is_file())
            self.assertIn(
                DEVELOPMENT_OBSERVED_ROLE,
                markdown_path.read_text(encoding="ascii"),
            )
            self.assertIn(
                LOCKED_EXTERNAL_ROLE,
                __import__("json").loads(
                    json_path.read_text(encoding="ascii")
                )["evaluation"]["split_roles"],
            )
        with self.assertRaises(ValueError):
            validate_split_roles(
                {
                    "development": ["S01"],
                    "controller_validation": ["S01"],
                }
            )

    def test_recording_qc_summary_reports_excluded_subjects(self) -> None:
        summary = summarize_recording_qc(
            (
                {"status": "accepted", "subject": "S01"},
                {
                    "status": "rejected",
                    "subject": "S02",
                    "reason": "too many bad channels",
                },
            )
        )
        self.assertEqual(summary["accepted_recordings"], 1)
        self.assertEqual(summary["excluded_recordings_or_subjects"], 1)
        self.assertEqual(summary["excluded_subjects"], ["S02"])

    def test_features_domain_heads_quality_gate_and_metrics_are_finite(self) -> None:
        dataset = self._dataset()
        training = np.isin(dataset.subject_keys, [f"S{index:02d}" for index in range(8)])
        features = build_translational_features(dataset, training, training)
        ensemble = BranchEnsemble.fit(
            {
                name: values[training]
                for name, values in features.branches.items()
                if name != "quality"
            },
            dataset.labels[training],
            dataset.dataset_ids[training],
            seed=73,
        )
        probabilities = ensemble.predict_stage_probabilities(
            {
                name: values
                for name, values in features.branches.items()
                if name != "quality"
            },
            dataset.dataset_ids,
        )
        gate = QualityAvailabilityGate(0.95).fit(
            features.branches["quality"][training],
            dataset.dataset_ids[training],
        )
        available = gate.available(
            features.branches["quality"],
            dataset.dataset_ids,
        )
        controller = ~training
        threshold = select_window_threshold(
            dataset.labels[controller],
            probabilities.intent[controller],
            available[controller],
            max_false_activation_rate=0.25,
        )
        metrics = score_operating_point(
            dataset.labels[controller],
            StageProbabilities(
                probabilities.intent[controller],
                probabilities.direction[controller],
            ),
            available[controller],
            threshold,
        )
        self.assertTrue(np.isfinite(metrics["intent_balanced_accuracy"]))
        self.assertGreaterEqual(metrics["abstained_fraction"], 0.0)
        self.assertTrue(
            np.isfinite(
                expected_calibration_error(
                    dataset.labels != 0,
                    probabilities.intent,
                )
            )
        )

    def test_v7_feature_build_adds_graph_branch_only_when_requested(self) -> None:
        dataset = self._dataset()
        training = np.isin(
            dataset.subject_keys,
            [f"S{index:02d}" for index in range(8)],
        )
        baseline = build_translational_features(dataset, training, training)
        enhanced = build_translational_features(
            dataset,
            training,
            training,
            include_graph=True,
        )
        self.assertNotIn("gsp", baseline.branches)
        self.assertIn("gsp", enhanced.branches)
        self.assertEqual(
            enhanced.branches["gsp"].shape,
            (len(dataset.labels), 64),
        )
        self.assertEqual(len(enhanced.names["gsp"]), 64)
        self.assertFalse(
            enhanced.metadata["gsp"]["edge_parameters_learned_from_eeg"]
        )

    def test_external_features_use_no_external_individual_mu_calibration(self) -> None:
        reference = self._dataset()
        external = self._dataset()
        external.subject_keys = np.asarray([f"E{index:02d}" for index in range(len(external.labels))])
        training = np.arange(len(reference.labels)) < 24
        features = build_external_translational_features(external, reference, training)
        self.assertTrue(np.all(features.individualized_mu_hz == 10.0))
        self.assertEqual(features.metadata["mode"], "locked_external")
        self.assertEqual(features.metadata["riemann"]["reference_rows"], 24)

    def test_stage_fusion_domain_calibration_and_bootstrap(self) -> None:
        dataset = self._dataset()
        rng = np.random.default_rng(74)
        binary = dataset.labels != 0
        base_intent = np.clip(
            0.25 + 0.45 * binary + rng.normal(0, 0.12, len(binary)),
            0.01,
            0.99,
        )
        right = np.clip(
            0.25
            + 0.50 * (dataset.labels == 2)
            + rng.normal(0, 0.12, len(binary)),
            0.01,
            0.99,
        )
        branch_probabilities = {}
        for offset, name in enumerate(
            ("mrcp", "smr", "riemann", "fbcsp")
        ):
            shifted_intent = np.clip(
                base_intent + 0.01 * offset,
                0.01,
                0.99,
            )
            shifted_right = np.clip(
                right - 0.01 * offset,
                0.01,
                0.99,
            )
            branch_probabilities[name] = StageProbabilities(
                shifted_intent,
                np.column_stack((1.0 - shifted_right, shifted_right)),
            )
        calibration = np.arange(len(binary)) % 2 == 0
        fusion = StageSpecificFusion.select(
            branch_probabilities,
            dataset.labels,
            dataset.dataset_ids,
            calibration,
        )
        raw = fusion.predict(branch_probabilities)
        calibrator = DomainProbabilityCalibrator(seed=74).fit(
            StageProbabilities(
                raw.intent[calibration],
                raw.direction[calibration],
            ),
            dataset.labels[calibration],
            dataset.dataset_ids[calibration],
        )
        calibrator.select_blends(
            raw,
            dataset.labels,
            dataset.dataset_ids,
            ~calibration,
        )
        calibrated = calibrator.predict(raw, dataset.dataset_ids)
        self.assertTrue(np.all(np.isfinite(calibrated.intent)))
        self.assertTrue(np.allclose(calibrated.direction.sum(axis=1), 1.0))
        threshold = select_sensitivity_threshold(
            dataset.labels,
            calibrated.intent,
            np.ones(len(binary), dtype=bool),
            target_sensitivity=0.60,
        )
        threshold_values = domain_threshold_vector(
            dataset.dataset_ids,
            {"a": threshold, "b": threshold},
            threshold,
        )
        intervals = subject_bootstrap_confidence_intervals(
            dataset.labels,
            calibrated,
            np.ones(len(binary), dtype=bool),
            threshold_values,
            dataset.subject_keys,
            iterations=25,
            seed=74,
        )
        self.assertIn("intent_sensitivity", intervals)
        self.assertLessEqual(
            intervals["intent_sensitivity"]["lower_95"],
            intervals["intent_sensitivity"]["upper_95"],
        )

    def test_rapid_patient_adapter_uses_population_and_local_features(self) -> None:
        labels = np.tile(np.asarray([0, 1, 2]), 4)
        rng = np.random.default_rng(75)
        right = np.clip(0.2 + 0.6 * (labels == 2), 0.01, 0.99)
        population = StageProbabilities(
            np.clip(0.2 + 0.6 * (labels != 0), 0.01, 0.99),
            np.column_stack((1.0 - right, right)),
        )
        branches = {
            "population_stage": population_stage_features(population),
            "mrcp": rng.normal(size=(len(labels), 4)),
        }
        adapter = RapidPatientAdapter.fit(
            branches,
            labels,
            "stroke",
            branch_names=("population_stage", "mrcp"),
            blend_weight=0.5,
            seed=75,
        )
        adapted = adapter.predict(branches, "stroke", population)
        self.assertEqual(adapter.calibration_rows, len(labels))
        self.assertEqual(adapted.intent.shape, (len(labels),))
        self.assertTrue(np.all(np.isfinite(adapted.intent)))
        self.assertTrue(np.allclose(adapted.direction.sum(axis=1), 1.0))

    def test_intent_only_adapter_excludes_calibration_recording(self) -> None:
        labels = np.tile(np.asarray([0, 1, 2, 0, 1, 2]), 2)
        subjects = np.repeat(np.asarray(["A", "B"]), 6)
        recordings = np.tile(np.repeat(np.asarray([0, 1]), 3), 2)
        domains = np.asarray(["stroke"] * len(labels))
        rng = np.random.default_rng(76)
        population = np.clip(
            0.2 + 0.6 * (labels != 0) + rng.normal(0.0, 0.03, len(labels)),
            0.01,
            0.99,
        )
        branches = {
            "population_intent": population_intent_features(population),
            "mrcp": rng.normal(size=(len(labels), 3)),
        }
        adapted, evaluation = patient_adapted_intent_probabilities(
            branches,
            labels,
            domains,
            subjects,
            recordings,
            np.ones(len(labels), dtype=bool),
            population,
            branch_names=("population_intent", "mrcp"),
            budget_per_class=2,
            blend_weight=0.5,
            seed=76,
        )
        self.assertTrue(np.all(~evaluation[recordings == 0]))
        self.assertTrue(np.all(evaluation[recordings == 1]))
        self.assertTrue(np.allclose(adapted[recordings == 0], population[recordings == 0]))
        metrics = score_intent_operating_point(
            labels[evaluation],
            adapted[evaluation],
            np.ones(np.sum(evaluation), dtype=bool),
            0.5,
        )
        self.assertNotIn("conditional_direction_accuracy", metrics)

    def test_intent_only_models_merge_left_and_right(self) -> None:
        labels = np.tile(np.asarray([0, 1, 2]), 8)
        domains = np.repeat(np.asarray(["a", "b"]), len(labels) // 2)
        rng = np.random.default_rng(77)
        features = rng.normal(size=(len(labels), 4))
        features[:, 0] += 2.0 * (labels != 0)
        ensemble = IntentBranchEnsemble.fit(
            {"mrcp": features},
            labels,
            domains,
            branch_names=("mrcp",),
            seed=77,
        )
        intent = ensemble.predict_intent_probabilities(
            {"mrcp": features},
            domains,
        )
        self.assertEqual(intent.shape, (len(labels),))
        stage = StageProbabilities(intent, np.full((len(labels), 2), 0.5))
        fusion = IntentSpecificFusion.select(
            {"mrcp": stage},
            labels,
            domains,
            np.ones(len(labels), dtype=bool),
        )
        calibrator = DomainIntentProbabilityCalibrator(seed=77).fit(
            fusion.predict({"mrcp": stage}),
            labels,
            domains,
        )
        calibrator.select_blends(
            intent,
            labels,
            domains,
            np.ones(len(labels), dtype=bool),
        )
        calibrated = calibrator.predict(intent, domains)
        self.assertTrue(np.all(np.isfinite(calibrated)))
        self.assertTrue(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))

    def test_auxiliary_direction_adapter_exposes_only_binary_intent(self) -> None:
        labels = np.tile(np.asarray([0, 1, 2, 0, 1, 2]), 2)
        subjects = np.repeat(np.asarray(["A", "B"]), 6)
        recordings = np.tile(np.repeat(np.asarray([0, 1]), 3), 2)
        domains = np.asarray(["stroke"] * len(labels))
        rng = np.random.default_rng(78)
        population = np.clip(
            0.25 + 0.5 * (labels != 0) + rng.normal(0.0, 0.03, len(labels)),
            0.01,
            0.99,
        )
        branches = {
            "population_intent": population_intent_features(population),
            "mrcp": rng.normal(size=(len(labels), 3)),
        }
        adapted, evaluation = patient_adapted_auxiliary_intent_probabilities(
            branches,
            labels,
            domains,
            subjects,
            recordings,
            np.ones(len(labels), dtype=bool),
            population,
            branch_names=("population_intent", "mrcp"),
            budget_per_source_class=1,
            blend_weight=0.5,
            auxiliary_weight=0.25,
            seed=78,
        )
        self.assertTrue(np.all(~evaluation[recordings == 0]))
        self.assertTrue(np.all(evaluation[recordings == 1]))
        self.assertEqual(adapted.ndim, 1)
        self.assertTrue(np.all((adapted >= 0.0) & (adapted <= 1.0)))

        auxiliary = AuxiliaryIntentBranchEnsemble.fit(
            {"mrcp": branches["mrcp"]},
            labels,
            domains,
            branch_names=("mrcp",),
            seed=78,
        )
        probabilities = auxiliary.predict_intent_probabilities(
            {"mrcp": branches["mrcp"]},
            domains,
        )
        self.assertEqual(probabilities.shape, (len(labels),))

    def test_continuous_metric_counts_confirmed_events(self) -> None:
        result = continuous_false_activations(
            np.asarray([0.1, 0.9, 0.9, 0.2, 0.95, 0.95]),
            np.ones(6, dtype=bool),
            np.asarray(["A"] * 6),
            np.arange(6, dtype=float),
            0.8,
            confirmations=2,
            refractory_seconds=2.0,
        )
        self.assertEqual(result["false_activation_events"], 2)
        self.assertGreater(result["false_activations_per_minute"], 0.0)


class ExtendedFeatureTests(unittest.TestCase):
    def test_morlet_spatiotemporal_features_are_finite_and_cacheable(self) -> None:
        rng = np.random.default_rng(29)
        epochs = rng.normal(size=(5, 7, 200)).astype(np.float32)
        channels = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "morlet_features.npz"
            built = build_morlet_feature_cache(
                epochs,
                100.0,
                channels,
                "synthetic",
                path,
                batch_size=2,
            )
            cached = build_morlet_feature_cache(
                epochs,
                100.0,
                channels,
                "synthetic",
                path,
                batch_size=2,
            )
        self.assertEqual(built.values.shape, (5, 207))
        self.assertEqual(len(built.names), 207)
        self.assertEqual(len(set(built.names)), 207)
        self.assertTrue(np.all(np.isfinite(built.values)))
        np.testing.assert_allclose(cached.values, built.values)

    def test_morlet_power_shape_and_kernel_energy(self) -> None:
        rng = np.random.default_rng(30)
        epochs = rng.normal(size=(2, 7, 200))
        power = morlet_band_power(epochs, 100.0)
        self.assertEqual(
            power.shape,
            (2, 7, len(MORLET_BANDS), 200),
        )
        self.assertTrue(np.all(power >= 0.0))
        kernel = morlet_kernel(float(MORLET_FREQUENCIES[0]), 3.5, 100.0)
        self.assertAlmostEqual(float(np.sum(np.abs(kernel) ** 2)), 1.0, places=6)

    def test_atlas_derived_feature_dimensions_match_names(self) -> None:
        rng = np.random.default_rng(31)
        trial_count = 6
        atlas = FeatureAtlasDataset(
            slow_features=np.zeros((trial_count, 55), dtype=np.float32),
            morlet_features=rng.normal(size=(trial_count, 1080)).astype(np.float32),
            csd_features=rng.normal(size=(trial_count, 900)).astype(np.float32),
            covariances=np.tile(
                np.eye(15, dtype=np.float32),
                (trial_count, 5, 4, 1, 1),
            ),
            quality_features=np.zeros((trial_count, 8), dtype=np.float32),
            slow_audit=np.zeros((trial_count, 6, 8), dtype=np.float32),
            csd_topography=np.zeros((trial_count, 64, 5), dtype=np.float32),
            labels=np.asarray([0, 1, 2, 0, 1, 2]),
            subjects=np.ones(trial_count, dtype=np.int64),
            runs=np.asarray([4, 4, 4, 8, 8, 8]),
            trial_indices=np.arange(trial_count),
            individual_mu_hz=np.full(trial_count, 10.0, dtype=np.float32),
            channel_names=[f"EEG{index}" for index in range(64)],
            feature_names={},
            sfreq=160.0,
            fingerprint="synthetic",
        )
        asymmetry, asymmetry_names = derive_asymmetry_features(atlas)
        morphology, morphology_names = derive_morphology_features(atlas)
        self.assertEqual(asymmetry.shape, (trial_count, 528))
        self.assertEqual(morphology.shape, (trial_count, 264))
        self.assertEqual(asymmetry.shape[1], len(asymmetry_names))
        self.assertEqual(morphology.shape[1], len(morphology_names))
        self.assertTrue(np.all(np.isfinite(asymmetry)))
        self.assertTrue(np.all(np.isfinite(morphology)))

    def test_raw_derived_ssd_and_phase_features_are_finite(self) -> None:
        rng = np.random.default_rng(37)
        sfreq = 160.0
        samples = int(4.0 * sfreq)
        labels = np.tile(np.asarray([0, 1, 2]), 4)
        runs = np.repeat(np.asarray([4, 8]), 6)
        time = np.arange(samples) / sfreq
        signal = rng.normal(scale=0.5, size=(len(labels), 15, samples))
        signal += np.sin(2.0 * np.pi * 10.0 * time)[None, None, :]
        ssd, patterns, eigenvalues, ssd_names, band_names = extract_ssd_features(
            signal, labels, runs, sfreq, 10.0
        )
        phase, phase_names = extract_phase_features(signal, sfreq, 10.0)
        self.assertEqual(ssd.shape, (len(labels), 108))
        self.assertEqual(patterns.shape, (3, 15, 3))
        self.assertEqual(eigenvalues.shape, (3, 3))
        self.assertEqual(len(ssd_names), ssd.shape[1])
        self.assertEqual(band_names, ["individual_mu", "beta_low", "beta_high"])
        self.assertEqual(phase.shape, (len(labels), 120))
        self.assertEqual(len(phase_names), phase.shape[1])
        self.assertTrue(np.all(np.isfinite(ssd)))
        self.assertTrue(np.all(np.isfinite(phase)))


class ModelTests(unittest.TestCase):
    def test_decoder_bakeoff_models_emit_finite_binary_logits(self) -> None:
        signal = torch.zeros((3, 14, 200))
        for name in MODEL_FACTORIES:
            model = build_bakeoff_model(name)
            logits = model(signal)
            self.assertEqual(tuple(logits.shape), (3, 2))
            self.assertTrue(torch.all(torch.isfinite(logits)))
            self.assertLess(trainable_parameter_count(model), 25_000)
            logits.sum().backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.all(torch.isfinite(value)) for value in gradients))

    def test_mrcp_fbcnet_seed_ensemble_loads_eeg_only_payloads(self) -> None:
        channels = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (1, 2):
                torch.manual_seed(seed)
                model = build_bakeoff_model("mrcp_fbcnet")
                path = root / "seeds" / f"seed_{seed}" / "mrcp_fbcnet.pt"
                path.parent.mkdir(parents=True)
                torch.save(
                    {
                        "architecture": "mrcp_fbcnet",
                        "model_state": model.state_dict(),
                        "branch_scales": {"mrcp": 1.0, "smr": 1.0},
                        "channel_names": channels,
                        "neural_seed": seed,
                        "peripheral_inputs_required": False,
                    },
                    path,
                )
            rng = np.random.default_rng(9)
            mrcp = rng.normal(size=(5, 7, 200)).astype(np.float32)
            smr = rng.normal(size=(5, 7, 200)).astype(np.float32)
            mean, per_seed, metadata = predict_mrcp_fbcnet_ensemble(
                mrcp,
                smr,
                channels,
                root,
                seeds=(1, 2),
                batch_size=5,
            )
            self.assertEqual(mean.shape, (5,))
            self.assertEqual(set(per_seed), {1, 2})
            self.assertTrue(np.all(np.isfinite(mean)))
            self.assertFalse(metadata["peripheral_inputs_required"])

    def test_shallow_seed_ensemble_loads_completed_bakeoff_payloads(self) -> None:
        channels = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in (3, 4):
                model = build_bakeoff_model("shallow_convnet")
                path = root / "seeds" / f"seed_{seed}" / "shallow_convnet.pt"
                path.parent.mkdir(parents=True)
                torch.save(
                    {
                        "architecture": "shallow_convnet",
                        "model_state": model.state_dict(),
                        "branch_scales": {"mrcp": 1.0, "smr": 1.0},
                        "channel_names": channels,
                        "neural_seed": seed,
                        "peripheral_inputs_required": False,
                    },
                    path,
                )
            signal = np.zeros((4, 7, 200), dtype=np.float32)
            mean, per_seed, metadata = predict_bakeoff_ensemble(
                signal,
                signal,
                channels,
                root,
                architecture="shallow_convnet",
                seeds=(3, 4),
                batch_size=4,
                device="cpu",
            )
            self.assertEqual(mean.shape, (4,))
            self.assertEqual(set(per_seed), {3, 4})
            self.assertEqual(metadata["architecture"], "shallow_convnet")
            self.assertTrue(np.all(np.isfinite(mean)))

    def test_fbcsp_returns_stage_probabilities(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.tile(np.asarray([0, 1, 2]), 4)
        epochs = rng.normal(size=(len(labels), 4, 320)).astype(np.float32)
        epochs[labels == 1, 0] *= 1.5
        epochs[labels == 2, 3] *= 1.5
        model = HierarchicalFBCSP(160.0, components_per_band=2).fit(epochs, labels)
        intent, direction = model.predict_stage_probabilities(epochs)
        self.assertEqual(intent.shape, (len(labels),))
        self.assertEqual(direction.shape, (len(labels), 2))
        self.assertTrue(np.all(np.isfinite(intent)))

    def test_eegnet_forward_has_two_binary_heads(self) -> None:
        model = HierarchicalEEGNetModule(channels=9, dropout=0.0)
        intent, direction = model(torch.zeros((3, 9, 480)))
        self.assertEqual(tuple(intent.shape), (3, 2))
        self.assertEqual(tuple(direction.shape), (3, 2))

    def test_eegnet_initialization_is_reproducible(self) -> None:
        config = EEGNetTrainingConfig(epochs=1, random_seed=13)
        first = HierarchicalEEGNet(9, config)
        second = HierarchicalEEGNet(9, config)
        for first_value, second_value in zip(
            first.model.state_dict().values(), second.model.state_dict().values()
        ):
            self.assertTrue(torch.equal(first_value, second_value))

    def test_eegnet_payload_round_trip(self) -> None:
        config = EEGNetTrainingConfig(epochs=1, random_seed=17, device="cpu")
        original = HierarchicalEEGNet(9, config)
        original.mean_ = np.zeros((1, 9, 1), dtype=np.float32)
        original.std_ = np.ones((1, 9, 1), dtype=np.float32)
        restored = HierarchicalEEGNet.from_payload(original.payload())
        signal = np.zeros((2, 9, 480), dtype=np.float32)
        original_probabilities = original.predict_stage_probabilities(signal)
        restored_probabilities = restored.predict_stage_probabilities(signal)
        np.testing.assert_allclose(original_probabilities[0], restored_probabilities[0])
        np.testing.assert_allclose(original_probabilities[1], restored_probabilities[1])


if __name__ == "__main__":
    unittest.main()
