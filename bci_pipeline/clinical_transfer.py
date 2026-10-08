"""Harmonized hand-intent datasets for healthy-to-stroke transfer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .harmonized_motor_imagery import (
    HARMONIZED_SFREQ,
    HarmonizedMotorImageryDataset,
)
from .liu_stroke import (
    available_liu_subjects,
    build_subject_trials,
    liu_edf_path,
    load_participants,
    read_liu_raw,
    resolve_liu_root,
)
from .openbmi_benchmark import OpenBMIBenchmarkDataset
from .preprocessing import PreprocessingProfile, is_valid_epoch, preprocess_run


# C1/C2 are absent from the Liu montage. These seven central electrodes are the
# complete motor-focused intersection across PhysioNet, OpenBMI, and Liu.
CLINICAL_CHANNELS = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
CLINICAL_SFREQ = HARMONIZED_SFREQ
CLINICAL_EPOCH_DURATION_SECONDS = 2.0
CLINICAL_EPOCH_SAMPLES = int(
    round(CLINICAL_SFREQ * CLINICAL_EPOCH_DURATION_SECONDS)
)
LIU_INTENT_OFFSET_SECONDS = 1.0
LIU_REST_OFFSET_SECONDS = 0.0
LIU_CALIBRATION_TRIALS = 20


@dataclass
class HandIntentDataset:
    """Row-aligned hand-imagery and rest windows from one or more datasets."""

    dataset_name: str
    cohort: str
    epochs: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    recordings: np.ndarray
    blocks: np.ndarray
    trial_indices: np.ndarray
    dataset_ids: np.ndarray
    paralysis_sides: np.ndarray
    nihss: np.ndarray
    channel_names: list[str]
    sfreq: float
    fingerprint: str

    def validate(self) -> None:
        count = len(self.labels)
        arrays = (
            self.epochs,
            self.subjects,
            self.sessions,
            self.recordings,
            self.blocks,
            self.trial_indices,
            self.dataset_ids,
            self.paralysis_sides,
            self.nihss,
        )
        if any(len(values) != count for values in arrays):
            raise ValueError("Clinical-transfer dataset arrays are not row-aligned.")
        if self.epochs.shape != (
            count,
            len(CLINICAL_CHANNELS),
            CLINICAL_EPOCH_SAMPLES,
        ):
            raise ValueError("Clinical-transfer epochs have the wrong shape.")
        if tuple(self.channel_names) != tuple(CLINICAL_CHANNELS):
            raise ValueError("Clinical-transfer channels are not canonical.")
        if not np.isclose(self.sfreq, CLINICAL_SFREQ):
            raise ValueError("Clinical-transfer dataset has the wrong sampling rate.")
        if set(np.unique(self.labels)) != {0, 1, 2}:
            raise ValueError("Dataset must contain rest, left-hand, and right-hand rows.")
        if not np.all(np.isfinite(self.epochs)):
            raise ValueError("Clinical-transfer epochs contain non-finite values.")

    def covariance_interface(self) -> OpenBMIBenchmarkDataset:
        return OpenBMIBenchmarkDataset(
            epochs=self.epochs,
            labels=self.labels,
            subjects=self.subjects,
            sessions=self.sessions,
            recordings=self.recordings,
            event_indices=self.trial_indices,
            motor_channel_names=self.channel_names,
            sfreq=self.sfreq,
            cue_features={},
            cue_labels=np.empty(0, dtype=np.int64),
            cue_subjects=np.empty(0, dtype=np.int64),
            cue_sessions=np.empty(0, dtype=np.int8),
            cue_recordings=np.empty(0, dtype=np.int8),
            cue_feature_names={},
            fingerprint=self.fingerprint,
        )


def from_harmonized_healthy(
    dataset: HarmonizedMotorImageryDataset,
) -> HandIntentDataset:
    """Crop an existing healthy cache to the shared two-second window."""
    dataset.validate()
    count = len(dataset.labels)
    lookup = {
        name.upper(): index for index, name in enumerate(dataset.channel_names)
    }
    indices = [lookup[name.upper()] for name in CLINICAL_CHANNELS]
    fingerprint = json.dumps(
        {
            "schema_version": 1,
            "source": dataset.fingerprint,
            "task": "hand_imagery_rest",
            "epoch_duration_seconds": CLINICAL_EPOCH_DURATION_SECONDS,
        },
        sort_keys=True,
    )
    result = HandIntentDataset(
        dataset_name=dataset.dataset_name,
        cohort="healthy",
        epochs=dataset.epochs[:, indices, :CLINICAL_EPOCH_SAMPLES].astype(
            np.float32,
            copy=False,
        ),
        labels=dataset.labels.astype(np.int64, copy=False),
        subjects=dataset.subjects.astype(np.int64, copy=False),
        sessions=dataset.sessions.astype(np.int16, copy=False),
        recordings=dataset.recordings.astype(np.int8, copy=False),
        blocks=dataset.blocks.astype(np.int16, copy=False),
        trial_indices=dataset.trial_indices.astype(np.int64, copy=False),
        dataset_ids=np.full(count, dataset.dataset_name, dtype="<U16"),
        paralysis_sides=np.full(count, "not_applicable", dtype="<U16"),
        nihss=np.full(count, np.nan, dtype=np.float32),
        channel_names=list(CLINICAL_CHANNELS),
        sfreq=CLINICAL_SFREQ,
        fingerprint=fingerprint,
    )
    result.validate()
    return result


def _liu_fingerprint(root: Path, subjects: Sequence[int]) -> str:
    source_files = []
    for subject in subjects:
        path = liu_edf_path(root, subject)
        stat = path.stat()
        source_files.append(
            {
                "subject": int(subject),
                "size": int(stat.st_size),
                "modified_ns": int(stat.st_mtime_ns),
            }
        )
    return json.dumps(
        {
            "schema_version": 1,
            "dataset": "liu2024",
            "subjects": [int(subject) for subject in subjects],
            "channels": list(CLINICAL_CHANNELS),
            "sampling_hz": CLINICAL_SFREQ,
            "bandpass_hz": [4.0, 38.0],
            "reference": "full_eeg_common_average",
            "epoch_duration_seconds": CLINICAL_EPOCH_DURATION_SECONDS,
            "intent_offset_seconds": LIU_INTENT_OFFSET_SECONDS,
            "rest_offset_seconds": LIU_REST_OFFSET_SECONDS,
            "calibration_trials": LIU_CALIBRATION_TRIALS,
            "artifact_peak_to_peak_uv": 250.0,
            "source_files": source_files,
        },
        sort_keys=True,
    )


def _load_liu_cache(
    path: Path,
    fingerprint: str,
) -> HandIntentDataset | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["fingerprint"].item()) != fingerprint:
            return None
        result = HandIntentDataset(
            dataset_name="liu2024",
            cohort="stroke",
            epochs=cached["epochs"],
            labels=cached["labels"],
            subjects=cached["subjects"],
            sessions=cached["sessions"],
            recordings=cached["recordings"],
            blocks=cached["blocks"],
            trial_indices=cached["trial_indices"],
            dataset_ids=cached["dataset_ids"],
            paralysis_sides=cached["paralysis_sides"],
            nihss=cached["nihss"],
            channel_names=[str(name) for name in cached["channel_names"]],
            sfreq=float(cached["sfreq"].item()),
            fingerprint=fingerprint,
        )
    result.validate()
    return result


def _save_liu_cache(path: Path, dataset: HandIntentDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprint=np.asarray(dataset.fingerprint),
        epochs=dataset.epochs,
        labels=dataset.labels,
        subjects=dataset.subjects,
        sessions=dataset.sessions,
        recordings=dataset.recordings,
        blocks=dataset.blocks,
        trial_indices=dataset.trial_indices,
        dataset_ids=dataset.dataset_ids,
        paralysis_sides=dataset.paralysis_sides,
        nihss=dataset.nihss,
        channel_names=np.asarray(dataset.channel_names),
        sfreq=np.asarray(dataset.sfreq),
    )


def _signal_window(
    signal: np.ndarray,
    sfreq: float,
    onset_seconds: float,
) -> np.ndarray | None:
    start = int(round(onset_seconds * sfreq))
    stop = start + CLINICAL_EPOCH_SAMPLES
    if start < 0 or stop > signal.shape[1]:
        return None
    return signal[:, start:stop].copy()


def build_liu_hand_intent_dataset(
    data_dir: str | Path,
    cache_path: Path,
    *,
    subjects: Sequence[int] | None = None,
    rebuild_cache: bool = False,
) -> HandIntentDataset:
    """Create paired hand-imagery/rest windows from the Liu stroke recordings."""
    root = resolve_liu_root(data_dir)
    selected = (
        available_liu_subjects(root)
        if subjects is None
        else sorted(set(int(subject) for subject in subjects))
    )
    fingerprint = _liu_fingerprint(root, selected)
    if not rebuild_cache:
        cached = _load_liu_cache(cache_path, fingerprint)
        if cached is not None:
            print(f"[+] Loaded {len(cached.labels):,} Liu windows from {cache_path}")
            return cached

    profile = PreprocessingProfile(
        sfreq=CLINICAL_SFREQ,
        l_freq=4.0,
        h_freq=38.0,
        filter_order=4,
        reference="average",
        phase="zero",
        profile_version=1,
    )
    participants = load_participants(root)
    epoch_rows: list[np.ndarray] = []
    labels: list[int] = []
    subject_rows: list[int] = []
    sessions: list[int] = []
    recordings: list[int] = []
    blocks: list[int] = []
    trial_indices: list[int] = []
    paralysis_sides: list[str] = []
    nihss_rows: list[float] = []
    rejected_pairs = 0

    print(f"[+] Extracting harmonized Liu windows from {len(selected)} patients...")
    for position, subject in enumerate(selected, start=1):
        participant = participants[f"sub-{subject:02d}"]
        raw = read_liu_raw(root, subject, preload=False)
        trials = build_subject_trials(
            root,
            subject,
            raw=raw,
            participant=participant,
        )
        processed = preprocess_run(raw, profile, CLINICAL_CHANNELS)
        signal = processed.get_data()
        sfreq = float(processed.info["sfreq"])
        for trial in trials:
            intent = _signal_window(
                signal,
                sfreq,
                trial.intent_start_s + LIU_INTENT_OFFSET_SECONDS,
            )
            rest = _signal_window(
                signal,
                sfreq,
                trial.rest_start_s + LIU_REST_OFFSET_SECONDS,
            )
            if intent is None or rest is None or not is_valid_epoch(
                intent,
                rest,
                peak_to_peak_uv=250.0,
            ):
                rejected_pairs += 1
                continue
            recording = int(trial.trial_index >= LIU_CALIBRATION_TRIALS)
            for epoch, label in ((intent, trial.native_label), (rest, 0)):
                epoch_rows.append(epoch.astype(np.float32))
                labels.append(int(label))
                subject_rows.append(subject)
                sessions.append(1)
                recordings.append(recording)
                blocks.append(trial.trial_index // LIU_CALIBRATION_TRIALS)
                trial_indices.append(trial.trial_index)
                paralysis_sides.append(trial.paralysis_side.strip().lower())
                nihss_rows.append(float(trial.nihss))
        if position % 10 == 0 or position == len(selected):
            print(
                f"    processed {position:2d}/{len(selected)} patients; "
                f"kept {len(epoch_rows):,} windows"
            )

    if not epoch_rows:
        raise RuntimeError("No valid Liu hand-intent windows were extracted.")
    result = HandIntentDataset(
        dataset_name="liu2024",
        cohort="stroke",
        epochs=np.stack(epoch_rows),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subject_rows, dtype=np.int64),
        sessions=np.asarray(sessions, dtype=np.int16),
        recordings=np.asarray(recordings, dtype=np.int8),
        blocks=np.asarray(blocks, dtype=np.int16),
        trial_indices=np.asarray(trial_indices, dtype=np.int64),
        dataset_ids=np.full(len(labels), "liu2024", dtype="<U16"),
        paralysis_sides=np.asarray(paralysis_sides, dtype="<U16"),
        nihss=np.asarray(nihss_rows, dtype=np.float32),
        channel_names=list(CLINICAL_CHANNELS),
        sfreq=CLINICAL_SFREQ,
        fingerprint=fingerprint,
    )
    result.validate()
    _save_liu_cache(cache_path, result)
    print(
        f"[+] Saved Liu cache: {cache_path} "
        f"({len(result.labels):,} windows; {rejected_pairs} rejected trial pairs)"
    )
    return result


def combine_hand_intent_datasets(
    datasets: Sequence[HandIntentDataset],
    *,
    subject_offsets: Sequence[int],
) -> HandIntentDataset:
    """Concatenate datasets while making subject identifiers globally unique."""
    if len(datasets) != len(subject_offsets) or not datasets:
        raise ValueError("Datasets and subject offsets must be non-empty and aligned.")
    for dataset in datasets:
        dataset.validate()
    fingerprints = {
        dataset.dataset_name: dataset.fingerprint for dataset in datasets
    }
    result = HandIntentDataset(
        dataset_name="combined",
        cohort="mixed",
        epochs=np.concatenate([dataset.epochs for dataset in datasets]),
        labels=np.concatenate([dataset.labels for dataset in datasets]),
        subjects=np.concatenate(
            [
                dataset.subjects + int(offset)
                for dataset, offset in zip(datasets, subject_offsets)
            ]
        ),
        sessions=np.concatenate([dataset.sessions for dataset in datasets]),
        recordings=np.concatenate([dataset.recordings for dataset in datasets]),
        blocks=np.concatenate([dataset.blocks for dataset in datasets]),
        trial_indices=np.concatenate(
            [dataset.trial_indices for dataset in datasets]
        ),
        dataset_ids=np.concatenate([dataset.dataset_ids for dataset in datasets]),
        paralysis_sides=np.concatenate(
            [dataset.paralysis_sides for dataset in datasets]
        ),
        nihss=np.concatenate([dataset.nihss for dataset in datasets]),
        channel_names=list(CLINICAL_CHANNELS),
        sfreq=CLINICAL_SFREQ,
        fingerprint=json.dumps(
            {
                "schema_version": 1,
                "datasets": fingerprints,
                "subject_offsets": [int(offset) for offset in subject_offsets],
            },
            sort_keys=True,
        ),
    )
    result.validate()
    return result
