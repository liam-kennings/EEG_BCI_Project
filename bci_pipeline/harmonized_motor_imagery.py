"""Common PhysioNet EEGMMIDB and OpenBMI motor-imagery representation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .dataset import IMAGERY_RUNS, available_subjects, resolve_data_dir
from .epochs import EpochConfig, build_raw_epoch_dataset, epoch_fingerprint
from .openbmi_benchmark import (
    OpenBMIBenchmarkDataset,
    load_openbmi_benchmark_cache,
)
from .preprocessing import PreprocessingProfile
from .representations.riemann import TEMPORAL_CHANNELS


HARMONIZED_CHANNELS = TEMPORAL_CHANNELS
HARMONIZED_SFREQ = 100.0
HARMONIZED_EPOCH_START_SECONDS = 1.0
HARMONIZED_EPOCH_DURATION_SECONDS = 2.5


@dataclass
class HarmonizedMotorImageryDataset:
    dataset_name: str
    epochs: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    recordings: np.ndarray
    blocks: np.ndarray
    trial_indices: np.ndarray
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
        )
        if any(len(values) != count for values in arrays):
            raise ValueError("Harmonized dataset arrays are not row-aligned.")
        if self.epochs.shape[1] != len(HARMONIZED_CHANNELS):
            raise ValueError("Harmonized dataset has the wrong channel count.")
        if self.epochs.shape[2] != int(
            round(HARMONIZED_SFREQ * HARMONIZED_EPOCH_DURATION_SECONDS)
        ):
            raise ValueError("Harmonized dataset has the wrong epoch duration.")
        if tuple(self.channel_names) != tuple(HARMONIZED_CHANNELS):
            raise ValueError("Harmonized channels are not in the canonical order.")
        if set(np.unique(self.labels)) != {0, 1, 2}:
            raise ValueError("Harmonized dataset must contain rest, left, and right.")
        if not np.all(np.isfinite(self.epochs)):
            raise ValueError("Harmonized epochs contain non-finite values.")

    def covariance_interface(self) -> OpenBMIBenchmarkDataset:
        """Expose the common fields consumed by the covariance cache builder."""
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


def physionet_harmonized_config() -> EpochConfig:
    return EpochConfig(
        baseline_seconds=1.0,
        epoch_start_seconds=HARMONIZED_EPOCH_START_SECONDS,
        epoch_duration_seconds=HARMONIZED_EPOCH_DURATION_SECONDS,
        artifact_peak_to_peak_uv=250.0,
        channels=HARMONIZED_CHANNELS,
        runs=IMAGERY_RUNS,
        preprocessing=PreprocessingProfile(
            sfreq=HARMONIZED_SFREQ,
            l_freq=4.0,
            h_freq=38.0,
            filter_order=4,
            reference="average",
            phase="zero",
            profile_version=1,
        ),
    )


def build_physionet_harmonized_dataset(
    data_dir: str | Path,
    cache_path: Path,
    *,
    subjects: Sequence[int] | None = None,
    rebuild_cache: bool = False,
) -> HarmonizedMotorImageryDataset:
    root = resolve_data_dir(data_dir)
    selected = (
        available_subjects(root)
        if subjects is None
        else sorted(set(int(subject) for subject in subjects))
    )
    config = physionet_harmonized_config()
    raw = build_raw_epoch_dataset(
        root,
        selected,
        cache_path,
        config,
        rebuild_cache,
    )
    channel_lookup = {
        name.upper().replace(".", ""): index
        for index, name in enumerate(raw.channel_names)
    }
    indices = [channel_lookup[name.upper()] for name in HARMONIZED_CHANNELS]
    fingerprint = json.dumps(
        {
            "schema_version": 1,
            "dataset": "physionet_eegmmidb",
            "epoch_fingerprint": epoch_fingerprint(config, selected),
            "calibration_runs": [4, 8],
            "evaluation_run": 12,
        },
        sort_keys=True,
    )
    dataset = HarmonizedMotorImageryDataset(
        dataset_name="physionet",
        epochs=raw.epochs[:, indices].astype(np.float32, copy=False),
        labels=raw.labels,
        subjects=raw.subjects,
        sessions=np.ones(len(raw.labels), dtype=np.int8),
        recordings=np.where(np.isin(raw.runs, (4, 8)), 0, 1).astype(np.int8),
        blocks=raw.runs,
        trial_indices=raw.trial_indices,
        channel_names=list(HARMONIZED_CHANNELS),
        sfreq=raw.sfreq,
        fingerprint=fingerprint,
    )
    dataset.validate()
    return dataset


def load_openbmi_harmonized_dataset(
    controlled_cache_path: Path,
) -> HarmonizedMotorImageryDataset:
    raw = load_openbmi_benchmark_cache(controlled_cache_path)
    if raw is None:
        raise FileNotFoundError(
            f"Controlled OpenBMI cache not found: {controlled_cache_path}"
        )
    lookup = {
        name.upper(): index
        for index, name in enumerate(raw.motor_channel_names)
    }
    missing = [name for name in HARMONIZED_CHANNELS if name.upper() not in lookup]
    if missing:
        raise ValueError(f"OpenBMI controlled cache is missing channels: {missing}")
    indices = [lookup[name.upper()] for name in HARMONIZED_CHANNELS]
    fingerprint = json.dumps(
        {
            "schema_version": 1,
            "dataset": "openbmi",
            "source_fingerprint": raw.fingerprint,
            "channels": list(HARMONIZED_CHANNELS),
            "sampling_hz": HARMONIZED_SFREQ,
            "epoch_start_seconds": HARMONIZED_EPOCH_START_SECONDS,
            "epoch_duration_seconds": HARMONIZED_EPOCH_DURATION_SECONDS,
        },
        sort_keys=True,
    )
    dataset = HarmonizedMotorImageryDataset(
        dataset_name="openbmi",
        epochs=raw.epochs[:, indices].astype(np.float32, copy=False),
        labels=raw.labels,
        subjects=raw.subjects,
        sessions=raw.sessions,
        recordings=raw.recordings,
        blocks=10 * raw.sessions + raw.recordings,
        trial_indices=raw.event_indices,
        channel_names=list(HARMONIZED_CHANNELS),
        sfreq=raw.sfreq,
        fingerprint=fingerprint,
    )
    dataset.validate()
    return dataset


def combine_harmonized_datasets(
    physionet: HarmonizedMotorImageryDataset,
    openbmi: HarmonizedMotorImageryDataset,
    *,
    openbmi_subject_offset: int = 1000,
) -> HarmonizedMotorImageryDataset:
    physionet.validate()
    openbmi.validate()
    fingerprint = json.dumps(
        {
            "schema_version": 1,
            "dataset": "physionet_openbmi",
            "physionet": physionet.fingerprint,
            "openbmi": openbmi.fingerprint,
            "openbmi_subject_offset": openbmi_subject_offset,
        },
        sort_keys=True,
    )
    combined = HarmonizedMotorImageryDataset(
        dataset_name="combined",
        epochs=np.concatenate((physionet.epochs, openbmi.epochs)),
        labels=np.concatenate((physionet.labels, openbmi.labels)),
        subjects=np.concatenate(
            (
                physionet.subjects,
                openbmi.subjects + openbmi_subject_offset,
            )
        ),
        sessions=np.concatenate((physionet.sessions, openbmi.sessions)),
        recordings=np.concatenate(
            (physionet.recordings, openbmi.recordings)
        ),
        blocks=np.concatenate((physionet.blocks, openbmi.blocks)),
        trial_indices=np.concatenate(
            (physionet.trial_indices, openbmi.trial_indices)
        ),
        channel_names=list(HARMONIZED_CHANNELS),
        sfreq=HARMONIZED_SFREQ,
        fingerprint=fingerprint,
    )
    combined.validate()
    return combined
