"""Adapter for the synchronized MRCP hand-gesture EEG-EMG CSV dataset."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from .privileged_transfer import (
    PrivilegedSourceDataset,
    emg_window_features,
    shared_eeg_intent_features,
)


DATASET_FOLDER = "EEG and EMG Dataset for Analyzing Movement-Related"
EEG_SFREQ = 128.0
EMG_SFREQ = 440.0
OUTPUT_SFREQ = 100.0
WINDOW_SECONDS = 2.0
EMG_CONFIRMATION_SECONDS = 0.75
REST_GUARD_SECONDS = 0.10
CUE_TRIGGER = 7711
END_TRIGGER = 7712
EEG_CHANNELS = (
    "AF3",
    "AF4",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "PO3",
    "POz",
    "PO4",
    "O1",
    "O2",
)
TRANSFER_CHANNELS = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
EEG_FILE_PATTERN = re.compile(
    r"SUBJECT(?P<subject>\d+)_Trial_(?P<session>\d+)_EEG.*\.csv$",
    re.IGNORECASE,
)
EMG_FILE_PATTERN = re.compile(
    r"SUBJECT(?P<subject>\d+)_Trial_(?P<session>\d+)_EMG.*\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SynchronizedRecording:
    subject: int
    session: int
    eeg_path: Path
    emg_path: Path


@dataclass(frozen=True)
class EMGOnset:
    sample: int
    latency_seconds: float
    threshold: float
    peak_envelope: float
    peak_to_threshold_ratio: float


@dataclass
class SourceBuildResult:
    dataset: PrivilegedSourceDataset
    sessions: np.ndarray
    trial_indices: np.ndarray
    row_kinds: np.ndarray
    onset_latencies_seconds: np.ndarray
    qc: dict[str, object]


def resolve_mrcp_eeg_emg_root(path: str | Path | None = None) -> Path:
    candidates = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    candidates.extend((Path.cwd() / DATASET_FOLDER, Path.cwd()))
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "SUBJECTS").is_dir():
            return root
        nested = root / DATASET_FOLDER
        if (nested / "SUBJECTS").is_dir():
            return nested.resolve()
    raise FileNotFoundError(
        "Could not find the MRCP EEG-EMG dataset folder containing SUBJECTS/."
    )


def _indexed_files(root: Path, pattern: re.Pattern[str]) -> dict[tuple[int, int], Path]:
    indexed: dict[tuple[int, int], Path] = {}
    for path in sorted((root / "SUBJECTS").glob("SUBJECT*/*.csv")):
        match = pattern.match(path.name)
        if match is None:
            continue
        key = (int(match.group("subject")), int(match.group("session")))
        if key in indexed:
            raise ValueError(f"Duplicate recording for {key}: {indexed[key]} and {path}")
        indexed[key] = path
    return indexed


def discover_synchronized_recordings(
    root: str | Path,
) -> tuple[list[SynchronizedRecording], dict[str, object]]:
    dataset_root = resolve_mrcp_eeg_emg_root(root)
    eeg = _indexed_files(dataset_root, EEG_FILE_PATTERN)
    emg = _indexed_files(dataset_root, EMG_FILE_PATTERN)
    paired = sorted(set(eeg) & set(emg))
    recordings = [
        SynchronizedRecording(
            subject=subject,
            session=session,
            eeg_path=eeg[(subject, session)],
            emg_path=emg[(subject, session)],
        )
        for subject, session in paired
    ]
    return recordings, {
        "dataset_root": str(dataset_root),
        "eeg_files": len(eeg),
        "emg_files": len(emg),
        "paired_recordings": len(recordings),
        "paired_subjects": len({recording.subject for recording in recordings}),
        "missing_eeg": [
            {"subject": subject, "session": session, "emg_path": str(emg[(subject, session)])}
            for subject, session in sorted(set(emg) - set(eeg))
        ],
        "missing_emg": [
            {"subject": subject, "session": session, "eeg_path": str(eeg[(subject, session)])}
            for subject, session in sorted(set(eeg) - set(emg))
        ],
    }


def _read_eeg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    expected = [str(index) for index in range(2, 34)]
    missing = [name for name in expected if name not in frame]
    if missing or "Triggers" not in frame:
        raise ValueError(f"Unexpected EEG columns in {path}: missing {missing}")
    values_uv = frame.loc[:, expected].to_numpy(dtype=np.float64).T
    triggers = frame["Triggers"].to_numpy(dtype=np.int64)
    if not np.all(np.isfinite(values_uv)):
        raise ValueError(f"Non-finite EEG values in {path}")
    return values_uv * 1e-6, triggers


def _read_emg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "Triggers" not in frame or "0" not in frame:
        raise ValueError(f"Unexpected EMG columns in {path}")
    values = frame["0"].to_numpy(dtype=np.float64)
    triggers = frame["Triggers"].to_numpy(dtype=np.int64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite EMG values in {path}")
    return values, triggers


def _filtered_emg(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filtered = sosfiltfilt(
        butter(4, (20.0, 200.0), btype="bandpass", fs=EMG_SFREQ, output="sos"),
        np.asarray(signal, dtype=np.float64),
    )
    window = max(3, int(round(0.05 * EMG_SFREQ)))
    envelope = np.convolve(np.abs(filtered), np.ones(window) / window, mode="same")
    return filtered, envelope


def detect_emg_onset(envelope: np.ndarray, cue_sample: int) -> EMGOnset | None:
    """Detect sustained post-cue EMG using a robust pre-cue baseline."""
    baseline_start = cue_sample - int(round(0.50 * EMG_SFREQ))
    if baseline_start < 0:
        return None
    baseline = envelope[baseline_start:cue_sample]
    median = float(np.median(baseline))
    robust_sigma = float(1.4826 * np.median(np.abs(baseline - median)))
    threshold = median + 3.0 * max(robust_sigma, 1e-9)
    search_start = cue_sample + int(round(0.10 * EMG_SFREQ))
    search_stop = min(len(envelope), cue_sample + int(round(2.0 * EMG_SFREQ)))
    hold = max(3, int(round(0.025 * EMG_SFREQ)))
    above = envelope[search_start:search_stop] > threshold
    onset_offset = next(
        (
            index
            for index in range(max(0, len(above) - hold + 1))
            if np.all(above[index : index + hold])
        ),
        None,
    )
    if onset_offset is None:
        return None
    sample = search_start + onset_offset
    peak = float(np.max(envelope[sample:search_stop]))
    return EMGOnset(
        sample=sample,
        latency_seconds=float((sample - cue_sample) / EMG_SFREQ),
        threshold=threshold,
        peak_envelope=peak,
        peak_to_threshold_ratio=float(peak / max(threshold, 1e-12)),
    )


def _eeg_branches(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    car = eeg - np.mean(eeg, axis=0, keepdims=True)
    mrcp = sosfiltfilt(
        butter(4, (0.1, 4.0), btype="bandpass", fs=EEG_SFREQ, output="sos"),
        car,
        axis=-1,
    )
    smr = sosfiltfilt(
        butter(4, (7.0, 35.0), btype="bandpass", fs=EEG_SFREQ, output="sos"),
        car,
        axis=-1,
    )
    selected = [EEG_CHANNELS.index(name) for name in TRANSFER_CHANNELS]
    return (
        resample_poly(mrcp[selected], 25, 32, axis=-1),
        resample_poly(smr[selected], 25, 32, axis=-1),
    )


def _window(signal: np.ndarray, end_seconds: float, duration_seconds: float) -> np.ndarray | None:
    stop = int(round(end_seconds * OUTPUT_SFREQ))
    start = stop - int(round(duration_seconds * OUTPUT_SFREQ))
    if start < 0 or stop > signal.shape[-1]:
        return None
    values = signal[:, start:stop]
    expected = int(round(duration_seconds * OUTPUT_SFREQ))
    if values.shape[-1] != expected or not np.all(np.isfinite(values)):
        return None
    return values.copy()


def _emg_window(signal: np.ndarray, start: int, samples: int) -> np.ndarray | None:
    stop = start + samples
    if start < 0 or stop > len(signal):
        return None
    return signal[start:stop][None, :].copy()


def build_mrcp_privileged_source(
    root: str | Path,
    *,
    subjects: Sequence[int] | None = None,
) -> SourceBuildResult:
    recordings, discovery = discover_synchronized_recordings(root)
    selected_subjects = None if subjects is None else {int(value) for value in subjects}
    if selected_subjects is not None:
        recordings = [
            recording for recording in recordings if recording.subject in selected_subjects
        ]
    mrcp_epochs: list[np.ndarray] = []
    smr_epochs: list[np.ndarray] = []
    emg_epochs: list[np.ndarray] = []
    labels: list[int] = []
    subject_keys: list[str] = []
    sessions: list[int] = []
    trial_indices: list[int] = []
    row_kinds: list[str] = []
    premovement_end: list[float] = []
    onset_latencies: list[float] = []
    recording_qc: list[dict[str, object]] = []
    rejected: dict[str, int] = {
        "unmatched_cue": 0,
        "emg_onset_not_confirmed": 0,
        "window_out_of_bounds": 0,
        "eeg_amplitude": 0,
    }

    for recording in recordings:
        try:
            eeg, eeg_triggers = _read_eeg(recording.eeg_path)
            emg, emg_triggers = _read_emg(recording.emg_path)
            mrcp, smr = _eeg_branches(eeg)
            filtered_emg, envelope = _filtered_emg(emg)
        except (ValueError, OSError) as error:
            recording_qc.append(
                {
                    "subject": recording.subject,
                    "session": recording.session,
                    "status": "rejected",
                    "reason": str(error),
                }
            )
            continue
        eeg_cues = np.flatnonzero(eeg_triggers == CUE_TRIGGER)
        emg_cues = np.flatnonzero(emg_triggers == CUE_TRIGGER)
        paired_cues = min(len(eeg_cues), len(emg_cues))
        rejected["unmatched_cue"] += abs(len(eeg_cues) - len(emg_cues))
        accepted_events = 0
        for trial, (eeg_cue, emg_cue) in enumerate(
            zip(eeg_cues[:paired_cues], emg_cues[:paired_cues]),
            start=1,
        ):
            onset = detect_emg_onset(envelope, int(emg_cue))
            if onset is None:
                rejected["emg_onset_not_confirmed"] += 1
                continue
            positive_end = float(eeg_cue / EEG_SFREQ + onset.latency_seconds)
            negative_end = float(eeg_cue / EEG_SFREQ - REST_GUARD_SECONDS)
            positive_mrcp = _window(mrcp, positive_end, WINDOW_SECONDS)
            positive_smr = _window(smr, positive_end, WINDOW_SECONDS)
            negative_mrcp = _window(mrcp, negative_end, WINDOW_SECONDS)
            negative_smr = _window(smr, negative_end, WINDOW_SECONDS)
            emg_samples = int(round(EMG_CONFIRMATION_SECONDS * EMG_SFREQ))
            positive_emg = _emg_window(filtered_emg, onset.sample, emg_samples)
            negative_emg = _emg_window(
                filtered_emg,
                int(round(emg_cue - (REST_GUARD_SECONDS + EMG_CONFIRMATION_SECONDS) * EMG_SFREQ)),
                emg_samples,
            )
            windows = (
                positive_mrcp,
                positive_smr,
                negative_mrcp,
                negative_smr,
                positive_emg,
                negative_emg,
            )
            if any(window is None for window in windows):
                rejected["window_out_of_bounds"] += 1
                continue
            assert positive_mrcp is not None
            assert positive_smr is not None
            assert negative_mrcp is not None
            assert negative_smr is not None
            assert positive_emg is not None
            assert negative_emg is not None
            if max(np.ptp(positive_mrcp), np.ptp(negative_mrcp)) * 1e6 > 1000.0:
                rejected["eeg_amplitude"] += 1
                continue
            for label, kind, mrcp_epoch, smr_epoch, emg_epoch, end in (
                (1, "premovement_intent", positive_mrcp, positive_smr, positive_emg, 0.0),
                (0, "verified_rest", negative_mrcp, negative_smr, negative_emg, -REST_GUARD_SECONDS),
            ):
                mrcp_epochs.append(mrcp_epoch)
                smr_epochs.append(smr_epoch)
                emg_epochs.append(emg_epoch)
                labels.append(label)
                subject_keys.append(f"mrcp_eeg_emg:S{recording.subject:02d}")
                sessions.append(recording.session)
                trial_indices.append(trial)
                row_kinds.append(kind)
                premovement_end.append(end)
                onset_latencies.append(onset.latency_seconds)
            accepted_events += 1
        recording_qc.append(
            {
                "subject": recording.subject,
                "session": recording.session,
                "status": "accepted" if accepted_events else "no_usable_events",
                "eeg_cues": int(len(eeg_cues)),
                "emg_cues": int(len(emg_cues)),
                "accepted_movement_events": accepted_events,
                "eeg_path": str(recording.eeg_path),
                "emg_path": str(recording.emg_path),
            }
        )
    if not labels:
        raise RuntimeError("No synchronized EEG-EMG windows passed source QC.")
    mrcp_array = np.stack(mrcp_epochs).astype(np.float32)
    smr_array = np.stack(smr_epochs).astype(np.float32)
    eeg_features, eeg_names = shared_eeg_intent_features(
        mrcp_array,
        smr_array,
        OUTPUT_SFREQ,
        TRANSFER_CHANNELS,
    )
    emg_features, emg_names = emg_window_features(
        np.stack(emg_epochs),
        ("right_forearm",),
    )
    dataset = PrivilegedSourceDataset(
        eeg_features=eeg_features,
        emg_features=emg_features,
        labels=np.asarray(labels, dtype=np.int64),
        subject_keys=np.asarray(subject_keys),
        premovement_end_seconds=np.asarray(premovement_end, dtype=np.float32),
        eeg_feature_names=eeg_names,
        emg_feature_names=emg_names,
        dataset_id="mrcp_eeg_emg_2025",
    )
    dataset.validate()
    accepted_recordings = [
        entry for entry in recording_qc if entry["status"] == "accepted"
    ]
    qc = {
        "discovery": discovery,
        "configuration": {
            "eeg_sampling_hz": EEG_SFREQ,
            "emg_sampling_hz": EMG_SFREQ,
            "output_sampling_hz": OUTPUT_SFREQ,
            "window_seconds": WINDOW_SECONDS,
            "emg_confirmation_seconds": EMG_CONFIRMATION_SECONDS,
            "eeg_channels": list(EEG_CHANNELS),
            "transfer_channels": list(TRANSFER_CHANNELS),
            "cue_trigger": CUE_TRIGGER,
        },
        "rows": len(labels),
        "accepted_movement_events": int(np.sum(np.asarray(labels) == 1)),
        "accepted_subjects": len(set(subject_keys)),
        "accepted_recordings": len(accepted_recordings),
        "rejected_events": rejected,
        "onset_latency_seconds": {
            "median": float(np.median(np.asarray(onset_latencies)[np.asarray(labels) == 1])),
            "p05": float(np.quantile(np.asarray(onset_latencies)[np.asarray(labels) == 1], 0.05)),
            "p95": float(np.quantile(np.asarray(onset_latencies)[np.asarray(labels) == 1], 0.95)),
        },
        "recordings": recording_qc,
    }
    return SourceBuildResult(
        dataset=dataset,
        sessions=np.asarray(sessions, dtype=np.int16),
        trial_indices=np.asarray(trial_indices, dtype=np.int16),
        row_kinds=np.asarray(row_kinds),
        onset_latencies_seconds=np.asarray(onset_latencies, dtype=np.float32),
        qc=qc,
    )


def source_result_as_dict(result: SourceBuildResult) -> dict[str, object]:
    return {
        "dataset": {
            "dataset_id": result.dataset.dataset_id,
            "rows": len(result.dataset.labels),
            "subjects": len(np.unique(result.dataset.subject_keys)),
            "eeg_features": result.dataset.eeg_features.shape[1],
            "emg_features": result.dataset.emg_features.shape[1],
        },
        "qc": result.qc,
    }
