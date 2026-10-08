"""Controlled OpenBMI reproduction cache and matrix-feature utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import welch

from .openbmi import (
    DATASET_KEY,
    LABEL_IDS,
    OPENBMI_OCCIPITAL_CHANNELS,
    OPENBMI_PAPER_MOTOR_CHANNELS,
    available_openbmi_subjects,
    discover_openbmi_recordings,
    load_openbmi_events,
    read_openbmi_raw,
    resolve_openbmi_root,
)
from .preprocessing import PreprocessingProfile, is_valid_epoch, preprocess_run


@dataclass(frozen=True)
class OpenBMIBenchmarkConfig:
    sfreq: float = 100.0
    l_freq: float = 4.0
    h_freq: float = 38.0
    filter_order: int = 4
    motor_start_seconds: float = 1.0
    motor_duration_seconds: float = 2.5
    rest_start_seconds: float = 5.0
    cue_window_seconds: float = 0.5
    baseline_seconds: float = 1.0
    artifact_peak_to_peak_uv: float = 250.0
    motor_channels: tuple[str, ...] = OPENBMI_PAPER_MOTOR_CHANNELS
    occipital_channels: tuple[str, ...] = OPENBMI_OCCIPITAL_CHANNELS

    @property
    def preprocessing(self) -> PreprocessingProfile:
        return PreprocessingProfile(
            sfreq=self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            filter_order=self.filter_order,
            reference="average",
            phase="zero",
            profile_version=1,
        )


@dataclass
class OpenBMIBenchmarkDataset:
    epochs: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    recordings: np.ndarray
    event_indices: np.ndarray
    motor_channel_names: list[str]
    sfreq: float
    cue_features: dict[str, np.ndarray]
    cue_labels: np.ndarray
    cue_subjects: np.ndarray
    cue_sessions: np.ndarray
    cue_recordings: np.ndarray
    cue_feature_names: dict[str, list[str]]
    fingerprint: str


def benchmark_fingerprint(
    subjects: Sequence[int],
    config: OpenBMIBenchmarkConfig,
) -> str:
    payload = {
        "schema_version": 1,
        "dataset": DATASET_KEY,
        "subjects": sorted(set(int(subject) for subject in subjects)),
        "config": asdict(config),
        "recording_codes": {"train": 0, "test": 1},
        "label_mapping": {"rest": 0, **LABEL_IDS},
        "window_policy": {
            "intent_baseline": "one_second_pre_cue",
            "rest_baseline": "four_to_five_seconds_post_cue",
            "paired_rest": "five_seconds_post_cue",
        },
    }
    return json.dumps(payload, sort_keys=True)


def _summary_feature_names(
    channels: Sequence[str],
    *,
    time_bins: int = 10,
) -> list[str]:
    names: list[str] = []
    for channel in channels:
        names.extend(f"{channel}:time_bin_{index + 1:02d}" for index in range(time_bins))
        names.extend(
            (
                f"{channel}:log_theta_power",
                f"{channel}:log_mu_power",
                f"{channel}:log_beta_power",
                f"{channel}:standard_deviation",
                f"{channel}:peak_to_peak",
            )
        )
    return names


def summarize_signal_window(
    signal: np.ndarray,
    sfreq: float,
    *,
    time_bins: int = 10,
) -> np.ndarray:
    """Create compact time-amplitude and spectral controls for cue audits."""
    if signal.ndim != 2 or signal.shape[1] < time_bins:
        raise ValueError("Cue-control windows must be channels by time.")
    trimmed = signal[:, : signal.shape[1] - signal.shape[1] % time_bins]
    temporal = trimmed.reshape(signal.shape[0], time_bins, -1).mean(axis=2) * 1e6
    nperseg = min(signal.shape[1], max(16, int(round(sfreq / 2.0))))
    frequencies, density = welch(signal, fs=sfreq, nperseg=nperseg, axis=-1)
    spectral: list[np.ndarray] = []
    for low, high in ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0)):
        selected = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(
            density[:, selected],
            frequencies[selected],
            axis=1,
        )
        spectral.append(np.log(np.clip(power, 1e-20, None)))
    statistics = np.column_stack(
        (
            *spectral,
            np.std(signal, axis=1) * 1e6,
            np.ptp(signal, axis=1) * 1e6,
        )
    )
    return np.concatenate((temporal, statistics), axis=1).reshape(-1).astype(np.float32)


def save_openbmi_benchmark_cache(
    path: Path,
    dataset: OpenBMIBenchmarkDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        epochs=dataset.epochs,
        labels=dataset.labels,
        subjects=dataset.subjects,
        sessions=dataset.sessions,
        recordings=dataset.recordings,
        event_indices=dataset.event_indices,
        motor_channel_names=np.asarray(dataset.motor_channel_names),
        sfreq=np.asarray(dataset.sfreq),
        cue_feature_keys=np.asarray(sorted(dataset.cue_features)),
        **{
            f"cue_features_{key}": dataset.cue_features[key]
            for key in sorted(dataset.cue_features)
        },
        cue_labels=dataset.cue_labels,
        cue_subjects=dataset.cue_subjects,
        cue_sessions=dataset.cue_sessions,
        cue_recordings=dataset.cue_recordings,
        cue_feature_names=np.asarray(json.dumps(dataset.cue_feature_names)),
        fingerprint=np.asarray(dataset.fingerprint),
    )


def load_openbmi_benchmark_cache(
    path: Path,
    fingerprint: str | None = None,
) -> OpenBMIBenchmarkDataset | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        stored = str(cached["fingerprint"].item())
        if fingerprint is not None and stored != fingerprint:
            return None
        keys = [str(key) for key in cached["cue_feature_keys"]]
        return OpenBMIBenchmarkDataset(
            epochs=cached["epochs"],
            labels=cached["labels"],
            subjects=cached["subjects"],
            sessions=cached["sessions"],
            recordings=cached["recordings"],
            event_indices=cached["event_indices"],
            motor_channel_names=[
                str(channel) for channel in cached["motor_channel_names"]
            ],
            sfreq=float(cached["sfreq"].item()),
            cue_features={
                key: cached[f"cue_features_{key}"]
                for key in keys
            },
            cue_labels=cached["cue_labels"],
            cue_subjects=cached["cue_subjects"],
            cue_sessions=cached["cue_sessions"],
            cue_recordings=cached["cue_recordings"],
            cue_feature_names=json.loads(str(cached["cue_feature_names"].item())),
            fingerprint=stored,
        )


def _slice(
    signal: np.ndarray,
    start_seconds: float,
    duration_seconds: float,
    sfreq: float,
) -> np.ndarray | None:
    start = int(round(start_seconds * sfreq))
    stop = start + int(round(duration_seconds * sfreq))
    if start < 0 or stop > signal.shape[1]:
        return None
    return signal[:, start:stop].copy()


def build_openbmi_benchmark_dataset(
    root: str | Path,
    subjects: Sequence[int],
    cache_path: Path,
    *,
    config: OpenBMIBenchmarkConfig = OpenBMIBenchmarkConfig(),
    rebuild_cache: bool = False,
) -> OpenBMIBenchmarkDataset:
    dataset_root = resolve_openbmi_root(root)
    selected = sorted(set(int(subject) for subject in subjects))
    complete = set(available_openbmi_subjects(dataset_root, complete_only=True))
    missing = sorted(set(selected) - complete)
    if missing:
        raise FileNotFoundError(
            f"OpenBMI subjects without all four BDF recordings: {missing}"
        )
    fingerprint = benchmark_fingerprint(selected, config)
    if not rebuild_cache:
        cached = load_openbmi_benchmark_cache(cache_path, fingerprint)
        if cached is not None:
            print(
                f"[+] Loaded {len(cached.labels):,} controlled OpenBMI windows "
                f"from {cache_path}"
            )
            return cached

    selected_channels = tuple(
        dict.fromkeys((*config.motor_channels, *config.occipital_channels))
    )
    motor_indices = np.asarray(
        [selected_channels.index(channel) for channel in config.motor_channels]
    )
    occipital_indices = np.asarray(
        [selected_channels.index(channel) for channel in config.occipital_channels]
    )
    rows: list[np.ndarray] = []
    labels: list[int] = []
    subject_rows: list[int] = []
    session_rows: list[int] = []
    recording_rows: list[int] = []
    event_rows: list[int] = []
    cue_rows: dict[str, list[np.ndarray]] = {
        "pre_cue_occipital": [],
        "early_cue_occipital": [],
        "early_cue_motor": [],
        "late_occipital": [],
        "late_motor": [],
    }
    cue_labels: list[int] = []
    cue_subjects: list[int] = []
    cue_sessions: list[int] = []
    cue_recordings: list[int] = []
    rejected = 0
    recordings = [
        recording
        for recording in discover_openbmi_recordings(dataset_root)
        if recording.subject in set(selected)
    ]
    print(
        f"[+] Building controlled OpenBMI cache from {len(selected)} subjects "
        f"and {len(recordings)} recordings..."
    )
    for position, recording in enumerate(recordings, start=1):
        raw = read_openbmi_raw(recording, preload=False)
        processed = preprocess_run(
            raw,
            config.preprocessing,
            selected_channels,
        )
        signal = processed.get_data()
        sfreq = float(processed.info["sfreq"])
        recording_code = 0 if recording.recording == "train" else 1
        for event in load_openbmi_events(recording):
            onset = float(event["onset"])
            intent_baseline = _slice(
                signal,
                onset - config.baseline_seconds,
                config.baseline_seconds,
                sfreq,
            )
            motor = _slice(
                signal,
                onset + config.motor_start_seconds,
                config.motor_duration_seconds,
                sfreq,
            )
            rest_baseline = _slice(
                signal,
                onset + config.rest_start_seconds - config.baseline_seconds,
                config.baseline_seconds,
                sfreq,
            )
            rest = _slice(
                signal,
                onset + config.rest_start_seconds,
                config.motor_duration_seconds,
                sfreq,
            )
            pre_cue = _slice(
                signal,
                onset - config.cue_window_seconds,
                config.cue_window_seconds,
                sfreq,
            )
            early_cue = _slice(
                signal,
                onset,
                config.cue_window_seconds,
                sfreq,
            )
            windows = (intent_baseline, motor, rest_baseline, rest, pre_cue, early_cue)
            if any(window is None for window in windows):
                rejected += 1
                continue
            assert intent_baseline is not None
            assert motor is not None
            assert rest_baseline is not None
            assert rest is not None
            assert pre_cue is not None
            assert early_cue is not None
            if not is_valid_epoch(
                intent_baseline,
                motor,
                rest_baseline,
                rest,
                peak_to_peak_uv=config.artifact_peak_to_peak_uv,
            ):
                rejected += 1
                continue

            intent_mean = np.mean(intent_baseline, axis=1, keepdims=True)
            rest_mean = np.mean(rest_baseline, axis=1, keepdims=True)
            motor -= intent_mean
            rest -= rest_mean
            pre_cue -= intent_mean
            early_cue -= intent_mean
            label = LABEL_IDS[str(event["trial_type"])]
            event_index = int(event["trial_index"])
            for canonical_label, epoch in ((label, motor), (0, rest)):
                rows.append(epoch[motor_indices].astype(np.float32))
                labels.append(canonical_label)
                subject_rows.append(recording.subject)
                session_rows.append(recording.session)
                recording_rows.append(recording_code)
                event_rows.append(event_index)
            cue_windows = {
                "pre_cue_occipital": pre_cue[occipital_indices],
                "early_cue_occipital": early_cue[occipital_indices],
                "early_cue_motor": early_cue[motor_indices],
                "late_occipital": motor[occipital_indices],
                "late_motor": motor[motor_indices],
            }
            for key, window in cue_windows.items():
                cue_rows[key].append(summarize_signal_window(window, sfreq))
            cue_labels.append(label)
            cue_subjects.append(recording.subject)
            cue_sessions.append(recording.session)
            cue_recordings.append(recording_code)
        del raw, processed, signal
        if position % 4 == 0 or position == len(recordings):
            print(
                f"    processed {position:3d}/{len(recordings)} recordings; "
                f"kept {len(labels):,} paired windows"
            )

    if not rows:
        raise RuntimeError("No valid OpenBMI benchmark windows were extracted.")
    cue_feature_names = {
        "pre_cue_occipital": _summary_feature_names(config.occipital_channels),
        "early_cue_occipital": _summary_feature_names(config.occipital_channels),
        "early_cue_motor": _summary_feature_names(config.motor_channels),
        "late_occipital": _summary_feature_names(config.occipital_channels),
        "late_motor": _summary_feature_names(config.motor_channels),
    }
    dataset = OpenBMIBenchmarkDataset(
        epochs=np.stack(rows),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subject_rows, dtype=np.int64),
        sessions=np.asarray(session_rows, dtype=np.int64),
        recordings=np.asarray(recording_rows, dtype=np.int8),
        event_indices=np.asarray(event_rows, dtype=np.int16),
        motor_channel_names=list(config.motor_channels),
        sfreq=config.sfreq,
        cue_features={
            key: np.stack(values).astype(np.float32)
            for key, values in cue_rows.items()
        },
        cue_labels=np.asarray(cue_labels, dtype=np.int64),
        cue_subjects=np.asarray(cue_subjects, dtype=np.int64),
        cue_sessions=np.asarray(cue_sessions, dtype=np.int8),
        cue_recordings=np.asarray(cue_recordings, dtype=np.int8),
        cue_feature_names=cue_feature_names,
        fingerprint=fingerprint,
    )
    save_openbmi_benchmark_cache(cache_path, dataset)
    print(
        f"[+] Saved {len(dataset.labels):,} paired windows; "
        f"rejected {rejected:,} events"
    )
    return dataset


def matrix_invsqrt(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    values = np.clip(values, 1e-10, None)
    return (vectors * (1.0 / np.sqrt(values))[..., None, :]) @ np.swapaxes(
        vectors, -1, -2
    )


def matrix_log(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    values = np.clip(values, 1e-10, None)
    return (vectors * np.log(values)[..., None, :]) @ np.swapaxes(
        vectors, -1, -2
    )


def matrix_exp(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    return (vectors * np.exp(values)[..., None, :]) @ np.swapaxes(
        vectors, -1, -2
    )


def log_euclidean_mean(matrices: np.ndarray) -> np.ndarray:
    return matrix_exp(np.mean(matrix_log(matrices), axis=0))


def upper_triangle_features(matrices: np.ndarray) -> np.ndarray:
    logarithms = matrix_log(matrices)
    channels = matrices.shape[-1]
    rows, columns = np.triu_indices(channels)
    features = logarithms[..., rows, columns]
    features[..., rows != columns] *= np.sqrt(2.0)
    return features
