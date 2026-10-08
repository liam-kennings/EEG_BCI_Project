"""MIND EEG-fNIRS adapter and V7-H feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch

from .dataset import normalize_channel_name


MIND_EEG_SAMPLING_HZ = 250.0
MIND_FNIRS_SAMPLING_HZ = 10.0
MIND_FNIRS_CHANNELS = 51
MIND_FNIRS_SAMPLES = 301
MIND_MOTOR_CHANNELS = ("FC3", "FC4", "C3", "CZ", "C4", "CP3", "CP4")
MIND_CORRUPT_FNIRS_SUBJECTS = {"sub-13"}


@dataclass(frozen=True)
class MINDDatasetSummary:
    root: Path
    eeg_motor_files: int
    eeg_rest_files: int
    hbo_files: int
    hbr_files: int
    paired_subjects: tuple[str, ...]
    excluded_subjects: tuple[str, ...]

    @property
    def adapter_ready(self) -> bool:
        return bool(self.paired_subjects)

    def to_text(self) -> str:
        return "\n".join(
            (
                f"MIND root: {self.root}",
                f"Paired usable subjects: {len(self.paired_subjects)}",
                f"EEG motor/rest derivatives: {self.eeg_motor_files}/{self.eeg_rest_files}",
                f"fNIRS HbO/HbR derivatives: {self.hbo_files}/{self.hbr_files}",
                "Excluded incomplete/corrupt subjects: "
                + (", ".join(self.excluded_subjects) or "none"),
            )
        )


@dataclass
class MINDHybridFeatureDataset:
    """Row-aligned EEG and fNIRS features for binary motor intent."""

    eeg_features: np.ndarray
    fnirs_features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    fnirs_quality: np.ndarray
    source_trials: np.ndarray
    task_labels: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.eeg_features,
            self.fnirs_features,
            self.labels,
            self.subjects,
            self.sessions,
            self.fnirs_quality,
            self.source_trials,
            self.task_labels,
        )
        row_count = len(self.labels)
        if any(len(array) != row_count for array in arrays):
            raise ValueError("MIND feature arrays must have matching rows.")
        if self.eeg_features.ndim != 2 or self.fnirs_features.ndim != 2:
            raise ValueError("MIND EEG and fNIRS features must be 2-D.")
        if not set(np.unique(self.labels)).issubset({0, 1}):
            raise ValueError("MIND labels must be binary REST/INTENT values.")

    def subset(self, mask: np.ndarray) -> "MINDHybridFeatureDataset":
        selected = np.asarray(mask)
        return MINDHybridFeatureDataset(
            eeg_features=self.eeg_features[selected],
            fnirs_features=self.fnirs_features[selected],
            labels=self.labels[selected],
            subjects=self.subjects[selected],
            sessions=self.sessions[selected],
            fnirs_quality=self.fnirs_quality[selected],
            source_trials=self.source_trials[selected],
            task_labels=self.task_labels[selected],
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            eeg_features=self.eeg_features.astype(np.float32),
            fnirs_features=self.fnirs_features.astype(np.float32),
            labels=self.labels.astype(np.int8),
            subjects=self.subjects.astype(str),
            sessions=self.sessions.astype(np.int16),
            fnirs_quality=self.fnirs_quality.astype(np.float32),
            source_trials=self.source_trials.astype(str),
            task_labels=self.task_labels.astype(np.int16),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "MINDHybridFeatureDataset":
        with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as data:
            return cls(
                eeg_features=np.asarray(data["eeg_features"], dtype=np.float64),
                fnirs_features=np.asarray(data["fnirs_features"], dtype=np.float64),
                labels=np.asarray(data["labels"], dtype=np.int64),
                subjects=np.asarray(data["subjects"]).astype(str),
                sessions=np.asarray(data["sessions"], dtype=np.int64),
                fnirs_quality=np.asarray(data["fnirs_quality"], dtype=np.float64),
                source_trials=np.asarray(data["source_trials"]).astype(str),
                task_labels=np.asarray(data["task_labels"], dtype=np.int64),
            )


def resolve_mind_root(path: str | Path) -> Path:
    supplied = Path(path).expanduser().resolve()
    candidates = (
        supplied,
        supplied / "MIND_BIDS",
        supplied / "V3" / "MIND_BIDS",
    )
    for candidate in candidates:
        if (candidate / "derivatives" / "preproc-v1").is_dir():
            return candidate
    raise FileNotFoundError(
        "MIND derivatives/preproc-v1 was not found below " f"{supplied}."
    )


def inspect_mind_dataset(path: str | Path) -> MINDDatasetSummary:
    root = resolve_mind_root(path)
    derivatives = root / "derivatives" / "preproc-v1"
    motor = sorted(derivatives.glob("sub-*/eeg/*desc-motorimagery_eeg.mat"))
    rest = sorted(derivatives.glob("sub-*/eeg/*desc-rest_eeg.mat"))
    hbo = sorted(derivatives.glob("sub-*/nirs/*desc-hbo_timeseries.csv"))
    hbr = sorted(derivatives.glob("sub-*/nirs/*desc-hbr_timeseries.csv"))

    def subjects(paths: Iterable[Path]) -> set[str]:
        return {next(part for part in path.parts if part.startswith("sub-")) for path in paths}

    modality_subjects = (
        subjects(motor),
        subjects(rest),
        subjects(hbo),
        subjects(hbr),
    )
    observed = set().union(*modality_subjects)
    paired = set.intersection(*modality_subjects)
    paired -= MIND_CORRUPT_FNIRS_SUBJECTS
    excluded = sorted(observed - paired)
    return MINDDatasetSummary(
        root=root,
        eeg_motor_files=len(motor),
        eeg_rest_files=len(rest),
        hbo_files=len(hbo),
        hbr_files=len(hbr),
        paired_subjects=tuple(sorted(paired)),
        excluded_subjects=tuple(excluded),
    )


def _mat_value(payload: dict[str, object], names: tuple[str, ...]) -> object:
    for name in names:
        if name in payload:
            return payload[name]
    raise KeyError(f"MAT file is missing all expected keys: {', '.join(names)}")


def _channel_names(payload: dict[str, object], count: int) -> tuple[str, ...]:
    raw = payload.get("ch_names")
    if raw is None:
        return tuple(f"CH{index + 1}" for index in range(count))
    values = np.asarray(raw, dtype=object).reshape(-1)
    names = []
    for value in values:
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        names.append(str(value).strip())
    if len(names) != count:
        return tuple(f"CH{index + 1}" for index in range(count))
    return tuple(names)


def _orient_eeg(data: np.ndarray, expected_trials: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected 3-D EEG epochs, received shape {values.shape}.")
    trial_axes = [axis for axis, size in enumerate(values.shape) if size == expected_trials]
    if not trial_axes:
        raise ValueError(
            f"No EEG axis matches {expected_trials} trials in shape {values.shape}."
        )
    trial_axis = trial_axes[-1]
    remaining = [axis for axis in range(3) if axis != trial_axis]
    time_axis = max(remaining, key=lambda axis: values.shape[axis])
    channel_axis = next(axis for axis in remaining if axis != time_axis)
    return np.transpose(values, (trial_axis, channel_axis, time_axis))


def _onehot_labels(payload: dict[str, object], trials: int) -> np.ndarray:
    raw = np.asarray(payload.get("label", np.ones(trials)), dtype=np.float64)
    if raw.ndim == 2 and trials in raw.shape:
        if raw.shape[1] == trials:
            return np.argmax(raw, axis=0).astype(np.int64) + 4
        return np.argmax(raw, axis=1).astype(np.int64) + 4
    values = raw.reshape(-1)
    if len(values) == trials:
        return values.astype(np.int64)
    return np.full(trials, 4, dtype=np.int64)


def _load_fnirs_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    matrix = matrix[np.isfinite(matrix[:, -1])]
    if matrix.shape[1] != MIND_FNIRS_CHANNELS * MIND_FNIRS_SAMPLES + 1:
        raise ValueError(
            f"Unexpected fNIRS derivative shape {matrix.shape} in {path.name}."
        )
    labels = matrix[:, -1].astype(np.int64)
    epochs = np.stack(
        [
            row[:-1].reshape(
                MIND_FNIRS_CHANNELS,
                MIND_FNIRS_SAMPLES,
                order="F",
            )
            for row in matrix
        ]
    )
    return epochs, labels


def _fill_nonfinite(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64).copy()
    for channel in range(values.shape[0]):
        finite = np.isfinite(values[channel])
        replacement = float(np.median(values[channel, finite])) if np.any(finite) else 0.0
        values[channel, ~finite] = replacement
    return values


def _bandpass(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyquist = fs / 2.0
    low_value = max(low, 0.01)
    high_value = min(high, nyquist * 0.95)
    sos = butter(3, (low_value, high_value), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def _log_covariance(signal: np.ndarray) -> np.ndarray:
    covariance = np.cov(signal) + np.eye(signal.shape[0]) * 1e-6
    covariance /= max(float(np.trace(covariance)), 1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    logged = (eigenvectors * np.log(np.clip(eigenvalues, 1e-8, None))) @ eigenvectors.T
    rows, columns = np.triu_indices_from(logged)
    values = logged[rows, columns]
    values[rows != columns] *= np.sqrt(2.0)
    return values


def _band_power(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    frequencies, spectrum = welch(
        signal,
        fs=fs,
        axis=-1,
        nperseg=min(signal.shape[-1], max(32, int(round(fs * 2.0)))),
    )
    selected = (frequencies >= low) & (frequencies <= high)
    if not np.any(selected):
        return np.full(signal.shape[0], 1e-12)
    return np.maximum(np.trapezoid(spectrum[:, selected], frequencies[selected], axis=1), 1e-12)


def extract_v7_compatible_eeg_features(
    task: np.ndarray,
    baseline: np.ndarray,
    *,
    sampling_hz: float,
) -> np.ndarray:
    """Extract MRCP, SMR, and covariance features matching V7's branches."""
    task_values = _fill_nonfinite(task)
    baseline_values = _fill_nonfinite(baseline)
    task_values -= np.mean(task_values, axis=0, keepdims=True)
    baseline_values -= np.mean(baseline_values, axis=0, keepdims=True)

    mrcp = _bandpass(task_values, sampling_hz, 0.1, 4.0)
    times = np.arange(mrcp.shape[1], dtype=np.float64) / sampling_hz
    centered_time = times - np.mean(times)
    denominator = max(float(np.sum(centered_time**2)), 1e-12)
    slopes = (mrcp @ centered_time) / denominator
    mrcp_features = np.column_stack(
        (
            np.mean(mrcp, axis=1),
            np.min(mrcp, axis=1),
            np.max(mrcp, axis=1),
            slopes,
            np.trapezoid(mrcp, times, axis=1),
        )
    ).reshape(-1)

    spectral = []
    for low, high in ((4.0, 7.0), (8.0, 13.0), (13.0, 30.0)):
        task_power = _band_power(task_values, sampling_hz, low, high)
        baseline_power = _band_power(baseline_values, sampling_hz, low, high)
        spectral.extend((np.log(task_power), np.log(task_power / baseline_power)))

    covariance = np.concatenate(
        [
            _log_covariance(_bandpass(task_values, sampling_hz, low, high))
            for low, high in ((0.1, 4.0), (7.0, 13.0), (13.0, 30.0), (7.0, 35.0))
        ]
    )
    return np.nan_to_num(
        np.concatenate((mrcp_features, *spectral, covariance)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _fnirs_modality_features(task: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    task_values = _fill_nonfinite(task)
    baseline_values = _fill_nonfinite(baseline)
    centered = task_values - np.mean(baseline_values, axis=1, keepdims=True)
    times = np.arange(centered.shape[1], dtype=np.float64) / MIND_FNIRS_SAMPLING_HZ
    centered_time = times - np.mean(times)
    denominator = max(float(np.sum(centered_time**2)), 1e-12)
    slopes = (centered @ centered_time) / denominator
    split = max(1, centered.shape[1] // 2)
    return np.column_stack(
        (
            np.mean(centered, axis=1),
            slopes,
            np.trapezoid(centered, times, axis=1),
            np.std(centered, axis=1),
            np.mean(centered[:, split:], axis=1)
            - np.mean(centered[:, :split], axis=1),
        )
    ).reshape(-1)


def fnirs_quality_score(hbo: np.ndarray, hbr: np.ndarray) -> float:
    """Derivative-level quality proxy; raw SCI remains preferable when available."""
    values = np.concatenate((np.asarray(hbo).reshape(-1), np.asarray(hbr).reshape(-1)))
    finite_fraction = float(np.mean(np.isfinite(values)))
    if finite_fraction == 0.0:
        return 0.0
    clean = values[np.isfinite(values)]
    differences = np.abs(np.diff(clean))
    if not len(differences):
        return finite_fraction
    ratio = float(np.percentile(differences, 95)) / max(
        float(np.median(differences)),
        1e-12,
    )
    motion_score = float(np.clip(1.0 - (ratio - 5.0) / 45.0, 0.0, 1.0))
    return finite_fraction * motion_score


def extract_fnirs_features(
    hbo_task: np.ndarray,
    hbo_baseline: np.ndarray,
    hbr_task: np.ndarray,
    hbr_baseline: np.ndarray,
) -> tuple[np.ndarray, float]:
    features = np.concatenate(
        (
            _fnirs_modality_features(hbo_task, hbo_baseline),
            _fnirs_modality_features(hbr_task, hbr_baseline),
        )
    )
    quality = fnirs_quality_score(hbo_task, hbr_task)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), quality


def _select_motor_channels(
    epochs: np.ndarray,
    channel_names: tuple[str, ...],
) -> np.ndarray:
    normalized = [normalize_channel_name(name) for name in channel_names]
    indexes = [normalized.index(name) for name in MIND_MOTOR_CHANNELS if name in normalized]
    if len(indexes) < 3:
        indexes = list(range(min(7, epochs.shape[1])))
    selected = epochs[:, indexes, :]
    if selected.shape[1] < 7:
        padding = np.repeat(selected[:, -1:, :], 7 - selected.shape[1], axis=1)
        selected = np.concatenate((selected, padding), axis=1)
    return selected[:, :7, :]


def _subject_paths(root: Path, subject: str) -> dict[str, Path]:
    base = root / "derivatives" / "preproc-v1" / subject
    return {
        "motor": next((base / "eeg").glob("*desc-motorimagery_eeg.mat")),
        "rest": next((base / "eeg").glob("*desc-rest_eeg.mat")),
        "hbo": next((base / "nirs").glob("*desc-hbo_timeseries.csv")),
        "hbr": next((base / "nirs").glob("*desc-hbr_timeseries.csv")),
    }


def build_mind_hybrid_feature_dataset(
    path: str | Path,
    *,
    max_subjects: int | None = None,
    rest_windows_per_trial: int = 10,
) -> MINDHybridFeatureDataset:
    """Build aligned binary REST/INTENT features from MIND derivatives."""
    if rest_windows_per_trial < 1 or rest_windows_per_trial > 10:
        raise ValueError("rest_windows_per_trial must be between 1 and 10.")
    summary = inspect_mind_dataset(path)
    subjects = list(summary.paired_subjects)
    if max_subjects is not None:
        subjects = subjects[: int(max_subjects)]
    if not subjects:
        raise RuntimeError("No usable MIND EEG-fNIRS derivative pairs were found.")

    eeg_rows: list[np.ndarray] = []
    fnirs_rows: list[np.ndarray] = []
    labels: list[int] = []
    row_subjects: list[str] = []
    sessions: list[int] = []
    qualities: list[float] = []
    source_trials: list[str] = []
    task_labels: list[int] = []

    for subject in subjects:
        paths = _subject_paths(summary.root, subject)
        motor_payload = loadmat(paths["motor"], squeeze_me=True)
        rest_payload = loadmat(paths["rest"], squeeze_me=True)
        hbo, hbo_labels = _load_fnirs_csv(paths["hbo"])
        hbr, hbr_labels = _load_fnirs_csv(paths["hbr"])
        if not np.array_equal(hbo_labels, hbr_labels):
            raise ValueError(f"HbO/HbR labels do not align for {subject}.")

        intent_mask = np.isin(hbo_labels, (4, 5, 6, 7))
        rest_mask = np.isin(hbo_labels, (1, 2))
        hbo_intent, hbr_intent = hbo[intent_mask], hbr[intent_mask]
        hbo_rest, hbr_rest = hbo[rest_mask], hbr[rest_mask]
        fnirs_intent_labels = hbo_labels[intent_mask]

        motor_data_raw = np.asarray(
            _mat_value(motor_payload, ("preprocessed_data", "data"))
        )
        motor_epochs = _orient_eeg(motor_data_raw, len(hbo_intent))
        motor_names = _channel_names(motor_payload, motor_epochs.shape[1])
        motor_epochs = _select_motor_channels(motor_epochs, motor_names)
        motor_labels = _onehot_labels(motor_payload, len(motor_epochs))
        motor_fs = float(np.asarray(motor_payload.get("srate", MIND_EEG_SAMPLING_HZ)).squeeze())
        cue_sample = int(round(5.0 * motor_fs))
        baseline_start = max(0, cue_sample - int(round(2.0 * motor_fs)))
        task_end = cue_sample + int(round(3.0 * motor_fs))
        if task_end > motor_epochs.shape[-1]:
            raise ValueError(f"Motor epoch is too short for {subject}.")

        fnirs_zero = int(round(5.0 * MIND_FNIRS_SAMPLING_HZ))
        fnirs_task_end = fnirs_zero + int(round(6.0 * MIND_FNIRS_SAMPLING_HZ))
        for trial in range(min(len(motor_epochs), len(hbo_intent))):
            eeg_feature = extract_v7_compatible_eeg_features(
                motor_epochs[trial, :, cue_sample:task_end],
                motor_epochs[trial, :, baseline_start:cue_sample],
                sampling_hz=motor_fs,
            )
            fnirs_feature, quality = extract_fnirs_features(
                hbo_intent[trial, :, fnirs_zero:fnirs_task_end],
                hbo_intent[trial, :, :fnirs_zero],
                hbr_intent[trial, :, fnirs_zero:fnirs_task_end],
                hbr_intent[trial, :, :fnirs_zero],
            )
            eeg_rows.append(eeg_feature)
            fnirs_rows.append(fnirs_feature)
            labels.append(1)
            row_subjects.append(subject)
            sessions.append(min(2, trial // 40))
            qualities.append(quality)
            source_trials.append(f"{subject}:intent:{trial:03d}")
            task_labels.append(int(fnirs_intent_labels[trial] if trial < len(fnirs_intent_labels) else motor_labels[trial]))

        rest_data_raw = np.asarray(_mat_value(rest_payload, ("preprocessed_data", "data")))
        rest_epochs = _orient_eeg(rest_data_raw, len(hbo_rest))
        rest_names = _channel_names(rest_payload, rest_epochs.shape[1])
        rest_epochs = _select_motor_channels(rest_epochs, rest_names)
        rest_fs = float(np.asarray(rest_payload.get("srate", MIND_EEG_SAMPLING_HZ)).squeeze())
        starts = np.linspace(2.0, 20.0, rest_windows_per_trial)
        for trial in range(min(len(rest_epochs), len(hbo_rest))):
            for window_index, start_seconds in enumerate(starts):
                eeg_start = int(round(start_seconds * rest_fs))
                eeg_end = eeg_start + int(round(3.0 * rest_fs))
                eeg_baseline_start = max(0, eeg_start - int(round(2.0 * rest_fs)))
                fnirs_start = int(round((5.0 + start_seconds) * MIND_FNIRS_SAMPLING_HZ))
                fnirs_end = fnirs_start + int(round(6.0 * MIND_FNIRS_SAMPLING_HZ))
                fnirs_baseline_start = max(0, fnirs_start - int(round(2.0 * MIND_FNIRS_SAMPLING_HZ)))
                if eeg_end > rest_epochs.shape[-1] or fnirs_end > MIND_FNIRS_SAMPLES:
                    continue
                eeg_feature = extract_v7_compatible_eeg_features(
                    rest_epochs[trial, :, eeg_start:eeg_end],
                    rest_epochs[trial, :, eeg_baseline_start:eeg_start],
                    sampling_hz=rest_fs,
                )
                fnirs_feature, quality = extract_fnirs_features(
                    hbo_rest[trial, :, fnirs_start:fnirs_end],
                    hbo_rest[trial, :, fnirs_baseline_start:fnirs_start],
                    hbr_rest[trial, :, fnirs_start:fnirs_end],
                    hbr_rest[trial, :, fnirs_baseline_start:fnirs_start],
                )
                eeg_rows.append(eeg_feature)
                fnirs_rows.append(fnirs_feature)
                labels.append(0)
                row_subjects.append(subject)
                sessions.append(min(2, trial // 2))
                qualities.append(quality)
                source_trials.append(
                    f"{subject}:rest:{trial:02d}:window:{window_index:02d}"
                )
                task_labels.append(0)

    return MINDHybridFeatureDataset(
        eeg_features=np.asarray(eeg_rows, dtype=np.float64),
        fnirs_features=np.asarray(fnirs_rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(row_subjects, dtype=str),
        sessions=np.asarray(sessions, dtype=np.int64),
        fnirs_quality=np.asarray(qualities, dtype=np.float64),
        source_trials=np.asarray(source_trials, dtype=str),
        task_labels=np.asarray(task_labels, dtype=np.int64),
    )


def synthetic_mind_feature_dataset(
    *,
    subjects: int = 18,
    seed: int = 42,
) -> MINDHybridFeatureDataset:
    """Create a grouped synthetic dataset for deterministic pipeline tests."""
    if subjects < 9:
        raise ValueError("Synthetic V7-H testing requires at least 9 subjects.")
    rng = np.random.default_rng(seed)
    eeg_rows: list[np.ndarray] = []
    fnirs_rows: list[np.ndarray] = []
    labels: list[int] = []
    subject_rows: list[str] = []
    sessions: list[int] = []
    qualities: list[float] = []
    source_trials: list[str] = []
    task_labels: list[int] = []
    for subject_index in range(subjects):
        subject = f"sub-{subject_index + 1:02d}"
        eeg_offset = rng.normal(0.0, 0.35, 40)
        fnirs_offset = rng.normal(0.0, 0.35, 60)
        for session in range(3):
            for label, count in ((0, 12), (1, 24)):
                for trial in range(count):
                    eeg = rng.normal(0.0, 1.0, 40) + eeg_offset
                    fnirs = rng.normal(0.0, 1.0, 60) + fnirs_offset
                    if label:
                        eeg[:10] += 0.75
                        fnirs[:15] += 0.95
                        if trial % 4 == 0:
                            eeg[10:15] -= 0.4
                    quality = float(np.clip(rng.normal(0.88, 0.09), 0.0, 1.0))
                    if rng.random() < 0.05:
                        quality *= 0.25
                        fnirs += rng.normal(0.0, 2.0, 60)
                    eeg_rows.append(eeg)
                    fnirs_rows.append(fnirs)
                    labels.append(label)
                    subject_rows.append(subject)
                    sessions.append(session)
                    qualities.append(quality)
                    source_trials.append(
                        f"{subject}:{'intent' if label else 'rest'}:{session}:{trial}"
                    )
                    task_labels.append(4 + trial % 4 if label else 0)
    return MINDHybridFeatureDataset(
        eeg_features=np.asarray(eeg_rows),
        fnirs_features=np.asarray(fnirs_rows),
        labels=np.asarray(labels),
        subjects=np.asarray(subject_rows),
        sessions=np.asarray(sessions),
        fnirs_quality=np.asarray(qualities),
        source_trials=np.asarray(source_trials),
        task_labels=np.asarray(task_labels),
    )
