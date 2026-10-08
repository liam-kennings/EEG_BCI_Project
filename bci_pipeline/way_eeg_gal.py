"""Leakage-safe WAY-EEG-GAL adapter for privileged EEG intent training."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from .privileged_transfer import (
    PrivilegedSourceDataset,
    emg_window_features,
    shared_eeg_intent_features,
)


DATASET_FOLDER = "WAY-EEG-GAL"
OUTPUT_SFREQ = 100.0
WINDOW_SECONDS = 2.0
PREMOVEMENT_GUARD_SECONDS = 0.10
REST_GUARD_SECONDS = 0.10
EMG_CONFIRMATION_DELAY_SECONDS = 0.05
EMG_CONFIRMATION_SECONDS = 0.75
TRANSFER_CHANNELS = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")
WAY_CHANNEL_MAP = {
    "FC3": ("FC1", "FC5"),
    "FC4": ("FC2", "FC6"),
    "C3": ("C3",),
    "Cz": ("Cz",),
    "C4": ("C4",),
    "CP3": ("CP1", "CP5"),
    "CP4": ("CP2", "CP6"),
}


@dataclass(frozen=True)
class WAYSeries:
    participant: int
    series: int
    path: Path


@dataclass
class WAYSourceBuildResult:
    dataset: PrivilegedSourceDataset
    series: np.ndarray
    trial_indices: np.ndarray
    row_kinds: np.ndarray
    onset_latencies_seconds: np.ndarray
    weights: np.ndarray
    surfaces: np.ndarray
    qc: dict[str, object]


def resolve_way_eeg_gal_root(path: str | Path | None = None) -> Path:
    """Resolve either the dataset root or its participants directory."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    candidates.extend((Path.cwd() / DATASET_FOLDER, Path.cwd()))
    for candidate in candidates:
        root = candidate.resolve()
        if root.name == "participants" and any(root.glob("P*/HS_P*_S*.mat")):
            return root.parent
        if any((root / "participants").glob("P*/HS_P*_S*.mat")):
            return root
        nested = root / DATASET_FOLDER
        if any((nested / "participants").glob("P*/HS_P*_S*.mat")):
            return nested.resolve()
    raise FileNotFoundError(
        "Could not find WAY-EEG-GAL/participants with continuous HS MAT files."
    )


def discover_way_series(root: str | Path) -> list[WAYSeries]:
    dataset_root = resolve_way_eeg_gal_root(root)
    discovered: list[WAYSeries] = []
    for participant_dir in sorted((dataset_root / "participants").glob("P[0-9][0-9]")):
        participant = int(participant_dir.name[1:])
        for path in sorted(participant_dir.glob(f"HS_P{participant}_S[0-9]*.mat")):
            suffix = path.stem.rsplit("_S", maxsplit=1)[-1]
            if not suffix.isdigit():
                continue
            discovered.append(
                WAYSeries(participant=participant, series=int(suffix), path=path)
            )
    return discovered


def _event_table(root: Path, participant: int) -> tuple[np.ndarray, tuple[str, ...]]:
    path = root / "participants" / f"P{participant:02d}" / f"P{participant}_AllLifts.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Missing WAY event table: {path}")
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)["P"]
    values = np.atleast_2d(np.asarray(payload.AllLifts, dtype=np.float64))
    names = tuple(str(value).strip() for value in np.atleast_1d(payload.ColNames))
    if values.shape[1] != len(names):
        raise ValueError(f"Event columns do not align in {path}")
    return values, names


def _required_event_indices(names: Sequence[str]) -> dict[str, int]:
    indexed = {str(name): index for index, name in enumerate(names)}
    required = ("Run", "Lift", "StartTime", "LEDOn", "tHandStart", "CurW", "CurS")
    missing = [name for name in required if name not in indexed]
    if missing:
        raise ValueError(f"WAY event table is missing columns: {missing}")
    return indexed


def _channel_names(values: object) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in np.atleast_1d(values))


def _select_transfer_channels(
    eeg_microvolts: np.ndarray,
    channel_names: Sequence[str],
) -> np.ndarray:
    signal = np.asarray(eeg_microvolts, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != len(channel_names):
        raise ValueError("WAY EEG must have shape (samples, channels).")
    if not np.all(np.isfinite(signal)):
        raise ValueError("WAY EEG contains non-finite values.")
    indexed = {str(name): index for index, name in enumerate(channel_names)}
    required = sorted({name for sources in WAY_CHANNEL_MAP.values() for name in sources})
    missing = [name for name in required if name not in indexed]
    if missing:
        raise ValueError(f"WAY EEG montage is missing interpolation channels: {missing}")

    signal_volts = signal * 1e-6
    car = signal_volts - np.mean(signal_volts, axis=1, keepdims=True)
    return np.stack(
        [
            np.mean(car[:, [indexed[source] for source in WAY_CHANNEL_MAP[target]]], axis=1)
            for target in TRANSFER_CHANNELS
        ],
        axis=0,
    )


def _resample(signal: np.ndarray, source_sfreq: float) -> np.ndarray:
    ratio = Fraction(OUTPUT_SFREQ / float(source_sfreq)).limit_denominator(1000)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)


def _eeg_branches(
    eeg_microvolts: np.ndarray,
    channel_names: Sequence[str],
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isclose(sfreq, 500.0):
        raise ValueError(f"Expected 500 Hz WAY EEG, received {sfreq:g} Hz.")
    selected = _select_transfer_channels(eeg_microvolts, channel_names)
    mrcp = sosfiltfilt(
        butter(4, (0.1, 4.0), btype="bandpass", fs=sfreq, output="sos"),
        selected,
        axis=-1,
    )
    smr = sosfiltfilt(
        butter(4, (7.0, 35.0), btype="bandpass", fs=sfreq, output="sos"),
        selected,
        axis=-1,
    )
    return _resample(mrcp, sfreq), _resample(smr, sfreq)


def _filtered_emg(emg: np.ndarray, sfreq: float) -> np.ndarray:
    signal = np.asarray(emg, dtype=np.float64)
    if signal.ndim != 2 or not np.all(np.isfinite(signal)):
        raise ValueError("WAY EMG must be a finite (samples, channels) matrix.")
    if not np.isclose(sfreq, 4000.0):
        raise ValueError(f"Expected 4000 Hz WAY EMG, received {sfreq:g} Hz.")
    centered = signal - np.median(signal, axis=0, keepdims=True)
    return sosfiltfilt(
        butter(4, (20.0, 450.0), btype="bandpass", fs=sfreq, output="sos"),
        centered,
        axis=0,
    ).T


def _window(
    signal: np.ndarray,
    end_seconds: float,
    duration_seconds: float,
    sfreq: float,
) -> np.ndarray | None:
    stop = int(round(end_seconds * sfreq))
    start = stop - int(round(duration_seconds * sfreq))
    if start < 0 or stop > signal.shape[-1]:
        return None
    values = signal[:, start:stop]
    expected = int(round(duration_seconds * sfreq))
    if values.shape[-1] != expected or not np.all(np.isfinite(values)):
        return None
    return values.copy()


def _load_continuous_series(
    series: WAYSeries,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], float, float]:
    payload = loadmat(series.path, squeeze_me=True, struct_as_record=False)["hs"]
    eeg = np.asarray(payload.eeg.sig, dtype=np.float64)
    emg = np.asarray(payload.emg.sig, dtype=np.float64)
    eeg_names = _channel_names(payload.eeg.names)
    emg_names = _channel_names(payload.emg.names)
    eeg_sfreq = float(payload.eeg.samplingrate)
    emg_sfreq = float(payload.emg.samplingrate)
    mrcp, smr = _eeg_branches(eeg, eeg_names, eeg_sfreq)
    filtered_emg = _filtered_emg(emg, emg_sfreq)
    return mrcp, smr, filtered_emg, eeg_names, emg_names, eeg_sfreq, emg_sfreq


def build_way_privileged_source(
    root: str | Path,
    *,
    subjects: Sequence[int] | None = None,
) -> WAYSourceBuildResult:
    """Build balanced pre-movement/rest EEG rows with training-only EMG evidence."""
    dataset_root = resolve_way_eeg_gal_root(root)
    selected_subjects = None if subjects is None else {int(value) for value in subjects}
    discovered = discover_way_series(dataset_root)
    if selected_subjects is not None:
        discovered = [entry for entry in discovered if entry.participant in selected_subjects]
    if not discovered:
        raise RuntimeError("No selected WAY continuous multimodal series were found.")

    event_tables: dict[int, tuple[np.ndarray, tuple[str, ...]]] = {}
    mrcp_epochs: list[np.ndarray] = []
    smr_epochs: list[np.ndarray] = []
    emg_feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    subject_keys: list[str] = []
    series_values: list[int] = []
    trial_indices: list[int] = []
    row_kinds: list[str] = []
    premovement_end: list[float] = []
    onset_latencies: list[float] = []
    weights: list[float] = []
    surfaces: list[float] = []
    series_qc: list[dict[str, object]] = []
    emg_names_reference: tuple[str, ...] | None = None
    emg_feature_names: tuple[str, ...] | None = None
    rejected = {
        "invalid_event_timing": 0,
        "window_out_of_bounds": 0,
        "eeg_amplitude": 0,
        "series_load_or_montage": 0,
    }

    for entry in discovered:
        if entry.participant not in event_tables:
            event_tables[entry.participant] = _event_table(dataset_root, entry.participant)
        table, column_names = event_tables[entry.participant]
        columns = _required_event_indices(column_names)
        event_rows = table[np.asarray(table[:, columns["Run"]], dtype=int) == entry.series]
        event_rows = event_rows[np.argsort(event_rows[:, columns["Lift"]])]
        try:
            mrcp, smr, filtered_emg, _, emg_names, _, emg_sfreq = _load_continuous_series(entry)
            if emg_names_reference is None:
                emg_names_reference = emg_names
            elif emg_names != emg_names_reference:
                raise ValueError("EMG channel names changed between WAY series.")
        except (KeyError, OSError, ValueError) as error:
            rejected["series_load_or_montage"] += len(event_rows)
            series_qc.append(
                {
                    "participant": entry.participant,
                    "series": entry.series,
                    "status": "rejected",
                    "reason": str(error),
                    "path": str(entry.path),
                }
            )
            continue

        accepted_events = 0
        for fallback_trial, row in enumerate(event_rows, start=1):
            start_time = float(row[columns["StartTime"]])
            cue_time = start_time + float(row[columns["LEDOn"]])
            hand_time = start_time + float(row[columns["tHandStart"]])
            onset_latency = hand_time - cue_time
            if not np.isfinite(onset_latency) or onset_latency <= 0.0:
                rejected["invalid_event_timing"] += 1
                continue
            positive_end = hand_time - PREMOVEMENT_GUARD_SECONDS
            negative_end = cue_time - REST_GUARD_SECONDS
            positive_mrcp = _window(mrcp, positive_end, WINDOW_SECONDS, OUTPUT_SFREQ)
            positive_smr = _window(smr, positive_end, WINDOW_SECONDS, OUTPUT_SFREQ)
            negative_mrcp = _window(mrcp, negative_end, WINDOW_SECONDS, OUTPUT_SFREQ)
            negative_smr = _window(smr, negative_end, WINDOW_SECONDS, OUTPUT_SFREQ)
            positive_emg = _window(
                filtered_emg,
                hand_time + EMG_CONFIRMATION_DELAY_SECONDS + EMG_CONFIRMATION_SECONDS,
                EMG_CONFIRMATION_SECONDS,
                emg_sfreq,
            )
            negative_emg = _window(
                filtered_emg,
                cue_time - REST_GUARD_SECONDS,
                EMG_CONFIRMATION_SECONDS,
                emg_sfreq,
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
            if (
                max(np.ptp(positive_mrcp), np.ptp(negative_mrcp)) * 1e6 > 1000.0
                or max(np.ptp(positive_smr), np.ptp(negative_smr)) * 1e6 > 500.0
            ):
                rejected["eeg_amplitude"] += 1
                continue
            pair_emg_features, pair_emg_names = emg_window_features(
                np.stack((positive_emg, negative_emg)),
                emg_names,
            )
            if emg_feature_names is None:
                emg_feature_names = pair_emg_names
            elif pair_emg_names != emg_feature_names:
                raise ValueError("EMG feature definitions changed between WAY series.")

            trial = int(round(row[columns["Lift"]])) if np.isfinite(row[columns["Lift"]]) else fallback_trial
            for label, kind, mrcp_epoch, smr_epoch, emg_features, end in (
                (
                    1,
                    "premovement_intent",
                    positive_mrcp,
                    positive_smr,
                    pair_emg_features[0],
                    -PREMOVEMENT_GUARD_SECONDS,
                ),
                (
                    0,
                    "verified_rest",
                    negative_mrcp,
                    negative_smr,
                    pair_emg_features[1],
                    -(onset_latency + REST_GUARD_SECONDS),
                ),
            ):
                mrcp_epochs.append(mrcp_epoch.astype(np.float32))
                smr_epochs.append(smr_epoch.astype(np.float32))
                emg_feature_rows.append(emg_features.astype(np.float32))
                labels.append(label)
                subject_keys.append(f"way_eeg_gal:P{entry.participant:02d}")
                series_values.append(entry.series)
                trial_indices.append(trial)
                row_kinds.append(kind)
                premovement_end.append(end)
                onset_latencies.append(onset_latency)
                weights.append(float(row[columns["CurW"]]))
                surfaces.append(float(row[columns["CurS"]]))
            accepted_events += 1
        series_qc.append(
            {
                "participant": entry.participant,
                "series": entry.series,
                "status": "accepted" if accepted_events else "no_usable_events",
                "events": len(event_rows),
                "accepted_movement_events": accepted_events,
                "path": str(entry.path),
            }
        )

    if not labels or emg_feature_names is None:
        raise RuntimeError("No WAY EEG-EMG windows passed source QC.")
    label_array = np.asarray(labels, dtype=np.int64)
    eeg_features, eeg_feature_names = shared_eeg_intent_features(
        np.stack(mrcp_epochs),
        np.stack(smr_epochs),
        OUTPUT_SFREQ,
        TRANSFER_CHANNELS,
    )
    dataset = PrivilegedSourceDataset(
        eeg_features=eeg_features,
        emg_features=np.stack(emg_feature_rows).astype(np.float32),
        labels=label_array,
        subject_keys=np.asarray(subject_keys),
        premovement_end_seconds=np.asarray(premovement_end, dtype=np.float32),
        eeg_feature_names=eeg_feature_names,
        emg_feature_names=emg_feature_names,
        dataset_id="way_eeg_gal_2014",
    )
    dataset.validate()
    positive_latencies = np.asarray(onset_latencies, dtype=np.float64)[label_array == 1]
    accepted_series = [entry for entry in series_qc if entry["status"] == "accepted"]
    qc = {
        "dataset_root": str(dataset_root),
        "configuration": {
            "output_sampling_hz": OUTPUT_SFREQ,
            "window_seconds": WINDOW_SECONDS,
            "premovement_guard_seconds": PREMOVEMENT_GUARD_SECONDS,
            "rest_guard_seconds": REST_GUARD_SECONDS,
            "emg_confirmation_delay_seconds": EMG_CONFIRMATION_DELAY_SECONDS,
            "emg_confirmation_seconds": EMG_CONFIRMATION_SECONDS,
            "transfer_channels": list(TRANSFER_CHANNELS),
            "spatial_channel_map": {
                target: list(sources) for target, sources in WAY_CHANNEL_MAP.items()
            },
        },
        "discovered_series": len(discovered),
        "accepted_series": len(accepted_series),
        "accepted_subjects": len(set(subject_keys)),
        "accepted_movement_events": int(np.sum(label_array == 1)),
        "rows": len(label_array),
        "rejected_events": rejected,
        "cue_to_hand_start_seconds": {
            "median": float(np.median(positive_latencies)),
            "p05": float(np.quantile(positive_latencies, 0.05)),
            "p95": float(np.quantile(positive_latencies, 0.95)),
        },
        "limitations": {
            "right_hand_execution_only": True,
            "visual_cue_inside_some_positive_windows": True,
            "interpolated_transfer_channels": ["FC3", "FC4", "CP3", "CP4"],
            "withheld_eeg_only_challenge_series_excluded": True,
        },
        "series": series_qc,
    }
    return WAYSourceBuildResult(
        dataset=dataset,
        series=np.asarray(series_values, dtype=np.int16),
        trial_indices=np.asarray(trial_indices, dtype=np.int16),
        row_kinds=np.asarray(row_kinds),
        onset_latencies_seconds=np.asarray(onset_latencies, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        surfaces=np.asarray(surfaces, dtype=np.float32),
        qc=qc,
    )
