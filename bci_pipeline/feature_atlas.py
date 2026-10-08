"""Leakage-safe, physiology-first feature extraction for EEGMMIDB imagery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import mne
import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt, welch

from .dataset import (
    IMAGERY_RUNS,
    LABEL_NAMES,
    find_run_file,
    map_task_annotation,
    normalize_channel_name,
)
from .representations.riemann import FILTER_BANK, MOTOR_CHANNELS


ATLAS_SCHEMA_VERSION = 2
CALIBRATION_RUNS = (4, 8)
EVALUATION_RUN = 12
SLOW_AUDIT_CHANNELS = ("C3", "Cz", "C4", "O1", "Oz", "O2")
MORLET_FREQUENCIES = np.arange(6.0, 41.0, 2.0)
MORLET_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("theta", 6.0, 8.0),
    ("mu_low", 8.0, 10.0),
    ("mu_high", 10.0, 14.0),
    ("beta_low", 14.0, 22.0),
    ("beta_high", 22.0, 32.0),
)
TIME_EDGES = np.arange(0.5, 3.5001, 0.25)
SLOW_TIME_EDGES = np.arange(0.0, 4.0001, 0.5)
COVARIANCE_PHASES: Tuple[Tuple[str, float, float], ...] = (
    ("early", 0.5, 1.25),
    ("early_middle", 1.25, 2.0),
    ("late_middle", 2.0, 2.75),
    ("late", 2.75, 3.5),
)


@dataclass
class FeatureAtlasDataset:
    slow_features: np.ndarray
    morlet_features: np.ndarray
    csd_features: np.ndarray
    covariances: np.ndarray
    quality_features: np.ndarray
    slow_audit: np.ndarray
    csd_topography: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    trial_indices: np.ndarray
    individual_mu_hz: np.ndarray
    channel_names: List[str]
    feature_names: Dict[str, List[str]]
    sfreq: float
    fingerprint: str

    def validate(self) -> None:
        trial_count = len(self.labels)
        arrays = (
            self.slow_features,
            self.morlet_features,
            self.csd_features,
            self.covariances,
            self.quality_features,
            self.slow_audit,
            self.csd_topography,
            self.subjects,
            self.runs,
            self.trial_indices,
            self.individual_mu_hz,
        )
        if any(len(values) != trial_count for values in arrays):
            raise ValueError("Feature-atlas arrays do not contain the same number of trials.")
        if trial_count == 0:
            raise ValueError("Feature atlas is empty.")
        for values in arrays:
            if not np.all(np.isfinite(values)):
                raise ValueError("Feature atlas contains non-finite values.")


@dataclass
class RunEpochs:
    car: np.ndarray
    slow: np.ndarray
    csd: np.ndarray
    labels: np.ndarray
    trial_indices: np.ndarray
    quality: np.ndarray
    qc: Dict[str, object]
    channel_names: List[str]
    sfreq: float


def atlas_fingerprint(subjects: Sequence[int]) -> str:
    payload = {
        "schema_version": ATLAS_SCHEMA_VERSION,
        "subjects": [int(subject) for subject in subjects],
        "runs": list(IMAGERY_RUNS),
        "calibration_runs": list(CALIBRATION_RUNS),
        "evaluation_run": EVALUATION_RUN,
        "motor_channels": list(MOTOR_CHANNELS),
        "morlet_frequencies": MORLET_FREQUENCIES.tolist(),
        "morlet_bands": [list(band) for band in MORLET_BANDS],
        "time_edges": TIME_EDGES.tolist(),
        "slow_time_edges": SLOW_TIME_EDGES.tolist(),
        "covariance_phases": [list(phase) for phase in COVARIANCE_PHASES],
        "raw_qc_before_reference": True,
        "bad_channel_repair": "spherical_spline_interpolation",
        "reference": "common_average_after_bad_channel_repair",
        "spatial_branch": "surface_laplacian_csd",
        "rest_reference": "runs_4_8_post_T0_0.5_to_3.5_seconds",
    }
    return json.dumps(payload, sort_keys=True)


def save_feature_atlas(path: Path, dataset: FeatureAtlasDataset) -> None:
    dataset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        slow_features=dataset.slow_features,
        morlet_features=dataset.morlet_features,
        csd_features=dataset.csd_features,
        covariances=dataset.covariances,
        quality_features=dataset.quality_features,
        slow_audit=dataset.slow_audit,
        csd_topography=dataset.csd_topography,
        labels=dataset.labels,
        subjects=dataset.subjects,
        runs=dataset.runs,
        trial_indices=dataset.trial_indices,
        individual_mu_hz=dataset.individual_mu_hz,
        channel_names=np.asarray(dataset.channel_names),
        feature_names=np.asarray(json.dumps(dataset.feature_names)),
        sfreq=np.asarray(dataset.sfreq),
        fingerprint=np.asarray(dataset.fingerprint),
    )


def load_feature_atlas(
    path: Path, expected_fingerprint: Optional[str] = None
) -> Optional[FeatureAtlasDataset]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        fingerprint = str(cached["fingerprint"].item())
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            return None
        dataset = FeatureAtlasDataset(
            slow_features=cached["slow_features"],
            morlet_features=cached["morlet_features"],
            csd_features=cached["csd_features"],
            covariances=cached["covariances"],
            quality_features=cached["quality_features"],
            slow_audit=cached["slow_audit"],
            csd_topography=cached["csd_topography"],
            labels=cached["labels"],
            subjects=cached["subjects"],
            runs=cached["runs"],
            trial_indices=cached["trial_indices"],
            individual_mu_hz=cached["individual_mu_hz"],
            channel_names=[str(name) for name in cached["channel_names"]],
            feature_names=json.loads(str(cached["feature_names"].item())),
            sfreq=float(cached["sfreq"].item()),
            fingerprint=fingerprint,
        )
    dataset.validate()
    return dataset


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    return 0.6745 * (values - median) / max(float(mad), 1e-12)


def _band_power(spectrum: np.ndarray, frequencies: np.ndarray, low: float, high: float) -> np.ndarray:
    selected = (frequencies >= low) & (frequencies < high)
    if not np.any(selected):
        return np.zeros(spectrum.shape[:-1], dtype=np.float64)
    return np.trapezoid(spectrum[..., selected], frequencies[selected], axis=-1)


def recording_quality(signal: np.ndarray, sfreq: float, channel_names: Sequence[str]) -> Dict[str, object]:
    """Detect gross channel failures before referencing can spread them globally."""
    signal = np.asarray(signal, dtype=np.float64)
    finite = np.all(np.isfinite(signal), axis=1)
    clean = np.nan_to_num(signal, copy=True)
    rms_uv = np.sqrt(np.mean(clean**2, axis=1)) * 1e6
    differences = np.diff(clean, axis=1)
    flat_fraction = np.mean(np.abs(differences) < 1e-10, axis=1)
    frequencies, spectrum = welch(
        clean,
        fs=sfreq,
        nperseg=min(int(round(2.0 * sfreq)), clean.shape[1]),
        axis=-1,
    )
    total = _band_power(spectrum, frequencies, 1.0, min(75.0, sfreq / 2.0))
    high_frequency = _band_power(spectrum, frequencies, 35.0, min(75.0, sfreq / 2.0))
    line = _band_power(spectrum, frequencies, 58.0, min(62.0, sfreq / 2.0))
    high_ratio = high_frequency / np.clip(total, 1e-30, None)
    line_ratio = line / np.clip(total, 1e-30, None)

    decimation = max(1, int(round(sfreq / 40.0)))
    centered = clean[:, ::decimation]
    centered -= np.mean(centered, axis=1, keepdims=True)
    scales = np.std(centered, axis=1, keepdims=True)
    normalized = centered / np.clip(scales, 1e-20, None)
    correlation = normalized @ normalized.T / max(normalized.shape[1] - 1, 1)
    np.fill_diagonal(correlation, 0.0)
    maximum_correlation = np.max(np.abs(correlation), axis=1)

    low_correlation = (
        maximum_correlation < 0.20
        if len(channel_names) >= 8
        else np.zeros(len(channel_names), dtype=bool)
    )
    bad = (
        ~finite
        # EDF quantization naturally creates short repeated-sample runs. Only
        # flag a channel when this becomes a gross recording failure.
        | (flat_fraction > 0.20)
        | (np.abs(_robust_z(np.log10(np.clip(rms_uv, 1e-6, None)))) > 6.0)
        | (_robust_z(high_ratio) > 6.0)
        | (_robust_z(line_ratio) > 6.0)
        | low_correlation
    )
    bad_names = [str(name) for name, rejected in zip(channel_names, bad) if rejected]
    return {
        "bad_channels": bad_names,
        "bad_channel_count": int(np.sum(bad)),
        "bad_channel_fraction": float(np.mean(bad)),
        "median_rms_uv": float(np.median(rms_uv)),
        "maximum_rms_uv": float(np.max(rms_uv)),
        "median_high_frequency_ratio": float(np.median(high_ratio)),
        "maximum_high_frequency_ratio": float(np.max(high_ratio)),
        "median_line_ratio": float(np.median(line_ratio)),
        "maximum_line_ratio": float(np.max(line_ratio)),
        "minimum_channel_correlation": float(np.min(maximum_correlation)),
        "maximum_flat_fraction": float(np.max(flat_fraction)),
    }


def _standardize_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    raw = raw.copy().pick("eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    montage_lookup = {
        normalize_channel_name(name): name for name in montage.ch_names
    }
    rename = {}
    missing = []
    for name in raw.ch_names:
        normalized = normalize_channel_name(name)
        if normalized not in montage_lookup:
            missing.append(name)
        else:
            rename[name] = montage_lookup[normalized]
    if missing:
        raise ValueError(f"Channels absent from standard_1005 montage: {missing}")
    raw.rename_channels(rename)
    raw.set_montage(montage, on_missing="raise")
    return raw


def _positions(names: Sequence[str], requested: Sequence[str]) -> List[int]:
    lookup = {normalize_channel_name(name): index for index, name in enumerate(names)}
    missing = [name for name in requested if normalize_channel_name(name) not in lookup]
    if missing:
        raise ValueError(f"Required channels are missing: {missing}")
    return [lookup[normalize_channel_name(name)] for name in requested]


def _trial_quality(epoch: np.ndarray, sfreq: float, names: Sequence[str], bad_fraction: float) -> np.ndarray:
    frequencies, spectrum = welch(
        epoch,
        fs=sfreq,
        nperseg=min(int(round(sfreq)), epoch.shape[-1]),
        axis=-1,
    )
    total = _band_power(spectrum, frequencies, 1.0, min(75.0, sfreq / 2.0))
    high = _band_power(spectrum, frequencies, 35.0, min(75.0, sfreq / 2.0))
    line = _band_power(spectrum, frequencies, 58.0, min(62.0, sfreq / 2.0))
    frontal_requested = ("Fp1", "Fp2", "AF7", "AF8")
    frontal = [index for index in _positions(names, frontal_requested)]
    ptp_uv = np.ptp(epoch, axis=-1) * 1e6
    rms_uv = np.sqrt(np.mean(epoch**2, axis=-1)) * 1e6
    return np.asarray(
        [
            np.max(ptp_uv),
            np.median(ptp_uv),
            np.median(rms_uv),
            np.max(np.abs(epoch)) * 1e6,
            np.max(ptp_uv[frontal]),
            np.median(high / np.clip(total, 1e-30, None)),
            np.median(line / np.clip(total, 1e-30, None)),
            bad_fraction,
        ],
        dtype=np.float32,
    )


def extract_run_epochs(file_path: Path, subject: int, run_id: int) -> RunEpochs:
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = _standardize_montage(raw)
    sfreq = float(raw.info["sfreq"])
    if not np.isclose(sfreq, 160.0):
        raw.resample(160.0)
        sfreq = 160.0
    qc = recording_quality(raw.get_data(), sfreq, raw.ch_names)
    qc["qc_level"] = (
        "warning" if float(qc["bad_channel_fraction"]) >= 0.20 else "pass"
    )
    qc.update({"subject": int(subject), "run": int(run_id), "file": str(file_path)})
    if int(qc["bad_channel_count"]) > int(round(0.25 * len(raw.ch_names))):
        raise RuntimeError(
            f"S{subject:03d}R{run_id:02d} has too many bad channels "
            f"({qc['bad_channel_count']}/{len(raw.ch_names)})."
        )

    cleaned = raw.copy()
    cleaned.info["bads"] = list(qc["bad_channels"])
    if cleaned.info["bads"]:
        cleaned.interpolate_bads(reset_bads=True, verbose=False)
    cleaned.set_eeg_reference("average", projection=False, verbose=False)
    broadband = cleaned.copy()
    motor = cleaned.copy().filter(4.0, 40.0, method="iir", verbose=False)
    slow = cleaned.copy().filter(0.1, 4.0, method="iir", verbose=False)
    csd = mne.preprocessing.compute_current_source_density(cleaned.copy(), verbose=False)
    csd.filter(4.0, 40.0, method="iir", verbose=False)

    names = list(cleaned.ch_names)
    motor_positions = _positions(names, MOTOR_CHANNELS)
    slow_positions = _positions(names, SLOW_AUDIT_CHANNELS)
    broadband_signal = broadband.get_data()
    motor_signal = motor.get_data(picks=motor_positions)
    slow_signal = slow.get_data(picks=slow_positions)
    csd_signal = csd.get_data()
    samples = int(round(4.0 * sfreq))
    car_rows: List[np.ndarray] = []
    slow_rows: List[np.ndarray] = []
    csd_rows: List[np.ndarray] = []
    quality_rows: List[np.ndarray] = []
    labels: List[int] = []
    trial_indices: List[int] = []
    trial_index = 0
    for annotation in cleaned.annotations:
        label = map_task_annotation(str(annotation["description"]))
        if label is None:
            continue
        start = int(round(float(annotation["onset"]) * sfreq))
        stop = start + samples
        if start < 0 or stop > broadband_signal.shape[1]:
            continue
        raw_epoch = broadband_signal[:, start:stop]
        car_epoch = motor_signal[:, start:stop]
        slow_epoch = slow_signal[:, start:stop]
        csd_epoch = csd_signal[:, start:stop]
        if not (
            np.all(np.isfinite(raw_epoch))
            and np.all(np.isfinite(car_epoch))
            and np.all(np.isfinite(csd_epoch))
        ):
            continue
        car_rows.append(car_epoch.astype(np.float32))
        slow_rows.append(slow_epoch.astype(np.float32))
        csd_rows.append(csd_epoch.astype(np.float32))
        quality_rows.append(
            _trial_quality(raw_epoch, sfreq, names, float(qc["bad_channel_fraction"]))
        )
        labels.append(int(label))
        trial_indices.append(trial_index)
        trial_index += 1
    if not labels:
        raise RuntimeError(f"No complete annotated trials found in {file_path}.")
    return RunEpochs(
        car=np.stack(car_rows),
        slow=np.stack(slow_rows),
        csd=np.stack(csd_rows),
        labels=np.asarray(labels, dtype=np.int64),
        trial_indices=np.asarray(trial_indices, dtype=np.int64),
        quality=np.stack(quality_rows),
        qc=qc,
        channel_names=names,
        sfreq=sfreq,
    )


def _morlet_kernel(frequency: float, cycles: float, sfreq: float) -> np.ndarray:
    sigma = cycles / (2.0 * np.pi * frequency)
    half_width = max(1.0, 3.5 * sigma)
    time = np.arange(-half_width, half_width + 1.0 / sfreq, 1.0 / sfreq)
    kernel = np.exp(2j * np.pi * frequency * time) * np.exp(
        -(time**2) / (2.0 * sigma**2)
    )
    return kernel / np.sqrt(np.sum(np.abs(kernel) ** 2))


def morlet_power(signals: np.ndarray, sfreq: float) -> np.ndarray:
    output = np.empty(
        (*signals.shape[:2], len(MORLET_FREQUENCIES), signals.shape[-1]),
        dtype=np.float32,
    )
    cycles = np.linspace(3.0, 8.0, len(MORLET_FREQUENCIES))
    for index, (frequency, cycle_count) in enumerate(
        zip(MORLET_FREQUENCIES, cycles)
    ):
        convolution = fftconvolve(
            signals,
            _morlet_kernel(float(frequency), float(cycle_count), sfreq)[
                None, None, :
            ],
            mode="same",
            axes=-1,
        )
        output[:, :, index] = np.abs(convolution) ** 2
    return output


def _time_bin_means(values: np.ndarray, sfreq: float, edges: np.ndarray) -> np.ndarray:
    blocks = []
    for start, stop in zip(edges[:-1], edges[1:]):
        first = int(round(float(start) * sfreq))
        last = int(round(float(stop) * sfreq))
        blocks.append(np.mean(values[..., first:last], axis=-1))
    return np.stack(blocks, axis=-1)


def _regularized_covariance(signal: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    centered = signal - np.mean(signal, axis=-1, keepdims=True)
    covariance = centered @ centered.T / max(signal.shape[-1] - 1, 1)
    scale = float(np.trace(covariance)) / covariance.shape[0]
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(
        covariance.shape[0]
    )
    covariance /= max(float(np.trace(covariance)), 1e-20)
    return (0.5 * (covariance + covariance.T)).astype(np.float32)


def _feature_names() -> Dict[str, List[str]]:
    slow_names = []
    slow_signals = ("C3", "Cz", "C4", "central_mean", "C3_minus_C4")
    for channel in slow_signals:
        slow_names.extend(
            [
                f"{channel}_slow_{start:.2f}_{stop:.2f}s"
                for start, stop in zip(SLOW_TIME_EDGES[:-1], SLOW_TIME_EDGES[1:])
            ]
        )
        slow_names.extend(
            [f"{channel}_slow_slope", f"{channel}_slow_minimum", f"{channel}_slow_maximum"]
        )
    morlet_names = [
        f"{channel}_{band}_{start:.2f}_{stop:.2f}s_erd_db"
        for channel in MOTOR_CHANNELS
        for band, _, _ in MORLET_BANDS
        for start, stop in zip(TIME_EDGES[:-1], TIME_EDGES[1:])
    ]
    morlet_names.extend(
        [
            f"{channel}_individual_mu_{start:.2f}_{stop:.2f}s_erd_db"
            for channel in MOTOR_CHANNELS
            for start, stop in zip(TIME_EDGES[:-1], TIME_EDGES[1:])
        ]
    )
    csd_names = [
        f"{channel}_csd_{band}_{start:.2f}_{stop:.2f}s_erd_db"
        for channel in MOTOR_CHANNELS
        for band, _, _ in MORLET_BANDS
        for start, stop in zip(TIME_EDGES[:-1], TIME_EDGES[1:])
    ]
    quality_names = [
        "max_peak_to_peak_uv",
        "median_peak_to_peak_uv",
        "median_rms_uv",
        "max_absolute_uv",
        "frontal_max_peak_to_peak_uv",
        "high_frequency_power_ratio",
        "line_power_ratio",
        "interpolated_channel_fraction",
    ]
    return {
        "slow": slow_names,
        "morlet": morlet_names,
        "csd": csd_names,
        "quality": quality_names,
    }


def extract_subject_features(
    subject: int, runs: Mapping[int, RunEpochs]
) -> Dict[str, np.ndarray]:
    car = np.concatenate([runs[run].car for run in IMAGERY_RUNS])
    slow = np.concatenate([runs[run].slow for run in IMAGERY_RUNS])
    csd = np.concatenate([runs[run].csd for run in IMAGERY_RUNS])
    labels = np.concatenate([runs[run].labels for run in IMAGERY_RUNS])
    run_ids = np.concatenate(
        [np.full(len(runs[run].labels), run, dtype=np.int64) for run in IMAGERY_RUNS]
    )
    trial_indices = np.concatenate([runs[run].trial_indices for run in IMAGERY_RUNS])
    quality = np.concatenate([runs[run].quality for run in IMAGERY_RUNS])
    sfreq = next(iter(runs.values())).sfreq
    calibration_rest = np.isin(run_ids, CALIBRATION_RUNS) & (labels == 0)
    reference_time = slice(int(round(0.5 * sfreq)), int(round(3.5 * sfreq)))
    if not np.any(calibration_rest):
        raise RuntimeError(f"Subject S{subject:03d} has no calibration rest trials.")

    slow_reference = np.median(
        slow[calibration_rest, :, reference_time], axis=(0, 2)
    )
    slow_scale = np.std(slow[calibration_rest, :, reference_time], axis=(0, 2))
    slow_z = (slow - slow_reference[None, :, None]) / np.clip(
        slow_scale[None, :, None], 1e-12, None
    )
    slow_audit = _time_bin_means(slow_z, sfreq, SLOW_TIME_EDGES)
    slow_lookup = {
        normalize_channel_name(name): index
        for index, name in enumerate(SLOW_AUDIT_CHANNELS)
    }
    # The slow audit contains only central and occipital channels. Use central
    # trajectories plus their spatial mean for compact decoder features.
    central = [
        slow_lookup[normalize_channel_name(name)] for name in ("C3", "Cz", "C4")
    ]
    central_bins = slow_audit[:, central]
    spatial_mean = np.mean(central_bins, axis=1, keepdims=True)
    spatial_difference = (
        central_bins[:, :1] - central_bins[:, 2:3]
    )
    compact_slow = np.concatenate(
        [central_bins, spatial_mean, spatial_difference], axis=1
    )
    slope_time = np.arange(compact_slow.shape[-1], dtype=np.float64)
    slope_time -= np.mean(slope_time)
    slopes = np.sum(compact_slow * slope_time, axis=-1) / np.sum(slope_time**2)
    slow_features = np.concatenate(
        [
            compact_slow.reshape(len(labels), -1),
            slopes,
            np.min(compact_slow, axis=-1),
            np.max(compact_slow, axis=-1),
        ],
        axis=1,
    ).astype(np.float32)

    power = morlet_power(car, sfreq)
    power_reference = np.median(
        power[calibration_rest, :, :, reference_time], axis=(0, 3)
    )
    ratio_db = 10.0 * np.log10(
        (power + np.finfo(np.float32).tiny)
        / (power_reference[None, :, :, None] + np.finfo(np.float32).tiny)
    )
    morlet_blocks = []
    for _, low, high in MORLET_BANDS:
        selected = (MORLET_FREQUENCIES >= low) & (MORLET_FREQUENCIES < high)
        morlet_blocks.append(
            _time_bin_means(np.mean(ratio_db[:, :, selected], axis=2), sfreq, TIME_EDGES)
        )
    morlet_binned = np.stack(morlet_blocks, axis=2)
    calibration_intent = np.isin(run_ids, CALIBRATION_RUNS) & (labels != 0)
    central_positions = _positions(MOTOR_CHANNELS, ("C3", "Cz", "C4"))
    task_relative = np.mean(
        ratio_db[calibration_intent][:, central_positions, :, reference_time],
        axis=(0, 1, 3),
    )
    mu_candidates = (MORLET_FREQUENCIES >= 8.0) & (MORLET_FREQUENCIES <= 14.0)
    individual_mu = float(
        MORLET_FREQUENCIES[mu_candidates][np.argmin(task_relative[mu_candidates])]
    )
    individual_selected = np.abs(MORLET_FREQUENCIES - individual_mu) <= 2.0
    individual_block = _time_bin_means(
        np.mean(ratio_db[:, :, individual_selected], axis=2), sfreq, TIME_EDGES
    )
    morlet_features = np.concatenate(
        [
            morlet_binned.reshape(len(labels), -1),
            individual_block.reshape(len(labels), -1),
        ],
        axis=1,
    ).astype(np.float32)
    del power, ratio_db

    motor_positions = _positions(next(iter(runs.values())).channel_names, MOTOR_CHANNELS)
    csd_features = np.empty(
        (len(labels), len(MOTOR_CHANNELS), len(MORLET_BANDS), len(TIME_EDGES) - 1),
        dtype=np.float32,
    )
    csd_topography = np.empty(
        (len(labels), csd.shape[1], len(MORLET_BANDS)), dtype=np.float32
    )
    covariances = np.empty(
        (
            len(labels),
            len(FILTER_BANK),
            len(COVARIANCE_PHASES),
            len(MOTOR_CHANNELS),
            len(MOTOR_CHANNELS),
        ),
        dtype=np.float32,
    )
    for band_index, (_, low, high) in enumerate(MORLET_BANDS):
        filtered = sosfiltfilt(
            butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos"),
            csd,
            axis=-1,
        )
        power_values = filtered**2
        reference = np.median(
            power_values[calibration_rest, :, reference_time], axis=(0, 2)
        )
        db = 10.0 * np.log10(
            (power_values + 1e-30) / (reference[None, :, None] + 1e-30)
        )
        binned = _time_bin_means(db, sfreq, TIME_EDGES)
        csd_features[:, :, band_index] = binned[:, motor_positions]
        csd_topography[:, :, band_index] = np.mean(
            db[:, :, reference_time], axis=-1
        )
        del filtered, power_values, db

    for band_index, (_, low, high) in enumerate(FILTER_BANK):
        filtered = sosfiltfilt(
            butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos"),
            car,
            axis=-1,
        )
        for phase_index, (_, start, stop) in enumerate(COVARIANCE_PHASES):
            first = int(round(start * sfreq))
            last = int(round(stop * sfreq))
            for trial in range(len(labels)):
                covariances[trial, band_index, phase_index] = _regularized_covariance(
                    filtered[trial, :, first:last]
                )

    return {
        "slow_features": slow_features,
        "morlet_features": morlet_features,
        "csd_features": csd_features.reshape(len(labels), -1),
        "covariances": covariances,
        "quality_features": quality.astype(np.float32),
        "slow_audit": slow_audit.astype(np.float32),
        "csd_topography": csd_topography,
        "labels": labels,
        "runs": run_ids,
        "trial_indices": trial_indices,
        "individual_mu_hz": np.full(len(labels), individual_mu, dtype=np.float32),
    }


def build_feature_atlas(
    data_dir: Path,
    subjects: Sequence[int],
    cache_path: Path,
    rebuild_cache: bool = False,
) -> Tuple[FeatureAtlasDataset, List[Dict[str, object]]]:
    fingerprint = atlas_fingerprint(subjects)
    if not rebuild_cache:
        cached = load_feature_atlas(cache_path, fingerprint)
        if cached is not None:
            return cached, []

    blocks: Dict[str, List[np.ndarray]] = {
        "slow_features": [],
        "morlet_features": [],
        "csd_features": [],
        "covariances": [],
        "quality_features": [],
        "slow_audit": [],
        "csd_topography": [],
        "labels": [],
        "subjects": [],
        "runs": [],
        "trial_indices": [],
        "individual_mu_hz": [],
    }
    recording_rows: List[Dict[str, object]] = []
    channel_names: Optional[List[str]] = None
    sfreq = 160.0
    print(f"[+] Building physiology feature atlas for {len(subjects)} subjects...")
    for position, subject in enumerate(subjects, start=1):
        run_epochs: Dict[int, RunEpochs] = {}
        for run_id in IMAGERY_RUNS:
            file_path = find_run_file(data_dir, int(subject), run_id)
            if file_path is None:
                raise FileNotFoundError(f"Missing S{int(subject):03d}R{run_id:02d}.edf")
            extracted = extract_run_epochs(file_path, int(subject), run_id)
            run_epochs[run_id] = extracted
            recording_rows.append(extracted.qc)
            if channel_names is None:
                channel_names = extracted.channel_names
                sfreq = extracted.sfreq
        subject_block = extract_subject_features(int(subject), run_epochs)
        count = len(subject_block["labels"])
        for name in blocks:
            if name == "subjects":
                blocks[name].append(np.full(count, int(subject), dtype=np.int64))
            else:
                blocks[name].append(subject_block[name])
        if position % 5 == 0 or position == len(subjects):
            kept = sum(len(values) for values in blocks["labels"])
            print(
                f"    processed {position:3d}/{len(subjects)} subjects; "
                f"derived {kept:5d} trials"
            )
    if channel_names is None:
        raise RuntimeError("No EEG recordings were processed.")
    dataset = FeatureAtlasDataset(
        slow_features=np.concatenate(blocks["slow_features"]),
        morlet_features=np.concatenate(blocks["morlet_features"]),
        csd_features=np.concatenate(blocks["csd_features"]),
        covariances=np.concatenate(blocks["covariances"]),
        quality_features=np.concatenate(blocks["quality_features"]),
        slow_audit=np.concatenate(blocks["slow_audit"]),
        csd_topography=np.concatenate(blocks["csd_topography"]),
        labels=np.concatenate(blocks["labels"]),
        subjects=np.concatenate(blocks["subjects"]),
        runs=np.concatenate(blocks["runs"]),
        trial_indices=np.concatenate(blocks["trial_indices"]),
        individual_mu_hz=np.concatenate(blocks["individual_mu_hz"]),
        channel_names=channel_names,
        feature_names=_feature_names(),
        sfreq=sfreq,
        fingerprint=fingerprint,
    )
    dataset.validate()
    save_feature_atlas(cache_path, dataset)
    return dataset, recording_rows


def label_counts(labels: np.ndarray) -> Dict[str, int]:
    return {
        LABEL_NAMES[label]: int(np.sum(labels == label)) for label in sorted(LABEL_NAMES)
    }
