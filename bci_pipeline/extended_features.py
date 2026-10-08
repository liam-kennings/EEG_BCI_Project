"""Development-stage physiological feature extensions for the imagery atlas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .dataset import IMAGERY_RUNS, find_run_file
from .feature_atlas import (
    CALIBRATION_RUNS,
    MORLET_BANDS,
    MOTOR_CHANNELS,
    TIME_EDGES,
    FeatureAtlasDataset,
    RunEpochs,
    extract_run_epochs,
)


EXTENDED_SCHEMA_VERSION = 1
SSD_COMPONENT_COUNT = 3
PHASE_TIME_EDGES = np.arange(0.5, 3.5001, 0.5)
SSD_FIXED_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("beta_low", 14.0, 22.0),
    ("beta_high", 22.0, 32.0),
)
PHASE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("C3", "C4"),
    ("C3", "Cz"),
    ("Cz", "C4"),
    ("FC3", "FC4"),
    ("CP3", "CP4"),
)
LEFT_MOTOR_CHANNELS = ("FC3", "FC1", "C3", "C1", "CP3", "CP1")
RIGHT_MOTOR_CHANNELS = ("FC2", "FC4", "C2", "C4", "CP2", "CP4")
MIDLINE_MOTOR_CHANNELS = ("FCz", "Cz", "CPz")


@dataclass
class ExtendedFeatureDataset:
    ssd_features: np.ndarray
    asymmetry_features: np.ndarray
    morphology_features: np.ndarray
    phase_features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    trial_indices: np.ndarray
    profile_subjects: np.ndarray
    patient_profiles: np.ndarray
    ssd_patterns: np.ndarray
    ssd_eigenvalues: np.ndarray
    feature_names: Dict[str, List[str]]
    profile_names: List[str]
    ssd_band_names: List[str]
    channel_names: List[str]
    fingerprint: str

    def validate(self) -> None:
        trial_count = len(self.labels)
        trial_arrays = (
            self.ssd_features,
            self.asymmetry_features,
            self.morphology_features,
            self.phase_features,
            self.subjects,
            self.runs,
            self.trial_indices,
        )
        if trial_count == 0 or any(len(values) != trial_count for values in trial_arrays):
            raise ValueError("Extended feature arrays do not contain aligned trials.")
        subject_count = len(self.profile_subjects)
        if any(
            len(values) != subject_count
            for values in (
                self.patient_profiles,
                self.ssd_patterns,
                self.ssd_eigenvalues,
            )
        ):
            raise ValueError("Extended subject-level arrays are not aligned.")
        subject_arrays = (
            self.patient_profiles,
            self.ssd_patterns,
            self.ssd_eigenvalues,
        )
        for values in (*trial_arrays, *subject_arrays):
            if not np.all(np.isfinite(values)):
                raise ValueError("Extended feature cache contains non-finite values.")


def extended_fingerprint(atlas: FeatureAtlasDataset) -> str:
    payload = {
        "schema_version": EXTENDED_SCHEMA_VERSION,
        "atlas_fingerprint": atlas.fingerprint,
        "calibration_runs": list(CALIBRATION_RUNS),
        "ssd_components": SSD_COMPONENT_COUNT,
        "ssd_fixed_bands": [list(values) for values in SSD_FIXED_BANDS],
        "phase_pairs": [list(values) for values in PHASE_PAIRS],
        "phase_edges": PHASE_TIME_EDGES.tolist(),
        "hemispheric_channels": {
            "left": list(LEFT_MOTOR_CHANNELS),
            "midline": list(MIDLINE_MOTOR_CHANNELS),
            "right": list(RIGHT_MOTOR_CHANNELS),
        },
        "subject_profile_usage": "audit_only_not_repeated_as_trial_features",
    }
    return json.dumps(payload, sort_keys=True)


def save_extended_features(path: Path, dataset: ExtendedFeatureDataset) -> None:
    dataset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ssd_features=dataset.ssd_features,
        asymmetry_features=dataset.asymmetry_features,
        morphology_features=dataset.morphology_features,
        phase_features=dataset.phase_features,
        labels=dataset.labels,
        subjects=dataset.subjects,
        runs=dataset.runs,
        trial_indices=dataset.trial_indices,
        profile_subjects=dataset.profile_subjects,
        patient_profiles=dataset.patient_profiles,
        ssd_patterns=dataset.ssd_patterns,
        ssd_eigenvalues=dataset.ssd_eigenvalues,
        feature_names=np.asarray(json.dumps(dataset.feature_names)),
        profile_names=np.asarray(json.dumps(dataset.profile_names)),
        ssd_band_names=np.asarray(dataset.ssd_band_names),
        channel_names=np.asarray(dataset.channel_names),
        fingerprint=np.asarray(dataset.fingerprint),
    )


def load_extended_features(
    path: Path, expected_fingerprint: Optional[str] = None
) -> Optional[ExtendedFeatureDataset]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        fingerprint = str(cached["fingerprint"].item())
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            return None
        dataset = ExtendedFeatureDataset(
            ssd_features=cached["ssd_features"],
            asymmetry_features=cached["asymmetry_features"],
            morphology_features=cached["morphology_features"],
            phase_features=cached["phase_features"],
            labels=cached["labels"],
            subjects=cached["subjects"],
            runs=cached["runs"],
            trial_indices=cached["trial_indices"],
            profile_subjects=cached["profile_subjects"],
            patient_profiles=cached["patient_profiles"],
            ssd_patterns=cached["ssd_patterns"],
            ssd_eigenvalues=cached["ssd_eigenvalues"],
            feature_names=json.loads(str(cached["feature_names"].item())),
            profile_names=json.loads(str(cached["profile_names"].item())),
            ssd_band_names=[str(value) for value in cached["ssd_band_names"]],
            channel_names=[str(value) for value in cached["channel_names"]],
            fingerprint=fingerprint,
        )
    dataset.validate()
    return dataset


def _channel_positions(requested: Sequence[str]) -> List[int]:
    lookup = {name: index for index, name in enumerate(MOTOR_CHANNELS)}
    return [lookup[name] for name in requested]


def _morlet_tensors(
    atlas: FeatureAtlasDataset,
) -> Tuple[np.ndarray, List[str]]:
    time_count = len(TIME_EDGES) - 1
    broad_count = len(MOTOR_CHANNELS) * len(MORLET_BANDS) * time_count
    broad = atlas.morlet_features[:, :broad_count].reshape(
        len(atlas.labels), len(MOTOR_CHANNELS), len(MORLET_BANDS), time_count
    )
    individual = atlas.morlet_features[:, broad_count:].reshape(
        len(atlas.labels), len(MOTOR_CHANNELS), 1, time_count
    )
    return (
        np.concatenate((broad, individual), axis=2),
        [name for name, _, _ in MORLET_BANDS] + ["individual_mu"],
    )


def derive_asymmetry_features(
    atlas: FeatureAtlasDataset,
) -> Tuple[np.ndarray, List[str]]:
    """Create signed sensor and hemisphere contrasts before classification."""
    time_count = len(TIME_EDGES) - 1
    morlet, morlet_bands = _morlet_tensors(atlas)
    csd = atlas.csd_features.reshape(
        len(atlas.labels), len(MOTOR_CHANNELS), len(MORLET_BANDS), time_count
    )
    left = _channel_positions(LEFT_MOTOR_CHANNELS)
    right = _channel_positions(RIGHT_MOTOR_CHANNELS)
    midline = _channel_positions(MIDLINE_MOTOR_CHANNELS)
    c3, cz, c4 = _channel_positions(("C3", "Cz", "C4"))
    blocks: List[np.ndarray] = []
    names: List[str] = []
    representations = (
        ("morlet", morlet, morlet_bands),
        ("csd", csd, [name for name, _, _ in MORLET_BANDS]),
    )
    contrast_functions = (
        ("C3_minus_C4", lambda values: values[:, c3] - values[:, c4]),
        (
            "left_minus_right_hemisphere",
            lambda values: np.mean(values[:, left], axis=1)
            - np.mean(values[:, right], axis=1),
        ),
        (
            "bilateral_minus_midline",
            lambda values: 0.5 * (values[:, c3] + values[:, c4])
            - values[:, cz],
        ),
        (
            "lateral_mean_minus_midline",
            lambda values: 0.5
            * (
                np.mean(values[:, left], axis=1)
                + np.mean(values[:, right], axis=1)
            )
            - np.mean(values[:, midline], axis=1),
        ),
    )
    for representation, values, band_names in representations:
        for contrast_name, function in contrast_functions:
            contrast = function(values)
            blocks.append(contrast.reshape(len(atlas.labels), -1))
            names.extend(
                f"{representation}_{contrast_name}_{band}_{start:.2f}_{stop:.2f}s_db"
                for band in band_names
                for start, stop in zip(TIME_EDGES[:-1], TIME_EDGES[1:])
            )
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def _curve_morphology(curves: np.ndarray) -> np.ndarray:
    """Summarize ERD depth, timing, persistence, and recovery."""
    centers = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    minima = np.min(curves, axis=-1)
    minimum_indices = np.argmin(curves, axis=-1)
    time_to_minimum = centers[minimum_indices]
    negative_area = np.mean(np.minimum(curves, 0.0), axis=-1)
    early = np.mean(curves[..., :4], axis=-1)
    late = np.mean(curves[..., -4:], axis=-1)
    recovery = curves[..., -1] - minima
    persistence = np.mean(curves < 0.0, axis=-1)
    return np.stack(
        (
            np.mean(curves, axis=-1),
            minima,
            time_to_minimum,
            negative_area,
            early,
            late,
            recovery,
            persistence,
        ),
        axis=-1,
    )


def derive_morphology_features(
    atlas: FeatureAtlasDataset,
) -> Tuple[np.ndarray, List[str]]:
    morlet, morlet_bands = _morlet_tensors(atlas)
    csd = atlas.csd_features.reshape(
        len(atlas.labels),
        len(MOTOR_CHANNELS),
        len(MORLET_BANDS),
        len(TIME_EDGES) - 1,
    )
    regions = (
        ("left_motor", _channel_positions(LEFT_MOTOR_CHANNELS)),
        ("midline_motor", _channel_positions(MIDLINE_MOTOR_CHANNELS)),
        ("right_motor", _channel_positions(RIGHT_MOTOR_CHANNELS)),
    )
    metrics = (
        "mean_db",
        "minimum_db",
        "time_to_minimum_s",
        "negative_area_db",
        "early_mean_db",
        "late_mean_db",
        "recovery_db",
        "fraction_below_rest",
    )
    blocks: List[np.ndarray] = []
    names: List[str] = []
    for representation, values, bands in (
        ("morlet", morlet, morlet_bands),
        ("csd", csd, [name for name, _, _ in MORLET_BANDS]),
    ):
        for region_name, positions in regions:
            morphology = _curve_morphology(np.mean(values[:, positions], axis=1))
            blocks.append(morphology.reshape(len(atlas.labels), -1))
            names.extend(
                f"{representation}_{region_name}_{band}_{metric}"
                for band in bands
                for metric in metrics
            )
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def _bandpass(
    signal: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    nyquist = sfreq / 2.0
    low = max(0.5, float(low))
    high = min(nyquist - 0.5, float(high))
    if high <= low + 0.25:
        raise ValueError(f"Invalid bandpass range: {low:.2f}-{high:.2f} Hz")
    coefficients = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(coefficients, signal, axis=-1)


def _aggregate_covariance(epochs: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    centered = epochs - np.mean(epochs, axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->cd", centered, centered)
    covariance /= max(epochs.shape[0] * (epochs.shape[-1] - 1), 1)
    scale = float(np.trace(covariance)) / covariance.shape[0]
    covariance = (1.0 - shrinkage) * covariance + shrinkage * scale * np.eye(
        covariance.shape[0]
    )
    return 0.5 * (covariance + covariance.T)


def _ssd_band_definition(
    name: str, low: float, high: float
) -> Tuple[Tuple[float, float], Tuple[Tuple[float, float], ...]]:
    if name == "individual_mu":
        flank_width = 1.5
        return (
            (low, high),
            ((low - 3.5, low - 0.5), (high + 0.5, high + 3.5)),
        )
    return (
        (low, high),
        ((low - 4.0, low - 0.5), (high + 0.5, high + 4.0)),
    )


def _fit_ssd(
    calibration: np.ndarray,
    sfreq: float,
    target_band: Tuple[float, float],
    noise_bands: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = _bandpass(calibration, sfreq, *target_band)
    target_covariance = _aggregate_covariance(target)
    noise_signals = [
        _bandpass(calibration, sfreq, max(4.1, low), min(39.5, high))
        for low, high in noise_bands
        if min(39.5, high) > max(4.1, low) + 0.25
    ]
    noise_covariance = _aggregate_covariance(
        np.concatenate(noise_signals, axis=-1)
    )
    eigenvalues, filters = eigh(
        target_covariance, noise_covariance, check_finite=False
    )
    order = np.argsort(eigenvalues)[::-1][:SSD_COMPONENT_COUNT]
    eigenvalues = eigenvalues[order]
    filters = filters[:, order]
    projected_covariance = filters.T @ target_covariance @ filters
    patterns = target_covariance @ filters @ np.linalg.pinv(projected_covariance)
    for component in range(patterns.shape[1]):
        anchor = int(np.argmax(np.abs(patterns[:, component])))
        if patterns[anchor, component] < 0.0:
            patterns[:, component] *= -1.0
            filters[:, component] *= -1.0
    return filters, patterns, eigenvalues


def extract_ssd_features(
    car: np.ndarray,
    labels: np.ndarray,
    runs: np.ndarray,
    sfreq: float,
    individual_mu: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    calibration = np.isin(runs, CALIBRATION_RUNS)
    calibration_rest = calibration & (labels == 0)
    if not np.any(calibration_rest):
        raise ValueError("SSD extraction requires calibration rest trials.")
    band_definitions = [
        ("individual_mu", individual_mu - 2.0, individual_mu + 2.0),
        *SSD_FIXED_BANDS,
    ]
    output_blocks: List[np.ndarray] = []
    patterns: List[np.ndarray] = []
    eigenvalues: List[np.ndarray] = []
    names: List[str] = []
    reference_slice = slice(int(round(0.5 * sfreq)), int(round(3.5 * sfreq)))
    for band_name, low, high in band_definitions:
        target_band, noise_bands = _ssd_band_definition(band_name, low, high)
        filters, band_patterns, band_eigenvalues = _fit_ssd(
            car[calibration], sfreq, target_band, noise_bands
        )
        filtered = _bandpass(car, sfreq, *target_band)
        components = np.einsum("ck,nct->nkt", filters, filtered)
        power = components**2
        reference = np.median(
            power[calibration_rest, :, reference_slice], axis=(0, 2)
        )
        db = 10.0 * np.log10(
            (power + 1e-30) / (reference[None, :, None] + 1e-30)
        )
        binned = []
        for start, stop in zip(TIME_EDGES[:-1], TIME_EDGES[1:]):
            first = int(round(float(start) * sfreq))
            last = int(round(float(stop) * sfreq))
            binned.append(np.mean(db[..., first:last], axis=-1))
        trajectories = np.stack(binned, axis=-1)
        output_blocks.append(trajectories.reshape(len(labels), -1))
        patterns.append(band_patterns)
        eigenvalues.append(band_eigenvalues)
        names.extend(
            f"ssd_{band_name}_component_{component + 1}_{start:.2f}_{stop:.2f}s_erd_db"
            for component in range(SSD_COMPONENT_COUNT)
            for start, stop in zip(TIME_EDGES[:-1], TIME_EDGES[1:])
        )
    return (
        np.concatenate(output_blocks, axis=1).astype(np.float32),
        np.stack(patterns).astype(np.float32),
        np.stack(eigenvalues).astype(np.float32),
        names,
        [name for name, _, _ in band_definitions],
    )


def extract_phase_features(
    car: np.ndarray,
    sfreq: float,
    individual_mu: float,
) -> Tuple[np.ndarray, List[str]]:
    pair_positions = [
        (MOTOR_CHANNELS.index(first), MOTOR_CHANNELS.index(second))
        for first, second in PHASE_PAIRS
    ]
    blocks: List[np.ndarray] = []
    names: List[str] = []
    for band_name, low, high in (
        ("individual_mu", individual_mu - 2.0, individual_mu + 2.0),
        ("beta_low", 14.0, 22.0),
    ):
        analytic = hilbert(_bandpass(car, sfreq, low, high), axis=-1)
        phase = np.angle(analytic)
        for (first_name, second_name), (first, second) in zip(
            PHASE_PAIRS, pair_positions
        ):
            difference = phase[:, first] - phase[:, second]
            pair_blocks = []
            for start, stop in zip(PHASE_TIME_EDGES[:-1], PHASE_TIME_EDGES[1:]):
                first_sample = int(round(float(start) * sfreq))
                last_sample = int(round(float(stop) * sfreq))
                unit_phase = np.exp(1j * difference[:, first_sample:last_sample])
                pair_blocks.append(
                    np.stack(
                        (
                            np.abs(np.mean(unit_phase, axis=-1)),
                            np.mean(np.sin(difference[:, first_sample:last_sample]), axis=-1),
                        ),
                        axis=-1,
                    )
                )
                names.extend(
                    (
                        f"{band_name}_{first_name}_{second_name}_{start:.2f}_{stop:.2f}s_plv",
                        (
                            f"{band_name}_{first_name}_{second_name}_"
                            f"{start:.2f}_{stop:.2f}s_signed_lag"
                        ),
                    )
                )
            blocks.append(np.stack(pair_blocks, axis=1).reshape(len(car), -1))
    return np.concatenate(blocks, axis=1).astype(np.float32), names


def _peak_and_snr(
    frequencies: np.ndarray, spectrum: np.ndarray, low: float, high: float
) -> Tuple[float, float]:
    selected = (frequencies >= low) & (frequencies <= high)
    local_frequencies = frequencies[selected]
    local_spectrum = spectrum[selected]
    peak_index = int(np.argmax(local_spectrum))
    peak_frequency = float(local_frequencies[peak_index])
    peak_power = float(local_spectrum[peak_index])
    background = np.median(
        spectrum[
            ((frequencies >= low - 3.0) & (frequencies < low))
            | ((frequencies > high) & (frequencies <= high + 3.0))
        ]
    )
    snr_db = 10.0 * np.log10((peak_power + 1e-30) / (float(background) + 1e-30))
    return peak_frequency, float(snr_db)


def extract_patient_profile(
    car: np.ndarray,
    labels: np.ndarray,
    runs: np.ndarray,
    sfreq: float,
    individual_mu: float,
) -> Tuple[np.ndarray, List[str]]:
    central = _channel_positions(("C3", "Cz", "C4"))
    rest = np.isin(runs, CALIBRATION_RUNS) & (labels == 0)
    central_rest = car[rest][:, central]
    frequencies, spectra = welch(
        central_rest,
        fs=sfreq,
        nperseg=min(int(round(4.0 * sfreq)), car.shape[-1]),
        axis=-1,
    )
    spectrum = np.median(spectra, axis=(0, 1))
    mu_peak, mu_snr = _peak_and_snr(frequencies, spectrum, 7.0, 14.0)
    beta_peak, beta_snr = _peak_and_snr(frequencies, spectrum, 14.0, 30.0)
    fit = (
        (frequencies >= 5.0)
        & (frequencies <= 35.0)
        & ~((frequencies >= 7.0) & (frequencies <= 14.0))
        & ~((frequencies >= 18.0) & (frequencies <= 30.0))
    )
    slope, intercept = np.polyfit(
        np.log10(frequencies[fit]), np.log10(spectrum[fit] + 1e-30), 1
    )
    mu_power = np.trapezoid(
        spectrum[(frequencies >= 8.0) & (frequencies < 14.0)],
        frequencies[(frequencies >= 8.0) & (frequencies < 14.0)],
    )
    beta_power = np.trapezoid(
        spectrum[(frequencies >= 14.0) & (frequencies < 30.0)],
        frequencies[(frequencies >= 14.0) & (frequencies < 30.0)],
    )
    run_spectra = []
    run_covariances = []
    for run in CALIBRATION_RUNS:
        selected = (runs == run) & (labels == 0)
        run_signal = car[selected][:, central]
        _, values = welch(
            run_signal,
            fs=sfreq,
            nperseg=min(int(round(4.0 * sfreq)), car.shape[-1]),
            axis=-1,
        )
        run_spectra.append(np.median(values, axis=(0, 1)))
        run_covariances.append(_aggregate_covariance(run_signal))
    spectral_selection = (frequencies >= 5.0) & (frequencies <= 35.0)
    spectral_correlation = float(
        np.corrcoef(
            np.log10(run_spectra[0][spectral_selection] + 1e-30),
            np.log10(run_spectra[1][spectral_selection] + 1e-30),
        )[0, 1]
    )
    normalized_covariances = [
        covariance / max(float(np.trace(covariance)), 1e-30)
        for covariance in run_covariances
    ]
    covariance_drift = float(
        np.linalg.norm(normalized_covariances[0] - normalized_covariances[1], ord="fro")
    )
    profile = np.asarray(
        (
            individual_mu,
            mu_peak,
            mu_snr,
            beta_peak,
            beta_snr,
            slope,
            intercept,
            10.0 * np.log10((mu_power + 1e-30) / (beta_power + 1e-30)),
            np.sqrt(np.mean(central_rest**2)) * 1e6,
            spectral_correlation,
            covariance_drift,
        ),
        dtype=np.float32,
    )
    names = [
        "task_relative_individual_mu_hz",
        "resting_mu_peak_hz",
        "resting_mu_peak_snr_db",
        "resting_beta_peak_hz",
        "resting_beta_peak_snr_db",
        "aperiodic_log_slope",
        "aperiodic_log_intercept",
        "resting_mu_to_beta_ratio_db",
        "central_rest_rms_uv",
        "run4_run8_rest_spectral_correlation",
        "run4_run8_rest_covariance_drift",
    ]
    return profile, names


def _stack_subject_runs(
    runs: Dict[int, RunEpochs],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    car = np.concatenate([runs[run].car for run in IMAGERY_RUNS])
    labels = np.concatenate([runs[run].labels for run in IMAGERY_RUNS])
    run_ids = np.concatenate(
        [np.full(len(runs[run].labels), run, dtype=np.int64) for run in IMAGERY_RUNS]
    )
    trial_indices = np.concatenate([runs[run].trial_indices for run in IMAGERY_RUNS])
    return car, labels, run_ids, trial_indices


def build_extended_features(
    data_dir: Path,
    atlas: FeatureAtlasDataset,
    cache_path: Path,
    rebuild_cache: bool = False,
) -> ExtendedFeatureDataset:
    fingerprint = extended_fingerprint(atlas)
    if not rebuild_cache:
        cached = load_extended_features(cache_path, fingerprint)
        if cached is not None:
            return cached

    asymmetry, asymmetry_names = derive_asymmetry_features(atlas)
    morphology, morphology_names = derive_morphology_features(atlas)
    ssd = np.empty((len(atlas.labels), 108), dtype=np.float32)
    phase = np.empty((len(atlas.labels), 120), dtype=np.float32)
    key_to_row = {
        (int(subject), int(run), int(trial)): row
        for row, (subject, run, trial) in enumerate(
            zip(atlas.subjects, atlas.runs, atlas.trial_indices)
        )
    }
    subjects = np.unique(atlas.subjects).astype(np.int64)
    profile_rows: List[np.ndarray] = []
    pattern_rows: List[np.ndarray] = []
    eigenvalue_rows: List[np.ndarray] = []
    ssd_names: List[str] = []
    phase_names: List[str] = []
    profile_names: List[str] = []
    ssd_band_names: List[str] = []
    print(f"[+] Building extended physiological features for {len(subjects)} subjects...")
    for position, subject in enumerate(subjects, start=1):
        run_epochs: Dict[int, RunEpochs] = {}
        for run in IMAGERY_RUNS:
            file_path = find_run_file(data_dir, int(subject), run)
            if file_path is None:
                raise FileNotFoundError(f"Missing S{int(subject):03d}R{run:02d}.edf")
            run_epochs[run] = extract_run_epochs(file_path, int(subject), run)
        car, labels, run_ids, trial_indices = _stack_subject_runs(run_epochs)
        individual_mu = float(
            np.median(atlas.individual_mu_hz[atlas.subjects == subject])
        )
        (
            subject_ssd,
            subject_patterns,
            subject_eigenvalues,
            current_ssd_names,
            current_band_names,
        ) = extract_ssd_features(
            car,
            labels,
            run_ids,
            run_epochs[IMAGERY_RUNS[0]].sfreq,
            individual_mu,
        )
        subject_phase, current_phase_names = extract_phase_features(
            car, run_epochs[IMAGERY_RUNS[0]].sfreq, individual_mu
        )
        subject_profile, current_profile_names = extract_patient_profile(
            car,
            labels,
            run_ids,
            run_epochs[IMAGERY_RUNS[0]].sfreq,
            individual_mu,
        )
        target_rows = np.asarray(
            [
                key_to_row[(int(subject), int(run), int(trial))]
                for run, trial in zip(run_ids, trial_indices)
            ],
            dtype=np.int64,
        )
        if not np.array_equal(atlas.labels[target_rows], labels):
            raise ValueError(f"Atlas alignment failed for subject S{int(subject):03d}.")
        ssd[target_rows] = subject_ssd
        phase[target_rows] = subject_phase
        profile_rows.append(subject_profile)
        pattern_rows.append(subject_patterns)
        eigenvalue_rows.append(subject_eigenvalues)
        ssd_names = current_ssd_names
        phase_names = current_phase_names
        profile_names = current_profile_names
        ssd_band_names = current_band_names
        if position % 5 == 0 or position == len(subjects):
            print(f"    processed {position:3d}/{len(subjects)} subjects")

    dataset = ExtendedFeatureDataset(
        ssd_features=ssd,
        asymmetry_features=asymmetry,
        morphology_features=morphology,
        phase_features=phase,
        labels=atlas.labels.copy(),
        subjects=atlas.subjects.copy(),
        runs=atlas.runs.copy(),
        trial_indices=atlas.trial_indices.copy(),
        profile_subjects=subjects,
        patient_profiles=np.stack(profile_rows),
        ssd_patterns=np.stack(pattern_rows),
        ssd_eigenvalues=np.stack(eigenvalue_rows),
        feature_names={
            "ssd": ssd_names,
            "asymmetry": asymmetry_names,
            "morphology": morphology_names,
            "phase": phase_names,
        },
        profile_names=profile_names,
        ssd_band_names=ssd_band_names,
        channel_names=list(MOTOR_CHANNELS),
        fingerprint=fingerprint,
    )
    dataset.validate()
    save_extended_features(cache_path, dataset)
    return dataset
