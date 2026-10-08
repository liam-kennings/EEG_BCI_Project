"""Compact classical EEG features for controlled motor-intent ablation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SPECTRAL_BANDS = (
    ("theta", 4.0, 8.0),
    ("mu", 8.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("high_frequency", 30.0, 38.0),
)
TEMPORAL_BANDS = (
    ("mu", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)
FAMILY_ORDER = ("statistical", "spectral", "temporal", "spatial")
CLASSICAL_MODEL_NAMES = (
    "logistic",
    "rbf_svm",
    "gradient_boosting",
    "random_forest",
    "decision_tree",
)


@dataclass
class ClassicalEEGFeatures:
    values: np.ndarray
    names: list[str]
    family_slices: dict[str, tuple[int, int]]
    fingerprint: str

    def validate(self, rows: int) -> None:
        if self.values.shape != (rows, len(self.names)):
            raise ValueError("Classical feature matrix and names are not aligned.")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Classical EEG features contain non-finite values.")
        if tuple(self.family_slices) != FAMILY_ORDER:
            raise ValueError("Classical EEG feature families are incomplete.")
        previous = 0
        for family in FAMILY_ORDER:
            start, stop = self.family_slices[family]
            if start != previous or stop <= start:
                raise ValueError("Classical feature-family slices are malformed.")
            previous = stop
        if previous != self.values.shape[1]:
            raise ValueError("Feature-family slices do not span the matrix.")

    def family(self, name: str) -> np.ndarray:
        start, stop = self.family_slices[name]
        return self.values[:, start:stop]


class ClassicalIntentDecoder:
    """Binary intent decoder with a shared interface for classical models."""

    def __init__(self, model_name: str, random_seed: int = 42):
        if model_name not in CLASSICAL_MODEL_NAMES:
            raise ValueError(f"Unsupported classical model: {model_name}")
        self.model_name = model_name
        self.random_seed = random_seed

    def _estimator(self):
        if self.model_name == "logistic":
            return LogisticRegression(
                C=0.25,
                max_iter=3000,
                solver="liblinear",
                random_state=self.random_seed,
            )
        if self.model_name == "rbf_svm":
            return SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                cache_size=2000,
                random_state=self.random_seed,
            )
        if self.model_name == "gradient_boosting":
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=self.random_seed,
            )
        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=1,
                random_state=self.random_seed,
            )
        return DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            random_state=self.random_seed,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "ClassicalIntentDecoder":
        labels = (np.asarray(labels) != 0).astype(np.int64)
        self.scaler_ = None
        transformed = features
        if self.model_name in {"logistic", "rbf_svm"}:
            self.scaler_ = StandardScaler().fit(features)
            transformed = self.scaler_.transform(features)
        self.estimator_ = self._estimator()
        self.estimator_.fit(
            transformed,
            labels,
            sample_weight=np.asarray(sample_weight, dtype=np.float64),
        )
        return self

    def predict_intent_probability(self, features: np.ndarray) -> np.ndarray:
        transformed = (
            self.scaler_.transform(features)
            if self.scaler_ is not None
            else features
        )
        classes = self.estimator_.classes_.tolist()
        return self.estimator_.predict_proba(transformed)[:, classes.index(1)]


def _safe_log(values: np.ndarray) -> np.ndarray:
    return np.log(np.clip(values, 1e-30, None))


def _band_integral(
    frequencies: np.ndarray,
    density: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    selected = (frequencies >= low) & (frequencies <= high)
    return trapezoid(
        density[..., selected],
        frequencies[selected],
        axis=-1,
    )


def _peak_features(
    frequencies: np.ndarray,
    density: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = (frequencies >= low) & (frequencies <= high)
    local_frequency = frequencies[selected]
    local_density = density[..., selected]
    peak_indices = np.argmax(local_density, axis=-1)
    peak_frequency = local_frequency[peak_indices]
    peak_power = np.take_along_axis(
        local_density,
        peak_indices[..., None],
        axis=-1,
    )[..., 0]
    background = np.median(local_density, axis=-1)
    peak_snr = 10.0 * np.log10(
        np.clip(peak_power, 1e-30, None)
        / np.clip(background, 1e-30, None)
    )
    return peak_frequency, peak_snr


def _statistical_features(
    epochs: np.ndarray,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    centered = epochs - np.mean(epochs, axis=-1, keepdims=True)
    differences = np.diff(centered, axis=-1)
    second_differences = np.diff(differences, axis=-1)
    variance = np.var(centered, axis=-1)
    difference_variance = np.var(differences, axis=-1)
    mobility = np.sqrt(
        difference_variance / np.clip(variance, 1e-30, None)
    )
    second_mobility = np.sqrt(
        np.var(second_differences, axis=-1)
        / np.clip(difference_variance, 1e-30, None)
    )
    values = np.stack(
        (
            _safe_log(np.std(centered, axis=-1)),
            skew(centered, axis=-1, bias=False),
            kurtosis(centered, axis=-1, fisher=True, bias=False),
            _safe_log(np.mean(np.abs(differences), axis=-1)),
            np.mean(centered[..., 1:] * centered[..., :-1] < 0.0, axis=-1),
            mobility,
            second_mobility / np.clip(mobility, 1e-12, None),
            _safe_log(np.ptp(centered, axis=-1)),
        ),
        axis=-1,
    )
    descriptors = (
        "log_standard_deviation",
        "skewness",
        "excess_kurtosis",
        "log_line_length",
        "zero_crossing_rate",
        "hjorth_mobility",
        "hjorth_complexity",
        "log_peak_to_peak",
    )
    names = [
        f"{channel}_{descriptor}"
        for channel in channel_names
        for descriptor in descriptors
    ]
    return values.reshape(len(epochs), -1), names


def _spectral_features(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    frequencies, density = welch(
        epochs,
        fs=sfreq,
        nperseg=epochs.shape[-1],
        axis=-1,
    )
    selected = (frequencies >= 4.0) & (frequencies <= 38.0)
    selected_density = density[..., selected]
    selected_frequencies = frequencies[selected]
    total_power = trapezoid(
        selected_density,
        selected_frequencies,
        axis=-1,
    )
    normalized = selected_density / np.clip(
        np.sum(selected_density, axis=-1, keepdims=True),
        1e-30,
        None,
    )
    entropy = -np.sum(
        normalized * np.log(np.clip(normalized, 1e-30, None)),
        axis=-1,
    ) / np.log(normalized.shape[-1])
    centroid = np.sum(
        normalized * selected_frequencies[None, None, :],
        axis=-1,
    )

    band_powers = {
        name: _band_integral(frequencies, density, low, high)
        for name, low, high in SPECTRAL_BANDS
    }
    mu_peak, mu_snr = _peak_features(frequencies, density, 8.0, 13.0)
    beta_peak, beta_snr = _peak_features(frequencies, density, 13.0, 30.0)
    per_channel = []
    descriptors = []
    for name, _, _ in SPECTRAL_BANDS:
        per_channel.append(_safe_log(band_powers[name]))
        descriptors.append(f"log_{name}_power")
    for name, _, _ in SPECTRAL_BANDS:
        per_channel.append(
            band_powers[name] / np.clip(total_power, 1e-30, None)
        )
        descriptors.append(f"relative_{name}_power")
    per_channel.extend((entropy, centroid, mu_peak, mu_snr, beta_peak, beta_snr))
    descriptors.extend(
        (
            "spectral_entropy",
            "spectral_centroid_hz",
            "mu_peak_hz",
            "mu_peak_snr_db",
            "beta_peak_hz",
            "beta_peak_snr_db",
        )
    )
    values = np.stack(per_channel, axis=-1)
    names = [
        f"{channel}_{descriptor}"
        for channel in channel_names
        for descriptor in descriptors
    ]
    return values.reshape(len(epochs), -1), names, band_powers


def _temporal_features(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    features = []
    names = []
    band_bins: dict[str, np.ndarray] = {}
    bin_edges = np.linspace(0, epochs.shape[-1], 5, dtype=int)
    time_axis = np.arange(4, dtype=np.float64)
    time_axis -= np.mean(time_axis)
    denominator = np.sum(time_axis**2)
    for band_name, low, high in TEMPORAL_BANDS:
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(sos, epochs, axis=-1)
        envelope_power = np.abs(hilbert(filtered, axis=-1)) ** 2
        bins = np.stack(
            [
                _safe_log(
                    np.mean(
                        envelope_power[..., bin_edges[index] : bin_edges[index + 1]],
                        axis=-1,
                    )
                )
                for index in range(4)
            ],
            axis=-1,
        )
        band_bins[band_name] = bins
        slope = np.sum(bins * time_axis[None, None, :], axis=-1) / denominator
        early_late = np.mean(bins[..., 2:], axis=-1) - np.mean(
            bins[..., :2],
            axis=-1,
        )
        variability = np.std(bins, axis=-1)
        values = np.concatenate(
            (bins, slope[..., None], early_late[..., None], variability[..., None]),
            axis=-1,
        )
        descriptors = (
            *(f"{band_name}_log_power_bin_{index + 1}" for index in range(4)),
            f"{band_name}_trajectory_slope",
            f"{band_name}_late_minus_early",
            f"{band_name}_temporal_variability",
        )
        features.append(values.reshape(len(epochs), -1))
        names.extend(
            f"{channel}_{descriptor}"
            for channel in channel_names
            for descriptor in descriptors
        )
    return np.concatenate(features, axis=1), names, band_bins


def _spatial_features(
    band_powers: Mapping[str, np.ndarray],
    temporal_bins: Mapping[str, np.ndarray],
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    lookup = {name.upper(): index for index, name in enumerate(channel_names)}
    pairs = (("FC3", "FC4"), ("C3", "C4"), ("CP3", "CP4"))
    values = []
    names = []
    for left, right in pairs:
        for band_name, _, _ in SPECTRAL_BANDS:
            power = _safe_log(band_powers[band_name])
            values.append(power[:, lookup[left]] - power[:, lookup[right]])
            names.append(f"{left}_minus_{right}_{band_name}_log_power")
        for band_name, bins in temporal_bins.items():
            for bin_index in range(bins.shape[-1]):
                values.append(
                    bins[:, lookup[left], bin_index]
                    - bins[:, lookup[right], bin_index]
                )
                names.append(
                    f"{left}_minus_{right}_{band_name}_bin_{bin_index + 1}"
                )
    for band_name, _, _ in SPECTRAL_BANDS:
        values.append(np.mean(_safe_log(band_powers[band_name]), axis=1))
        names.append(f"central_mean_{band_name}_log_power")
    return np.column_stack(values), names


def _extract_batch(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    statistical, statistical_names = _statistical_features(epochs, channel_names)
    spectral, spectral_names, band_powers = _spectral_features(
        epochs,
        sfreq,
        channel_names,
    )
    temporal, temporal_names, temporal_bins = _temporal_features(
        epochs,
        sfreq,
        channel_names,
    )
    spatial, spatial_names = _spatial_features(
        band_powers,
        temporal_bins,
        channel_names,
    )
    return (
        {
            "statistical": statistical,
            "spectral": spectral,
            "temporal": temporal,
            "spatial": spatial,
        },
        {
            "statistical": statistical_names,
            "spectral": spectral_names,
            "temporal": temporal_names,
            "spatial": spatial_names,
        },
    )


def _feature_fingerprint(
    dataset_fingerprint: str,
    channel_names: Sequence[str],
    sfreq: float,
) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "channels": list(channel_names),
            "sampling_hz": float(sfreq),
            "families": list(FAMILY_ORDER),
            "spectral_bands": list(SPECTRAL_BANDS),
            "temporal_bands": list(TEMPORAL_BANDS),
            "welch_nperseg": "full_epoch",
            "temporal_bins": 4,
        },
        sort_keys=True,
    )


def build_classical_feature_cache(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
    dataset_fingerprint: str,
    path: Path,
    *,
    rebuild_cache: bool = False,
    batch_size: int = 512,
) -> ClassicalEEGFeatures:
    fingerprint = _feature_fingerprint(
        dataset_fingerprint,
        channel_names,
        sfreq,
    )
    if path.is_file() and not rebuild_cache:
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) == fingerprint:
                result = ClassicalEEGFeatures(
                    values=cached["values"],
                    names=[str(name) for name in cached["names"]],
                    family_slices={
                        family: tuple(
                            int(value)
                            for value in cached[f"{family}_slice"].tolist()
                        )
                        for family in FAMILY_ORDER
                    },
                    fingerprint=fingerprint,
                )
                result.validate(len(epochs))
                print(f"[+] Loaded compact EEG features from {path}")
                return result

    family_blocks: dict[str, list[np.ndarray]] = {
        family: [] for family in FAMILY_ORDER
    }
    names: dict[str, list[str]] | None = None
    print("[+] Extracting compact statistical, spectral, and temporal features...")
    for start in range(0, len(epochs), batch_size):
        stop = min(start + batch_size, len(epochs))
        batch, batch_names = _extract_batch(
            epochs[start:stop],
            sfreq,
            channel_names,
        )
        if names is None:
            names = batch_names
        for family in FAMILY_ORDER:
            family_blocks[family].append(batch[family].astype(np.float32))
        if stop % (5 * batch_size) == 0 or stop == len(epochs):
            print(f"    extracted {stop:,}/{len(epochs):,} windows")
    if names is None:
        raise ValueError("No EEG epochs were supplied.")

    matrices = {
        family: np.concatenate(family_blocks[family], axis=0)
        for family in FAMILY_ORDER
    }
    slices: dict[str, tuple[int, int]] = {}
    all_names: list[str] = []
    cursor = 0
    for family in FAMILY_ORDER:
        stop = cursor + matrices[family].shape[1]
        slices[family] = (cursor, stop)
        cursor = stop
        all_names.extend(names[family])
    result = ClassicalEEGFeatures(
        values=np.concatenate([matrices[family] for family in FAMILY_ORDER], axis=1),
        names=all_names,
        family_slices=slices,
        fingerprint=fingerprint,
    )
    result.validate(len(epochs))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "values": result.values,
        "names": np.asarray(result.names),
        "fingerprint": np.asarray(result.fingerprint),
    }
    payload.update(
        {
            f"{family}_slice": np.asarray(result.family_slices[family])
            for family in FAMILY_ORDER
        }
    )
    np.savez_compressed(path, **payload)
    print(f"[+] Saved compact EEG features: {path}")
    return result


def calibration_robust_alignment(
    features: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    recordings: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Robustly normalize each subject/session from recording-zero windows."""
    aligned = np.empty_like(features, dtype=np.float32)
    groups = 0
    clipped_scales = 0
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        for session in np.unique(sessions[subject_mask]):
            group = subject_mask & (sessions == session)
            calibration = group & (recordings == 0)
            if not np.any(calibration):
                raise ValueError(
                    f"No calibration rows for subject {subject}, session {session}."
                )
            reference = np.median(features[calibration], axis=0)
            q25, q75 = np.quantile(features[calibration], (0.25, 0.75), axis=0)
            robust_scale = (q75 - q25) / 1.349
            fallback = np.std(features[calibration], axis=0)
            scale = np.where(robust_scale > 1e-8, robust_scale, fallback)
            clipped_scales += int(np.sum(scale <= 1e-8))
            scale = np.where(scale > 1e-8, scale, 1.0)
            aligned[group] = (features[group] - reference) / scale
            groups += 1
    if not np.all(np.isfinite(aligned)):
        raise ValueError("Aligned compact EEG features contain non-finite values.")
    return aligned, {
        "alignment": "calibration-only median/IQR",
        "groups": groups,
        "clipped_feature_scales": clipped_scales,
    }
