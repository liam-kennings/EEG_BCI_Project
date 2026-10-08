"""EMG-confirmed source training for an EEG-only motor-intent student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .translational_features import fixed_smr_features, mrcp_features


def _binary(values: np.ndarray, name: str) -> np.ndarray:
    labels = np.asarray(values).astype(np.int64)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{name} must be a one-dimensional binary array.")
    return labels


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite two-dimensional array.")
    return matrix


def _classifier(seed: int) -> Pipeline:
    return Pipeline(
        (
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        )
    )


def robust_domain_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match source feature medians and IQRs to target-training EEG only."""
    source_values = _finite_matrix(source, "source")
    target_values = _finite_matrix(target, "target")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("Source and target EEG feature counts do not match.")

    source_median = np.median(source_values, axis=0)
    target_median = np.median(target_values, axis=0)
    source_scale = np.subtract(*np.percentile(source_values, (75, 25), axis=0))
    target_scale = np.subtract(*np.percentile(target_values, (75, 25), axis=0))
    source_scale = np.where(source_scale > 1e-12, source_scale, 1.0)
    target_scale = np.where(target_scale > 1e-12, target_scale, 1.0)
    return (
        (source_values - source_median) / source_scale * target_scale
        + target_median
    )


def emg_window_features(
    epochs: np.ndarray,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract amplitude and morphology summaries from synchronized EMG windows."""
    signal = np.asarray(epochs, dtype=np.float64)
    if signal.ndim != 3 or not len(signal) or not np.all(np.isfinite(signal)):
        raise ValueError("EMG epochs must have shape (windows, channels, samples).")
    if signal.shape[-1] < 4:
        raise ValueError("EMG windows need at least four samples.")
    names = tuple(channel_names or (f"EMG{index + 1}" for index in range(signal.shape[1])))
    if len(names) != signal.shape[1]:
        raise ValueError("EMG channel names do not match the epoch channel count.")

    difference = np.diff(signal, axis=-1)
    rms = np.sqrt(np.mean(signal**2, axis=-1))
    mean_absolute = np.mean(np.abs(signal), axis=-1)
    waveform_length = np.sum(np.abs(difference), axis=-1)
    peak_to_peak = np.ptp(signal, axis=-1)
    zero_crossing = np.mean(signal[..., 1:] * signal[..., :-1] < 0.0, axis=-1)
    slope_sign = np.mean(difference[..., 1:] * difference[..., :-1] < 0.0, axis=-1)
    blocks = (rms, mean_absolute, waveform_length, peak_to_peak, zero_crossing, slope_sign)
    suffixes = ("rms", "mean_absolute", "waveform_length", "peak_to_peak", "zero_crossing_rate", "slope_sign_rate")
    feature_names = tuple(
        f"{channel}_{suffix}"
        for suffix in suffixes
        for channel in names
    )
    return np.concatenate(blocks, axis=1).astype(np.float32), feature_names


def shared_eeg_intent_features(
    mrcp_epochs: np.ndarray,
    smr_epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build identically defined MRCP and fixed-SMR source/target features."""
    mrcp_values = np.asarray(mrcp_epochs, dtype=np.float64)
    smr_values = np.asarray(smr_epochs, dtype=np.float64)
    channels = tuple(str(name) for name in channel_names)
    if mrcp_values.shape != smr_values.shape or mrcp_values.ndim != 3:
        raise ValueError("MRCP and SMR epochs must have matching 3-D shapes.")
    if mrcp_values.shape[1] != len(channels):
        raise ValueError("EEG channel names do not match the epoch channel count.")
    mrcp, mrcp_names = mrcp_features(mrcp_values, sfreq, channels)
    smr, smr_names = fixed_smr_features(smr_values, sfreq, channels)
    return (
        np.concatenate((mrcp, smr), axis=1).astype(np.float32),
        tuple(mrcp_names) + tuple(smr_names),
    )


@dataclass(frozen=True)
class PrivilegedSourceDataset:
    """Standardized synchronized source cache used only during training."""

    eeg_features: np.ndarray
    emg_features: np.ndarray
    labels: np.ndarray
    subject_keys: np.ndarray
    premovement_end_seconds: np.ndarray
    eeg_feature_names: tuple[str, ...]
    emg_feature_names: tuple[str, ...]
    dataset_id: str

    def validate(self) -> None:
        eeg = _finite_matrix(self.eeg_features, "source EEG features")
        emg = _finite_matrix(self.emg_features, "source EMG features")
        labels = _binary(self.labels, "source labels")
        count = len(labels)
        if len(emg) != count or len(eeg) != count:
            raise ValueError("Source EEG, EMG, and labels are not row-aligned.")
        if len(self.subject_keys) != count or len(self.premovement_end_seconds) != count:
            raise ValueError("Source metadata are not row-aligned.")
        if eeg.shape[1] != len(self.eeg_feature_names):
            raise ValueError("Source EEG feature names do not align.")
        if emg.shape[1] != len(self.emg_feature_names):
            raise ValueError("Source EMG feature names do not align.")
        end_times = np.asarray(self.premovement_end_seconds, dtype=np.float64)
        if not np.all(np.isfinite(end_times)):
            raise ValueError("Premovement end times must be finite.")
        if np.any(end_times[labels == 1] > 1e-9):
            raise ValueError(
                "Positive source EEG windows must end at or before confirmed movement onset."
            )


@dataclass(frozen=True)
class TransferFitSummary:
    source_rows: int
    accepted_source_rows: int
    source_acceptance_fraction: float
    emg_cross_validated_balanced_accuracy: float
    emg_cross_validated_auroc: float
    source_weight: float
    confidence_floor: float


def _cross_fitted_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    probabilities = np.empty(len(labels), dtype=np.float64)
    if len(unique_groups) >= 2:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        folds = splitter.split(features, labels, groups)
    else:
        class_count = min(int(np.sum(labels == 0)), int(np.sum(labels == 1)))
        if class_count < 2:
            raise ValueError("At least two source examples per class are required.")
        splitter = StratifiedKFold(
            n_splits=min(5, class_count),
            shuffle=True,
            random_state=seed,
        )
        folds = splitter.split(features, labels)
    for fold, (train, test) in enumerate(folds):
        if len(np.unique(labels[train])) != 2:
            raise ValueError("Each EMG verification fold must contain both classes.")
        model = _classifier(seed + fold).fit(features[train], labels[train])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
    return probabilities


class EMGConfirmedEEGTransfer:
    """Train an EEG-only student with EMG as privileged source information."""

    def __init__(
        self,
        *,
        seed: int = 42,
        source_weight: float = 0.35,
        confidence_floor: float = 0.60,
        align_source: bool = True,
    ) -> None:
        if not 0.0 < source_weight <= 1.0:
            raise ValueError("source_weight must be in (0, 1].")
        if not 0.5 <= confidence_floor < 1.0:
            raise ValueError("confidence_floor must be in [0.5, 1).")
        self.seed = seed
        self.source_weight = source_weight
        self.confidence_floor = confidence_floor
        self.align_source = align_source

    def fit(
        self,
        source: PrivilegedSourceDataset,
        target_eeg_features: np.ndarray,
        target_labels: np.ndarray,
        target_feature_names: Sequence[str],
        target_sample_weight: np.ndarray | None = None,
    ) -> "EMGConfirmedEEGTransfer":
        source.validate()
        target_eeg = _finite_matrix(target_eeg_features, "target EEG features")
        target_y = _binary(target_labels, "target labels")
        if len(target_eeg) != len(target_y):
            raise ValueError("Target EEG and labels are not row-aligned.")
        target_weight = (
            np.ones(len(target_y), dtype=np.float64)
            if target_sample_weight is None
            else np.asarray(target_sample_weight, dtype=np.float64)
        )
        if (
            target_weight.ndim != 1
            or len(target_weight) != len(target_y)
            or not np.all(np.isfinite(target_weight))
            or np.any(target_weight <= 0.0)
        ):
            raise ValueError(
                "Target sample weights must be finite, positive, and row-aligned."
            )
        target_names = tuple(str(name) for name in target_feature_names)
        if source.eeg_feature_names != target_names:
            raise ValueError("Source and target EEG feature definitions must match exactly.")

        source_y = np.asarray(source.labels, dtype=np.int64)
        groups = np.asarray(source.subject_keys).astype(str)
        cross_fitted = _cross_fitted_probabilities(
            np.asarray(source.emg_features, dtype=np.float64),
            source_y,
            groups,
            self.seed,
        )
        confirmation = np.where(source_y == 1, cross_fitted, 1.0 - cross_fitted)
        accepted = confirmation >= self.confidence_floor
        if len(np.unique(source_y[accepted])) != 2:
            raise ValueError(
                "The EMG confidence floor rejected an entire source class; lower it or inspect EMG quality."
            )

        self.emg_verifier_ = _classifier(self.seed).fit(
            source.emg_features,
            source_y,
        )
        source_eeg = np.asarray(source.eeg_features, dtype=np.float64)[accepted]
        if self.align_source:
            source_eeg = robust_domain_align(source_eeg, target_eeg)
        combined_eeg = np.concatenate((target_eeg, source_eeg), axis=0)
        combined_y = np.concatenate((target_y, source_y[accepted]), axis=0)
        accepted_groups = groups[accepted]
        group_names, group_counts = np.unique(accepted_groups, return_counts=True)
        group_mass = {
            group: len(accepted_groups) / (len(group_names) * count)
            for group, count in zip(group_names, group_counts, strict=True)
        }
        source_subject_weight = np.asarray(
            [group_mass[group] for group in accepted_groups],
            dtype=np.float64,
        )
        sample_weight = np.concatenate(
            (
                target_weight,
                self.source_weight
                * confirmation[accepted]
                * source_subject_weight,
            )
        )
        self.target_only_model_ = _classifier(self.seed + 1000).fit(
            target_eeg,
            target_y,
            model__sample_weight=target_weight,
        )
        self.student_model_ = _classifier(self.seed + 2000).fit(
            combined_eeg,
            combined_y,
            model__sample_weight=sample_weight,
        )
        self.feature_names_ = target_names
        self.summary_ = TransferFitSummary(
            source_rows=len(source_y),
            accepted_source_rows=int(np.sum(accepted)),
            source_acceptance_fraction=float(np.mean(accepted)),
            emg_cross_validated_balanced_accuracy=float(
                balanced_accuracy_score(source_y, cross_fitted >= 0.5)
            ),
            emg_cross_validated_auroc=float(roc_auc_score(source_y, cross_fitted)),
            source_weight=self.source_weight,
            confidence_floor=self.confidence_floor,
        )
        return self

    def _validate_prediction(self, eeg_features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "student_model_"):
            raise RuntimeError("The transfer model has not been fitted.")
        values = _finite_matrix(eeg_features, "EEG features")
        if values.shape[1] != len(self.feature_names_):
            raise ValueError("EEG feature count does not match the fitted student.")
        return values

    def predict_intent_probability(self, eeg_features: np.ndarray) -> np.ndarray:
        """Predict from EEG only; EMG is intentionally absent from this API."""
        values = self._validate_prediction(eeg_features)
        return self.student_model_.predict_proba(values)[:, 1]

    def predict_target_only_probability(self, eeg_features: np.ndarray) -> np.ndarray:
        """Return the matched pure-target EEG baseline."""
        values = self._validate_prediction(eeg_features)
        return self.target_only_model_.predict_proba(values)[:, 1]
