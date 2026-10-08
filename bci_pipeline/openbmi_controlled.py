"""Controlled OpenBMI decoders, alignment, calibration, and policy selection."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import command_policy_metrics
from .models.fbcsp import BinaryFBCSP
from .openbmi_benchmark import (
    OpenBMIBenchmarkDataset,
    log_euclidean_mean,
    matrix_invsqrt,
    upper_triangle_features,
)
from .representations.riemann import FILTER_BANK


@dataclass
class CovarianceCache:
    covariances: np.ndarray
    filter_bank: tuple[tuple[str, float, float], ...]
    fingerprint: str


@dataclass
class StageProbabilities:
    intent: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class CommandPolicy:
    intent_threshold: float
    direction_margin: float


class HierarchicalFeatureDecoder:
    """Two-stage regularized linear decoder for fixed feature matrices."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    @staticmethod
    def _model(random_seed: int):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=random_seed,
            ),
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "HierarchicalFeatureDecoder":
        labels = np.asarray(labels)
        movement = labels != 0
        intent_fit: dict[str, np.ndarray] = {}
        direction_fit: dict[str, np.ndarray] = {}
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if len(sample_weight) != len(labels):
                raise ValueError("Sample weights must align with feature rows.")
            intent_fit["logisticregression__sample_weight"] = sample_weight
            direction_fit["logisticregression__sample_weight"] = sample_weight[
                movement
            ]
        self.intent_model_ = self._model(self.random_seed).fit(
            features,
            movement.astype(np.int64),
            **intent_fit,
        )
        self.direction_model_ = self._model(self.random_seed + 1).fit(
            features[movement],
            labels[movement],
            **direction_fit,
        )
        return self

    def predict_stage_probabilities(
        self,
        features: np.ndarray,
    ) -> StageProbabilities:
        intent_classes = self.intent_model_.classes_.tolist()
        direction_classes = self.direction_model_.classes_.tolist()
        intent = self.intent_model_.predict_proba(features)[
            :, intent_classes.index(1)
        ]
        direction_raw = self.direction_model_.predict_proba(features)
        direction = np.column_stack(
            (
                direction_raw[:, direction_classes.index(1)],
                direction_raw[:, direction_classes.index(2)],
            )
        )
        return StageProbabilities(intent=intent, direction=direction)


class HierarchicalCSPDecoder:
    """CSP hierarchy supporting the original LDA and regularized LR heads."""

    def __init__(
        self,
        sfreq: float,
        *,
        filter_bank: Sequence[tuple[str, float, float]],
        components_per_band: int = 4,
        classifier: str = "lda",
        random_seed: int = 42,
    ):
        if classifier not in {"lda", "logistic"}:
            raise ValueError("classifier must be 'lda' or 'logistic'.")
        self.sfreq = sfreq
        self.filter_bank = tuple(filter_bank)
        self.components_per_band = components_per_band
        self.classifier = classifier
        self.random_seed = random_seed

    def _model(self):
        if self.classifier == "lda":
            estimator = LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            )
        else:
            estimator = LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=self.random_seed,
            )
        return make_pipeline(StandardScaler(), estimator)

    def fit(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
    ) -> "HierarchicalCSPDecoder":
        movement = labels != 0
        intent_labels = movement.astype(np.int64)
        self.intent_csp_ = BinaryFBCSP(
            self.sfreq,
            self.components_per_band,
            self.filter_bank,
        )
        intent_features = self.intent_csp_.fit_transform(epochs, intent_labels)
        self.intent_model_ = self._model().fit(intent_features, intent_labels)
        self.direction_csp_ = BinaryFBCSP(
            self.sfreq,
            self.components_per_band,
            self.filter_bank,
        )
        direction_features = self.direction_csp_.fit_transform(
            epochs[movement],
            labels[movement],
        )
        self.direction_model_ = self._model().fit(
            direction_features,
            labels[movement],
        )
        return self

    def predict_stage_probabilities(
        self,
        epochs: np.ndarray,
    ) -> StageProbabilities:
        intent_features = self.intent_csp_.transform(epochs)
        direction_features = self.direction_csp_.transform(epochs)
        intent_classes = self.intent_model_.classes_.tolist()
        direction_classes = self.direction_model_.classes_.tolist()
        intent = self.intent_model_.predict_proba(intent_features)[
            :, intent_classes.index(1)
        ]
        direction_raw = self.direction_model_.predict_proba(direction_features)
        direction = np.column_stack(
            (
                direction_raw[:, direction_classes.index(1)],
                direction_raw[:, direction_classes.index(2)],
            )
        )
        return StageProbabilities(intent=intent, direction=direction)


def _covariance_fingerprint(
    dataset: OpenBMIBenchmarkDataset,
    filter_bank: Sequence[tuple[str, float, float]],
) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "dataset_fingerprint": dataset.fingerprint,
            "filter_bank": list(filter_bank),
            "covariance": "trace_normalized_sample_covariance_shrinkage_0.05",
        },
        sort_keys=True,
    )


def _regularized_covariances(epochs: np.ndarray) -> np.ndarray:
    centered = epochs - np.mean(epochs, axis=-1, keepdims=True)
    covariances = np.einsum(
        "nct,ndt->ncd",
        centered,
        centered,
        optimize=True,
    )
    covariances /= max(epochs.shape[-1] - 1, 1)
    channels = epochs.shape[1]
    traces = np.trace(covariances, axis1=1, axis2=2)
    covariances /= np.clip(traces[:, None, None], 1e-12, None)
    identity = np.eye(channels, dtype=covariances.dtype) / channels
    return (0.95 * covariances + 0.05 * identity).astype(np.float32)


def build_covariance_cache(
    dataset: OpenBMIBenchmarkDataset,
    path: Path,
    *,
    filter_bank: Sequence[tuple[str, float, float]] = FILTER_BANK,
    rebuild_cache: bool = False,
    batch_size: int = 256,
) -> CovarianceCache:
    selected_bank = tuple(filter_bank)
    fingerprint = _covariance_fingerprint(dataset, selected_bank)
    if path.is_file() and not rebuild_cache:
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) == fingerprint:
                print(f"[+] Loaded filter-bank covariance cache from {path}")
                return CovarianceCache(
                    covariances=cached["covariances"],
                    filter_bank=selected_bank,
                    fingerprint=fingerprint,
                )

    count, channels = dataset.epochs.shape[:2]
    covariances = np.empty(
        (count, len(selected_bank), channels, channels),
        dtype=np.float32,
    )
    print(f"[+] Computing {len(selected_bank)}-band covariance representations...")
    for band_index, (name, low, high) in enumerate(selected_bank):
        sos = butter(
            4,
            (low, high),
            btype="bandpass",
            fs=dataset.sfreq,
            output="sos",
        )
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            filtered = sosfiltfilt(
                sos,
                dataset.epochs[start:stop],
                axis=-1,
            )
            covariances[start:stop, band_index] = _regularized_covariances(
                filtered
            )
        print(f"    covariance band {band_index + 1}/{len(selected_bank)}: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        covariances=covariances,
        filter_bank=np.asarray(selected_bank),
        fingerprint=np.asarray(fingerprint),
    )
    print(f"[+] Saved covariance cache: {path}")
    return CovarianceCache(
        covariances=covariances,
        filter_bank=selected_bank,
        fingerprint=fingerprint,
    )


def _congruence(
    matrices: np.ndarray,
    transforms: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "...ij,...njk,...lk->...nil",
        transforms,
        matrices,
        transforms,
        optimize=True,
    )


def tangent_features(
    covariances: np.ndarray,
    training_mask: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    recordings: np.ndarray,
    *,
    alignment: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map covariances using calibration-only subject/session alignment."""
    if alignment not in {"none", "euclidean", "log_euclidean"}:
        raise ValueError("Unsupported alignment.")
    aligned = covariances.astype(np.float64, copy=True)
    alignment_groups = 0
    if alignment != "none":
        for subject in np.unique(subjects):
            for session in np.unique(sessions[subjects == subject]):
                group = (subjects == subject) & (sessions == session)
                calibration = group & (recordings == 0)
                if not np.any(calibration):
                    raise ValueError(
                        f"No calibration recording for subject {subject}, "
                        f"session {session}."
                    )
                for band in range(covariances.shape[1]):
                    if alignment == "euclidean":
                        reference = np.mean(
                            covariances[calibration, band],
                            axis=0,
                        )
                    else:
                        reference = log_euclidean_mean(
                            covariances[calibration, band]
                        )
                    whitening = matrix_invsqrt(reference)
                    aligned[group, band] = _congruence(
                        covariances[group, band],
                        whitening,
                    )
                alignment_groups += 1

    blocks: list[np.ndarray] = []
    for band in range(covariances.shape[1]):
        global_reference = np.mean(aligned[training_mask, band], axis=0)
        whitening = matrix_invsqrt(global_reference)
        tangent_covariances = _congruence(aligned[:, band], whitening)
        blocks.append(upper_triangle_features(tangent_covariances))
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    return features, {
        "alignment": alignment,
        "alignment_reference": "calibration recording only",
        "alignment_groups": alignment_groups,
        "feature_count": int(features.shape[1]),
    }


def tangent_features_from_reference(
    covariances: np.ndarray,
    reference_covariances: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Map unseen covariances using a reference fitted before cohort access."""
    values = np.asarray(covariances, dtype=np.float64)
    reference_values = np.asarray(reference_covariances, dtype=np.float64)
    if values.ndim != 4 or reference_values.ndim != 4:
        raise ValueError("Covariances must have rows, bands, channels, channels.")
    if values.shape[1:] != reference_values.shape[1:]:
        raise ValueError("External and reference covariance shapes do not match.")
    if not len(reference_values):
        raise ValueError("At least one frozen-reference covariance is required.")
    blocks: list[np.ndarray] = []
    for band in range(values.shape[1]):
        global_reference = np.mean(reference_values[:, band], axis=0)
        whitening = matrix_invsqrt(global_reference)
        tangent_covariances = _congruence(values[:, band], whitening)
        blocks.append(upper_triangle_features(tangent_covariances))
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    return features, {
        "alignment": "none",
        "reference": "frozen_development_covariance_mean",
        "reference_rows": int(len(reference_values)),
        "feature_count": int(features.shape[1]),
    }


def select_balanced_policy(
    labels: np.ndarray,
    probabilities: StageProbabilities,
) -> CommandPolicy:
    true_intent = labels != 0
    candidates = np.linspace(0.1, 0.9, 161)
    scores = np.asarray(
        [
            balanced_accuracy_score(
                true_intent,
                probabilities.intent >= threshold,
            )
            for threshold in candidates
        ]
    )
    tied = candidates[np.isclose(scores, np.max(scores))]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return CommandPolicy(intent_threshold=threshold, direction_margin=0.0)


def select_safety_policy(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    max_false_activation_rate: float,
) -> CommandPolicy:
    labels = np.asarray(labels)
    true_intent = labels != 0
    direction_predictions = np.argmax(probabilities.direction, axis=1) + 1
    direction_confidence = np.max(probabilities.direction, axis=1)
    candidates: list[tuple[float, float, float, float, float]] = []
    for threshold in np.linspace(0.1, 0.95, 171):
        for margin in np.linspace(0.0, 0.8, 17):
            accepted = (probabilities.intent >= threshold) & (
                direction_confidence >= 0.5 + margin / 2.0
            )
            false_activation = float(np.mean(accepted[~true_intent]))
            if false_activation > max_false_activation_rate + 1e-12:
                continue
            sensitivity = float(np.mean(accepted[true_intent]))
            intent_balanced_accuracy = 0.5 * (
                sensitivity + (1.0 - false_activation)
            )
            predictions = np.where(accepted, direction_predictions, 0)
            recalls = [
                float(np.mean(predictions[labels == label] == label))
                for label in (0, 1, 2)
            ]
            three_class_balanced_accuracy = float(np.mean(recalls))
            candidates.append(
                (
                    float(threshold),
                    float(margin),
                    intent_balanced_accuracy,
                    three_class_balanced_accuracy,
                    1.0 - sensitivity,
                )
            )
    if not candidates:
        return CommandPolicy(intent_threshold=1.0, direction_margin=0.0)
    candidates.sort(
        key=lambda row: (
            row[2],
            row[3],
            -row[4],
            -abs(row[0] - 0.5),
        ),
        reverse=True,
    )
    threshold, margin = candidates[0][:2]
    return CommandPolicy(intent_threshold=threshold, direction_margin=margin)


def evaluate_policy(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    policy: CommandPolicy,
) -> dict[str, object]:
    return command_policy_metrics(
        labels,
        probabilities.intent,
        probabilities.direction,
        policy.intent_threshold,
        policy.direction_margin,
    )


def balanced_calibration_indices(
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    budget_per_class: int,
    *,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    for label in (0, 1, 2):
        candidates = np.flatnonzero(candidate_mask & (labels == label))
        rng.shuffle(candidates)
        selected.extend(candidates[: min(budget_per_class, len(candidates))])
    return np.asarray(sorted(selected), dtype=np.int64)


def locally_adapted_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    evaluation_mask: np.ndarray,
    global_decoder: HierarchicalFeatureDecoder,
    *,
    budget_per_class: int,
    blend_weight: float,
    random_seed: int,
) -> StageProbabilities:
    global_probabilities = global_decoder.predict_stage_probabilities(features)
    intent = global_probabilities.intent.copy()
    direction = global_probabilities.direction.copy()
    for subject in np.unique(subjects[evaluation_mask]):
        subject_evaluation = evaluation_mask & (subjects == subject)
        calibration_pool = (subjects == subject) & (recordings == 0)
        selected = balanced_calibration_indices(
            labels,
            calibration_pool,
            budget_per_class,
            random_seed=random_seed + int(subject),
        )
        if len(np.unique(labels[selected])) < 3:
            continue
        local_decoder = HierarchicalFeatureDecoder(
            random_seed=random_seed + int(subject)
        ).fit(features[selected], labels[selected])
        local = local_decoder.predict_stage_probabilities(
            features[subject_evaluation]
        )
        intent[subject_evaluation] = (
            (1.0 - blend_weight) * intent[subject_evaluation]
            + blend_weight * local.intent
        )
        direction[subject_evaluation] = (
            (1.0 - blend_weight) * direction[subject_evaluation]
            + blend_weight * local.direction
        )
    return StageProbabilities(intent=intent, direction=direction)


def subject_bootstrap_interval(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    subjects: np.ndarray,
    policy: CommandPolicy,
    *,
    repetitions: int = 2000,
    random_seed: int = 42,
) -> dict[str, list[float]]:
    unique_subjects = np.unique(subjects)
    fields = (
        "balanced_accuracy",
        "intent_balanced_accuracy",
        "direction_balanced_accuracy",
        "false_activation_rate",
        "missed_intent_rate",
    )
    values = {field: np.empty(repetitions) for field in fields}
    rng = np.random.default_rng(random_seed)
    for repetition in range(repetitions):
        sampled = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        indices = np.concatenate(
            [np.flatnonzero(subjects == subject) for subject in sampled]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="y_pred contains classes not in y_true",
                category=UserWarning,
            )
            metrics = evaluate_policy(
                labels[indices],
                StageProbabilities(
                    intent=probabilities.intent[indices],
                    direction=probabilities.direction[indices],
                ),
                policy,
            )
        for field in fields:
            values[field][repetition] = float(metrics[field])
    return {
        field: [
            float(np.quantile(field_values, 0.025)),
            float(np.quantile(field_values, 0.975)),
        ]
        for field, field_values in values.items()
    }
