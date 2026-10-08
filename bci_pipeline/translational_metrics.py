"""Operating-point metrics for abstaining trial and continuous EEG decoders."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from .openbmi_controlled import StageProbabilities


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(selected):
            error += (
                np.sum(selected)
                / total
                * abs(
                    float(np.mean(probabilities[selected]))
                    - float(np.mean(labels[selected]))
                )
            )
    return float(error)


def select_window_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    available: np.ndarray,
    *,
    max_false_activation_rate: float,
) -> float:
    true_intent = np.asarray(labels) != 0
    probabilities = np.asarray(probabilities)
    available = np.asarray(available, dtype=bool)
    candidates = []
    for threshold in np.linspace(0.05, 0.99, 189):
        predicted = available & (probabilities >= threshold)
        rest = ~true_intent
        false_activation = float(np.mean(predicted[rest]))
        sensitivity = float(np.mean(predicted[true_intent]))
        if false_activation <= max_false_activation_rate + 1e-12:
            candidates.append((float(threshold), sensitivity, false_activation))
    if not candidates:
        return 1.0
    return max(
        candidates,
        key=lambda value: (value[1], -value[2], -abs(value[0] - 0.5)),
    )[0]


def select_sensitivity_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    available: np.ndarray,
    *,
    target_sensitivity: float,
) -> float:
    """Find the most specific threshold that reaches target sensitivity."""
    if not 0.0 < target_sensitivity <= 1.0:
        raise ValueError("target_sensitivity must be between zero and one.")
    true_intent = np.asarray(labels) != 0
    probabilities = np.asarray(probabilities)
    available = np.asarray(available, dtype=bool)
    candidates = []
    for threshold in np.linspace(0.05, 0.99, 189):
        predicted = available & (probabilities >= threshold)
        sensitivity = float(np.mean(predicted[true_intent]))
        false_activation = float(np.mean(predicted[~true_intent]))
        if sensitivity + 1e-12 >= target_sensitivity:
            candidates.append(
                (float(threshold), false_activation, sensitivity)
            )
    if not candidates:
        return 0.05
    return min(
        candidates,
        key=lambda value: (value[1], -value[0], -value[2]),
    )[0]


def domain_threshold_vector(
    domains: np.ndarray,
    thresholds: Mapping[str, float],
    fallback: float,
) -> np.ndarray:
    """Expand a dataset-threshold mapping into row-aligned thresholds."""
    domains = np.asarray(domains).astype(str)
    return np.asarray(
        [float(thresholds.get(domain, fallback)) for domain in domains],
        dtype=np.float64,
    )


def score_operating_point(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    available: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, float | int | str]:
    labels = np.asarray(labels)
    available = np.asarray(available, dtype=bool)
    true_intent = labels != 0
    threshold_values = np.asarray(threshold, dtype=np.float64)
    if threshold_values.ndim == 0:
        applied_threshold: float | str = float(threshold_values)
    else:
        if len(threshold_values) != len(labels):
            raise ValueError("Row-specific thresholds must align with labels.")
        applied_threshold = "row_specific"
    predicted_intent = available & (
        probabilities.intent >= threshold_values
    )
    rest = ~true_intent
    intent = true_intent
    false_activation = float(np.mean(predicted_intent[rest]))
    sensitivity = float(np.mean(predicted_intent[intent]))
    direction_selected = intent & predicted_intent
    if np.any(direction_selected):
        direction_prediction = (
            np.argmax(probabilities.direction[direction_selected], axis=1) + 1
        )
        direction_accuracy = float(
            np.mean(direction_prediction == labels[direction_selected])
        )
        direction_classes = set(np.unique(labels[direction_selected]))
        direction_balanced: float | str = (
            float(
                balanced_accuracy_score(
                    labels[direction_selected],
                    direction_prediction,
                )
            )
            if direction_classes == {1, 2}
            else "not_evaluable_single_class"
        )
    else:
        direction_accuracy = 0.0
        direction_balanced = "not_evaluable_no_accepted_intent"
    binary = true_intent.astype(np.int64)
    return {
        "rows": int(len(labels)),
        "intent_threshold": applied_threshold,
        "intent_balanced_accuracy": float(
            balanced_accuracy_score(true_intent, predicted_intent)
        ),
        "intent_sensitivity": sensitivity,
        "intent_specificity": 1.0 - false_activation,
        "false_activation_rate_per_window": false_activation,
        "missed_intent_rate": 1.0 - sensitivity,
        "available_fraction": float(np.mean(available)),
        "abstained_fraction": float(np.mean(~available)),
        "conditional_direction_accuracy": direction_accuracy,
        "conditional_direction_balanced_accuracy": direction_balanced,
        "intent_auroc": float(roc_auc_score(binary, probabilities.intent)),
        "intent_auprc": float(
            average_precision_score(binary, probabilities.intent)
        ),
        "intent_brier_score": float(
            brier_score_loss(binary, probabilities.intent)
        ),
        "intent_expected_calibration_error": expected_calibration_error(
            binary,
            probabilities.intent,
        ),
        "continuous_false_activations_per_minute": "not_evaluated",
    }


def score_intent_operating_point(
    labels: np.ndarray,
    probabilities: np.ndarray,
    available: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, float | int | str]:
    """Score a strict binary rest-versus-intent decoder."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    true_intent = labels != 0
    threshold_values = np.asarray(threshold, dtype=np.float64)
    if threshold_values.ndim == 0:
        applied_threshold: float | str = float(threshold_values)
    else:
        if len(threshold_values) != len(labels):
            raise ValueError("Row-specific thresholds must align with labels.")
        applied_threshold = "row_specific"
    predicted_intent = available & (probabilities >= threshold_values)
    rest = ~true_intent
    false_activation = float(np.mean(predicted_intent[rest]))
    sensitivity = float(np.mean(predicted_intent[true_intent]))
    binary = true_intent.astype(np.int64)
    return {
        "rows": int(len(labels)),
        "intent_threshold": applied_threshold,
        "intent_balanced_accuracy": float(
            balanced_accuracy_score(true_intent, predicted_intent)
        ),
        "intent_sensitivity": sensitivity,
        "intent_specificity": 1.0 - false_activation,
        "false_activation_rate_per_window": false_activation,
        "missed_intent_rate": 1.0 - sensitivity,
        "available_fraction": float(np.mean(available)),
        "abstained_fraction": float(np.mean(~available)),
        "intent_auroc": float(roc_auc_score(binary, probabilities)),
        "intent_auprc": float(average_precision_score(binary, probabilities)),
        "intent_brier_score": float(brier_score_loss(binary, probabilities)),
        "intent_expected_calibration_error": expected_calibration_error(
            binary,
            probabilities,
        ),
        "continuous_false_activations_per_minute": "not_evaluated",
    }


def subject_bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    available: np.ndarray,
    thresholds: float | np.ndarray,
    subjects: np.ndarray,
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap complete subjects to retain within-subject dependence."""
    labels = np.asarray(labels)
    available = np.asarray(available, dtype=bool)
    subjects = np.asarray(subjects).astype(str)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError("Subject bootstrap requires at least two subjects.")
    metrics = (
        "intent_balanced_accuracy",
        "intent_sensitivity",
        "intent_specificity",
        "conditional_direction_balanced_accuracy",
        "intent_auroc",
        "intent_auprc",
    )
    point = score_operating_point(
        labels,
        probabilities,
        available,
        threshold_values,
    )
    samples = {metric: [] for metric in metrics}
    indices = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected_subjects = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        selected = np.concatenate(
            [indices[subject] for subject in selected_subjects]
        )
        local_thresholds: float | np.ndarray = (
            float(threshold_values)
            if threshold_values.ndim == 0
            else threshold_values[selected]
        )
        result = score_operating_point(
            labels[selected],
            StageProbabilities(
                probabilities.intent[selected],
                probabilities.direction[selected],
            ),
            available[selected],
            local_thresholds,
        )
        for metric in metrics:
            value = result[metric]
            if isinstance(value, (float, int)):
                samples[metric].append(float(value))
    intervals = {}
    for metric in metrics:
        values = np.asarray(samples[metric], dtype=np.float64)
        intervals[metric] = {
            "estimate": float(point[metric]),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
    return intervals


def subject_bootstrap_intent_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    available: np.ndarray,
    thresholds: float | np.ndarray,
    subjects: np.ndarray,
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap binary intent metrics over complete participants."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    subjects = np.asarray(subjects).astype(str)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError("Subject bootstrap requires at least two subjects.")
    metrics = (
        "intent_balanced_accuracy",
        "intent_sensitivity",
        "intent_specificity",
        "intent_auroc",
        "intent_auprc",
    )
    point = score_intent_operating_point(
        labels,
        probabilities,
        available,
        threshold_values,
    )
    samples = {metric: [] for metric in metrics}
    indices = {
        subject: np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        selected_subjects = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        selected = np.concatenate(
            [indices[subject] for subject in selected_subjects]
        )
        local_thresholds: float | np.ndarray = (
            float(threshold_values)
            if threshold_values.ndim == 0
            else threshold_values[selected]
        )
        result = score_intent_operating_point(
            labels[selected],
            probabilities[selected],
            available[selected],
            local_thresholds,
        )
        for metric in metrics:
            samples[metric].append(float(result[metric]))
    return {
        metric: {
            "estimate": float(point[metric]),
            "lower_95": float(np.quantile(samples[metric], 0.025)),
            "upper_95": float(np.quantile(samples[metric], 0.975)),
        }
        for metric in metrics
    }


def continuous_false_activations(
    probabilities: np.ndarray,
    available: np.ndarray,
    groups: Sequence[str],
    onsets_seconds: np.ndarray,
    threshold: float,
    *,
    confirmations: int = 2,
    refractory_seconds: float = 2.0,
    window_seconds: float = 2.0,
) -> dict[str, float | int]:
    """Count discrete commands in explicitly continuous no-control recordings."""
    probabilities = np.asarray(probabilities)
    available = np.asarray(available, dtype=bool)
    groups = np.asarray(groups)
    onsets_seconds = np.asarray(onsets_seconds, dtype=float)
    events = 0
    total_seconds = 0.0
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        order = selected[np.argsort(onsets_seconds[selected])]
        if not len(order):
            continue
        total_seconds += (
            float(onsets_seconds[order[-1]] - onsets_seconds[order[0]])
            + window_seconds
        )
        consecutive = 0
        last_event = -np.inf
        for index in order:
            active = available[index] and probabilities[index] >= threshold
            consecutive = consecutive + 1 if active else 0
            onset = float(onsets_seconds[index])
            if (
                consecutive >= confirmations
                and onset - last_event >= refractory_seconds
            ):
                events += 1
                last_event = onset
                consecutive = 0
    minutes = total_seconds / 60.0
    return {
        "false_activation_events": int(events),
        "rest_minutes": float(minutes),
        "false_activations_per_minute": float(events / max(minutes, 1e-12)),
        "confirmations": int(confirmations),
        "refractory_seconds": float(refractory_seconds),
    }


def select_continuous_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    available: np.ndarray,
    intent_mask: np.ndarray,
    hard_negative_mask: np.ndarray,
    groups: Sequence[str],
    onsets_seconds: np.ndarray,
    *,
    max_false_activations_per_minute: float,
    confirmations: int = 2,
    refractory_seconds: float = 2.0,
) -> tuple[float, dict[str, float | int] | None]:
    """Maximize intent sensitivity under a measured continuous-no-control budget."""
    if not np.any(hard_negative_mask):
        return 1.0, None
    candidates = []
    for threshold in np.linspace(0.05, 0.99, 189):
        false_activations = continuous_false_activations(
            probabilities[hard_negative_mask],
            available[hard_negative_mask],
            np.asarray(groups)[hard_negative_mask],
            onsets_seconds[hard_negative_mask],
            float(threshold),
            confirmations=confirmations,
            refractory_seconds=refractory_seconds,
        )
        if (
            float(false_activations["false_activations_per_minute"])
            > max_false_activations_per_minute + 1e-12
        ):
            continue
        sensitivity = float(
            np.mean(
                available[intent_mask]
                & (probabilities[intent_mask] >= threshold)
            )
        )
        candidates.append(
            (
                float(threshold),
                sensitivity,
                float(false_activations["false_activations_per_minute"]),
                false_activations,
            )
        )
    if not candidates:
        fallback = continuous_false_activations(
            probabilities[hard_negative_mask],
            available[hard_negative_mask],
            np.asarray(groups)[hard_negative_mask],
            onsets_seconds[hard_negative_mask],
            1.0,
            confirmations=confirmations,
            refractory_seconds=refractory_seconds,
        )
        return 1.0, fallback
    selected = max(candidates, key=lambda value: (value[1], -value[2], value[0]))
    return selected[0], selected[3]
