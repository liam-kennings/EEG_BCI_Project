"""Shared trial-level motor-intent and safety metrics."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .dataset import LABEL_NAMES


def hierarchical_metrics(
    labels: np.ndarray,
    intent_probabilities: np.ndarray,
    direction_probabilities: np.ndarray,
    intent_threshold: float,
    label_names: Optional[Mapping[int, str]] = None,
) -> Dict[str, object]:
    labels = np.asarray(labels)
    intent_predictions = np.asarray(intent_probabilities) >= intent_threshold
    direction_predictions = np.argmax(direction_probabilities, axis=1) + 1
    predictions = np.where(intent_predictions, direction_predictions, 0)
    true_intent = labels != 0
    resolved_label_names = LABEL_NAMES if label_names is None else label_names
    label_ids = sorted(resolved_label_names)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "intent_balanced_accuracy": float(balanced_accuracy_score(true_intent, intent_predictions)),
        "direction_balanced_accuracy": float(
            balanced_accuracy_score(labels[true_intent], direction_predictions[true_intent])
        ),
        "false_activation_rate": float(np.mean(intent_predictions[~true_intent]))
        if np.any(~true_intent)
        else 0.0,
        "missed_intent_rate": float(np.mean(~intent_predictions[true_intent]))
        if np.any(true_intent)
        else 0.0,
        "intent_threshold": float(intent_threshold),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_ids).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=[resolved_label_names[label] for label in label_ids],
            zero_division=0,
            output_dict=True,
        ),
    }


def command_policy_metrics(
    labels: np.ndarray,
    intent_probabilities: np.ndarray,
    direction_probabilities: np.ndarray,
    intent_threshold: float,
    direction_margin: float,
) -> Dict[str, object]:
    """Score a two-stage command policy with an explicit direction reject region."""
    labels = np.asarray(labels)
    intent_probabilities = np.asarray(intent_probabilities)
    direction_probabilities = np.asarray(direction_probabilities)
    if direction_probabilities.ndim != 2 or direction_probabilities.shape[1] != 2:
        raise ValueError("Direction probabilities must have shape (trials, 2).")
    if len(labels) != len(intent_probabilities) or len(labels) != len(direction_probabilities):
        raise ValueError("Labels and probability arrays must contain the same number of trials.")

    direction_predictions = np.argmax(direction_probabilities, axis=1) + 1
    direction_confidence = np.max(direction_probabilities, axis=1)
    accepted = (intent_probabilities >= intent_threshold) & (
        direction_confidence >= 0.5 + direction_margin / 2.0
    )
    predictions = np.where(accepted, direction_predictions, 0)
    true_intent = labels != 0
    label_ids = sorted(LABEL_NAMES)
    direction_labels = labels[true_intent]
    direction_score = (
        float(balanced_accuracy_score(direction_labels, direction_predictions[true_intent]))
        if len(np.unique(direction_labels)) == 2
        else 0.0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "intent_balanced_accuracy": float(balanced_accuracy_score(true_intent, accepted)),
        "direction_balanced_accuracy": direction_score,
        "false_activation_rate": float(np.mean(accepted[~true_intent]))
        if np.any(~true_intent)
        else 0.0,
        "missed_intent_rate": float(np.mean(~accepted[true_intent]))
        if np.any(true_intent)
        else 0.0,
        "wrong_direction_rate": float(
            np.mean(accepted[true_intent] & (direction_predictions[true_intent] != labels[true_intent]))
        )
        if np.any(true_intent)
        else 0.0,
        "intent_threshold": float(intent_threshold),
        "direction_margin": float(direction_margin),
        "command_rate": float(np.mean(accepted)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_ids).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_ids,
            target_names=[LABEL_NAMES[label] for label in label_ids],
            zero_division=0,
            output_dict=True,
        ),
    }


def temporal_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    latencies: Sequence[Optional[float]],
    rejection: np.ndarray,
    trial_seconds: float,
) -> Dict[str, object]:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    true_intent, predicted_intent = labels != 0, predictions != 0
    rest_count = max(int(np.sum(~true_intent)), 1)
    movement_count = max(int(np.sum(true_intent)), 1)
    emitted_latencies = [value for value in latencies if value is not None]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "false_activation_rate": float(np.sum(predicted_intent & ~true_intent) / rest_count),
        "missed_intent_rate": float(np.sum(~predicted_intent & true_intent) / movement_count),
        "wrong_direction_rate": float(
            np.sum(predicted_intent & true_intent & (predictions != labels)) / movement_count
        ),
        "false_activations_per_minute": float(
            np.sum(predicted_intent & ~true_intent) / (rest_count * trial_seconds / 60.0)
        ),
        "median_latency_seconds": float(np.median(emitted_latencies)) if emitted_latencies else None,
        "mean_ood_window_rejection_rate": float(np.mean(rejection)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
    }
