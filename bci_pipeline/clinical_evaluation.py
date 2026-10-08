"""Shared evaluation helpers for healthy-to-stroke intent experiments."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from .classical_eeg_features import ClassicalIntentDecoder
from .clinical_transfer import CLINICAL_EPOCH_DURATION_SECONDS, HandIntentDataset
from .splits import SubjectPartitions


PHYSIONET_OFFSET = 0
OPENBMI_OFFSET = 1000
LIU_OFFSET = 2000


def build_transfer_masks(
    dataset: HandIntentDataset,
    partitions: Mapping[str, SubjectPartitions],
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for dataset_name, offset in (
        ("physionet", PHYSIONET_OFFSET),
        ("openbmi", OPENBMI_OFFSET),
        ("liu2024", LIU_OFFSET),
    ):
        partition = partitions[dataset_name]
        groups = {
            "development": np.asarray(partition.development) + offset,
            "controller": np.asarray(partition.controller_validation) + offset,
            "final": np.asarray(partition.final_target) + offset,
        }
        source = dataset.dataset_ids == dataset_name
        for role in ("development", "controller", "final"):
            masks[f"{dataset_name}_{role}_all"] = source & np.isin(
                dataset.subjects,
                groups[role],
            )
        masks[f"{dataset_name}_development"] = masks[
            f"{dataset_name}_development_all"
        ]
        if dataset_name != "liu2024":
            masks[f"{dataset_name}_development"] &= dataset.recordings == 0
        masks[f"{dataset_name}_controller"] = masks[
            f"{dataset_name}_controller_all"
        ] & (dataset.recordings == 1)
        masks[f"{dataset_name}_final"] = masks[
            f"{dataset_name}_final_all"
        ] & (dataset.recordings == 1)
    masks["mixed_development"] = (
        masks["physionet_development"]
        | masks["openbmi_development"]
        | masks["liu2024_development"]
    )
    masks["healthy_final"] = (
        masks["physionet_final"] | masks["openbmi_final"]
    )
    return masks


def transfer_training_weights(
    labels: np.ndarray,
    dataset_ids: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float64)
    datasets = np.unique(dataset_ids[mask])
    total = int(np.sum(mask))
    for dataset_name in datasets:
        selected = mask & (dataset_ids == dataset_name)
        weights[selected] = total / (len(datasets) * int(np.sum(selected)))
    binary = labels != 0
    selected_weights = weights[mask]
    selected_binary = binary[mask]
    for intent_value in (False, True):
        class_weight = np.sum(selected_weights[selected_binary == intent_value])
        weights[mask & (binary == intent_value)] *= (
            np.sum(selected_weights) / (2.0 * class_weight)
        )
    return weights


def select_intent_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_false_activation: float | None,
) -> float:
    true_intent = labels != 0
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = probabilities >= threshold
        false_activation = float(np.mean(predicted[~true_intent]))
        sensitivity = float(np.mean(predicted[true_intent]))
        balanced = 0.5 * (sensitivity + 1.0 - false_activation)
        if (
            max_false_activation is None
            or false_activation <= max_false_activation + 1e-12
        ):
            candidates.append(
                (float(threshold), balanced, sensitivity, false_activation)
            )
    if not candidates:
        return 1.0
    return max(
        candidates,
        key=lambda row: (row[1], row[2], -abs(row[0] - 0.5)),
    )[0]


def score_intent(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    true_intent = labels != 0
    predicted = probabilities >= threshold
    false_activation = float(np.mean(predicted[~true_intent]))
    missed = float(np.mean(~predicted[true_intent]))
    return {
        "rows": int(len(labels)),
        "intent_balanced_accuracy": float(
            balanced_accuracy_score(true_intent, predicted)
        ),
        "intent_sensitivity": 1.0 - missed,
        "intent_specificity": 1.0 - false_activation,
        "false_activation_rate": false_activation,
        "missed_intent_rate": missed,
        "intent_auroc": float(roc_auc_score(true_intent, probabilities)),
        "intent_auprc": float(
            average_precision_score(true_intent, probabilities)
        ),
        "intent_brier_score": float(
            brier_score_loss(true_intent, probabilities)
        ),
        "intent_threshold": float(threshold),
        "estimated_false_activations_per_rest_minute": float(
            false_activation * 60.0 / CLINICAL_EPOCH_DURATION_SECONDS
        ),
    }


def fit_intent_candidate(
    features: np.ndarray,
    dataset: HandIntentDataset,
    masks: Mapping[str, np.ndarray],
    weights: np.ndarray,
    *,
    seed: int,
    max_false_activation: float,
) -> tuple[ClassicalIntentDecoder, np.ndarray, float, dict[str, float | int]]:
    train = masks["mixed_development"]
    controller = masks["liu2024_controller"]
    model = ClassicalIntentDecoder("logistic", seed).fit(
        features[train],
        dataset.labels[train],
        weights[train],
    )
    probabilities = model.predict_intent_probability(features)
    threshold = select_intent_threshold(
        dataset.labels[controller],
        probabilities[controller],
        max_false_activation=max_false_activation,
    )
    return (
        model,
        probabilities,
        threshold,
        score_intent(
            dataset.labels[controller],
            probabilities[controller],
            threshold,
        ),
    )


def _calibration_indices(
    labels: np.ndarray,
    mask: np.ndarray,
    budget_per_class: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    binary = labels != 0
    selected = []
    for intent_value in (False, True):
        candidates = np.flatnonzero(mask & (binary == intent_value))
        rng.shuffle(candidates)
        selected.extend(candidates[: min(budget_per_class, len(candidates))])
    return np.asarray(sorted(selected), dtype=np.int64)


def adapted_intent_probabilities(
    features: np.ndarray,
    dataset: HandIntentDataset,
    evaluation_mask: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    budget_per_class: int,
    blend_weight: float,
    seed: int,
) -> np.ndarray:
    result = global_probabilities.copy()
    for subject in np.unique(dataset.subjects[evaluation_mask]):
        evaluate = evaluation_mask & (dataset.subjects == subject)
        calibration = (dataset.subjects == subject) & (dataset.recordings == 0)
        selected = _calibration_indices(
            dataset.labels,
            calibration,
            budget_per_class,
            seed=seed + int(subject),
        )
        if len(np.unique((dataset.labels[selected] != 0).astype(int))) < 2:
            continue
        local = ClassicalIntentDecoder("logistic", seed + int(subject)).fit(
            features[selected],
            dataset.labels[selected],
            np.ones(len(selected), dtype=np.float64),
        )
        local_probability = local.predict_intent_probability(features[evaluate])
        result[evaluate] = (
            (1.0 - blend_weight) * result[evaluate]
            + blend_weight * local_probability
        )
    return result


def select_patient_calibration(
    features: np.ndarray,
    dataset: HandIntentDataset,
    masks: Mapping[str, np.ndarray],
    global_probabilities: np.ndarray,
    *,
    seed: int,
    max_false_activation: float,
) -> tuple[dict[str, object], np.ndarray, float]:
    controller = masks["liu2024_controller"]
    configurations = []
    for budget in (5, 10, 15):
        for blend in (0.25, 0.5, 0.75):
            probabilities = adapted_intent_probabilities(
                features,
                dataset,
                controller,
                global_probabilities,
                budget_per_class=budget,
                blend_weight=blend,
                seed=seed,
            )
            threshold = select_intent_threshold(
                dataset.labels[controller],
                probabilities[controller],
                max_false_activation=max_false_activation,
            )
            configurations.append(
                {
                    "budget_per_binary_class": budget,
                    "blend_weight": blend,
                    "threshold": threshold,
                    "controller": score_intent(
                        dataset.labels[controller],
                        probabilities[controller],
                        threshold,
                    ),
                }
            )
    selected = max(
        configurations,
        key=lambda row: (
            float(row["controller"]["intent_balanced_accuracy"]),
            float(row["controller"]["intent_sensitivity"]),
            -float(row["controller"]["false_activation_rate"]),
        ),
    )
    final_probabilities = adapted_intent_probabilities(
        features,
        dataset,
        masks["liu2024_final"],
        global_probabilities,
        budget_per_class=int(selected["budget_per_binary_class"]),
        blend_weight=float(selected["blend_weight"]),
        seed=seed,
    )
    return (
        {
            "selected_on_controller": selected,
            "all_controller_configurations": configurations,
        },
        final_probabilities,
        float(selected["threshold"]),
    )


def paired_subject_bootstrap(
    labels: np.ndarray,
    subjects: np.ndarray,
    baseline_probabilities: np.ndarray,
    baseline_threshold: float,
    candidate_probabilities: np.ndarray,
    candidate_threshold: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    unique_subjects = np.unique(subjects)
    fields = (
        "intent_balanced_accuracy",
        "intent_sensitivity",
        "false_activation_rate",
        "missed_intent_rate",
    )
    changes = {field: np.empty(repetitions) for field in fields}
    rng = np.random.default_rng(seed)
    for repetition in range(repetitions):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(subjects == subject) for subject in sampled]
        )
        baseline = score_intent(
            labels[indices],
            baseline_probabilities[indices],
            baseline_threshold,
        )
        candidate = score_intent(
            labels[indices],
            candidate_probabilities[indices],
            candidate_threshold,
        )
        for field in fields:
            changes[field][repetition] = (
                float(candidate[field]) - float(baseline[field])
            )
    return {
        field: {
            "mean_change": float(np.mean(values)),
            "ci_95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for field, values in changes.items()
    }
