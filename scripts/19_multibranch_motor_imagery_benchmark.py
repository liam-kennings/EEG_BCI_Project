#!/usr/bin/env python3
"""Train and evaluate a locked multibranch motor-imagery intent decoder."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.dataset import LABEL_NAMES  # noqa: E402
from bci_pipeline.feature_atlas import FeatureAtlasDataset, load_feature_atlas  # noqa: E402
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.splits import SubjectPartitions, load_partitions  # noqa: E402


BRANCH_COLORS = {
    "slow": "#6C757D",
    "morlet": "#D66A3A",
    "csd": "#147D92",
    "covariance": "#2D6A4F",
    "fusion": "#E9A23B",
    "adapted": "#8E5572",
}


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F8FAFC",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def locked_masks(
    dataset: FeatureAtlasDataset, partitions: SubjectPartitions
) -> Dict[str, np.ndarray]:
    return {
        "development": np.isin(dataset.subjects, partitions.development),
        "validation": np.isin(dataset.subjects, partitions.controller_validation)
        & (dataset.runs == 12),
        "test": np.isin(dataset.subjects, partitions.final_target)
        & (dataset.runs == 12),
    }


def validate_masks(masks: Mapping[str, np.ndarray], labels: np.ndarray) -> None:
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"The atlas contains no {name} trials for the fixed split.")
        if len(np.unique(labels[mask])) != 3:
            raise ValueError(f"The {name} split does not contain all three classes.")


def _matrix_invsqrt(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    values = np.clip(values, 1e-10, None)
    return (vectors * (1.0 / np.sqrt(values))[..., None, :]) @ np.swapaxes(
        vectors, -1, -2
    )


def _log_upper_features(matrices: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrices)
    values = np.clip(values, 1e-10, None)
    logarithms = (vectors * np.log(values)[..., None, :]) @ np.swapaxes(
        vectors, -1, -2
    )
    channel_count = matrices.shape[-1]
    rows, columns = np.triu_indices(channel_count)
    features = logarithms[..., rows, columns]
    off_diagonal = rows != columns
    features[..., off_diagonal] *= np.sqrt(2.0)
    return features


def aligned_covariance_features(
    dataset: FeatureAtlasDataset, cache_path: Path, rebuild: bool
) -> np.ndarray:
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) == dataset.fingerprint:
                features = cached["features"]
                if len(features) == len(dataset.labels):
                    print(f"[+] Loaded {features.shape[1]} aligned covariance features.")
                    return features
    covariances = dataset.covariances.reshape(
        len(dataset.labels),
        -1,
        dataset.covariances.shape[-1],
        dataset.covariances.shape[-1],
    ).astype(np.float64)
    output = np.empty(
        (
            len(dataset.labels),
            covariances.shape[1],
            covariances.shape[-1] * (covariances.shape[-1] + 1) // 2,
        ),
        dtype=np.float32,
    )
    print("[+] Euclidean-aligning and log-mapping covariance trajectories...")
    for position, subject in enumerate(np.unique(dataset.subjects), start=1):
        selected = dataset.subjects == subject
        calibration = selected & np.isin(dataset.runs, (4, 8))
        reference = np.mean(covariances[calibration], axis=0)
        reference = 0.5 * (reference + np.swapaxes(reference, -1, -2))
        whitening = _matrix_invsqrt(reference)
        aligned = whitening[None] @ covariances[selected] @ whitening[None]
        aligned = 0.5 * (aligned + np.swapaxes(aligned, -1, -2))
        output[selected] = _log_upper_features(aligned).astype(np.float32)
        if position % 20 == 0 or position == len(np.unique(dataset.subjects)):
            print(
                f"    aligned {position:3d}/{len(np.unique(dataset.subjects))} subjects"
            )
    features = output.reshape(len(dataset.labels), -1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        fingerprint=np.asarray(dataset.fingerprint),
    )
    return features


def make_model(feature_count: int, selected_count: int, c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("selection", SelectKBest(f_classif, k=min(selected_count, feature_count))),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def intent_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, float]:
    truth = labels != 0
    prediction = probabilities >= threshold
    rest = ~truth
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "false_activation_rate": float(np.mean(prediction[rest])),
        "missed_intent_rate": float(np.mean(~prediction[truth])),
        "intent_sensitivity": float(np.mean(prediction[truth])),
        "threshold": float(threshold),
    }


def balanced_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 161)
    scores = [
        balanced_accuracy_score(labels != 0, probabilities >= threshold)
        for threshold in candidates
    ]
    maximum = np.max(scores)
    tied = candidates[np.isclose(scores, maximum)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def safety_threshold(
    labels: np.ndarray, probabilities: np.ndarray, maximum_false_activation: float
) -> float:
    rest = probabilities[labels == 0]
    return float(
        np.nextafter(
            np.quantile(rest, 1.0 - maximum_false_activation, method="higher"),
            np.inf,
        )
    )


def command_metrics(
    labels: np.ndarray,
    intent_probability: np.ndarray,
    direction_probability: np.ndarray,
    threshold: float,
    allowed: np.ndarray | None = None,
) -> Dict[str, object]:
    if allowed is None:
        allowed = np.ones(len(labels), dtype=bool)
    adjusted = np.where(allowed, intent_probability, 0.0)
    metrics = hierarchical_metrics(
        labels, adjusted, direction_probability, threshold, LABEL_NAMES
    )
    metrics["intent_sensitivity"] = 1.0 - float(metrics["missed_intent_rate"])
    metrics["rejected_trial_rate"] = float(np.mean(~allowed))
    return metrics


def fit_branch(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    masks: Mapping[str, np.ndarray],
    seed: int,
) -> Dict[str, object]:
    candidates = []
    feature_counts = sorted(
        {min(value, features.shape[1]) for value in (60, 120, 240)}
    )
    for selected_count, c_value in itertools.product(feature_counts, (0.05, 0.20)):
        intent_model = make_model(features.shape[1], selected_count, c_value, seed)
        intent_model.fit(features[masks["development"]], labels[masks["development"]] != 0)
        probability = intent_model.predict_proba(features[masks["validation"]])[:, 1]
        threshold = balanced_threshold(labels[masks["validation"]], probability)
        metrics = intent_metrics(labels[masks["validation"]], probability, threshold)
        candidates.append(
            {
                "selected_count": selected_count,
                "c_value": c_value,
                "model": intent_model,
                "threshold": threshold,
                "metrics": metrics,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["balanced_accuracy"],
            -row["metrics"]["false_activation_rate"],
            -row["selected_count"],
        ),
    )
    movement_train = masks["development"] & (labels != 0)
    direction_model = make_model(
        features.shape[1],
        int(selected["selected_count"]),
        float(selected["c_value"]),
        seed + 100,
    )
    direction_model.fit(features[movement_train], labels[movement_train])
    intent_all = selected["model"].predict_proba(features)[:, 1]
    direction_all = direction_model.predict_proba(features)
    validation_metrics = command_metrics(
        labels[masks["validation"]],
        intent_all[masks["validation"]],
        direction_all[masks["validation"]],
        float(selected["threshold"]),
    )
    final_metrics = command_metrics(
        labels[masks["test"]],
        intent_all[masks["test"]],
        direction_all[masks["test"]],
        float(selected["threshold"]),
    )
    print(
        f"    {name:10s} | controller intent BA="
        f"{validation_metrics['intent_balanced_accuracy'] * 100:5.1f}% | "
        f"final={final_metrics['intent_balanced_accuracy'] * 100:5.1f}%"
    )
    return {
        "name": name,
        "intent_model": selected["model"],
        "direction_model": direction_model,
        "intent_probability": intent_all,
        "direction_probability": direction_all,
        "selected_count": int(selected["selected_count"]),
        "c_value": float(selected["c_value"]),
        "threshold": float(selected["threshold"]),
        "controller_validation": validation_metrics,
        "final": final_metrics,
    }


def select_fusion(
    branch_results: Mapping[str, Mapping[str, object]],
    labels: np.ndarray,
    validation_mask: np.ndarray,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    names = list(branch_results)
    frontier = []
    for count in range(1, len(names) + 1):
        for subset in itertools.combinations(names, count):
            intent = np.mean(
                [branch_results[name]["intent_probability"] for name in subset],
                axis=0,
            )
            direction = np.mean(
                [branch_results[name]["direction_probability"] for name in subset],
                axis=0,
            )
            threshold = balanced_threshold(labels[validation_mask], intent[validation_mask])
            metrics = command_metrics(
                labels[validation_mask],
                intent[validation_mask],
                direction[validation_mask],
                threshold,
            )
            frontier.append(
                {
                    "branches": list(subset),
                    "threshold": threshold,
                    "controller_validation": metrics,
                }
            )
    selected = max(
        frontier,
        key=lambda row: (
            row["controller_validation"]["intent_balanced_accuracy"],
            row["controller_validation"]["balanced_accuracy"],
            -row["controller_validation"]["false_activation_rate"],
            -len(row["branches"]),
        ),
    )
    return selected, frontier


def robust_quality_score(
    quality: np.ndarray, development_mask: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    columns = np.asarray([0, 3, 4, 5, 6, 7], dtype=np.int64)
    train = quality[development_mask][:, columns].astype(np.float64)
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    scale = np.clip(1.4826 * mad, 1e-8, None)
    standardized = (quality[:, columns] - median) / scale
    return np.max(standardized, axis=1), {
        "columns": columns,
        "median": median,
        "scale": scale,
    }


def select_evidence_ood_policy(
    labels: np.ndarray,
    intent: np.ndarray,
    direction: np.ndarray,
    disagreement: np.ndarray,
    quality_score: np.ndarray,
    development_mask: np.ndarray,
    validation_mask: np.ndarray,
    maximum_false_activation: float,
) -> Dict[str, object]:
    quality_thresholds = [
        float(np.quantile(quality_score[development_mask], quantile))
        for quantile in (0.95, 0.975, 0.99)
    ] + [1e12]
    disagreement_thresholds = [
        float(np.quantile(disagreement[validation_mask], quantile))
        for quantile in (0.70, 0.80, 0.90, 0.95, 1.0)
    ]
    candidates = []
    for threshold, disagreement_limit, quality_limit in itertools.product(
        np.linspace(0.20, 0.90, 141),
        disagreement_thresholds,
        quality_thresholds,
    ):
        allowed = (
            (disagreement[validation_mask] <= disagreement_limit)
            & (quality_score[validation_mask] <= quality_limit)
        )
        metrics = command_metrics(
            labels[validation_mask],
            intent[validation_mask],
            direction[validation_mask],
            float(threshold),
            allowed,
        )
        if metrics["false_activation_rate"] <= maximum_false_activation:
            candidates.append(
                {
                    "threshold": float(threshold),
                    "disagreement_limit": disagreement_limit,
                    "quality_limit": quality_limit,
                    "controller_validation": metrics,
                }
            )
    if not candidates:
        raise RuntimeError("No evidence/OOD policy met the controller false-activation limit.")
    return max(
        candidates,
        key=lambda row: (
            row["controller_validation"]["intent_balanced_accuracy"],
            row["controller_validation"]["balanced_accuracy"],
            -row["controller_validation"]["false_activation_rate"],
            -row["controller_validation"]["rejected_trial_rate"],
        ),
    )


def calibration_indices(
    labels: np.ndarray, runs: np.ndarray, trials: np.ndarray, budget: int
) -> np.ndarray:
    chosen: List[int] = []
    for label in (0, 1, 2):
        candidates = np.flatnonzero(np.isin(runs, (4, 8)) & (labels == label))
        order = np.lexsort((trials[candidates], runs[candidates]))
        if len(candidates) < budget:
            return np.asarray([], dtype=np.int64)
        chosen.extend(candidates[order][:budget].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def patient_adapted_probabilities(
    target_subjects: Iterable[int],
    dataset: FeatureAtlasDataset,
    branch_intent: np.ndarray,
    branch_direction: np.ndarray,
    branch_indices: Sequence[int],
    budget: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    output: Dict[str, List[np.ndarray]] = {
        "labels": [],
        "subjects": [],
        "global_intent": [],
        "local_intent": [],
        "global_direction": [],
        "local_direction": [],
    }
    for subject in target_subjects:
        selected = dataset.subjects == int(subject)
        local_labels = dataset.labels[selected]
        local_runs = dataset.runs[selected]
        local_trials = dataset.trial_indices[selected]
        calibration = calibration_indices(local_labels, local_runs, local_trials, budget)
        evaluation = np.flatnonzero(local_runs == 12)
        if not len(calibration) or not len(evaluation):
            continue
        intent_x = branch_intent[selected][:, branch_indices]
        direction_x = branch_direction[selected][:, branch_indices, 1]
        intent_model = LogisticRegression(
            C=0.10,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed + int(subject),
        ).fit(intent_x[calibration], local_labels[calibration] != 0)
        movement_calibration = calibration[local_labels[calibration] != 0]
        direction_model = LogisticRegression(
            C=0.10,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed + 1000 + int(subject),
        ).fit(direction_x[movement_calibration], local_labels[movement_calibration])
        local_direction = direction_model.predict_proba(direction_x[evaluation])
        output["labels"].append(local_labels[evaluation])
        output["subjects"].append(
            np.full(len(evaluation), int(subject), dtype=np.int64)
        )
        output["global_intent"].append(np.mean(intent_x[evaluation], axis=1))
        output["local_intent"].append(intent_model.predict_proba(intent_x[evaluation])[:, 1])
        output["global_direction"].append(
            np.mean(branch_direction[selected][evaluation][:, branch_indices], axis=1)
        )
        output["local_direction"].append(local_direction)
    if not output["labels"]:
        raise RuntimeError("No target subjects had sufficient calibration trials.")
    return {name: np.concatenate(rows) for name, rows in output.items()}


def patient_calibration_experiment(
    dataset: FeatureAtlasDataset,
    partitions: SubjectPartitions,
    branch_names: Sequence[str],
    branch_results: Mapping[str, Mapping[str, object]],
    selected_branches: Sequence[str],
    budgets: Sequence[int],
    maximum_false_activation: float,
    seed: int,
) -> Dict[str, object]:
    intent_matrix = np.column_stack(
        [branch_results[name]["intent_probability"] for name in branch_names]
    )
    direction_matrix = np.stack(
        [branch_results[name]["direction_probability"] for name in branch_names],
        axis=1,
    )
    branch_indices = [branch_names.index(name) for name in selected_branches]
    results: Dict[str, object] = {}
    for budget in budgets:
        validation = patient_adapted_probabilities(
            partitions.controller_validation,
            dataset,
            intent_matrix,
            direction_matrix,
            branch_indices,
            budget,
            seed,
        )
        candidates = []
        for weight in (0.0, 0.25, 0.50, 0.75):
            intent = (
                (1.0 - weight) * validation["global_intent"]
                + weight * validation["local_intent"]
            )
            direction = (
                (1.0 - weight) * validation["global_direction"]
                + weight * validation["local_direction"]
            )
            threshold = balanced_threshold(validation["labels"], intent)
            metrics = command_metrics(
                validation["labels"], intent, direction, threshold
            )
            candidates.append(
                {
                    "weight": weight,
                    "threshold": threshold,
                    "metrics": metrics,
                    "intent": intent,
                }
            )
        selected = max(
            candidates,
            key=lambda row: (
                row["metrics"]["intent_balanced_accuracy"],
                row["metrics"]["balanced_accuracy"],
                -row["metrics"]["false_activation_rate"],
            ),
        )
        safety = safety_threshold(
            validation["labels"], selected["intent"], maximum_false_activation
        )
        final = patient_adapted_probabilities(
            partitions.final_target,
            dataset,
            intent_matrix,
            direction_matrix,
            branch_indices,
            budget,
            seed,
        )
        final_intent = (
            (1.0 - selected["weight"]) * final["global_intent"]
            + selected["weight"] * final["local_intent"]
        )
        final_direction = (
            (1.0 - selected["weight"]) * final["global_direction"]
            + selected["weight"] * final["local_direction"]
        )
        results[str(budget)] = {
            "trials_per_original_class": int(budget),
            "total_labeled_trials_per_patient": int(3 * budget),
            "selected_local_weight": float(selected["weight"]),
            "balanced_threshold": float(selected["threshold"]),
            "safety_threshold": float(safety),
            "controller_validation": selected["metrics"],
            "final_balanced": command_metrics(
                final["labels"],
                final_intent,
                final_direction,
                float(selected["threshold"]),
            ),
            "final_safety": command_metrics(
                final["labels"], final_intent, final_direction, safety
            ),
            "final_subjects": final["subjects"],
            "final_labels": final["labels"],
            "final_intent": final_intent,
            "final_direction": final_direction,
        }
    selected_budget = max(
        results,
        key=lambda key: (
            results[key]["controller_validation"]["intent_balanced_accuracy"],
            results[key]["controller_validation"]["balanced_accuracy"],
            -results[key]["controller_validation"]["false_activation_rate"],
        ),
    )
    return {"selected_budget": int(selected_budget), "budgets": results}


def final_policy_predictions(
    intent: np.ndarray,
    direction: np.ndarray,
    threshold: float,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    if allowed is None:
        allowed = np.ones(len(intent), dtype=bool)
    accepted = (intent >= threshold) & allowed
    direction_label = np.argmax(direction, axis=1) + 1
    return np.where(accepted, direction_label, 0)


def subject_uncertainty(
    labels: np.ndarray,
    subjects: np.ndarray,
    intent: np.ndarray,
    direction: np.ndarray,
    threshold: float,
    allowed: np.ndarray,
    seed: int,
    bootstrap_repetitions: int = 2000,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    unique_subjects = np.unique(subjects)
    per_subject: Dict[str, Dict[str, float]] = {}
    fields = (
        "balanced_accuracy",
        "intent_balanced_accuracy",
        "direction_balanced_accuracy",
        "false_activation_rate",
        "missed_intent_rate",
    )
    for subject in unique_subjects:
        selected = subjects == subject
        metrics = command_metrics(
            labels[selected],
            intent[selected],
            direction[selected],
            threshold,
            allowed[selected],
        )
        per_subject[f"S{int(subject):03d}"] = {
            field: float(metrics[field]) for field in fields
        }

    rng = np.random.default_rng(seed)
    bootstrap = {field: np.empty(bootstrap_repetitions) for field in fields}
    subject_indices = {
        int(subject): np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    for repetition in range(bootstrap_repetitions):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_indices[int(subject)] for subject in sampled])
        metrics = command_metrics(
            labels[indices],
            intent[indices],
            direction[indices],
            threshold,
            allowed[indices],
        )
        for field in fields:
            bootstrap[field][repetition] = float(metrics[field])
    intervals = {
        field: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for field, values in bootstrap.items()
    }
    return per_subject, intervals


def plot_branch_comparison(
    output_dir: Path,
    branch_results: Mapping[str, Mapping[str, object]],
    fusion_metrics: Mapping[str, object],
) -> None:
    names = list(branch_results) + ["fusion"]
    metrics = [branch_results[name]["final"] for name in branch_results] + [
        fusion_metrics
    ]
    colors = [BRANCH_COLORS[name] for name in branch_results] + [
        BRANCH_COLORS["fusion"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    fields = (
        ("intent_balanced_accuracy", "Intent Balanced Accuracy"),
        ("false_activation_rate", "False Activation During Rest"),
        ("missed_intent_rate", "Missed Intent"),
    )
    x = np.arange(len(names))
    for axis, (field, title) in zip(axes, fields):
        values = [float(row[field]) for row in metrics]
        bars = axis.bar(x, values, color=colors)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value * 100:.1f}%",
                ha="center",
                fontsize=8,
                weight="bold",
            )
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.set_ylim(0, 1.03)
        axis.set_ylabel("Locked final-subject rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle("Physiology Branches and Late Fusion", fontsize=14, weight="bold")
    fig.savefig(output_dir / "01_branch_and_fusion_comparison.png", dpi=300)
    plt.close(fig)


def plot_operating_points(
    output_dir: Path,
    policies: Mapping[str, Mapping[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    colors = {"balanced": "#147D92", "safety": "#D66A3A", "evidence_ood": "#2D6A4F"}
    offsets = {
        "balanced": (10, 18),
        "safety": (10, 22),
        "evidence_ood": (10, -30),
    }
    for name, metrics in policies.items():
        ax.scatter(
            metrics["false_activation_rate"],
            metrics["intent_sensitivity"],
            s=120,
            color=colors[name],
            edgecolor="white",
            linewidth=1.0,
            label=name.replace("_", " "),
        )
        ax.annotate(
            f"FA {metrics['false_activation_rate'] * 100:.1f}%\n"
            f"Sensitivity {metrics['intent_sensitivity'] * 100:.1f}%",
            (metrics["false_activation_rate"], metrics["intent_sensitivity"]),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvspan(0.0, 0.10, color="#DDEFE9", alpha=0.65, label="Target FA <=10%")
    ax.set_xlim(-0.02, max(0.55, max(row["false_activation_rate"] for row in policies.values()) + 0.08))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False-activation rate during rest")
    ax.set_ylabel("Motor-intent sensitivity")
    ax.set_title("Locked Safety-Responsiveness Operating Points")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "02_safety_responsiveness_operating_points.png", dpi=300)
    plt.close(fig)


def plot_confusions(
    output_dir: Path,
    policies: Mapping[str, Mapping[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    image = None
    for axis, (name, metrics) in zip(axes, policies.items()):
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        normalized = matrix / np.clip(np.sum(matrix, axis=1, keepdims=True), 1, None)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}\n{normalized[row, column] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if normalized[row, column] > 0.55 else "#1F2933",
                    fontsize=8,
                )
        axis.set_xticks(range(3), ["rest", "left", "right"], rotation=20)
        axis.set_yticks(range(3), ["rest", "left", "right"])
        axis.set_xlabel("Predicted command")
        axis.set_ylabel("True event")
        axis.set_title(name.replace("_", " ").title())
    fig.colorbar(image, ax=axes, label="Row-normalized proportion", shrink=0.82)
    fig.suptitle("Final Unseen-Subject Confusion Matrices", fontsize=14, weight="bold")
    fig.savefig(output_dir / "03_final_policy_confusion_matrices.png", dpi=300)
    plt.close(fig)


def plot_calibration(
    output_dir: Path,
    calibration: Mapping[str, object],
    global_balanced: Mapping[str, object],
) -> None:
    rows = calibration["budgets"]
    budgets = sorted(int(value) for value in rows)
    balanced = [
        rows[str(value)]["final_balanced"]["intent_balanced_accuracy"]
        for value in budgets
    ]
    false_activation = [
        rows[str(value)]["final_balanced"]["false_activation_rate"]
        for value in budgets
    ]
    missed = [
        rows[str(value)]["final_balanced"]["missed_intent_rate"] for value in budgets
    ]
    weights = [rows[str(value)]["selected_local_weight"] for value in budgets]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    axes[0].plot(budgets, balanced, marker="o", linewidth=2.4, color=BRANCH_COLORS["adapted"])
    axes[0].axhline(
        global_balanced["intent_balanced_accuracy"],
        color=BRANCH_COLORS["fusion"],
        linestyle="--",
        linewidth=1.8,
        label="Unadapted fusion",
    )
    for x, value, weight in zip(budgets, balanced, weights):
        axes[0].text(
            x,
            value + 0.02,
            f"{value * 100:.1f}%\nweight {weight:.2f}",
            ha="center",
            fontsize=8,
        )
    axes[0].set_ylim(0.35, 0.88)
    axes[0].set_xlabel("Labeled calibration trials per rest/left/right class")
    axes[0].set_ylabel("Final intent balanced accuracy")
    axes[0].set_title("Patient Calibration Benefit")
    axes[0].grid(alpha=0.20)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(budgets, false_activation, marker="o", linewidth=2.2, label="False activation")
    axes[1].plot(budgets, missed, marker="s", linewidth=2.2, label="Missed intent")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Labeled calibration trials per original class")
    axes[1].set_ylabel("Final error rate")
    axes[1].set_title("Calibration Safety Tradeoff")
    axes[1].grid(alpha=0.20)
    axes[1].legend(frameon=False)
    fig.suptitle("Few-Shot Patient Adaptation: Runs 4/8 to Run 12", fontsize=14, weight="bold")
    fig.savefig(output_dir / "04_patient_calibration_results.png", dpi=300)
    plt.close(fig)


def plot_subject_variability(
    output_dir: Path, balanced_policy: Mapping[str, object]
) -> None:
    rows = balanced_policy["per_subject"]
    ordered = sorted(
        rows,
        key=lambda subject: rows[subject]["intent_balanced_accuracy"],
    )
    intent_ba = np.asarray(
        [rows[subject]["intent_balanced_accuracy"] for subject in ordered]
    )
    false_activation = np.asarray(
        [rows[subject]["false_activation_rate"] for subject in ordered]
    )
    missed = np.asarray([rows[subject]["missed_intent_rate"] for subject in ordered])
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.2), sharex=True, constrained_layout=True)
    axes[0].bar(x, intent_ba, color=BRANCH_COLORS["fusion"])
    axes[0].axhline(0.5, color="#4B5563", linestyle="--", linewidth=1.2, label="Chance intent BA")
    axes[0].axhline(
        np.median(intent_ba),
        color="#8E5572",
        linewidth=1.5,
        label=f"Subject median {np.median(intent_ba) * 100:.1f}%",
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Intent balanced accuracy")
    axes[0].set_title("Balanced Fusion Performance by Final Subject")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.20)
    axes[1].plot(
        x,
        false_activation,
        marker="o",
        linewidth=1.8,
        color="#D66A3A",
        label="False activation",
    )
    axes[1].plot(
        x,
        missed,
        marker="s",
        linewidth=1.8,
        color="#147D92",
        label="Missed intent",
    )
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Subject error rate")
    axes[1].set_xlabel("Final subjects ordered by intent balanced accuracy")
    axes[1].set_xticks(x, ordered, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.20)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Generalization Is Heterogeneous Across Unseen Patients", fontsize=14, weight="bold")
    fig.savefig(output_dir / "05_final_subject_variability.png", dpi=300)
    plt.close(fig)


def write_predictions(
    path: Path,
    dataset: FeatureAtlasDataset,
    test_mask: np.ndarray,
    intent: np.ndarray,
    direction: np.ndarray,
    disagreement: np.ndarray,
    quality_score: np.ndarray,
    balanced_value: float,
    safety_value: float,
    evidence_policy: Mapping[str, object],
) -> None:
    evidence_allowed = (
        (disagreement[test_mask] <= evidence_policy["disagreement_limit"])
        & (quality_score[test_mask] <= evidence_policy["quality_limit"])
    )
    predictions = {
        "balanced": final_policy_predictions(
            intent[test_mask], direction[test_mask], balanced_value
        ),
        "safety": final_policy_predictions(
            intent[test_mask], direction[test_mask], safety_value
        ),
        "evidence_ood": final_policy_predictions(
            intent[test_mask],
            direction[test_mask],
            float(evidence_policy["threshold"]),
            evidence_allowed,
        ),
    }
    with path.open("w", newline="", encoding="ascii") as handle:
        fields = [
            "subject",
            "run",
            "trial_index",
            "true_label",
            "true_label_name",
            "intent_probability",
            "left_probability",
            "right_probability",
            "branch_disagreement",
            "quality_score",
            "evidence_ood_allowed",
            "balanced_prediction",
            "safety_prediction",
            "evidence_ood_prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        indices = np.flatnonzero(test_mask)
        for local, index in enumerate(indices):
            writer.writerow(
                {
                    "subject": int(dataset.subjects[index]),
                    "run": int(dataset.runs[index]),
                    "trial_index": int(dataset.trial_indices[index]),
                    "true_label": int(dataset.labels[index]),
                    "true_label_name": LABEL_NAMES[int(dataset.labels[index])],
                    "intent_probability": float(intent[index]),
                    "left_probability": float(direction[index, 0]),
                    "right_probability": float(direction[index, 1]),
                    "branch_disagreement": float(disagreement[index]),
                    "quality_score": float(quality_score[index]),
                    "evidence_ood_allowed": int(evidence_allowed[local]),
                    "balanced_prediction": int(predictions["balanced"][local]),
                    "safety_prediction": int(predictions["safety"][local]),
                    "evidence_ood_prediction": int(predictions["evidence_ood"][local]),
                }
            )


def report_payload(
    dataset: FeatureAtlasDataset,
    partitions: SubjectPartitions,
    branch_results: Mapping[str, Mapping[str, object]],
    selected_fusion: Mapping[str, object],
    fusion_frontier: Sequence[Mapping[str, object]],
    policies: Mapping[str, Mapping[str, object]],
    evidence_policy: Mapping[str, object],
    calibration: Mapping[str, object],
    elapsed_minutes: float,
) -> Dict[str, object]:
    branch_public = {
        name: {
            "selected_features": row["selected_count"],
            "logistic_c": row["c_value"],
            "controller_validation": row["controller_validation"],
            "final": row["final"],
        }
        for name, row in branch_results.items()
    }
    calibration_public: Dict[str, object] = {
        "selected_budget": calibration["selected_budget"],
        "budgets": {},
    }
    for budget, row in calibration["budgets"].items():
        calibration_public["budgets"][budget] = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "final_subjects",
                "final_labels",
                "final_intent",
                "final_direction",
            }
        }
    return {
        "schema_version": 1,
        "protocol": {
            "development_subjects": len(partitions.development),
            "controller_validation_subjects": len(partitions.controller_validation),
            "final_untouched_subjects": len(partitions.final_target),
            "global_training": "all Runs 4/8/12 from development subjects",
            "controller_selection": "Run 12 from controller-validation subjects",
            "patient_calibration": "first labeled trials/class in Runs 4/8",
            "final_evaluation": "Run 12 from final target subjects",
            "selection_guardrail": (
                "All feature, fusion, threshold, OOD, and calibration choices are "
                "fixed on development/controller data before final evaluation."
            ),
        },
        "dataset": {
            "trials": int(len(dataset.labels)),
            "subjects": int(len(np.unique(dataset.subjects))),
        },
        "branches": branch_public,
        "selected_fusion": {
            "branches": selected_fusion["branches"],
            "controller_validation": selected_fusion["controller_validation"],
        },
        "fusion_candidates": len(fusion_frontier),
        "policies_final": policies,
        "evidence_ood_selection": {
            key: value
            for key, value in evidence_policy.items()
            if key != "controller_validation"
        },
        "patient_calibration": calibration_public,
        "runtime_minutes": elapsed_minutes,
    }


def write_markdown(output_dir: Path, report: Mapping[str, object]) -> None:
    selected = report["selected_fusion"]["branches"]
    policies = report["policies_final"]
    calibration = report["patient_calibration"]
    chosen_budget = str(calibration["selected_budget"])
    adapted = calibration["budgets"][chosen_budget]["final_balanced"]
    balanced = policies["balanced"]
    safety = policies["safety"]
    evidence = policies["evidence_ood"]
    balanced_ci = balanced["subject_bootstrap_95_ci"]

    def rates(metrics: Mapping[str, object]) -> str:
        return (
            f"intent BA {metrics['intent_balanced_accuracy'] * 100:.1f}%, "
            f"false activation {metrics['false_activation_rate'] * 100:.1f}%, "
            f"missed intent {metrics['missed_intent_rate'] * 100:.1f}%, "
            f"direction BA {metrics['direction_balanced_accuracy'] * 100:.1f}%"
        )

    text = f"""# Multibranch Motor-Imagery Benchmark

## Locked Protocol

The model is trained on development subjects, all branch/fusion/controller choices
are made on the separate controller-validation subjects, and final percentages
come only from Run 12 of the 22 final subjects. Runs 4/8 provide each final
patient's unsupervised signal reference and the explicitly reported labeled
calibration trials.

## Selected Model

- Late-fusion branches: {", ".join(selected)}
- Balanced policy: {rates(policies["balanced"])}
- Safety policy: {rates(policies["safety"])}
- Evidence + OOD policy: {rates(policies["evidence_ood"])}
- Selected patient calibration ({chosen_budget} trials/class):
  {rates(adapted)}
- Subject-clustered 95% CI for balanced intent BA:
  {balanced_ci["intent_balanced_accuracy"]["lower"] * 100:.1f}% to
  {balanced_ci["intent_balanced_accuracy"]["upper"] * 100:.1f}%
- Subject-clustered 95% CI for balanced false activation:
  {balanced_ci["false_activation_rate"]["lower"] * 100:.1f}% to
  {balanced_ci["false_activation_rate"]["upper"] * 100:.1f}%

## What Improved

Relative to the previous strongest 65.0% trial-level intent result, the selected
fusion reached {balanced["intent_balanced_accuracy"] * 100:.1f}% while reducing
false activation from 43.7% to {balanced["false_activation_rate"] * 100:.1f}%.
This came with more missed intent
({balanced["missed_intent_rate"] * 100:.1f}% rather than 26.3%), so it is a
safer balanced operating point rather than a universal improvement.

## Negative Findings

- The controller-selected patient calibration did not improve the locked final
  cohort: intent BA changed from {balanced["intent_balanced_accuracy"] * 100:.1f}%
  to {adapted["intent_balanced_accuracy"] * 100:.1f}% and false activation changed
  from {balanced["false_activation_rate"] * 100:.1f}% to
  {adapted["false_activation_rate"] * 100:.1f}%.
- Evidence/OOD rejection was effectively tied with the threshold-only safety
  policy: false activation changed from {safety["false_activation_rate"] * 100:.1f}%
  to {evidence["false_activation_rate"] * 100:.1f}%, while missed intent changed
  from {safety["missed_intent_rate"] * 100:.1f}% to
  {evidence["missed_intent_rate"] * 100:.1f}%.

## Interpretation

The balanced, safety, and evidence/OOD operating points answer different design
questions. A rehabilitation controller should not report only accuracy: false
activation, missed intent, direction accuracy, trial rejection, and eventually
false commands per minute and latency must all be shown. This script remains a
cue-locked trial benchmark; it does not replace the continuous controller test.
"""
    (output_dir / "MULTIBRANCH_BENCHMARK_RESULTS.md").write_text(text, encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-path",
        default="artifacts/feature_atlas/motor_imagery_feature_atlas.npz",
    )
    parser.add_argument("--split-path", default="artifacts/dataset/splits.json")
    parser.add_argument(
        "--covariance-cache",
        default="artifacts/multibranch_benchmark/aligned_covariance_features.npz",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/multibranch_benchmark"
    )
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--calibration-budgets", default="3,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-covariance-cache", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.max_false_activation < 0.5:
        raise ValueError("--max-false-activation must be between 0 and 0.5.")
    budgets = [int(value) for value in args.calibration_budgets.split(",")]

    configure_plots()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_feature_atlas(Path(args.atlas_path))
    if dataset is None:
        raise FileNotFoundError(
            f"Feature atlas not found at {args.atlas_path}; run script 18 first."
        )
    partitions = load_partitions(Path(args.split_path))
    masks = locked_masks(dataset, partitions)
    validate_masks(masks, dataset.labels)
    print("=== Locked Multibranch Motor-Imagery Benchmark ===")
    print(
        f"Atlas: {len(dataset.labels)} trials | "
        f"{len(np.unique(dataset.subjects))} subjects"
    )
    print(
        f"Split trials: development={np.sum(masks['development'])} | "
        f"controller={np.sum(masks['validation'])} | final={np.sum(masks['test'])}"
    )
    covariance_features = aligned_covariance_features(
        dataset,
        Path(args.covariance_cache),
        args.rebuild_covariance_cache,
    )
    features = {
        "slow": dataset.slow_features,
        "morlet": dataset.morlet_features,
        "csd": dataset.csd_features,
        "covariance": covariance_features,
    }
    branch_results = {
        name: fit_branch(name, values, dataset.labels, masks, args.seed)
        for name, values in features.items()
    }
    selected_fusion, frontier = select_fusion(
        branch_results, dataset.labels, masks["validation"]
    )
    selected_names = selected_fusion["branches"]
    intent_stack = np.stack(
        [branch_results[name]["intent_probability"] for name in selected_names],
        axis=1,
    )
    direction_stack = np.stack(
        [branch_results[name]["direction_probability"] for name in selected_names],
        axis=1,
    )
    fusion_intent = np.mean(intent_stack, axis=1)
    fusion_direction = np.mean(direction_stack, axis=1)
    disagreement = np.std(intent_stack, axis=1)
    quality_score, quality_transform = robust_quality_score(
        dataset.quality_features, masks["development"]
    )
    balanced_value = float(selected_fusion["threshold"])
    safety_value = safety_threshold(
        dataset.labels[masks["validation"]],
        fusion_intent[masks["validation"]],
        args.max_false_activation,
    )
    evidence_policy = select_evidence_ood_policy(
        dataset.labels,
        fusion_intent,
        fusion_direction,
        disagreement,
        quality_score,
        masks["development"],
        masks["validation"],
        args.max_false_activation,
    )
    evidence_allowed = (
        (disagreement[masks["test"]] <= evidence_policy["disagreement_limit"])
        & (quality_score[masks["test"]] <= evidence_policy["quality_limit"])
    )
    policies = {
        "balanced": command_metrics(
            dataset.labels[masks["test"]],
            fusion_intent[masks["test"]],
            fusion_direction[masks["test"]],
            balanced_value,
        ),
        "safety": command_metrics(
            dataset.labels[masks["test"]],
            fusion_intent[masks["test"]],
            fusion_direction[masks["test"]],
            safety_value,
        ),
        "evidence_ood": command_metrics(
            dataset.labels[masks["test"]],
            fusion_intent[masks["test"]],
            fusion_direction[masks["test"]],
            float(evidence_policy["threshold"]),
            evidence_allowed,
        ),
    }
    final_labels = dataset.labels[masks["test"]]
    final_subjects = dataset.subjects[masks["test"]]
    policy_settings = {
        "balanced": (
            balanced_value,
            np.ones(np.sum(masks["test"]), dtype=bool),
        ),
        "safety": (
            safety_value,
            np.ones(np.sum(masks["test"]), dtype=bool),
        ),
        "evidence_ood": (
            float(evidence_policy["threshold"]),
            evidence_allowed,
        ),
    }
    for offset, (name, (threshold, allowed)) in enumerate(policy_settings.items()):
        per_subject, intervals = subject_uncertainty(
            final_labels,
            final_subjects,
            fusion_intent[masks["test"]],
            fusion_direction[masks["test"]],
            threshold,
            allowed,
            args.seed + 5000 + offset,
        )
        policies[name]["per_subject"] = per_subject
        policies[name]["subject_bootstrap_95_ci"] = intervals
    calibration = patient_calibration_experiment(
        dataset,
        partitions,
        list(branch_results),
        branch_results,
        selected_names,
        budgets,
        args.max_false_activation,
        args.seed,
    )
    elapsed = (time.time() - started) / 60.0
    report = report_payload(
        dataset,
        partitions,
        branch_results,
        selected_fusion,
        frontier,
        policies,
        evidence_policy,
        calibration,
        elapsed,
    )
    (output_dir / "multibranch_metrics.json").write_text(
        json.dumps(report, indent=2, default=json_default) + "\n", encoding="ascii"
    )
    joblib.dump(
        {
            "branch_models": {
                name: {
                    "intent": row["intent_model"],
                    "direction": row["direction_model"],
                }
                for name, row in branch_results.items()
            },
            "selected_branches": selected_names,
            "balanced_threshold": balanced_value,
            "safety_threshold": safety_value,
            "evidence_ood_policy": {
                key: value
                for key, value in evidence_policy.items()
                if key != "controller_validation"
            },
            "quality_transform": quality_transform,
            "atlas_fingerprint": dataset.fingerprint,
        },
        output_dir / "multibranch_motor_imagery_models.joblib",
    )
    write_predictions(
        output_dir / "final_trial_predictions.csv",
        dataset,
        masks["test"],
        fusion_intent,
        fusion_direction,
        disagreement,
        quality_score,
        balanced_value,
        safety_value,
        evidence_policy,
    )
    plot_branch_comparison(output_dir, branch_results, policies["balanced"])
    plot_operating_points(output_dir, policies)
    plot_confusions(output_dir, policies)
    plot_calibration(output_dir, calibration, policies["balanced"])
    plot_subject_variability(output_dir, policies["balanced"])
    write_markdown(output_dir, report)
    print(f"[+] Selected fusion: {', '.join(selected_names)}")
    for name, metrics in policies.items():
        print(
            f"    {name:12s} | intent BA={metrics['intent_balanced_accuracy'] * 100:5.1f}% | "
            f"FA={metrics['false_activation_rate'] * 100:5.1f}% | "
            f"missed={metrics['missed_intent_rate'] * 100:5.1f}%"
        )
    print(f"[+] Saved locked results and five figures to {output_dir.resolve()}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
