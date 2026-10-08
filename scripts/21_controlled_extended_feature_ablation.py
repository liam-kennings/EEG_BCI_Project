#!/usr/bin/env python3
"""Run a validation-selected, final-locked morphology/SSD ablation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

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
from bci_pipeline.extended_features import (  # noqa: E402
    ExtendedFeatureDataset,
    extended_fingerprint,
    load_extended_features,
)
from bci_pipeline.feature_atlas import (  # noqa: E402
    FeatureAtlasDataset,
    load_feature_atlas,
)
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.splits import SubjectPartitions, load_partitions  # noqa: E402


CONFIGURATIONS: Dict[str, Tuple[str, ...]] = {
    "reference": ("slow", "covariance"),
    "reference+morphology": ("slow", "covariance", "morphology"),
    "reference+ssd": ("slow", "covariance", "ssd"),
    "reference+morphology+ssd": (
        "slow",
        "covariance",
        "morphology",
        "ssd",
    ),
}
COLORS = {
    "reference": "#5D6770",
    "reference+morphology": "#D66A3A",
    "reference+ssd": "#2D6A4F",
    "reference+morphology+ssd": "#147D92",
    "positive": "#2D6A4F",
    "negative": "#C75C3A",
}
DISPLAY_NAMES = {
    "reference": "Reference",
    "reference+morphology": "+ Morphology",
    "reference+ssd": "+ SSD",
    "reference+morphology+ssd": "+ Morphology + SSD",
}
METRIC_FIELDS = (
    "balanced_accuracy",
    "intent_balanced_accuracy",
    "direction_balanced_accuracy",
    "false_activation_rate",
    "missed_intent_rate",
)


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def locked_masks(
    dataset: FeatureAtlasDataset, partitions: SubjectPartitions
) -> Dict[str, np.ndarray]:
    return {
        "development": np.isin(dataset.subjects, partitions.development),
        "validation": np.isin(
            dataset.subjects, partitions.controller_validation
        )
        & (dataset.runs == 12),
        "test": np.isin(dataset.subjects, partitions.final_target)
        & (dataset.runs == 12),
    }


def validate_inputs(
    atlas: FeatureAtlasDataset,
    extended: ExtendedFeatureDataset,
    masks: Mapping[str, np.ndarray],
) -> None:
    aligned = (
        np.array_equal(atlas.labels, extended.labels)
        and np.array_equal(atlas.subjects, extended.subjects)
        and np.array_equal(atlas.runs, extended.runs)
        and np.array_equal(atlas.trial_indices, extended.trial_indices)
    )
    if not aligned:
        raise ValueError("Extended features are not row-aligned with the base atlas.")
    for name, mask in masks.items():
        if not np.any(mask) or len(np.unique(atlas.labels[mask])) != 3:
            raise ValueError(f"The fixed {name} split is empty or missing a class.")


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
    features[..., rows != columns] *= np.sqrt(2.0)
    return features


def aligned_covariance_features(
    dataset: FeatureAtlasDataset, cache_path: Path, rebuild: bool
) -> np.ndarray:
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            fingerprint = str(cached["fingerprint"].item())
            features = cached["features"]
            if fingerprint == dataset.fingerprint and len(features) == len(dataset.labels):
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
    print("[+] Rebuilding aligned covariance features...")
    unique_subjects = np.unique(dataset.subjects)
    for position, subject in enumerate(unique_subjects, start=1):
        selected = dataset.subjects == subject
        calibration = selected & np.isin(dataset.runs, (4, 8))
        reference = np.mean(covariances[calibration], axis=0)
        reference = 0.5 * (reference + np.swapaxes(reference, -1, -2))
        whitening = _matrix_invsqrt(reference)
        aligned = whitening[None] @ covariances[selected] @ whitening[None]
        aligned = 0.5 * (aligned + np.swapaxes(aligned, -1, -2))
        output[selected] = _log_upper_features(aligned).astype(np.float32)
        if position % 20 == 0 or position == len(unique_subjects):
            print(f"    aligned {position:3d}/{len(unique_subjects)} subjects")
    features = output.reshape(len(dataset.labels), -1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        fingerprint=np.asarray(dataset.fingerprint),
    )
    return features


def make_model(
    feature_count: int, selected_count: int, c_value: float, seed: int
) -> Pipeline:
    return Pipeline(
        [
            (
                "selection",
                SelectKBest(f_classif, k=min(selected_count, feature_count)),
            ),
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
    labels: np.ndarray,
    probabilities: np.ndarray,
    maximum_false_activation: float,
) -> float:
    rest = probabilities[labels == 0]
    return float(
        np.nextafter(
            np.quantile(
                rest,
                1.0 - maximum_false_activation,
                method="higher",
            ),
            np.inf,
        )
    )


def command_metrics(
    labels: np.ndarray,
    intent_probability: np.ndarray,
    direction_probability: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    metrics = hierarchical_metrics(
        labels,
        intent_probability,
        direction_probability,
        threshold,
        LABEL_NAMES,
    )
    metrics["intent_sensitivity"] = 1.0 - float(metrics["missed_intent_rate"])
    metrics["threshold"] = float(threshold)
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
    for selected_count, c_value in itertools.product(
        feature_counts, (0.05, 0.20)
    ):
        intent_model = make_model(
            features.shape[1], selected_count, c_value, seed
        )
        intent_model.fit(
            features[masks["development"]],
            labels[masks["development"]] != 0,
        )
        probability = intent_model.predict_proba(
            features[masks["validation"]]
        )[:, 1]
        threshold = balanced_threshold(
            labels[masks["validation"]], probability
        )
        metrics = command_metrics(
            labels[masks["validation"]],
            probability,
            np.full((len(probability), 2), 0.5),
            threshold,
        )
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
            row["metrics"]["intent_balanced_accuracy"],
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
    intent_probability = selected["model"].predict_proba(features)[:, 1]
    direction_probability = direction_model.predict_proba(features)
    print(
        f"    {name:10s} | features={int(selected['selected_count']):3d} | "
        f"C={float(selected['c_value']):.2f} | validation intent BA="
        f"{selected['metrics']['intent_balanced_accuracy'] * 100:5.1f}%"
    )
    return {
        "intent_model": selected["model"],
        "direction_model": direction_model,
        "intent_probability": intent_probability,
        "direction_probability": direction_probability,
        "selected_count": int(selected["selected_count"]),
        "c_value": float(selected["c_value"]),
    }


def fusion_probabilities(
    branches: Sequence[str],
    branch_results: Mapping[str, Mapping[str, object]],
) -> Tuple[np.ndarray, np.ndarray]:
    intent = np.mean(
        [branch_results[name]["intent_probability"] for name in branches],
        axis=0,
    )
    direction = np.mean(
        [branch_results[name]["direction_probability"] for name in branches],
        axis=0,
    )
    return intent, direction


def validation_ablation(
    labels: np.ndarray,
    validation_mask: np.ndarray,
    branch_results: Mapping[str, Mapping[str, object]],
    maximum_false_activation: float,
) -> Tuple[Dict[str, Dict[str, object]], str]:
    results: Dict[str, Dict[str, object]] = {}
    for name, branches in CONFIGURATIONS.items():
        intent, direction = fusion_probabilities(branches, branch_results)
        balanced_value = balanced_threshold(
            labels[validation_mask], intent[validation_mask]
        )
        safety_value = safety_threshold(
            labels[validation_mask],
            intent[validation_mask],
            maximum_false_activation,
        )
        results[name] = {
            "branches": list(branches),
            "balanced_threshold": balanced_value,
            "safety_threshold": safety_value,
            "balanced": command_metrics(
                labels[validation_mask],
                intent[validation_mask],
                direction[validation_mask],
                balanced_value,
            ),
            "safety": command_metrics(
                labels[validation_mask],
                intent[validation_mask],
                direction[validation_mask],
                safety_value,
            ),
            "intent_probability": intent,
            "direction_probability": direction,
        }
    selected = max(
        results,
        key=lambda name: (
            results[name]["balanced"]["intent_balanced_accuracy"],
            results[name]["balanced"]["balanced_accuracy"],
            -results[name]["balanced"]["false_activation_rate"],
            -len(results[name]["branches"]),
        ),
    )
    return results, selected


def final_evaluation(
    labels: np.ndarray,
    test_mask: np.ndarray,
    validation_results: Mapping[str, Mapping[str, object]],
    selected_name: str,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    output: Dict[str, Dict[str, Dict[str, object]]] = {}
    for operating_point, threshold_key in (
        ("balanced", "balanced_threshold"),
        ("safety", "safety_threshold"),
    ):
        output[operating_point] = {}
        for name in ("reference", selected_name):
            row = validation_results[name]
            output[operating_point][name] = command_metrics(
                labels[test_mask],
                row["intent_probability"][test_mask],
                row["direction_probability"][test_mask],
                float(row[threshold_key]),
            )
    return output


def subject_bootstrap_comparison(
    labels: np.ndarray,
    subjects: np.ndarray,
    reference_intent: np.ndarray,
    reference_direction: np.ndarray,
    selected_intent: np.ndarray,
    selected_direction: np.ndarray,
    reference_threshold: float,
    selected_threshold: float,
    seed: int,
    repetitions: int,
) -> Dict[str, object]:
    unique_subjects = np.unique(subjects)
    subject_indices = {
        int(subject): np.flatnonzero(subjects == subject)
        for subject in unique_subjects
    }
    per_subject = {}
    for subject in unique_subjects:
        indices = subject_indices[int(subject)]
        reference = command_metrics(
            labels[indices],
            reference_intent[indices],
            reference_direction[indices],
            reference_threshold,
        )
        selected = command_metrics(
            labels[indices],
            selected_intent[indices],
            selected_direction[indices],
            selected_threshold,
        )
        per_subject[f"S{int(subject):03d}"] = {
            "reference": {
                field: float(reference[field]) for field in METRIC_FIELDS
            },
            "selected": {
                field: float(selected[field]) for field in METRIC_FIELDS
            },
            "delta_selected_minus_reference": {
                field: float(selected[field]) - float(reference[field])
                for field in METRIC_FIELDS
            },
        }

    rng = np.random.default_rng(seed)
    samples = {
        model: {
            field: np.empty(repetitions, dtype=np.float64)
            for field in METRIC_FIELDS
        }
        for model in ("reference", "selected", "delta")
    }
    for repetition in range(repetitions):
        sampled = rng.choice(
            unique_subjects, size=len(unique_subjects), replace=True
        )
        indices = np.concatenate(
            [subject_indices[int(subject)] for subject in sampled]
        )
        reference = command_metrics(
            labels[indices],
            reference_intent[indices],
            reference_direction[indices],
            reference_threshold,
        )
        selected = command_metrics(
            labels[indices],
            selected_intent[indices],
            selected_direction[indices],
            selected_threshold,
        )
        for field in METRIC_FIELDS:
            samples["reference"][field][repetition] = float(reference[field])
            samples["selected"][field][repetition] = float(selected[field])
            samples["delta"][field][repetition] = (
                float(selected[field]) - float(reference[field])
            )
    intervals = {
        model: {
            field: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for field, values in fields.items()
        }
        for model, fields in samples.items()
    }
    return {"per_subject": per_subject, "subject_bootstrap_95_ci": intervals}


def public_validation_results(
    results: Mapping[str, Mapping[str, object]]
) -> Dict[str, Dict[str, object]]:
    return {
        name: {
            "branches": row["branches"],
            "balanced_threshold": row["balanced_threshold"],
            "safety_threshold": row["safety_threshold"],
            "balanced": row["balanced"],
            "safety": row["safety"],
        }
        for name, row in results.items()
    }


def plot_validation_ablation(
    output_dir: Path,
    results: Mapping[str, Mapping[str, object]],
    selected_name: str,
) -> None:
    names = list(CONFIGURATIONS)
    panels = (
        ("intent_balanced_accuracy", "Intent Balanced Accuracy"),
        ("false_activation_rate", "False Activation During Rest"),
        ("missed_intent_rate", "Missed Intent"),
        ("direction_balanced_accuracy", "Direction Balanced Accuracy"),
    )
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for axis, (field, title) in zip(axes.ravel(), panels):
        values = [results[name]["balanced"][field] for name in names]
        bars = axis.bar(x, values, color=[COLORS[name] for name in names])
        bars[names.index(selected_name)].set_edgecolor("#111111")
        bars[names.index(selected_name)].set_linewidth(2.5)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value * 100:.1f}%",
                ha="center",
                fontsize=8,
                weight="bold",
            )
        axis.set_xticks(
            x,
            [DISPLAY_NAMES[name] for name in names],
            rotation=20,
            ha="right",
        )
        axis.set_ylim(0, 1.03)
        axis.set_ylabel("Controller-validation rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle(
        "Controlled Feature Ablation: Validation Selection",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "01_validation_ablation_selection.png", dpi=300)
    plt.close(fig)


def plot_final_comparison(
    output_dir: Path,
    final_results: Mapping[str, Mapping[str, Mapping[str, object]]],
    uncertainty: Mapping[str, Mapping[str, object]],
    selected_name: str,
) -> None:
    panels = (
        ("intent_balanced_accuracy", "Intent Balanced Accuracy"),
        ("false_activation_rate", "False Activation During Rest"),
        ("missed_intent_rate", "Missed Intent"),
        ("direction_balanced_accuracy", "Direction Balanced Accuracy"),
    )
    operating_points = ("balanced", "safety")
    x = np.arange(len(operating_points))
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    for axis, (field, title) in zip(axes.ravel(), panels):
        for offset, (model, label, color) in enumerate(
            (
                ("reference", "Reference", COLORS["reference"]),
                ("selected", DISPLAY_NAMES[selected_name], COLORS[selected_name]),
            )
        ):
            values = [
                final_results[point][
                    "reference" if model == "reference" else selected_name
                ][field]
                for point in operating_points
            ]
            interval_values = [
                uncertainty[point]["subject_bootstrap_95_ci"][model][field]
                for point in operating_points
            ]
            lower = [
                max(0.0, value - interval["lower"])
                for value, interval in zip(values, interval_values)
            ]
            upper = [
                max(0.0, interval["upper"] - value)
                for value, interval in zip(values, interval_values)
            ]
            positions = x + (offset - 0.5) * width
            bars = axis.bar(
                positions,
                values,
                width,
                color=color,
                label=label,
                yerr=np.asarray((lower, upper)),
                capsize=4,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max(upper) + 0.025,
                    f"{value * 100:.1f}%",
                    ha="center",
                    fontsize=8,
                )
        axis.set_xticks(x, ["Balanced", "Safety"])
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Locked final-subject rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.20)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Locked Final Comparison: Reference Versus Validation Winner",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "02_locked_final_reference_vs_winner.png", dpi=300)
    plt.close(fig)


def plot_subject_deltas(
    output_dir: Path,
    per_subject: Mapping[str, Mapping[str, Mapping[str, float]]],
    selected_name: str,
) -> None:
    ordered = sorted(
        per_subject,
        key=lambda subject: per_subject[subject][
            "delta_selected_minus_reference"
        ]["intent_balanced_accuracy"],
    )
    intent_delta = np.asarray(
        [
            per_subject[subject]["delta_selected_minus_reference"][
                "intent_balanced_accuracy"
            ]
            for subject in ordered
        ]
    )
    false_activation_delta = np.asarray(
        [
            per_subject[subject]["delta_selected_minus_reference"][
                "false_activation_rate"
            ]
            for subject in ordered
        ]
    )
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(
        2, 1, figsize=(12.0, 7.0), sharex=True, constrained_layout=True
    )
    axes[0].bar(
        x,
        intent_delta,
        color=np.where(
            intent_delta >= 0.0,
            COLORS["positive"],
            COLORS["negative"],
        ),
    )
    axes[0].axhline(0.0, color="#222222", linewidth=0.9)
    axes[0].set_ylabel("Intent BA change")
    axes[0].set_title("Selected Minus Reference: Intent Performance")
    axes[0].grid(axis="y", alpha=0.20)
    axes[1].bar(
        x,
        false_activation_delta,
        color=np.where(
            false_activation_delta <= 0.0,
            COLORS["positive"],
            COLORS["negative"],
        ),
    )
    axes[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes[1].set_ylabel("False-activation change")
    axes[1].set_xlabel("Final subjects ordered by intent-BA change")
    axes[1].set_xticks(x, ordered, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.20)
    fig.suptitle(
        f"Subject-Level Effect of {DISPLAY_NAMES[selected_name].lstrip('+ ')}",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "03_final_subject_paired_deltas.png", dpi=300)
    plt.close(fig)


def plot_final_confusions(
    output_dir: Path,
    final_results: Mapping[str, Mapping[str, Mapping[str, object]]],
    selected_name: str,
) -> None:
    panels = (
        ("balanced", "reference", "Reference | Balanced"),
        ("balanced", selected_name, "Selected | Balanced"),
        ("safety", "reference", "Reference | Safety"),
        ("safety", selected_name, "Selected | Safety"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.8), constrained_layout=True)
    image = None
    for axis, (point, name, title) in zip(axes.ravel(), panels):
        matrix = np.asarray(final_results[point][name]["confusion_matrix"], dtype=float)
        normalized = matrix / np.clip(
            np.sum(matrix, axis=1, keepdims=True), 1, None
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}\n"
                    f"{normalized[row, column] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if normalized[row, column] > 0.55
                        else "#1F2933"
                    ),
                    fontsize=8,
                )
        axis.set_xticks(range(3), ["rest", "left", "right"])
        axis.set_yticks(range(3), ["rest", "left", "right"])
        axis.set_xlabel("Predicted command")
        axis.set_ylabel("True event")
        axis.set_title(title)
    fig.colorbar(
        image,
        ax=axes,
        label="Row-normalized proportion",
        shrink=0.82,
    )
    fig.suptitle(
        "Final Unseen-Subject Confusion Matrices",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "04_locked_final_confusion_matrices.png", dpi=300)
    plt.close(fig)


def write_validation_csv(
    path: Path,
    results: Mapping[str, Mapping[str, object]],
    selected_name: str,
) -> None:
    fields = [
        "configuration",
        "selected",
        "branches",
        "operating_point",
        "threshold",
        *METRIC_FIELDS,
    ]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in results.items():
            for operating_point in ("balanced", "safety"):
                metrics = row[operating_point]
                writer.writerow(
                    {
                        "configuration": name,
                        "selected": int(name == selected_name),
                        "branches": "+".join(row["branches"]),
                        "operating_point": operating_point,
                        "threshold": metrics["threshold"],
                        **{
                            field: metrics[field] for field in METRIC_FIELDS
                        },
                    }
                )


def write_subject_csv(
    path: Path,
    per_subject: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> None:
    fields = ["subject"]
    for prefix in ("reference", "selected", "delta_selected_minus_reference"):
        fields.extend(f"{prefix}_{field}" for field in METRIC_FIELDS)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject, row in per_subject.items():
            values = {"subject": subject}
            for prefix in (
                "reference",
                "selected",
                "delta_selected_minus_reference",
            ):
                values.update(
                    {
                        f"{prefix}_{field}": row[prefix][field]
                        for field in METRIC_FIELDS
                    }
                )
            writer.writerow(values)


def write_final_predictions(
    path: Path,
    atlas: FeatureAtlasDataset,
    test_mask: np.ndarray,
    validation_results: Mapping[str, Mapping[str, object]],
    selected_name: str,
) -> None:
    reference = validation_results["reference"]
    selected = validation_results[selected_name]
    indices = np.flatnonzero(test_mask)

    def predictions(row: Mapping[str, object], threshold_key: str) -> np.ndarray:
        intent = row["intent_probability"][test_mask]
        direction = row["direction_probability"][test_mask]
        accepted = intent >= float(row[threshold_key])
        return np.where(accepted, np.argmax(direction, axis=1) + 1, 0)

    columns = {
        "reference_balanced_prediction": predictions(
            reference, "balanced_threshold"
        ),
        "reference_safety_prediction": predictions(
            reference, "safety_threshold"
        ),
        "selected_balanced_prediction": predictions(
            selected, "balanced_threshold"
        ),
        "selected_safety_prediction": predictions(
            selected, "safety_threshold"
        ),
    }
    with path.open("w", newline="", encoding="ascii") as handle:
        fields = [
            "subject",
            "run",
            "trial_index",
            "true_label",
            "true_label_name",
            "reference_intent_probability",
            "selected_intent_probability",
            *columns,
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for local, index in enumerate(indices):
            writer.writerow(
                {
                    "subject": int(atlas.subjects[index]),
                    "run": int(atlas.runs[index]),
                    "trial_index": int(atlas.trial_indices[index]),
                    "true_label": int(atlas.labels[index]),
                    "true_label_name": LABEL_NAMES[int(atlas.labels[index])],
                    "reference_intent_probability": float(
                        reference["intent_probability"][index]
                    ),
                    "selected_intent_probability": float(
                        selected["intent_probability"][index]
                    ),
                    **{
                        name: int(values[local])
                        for name, values in columns.items()
                    },
                }
            )


def format_rates(metrics: Mapping[str, object]) -> str:
    return (
        f"{metrics['intent_balanced_accuracy'] * 100:.1f}% | "
        f"{metrics['false_activation_rate'] * 100:.1f}% | "
        f"{metrics['missed_intent_rate'] * 100:.1f}% | "
        f"{metrics['direction_balanced_accuracy'] * 100:.1f}%"
    )


def write_markdown(path: Path, report: Mapping[str, object]) -> None:
    validation = report["validation_ablation"]
    selected_name = report["selected_configuration"]
    final_results = report["locked_final"]
    uncertainty = report["paired_subject_uncertainty"]
    lines = [
        "# Controlled Extended-Feature Ablation",
        "",
        "## Protocol",
        "",
        "The development cohort fits every branch. The controller-validation cohort "
        "selects branch hyperparameters, balanced and safety thresholds, and one of "
        "four prespecified equal-weight fusions. The locked final cohort is evaluated "
        "only for the original reference and the validation-selected configuration.",
        "",
        "No patient calibration, OOD rejection, temporal confirmation, or fusion-weight "
        "search is included, so morphology/SSD are the only experimental variables.",
        "",
        "## Controller-Validation Selection",
        "",
        "| Configuration | Intent BA | False activation | Missed intent | Direction BA |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in CONFIGURATIONS:
        metrics = validation[name]["balanced"]
        marker = " **(selected)**" if name == selected_name else ""
        lines.append(
            f"| {DISPLAY_NAMES[name]}{marker} | "
            f"{metrics['intent_balanced_accuracy'] * 100:.1f}% | "
            f"{metrics['false_activation_rate'] * 100:.1f}% | "
            f"{metrics['missed_intent_rate'] * 100:.1f}% | "
            f"{metrics['direction_balanced_accuracy'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            f"Validation selected **{DISPLAY_NAMES[selected_name]}**.",
            "",
            "## Locked Final Comparison",
            "",
            "| Operating point | Model | Intent BA | False activation | Missed intent | Direction BA |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in ("balanced", "safety"):
        for name, model_label in (
            ("reference", "Reference"),
            (selected_name, DISPLAY_NAMES[selected_name]),
        ):
            metrics = final_results[point][name]
            values = format_rates(metrics).split(" | ")
            lines.append(
                f"| {point.title()} | {model_label} | "
                + " | ".join(values)
                + " |"
            )
    balanced_delta = uncertainty["balanced"]["subject_bootstrap_95_ci"]["delta"]
    per_subject = uncertainty["balanced"]["per_subject"]
    final_reference = final_results["balanced"]["reference"]
    final_selected = final_results["balanced"][selected_name]
    intent_delta = (
        final_selected["intent_balanced_accuracy"]
        - final_reference["intent_balanced_accuracy"]
    )
    false_delta = (
        final_selected["false_activation_rate"]
        - final_reference["false_activation_rate"]
    )
    intent_interval = balanced_delta["intent_balanced_accuracy"]
    false_interval = balanced_delta["false_activation_rate"]
    intent_subject_deltas = np.asarray(
        [
            row["delta_selected_minus_reference"]["intent_balanced_accuracy"]
            for row in per_subject.values()
        ]
    )
    missed_subject_deltas = np.asarray(
        [
            row["delta_selected_minus_reference"]["missed_intent_rate"]
            for row in per_subject.values()
        ]
    )
    false_subject_deltas = np.asarray(
        [
            row["delta_selected_minus_reference"]["false_activation_rate"]
            for row in per_subject.values()
        ]
    )
    if intent_interval["lower"] > 0.0:
        conclusion = (
            "The selected extension improved final intent balanced accuracy with a "
            "positive subject-clustered 95% interval."
        )
    elif intent_interval["upper"] < 0.0:
        conclusion = (
            "The selected extension reduced final intent balanced accuracy and should "
            "not replace the reference."
        )
    else:
        conclusion = (
            "The final intent-accuracy difference is uncertain because its "
            "subject-clustered 95% interval includes zero."
        )
    if intent_interval["lower"] <= 0.0 <= intent_interval["upper"]:
        recommendation = (
            "Keep the slow-plus-covariance reference as the default decoder. "
            "Morphology may remain an exploratory sensitivity-oriented proposal "
            "signal, but it should not replace the reference safety policy."
        )
    else:
        recommendation = (
            "Advance the selected configuration only after confirming the same "
            "direction of effect in a new session or external dataset."
        )
    lines.extend(
        [
            "",
            "## Paired Interpretation",
            "",
            f"- Balanced intent-BA change: **{intent_delta * 100:+.1f} percentage "
            f"points** (95% CI {intent_interval['lower'] * 100:+.1f} to "
            f"{intent_interval['upper'] * 100:+.1f}).",
            f"- Balanced false-activation change: **{false_delta * 100:+.1f} "
            f"percentage points** (95% CI {false_interval['lower'] * 100:+.1f} "
            f"to {false_interval['upper'] * 100:+.1f}).",
            f"- Intent BA improved for **{int(np.sum(intent_subject_deltas > 1e-12))}** "
            f"subjects, worsened for **{int(np.sum(intent_subject_deltas < -1e-12))}**, "
            f"and was unchanged for **{int(np.sum(np.abs(intent_subject_deltas) <= 1e-12))}**.",
            f"- Missed intent decreased for **{int(np.sum(missed_subject_deltas < -1e-12))}** "
            f"subjects, while false activation increased for "
            f"**{int(np.sum(false_subject_deltas > 1e-12))}**.",
            f"- {conclusion}",
            f"- **Recommendation:** {recommendation}",
            "",
            "A new branch should advance to the continuous controller only if the "
            "validation-selected result offers a useful final safety/sensitivity "
            "tradeoff without relying on an isolated pooled-accuracy increase.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-path",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "feature_atlas"
        / "motor_imagery_feature_atlas.npz",
    )
    parser.add_argument(
        "--extended-path",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "extended_features"
        / "extended_motor_imagery_features.npz",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "dataset" / "splits.json",
    )
    parser.add_argument(
        "--covariance-cache",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "multibranch_benchmark"
        / "aligned_covariance_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "controlled_ablation",
    )
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-covariance-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.max_false_activation < 0.5:
        raise ValueError("--max-false-activation must be between 0 and 0.5.")
    configure_plots()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atlas = load_feature_atlas(args.atlas_path)
    if atlas is None:
        raise FileNotFoundError("Base feature atlas is missing; run script 18.")
    extended = load_extended_features(
        args.extended_path, extended_fingerprint(atlas)
    )
    if extended is None:
        raise FileNotFoundError(
            "Extended feature cache is missing or stale; run script 20."
        )
    partitions = load_partitions(args.split_path)
    masks = locked_masks(atlas, partitions)
    validate_inputs(atlas, extended, masks)
    print("\n=== Controlled Morphology/SSD Ablation ===")
    print(
        f"Trials: {len(atlas.labels)} | subjects: "
        f"{len(np.unique(atlas.subjects))}"
    )
    print(
        f"Split trials: development={np.sum(masks['development'])} | "
        f"validation={np.sum(masks['validation'])} | "
        f"final={np.sum(masks['test'])}"
    )
    covariance = aligned_covariance_features(
        atlas, args.covariance_cache, args.rebuild_covariance_cache
    )
    feature_matrices = {
        "slow": atlas.slow_features,
        "covariance": covariance,
        "morphology": extended.morphology_features,
        "ssd": extended.ssd_features,
    }
    print("[+] Fitting branch models with the reference hyperparameter grid...")
    branch_results = {
        name: fit_branch(name, features, atlas.labels, masks, args.seed)
        for name, features in feature_matrices.items()
    }
    validation_results, selected_name = validation_ablation(
        atlas.labels,
        masks["validation"],
        branch_results,
        args.max_false_activation,
    )
    print(
        f"[+] Controller validation selected: {DISPLAY_NAMES[selected_name]}"
    )
    final_results = final_evaluation(
        atlas.labels, masks["test"], validation_results, selected_name
    )
    test_labels = atlas.labels[masks["test"]]
    test_subjects = atlas.subjects[masks["test"]]
    reference = validation_results["reference"]
    selected = validation_results[selected_name]
    uncertainty = {}
    for offset, (point, threshold_key) in enumerate(
        (
            ("balanced", "balanced_threshold"),
            ("safety", "safety_threshold"),
        )
    ):
        uncertainty[point] = subject_bootstrap_comparison(
            test_labels,
            test_subjects,
            reference["intent_probability"][masks["test"]],
            reference["direction_probability"][masks["test"]],
            selected["intent_probability"][masks["test"]],
            selected["direction_probability"][masks["test"]],
            float(reference[threshold_key]),
            float(selected[threshold_key]),
            args.seed + 9000 + offset,
            args.bootstrap_repetitions,
        )
    elapsed_minutes = (time.time() - started) / 60.0
    report = {
        "schema_version": 1,
        "protocol": {
            "development_subjects": len(partitions.development),
            "controller_validation_subjects": len(
                partitions.controller_validation
            ),
            "final_subjects": len(partitions.final_target),
            "global_training": "all runs from development subjects",
            "controller_selection": (
                "Run 12 from controller-validation subjects; equal branch weights"
            ),
            "final_evaluation": (
                "Run 12 reference versus validation-selected configuration only"
            ),
            "maximum_validation_false_activation_for_safety": (
                args.max_false_activation
            ),
        },
        "branch_hyperparameters": {
            name: {
                "selected_features": row["selected_count"],
                "logistic_c": row["c_value"],
            }
            for name, row in branch_results.items()
        },
        "validation_ablation": public_validation_results(validation_results),
        "selected_configuration": selected_name,
        "locked_final": final_results,
        "paired_subject_uncertainty": uncertainty,
        "runtime_minutes": elapsed_minutes,
    }
    (args.output_dir / "controlled_ablation_metrics.json").write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
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
            "configurations": CONFIGURATIONS,
            "selected_configuration": selected_name,
            "balanced_thresholds": {
                name: row["balanced_threshold"]
                for name, row in validation_results.items()
            },
            "safety_thresholds": {
                name: row["safety_threshold"]
                for name, row in validation_results.items()
            },
            "atlas_fingerprint": atlas.fingerprint,
            "extended_fingerprint": extended.fingerprint,
        },
        args.output_dir / "controlled_ablation_models.joblib",
    )
    write_validation_csv(
        args.output_dir / "validation_ablation.csv",
        validation_results,
        selected_name,
    )
    write_subject_csv(
        args.output_dir / "final_subject_paired_metrics.csv",
        uncertainty["balanced"]["per_subject"],
    )
    write_final_predictions(
        args.output_dir / "final_trial_predictions.csv",
        atlas,
        masks["test"],
        validation_results,
        selected_name,
    )
    plot_validation_ablation(
        args.output_dir, validation_results, selected_name
    )
    plot_final_comparison(
        args.output_dir, final_results, uncertainty, selected_name
    )
    plot_subject_deltas(
        args.output_dir,
        uncertainty["balanced"]["per_subject"],
        selected_name,
    )
    plot_final_confusions(args.output_dir, final_results, selected_name)
    write_markdown(
        args.output_dir / "CONTROLLED_ABLATION_RESULTS.md", report
    )
    for point in ("balanced", "safety"):
        reference_metrics = final_results[point]["reference"]
        selected_metrics = final_results[point][selected_name]
        print(
            f"    {point:8s} reference | intent BA="
            f"{reference_metrics['intent_balanced_accuracy'] * 100:5.1f}% | "
            f"FA={reference_metrics['false_activation_rate'] * 100:5.1f}% | "
            f"missed={reference_metrics['missed_intent_rate'] * 100:5.1f}%"
        )
        print(
            f"    {point:8s} selected  | intent BA="
            f"{selected_metrics['intent_balanced_accuracy'] * 100:5.1f}% | "
            f"FA={selected_metrics['false_activation_rate'] * 100:5.1f}% | "
            f"missed={selected_metrics['missed_intent_rate'] * 100:5.1f}%"
        )
    print(
        f"[+] Saved controlled ablation to {args.output_dir.resolve()}"
    )
    print(f"[+] Runtime: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
