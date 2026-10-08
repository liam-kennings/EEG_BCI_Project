#!/usr/bin/env python3
"""Matched unilateral/bilateral execution and imagery benchmark with presentation visuals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402

from bci_pipeline.dataset import (  # noqa: E402
    ALL_TASK_RUNS,
    BILATERAL_EXECUTION_RUNS,
    BILATERAL_IMAGERY_RUNS,
    BILATERAL_LABEL_NAMES,
    UNILATERAL_EXECUTION_RUNS,
    UNILATERAL_IMAGERY_RUNS,
    UNILATERAL_LABEL_NAMES,
    available_subjects,
    find_run_file,
    map_task_annotation,
    normalize_channel_name,
    parse_int_list,
    resolve_data_dir,
)
from bci_pipeline.epochs import EpochConfig, RawEpochDataset, build_raw_epoch_dataset  # noqa: E402
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.models.eegnet import (  # noqa: E402
    select_balanced_intent_threshold,
    select_safety_intent_threshold,
)
from bci_pipeline.models.fbcsp import HierarchicalFBCSP  # noqa: E402
from bci_pipeline.preprocessing import PreprocessingProfile, preprocess_run  # noqa: E402
from bci_pipeline.splits import SubjectPartitions, load_partitions, three_way_subject_split  # noqa: E402


MODE_COLORS = {"execution": "#1D4E89", "imagery": "#D05A3A"}
CONDITION_COLORS = {
    "unilateral_execution": "#174A7E",
    "unilateral_imagery": "#D05A3A",
    "bilateral_execution": "#3B7F70",
    "bilateral_imagery": "#D89A32",
}
TASK_SPECS: Dict[str, Dict[str, object]] = {
    "unilateral_execution": {
        "family": "unilateral",
        "mode": "execution",
        "title": "Unilateral movement",
        "runs": UNILATERAL_EXECUTION_RUNS,
        "final_run": 11,
        "labels": UNILATERAL_LABEL_NAMES,
    },
    "unilateral_imagery": {
        "family": "unilateral",
        "mode": "imagery",
        "title": "Unilateral imagery",
        "runs": UNILATERAL_IMAGERY_RUNS,
        "final_run": 12,
        "labels": UNILATERAL_LABEL_NAMES,
    },
    "bilateral_execution": {
        "family": "bilateral",
        "mode": "execution",
        "title": "Bilateral movement",
        "runs": BILATERAL_EXECUTION_RUNS,
        "final_run": 13,
        "labels": BILATERAL_LABEL_NAMES,
    },
    "bilateral_imagery": {
        "family": "bilateral",
        "mode": "imagery",
        "title": "Bilateral imagery",
        "runs": BILATERAL_IMAGERY_RUNS,
        "final_run": 14,
        "labels": BILATERAL_LABEL_NAMES,
    },
}
CONDITION_ORDER = tuple(TASK_SPECS)
# Retained for older helper functions; expanded figures use CONDITION_COLORS.
TASK_COLORS = MODE_COLORS
TASK_LABELS = {"execution": "Physical movement", "imagery": "Motor imagery"}
LABEL_IDS = (0, 1, 2)


def configure_plot_style() -> None:
    """Use one restrained, colorblind-aware visual style across every figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "savefig.facecolor": "white",
        }
    )


def load_dataset(cache_path: Path) -> Tuple[RawEpochDataset, Dict[str, object]]:
    with np.load(cache_path, allow_pickle=False) as cache:
        fingerprint = json.loads(str(cache["fingerprint"].item()))
        dataset = RawEpochDataset(
            epochs=cache["epochs"],
            baselines=cache["baselines"],
            labels=cache["labels"],
            subjects=cache["subjects"],
            runs=cache["runs"],
            trial_indices=cache["trial_indices"],
            channel_names=[str(name) for name in cache["channel_names"]],
            sfreq=float(cache["sfreq"].item()),
        )
    return dataset, fingerprint


def task_masks(dataset: RawEpochDataset, partitions: SubjectPartitions) -> Dict[str, np.ndarray]:
    subject_masks = {
        "development": np.isin(dataset.subjects, partitions.development),
        "validation": np.isin(dataset.subjects, partitions.controller_validation),
        "test": np.isin(dataset.subjects, partitions.final_target),
    }
    masks: Dict[str, np.ndarray] = {}
    for condition, spec in TASK_SPECS.items():
        runs = spec["runs"]
        final_run = spec["final_run"]
        masks[f"{condition}_development"] = subject_masks["development"] & np.isin(dataset.runs, runs)
        masks[f"{condition}_validation"] = subject_masks["validation"] & (dataset.runs == final_run)
        masks[f"{condition}_test"] = subject_masks["test"] & (dataset.runs == final_run)
    final_repeats = tuple(int(TASK_SPECS[key]["final_run"]) for key in CONDITION_ORDER)
    masks["pooled_intent_development"] = subject_masks["development"] & np.isin(
        dataset.runs, ALL_TASK_RUNS
    )
    masks["pooled_intent_validation"] = subject_masks["validation"] & np.isin(
        dataset.runs, final_repeats
    )
    masks["pooled_intent_test"] = subject_masks["test"] & np.isin(
        dataset.runs, final_repeats
    )
    return masks


def restrict_partitions_to_cache(
    partitions: SubjectPartitions, available_subjects: set[int]
) -> Tuple[SubjectPartitions, List[int]]:
    """Keep a cached cohort reproducible when a recording was unavailable at cache creation."""
    requested = set(
        partitions.development + partitions.controller_validation + partitions.final_target
    )
    missing = sorted(requested - available_subjects)
    effective = SubjectPartitions(
        development=tuple(subject for subject in partitions.development if subject in available_subjects),
        controller_validation=tuple(
            subject for subject in partitions.controller_validation if subject in available_subjects
        ),
        final_target=tuple(subject for subject in partitions.final_target if subject in available_subjects),
        random_seed=partitions.random_seed,
        schema_version=partitions.schema_version,
    )
    effective.validate()
    return effective, missing


def labels_from_probabilities(
    intent_probabilities: np.ndarray,
    direction_probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    direction = np.argmax(direction_probabilities, axis=1) + 1
    return np.where(intent_probabilities >= threshold, direction, 0).astype(np.int64)


def per_subject_metrics(
    labels: np.ndarray,
    intent_probabilities: np.ndarray,
    direction_probabilities: np.ndarray,
    subjects: np.ndarray,
    threshold: float,
    label_names: Dict[int, str],
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        results[f"S{int(subject):03d}"] = hierarchical_metrics(
            labels[mask], intent_probabilities[mask], direction_probabilities[mask], threshold,
            label_names,
        )
    return results


def threshold_curve(
    labels: np.ndarray,
    intent_probabilities: np.ndarray,
    direction_probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        metrics = hierarchical_metrics(labels, intent_probabilities, direction_probabilities, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "false_activation_rate": float(metrics["false_activation_rate"]),
                "intent_sensitivity": float(1.0 - metrics["missed_intent_rate"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
    return rows


def train_task(
    condition: str,
    dataset: RawEpochDataset,
    masks: Dict[str, np.ndarray],
    max_false_activation: float,
    seed: int,
) -> Tuple[HierarchicalFBCSP, Dict[str, object], List[Dict[str, object]]]:
    spec = TASK_SPECS[condition]
    label_names = spec["labels"]
    train = masks[f"{condition}_development"]
    validation = masks[f"{condition}_validation"]
    test = masks[f"{condition}_test"]
    model = HierarchicalFBCSP(dataset.sfreq, random_seed=seed).fit(
        dataset.epochs[train], dataset.labels[train]
    )
    validation_intent, _ = model.predict_stage_probabilities(dataset.epochs[validation])
    balanced_threshold = select_balanced_intent_threshold(dataset.labels[validation], validation_intent)
    safety_threshold = select_safety_intent_threshold(
        dataset.labels[validation], validation_intent, max_false_activation
    )
    test_intent, test_direction = model.predict_stage_probabilities(dataset.epochs[test])
    balanced_predictions = labels_from_probabilities(test_intent, test_direction, balanced_threshold)
    safety_predictions = labels_from_probabilities(test_intent, test_direction, safety_threshold)
    balanced_metrics = hierarchical_metrics(
        dataset.labels[test], test_intent, test_direction, balanced_threshold, label_names
    )
    safety_metrics = hierarchical_metrics(
        dataset.labels[test], test_intent, test_direction, safety_threshold, label_names
    )
    predictions: List[Dict[str, object]] = []
    for index in range(int(np.sum(test))):
        predictions.append(
            {
                "condition": condition,
                "task_family": spec["family"],
                "modality": spec["mode"],
                "subject": int(dataset.subjects[test][index]),
                "run": int(dataset.runs[test][index]),
                "trial_index": int(dataset.trial_indices[test][index]),
                "true_label": label_names[int(dataset.labels[test][index])],
                "intent_probability": float(test_intent[index]),
                "class_1_name": label_names[1],
                "class_1_probability": float(test_direction[index, 0]),
                "class_2_name": label_names[2],
                "class_2_probability": float(test_direction[index, 1]),
                "balanced_prediction": label_names[int(balanced_predictions[index])],
                "safety_prediction": label_names[int(safety_predictions[index])],
            }
        )
    result: Dict[str, object] = {
        "condition": condition,
        "task_family": spec["family"],
        "modality": spec["mode"],
        "runs": list(spec["runs"]),
        "final_run": spec["final_run"],
        "label_names": label_names,
        "model": "Hierarchical FBCSP + regularized logistic regression",
        "train_trials": int(np.sum(train)),
        "validation_trials": int(np.sum(validation)),
        "test_trials": int(np.sum(test)),
        "balanced_threshold": float(balanced_threshold),
        "safety_threshold": float(safety_threshold),
        "balanced": balanced_metrics,
        "safety": safety_metrics,
        "per_subject": per_subject_metrics(
            dataset.labels[test], test_intent, test_direction, dataset.subjects[test], balanced_threshold,
            label_names,
        ),
        "threshold_curve_final_test_descriptive_only": threshold_curve(
            dataset.labels[test], test_intent, test_direction, np.linspace(0.05, 0.95, 19)
        ),
    }
    return model, result, predictions


def intent_only_metrics(
    labels: np.ndarray, intent_probabilities: np.ndarray, threshold: float
) -> Dict[str, object]:
    true_intent = (labels != 0).astype(np.int64)
    predicted_intent = (intent_probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(true_intent, predicted_intent, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    rest_total = max(true_negative + false_positive, 1)
    intent_total = max(true_positive + false_negative, 1)
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(true_intent, predicted_intent)),
        "roc_auc": float(roc_auc_score(true_intent, intent_probabilities)),
        "specificity": float(true_negative / rest_total),
        "intent_sensitivity": float(true_positive / intent_total),
        "false_activation_rate": float(false_positive / rest_total),
        "missed_intent_rate": float(false_negative / intent_total),
        "confusion_matrix": matrix.tolist(),
    }


def intent_only_threshold_curve(
    labels: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float]
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        metrics = intent_only_metrics(labels, probabilities, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "false_activation_rate": float(metrics["false_activation_rate"]),
                "intent_sensitivity": float(metrics["intent_sensitivity"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
            }
        )
    return rows


def train_pooled_intent(
    dataset: RawEpochDataset,
    masks: Dict[str, np.ndarray],
    max_false_activation: float,
    seed: int,
) -> Tuple[HierarchicalFBCSP, Dict[str, object], List[Dict[str, object]]]:
    train = masks["pooled_intent_development"]
    validation = masks["pooled_intent_validation"]
    test = masks["pooled_intent_test"]
    model = HierarchicalFBCSP(dataset.sfreq, random_seed=seed).fit(
        dataset.epochs[train], dataset.labels[train]
    )
    validation_intent, _ = model.predict_stage_probabilities(dataset.epochs[validation])
    balanced_threshold = select_balanced_intent_threshold(dataset.labels[validation], validation_intent)
    safety_threshold = select_safety_intent_threshold(
        dataset.labels[validation], validation_intent, max_false_activation
    )
    test_intent, _ = model.predict_stage_probabilities(dataset.epochs[test])
    balanced = intent_only_metrics(dataset.labels[test], test_intent, balanced_threshold)
    safety = intent_only_metrics(dataset.labels[test], test_intent, safety_threshold)
    per_subject: Dict[str, Dict[str, object]] = {}
    for subject in sorted(np.unique(dataset.subjects[test]).tolist()):
        subject_mask = dataset.subjects[test] == subject
        per_subject[f"S{int(subject):03d}"] = intent_only_metrics(
            dataset.labels[test][subject_mask], test_intent[subject_mask], balanced_threshold
        )
    rows: List[Dict[str, object]] = []
    for index in range(int(np.sum(test))):
        run = int(dataset.runs[test][index])
        condition = next(
            key for key, spec in TASK_SPECS.items() if run == int(spec["final_run"])
        )
        rows.append(
            {
                "condition": condition,
                "subject": int(dataset.subjects[test][index]),
                "run": run,
                "trial_index": int(dataset.trial_indices[test][index]),
                "true_state": "intent" if dataset.labels[test][index] != 0 else "rest",
                "intent_probability": float(test_intent[index]),
                "balanced_prediction": "intent" if test_intent[index] >= balanced_threshold else "rest",
                "safety_prediction": "intent" if test_intent[index] >= safety_threshold else "rest",
            }
        )
    result: Dict[str, object] = {
        "task": "pooled rest versus motor intent across all unilateral and bilateral task runs",
        "model": "Pooled FBCSP intent gate + regularized logistic regression",
        "train_trials": int(np.sum(train)),
        "validation_trials": int(np.sum(validation)),
        "test_trials": int(np.sum(test)),
        "balanced_threshold": float(balanced_threshold),
        "safety_threshold": float(safety_threshold),
        "balanced": balanced,
        "safety": safety,
        "per_subject": per_subject,
        "threshold_curve_final_test_descriptive_only": intent_only_threshold_curve(
            dataset.labels[test], test_intent, np.linspace(0.05, 0.95, 19)
        ),
    }
    return model, result, rows


def plot_pipeline_overview(output_dir: Path, partitions: SubjectPartitions) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.5))
    ax.set_axis_off()
    columns = [
        ("Task-specific EEG", ""),
        ("Shared preprocessing", "4-35 Hz, CAR, artifact\nQC, baseline centering"),
        ("Mu/beta features", "9 sensorimotor channels\nfilter-bank CSP"),
        ("1. Intent gate", "Rest vs. motor intent\nselected threshold"),
        ("2. Direction head", "Left vs. right\nif intent is accepted"),
        ("Subject-isolated test", ""),
    ]
    x_positions = np.linspace(0.14, 0.84, len(columns))
    task_details = {
        "execution": "Physical movement\nRuns 3 / 7 / 11",
        "imagery": "Imagined movement\nRuns 4 / 8 / 12",
    }
    test_details = {
        "execution": f"Run 11 | {len(partitions.final_target)}\nunseen subjects",
        "imagery": f"Run 12 | {len(partitions.final_target)}\nunseen subjects",
    }
    for row, task in enumerate(("execution", "imagery")):
        y = 0.67 if task == "execution" else 0.22
        ax.text(0.01, y + 0.075, TASK_LABELS[task], color=TASK_COLORS[task], weight="bold", fontsize=12)
        for index, ((title, detail), x) in enumerate(zip(columns, x_positions)):
            if index == 0:
                detail = task_details[task]
            elif index == len(columns) - 1:
                detail = test_details[task]
            patch = FancyBboxPatch(
                (x, y), 0.115, 0.19, boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.5, edgecolor=TASK_COLORS[task], facecolor="#F7FAFC"
            )
            ax.add_patch(patch)
            ax.text(x + 0.0575, y + 0.132, title, ha="center", va="center", weight="bold", fontsize=8.2)
            ax.text(x + 0.0575, y + 0.060, detail, ha="center", va="center", color="#44546A", fontsize=7.5)
            if index < len(columns) - 1:
                ax.annotate(
                    "", xy=(x + 0.135, y + 0.095), xytext=(x + 0.118, y + 0.095),
                    arrowprops={"arrowstyle": "->", "color": "#6B7C93", "lw": 1.4}
                )
    ax.text(
        0.5, 0.96, "Matched Dual Pipeline: Physical Movement vs. Motor Imagery",
        ha="center", va="center", fontsize=16, weight="bold", color="#152536"
    )
    ax.text(
        0.5, 0.03,
        "Only the task condition differs. Preprocessing, channels, features, classifier, and subject-isolated evaluation are identical.",
        ha="center", va="center", fontsize=10, color="#44546A"
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(output_dir / "dual_pipeline_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_isolation(partitions: SubjectPartitions, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 4.2))
    ax.set_axis_off()
    stages = [
        (
            "Development",
            len(partitions.development),
            "Fit CSP spatial filters,\nscalers, and classifiers",
            "All execution + imagery runs\n3 through 14",
            "#DCE9F5",
        ),
        (
            "Controller validation",
            len(partitions.controller_validation),
            "Select intent thresholds\nwithout final-test access",
            "Final repeats\nRuns 11/12/13/14",
            "#F7E8CE",
        ),
        (
            "Final unseen test",
            len(partitions.final_target),
            "Report headline metrics\nand error patterns",
            "Final repeats\nRuns 11/12/13/14",
            "#DDEFE9",
        ),
    ]
    x_positions = (0.05, 0.37, 0.69)
    for index, ((title, count, role, runs, color), x) in enumerate(zip(stages, x_positions)):
        patch = FancyBboxPatch(
            (x, 0.22), 0.25, 0.55, boxstyle="round,pad=0.016,rounding_size=0.025",
            linewidth=1.4, edgecolor="#52677A", facecolor=color
        )
        ax.add_patch(patch)
        ax.text(x + 0.125, 0.68, title, ha="center", weight="bold", fontsize=12, color="#152536")
        ax.text(x + 0.125, 0.54, f"{count} subjects", ha="center", weight="bold", fontsize=16, color="#1D4E89")
        ax.text(x + 0.125, 0.40, role, ha="center", va="center", fontsize=9, color="#44546A")
        ax.text(x + 0.125, 0.27, runs, ha="center", va="center", fontsize=8, color="#44546A")
        if index < len(stages) - 1:
            ax.annotate(
                "", xy=(x + 0.31, 0.50), xytext=(x + 0.27, 0.50),
                arrowprops={"arrowstyle": "->", "color": "#52677A", "lw": 1.5}
            )
            ax.text(x + 0.29, 0.53, "No overlap", ha="center", fontsize=7.5, color="#52677A")
    ax.text(0.5, 0.92, "Subject-Isolated Model Development and Evaluation", ha="center", fontsize=16, weight="bold", color="#152536")
    ax.text(0.5, 0.08, "Final-test participants never contribute to feature fitting, model training, or threshold selection.", ha="center", fontsize=10, color="#44546A")
    fig.tight_layout(pad=0.4)
    fig.savefig(output_dir / "dual_task_subject_isolation_protocol.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_abstract_key_findings(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    tasks = ("execution", "imagery")
    metrics = ("balanced_accuracy", "intent_balanced_accuracy", "direction_balanced_accuracy")
    labels = ("Three-class\nbalanced accuracy", "Rest vs. intent\nbalanced accuracy", "Left vs. right\nbalanced accuracy")
    positions = np.arange(len(metrics))
    width = 0.33
    for offset, task in zip((-width / 2, width / 2), tasks):
        values = [results[task]["balanced"][metric] for metric in metrics]
        bars = axes[0].bar(positions + offset, values, width, color=TASK_COLORS[task], label=TASK_LABELS[task])
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.022, f"{value * 100:.1f}%", ha="center", fontsize=9, weight="bold")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("Final unseen-subject rate")
    axes[0].set_title("Where Imagery Performance Is Lost")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")

    error_metrics = ("false_activation_rate", "missed_intent_rate")
    error_labels = ("False activation\nduring rest", "Missed\nmotor intent")
    positions = np.arange(len(error_metrics))
    for offset, task in zip((-width / 2, width / 2), tasks):
        values = [results[task]["balanced"][metric] for metric in error_metrics]
        bars = axes[1].bar(positions + offset, values, width, color=TASK_COLORS[task], label=TASK_LABELS[task])
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.022, f"{value * 100:.1f}%", ha="center", fontsize=9, weight="bold")
    axes[1].set_xticks(positions, error_labels)
    axes[1].set_ylim(0, 0.9)
    axes[1].set_ylabel("Final unseen-subject rate")
    axes[1].set_title("Safety-Critical Intent-Gate Errors")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="upper right")

    intent_gap = 100 * (
        results["execution"]["balanced"]["intent_balanced_accuracy"]
        - results["imagery"]["balanced"]["intent_balanced_accuracy"]
    )
    direction_gap = 100 * (
        results["execution"]["balanced"]["direction_balanced_accuracy"]
        - results["imagery"]["balanced"]["direction_balanced_accuracy"]
    )
    fig.suptitle(
        f"Key Finding: Intent Gap {intent_gap:.1f} points vs. Direction Gap {direction_gap:.1f} points",
        fontsize=15, weight="bold", color="#152536"
    )
    fig.savefig(output_dir / "dual_task_abstract_key_findings.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), constrained_layout=True)
    class_labels = ["Rest\n(no command)", "Left fist", "Right fist"]
    for ax, task in zip(axes, ("execution", "imagery")):
        metrics = results[task]["balanced"]
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
        image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(3):
            for column in range(3):
                color = "white" if normalized[row, column] >= 0.55 else "#152536"
                ax.text(
                    column, row, f"{normalized[row, column] * 100:.1f}%\n(n={int(matrix[row, column])})",
                    ha="center", va="center", color=color, fontsize=10, weight="bold"
                )
        ax.set_xticks(range(3), class_labels)
        ax.set_yticks(range(3), class_labels)
        ax.set_xlabel("Predicted state")
        ax.set_ylabel("True state")
        ax.set_title(
            f"{TASK_LABELS[task]}\nBA {metrics['balanced_accuracy'] * 100:.1f}% | "
            f"n = {results[task]['test_trials']}",
            color=TASK_COLORS[task]
        )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colorbar.set_label("Within-class percentage")
    fig.suptitle("Three-Class Performance on Completely Unseen Subjects", fontsize=15, weight="bold")
    fig.savefig(output_dir / "dual_task_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_performance(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1), constrained_layout=True)
    tasks = ("execution", "imagery")
    x = np.arange(2)
    colors = [TASK_COLORS[task] for task in tasks]
    metric_specs = (
        ("Three-class decoding", ["Balanced\naccuracy", "Macro F1", "Intent BA", "Direction BA"], [
            "balanced_accuracy", "macro_f1", "intent_balanced_accuracy", "direction_balanced_accuracy"
        ]),
        ("Command safety", ["False\nactivation", "Missed\nintent"], ["false_activation_rate", "missed_intent_rate"]),
    )
    for axis, (title, labels, keys) in zip(axes[:2], metric_specs):
        width = 0.32
        positions = np.arange(len(keys))
        for offset, task in zip((-width / 2, width / 2), tasks):
            metrics = results[task]["balanced"]
            values = [metrics[key] for key in keys]
            bars = axis.bar(positions + offset, values, width, color=TASK_COLORS[task], label=TASK_LABELS[task])
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value * 100:.1f}", ha="center", fontsize=9, weight="bold")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, loc="upper right", fontsize=8)

    class_names = [UNILATERAL_LABEL_NAMES[label] for label in LABEL_IDS]
    positions = np.arange(len(class_names))
    width = 0.32
    for offset, task in zip((-width / 2, width / 2), tasks):
        report = results[task]["balanced"]["classification_report"]
        values = [report[name]["f1-score"] for name in class_names]
        bars = axes[2].bar(positions + offset, values, width, color=TASK_COLORS[task], label=TASK_LABELS[task])
        for bar, value in zip(bars, values):
            axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value * 100:.1f}", ha="center", fontsize=9, weight="bold")
    axes[2].set_xticks(positions, ["Rest\n(no command)", "Left fist", "Right fist"])
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("F1 score")
    axes[2].set_title("Class-wise F1 score")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Matched FBCSP Decoder: Final Unseen-Subject Comparison", fontsize=15, weight="bold")
    fig.savefig(output_dir / "dual_task_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_comparison(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    execution = results["execution"]["per_subject"]
    imagery = results["imagery"]["per_subject"]
    subjects = sorted(set(execution) & set(imagery))
    execution_values = np.asarray([execution[subject]["balanced_accuracy"] for subject in subjects])
    imagery_values = np.asarray([imagery[subject]["balanced_accuracy"] for subject in subjects])
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(execution_values, imagery_values, s=55, color="#386FA4", alpha=0.82, edgecolor="white", linewidth=0.7)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.6, color="#697B8C", label="Equal performance")
    ax.axvline(np.median(execution_values), color=TASK_COLORS["execution"], linewidth=1.5, alpha=0.75)
    ax.axhline(np.median(imagery_values), color=TASK_COLORS["imagery"], linewidth=1.5, alpha=0.75)
    for subject, x_value, y_value in zip(subjects, execution_values, imagery_values):
        if abs(x_value - y_value) > 0.19:
            ax.annotate(subject, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#44546A")
    ax.set_xlim(0.20, 0.95)
    ax.set_ylim(0.20, 0.95)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Physical-movement balanced accuracy")
    ax.set_ylabel("Motor-imagery balanced accuracy")
    ax.set_title(f"Per-Subject Decoding Gap (n = {len(subjects)} unseen subjects)")
    ax.grid(alpha=0.23)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98, 0.04,
        f"Median movement: {np.median(execution_values) * 100:.1f}%\nMedian imagery: {np.median(imagery_values) * 100:.1f}%",
        ha="right", va="bottom", transform=ax.transAxes, fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D9E2EC"}
    )
    fig.tight_layout()
    fig.savefig(output_dir / "dual_task_per_subject_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_safety_tradeoff(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.7, 5.8))
    for task in ("execution", "imagery"):
        curve = results[task]["threshold_curve_final_test_descriptive_only"]
        x_values = [point["false_activation_rate"] for point in curve]
        y_values = [point["intent_sensitivity"] for point in curve]
        ax.plot(x_values, y_values, color=TASK_COLORS[task], linewidth=2.5, label=TASK_LABELS[task])
        for operating_point, marker, label in (("balanced", "o", "balanced"), ("safety", "s", "safety")):
            metrics = results[task][operating_point]
            ax.scatter(
                metrics["false_activation_rate"], 1.0 - metrics["missed_intent_rate"],
                color=TASK_COLORS[task], marker=marker, s=75, edgecolor="white", linewidth=0.8,
                zorder=4, label=f"{TASK_LABELS[task]} {label} threshold"
            )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False-activation rate during rest")
    ax.set_ylabel("Intent sensitivity (1 - missed-intent rate)")
    ax.set_title("Safety Tradeoff on Final Unseen Subjects")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(
        0.5, 0.012, "Curves are descriptive; thresholds were selected on separate controller subjects.",
        ha="center", fontsize=8, color="#44546A"
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "dual_task_safety_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expanded_pipeline(output_dir: Path, partitions: SubjectPartitions) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 6.0))
    ax.set_axis_off()
    rows = (
        ("unilateral_execution", 0.72),
        ("unilateral_imagery", 0.52),
        ("bilateral_execution", 0.32),
        ("bilateral_imagery", 0.12),
    )
    columns = (0.03, 0.25, 0.47, 0.69, 0.87)
    for condition, y in rows:
        spec = TASK_SPECS[condition]
        labels = spec["labels"]
        details = (
            f"{spec['title']}\nRuns {' / '.join(str(run) for run in spec['runs'])}\n"
            f"rest | {labels[1].replace('_', ' ')} | {labels[2].replace('_', ' ')}",
            "4-35 Hz + CAR\nartifact QC + baseline",
            "Mu/beta FBCSP\n9 sensorimotor channels",
            "1. Rest vs. intent\n2. Task-specific class",
            f"Run {spec['final_run']}\n{len(partitions.final_target)} unseen subjects",
        )
        titles = ("Task EEG", "Shared preprocessing", "Shared features", "Hierarchical decoder", "Final test")
        for index, (x, title, detail) in enumerate(zip(columns, titles, details)):
            width = 0.15 if index < 4 else 0.11
            patch = FancyBboxPatch(
                (x, y), width, 0.13, boxstyle="round,pad=0.009,rounding_size=0.015",
                linewidth=1.35, edgecolor=CONDITION_COLORS[condition], facecolor="#F8FAFC"
            )
            ax.add_patch(patch)
            ax.text(x + width / 2, y + 0.092, title, ha="center", va="center", weight="bold", fontsize=8.3)
            ax.text(x + width / 2, y + 0.043, detail, ha="center", va="center", fontsize=6.8, color="#44546A")
            if index < len(columns) - 1:
                next_x = columns[index + 1]
                ax.annotate(
                    "", xy=(next_x - 0.008, y + 0.065), xytext=(x + width + 0.008, y + 0.065),
                    arrowprops={"arrowstyle": "->", "color": "#6B7C93", "lw": 1.2}
                )
    ax.text(0.5, 0.95, "Expanded EEGMMIDB Movement-vs.-Imagery Benchmark", ha="center", fontsize=17, weight="bold", color="#152536")
    ax.text(
        0.5, 0.015,
        "A pooled binary intent gate also combines all correctly mapped task runs; task-specific class labels are never mixed across families.",
        ha="center", fontsize=9.5, color="#44546A"
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(output_dir / "dual_pipeline_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def signal_example(
    data_dir: Path,
    subject: int,
    run: int,
    profile: PreprocessingProfile,
    channels: Tuple[str, ...] = ("C3", "Cz", "C4"),
) -> Dict[str, object]:
    """Load one cue-centered epoch before and after the benchmark preprocessing."""
    file_path = find_run_file(data_dir, subject, run)
    if file_path is None:
        raise FileNotFoundError(f"Missing S{subject:03d}R{run:02d}.edf")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw_eeg = raw.copy().pick("eeg").load_data()
    lookup = {normalize_channel_name(name): name for name in raw_eeg.ch_names}
    raw_selected = raw_eeg.pick([lookup[normalize_channel_name(name)] for name in channels])
    processed = preprocess_run(raw, profile, channels)
    event = next(
        annotation
        for annotation in raw.annotations
        if map_task_annotation(str(annotation["description"])) in (1, 2)
    )
    event_label = int(map_task_annotation(str(event["description"])))
    onset = float(event["onset"])
    start_seconds = max(0.0, onset - 1.0)
    stop_seconds = min(float(raw.times[-1]), onset + 3.5)
    raw_start = int(round(start_seconds * float(raw_selected.info["sfreq"])))
    raw_stop = int(round(stop_seconds * float(raw_selected.info["sfreq"])))
    processed_start = int(round(start_seconds * float(processed.info["sfreq"])))
    processed_stop = int(round(stop_seconds * float(processed.info["sfreq"])))
    raw_uv = raw_selected.get_data(start=raw_start, stop=raw_stop) * 1e6
    processed_uv = processed.get_data(start=processed_start, stop=processed_stop) * 1e6
    sample_count = min(raw_uv.shape[1], processed_uv.shape[1])
    raw_uv = raw_uv[:, :sample_count]
    processed_uv = processed_uv[:, :sample_count]
    times = np.arange(sample_count) / profile.sfreq + start_seconds - onset
    baseline = times < 0.0
    # Per-channel offsets make the voltage traces comparable without filtering the raw panel.
    raw_uv -= np.mean(raw_uv[:, baseline], axis=1, keepdims=True)
    processed_uv -= np.mean(processed_uv[:, baseline], axis=1, keepdims=True)
    return {
        "raw_uv": raw_uv,
        "processed_uv": processed_uv,
        "times": times,
        "event_label": event_label,
        "channels": channels,
    }


def plot_raw_vs_processed(
    data_dir: Path,
    subject: int,
    profile: PreprocessingProfile,
    output_dir: Path,
) -> None:
    examples = (
        ("unilateral_execution", signal_example(data_dir, subject, 11, profile)),
        ("unilateral_imagery", signal_example(data_dir, subject, 12, profile)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), sharex=True, constrained_layout=True)
    for row, (condition, example) in enumerate(examples):
        label_names = TASK_SPECS[condition]["labels"]
        event_name = str(label_names[example["event_label"]]).replace("_", " ").title()
        combined = np.concatenate((example["raw_uv"], example["processed_uv"]), axis=1)
        robust_amplitude = max(float(np.percentile(np.abs(combined), 99.0)), 12.0)
        spacing = robust_amplitude * 2.4
        scale_uv = 50.0 if robust_amplitude >= 35.0 else 25.0
        offsets = np.arange(len(example["channels"]) - 1, -1, -1) * spacing
        for column, (signal_key, panel_title) in enumerate(
            (
                ("raw_uv", "Raw EEG (display offset only)"),
                ("processed_uv", "Processed EEG (4-35 Hz + CAR + baseline)"),
            )
        ):
            axis = axes[row, column]
            for channel_index, (channel, offset) in enumerate(zip(example["channels"], offsets)):
                axis.plot(
                    example["times"], example[signal_key][channel_index] + offset,
                    color=CONDITION_COLORS[condition], linewidth=0.85,
                )
            axis.axvline(0.0, color="#C23B22", linewidth=1.5, linestyle="--", label="Cue onset")
            axis.axvspan(0.5, 3.5, color="#DCE9F5", alpha=0.32, label="Analyzed window")
            axis.set_yticks(offsets, example["channels"])
            axis.set_ylim(-spacing, offsets[0] + spacing)
            axis.set_xlim(-1.0, 3.5)
            scale_base = -0.78 * spacing
            axis.plot((3.22, 3.22), (scale_base, scale_base + scale_uv), color="#152536", linewidth=2.0)
            axis.text(3.17, scale_base + scale_uv / 2.0, f"{scale_uv:.0f} uV", ha="right", va="center", fontsize=7.5)
            axis.set_title(panel_title)
            axis.grid(axis="x", alpha=0.18)
            axis.set_ylabel(f"{TASK_SPECS[condition]['title']}\n{event_name}\nChannel")
            if row == 1:
                axis.set_xlabel("Time from cue onset (seconds)")
            if row == 0 and column == 1:
                axis.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        f"Representative Raw and Processed Sensorimotor EEG | Development Subject S{subject:03d}",
        fontsize=15, weight="bold", color="#152536",
    )
    fig.savefig(output_dir / "raw_vs_processed_eeg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def full_trial_band_example(
    data_dir: Path,
    subject: int,
    run: int,
    profile: PreprocessingProfile,
    channels: Tuple[str, ...] = ("C3", "Cz", "C4"),
) -> Dict[str, object]:
    """Return the first complete T0-to-task cycle and its mu/beta waveforms."""
    file_path = find_run_file(data_dir, subject, run)
    if file_path is None:
        raise FileNotFoundError(f"Missing S{subject:03d}R{run:02d}.edf")
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    processed = preprocess_run(raw, profile, channels)
    annotations = list(raw.annotations)
    task_index = next(
        index
        for index, annotation in enumerate(annotations)
        if index > 0
        and map_task_annotation(str(annotation["description"])) in (1, 2)
        and map_task_annotation(str(annotations[index - 1]["description"])) == 0
    )
    rest_event = annotations[task_index - 1]
    task_event = annotations[task_index]
    rest_onset = float(rest_event["onset"])
    task_onset = float(task_event["onset"])
    task_duration = float(task_event["duration"])
    trial_stop = min(float(processed.times[-1]), task_onset + task_duration)
    sfreq = float(processed.info["sfreq"])
    start_sample = int(round(rest_onset * sfreq))
    stop_sample = int(round(trial_stop * sfreq))
    full_signal = processed.get_data()
    mu_sos = butter(4, (7.0, 13.0), btype="bandpass", fs=sfreq, output="sos")
    beta_sos = butter(4, (13.0, 30.0), btype="bandpass", fs=sfreq, output="sos")
    signals = {
        "broadband": full_signal[:, start_sample:stop_sample].copy(),
        "mu": sosfiltfilt(mu_sos, full_signal, axis=-1)[:, start_sample:stop_sample],
        "beta": sosfiltfilt(beta_sos, full_signal, axis=-1)[:, start_sample:stop_sample],
    }
    times = np.arange(stop_sample - start_sample) / sfreq
    task_onset_relative = task_onset - rest_onset
    baseline = (times >= task_onset_relative - 1.0) & (times < task_onset_relative)
    for key in signals:
        signals[key] = signals[key] * 1e6
        signals[key] -= np.mean(signals[key][:, baseline], axis=1, keepdims=True)
    return {
        **signals,
        "times": times,
        "task_onset": task_onset_relative,
        "task_stop": trial_stop - rest_onset,
        "event_label": int(map_task_annotation(str(task_event["description"]))),
        "event_code": str(task_event["description"]).strip().upper(),
        "channels": channels,
    }


def plot_full_trial_mu_beta(
    data_dir: Path,
    subject: int,
    profile: PreprocessingProfile,
    output_dir: Path,
) -> None:
    examples = {
        "unilateral_execution": full_trial_band_example(data_dir, subject, 11, profile),
        "unilateral_imagery": full_trial_band_example(data_dir, subject, 12, profile),
    }
    band_specs = (
        ("broadband", "Processed broadband EEG", "4-35 Hz"),
        ("mu", "Mu-band EEG", "7-13 Hz"),
        ("beta", "Beta-band EEG", "13-30 Hz"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 10.0), sharex="col", constrained_layout=True)
    for row, (signal_key, band_title, band_range) in enumerate(band_specs):
        row_amplitudes = np.concatenate(
            [np.ravel(examples[condition][signal_key]) for condition in examples]
        )
        robust_amplitude = max(float(np.percentile(np.abs(row_amplitudes), 99.0)), 6.0)
        spacing = robust_amplitude * 2.5
        scale_uv = 25.0 if robust_amplitude >= 16.0 else 10.0
        for column, condition in enumerate(("unilateral_execution", "unilateral_imagery")):
            example = examples[condition]
            axis = axes[row, column]
            offsets = np.arange(len(example["channels"]) - 1, -1, -1) * spacing
            for channel_index, offset in enumerate(offsets):
                axis.plot(
                    example["times"], example[signal_key][channel_index] + offset,
                    color=CONDITION_COLORS[condition], linewidth=0.8,
                )
            onset = float(example["task_onset"])
            stop = float(example["task_stop"])
            axis.axvspan(0.0, onset, color="#E9EDF2", alpha=0.62, label="T0 rest")
            axis.axvspan(onset, stop, color=CONDITION_COLORS[condition], alpha=0.10, label="T1/T2 task")
            axis.axvline(onset, color="#C23B22", linewidth=1.5, linestyle="--")
            axis.text(
                onset + 0.08, offsets[0] + 0.72 * spacing,
                f"{example['event_code']} onset = {onset:.2f} s",
                color="#9F2D1B", fontsize=8, weight="bold", va="top",
            )
            axis.set_yticks(offsets, example["channels"])
            axis.set_ylim(-spacing, offsets[0] + spacing)
            axis.set_xlim(0.0, max(float(value["task_stop"]) for value in examples.values()))
            axis.set_ylabel(f"{band_title}\n({band_range})\nChannel")
            axis.grid(axis="x", alpha=0.18)
            scale_base = -0.78 * spacing
            scale_x = stop - 0.35
            axis.plot((scale_x, scale_x), (scale_base, scale_base + scale_uv), color="#152536", linewidth=2.0)
            axis.text(scale_x - 0.06, scale_base + scale_uv / 2.0, f"{scale_uv:.0f} uV", ha="right", va="center", fontsize=7.5)
            if row == 0:
                label_names = TASK_SPECS[condition]["labels"]
                event_name = str(label_names[example["event_label"]]).replace("_", " ").title()
                axis.set_title(f"{TASK_SPECS[condition]['title']} | {event_name}")
            if row == 2:
                axis.set_xlabel("Time from start of full T0-to-task cycle (seconds)")
            if row == 0 and column == 1:
                axis.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        f"Full EEG Trial With Motor-Relevant Mu and Beta Bands | Development Subject S{subject:03d}",
        fontsize=15, weight="bold", color="#152536",
    )
    fig.savefig(output_dir / "full_trial_mu_beta_eeg.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expanded_summary(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)
    positions = np.arange(len(CONDITION_ORDER))
    labels = [str(TASK_SPECS[key]["title"]).replace(" ", "\n", 1) for key in CONDITION_ORDER]
    metrics = (
        ("balanced_accuracy", "Three-class BA", "o"),
        ("intent_balanced_accuracy", "Intent BA", "s"),
        ("direction_balanced_accuracy", "Within-task class BA", "^"),
    )
    for key, label, marker in metrics:
        values = [results[condition]["balanced"][key] for condition in CONDITION_ORDER]
        axes[0].plot(positions, values, marker=marker, linewidth=2.0, markersize=7, label=label)
        for x, value in zip(positions, values):
            axes[0].text(x, value + 0.025, f"{value * 100:.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0.25, 0.95)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_title("Decoding Performance")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8)

    width = 0.34
    for offset, metric, label in (
        (-width / 2, "false_activation_rate", "False activation"),
        (width / 2, "missed_intent_rate", "Missed intent"),
    ):
        values = [results[condition]["balanced"][metric] for condition in CONDITION_ORDER]
        bars = axes[1].bar(
            positions + offset, values, width,
            color=[CONDITION_COLORS[key] for key in CONDITION_ORDER],
            alpha=1.0 if metric == "false_activation_rate" else 0.58,
            label=label,
        )
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value * 100:.1f}", ha="center", fontsize=8)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Error rate")
    axes[1].set_title("Safety-Critical Intent Errors")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Movement and Imagery Across Unilateral and Bilateral Tasks", fontsize=15, weight="bold", color="#152536")
    fig.savefig(output_dir / "dual_task_abstract_key_findings.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "dual_task_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expanded_confusions(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.0), constrained_layout=True)
    image = None
    for ax, condition in zip(axes.flat, CONDITION_ORDER):
        result = results[condition]
        matrix = np.asarray(result["balanced"]["confusion_matrix"], dtype=float)
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
        image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(3):
            for column in range(3):
                ax.text(
                    column, row, f"{normalized[row, column] * 100:.1f}%\n(n={int(matrix[row, column])})",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if normalized[row, column] >= 0.55 else "#152536",
                    weight="bold",
                )
        label_names = result["label_names"]
        axis_labels = [str(label_names[index]).replace("_", " ").title() for index in LABEL_IDS]
        ax.set_xticks(range(3), axis_labels)
        ax.set_yticks(range(3), axis_labels)
        ax.set_xlabel("Predicted state")
        ax.set_ylabel("True state")
        ax.set_title(
            f"{TASK_SPECS[condition]['title']} | BA {result['balanced']['balanced_accuracy'] * 100:.1f}%",
            color=CONDITION_COLORS[condition],
        )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Within-class percentage")
    fig.suptitle("Task-Specific Three-Class Results on Unseen Subjects", fontsize=15, weight="bold")
    fig.savefig(output_dir / "dual_task_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expanded_subject_comparison(results: Dict[str, Dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), constrained_layout=True)
    for ax, family in zip(axes, ("unilateral", "bilateral")):
        execution = results[f"{family}_execution"]["per_subject"]
        imagery = results[f"{family}_imagery"]["per_subject"]
        subjects = sorted(set(execution) & set(imagery))
        x_values = np.asarray([execution[subject]["balanced_accuracy"] for subject in subjects])
        y_values = np.asarray([imagery[subject]["balanced_accuracy"] for subject in subjects])
        ax.scatter(x_values, y_values, s=48, color="#386FA4", alpha=0.82, edgecolor="white", linewidth=0.6)
        ax.plot([0, 1], [0, 1], "--", color="#697B8C", linewidth=1.4, label="Equal performance")
        ax.set_xlim(0.15, 0.95)
        ax.set_ylim(0.15, 0.95)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Movement balanced accuracy")
        ax.set_ylabel("Imagery balanced accuracy")
        ax.set_title(f"{family.title()} tasks (n = {len(subjects)})")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
        ax.text(
            0.97, 0.04,
            f"Median movement {np.median(x_values) * 100:.1f}%\nMedian imagery {np.median(y_values) * 100:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D9E2EC"},
        )
    fig.suptitle("Per-Subject Movement-to-Imagery Decoding Gap", fontsize=15, weight="bold")
    fig.savefig(output_dir / "dual_task_per_subject_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_expanded_safety_tradeoff(
    results: Dict[str, Dict[str, object]], pooled_intent: Dict[str, object], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for condition in CONDITION_ORDER:
        curve = results[condition]["threshold_curve_final_test_descriptive_only"]
        ax.plot(
            [point["false_activation_rate"] for point in curve],
            [point["intent_sensitivity"] for point in curve],
            color=CONDITION_COLORS[condition], linewidth=2.0,
            label=str(TASK_SPECS[condition]["title"]),
        )
    pooled_curve = pooled_intent["threshold_curve_final_test_descriptive_only"]
    ax.plot(
        [point["false_activation_rate"] for point in pooled_curve],
        [point["intent_sensitivity"] for point in pooled_curve],
        color="#172B3A", linewidth=2.8, linestyle="--", label="Pooled intent gate",
    )
    pooled_metrics = pooled_intent["balanced"]
    ax.scatter(
        pooled_metrics["false_activation_rate"], pooled_metrics["intent_sensitivity"],
        s=90, marker="*", color="#172B3A", zorder=5, label="Pooled selected threshold",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False-activation rate during rest")
    ax.set_ylabel("Intent sensitivity")
    ax.set_title("Intent-Gate Safety Tradeoff on Final Unseen Subjects")
    ax.grid(alpha=0.23)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.5, 0.015, "Curves are descriptive; all selected thresholds use separate controller-validation subjects.", ha="center", fontsize=8, color="#44546A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "dual_task_safety_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pooled_intent_summary(pooled_intent: Dict[str, object], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    metrics = pooled_intent["balanced"]
    names = ("Balanced\naccuracy", "ROC AUC", "Rest\nspecificity", "Intent\nsensitivity")
    values = (
        metrics["balanced_accuracy"], metrics["roc_auc"], metrics["specificity"], metrics["intent_sensitivity"]
    )
    bars = axes[0].bar(np.arange(len(values)), values, color=("#1D4E89", "#3B7F70", "#697B8C", "#D05A3A"))
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value * 100:.1f}%", ha="center", weight="bold", fontsize=9)
    axes[0].set_xticks(np.arange(len(values)), names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Final unseen-subject rate")
    axes[0].set_title("Pooled Rest-vs.-Intent Performance")
    axes[0].grid(axis="y", alpha=0.22)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    image = axes[1].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column, row, f"{normalized[row, column] * 100:.1f}%\n(n={int(matrix[row, column])})",
                ha="center", va="center", weight="bold",
                color="white" if normalized[row, column] >= 0.55 else "#152536",
            )
    axes[1].set_xticks((0, 1), ("Rest", "Any intent"))
    axes[1].set_yticks((0, 1), ("Rest", "Any intent"))
    axes[1].set_xlabel("Predicted state")
    axes[1].set_ylabel("True state")
    axes[1].set_title(f"Selected threshold = {pooled_intent['balanced_threshold']:.2f}")
    fig.colorbar(image, ax=axes[1], shrink=0.82, label="Within-class percentage")
    fig.suptitle("Generalized Intent Gate Across All EEGMMIDB Motor Tasks", fontsize=15, weight="bold")
    fig.savefig(output_dir / "pooled_intent_gate_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_predictions(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, report: Dict[str, object]) -> None:
    lines = [
        "# Expanded Physical-Movement and Motor-Imagery Benchmark",
        "",
        "## Design",
        "",
        "Four matched three-class decoders compare physical movement with motor imagery for both EEGMMIDB task families. Unilateral models identify `rest`, `left_fist`, and `right_fist`; bilateral models identify `rest`, `both_fists`, and `both_feet`.",
        "",
        "All conditions use identical cue-locked epochs, preprocessing, central motor channels, filter-bank CSP features, hierarchical logistic-regression heads, and subject-isolated partitions. Event codes are resolved within their task family, so bilateral `T1/T2` events are never mislabeled as left/right fist.",
        "",
        "A fifth, pooled binary model combines all task runs for generalized `rest` versus `motor_intent` detection. Its task-specific second-stage output is intentionally ignored.",
        "",
        "## Final Three-Class Results",
        "",
        "| Condition | Balanced accuracy | Intent BA | Within-task class BA | False activation | Missed intent |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITION_ORDER:
        metrics = report["results"][condition]["balanced"]
        lines.append(
            f"| {TASK_SPECS[condition]['title']} | {metrics['balanced_accuracy'] * 100:.1f}% | "
            f"{metrics['intent_balanced_accuracy'] * 100:.1f}% | "
            f"{metrics['direction_balanced_accuracy'] * 100:.1f}% | "
            f"{metrics['false_activation_rate'] * 100:.1f}% | {metrics['missed_intent_rate'] * 100:.1f}% |"
        )
    pooled = report["pooled_intent"]["balanced"]
    lines.extend(
        [
            "",
            "## Pooled Intent Gate",
            "",
            f"Across all correctly mapped task runs, the pooled rest-versus-intent model achieved {pooled['balanced_accuracy'] * 100:.1f}% balanced accuracy, {pooled['false_activation_rate'] * 100:.1f}% false activation, and {pooled['missed_intent_rate'] * 100:.1f}% missed intent on final unseen subjects.",
            "",
            "## Interpretation",
            "",
            "Movement-to-imagery gaps are interpreted separately for unilateral and bilateral tasks. This tests whether weak imagery intent is a general phenomenon while preserving the distinct neurophysiology and event semantics of hand-laterality versus hand/foot discrimination.",
            "",
            "## Presentation Figures",
            "",
            "- `dual_pipeline_overview.png`: four-condition pipeline schematic.",
            "- `raw_vs_processed_eeg.png`: representative raw and processed sensorimotor traces.",
            "- `full_trial_mu_beta_eeg.png`: complete T0-to-task cycle with mu/beta waveforms.",
            "- `dual_task_subject_isolation_protocol.png`: leakage-safe subject split.",
            "- `dual_task_abstract_key_findings.png`: four-condition performance and safety summary.",
            "- `dual_task_confusion_matrices.png`: task-specific three-class error patterns.",
            "- `dual_task_per_subject_comparison.png`: per-subject gaps for both task families.",
            "- `dual_task_safety_tradeoff.png`: task-specific and pooled intent tradeoffs.",
            "- `pooled_intent_gate_summary.png`: generalized binary intent result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--cache-path", default="artifacts/cross_task/all_task_epochs.npz")
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--output-dir", default="artifacts/dual_task_benchmark")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_plot_style()
    started = time.time()
    cache_path, output_dir = Path(args.cache_path), Path(args.output_dir)
    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir, parse_int_list(args.subjects), args.max_subjects)
    if args.subjects or args.max_subjects:
        partitions = three_way_subject_split(subjects, random_seed=args.seed)
    else:
        partitions = load_partitions(Path(args.splits))
        expected_subjects = set(
            partitions.development + partitions.controller_validation + partitions.final_target
        )
        if set(subjects) != expected_subjects:
            raise ValueError("Saved subject partitions do not match the discovered dataset cohort.")
    epoch_config = EpochConfig(runs=ALL_TASK_RUNS)
    dataset = build_raw_epoch_dataset(
        data_dir, subjects, cache_path, epoch_config, args.rebuild_cache
    )
    discovered_subjects = set(np.unique(dataset.subjects).astype(int).tolist())
    partitions, excluded_subjects = restrict_partitions_to_cache(partitions, discovered_subjects)
    if excluded_subjects:
        print(
            "[!] Excluding subjects absent from the existing cache: "
            + ", ".join(f"S{subject:03d}" for subject in excluded_subjects)
        )
    masks = task_masks(dataset, partitions)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== Expanded Execution vs. Motor-Imagery Benchmark ===")
    print("[+] Shared preprocessing: 4-35 Hz, CAR, artifact QC, pre-cue baseline centering")
    print("[+] Shared decoder: filter-bank CSP plus hierarchical logistic regression")
    for name, mask in masks.items():
        print(f"    {name:24s}: {int(np.sum(mask))} trials")

    models: Dict[str, HierarchicalFBCSP] = {}
    results: Dict[str, Dict[str, object]] = {}
    prediction_rows: List[Dict[str, object]] = []
    for condition in CONDITION_ORDER:
        print(f"[+] Training {str(TASK_SPECS[condition]['title']).lower()} decoder...")
        model, result, rows = train_task(
            condition, dataset, masks, args.max_false_activation, args.seed
        )
        models[condition] = model
        results[condition] = result
        prediction_rows.extend(rows)
        balanced = result["balanced"]
        print(
            f"    BA={balanced['balanced_accuracy'] * 100:.2f}% | "
            f"intent BA={balanced['intent_balanced_accuracy'] * 100:.2f}% | "
            f"direction BA={balanced['direction_balanced_accuracy'] * 100:.2f}% | "
            f"FP={balanced['false_activation_rate'] * 100:.2f}% | "
            f"miss={balanced['missed_intent_rate'] * 100:.2f}%"
        )

    print("[+] Training pooled rest-versus-intent gate across all task families...")
    pooled_model, pooled_intent, pooled_rows = train_pooled_intent(
        dataset, masks, args.max_false_activation, args.seed
    )
    models["pooled_intent"] = pooled_model
    pooled_balanced = pooled_intent["balanced"]
    print(
        f"    BA={pooled_balanced['balanced_accuracy'] * 100:.2f}% | "
        f"AUC={pooled_balanced['roc_auc'] * 100:.2f}% | "
        f"FP={pooled_balanced['false_activation_rate'] * 100:.2f}% | "
        f"miss={pooled_balanced['missed_intent_rate'] * 100:.2f}%"
    )

    report: Dict[str, object] = {
        "protocol": {
            "schema_version": 2,
            "task": "matched unilateral and bilateral movement-versus-imagery benchmark",
            "task_conditions": {
                condition: {
                    "runs": list(spec["runs"]),
                    "final_run": spec["final_run"],
                    "labels": spec["labels"],
                }
                for condition, spec in TASK_SPECS.items()
            },
            "event_semantics": "T0=rest; T1/T2 names are resolved from unilateral versus bilateral run family",
            "pooled_intent_runs": list(ALL_TASK_RUNS),
            "development_subjects": list(partitions.development),
            "controller_validation_subjects": list(partitions.controller_validation),
            "final_target_subjects": list(partitions.final_target),
            "excluded_from_saved_partition_absent_from_cache": excluded_subjects,
            "preprocessing": epoch_config.preprocessing.fingerprint(),
            "epoch_start_seconds": epoch_config.epoch_start_seconds,
            "epoch_duration_seconds": epoch_config.epoch_duration_seconds,
            "channels": dataset.channel_names,
            "model": "Hierarchical FBCSP + regularized logistic regression",
            "threshold_selection": "condition-matched final repeat from controller subjects (runs 11-14)",
            "primary_operating_point": "balanced controller-selected intent threshold",
            "safety_operating_point": f"controller-selected threshold targeting <= {args.max_false_activation:.0%} false activations",
        },
        "dataset": {"trials": int(len(dataset.labels)), "subjects": int(len(discovered_subjects)), "sfreq": dataset.sfreq},
        "results": results,
        "pooled_intent": pooled_intent,
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    joblib.dump(models, output_dir / "dual_task_fbcsp_models.joblib")
    write_predictions(output_dir / "dual_task_final_predictions.csv", prediction_rows)
    write_predictions(output_dir / "pooled_intent_final_predictions.csv", pooled_rows)
    (output_dir / "dual_task_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    plot_expanded_pipeline(output_dir, partitions)
    plot_raw_vs_processed(
        data_dir, int(partitions.development[0]), epoch_config.preprocessing, output_dir
    )
    plot_full_trial_mu_beta(
        data_dir, int(partitions.development[0]), epoch_config.preprocessing, output_dir
    )
    plot_subject_isolation(partitions, output_dir)
    plot_expanded_summary(results, output_dir)
    plot_expanded_confusions(results, output_dir)
    plot_expanded_subject_comparison(results, output_dir)
    plot_expanded_safety_tradeoff(results, pooled_intent, output_dir)
    plot_pooled_intent_summary(pooled_intent, output_dir)
    write_report(output_dir / "DUAL_TASK_BENCHMARK_RESULTS.md", report)
    print(f"[+] Saved metrics, models, predictions, report, and figures: {output_dir.resolve()}")
    print(f"[+] Runtime: {report['runtime_minutes']:.2f} minutes")


if __name__ == "__main__":
    main()
