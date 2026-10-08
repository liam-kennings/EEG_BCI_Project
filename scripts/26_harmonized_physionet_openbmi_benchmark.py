#!/usr/bin/env python3
"""Compare PhysioNet EEGMMIDB and OpenBMI through one locked pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bci_pipeline.dataset import LABEL_NAMES  # noqa: E402
from bci_pipeline.harmonized_motor_imagery import (  # noqa: E402
    HARMONIZED_CHANNELS,
    HARMONIZED_EPOCH_DURATION_SECONDS,
    HARMONIZED_EPOCH_START_SECONDS,
    HARMONIZED_SFREQ,
    HarmonizedMotorImageryDataset,
    build_physionet_harmonized_dataset,
    combine_harmonized_datasets,
    load_openbmi_harmonized_dataset,
)
from bci_pipeline.openbmi_controlled import (  # noqa: E402
    CommandPolicy,
    HierarchicalFeatureDecoder,
    StageProbabilities,
    balanced_calibration_indices,
    build_covariance_cache,
    evaluate_policy,
    locally_adapted_probabilities,
    select_balanced_policy,
    select_safety_policy,
    subject_bootstrap_interval,
    tangent_features,
)
from bci_pipeline.splits import (  # noqa: E402
    SubjectPartitions,
    save_partitions,
    three_way_subject_split,
)


DATASET_NAMES = {
    "physionet": "PhysioNet EEGMMIDB",
    "openbmi": "OpenBMI",
}
COLORS = {
    "physionet": "#184E77",
    "openbmi": "#D66A3A",
    "within": "#5D6770",
    "pooled": "#2D6A4F",
    "adapted": "#E9A23B",
    "false_activation": "#C8553D",
    "missed_intent": "#64748B",
}
OPENBMI_SUBJECT_OFFSET = 1000


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F5F7F8",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--physionet-dir",
        default="files/eegmmidb/1.0.0",
    )
    parser.add_argument(
        "--openbmi-cache",
        default="artifacts/openbmi_controlled/openbmi_controlled_epochs.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/harmonized_physionet_openbmi",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--rebuild-physionet-cache", action="store_true")
    parser.add_argument("--rebuild-covariances", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def dataset_masks(
    dataset: HarmonizedMotorImageryDataset,
    partitions: SubjectPartitions,
) -> dict[str, np.ndarray]:
    return {
        "development": np.isin(dataset.subjects, partitions.development)
        & (dataset.recordings == 0),
        "controller": np.isin(
            dataset.subjects,
            partitions.controller_validation,
        )
        & (dataset.recordings == 1),
        "final": np.isin(dataset.subjects, partitions.final_target)
        & (dataset.recordings == 1),
        "controller_all": np.isin(
            dataset.subjects,
            partitions.controller_validation,
        ),
        "final_all": np.isin(dataset.subjects, partitions.final_target),
    }


def validate_masks(
    dataset: HarmonizedMotorImageryDataset,
    masks: Mapping[str, np.ndarray],
) -> None:
    for name in ("development", "controller", "final"):
        if not np.any(masks[name]):
            raise ValueError(f"{dataset.dataset_name} {name} split is empty.")
        if set(np.unique(dataset.labels[masks[name]])) != {0, 1, 2}:
            raise ValueError(
                f"{dataset.dataset_name} {name} split is missing a class."
            )


def combined_masks(
    physionet: HarmonizedMotorImageryDataset,
    openbmi: HarmonizedMotorImageryDataset,
    physionet_masks: Mapping[str, np.ndarray],
    openbmi_masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count_physionet = len(physionet.labels)
    count_openbmi = len(openbmi.labels)
    result: dict[str, np.ndarray] = {
        "is_physionet": np.concatenate(
            (
                np.ones(count_physionet, dtype=bool),
                np.zeros(count_openbmi, dtype=bool),
            )
        ),
        "is_openbmi": np.concatenate(
            (
                np.zeros(count_physionet, dtype=bool),
                np.ones(count_openbmi, dtype=bool),
            )
        ),
    }
    for name in physionet_masks:
        result[f"physionet_{name}"] = np.concatenate(
            (
                physionet_masks[name],
                np.zeros(count_openbmi, dtype=bool),
            )
        )
        result[f"openbmi_{name}"] = np.concatenate(
            (
                np.zeros(count_physionet, dtype=bool),
                openbmi_masks[name],
            )
        )
    for name in ("development", "controller", "final", "controller_all", "final_all"):
        result[f"pooled_{name}"] = (
            result[f"physionet_{name}"] | result[f"openbmi_{name}"]
        )
    return result


def subset_probabilities(
    probabilities: StageProbabilities,
    mask: np.ndarray,
) -> StageProbabilities:
    return StageProbabilities(
        intent=probabilities.intent[mask],
        direction=probabilities.direction[mask],
    )


def fit_protocol(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[HierarchicalFeatureDecoder, StageProbabilities]:
    model = HierarchicalFeatureDecoder(seed).fit(
        features[train_mask],
        labels[train_mask],
        None if sample_weight is None else sample_weight[train_mask],
    )
    return model, model.predict_stage_probabilities(features)


def policy_results(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    controller_mask: np.ndarray,
    final_masks: Mapping[str, np.ndarray],
    *,
    max_false_activation: float,
) -> dict[str, object]:
    controller = subset_probabilities(probabilities, controller_mask)
    balanced_policy = select_balanced_policy(
        labels[controller_mask],
        controller,
    )
    safety_policy = select_safety_policy(
        labels[controller_mask],
        controller,
        max_false_activation,
    )
    result: dict[str, object] = {
        "balanced_policy": asdict(balanced_policy),
        "safety_policy": asdict(safety_policy),
        "controller_balanced": evaluate_policy(
            labels[controller_mask],
            controller,
            balanced_policy,
        ),
        "controller_safety": evaluate_policy(
            labels[controller_mask],
            controller,
            safety_policy,
        ),
    }
    for name, mask in final_masks.items():
        final = subset_probabilities(probabilities, mask)
        result[f"{name}_balanced"] = evaluate_policy(
            labels[mask],
            final,
            balanced_policy,
        )
        result[f"{name}_safety"] = evaluate_policy(
            labels[mask],
            final,
            safety_policy,
        )
    return result


def calibration_counts(
    labels: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    target_mask: np.ndarray,
    budget: int,
    *,
    seed: int,
) -> dict[str, object]:
    rows = []
    for subject in np.unique(subjects[target_mask]):
        selected = balanced_calibration_indices(
            labels,
            (subjects == subject) & (recordings == 0),
            budget,
            random_seed=seed + int(subject),
        )
        counts = {
            str(label): int(np.sum(labels[selected] == label))
            for label in (0, 1, 2)
        }
        rows.append({"subject": int(subject), "counts": counts})
    per_class = {
        str(label): [
            int(row["counts"][str(label)])
            for row in rows
        ]
        for label in (0, 1, 2)
    }
    return {
        "requested_budget_per_class": budget,
        "minimum_actual_per_class": {
            label: int(np.min(values))
            for label, values in per_class.items()
        },
        "median_actual_per_class": {
            label: float(np.median(values))
            for label, values in per_class.items()
        },
        "subjects": rows,
    }


def calibration_protocol(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    global_model: HierarchicalFeatureDecoder,
    controller_subject_mask: np.ndarray,
    controller_eval_mask: np.ndarray,
    final_subject_mask: np.ndarray,
    final_eval_masks: Mapping[str, np.ndarray],
    *,
    max_false_activation: float,
    seed: int,
) -> tuple[dict[str, object], StageProbabilities]:
    global_probabilities = global_model.predict_stage_probabilities(features)
    configurations: list[dict[str, object]] = []
    controller_cache: dict[tuple[int, float], StageProbabilities] = {}
    for budget in (5, 10, 20):
        for blend in (0.25, 0.5, 0.75, 1.0):
            adapted = locally_adapted_probabilities(
                features,
                labels,
                subjects,
                recordings,
                controller_subject_mask,
                global_model,
                budget_per_class=budget,
                blend_weight=blend,
                random_seed=seed,
            )
            controller_cache[(budget, blend)] = adapted
            controller = subset_probabilities(adapted, controller_eval_mask)
            policy = select_safety_policy(
                labels[controller_eval_mask],
                controller,
                max_false_activation,
            )
            metrics = evaluate_policy(
                labels[controller_eval_mask],
                controller,
                policy,
            )
            configurations.append(
                {
                    "budget_per_class": budget,
                    "blend_weight": blend,
                    "safety_policy": asdict(policy),
                    "controller_safety": metrics,
                    "calibration_counts": calibration_counts(
                        labels,
                        subjects,
                        recordings,
                        controller_subject_mask,
                        budget,
                        seed=seed,
                    ),
                }
            )
    selected = max(
        configurations,
        key=lambda row: (
            float(row["controller_safety"]["intent_balanced_accuracy"]),
            float(row["controller_safety"]["balanced_accuracy"]),
            -float(row["controller_safety"]["missed_intent_rate"]),
        ),
    )
    final_budget_curve: list[dict[str, object]] = []
    final_probability_cache: dict[tuple[int, float], StageProbabilities] = {}
    for curve_budget in (5, 10, 20):
        budget_candidates = [
            row
            for row in configurations
            if int(row["budget_per_class"]) == curve_budget
        ]
        budget_selected = max(
            budget_candidates,
            key=lambda row: (
                float(row["controller_safety"]["intent_balanced_accuracy"]),
                float(row["controller_safety"]["balanced_accuracy"]),
                -float(row["controller_safety"]["missed_intent_rate"]),
            ),
        )
        curve_blend = float(budget_selected["blend_weight"])
        curve_probabilities = locally_adapted_probabilities(
            features,
            labels,
            subjects,
            recordings,
            final_subject_mask,
            global_model,
            budget_per_class=curve_budget,
            blend_weight=curve_blend,
            random_seed=seed,
        )
        final_probability_cache[(curve_budget, curve_blend)] = curve_probabilities
        curve_policy = CommandPolicy(**budget_selected["safety_policy"])
        curve_row: dict[str, object] = {
            "budget_per_class": curve_budget,
            "blend_weight": curve_blend,
            "safety_policy": asdict(curve_policy),
            "controller_safety": budget_selected["controller_safety"],
            "calibration_counts": calibration_counts(
                labels,
                subjects,
                recordings,
                final_subject_mask,
                curve_budget,
                seed=seed,
            ),
        }
        for name, mask in final_eval_masks.items():
            curve_row[f"{name}_safety"] = evaluate_policy(
                labels[mask],
                subset_probabilities(curve_probabilities, mask),
                curve_policy,
            )
        final_budget_curve.append(curve_row)

    budget = int(selected["budget_per_class"])
    blend = float(selected["blend_weight"])
    controller_adapted = controller_cache[(budget, blend)]
    controller = subset_probabilities(controller_adapted, controller_eval_mask)
    balanced_policy = select_balanced_policy(
        labels[controller_eval_mask],
        controller,
    )
    safety_policy = select_safety_policy(
        labels[controller_eval_mask],
        controller,
        max_false_activation,
    )
    final_adapted = final_probability_cache.get(
        (budget, blend)
    )
    if final_adapted is None:
        final_adapted = locally_adapted_probabilities(
            features,
            labels,
            subjects,
            recordings,
            final_subject_mask,
            global_model,
            budget_per_class=budget,
            blend_weight=blend,
            random_seed=seed,
        )
    report: dict[str, object] = {
        "selected_on_controller": selected,
        "all_controller_configurations": configurations,
        "final_budget_curve": final_budget_curve,
        "balanced_policy": asdict(balanced_policy),
        "safety_policy": asdict(safety_policy),
        "final_calibration_counts": calibration_counts(
            labels,
            subjects,
            recordings,
            final_subject_mask,
            budget,
            seed=seed,
        ),
    }
    for name, mask in final_eval_masks.items():
        final = subset_probabilities(final_adapted, mask)
        report[f"{name}_balanced"] = evaluate_policy(
            labels[mask],
            final,
            balanced_policy,
        )
        report[f"{name}_safety"] = evaluate_policy(
            labels[mask],
            final,
            safety_policy,
        )
    return report, final_adapted


def per_subject_metrics(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    subjects: np.ndarray,
    mask: np.ndarray,
    policy: CommandPolicy,
    dataset_name: str,
) -> list[dict[str, object]]:
    rows = []
    for subject in np.unique(subjects[mask]):
        selected = mask & (subjects == subject)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="y_pred contains classes not in y_true",
                category=UserWarning,
            )
            metrics = evaluate_policy(
                labels[selected],
                subset_probabilities(probabilities, selected),
                policy,
            )
        rows.append(
            {
                "dataset": dataset_name,
                "subject": int(
                    subject - OPENBMI_SUBJECT_OFFSET
                    if dataset_name == "openbmi"
                    else subject
                ),
                **metrics,
            }
        )
    return rows


def plot_intent_comparison(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    datasets = ("physionet", "openbmi")
    strategies = (
        ("within_global", "Within-dataset global", COLORS["within"]),
        ("within_adapted", "Within + calibration", "#147D92"),
        ("pooled_global", "Pooled global", COLORS["pooled"]),
        ("pooled_adapted", "Pooled + calibration", COLORS["adapted"]),
    )
    positions = np.arange(2)
    width = 0.18
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, (key, label, color) in enumerate(strategies):
        values = [
            100
            * float(
                report["summary"][dataset][key][
                    "intent_balanced_accuracy"
                ]
            )
            for dataset in datasets
        ]
        bars = axis.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            color=color,
            label=label,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    axis.axhline(50, color="#475569", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, [DATASET_NAMES[name] for name in datasets])
    axis.set_ylabel("Intent Balanced Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Matched Pipeline: Intent-versus-Rest Detection")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "01_harmonized_intent_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_safety_comparison(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for axis, dataset in zip(axes, ("physionet", "openbmi")):
        labels = (
            ("within_global", "Within global", "o", COLORS["within"]),
            ("within_adapted", "Within + calibration", "D", "#147D92"),
            ("pooled_global", "Pooled global", "s", COLORS["pooled"]),
            ("pooled_adapted", "Pooled + calibration", "^", COLORS["adapted"]),
        )
        for key, display, marker, color in labels:
            metrics = report["summary"][dataset][key]
            x = 100 * float(metrics["false_activation_rate"])
            y = 100 * (1.0 - float(metrics["missed_intent_rate"]))
            axis.scatter(
                x,
                y,
                s=100,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=display,
            )
            axis.annotate(
                f"{x:.1f}%, {y:.1f}%",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(DATASET_NAMES[dataset])
        axis.set_xlabel("False Activation on Rest (%)")
        axis.set_xlim(left=0)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Intent Sensitivity (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Safety Trade-off on Untouched Subjects", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "02_harmonized_safety_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_direction_comparison(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    datasets = ("physionet", "openbmi")
    strategies = (
        ("within_global", "Within-dataset global", COLORS["within"]),
        ("within_adapted", "Within + calibration", "#147D92"),
        ("pooled_global", "Pooled global", COLORS["pooled"]),
        ("pooled_adapted", "Pooled + calibration", COLORS["adapted"]),
    )
    positions = np.arange(2)
    width = 0.18
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for offset, (key, display, color) in enumerate(strategies):
        values = [
            100
            * float(
                report["summary"][dataset][key][
                    "direction_balanced_accuracy"
                ]
            )
            for dataset in datasets
        ]
        bars = axis.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            color=color,
            label=display,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
    axis.axhline(50, color="#475569", linestyle="--", linewidth=1, label="Chance")
    axis.set_xticks(positions, [DATASET_NAMES[name] for name in datasets])
    axis.set_ylabel("Left-vs-Right Balanced Accuracy (%)")
    axis.set_ylim(35, 80)
    axis.set_title("Matched Pipeline: Direction Decoding")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "03_harmonized_direction_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_cross_transfer(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    fields = (
        ("intent_balanced_accuracy", "Intent BA"),
        ("direction_balanced_accuracy", "Direction BA"),
        ("balanced_accuracy", "3-Class BA"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for axis, (field, title) in zip(axes, fields):
        matrix = np.asarray(
            [
                [
                    report["within"]["physionet"]["physionet_final_safety"][field],
                    report["cross_dataset"]["physionet_to_openbmi"][
                        "openbmi_final_safety"
                    ][field],
                ],
                [
                    report["cross_dataset"]["openbmi_to_physionet"][
                        "physionet_final_safety"
                    ][field],
                    report["within"]["openbmi"]["openbmi_final_safety"][field],
                ],
            ],
            dtype=float,
        )
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0.3, vmax=0.9)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{100 * matrix[row, column]:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > 0.68 else "#102A43",
                    fontweight="bold",
                )
        axis.set_xticks(range(2), ("PhysioNet", "OpenBMI"))
        axis.set_yticks(range(2), ("PhysioNet", "OpenBMI"))
        axis.set_xlabel("Final Evaluation Dataset")
        axis.set_ylabel("Training Dataset")
        axis.set_title(title)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04, label="Balanced Accuracy")
    fig.suptitle("Label-Free Aligned Cross-Dataset Transfer", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.18, top=0.82, wspace=0.4)
    fig.savefig(output_dir / "04_cross_dataset_transfer.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, dataset in zip(axes, ("physionet", "openbmi")):
        best = report["calibration"][dataset]["final_budget_curve"]
        global_metrics = report["summary"][dataset]["within_global"]
        budgets = [0, 5, 10, 20]
        intent = [100 * global_metrics["intent_balanced_accuracy"]] + [
            100
            * row[f"{dataset}_final_safety"]["intent_balanced_accuracy"]
            for row in best
        ]
        false_activation = [100 * global_metrics["false_activation_rate"]] + [
            100 * row[f"{dataset}_final_safety"]["false_activation_rate"]
            for row in best
        ]
        missed = [100 * global_metrics["missed_intent_rate"]] + [
            100 * row[f"{dataset}_final_safety"]["missed_intent_rate"]
            for row in best
        ]
        axis.plot(
            budgets,
            intent,
            marker="o",
            color=COLORS["adapted"],
            linewidth=2,
            label="Intent BA",
        )
        axis.plot(
            budgets,
            false_activation,
            marker="s",
            color=COLORS["false_activation"],
            linewidth=2,
            label="False activation",
        )
        axis.plot(
            budgets,
            missed,
            marker="^",
            color=COLORS["missed_intent"],
            linewidth=2,
            label="Missed intent",
        )
        axis.set_title(DATASET_NAMES[dataset])
        axis.set_xlabel("Requested Calibration Trials per Class")
        axis.set_xticks(budgets)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Rate (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("Untouched-Subject Calibration Curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "05_matched_calibration_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusions(
    output_dir: Path,
    report: Mapping[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, dataset in zip(axes, ("physionet", "openbmi")):
        matrix = np.asarray(
            report["summary"][dataset]["pooled_adapted"]["confusion_matrix"],
            dtype=float,
        )
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}\n{100 * normalized[row, column]:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if normalized[row, column] > 0.55 else "#102A43",
                    fontsize=9,
                )
        axis.set_xticks(range(3), ("Rest", "Left", "Right"))
        axis.set_yticks(range(3), ("Rest", "Left", "Right"))
        axis.set_xlabel("Predicted Command")
        axis.set_ylabel("True State")
        axis.set_title(DATASET_NAMES[dataset])
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04, label="Row Fraction")
    fig.suptitle("Pooled + Calibrated Model on Untouched Subjects", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.14, top=0.82, wspace=0.3)
    fig.savefig(output_dir / "06_pooled_calibrated_confusions.png", bbox_inches="tight")
    plt.close(fig)


def plot_subject_variability(
    output_dir: Path,
    rows: list[Mapping[str, object]],
) -> None:
    rows = sorted(rows, key=lambda row: (str(row["dataset"]), int(row["subject"])))
    positions = np.arange(len(rows))
    labels = [
        f"{'P' if row['dataset'] == 'physionet' else 'O'}{int(row['subject']):03d}"
        for row in rows
    ]
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.bar(
        positions - 0.18,
        [100 * float(row["false_activation_rate"]) for row in rows],
        0.36,
        color=COLORS["false_activation"],
        label="False activation",
    )
    axis.bar(
        positions + 0.18,
        [100 * float(row["missed_intent_rate"]) for row in rows],
        0.36,
        color=COLORS["missed_intent"],
        label="Missed intent",
    )
    boundary = sum(row["dataset"] == "openbmi" for row in rows)
    axis.axvline(boundary - 0.5, color="#94A3B8", linewidth=1)
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Pooled + Calibrated Safety Variability")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "07_final_subject_variability.png", bbox_inches="tight")
    plt.close(fig)


def write_predictions(
    path: Path,
    dataset: HarmonizedMotorImageryDataset,
    dataset_membership: np.ndarray,
    final_mask: np.ndarray,
    probabilities: StageProbabilities,
    policy: CommandPolicy,
) -> None:
    rows = []
    direction_predictions = np.argmax(probabilities.direction, axis=1) + 1
    confidence = np.max(probabilities.direction, axis=1)
    for index in np.flatnonzero(final_mask):
        dataset_name = "physionet" if dataset_membership[index] else "openbmi"
        subject = int(dataset.subjects[index])
        if dataset_name == "openbmi":
            subject -= OPENBMI_SUBJECT_OFFSET
        accepted = (
            probabilities.intent[index] >= policy.intent_threshold
            and confidence[index] >= 0.5 + policy.direction_margin / 2.0
        )
        prediction = int(direction_predictions[index]) if accepted else 0
        rows.append(
            {
                "dataset": dataset_name,
                "subject": subject,
                "block": int(dataset.blocks[index]),
                "trial_index": int(dataset.trial_indices[index]),
                "true_label": LABEL_NAMES[int(dataset.labels[index])],
                "intent_probability": float(probabilities.intent[index]),
                "left_probability": float(probabilities.direction[index, 0]),
                "right_probability": float(probabilities.direction[index, 1]),
                "prediction": LABEL_NAMES[prediction],
            }
        )
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: Mapping[str, object]) -> None:
    p = report["summary"]["physionet"]
    o = report["summary"]["openbmi"]
    p_counts = report["calibration"]["physionet"]["final_calibration_counts"]
    lines = [
        "# Harmonized PhysioNet-OpenBMI Motor-Imagery Benchmark",
        "",
        "## Matched Protocol",
        "",
        (
            f"- Common montage: **{len(HARMONIZED_CHANNELS)} channels** "
            f"(`{', '.join(HARMONIZED_CHANNELS)}`)."
        ),
        (
            f"- Common signal processing: **4-38 Hz CAR**, "
            f"**{HARMONIZED_SFREQ:.0f} Hz**, and "
            f"**{HARMONIZED_EPOCH_START_SECONDS:.1f}-"
            f"{HARMONIZED_EPOCH_START_SECONDS + HARMONIZED_EPOCH_DURATION_SECONDS:.1f} s**."
        ),
        "- Common representation: five-band covariance, calibration-only log-Euclidean alignment, and a hierarchical intent/direction decoder.",
        "- Pooled training assigns equal total weight to each dataset rather than allowing the larger OpenBMI trial count to dominate.",
        "- Common evaluation: development subjects, controller-validation subjects, untouched final subjects, and a controller-selected safety policy.",
        "",
        "## Untouched-Subject Results",
        "",
        "| Dataset | Strategy | 3-class BA | Intent BA | Direction BA | False activation | Missed intent |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, rows in (("PhysioNet EEGMMIDB", p), ("OpenBMI", o)):
        for key, display in (
            ("within_global", "Within global"),
            ("within_adapted", "Within + calibration"),
            ("pooled_global", "Pooled global"),
            ("pooled_adapted", "Pooled + calibration"),
        ):
            metrics = rows[key]
            lines.append(
                f"| {dataset_name} | {display} "
                f"| {100 * metrics['balanced_accuracy']:.1f}% "
                f"| {100 * metrics['intent_balanced_accuracy']:.1f}% "
                f"| {100 * metrics['direction_balanced_accuracy']:.1f}% "
                f"| {100 * metrics['false_activation_rate']:.1f}% "
                f"| {100 * metrics['missed_intent_rate']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"The matched within-dataset pipeline produced "
                f"**{100 * p['within_global']['intent_balanced_accuracy']:.1f}%** "
                f"PhysioNet and **{100 * o['within_global']['intent_balanced_accuracy']:.1f}%** "
                "OpenBMI intent balanced accuracy. Because channels, windows, filters, "
                "features, alignment, and decoder are now matched, the remaining difference "
                "is more plausibly associated with protocol, rest definition, trial count, "
                "and cohort characteristics."
            ),
            (
                f"Within-dataset calibration reached "
                f"**{100 * p['within_adapted']['intent_balanced_accuracy']:.1f}%** "
                f"intent BA on PhysioNet and "
                f"**{100 * o['within_adapted']['intent_balanced_accuracy']:.1f}%** "
                "on OpenBMI. The pooled calibrated model reached "
                f"**{100 * p['pooled_adapted']['intent_balanced_accuracy']:.1f}%** "
                f"and **{100 * o['pooled_adapted']['intent_balanced_accuracy']:.1f}%**, "
                "respectively; pooling therefore helped OpenBMI but did not improve "
                "PhysioNet intent gating over dataset-specific calibration."
            ),
            (
                f"Direction remained **{100 * p['pooled_adapted']['direction_balanced_accuracy']:.1f}%** "
                f"on PhysioNet and **{100 * o['pooled_adapted']['direction_balanced_accuracy']:.1f}%** "
                "on OpenBMI, so adding datasets does not by itself establish a robust "
                "cross-subject left/right command."
            ),
            (
                "PhysioNet cannot supply a literal 20 examples for every class without "
                "using the held-out run. At the selected budget, the minimum final-subject "
                f"counts were rest={p_counts['minimum_actual_per_class']['0']}, "
                f"left={p_counts['minimum_actual_per_class']['1']}, and "
                f"right={p_counts['minimum_actual_per_class']['2']}; the held-out run "
                "was preserved."
            ),
            "",
            "## Limits",
            "",
            "- PhysioNet rest comes from T0 annotation periods; OpenBMI rest is a paired post-imagery window. This semantic difference is intentionally retained and reported.",
            "- False activation is trial-level on cue-locked rest windows, not continuous false activations per minute.",
            "- OpenBMI remains an interim 29-subject cohort because only those subjects have complete local BDF recordings.",
            "- The reused OpenBMI cache applied artifact rejection across its 20 motor and 9 visual-control channels, whereas PhysioNet rejection used the common 9-channel montage.",
            "- Label-free transfer uses unlabeled target calibration recordings for alignment, but no target labels for decoder or threshold fitting; it is not strict zero-shot transfer.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/26_harmonized_physionet_openbmi_benchmark.py",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="ascii")


def main() -> None:
    args = parse_args()
    configure_plots()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Loading harmonized PhysioNet EEGMMIDB...")
    physionet = build_physionet_harmonized_dataset(
        args.physionet_dir,
        output_dir / "physionet_harmonized_epochs.npz",
        rebuild_cache=args.rebuild_physionet_cache,
    )
    print("[+] Loading harmonized OpenBMI...")
    openbmi = load_openbmi_harmonized_dataset(Path(args.openbmi_cache))
    combined = combine_harmonized_datasets(
        physionet,
        openbmi,
        openbmi_subject_offset=OPENBMI_SUBJECT_OFFSET,
    )
    physionet_partitions = three_way_subject_split(
        np.unique(physionet.subjects),
        random_seed=args.seed,
    )
    openbmi_partitions = three_way_subject_split(
        np.unique(openbmi.subjects),
        random_seed=args.seed,
    )
    save_partitions(
        physionet_partitions,
        output_dir / "physionet_subject_partitions.json",
    )
    save_partitions(
        openbmi_partitions,
        output_dir / "openbmi_subject_partitions.json",
    )
    physionet_masks = dataset_masks(physionet, physionet_partitions)
    openbmi_masks = dataset_masks(openbmi, openbmi_partitions)
    validate_masks(physionet, physionet_masks)
    validate_masks(openbmi, openbmi_masks)
    masks = combined_masks(
        physionet,
        openbmi,
        physionet_masks,
        openbmi_masks,
    )
    print(
        f"[+] PhysioNet: {len(np.unique(physionet.subjects))} subjects, "
        f"{len(physionet.labels):,} trials"
    )
    print(
        f"[+] OpenBMI: {len(np.unique(openbmi.subjects))} subjects, "
        f"{len(openbmi.labels):,} paired windows"
    )

    physionet_covariances = build_covariance_cache(
        physionet.covariance_interface(),
        output_dir / "physionet_covariances.npz",
        rebuild_cache=args.rebuild_covariances,
    )
    openbmi_covariances = build_covariance_cache(
        openbmi.covariance_interface(),
        output_dir / "openbmi_covariances.npz",
        rebuild_cache=args.rebuild_covariances,
    )
    covariances = np.concatenate(
        (
            physionet_covariances.covariances,
            openbmi_covariances.covariances,
        )
    )

    feature_sets: dict[str, np.ndarray] = {}
    feature_metadata: dict[str, object] = {}
    for protocol, train_mask in (
        ("physionet", masks["physionet_development"]),
        ("openbmi", masks["openbmi_development"]),
        ("pooled", masks["pooled_development"]),
    ):
        print(f"[+] Computing {protocol} calibration-aligned tangent space...")
        features, metadata = tangent_features(
            covariances,
            train_mask,
            combined.subjects,
            combined.sessions,
            combined.recordings,
            alignment="log_euclidean",
        )
        feature_sets[protocol] = features
        feature_metadata[protocol] = metadata

    models: dict[str, HierarchicalFeatureDecoder] = {}
    probabilities: dict[str, StageProbabilities] = {}
    for protocol in ("physionet", "openbmi", "pooled"):
        sample_weight = None
        if protocol == "pooled":
            sample_weight = np.ones(len(combined.labels), dtype=np.float64)
            physionet_train = masks["physionet_development"]
            openbmi_train = masks["openbmi_development"]
            total = int(np.sum(physionet_train) + np.sum(openbmi_train))
            sample_weight[physionet_train] = total / (
                2.0 * np.sum(physionet_train)
            )
            sample_weight[openbmi_train] = total / (
                2.0 * np.sum(openbmi_train)
            )
        models[protocol], probabilities[protocol] = fit_protocol(
            feature_sets[protocol],
            combined.labels,
            masks[f"{protocol}_development"],
            seed=args.seed,
            sample_weight=sample_weight,
        )

    within = {
        "physionet": policy_results(
            combined.labels,
            probabilities["physionet"],
            masks["physionet_controller"],
            {"physionet_final": masks["physionet_final"]},
            max_false_activation=args.max_false_activation,
        ),
        "openbmi": policy_results(
            combined.labels,
            probabilities["openbmi"],
            masks["openbmi_controller"],
            {"openbmi_final": masks["openbmi_final"]},
            max_false_activation=args.max_false_activation,
        ),
    }
    pooled = policy_results(
        combined.labels,
        probabilities["pooled"],
        masks["pooled_controller"],
        {
            "physionet_final": masks["physionet_final"],
            "openbmi_final": masks["openbmi_final"],
            "pooled_final": masks["pooled_final"],
        },
        max_false_activation=args.max_false_activation,
    )
    cross_dataset = {
        "physionet_to_openbmi": policy_results(
            combined.labels,
            probabilities["physionet"],
            masks["physionet_controller"],
            {"openbmi_final": masks["openbmi_final"]},
            max_false_activation=args.max_false_activation,
        ),
        "openbmi_to_physionet": policy_results(
            combined.labels,
            probabilities["openbmi"],
            masks["openbmi_controller"],
            {"physionet_final": masks["physionet_final"]},
            max_false_activation=args.max_false_activation,
        ),
    }

    calibration: dict[str, object] = {}
    adapted_probabilities: dict[str, StageProbabilities] = {}
    calibration["physionet"], adapted_probabilities["physionet"] = (
        calibration_protocol(
            feature_sets["physionet"],
            combined.labels,
            combined.subjects,
            combined.recordings,
            models["physionet"],
            masks["physionet_controller_all"],
            masks["physionet_controller"],
            masks["physionet_final_all"],
            {"physionet_final": masks["physionet_final"]},
            max_false_activation=args.max_false_activation,
            seed=args.seed,
        )
    )
    calibration["openbmi"], adapted_probabilities["openbmi"] = (
        calibration_protocol(
            feature_sets["openbmi"],
            combined.labels,
            combined.subjects,
            combined.recordings,
            models["openbmi"],
            masks["openbmi_controller_all"],
            masks["openbmi_controller"],
            masks["openbmi_final_all"],
            {"openbmi_final": masks["openbmi_final"]},
            max_false_activation=args.max_false_activation,
            seed=args.seed,
        )
    )
    calibration["pooled"], adapted_probabilities["pooled"] = calibration_protocol(
        feature_sets["pooled"],
        combined.labels,
        combined.subjects,
        combined.recordings,
        models["pooled"],
        masks["pooled_controller_all"],
        masks["pooled_controller"],
        masks["pooled_final_all"],
        {
            "physionet_final": masks["physionet_final"],
            "openbmi_final": masks["openbmi_final"],
            "pooled_final": masks["pooled_final"],
        },
        max_false_activation=args.max_false_activation,
        seed=args.seed,
    )

    pooled_policy = CommandPolicy(**calibration["pooled"]["safety_policy"])
    final_subject_rows = (
        per_subject_metrics(
            combined.labels,
            adapted_probabilities["pooled"],
            combined.subjects,
            masks["physionet_final"],
            pooled_policy,
            "physionet",
        )
        + per_subject_metrics(
            combined.labels,
            adapted_probabilities["pooled"],
            combined.subjects,
            masks["openbmi_final"],
            pooled_policy,
            "openbmi",
        )
    )
    for dataset_name, mask in (
        ("physionet", masks["physionet_final"]),
        ("openbmi", masks["openbmi_final"]),
    ):
        calibration["pooled"][f"{dataset_name}_final_safety"][
            "subject_bootstrap_95_ci"
        ] = subject_bootstrap_interval(
            combined.labels[mask],
            subset_probabilities(adapted_probabilities["pooled"], mask),
            combined.subjects[mask],
            pooled_policy,
            repetitions=args.bootstrap_repetitions,
            random_seed=args.seed,
        )

    summary = {
        "physionet": {
            "within_global": within["physionet"]["physionet_final_safety"],
            "within_adapted": calibration["physionet"][
                "physionet_final_safety"
            ],
            "pooled_global": pooled["physionet_final_safety"],
            "pooled_adapted": calibration["pooled"][
                "physionet_final_safety"
            ],
        },
        "openbmi": {
            "within_global": within["openbmi"]["openbmi_final_safety"],
            "within_adapted": calibration["openbmi"]["openbmi_final_safety"],
            "pooled_global": pooled["openbmi_final_safety"],
            "pooled_adapted": calibration["pooled"]["openbmi_final_safety"],
        },
    }
    elapsed = (time.time() - started) / 60.0
    report = {
        "schema_version": 1,
        "experiment": "harmonized_physionet_openbmi_motor_imagery",
        "harmonization": {
            "channels": list(HARMONIZED_CHANNELS),
            "sampling_hz": HARMONIZED_SFREQ,
            "bandpass_hz": [4.0, 38.0],
            "reference": "common_average",
            "epoch_seconds": [
                HARMONIZED_EPOCH_START_SECONDS,
                HARMONIZED_EPOCH_START_SECONDS
                + HARMONIZED_EPOCH_DURATION_SECONDS,
            ],
            "filter_bank": "shared five-band covariance",
            "alignment": "calibration-only log-Euclidean",
            "decoder": "hierarchical regularized logistic regression",
            "pooled_dataset_weighting": "equal total training weight per dataset",
            "max_controller_false_activation": args.max_false_activation,
        },
        "datasets": {
            "physionet": {
                "subjects": int(len(np.unique(physionet.subjects))),
                "rows": int(len(physionet.labels)),
                "class_counts": {
                    str(label): int(np.sum(physionet.labels == label))
                    for label in (0, 1, 2)
                },
                "rest_semantics": "T0 annotation epochs",
                "calibration_blocks": [4, 8],
                "evaluation_block": 12,
                "fingerprint": physionet.fingerprint,
            },
            "openbmi": {
                "subjects": int(len(np.unique(openbmi.subjects))),
                "rows": int(len(openbmi.labels)),
                "class_counts": {
                    str(label): int(np.sum(openbmi.labels == label))
                    for label in (0, 1, 2)
                },
                "rest_semantics": "paired 5.0-7.5 s post-imagery windows",
                "calibration_blocks": "training recording in each session",
                "evaluation_block": "test recording in each session",
                "fingerprint": openbmi.fingerprint,
            },
        },
        "partitions": {
            "physionet": asdict(physionet_partitions),
            "openbmi": asdict(openbmi_partitions),
        },
        "feature_metadata": feature_metadata,
        "within": within,
        "pooled": pooled,
        "cross_dataset": cross_dataset,
        "calibration": calibration,
        "summary": summary,
        "pooled_calibrated_per_subject": final_subject_rows,
        "runtime_minutes": elapsed,
    }
    metrics_path = output_dir / "harmonized_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    joblib.dump(
        {
            "models": models,
            "partitions": {
                "physionet": physionet_partitions,
                "openbmi": openbmi_partitions,
            },
            "pooled_calibrated_policy": pooled_policy,
            "fingerprint": combined.fingerprint,
        },
        output_dir / "harmonized_models.joblib",
    )
    write_predictions(
        output_dir / "pooled_calibrated_final_predictions.csv",
        combined,
        masks["is_physionet"],
        masks["pooled_final"],
        adapted_probabilities["pooled"],
        pooled_policy,
    )
    plot_intent_comparison(output_dir, report)
    plot_safety_comparison(output_dir, report)
    plot_direction_comparison(output_dir, report)
    plot_cross_transfer(output_dir, report)
    plot_calibration_curves(output_dir, report)
    plot_confusions(output_dir, report)
    plot_subject_variability(output_dir, final_subject_rows)
    write_markdown(output_dir / "HARMONIZED_BENCHMARK_REPORT.md", report)

    print("\n=== Harmonized Untouched-Subject Results ===")
    for dataset_name in ("physionet", "openbmi"):
        print(f"{DATASET_NAMES[dataset_name]}:")
        for strategy in (
            "within_global",
            "within_adapted",
            "pooled_global",
            "pooled_adapted",
        ):
            metrics = summary[dataset_name][strategy]
            print(
                f"    {strategy:15s} | intent BA="
                f"{100 * metrics['intent_balanced_accuracy']:5.1f}% | "
                f"direction BA={100 * metrics['direction_balanced_accuracy']:5.1f}% | "
                f"FA={100 * metrics['false_activation_rate']:5.1f}% | "
                f"missed={100 * metrics['missed_intent_rate']:5.1f}%"
            )
    print(f"[+] Saved report and seven figures to {output_dir}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
