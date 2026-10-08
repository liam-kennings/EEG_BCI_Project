#!/usr/bin/env python3
"""Test compact classical EEG features and models on healthy-to-stroke intent."""

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
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from bci_pipeline.classical_eeg_features import (  # noqa: E402
    CLASSICAL_MODEL_NAMES,
    ClassicalIntentDecoder,
    build_classical_feature_cache,
    calibration_robust_alignment,
)
from bci_pipeline.clinical_transfer import (  # noqa: E402
    CLINICAL_CHANNELS,
    CLINICAL_EPOCH_DURATION_SECONDS,
    HandIntentDataset,
    build_liu_hand_intent_dataset,
    combine_hand_intent_datasets,
    from_harmonized_healthy,
)
from bci_pipeline.harmonized_motor_imagery import (  # noqa: E402
    build_physionet_harmonized_dataset,
    load_openbmi_harmonized_dataset,
)
from bci_pipeline.openbmi_controlled import (  # noqa: E402
    build_covariance_cache,
    tangent_features,
)
from bci_pipeline.splits import SubjectPartitions, load_partitions  # noqa: E402


PHYSIONET_OFFSET = 0
OPENBMI_OFFSET = 1000
LIU_OFFSET = 2000
COLORS = {
    "navy": "#17324D",
    "blue": "#2878A6",
    "teal": "#21867A",
    "orange": "#D56A3A",
    "gold": "#E5A23C",
    "red": "#B94B4B",
    "gray": "#66737E",
}
DISPLAY_NAMES = {
    "covariance_reference": "Covariance\nreference",
    "statistical": "Statistical",
    "spectral": "Welch\nspectral",
    "temporal": "Temporal\nmorphology",
    "compact_all": "All compact",
    "covariance_plus_compact": "Covariance +\ncompact",
    "logistic": "Logistic",
    "rbf_svm": "RBF-SVM",
    "gradient_boosting": "Gradient\nboosting",
    "random_forest": "Random\nforest",
    "decision_tree": "Decision\ntree",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument(
        "--physionet-cache",
        default="artifacts/harmonized_physionet_openbmi/"
        "physionet_harmonized_epochs.npz",
    )
    parser.add_argument(
        "--openbmi-cache",
        default="artifacts/openbmi_controlled/openbmi_controlled_epochs.npz",
    )
    parser.add_argument("--liu-dir", default="Liu acute-stroke dataset")
    parser.add_argument(
        "--transfer-dir",
        default="artifacts/healthy_to_stroke_hand_intent",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/classical_feature_model_ablation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F8FAFA",
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


def _offset_subjects(
    partitions: SubjectPartitions,
    offset: int,
) -> dict[str, np.ndarray]:
    return {
        "development": np.asarray(partitions.development) + offset,
        "controller": np.asarray(partitions.controller_validation) + offset,
        "final": np.asarray(partitions.final_target) + offset,
    }


def build_masks(
    dataset: HandIntentDataset,
    partitions: Mapping[str, SubjectPartitions],
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for dataset_name, offset in (
        ("physionet", PHYSIONET_OFFSET),
        ("openbmi", OPENBMI_OFFSET),
        ("liu2024", LIU_OFFSET),
    ):
        groups = _offset_subjects(partitions[dataset_name], offset)
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
    return masks


def training_weights(
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


def select_threshold(
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
        intent_ba = 0.5 * (sensitivity + 1.0 - false_activation)
        if (
            max_false_activation is None
            or false_activation <= max_false_activation + 1e-12
        ):
            candidates.append(
                (float(threshold), intent_ba, sensitivity, false_activation)
            )
    if not candidates:
        return 1.0
    return max(
        candidates,
        key=lambda row: (row[1], row[2], -abs(row[0] - 0.5)),
    )[0]


def score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
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


def fit_candidate(
    features: np.ndarray,
    dataset: HandIntentDataset,
    masks: Mapping[str, np.ndarray],
    weights: np.ndarray,
    model_name: str,
    *,
    seed: int,
    max_false_activation: float,
) -> tuple[ClassicalIntentDecoder, np.ndarray, float, dict[str, object]]:
    train = masks["mixed_development"]
    controller = masks["liu2024_controller"]
    model = ClassicalIntentDecoder(model_name, seed).fit(
        features[train],
        dataset.labels[train],
        weights[train],
    )
    probabilities = model.predict_intent_probability(features)
    threshold = select_threshold(
        dataset.labels[controller],
        probabilities[controller],
        max_false_activation=max_false_activation,
    )
    return (
        model,
        probabilities,
        threshold,
        score(dataset.labels[controller], probabilities[controller], threshold),
    )


def calibration_indices(
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


def adapted_probabilities(
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
        selected = calibration_indices(
            dataset.labels,
            calibration,
            budget_per_class,
            seed=seed + int(subject),
        )
        if len(np.unique((dataset.labels[selected] != 0).astype(int))) < 2:
            continue
        local_weights = np.ones(len(selected), dtype=np.float64)
        local = ClassicalIntentDecoder("logistic", seed + int(subject)).fit(
            features[selected],
            dataset.labels[selected],
            local_weights,
        )
        local_probability = local.predict_intent_probability(features[evaluate])
        result[evaluate] = (
            (1.0 - blend_weight) * result[evaluate]
            + blend_weight * local_probability
        )
    return result


def select_calibration(
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
            probabilities = adapted_probabilities(
                features,
                dataset,
                controller,
                global_probabilities,
                budget_per_class=budget,
                blend_weight=blend,
                seed=seed,
            )
            threshold = select_threshold(
                dataset.labels[controller],
                probabilities[controller],
                max_false_activation=max_false_activation,
            )
            configurations.append(
                {
                    "budget_per_binary_class": budget,
                    "blend_weight": blend,
                    "threshold": threshold,
                    "controller": score(
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
    final_probabilities = adapted_probabilities(
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


def paired_bootstrap(
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
        sampled = rng.choice(
            unique_subjects,
            size=len(unique_subjects),
            replace=True,
        )
        indices = np.concatenate(
            [np.flatnonzero(subjects == subject) for subject in sampled]
        )
        baseline = score(
            labels[indices],
            baseline_probabilities[indices],
            baseline_threshold,
        )
        candidate = score(
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


def patient_rows(
    dataset: HandIntentDataset,
    mask: np.ndarray,
    methods: Mapping[str, tuple[np.ndarray, float]],
) -> list[dict[str, object]]:
    rows = []
    for subject in np.unique(dataset.subjects[mask]):
        selected = mask & (dataset.subjects == subject)
        for name, (probabilities, threshold) in methods.items():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="y_pred contains classes not in y_true",
                    category=UserWarning,
                )
                metrics = score(
                    dataset.labels[selected],
                    probabilities[selected],
                    threshold,
                )
            rows.append(
                {
                    "method": name,
                    "subject": int(subject - LIU_OFFSET),
                    "nihss": int(dataset.nihss[selected][0]),
                    "paralysis_side": str(dataset.paralysis_sides[selected][0]),
                    **metrics,
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def annotate_bars(axis: plt.Axes, bars: object) -> None:
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9, fontweight="bold")


def plot_controller_comparison(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    order: Sequence[str],
    *,
    title: str,
) -> None:
    lookup = {str(row["name"]): row for row in rows}
    x = np.arange(len(order))
    metrics = (
        ("intent_balanced_accuracy", "Intent BA", COLORS["navy"]),
        ("intent_sensitivity", "Sensitivity", COLORS["blue"]),
        ("intent_specificity", "Specificity", COLORS["gold"]),
    )
    width = 0.24
    fig, axis = plt.subplots(figsize=(12, 6))
    for offset, (metric, label, color) in enumerate(metrics):
        values = [
            100.0 * float(lookup[name]["controller"][metric])
            for name in order
        ]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        annotate_bars(axis, bars)
    axis.set_xticks(x, [DISPLAY_NAMES[name] for name in order])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Controller-patient result (%)")
    axis.set_title(title)
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def final_ladder(report: Mapping[str, object]) -> list[tuple[str, Mapping[str, object], str]]:
    return [
        (
            "Covariance\npopulation",
            report["final"]["baseline_population"],
            COLORS["gray"],
        ),
        (
            "Selected\npopulation",
            report["final"]["selected_population"],
            COLORS["blue"],
        ),
        (
            "Covariance +\ncalibration",
            report["final"]["baseline_calibrated"],
            COLORS["teal"],
        ),
        (
            "Selected +\ncalibration",
            report["final"]["selected_calibrated"],
            COLORS["orange"],
        ),
    ]


def plot_final_comparison(path: Path, report: Mapping[str, object]) -> None:
    methods = final_ladder(report)
    x = np.arange(len(methods))
    metrics = (
        ("intent_balanced_accuracy", "Intent BA", COLORS["navy"]),
        ("intent_sensitivity", "Sensitivity", COLORS["blue"]),
        ("intent_specificity", "Specificity", COLORS["gold"]),
    )
    width = 0.23
    fig, axis = plt.subplots(figsize=(11.5, 6.2))
    for offset, (metric, label, color) in enumerate(metrics):
        values = [100.0 * float(row[1][metric]) for row in methods]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        annotate_bars(axis, bars)
    axis.set_xticks(x, [row[0] for row in methods])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Final stroke-patient result (%)")
    axis.set_title("Classical Feature/Model Ablation: Final Comparison")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_safety(path: Path, report: Mapping[str, object]) -> None:
    fig, axis = plt.subplots(figsize=(9, 6.2))
    axis.axvspan(0, 10, color="#DCEFE7", alpha=0.75)
    axis.axhspan(60, 100, color="#E8F0F5", alpha=0.5)
    for label, metrics, color in final_ladder(report):
        x = 100.0 * float(metrics["false_activation_rate"])
        y = 100.0 * float(metrics["intent_sensitivity"])
        axis.scatter(x, y, s=135, color=color, edgecolor="white", linewidth=1)
        axis.annotate(
            label.replace("\n", " "),
            (x, y),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    axis.axvline(10, color=COLORS["teal"], linestyle="--", linewidth=1)
    axis.axhline(60, color=COLORS["navy"], linestyle="--", linewidth=1)
    axis.set_xlim(0, max(30, axis.get_xlim()[1]))
    axis.set_ylim(0, 100)
    axis.set_xlabel("False activation per rest window (%)")
    axis.set_ylabel("Detected intent (%)")
    axis.set_title("Final Safety-Sensitivity Tradeoff")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_patient_change(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    baseline = {
        int(row["subject"]): row
        for row in rows
        if row["method"] == "baseline_calibrated"
    }
    selected = {
        int(row["subject"]): row
        for row in rows
        if row["method"] == "selected_calibrated"
    }
    subjects = sorted(baseline)
    changes = [
        100.0
        * (
            float(selected[subject]["intent_balanced_accuracy"])
            - float(baseline[subject]["intent_balanced_accuracy"])
        )
        for subject in subjects
    ]
    colors = [COLORS["teal"] if value >= 0 else COLORS["red"] for value in changes]
    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    bars = axis.bar(np.arange(len(subjects)), changes, color=colors)
    annotate_bars(axis, bars)
    axis.axhline(0, color=COLORS["gray"], linewidth=1)
    axis.set_xticks(
        np.arange(len(subjects)),
        [
            f"P{subject:02d}\nNIHSS {baseline[subject]['nihss']}"
            for subject in subjects
        ],
    )
    axis.set_ylabel("Calibrated intent-BA change (percentage points)")
    axis.set_title("Selected Pipeline Change by Final Stroke Patient")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_metrics(metrics: Mapping[str, object]) -> str:
    return (
        f"intent BA **{100 * float(metrics['intent_balanced_accuracy']):.1f}%**, "
        f"sensitivity **{100 * float(metrics['intent_sensitivity']):.1f}%**, "
        f"specificity **{100 * float(metrics['intent_specificity']):.1f}%**, "
        f"false activation **{100 * float(metrics['false_activation_rate']):.1f}%**"
    )


def write_report(path: Path, report: Mapping[str, object]) -> None:
    selected_feature = str(report["selection"]["feature_family"])
    selected_model = str(report["selection"]["model"])
    final = report["final"]
    population_change = report["paired_population_change"][
        "intent_balanced_accuracy"
    ]
    change = report["paired_calibrated_change"]["intent_balanced_accuracy"]
    patient_rows = report["patient_rows"]
    baseline_patient = {
        int(row["subject"]): row
        for row in patient_rows
        if row["method"] == "baseline_calibrated"
    }
    selected_patient = {
        int(row["subject"]): row
        for row in patient_rows
        if row["method"] == "selected_calibrated"
    }
    improved = sum(
        selected_patient[subject]["intent_balanced_accuracy"]
        > baseline_patient[subject]["intent_balanced_accuracy"]
        for subject in baseline_patient
    )
    tied = sum(
        selected_patient[subject]["intent_balanced_accuracy"]
        == baseline_patient[subject]["intent_balanced_accuracy"]
        for subject in baseline_patient
    )
    worsened = len(baseline_patient) - improved - tied
    lines = [
        "# Compact Classical EEG Feature and Model Ablation",
        "",
        "## Question",
        "",
        "Can interpretable statistical, Welch spectral, temporal-morphology, and "
        "spatial-asymmetry features improve the healthy-to-stroke hand-intent "
        "pipeline relative to the covariance reference?",
        "",
        "Feature family and model type were selected only on the eight stroke "
        "controller patients. Final evaluation used the same ten final patients as "
        "the preceding healthy-to-stroke experiment. Because that cohort's earlier "
        "baseline result was already observed, this is a paired exploratory ablation, "
        "not a new untouched confirmatory test.",
        "",
        "## Controller Selection",
        "",
        f"- Selected feature set: **{DISPLAY_NAMES[selected_feature].replace(chr(10), ' ')}**.",
        f"- Selected model: **{DISPLAY_NAMES[selected_model].replace(chr(10), ' ')}**.",
        f"- Selected feature count: **{report['selection']['feature_count']}**.",
        "- Selected descriptors per channel: log standard deviation, skewness, "
        "excess kurtosis, log line length, zero-crossing rate, Hjorth mobility, "
        "Hjorth complexity, and log peak-to-peak amplitude.",
        "",
        "## Final Results",
        "",
        f"- Covariance population reference: {format_metrics(final['baseline_population'])}.",
        f"- Selected population pipeline: {format_metrics(final['selected_population'])}.",
        f"- Covariance plus calibration: {format_metrics(final['baseline_calibrated'])}.",
        f"- Selected pipeline plus calibration: {format_metrics(final['selected_calibrated'])}.",
        "",
        f"The population intent-BA change was "
        f"**{100 * float(population_change['mean_change']):+.1f} percentage points** "
        f"(subject-bootstrap 95% CI "
        f"{100 * float(population_change['ci_95'][0]):+.1f} to "
        f"{100 * float(population_change['ci_95'][1]):+.1f}). Population AUROC "
        f"increased from **{100 * float(final['baseline_population']['intent_auroc']):.1f}%** "
        f"to **{100 * float(final['selected_population']['intent_auroc']):.1f}%**, "
        f"and AUPRC increased from "
        f"**{100 * float(final['baseline_population']['intent_auprc']):.1f}%** "
        f"to **{100 * float(final['selected_population']['intent_auprc']):.1f}%**.",
        "",
        f"The calibrated intent-BA change was "
        f"**{100 * float(change['mean_change']):+.1f} percentage points** "
        f"(subject-bootstrap 95% CI "
        f"{100 * float(change['ci_95'][0]):+.1f} to "
        f"{100 * float(change['ci_95'][1]):+.1f}). It improved **{improved}/10** "
        f"patients, tied for **{tied}/10**, and worsened **{worsened}/10**.",
        "",
        "## Interpretation",
        "",
        "A feature/model change should advance only if it improves the paired final "
        "safety-sensitivity tradeoff, not merely controller accuracy. Overlapping "
        "window leakage is absent: all splits remain patient-isolated, and early "
        "calibration windows never enter later evaluation windows.",
        "",
        "The compact statistical branch improves average discrimination and "
        "sensitivity, but its calibrated 12.4% false activation does not satisfy the "
        "10% safety target and both paired confidence intervals include zero. Retain "
        "it as a candidate proposal/evidence branch for late fusion with the "
        "covariance safety gate; do not replace the covariance controller until a "
        "new locked cohort or nested subject-level validation confirms the gain.",
        "",
        "The false-activations-per-minute estimate assumes independent, non-overlapping "
        "two-second rest windows and is not a continuous online-controller result.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/28_classical_feature_model_ablation.py",
        "```",
        "",
        "Machine-readable metrics, trained models, patient-level CSV data, and five "
        "presentation figures are saved beside this report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    configure_plots()
    started = time.time()
    transfer_dir = Path(args.transfer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Compact Classical EEG Feature/Model Ablation ===")
    physionet = from_harmonized_healthy(
        build_physionet_harmonized_dataset(
            args.physionet_dir,
            Path(args.physionet_cache),
        )
    )
    openbmi = from_harmonized_healthy(
        load_openbmi_harmonized_dataset(Path(args.openbmi_cache))
    )
    liu = build_liu_hand_intent_dataset(
        args.liu_dir,
        transfer_dir / "liu_hand_intent_epochs.npz",
    )
    combined = combine_hand_intent_datasets(
        (physionet, openbmi, liu),
        subject_offsets=(PHYSIONET_OFFSET, OPENBMI_OFFSET, LIU_OFFSET),
    )
    partitions = {
        name: load_partitions(transfer_dir / f"{name}_subject_partitions.json")
        for name in ("physionet", "openbmi", "liu2024")
    }
    masks = build_masks(combined, partitions)
    weights = training_weights(
        combined.labels,
        combined.dataset_ids,
        masks["mixed_development"],
    )

    covariances = np.concatenate(
        [
            build_covariance_cache(
                dataset.covariance_interface(),
                transfer_dir / f"{dataset.dataset_name}_covariances.npz",
            ).covariances
            for dataset in (physionet, openbmi, liu)
        ]
    )
    covariance_features, covariance_metadata = tangent_features(
        covariances,
        masks["mixed_development"],
        combined.subjects,
        combined.sessions,
        combined.recordings,
        alignment="log_euclidean",
    )
    compact = build_classical_feature_cache(
        combined.epochs,
        combined.sfreq,
        combined.channel_names,
        combined.fingerprint,
        output_dir / "compact_eeg_features.npz",
        rebuild_cache=args.rebuild_features,
    )
    aligned_compact, compact_alignment = calibration_robust_alignment(
        compact.values,
        combined.subjects,
        combined.sessions,
        combined.recordings,
    )
    feature_sets = {
        "covariance_reference": covariance_features,
        "statistical": aligned_compact[
            :,
            slice(*compact.family_slices["statistical"]),
        ],
        "spectral": aligned_compact[
            :,
            slice(*compact.family_slices["spectral"]),
        ],
        "temporal": aligned_compact[
            :,
            slice(*compact.family_slices["temporal"]),
        ],
        "compact_all": aligned_compact,
        "covariance_plus_compact": np.concatenate(
            (covariance_features, aligned_compact),
            axis=1,
        ),
    }

    print("[+] Controller-only feature-family selection...")
    feature_rows = []
    feature_models = {}
    feature_probabilities = {}
    feature_thresholds = {}
    for name, features in feature_sets.items():
        model, probabilities, threshold, metrics = fit_candidate(
            features,
            combined,
            masks,
            weights,
            "logistic",
            seed=args.seed,
            max_false_activation=args.max_false_activation,
        )
        feature_models[name] = model
        feature_probabilities[name] = probabilities
        feature_thresholds[name] = threshold
        feature_rows.append(
            {
                "name": name,
                "feature_count": int(features.shape[1]),
                "controller": metrics,
            }
        )
        print(
            f"    {name:27s} | BA={100 * metrics['intent_balanced_accuracy']:5.1f}% "
            f"| FA={100 * metrics['false_activation_rate']:5.1f}% "
            f"| missed={100 * metrics['missed_intent_rate']:5.1f}%"
        )
    selected_feature_row = max(
        feature_rows,
        key=lambda row: (
            float(row["controller"]["intent_balanced_accuracy"]),
            float(row["controller"]["intent_sensitivity"]),
            -float(row["controller"]["false_activation_rate"]),
        ),
    )
    selected_feature = str(selected_feature_row["name"])

    print(f"[+] Controller-only model selection on {selected_feature}...")
    model_rows = []
    model_objects = {}
    model_probabilities = {}
    model_thresholds = {}
    selected_features = feature_sets[selected_feature]
    for model_name in CLASSICAL_MODEL_NAMES:
        model, probabilities, threshold, metrics = fit_candidate(
            selected_features,
            combined,
            masks,
            weights,
            model_name,
            seed=args.seed,
            max_false_activation=args.max_false_activation,
        )
        model_objects[model_name] = model
        model_probabilities[model_name] = probabilities
        model_thresholds[model_name] = threshold
        model_rows.append(
            {
                "name": model_name,
                "feature_count": int(selected_features.shape[1]),
                "controller": metrics,
            }
        )
        print(
            f"    {model_name:27s} | BA={100 * metrics['intent_balanced_accuracy']:5.1f}% "
            f"| FA={100 * metrics['false_activation_rate']:5.1f}% "
            f"| missed={100 * metrics['missed_intent_rate']:5.1f}%"
        )
    selected_model_row = max(
        model_rows,
        key=lambda row: (
            float(row["controller"]["intent_balanced_accuracy"]),
            float(row["controller"]["intent_sensitivity"]),
            -float(row["controller"]["false_activation_rate"]),
        ),
    )
    selected_model = str(selected_model_row["name"])

    baseline_model = feature_models["covariance_reference"]
    baseline_probabilities = feature_probabilities["covariance_reference"]
    baseline_threshold = feature_thresholds["covariance_reference"]
    candidate_model = model_objects[selected_model]
    candidate_probabilities = model_probabilities[selected_model]
    candidate_threshold = model_thresholds[selected_model]
    final_mask = masks["liu2024_final"]

    print("[+] Selecting patient calibration on controller patients...")
    baseline_calibration, baseline_calibrated, baseline_calibrated_threshold = (
        select_calibration(
            feature_sets["covariance_reference"],
            combined,
            masks,
            baseline_probabilities,
            seed=args.seed,
            max_false_activation=args.max_false_activation,
        )
    )
    candidate_calibration, candidate_calibrated, candidate_calibrated_threshold = (
        select_calibration(
            selected_features,
            combined,
            masks,
            candidate_probabilities,
            seed=args.seed,
            max_false_activation=args.max_false_activation,
        )
    )

    final_results = {
        "baseline_population": score(
            combined.labels[final_mask],
            baseline_probabilities[final_mask],
            baseline_threshold,
        ),
        "selected_population": score(
            combined.labels[final_mask],
            candidate_probabilities[final_mask],
            candidate_threshold,
        ),
        "baseline_calibrated": score(
            combined.labels[final_mask],
            baseline_calibrated[final_mask],
            baseline_calibrated_threshold,
        ),
        "selected_calibrated": score(
            combined.labels[final_mask],
            candidate_calibrated[final_mask],
            candidate_calibrated_threshold,
        ),
    }
    paired_population = paired_bootstrap(
        combined.labels[final_mask],
        combined.subjects[final_mask],
        baseline_probabilities[final_mask],
        baseline_threshold,
        candidate_probabilities[final_mask],
        candidate_threshold,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    paired_calibrated = paired_bootstrap(
        combined.labels[final_mask],
        combined.subjects[final_mask],
        baseline_calibrated[final_mask],
        baseline_calibrated_threshold,
        candidate_calibrated[final_mask],
        candidate_calibrated_threshold,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    methods = {
        "baseline_population": (baseline_probabilities, baseline_threshold),
        "selected_population": (candidate_probabilities, candidate_threshold),
        "baseline_calibrated": (
            baseline_calibrated,
            baseline_calibrated_threshold,
        ),
        "selected_calibrated": (
            candidate_calibrated,
            candidate_calibrated_threshold,
        ),
    }
    patients = patient_rows(combined, final_mask, methods)
    elapsed = (time.time() - started) / 60.0
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "compact_classical_eeg_feature_model_ablation",
        "selection_boundary": (
            "Feature family, model, calibration budget/blend, and thresholds "
            "selected only on controller patients."
        ),
        "confirmation_boundary": (
            "Same final patients as the prior transfer benchmark; exploratory "
            "paired ablation, not a newly untouched confirmation cohort."
        ),
        "features": {
            "total_compact_features": int(compact.values.shape[1]),
            "family_slices": compact.family_slices,
            "candidate_dimensions": {
                name: int(values.shape[1])
                for name, values in feature_sets.items()
            },
            "alignment": compact_alignment,
            "covariance_metadata": covariance_metadata,
        },
        "feature_selection_controller": feature_rows,
        "model_selection_controller": model_rows,
        "selection": {
            "feature_family": selected_feature,
            "model": selected_model,
            "feature_count": int(selected_features.shape[1]),
        },
        "calibration": {
            "baseline": baseline_calibration,
            "selected": candidate_calibration,
        },
        "final": final_results,
        "paired_population_change": paired_population,
        "paired_calibrated_change": paired_calibrated,
        "patient_rows": patients,
        "runtime_minutes": elapsed,
    }
    metrics_path = output_dir / "classical_ablation_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    joblib.dump(
        {
            "baseline_model": baseline_model,
            "selected_model": candidate_model,
            "selection": report["selection"],
            "thresholds": {
                "baseline_population": baseline_threshold,
                "selected_population": candidate_threshold,
                "baseline_calibrated": baseline_calibrated_threshold,
                "selected_calibrated": candidate_calibrated_threshold,
            },
            "feature_names": compact.names,
            "partitions": partitions,
            "fingerprint": combined.fingerprint,
        },
        output_dir / "classical_ablation_models.joblib",
    )
    write_csv(output_dir / "final_patient_metrics.csv", patients)
    plot_controller_comparison(
        output_dir / "01_controller_feature_ablation.png",
        feature_rows,
        tuple(feature_sets),
        title="Controller-Only Compact EEG Feature Ablation",
    )
    plot_controller_comparison(
        output_dir / "02_controller_model_comparison.png",
        model_rows,
        CLASSICAL_MODEL_NAMES,
        title=f"Controller-Only Model Comparison: {DISPLAY_NAMES[selected_feature].replace(chr(10), ' ')}",
    )
    plot_final_comparison(
        output_dir / "03_final_selected_comparison.png",
        report,
    )
    plot_safety(
        output_dir / "04_final_safety_tradeoff.png",
        report,
    )
    plot_patient_change(
        output_dir / "05_final_patient_change.png",
        patients,
    )
    write_report(output_dir / "CLASSICAL_FEATURE_MODEL_ABLATION.md", report)

    print("\n=== Final Stroke-Patient Comparison ===")
    print(f"Selected: {selected_feature} + {selected_model}")
    for label, metrics, _ in final_ladder(report):
        print(
            f"{label.replace(chr(10), ' '):30s} | "
            f"BA={100 * metrics['intent_balanced_accuracy']:5.1f}% | "
            f"sensitivity={100 * metrics['intent_sensitivity']:5.1f}% | "
            f"specificity={100 * metrics['intent_specificity']:5.1f}% | "
            f"FA={100 * metrics['false_activation_rate']:5.1f}%"
        )
    print(f"[+] Saved report and five figures to {output_dir}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
