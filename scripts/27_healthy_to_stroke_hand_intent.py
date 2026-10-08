#!/usr/bin/env python3
"""Run the hand-only healthy-to-stroke weak motor-intent study."""

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

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import trapezoid  # noqa: E402
from scipy.signal import welch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.clinical_transfer import (  # noqa: E402
    CLINICAL_CHANNELS,
    CLINICAL_EPOCH_DURATION_SECONDS,
    CLINICAL_SFREQ,
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
    CommandPolicy,
    HierarchicalFeatureDecoder,
    StageProbabilities,
    build_covariance_cache,
    evaluate_policy,
    locally_adapted_probabilities,
    subject_bootstrap_interval,
    tangent_features,
)
from bci_pipeline.splits import (  # noqa: E402
    SubjectPartitions,
    save_partitions,
    three_way_subject_split,
)


PHYSIONET_OFFSET = 0
OPENBMI_OFFSET = 1000
LIU_OFFSET = 2000
DATASET_LABELS = {
    "physionet": "PhysioNet healthy",
    "openbmi": "OpenBMI healthy",
    "liu2024": "Liu acute stroke",
}
COLORS = {
    "navy": "#17324D",
    "blue": "#2878A6",
    "teal": "#21867A",
    "orange": "#D56A3A",
    "gold": "#E5A23C",
    "red": "#B94B4B",
    "gray": "#66737E",
    "light": "#EEF2F3",
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
        "--healthy-reference",
        default="artifacts/dual_task_benchmark/dual_task_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/healthy_to_stroke_hand_intent",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--max-liu-subjects", type=int)
    parser.add_argument("--rebuild-physionet-cache", action="store_true")
    parser.add_argument("--rebuild-liu-cache", action="store_true")
    parser.add_argument("--rebuild-covariances", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
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


def subset_probabilities(
    probabilities: StageProbabilities,
    mask: np.ndarray,
) -> StageProbabilities:
    return StageProbabilities(
        intent=probabilities.intent[mask],
        direction=probabilities.direction[mask],
    )


def fit_decoder(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    dataset_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[HierarchicalFeatureDecoder, StageProbabilities, np.ndarray]:
    train_datasets = np.unique(dataset_ids[train_mask])
    weights = np.ones(len(labels), dtype=np.float64)
    train_total = int(np.sum(train_mask))
    for dataset_name in train_datasets:
        selected = train_mask & (dataset_ids == dataset_name)
        weights[selected] = train_total / (
            len(train_datasets) * int(np.sum(selected))
        )
    model = HierarchicalFeatureDecoder(seed).fit(
        features[train_mask],
        labels[train_mask],
        weights[train_mask],
    )
    return model, model.predict_stage_probabilities(features), weights


def intent_policy(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    *,
    max_false_activation: float | None,
) -> CommandPolicy:
    true_intent = labels != 0
    candidates: list[tuple[float, float, float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = probabilities.intent >= threshold
        false_activation = float(np.mean(predicted[~true_intent]))
        sensitivity = float(np.mean(predicted[true_intent]))
        score = 0.5 * (sensitivity + 1.0 - false_activation)
        if (
            max_false_activation is not None
            and false_activation > max_false_activation + 1e-12
        ):
            continue
        candidates.append((float(threshold), score, sensitivity, false_activation))
    if not candidates:
        return CommandPolicy(intent_threshold=1.0, direction_margin=0.0)
    selected = max(
        candidates,
        key=lambda row: (
            row[1],
            row[2],
            -abs(row[0] - 0.5),
        ),
    )
    return CommandPolicy(intent_threshold=selected[0], direction_margin=0.0)


def score(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    policy: CommandPolicy,
) -> dict[str, object]:
    result = evaluate_policy(labels, probabilities, policy)
    true_intent = (labels != 0).astype(np.int64)
    predicted = probabilities.intent >= policy.intent_threshold
    result.update(
        {
            "rows": int(len(labels)),
            "rest_rows": int(np.sum(true_intent == 0)),
            "intent_rows": int(np.sum(true_intent == 1)),
            "intent_sensitivity": float(np.mean(predicted[true_intent == 1])),
            "intent_specificity": float(np.mean(~predicted[true_intent == 0])),
            "intent_auroc": float(
                roc_auc_score(true_intent, probabilities.intent)
            ),
            "intent_auprc": float(
                average_precision_score(true_intent, probabilities.intent)
            ),
            "intent_brier_score": float(
                brier_score_loss(true_intent, probabilities.intent)
            ),
            "estimated_false_activations_per_rest_minute": float(
                result["false_activation_rate"]
                * 60.0
                / CLINICAL_EPOCH_DURATION_SECONDS
            ),
        }
    )
    return result


def policies_and_scores(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    controller_mask: np.ndarray,
    evaluation_masks: Mapping[str, np.ndarray],
    *,
    max_false_activation: float,
) -> tuple[dict[str, object], CommandPolicy, CommandPolicy]:
    controller = subset_probabilities(probabilities, controller_mask)
    balanced = intent_policy(
        labels[controller_mask],
        controller,
        max_false_activation=None,
    )
    safety = intent_policy(
        labels[controller_mask],
        controller,
        max_false_activation=max_false_activation,
    )
    results: dict[str, object] = {
        "balanced_policy": asdict(balanced),
        "safety_policy": asdict(safety),
        "controller_balanced": score(
            labels[controller_mask],
            controller,
            balanced,
        ),
        "controller_safety": score(
            labels[controller_mask],
            controller,
            safety,
        ),
    }
    for name, mask in evaluation_masks.items():
        selected = subset_probabilities(probabilities, mask)
        results[f"{name}_balanced"] = score(labels[mask], selected, balanced)
        results[f"{name}_safety"] = score(labels[mask], selected, safety)
    return results, balanced, safety


def local_subjects(
    partitions: SubjectPartitions,
    offset: int,
) -> dict[str, np.ndarray]:
    return {
        "development": np.asarray(partitions.development, dtype=np.int64) + offset,
        "controller": np.asarray(
            partitions.controller_validation,
            dtype=np.int64,
        )
        + offset,
        "final": np.asarray(partitions.final_target, dtype=np.int64) + offset,
    }


def build_masks(
    dataset: HandIntentDataset,
    partitions: Mapping[str, SubjectPartitions],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for dataset_name, offset in (
        ("physionet", PHYSIONET_OFFSET),
        ("openbmi", OPENBMI_OFFSET),
        ("liu2024", LIU_OFFSET),
    ):
        identifiers = local_subjects(partitions[dataset_name], offset)
        dataset_mask = dataset.dataset_ids == dataset_name
        result[f"{dataset_name}_development"] = dataset_mask & np.isin(
            dataset.subjects,
            identifiers["development"],
        )
        result[f"{dataset_name}_development_calibration"] = result[
            f"{dataset_name}_development"
        ] & (dataset.recordings == 0)
        result[f"{dataset_name}_controller_all"] = dataset_mask & np.isin(
            dataset.subjects,
            identifiers["controller"],
        )
        result[f"{dataset_name}_controller_calibration"] = result[
            f"{dataset_name}_controller_all"
        ] & (dataset.recordings == 0)
        result[f"{dataset_name}_controller"] = result[
            f"{dataset_name}_controller_all"
        ] & (dataset.recordings == 1)
        result[f"{dataset_name}_final_all"] = dataset_mask & np.isin(
            dataset.subjects,
            identifiers["final"],
        )
        result[f"{dataset_name}_final_calibration"] = result[
            f"{dataset_name}_final_all"
        ] & (dataset.recordings == 0)
        result[f"{dataset_name}_final"] = result[
            f"{dataset_name}_final_all"
        ] & (dataset.recordings == 1)

    result["healthy_development"] = (
        result["physionet_development_calibration"]
        | result["openbmi_development_calibration"]
    )
    result["healthy_controller"] = (
        result["physionet_controller"] | result["openbmi_controller"]
    )
    result["healthy_final"] = (
        result["physionet_final"] | result["openbmi_final"]
    )
    result["mixed_development"] = (
        result["healthy_development"] | result["liu2024_development"]
    )
    return result


def validate_masks(
    labels: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> None:
    required = (
        "healthy_development",
        "healthy_controller",
        "healthy_final",
        "liu2024_development",
        "liu2024_controller",
        "liu2024_final",
    )
    for name in required:
        if not np.any(masks[name]):
            raise ValueError(f"Required split is empty: {name}")
        if set(np.unique(labels[masks[name]])) != {0, 1, 2}:
            raise ValueError(f"Required split is missing a class: {name}")


def select_patient_calibration(
    features: np.ndarray,
    dataset: HandIntentDataset,
    masks: Mapping[str, np.ndarray],
    global_model: HierarchicalFeatureDecoder,
    *,
    max_false_activation: float,
    seed: int,
) -> tuple[dict[str, object], StageProbabilities, CommandPolicy]:
    configurations: list[dict[str, object]] = []
    for budget in (2, 5, 10):
        for blend in (0.25, 0.5, 0.75, 1.0):
            probabilities = locally_adapted_probabilities(
                features,
                dataset.labels,
                dataset.subjects,
                dataset.recordings,
                masks["liu2024_controller"],
                global_model,
                budget_per_class=budget,
                blend_weight=blend,
                random_seed=seed,
            )
            controller = subset_probabilities(
                probabilities,
                masks["liu2024_controller"],
            )
            policy = intent_policy(
                dataset.labels[masks["liu2024_controller"]],
                controller,
                max_false_activation=max_false_activation,
            )
            metrics = score(
                dataset.labels[masks["liu2024_controller"]],
                controller,
                policy,
            )
            configurations.append(
                {
                    "budget_per_class": budget,
                    "blend_weight": blend,
                    "policy": asdict(policy),
                    "controller": metrics,
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
    budget = int(selected["budget_per_class"])
    blend = float(selected["blend_weight"])
    final_probabilities = locally_adapted_probabilities(
        features,
        dataset.labels,
        dataset.subjects,
        dataset.recordings,
        masks["liu2024_final"],
        global_model,
        budget_per_class=budget,
        blend_weight=blend,
        random_seed=seed,
    )
    policy = CommandPolicy(**selected["policy"])
    report = {
        "selected_on_stroke_controller": selected,
        "all_controller_configurations": configurations,
        "final": score(
            dataset.labels[masks["liu2024_final"]],
            subset_probabilities(
                final_probabilities,
                masks["liu2024_final"],
            ),
            policy,
        ),
    }
    return report, final_probabilities, policy


def load_healthy_reference(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    required = ("unilateral_execution", "unilateral_imagery")
    if not all(name in results for name in required):
        return None
    return {
        name: {
            "balanced": results[name]["balanced"],
            "safety": results[name]["safety"],
            "train_trials": results[name]["train_trials"],
            "validation_trials": results[name]["validation_trials"],
            "test_trials": results[name]["test_trials"],
        }
        for name in required
    }


def _bandpower(
    epochs: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    frequencies, psd = welch(
        epochs,
        fs=CLINICAL_SFREQ,
        nperseg=epochs.shape[-1],
        axis=-1,
    )
    band = (frequencies >= low) & (frequencies <= high)
    power = trapezoid(psd[..., band], frequencies[band], axis=-1)
    return 10.0 * np.log10(np.clip(power, 1e-20, None))


def physiology_rows(
    dataset: HandIntentDataset,
    evaluation_mask: np.ndarray,
    features: np.ndarray,
    healthy_training_mask: np.ndarray,
) -> list[dict[str, object]]:
    indices = np.flatnonzero(evaluation_mask)
    mu = _bandpower(dataset.epochs[indices], 8.0, 13.0)
    beta = _bandpower(dataset.epochs[indices], 13.0, 30.0)
    scaler = StandardScaler().fit(features[healthy_training_mask])
    standardized = scaler.transform(features[indices])
    ood = np.linalg.norm(standardized, axis=1) / np.sqrt(features.shape[1])
    c3 = list(CLINICAL_CHANNELS).index("C3")
    c4 = list(CLINICAL_CHANNELS).index("C4")
    rows: list[dict[str, object]] = []
    for subject in np.unique(dataset.subjects[indices]):
        local = dataset.subjects[indices] == subject
        labels = dataset.labels[indices][local]
        rest = labels == 0
        intent = labels != 0
        dataset_name = str(dataset.dataset_ids[indices][local][0])
        rest_mu = np.median(mu[local][rest], axis=0)
        rest_beta = np.median(beta[local][rest], axis=0)
        intent_mu = mu[local][intent] - rest_mu
        intent_beta = beta[local][intent] - rest_beta
        intent_labels = labels[intent]
        contra_mu = np.where(
            intent_labels[:, None] == 1,
            intent_mu[:, [c4]],
            intent_mu[:, [c3]],
        )[:, 0]
        ipsi_mu = np.where(
            intent_labels[:, None] == 1,
            intent_mu[:, [c3]],
            intent_mu[:, [c4]],
        )[:, 0]
        offset = {
            "physionet": PHYSIONET_OFFSET,
            "openbmi": OPENBMI_OFFSET,
            "liu2024": LIU_OFFSET,
        }[dataset_name]
        rows.append(
            {
                "dataset": dataset_name,
                "cohort": "stroke" if dataset_name == "liu2024" else "healthy",
                "subject": int(subject - offset),
                "mu_intent_minus_rest_db": float(np.median(intent_mu)),
                "beta_intent_minus_rest_db": float(np.median(intent_beta)),
                "mu_contralateral_minus_ipsilateral_db": float(
                    np.median(contra_mu - ipsi_mu)
                ),
                "healthy_feature_distance": float(np.median(ood[local])),
                "paralysis_side": str(
                    dataset.paralysis_sides[indices][local][0]
                ),
                "nihss": (
                    float(dataset.nihss[indices][local][0])
                    if np.isfinite(dataset.nihss[indices][local][0])
                    else ""
                ),
            }
        )
    return rows


def per_patient_rows(
    dataset: HandIntentDataset,
    mask: np.ndarray,
    methods: Mapping[
        str,
        tuple[StageProbabilities, CommandPolicy],
    ],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subject in np.unique(dataset.subjects[mask]):
        selected = mask & (dataset.subjects == subject)
        base = {
            "subject": int(subject - LIU_OFFSET),
            "paralysis_side": str(dataset.paralysis_sides[selected][0]),
            "nihss": int(dataset.nihss[selected][0]),
        }
        for method, (probabilities, policy) in methods.items():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="y_pred contains classes not in y_true",
                    category=UserWarning,
                )
                metrics = score(
                    dataset.labels[selected],
                    subset_probabilities(probabilities, selected),
                    policy,
                )
            rows.append({"method": method, **base, **metrics})
    return rows


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def predicted_labels(
    probabilities: StageProbabilities,
    policy: CommandPolicy,
) -> np.ndarray:
    intent = probabilities.intent >= policy.intent_threshold
    direction = np.argmax(probabilities.direction, axis=1) + 1
    return np.where(intent, direction, 0)


def write_predictions(
    path: Path,
    dataset: HandIntentDataset,
    mask: np.ndarray,
    methods: Mapping[str, tuple[StageProbabilities, CommandPolicy]],
) -> None:
    indices = np.flatnonzero(mask)
    predictions = {
        name: predicted_labels(probabilities, policy)
        for name, (probabilities, policy) in methods.items()
    }
    rows = []
    for index in indices:
        row: dict[str, object] = {
            "dataset": str(dataset.dataset_ids[index]),
            "subject": int(dataset.subjects[index] - LIU_OFFSET),
            "trial_index": int(dataset.trial_indices[index]),
            "true_label": int(dataset.labels[index]),
            "paralysis_side": str(dataset.paralysis_sides[index]),
            "nihss": int(dataset.nihss[index]),
        }
        for name, (probabilities, _) in methods.items():
            row[f"{name}_intent_probability"] = float(
                probabilities.intent[index]
            )
            row[f"{name}_prediction"] = int(predictions[name][index])
        rows.append(row)
    write_csv(path, rows)


def annotate_bars(axis: plt.Axes, bars: object) -> None:
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9, fontweight="bold")


def plot_healthy_reference(
    path: Path,
    reference: Mapping[str, object] | None,
) -> None:
    if reference is None:
        return
    conditions = ("unilateral_execution", "unilateral_imagery")
    labels = ("Physical movement", "Motor imagery")
    metrics = (
        ("intent_balanced_accuracy", "Intent balanced accuracy"),
        ("direction_balanced_accuracy", "Left/right balanced accuracy"),
        ("false_activation_rate", "False activation"),
        ("missed_intent_rate", "Missed intent"),
    )
    x = np.arange(len(metrics))
    width = 0.34
    fig, axis = plt.subplots(figsize=(11.2, 5.8))
    for offset, (condition, label, color) in enumerate(
        zip(conditions, labels, (COLORS["navy"], COLORS["orange"]))
    ):
        values = [
            100.0 * float(reference[condition]["balanced"][metric])
            for metric, _ in metrics
        ]
        bars = axis.bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=label,
            color=color,
        )
        annotate_bars(axis, bars)
    axis.set_xticks(x, [label for _, label in metrics])
    axis.set_ylabel("Final unseen-subject result (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Phase 1: Healthy Hand Movement as an Intent Reference")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_healthy_generalization(
    path: Path,
    results: Mapping[str, object],
) -> None:
    datasets = ("physionet_final_safety", "openbmi_final_safety")
    labels = ("PhysioNet", "OpenBMI")
    metrics = (
        ("intent_balanced_accuracy", "Intent BA"),
        ("intent_sensitivity", "Sensitivity"),
        ("intent_specificity", "Specificity"),
    )
    x = np.arange(len(labels))
    width = 0.22
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    for offset, (metric, label) in enumerate(metrics):
        values = [100.0 * float(results[name][metric]) for name in datasets]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=label,
            color=(COLORS["blue"], COLORS["teal"], COLORS["gray"])[offset],
        )
        annotate_bars(axis, bars)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Final unseen-subject result (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Phase 2: One Healthy Model Across Two EEG Datasets")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def stroke_ladder(
    report: Mapping[str, object],
) -> list[tuple[str, Mapping[str, object], str]]:
    return [
        (
            "Frozen healthy\n(strict)",
            report["zero_shot"]["strict_unaligned"]["liu_final_safety"],
            COLORS["gray"],
        ),
        (
            "Frozen healthy\n(label-free alignment)",
            report["zero_shot"]["label_free_aligned"]["liu_final_safety"],
            COLORS["blue"],
        ),
        (
            "Healthy + stroke\npopulation adaptation",
            report["stroke_adaptation"]["mixed_population"]["liu_final_safety"],
            COLORS["teal"],
        ),
        (
            "Population +\npatient calibration",
            report["stroke_adaptation"]["patient_calibration"]["final"],
            COLORS["orange"],
        ),
    ]


def plot_stroke_ladder(path: Path, report: Mapping[str, object]) -> None:
    methods = stroke_ladder(report)
    metrics = (
        ("intent_balanced_accuracy", "Intent BA"),
        ("intent_sensitivity", "Sensitivity"),
        ("intent_specificity", "Specificity"),
    )
    x = np.arange(len(methods))
    width = 0.23
    fig, axis = plt.subplots(figsize=(12.2, 6.2))
    for offset, (metric, label) in enumerate(metrics):
        values = [100.0 * float(row[1][metric]) for row in methods]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=label,
            color=(COLORS["navy"], COLORS["blue"], COLORS["gold"])[offset],
        )
        annotate_bars(axis, bars)
    axis.set_xticks(x, [row[0] for row in methods])
    axis.set_ylabel("Locked Liu final-patient result (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Phases 3-4: Healthy-to-Stroke Transfer and Adaptation")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_safety_tradeoff(path: Path, report: Mapping[str, object]) -> None:
    methods = stroke_ladder(report)
    fig, axis = plt.subplots(figsize=(9, 6.2))
    axis.axvspan(0, 10, color="#DCEFE7", alpha=0.75)
    axis.axhspan(60, 100, color="#E8F0F5", alpha=0.45)
    for label, metrics, color in methods:
        false_activation = 100.0 * float(metrics["false_activation_rate"])
        sensitivity = 100.0 * float(metrics["intent_sensitivity"])
        axis.scatter(
            false_activation,
            sensitivity,
            s=130,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        axis.annotate(
            label.replace("\n", " "),
            (false_activation, sensitivity),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    axis.axvline(10, color=COLORS["teal"], linestyle="--", linewidth=1.2)
    axis.axhline(60, color=COLORS["navy"], linestyle="--", linewidth=1.2)
    axis.set_xlim(0, max(35, axis.get_xlim()[1]))
    axis.set_ylim(0, 100)
    axis.set_xlabel("False activation per rest window (%)")
    axis.set_ylabel("Detected motor intent (%)")
    axis.set_title("Locked Stroke Safety-Sensitivity Tradeoff")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_shift(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    groups = ("physionet", "openbmi", "liu2024")
    labels = ("PhysioNet\nhealthy", "OpenBMI\nhealthy", "Liu\nstroke")
    metrics = (
        ("mu_intent_minus_rest_db", "Mu intent - rest", "Power change (dB)"),
        ("beta_intent_minus_rest_db", "Beta intent - rest", "Power change (dB)"),
        (
            "mu_contralateral_minus_ipsilateral_db",
            "Mu laterality",
            "Contralateral - ipsilateral (dB)",
        ),
        (
            "healthy_feature_distance",
            "Healthy-reference distance",
            "Standardized distance",
        ),
    )
    colors = (COLORS["navy"], COLORS["blue"], COLORS["orange"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.ravel(), metrics):
        values = [
            [float(row[metric]) for row in rows if row["dataset"] == group]
            for group in groups
        ]
        boxes = axis.boxplot(
            values,
            patch_artist=True,
            widths=0.58,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 2},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
        rng = np.random.default_rng(42)
        for index, (group_values, color) in enumerate(zip(values, colors), start=1):
            jitter = rng.normal(index, 0.045, len(group_values))
            axis.scatter(
                jitter,
                group_values,
                s=18,
                color=color,
                alpha=0.5,
                edgecolor="none",
            )
        axis.axhline(0, color=COLORS["gray"], linewidth=0.8, alpha=0.6)
        axis.set_xticks((1, 2, 3), labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Exploratory Healthy-Stroke Feature Comparison",
        fontsize=16,
        fontweight="bold",
        color=COLORS["navy"],
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_patient_variability(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    methods = (
        ("strict_zero_shot", "Frozen healthy", COLORS["gray"]),
        ("aligned_zero_shot", "Label-free alignment", COLORS["blue"]),
        ("mixed_population", "Population adapted", COLORS["teal"]),
        ("patient_calibrated", "Patient calibrated", COLORS["orange"]),
    )
    patient_rows = {
        int(row["subject"]): row
        for row in rows
        if row["method"] == "patient_calibrated"
    }
    order = sorted(
        patient_rows,
        key=lambda subject: float(patient_rows[subject]["intent_balanced_accuracy"]),
    )
    x = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(12.5, 6))
    for method, label, color in methods:
        lookup = {
            int(row["subject"]): row
            for row in rows
            if row["method"] == method
        }
        values = [
            100.0 * float(lookup[subject]["intent_balanced_accuracy"])
            for subject in order
        ]
        axis.plot(
            x,
            values,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=label,
        )
    axis.axhline(50, color=COLORS["gray"], linestyle=":", linewidth=1)
    axis.set_xticks(
        x,
        [
            f"P{subject:02d}\nNIHSS {int(patient_rows[subject]['nihss'])}"
            for subject in order
        ],
    )
    axis.set_ylabel("Intent balanced accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Locked Stroke-Patient Variability")
    axis.legend(frameon=False, ncol=4, loc="lower center")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_result(metrics: Mapping[str, object]) -> str:
    return (
        f"intent BA **{100 * float(metrics['intent_balanced_accuracy']):.1f}%**, "
        f"sensitivity **{100 * float(metrics['intent_sensitivity']):.1f}%**, "
        f"specificity **{100 * float(metrics['intent_specificity']):.1f}%**, "
        f"false activation **{100 * float(metrics['false_activation_rate']):.1f}%**, "
        f"missed intent **{100 * float(metrics['missed_intent_rate']):.1f}%**"
    )


def write_report(path: Path, report: Mapping[str, object]) -> None:
    zero_strict = report["zero_shot"]["strict_unaligned"]["liu_final_safety"]
    zero_aligned = report["zero_shot"]["label_free_aligned"]["liu_final_safety"]
    mixed = report["stroke_adaptation"]["mixed_population"]["liu_final_safety"]
    calibrated = report["stroke_adaptation"]["patient_calibration"]["final"]
    lines = [
        "# Healthy-to-Stroke Weak Hand-Intent Study",
        "",
        "## Objective",
        "",
        "Develop a hand-only `motor_intent` versus `rest` process with healthy EEG, "
        "test the frozen process on unseen stroke patients, and then quantify the "
        "gain from population-level stroke adaptation and limited patient calibration. "
        "PhysioNet bilateral fist/foot runs are excluded.",
        "",
        "## Locked Study Order",
        "",
        "1. Healthy movement-versus-imagery reference on PhysioNet unilateral tasks.",
        "2. Healthy intent-model development with PhysioNet and OpenBMI.",
        "3. Zero-label transfer of the frozen healthy model to final Liu patients.",
        "4. Stroke-domain adaptation using separate Liu development/controller patients.",
        "5. Limited calibration from early trials of final patients, evaluated on later trials.",
        "",
        "## Primary Results",
        "",
        f"- Strict frozen healthy transfer: {format_result(zero_strict)}.",
        f"- Frozen healthy plus label-free alignment: {format_result(zero_aligned)}.",
        f"- Mixed healthy/stroke population adaptation: {format_result(mixed)}.",
        f"- Population model plus patient calibration: {format_result(calibrated)}.",
        "",
        "The estimated false-activations-per-minute values assume independent, "
        f"non-overlapping {CLINICAL_EPOCH_DURATION_SECONDS:.0f}-second rest windows. "
        "They are descriptive offline rates, not continuous-controller measurements.",
        "",
        "## Harmonization",
        "",
        f"- Channels: `{', '.join(CLINICAL_CHANNELS)}`.",
        f"- Sampling: **{CLINICAL_SFREQ:.0f} Hz**.",
        "- Preprocessing: full-EEG CAR and 4-38 Hz zero-phase offline filtering.",
        f"- Window length: **{CLINICAL_EPOCH_DURATION_SECONDS:.1f} seconds**.",
        "- Representation: five-band regularized covariance with tangent-space features.",
        "- Decoder: hierarchical regularized logistic regression; intent is primary.",
        "- PhysioNet: unilateral imagery runs 4/8/12 only.",
        "- OpenBMI: left/right hand imagery and paired post-imagery rest.",
        "- Liu: left/right hand imagery and stage-3 break/rest; instruction is excluded.",
        "",
        "## Interpretation Boundary",
        "",
        "Healthy/stroke differences remain confounded with equipment, referencing, "
        "cue timing, and protocol. The feature comparison is therefore exploratory. "
        "A causal physiological comparison requires prospective healthy controls and "
        "stroke participants recorded with identical hardware and synchronized cues.",
        "",
        "## Figures",
        "",
        "- `01_healthy_movement_imagery_reference.png`: hand movement/imagery reference.",
        "- `02_healthy_cross_dataset_generalization.png`: healthy final-subject results.",
        "- `03_stroke_transfer_ladder.png`: zero-shot through patient-adapted results.",
        "- `04_stroke_safety_tradeoff.png`: false activation versus detected intent.",
        "- `05_healthy_stroke_feature_shift.png`: exploratory feature distributions.",
        "- `06_stroke_patient_variability.png`: final-patient variability and NIHSS.",
        "",
        "Machine-readable results are in `healthy_to_stroke_metrics.json`; final "
        "stroke predictions, physiology summaries, and patient metrics are CSV files.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    configure_plots()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Healthy-to-Stroke Weak Hand-Intent Study ===")
    print("[+] Loading hand-only healthy datasets...")
    physionet = from_harmonized_healthy(
        build_physionet_harmonized_dataset(
            args.physionet_dir,
            Path(args.physionet_cache),
            rebuild_cache=args.rebuild_physionet_cache,
        )
    )
    openbmi = from_harmonized_healthy(
        load_openbmi_harmonized_dataset(Path(args.openbmi_cache))
    )
    liu_subjects = None
    if args.max_liu_subjects is not None:
        liu_subjects = list(range(1, args.max_liu_subjects + 1))
    liu = build_liu_hand_intent_dataset(
        args.liu_dir,
        output_dir / "liu_hand_intent_epochs.npz",
        subjects=liu_subjects,
        rebuild_cache=args.rebuild_liu_cache,
    )
    combined = combine_hand_intent_datasets(
        (physionet, openbmi, liu),
        subject_offsets=(PHYSIONET_OFFSET, OPENBMI_OFFSET, LIU_OFFSET),
    )
    print(
        f"[+] Rows: PhysioNet {len(physionet.labels):,}, "
        f"OpenBMI {len(openbmi.labels):,}, Liu {len(liu.labels):,}"
    )

    partitions = {
        "physionet": three_way_subject_split(
            np.unique(physionet.subjects),
            random_seed=args.seed,
        ),
        "openbmi": three_way_subject_split(
            np.unique(openbmi.subjects),
            random_seed=args.seed,
        ),
        "liu2024": three_way_subject_split(
            np.unique(liu.subjects),
            random_seed=args.seed,
        ),
    }
    for name, partition in partitions.items():
        save_partitions(partition, output_dir / f"{name}_subject_partitions.json")
    masks = build_masks(combined, partitions)
    validate_masks(combined.labels, masks)

    covariance_parts = []
    for dataset in (physionet, openbmi, liu):
        covariance_parts.append(
            build_covariance_cache(
                dataset.covariance_interface(),
                output_dir / f"{dataset.dataset_name}_covariances.npz",
                rebuild_cache=args.rebuild_covariances,
            ).covariances
        )
    covariances = np.concatenate(covariance_parts)

    print("[+] Phase 2: fitting the healthy population model...")
    strict_features, strict_metadata = tangent_features(
        covariances,
        masks["healthy_development"],
        combined.subjects,
        combined.sessions,
        combined.recordings,
        alignment="none",
    )
    aligned_features, aligned_metadata = tangent_features(
        covariances,
        masks["healthy_development"],
        combined.subjects,
        combined.sessions,
        combined.recordings,
        alignment="log_euclidean",
    )
    strict_model, strict_probabilities, strict_weights = fit_decoder(
        strict_features,
        combined.labels,
        masks["healthy_development"],
        combined.dataset_ids,
        seed=args.seed,
    )
    aligned_model, aligned_probabilities, aligned_weights = fit_decoder(
        aligned_features,
        combined.labels,
        masks["healthy_development"],
        combined.dataset_ids,
        seed=args.seed,
    )
    strict_results, strict_balanced, strict_safety = policies_and_scores(
        combined.labels,
        strict_probabilities,
        masks["healthy_controller"],
        {
            "physionet_final": masks["physionet_final"],
            "openbmi_final": masks["openbmi_final"],
            "healthy_final": masks["healthy_final"],
            "liu_final": masks["liu2024_final"],
        },
        max_false_activation=args.max_false_activation,
    )
    aligned_results, aligned_balanced, aligned_safety = policies_and_scores(
        combined.labels,
        aligned_probabilities,
        masks["healthy_controller"],
        {
            "physionet_final": masks["physionet_final"],
            "openbmi_final": masks["openbmi_final"],
            "healthy_final": masks["healthy_final"],
            "liu_final": masks["liu2024_final"],
        },
        max_false_activation=args.max_false_activation,
    )

    print("[+] Phase 4: fitting population-level stroke adaptation...")
    mixed_features, mixed_metadata = tangent_features(
        covariances,
        masks["mixed_development"],
        combined.subjects,
        combined.sessions,
        combined.recordings,
        alignment="log_euclidean",
    )
    mixed_model, mixed_probabilities, mixed_weights = fit_decoder(
        mixed_features,
        combined.labels,
        masks["mixed_development"],
        combined.dataset_ids,
        seed=args.seed,
    )
    mixed_results, mixed_balanced, mixed_safety = policies_and_scores(
        combined.labels,
        mixed_probabilities,
        masks["liu2024_controller"],
        {
            "liu_final": masks["liu2024_final"],
            "physionet_final": masks["physionet_final"],
            "openbmi_final": masks["openbmi_final"],
        },
        max_false_activation=args.max_false_activation,
    )
    calibration_report, calibrated_probabilities, calibrated_policy = (
        select_patient_calibration(
            mixed_features,
            combined,
            masks,
            mixed_model,
            max_false_activation=args.max_false_activation,
            seed=args.seed,
        )
    )

    methods = {
        "strict_zero_shot": (strict_probabilities, strict_safety),
        "aligned_zero_shot": (aligned_probabilities, aligned_safety),
        "mixed_population": (mixed_probabilities, mixed_safety),
        "patient_calibrated": (
            calibrated_probabilities,
            calibrated_policy,
        ),
    }
    patient_rows = per_patient_rows(
        combined,
        masks["liu2024_final"],
        methods,
    )
    physiology = physiology_rows(
        combined,
        masks["healthy_final"] | masks["liu2024_final"],
        aligned_features,
        masks["healthy_development"],
    )

    for metrics, probabilities, policy in (
        (
            strict_results["liu_final_safety"],
            strict_probabilities,
            strict_safety,
        ),
        (
            aligned_results["liu_final_safety"],
            aligned_probabilities,
            aligned_safety,
        ),
        (
            mixed_results["liu_final_safety"],
            mixed_probabilities,
            mixed_safety,
        ),
        (
            calibration_report["final"],
            calibrated_probabilities,
            calibrated_policy,
        ),
    ):
        metrics["subject_bootstrap_95_ci"] = subject_bootstrap_interval(
            combined.labels[masks["liu2024_final"]],
            subset_probabilities(probabilities, masks["liu2024_final"]),
            combined.subjects[masks["liu2024_final"]],
            policy,
            repetitions=args.bootstrap_repetitions,
            random_seed=args.seed,
        )

    healthy_reference = load_healthy_reference(Path(args.healthy_reference))
    elapsed = (time.time() - started) / 60.0
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "healthy_to_stroke_weak_hand_intent",
        "primary_endpoint": "binary hand motor-intent versus rest",
        "secondary_endpoint": "conditional left-versus-right hand imagery",
        "excluded_tasks": "PhysioNet bilateral fist/foot runs 5/6/9/10/13/14",
        "harmonization": {
            "channels": list(CLINICAL_CHANNELS),
            "sampling_hz": CLINICAL_SFREQ,
            "bandpass_hz": [4.0, 38.0],
            "reference": "full-EEG common average",
            "epoch_duration_seconds": CLINICAL_EPOCH_DURATION_SECONDS,
            "representation": "five-band regularized covariance tangent space",
            "decoder": "hierarchical regularized logistic regression",
            "max_controller_false_activation": args.max_false_activation,
        },
        "datasets": {
            dataset.dataset_name: {
                "cohort": dataset.cohort,
                "subjects": int(len(np.unique(dataset.subjects))),
                "rows": int(len(dataset.labels)),
                "class_counts": {
                    str(label): int(np.sum(dataset.labels == label))
                    for label in (0, 1, 2)
                },
                "fingerprint": dataset.fingerprint,
            }
            for dataset in (physionet, openbmi, liu)
        },
        "partitions": {
            name: asdict(partition) for name, partition in partitions.items()
        },
        "healthy_reference": healthy_reference,
        "healthy_model": {
            "strict_unaligned": strict_results,
            "label_free_aligned": aligned_results,
        },
        "zero_shot": {
            "strict_unaligned": strict_results,
            "label_free_aligned": aligned_results,
        },
        "stroke_adaptation": {
            "mixed_population": mixed_results,
            "patient_calibration": calibration_report,
        },
        "feature_metadata": {
            "strict_unaligned": strict_metadata,
            "healthy_label_free_aligned": aligned_metadata,
            "mixed_population": mixed_metadata,
        },
        "training_weight_totals": {
            "strict_healthy": {
                name: float(
                    np.sum(
                        strict_weights[
                            masks["healthy_development"]
                            & (combined.dataset_ids == name)
                        ]
                    )
                )
                for name in ("physionet", "openbmi")
            },
            "aligned_healthy": {
                name: float(
                    np.sum(
                        aligned_weights[
                            masks["healthy_development"]
                            & (combined.dataset_ids == name)
                        ]
                    )
                )
                for name in ("physionet", "openbmi")
            },
            "mixed": {
                name: float(
                    np.sum(
                        mixed_weights[
                            masks["mixed_development"]
                            & (combined.dataset_ids == name)
                        ]
                    )
                )
                for name in ("physionet", "openbmi", "liu2024")
            },
        },
        "physiology_subject_rows": physiology,
        "stroke_final_patient_rows": patient_rows,
        "runtime_minutes": elapsed,
    }

    metrics_path = output_dir / "healthy_to_stroke_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    joblib.dump(
        {
            "strict_healthy_model": strict_model,
            "aligned_healthy_model": aligned_model,
            "mixed_population_model": mixed_model,
            "policies": {
                "strict_balanced": strict_balanced,
                "strict_safety": strict_safety,
                "aligned_balanced": aligned_balanced,
                "aligned_safety": aligned_safety,
                "mixed_balanced": mixed_balanced,
                "mixed_safety": mixed_safety,
                "patient_calibrated": calibrated_policy,
            },
            "partitions": partitions,
            "fingerprint": combined.fingerprint,
        },
        output_dir / "healthy_to_stroke_models.joblib",
    )
    write_csv(output_dir / "healthy_stroke_physiology.csv", physiology)
    write_csv(output_dir / "stroke_final_patient_metrics.csv", patient_rows)
    write_predictions(
        output_dir / "stroke_final_predictions.csv",
        combined,
        masks["liu2024_final"],
        methods,
    )
    plot_healthy_reference(
        output_dir / "01_healthy_movement_imagery_reference.png",
        healthy_reference,
    )
    plot_healthy_generalization(
        output_dir / "02_healthy_cross_dataset_generalization.png",
        aligned_results,
    )
    plot_stroke_ladder(
        output_dir / "03_stroke_transfer_ladder.png",
        report,
    )
    plot_safety_tradeoff(
        output_dir / "04_stroke_safety_tradeoff.png",
        report,
    )
    plot_feature_shift(
        output_dir / "05_healthy_stroke_feature_shift.png",
        physiology,
    )
    plot_patient_variability(
        output_dir / "06_stroke_patient_variability.png",
        patient_rows,
    )
    write_report(output_dir / "HEALTHY_TO_STROKE_REPORT.md", report)

    print("\n=== Locked Liu Final-Patient Results ===")
    for label, metrics, _ in stroke_ladder(report):
        print(
            f"{label.replace(chr(10), ' '):38s} | "
            f"intent BA={100 * metrics['intent_balanced_accuracy']:5.1f}% | "
            f"sensitivity={100 * metrics['intent_sensitivity']:5.1f}% | "
            f"specificity={100 * metrics['intent_specificity']:5.1f}% | "
            f"FA={100 * metrics['false_activation_rate']:5.1f}% | "
            f"missed={100 * metrics['missed_intent_rate']:5.1f}%"
        )
    print(f"[+] Saved report, models, CSV data, and six figures to {output_dir}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
