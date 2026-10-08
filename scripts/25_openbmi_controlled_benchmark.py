#!/usr/bin/env python3
"""Run a subject-isolated OpenBMI reproduction and safety benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.dataset import LABEL_NAMES, parse_int_list  # noqa: E402
from bci_pipeline.models.eegnet import (  # noqa: E402
    EEGNetTrainingConfig,
    HierarchicalEEGNet,
)
from bci_pipeline.models.fbcsp import BinaryFBCSP  # noqa: E402
from bci_pipeline.openbmi import (  # noqa: E402
    available_openbmi_subjects,
    resolve_openbmi_root,
)
from bci_pipeline.openbmi_benchmark import (  # noqa: E402
    OpenBMIBenchmarkConfig,
    OpenBMIBenchmarkDataset,
    build_openbmi_benchmark_dataset,
)
from bci_pipeline.openbmi_controlled import (  # noqa: E402
    CommandPolicy,
    HierarchicalCSPDecoder,
    HierarchicalFeatureDecoder,
    StageProbabilities,
    build_covariance_cache,
    evaluate_policy,
    locally_adapted_probabilities,
    select_balanced_policy,
    select_safety_policy,
    subject_bootstrap_interval,
    tangent_features,
)
from bci_pipeline.representations.riemann import FILTER_BANK  # noqa: E402
from bci_pipeline.splits import (  # noqa: E402
    SubjectPartitions,
    save_partitions,
    three_way_subject_split,
)


METHOD_NAMES = {
    "paper_csp_lda": "CSP + LDA",
    "fbcsp_lr": "FBCSP + LR",
    "tangent_unaligned": "Tangent (unaligned)",
    "tangent_euclidean": "Tangent + EA",
    "tangent_log_euclidean": "Tangent + log-EA",
    "eegnet": "EEGNet",
}
COLORS = {
    "paper_csp_lda": "#184E77",
    "fbcsp_lr": "#1E7A8A",
    "tangent_unaligned": "#6C757D",
    "tangent_euclidean": "#2D6A4F",
    "tangent_log_euclidean": "#52A675",
    "eegnet": "#D66A3A",
    "adapted": "#E9A23B",
    "balanced": "#227C9D",
    "safety": "#D95D39",
}


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
        "--data-dir",
        default="files/external/openbmi/nm000338",
        help="OpenBMI nm000338 BIDS root.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/openbmi_controlled",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma/range selection, for example 4-9,32-54.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-covariances", action="store_true")
    parser.add_argument("--skip-eegnet", action="store_true")
    parser.add_argument("--eegnet-epochs", type=int, default=25)
    parser.add_argument("--eegnet-device", default="auto")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def locked_masks(
    dataset: OpenBMIBenchmarkDataset,
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
        "controller_calibration": np.isin(
            dataset.subjects,
            partitions.controller_validation,
        )
        & (dataset.recordings == 0),
        "final_calibration": np.isin(
            dataset.subjects,
            partitions.final_target,
        )
        & (dataset.recordings == 0),
    }


def validate_masks(
    labels: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> None:
    for name in ("development", "controller", "final"):
        if not np.any(masks[name]) or len(np.unique(labels[masks[name]])) != 3:
            raise ValueError(f"{name} split is empty or missing a class.")


def subset_probabilities(
    probabilities: StageProbabilities,
    mask: np.ndarray,
) -> StageProbabilities:
    return StageProbabilities(
        intent=probabilities.intent[mask],
        direction=probabilities.direction[mask],
    )


def fit_raw_models(
    dataset: OpenBMIBenchmarkDataset,
    masks: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[dict[str, StageProbabilities], dict[str, object]]:
    probabilities: dict[str, StageProbabilities] = {}
    models: dict[str, object] = {}
    raw_specs = {
        "paper_csp_lda": {
            "filter_bank": (("mu_beta", 8.0, 30.0),),
            "components_per_band": 4,
            "classifier": "lda",
        },
        "fbcsp_lr": {
            "filter_bank": FILTER_BANK,
            "components_per_band": 4,
            "classifier": "logistic",
        },
    }
    for name, spec in raw_specs.items():
        print(f"[+] Fitting {METHOD_NAMES[name]}...")
        model = HierarchicalCSPDecoder(
            dataset.sfreq,
            random_seed=seed,
            **spec,
        ).fit(
            dataset.epochs[masks["development"]],
            dataset.labels[masks["development"]],
        )
        probabilities[name] = model.predict_stage_probabilities(dataset.epochs)
        models[name] = model
    return probabilities, models


def fit_covariance_models(
    dataset: OpenBMIBenchmarkDataset,
    masks: Mapping[str, np.ndarray],
    covariance_path: Path,
    *,
    seed: int,
    rebuild: bool,
) -> tuple[
    dict[str, StageProbabilities],
    dict[str, object],
    dict[str, np.ndarray],
    dict[str, object],
]:
    covariance_cache = build_covariance_cache(
        dataset,
        covariance_path,
        rebuild_cache=rebuild,
    )
    probabilities: dict[str, StageProbabilities] = {}
    models: dict[str, object] = {}
    feature_matrices: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}
    for alignment, suffix in (
        ("none", "unaligned"),
        ("euclidean", "euclidean"),
        ("log_euclidean", "log_euclidean"),
    ):
        name = f"tangent_{suffix}"
        print(f"[+] Fitting {METHOD_NAMES[name]}...")
        features, feature_metadata = tangent_features(
            covariance_cache.covariances,
            masks["development"],
            dataset.subjects,
            dataset.sessions,
            dataset.recordings,
            alignment=alignment,
        )
        model = HierarchicalFeatureDecoder(seed).fit(
            features[masks["development"]],
            dataset.labels[masks["development"]],
        )
        probabilities[name] = model.predict_stage_probabilities(features)
        models[name] = model
        feature_matrices[name] = features
        metadata[name] = feature_metadata
    return probabilities, models, feature_matrices, metadata


def fit_eegnet(
    dataset: OpenBMIBenchmarkDataset,
    masks: Mapping[str, np.ndarray],
    *,
    epochs: int,
    device: str,
    seed: int,
) -> tuple[StageProbabilities, HierarchicalEEGNet]:
    print("[+] Fitting compact hierarchical EEGNet...")
    model = HierarchicalEEGNet(
        dataset.epochs.shape[1],
        EEGNetTrainingConfig(
            epochs=epochs,
            batch_size=256,
            patience=6,
            random_seed=seed,
            device=device,
        ),
    ).fit(
        dataset.epochs[masks["development"]],
        dataset.labels[masks["development"]],
        dataset.epochs[masks["controller"]],
        dataset.labels[masks["controller"]],
    )
    intent, direction = model.predict_stage_probabilities(dataset.epochs)
    return StageProbabilities(intent=intent, direction=direction), model


def evaluate_methods(
    dataset: OpenBMIBenchmarkDataset,
    masks: Mapping[str, np.ndarray],
    probabilities: Mapping[str, StageProbabilities],
    *,
    max_false_activation: float,
) -> tuple[dict[str, object], str]:
    results: dict[str, object] = {}
    for name, method_probabilities in probabilities.items():
        validation = subset_probabilities(method_probabilities, masks["controller"])
        final = subset_probabilities(method_probabilities, masks["final"])
        balanced_policy = select_balanced_policy(
            dataset.labels[masks["controller"]],
            validation,
        )
        safety_policy = select_safety_policy(
            dataset.labels[masks["controller"]],
            validation,
            max_false_activation,
        )
        results[name] = {
            "display_name": METHOD_NAMES[name],
            "balanced_policy": asdict(balanced_policy),
            "safety_policy": asdict(safety_policy),
            "controller_balanced": evaluate_policy(
                dataset.labels[masks["controller"]],
                validation,
                balanced_policy,
            ),
            "controller_safety": evaluate_policy(
                dataset.labels[masks["controller"]],
                validation,
                safety_policy,
            ),
            "final_balanced": evaluate_policy(
                dataset.labels[masks["final"]],
                final,
                balanced_policy,
            ),
            "final_safety": evaluate_policy(
                dataset.labels[masks["final"]],
                final,
                safety_policy,
            ),
        }
    selected = max(
        results,
        key=lambda name: (
            float(results[name]["controller_safety"]["intent_balanced_accuracy"]),
            float(results[name]["controller_safety"]["balanced_accuracy"]),
            -float(results[name]["controller_safety"]["missed_intent_rate"]),
        ),
    )
    return results, selected


def cue_leakage_audit(
    dataset: OpenBMIBenchmarkDataset,
    partitions: SubjectPartitions,
) -> dict[str, object]:
    train = np.isin(dataset.cue_subjects, partitions.development) & (
        dataset.cue_recordings == 0
    )
    controller = np.isin(
        dataset.cue_subjects,
        partitions.controller_validation,
    ) & (dataset.cue_recordings == 1)
    final = np.isin(dataset.cue_subjects, partitions.final_target) & (
        dataset.cue_recordings == 1
    )
    rows: dict[str, object] = {}
    for name, features in dataset.cue_features.items():
        model = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ).fit(features[train], dataset.cue_labels[train])
        rows[name] = {
            "controller_direction_balanced_accuracy": float(
                balanced_accuracy_score(
                    dataset.cue_labels[controller],
                    model.predict(features[controller]),
                )
            ),
            "final_direction_balanced_accuracy": float(
                balanced_accuracy_score(
                    dataset.cue_labels[final],
                    model.predict(features[final]),
                )
            ),
            "feature_count": int(features.shape[1]),
        }
    return rows


def subject_specific_reproduction(
    dataset: OpenBMIBenchmarkDataset,
) -> dict[str, object]:
    """Reproduce the easier within-subject offline-to-online direction task."""
    rows: list[dict[str, object]] = []
    for subject in np.unique(dataset.subjects):
        for session in np.unique(dataset.sessions[dataset.subjects == subject]):
            train = (
                (dataset.subjects == subject)
                & (dataset.sessions == session)
                & (dataset.recordings == 0)
                & (dataset.labels != 0)
            )
            test = (
                (dataset.subjects == subject)
                & (dataset.sessions == session)
                & (dataset.recordings == 1)
                & (dataset.labels != 0)
            )
            csp = BinaryFBCSP(
                dataset.sfreq,
                components_per_band=4,
                filter_bank=(("mu_beta", 8.0, 30.0),),
            )
            train_features = csp.fit_transform(
                dataset.epochs[train],
                dataset.labels[train],
            )
            model = make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ).fit(train_features, dataset.labels[train])
            score = balanced_accuracy_score(
                dataset.labels[test],
                model.predict(csp.transform(dataset.epochs[test])),
            )
            rows.append(
                {
                    "subject": int(subject),
                    "session": int(session),
                    "direction_balanced_accuracy": float(score),
                }
            )
    values = np.asarray(
        [row["direction_balanced_accuracy"] for row in rows],
        dtype=float,
    )
    return {
        "protocol": "same-subject train recording to test recording; intent trials only",
        "mean_direction_balanced_accuracy": float(np.mean(values)),
        "median_direction_balanced_accuracy": float(np.median(values)),
        "standard_deviation": float(np.std(values)),
        "rows": rows,
    }


def calibration_experiment(
    dataset: OpenBMIBenchmarkDataset,
    partitions: SubjectPartitions,
    masks: Mapping[str, np.ndarray],
    features: np.ndarray,
    global_decoder: HierarchicalFeatureDecoder,
    *,
    max_false_activation: float,
    seed: int,
) -> tuple[dict[str, object], StageProbabilities, CommandPolicy, CommandPolicy]:
    controller_subject_mask = np.isin(
        dataset.subjects,
        partitions.controller_validation,
    )
    final_subject_mask = np.isin(dataset.subjects, partitions.final_target)
    configurations: list[dict[str, object]] = []
    probability_cache: dict[tuple[int, float], StageProbabilities] = {}
    for budget in (5, 10, 20):
        for blend in (0.25, 0.5, 0.75, 1.0):
            adapted = locally_adapted_probabilities(
                features,
                dataset.labels,
                dataset.subjects,
                dataset.recordings,
                controller_subject_mask,
                global_decoder,
                budget_per_class=budget,
                blend_weight=blend,
                random_seed=seed,
            )
            probability_cache[(budget, blend)] = adapted
            validation = subset_probabilities(adapted, masks["controller"])
            safety_policy = select_safety_policy(
                dataset.labels[masks["controller"]],
                validation,
                max_false_activation,
            )
            metrics = evaluate_policy(
                dataset.labels[masks["controller"]],
                validation,
                safety_policy,
            )
            configurations.append(
                {
                    "budget_per_class": budget,
                    "blend_weight": blend,
                    "safety_policy": asdict(safety_policy),
                    "controller_safety": metrics,
                }
            )
    best = max(
        configurations,
        key=lambda row: (
            float(row["controller_safety"]["intent_balanced_accuracy"]),
            float(row["controller_safety"]["balanced_accuracy"]),
            -float(row["controller_safety"]["missed_intent_rate"]),
        ),
    )
    budget = int(best["budget_per_class"])
    blend = float(best["blend_weight"])
    final_adapted = locally_adapted_probabilities(
        features,
        dataset.labels,
        dataset.subjects,
        dataset.recordings,
        final_subject_mask,
        global_decoder,
        budget_per_class=budget,
        blend_weight=blend,
        random_seed=seed,
    )
    controller_adapted = probability_cache[(budget, blend)]
    controller_subset = subset_probabilities(
        controller_adapted,
        masks["controller"],
    )
    balanced_policy = select_balanced_policy(
        dataset.labels[masks["controller"]],
        controller_subset,
    )
    safety_policy = select_safety_policy(
        dataset.labels[masks["controller"]],
        controller_subset,
        max_false_activation,
    )
    final_subset = subset_probabilities(final_adapted, masks["final"])
    report = {
        "selected_on_controller_subjects": best,
        "all_controller_configurations": configurations,
        "final_balanced": evaluate_policy(
            dataset.labels[masks["final"]],
            final_subset,
            balanced_policy,
        ),
        "final_safety": evaluate_policy(
            dataset.labels[masks["final"]],
            final_subset,
            safety_policy,
        ),
        "balanced_policy": asdict(balanced_policy),
        "safety_policy": asdict(safety_policy),
    }
    return report, final_adapted, balanced_policy, safety_policy


def plot_method_comparison(
    output_dir: Path,
    results: Mapping[str, object],
    selected: str,
) -> None:
    names = list(results)
    positions = np.arange(len(names))
    width = 0.24
    metrics = (
        ("balanced_accuracy", "3-Class BA"),
        ("intent_balanced_accuracy", "Intent BA"),
        ("direction_balanced_accuracy", "Direction BA"),
    )
    metric_colors = ("#184E77", "#52A675", "#E9A23B")
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for offset, ((field, label), color) in enumerate(zip(metrics, metric_colors)):
        values = [
            100.0 * float(results[name]["final_balanced"][field])
            for name in names
        ]
        axis.bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axis.axhline(
        50,
        color="#5D6770",
        linestyle="--",
        linewidth=1,
        label="Binary chance (50%)",
    )
    axis.axhline(
        100 / 3,
        color="#8D99AE",
        linestyle=":",
        linewidth=1,
        label="3-class chance (33.3%)",
    )
    axis.set_xticks(
        positions,
        [
            METHOD_NAMES[name] + ("\n(selected)" if name == selected else "")
            for name in names
        ],
    )
    axis.set_ylabel("Balanced Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Unseen-Subject OpenBMI Decoder Comparison")
    axis.legend(frameon=False, ncol=5, loc="upper center")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "01_decoder_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_safety_tradeoff(
    output_dir: Path,
    results: Mapping[str, object],
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 6.5))
    annotation_offsets = {
        "paper_csp_lda": (5, 4),
        "fbcsp_lr": (5, 4),
        "tangent_unaligned": (5, 4),
        "tangent_euclidean": (5, 8),
        "tangent_log_euclidean": (5, -12),
        "eegnet": (5, 4),
    }
    for name, row in results.items():
        for policy_name, marker in (("balanced", "o"), ("safety", "s")):
            metrics = row[f"final_{policy_name}"]
            axis.scatter(
                100.0 * float(metrics["false_activation_rate"]),
                100.0 * (1.0 - float(metrics["missed_intent_rate"])),
                s=75,
                marker=marker,
                color=COLORS[name],
                edgecolor="white",
                linewidth=0.8,
            )
            if policy_name == "safety":
                axis.annotate(
                    METHOD_NAMES[name],
                    (
                        100.0 * float(metrics["false_activation_rate"]),
                        100.0 * (1.0 - float(metrics["missed_intent_rate"])),
                    ),
                    xytext=annotation_offsets[name],
                    textcoords="offset points",
                    fontsize=8,
                )
    axis.set_xlabel("False Activation Rate on Rest Windows (%)")
    axis.set_ylabel("Intent Sensitivity (%)")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 100)
    axis.set_title("Safety Trade-off on Untouched Subjects")
    axis.grid(alpha=0.25)
    axis.text(
        0.99,
        0.02,
        "Circles: balanced policy   Squares: safety policy",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "02_safety_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusions(
    output_dir: Path,
    selected_result: Mapping[str, object],
    selected_name: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for axis, policy_name in zip(axes, ("balanced", "safety")):
        matrix = np.asarray(
            selected_result[f"final_{policy_name}"]["confusion_matrix"],
            dtype=float,
        )
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}\n{normalized[row, column] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if normalized[row, column] > 0.55 else "#102A43",
                    fontsize=9,
                )
        axis.set_xticks(range(3), ("Rest", "Left", "Right"))
        axis.set_yticks(range(3), ("Rest", "Left", "Right"))
        axis.set_xlabel("Predicted Command")
        axis.set_ylabel("True State")
        axis.set_title(f"{policy_name.title()} Policy")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04, label="Row Fraction")
    fig.suptitle(f"{METHOD_NAMES[selected_name]}: Untouched-Subject Confusions", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.14, top=0.83, wspace=0.3)
    fig.savefig(output_dir / "03_selected_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def plot_cue_audit(output_dir: Path, cue_results: Mapping[str, object]) -> None:
    display = {
        "pre_cue_occipital": "Pre-cue\noccipital",
        "early_cue_occipital": "0-0.5 s\noccipital",
        "early_cue_motor": "0-0.5 s\nmotor",
        "late_occipital": "1-3.5 s\noccipital",
        "late_motor": "1-3.5 s\nmotor",
    }
    names = list(display)
    values = [
        100.0 * float(cue_results[name]["final_direction_balanced_accuracy"])
        for name in names
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(
        np.arange(len(names)),
        values,
        color=("#6C757D", "#D66A3A", "#E9A23B", "#B56576", "#2D6A4F"),
    )
    axis.axhline(50, color="#102A43", linestyle="--", linewidth=1.2, label="Chance")
    axis.set_xticks(np.arange(len(names)), [display[name] for name in names])
    axis.set_ylabel("Left-vs-Right Balanced Accuracy (%)")
    axis.set_ylim(35, max(75, max(values) + 8))
    axis.set_title("Visual-Cue Leakage Control on Untouched Subjects")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "04_visual_cue_control.png", bbox_inches="tight")
    plt.close(fig)


def per_subject_metrics(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    subjects: np.ndarray,
    policy: CommandPolicy,
) -> list[dict[str, object]]:
    rows = []
    for subject in np.unique(subjects):
        selected = subjects == subject
        metrics = evaluate_policy(
            labels[selected],
            subset_probabilities(probabilities, selected),
            policy,
        )
        rows.append({"subject": int(subject), **metrics})
    return rows


def plot_subject_variability(
    output_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    rows = sorted(rows, key=lambda row: int(row["subject"]))
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        positions - 0.18,
        [100 * float(row["false_activation_rate"]) for row in rows],
        0.36,
        color=COLORS["safety"],
        label="False activation",
    )
    axis.bar(
        positions + 0.18,
        [100 * float(row["missed_intent_rate"]) for row in rows],
        0.36,
        color="#E9A23B",
        label="Missed intent",
    )
    axis.set_xticks(positions, [f"S{int(row['subject']):02d}" for row in rows])
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Safety-Policy Variability Across Untouched Subjects")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "05_subject_variability.png", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    output_dir: Path,
    calibration: Mapping[str, object],
    global_safety: Mapping[str, object],
) -> None:
    rows = calibration["all_controller_configurations"]
    budgets = sorted({int(row["budget_per_class"]) for row in rows})
    best_by_budget = []
    for budget in budgets:
        candidates = [row for row in rows if int(row["budget_per_class"]) == budget]
        best_by_budget.append(
            max(
                candidates,
                key=lambda row: float(
                    row["controller_safety"]["intent_balanced_accuracy"]
                ),
            )
        )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [0] + budgets,
        [100 * float(global_safety["intent_balanced_accuracy"])]
        + [
            100 * float(row["controller_safety"]["intent_balanced_accuracy"])
            for row in best_by_budget
        ],
        marker="o",
        linewidth=2.2,
        color=COLORS["adapted"],
        label="Intent balanced accuracy",
    )
    axis.plot(
        [0] + budgets,
        [100 * float(global_safety["false_activation_rate"])]
        + [
            100 * float(row["controller_safety"]["false_activation_rate"])
            for row in best_by_budget
        ],
        marker="s",
        linewidth=2.2,
        color=COLORS["safety"],
        label="False activation",
    )
    axis.plot(
        [0] + budgets,
        [100 * float(global_safety["missed_intent_rate"])]
        + [
            100 * float(row["controller_safety"]["missed_intent_rate"])
            for row in best_by_budget
        ],
        marker="^",
        linewidth=2.2,
        color="#5D6770",
        label="Missed intent",
    )
    axis.set_xlabel("Calibration Trials per Class")
    axis.set_ylabel("Rate (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Controller-Subject Calibration Ablation")
    axis.legend(frameon=False, ncol=3)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "06_calibration_ablation.png", bbox_inches="tight")
    plt.close(fig)


def plot_reproduction(
    output_dir: Path,
    reproduction: Mapping[str, object],
) -> None:
    rows = reproduction["rows"]
    subjects = sorted({int(row["subject"]) for row in rows})
    session_one = {
        int(row["subject"]): 100 * float(row["direction_balanced_accuracy"])
        for row in rows
        if int(row["session"]) == 1
    }
    session_two = {
        int(row["subject"]): 100 * float(row["direction_balanced_accuracy"])
        for row in rows
        if int(row["session"]) == 2
    }
    positions = np.arange(len(subjects))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        positions,
        [session_one[subject] for subject in subjects],
        marker="o",
        linewidth=1.4,
        color="#184E77",
        label="Session 1",
    )
    axis.plot(
        positions,
        [session_two[subject] for subject in subjects],
        marker="s",
        linewidth=1.4,
        color="#52A675",
        label="Session 2",
    )
    axis.axhline(50, color="#5D6770", linestyle="--", linewidth=1, label="Chance")
    axis.axhline(
        100 * float(reproduction["mean_direction_balanced_accuracy"]),
        color="#D66A3A",
        linestyle="-.",
        linewidth=1.4,
        label="Available-cohort mean",
    )
    axis.set_xticks(positions, [f"S{subject:02d}" for subject in subjects], rotation=45)
    axis.set_ylabel("Direction Balanced Accuracy (%)")
    axis.set_ylim(20, 100)
    axis.set_title("Within-Subject CSP/LDA Reproduction")
    axis.legend(frameon=False, ncol=4)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "07_within_subject_reproduction.png", bbox_inches="tight")
    plt.close(fig)


def write_predictions(
    path: Path,
    dataset: OpenBMIBenchmarkDataset,
    mask: np.ndarray,
    probabilities: StageProbabilities,
    balanced_policy: CommandPolicy,
    safety_policy: CommandPolicy,
) -> None:
    direction = np.argmax(probabilities.direction[mask], axis=1) + 1
    direction_confidence = np.max(probabilities.direction[mask], axis=1)
    rows = []
    for index, dataset_index in enumerate(np.flatnonzero(mask)):
        row = {
            "subject": int(dataset.subjects[dataset_index]),
            "session": int(dataset.sessions[dataset_index]),
            "recording": "test",
            "event_index": int(dataset.event_indices[dataset_index]),
            "true_label": LABEL_NAMES[int(dataset.labels[dataset_index])],
            "intent_probability": float(probabilities.intent[dataset_index]),
            "left_probability": float(probabilities.direction[dataset_index, 0]),
            "right_probability": float(probabilities.direction[dataset_index, 1]),
        }
        for name, policy in (
            ("balanced", balanced_policy),
            ("safety", safety_policy),
        ):
            accepted = (
                probabilities.intent[dataset_index] >= policy.intent_threshold
                and direction_confidence[index]
                >= 0.5 + policy.direction_margin / 2.0
            )
            row[f"{name}_prediction"] = LABEL_NAMES[
                int(direction[index]) if accepted else 0
            ]
        rows.append(row)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: Mapping[str, object]) -> None:
    selected = str(report["selected_method"])
    result = report["methods"][selected]
    calibration = report["calibration"]
    cue = report["cue_leakage_audit"]
    reproduction = report["within_subject_reproduction"]
    final_safety = result["final_safety"]
    lines = [
        "# Controlled OpenBMI Motor-Intent Benchmark",
        "",
        "## Protocol",
        "",
        (
            f"- Available complete cohort: **{report['dataset']['subjects']} subjects** "
            f"({report['dataset']['rows']:,} paired intent/rest windows)."
        ),
        (
            f"- Subject isolation: **{report['partitions']['development_count']} development**, "
            f"**{report['partitions']['controller_count']} controller-validation**, and "
            f"**{report['partitions']['final_count']} untouched final subjects**."
        ),
        "- Motor window: 1.0-3.5 s after cue; paired rest window: 5.0-7.5 s.",
        "- Target alignment uses each subject/session training recording only.",
        "",
        "## Locked Result",
        "",
        (
            f"Controller-selected decoder: **{METHOD_NAMES[selected]}**. On untouched "
            f"subjects, its safety policy produced **{100 * float(final_safety['intent_balanced_accuracy']):.1f}% "
            f"intent balanced accuracy**, **{100 * float(final_safety['false_activation_rate']):.1f}% "
            f"false activation**, and **{100 * float(final_safety['missed_intent_rate']):.1f}% "
            "missed intent**."
        ),
        "",
        "| Decoder | 3-class BA | Intent BA | Direction BA | False activation | Missed intent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["methods"].items():
        metrics = row["final_safety"]
        lines.append(
            f"| {METHOD_NAMES[name]} | {100 * float(metrics['balanced_accuracy']):.1f}% "
            f"| {100 * float(metrics['intent_balanced_accuracy']):.1f}% "
            f"| {100 * float(metrics['direction_balanced_accuracy']):.1f}% "
            f"| {100 * float(metrics['false_activation_rate']):.1f}% "
            f"| {100 * float(metrics['missed_intent_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Controls And Calibration",
            "",
            (
                f"- Pre-cue occipital left/right decoding was "
                f"**{100 * float(cue['pre_cue_occipital']['final_direction_balanced_accuracy']):.1f}%**; "
                f"early-cue occipital decoding was "
                f"**{100 * float(cue['early_cue_occipital']['final_direction_balanced_accuracy']):.1f}%**. "
                "Values materially above 50% indicate visual-cue information that must not be "
                "mistaken for motor imagery."
            ),
            (
                f"- The easier within-subject CSP/LDA direction reproduction reached "
                f"**{100 * float(reproduction['mean_direction_balanced_accuracy']):.1f}%** mean "
                "balanced accuracy, which should not be compared directly with the harder "
                "cross-subject rest/left/right command task."
            ),
            (
                f"- Selected calibration: **{calibration['selected_on_controller_subjects']['budget_per_class']} "
                f"trials/class** with blend "
                f"**{calibration['selected_on_controller_subjects']['blend_weight']:.2f}**. "
                f"Untouched-subject calibrated safety intent BA was "
                f"**{100 * float(calibration['final_safety']['intent_balanced_accuracy']):.1f}%**, "
                f"with **{100 * float(calibration['final_safety']['false_activation_rate']):.1f}%** "
                f"false activation, **{100 * float(calibration['final_safety']['missed_intent_rate']):.1f}%** "
                f"missed intent, and **{100 * float(calibration['final_safety']['direction_balanced_accuracy']):.1f}%** "
                "direction balanced accuracy."
            ),
            "- The strongest result is therefore intent-versus-rest gating; cross-subject left/right direction remains near chance and is not deployment-ready.",
            "",
            "## Interpretation Limits",
            "",
            "- False activation is measured per paired 2.5-second rest window, not per minute of continuous use.",
            "- The local installation is incomplete, so these are interim 29-subject results rather than the full 54-subject OpenBMI benchmark.",
            "- EEGNet and all thresholds are selected without final-subject labels; final results remain a single locked evaluation.",
            "",
            "## Figures",
            "",
            "1. `01_decoder_comparison.png`",
            "2. `02_safety_tradeoff.png`",
            "3. `03_selected_confusion_matrices.png`",
            "4. `04_visual_cue_control.png`",
            "5. `05_subject_variability.png`",
            "6. `06_calibration_ablation.png`",
            "7. `07_within_subject_reproduction.png`",
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
    data_root = resolve_openbmi_root(args.data_dir)
    complete = available_openbmi_subjects(data_root, complete_only=True)
    requested = parse_int_list(args.subjects)
    subjects = complete if requested is None else sorted(set(complete) & set(requested))
    if len(subjects) < 10:
        raise ValueError("Use at least 10 complete subjects for the controlled split.")

    config = OpenBMIBenchmarkConfig()
    dataset = build_openbmi_benchmark_dataset(
        data_root,
        subjects,
        output_dir / "openbmi_controlled_epochs.npz",
        config=config,
        rebuild_cache=args.rebuild_cache,
    )
    partitions = three_way_subject_split(
        subjects,
        target_fraction=0.2,
        controller_fraction=0.2,
        random_seed=args.seed,
    )
    save_partitions(partitions, output_dir / "subject_partitions.json")
    masks = locked_masks(dataset, partitions)
    validate_masks(dataset.labels, masks)
    print(
        f"[+] Subject split: {len(partitions.development)} development, "
        f"{len(partitions.controller_validation)} controller, "
        f"{len(partitions.final_target)} final"
    )

    probabilities, models = fit_raw_models(dataset, masks, seed=args.seed)
    covariance_probabilities, covariance_models, features, feature_metadata = (
        fit_covariance_models(
            dataset,
            masks,
            output_dir / "openbmi_filterbank_covariances.npz",
            seed=args.seed,
            rebuild=args.rebuild_covariances,
        )
    )
    probabilities.update(covariance_probabilities)
    models.update(covariance_models)
    eegnet_model = None
    if not args.skip_eegnet:
        eegnet_probabilities, eegnet_model = fit_eegnet(
            dataset,
            masks,
            epochs=args.eegnet_epochs,
            device=args.eegnet_device,
            seed=args.seed,
        )
        probabilities["eegnet"] = eegnet_probabilities

    methods, selected = evaluate_methods(
        dataset,
        masks,
        probabilities,
        max_false_activation=args.max_false_activation,
    )
    print(f"[+] Controller-selected method: {METHOD_NAMES[selected]}")

    cue_results = cue_leakage_audit(dataset, partitions)
    reproduction = subject_specific_reproduction(dataset)
    adaptation_features = features["tangent_log_euclidean"]
    adaptation_model = covariance_models["tangent_log_euclidean"]
    calibration, _, _, _ = calibration_experiment(
        dataset,
        partitions,
        masks,
        adaptation_features,
        adaptation_model,
        max_false_activation=args.max_false_activation,
        seed=args.seed,
    )

    selected_probabilities = probabilities[selected]
    selected_safety_policy = CommandPolicy(
        **methods[selected]["safety_policy"]
    )
    selected_balanced_policy = CommandPolicy(
        **methods[selected]["balanced_policy"]
    )
    final_probabilities = subset_probabilities(
        selected_probabilities,
        masks["final"],
    )
    methods[selected]["final_safety"]["subject_bootstrap_95_ci"] = (
        subject_bootstrap_interval(
            dataset.labels[masks["final"]],
            final_probabilities,
            dataset.subjects[masks["final"]],
            selected_safety_policy,
            repetitions=args.bootstrap_repetitions,
            random_seed=args.seed,
        )
    )
    subject_rows = per_subject_metrics(
        dataset.labels[masks["final"]],
        final_probabilities,
        dataset.subjects[masks["final"]],
        selected_safety_policy,
    )
    methods[selected]["final_safety"]["per_subject"] = subject_rows

    elapsed = (time.time() - started) / 60.0
    report = {
        "schema_version": 1,
        "experiment": "controlled_openbmi_motor_intent",
        "dataset": {
            "root": str(data_root),
            "subjects": len(subjects),
            "complete_subject_ids": subjects,
            "rows": int(len(dataset.labels)),
            "intent_rows": int(np.sum(dataset.labels != 0)),
            "rest_rows": int(np.sum(dataset.labels == 0)),
            "channels": dataset.motor_channel_names,
            "sampling_hz": dataset.sfreq,
            "motor_window_seconds": [
                config.motor_start_seconds,
                config.motor_start_seconds + config.motor_duration_seconds,
            ],
            "rest_window_seconds": [
                config.rest_start_seconds,
                config.rest_start_seconds + config.motor_duration_seconds,
            ],
            "fingerprint": dataset.fingerprint,
        },
        "partitions": {
            **asdict(partitions),
            "development_count": len(partitions.development),
            "controller_count": len(partitions.controller_validation),
            "final_count": len(partitions.final_target),
        },
        "selection_rule": (
            "Maximum controller safety-policy intent balanced accuracy, then "
            "3-class balanced accuracy, with false activation constrained to "
            f"{args.max_false_activation:.3f}."
        ),
        "selected_method": selected,
        "methods": methods,
        "feature_metadata": feature_metadata,
        "cue_leakage_audit": cue_results,
        "within_subject_reproduction": reproduction,
        "calibration": calibration,
        "runtime_minutes": elapsed,
    }
    metrics_path = output_dir / "openbmi_controlled_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    joblib.dump(
        {
            "models": models,
            "selected_method": selected,
            "balanced_policy": selected_balanced_policy,
            "safety_policy": selected_safety_policy,
            "partitions": partitions,
            "dataset_fingerprint": dataset.fingerprint,
        },
        output_dir / "openbmi_controlled_models.joblib",
    )
    if eegnet_model is not None:
        import torch

        torch.save(
            eegnet_model.payload(),
            output_dir / "openbmi_eegnet.pt",
        )
    write_predictions(
        output_dir / "final_trial_predictions.csv",
        dataset,
        masks["final"],
        selected_probabilities,
        selected_balanced_policy,
        selected_safety_policy,
    )
    plot_method_comparison(output_dir, methods, selected)
    plot_safety_tradeoff(output_dir, methods)
    plot_confusions(output_dir, methods[selected], selected)
    plot_cue_audit(output_dir, cue_results)
    plot_subject_variability(output_dir, subject_rows)
    plot_calibration(
        output_dir,
        calibration,
        methods["tangent_log_euclidean"]["controller_safety"],
    )
    plot_reproduction(output_dir, reproduction)
    write_markdown(output_dir / "OPENBMI_CONTROLLED_REPORT.md", report)

    print("\n=== Locked Untouched-Subject Results ===")
    for name, row in methods.items():
        metrics = row["final_safety"]
        print(
            f"{METHOD_NAMES[name]:22s} | intent BA="
            f"{100 * float(metrics['intent_balanced_accuracy']):5.1f}% | "
            f"direction BA={100 * float(metrics['direction_balanced_accuracy']):5.1f}% | "
            f"FA={100 * float(metrics['false_activation_rate']):5.1f}% | "
            f"missed={100 * float(metrics['missed_intent_rate']):5.1f}%"
        )
    print(f"[+] Saved metrics, models, predictions, report, and seven figures to {output_dir}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
