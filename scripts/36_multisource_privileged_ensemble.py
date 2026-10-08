#!/usr/bin/env python3
"""Fuse target-only, MRCP-assisted, and WAY-assisted EEG intent students."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from bci_pipeline.clinical_evaluation import select_intent_threshold  # noqa: E402
from bci_pipeline.multisource_transfer import (  # noqa: E402
    COMPONENT_NAMES,
    GLOBAL_DOMAIN,
    DomainFusionRule,
    MultiSourceEEGEnsemble,
    fit_cross_fitted_stacker,
    logit_component_features,
)
from bci_pipeline.privileged_transfer import (  # noqa: E402
    EMGConfirmedEEGTransfer,
    PrivilegedSourceDataset,
)
from bci_pipeline.translational_decoder import dataset_balanced_weights  # noqa: E402


INK = "#17324D"
TEAL = "#087E8B"
GOLD = "#D99B24"
CORAL = "#C64B42"
BLUE = "#3F6C8A"
MODEL_LABELS = {
    "target_only": "Target-only EEG",
    "mrcp_student": "MRCP-assisted EEG",
    "way_student": "WAY-assisted EEG",
    "multisource": "Validation-selected multi-source",
}
MODEL_COLORS = {
    "target_only": GOLD,
    "mrcp_student": BLUE,
    "way_student": TEAL,
    "multisource": CORAL,
}


def _strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).tolist())


def load_source_cache(path: Path) -> PrivilegedSourceDataset:
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "eeg_features",
            "emg_features",
            "labels",
            "subject_keys",
            "premovement_end_seconds",
            "eeg_feature_names",
            "emg_feature_names",
            "dataset_id",
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"Source cache is missing keys: {sorted(missing)}")
        dataset = PrivilegedSourceDataset(
            eeg_features=cached["eeg_features"],
            emg_features=cached["emg_features"],
            labels=cached["labels"],
            subject_keys=cached["subject_keys"],
            premovement_end_seconds=cached["premovement_end_seconds"],
            eeg_feature_names=_strings(cached["eeg_feature_names"]),
            emg_feature_names=_strings(cached["emg_feature_names"]),
            dataset_id=str(np.asarray(cached["dataset_id"]).item()),
        )
    dataset.validate()
    return dataset


def load_target_cache(path: Path) -> dict[str, np.ndarray | tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "eeg_features",
            "labels",
            "subject_keys",
            "dataset_ids",
            "roles",
            "eeg_feature_names",
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"Target cache is missing keys: {sorted(missing)}")
        result: dict[str, np.ndarray | tuple[str, ...]] = {
            "eeg_features": cached["eeg_features"],
            "labels": cached["labels"].astype(np.int64),
            "subject_keys": cached["subject_keys"].astype(str),
            "dataset_ids": cached["dataset_ids"].astype(str),
            "roles": cached["roles"].astype(str),
            "eeg_feature_names": _strings(cached["eeg_feature_names"]),
        }
    features = np.asarray(result["eeg_features"], dtype=np.float64)
    labels = np.asarray(result["labels"], dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Target EEG features and labels are not row-aligned.")
    if not np.all(np.isfinite(features)) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Target cache requires finite EEG and binary labels.")
    roles = set(np.asarray(result["roles"]).astype(str))
    required_roles = {"development", "controller_validation", "development_observed"}
    if not required_roles.issubset(roles):
        raise ValueError(f"Target cache requires roles: {sorted(required_roles)}")
    return result


def score(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float | np.ndarray,
) -> dict[str, float | int]:
    true = np.asarray(labels, dtype=np.int64) != 0
    probabilities = np.asarray(probability, dtype=np.float64)
    thresholds = np.asarray(threshold, dtype=np.float64)
    predicted = probabilities >= thresholds
    return {
        "rows": int(len(true)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "sensitivity": float(np.mean(predicted[true])),
        "specificity": float(np.mean(~predicted[~true])),
        "missed_intent_rate": float(np.mean(~predicted[true])),
        "false_activation_rate_per_window": float(np.mean(predicted[~true])),
        "auroc": float(roc_auc_score(true, probabilities)),
        "auprc": float(average_precision_score(true, probabilities)),
        "brier_score": float(brier_score_loss(true, probabilities)),
    }


def _selection_key(metrics: dict[str, float | int], complexity: int) -> tuple[float, ...]:
    return (
        float(metrics["balanced_accuracy"]),
        float(metrics["sensitivity"]),
        -float(metrics["false_activation_rate_per_window"]),
        -float(complexity),
    )


def fit_source_student(
    source: PrivilegedSourceDataset,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: tuple[str, ...],
    development: np.ndarray,
    controller: np.ndarray,
    target_weight: np.ndarray,
    source_weights: list[float],
    confidence_floor: float,
    max_false_activation: float,
    seed: int,
) -> tuple[EMGConfirmedEEGTransfer, list[dict[str, Any]]]:
    selected_model: EMGConfirmedEEGTransfer | None = None
    selected_key: tuple[float, ...] | None = None
    candidates: list[dict[str, Any]] = []
    for source_weight in source_weights:
        model = EMGConfirmedEEGTransfer(
            seed=seed,
            source_weight=source_weight,
            confidence_floor=confidence_floor,
        ).fit(
            source,
            features[development],
            labels[development],
            feature_names,
            target_sample_weight=target_weight,
        )
        probability = model.predict_intent_probability(features[controller])
        threshold = select_intent_threshold(
            labels[controller],
            probability,
            max_false_activation=max_false_activation,
        )
        metrics = score(labels[controller], probability, threshold)
        candidates.append(
            {
                "source_weight": source_weight,
                "threshold": threshold,
                "metrics": metrics,
                "fit_summary": asdict(model.summary_),
            }
        )
        key = _selection_key(metrics, 0)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_model = model
    if selected_model is None:
        raise RuntimeError(f"No model was fitted for source {source.dataset_id}.")
    return selected_model, candidates


def _component_matrix(
    mrcp_model: EMGConfirmedEEGTransfer,
    way_model: EMGConfirmedEEGTransfer,
    features: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            mrcp_model.predict_target_only_probability(features),
            mrcp_model.predict_intent_probability(features),
            way_model.predict_intent_probability(features),
        )
    )


def fit_domain_fusion(
    components: np.ndarray,
    labels: np.ndarray,
    dataset_ids: np.ndarray,
    subject_keys: np.ndarray,
    max_false_activation: float,
    seed: int,
) -> tuple[dict[str, object], dict[str, DomainFusionRule], dict[str, Any]]:
    stackers: dict[str, object] = {}
    rules: dict[str, DomainFusionRule] = {}
    audit: dict[str, Any] = {}
    domain_values = [str(value) for value in np.unique(dataset_ids)] + [GLOBAL_DOMAIN]
    for offset, domain in enumerate(domain_values):
        selected = np.ones(len(labels), dtype=bool) if domain == GLOBAL_DOMAIN else dataset_ids == domain
        group_values = np.asarray(
            [
                f"{dataset}:{subject}"
                for dataset, subject in zip(
                    dataset_ids[selected],
                    subject_keys[selected],
                    strict=True,
                )
            ]
        )
        stacker, held_out = fit_cross_fitted_stacker(
            components[selected],
            labels[selected],
            group_values,
            seed=seed + offset * 100,
        )
        stackers[domain] = stacker
        candidate_probabilities = {
            name: components[selected, index]
            for index, name in enumerate(COMPONENT_NAMES)
        }
        candidate_probabilities["stacked"] = held_out
        candidate_results: dict[str, Any] = {}
        selected_method: str | None = None
        selected_threshold = 0.5
        selected_metrics: dict[str, float | int] | None = None
        selected_key: tuple[float, ...] | None = None
        for complexity, method in enumerate((*COMPONENT_NAMES, "stacked")):
            probability = candidate_probabilities[method]
            threshold = select_intent_threshold(
                labels[selected],
                probability,
                max_false_activation=max_false_activation,
            )
            metrics = score(labels[selected], probability, threshold)
            candidate_results[method] = {"threshold": threshold, "metrics": metrics}
            key = _selection_key(metrics, complexity)
            if selected_key is None or key > selected_key:
                selected_key = key
                selected_method = method
                selected_threshold = threshold
                selected_metrics = metrics
        assert selected_method is not None
        assert selected_metrics is not None
        rules[domain] = DomainFusionRule(
            method=selected_method,
            threshold=selected_threshold,
            controller_balanced_accuracy=float(selected_metrics["balanced_accuracy"]),
            controller_sensitivity=float(selected_metrics["sensitivity"]),
            controller_specificity=float(selected_metrics["specificity"]),
        )
        audit[domain] = {
            "rows": int(np.sum(selected)),
            "subjects": int(len(np.unique(group_values))),
            "candidates": candidate_results,
            "selected": asdict(rules[domain]),
        }
    return stackers, rules, audit


def component_thresholds(
    components: np.ndarray,
    labels: np.ndarray,
    dataset_ids: np.ndarray,
    max_false_activation: float,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for index, method in enumerate(COMPONENT_NAMES):
        thresholds[method] = {}
        for domain in np.unique(dataset_ids):
            selected = dataset_ids == domain
            thresholds[method][str(domain)] = select_intent_threshold(
                labels[selected],
                components[selected, index],
                max_false_activation=max_false_activation,
            )
    return thresholds


def _threshold_vector(
    dataset_ids: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    return np.asarray([thresholds[str(domain)] for domain in dataset_ids], dtype=np.float64)


def participant_analysis(
    labels: np.ndarray,
    baseline_probability: np.ndarray,
    ensemble_probability: np.ndarray,
    baseline_threshold: np.ndarray,
    ensemble_threshold: np.ndarray,
    subject_keys: np.ndarray,
    dataset_ids: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    paired = np.asarray(
        [f"{dataset}:{subject}" for dataset, subject in zip(dataset_ids, subject_keys, strict=True)]
    )
    for key in np.unique(paired):
        selected = paired == key
        if len(np.unique(labels[selected])) != 2:
            continue
        baseline = score(
            labels[selected],
            baseline_probability[selected],
            baseline_threshold[selected],
        )
        ensemble = score(
            labels[selected],
            ensemble_probability[selected],
            ensemble_threshold[selected],
        )
        dataset, subject = key.split(":", maxsplit=1)
        records.append(
            {
                "dataset_id": dataset,
                "subject_key": subject,
                "rows": int(np.sum(selected)),
                "target_only_balanced_accuracy": baseline["balanced_accuracy"],
                "multisource_balanced_accuracy": ensemble["balanced_accuracy"],
                "balanced_accuracy_delta": ensemble["balanced_accuracy"]
                - baseline["balanced_accuracy"],
                "target_only_sensitivity": baseline["sensitivity"],
                "multisource_sensitivity": ensemble["sensitivity"],
            }
        )
    deltas = np.asarray([record["balanced_accuracy_delta"] for record in records])
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True)
    bootstrap = np.mean(samples, axis=1)
    return {
        "participants": len(records),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "median_balanced_accuracy_delta": float(np.median(deltas)),
        "bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "participants_improved_fraction": float(np.mean(deltas > 0.0)),
        "bootstrap_iterations": iterations,
        "records": records,
    }


def _plot_overall(evaluation: dict[str, Any], output: Path) -> None:
    models = tuple(MODEL_LABELS)
    metrics = ("balanced_accuracy", "sensitivity", "specificity", "auroc")
    labels = ("Balanced accuracy", "Sensitivity", "Specificity", "AUROC")
    positions = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.5, 6.7), constrained_layout=True)
    for index, model in enumerate(models):
        values = 100.0 * np.asarray([evaluation[model][metric] for metric in metrics])
        offset = (index - 1.5) * width
        ax.bar(
            positions + offset,
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
        for position, value in zip(positions + offset, values, strict=True):
            ax.text(position, value + 1.0, f"{value:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Development-observed result (%)")
    ax.set_title(
        "Privileged-Source Fusion for EEG-Only Motor-Intent Detection",
        loc="left",
        weight="bold",
        color=INK,
    )
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "01_multisource_overall_comparison.png", dpi=220)
    plt.close(fig)


def _plot_domains(
    evaluation_by_dataset: dict[str, Any],
    fusion_audit: dict[str, Any],
    participant: dict[str, Any],
    output: Path,
) -> None:
    domains = tuple(evaluation_by_dataset)
    models = tuple(MODEL_LABELS)
    positions = np.arange(len(domains))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    for index, model in enumerate(models):
        values = 100.0 * np.asarray(
            [evaluation_by_dataset[domain][model]["balanced_accuracy"] for domain in domains]
        )
        offset = (index - 1.5) * width
        axes[0].bar(
            positions + offset,
            values,
            width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
    domain_labels = [domain.upper() if domain == "openbmi" else domain.title() for domain in domains]
    axes[0].set_xticks(positions, domain_labels)
    axes[0].set_ylim(45, 70)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_title("Unseen-subject performance by target domain", loc="left", weight="bold", color=INK)
    axes[0].legend(frameon=False, fontsize=8, ncols=2)
    for index, domain in enumerate(domains):
        method = fusion_audit[domain]["selected"]["method"]
        axes[0].text(index, 46.0, f"Selected: {method.replace('_', ' ')}", ha="center", fontsize=8, color=INK)

    color_map = {"liu2024": CORAL, "openbmi": TEAL, "physionet": GOLD}
    records = participant["records"]
    for index, domain in enumerate(domains):
        deltas = 100.0 * np.asarray(
            [record["balanced_accuracy_delta"] for record in records if record["dataset_id"] == domain]
        )
        jitter = np.linspace(-0.13, 0.13, len(deltas)) if len(deltas) > 1 else np.zeros(len(deltas))
        axes[1].scatter(
            np.full(len(deltas), index) + jitter,
            deltas,
            color=color_map.get(domain, BLUE),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            s=44,
        )
        if len(deltas):
            axes[1].plot([index - 0.18, index + 0.18], [np.mean(deltas)] * 2, color=INK, linewidth=3)
    axes[1].axhline(0.0, color=INK, linewidth=1.0)
    axes[1].set_xticks(positions, domain_labels)
    axes[1].set_ylabel("Participant BA change vs target-only (points)")
    axes[1].set_title("Paired unseen-participant effects", loc="left", weight="bold", color=INK)
    for axis in axes:
        axis.grid(axis="y", color="#D5DEE2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    low, high = participant["bootstrap_95_ci"]
    fig.suptitle(
        f"Dataset-Calibrated Multi-Source Transfer | Mean participant change "
        f"{100 * participant['mean_balanced_accuracy_delta']:+.1f} points "
        f"(95% CI {100 * low:+.1f} to {100 * high:+.1f})",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.savefig(output / "02_multisource_domain_and_participants.png", dpi=220)
    plt.close(fig)


def _write_report(payload: dict[str, Any], output: Path) -> None:
    evaluation = payload["evaluation"]
    participant = payload["participant_analysis"]
    low, high = participant["bootstrap_95_ci"]
    lines = [
        "# Multi-Source Privileged EEG Ensemble",
        "",
        "## Leakage Boundary",
        "",
        "- MRCP EMG and WAY EMG/kinematics are used only to verify synchronized source examples.",
        "- Every deployed component receives the same 144 EEG features; the fusion API additionally receives only the dataset identifier.",
        "- Source weights are selected on controller-validation subjects.",
        "- Dataset-specific stackers and thresholds use group-held controller predictions.",
        "- Development-observed subjects are evaluated once and never used for model, fusion, or threshold selection.",
        "- The separate left/right direction head is unchanged.",
        "",
        "## Development-Observed Result",
        "",
        "| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_LABELS:
        result = evaluation[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {100 * result['balanced_accuracy']:.2f}% | "
            f"{100 * result['sensitivity']:.2f}% | {100 * result['specificity']:.2f}% | "
            f"{result['auroc']:.3f} | {result['auprc']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Across **{participant['participants']}** paired unseen participants, the equal-participant mean balanced-accuracy change versus the domain-thresholded target-only baseline was **{100 * participant['mean_balanced_accuracy_delta']:+.2f} points** (95% participant-bootstrap CI **{100 * low:+.2f} to {100 * high:+.2f}**).",
            "",
            "## Dataset Policies And Results",
            "",
            "| Dataset | Controller-selected policy | Target-only BA | Multi-source BA | Sensitivity | Specificity |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for domain, comparisons in payload["evaluation_by_dataset"].items():
        selected = payload["fusion_selection"][domain]["selected"]
        baseline = comparisons["target_only"]
        ensemble = comparisons["multisource"]
        lines.append(
            f"| {domain} | {selected['method']} at {selected['threshold']:.3f} | "
            f"{100 * baseline['balanced_accuracy']:.2f}% | "
            f"{100 * ensemble['balanced_accuracy']:.2f}% | "
            f"{100 * ensemble['sensitivity']:.2f}% | "
            f"{100 * ensemble['specificity']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The ensemble is an exploratory cue-locked intent gate, not an asynchronous prosthetic controller. Improvement must be judged against the paired participant interval and domain results, not only pooled window accuracy. Continuous no-control recordings and a still-locked clinical cohort remain required before a device claim.",
        ]
    )
    (output / "MULTISOURCE_PRIVILEGED_ENSEMBLE_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrcp-source-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer" / "mrcp_eeg_emg_source_features.npz",
    )
    parser.add_argument(
        "--way-source-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "way_eeg_gal_transfer" / "way_eeg_gal_source_features.npz",
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer" / "pure_eeg_target_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "multisource_privileged_ensemble",
    )
    parser.add_argument("--source-weight-grid", default="0.05,0.10,0.20,0.35")
    parser.add_argument("--confidence-floor", type=float, default=0.60)
    parser.add_argument("--max-controller-false-activation", type=float, default=0.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source_weights = sorted(
        {float(value.strip()) for value in args.source_weight_grid.split(",")}
    )
    if not source_weights or any(not 0.0 < value <= 1.0 for value in source_weights):
        parser.error("Every source weight must be in (0, 1].")

    mrcp_source = load_source_cache(args.mrcp_source_cache)
    way_source = load_source_cache(args.way_source_cache)
    target = load_target_cache(args.target_cache)
    features = np.asarray(target["eeg_features"], dtype=np.float64)
    labels = np.asarray(target["labels"], dtype=np.int64)
    roles = np.asarray(target["roles"]).astype(str)
    datasets = np.asarray(target["dataset_ids"]).astype(str)
    subjects = np.asarray(target["subject_keys"]).astype(str)
    feature_names = tuple(target["eeg_feature_names"])
    development = roles == "development"
    controller = roles == "controller_validation"
    observed = roles == "development_observed"
    target_weight = dataset_balanced_weights(labels[development], datasets[development])

    print("[+] Selecting MRCP and WAY source weights on controller-validation subjects...")
    mrcp_model, mrcp_candidates = fit_source_student(
        mrcp_source,
        features,
        labels,
        feature_names,
        development,
        controller,
        target_weight,
        source_weights,
        args.confidence_floor,
        args.max_controller_false_activation,
        args.seed,
    )
    way_model, way_candidates = fit_source_student(
        way_source,
        features,
        labels,
        feature_names,
        development,
        controller,
        target_weight,
        source_weights,
        args.confidence_floor,
        args.max_controller_false_activation,
        args.seed + 5000,
    )
    controller_components = _component_matrix(mrcp_model, way_model, features[controller])
    observed_components = _component_matrix(mrcp_model, way_model, features[observed])
    print("[+] Fitting group-held domain fusion and selecting safety thresholds...")
    stackers, rules, fusion_audit = fit_domain_fusion(
        controller_components,
        labels[controller],
        datasets[controller],
        subjects[controller],
        args.max_controller_false_activation,
        args.seed + 10000,
    )
    ensemble = MultiSourceEEGEnsemble(
        mrcp_model=mrcp_model,
        way_model=way_model,
        stackers=stackers,
        rules=rules,
        feature_names=feature_names,
    )
    component_policy_thresholds = component_thresholds(
        controller_components,
        labels[controller],
        datasets[controller],
        args.max_controller_false_activation,
    )

    observed_datasets = datasets[observed]
    probabilities = {
        name: observed_components[:, index]
        for index, name in enumerate(COMPONENT_NAMES)
    }
    probabilities["multisource"] = ensemble.predict_intent_probability(
        features[observed],
        observed_datasets,
    )
    threshold_vectors = {
        name: _threshold_vector(observed_datasets, component_policy_thresholds[name])
        for name in COMPONENT_NAMES
    }
    threshold_vectors["multisource"] = ensemble.threshold_vector(observed_datasets)
    evaluation = {
        model: score(labels[observed], probabilities[model], threshold_vectors[model])
        for model in MODEL_LABELS
    }
    evaluation_by_dataset: dict[str, Any] = {}
    for domain in np.unique(observed_datasets):
        selected = observed_datasets == domain
        evaluation_by_dataset[str(domain)] = {
            model: score(
                labels[observed][selected],
                probabilities[model][selected],
                threshold_vectors[model][selected],
            )
            for model in MODEL_LABELS
        }
    participant = participant_analysis(
        labels[observed],
        probabilities["target_only"],
        probabilities["multisource"],
        threshold_vectors["target_only"],
        threshold_vectors["multisource"],
        subjects[observed],
        observed_datasets,
        seed=args.seed,
        iterations=args.bootstrap_iterations,
    )
    payload = {
        "schema_version": 1,
        "experiment": "dataset_calibrated_multisource_privileged_eeg_ensemble",
        "source_datasets": [mrcp_source.dataset_id, way_source.dataset_id],
        "target_datasets": sorted(np.unique(datasets).tolist()),
        "leakage_boundary": {
            "peripheral_signals_used_at_target_inference": False,
            "positive_source_eeg_ends_before_confirmed_movement": True,
            "source_weights_selected_on": "controller_validation",
            "fusion_selected_on": "group-held controller_validation predictions",
            "evaluated_on": "development_observed",
            "direction_head_modified": False,
        },
        "selected_source_weights": {
            "mrcp": mrcp_model.source_weight,
            "way": way_model.source_weight,
        },
        "source_fit_summaries": {
            "mrcp": asdict(mrcp_model.summary_),
            "way": asdict(way_model.summary_),
        },
        "source_weight_selection": {
            "mrcp": mrcp_candidates,
            "way": way_candidates,
        },
        "component_domain_thresholds": component_policy_thresholds,
        "fusion_selection": fusion_audit,
        "evaluation": evaluation,
        "evaluation_by_dataset": evaluation_by_dataset,
        "participant_analysis": participant,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "multisource_privileged_ensemble_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(ensemble, args.output_dir / "multisource_eeg_intent_ensemble.joblib")
    _plot_overall(evaluation, args.output_dir)
    _plot_domains(evaluation_by_dataset, fusion_audit, participant, args.output_dir)
    _write_report(payload, args.output_dir)
    selected_text = ", ".join(
        f"{domain}={rules[domain].method}" for domain in sorted(np.unique(datasets))
    )
    print(f"[+] Selected domain policies: {selected_text}")
    print(
        "[+] Development-observed multi-source result: "
        f"BA={100 * evaluation['multisource']['balanced_accuracy']:.2f}% | "
        f"sensitivity={100 * evaluation['multisource']['sensitivity']:.2f}% | "
        f"specificity={100 * evaluation['multisource']['specificity']:.2f}% | "
        f"AUROC={evaluation['multisource']['auroc']:.3f}"
    )
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
