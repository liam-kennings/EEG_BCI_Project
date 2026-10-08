#!/usr/bin/env python3
"""Evaluate an imagery-only, patient-adaptive two-stage BCI command controller.

The classifier first gates rest versus motor imagery, then emits left/right only
when the direction head is sufficiently confident. Controller policy parameters
are selected on a dedicated patient cohort and reported on untouched patients.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bci_pipeline.dataset import LABEL_NAMES, resolve_data_dir  # noqa: E402
from bci_pipeline.metrics import command_policy_metrics  # noqa: E402
from bci_pipeline.splits import load_partitions  # noqa: E402
from patient_adaptive_bci import (  # noqa: E402
    Config,
    CovarianceDataset,
    align_source_subjects,
    align_with_references,
    baseline_relative_covariances,
    build_covariance_dataset,
    calibration_band_weights,
    decoder_stage_probabilities,
    fit_decoder,
    fit_feature_space,
    nearest_source_subjects,
    threshold_for_false_activation_limit,
    transform_features,
    tune_intent_threshold,
)


@dataclass(frozen=True)
class ControllerPolicy:
    calibration_rest_false_activation_limit: float
    direction_margin: float
    source_intent_floor: float


@dataclass
class PatientScores:
    subject: int
    labels: np.ndarray
    intent: np.ndarray
    direction: np.ndarray
    calibration_labels: np.ndarray
    calibration_intent: np.ndarray


def subject_scores(
    dataset: CovarianceDataset,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_groups: np.ndarray,
    source_references: Dict[int, List[np.ndarray]],
    tangent_spaces: Sequence[object],
    scaler: object,
    pca: object,
    global_decoder: object,
    subject: int,
    calibration_budget: int,
    rest_calibration_budget: int,
    seed: int,
    source_neighbors: int,
) -> PatientScores | None:
    calibration_idx = select_controller_calibration_indices(
        dataset,
        subject,
        direction_budget=calibration_budget,
        rest_budget=rest_calibration_budget,
    )
    evaluation_idx = np.flatnonzero((dataset.subjects == subject) & (dataset.runs == 12))
    if calibration_idx.size == 0 or evaluation_idx.size == 0:
        return None
    if set(dataset.labels[evaluation_idx].tolist()) != set(LABEL_NAMES):
        return None

    # Keep the same regularized reference calculation used in the original experiment.
    from patient_adaptive_bci import alignment_reference

    references = [
        alignment_reference(dataset.covariances[calibration_idx, band])
        for band in range(dataset.covariances.shape[1])
    ]
    band_weights = calibration_band_weights(dataset, calibration_idx)
    nearby_subjects = nearest_source_subjects(
        references,
        source_references,
        band_weights,
        source_neighbors,
    )
    nearby_mask = np.isin(source_groups, nearby_subjects)
    nearby_decoder = fit_decoder(
        source_features[nearby_mask],
        source_labels[nearby_mask],
        seed + subject,
        c_value=1.0,
    )

    calibration_aligned = align_with_references(dataset.covariances[calibration_idx], references)
    evaluation_aligned = align_with_references(dataset.covariances[evaluation_idx], references)
    calibration_relative = baseline_relative_covariances(
        dataset.covariances[calibration_idx], dataset.baseline_covariances[calibration_idx]
    )
    evaluation_relative = baseline_relative_covariances(
        dataset.covariances[evaluation_idx], dataset.baseline_covariances[evaluation_idx]
    )
    calibration_views = np.concatenate((calibration_aligned, calibration_relative), axis=1)
    evaluation_views = np.concatenate((evaluation_aligned, evaluation_relative), axis=1)
    calibration_features = transform_features(calibration_views, tangent_spaces, scaler, pca)
    evaluation_features = transform_features(evaluation_views, tangent_spaces, scaler, pca)

    patient_decoder = fit_decoder(
        calibration_features,
        dataset.labels[calibration_idx],
        seed + subject,
        c_value=0.1,
        max_features=max(2, min(8, 2 * calibration_budget)),
    )
    nearby_intent, _ = decoder_stage_probabilities(nearby_decoder, evaluation_features)
    patient_intent, _ = decoder_stage_probabilities(patient_decoder, evaluation_features)
    nearby_calibration_intent, _ = decoder_stage_probabilities(nearby_decoder, calibration_features)
    patient_calibration_intent, _ = decoder_stage_probabilities(patient_decoder, calibration_features)
    _, global_direction = decoder_stage_probabilities(global_decoder, evaluation_features)

    adapter_weight = min(0.7, calibration_budget / (calibration_budget + 5.0))
    return PatientScores(
        subject=subject,
        labels=dataset.labels[evaluation_idx],
        intent=(1.0 - adapter_weight) * nearby_intent + adapter_weight * patient_intent,
        direction=global_direction,
        calibration_labels=dataset.labels[calibration_idx],
        calibration_intent=(1.0 - adapter_weight) * nearby_calibration_intent
        + adapter_weight * patient_calibration_intent,
    )


def select_controller_calibration_indices(
    dataset: CovarianceDataset,
    subject: int,
    direction_budget: int,
    rest_budget: int,
) -> np.ndarray:
    """Use extra labeled rest trials to estimate the individual safety threshold."""
    selected: List[int] = []
    for label in sorted(LABEL_NAMES):
        budget = rest_budget if label == 0 else direction_budget
        candidates = np.flatnonzero(
            (dataset.subjects == subject)
            & np.isin(dataset.runs, (4, 8))
            & (dataset.labels == label)
        )
        order = np.lexsort((dataset.trial_indices[candidates], dataset.runs[candidates]))
        if len(candidates) < budget:
            return np.asarray([], dtype=np.int64)
        selected.extend(candidates[order][:budget].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def evaluate_policy(scores: Iterable[PatientScores], policy: ControllerPolicy) -> Dict[str, object]:
    per_subject: Dict[str, object] = {}
    thresholds: List[float] = []
    for item in scores:
        patient_threshold = max(
            policy.source_intent_floor,
            threshold_for_false_activation_limit(
                item.calibration_labels,
                item.calibration_intent,
                policy.calibration_rest_false_activation_limit,
            ),
        )
        metrics = command_policy_metrics(
            item.labels,
            item.intent,
            item.direction,
            patient_threshold,
            policy.direction_margin,
        )
        per_subject[f"S{item.subject:03d}"] = metrics
        thresholds.append(patient_threshold)

    if not per_subject:
        raise RuntimeError("No patients had sufficient calibration and Run 12 evaluation trials.")
    macro = {
        key: float(np.mean([metrics[key] for metrics in per_subject.values()]))
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "intent_balanced_accuracy",
            "direction_balanced_accuracy",
            "false_activation_rate",
            "missed_intent_rate",
            "wrong_direction_rate",
            "command_rate",
        )
    }
    return {
        "policy": asdict(policy),
        "evaluated_subjects": len(per_subject),
        "mean_patient_metrics": macro,
        "mean_patient_threshold": float(np.mean(thresholds)),
        "patient_thresholds": {key: float(value) for key, value in zip(per_subject, thresholds)},
        "per_subject": per_subject,
    }


def choose_policy(
    controller_scores: Sequence[PatientScores],
    source_floors: Dict[float, float],
    operating_limit: float,
) -> tuple[ControllerPolicy, List[Dict[str, object]]]:
    candidates: List[Dict[str, object]] = []
    for rest_limit, source_floor in source_floors.items():
        for margin in np.arange(0.0, 0.31, 0.05):
            policy = ControllerPolicy(float(rest_limit), float(margin), float(source_floor))
            result = evaluate_policy(controller_scores, policy)
            metrics = result["mean_patient_metrics"]
            candidates.append(
                {
                    "policy": asdict(policy),
                    "false_activation_rate": metrics["false_activation_rate"],
                    "intent_sensitivity": 1.0 - metrics["missed_intent_rate"],
                    "direction_balanced_accuracy": metrics["direction_balanced_accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )

    feasible = [row for row in candidates if row["false_activation_rate"] <= operating_limit]
    pool = feasible or candidates
    chosen_row = max(
        pool,
        key=lambda row: (
            row["intent_sensitivity"],
            row["direction_balanced_accuracy"],
            -row["false_activation_rate"],
        ),
    )
    values = chosen_row["policy"]
    return ControllerPolicy(**values), candidates


def save_plots(
    output_dir: Path,
    frontiers: Dict[str, List[Dict[str, object]]],
    selected: Dict[str, ControllerPolicy],
    final_results: Dict[str, Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"conservative": "#2d5d7b", "balanced": "#4f9c70", "responsive": "#bc7a32"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, rows in frontiers.items():
        ax.scatter(
            [100 * row["false_activation_rate"] for row in rows],
            [100 * row["intent_sensitivity"] for row in rows],
            s=22,
            alpha=0.35,
            color=colors[name],
        )
        policy = asdict(selected[name])
        selected_row = next(row for row in rows if row["policy"] == policy)
        ax.scatter(
            100 * selected_row["false_activation_rate"],
            100 * selected_row["intent_sensitivity"],
            s=100,
            color=colors[name],
            edgecolor="black",
            label=f"{name.title()} selected",
            zorder=3,
        )
    ax.set_xlabel("Controller-validation false activation rate (%)")
    ax.set_ylabel("Controller-validation intent sensitivity (%)")
    ax.set_title("Patient-adaptive imagery controller operating points")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "controller_tradeoff_frontier.png", dpi=180)
    plt.close(fig)

    names = list(final_results)
    metric_names = [
        ("false_activation_rate", "False activation", -1),
        ("missed_intent_rate", "Missed intent", -1),
        ("direction_balanced_accuracy", "Direction BA", 1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    for axis, (key, title, _) in zip(axes, metric_names):
        values = [100 * final_results[name]["mean_patient_metrics"][key] for name in names]
        bars = axis.bar(names, values, color=[colors[name] for name in names])
        axis.set_title(title)
        axis.set_ylim(0, 100)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    axes[0].set_ylabel("Mean patient rate (%)")
    fig.suptitle("Final untouched-patient imagery-only controller results", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "final_operating_points.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    balanced = final_results["balanced"]["per_subject"]
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    x = [100 * metrics["false_activation_rate"] for metrics in balanced.values()]
    y = [100 * (1.0 - metrics["missed_intent_rate"]) for metrics in balanced.values()]
    ax.scatter(x, y, s=55, color=colors["balanced"], alpha=0.8)
    ax.axvline(15, color="black", linestyle="--", linewidth=1, label="15% design target")
    ax.set_xlabel("Per-patient false activation rate (%)")
    ax.set_ylabel("Per-patient intent sensitivity (%)")
    ax.set_title("Final-patient safety versus sensitivity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_patient_tradeoff.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--cache-path", default="artifacts/patient_adaptive/covariance_cache.npz")
    parser.add_argument("--output-dir", default="artifacts/imagery_intent_controller")
    parser.add_argument("--calibration-budget", type=int, default=12)
    parser.add_argument("--rest-calibration-budget", type=int, default=18)
    parser.add_argument("--source-neighbors", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.calibration_budget < 1 or args.rest_calibration_budget < 1:
        raise ValueError("Calibration budgets must be positive.")

    partitions = load_partitions(Path(args.splits))
    all_subjects = sorted(
        partitions.development + partitions.controller_validation + partitions.final_target
    )
    config = Config(
        data_dir=resolve_data_dir(args.data_dir),
        subjects=all_subjects,
        cache_path=Path(args.cache_path),
        rebuild_cache=args.rebuild_cache,
        random_seed=args.seed,
        source_neighbors=args.source_neighbors,
    )
    dataset = build_covariance_dataset(config)
    observed_subjects = set(np.unique(dataset.subjects).astype(int).tolist())
    development_subjects = tuple(
        subject for subject in partitions.development if subject in observed_subjects
    )
    if not set(partitions.controller_validation).issubset(observed_subjects):
        raise RuntimeError("A locked controller-validation subject is absent from the cached dataset.")
    if not set(partitions.final_target).issubset(observed_subjects):
        raise RuntimeError("A locked final-target subject is absent from the cached dataset.")
    unavailable_development = sorted(set(partitions.development) - observed_subjects)
    if unavailable_development:
        print(f"[!] Excluding unavailable development subjects: {unavailable_development}")

    development_mask = np.isin(dataset.subjects, development_subjects)
    source_covariances = dataset.covariances[development_mask]
    source_baselines = dataset.baseline_covariances[development_mask]
    source_labels = dataset.labels[development_mask]
    source_groups = dataset.subjects[development_mask]
    aligned_source, source_references = align_source_subjects(source_covariances, source_groups)
    relative_source = baseline_relative_covariances(source_covariances, source_baselines)
    source_views = np.concatenate((aligned_source, relative_source), axis=1)
    tangent_spaces, scaler, pca, source_features = fit_feature_space(source_views, config)

    rest_limits = (0.05, 0.10, 0.15, 0.20, 0.25)
    source_floors = {
        limit: tune_intent_threshold(
            source_features,
            source_labels,
            source_groups,
            args.seed,
            limit,
        )[1]
        for limit in rest_limits
    }
    global_decoder = fit_decoder(source_features, source_labels, args.seed, c_value=1.0)
    print("=== Imagery-Only Patient-Adaptive Intent Controller ===")
    print(
        f"[+] Development subjects: {len(development_subjects)} | controller validation: "
        f"{len(partitions.controller_validation)} | final patients: {len(partitions.final_target)}"
    )
    print(
        f"[+] Calibration: Runs 4/8, {args.calibration_budget} trials per imagery class and "
        f"{args.rest_calibration_budget} rest trials; evaluation: Run 12"
    )

    def collect(subjects: Sequence[int]) -> List[PatientScores]:
        rows = [
            subject_scores(
                dataset,
                source_features,
                source_labels,
                source_groups,
                source_references,
                tangent_spaces,
                scaler,
                pca,
                global_decoder,
                subject,
                args.calibration_budget,
                args.rest_calibration_budget,
                args.seed,
                args.source_neighbors,
            )
            for subject in subjects
        ]
        return [row for row in rows if row is not None]

    controller_scores = collect(partitions.controller_validation)
    final_scores = collect(partitions.final_target)
    excluded_controller_subjects = sorted(
        set(partitions.controller_validation) - {item.subject for item in controller_scores}
    )
    if excluded_controller_subjects:
        print(
            "[!] Excluding controller-validation subjects without all three Run 12 labels: "
            f"{excluded_controller_subjects}"
        )
    if not controller_scores:
        raise RuntimeError("No controller-validation patients had complete calibration and evaluation data.")
    if len(final_scores) != len(partitions.final_target):
        raise RuntimeError("A final patient lacked the requested calibration data.")

    design_limits = {"conservative": 0.10, "balanced": 0.15, "responsive": 0.20}
    selected: Dict[str, ControllerPolicy] = {}
    frontiers: Dict[str, List[Dict[str, object]]] = {}
    final_results: Dict[str, Dict[str, object]] = {}
    controller_results: Dict[str, Dict[str, object]] = {}
    for name, limit in design_limits.items():
        policy, frontier = choose_policy(controller_scores, source_floors, limit)
        selected[name] = policy
        frontiers[name] = frontier
        controller_results[name] = evaluate_policy(controller_scores, policy)
        final_results[name] = evaluate_policy(final_scores, policy)
        metrics = final_results[name]["mean_patient_metrics"]
        print(
            f"    {name:12s} | calibration rest limit={policy.calibration_rest_false_activation_limit:.0%} "
            f"| direction margin={policy.direction_margin:.2f} | final FP="
            f"{metrics['false_activation_rate'] * 100:.1f}% | miss="
            f"{metrics['missed_intent_rate'] * 100:.1f}% | direction BA="
            f"{metrics['direction_balanced_accuracy'] * 100:.1f}%"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "task": "motor imagery only; rest versus left/right fist imagery",
            "development_subjects": list(development_subjects),
            "unavailable_development_subjects": unavailable_development,
            "controller_validation_subjects": list(partitions.controller_validation),
            "excluded_controller_validation_subjects": excluded_controller_subjects,
            "final_target_subjects": list(partitions.final_target),
            "calibration_runs": [4, 8],
            "evaluation_run": [12],
            "calibration_trials_per_imagery_class": args.calibration_budget,
            "calibration_rest_trials": args.rest_calibration_budget,
            "subject_overlap": False,
            "controller_selection": "maximize mean patient intent sensitivity under a mean patient false-activation budget",
            "command_policy": "patient-calibrated intent gate plus direction-confidence reject option",
        },
        "dataset": {
            "trials": int(len(dataset.labels)),
            "channels": dataset.channel_names,
            "representation": "filter-bank covariance, Euclidean alignment, baseline-relative Riemannian tangent features",
        },
        "source_intent_floors": {str(key): value for key, value in source_floors.items()},
        "selected_policies": {name: asdict(policy) for name, policy in selected.items()},
        "controller_validation": controller_results,
        "final_target": final_results,
        "controller_frontiers": frontiers,
    }
    with (output_dir / "imagery_intent_controller_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    joblib.dump(
        {
            "label_names": LABEL_NAMES,
            "source_subjects": list(development_subjects),
            "source_references": source_references,
            "source_features": source_features.astype(np.float32),
            "source_labels": source_labels,
            "source_groups": source_groups,
            "tangent_spaces": tangent_spaces,
            "scaler": scaler,
            "pca": pca,
            "global_decoder": global_decoder,
            "selected_policies": selected,
            "calibration_budget": args.calibration_budget,
            "rest_calibration_budget": args.rest_calibration_budget,
        },
        output_dir / "imagery_intent_controller.joblib",
    )
    save_plots(output_dir, frontiers, selected, final_results)
    print(f"[+] Saved report and model to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
