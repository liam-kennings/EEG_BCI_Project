#!/usr/bin/env python3
"""Patient-calibrated rest-aware motor-imagery intent gate with sequential confirmation.

This is a windowed controller simulation, not a continuous clinical-device
claim.  It uses early/middle/terminal Morlet-ERD windows from cue-locked trials,
selects all policy parameters on controller subjects, and reports untouched
final-subject safety and latency metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.metrics import temporal_metrics  # noqa: E402
from bci_pipeline.splits import load_partitions  # noqa: E402


PHASES = ("early", "middle", "terminal")
PHASE_END_SECONDS = {"early": 1.5, "middle": 2.5, "terminal": 3.5}
STRATEGIES = ("single_early", "single_middle", "single_terminal", "temporal_2of3", "temporal_3of3")
TASK_COLOR = "#D05A3A"
BASELINE_COLOR = "#75869A"
REST_COLOR = "#2A9D8F"


@dataclass(frozen=True)
class Policy:
    strategy: str
    adapter_weight: float
    calibration_rest_false_activation_limit: float
    ood_rest_quantile: float


@dataclass
class PhaseScores:
    labels: np.ndarray
    global_evaluation: np.ndarray
    local_evaluation: np.ndarray
    global_calibration: np.ndarray
    local_calibration: np.ndarray
    calibration_labels: np.ndarray
    evaluation_ood_distance: np.ndarray
    calibration_ood_distance: np.ndarray


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "savefig.facecolor": "white",
        }
    )


def phase_feature_names(fieldnames: Sequence[str], phase: str) -> List[str]:
    suffixes = (f"_{phase}_erd_db", f"_{phase}_db")
    erd_names = [name for name in fieldnames if name.endswith(suffixes)]
    # These coarse signal-quality features provide a rest-specific artifact context.
    rest_context = ["mean_peak_to_peak_uv", "max_peak_to_peak_uv", "mean_rms_uv"]
    missing = [name for name in rest_context if name not in fieldnames]
    if missing:
        raise ValueError(f"Missing rest-context feature columns: {missing}")
    if not erd_names:
        raise ValueError(f"No ERD feature columns found for phase '{phase}'.")
    return erd_names + rest_context


def load_feature_table(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Feature table has no header.")
        rows = list(reader)
        names = {phase: phase_feature_names(reader.fieldnames, phase) for phase in PHASES}
    phase_data = {
        phase: np.asarray([[float(row[name]) for name in columns] for row in rows], dtype=np.float64)
        for phase, columns in names.items()
    }
    labels = np.asarray([int(row["label"]) != 0 for row in rows], dtype=np.int64)
    subjects = np.asarray([int(row["subject"]) for row in rows], dtype=np.int64)
    runs = np.asarray([int(row["run"]) for row in rows], dtype=np.int64)
    trials = np.asarray([int(row["trial_index"]) for row in rows], dtype=np.int64)
    if any(np.any(~np.isfinite(values)) for values in phase_data.values()):
        raise ValueError("Morlet feature table contains non-finite values.")
    return phase_data, labels, subjects, runs, trials, names


def make_global_model(feature_count: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selection", SelectKBest(f_classif, k=min(50, feature_count))),
            (
                "model",
                LogisticRegression(
                    C=0.2,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def selected_features(model: Pipeline, values: np.ndarray) -> np.ndarray:
    imputed = model.named_steps["imputer"].transform(values)
    scaled = model.named_steps["scaler"].transform(imputed)
    return model.named_steps["selection"].transform(scaled)


def target_align(calibration: np.ndarray, values: np.ndarray, model: Pipeline) -> np.ndarray:
    source_mean = model.named_steps["scaler"].mean_
    source_std = np.sqrt(model.named_steps["scaler"].var_)
    source_std = np.clip(source_std, 1e-6, None)
    target_mean = np.mean(calibration, axis=0)
    target_std = np.clip(np.std(calibration, axis=0), 1e-6, None)
    return ((values - target_mean) / target_std) * source_std + source_mean


def select_calibration_indices(
    labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    trials: np.ndarray,
    subject: int,
    imagery_budget: int,
    rest_budget: int,
) -> np.ndarray:
    selected: List[int] = []
    for label, budget in ((0, rest_budget), (1, imagery_budget)):
        candidates = np.flatnonzero(
            (subjects == subject) & np.isin(runs, (4, 8)) & (labels == label)
        )
        order = np.lexsort((trials[candidates], runs[candidates]))
        candidates = candidates[order]
        if len(candidates) < budget:
            return np.asarray([], dtype=np.int64)
        selected.extend(candidates[:budget].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def phase_scores_for_subject(
    model: Pipeline,
    data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    trials: np.ndarray,
    subject: int,
    imagery_budget: int,
    rest_budget: int,
    seed: int,
) -> PhaseScores | None:
    calibration = select_calibration_indices(
        labels, subjects, runs, trials, subject, imagery_budget, rest_budget
    )
    evaluation = np.flatnonzero((subjects == subject) & (runs == 12))
    if not len(calibration) or not len(evaluation):
        return None
    if set(labels[evaluation].tolist()) != {0, 1}:
        return None
    aligned_calibration = target_align(data[calibration], data[calibration], model)
    aligned_evaluation = target_align(data[calibration], data[evaluation], model)
    calibration_features = selected_features(model, aligned_calibration)
    evaluation_features = selected_features(model, aligned_evaluation)
    local_model = LogisticRegression(
        C=0.05,
        class_weight="balanced",
        max_iter=3000,
        random_state=seed + subject,
    ).fit(calibration_features, labels[calibration])
    rest_features = calibration_features[labels[calibration] == 0]
    rest_detector = LedoitWolf().fit(rest_features)
    return PhaseScores(
        labels=labels[evaluation],
        global_evaluation=model.predict_proba(aligned_evaluation)[:, 1],
        local_evaluation=local_model.predict_proba(evaluation_features)[:, 1],
        global_calibration=model.predict_proba(aligned_calibration)[:, 1],
        local_calibration=local_model.predict_proba(calibration_features)[:, 1],
        calibration_labels=labels[calibration],
        evaluation_ood_distance=rest_detector.mahalanobis(evaluation_features),
        calibration_ood_distance=rest_detector.mahalanobis(calibration_features),
    )


def phase_decisions(scores: PhaseScores, policy: Policy) -> Tuple[np.ndarray, np.ndarray, float]:
    calibration_probability = (
        (1.0 - policy.adapter_weight) * scores.global_calibration
        + policy.adapter_weight * scores.local_calibration
    )
    evaluation_probability = (
        (1.0 - policy.adapter_weight) * scores.global_evaluation
        + policy.adapter_weight * scores.local_evaluation
    )
    rest_probability = calibration_probability[scores.calibration_labels == 0]
    threshold = float(
        np.nextafter(
            np.quantile(rest_probability, 1.0 - policy.calibration_rest_false_activation_limit, method="higher"),
            np.inf,
        )
    )
    if policy.ood_rest_quantile >= 1.0:
        rejected = np.zeros(len(evaluation_probability), dtype=bool)
    else:
        rest_distance = scores.calibration_ood_distance[scores.calibration_labels == 0]
        ood_limit = float(np.quantile(rest_distance, policy.ood_rest_quantile, method="higher"))
        rejected = scores.evaluation_ood_distance > ood_limit
    return (evaluation_probability >= threshold) & ~rejected, rejected, threshold


def sequential_predictions(
    phase_scores: Dict[str, PhaseScores], policy: Policy
) -> Tuple[np.ndarray, List[float | None], np.ndarray, List[float]]:
    commands: List[np.ndarray] = []
    rejected: List[np.ndarray] = []
    thresholds: List[float] = []
    for phase in PHASES:
        accepted, ood_rejected, threshold = phase_decisions(phase_scores[phase], policy)
        commands.append(accepted)
        rejected.append(ood_rejected)
        thresholds.append(threshold)
    labels = phase_scores["early"].labels
    if policy.strategy.startswith("single_"):
        phase = policy.strategy.removeprefix("single_")
        index = PHASES.index(phase)
        predictions = commands[index]
        latencies: List[float | None] = [
            PHASE_END_SECONDS[phase] if prediction and label == 1 else None
            for prediction, label in zip(predictions, labels)
        ]
    elif policy.strategy == "temporal_2of3":
        stacked = np.stack(commands)
        predictions = np.sum(stacked, axis=0) >= 2
        latencies = []
        for row, label in zip(stacked.T, labels):
            second_positive = np.flatnonzero(np.cumsum(row) >= 2)
            latencies.append(
                PHASE_END_SECONDS[PHASES[int(second_positive[0])]]
                if len(second_positive) and label == 1
                else None
            )
    elif policy.strategy == "temporal_3of3":
        stacked = np.stack(commands)
        predictions = np.all(stacked, axis=0)
        latencies = [3.5 if prediction and label == 1 else None for prediction, label in zip(predictions, labels)]
    else:
        raise ValueError(f"Unknown strategy: {policy.strategy}")
    return predictions.astype(np.int64), latencies, np.mean(np.stack(rejected), axis=0), thresholds


def evaluate_policy(
    scores: Dict[int, Dict[str, PhaseScores]], policy: Policy, trial_seconds: float
) -> Dict[str, object]:
    per_subject: Dict[str, Dict[str, object]] = {}
    for subject, phase_score in scores.items():
        predictions, latencies, rejection, thresholds = sequential_predictions(phase_score, policy)
        metrics = temporal_metrics(
            phase_score["early"].labels,
            predictions,
            latencies,
            rejection,
            trial_seconds,
        )
        metrics["command_rate"] = float(np.mean(predictions))
        metrics["mean_intent_threshold"] = float(np.mean(thresholds))
        metrics["accepted_true_intent_count"] = int(np.sum((phase_score["early"].labels == 1) & (predictions == 1)))
        per_subject[f"S{subject:03d}"] = metrics
    if not per_subject:
        raise RuntimeError("No complete patient records were available.")
    summary_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "false_activation_rate",
        "missed_intent_rate",
        "wrong_direction_rate",
        "false_activations_per_minute",
        "mean_ood_window_rejection_rate",
        "command_rate",
    )
    mean_metrics = {
        key: float(np.mean([metrics[key] for metrics in per_subject.values()]))
        for key in summary_keys
    }
    true_latencies = [
        metrics["median_latency_seconds"]
        for metrics in per_subject.values()
        if metrics["median_latency_seconds"] is not None
    ]
    mean_metrics["median_true_intent_latency_seconds"] = (
        float(np.median(true_latencies)) if true_latencies else None
    )
    return {"policy": asdict(policy), "mean_patient_metrics": mean_metrics, "per_subject": per_subject}


def choose_policy(
    controller_scores: Dict[int, Dict[str, PhaseScores]],
    max_false_activation: float,
    trial_seconds: float,
) -> Tuple[Policy, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for strategy in STRATEGIES:
        for weight in (0.0, 0.35, 0.70):
            for rest_limit in (0.05, 0.10, 0.15, 0.20):
                for ood_quantile in (1.0, 0.99, 0.95, 0.90):
                    policy = Policy(strategy, weight, rest_limit, ood_quantile)
                    result = evaluate_policy(controller_scores, policy, trial_seconds)
                    metrics = result["mean_patient_metrics"]
                    rows.append({"policy": asdict(policy), **metrics})
    feasible = [row for row in rows if row["false_activation_rate"] <= max_false_activation]
    pool = feasible or rows
    selected = max(
        pool,
        key=lambda row: (
            1.0 - row["missed_intent_rate"],
            row["balanced_accuracy"],
            -row["false_activation_rate"],
        ),
    )
    return Policy(**selected["policy"]), rows


def save_overview(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_axis_off()
    blocks = [
        ("Patient calibration", "Runs 4/8\nrest + imagery labels"),
        ("Rest-aware gate", "Morlet ERD + signal\nquality context"),
        ("OOD screen", "Reject windows outside\npersonal rest distribution"),
        ("Sequential evidence", "Early -> middle -> terminal\nconfirm 2 of 3 windows"),
        ("Final evaluation", "Run 12, unseen for\nthat patient"),
    ]
    xs = np.linspace(0.04, 0.81, len(blocks))
    for index, ((title, detail), x) in enumerate(zip(blocks, xs)):
        patch = FancyBboxPatch(
            (x, 0.34), 0.15, 0.30, boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5, edgecolor=TASK_COLOR, facecolor="#FFF9F6"
        )
        ax.add_patch(patch)
        ax.text(x + 0.075, 0.54, title, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x + 0.075, 0.42, detail, ha="center", va="center", color="#44546A", fontsize=8.5)
        if index < len(blocks) - 1:
            ax.annotate("", xy=(x + 0.18, 0.49), xytext=(x + 0.153, 0.49), arrowprops={"arrowstyle": "->", "color": "#75869A", "lw": 1.4})
    ax.text(0.5, 0.90, "Rest-Aware, Patient-Calibrated Motor-Imagery Intent Gate", ha="center", fontsize=16, weight="bold", color="#152536")
    ax.text(0.5, 0.09, "The direction decoder is unchanged; this experiment isolates the rest-versus-intent safety problem.", ha="center", color="#44546A", fontsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(output_dir / "rest_aware_controller_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_frontier(
    output_dir: Path,
    frontier: List[Dict[str, object]],
    responsive: Policy,
    safety: Policy,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    colors = {"single_early": "#427AA1", "single_middle": "#5AA469", "single_terminal": "#7A6F9B", "temporal_2of3": "#D05A3A", "temporal_3of3": "#9B5D41"}
    for strategy in STRATEGIES:
        rows = [row for row in frontier if row["policy"]["strategy"] == strategy]
        ax.scatter(
            [100 * row["false_activation_rate"] for row in rows],
            [100 * (1.0 - row["missed_intent_rate"]) for row in rows],
            s=20, alpha=0.30, color=colors[strategy], label=strategy.replace("_", " ")
        )
    for policy, marker, label in (
        (responsive, "o", "responsive selected"),
        (safety, "s", "safety selected"),
    ):
        row = next(row for row in frontier if row["policy"] == asdict(policy))
        ax.scatter(
            100 * row["false_activation_rate"],
            100 * (1.0 - row["missed_intent_rate"]),
            s=110,
            marker=marker,
            color="#152536",
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=5,
        )
    ax.axvline(15, linestyle="--", color="#75869A", linewidth=1.2, label="15% FP target")
    ax.set_xlabel("Controller-subject false activation rate (%)")
    ax.set_ylabel("Controller-subject intent sensitivity (%)")
    ax.set_title("Policy Selection: Safety Versus Intent Sensitivity")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "rest_aware_policy_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_comparison(output_dir: Path, results: Dict[str, Dict[str, object]]) -> None:
    keys = ("balanced_accuracy", "false_activation_rate", "missed_intent_rate")
    labels = ("Balanced\naccuracy", "False\nactivation", "Missed\nintent")
    names = list(results)
    colors = [BASELINE_COLOR, TASK_COLOR, REST_COLOR]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    positions = np.arange(len(keys))
    width = 0.23
    offsets = (-width, 0.0, width)
    for name, color, offset in zip(names, colors, offsets):
        values = [results[name]["mean_patient_metrics"][key] for key in keys]
        bars = ax.bar(positions + offset, values, width, color=color, label=name)
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value * 100:.1f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean final-subject rate")
    ax.set_title("Final Unseen-Subject Intent-Gate Comparison")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    fa_minutes = " | ".join(
        f"{name}: {result['mean_patient_metrics']['false_activations_per_minute']:.2f} FA/min"
        for name, result in results.items()
    )
    fig.text(0.5, 0.01, fa_minutes + " (3-second decision coverage assumption)", ha="center", fontsize=8, color="#44546A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_dir / "rest_aware_final_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_subject_tradeoff(output_dir: Path, result: Dict[str, object], title: str) -> None:
    per_subject = result["per_subject"]
    x_values = [100 * value["false_activation_rate"] for value in per_subject.values()]
    y_values = [100 * (1.0 - value["missed_intent_rate"]) for value in per_subject.values()]
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.scatter(x_values, y_values, color=TASK_COLOR, s=58, alpha=0.82, edgecolor="white", linewidth=0.8)
    ax.axvline(15, linestyle="--", color="#75869A", linewidth=1.2, label="15% design target")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Per-subject false activation rate (%)")
    ax.set_ylabel("Per-subject intent sensitivity (%)")
    ax.set_title(f"{title} Across {len(per_subject)} Final Unseen Subjects")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "rest_aware_final_subject_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, report: Dict[str, object]) -> None:
    baseline = report["final_target"]["single_window_no_ood"]["mean_patient_metrics"]
    responsive = report["final_target"]["responsive_rest_aware_policy"]["mean_patient_metrics"]
    safety = report["final_target"]["safety_rest_aware_policy"]["mean_patient_metrics"]
    policies = report["selected_policies"]
    lines = [
        "# Rest-Aware Sequential Motor-Imagery Intent Gate",
        "",
        "## Purpose",
        "",
        "This experiment improves only the motor-imagery rest-versus-intent gate. The left/right direction decoder is intentionally unchanged. The dual execution-versus-imagery benchmark remains the fixed reference; this is a separate patient-calibrated controller study.",
        "",
        "## Protocol",
        "",
        "- Features: baseline-normalized Morlet mu/beta ERD from central motor channels, plus peak-to-peak and RMS signal-quality context.",
        "- Calibration: each subject contributes labeled rest and imagery trials from Runs 4/8.",
        "- Evaluation: Run 12 for the same subject is held out from that subject's calibration.",
        "- OOD rejection: a Ledoit-Wolf rest-distribution model rejects windows outside a selected calibration-rest distance quantile.",
        "- Sequential confirmation: early (0.5-1.5 s), middle (1.5-2.5 s), and terminal (2.5-3.5 s) windows can require 2-of-3 or 3-of-3 positive evidence before a command.",
        "- Selection: all policy parameters are selected only on controller subjects; final subjects are untouched.",
        "",
        "## Selected Policies",
        "",
        f"- Responsive policy (`{policies['responsive']['strategy']}`): blend `{policies['responsive']['adapter_weight']:.2f}`, rest target `{policies['responsive']['calibration_rest_false_activation_limit']:.0%}`, OOD quantile `{policies['responsive']['ood_rest_quantile']:.2f}`.",
        f"- Safety policy (`{policies['safety']['strategy']}`): blend `{policies['safety']['adapter_weight']:.2f}`, rest target `{policies['safety']['calibration_rest_false_activation_limit']:.0%}`, OOD quantile `{policies['safety']['ood_rest_quantile']:.2f}`.",
        "",
        "## Final Unseen-Subject Results",
        "",
        "| Measure | Single early, no OOD | Responsive policy | Safety policy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Balanced accuracy", "balanced_accuracy"),
        ("False activation rate", "false_activation_rate"),
        ("Missed intent rate", "missed_intent_rate"),
        ("OOD rejection rate", "mean_ood_window_rejection_rate"),
    ):
        lines.append(
            f"| {label} | {baseline[key] * 100:.1f}% | "
            f"{responsive[key] * 100:.1f}% | {safety[key] * 100:.1f}% |"
        )
    lines.append(
        "| False activations per minute | "
        f"{baseline['false_activations_per_minute']:.2f} | {responsive['false_activations_per_minute']:.2f} | {safety['false_activations_per_minute']:.2f} |"
    )
    latency = safety["median_true_intent_latency_seconds"]
    lines.extend(
        [
            "",
            f"Median accepted true-intent latency for the safety policy: `{latency:.2f} s`." if latency is not None else "No true intents were accepted by the safety policy.",
            "",
            "## Interpretation",
            "",
            "The comparison separates a responsive operating point from a stricter safety operating point. A lower false-activation rate is useful only if the missed-intent and latency costs remain acceptable. If OOD rejection is not selected, that is a negative result: this rest-distance definition did not improve the controller cohort under the tested policy grid. This is still a cue-locked windowed simulation, not a continuous clinical-device validation; its false-activations-per-minute value assumes three seconds of decision coverage per rest trial.",
            "",
            "## Outputs",
            "",
            "- `rest_aware_controller_overview.png`: method schematic.",
            "- `rest_aware_policy_frontier.png`: controller-subject policy selection.",
            "- `rest_aware_final_comparison.png`: baseline versus selected final metrics.",
            "- `rest_aware_final_subject_tradeoff.png`: final subject safety/sensitivity variation.",
            "- `rest_aware_controller_metrics.json`: reproducible metrics and selected policy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", default="artifacts/morlet_fullcue_audit/morlet_erd_trial_features.csv")
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--output-dir", default="artifacts/rest_aware_sequential_controller")
    parser.add_argument("--imagery-calibration", type=int, default=12)
    parser.add_argument("--rest-calibration", type=int, default=18)
    parser.add_argument("--max-controller-fp", type=float, default=0.15)
    parser.add_argument("--safety-controller-fp", type=float, default=0.05)
    parser.add_argument("--trial-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.imagery_calibration < 1 or args.rest_calibration < 2:
        raise ValueError("Imagery calibration must be positive and rest calibration must be at least two trials.")
    if not 0 < args.safety_controller_fp <= args.max_controller_fp < 1:
        raise ValueError("Safety and responsive controller false-activation limits must be in (0, 1).")

    configure_plot_style()
    phase_data, labels, subjects, runs, trials, feature_names = load_feature_table(Path(args.features_path))
    partitions = load_partitions(Path(args.splits))
    observed = set(np.unique(subjects).astype(int).tolist())
    development_subjects = tuple(subject for subject in partitions.development if subject in observed)
    controller_subjects = tuple(subject for subject in partitions.controller_validation if subject in observed)
    final_subjects = tuple(subject for subject in partitions.final_target if subject in observed)
    if len(controller_subjects) != len(partitions.controller_validation) or len(final_subjects) != len(partitions.final_target):
        raise RuntimeError("A locked controller or final subject is absent from the feature table.")
    development = np.isin(subjects, development_subjects)
    models: Dict[str, Pipeline] = {}
    for phase in PHASES:
        model = make_global_model(phase_data[phase].shape[1], args.seed)
        model.fit(phase_data[phase][development], labels[development])
        models[phase] = model

    def collect(subject_ids: Sequence[int]) -> Dict[int, Dict[str, PhaseScores]]:
        collected: Dict[int, Dict[str, PhaseScores]] = {}
        for subject in subject_ids:
            by_phase = {
                phase: phase_scores_for_subject(
                    models[phase], phase_data[phase], labels, subjects, runs, trials, subject,
                    args.imagery_calibration, args.rest_calibration, args.seed,
                )
                for phase in PHASES
            }
            if all(value is not None for value in by_phase.values()):
                collected[subject] = {phase: value for phase, value in by_phase.items() if value is not None}
        return collected

    controller_scores = collect(controller_subjects)
    final_scores = collect(final_subjects)
    excluded_controller_subjects = sorted(set(controller_subjects) - set(controller_scores))
    excluded_final_subjects = sorted(set(final_subjects) - set(final_scores))
    if excluded_final_subjects:
        raise RuntimeError(
            "A final subject lacks complete calibration or Run 12 labels: "
            f"{excluded_final_subjects}"
        )
    if excluded_controller_subjects:
        print(
            "[!] Excluding controller subjects without complete Run 12 rest/intent labels: "
            f"{excluded_controller_subjects}"
        )

    print("=== Rest-Aware Sequential Motor-Imagery Intent Gate ===")
    print(f"[+] Development subjects: {len(development_subjects)} | controller subjects: {len(controller_subjects)} | final subjects: {len(final_subjects)}")
    print(f"[+] Calibration: Runs 4/8, rest={args.rest_calibration}, imagery={args.imagery_calibration}; evaluation: Run 12")
    responsive_policy, frontier = choose_policy(
        controller_scores, args.max_controller_fp, args.trial_seconds
    )
    safety_policy, _ = choose_policy(
        controller_scores, args.safety_controller_fp, args.trial_seconds
    )
    baseline_policy = Policy("single_early", 0.70, 0.10, 1.0)
    controller_baseline = evaluate_policy(controller_scores, baseline_policy, args.trial_seconds)
    controller_responsive = evaluate_policy(controller_scores, responsive_policy, args.trial_seconds)
    controller_safety = evaluate_policy(controller_scores, safety_policy, args.trial_seconds)
    final_baseline = evaluate_policy(final_scores, baseline_policy, args.trial_seconds)
    final_responsive = evaluate_policy(final_scores, responsive_policy, args.trial_seconds)
    final_safety = evaluate_policy(final_scores, safety_policy, args.trial_seconds)
    for name, result in (
        ("single early/no OOD", final_baseline),
        ("responsive policy", final_responsive),
        ("safety policy", final_safety),
    ):
        metrics = result["mean_patient_metrics"]
        print(
            f"    {name:20s} | BA={metrics['balanced_accuracy'] * 100:.1f}% | "
            f"FP={metrics['false_activation_rate'] * 100:.1f}% | miss={metrics['missed_intent_rate'] * 100:.1f}% | "
            f"FA/min={metrics['false_activations_per_minute']:.2f}"
        )
    print(
        f"[+] Responsive: {responsive_policy.strategy}; blend={responsive_policy.adapter_weight:.2f}; "
        f"rest limit={responsive_policy.calibration_rest_false_activation_limit:.0%}; "
        f"OOD q={responsive_policy.ood_rest_quantile:.2f}"
    )
    print(
        f"[+] Safety: {safety_policy.strategy}; blend={safety_policy.adapter_weight:.2f}; "
        f"rest limit={safety_policy.calibration_rest_false_activation_limit:.0%}; "
        f"OOD q={safety_policy.ood_rest_quantile:.2f}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "protocol": {
            "task": "motor imagery rest versus intent gate; direction head intentionally unchanged",
            "development_subjects": list(development_subjects),
            "controller_validation_subjects": list(controller_subjects),
            "excluded_controller_validation_subjects": excluded_controller_subjects,
            "final_target_subjects": list(final_subjects),
            "calibration_runs": [4, 8],
            "evaluation_run": [12],
            "imagery_calibration_trials": args.imagery_calibration,
            "rest_calibration_trials": args.rest_calibration,
            "windows_seconds_after_cue": {"early": [0.5, 1.5], "middle": [1.5, 2.5], "terminal": [2.5, 3.5]},
            "controller_simulation": "sequential cue-locked windows; false activations per minute assumes 3 seconds per rest trial",
            "features": "phase-specific Morlet ERD plus peak-to-peak and RMS rest-context features",
            "ood_method": "patient rest-calibration Ledoit-Wolf Mahalanobis distance",
            "selection": "controller subjects only; maximize sensitivity under responsive and safety false-activation limits",
            "responsive_controller_false_activation_limit": args.max_controller_fp,
            "safety_controller_false_activation_limit": args.safety_controller_fp,
        },
        "feature_names": feature_names,
        "selected_policies": {"responsive": asdict(responsive_policy), "safety": asdict(safety_policy)},
        "controller_validation": {
            "single_window_no_ood": controller_baseline,
            "responsive_rest_aware_policy": controller_responsive,
            "safety_rest_aware_policy": controller_safety,
        },
        "final_target": {
            "single_window_no_ood": final_baseline,
            "responsive_rest_aware_policy": final_responsive,
            "safety_rest_aware_policy": final_safety,
        },
        "frontier": frontier,
    }
    with (output_dir / "rest_aware_controller_metrics.json").open("w", encoding="ascii") as handle:
        json.dump(report, handle, indent=2)
    joblib.dump(
        {"phase_models": models, "selected_policies": report["selected_policies"], "feature_names": feature_names},
        output_dir / "rest_aware_controller_models.joblib",
    )
    save_overview(output_dir)
    save_frontier(output_dir, frontier, responsive_policy, safety_policy)
    save_comparison(
        output_dir,
        {
            "Single early, no OOD": final_baseline,
            "Responsive selected": final_responsive,
            "Safety selected": final_safety,
        },
    )
    save_subject_tradeoff(output_dir, final_safety, "Safety Policy")
    write_report(output_dir / "REST_AWARE_SEQUENTIAL_CONTROLLER_RESULTS.md", report)
    print(f"[+] Saved metrics, models, report, and figures: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
