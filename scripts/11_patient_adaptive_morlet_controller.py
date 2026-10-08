#!/usr/bin/env python3
"""Select a patient-adaptive Morlet-ERD intent controller on locked subjects.

Runs 4/8 provide unsupervised feature alignment and small labeled calibration.
Run 12 is reserved for each patient's evaluation. The controller cohort chooses
the time window, adaptation blend, safety quantile, and optional 2-of-3 vote.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.splits import load_partitions  # noqa: E402


PHASES = ("early", "middle", "terminal", "full_cue")
TEMPORAL_PHASES = ("early", "middle", "terminal")


@dataclass(frozen=True)
class Policy:
    strategy: str
    adapter_weight: float
    calibration_rest_false_activation_limit: float


@dataclass
class PhaseScores:
    labels: np.ndarray
    global_eval: np.ndarray
    local_eval: np.ndarray
    global_calibration: np.ndarray
    local_calibration: np.ndarray
    calibration_labels: np.ndarray


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Feature table has no header.")
        names = [
            name
            for name in reader.fieldnames
            if name.endswith("_erd_db") or name.startswith("c3_minus_c4_") and name.endswith("_db")
        ]
        rows = list(reader)
    features = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["label"]) != 0 for row in rows], dtype=np.int64)
    subjects = np.asarray([int(row["subject"]) for row in rows], dtype=np.int64)
    runs = np.asarray([int(row["run"]) for row in rows], dtype=np.int64)
    return features, labels, subjects, runs, names


def phase_indices(names: Sequence[str], phase: str) -> np.ndarray:
    suffixes = (f"_{phase}_erd_db", f"_{phase}_db")
    indices = np.asarray([index for index, name in enumerate(names) if name.endswith(suffixes)])
    if not len(indices):
        raise ValueError(f"No features for phase '{phase}'. Re-run the extended Morlet audit first.")
    return indices


def make_global_model(feature_count: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selection", SelectKBest(f_classif, k=min(40, feature_count))),
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


def transform_for_local_model(model: Pipeline, values: np.ndarray) -> np.ndarray:
    values = model.named_steps["imputer"].transform(values)
    values = model.named_steps["scaler"].transform(values)
    return model.named_steps["selection"].transform(values)


def target_align(
    calibration: np.ndarray,
    values: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
) -> np.ndarray:
    target_mean = np.mean(calibration, axis=0)
    target_std = np.clip(np.std(calibration, axis=0), 1e-6, None)
    return ((values - target_mean) / target_std) * source_std + source_mean


def select_calibration(
    labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    subject: int,
    imagery_budget: int,
    rest_budget: int,
) -> np.ndarray:
    selected: List[int] = []
    for label, budget in ((0, rest_budget), (1, imagery_budget)):
        candidates = np.flatnonzero((subjects == subject) & np.isin(runs, (4, 8)) & (labels == label))
        if len(candidates) < budget:
            return np.asarray([], dtype=np.int64)
        selected.extend(candidates[:budget].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def score_patient_phase(
    model: Pipeline,
    phase_data: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    subject: int,
    imagery_budget: int,
    rest_budget: int,
    seed: int,
) -> PhaseScores | None:
    calibration_idx = select_calibration(
        labels, subjects, runs, subject, imagery_budget, rest_budget
    )
    evaluation_idx = np.flatnonzero((subjects == subject) & (runs == 12))
    if not len(calibration_idx) or not len(evaluation_idx):
        return None
    if set(labels[evaluation_idx].tolist()) != {0, 1}:
        return None
    source_mean = model.named_steps["scaler"].mean_
    source_std = np.sqrt(model.named_steps["scaler"].var_)
    source_std = np.clip(source_std, 1e-6, None)
    aligned_calibration = target_align(
        phase_data[calibration_idx], phase_data[calibration_idx], source_mean, source_std
    )
    aligned_evaluation = target_align(
        phase_data[calibration_idx], phase_data[evaluation_idx], source_mean, source_std
    )
    global_calibration = model.predict_proba(aligned_calibration)[:, 1]
    global_evaluation = model.predict_proba(aligned_evaluation)[:, 1]
    local_model = LogisticRegression(
        C=0.05,
        class_weight="balanced",
        max_iter=3000,
        random_state=seed + subject,
    ).fit(transform_for_local_model(model, aligned_calibration), labels[calibration_idx])
    local_calibration = local_model.predict_proba(transform_for_local_model(model, aligned_calibration))[:, 1]
    local_evaluation = local_model.predict_proba(transform_for_local_model(model, aligned_evaluation))[:, 1]
    return PhaseScores(
        labels=labels[evaluation_idx],
        global_eval=global_evaluation,
        local_eval=local_evaluation,
        global_calibration=global_calibration,
        local_calibration=local_calibration,
        calibration_labels=labels[calibration_idx],
    )


def thresholds_and_predictions(
    scores: PhaseScores,
    weight: float,
    rest_limit: float,
) -> tuple[np.ndarray, float]:
    calibration = (1.0 - weight) * scores.global_calibration + weight * scores.local_calibration
    evaluation = (1.0 - weight) * scores.global_eval + weight * scores.local_eval
    threshold = float(
        np.nextafter(
            np.quantile(calibration[scores.calibration_labels == 0], 1.0 - rest_limit, method="higher"),
            np.inf,
        )
    )
    return evaluation >= threshold, threshold


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    rest, intent = labels == 0, labels == 1
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "false_activation_rate": float(np.mean(predictions[rest])),
        "missed_intent_rate": float(np.mean(~predictions[intent])),
        "intent_sensitivity": float(np.mean(predictions[intent])),
        "command_rate": float(np.mean(predictions)),
    }


def evaluate_policy(
    scores: Dict[int, Dict[str, PhaseScores]], policy: Policy
) -> Dict[str, object]:
    per_subject: Dict[str, Dict[str, float]] = {}
    for subject, by_phase in scores.items():
        if policy.strategy == "temporal_2of3":
            phase_predictions = [
                thresholds_and_predictions(
                    by_phase[phase], policy.adapter_weight, policy.calibration_rest_false_activation_limit
                )[0]
                for phase in TEMPORAL_PHASES
            ]
            predictions = np.sum(phase_predictions, axis=0) >= 2
            labels = by_phase[TEMPORAL_PHASES[0]].labels
        else:
            predictions, _ = thresholds_and_predictions(
                by_phase[policy.strategy],
                policy.adapter_weight,
                policy.calibration_rest_false_activation_limit,
            )
            labels = by_phase[policy.strategy].labels
        per_subject[f"S{subject:03d}"] = binary_metrics(labels, predictions)
    mean_metrics = {
        key: float(np.mean([item[key] for item in per_subject.values()]))
        for key in next(iter(per_subject.values()))
    }
    return {"policy": asdict(policy), "mean_patient_metrics": mean_metrics, "per_subject": per_subject}


def choose_policy(scores: Dict[int, Dict[str, PhaseScores]], max_fp: float) -> tuple[Policy, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    for strategy in (*PHASES, "temporal_2of3"):
        for weight in (0.0, 0.35, 0.7):
            for rest_limit in (0.05, 0.10, 0.15, 0.20):
                policy = Policy(strategy, weight, rest_limit)
                result = evaluate_policy(scores, policy)
                metrics = result["mean_patient_metrics"]
                rows.append({"policy": asdict(policy), **metrics})
    feasible = [row for row in rows if row["false_activation_rate"] <= max_fp]
    pool = feasible or rows
    selected = max(
        pool,
        key=lambda row: (
            row["intent_sensitivity"],
            row["balanced_accuracy"],
            -row["false_activation_rate"],
        ),
    )
    return Policy(**selected["policy"]), rows


def save_plots(output_dir: Path, frontier: List[Dict[str, object]], final: Dict[str, object]) -> None:
    selected = final["policy"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    strategies = sorted({row["policy"]["strategy"] for row in frontier})
    palette = dict(zip(strategies, ("#39739d", "#5aa469", "#bd7b31", "#9b5d91", "#555555")))
    for strategy in strategies:
        rows = [row for row in frontier if row["policy"]["strategy"] == strategy]
        ax.scatter(
            [100 * row["false_activation_rate"] for row in rows],
            [100 * row["intent_sensitivity"] for row in rows],
            label=strategy,
            alpha=0.55,
            color=palette[strategy],
        )
    selected_row = next(row for row in frontier if row["policy"] == selected)
    ax.scatter(
        100 * selected_row["false_activation_rate"],
        100 * selected_row["intent_sensitivity"],
        s=140,
        color="black",
        marker="*",
        label="selected",
        zorder=5,
    )
    ax.set_xlabel("Controller false activation rate (%)")
    ax.set_ylabel("Controller intent sensitivity (%)")
    ax.set_title("Morlet time-window and temporal-vote selection")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "morlet_patient_controller_frontier.png", dpi=180)
    plt.close(fig)

    metrics = final["mean_patient_metrics"]
    labels = ["Balanced\naccuracy", "False\nactivation", "Missed\nintent", "Sensitivity"]
    keys = ["balanced_accuracy", "false_activation_rate", "missed_intent_rate", "intent_sensitivity"]
    values = [100 * metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color=["#39739d", "#b65c4a", "#b65c4a", "#5aa469"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean final-patient rate (%)")
    ax.set_title(f"Final patient-adaptive Morlet controller: {selected['strategy']}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "morlet_patient_controller_final.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path", default="artifacts/morlet_fullcue_audit/morlet_erd_trial_features.csv"
    )
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--output-dir", default="artifacts/morlet_patient_controller")
    parser.add_argument("--imagery-calibration", type=int, default=12)
    parser.add_argument("--rest-calibration", type=int, default=18)
    parser.add_argument("--max-controller-fp", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, labels, subjects, runs, names = load_table(Path(args.features_path))
    partitions = load_partitions(Path(args.splits))
    development = np.isin(subjects, partitions.development)
    phase_models: Dict[str, Pipeline] = {}
    phase_data: Dict[str, np.ndarray] = {}
    for phase in PHASES:
        data = features[:, phase_indices(names, phase)]
        model = make_global_model(data.shape[1], args.seed)
        model.fit(data[development], labels[development])
        phase_data[phase], phase_models[phase] = data, model

    def collect(patient_ids: Sequence[int]) -> Dict[int, Dict[str, PhaseScores]]:
        result: Dict[int, Dict[str, PhaseScores]] = {}
        for subject in patient_ids:
            scores = {
                phase: score_patient_phase(
                    phase_models[phase],
                    phase_data[phase],
                    labels,
                    subjects,
                    runs,
                    subject,
                    args.imagery_calibration,
                    args.rest_calibration,
                    args.seed,
                )
                for phase in PHASES
            }
            if all(value is not None for value in scores.values()):
                result[subject] = {phase: value for phase, value in scores.items() if value is not None}
        return result

    controller_scores = collect(partitions.controller_validation)
    final_scores = collect(partitions.final_target)
    if not controller_scores or not final_scores:
        raise RuntimeError("No complete patient records were available for controller selection/evaluation.")
    policy, frontier = choose_policy(controller_scores, args.max_controller_fp)
    controller_result = evaluate_policy(controller_scores, policy)
    final_result = evaluate_policy(final_scores, policy)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "task": "rest versus unilateral motor imagery intent",
            "calibration_runs": [4, 8],
            "evaluation_run": [12],
            "imagery_trials_per_calibration_class": args.imagery_calibration,
            "rest_calibration_trials": args.rest_calibration,
            "target_alignment": "unlabeled target mean/standard deviation from calibration recordings",
            "selection_cohort": "controller-validation subjects only",
            "final_cohort": "untouched final-target subjects",
        },
        "selected_policy": asdict(policy),
        "controller_validation": controller_result,
        "final_target": final_result,
        "frontier": frontier,
    }
    with (output_dir / "morlet_patient_controller_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    joblib.dump(
        {"phase_models": phase_models, "policy": policy, "phase_feature_names": {phase: [names[index] for index in phase_indices(names, phase)] for phase in PHASES}},
        output_dir / "morlet_patient_controller_models.joblib",
    )
    save_plots(output_dir, frontier, final_result)
    metrics = final_result["mean_patient_metrics"]
    print("=== Patient-Adaptive Morlet Intent Controller ===")
    print(f"[+] Selected strategy: {policy.strategy}; adapter weight={policy.adapter_weight:.2f}; rest limit={policy.calibration_rest_false_activation_limit:.0%}")
    print(
        f"[+] Final mean patient BA={metrics['balanced_accuracy'] * 100:.1f}% | "
        f"FP={metrics['false_activation_rate'] * 100:.1f}% | "
        f"miss={metrics['missed_intent_rate'] * 100:.1f}%"
    )
    print(f"[+] Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
