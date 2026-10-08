#!/usr/bin/env python3
"""Audit patient-calibrated quality rejection after the locked Cho2017 run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bci_pipeline.translational_dataset import load_translational_dataset  # noqa: E402
from bci_pipeline.translational_decoder import QualityAvailabilityGate  # noqa: E402


LOCKED_DIR = PROJECT_ROOT / "artifacts" / "cho2017_locked_external_v7"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cho2017_post_lock_quality_audit"
INK = "#17324D"
BLUE = "#176B87"
GREEN = "#218F7A"
ORANGE = "#E9A22B"
RED = "#D96832"
PALE = "#E8EEF1"


def _score(
    labels: np.ndarray,
    probability: np.ndarray,
    available: np.ndarray,
    threshold: float,
    selected: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(labels)[selected] != 0
    scores = np.asarray(probability)[selected]
    local_available = np.asarray(available)[selected]
    issued = local_available & (scores >= threshold)
    sensitivity = float(np.mean(issued[y]))
    specificity = float(np.mean(~issued[~y]))
    return {
        "rows": int(np.sum(selected)),
        "threshold": float(threshold),
        "intent_balanced_accuracy": (sensitivity + specificity) / 2.0,
        "intent_sensitivity": sensitivity,
        "rest_specificity": specificity,
        "false_activation_fraction": 1.0 - specificity,
        "missed_intent_fraction": 1.0 - sensitivity,
        "available_fraction": float(np.mean(local_available)),
        "abstained_fraction": float(np.mean(~local_available)),
        "auroc": float(roc_auc_score(y.astype(np.int64), scores)),
    }


def _patient_calibrated_availability(
    dataset: object,
    quantile: float = 0.99,
) -> np.ndarray:
    available = np.zeros(len(dataset.labels), dtype=bool)
    for subject in np.unique(dataset.subject_keys):
        calibration = np.flatnonzero(
            (dataset.subject_keys == subject) & (dataset.recordings == 0)
        )
        evaluation = np.flatnonzero(
            (dataset.subject_keys == subject) & (dataset.recordings != 0)
        )
        gate = QualityAvailabilityGate(quantile).fit(
            dataset.quality_features[calibration]
        )
        available[evaluation] = gate.available(
            dataset.quality_features[evaluation]
        )
    return available


def _bootstrap(
    dataset: object,
    probability: np.ndarray,
    available: np.ndarray,
    threshold: float,
    selected: np.ndarray,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    subjects = np.unique(dataset.subject_keys[selected])
    by_subject = {
        subject: np.flatnonzero(selected & (dataset.subject_keys == subject))
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    names = (
        "intent_balanced_accuracy",
        "intent_sensitivity",
        "rest_specificity",
        "false_activation_fraction",
        "available_fraction",
    )
    values = {name: [] for name in names}
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([by_subject[subject] for subject in sampled])
        local = _score(
            dataset.labels[indices],
            probability[indices],
            available[indices],
            threshold,
            np.ones(len(indices), dtype=bool),
        )
        for name in names:
            values[name].append(float(local[name]))
    return {
        name: {
            "low": float(np.percentile(samples, 2.5)),
            "high": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def _plot(path: Path, results: dict[str, dict[str, float]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    configurations = (
        ("Frozen OOD gate\nconservative", results["locked_conservative"], BLUE),
        ("Patient-calibrated gate\nconservative", results["recalibrated_conservative"], ORANGE),
        ("Patient-calibrated gate\nsensitivity-targeted", results["recalibrated_sensitivity_targeted"], GREEN),
        ("Gate bypass diagnostic\nsensitivity-targeted", results["bypass_sensitivity_targeted"], RED),
    )
    metrics = (
        ("Balanced accuracy", "intent_balanced_accuracy"),
        ("Intent sensitivity", "intent_sensitivity"),
        ("Rest specificity", "rest_specificity"),
        ("Availability", "available_fraction"),
    )
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    for index, (label, values, color) in enumerate(configurations):
        heights = [100 * float(values[key]) for _, key in metrics]
        bars = ax.bar(
            x + (index - 1.5) * width,
            heights,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9, fontweight="bold")
    ax.set_xticks(x, [label for label, _ in metrics])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Result (%)")
    ax.set_title(
        "Post-Lock Audit: External Failure Was Dominated by the Frozen OOD Gate",
        color=INK,
        fontweight="bold",
    )
    ax.grid(axis="y", color=PALE)
    ax.legend(
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics_path = LOCKED_DIR / "cho2017_v7_locked_metrics.json"
    predictions_path = LOCKED_DIR / "cho2017_v7_locked_predictions.csv"
    cache_path = LOCKED_DIR / "cho2017_v7_windows.npz"
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Run the locked Cho2017 V7 evaluation first.")
    locked = json.loads(metrics_path.read_text(encoding="utf-8"))
    dataset = load_translational_dataset(cache_path)
    if dataset is None:
        raise RuntimeError("The locked Cho2017 window cache is unavailable.")
    predictions = pd.read_csv(predictions_path)
    if len(predictions) != len(dataset.labels):
        raise RuntimeError("Locked predictions and window cache are not aligned.")
    probability = predictions["adapted_probability"].to_numpy(dtype=np.float64)
    selected = predictions["adapted_evaluation_row"].to_numpy(dtype=bool)
    frozen_available = predictions["available"].to_numpy(dtype=bool)
    patient_available = _patient_calibrated_availability(dataset)
    bypass_available = np.ones(len(dataset.labels), dtype=bool)
    thresholds = locked["thresholds"]

    results = {
        "locked_conservative": _score(
            dataset.labels,
            probability,
            frozen_available,
            float(thresholds["conservative"]),
            selected,
        ),
        "recalibrated_conservative": _score(
            dataset.labels,
            probability,
            patient_available,
            float(thresholds["conservative"]),
            selected,
        ),
        "recalibrated_sensitivity_targeted": _score(
            dataset.labels,
            probability,
            patient_available,
            float(thresholds["sensitivity_targeted"]),
            selected,
        ),
        "bypass_sensitivity_targeted": _score(
            dataset.labels,
            probability,
            bypass_available,
            float(thresholds["sensitivity_targeted"]),
            selected,
        ),
    }
    primary = results["recalibrated_sensitivity_targeted"]
    confidence = _bootstrap(
        dataset,
        probability,
        patient_available,
        float(thresholds["sensitivity_targeted"]),
        selected,
    )
    report = {
        "experiment": "cho2017_post_lock_quality_audit",
        "status": "exploratory_post_lock_not_confirmatory",
        "locked_result_unchanged": True,
        "quality_calibration": {
            "inputs": "30 frozen patient-calibration windows only",
            "evaluation_labels_used": False,
            "gate": "QualityAvailabilityGate",
            "rejection_quantile": 0.99,
        },
        "results": results,
        "participant_bootstrap_95ci": confidence,
        "interpretation": [
            "The frozen population OOD gate rejected every Cho2017 evaluation window.",
            "Patient-calibrated quality normalization restored availability without evaluation labels.",
            "This audit was designed after observing the locked failure and is exploratory.",
            "A future external cohort must pre-register this quality-calibration step before inference.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cho2017_post_lock_quality_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="ascii",
    )
    _plot(OUTPUT_DIR / "01_quality_gate_recalibration.png", results)
    markdown = "\n".join(
        [
            "# Cho2017 Post-Lock Quality-Gate Audit",
            "",
            "> Exploratory only. The locked result remains 0% sensitivity because the",
            "> frozen OOD gate abstained on 100% of external windows.",
            "",
            "## Calibration-Only Diagnostic",
            "",
            "| Configuration | BA | Sensitivity | Specificity | False activation/rest window | Availability | AUROC |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {name.replace('_', ' ').title()} | {values['intent_balanced_accuracy']:.1%} | "
                f"{values['intent_sensitivity']:.1%} | {values['rest_specificity']:.1%} | "
                f"{values['false_activation_fraction']:.1%} | {values['available_fraction']:.1%} | "
                f"{values['auroc']:.3f} |"
                for name, values in results.items()
            ],
            "",
            "The patient-calibrated gate was fitted from each participant's 30 reserved",
            "calibration windows only. No evaluation labels or evaluation probabilities",
            "were used to fit it. Nevertheless, this method was proposed after the locked",
            "failure was observed, so these values are not confirmatory.",
            "",
            "## Recommended Protocol Change",
            "",
            "Pre-register hardware/session-specific quality normalization as part of the",
            "30-window calibration session, while preserving the population intent model,",
            "probability calibration, and operating thresholds. Confirm that revision on a",
            "new untouched dataset or later-session cohort.",
            "",
        ]
    )
    (OUTPUT_DIR / "CHO2017_POST_LOCK_QUALITY_AUDIT.md").write_text(
        markdown,
        encoding="ascii",
    )
    print("\n=== Cho2017 Post-Lock Quality Audit ===")
    print(
        "Patient-calibrated gate + sensitivity threshold | "
        f"BA={primary['intent_balanced_accuracy']:.1%} | "
        f"sensitivity={primary['intent_sensitivity']:.1%} | "
        f"specificity={primary['rest_specificity']:.1%} | "
        f"availability={primary['available_fraction']:.1%}"
    )
    print(f"[+] Saved exploratory audit to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
