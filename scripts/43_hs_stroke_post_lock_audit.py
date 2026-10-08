#!/usr/bin/env python3
"""Explain a completed HS-Stroke lock result without changing it."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from bci_pipeline.translational_dataset import load_translational_dataset


OUTPUT = PROJECT_ROOT / "artifacts" / "hs_stroke_locked_external"
MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "prospective_cue_controller"
    / "frozen_model"
    / "translational_v4_neural_models.joblib"
)


def main() -> None:
    metrics = json.loads(
        (OUTPUT / "hs_stroke_locked_external_metrics.json").read_text(encoding="utf-8")
    )
    dataset = load_translational_dataset(OUTPUT / "hs_stroke_locked_windows.npz")
    if dataset is None:
        raise RuntimeError("Locked window cache is unavailable.")
    bundle = joblib.load(MODEL)
    gate = bundle["quality_gate"]
    scores = gate.score(dataset.quality_features, dataset.dataset_ids)
    median, scale, threshold = gate.statistics_["__global__"]
    component_z = np.median(
        np.abs((dataset.quality_features - median) / scale),
        axis=0,
    )

    with (OUTPUT / "hs_stroke_locked_predictions.csv").open(
        "r", encoding="ascii", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    labels = np.asarray([int(row["true_numeric_label"]) for row in rows])
    intent = np.asarray([float(row["intent_probability"]) for row in rows])
    direction = np.asarray(
        [
            1
            if float(row["class1_probability"]) >= float(row["class2_probability"])
            else 2
            for row in rows
        ]
    )
    thresholds = metrics["thresholds"]
    ungated = {}
    for name, value in thresholds.items():
        issued = intent >= value
        ungated[name] = {
            "intent_sensitivity": float(np.mean(issued[labels != 0])),
            "preparation_specificity": float(np.mean(~issued[labels == 0])),
            "false_activation_fraction": float(np.mean(issued[labels == 0])),
        }
    direction_ba = float(
        balanced_accuracy_score(labels[labels != 0], direction[labels != 0])
    )
    theoretical_windows = 2 * 23382
    qc = metrics["quality_control"]
    audit = {
        "status": "post_lock_diagnostic_not_confirmatory_and_not_for_selection",
        "locked_metrics_sha256": json.loads(
            (OUTPUT / "locked_execution_receipt.json").read_text(encoding="utf-8")
        )["metrics_sha256"],
        "sessions_total": 278,
        "sessions_accepted_by_recording_qc": int(qc["accepted_recordings"]),
        "sessions_rejected_by_recording_qc": int(qc["excluded_recordings_or_subjects"]),
        "windows_theoretical": theoretical_windows,
        "windows_retained": int(len(dataset.labels)),
        "window_retention_fraction": float(len(dataset.labels) / theoretical_windows),
        "frozen_quality_gate": {
            "threshold": float(threshold),
            "minimum_external_score": float(np.min(scores)),
            "median_external_score": float(np.median(scores)),
            "available_fraction": float(np.mean(scores <= threshold)),
            "component_median_absolute_z": {
                name: float(value)
                for name, value in zip(dataset.quality_feature_names, component_z)
            },
            "root_cause": (
                "auxiliary_available and csd_available were invariant at one in the "
                "frozen gate reference but are zero for HS-Stroke"
            ),
        },
        "frozen_probabilities_post_hoc_description": {
            "intent_auroc": metrics["intent_auroc_descriptive_not_for_selection"],
            "numeric_direction_balanced_accuracy": direction_ba,
            "rest_quantiles_05_50_95": np.quantile(
                intent[labels == 0], [0.05, 0.5, 0.95]
            ).tolist(),
            "imagery_quantiles_05_50_95": np.quantile(
                intent[labels != 0], [0.05, 0.5, 0.95]
            ).tolist(),
            "ungated_frozen_thresholds": ungated,
        },
        "interpretation": (
            "The frozen system failed external transfer. The availability gate rejects "
            "the montage deterministically, while bypassing it would issue commands on "
            "almost every rest window because probabilities are shifted high and only "
            "weakly discriminate imagery from preparation."
        ),
    }
    (OUTPUT / "POST_LOCK_DIAGNOSTIC_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="ascii",
    )
    markdown = f"""# HS-Stroke Post-Lock Diagnostic Audit

> This document explains the immutable locked result. It is post hoc, is not a
> second confirmatory analysis, and must not be used to replace the saved locked
> metrics.

## What Happened

- The corrected release supplied 278 sessions from 14 patients.
- Frozen recording QC accepted **{qc['accepted_recordings']}/278 ({qc['accepted_recordings']/278:.1%})** sessions.
- The hard window rules retained **{len(dataset.labels):,}/{theoretical_windows:,} ({len(dataset.labels)/theoretical_windows:.1%})** theoretical paired windows.
- The frozen availability gate accepted **0.0%** of retained windows.
- Consequently, both locked operating points issued zero commands: 0% sensitivity, 100% structured-rest specificity, and 50% binary balanced accuracy.

## Root Cause

The gate treated `auxiliary_available` and `csd_available` as continuous quality
features. Both had training median 1 and a clipped MAD scale of `1e-6`. HS-Stroke
has no auxiliary EOG/EMG channels and only 11 EEG channels, so both indicators are
0. Each becomes a **1,000,000-MAD** deviation, forcing every quality score above
the frozen threshold ({threshold:,.1f}). This is a quality-gate design flaw exposed
by the new montage, not evidence that every HS window is physiologically unusable.

## Why Removing the Gate Is Not Enough

- Descriptive intent AUROC: **{metrics['intent_auroc_descriptive_not_for_selection']:.1%}**.
- Numeric class-1/class-2 direction BA: **{direction_ba:.1%}**.
- Ungated conservative sensitivity/specificity: **{ungated['conservative']['intent_sensitivity']:.1%} / {ungated['conservative']['preparation_specificity']:.2%}**.
- Ungated sensitivity-targeted sensitivity/specificity: **{ungated['sensitivity_targeted']['intent_sensitivity']:.1%} / {ungated['sensitivity_targeted']['preparation_specificity']:.2%}**.

The external probabilities are shifted high for both preparation and imagery.
Bypassing the gate would therefore replace total abstention with near-continuous
false activation, while discrimination remains near chance.

## Research Consequence

HS-Stroke is now an observed development/adaptation cohort. We can use it to fix
modality-aware quality gating, domain calibration, lesion-aware normalization, and
early-session patient adaptation, but those changes require a different untouched
dataset or a prospective cohort for confirmation.
"""
    (OUTPUT / "POST_LOCK_DIAGNOSTIC_AUDIT.md").write_text(markdown, encoding="ascii")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    order = np.argsort(component_z)[::-1]
    names = np.asarray(dataset.quality_feature_names)[order]
    values = component_z[order]
    axes[0].barh(np.arange(len(names)), values, color=["#C64B4B" if value > 100 else "#176B87" for value in values])
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Median absolute deviation from frozen reference (log scale)")
    axes[0].set_title("Why the Quality Gate Rejected Every Window")
    axes[0].grid(axis="x", color="#EAF1F5")
    axes[1].boxplot(
        [intent[labels == 0], intent[labels != 0]],
        tick_labels=["Preparation rest", "Motor imagery"],
        patch_artist=True,
        boxprops={"facecolor": "#D9EAF0"},
        medianprops={"color": "#17324D", "linewidth": 2},
        showfliers=False,
    )
    axes[1].axhline(thresholds["conservative"], color="#176B87", linestyle="--", label="Conservative threshold")
    axes[1].axhline(thresholds["sensitivity_targeted"], color="#E07A2F", linestyle=":", label="Sensitivity-targeted threshold")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("Frozen calibrated intent probability")
    axes[1].set_title("High Probability Shift Without Separation")
    axes[1].grid(axis="y", color="#EAF1F5")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("HS-Stroke Locked-Result Diagnostic", fontsize=15, color="#17324D")
    fig.tight_layout()
    fig.savefig(OUTPUT / "05_post_lock_failure_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved post-lock audit to {OUTPUT}")


if __name__ == "__main__":
    main()
