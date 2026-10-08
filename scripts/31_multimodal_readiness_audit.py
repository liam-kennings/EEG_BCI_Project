#!/usr/bin/env python3
"""Create a reproducible EEG-EMG translational-readiness audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bci_pipeline.dataset_registry import dataset_catalog
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v3_calibrated"
    / "translational_v3_metrics.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "multimodal_readiness"

INK = "#17324D"
TEAL = "#087E8B"
GOLD = "#D99B24"
CORAL = "#C64B42"
GREEN = "#2D7D5B"
PALE = "#EEF3F5"


@dataclass(frozen=True)
class AuditCheck:
    key: str
    label: str
    status: str
    evidence: str
    next_action: str

    @property
    def score(self) -> float:
        return {"ready": 1.0, "prepared": 0.5, "gap": 0.0}[self.status]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["patient_calibration"]["development_observed_conservative"][key])


def _build_checks(metrics: dict[str, Any]) -> list[AuditCheck]:
    quality_source = (PROJECT_ROOT / "bci_pipeline" / "quality_preprocessing.py").read_text(
        encoding="utf-8"
    )
    openbmi_source = (PROJECT_ROOT / "bci_pipeline" / "openbmi.py").read_text(
        encoding="utf-8"
    )
    script_source = (
        PROJECT_ROOT / "scripts" / "30_translational_motor_intent_v3.py"
    ).read_text(encoding="utf-8")
    locked = bool(metrics["confirmatory_status"]["locked_external_available"])
    continuous = "not evaluated" not in metrics["continuous_no_control_status"].lower()
    causal = str(metrics["preprocessing"]["phase"]).lower() != "zero"
    modality_specific = (
        "emg_available" in quality_source and "eog_available" in quality_source
    )
    quality_threshold = float(metrics["quality_gate"]["threshold"])
    robust_quality = quality_threshold < 100.0
    nested = "split_controller_subroles" in script_source
    has_openbmi_emg = "EXPECTED_EMG_CHANNELS = 4" in openbmi_source
    multimodal_sources = [
        dataset for dataset in dataset_catalog() if "EMG" in dataset.modalities
    ]
    return [
        AuditCheck(
            "subject_isolation",
            "Subject-isolated development roles",
            "ready",
            f"{metrics['dataset']['subjects']} subjects are assigned by subject role.",
            "Preserve subject and session grouping in every new adapter.",
        ),
        AuditCheck(
            "nested_controller",
            "Nested controller selection",
            "ready" if nested else "gap",
            "Fusion/calibration and operating-point selection use separate controller subroles."
            if nested
            else "No controller subrole split was found.",
            "Keep all fusion, calibration, and threshold choices outside observed test subjects.",
        ),
        AuditCheck(
            "native_qc",
            "Full-montage QC before reference",
            "ready",
            f"{metrics['dataset']['quality_control']['accepted_recordings']} recordings accepted; "
            f"{metrics['dataset']['quality_control']['excluded_recordings_or_subjects']} excluded.",
            "Continue reporting exclusions and sensitivity to QC thresholds.",
        ),
        AuditCheck(
            "openbmi_emg",
            "Existing synchronized EMG source",
            "prepared" if has_openbmi_emg else "gap",
            "OpenBMI loader recognizes four EMG channels, but v3 stores only aggregate auxiliary quality."
            if has_openbmi_emg
            else "No synchronized EMG source is recognized locally.",
            "Retain EMG waveforms and derive silence, onset, RMS, and waveform-length labels.",
        ),
        AuditCheck(
            "multimodal_catalog",
            "External multimodal teacher catalog",
            "prepared" if len(multimodal_sources) >= 5 else "gap",
            f"{len(multimodal_sources)} EEG-EMG sources are cataloged with modality-specific roles.",
            "Implement the MRCP EEG-EMG adapter first; do not pool execution labels with imagery.",
        ),
        AuditCheck(
            "modality_identity",
            "Modality-specific auxiliary processing",
            "ready" if modality_specific else "gap",
            "EMG and EOG are currently merged into generic auxiliary peak-to-peak/RMS features.",
            "Split EMG, EOG, and device channels before feature extraction and gating.",
        ),
        AuditCheck(
            "robust_quality_scaling",
            "Robustly scaled quality gate",
            "ready" if robust_quality else "gap",
            f"The current heterogeneous-distance threshold is {quality_threshold:,.1f}, indicating scale domination.",
            "Fit median/IQR scaling per feature and report per-domain rejection rates.",
        ),
        AuditCheck(
            "causal_preprocessing",
            "Causal deployable preprocessing",
            "ready" if causal else "gap",
            f"Final reported profile uses phase={metrics['preprocessing']['phase']} filtering.",
            "Re-run the locked candidate with causal filters and measure added latency/performance loss.",
        ),
        AuditCheck(
            "continuous_safety",
            "Continuous no-control safety",
            "ready" if continuous else "gap",
            metrics["continuous_no_control_status"],
            "Add continuous idle, gaze, speech, jaw, posture, and device-motion hard negatives.",
        ),
        AuditCheck(
            "locked_external",
            "Locked external confirmation",
            "ready" if locked else "gap",
            "A locked external cohort is available."
            if locked
            else "The final cohort remains development-observed.",
            "Freeze v4, then evaluate once on untouched subjects/sessions.",
        ),
    ]


def _plot_scorecard(checks: list[AuditCheck], output: Path) -> None:
    labels = [check.label for check in checks][::-1]
    scores = [check.score for check in checks][::-1]
    colors = [
        GREEN if check.status == "ready" else GOLD if check.status == "prepared" else CORAL
        for check in checks
    ][::-1]
    fig, ax = plt.subplots(figsize=(12, 7.2), constrained_layout=True)
    ax.barh(labels, scores, color=colors, height=0.62)
    ax.set_xlim(0, 1.05)
    ax.set_xticks((0, 0.5, 1.0), ("Gap", "Prepared", "Ready"))
    ax.set_title("Motor-Intent Pipeline Translational Readiness", loc="left", weight="bold", color=INK)
    ax.set_xlabel("Implementation status", color=INK)
    ax.grid(axis="x", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=INK)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(output / "01_multimodal_readiness_scorecard.png", dpi=220)
    plt.close(fig)


def _plot_context_table(metrics: dict[str, Any], output: Path) -> None:
    rows = [
        [
            "This project v3",
            f"{100 * _metric(metrics, 'intent_balanced_accuracy'):.1f}% BA",
            "Intent vs rest; patient-adapted; development-observed",
            "94.5% cue-locked specificity; continuous FPM unavailable",
        ],
        [
            "OpenBMI reference",
            "71.1% accuracy",
            "Within-subject left/right imagery; all 54 participants",
            "Different endpoint and validation design",
        ],
        [
            "Severe-stroke EEG+EMG",
            "69.7% accuracy",
            "Within-patient attempted movement vs rest; 35 patients",
            "73.8% among responders; fusion includes peripheral evidence",
        ],
        [
            "HS-Stroke report",
            "82.65% accuracy",
            "Cross-trial left/right imagery; 14 patients, many sessions",
            "Not subject- or session-held-out confirmation",
        ],
        [
            "Cue-gated async BCI",
            "84.88% TPR",
            "Event detection during enabled cue intervals; 9 patients",
            "0.52 false activations/minute",
        ],
    ]
    fig, ax = plt.subplots(figsize=(15, 5.2), constrained_layout=True)
    ax.axis("off")
    ax.set_title(
        "Published Context: Similar Percentages, Different Scientific Questions",
        loc="left",
        weight="bold",
        color=INK,
        pad=16,
    )
    table = ax.table(
        cellText=rows,
        colLabels=("Study", "Reported result", "Endpoint / validation", "Interpretation"),
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=(0.17, 0.13, 0.34, 0.36),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PALE if row % 2 else "white")
            cell.get_text().set_color(INK)
            if column == 1:
                cell.get_text().set_weight("bold")
    fig.text(
        0.01,
        0.01,
        "Values are contextual only and must not be ranked as a common leaderboard.",
        color=CORAL,
        fontsize=9,
        weight="bold",
    )
    fig.savefig(output / "02_published_results_context.png", dpi=220)
    plt.close(fig)


def _plot_dataset_matrix(output: Path) -> None:
    selected_keys = (
        "mrcp_eeg_emg_2025",
        "way_eeg_gal",
        "wristmotion_2026",
        "nebula_reaching_2025",
        "prehension_eeg_emg_2026",
        "stroke_attempt_eeg_emg_2025",
    )
    lookup = {dataset.key: dataset for dataset in dataset_catalog()}
    modalities = ("EEG", "EMG", "kinematics", "force/device", "stroke")
    matrix: list[list[int]] = []
    labels: list[str] = []
    for key in selected_keys:
        dataset = lookup[key]
        joined = " ".join(dataset.modalities).lower()
        matrix.append(
            [
                int("eeg" in joined),
                int("emg" in joined),
                int("kinematic" in joined or "encoder" in joined),
                int("force" in joined or "device" in joined or "box" in joined or "telemetry" in joined),
                int("stroke" in dataset.cohort.lower()),
            ]
        )
        labels.append(dataset.name)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(("#E7EDF0", TEAL)), aspect="auto")
    ax.set_xticks(range(len(modalities)), modalities)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Prioritized Synchronized EEG-EMG Resources", loc="left", weight="bold", color=INK)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            ax.text(column, row, "Included" if value else "-", ha="center", va="center", color="white" if value else "#8A9BA3", fontsize=8.5, weight="bold" if value else "normal")
    ax.tick_params(colors=INK, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "03_multimodal_dataset_matrix.png", dpi=220)
    plt.close(fig)


def _write_report(metrics: dict[str, Any], checks: list[AuditCheck], output: Path) -> None:
    ready = sum(check.status == "ready" for check in checks)
    prepared = sum(check.status == "prepared" for check in checks)
    gaps = sum(check.status == "gap" for check in checks)
    lines = [
        "# Multimodal Motor-Intent Readiness Audit",
        "",
        "## Current Result",
        "",
        f"- Patient-adapted conservative balanced accuracy: **{100 * _metric(metrics, 'intent_balanced_accuracy'):.1f}%**.",
        f"- Intent sensitivity: **{100 * _metric(metrics, 'intent_sensitivity'):.1f}%**.",
        f"- Cue-locked specificity: **{100 * _metric(metrics, 'intent_specificity'):.1f}%**.",
        f"- Conditional direction balanced accuracy: **{100 * _metric(metrics, 'conditional_direction_balanced_accuracy'):.1f}%**.",
        "- Status: development-observed; continuous false activations per minute are not evaluated.",
        "",
        "## Readiness Summary",
        "",
        f"- Ready: **{ready}** checks.",
        f"- Prepared: **{prepared}** checks.",
        f"- Gaps: **{gaps}** checks.",
        "",
        "| Check | Status | Evidence | Required action |",
        "|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check.label} | {check.status.upper()} | {check.evidence} | {check.next_action} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The next improvement should not be a single fused classifier trained on mixed imagery and execution labels. Build an EEG-only cortical-intent track, an EMG/kinematic teacher and validation track, and an optional hybrid control track. Require EEG-only, EMG-only, and fused ablations for every multimodal claim.",
            "",
            "See `docs/final/EEG_EMG_MULTIMODAL_PROJECT_AUDIT_2026.md` for the cited research comparison and implementation sequence.",
            "",
            "## Figures",
            "",
            "- `01_multimodal_readiness_scorecard.png`",
            "- `02_published_results_context.png`",
            "- `03_multimodal_dataset_matrix.png`",
        ]
    )
    (output / "MULTIMODAL_READINESS_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metrics = _load_json(args.metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    checks = _build_checks(metrics)
    payload = {
        "schema_version": 1,
        "source_metrics": str(args.metrics.resolve()),
        "current_result": {
            key: _metric(metrics, key)
            for key in (
                "intent_balanced_accuracy",
                "intent_sensitivity",
                "intent_specificity",
                "conditional_direction_balanced_accuracy",
            )
        },
        "checks": [asdict(check) | {"score": check.score} for check in checks],
        "multimodal_datasets": [
            dataset.as_dict()
            for dataset in dataset_catalog()
            if "EMG" in dataset.modalities
        ],
    }
    (args.output / "multimodal_readiness.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    _plot_scorecard(checks, args.output)
    _plot_context_table(metrics, args.output)
    _plot_dataset_matrix(args.output)
    _write_report(metrics, checks, args.output)
    print(f"[+] Saved multimodal readiness audit to {args.output.resolve()}")


if __name__ == "__main__":
    main()
