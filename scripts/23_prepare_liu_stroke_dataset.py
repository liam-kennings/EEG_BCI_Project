#!/usr/bin/env python3
"""Validate and harmonize the Liu acute-stroke motor-imagery dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bci_pipeline.dataset import parse_int_list  # noqa: E402
from bci_pipeline.liu_stroke import (  # noqa: E402
    EXPECTED_SUBJECTS,
    EXPECTED_TRIALS_PER_SUBJECT,
    available_liu_subjects,
    iter_liu_dataset,
    load_participants,
    resolve_liu_root,
)


COLORS = {
    "navy": "#16324F",
    "blue": "#147D92",
    "orange": "#D66A3A",
    "gold": "#E9A23B",
    "green": "#2D6A4F",
    "gray": "#65717C",
    "background": "#F6F3EC",
}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FBFAF7",
            "figure.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
        }
    )


def plot_overview(
    output_path: Path,
    participants: Mapping[str, Mapping[str, str]],
    trial_rows: Sequence[Mapping[str, object]],
) -> None:
    configure_plots()
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 8.2), constrained_layout=True)
    figure.suptitle(
        "Liu Acute-Stroke Motor-Imagery Dataset",
        fontsize=18,
        fontweight="bold",
        color=COLORS["navy"],
    )

    timeline = axes[0, 0]
    segments = (
        (0.0, 2.0, "Instruction", COLORS["gold"]),
        (2.0, 4.0, "Motor imagery", COLORS["blue"]),
        (6.0, 2.0, "Break / rest", COLORS["green"]),
    )
    for start, width, label, color in segments:
        timeline.barh(0, width, left=start, height=0.52, color=color, label=label)
        timeline.text(
            start + width / 2.0,
            0,
            f"{label}\n{width:.0f} s",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    timeline.set_xlim(0, 8)
    timeline.set_yticks([])
    timeline.set_xlabel("Time from trial cue (s)")
    timeline.set_title("One 8-second Trial")
    timeline.grid(axis="x", alpha=0.2)

    paralysis = Counter(row["ParalysisSide"].strip().lower() for row in participants.values())
    sides = ["left", "right"]
    values = [paralysis.get(side, 0) for side in sides]
    bars = axes[0, 1].bar(
        ["Left-side\nparalysis", "Right-side\nparalysis"],
        values,
        color=(COLORS["orange"], COLORS["blue"]),
    )
    axes[0, 1].bar_label(bars, fontweight="bold")
    axes[0, 1].set_ylabel("Patients")
    axes[0, 1].set_title("Clinical Cohort")
    axes[0, 1].grid(axis="y", alpha=0.2)

    ages = np.asarray([int(row["Age"]) for row in participants.values()])
    nihss = np.asarray([int(row["NIHSS"]) for row in participants.values()])
    axes[1, 0].hist(
        ages,
        bins=np.arange(25, 81, 5),
        color=COLORS["navy"],
        edgecolor="white",
    )
    axes[1, 0].axvline(np.median(ages), color=COLORS["gold"], linewidth=2)
    axes[1, 0].set_xlabel("Age (years)")
    axes[1, 0].set_ylabel("Patients")
    axes[1, 0].set_title(f"Age Distribution (median {np.median(ages):.0f} years)")
    axes[1, 0].grid(axis="y", alpha=0.2)

    labels = Counter(str(row["canonical_label"]) for row in trial_rows)
    class_names = ["left_hand", "right_hand"]
    class_counts = [labels[name] for name in class_names]
    bars = axes[1, 1].bar(
        ["Left-hand\nimagery", "Right-hand\nimagery"],
        class_counts,
        color=(COLORS["blue"], COLORS["orange"]),
    )
    axes[1, 1].bar_label(bars, fontweight="bold", fmt="%d")
    axes[1, 1].set_ylabel("Trials")
    axes[1, 1].set_title(
        f"Balanced Labels: {len(trial_rows):,} Trials | Median NIHSS {np.median(nihss):.0f}"
    )
    axes[1, 1].grid(axis="y", alpha=0.2)

    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Liu Acute-Stroke Dataset Import",
        "",
        "## Structural Validation",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Valid recordings: **{summary['valid_recordings']}/{summary['recordings']}**",
        f"- Recordings requiring trigger fallback: "
        f"**{summary['recordings_with_trigger_fallback']}**",
        f"- Recordings containing extra trigger pulses: "
        f"**{summary['recordings_with_extra_pulses']}**",
        f"- Repaired/missing stage pulses: **{summary['trigger_fallbacks']}**; "
        f"ignored extra pulses: **{summary['ignored_trigger_pulses']}**",
        f"- Patients: **{summary['subjects']}**",
        f"- Harmonized trials: **{summary['trials']}**",
        f"- Left/right trials: **{summary['left_trials']} / {summary['right_trials']}**",
        f"- Native sampling: **{summary['sampling_hz']:.0f} Hz**",
        f"- Montage: **{summary['eeg_channels']} EEG + {summary['eog_channels']} EOG**",
        "",
        "## Timing Interpretation",
        "",
        "Subject-specific EDF trigger pulses were used when a correctly coded pulse "
        "occurred near the expected protocol stage. Isolated missing or extra pulses "
        "were reconciled against the fixed 8-second event schedule and explicitly "
        "flagged in the manifests. Every trial preserves cue onset, motor-imagery "
        "start/end, and break/rest timing. The instruction interval is not silently "
        "labeled as rest because it contains visual and preparatory activity.",
        "",
        "## Channel Interpretation",
        "",
        "The distributed EDF stores CPz as a flat online-reference channel, HEOL/HEOR "
        "as EOG, and one unnamed signal as the trigger channel. The adapter corrects "
        "these roles in memory and leaves the source files unchanged.",
        "",
        "## Research Boundary",
        "",
        "This import report establishes data integrity, not decoding accuracy. The Liu "
        "cohort should first be used for zero-shot external validation of a frozen "
        "healthy-subject model. Any patient-specific calibration must use earlier trials "
        "and preserve later trials for evaluation.",
        "",
        f"Overview figure: `{summary['overview_figure']}`",
        f"Trial manifest: `{summary['trial_manifest']}`",
        f"Recording inventory: `{summary['recording_inventory']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="Liu acute-stroke dataset",
        help="Liu dataset root containing edffile/ and participants.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "liu_stroke",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated patients or ranges, for example 1-5,12.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    root = resolve_liu_root(args.data_dir)
    discovered = available_liu_subjects(root)
    requested = parse_int_list(args.subjects)
    subjects = requested if requested is not None else discovered
    missing = sorted(set(subjects) - set(discovered))
    if missing:
        raise FileNotFoundError(f"Requested Liu subjects are missing: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    participants = load_participants(root)
    recording_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    print("\n=== Liu Acute-Stroke Dataset Preparation ===")
    print(f"Data: {root}")
    print(f"[+] Validating {len(subjects)} patient recordings...")
    for position, (recording, trials) in enumerate(
        iter_liu_dataset(root, subjects),
        start=1,
    ):
        recording_rows.append(recording.as_dict())
        trial_rows.extend(trial.as_dict() for trial in trials)
        if position % 10 == 0 or position == len(subjects):
            print(
                f"    checked {position:2d}/{len(subjects)} patients; "
                f"harmonized {len(trial_rows):,} trials"
            )

    trial_manifest = args.output_dir / "liu_trial_manifest.csv"
    recording_inventory = args.output_dir / "liu_recording_inventory.csv"
    overview_figure = args.output_dir / "liu_dataset_overview.png"
    write_csv(trial_manifest, trial_rows)
    write_csv(recording_inventory, recording_rows)
    plot_overview(overview_figure, participants, trial_rows)

    label_counts = Counter(str(row["canonical_label"]) for row in trial_rows)
    fatal_flags = {
        "unexpected_sampling_rate",
        "unexpected_channel_count",
        "unexpected_eeg_count",
        "unexpected_eog_count",
        "event_outside_recording",
    }
    valid_recordings = sum(
        not (set(str(row["quality_flags"]).split(";")) & fatal_flags)
        for row in recording_rows
    )
    recordings_with_trigger_fallback = sum(
        int(row["trigger_fallbacks"]) > 0 for row in recording_rows
    )
    recordings_with_extra_pulses = sum(
        int(row["ignored_trigger_pulses"]) > 0 for row in recording_rows
    )
    trigger_fallbacks = sum(int(row["trigger_fallbacks"]) for row in recording_rows)
    ignored_trigger_pulses = sum(
        int(row["ignored_trigger_pulses"]) for row in recording_rows
    )
    summary = {
        "schema_version": 1,
        "dataset": "liu2024",
        "dataset_root": str(root),
        "subjects": len(subjects),
        "recordings": len(recording_rows),
        "valid_recordings": valid_recordings,
        "recordings_with_trigger_fallback": recordings_with_trigger_fallback,
        "recordings_with_extra_pulses": recordings_with_extra_pulses,
        "trigger_fallbacks": trigger_fallbacks,
        "ignored_trigger_pulses": ignored_trigger_pulses,
        "trials": len(trial_rows),
        "expected_complete_trials": len(subjects) * EXPECTED_TRIALS_PER_SUBJECT,
        "left_trials": label_counts["left_hand"],
        "right_trials": label_counts["right_hand"],
        "sampling_hz": 500.0,
        "eeg_channels": 29,
        "eog_channels": 2,
        "complete_download": (
            len(discovered) == EXPECTED_SUBJECTS
            and len(subjects) == EXPECTED_SUBJECTS
            and valid_recordings == EXPECTED_SUBJECTS
        ),
        "timing_source": "EDF STIM pulses with flagged protocol fallback",
        "rest_definition": "stage-3 break only; instruction is retained separately",
        "trial_manifest": str(trial_manifest.resolve()),
        "recording_inventory": str(recording_inventory.resolve()),
        "overview_figure": str(overview_figure.resolve()),
        "runtime_seconds": time.time() - started,
    }
    summary_path = args.output_dir / "liu_dataset_summary.json"
    report_path = args.output_dir / "LIU_STROKE_IMPORT_REPORT.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    write_report(report_path, summary)

    print(
        f"[+] Valid recordings: {valid_recordings}/{len(recording_rows)} | "
        f"trials: {len(trial_rows):,} | "
        f"left/right: {label_counts['left_hand']}/{label_counts['right_hand']}"
    )
    print(f"[+] Saved report: {report_path.resolve()}")
    print(f"[+] Saved overview: {overview_figure.resolve()}")


if __name__ == "__main__":
    main()
