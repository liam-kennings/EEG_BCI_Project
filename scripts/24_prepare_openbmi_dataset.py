#!/usr/bin/env python3
"""Validate OpenBMI BIDS data and optionally build a model-ready epoch cache."""

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

from bci_pipeline.dataset import parse_int_list  # noqa: E402
from bci_pipeline.openbmi import (  # noqa: E402
    EXPECTED_RECORDINGS_PER_SUBJECT,
    EXPECTED_SUBJECTS,
    available_openbmi_subjects,
    build_openbmi_raw_epoch_dataset,
    discover_openbmi_recordings,
    iter_openbmi_dataset,
    metadata_subjects,
    resolve_openbmi_root,
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


def plot_overview(
    path: Path,
    summary: Mapping[str, object],
    trial_rows: Sequence[Mapping[str, object]],
) -> None:
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
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.0), constrained_layout=True)
    figure.suptitle(
        "OpenBMI Motor-Imagery Adapter Audit",
        fontsize=18,
        fontweight="bold",
        color=COLORS["navy"],
    )

    axes[0, 0].bar(
        ["Subjects", "BDF recordings"],
        [summary["complete_subjects"], summary["available_recordings"]],
        color=(COLORS["blue"], COLORS["green"]),
        label="Available",
    )
    axes[0, 0].bar(
        ["Subjects", "BDF recordings"],
        [
            summary["metadata_subjects"] - summary["complete_subjects"],
            summary["expected_recordings"] - summary["available_recordings"],
        ],
        bottom=[summary["complete_subjects"], summary["available_recordings"]],
        color=COLORS["gray"],
        alpha=0.38,
        label="Missing signal files",
    )
    axes[0, 0].set_title("Local Archive Completeness")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.2)

    labels = Counter(str(row["canonical_label"]) for row in trial_rows)
    bars = axes[0, 1].bar(
        ["Left hand", "Right hand"],
        [labels["left_hand"], labels["right_hand"]],
        color=(COLORS["blue"], COLORS["orange"]),
    )
    axes[0, 1].bar_label(bars, fmt="%d", fontweight="bold")
    axes[0, 1].set_title("Available Intent Trials")
    axes[0, 1].set_ylabel("Trials")
    axes[0, 1].grid(axis="y", alpha=0.2)

    timeline = axes[1, 0]
    segments = (
        (0.0, 4.0, "Motor imagery", COLORS["blue"]),
        (4.0, 1.0, "Buffer", COLORS["gold"]),
        (5.0, 4.0, "Rest window", COLORS["green"]),
    )
    for start, width, label, color in segments:
        timeline.barh(0, width, left=start, height=0.5, color=color)
        timeline.text(
            start + width / 2,
            0,
            label,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    timeline.set_xlim(0, 9)
    timeline.set_yticks([])
    timeline.set_xlabel("Time from imagery cue (s)")
    timeline.set_title("Adapter Window Definitions")
    timeline.grid(axis="x", alpha=0.2)

    session_counts = Counter(
        (str(row["session"]), str(row["recording"])) for row in trial_rows
    )
    names = ["S1 train", "S1 test", "S2 train", "S2 test"]
    keys = [
        ("ses-1", "train"),
        ("ses-1", "test"),
        ("ses-2", "train"),
        ("ses-2", "test"),
    ]
    bars = axes[1, 1].bar(
        names,
        [session_counts[key] for key in keys],
        color=(COLORS["navy"], COLORS["blue"], COLORS["gold"], COLORS["orange"]),
    )
    axes[1, 1].bar_label(bars, fmt="%d", fontweight="bold")
    axes[1, 1].set_title("Session and Recording Coverage")
    axes[1, 1].set_ylabel("Intent trials")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# OpenBMI Adapter Import",
        "",
        "## Structural Validation",
        "",
        f"- BIDS root: `{summary['dataset_root']}`",
        f"- Metadata subjects: **{summary['metadata_subjects']}**",
        f"- Subjects with all four BDFs: **{summary['complete_subjects']}**",
        f"- Available/expected BDF recordings: "
        f"**{summary['available_recordings']}/{summary['expected_recordings']}**",
        f"- Available intent trials: **{summary['intent_trials']}**",
        f"- Model-ready intent + paired-rest windows before artifact rejection: "
        f"**{summary['potential_model_windows']}**",
        "",
        "## Adapter Rules",
        "",
        "- Labels are mapped from `trial_type`, not numeric `value`, because the "
        "distributed README and BIDS values disagree.",
        "- EMG1-EMG4 are typed as EMG and excluded from EEG referencing/features.",
        "- Session 1 train/test map to pipeline Runs 4/8 for calibration and "
        "development; both Session 2 recordings map to Run 12 for held-out "
        "cross-session evaluation.",
        "- Imagery uses 0.5-3.5 seconds after cue. Rest uses 5.5-8.5 seconds, "
        "after a one-second post-imagery buffer.",
        "- Preprocessing uses all available EEG for CAR, then selects the common "
        "nine-channel sensorimotor montage and resamples to 160 Hz.",
        "",
        "## Limitation",
        "",
        f"The local archive contains complete signal data for "
        f"{summary['complete_subjects']} of {summary['metadata_subjects']} subjects. "
        "Adapter development can proceed, but final OpenBMI claims must wait until "
        "the missing BDF recordings are retrieved.",
        "",
        f"Overview: `{summary['overview_figure']}`",
        f"Trial manifest: `{summary['trial_manifest']}`",
        f"Recording inventory: `{summary['recording_inventory']}`",
    ]
    if summary.get("epoch_cache"):
        lines.append(f"Raw-window cache: `{summary['epoch_cache']}`")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="files/external/openbmi/nm000338",
        help="OpenBMI nm000338 BIDS root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "openbmi",
    )
    parser.add_argument("--subjects", help="Optional subjects, for example 4-9,32.")
    parser.add_argument(
        "--build-epoch-cache",
        action="store_true",
        help="Read BDF signals and build the common RawEpochDataset cache.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    root = resolve_openbmi_root(args.data_dir)
    metadata = metadata_subjects(root)
    complete = available_openbmi_subjects(root, complete_only=True)
    requested = parse_int_list(args.subjects)
    subjects = requested if requested is not None else complete
    unavailable = sorted(set(subjects) - set(complete))
    if unavailable:
        raise FileNotFoundError(
            f"Subjects without all four local BDF recordings: {unavailable}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recording_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    print("\n=== OpenBMI BIDS Adapter Preparation ===")
    print(f"Data: {root}")
    print(
        f"[+] Complete local subjects: {len(complete)}/{len(metadata)} | "
        f"selected: {len(subjects)}"
    )
    for position, (recording, trials) in enumerate(
        iter_openbmi_dataset(root, subjects),
        start=1,
    ):
        recording_rows.append(recording.as_dict())
        trial_rows.extend(trial.as_dict() for trial in trials)
        if position % 20 == 0:
            print(
                f"    checked {position:3d} recordings; "
                f"harmonized {len(trial_rows):,} intent trials"
            )

    trial_manifest = args.output_dir / "openbmi_trial_manifest.csv"
    inventory = args.output_dir / "openbmi_recording_inventory.csv"
    overview = args.output_dir / "openbmi_dataset_overview.png"
    write_csv(trial_manifest, trial_rows)
    write_csv(inventory, recording_rows)
    expected_recordings = len(metadata) * EXPECTED_RECORDINGS_PER_SUBJECT
    available_recordings = len(discover_openbmi_recordings(root))
    summary: dict[str, object] = {
        "schema_version": 1,
        "dataset": "openbmi",
        "dataset_root": str(root),
        "metadata_subjects": len(metadata),
        "complete_subjects": len(complete),
        "selected_subjects": len(subjects),
        "available_recordings": available_recordings,
        "expected_recordings": expected_recordings,
        "missing_recordings": expected_recordings - available_recordings,
        "intent_trials": len(trial_rows),
        "left_trials": sum(
            row["canonical_label"] == "left_hand" for row in trial_rows
        ),
        "right_trials": sum(
            row["canonical_label"] == "right_hand" for row in trial_rows
        ),
        "potential_model_windows": 2 * len(trial_rows),
        "native_sampling_hz": 1000.0,
        "target_sampling_hz": 160.0,
        "eeg_channels": 62,
        "emg_channels": 4,
        "common_motor_channels": 9,
        "trial_manifest": str(trial_manifest.resolve()),
        "recording_inventory": str(inventory.resolve()),
        "overview_figure": str(overview.resolve()),
        "epoch_cache": None,
    }
    if args.build_epoch_cache:
        cache = args.output_dir / "openbmi_raw_epochs.npz"
        dataset = build_openbmi_raw_epoch_dataset(
            root,
            subjects,
            cache,
            rebuild_cache=args.rebuild_cache,
        )
        summary["epoch_cache"] = str(cache.resolve())
        summary["model_windows_kept"] = len(dataset.labels)
        summary["model_label_counts"] = {
            "rest": int((dataset.labels == 0).sum()),
            "left_hand": int((dataset.labels == 1).sum()),
            "right_hand": int((dataset.labels == 2).sum()),
        }
    plot_overview(overview, summary, trial_rows)
    summary["runtime_seconds"] = time.time() - started
    summary_path = args.output_dir / "openbmi_dataset_summary.json"
    report_path = args.output_dir / "OPENBMI_IMPORT_REPORT.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    write_report(report_path, summary)
    print(
        f"[+] Validated {len(recording_rows)} recordings and "
        f"{len(trial_rows):,} intent trials."
    )
    print(f"[+] Saved report: {report_path.resolve()}")
    print(f"[+] Saved overview: {overview.resolve()}")


if __name__ == "__main__":
    main()
