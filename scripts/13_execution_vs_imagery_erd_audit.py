#!/usr/bin/env python3
"""Compare executed and imagined unilateral fist movement using baseline-normalized ERD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt  # noqa: E402
from scipy.stats import ttest_rel  # noqa: E402

from bci_pipeline.dataset import EXECUTION_RUNS, IMAGERY_RUNS, LABEL_NAMES, normalize_channel_name  # noqa: E402


def channel_index(names: Sequence[str], requested: str) -> int:
    requested = normalize_channel_name(requested)
    for index, name in enumerate(names):
        if normalize_channel_name(name) == requested:
            return index
    raise ValueError(f"Missing channel {requested}.")


def mode_name(run: int) -> str:
    if run in EXECUTION_RUNS:
        return "execution"
    if run in IMAGERY_RUNS:
        return "imagery"
    raise ValueError(f"Unexpected unilateral run {run}.")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default="artifacts/cross_task/execution_imagery_epochs.npz")
    parser.add_argument("--output-dir", default="artifacts/execution_imagery_erd_audit")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()
    with np.load(Path(args.cache_path), allow_pickle=False) as cache:
        epochs = cache["epochs"]
        baselines = cache["baselines"]
        labels = cache["labels"]
        subjects = cache["subjects"]
        runs = cache["runs"]
        channel_names = [str(name) for name in cache["channel_names"]]
        sfreq = float(cache["sfreq"].item())

    c3, c4 = channel_index(channel_names, "C3"), channel_index(channel_names, "C4")
    motor_indices = [channel_index(channel_names, name) for name in ("FC3", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP4")]
    times = 0.5 + np.arange(epochs.shape[-1]) / sfreq
    summary_window = (times >= 0.5) & (times <= 3.5)
    conditions = [("execution", 1), ("execution", 2), ("imagery", 1), ("imagery", 2)]
    group_sum = {
        band: {condition: np.zeros(epochs.shape[-1], dtype=np.float64) for condition in conditions}
        for band in ("mu", "beta")
    }
    group_count = {condition: 0 for condition in conditions}
    summaries = {band: np.empty((len(labels), len(channel_names)), dtype=np.float32) for band in ("mu", "beta")}

    for band, (low, high) in {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}.items():
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        for start in range(0, len(labels), args.batch_size):
            stop = min(start + args.batch_size, len(labels))
            task_power = np.abs(hilbert(sosfiltfilt(sos, epochs[start:stop], axis=-1), axis=-1)) ** 2
            baseline_power = np.mean(
                np.abs(hilbert(sosfiltfilt(sos, baselines[start:stop], axis=-1), axis=-1)) ** 2,
                axis=-1,
            )
            erd = 10.0 * np.log10((task_power + 1e-20) / (baseline_power[..., None] + 1e-20))
            summaries[band][start:stop] = np.mean(erd[:, :, summary_window], axis=-1)
            if band == "mu":
                for local, trial in enumerate(range(start, stop)):
                    if labels[trial] == 0:
                        continue
                    condition = (mode_name(int(runs[trial])), int(labels[trial]))
                    group_count[condition] += 1
            for condition in conditions:
                mode, side = condition
                mask = (labels[start:stop] == side) & np.isin(
                    runs[start:stop], EXECUTION_RUNS if mode == "execution" else IMAGERY_RUNS
                )
                if np.any(mask):
                    group_sum[band][condition] += np.sum(erd[mask][:, c4 if side == 1 else c3], axis=0)

    group_average = {
        band: {condition: values / max(group_count[condition], 1) for condition, values in rows.items()}
        for band, rows in group_sum.items()
    }
    subject_rows: List[Dict[str, object]] = []
    for subject in np.unique(subjects):
        for mode, side in conditions:
            mask = (
                (subjects == subject)
                & (labels == side)
                & np.isin(runs, EXECUTION_RUNS if mode == "execution" else IMAGERY_RUNS)
            )
            if not np.any(mask):
                continue
            contralateral = c4 if side == 1 else c3
            ipsilateral = c3 if side == 1 else c4
            row: Dict[str, object] = {
                "subject": int(subject),
                "mode": mode,
                "side": LABEL_NAMES[side],
                "trials": int(np.sum(mask)),
            }
            for band in ("mu", "beta"):
                row[f"contralateral_{band}_erd_db"] = float(np.mean(summaries[band][mask, contralateral]))
                row[f"ipsilateral_{band}_erd_db"] = float(np.mean(summaries[band][mask, ipsilateral]))
                row[f"motor_mean_{band}_erd_db"] = float(np.mean(summaries[band][mask][:, motor_indices]))
                row[f"lateralization_{band}_db"] = float(
                    np.mean(summaries[band][mask, ipsilateral] - summaries[band][mask, contralateral])
                )
            subject_rows.append(row)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "execution_imagery_subject_features.csv", subject_rows)

    paired_stats: Dict[str, object] = {}
    for side in ("left_fist", "right_fist"):
        for band in ("mu", "beta"):
            execution = {
                row["subject"]: -float(row[f"contralateral_{band}_erd_db"])
                for row in subject_rows
                if row["mode"] == "execution" and row["side"] == side
            }
            imagery = {
                row["subject"]: -float(row[f"contralateral_{band}_erd_db"])
                for row in subject_rows
                if row["mode"] == "imagery" and row["side"] == side
            }
            shared = sorted(set(execution) & set(imagery))
            execution_values = np.asarray([execution[subject] for subject in shared])
            imagery_values = np.asarray([imagery[subject] for subject in shared])
            statistic, p_value = ttest_rel(execution_values, imagery_values)
            key = f"{side}_{band}"
            paired_stats[key] = {
                "subjects": len(shared),
                "execution_mean_erd_magnitude_db": float(np.mean(execution_values)),
                "imagery_mean_erd_magnitude_db": float(np.mean(imagery_values)),
                "mean_execution_minus_imagery_db": float(np.mean(execution_values - imagery_values)),
                "paired_t": float(statistic),
                "two_sided_p": float(p_value),
            }

    colors = {"execution": "#b65c4a", "imagery": "#39739d"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for row, side in enumerate((1, 2)):
        for column, band in enumerate(("mu", "beta")):
            axis = axes[row, column]
            for mode in ("execution", "imagery"):
                condition = (mode, side)
                axis.plot(times, group_average[band][condition], color=colors[mode], label=mode, linewidth=2)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{LABEL_NAMES[side]} | contralateral {band}")
            axis.set_ylabel("ERD/ERS (dB)")
    for axis in axes[-1]:
        axis.set_xlabel("Seconds after cue")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Execution versus imagery: contralateral sensorimotor ERD", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "execution_vs_imagery_erd_timecourses.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for axis, band in zip(axes, ("mu", "beta")):
        execution_values, imagery_values = [], []
        for subject in sorted(set(row["subject"] for row in subject_rows)):
            execution = [
                -float(row[f"contralateral_{band}_erd_db"])
                for row in subject_rows
                if row["subject"] == subject and row["mode"] == "execution"
            ]
            imagery = [
                -float(row[f"contralateral_{band}_erd_db"])
                for row in subject_rows
                if row["subject"] == subject and row["mode"] == "imagery"
            ]
            if execution and imagery:
                execution_values.append(float(np.mean(execution)))
                imagery_values.append(float(np.mean(imagery)))
        for execution, imagery in zip(execution_values, imagery_values):
            axis.plot((0, 1), (execution, imagery), color="#999999", alpha=0.25)
        axis.scatter(np.zeros(len(execution_values)), execution_values, color=colors["execution"], alpha=0.75)
        axis.scatter(np.ones(len(imagery_values)), imagery_values, color=colors["imagery"], alpha=0.75)
        axis.set_xticks((0, 1), ("Execution", "Imagery"))
        axis.set_title(f"Contralateral {band} ERD magnitude")
        axis.set_ylabel("-ERD (dB; higher means stronger desynchronization)")
    fig.suptitle("Within-subject execution-to-imagery attenuation", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "execution_vs_imagery_paired_erd.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "protocol": {
            "execution_runs": list(EXECUTION_RUNS),
            "imagery_runs": list(IMAGERY_RUNS),
            "task": "unilateral left/right fist movement or imagery",
            "post_cue_epoch_seconds": [0.5, 3.5],
            "normalization": "band power relative to each trial pre-cue baseline",
            "primary_measure": "contralateral mu/beta ERD magnitude",
        },
        "dataset": {
            "trials": int(len(labels)),
            "subjects": int(len(np.unique(subjects))),
            "condition_trial_counts": {
                f"{mode}_{LABEL_NAMES[side]}": int(group_count[(mode, side)])
                for mode, side in conditions
            },
        },
        "paired_execution_imagery_statistics": paired_stats,
    }
    with (output_dir / "execution_imagery_erd_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("=== Execution versus Imagery ERD Audit ===")
    for key, stats in paired_stats.items():
        print(
            f"[+] {key}: execution={stats['execution_mean_erd_magnitude_db']:.2f} dB | "
            f"imagery={stats['imagery_mean_erd_magnitude_db']:.2f} dB | "
            f"difference={stats['mean_execution_minus_imagery_db']:.2f} dB | p={stats['two_sided_p']:.4g}"
        )
    print(f"[+] Saved audit to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
