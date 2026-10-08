#!/usr/bin/env python3
"""Create an interpretable Morlet/ERD feature audit for unilateral motor imagery.

This script deliberately stops before classification. It converts cached raw
imagery trials into baseline-normalized time-frequency maps and a trial-level
feature table, then saves figures that make the sensorimotor physiology visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

from bci_pipeline.dataset import (  # noqa: E402
    LABEL_NAMES,
    available_subjects,
    normalize_channel_name,
    parse_int_list,
    resolve_data_dir,
)
from bci_pipeline.epochs import EpochConfig, RawEpochDataset, build_raw_epoch_dataset  # noqa: E402


BANDS = {
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 30.0),
}
PRIMARY_BANDS = ("mu", "beta_low", "beta_high")


def morlet_kernel(freq_hz: float, cycles: float, sfreq: float) -> np.ndarray:
    """Return a unit-energy complex Morlet kernel."""
    sigma_seconds = cycles / (2.0 * np.pi * freq_hz)
    half_width = max(1.0, 3.5 * sigma_seconds)
    time = np.arange(-half_width, half_width + 1.0 / sfreq, 1.0 / sfreq)
    kernel = np.exp(2j * np.pi * freq_hz * time) * np.exp(
        -(time**2) / (2.0 * sigma_seconds**2)
    )
    return kernel / np.sqrt(np.sum(np.abs(kernel) ** 2))


def morlet_power_batch(
    signals: np.ndarray,
    freqs: np.ndarray,
    cycles: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Compute complex-Morlet power for ``trials x channels x samples`` signals."""
    power = np.empty((*signals.shape[:2], len(freqs), signals.shape[-1]), dtype=np.float32)
    for frequency_index, (freq, cycle_count) in enumerate(zip(freqs, cycles)):
        convolved = fftconvolve(
            signals,
            morlet_kernel(float(freq), float(cycle_count), sfreq)[None, None, :],
            mode="same",
            axes=-1,
        )
        power[:, :, frequency_index] = np.abs(convolved) ** 2
    return power


def db_baseline_ratio(task_power: np.ndarray, baseline_power: np.ndarray) -> np.ndarray:
    """Express time-frequency power as dB change from each trial's own baseline."""
    epsilon = np.finfo(np.float32).tiny
    return 10.0 * np.log10((task_power + epsilon) / (baseline_power[..., None] + epsilon))


def channel_index(channel_names: Sequence[str], requested: str) -> int | None:
    normalized = normalize_channel_name(requested)
    for index, name in enumerate(channel_names):
        if normalize_channel_name(name) == normalized:
            return index
    return None


def montage_channel_name(name: str) -> str:
    """Convert EDF naming quirks such as ``Fc3.`` to standard montage labels."""
    normalized = normalize_channel_name(name)
    if normalized.endswith("Z"):
        return normalized[:-1] + "z"
    return normalized


def band_indices(freqs: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        name: np.flatnonzero((freqs >= low) & (freqs < high))
        for name, (low, high) in BANDS.items()
    }


def add_feature_rows(
    rows: List[Dict[str, object]],
    dataset: RawEpochDataset,
    trial_indices: np.ndarray,
    ratios_db: np.ndarray,
    freq_indices: Dict[str, np.ndarray],
    times: np.ndarray,
) -> None:
    early = (times >= 0.5) & (times < 1.5)
    late = (times >= 1.5) & (times <= 3.5)
    middle = (times >= 1.5) & (times < 2.5)
    terminal = (times >= 2.5) & (times <= 3.5)
    full_cue = (times >= 0.5) & (times <= min(4.0, float(times[-1])))
    windows = {
        "middle": middle,
        "terminal": terminal,
        "full_cue": full_cue,
    }
    channel_names = dataset.channel_names
    c3, c4 = channel_index(channel_names, "C3"), channel_index(channel_names, "C4")
    ptp_uv = np.ptp(dataset.epochs[trial_indices], axis=2) * 1e6
    rms_uv = np.sqrt(np.mean(dataset.epochs[trial_indices] ** 2, axis=2)) * 1e6

    for local_index, trial_index in enumerate(trial_indices):
        row: Dict[str, object] = {
            "subject": int(dataset.subjects[trial_index]),
            "run": int(dataset.runs[trial_index]),
            "trial_index": int(dataset.trial_indices[trial_index]),
            "label": int(dataset.labels[trial_index]),
            "label_name": LABEL_NAMES[int(dataset.labels[trial_index])],
            "mean_peak_to_peak_uv": float(np.mean(ptp_uv[local_index])),
            "max_peak_to_peak_uv": float(np.max(ptp_uv[local_index])),
            "mean_rms_uv": float(np.mean(rms_uv[local_index])),
        }
        for channel_position, channel_name in enumerate(channel_names):
            safe_name = normalize_channel_name(channel_name).lower()
            for band_name in PRIMARY_BANDS:
                band_power = np.mean(
                    ratios_db[local_index, channel_position, freq_indices[band_name]], axis=0
                )
                row[f"{safe_name}_{band_name}_early_erd_db"] = float(np.mean(band_power[early]))
                row[f"{safe_name}_{band_name}_late_erd_db"] = float(np.mean(band_power[late]))
                row[f"{safe_name}_{band_name}_minimum_erd_db"] = float(np.min(band_power))
                row[f"{safe_name}_{band_name}_mean_erd_db"] = float(np.mean(band_power))
                for window_name, window_mask in windows.items():
                    row[f"{safe_name}_{band_name}_{window_name}_erd_db"] = float(
                        np.mean(band_power[window_mask])
                    )

        if c3 is not None and c4 is not None:
            for band_name in PRIMARY_BANDS:
                c3_band = np.mean(ratios_db[local_index, c3, freq_indices[band_name]], axis=0)
                c4_band = np.mean(ratios_db[local_index, c4, freq_indices[band_name]], axis=0)
                difference = c3_band - c4_band
                row[f"c3_minus_c4_{band_name}_early_db"] = float(np.mean(difference[early]))
                row[f"c3_minus_c4_{band_name}_late_db"] = float(np.mean(difference[late]))
                for window_name, window_mask in windows.items():
                    row[f"c3_minus_c4_{band_name}_{window_name}_db"] = float(
                        np.mean(difference[window_mask])
                    )
        rows.append(row)


def plot_tfrs(
    output_dir: Path,
    average_tfr: Dict[int, np.ndarray],
    counts: Dict[int, int],
    channel_names: Sequence[str],
    freqs: np.ndarray,
    times: np.ndarray,
) -> None:
    channels = [("C3", channel_index(channel_names, "C3")), ("C4", channel_index(channel_names, "C4"))]
    values = [
        average_tfr[label][channel]
        for label in sorted(LABEL_NAMES)
        for _, channel in channels
        if channel is not None
    ]
    limit = max(float(np.percentile(np.abs(np.concatenate([value.ravel() for value in values])), 98)), 1.0)
    fig, axes = plt.subplots(3, len(channels), figsize=(12, 10), sharex=True, sharey=True)
    image = None
    for row, label in enumerate(sorted(LABEL_NAMES)):
        for column, (name, channel) in enumerate(channels):
            axis = axes[row, column]
            if channel is None:
                axis.set_visible(False)
                continue
            image = axis.pcolormesh(
                times,
                freqs,
                average_tfr[label][channel],
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.axvline(1.5, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{LABEL_NAMES[label]} | {name} | n={counts[label]}")
            axis.set_ylabel("Frequency (Hz)")
    for axis in axes[-1]:
        axis.set_xlabel("Seconds after cue")
    fig.subplots_adjust(top=0.93, right=0.88, hspace=0.28, wspace=0.12)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Power change from baseline (dB)", shrink=0.88)
    fig.suptitle("Baseline-normalized Morlet ERD/ERS maps", y=1.01)
    fig.savefig(output_dir / "morlet_erd_tfr_c3_c4.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_timecourses(
    output_dir: Path,
    average_tfr: Dict[int, np.ndarray],
    channel_names: Sequence[str],
    freq_indices: Dict[str, np.ndarray],
    times: np.ndarray,
) -> None:
    colors = {0: "#5b6770", 1: "#3b7ea1", 2: "#b65c4a"}
    channels = [("C3", channel_index(channel_names, "C3")), ("C4", channel_index(channel_names, "C4"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for row, (channel_name, channel) in enumerate(channels):
        for column, band_name in enumerate(("mu", "beta_low")):
            axis = axes[row, column]
            if channel is None:
                axis.set_visible(False)
                continue
            for label in sorted(LABEL_NAMES):
                curve = np.mean(average_tfr[label][channel, freq_indices[band_name]], axis=0)
                axis.plot(times, curve, color=colors[label], label=LABEL_NAMES[label], linewidth=2)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(1.5, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{channel_name} {band_name.replace('_', ' ')}")
            axis.set_ylabel("ERD/ERS (dB)")
    for axis in axes[-1]:
        axis.set_xlabel("Seconds after cue")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Sensorimotor ERD time courses", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "erd_timecourses_c3_c4.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_topomaps(
    output_dir: Path,
    average_tfr: Dict[int, np.ndarray],
    channel_names: Sequence[str],
    freq_indices: Dict[str, np.ndarray],
    times: np.ndarray,
    sfreq: float,
) -> None:
    window = (times >= 0.5) & (times <= 3.5)
    montage_names = [montage_channel_name(name) for name in channel_names]
    info = mne.create_info(montage_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage("standard_1005", on_missing="raise")
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    image = None
    for row, label in enumerate((1, 2)):
        for column, band_name in enumerate(("mu", "beta_low")):
            values = np.mean(average_tfr[label][:, freq_indices[band_name]][:, :, window], axis=(1, 2))
            image, _ = mne.viz.plot_topomap(
                values,
                info,
                axes=axes[row, column],
                show=False,
                cmap="RdBu_r",
                vlim=(-3, 3),
            )
            axes[row, column].set_title(f"{LABEL_NAMES[label]} | {band_name.replace('_', ' ')}")
    fig.subplots_adjust(top=0.88, right=0.88, hspace=0.35, wspace=0.25)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Mean ERD/ERS (dB)", shrink=0.8)
    fig.suptitle("Motor-imagery scalp distributions", y=0.98)
    fig.savefig(output_dir / "erd_topomaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    labels = sorted(LABEL_NAMES)
    colors = ["#5b6770", "#3b7ea1", "#b65c4a"]
    feature_names = ("c3_minus_c4_mu_early_db", "c3_minus_c4_beta_low_early_db")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for axis, feature_name in zip(axes, feature_names):
        values = [
            [float(row[feature_name]) for row in rows if int(row["label"]) == label]
            for label in labels
        ]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticklabels([LABEL_NAMES[label] for label in labels], rotation=20)
        axis.set_title(feature_name.replace("_", " "))
        axis.set_ylabel("C3 minus C4 ERD (dB)")
    fig.suptitle("Per-trial motor-cortex lateralization features", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "lateralization_feature_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_quality(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    labels = sorted(LABEL_NAMES)
    values = [
        [float(row["max_peak_to_peak_uv"]) for row in rows if int(row["label"]) == label]
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    boxes = ax.boxplot(values, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], ("#5b6770", "#3b7ea1", "#b65c4a")):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(250, color="#b43c39", linestyle="--", label="250 uV rejection limit")
    ax.set_xticklabels([LABEL_NAMES[label] for label in labels], rotation=20)
    ax.set_ylabel("Largest channel peak-to-peak amplitude (uV)")
    ax.set_title("Trial quality after preprocessing")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "trial_quality_peak_to_peak.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No feature rows were generated.")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--cache-path", default="artifacts/benchmark/raw_epoch_cache.npz")
    parser.add_argument("--output-dir", default="artifacts/morlet_feature_audit")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epoch-start-seconds", type=float, default=0.5)
    parser.add_argument("--epoch-duration-seconds", type=float, default=3.0)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir, parse_int_list(args.subjects), args.max_subjects)
    epoch_config = EpochConfig(
        epoch_start_seconds=args.epoch_start_seconds,
        epoch_duration_seconds=args.epoch_duration_seconds,
    )
    dataset = build_raw_epoch_dataset(
        data_dir,
        subjects,
        Path(args.cache_path),
        epoch_config,
        args.rebuild_cache,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.arange(4.0, 31.0, 1.0)
    cycles = np.linspace(3.0, 10.0, len(freqs))
    frequency_indices = band_indices(freqs)
    times = args.epoch_start_seconds + np.arange(dataset.epochs.shape[-1]) / dataset.sfreq
    sums = {label: np.zeros((len(dataset.channel_names), len(freqs), len(times)), dtype=np.float64) for label in LABEL_NAMES}
    counts = {label: 0 for label in LABEL_NAMES}
    rows: List[Dict[str, object]] = []

    print("=== Motor-Imagery Morlet ERD Feature Audit ===")
    print(f"[+] Trials: {len(dataset.labels)} | subjects: {len(np.unique(dataset.subjects))} | channels: {len(dataset.channel_names)}")
    for start in range(0, len(dataset.labels), args.batch_size):
        stop = min(start + args.batch_size, len(dataset.labels))
        indices = np.arange(start, stop)
        task_power = morlet_power_batch(dataset.epochs[indices], freqs, cycles, dataset.sfreq)
        baseline_power = np.mean(
            morlet_power_batch(dataset.baselines[indices], freqs, cycles, dataset.sfreq),
            axis=-1,
        )
        ratios_db = db_baseline_ratio(task_power, baseline_power)
        add_feature_rows(rows, dataset, indices, ratios_db, frequency_indices, times)
        for label in LABEL_NAMES:
            label_mask = dataset.labels[indices] == label
            if np.any(label_mask):
                sums[label] += np.sum(ratios_db[label_mask], axis=0)
                counts[label] += int(np.sum(label_mask))
        if stop % (args.batch_size * 25) == 0 or stop == len(dataset.labels):
            print(f"    processed {stop:5d}/{len(dataset.labels)} trials")

    average_tfr = {label: sums[label] / max(counts[label], 1) for label in LABEL_NAMES}
    write_csv(output_dir / "morlet_erd_trial_features.csv", rows)
    plot_tfrs(output_dir, average_tfr, counts, dataset.channel_names, freqs, times)
    plot_timecourses(output_dir, average_tfr, dataset.channel_names, frequency_indices, times)
    plot_topomaps(output_dir, average_tfr, dataset.channel_names, frequency_indices, times, dataset.sfreq)
    plot_feature_distributions(output_dir, rows)
    plot_quality(output_dir, rows)
    summary = {
        "protocol": {
            "task": "unilateral motor imagery only",
            "baseline_seconds": 1.0,
            "post_cue_epoch_seconds": [
                args.epoch_start_seconds,
                args.epoch_start_seconds + args.epoch_duration_seconds,
            ],
            "normalization": "10 * log10(task Morlet power / trial pre-cue Morlet power)",
            "frequencies_hz": freqs.tolist(),
            "morlet_cycles": cycles.tolist(),
        },
        "dataset": {
            "trials": int(len(dataset.labels)),
            "subjects": int(len(np.unique(dataset.subjects))),
            "channels": dataset.channel_names,
            "condition_counts": {LABEL_NAMES[label]: count for label, count in counts.items()},
        },
        "feature_table": "morlet_erd_trial_features.csv",
    }
    with (output_dir / "morlet_erd_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[+] Saved feature table, figures, and summary to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
