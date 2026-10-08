#!/usr/bin/env python3
"""Evaluate Welch PSD and Morlet spatial-temporal fusion for motor intent."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba"))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.signal import welch  # noqa: E402

from bci_pipeline.classical_eeg_features import (  # noqa: E402
    build_classical_feature_cache,
    calibration_robust_alignment,
)
from bci_pipeline.clinical_evaluation import (  # noqa: E402
    build_transfer_masks,
    fit_intent_candidate,
    paired_subject_bootstrap,
    score_intent,
    select_patient_calibration,
    transfer_training_weights,
)
from bci_pipeline.clinical_transfer import (  # noqa: E402
    HandIntentDataset,
    build_liu_hand_intent_dataset,
    combine_hand_intent_datasets,
    from_harmonized_healthy,
)
from bci_pipeline.harmonized_motor_imagery import (  # noqa: E402
    build_physionet_harmonized_dataset,
    load_openbmi_harmonized_dataset,
)
from bci_pipeline.openbmi_controlled import (  # noqa: E402
    build_covariance_cache,
    tangent_features,
)
from bci_pipeline.splits import load_partitions  # noqa: E402
from bci_pipeline.time_frequency_features import (  # noqa: E402
    MORLET_BANDS,
    MORLET_FREQUENCIES,
    build_morlet_feature_cache,
    morlet_frequency_power,
)


PHYSIONET_OFFSET = 0
OPENBMI_OFFSET = 1000
LIU_OFFSET = 2000
COLORS = {
    "navy": "#17324D",
    "blue": "#2878A6",
    "teal": "#21867A",
    "orange": "#D56A3A",
    "gold": "#E5A23C",
    "red": "#B94B4B",
    "gray": "#66737E",
}
DISPLAY_NAMES = {
    "covariance_reference": "Covariance",
    "statistical_reference": "Statistical",
    "welch_psd": "Welch PSD",
    "morlet_spatiotemporal": "Morlet",
    "psd_morlet": "PSD + Morlet",
    "statistical_morlet": "Statistical + Morlet",
    "full_feature_fusion": "Statistical + PSD + Morlet",
    "covariance_full_fusion": "Covariance + all features",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument(
        "--physionet-cache",
        default="artifacts/harmonized_physionet_openbmi/"
        "physionet_harmonized_epochs.npz",
    )
    parser.add_argument(
        "--openbmi-cache",
        default="artifacts/openbmi_controlled/openbmi_controlled_epochs.npz",
    )
    parser.add_argument("--liu-dir", default="Liu acute-stroke dataset")
    parser.add_argument(
        "--transfer-dir",
        default="artifacts/healthy_to_stroke_hand_intent",
    )
    parser.add_argument(
        "--classical-dir",
        default="artifacts/classical_feature_model_ablation",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/psd_morlet_fusion_ablation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--skip-animation", action="store_true")
    return parser.parse_args()


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F8FAFA",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_controller_ablation(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    x = np.arange(len(rows))
    width = 0.24
    fig, axis = plt.subplots(figsize=(14, 6.2))
    for offset, (field, label, color) in enumerate(
        (
            ("intent_balanced_accuracy", "Intent BA", COLORS["navy"]),
            ("intent_sensitivity", "Sensitivity", COLORS["blue"]),
            ("intent_specificity", "Specificity", COLORS["gold"]),
        )
    ):
        values = [100.0 * float(row["controller"][field]) for row in rows]
        bars = axis.bar(x + (offset - 1) * width, values, width, color=color, label=label)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, fontweight="bold")
    axis.axhline(60, color=COLORS["gray"], linewidth=1, linestyle=":")
    axis.set_xticks(
        x,
        [
            DISPLAY_NAMES[str(row["name"])].replace(" + ", "\n+\n")
            for row in rows
        ],
    )
    axis.set_ylim(0, 100)
    axis.set_ylabel("Controller stroke-patient result (%)")
    axis.set_title("Controller-Only PSD and Morlet Feature-Fusion Ablation")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(
    path: Path,
    final: Mapping[str, Mapping[str, object]],
) -> None:
    order = (
        "covariance_population",
        "statistical_population",
        "selected_population",
        "covariance_calibrated",
        "statistical_calibrated",
        "selected_calibrated",
    )
    labels = (
        "Covariance\npopulation",
        "Statistical\npopulation",
        "Selected fusion\npopulation",
        "Covariance\ncalibrated",
        "Statistical\ncalibrated",
        "Selected fusion\ncalibrated",
    )
    x = np.arange(len(order))
    width = 0.24
    fig, axis = plt.subplots(figsize=(13, 6.4))
    for offset, (field, label, color) in enumerate(
        (
            ("intent_balanced_accuracy", "Intent BA", COLORS["navy"]),
            ("intent_sensitivity", "Sensitivity", COLORS["blue"]),
            ("intent_specificity", "Specificity", COLORS["gold"]),
        )
    ):
        values = [100.0 * float(final[name][field]) for name in order]
        bars = axis.bar(x + (offset - 1) * width, values, width, color=color, label=label)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8, fontweight="bold")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Final stroke-patient result (%)")
    axis.set_title("Final PSD-Morlet Fusion Comparison")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_psd(
    path: Path,
    dataset: HandIntentDataset,
    masks: Mapping[str, np.ndarray],
) -> None:
    central = [
        dataset.channel_names.index(channel) for channel in ("C3", "Cz", "C4")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for axis, (title, cohort_mask) in zip(
        axes,
        (
            ("Healthy final subjects", masks["healthy_final"]),
            ("Stroke final patients", masks["liu2024_final"]),
        ),
    ):
        for intent_value, label, color in (
            (False, "Rest", COLORS["gray"]),
            (True, "Motor intent", COLORS["orange"]),
        ):
            selected = cohort_mask & ((dataset.labels != 0) == intent_value)
            frequencies, density = welch(
                dataset.epochs[selected][:, central],
                fs=dataset.sfreq,
                nperseg=dataset.epochs.shape[-1],
                axis=-1,
            )
            mean_density = np.mean(density, axis=(0, 1))
            shown = (frequencies >= 4.0) & (frequencies <= 38.0)
            axis.plot(
                frequencies[shown],
                10.0 * np.log10(np.clip(mean_density[shown], 1e-30, None)),
                color=color,
                linewidth=2.2,
                label=label,
            )
        axis.axvspan(8, 13, color=COLORS["blue"], alpha=0.09)
        axis.axvspan(13, 30, color=COLORS["gold"], alpha=0.08)
        axis.text(10.5, 0.98, "Mu", transform=axis.get_xaxis_transform(), ha="center", va="top")
        axis.text(21.5, 0.98, "Beta", transform=axis.get_xaxis_transform(), ha="center", va="top")
        axis.set_title(title)
        axis.set_xlabel("Frequency (Hz)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Central-channel PSD (dB/Hz)")
    axes[1].legend(frameon=False)
    fig.suptitle("Welch PSD Before Classification: C3/Cz/C4 Average", weight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def aggregate_morlet_contrast(
    dataset: HandIntentDataset,
    mask: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    sums = np.zeros(
        (
            2,
            len(dataset.channel_names),
            len(MORLET_FREQUENCIES),
            dataset.epochs.shape[-1],
        ),
        dtype=np.float64,
    )
    counts = np.zeros(2, dtype=np.int64)
    indices = np.flatnonzero(mask)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        power = morlet_frequency_power(
            dataset.epochs[batch_indices],
            dataset.sfreq,
        )
        binary = dataset.labels[batch_indices] != 0
        for class_index, intent_value in enumerate((False, True)):
            selected = binary == intent_value
            if np.any(selected):
                sums[class_index] += np.sum(power[selected], axis=0)
                counts[class_index] += int(np.sum(selected))
    means = sums / counts[:, None, None, None]
    return 10.0 * np.log10(
        np.clip(means[1], 1e-30, None)
        / np.clip(means[0], 1e-30, None)
    )


def plot_morlet_spectrograms(
    path: Path,
    dataset: HandIntentDataset,
    healthy_contrast: np.ndarray,
    stroke_contrast: np.ndarray,
) -> None:
    central = [
        dataset.channel_names.index(channel) for channel in ("C3", "Cz", "C4")
    ]
    maps = (
        np.mean(healthy_contrast[central], axis=0),
        np.mean(stroke_contrast[central], axis=0),
    )
    limit = max(0.5, float(np.quantile(np.abs(np.concatenate([row.ravel() for row in maps])), 0.98)))
    times = np.arange(dataset.epochs.shape[-1]) / dataset.sfreq
    fig = plt.figure(figsize=(12.5, 5.3))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.045), wspace=0.18)
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
        ]
    )
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    color_axis = fig.add_subplot(grid[0, 2])
    image = None
    for axis, values, title in zip(
        axes,
        maps,
        ("Healthy final subjects", "Stroke final patients"),
    ):
        image = axis.pcolormesh(
            times,
            MORLET_FREQUENCIES,
            values,
            shading="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.axhline(13, color="white", linewidth=0.8, alpha=0.8)
        axis.axhline(21, color="white", linewidth=0.8, alpha=0.8)
        axis.set_title(title)
        axis.set_xlabel("Time within two-second window (s)")
    axes[0].set_ylabel("Morlet frequency (Hz)")
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Intent minus rest power (dB)")
    fig.suptitle("Variable-Cycle Morlet Contrast: C3/Cz/C4 Average", weight="bold")
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.12, top=0.86)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _topomap_info(channel_names: Sequence[str], sfreq: float) -> mne.Info:
    info = mne.create_info(list(channel_names), sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"))
    return info


def band_contrast(
    contrast: np.ndarray,
    band_index: int,
) -> np.ndarray:
    _, low, high = MORLET_BANDS[band_index]
    selected = (MORLET_FREQUENCIES >= low) & (MORLET_FREQUENCIES < high)
    return np.mean(contrast[:, selected], axis=1)


def plot_scalp_maps(
    path: Path,
    dataset: HandIntentDataset,
    healthy_contrast: np.ndarray,
    stroke_contrast: np.ndarray,
) -> None:
    info = _topomap_info(dataset.channel_names, dataset.sfreq)
    late = slice(dataset.epochs.shape[-1] // 2, None)
    maps = []
    for contrast in (healthy_contrast, stroke_contrast):
        maps.extend(
            [
                np.mean(band_contrast(contrast, band_index)[:, late], axis=-1)
                for band_index in (0, 1)
            ]
        )
    limit = max(0.25, float(np.quantile(np.abs(np.concatenate(maps)), 0.95)))
    fig = plt.figure(figsize=(9.5, 8.2))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        wspace=0.16,
        hspace=0.24,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    color_axis = fig.add_subplot(grid[:, 2])
    image = None
    for axis, values, title in zip(
        axes.ravel(),
        maps,
        (
            "Healthy: mu (8-13 Hz)",
            "Healthy: low beta (13-21 Hz)",
            "Stroke: mu (8-13 Hz)",
            "Stroke: low beta (13-21 Hz)",
        ),
    ):
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            contours=4,
            cmap="RdBu_r",
            vlim=(-limit, limit),
            sensors=True,
        )
        axis.set_title(title)
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Intent minus rest power (dB)")
    fig.suptitle("Late-Window Sensorimotor Morlet Scalp Contrast (1-2 s)", weight="bold")
    fig.subplots_adjust(left=0.04, right=0.92, bottom=0.04, top=0.91)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_scalp_animation(
    path: Path,
    dataset: HandIntentDataset,
    healthy_contrast: np.ndarray,
    stroke_contrast: np.ndarray,
) -> None:
    info = _topomap_info(dataset.channel_names, dataset.sfreq)
    band_maps = {
        "healthy_mu": band_contrast(healthy_contrast, 0),
        "healthy_beta": band_contrast(healthy_contrast, 1),
        "stroke_mu": band_contrast(stroke_contrast, 0),
        "stroke_beta": band_contrast(stroke_contrast, 1),
    }
    limit = max(
        0.25,
        float(
            np.quantile(
                np.abs(np.concatenate([values.ravel() for values in band_maps.values()])),
                0.98,
            )
        ),
    )
    frame_indices = np.linspace(
        0,
        dataset.epochs.shape[-1] - 1,
        20,
        dtype=int,
    )
    fig = plt.figure(figsize=(8.4, 7.4))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        wspace=0.12,
        hspace=0.18,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    color_axis = fig.add_subplot(grid[:, 2])
    titles = (
        ("healthy_mu", "Healthy: mu"),
        ("healthy_beta", "Healthy: low beta"),
        ("stroke_mu", "Stroke: mu"),
        ("stroke_beta", "Stroke: low beta"),
    )
    scalar = ScalarMappable(norm=Normalize(-limit, limit), cmap="RdBu_r")
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("Intent minus rest power (dB)")

    def draw(frame: int) -> list[object]:
        sample = int(frame_indices[frame])
        for axis, (key, title) in zip(axes.ravel(), titles):
            axis.clear()
            mne.viz.plot_topomap(
                band_maps[key][:, sample],
                info,
                axes=axis,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-limit, limit),
                sensors=True,
            )
            axis.set_title(title, fontsize=10, weight="bold")
        fig.suptitle(
            f"Morlet Intent-Minus-Rest Scalp Dynamics | t = {sample / dataset.sfreq:.2f} s",
            fontsize=13,
            weight="bold",
        )
        return list(axes.ravel())

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), interval=180, blit=False)
    animation.save(path, writer=PillowWriter(fps=5), dpi=120)
    plt.close(fig)


def plot_safety(
    path: Path,
    final: Mapping[str, Mapping[str, object]],
) -> None:
    points = (
        ("Covariance calibrated", final["covariance_calibrated"], COLORS["gray"]),
        ("Statistical calibrated", final["statistical_calibrated"], COLORS["blue"]),
        ("Selected fusion calibrated", final["selected_calibrated"], COLORS["orange"]),
    )
    fig, axis = plt.subplots(figsize=(9, 6.2))
    axis.axvspan(0, 10, color="#DCEFE7", alpha=0.75)
    axis.axhspan(60, 100, color="#E8F0F5", alpha=0.5)
    for label, metrics, color in points:
        x = 100.0 * float(metrics["false_activation_rate"])
        y = 100.0 * float(metrics["intent_sensitivity"])
        axis.scatter(x, y, s=145, color=color, edgecolor="white", linewidth=1)
        axis.annotate(label, (x, y), xytext=(7, 6), textcoords="offset points", fontsize=9)
    axis.axvline(10, color=COLORS["teal"], linestyle="--", linewidth=1)
    axis.axhline(60, color=COLORS["navy"], linestyle="--", linewidth=1)
    axis.set_xlim(0, max(25, axis.get_xlim()[1]))
    axis.set_ylim(0, 100)
    axis.set_xlabel("False activation per rest window (%)")
    axis.set_ylabel("Detected intent (%)")
    axis.set_title("Calibrated Safety-Sensitivity Tradeoff")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def metric_text(metrics: Mapping[str, object]) -> str:
    return (
        f"BA {100 * float(metrics['intent_balanced_accuracy']):.1f}%, "
        f"sensitivity {100 * float(metrics['intent_sensitivity']):.1f}%, "
        f"specificity {100 * float(metrics['intent_specificity']):.1f}%, "
        f"false activation {100 * float(metrics['false_activation_rate']):.1f}%"
    )


def write_report(path: Path, report: Mapping[str, object]) -> None:
    selection = report["selection"]
    final = report["final"]
    change = report["paired_vs_statistical"]["calibrated"]["intent_balanced_accuracy"]
    lines = [
        "# PSD and Morlet Spatial-Temporal Fusion Ablation",
        "",
        "## What Changed",
        "",
        "Welch PSD remains a feature-extraction and visualization operation, not a "
        "denoising filter. The new branch combines calibration-normalized Welch "
        "spectral summaries with variable-cycle Morlet mu/beta trajectories and "
        "left-right sensorimotor contrasts. The existing covariance and compact "
        "statistical branches remain independent references.",
        "",
        "All feature choices and safety thresholds were selected on the eight Liu "
        "controller patients. The ten final Liu patients are the same cohort used "
        "by earlier exploratory ablations, so this is not a new untouched "
        "confirmatory result.",
        "",
        "## Controller Selection",
        "",
        f"- Selected branch: **{DISPLAY_NAMES[str(selection['name'])]}**.",
        f"- Feature count: **{selection['feature_count']}**.",
        f"- Controller result: {metric_text(selection['controller'])}.",
        "",
        "## Final Stroke Results",
        "",
        f"- Covariance population: {metric_text(final['covariance_population'])}.",
        f"- Statistical population: {metric_text(final['statistical_population'])}.",
        f"- Selected fusion population: {metric_text(final['selected_population'])}.",
        f"- Covariance calibrated: {metric_text(final['covariance_calibrated'])}.",
        f"- Statistical calibrated: {metric_text(final['statistical_calibrated'])}.",
        f"- Selected fusion calibrated: {metric_text(final['selected_calibrated'])}.",
        "",
        f"Relative to the compact statistical branch, calibrated intent BA changed "
        f"by **{100 * float(change['mean_change']):+.1f} percentage points** "
        f"(paired subject-bootstrap 95% CI "
        f"{100 * float(change['ci_95'][0]):+.1f} to "
        f"{100 * float(change['ci_95'][1]):+.1f}).",
        "",
        "## Visualization Interpretation",
        "",
        "The PSD figure shows stable frequency-domain class summaries. The Morlet "
        "spectrogram and scalp maps preserve timing and spatial distribution. "
        "Values are intent-minus-rest power in dB; negative values indicate lower "
        "power during intent. Because these are two-second task/rest windows without "
        "a trial-specific pre-cue baseline, they are class contrasts rather than "
        "strict event-related desynchronization (ERD).",
        "",
        "The animated GIF is descriptive only. It must not be interpreted as "
        "real-time source localization, and seven shared channels limit scalp "
        "interpolation resolution.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/29_psd_morlet_fusion_ablation.py",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    configure_plots()
    started = time.time()
    transfer_dir = Path(args.transfer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== PSD + Morlet Spatial-Temporal Fusion Ablation ===")
    physionet = from_harmonized_healthy(
        build_physionet_harmonized_dataset(
            args.physionet_dir,
            Path(args.physionet_cache),
        )
    )
    openbmi = from_harmonized_healthy(
        load_openbmi_harmonized_dataset(Path(args.openbmi_cache))
    )
    liu = build_liu_hand_intent_dataset(
        args.liu_dir,
        transfer_dir / "liu_hand_intent_epochs.npz",
    )
    combined = combine_hand_intent_datasets(
        (physionet, openbmi, liu),
        subject_offsets=(PHYSIONET_OFFSET, OPENBMI_OFFSET, LIU_OFFSET),
    )
    partitions = {
        name: load_partitions(transfer_dir / f"{name}_subject_partitions.json")
        for name in ("physionet", "openbmi", "liu2024")
    }
    masks = build_transfer_masks(combined, partitions)
    weights = transfer_training_weights(
        combined.labels,
        combined.dataset_ids,
        masks["mixed_development"],
    )

    covariances = np.concatenate(
        [
            build_covariance_cache(
                dataset.covariance_interface(),
                transfer_dir / f"{dataset.dataset_name}_covariances.npz",
            ).covariances
            for dataset in (physionet, openbmi, liu)
        ]
    )
    covariance_features, covariance_metadata = tangent_features(
        covariances,
        masks["mixed_development"],
        combined.subjects,
        combined.sessions,
        combined.recordings,
        alignment="log_euclidean",
    )
    compact = build_classical_feature_cache(
        combined.epochs,
        combined.sfreq,
        combined.channel_names,
        combined.fingerprint,
        Path(args.classical_dir) / "compact_eeg_features.npz",
    )
    aligned_compact, compact_alignment = calibration_robust_alignment(
        compact.values,
        combined.subjects,
        combined.sessions,
        combined.recordings,
    )
    morlet = build_morlet_feature_cache(
        combined.epochs,
        combined.sfreq,
        combined.channel_names,
        combined.fingerprint,
        output_dir / "morlet_spatiotemporal_features.npz",
        rebuild_cache=args.rebuild_features,
    )
    aligned_morlet, morlet_alignment = calibration_robust_alignment(
        morlet.values,
        combined.subjects,
        combined.sessions,
        combined.recordings,
    )
    statistical = aligned_compact[:, slice(*compact.family_slices["statistical"])]
    spectral = aligned_compact[:, slice(*compact.family_slices["spectral"])]
    feature_sets = {
        "covariance_reference": covariance_features,
        "statistical_reference": statistical,
        "welch_psd": spectral,
        "morlet_spatiotemporal": aligned_morlet,
        "psd_morlet": np.concatenate((spectral, aligned_morlet), axis=1),
        "statistical_morlet": np.concatenate((statistical, aligned_morlet), axis=1),
        "full_feature_fusion": np.concatenate(
            (statistical, spectral, aligned_morlet),
            axis=1,
        ),
        "covariance_full_fusion": np.concatenate(
            (covariance_features, statistical, spectral, aligned_morlet),
            axis=1,
        ),
    }

    print("[+] Controller-only PSD/Morlet fusion selection...")
    rows = []
    models = {}
    probabilities = {}
    thresholds = {}
    for name, features in feature_sets.items():
        model, probability, threshold, metrics = fit_intent_candidate(
            features,
            combined,
            masks,
            weights,
            seed=args.seed,
            max_false_activation=args.max_false_activation,
        )
        models[name] = model
        probabilities[name] = probability
        thresholds[name] = threshold
        rows.append(
            {
                "name": name,
                "feature_count": int(features.shape[1]),
                "controller": metrics,
            }
        )
        print(
            f"    {name:28s} | BA={100 * metrics['intent_balanced_accuracy']:5.1f}% "
            f"| FA={100 * metrics['false_activation_rate']:5.1f}% "
            f"| missed={100 * metrics['missed_intent_rate']:5.1f}%"
        )
    selected_row = max(
        rows,
        key=lambda row: (
            float(row["controller"]["intent_balanced_accuracy"]),
            float(row["controller"]["intent_sensitivity"]),
            -float(row["controller"]["false_activation_rate"]),
        ),
    )
    selected_name = str(selected_row["name"])
    final_mask = masks["liu2024_final"]

    print(f"[+] Calibrating references and selected branch: {selected_name}")
    calibrations = {}
    adapted = {}
    adapted_thresholds = {}
    for name in ("covariance_reference", "statistical_reference", selected_name):
        calibration, adapted_probability, adapted_threshold = select_patient_calibration(
            feature_sets[name],
            combined,
            masks,
            probabilities[name],
            seed=args.seed,
            max_false_activation=args.max_false_activation,
        )
        calibrations[name] = calibration
        adapted[name] = adapted_probability
        adapted_thresholds[name] = adapted_threshold

    final = {
        "covariance_population": score_intent(
            combined.labels[final_mask],
            probabilities["covariance_reference"][final_mask],
            thresholds["covariance_reference"],
        ),
        "statistical_population": score_intent(
            combined.labels[final_mask],
            probabilities["statistical_reference"][final_mask],
            thresholds["statistical_reference"],
        ),
        "selected_population": score_intent(
            combined.labels[final_mask],
            probabilities[selected_name][final_mask],
            thresholds[selected_name],
        ),
        "covariance_calibrated": score_intent(
            combined.labels[final_mask],
            adapted["covariance_reference"][final_mask],
            adapted_thresholds["covariance_reference"],
        ),
        "statistical_calibrated": score_intent(
            combined.labels[final_mask],
            adapted["statistical_reference"][final_mask],
            adapted_thresholds["statistical_reference"],
        ),
        "selected_calibrated": score_intent(
            combined.labels[final_mask],
            adapted[selected_name][final_mask],
            adapted_thresholds[selected_name],
        ),
    }
    paired_vs_statistical = {
        "population": paired_subject_bootstrap(
            combined.labels[final_mask],
            combined.subjects[final_mask],
            probabilities["statistical_reference"][final_mask],
            thresholds["statistical_reference"],
            probabilities[selected_name][final_mask],
            thresholds[selected_name],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        ),
        "calibrated": paired_subject_bootstrap(
            combined.labels[final_mask],
            combined.subjects[final_mask],
            adapted["statistical_reference"][final_mask],
            adapted_thresholds["statistical_reference"],
            adapted[selected_name][final_mask],
            adapted_thresholds[selected_name],
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        ),
    }

    print("[+] Building PSD, Morlet, scalp, and temporal visualizations...")
    plot_controller_ablation(output_dir / "01_controller_fusion_ablation.png", rows)
    plot_final_comparison(output_dir / "02_final_fusion_comparison.png", final)
    plot_psd(output_dir / "03_welch_psd_intent_vs_rest.png", combined, masks)
    healthy_contrast = aggregate_morlet_contrast(combined, masks["healthy_final"])
    stroke_contrast = aggregate_morlet_contrast(combined, masks["liu2024_final"])
    plot_morlet_spectrograms(
        output_dir / "04_morlet_time_frequency_contrast.png",
        combined,
        healthy_contrast,
        stroke_contrast,
    )
    plot_scalp_maps(
        output_dir / "05_morlet_scalp_contrast.png",
        combined,
        healthy_contrast,
        stroke_contrast,
    )
    if not args.skip_animation:
        save_scalp_animation(
            output_dir / "06_morlet_scalp_dynamics.gif",
            combined,
            healthy_contrast,
            stroke_contrast,
        )
    plot_safety(output_dir / "07_calibrated_safety_tradeoff.png", final)

    elapsed = (time.time() - started) / 60.0
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "psd_morlet_spatiotemporal_fusion_ablation",
        "selection_boundary": (
            "Feature branch, calibration, and thresholds selected only on "
            "controller stroke patients."
        ),
        "confirmation_boundary": (
            "Exploratory paired analysis on the previously observed final cohort."
        ),
        "feature_definitions": {
            "welch_psd_features": int(spectral.shape[1]),
            "morlet_features": int(aligned_morlet.shape[1]),
            "morlet_frequencies_hz": MORLET_FREQUENCIES,
            "morlet_bands": MORLET_BANDS,
            "candidate_dimensions": {
                name: int(values.shape[1]) for name, values in feature_sets.items()
            },
            "compact_alignment": compact_alignment,
            "morlet_alignment": morlet_alignment,
            "covariance_metadata": covariance_metadata,
        },
        "controller_candidates": rows,
        "selection": selected_row,
        "calibration": calibrations,
        "final": final,
        "paired_vs_statistical": paired_vs_statistical,
        "runtime_minutes": elapsed,
    }
    (output_dir / "psd_morlet_fusion_metrics.json").write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    write_csv(
        output_dir / "controller_candidate_metrics.csv",
        [
            {
                "candidate": row["name"],
                "features": row["feature_count"],
                **row["controller"],
            }
            for row in rows
        ],
    )
    joblib.dump(
        {
            "models": models,
            "selected_name": selected_name,
            "thresholds": thresholds,
            "adapted_thresholds": adapted_thresholds,
            "morlet_feature_names": morlet.names,
            "fingerprint": combined.fingerprint,
        },
        output_dir / "psd_morlet_fusion_models.joblib",
    )
    write_report(output_dir / "PSD_MORLET_FUSION_ABLATION.md", report)

    print("\n=== Final Stroke-Patient Results ===")
    for name, metrics in final.items():
        print(
            f"{name:28s} | BA={100 * metrics['intent_balanced_accuracy']:5.1f}% "
            f"| sensitivity={100 * metrics['intent_sensitivity']:5.1f}% "
            f"| specificity={100 * metrics['intent_specificity']:5.1f}% "
            f"| FA={100 * metrics['false_activation_rate']:5.1f}%"
        )
    print(f"[+] Saved report, data, models, and visualizations to {output_dir}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
