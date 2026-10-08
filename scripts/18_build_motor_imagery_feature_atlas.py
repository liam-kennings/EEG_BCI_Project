#!/usr/bin/env python3
"""Build and visualize a leakage-safe, multibranch motor-imagery feature atlas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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

from bci_pipeline.dataset import (  # noqa: E402
    LABEL_NAMES,
    available_subjects,
    parse_int_list,
    resolve_data_dir,
)
from bci_pipeline.feature_atlas import (  # noqa: E402
    CALIBRATION_RUNS,
    COVARIANCE_PHASES,
    MORLET_BANDS,
    MOTOR_CHANNELS,
    SLOW_AUDIT_CHANNELS,
    SLOW_TIME_EDGES,
    TIME_EDGES,
    FeatureAtlasDataset,
    build_feature_atlas,
    label_counts,
)
from bci_pipeline.representations.riemann import FILTER_BANK  # noqa: E402
from bci_pipeline.splits import load_partitions  # noqa: E402


COLORS = {
    0: "#5D6770",
    1: "#147D92",
    2: "#D66A3A",
    "accent": "#E9A23B",
    "quality": "#2D6A4F",
}


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F8FAFC",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_recording_quality(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    integer_fields = {"bad_channel_count", "subject", "run"}
    text_fields = {"bad_channels", "qc_level", "file"}
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="ascii") as handle:
        for source in csv.DictReader(handle):
            row: Dict[str, object] = {}
            for name, value in source.items():
                if name == "bad_channels":
                    row[name] = value.split(";") if value else []
                elif name in integer_fields:
                    row[name] = int(value)
                elif name in text_fields:
                    row[name] = value
                else:
                    row[name] = float(value)
            rows.append(row)
    return rows


def audit_mask(dataset: FeatureAtlasDataset, split_path: Path) -> np.ndarray:
    partitions = load_partitions(split_path)
    mask = np.isin(dataset.subjects, partitions.development)
    if not np.any(mask):
        print("[!] Requested subjects do not include development subjects; using all for smoke plots.")
        return np.ones(len(dataset.labels), dtype=bool)
    return mask


def plot_recording_quality(
    output_dir: Path,
    recording_rows: Sequence[Mapping[str, object]],
    dataset: FeatureAtlasDataset,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    if recording_rows:
        bad_counts = np.asarray([row["bad_channel_count"] for row in recording_rows])
        high = np.asarray(
            [row["median_high_frequency_ratio"] for row in recording_rows]
        )
        line = np.asarray([row["median_line_ratio"] for row in recording_rows])
        rms = np.asarray([row["median_rms_uv"] for row in recording_rows])
        axes[0, 0].hist(bad_counts, bins=np.arange(-0.5, 16.5), color=COLORS["quality"])
        axes[0, 0].set_xlabel("Channels interpolated per recording")
        axes[0, 0].set_ylabel("Recordings")
        axes[0, 0].set_title("Pre-reference Channel QC")
        axes[0, 1].scatter(high, line, s=22, alpha=0.65, color="#277DA1")
        axes[0, 1].set_xlabel("Median 35-75 Hz / 1-75 Hz power")
        axes[0, 1].set_ylabel("Median 58-62 Hz / 1-75 Hz power")
        axes[0, 1].set_title("Muscle and Line-Noise Indicators")
        axes[1, 0].hist(rms, bins=24, color="#90BE6D")
        axes[1, 0].set_xlabel("Median channel RMS (uV)")
        axes[1, 0].set_ylabel("Recordings")
        axes[1, 0].set_title("Broadband Recording Amplitude")
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "Recording-level rows are retained\nfrom the extraction run.",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 1].set_visible(False)
        axes[1, 0].set_visible(False)
    quality = dataset.quality_features
    axes[1, 1].scatter(
        quality[:, 0],
        quality[:, 5],
        c=[COLORS[int(label)] for label in dataset.labels],
        s=10,
        alpha=0.35,
    )
    axes[1, 1].set_xlabel("Trial maximum peak-to-peak (uV)")
    axes[1, 1].set_ylabel("Trial high-frequency power ratio")
    axes[1, 1].set_title("Trial-Level Quality Context")
    fig.suptitle("EEG Quality Audit Before Classification", fontsize=15, weight="bold")
    fig.savefig(output_dir / "01_recording_quality_dashboard.png", dpi=300)
    plt.close(fig)


def plot_slow_potentials(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> None:
    times = 0.5 * (SLOW_TIME_EDGES[:-1] + SLOW_TIME_EDGES[1:])
    central = [0, 1, 2]
    occipital = [3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True, constrained_layout=True)
    for axis, indices, title in (
        (axes[0], central, "Central: C3 / Cz / C4"),
        (axes[1], occipital, "Occipital Control: O1 / Oz / O2"),
    ):
        for label in sorted(LABEL_NAMES):
            selected = mask & (dataset.labels == label)
            trials = np.mean(dataset.slow_audit[selected][:, indices], axis=1)
            mean = np.mean(trials, axis=0)
            sem = np.std(trials, axis=0) / np.sqrt(max(len(trials), 1))
            axis.plot(times, mean, color=COLORS[label], linewidth=2.3, label=LABEL_NAMES[label])
            axis.fill_between(times, mean - sem, mean + sem, color=COLORS[label], alpha=0.14)
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.axvspan(0.0, 0.5, color="#F6BD60", alpha=0.18, label="Visual-cue interval")
        axis.set_xlabel("Seconds after event marker")
        axis.set_title(title)
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("Slow potential relative to calibration rest (z)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Cue-Locked Slow Potentials: Motor Signal Versus Visual Confound",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output_dir / "02_slow_potential_motor_vs_visual.png", dpi=300)
    plt.close(fig)


def morlet_tensor(dataset: FeatureAtlasDataset) -> np.ndarray:
    broad_count = (
        len(MOTOR_CHANNELS) * len(MORLET_BANDS) * (len(TIME_EDGES) - 1)
    )
    return dataset.morlet_features[:, :broad_count].reshape(
        len(dataset.labels),
        len(MOTOR_CHANNELS),
        len(MORLET_BANDS),
        len(TIME_EDGES) - 1,
    )


def plot_morlet_trajectories(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> None:
    values = morlet_tensor(dataset)
    times = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    central = [MOTOR_CHANNELS.index(name) for name in ("C3", "Cz", "C4")]
    band_indices = {
        name: index for index, (name, _, _) in enumerate(MORLET_BANDS)
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True, constrained_layout=True)
    for axis, (band_name, title) in zip(
        axes, (("mu_high", "Upper Mu (10-14 Hz)"), ("beta_low", "Low Beta (14-22 Hz)"))
    ):
        band = band_indices[band_name]
        for label in sorted(LABEL_NAMES):
            selected = mask & (dataset.labels == label)
            trials = np.mean(values[selected][:, central, band], axis=1)
            mean = np.mean(trials, axis=0)
            sem = np.std(trials, axis=0) / np.sqrt(max(len(trials), 1))
            axis.plot(times, mean, color=COLORS[label], linewidth=2.3, label=LABEL_NAMES[label])
            axis.fill_between(times, mean - sem, mean + sem, color=COLORS[label], alpha=0.14)
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_xlabel("Seconds after cue")
        axis.set_title(title)
        axis.grid(alpha=0.20)
    axes[0].set_ylabel("Power relative to Runs 4/8 rest (dB)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Morlet Sensorimotor ERD/ERS Trajectories", fontsize=14, weight="bold")
    fig.savefig(output_dir / "03_morlet_erd_trajectories.png", dpi=300)
    plt.close(fig)


def plot_effect_atlas(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> None:
    values = morlet_tensor(dataset)
    central = [MOTOR_CHANNELS.index(name) for name in ("C3", "Cz", "C4")]
    rest = np.mean(values[mask & (dataset.labels == 0)][:, central], axis=(0, 1))
    intent = np.mean(values[mask & (dataset.labels != 0)][:, central], axis=(0, 1))
    effect = intent - rest
    limit = max(float(np.max(np.abs(effect))), 0.25)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    image = ax.imshow(effect, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(
        np.arange(len(MORLET_BANDS)),
        [name.replace("_", " ") for name, _, _ in MORLET_BANDS],
    )
    times = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    ax.set_xticks(np.arange(len(times))[::2], [f"{value:.2f}" for value in times[::2]])
    ax.set_xlabel("Seconds after cue")
    ax.set_ylabel("Frequency band")
    ax.set_title("Intent Minus Rest: Central Morlet Feature Effect Atlas")
    fig.colorbar(image, ax=ax, label="Difference in ERD/ERS (dB)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_morlet_effect_atlas.png", dpi=300)
    plt.close(fig)


def plot_csd_topographies(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> None:
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(dataset.channel_names, dataset.sfreq, "eeg")
    info.set_montage(montage, on_missing="raise")
    band_lookup = {name: index for index, (name, _, _) in enumerate(MORLET_BANDS)}
    panels = (
        (1, "mu_high", "Left imagery - rest | Mu"),
        (2, "mu_high", "Right imagery - rest | Mu"),
        (1, "beta_low", "Left imagery - rest | Beta"),
        (2, "beta_low", "Right imagery - rest | Beta"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.5))
    maps = []
    for label, band_name, _ in panels:
        band = band_lookup[band_name]
        target = np.mean(dataset.csd_topography[mask & (dataset.labels == label), :, band], axis=0)
        rest = np.mean(dataset.csd_topography[mask & (dataset.labels == 0), :, band], axis=0)
        maps.append(target - rest)
    limit = max(float(np.percentile(np.abs(np.concatenate(maps)), 98)), 0.25)
    image = None
    for axis, values, (_, _, title) in zip(axes.ravel(), maps, panels):
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            cmap="RdBu_r",
            vlim=(-limit, limit),
        )
        axis.set_title(title)
    fig.subplots_adjust(top=0.88, right=0.87, hspace=0.28, wspace=0.15)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="CSD power contrast (dB)", shrink=0.78)
    fig.suptitle("Surface-Laplacian Sensorimotor Spatial Contrast", fontsize=14, weight="bold")
    fig.savefig(output_dir / "05_csd_spatial_contrast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_covariance_trajectory(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> None:
    values = dataset.covariances[mask]
    labels = dataset.labels[mask]
    distances = np.zeros((len(FILTER_BANK), len(COVARIANCE_PHASES)))
    for band in range(len(FILTER_BANK)):
        for phase in range(len(COVARIANCE_PHASES)):
            rest = np.mean(values[labels == 0, band, phase], axis=0)
            intent = np.mean(values[labels != 0, band, phase], axis=0)
            distances[band, phase] = np.linalg.norm(intent - rest, ord="fro")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(COVARIANCE_PHASES))
    for band, (name, _, _) in enumerate(FILTER_BANK):
        ax.plot(x, distances[band], marker="o", linewidth=2.0, label=name.replace("_", " "))
    ax.set_xticks(x, [name.replace("_", " ") for name, _, _ in COVARIANCE_PHASES])
    ax.set_ylabel("Frobenius distance: mean intent vs rest")
    ax.set_xlabel("Cue-locked phase")
    ax.set_title("Phase-Resolved Sensorimotor Covariance Separation")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "06_covariance_phase_trajectory.png", dpi=300)
    plt.close(fig)


def plot_personalization(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> None:
    subject_peaks = [
        float(dataset.individual_mu_hz[dataset.subjects == subject][0])
        for subject in np.unique(dataset.subjects[mask])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    axes[0].hist(
        subject_peaks,
        bins=np.arange(6.5, 15.6, 1.0),
        color=COLORS["accent"],
        edgecolor="white",
    )
    axes[0].set_xlabel("Task-relative individual mu center (Hz)")
    axes[0].set_ylabel("Development subjects")
    axes[0].set_title("Calibration-Derived Mu Frequency")
    quality = dataset.quality_features[mask]
    labels = dataset.labels[mask]
    data = [quality[labels == label, 0] for label in sorted(LABEL_NAMES)]
    boxes = axes[1].boxplot(data, patch_artist=True, showfliers=False)
    for patch, label in zip(boxes["boxes"], sorted(LABEL_NAMES)):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.78)
    axes[1].set_xticklabels([LABEL_NAMES[label] for label in sorted(LABEL_NAMES)], rotation=15)
    axes[1].set_ylabel("Maximum trial peak-to-peak (uV)")
    axes[1].set_title("Quality Distribution by Event Class")
    fig.suptitle("Patient Frequency Calibration and Trial Quality", fontsize=14, weight="bold")
    fig.savefig(output_dir / "07_patient_frequency_and_quality.png", dpi=300)
    plt.close(fig)


def plot_calibration_reliability(
    output_dir: Path, dataset: FeatureAtlasDataset, mask: np.ndarray
) -> Dict[str, float]:
    morlet = morlet_tensor(dataset)
    central_motor = [MOTOR_CHANNELS.index(name) for name in ("C3", "Cz", "C4")]
    central_slow = [0, 1, 2]
    mu_index = [
        index for index, (name, _, _) in enumerate(MORLET_BANDS) if name == "mu_high"
    ][0]
    csd = dataset.csd_features.reshape(
        len(dataset.labels),
        len(MOTOR_CHANNELS),
        len(MORLET_BANDS),
        len(TIME_EDGES) - 1,
    )
    subjects = np.unique(dataset.subjects[mask])
    effects: Dict[str, Dict[int, List[float]]] = {
        name: {4: [], 8: []} for name in ("slow", "morlet", "csd", "covariance")
    }
    for subject in subjects:
        for run in (4, 8):
            selected = mask & (dataset.subjects == subject) & (dataset.runs == run)
            rest = selected & (dataset.labels == 0)
            intent = selected & (dataset.labels != 0)
            effects["slow"][run].append(
                float(
                    np.mean(dataset.slow_audit[intent][:, central_slow])
                    - np.mean(dataset.slow_audit[rest][:, central_slow])
                )
            )
            effects["morlet"][run].append(
                float(
                    np.mean(morlet[intent][:, central_motor, mu_index])
                    - np.mean(morlet[rest][:, central_motor, mu_index])
                )
            )
            effects["csd"][run].append(
                float(
                    np.mean(csd[intent][:, central_motor, mu_index])
                    - np.mean(csd[rest][:, central_motor, mu_index])
                )
            )
            rest_covariance = np.mean(dataset.covariances[rest], axis=0)
            intent_covariance = np.mean(dataset.covariances[intent], axis=0)
            effects["covariance"][run].append(
                float(np.mean(np.linalg.norm(intent_covariance - rest_covariance, axis=(-2, -1))))
            )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    correlations: Dict[str, float] = {}
    titles = {
        "slow": "Central Slow-Potential Contrast",
        "morlet": "Central Mu Morlet Contrast",
        "csd": "Central Mu CSD Contrast",
        "covariance": "Covariance Intent-Rest Distance",
    }
    for axis, name in zip(axes.ravel(), effects):
        run4 = np.asarray(effects[name][4])
        run8 = np.asarray(effects[name][8])
        correlation = float(np.corrcoef(run4, run8)[0, 1])
        correlations[name] = correlation
        axis.scatter(run4, run8, s=34, alpha=0.72, color=branch_color(name))
        lower = float(min(np.min(run4), np.min(run8)))
        upper = float(max(np.max(run4), np.max(run8)))
        padding = max((upper - lower) * 0.08, 1e-6)
        axis.plot(
            [lower - padding, upper + padding],
            [lower - padding, upper + padding],
            linestyle="--",
            color="#5D6770",
            linewidth=1.0,
        )
        axis.set_xlim(lower - padding, upper + padding)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_xlabel("Run 4 subject effect")
        axis.set_ylabel("Run 8 subject effect")
        axis.set_title(f"{titles[name]} | r={correlation:.2f}")
        axis.grid(alpha=0.20)
    fig.suptitle(
        "Calibration Repeatability Across Independent Imagery Runs",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output_dir / "08_calibration_feature_repeatability.png", dpi=300)
    plt.close(fig)
    return correlations


def branch_color(name: str) -> str:
    return {
        "slow": "#6C757D",
        "morlet": "#D66A3A",
        "csd": "#147D92",
        "covariance": "#2D6A4F",
    }[name]


def trial_quality_rows(dataset: FeatureAtlasDataset) -> List[Dict[str, object]]:
    rows = []
    names = dataset.feature_names["quality"]
    for index in range(len(dataset.labels)):
        row: Dict[str, object] = {
            "subject": int(dataset.subjects[index]),
            "run": int(dataset.runs[index]),
            "trial_index": int(dataset.trial_indices[index]),
            "label": int(dataset.labels[index]),
            "label_name": LABEL_NAMES[int(dataset.labels[index])],
            "individual_mu_hz": float(dataset.individual_mu_hz[index]),
        }
        row.update(
            {
                name: float(value)
                for name, value in zip(names, dataset.quality_features[index])
            }
        )
        rows.append(row)
    return rows


def write_report(
    output_dir: Path,
    dataset: FeatureAtlasDataset,
    subjects: Sequence[int],
    recording_rows: Sequence[Mapping[str, object]],
    reliability: Mapping[str, float],
    elapsed_minutes: float,
) -> None:
    report = {
        "schema_version": 1,
        "protocol": {
            "task": "unilateral motor imagery: rest vs left fist vs right fist",
            "runs": [4, 8, 12],
            "calibration_reference_runs": list(CALIBRATION_RUNS),
            "held_out_evaluation_run": 12,
            "reference": "bad-channel repair then common-average reference",
            "spatial_branch": "surface Laplacian / current-source density",
            "slow_band_hz": [0.1, 4.0],
            "morlet_bands": [list(values) for values in MORLET_BANDS],
            "covariance_filter_bank": [list(values) for values in FILTER_BANK],
            "important_limitation": (
                "EEGMMIDB markers are visual cue/task-state markers, not measured "
                "movement-onset markers; the slow branch is cue-locked, not a "
                "validated self-paced MRCP detector."
            ),
        },
        "dataset": {
            "subjects": len(set(int(value) for value in subjects)),
            "trials": int(len(dataset.labels)),
            "class_counts": label_counts(dataset.labels),
            "recordings": int(len(recording_rows)) if recording_rows else None,
        },
        "feature_dimensions": {
            "slow": int(dataset.slow_features.shape[1]),
            "morlet": int(dataset.morlet_features.shape[1]),
            "csd": int(dataset.csd_features.shape[1]),
            "covariance_views": list(dataset.covariances.shape[1:3]),
            "covariance_channels": int(dataset.covariances.shape[-1]),
            "quality_context": int(dataset.quality_features.shape[1]),
        },
        "runtime_minutes": elapsed_minutes,
        "run4_run8_feature_repeatability": {
            name: float(value) for name, value in reliability.items()
        },
        "artifacts": {
            "feature_cache": "motor_imagery_feature_atlas.npz",
            "recording_qc": "recording_quality.csv",
            "trial_quality": "trial_quality.csv",
            "figures": [f"{index:02d}" for index in range(1, 9)],
        },
    }
    (output_dir / "feature_atlas_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii"
    )
    markdown = f"""# Motor-Imagery Physiology Feature Atlas

## Purpose

This stage stops before machine learning. It verifies what signal information is
available after raw recording QC, bad-channel repair, CAR, CSD, slow-potential
extraction, Morlet ERD/ERS, and phase-resolved covariance.

## Dataset

- Subjects: {report["dataset"]["subjects"]}
- Trials: {report["dataset"]["trials"]}
- Rest / left / right: {report["dataset"]["class_counts"]["rest"]} /
  {report["dataset"]["class_counts"]["left_fist"]} /
  {report["dataset"]["class_counts"]["right_fist"]}
- Calibration reference: post-T0 rest from Runs 4 and 8
- Locked evaluation run for the next stage: Run 12

## Feature Blocks

- Cue-locked 0.1-4 Hz slow potentials: {dataset.slow_features.shape[1]} features
- Morlet time-frequency trajectories: {dataset.morlet_features.shape[1]} features
- Surface-Laplacian ERD trajectories: {dataset.csd_features.shape[1]} features
- Covariance trajectory: {dataset.covariances.shape[1]} bands x
  {dataset.covariances.shape[2]} phases x {dataset.covariances.shape[-1]} channels
- Independent quality context: {dataset.quality_features.shape[1]} values

## Run 4 Versus Run 8 Repeatability

- Slow-potential subject effect: r={reliability["slow"]:.2f}
- Morlet mu subject effect: r={reliability["morlet"]:.2f}
- CSD mu subject effect: r={reliability["csd"]:.2f}
- Covariance intent-rest distance: r={reliability["covariance"]:.2f}

## Interpretation Guardrail

EEGMMIDB markers indicate visual cue and task-state transitions. They are not
measured movement-onset markers. The 0.1-4 Hz branch can therefore be described
as a cue-locked slow-potential branch, but not yet as a validated self-paced MRCP
detector. The central-versus-occipital figure is included specifically to audit
visual-cue contamination before classification.
"""
    (output_dir / "FEATURE_ATLAS_RESULTS.md").write_text(markdown, encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--cache-path",
        default="artifacts/feature_atlas/motor_imagery_feature_atlas.npz",
    )
    parser.add_argument("--output-dir", default="artifacts/feature_atlas")
    parser.add_argument("--split-path", default="artifacts/dataset/splits.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    configure_plots()
    started = time.time()
    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(
        data_dir, parse_int_list(args.subjects), args.max_subjects
    )
    if not subjects:
        raise ValueError("No subjects selected.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== Motor-Imagery Physiology Feature Atlas ===")
    print(f"Data: {data_dir}")
    print("Protocol: Runs 4/8 calibration reference -> Run 12 held-out evaluation")
    dataset, recording_rows = build_feature_atlas(
        data_dir,
        subjects,
        Path(args.cache_path),
        rebuild_cache=args.rebuild_cache,
    )
    print(
        f"[+] Atlas: {len(dataset.labels)} trials | "
        f"{len(np.unique(dataset.subjects))} subjects"
    )
    if not recording_rows:
        recording_rows = read_recording_quality(
            output_dir / "recording_quality.csv"
        )
    if recording_rows:
        serializable = []
        for row in recording_rows:
            converted = dict(row)
            converted["bad_channels"] = ";".join(row["bad_channels"])
            serializable.append(converted)
        write_csv(output_dir / "recording_quality.csv", serializable)
    write_csv(output_dir / "trial_quality.csv", trial_quality_rows(dataset))
    mask = audit_mask(dataset, Path(args.split_path))
    plot_recording_quality(output_dir, recording_rows, dataset)
    plot_slow_potentials(output_dir, dataset, mask)
    plot_morlet_trajectories(output_dir, dataset, mask)
    plot_effect_atlas(output_dir, dataset, mask)
    plot_csd_topographies(output_dir, dataset, mask)
    plot_covariance_trajectory(output_dir, dataset, mask)
    plot_personalization(output_dir, dataset, mask)
    reliability = plot_calibration_reliability(output_dir, dataset, mask)
    elapsed = (time.time() - started) / 60.0
    write_report(
        output_dir,
        dataset,
        subjects,
        recording_rows,
        reliability,
        elapsed,
    )
    print(f"[+] Saved feature cache, audits, and eight figures to {output_dir.resolve()}")
    print(f"[+] Runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
