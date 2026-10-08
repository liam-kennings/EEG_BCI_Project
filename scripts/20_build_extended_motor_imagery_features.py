#!/usr/bin/env python3
"""Build and audit physiological feature extensions before ML classification."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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

from bci_pipeline.dataset import LABEL_NAMES, resolve_data_dir  # noqa: E402
from bci_pipeline.extended_features import (  # noqa: E402
    PHASE_TIME_EDGES,
    SSD_COMPONENT_COUNT,
    ExtendedFeatureDataset,
    build_extended_features,
)
from bci_pipeline.feature_atlas import (  # noqa: E402
    CALIBRATION_RUNS,
    TIME_EDGES,
    FeatureAtlasDataset,
    load_feature_atlas,
)
from bci_pipeline.splits import load_partitions  # noqa: E402


COLORS = {
    0: "#5D6770",
    1: "#147D92",
    2: "#D66A3A",
    "ssd": "#2D6A4F",
    "asymmetry": "#147D92",
    "morphology": "#D66A3A",
    "phase": "#8E5572",
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


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _feature_columns(names: Sequence[str], prefix: str) -> np.ndarray:
    selected = [index for index, name in enumerate(names) if name.startswith(prefix)]
    if not selected:
        raise ValueError(f"No features matched prefix: {prefix}")
    return np.asarray(selected, dtype=np.int64)


def _mean_and_sem(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.mean(values, axis=0),
        np.std(values, axis=0) / np.sqrt(max(len(values), 1)),
    )


def plot_ssd_audit(
    output_dir: Path,
    dataset: ExtendedFeatureDataset,
    trial_mask: np.ndarray,
    profile_mask: np.ndarray,
) -> None:
    montage = mne.channels.make_standard_montage("standard_1005")
    info = mne.create_info(dataset.channel_names, 160.0, "eeg")
    info.set_montage(montage, on_missing="raise")
    times = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), constrained_layout=True)
    normalized = dataset.ssd_patterns[profile_mask].copy()
    scale = np.max(np.abs(normalized), axis=2, keepdims=True)
    normalized /= np.clip(scale, 1e-12, None)
    for band, band_name in enumerate(dataset.ssd_band_names):
        pattern = np.mean(normalized[:, band, :, 0], axis=0)
        limit = max(float(np.max(np.abs(pattern))), 0.1)
        mne.viz.plot_topomap(
            pattern,
            info,
            axes=axes[0, band],
            show=False,
            cmap="RdBu_r",
            vlim=(-limit, limit),
            contours=4,
        )
        eigenvalue = np.median(dataset.ssd_eigenvalues[profile_mask, band, 0])
        axes[0, band].set_title(
            f"{band_name.replace('_', ' ').title()}\n"
            f"Component 1 | median ratio {eigenvalue:.2f}"
        )
        prefix = f"ssd_{band_name}_component_1_"
        columns = _feature_columns(dataset.feature_names["ssd"], prefix)
        for label in sorted(LABEL_NAMES):
            selected = trial_mask & (dataset.labels == label)
            mean, sem = _mean_and_sem(dataset.ssd_features[selected][:, columns])
            axes[1, band].plot(
                times,
                mean,
                color=COLORS[label],
                linewidth=2.2,
                label=LABEL_NAMES[label],
            )
            axes[1, band].fill_between(
                times, mean - sem, mean + sem, color=COLORS[label], alpha=0.14
            )
        axes[1, band].axhline(0.0, color="#222222", linewidth=0.8)
        axes[1, band].set_xlabel("Seconds after cue")
        axes[1, band].set_title("SSD Component 1 ERD/ERS")
        axes[1, band].grid(alpha=0.20)
    axes[1, 0].set_ylabel("Power relative to calibration rest (dB)")
    axes[1, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Spatio-Spectral Decomposition: Spatial Patterns and Time Courses",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "01_ssd_spatial_and_temporal_audit.png", dpi=300)
    plt.close(fig)


def plot_asymmetry_trajectories(
    output_dir: Path, dataset: ExtendedFeatureDataset, mask: np.ndarray
) -> None:
    times = 0.5 * (TIME_EDGES[:-1] + TIME_EDGES[1:])
    panels = (
        (
            "morlet_C3_minus_C4_individual_mu_",
            "C3 Minus C4",
            "C3 - C4 power (dB)",
        ),
        (
            "morlet_left_minus_right_hemisphere_individual_mu_",
            "Left Minus Right Sensorimotor Hemisphere",
            "Left - right power (dB)",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    for axis, (prefix, title, ylabel) in zip(axes, panels):
        columns = _feature_columns(dataset.feature_names["asymmetry"], prefix)
        for label in sorted(LABEL_NAMES):
            selected = mask & (dataset.labels == label)
            mean, sem = _mean_and_sem(dataset.asymmetry_features[selected][:, columns])
            axis.plot(
                times,
                mean,
                color=COLORS[label],
                linewidth=2.3,
                label=LABEL_NAMES[label],
            )
            axis.fill_between(
                times, mean - sem, mean + sem, color=COLORS[label], alpha=0.14
            )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_xlabel("Seconds after cue")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.20)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Explicit Hemispheric Asymmetry in Individual Mu",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "02_hemispheric_asymmetry_trajectories.png", dpi=300)
    plt.close(fig)


def plot_morphology_audit(
    output_dir: Path, dataset: ExtendedFeatureDataset, mask: np.ndarray
) -> None:
    names = dataset.feature_names["morphology"]
    panels = (
        (
            "negative_area_db",
            "Sustained ERD Area",
            "Mean negative ERD (dB)",
        ),
        (
            "minimum_db",
            "Peak ERD Depth",
            "Minimum ERD (dB)",
        ),
        (
            "time_to_minimum_s",
            "Peak ERD Timing",
            "Seconds after cue",
        ),
        (
            "fraction_below_rest",
            "ERD Persistence",
            "Fraction of analysis window",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.ravel(), panels):
        columns = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if name.startswith("morlet_")
                and "_individual_mu_" in name
                and name.endswith(metric)
                and ("left_motor" in name or "right_motor" in name)
            ],
            dtype=np.int64,
        )
        values = np.mean(dataset.morphology_features[:, columns], axis=1)
        groups = [values[mask & (dataset.labels == label)] for label in sorted(LABEL_NAMES)]
        box = axis.boxplot(
            groups,
            tick_labels=[LABEL_NAMES[label] for label in sorted(LABEL_NAMES)],
            patch_artist=True,
            showfliers=False,
            widths=0.65,
        )
        for patch, label in zip(box["boxes"], sorted(LABEL_NAMES)):
            patch.set_facecolor(COLORS[label])
            patch.set_alpha(0.78)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle(
        "Individual-Mu ERD Morphology Before Classification",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "03_erd_morphology_audit.png", dpi=300)
    plt.close(fig)


def plot_phase_audit(
    output_dir: Path, dataset: ExtendedFeatureDataset, mask: np.ndarray
) -> None:
    times = 0.5 * (PHASE_TIME_EDGES[:-1] + PHASE_TIME_EDGES[1:])
    panels = (
        ("individual_mu_C3_C4_", "plv", "Individual-Mu C3-C4 PLV", "PLV"),
        (
            "individual_mu_C3_C4_",
            "signed_lag",
            "Individual-Mu C3-C4 Signed Lag",
            "Mean sine of phase difference",
        ),
        ("beta_low_C3_C4_", "plv", "Low-Beta C3-C4 PLV", "PLV"),
        (
            "beta_low_C3_C4_",
            "signed_lag",
            "Low-Beta C3-C4 Signed Lag",
            "Mean sine of phase difference",
        ),
    )
    names = dataset.feature_names["phase"]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    for axis, (prefix, suffix, title, ylabel) in zip(axes.ravel(), panels):
        columns = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if name.startswith(prefix) and name.endswith(suffix)
            ],
            dtype=np.int64,
        )
        for label in sorted(LABEL_NAMES):
            selected = mask & (dataset.labels == label)
            mean, sem = _mean_and_sem(dataset.phase_features[selected][:, columns])
            axis.plot(
                times,
                mean,
                color=COLORS[label],
                linewidth=2.2,
                label=LABEL_NAMES[label],
            )
            axis.fill_between(
                times, mean - sem, mean + sem, color=COLORS[label], alpha=0.14
            )
        axis.set_xlabel("Seconds after cue")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.20)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Restricted Sensorimotor Phase Features",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "04_phase_coupling_audit.png", dpi=300)
    plt.close(fig)


def plot_patient_profiles(
    output_dir: Path,
    dataset: ExtendedFeatureDataset,
    profile_mask: np.ndarray,
) -> None:
    lookup = {name: index for index, name in enumerate(dataset.profile_names)}
    panels = (
        (
            "resting_mu_peak_hz",
            "Resting Mu Peak",
            "Frequency (Hz)",
            "#147D92",
        ),
        (
            "resting_mu_peak_snr_db",
            "Resting Mu Peak SNR",
            "Peak-to-background ratio (dB)",
            "#D66A3A",
        ),
        (
            "aperiodic_log_slope",
            "Aperiodic Spectral Slope",
            "Log-power / log-frequency slope",
            "#2D6A4F",
        ),
        (
            "run4_run8_rest_spectral_correlation",
            "Calibration Spectral Stability",
            "Run 4 versus Run 8 correlation",
            "#8E5572",
        ),
        (
            "run4_run8_rest_covariance_drift",
            "Calibration Spatial Drift",
            "Normalized covariance Frobenius distance",
            "#E9A23B",
        ),
        (
            "resting_mu_to_beta_ratio_db",
            "Resting Mu-to-Beta Balance",
            "Power ratio (dB)",
            "#577590",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.3), constrained_layout=True)
    for axis, (name, title, xlabel, color) in zip(axes.ravel(), panels):
        values = dataset.patient_profiles[profile_mask, lookup[name]]
        axis.hist(values, bins=16, color=color, alpha=0.88)
        median = float(np.median(values))
        axis.axvline(median, color="#222222", linestyle="--", linewidth=1.4)
        axis.text(
            0.97,
            0.92,
            f"Median {median:.2f}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Development participants")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle(
        "Participant Spectral and Calibration Profiles",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "05_patient_spectral_profiles.png", dpi=300)
    plt.close(fig)


def family_repeatability(
    dataset: ExtendedFeatureDataset,
    development_subjects: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    families = {
        "ssd": dataset.ssd_features,
        "asymmetry": dataset.asymmetry_features,
        "morphology": dataset.morphology_features,
        "phase": dataset.phase_features,
    }
    output: Dict[str, Dict[str, float]] = {}
    development = np.isin(dataset.subjects, development_subjects)
    calibration = development & np.isin(dataset.runs, CALIBRATION_RUNS)
    for family, raw_features in families.items():
        mean = np.mean(raw_features[calibration], axis=0)
        scale = np.std(raw_features[calibration], axis=0)
        features = (raw_features - mean) / np.clip(scale, 1e-6, None)
        run_effects: Dict[int, List[np.ndarray]] = {4: [], 8: []}
        magnitudes: Dict[int, List[float]] = {4: [], 8: []}
        used_subjects: List[int] = []
        cosines: List[float] = []
        for subject in development_subjects:
            effects = {}
            valid = True
            for run in CALIBRATION_RUNS:
                selected = (dataset.subjects == subject) & (dataset.runs == run)
                if family == "asymmetry":
                    first = features[selected & (dataset.labels == 1)]
                    second = features[selected & (dataset.labels == 2)]
                else:
                    first = features[selected & (dataset.labels != 0)]
                    second = features[selected & (dataset.labels == 0)]
                if len(first) == 0 or len(second) == 0:
                    valid = False
                    break
                effects[run] = np.mean(first, axis=0) - np.mean(second, axis=0)
            if not valid:
                continue
            first_effect = effects[4]
            second_effect = effects[8]
            denominator = np.linalg.norm(first_effect) * np.linalg.norm(second_effect)
            cosines.append(
                float(np.dot(first_effect, second_effect) / max(denominator, 1e-12))
            )
            for run in CALIBRATION_RUNS:
                run_effects[run].append(effects[run])
                magnitudes[run].append(
                    float(np.linalg.norm(effects[run]) / np.sqrt(features.shape[1]))
                )
            used_subjects.append(int(subject))
        correlation = float(np.corrcoef(magnitudes[4], magnitudes[8])[0, 1])
        output[family] = {
            "subject_count": len(used_subjects),
            "median_within_subject_effect_cosine": float(np.median(cosines)),
            "run4_run8_effect_magnitude_correlation": correlation,
            "median_run4_standardized_effect_magnitude": float(
                np.median(magnitudes[4])
            ),
            "median_run8_standardized_effect_magnitude": float(
                np.median(magnitudes[8])
            ),
        }
    return output


def plot_repeatability(
    output_dir: Path, repeatability: Mapping[str, Mapping[str, float]]
) -> None:
    families = list(repeatability)
    cosines = [
        repeatability[name]["median_within_subject_effect_cosine"]
        for name in families
    ]
    correlations = [
        repeatability[name]["run4_run8_effect_magnitude_correlation"]
        for name in families
    ]
    colors = [COLORS[name] for name in families]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), constrained_layout=True)
    axes[0].bar(families, cosines, color=colors)
    axes[0].axhline(0.0, color="#222222", linewidth=0.8)
    axes[0].set_ylim(min(-0.1, min(cosines) - 0.08), 1.0)
    axes[0].set_ylabel("Median cosine similarity")
    axes[0].set_title("Within-Subject Effect-Pattern Stability")
    axes[1].bar(families, correlations, color=colors)
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_ylim(min(-0.1, min(correlations) - 0.08), 1.0)
    axes[1].set_ylabel("Pearson correlation")
    axes[1].set_title("Across-Subject Effect-Magnitude Stability")
    for axis, values in zip(axes, (cosines, correlations)):
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.20)
        for index, value in enumerate(values):
            axis.text(index, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    fig.suptitle(
        "Run 4-to-Run 8 Development-Only Repeatability",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "06_extended_feature_repeatability.png", dpi=300)
    plt.close(fig)


def write_profile_csv(path: Path, dataset: ExtendedFeatureDataset) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        fieldnames = ["subject", *dataset.profile_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for subject, values in zip(
            dataset.profile_subjects, dataset.patient_profiles
        ):
            writer.writerow(
                {
                    "subject": int(subject),
                    **{
                        name: float(value)
                        for name, value in zip(dataset.profile_names, values)
                    },
                }
            )


def build_summary(
    dataset: ExtendedFeatureDataset,
    development_subjects: Sequence[int],
    repeatability: Mapping[str, Mapping[str, float]],
    runtime_minutes: float,
) -> Dict[str, object]:
    profile_mask = np.isin(dataset.profile_subjects, development_subjects)
    lookup = {name: index for index, name in enumerate(dataset.profile_names)}
    mu_snr = dataset.patient_profiles[
        profile_mask, lookup["resting_mu_peak_snr_db"]
    ]
    spectral_stability = dataset.patient_profiles[
        profile_mask, lookup["run4_run8_rest_spectral_correlation"]
    ]
    covariance_drift = dataset.patient_profiles[
        profile_mask, lookup["run4_run8_rest_covariance_drift"]
    ]
    return {
        "analysis_scope": (
            "Feature extraction and development-only physiological audit; "
            "no new classifier or final-cohort evaluation."
        ),
        "trial_count": len(dataset.labels),
        "subject_count": len(dataset.profile_subjects),
        "development_subject_count": int(np.sum(profile_mask)),
        "feature_counts": {
            name: len(dataset.feature_names[name])
            for name in ("ssd", "asymmetry", "morphology", "phase")
        },
        "total_new_trial_features": int(
            sum(len(values) for values in dataset.feature_names.values())
        ),
        "repeatability": repeatability,
        "development_patient_profiles": {
            "median_resting_mu_peak_snr_db": float(np.median(mu_snr)),
            "subjects_with_mu_peak_snr_at_least_3db_percent": float(
                100.0 * np.mean(mu_snr >= 3.0)
            ),
            "median_run4_run8_rest_spectral_correlation": float(
                np.median(spectral_stability)
            ),
            "subjects_with_rest_spectral_correlation_at_least_0_8_percent": float(
                100.0 * np.mean(spectral_stability >= 0.8)
            ),
            "median_run4_run8_rest_covariance_drift": float(
                np.median(covariance_drift)
            ),
        },
        "ssd_component_one_median_eigenvalue_ratio": {
            band: float(
                np.median(dataset.ssd_eigenvalues[profile_mask, index, 0])
            )
            for index, band in enumerate(dataset.ssd_band_names)
        },
        "runtime_minutes": runtime_minutes,
    }


def write_report(path: Path, summary: Mapping[str, object]) -> None:
    repeatability = summary["repeatability"]
    profiles = summary["development_patient_profiles"]
    feature_counts = summary["feature_counts"]
    ratios = summary["ssd_component_one_median_eigenvalue_ratio"]
    lines = [
        "# Extended Motor-Imagery Feature Audit",
        "",
        "## Scope",
        "",
        "This stage expands the pre-ML representation without reopening the locked "
        "final-subject benchmark. All label-based plots and repeatability checks use "
        "development subjects and calibration Runs 4/8 only. Run 12 is transformed "
        "with calibration-derived SSD filters and references but is not used to fit them.",
        "",
        "## Added Features",
        "",
        f"- SSD component ERD trajectories: **{feature_counts['ssd']}** features.",
        f"- Explicit sensor and hemispheric asymmetry: **{feature_counts['asymmetry']}** features.",
        f"- Compact ERD morphology: **{feature_counts['morphology']}** features.",
        f"- Restricted sensorimotor phase coupling: **{feature_counts['phase']}** features.",
        f"- Total new trial-level features: **{summary['total_new_trial_features']}**.",
        "- Eleven subject spectral/calibration profile values are retained as audit and "
        "calibration metadata, not repeated as classifier inputs.",
        "",
        "## Development Diagnostics",
        "",
    ]
    for family in ("ssd", "asymmetry", "morphology", "phase"):
        values = repeatability[family]
        display_name = "SSD" if family == "ssd" else family.title()
        lines.append(
            f"- {display_name}: median Run 4/8 effect-pattern cosine "
            f"**{values['median_within_subject_effect_cosine']:.2f}**; "
            "effect-magnitude correlation "
            f"**r={values['run4_run8_effect_magnitude_correlation']:.2f}**."
        )
    lines.extend(
        [
            "- Median resting mu-peak SNR: "
            f"**{profiles['median_resting_mu_peak_snr_db']:.2f} dB**; "
            f"**{profiles['subjects_with_mu_peak_snr_at_least_3db_percent']:.1f}%** "
            "of development subjects reached 3 dB.",
            f"- Median Run 4/8 resting spectral correlation: "
            f"**{profiles['median_run4_run8_rest_spectral_correlation']:.2f}**; "
            f"**{profiles['subjects_with_rest_spectral_correlation_at_least_0_8_percent']:.1f}%** "
            "reached 0.80.",
            f"- Median SSD component-1 target/flank ratios: individual mu "
            f"**{ratios['individual_mu']:.2f}**, low beta **{ratios['beta_low']:.2f}**, "
            f"high beta **{ratios['beta_high']:.2f}**.",
            "",
            "## Interpretation",
            "",
            "These diagnostics determine whether a feature family is sufficiently stable "
            "to justify a development-only ablation. High dimensionality alone is not "
            "evidence of usefulness. The next model experiment should add one family at a "
            "time to the fixed benchmark, tune only on development/controller-validation "
            "subjects, and evaluate the final cohort once after the design is frozen.",
            "",
            "No new classification accuracy is reported by this script by design.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="files/eegmmidb/1.0.0",
        help="EEGMMIDB root containing S001 ... S109.",
    )
    parser.add_argument(
        "--atlas-cache",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "feature_atlas"
        / "motor_imagery_feature_atlas.npz",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "dataset" / "splits.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "extended_features",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute raw-derived SSD, phase, and subject profile features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    configure_plots()
    data_dir = resolve_data_dir(args.data_dir)
    atlas = load_feature_atlas(args.atlas_cache)
    if atlas is None:
        raise FileNotFoundError(
            f"Base feature atlas not found at {args.atlas_cache}. "
            "Run scripts/18_build_motor_imagery_feature_atlas.py first."
        )
    partitions = load_partitions(args.split_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "extended_motor_imagery_features.npz"
    print("\n=== Extended Motor-Imagery Feature Audit ===")
    print(f"Data: {data_dir}")
    print(f"Base atlas: {args.atlas_cache}")
    dataset = build_extended_features(
        data_dir, atlas, cache_path, rebuild_cache=args.rebuild_cache
    )
    development_trials = (
        np.isin(dataset.subjects, partitions.development)
        & np.isin(dataset.runs, CALIBRATION_RUNS)
    )
    development_profiles = np.isin(
        dataset.profile_subjects, partitions.development
    )
    if not np.any(development_trials) or not np.any(development_profiles):
        raise ValueError("The extended cache does not contain fixed development subjects.")
    repeatability = family_repeatability(dataset, partitions.development)
    plot_ssd_audit(
        args.output_dir, dataset, development_trials, development_profiles
    )
    plot_asymmetry_trajectories(
        args.output_dir, dataset, development_trials
    )
    plot_morphology_audit(args.output_dir, dataset, development_trials)
    plot_phase_audit(args.output_dir, dataset, development_trials)
    plot_patient_profiles(args.output_dir, dataset, development_profiles)
    plot_repeatability(args.output_dir, repeatability)
    write_profile_csv(args.output_dir / "patient_spectral_profiles.csv", dataset)
    runtime_minutes = (time.time() - started) / 60.0
    summary = build_summary(
        dataset, partitions.development, repeatability, runtime_minutes
    )
    (args.output_dir / "extended_feature_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    write_report(args.output_dir / "EXTENDED_FEATURE_RESULTS.md", summary)
    print(
        f"[+] Added {summary['total_new_trial_features']} trial features across "
        f"{summary['subject_count']} subjects."
    )
    profile_summary = summary["development_patient_profiles"]
    print(
        "[+] Development resting mu SNR >= 3 dB: "
        f"{profile_summary['subjects_with_mu_peak_snr_at_least_3db_percent']:.1f}%"
    )
    print(f"[+] Saved report: {args.output_dir / 'EXTENDED_FEATURE_RESULTS.md'}")
    print(f"[+] Runtime: {runtime_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
