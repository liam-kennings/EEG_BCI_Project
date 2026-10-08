#!/usr/bin/env python3
"""Audit the visual-cue to motor-imagery sequence without conflating the signals.

Early posterior visual responses are measured 80-300 ms after the cue. Later
sensorimotor mu/beta ERD is measured 0.5-3.5 s after the cue. A left/right
decoder is fitted separately to each feature family to quantify cue confounding.
"""

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
from scipy.stats import pearsonr, ttest_1samp  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.dataset import available_subjects, normalize_channel_name, resolve_data_dir  # noqa: E402
from bci_pipeline.epochs import EpochConfig, build_raw_epoch_dataset  # noqa: E402
from bci_pipeline.preprocessing import PreprocessingProfile  # noqa: E402
from bci_pipeline.splits import load_partitions  # noqa: E402


VISUAL_CHANNELS = ("O1", "Oz", "O2", "PO3", "POz", "PO4", "P3", "Pz", "P4")
MOTOR_CHANNELS = (
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
)
CHANNELS = tuple(dict.fromkeys(VISUAL_CHANNELS + MOTOR_CHANNELS))


def channel_index(names: Sequence[str], requested: str) -> int:
    requested = normalize_channel_name(requested)
    for index, name in enumerate(names):
        if normalize_channel_name(name) == requested:
            return index
    raise ValueError(f"Required channel {requested} is absent from the cache.")


def band_power(signals: np.ndarray, low: float, high: float, sfreq: float) -> np.ndarray:
    sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, signals, axis=-1)
    return np.abs(hilbert(filtered, axis=-1)) ** 2


def residualize_by_subject_and_side(values: np.ndarray, subjects: np.ndarray, sides: np.ndarray) -> np.ndarray:
    residual = np.empty_like(values, dtype=np.float64)
    for subject in np.unique(subjects):
        subject_mask = subjects == subject
        for side in (0, 1):
            mask = subject_mask & (sides == side)
            if np.any(mask):
                residual[mask] = values[mask] - np.mean(values[mask])
    return residual


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_decoder(
    values: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    partitions: object,
) -> Dict[str, float]:
    movement = labels != 0
    development = movement & np.isin(subjects, partitions.development)
    controller = movement & np.isin(subjects, partitions.controller_validation) & (runs == 12)
    final = movement & np.isin(subjects, partitions.final_target) & (runs == 12)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=3000, random_state=42))
    model.fit(values[development], labels[development] - 1)
    return {
        "controller_accuracy": float(accuracy_score(labels[controller] - 1, model.predict(values[controller]))),
        "controller_balanced_accuracy": float(
            balanced_accuracy_score(labels[controller] - 1, model.predict(values[controller]))
        ),
        "final_accuracy": float(accuracy_score(labels[final] - 1, model.predict(values[final]))),
        "final_balanced_accuracy": float(
            balanced_accuracy_score(labels[final] - 1, model.predict(values[final]))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--cache-path", default="artifacts/cue_intent_audit/cue_epoch_cache.npz")
    parser.add_argument("--output-dir", default="artifacts/cue_intent_audit")
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir)
    profile = PreprocessingProfile(l_freq=1.0, h_freq=35.0, reference="average")
    dataset = build_raw_epoch_dataset(
        data_dir,
        subjects,
        Path(args.cache_path),
        EpochConfig(
            epoch_start_seconds=0.0,
            epoch_duration_seconds=4.0,
            channels=CHANNELS,
            preprocessing=profile,
        ),
        args.rebuild_cache,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.arange(dataset.epochs.shape[-1]) / dataset.sfreq
    visual_window = (times >= 0.08) & (times <= 0.30)
    motor_window = (times >= 0.5) & (times <= 3.5)
    visual_indices = [channel_index(dataset.channel_names, name) for name in VISUAL_CHANNELS]
    motor_indices = [channel_index(dataset.channel_names, name) for name in MOTOR_CHANNELS]
    c3_index, c4_index = channel_index(dataset.channel_names, "C3"), channel_index(dataset.channel_names, "C4")
    mu_task = band_power(dataset.epochs, 8.0, 13.0, dataset.sfreq)
    beta_task = band_power(dataset.epochs, 13.0, 30.0, dataset.sfreq)
    mu_baseline = np.mean(band_power(dataset.baselines, 8.0, 13.0, dataset.sfreq), axis=-1)
    beta_baseline = np.mean(band_power(dataset.baselines, 13.0, 30.0, dataset.sfreq), axis=-1)
    mu_erd = 10.0 * np.log10(
        (np.mean(mu_task[:, :, motor_window], axis=-1) + 1e-20)
        / (mu_baseline + 1e-20)
    )
    beta_erd = 10.0 * np.log10(
        (np.mean(beta_task[:, :, motor_window], axis=-1) + 1e-20)
        / (beta_baseline + 1e-20)
    )
    visual_evoked = dataset.epochs[:, visual_indices][:, :, visual_window]
    visual_rms = np.sqrt(np.mean(visual_evoked**2, axis=-1)) * 1e6
    visual_signed = np.mean(visual_evoked, axis=-1) * 1e6
    rows: List[Dict[str, object]] = []
    for index in range(len(dataset.labels)):
        row: Dict[str, object] = {
            "subject": int(dataset.subjects[index]),
            "run": int(dataset.runs[index]),
            "label": int(dataset.labels[index]),
            "visual_response_rms_uv": float(np.mean(visual_rms[index])),
            "visual_left_minus_right_uv": float(
                np.mean(visual_signed[index, (0, 3, 6)])
                - np.mean(visual_signed[index, (2, 5, 8)])
            ),
            "motor_mu_erd_db": float(np.mean(mu_erd[index, motor_indices])),
            "motor_beta_erd_db": float(np.mean(beta_erd[index, motor_indices])),
            "motor_c3_minus_c4_mu_db": float(mu_erd[index, c3_index] - mu_erd[index, c4_index]),
            "motor_c3_minus_c4_beta_db": float(beta_erd[index, c3_index] - beta_erd[index, c4_index]),
        }
        for local, channel in enumerate(VISUAL_CHANNELS):
            row[f"visual_{normalize_channel_name(channel).lower()}_rms_uv"] = float(visual_rms[index, local])
            row[f"visual_{normalize_channel_name(channel).lower()}_mean_uv"] = float(visual_signed[index, local])
        for channel in MOTOR_CHANNELS:
            channel_position = channel_index(dataset.channel_names, channel)
            safe_name = normalize_channel_name(channel).lower()
            row[f"motor_{safe_name}_mu_erd_db"] = float(mu_erd[index, channel_position])
            row[f"motor_{safe_name}_beta_erd_db"] = float(beta_erd[index, channel_position])
        rows.append(row)
    write_csv(output_dir / "cue_intent_trial_features.csv", rows)

    movement = dataset.labels != 0
    movement_subjects = dataset.subjects[movement]
    movement_sides = dataset.labels[movement] - 1
    visual_value = np.asarray([row["visual_response_rms_uv"] for row in rows])[movement]
    motor_value = -np.asarray([row["motor_mu_erd_db"] for row in rows])[movement]
    visual_residual = residualize_by_subject_and_side(visual_value, movement_subjects, movement_sides)
    motor_residual = residualize_by_subject_and_side(motor_value, movement_subjects, movement_sides)
    correlations: List[Dict[str, float]] = []
    for subject in np.unique(movement_subjects):
        mask = movement_subjects == subject
        if np.std(visual_residual[mask]) < 1e-9 or np.std(motor_residual[mask]) < 1e-9:
            continue
        correlation, p_value = pearsonr(visual_residual[mask], motor_residual[mask])
        correlations.append({"subject": int(subject), "r": float(correlation), "p_value": float(p_value)})
    correlation_values = np.asarray([item["r"] for item in correlations])
    t_statistic, group_p_value = ttest_1samp(correlation_values, 0.0)
    partitions = load_partitions(Path(args.splits))
    visual_columns = [name for name in rows[0] if name.startswith("visual_")]
    motor_columns = [name for name in rows[0] if name.startswith("motor_")]
    visual_features = np.asarray([[float(row[name]) for name in visual_columns] for row in rows])
    motor_features = np.asarray([[float(row[name]) for name in motor_columns] for row in rows])
    visual_decoder = fit_decoder(visual_features, dataset.labels, dataset.subjects, dataset.runs, partitions)
    motor_decoder = fit_decoder(motor_features, dataset.labels, dataset.subjects, dataset.runs, partitions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(visual_residual, motor_residual, s=7, alpha=0.16, color="#39739d")
    slope, intercept = np.polyfit(visual_residual, motor_residual, 1)
    x_values = np.linspace(np.min(visual_residual), np.max(visual_residual), 100)
    axes[0].plot(x_values, slope * x_values + intercept, color="#b65c4a", linewidth=2)
    axes[0].set_xlabel("Within-subject, within-side visual-response residual (uV RMS)")
    axes[0].set_ylabel("Within-subject, within-side motor mu ERD magnitude (dB)")
    axes[0].set_title("Cue response versus later motor ERD")
    axes[1].hist(correlation_values, bins=20, color="#5aa469", edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Per-subject residual correlation (r)")
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Patient-level cue-to-ERD correlations")
    fig.tight_layout()
    fig.savefig(output_dir / "cue_to_motor_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    names = ["Early visual\nfeatures", "Late motor\nERD features"]
    values = [100 * visual_decoder["final_balanced_accuracy"], 100 * motor_decoder["final_balanced_accuracy"]]
    bars = ax.bar(names, values, color=["#b97831", "#39739d"])
    ax.axhline(50, color="black", linestyle="--", linewidth=0.9, label="Chance")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Final unseen-subject left/right balanced accuracy (%)")
    ax.set_title("Cue-confound audit")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "visual_vs_motor_direction_decoding.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "protocol": {
            "visual_feature_window_seconds": [0.08, 0.30],
            "motor_feature_window_seconds": [0.5, 3.5],
            "correlation": "Pearson correlation after removing each subject and cue-side mean",
            "purpose": "cue-to-motor association and visual-cue confound audit, not a motor-intent classifier",
        },
        "dataset": {"trials": int(len(dataset.labels)), "subjects": int(len(np.unique(dataset.subjects)))},
        "correlation": {
            "evaluated_subjects": int(len(correlations)),
            "mean_r": float(np.mean(correlation_values)),
            "median_r": float(np.median(correlation_values)),
            "one_sample_t": float(t_statistic),
            "two_sided_p": float(group_p_value),
            "per_subject": correlations,
        },
        "left_right_confound_audit": {"visual_early": visual_decoder, "motor_late": motor_decoder},
    }
    with (output_dir / "cue_intent_correlation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("=== Cue-to-Motor Intent Correlation Audit ===")
    print(
        f"[+] Residual cue-to-motor association: mean r={report['correlation']['mean_r']:.3f}, "
        f"p={report['correlation']['two_sided_p']:.4f}"
    )
    print(
        f"[+] Final left/right BA: visual early={visual_decoder['final_balanced_accuracy'] * 100:.1f}% | "
        f"motor late={motor_decoder['final_balanced_accuracy'] * 100:.1f}%"
    )
    print(f"[+] Saved audit to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
