#!/usr/bin/env python3
"""Prepare WAY-EEG-GAL privileged source features for the EEG-only decoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bci_pipeline.way_eeg_gal import (  # noqa: E402
    DATASET_FOLDER,
    TRANSFER_CHANNELS,
    WAY_CHANNEL_MAP,
    build_way_privileged_source,
    resolve_way_eeg_gal_root,
)


INK = "#17324D"
TEAL = "#087E8B"
GOLD = "#D99B24"
CORAL = "#C64B42"


def _save_source(path: Path, result: object) -> None:
    dataset = result.dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eeg_features=dataset.eeg_features,
        emg_features=dataset.emg_features,
        labels=dataset.labels,
        subject_keys=dataset.subject_keys,
        premovement_end_seconds=dataset.premovement_end_seconds,
        eeg_feature_names=np.asarray(dataset.eeg_feature_names),
        emg_feature_names=np.asarray(dataset.emg_feature_names),
        dataset_id=np.asarray(dataset.dataset_id),
        series=result.series,
        trial_indices=result.trial_indices,
        row_kinds=result.row_kinds,
        onset_latencies_seconds=result.onset_latencies_seconds,
        weights=result.weights,
        surfaces=result.surfaces,
    )
    path.with_name(f"{path.stem}.qc.json").write_text(
        json.dumps(result.qc, indent=2) + "\n",
        encoding="utf-8",
    )


def _target_contract(path: Path, source: object) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Pure-EEG target cache not found: {path}. Run the existing EMG-supervised "
            "preparation workflow first."
        )
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "eeg_features",
            "labels",
            "subject_keys",
            "dataset_ids",
            "roles",
            "eeg_feature_names",
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"Target cache is missing keys: {sorted(missing)}")
        target_names = tuple(str(value) for value in cached["eeg_feature_names"].tolist())
        if target_names != source.dataset.eeg_feature_names:
            raise ValueError("WAY and target EEG feature definitions do not match exactly.")
        return {
            "path": str(path.resolve()),
            "rows": int(len(cached["labels"])),
            "subjects": int(len(np.unique(cached["subject_keys"]))),
            "features": int(cached["eeg_features"].shape[1]),
            "datasets": {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(cached["dataset_ids"].astype(str), return_counts=True),
                    strict=True,
                )
            },
            "roles": {
                str(name): int(count)
                for name, count in zip(
                    *np.unique(cached["roles"].astype(str), return_counts=True),
                    strict=True,
                )
            },
        }


def _effect_sizes(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rest = np.asarray(features)[np.asarray(labels) == 0]
    intent = np.asarray(features)[np.asarray(labels) == 1]
    difference = np.mean(intent, axis=0) - np.mean(rest, axis=0)
    pooled = np.sqrt(0.5 * (np.var(intent, axis=0) + np.var(rest, axis=0)))
    return difference / np.where(pooled > 1e-12, pooled, 1.0)


def _plot_source_qc(result: object, output: Path) -> None:
    dataset = result.dataset
    positive = dataset.labels == 1
    latencies = result.onset_latencies_seconds[positive]
    effects = _effect_sizes(dataset.eeg_features, dataset.labels)
    strongest = np.argsort(np.abs(effects))[-12:][::-1]
    names = [dataset.eeg_feature_names[index] for index in strongest]
    names = [name.replace("_log_power", "").replace("_mrcp_", "\n") for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    axes[0].axvspan(-2.1, -0.1, color=TEAL, alpha=0.24, label="Positive EEG")
    axes[0].axvline(0.0, color=CORAL, linewidth=2.5, label="Kinematic hand start")
    axes[0].axvspan(0.05, 0.80, color=GOLD, alpha=0.28, label="Training-only EMG")
    axes[0].set_xlim(-2.3, 1.0)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Time relative to hand start (seconds)")
    axes[0].set_title("Leakage-safe source timing", loc="left", weight="bold", color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    axes[1].hist(latencies, bins=24, color=TEAL, edgecolor="white")
    axes[1].axvline(
        np.median(latencies),
        color=GOLD,
        linewidth=2.5,
        label=f"Median {np.median(latencies):.2f} s",
    )
    axes[1].set_title("Cue-to-hand-start latency", loc="left", weight="bold", color=INK)
    axes[1].set_xlabel("Seconds after visual cue")
    axes[1].set_ylabel("Accepted grasp-and-lift trials")
    axes[1].legend(frameon=False)

    colors = [TEAL if effects[index] >= 0 else CORAL for index in strongest]
    axes[2].barh(np.arange(len(strongest)), effects[strongest], color=colors)
    axes[2].set_yticks(np.arange(len(strongest)), names, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].axvline(0.0, color=INK, linewidth=0.9)
    axes[2].set_title("Strongest source EEG effects", loc="left", weight="bold", color=INK)
    axes[2].set_xlabel("Standardized intent-minus-rest effect")
    for axis in axes:
        axis.grid(axis="x" if axis is axes[2] else "y", color="#D5DEE2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "WAY-EEG-GAL Privileged Source Quality Audit",
        fontsize=18,
        weight="bold",
        color=INK,
    )
    fig.savefig(output / "00_way_eeg_gal_source_qc.png", dpi=220)
    plt.close(fig)


def _write_report(
    output: Path,
    root: Path,
    result: object,
    target: dict[str, object],
    source_path: Path,
) -> None:
    qc = result.qc
    latency = qc["cue_to_hand_start_seconds"]
    rejected = qc["rejected_events"]
    interpolated = [
        f"{target_name}=mean({', '.join(WAY_CHANNEL_MAP[target_name])})"
        for target_name in ("FC3", "FC4", "CP3", "CP4")
    ]
    lines = [
        "# WAY-EEG-GAL Privileged-Transfer Adapter",
        "",
        "## Decoder Role",
        "",
        "WAY-EEG-GAL supplies additional **intent-versus-rest** source training. It cannot train the left/right direction head because every recorded action uses the right hand.",
        "",
        "## Source Cohort",
        "",
        f"- Root: `{root}`.",
        f"- Accepted series: **{qc['accepted_series']}/{qc['discovered_series']}** across **{qc['accepted_subjects']}** participants.",
        f"- Accepted grasp-and-lift events: **{qc['accepted_movement_events']}**.",
        f"- Balanced source rows: **{qc['rows']}** intent/rest examples.",
        f"- Cue-to-hand-start latency: median **{latency['median']:.2f} s**, 5th-95th percentile **{latency['p05']:.2f}-{latency['p95']:.2f} s**.",
        f"- Rejected event counts: `{rejected}`.",
        "- The 12 EEG-only challenge series are excluded because they do not contain the peripheral evidence required for privileged supervision.",
        "",
        "## Leakage Boundary",
        "",
        "- Positive EEG windows are two seconds long and end 100 ms before kinematic hand start.",
        "- Rest EEG windows end 100 ms before the visual cue.",
        "- EMG verification windows begin 50 ms after hand start and are available only to the source-label verifier.",
        "- The trained student and every target evaluation receive EEG features only.",
        "- Some positive windows contain the visual cue. The result therefore measures cue-conditioned motor preparation and must not be presented as asynchronous intention decoding.",
        "",
        "## Montage Harmonization",
        "",
        f"The target contract uses `{list(TRANSFER_CHANNELS)}`. WAY directly supplies C3, Cz, and C4; missing mid-lateral sites are reconstructed as `{interpolated}` after full-montage common-average reference.",
        "",
        "## Shared Target Contract",
        "",
        f"- Target rows: **{target['rows']}** from **{target['subjects']}** subjects.",
        f"- Shared EEG features: **{target['features']}**.",
        f"- Target datasets: `{target['datasets']}`.",
        f"- Locked roles: `{target['roles']}`.",
        "",
        "## Outputs",
        "",
        f"- Source cache: `{source_path}`.",
        f"- Existing target cache: `{target['path']}`.",
        "- Source audit figure: `00_way_eeg_gal_source_qc.png`.",
    ]
    (output / "WAY_EEG_GAL_ADAPTER_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / DATASET_FOLDER,
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer" / "pure_eeg_target_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "way_eeg_gal_transfer",
    )
    parser.add_argument(
        "--subjects",
        help="Optional comma-separated participant numbers for an adapter smoke test.",
    )
    args = parser.parse_args()
    subjects = None
    if args.subjects:
        try:
            subjects = [int(value.strip()) for value in args.subjects.split(",")]
        except ValueError as error:
            parser.error(f"Invalid --subjects list: {error}")

    root = resolve_way_eeg_gal_root(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Building WAY-EEG-GAL privileged source cache from {root}...")
    result = build_way_privileged_source(root, subjects=subjects)
    source_path = args.output_dir / "way_eeg_gal_source_features.npz"
    _save_source(source_path, result)
    target = _target_contract(args.target_cache, result)
    _plot_source_qc(result, args.output_dir)
    _write_report(args.output_dir, root, result, target, source_path)
    print(
        f"[+] Prepared {result.qc['accepted_movement_events']} pre-movement events "
        f"({result.qc['rows']} balanced rows) with {result.dataset.eeg_features.shape[1]} shared EEG features."
    )
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
