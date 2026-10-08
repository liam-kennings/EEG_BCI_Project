#!/usr/bin/env python3
"""Prepare synchronized MRCP EEG-EMG source and pure-EEG target caches."""

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

from bci_pipeline.mrcp_eeg_emg import (  # noqa: E402
    DATASET_FOLDER,
    build_mrcp_privileged_source,
    resolve_mrcp_eeg_emg_root,
)
from bci_pipeline.privileged_transfer import shared_eeg_intent_features  # noqa: E402
from bci_pipeline.translational_dataset import load_translational_dataset  # noqa: E402


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
        sessions=result.sessions,
        trial_indices=result.trial_indices,
        row_kinds=result.row_kinds,
        onset_latencies_seconds=result.onset_latencies_seconds,
    )
    path.with_name(f"{path.stem}.qc.json").write_text(
        json.dumps(result.qc, indent=2) + "\n",
        encoding="utf-8",
    )


def _role_vector(
    dataset_ids: np.ndarray,
    subject_keys: np.ndarray,
    role_path: Path,
) -> np.ndarray:
    role_contract = json.loads(role_path.read_text(encoding="utf-8"))
    roles = np.full(len(subject_keys), "unassigned", dtype="<U24")
    for dataset_id, assignments in role_contract.items():
        dataset_mask = np.asarray(dataset_ids).astype(str) == dataset_id
        for role, subjects in assignments.items():
            if role == "locked_external" and not subjects:
                continue
            selected = dataset_mask & np.isin(
                np.asarray(subject_keys).astype(str),
                np.asarray(subjects).astype(str),
            )
            if np.any(roles[selected] != "unassigned"):
                raise ValueError(f"Duplicate target role assignment in {dataset_id}.")
            roles[selected] = role
    if np.any(roles == "unassigned"):
        missing = np.unique(np.asarray(subject_keys)[roles == "unassigned"])
        raise ValueError(f"Target subjects are missing roles: {missing[:10].tolist()}")
    return roles


def _save_target(
    path: Path,
    translational_cache: Path,
    role_path: Path,
) -> dict[str, object]:
    dataset = load_translational_dataset(translational_cache)
    if dataset is None:
        raise FileNotFoundError(f"Target translational cache not found: {translational_cache}")
    features, feature_names = shared_eeg_intent_features(
        dataset.mrcp_epochs,
        dataset.smr_epochs,
        dataset.sfreq,
        dataset.channel_names,
    )
    roles = _role_vector(dataset.dataset_ids, dataset.subject_keys, role_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eeg_features=features,
        labels=(dataset.labels != 0).astype(np.int64),
        original_labels=dataset.labels,
        subject_keys=dataset.subject_keys,
        dataset_ids=dataset.dataset_ids,
        roles=roles,
        eeg_feature_names=np.asarray(feature_names),
    )
    return {
        "rows": len(dataset.labels),
        "subjects": len(np.unique(dataset.subject_keys)),
        "features": features.shape[1],
        "datasets": {
            str(dataset_id): {
                "rows": int(np.sum(dataset.dataset_ids == dataset_id)),
                "subjects": int(
                    len(np.unique(dataset.subject_keys[dataset.dataset_ids == dataset_id]))
                ),
            }
            for dataset_id in np.unique(dataset.dataset_ids)
        },
        "roles": {
            str(role): int(np.sum(roles == role))
            for role in np.unique(roles)
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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    axes[0].hist(latencies, bins=np.linspace(0.1, 2.0, 20), color=TEAL, edgecolor="white")
    axes[0].axvline(np.median(latencies), color=GOLD, linewidth=2.5, label=f"Median {np.median(latencies):.2f} s")
    axes[0].set_title("EMG-confirmed onset after cue", loc="left", weight="bold", color=INK)
    axes[0].set_xlabel("EMG onset latency (seconds)")
    axes[0].set_ylabel("Accepted movement events")
    axes[0].legend(frameon=False)

    colors = [TEAL if effects[index] >= 0 else CORAL for index in strongest]
    axes[1].barh(np.arange(len(strongest)), effects[strongest], color=colors)
    axes[1].set_yticks(np.arange(len(strongest)), names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color=INK, linewidth=0.9)
    axes[1].set_title("Strongest source EEG class effects", loc="left", weight="bold", color=INK)
    axes[1].set_xlabel("Standardized intent-minus-rest effect")
    for ax in axes:
        ax.grid(axis="x" if ax is axes[1] else "y", color="#D5DEE2", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("MRCP EEG-EMG Source Adapter Quality Audit", fontsize=18, weight="bold", color=INK)
    fig.savefig(output / "00_mrcp_eeg_emg_source_qc.png", dpi=220)
    plt.close(fig)


def _write_report(
    output: Path,
    root: Path,
    result: object,
    target_summary: dict[str, object],
    source_path: Path,
    target_path: Path,
) -> None:
    qc = result.qc
    discovery = qc["discovery"]
    latency = qc["onset_latency_seconds"]
    rejected = qc["rejected_events"]
    lines = [
        "# MRCP EEG-EMG Adapter Report",
        "",
        "## Source Data",
        "",
        f"- Root: `{root}`.",
        f"- EEG files: **{discovery['eeg_files']}**; EMG files: **{discovery['emg_files']}**.",
        f"- Paired recordings: **{discovery['paired_recordings']}** across **{discovery['paired_subjects']}** participants.",
        f"- Accepted movement events: **{qc['accepted_movement_events']}**.",
        f"- Balanced source rows after matched rest extraction: **{qc['rows']}**.",
        f"- EMG onset latency: median **{latency['median']:.2f} s**, 5th-95th percentile **{latency['p05']:.2f}-{latency['p95']:.2f} s** after cue.",
        f"- Rejected because EMG onset was not confirmed: **{rejected['emg_onset_not_confirmed']}** events.",
        f"- Rejected because synchronized windows were out of bounds: **{rejected['window_out_of_bounds']}** events.",
        "",
        "Subject 11 contains five EMG files but no EEG files and is excluded. Files with incomplete trigger sequences are retained only for individually paired cue events; missing cues are never imputed.",
        "",
        "## Leakage Controls",
        "",
        "- EEG positive windows are two seconds long and end at estimated EMG onset.",
        "- EMG confirmation features are taken after onset and are used only in source training.",
        "- Rest windows end before the movement cue.",
        "- Source and target use the same seven channels, MRCP features, and fixed mu/beta features.",
        "- The pure-EEG target cache contains no EMG features.",
        "",
        "## Target Cache",
        "",
        f"- Rows: **{target_summary['rows']}** from **{target_summary['subjects']}** subjects.",
        f"- Shared EEG features: **{target_summary['features']}**.",
        f"- Datasets: `{target_summary['datasets']}`.",
        f"- Roles: `{target_summary['roles']}`.",
        "",
        "## Outputs",
        "",
        f"- Source: `{source_path}`.",
        f"- Target: `{target_path}`.",
        "- Figure: `00_mrcp_eeg_emg_source_qc.png`.",
    ]
    (output / "MRCP_EEG_EMG_ADAPTER_REPORT.md").write_text(
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
        "--translational-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "gui_experiments" / "translational_v3" / "translational_windows_v3.npz",
    )
    parser.add_argument(
        "--subject-roles",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "gui_experiments" / "translational_v3_calibrated" / "subject_roles.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer",
    )
    args = parser.parse_args()
    root = resolve_mrcp_eeg_emg_root(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Building EMG-confirmed source cache from {root}...")
    result = build_mrcp_privileged_source(root)
    source_path = args.output_dir / "mrcp_eeg_emg_source_features.npz"
    target_path = args.output_dir / "pure_eeg_target_features.npz"
    _save_source(source_path, result)
    print("[+] Exporting identically defined pure-EEG target features...")
    target_summary = _save_target(
        target_path,
        args.translational_cache,
        args.subject_roles,
    )
    _plot_source_qc(result, args.output_dir)
    _write_report(
        args.output_dir,
        root,
        result,
        target_summary,
        source_path,
        target_path,
    )
    print(
        f"[+] Prepared {result.qc['accepted_movement_events']} EMG-confirmed movements "
        f"and {target_summary['rows']} pure-EEG target rows."
    )
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
