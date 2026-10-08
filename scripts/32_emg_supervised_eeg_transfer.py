#!/usr/bin/env python3
"""Compare target-only EEG with an EMG-confirmed EEG transfer student."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from bci_pipeline.clinical_evaluation import select_intent_threshold  # noqa: E402
from bci_pipeline.privileged_transfer import (  # noqa: E402
    EMGConfirmedEEGTransfer,
    PrivilegedSourceDataset,
)
from bci_pipeline.translational_decoder import dataset_balanced_weights  # noqa: E402


INK = "#17324D"
TEAL = "#087E8B"
GOLD = "#D99B24"
CORAL = "#C64B42"


def _strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).tolist())


def load_source_cache(path: Path) -> PrivilegedSourceDataset:
    with np.load(path, allow_pickle=False) as cached:
        required = {
            "eeg_features",
            "emg_features",
            "labels",
            "subject_keys",
            "premovement_end_seconds",
            "eeg_feature_names",
            "emg_feature_names",
            "dataset_id",
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"Source cache is missing keys: {sorted(missing)}")
        dataset = PrivilegedSourceDataset(
            eeg_features=cached["eeg_features"],
            emg_features=cached["emg_features"],
            labels=cached["labels"],
            subject_keys=cached["subject_keys"],
            premovement_end_seconds=cached["premovement_end_seconds"],
            eeg_feature_names=_strings(cached["eeg_feature_names"]),
            emg_feature_names=_strings(cached["emg_feature_names"]),
            dataset_id=str(np.asarray(cached["dataset_id"]).item()),
        )
    dataset.validate()
    return dataset


def load_target_cache(path: Path) -> dict[str, np.ndarray | tuple[str, ...]]:
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
        target: dict[str, np.ndarray | tuple[str, ...]] = {
            "eeg_features": cached["eeg_features"],
            "labels": cached["labels"],
            "subject_keys": cached["subject_keys"],
            "dataset_ids": cached["dataset_ids"].astype(str),
            "roles": cached["roles"].astype(str),
            "eeg_feature_names": _strings(cached["eeg_feature_names"]),
        }
    features = np.asarray(target["eeg_features"], dtype=np.float64)
    labels = np.asarray(target["labels"], dtype=np.int64)
    roles = np.asarray(target["roles"]).astype(str)
    subjects = np.asarray(target["subject_keys"]).astype(str)
    datasets = np.asarray(target["dataset_ids"]).astype(str)
    count = len(labels)
    if features.ndim != 2 or len(features) != count:
        raise ValueError("Target EEG features and labels are not row-aligned.")
    if len(roles) != count or len(subjects) != count or len(datasets) != count:
        raise ValueError("Target metadata are not row-aligned.")
    if features.shape[1] != len(target["eeg_feature_names"]):
        raise ValueError("Target EEG feature names do not align.")
    if not np.all(np.isfinite(features)):
        raise ValueError("Target EEG features contain non-finite values.")
    required_roles = {"development", "controller_validation", "development_observed"}
    if not required_roles.issubset(set(roles)):
        raise ValueError(f"Target cache requires roles: {sorted(required_roles)}")
    return target


def score(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    true = np.asarray(labels) != 0
    predicted = np.asarray(probability) >= threshold
    return {
        "rows": int(len(true)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "sensitivity": float(np.mean(predicted[true])),
        "specificity": float(np.mean(~predicted[~true])),
        "missed_intent_rate": float(np.mean(~predicted[true])),
        "false_activation_rate_per_window": float(np.mean(predicted[~true])),
        "auroc": float(roc_auc_score(true, probability)),
        "auprc": float(average_precision_score(true, probability)),
        "threshold": float(threshold),
    }


def participant_analysis(
    labels: np.ndarray,
    baseline_probability: np.ndarray,
    transfer_probability: np.ndarray,
    baseline_threshold: float,
    transfer_threshold: float,
    subject_keys: np.ndarray,
    dataset_ids: np.ndarray,
    seed: int,
    iterations: int = 2000,
) -> dict[str, Any]:
    records = []
    paired_keys = np.asarray(
        [f"{dataset}:{subject}" for dataset, subject in zip(dataset_ids, subject_keys, strict=True)]
    )
    for key in np.unique(paired_keys):
        selected = paired_keys == key
        subject_labels = labels[selected]
        if len(np.unique(subject_labels != 0)) != 2:
            continue
        baseline = score(
            subject_labels,
            baseline_probability[selected],
            baseline_threshold,
        )
        transfer = score(
            subject_labels,
            transfer_probability[selected],
            transfer_threshold,
        )
        dataset, subject = key.split(":", maxsplit=1)
        records.append(
            {
                "dataset_id": dataset,
                "subject_key": subject,
                "rows": int(np.sum(selected)),
                "target_only_balanced_accuracy": baseline["balanced_accuracy"],
                "emg_supervised_balanced_accuracy": transfer["balanced_accuracy"],
                "balanced_accuracy_delta": transfer["balanced_accuracy"]
                - baseline["balanced_accuracy"],
                "target_only_sensitivity": baseline["sensitivity"],
                "emg_supervised_sensitivity": transfer["sensitivity"],
            }
        )
    if not records:
        raise ValueError("No observed participant contains both intent classes.")
    deltas = np.asarray(
        [record["balanced_accuracy_delta"] for record in records],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True)
    bootstrap_mean = np.mean(samples, axis=1)
    return {
        "participants": len(records),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "median_balanced_accuracy_delta": float(np.median(deltas)),
        "bootstrap_95_ci": [
            float(np.percentile(bootstrap_mean, 2.5)),
            float(np.percentile(bootstrap_mean, 97.5)),
        ],
        "participants_improved_fraction": float(np.mean(deltas > 0.0)),
        "bootstrap_iterations": iterations,
        "records": records,
    }


def _plot(results: dict[str, Any], output: Path) -> None:
    metrics = ("balanced_accuracy", "sensitivity", "specificity")
    labels = ("Balanced accuracy", "Sensitivity", "Specificity")
    baseline = [100.0 * results["target_only"][metric] for metric in metrics]
    transfer = [100.0 * results["emg_supervised_eeg"][metric] for metric in metrics]
    positions = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    ax.bar(positions - width / 2, baseline, width, label="Target-only EEG", color=GOLD)
    ax.bar(positions + width / 2, transfer, width, label="Privileged-source EEG student", color=TEAL)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Development-observed pure-EEG result (%)", color=INK)
    ax.set_title("Does EMG-Confirmed Source Training Improve EEG-Only Intent Detection?", loc="left", weight="bold", color=INK)
    ax.grid(axis="y", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for offset, values in ((-width / 2, baseline), (width / 2, transfer)):
        for index, value in enumerate(values):
            ax.text(index + offset, value + 1.5, f"{value:.1f}%", ha="center", color=INK, weight="bold")
    delta = transfer[0] - baseline[0]
    ax.text(
        0.50,
        0.94,
        f"Balanced-accuracy change: {delta:+.1f} points",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=TEAL if delta >= 0 else CORAL,
        weight="bold",
    )
    fig.savefig(output / "01_emg_supervised_eeg_transfer.png", dpi=220)
    plt.close(fig)


def _plot_dataset_and_participants(
    dataset_results: dict[str, Any],
    participant_results: dict[str, Any],
    output: Path,
) -> None:
    datasets = tuple(dataset_results)
    positions = np.arange(len(datasets))
    width = 0.34
    baseline = np.asarray(
        [100.0 * dataset_results[name]["target_only"]["balanced_accuracy"] for name in datasets]
    )
    transfer = np.asarray(
        [100.0 * dataset_results[name]["emg_supervised_eeg"]["balanced_accuracy"] for name in datasets]
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    axes[0].bar(positions - width / 2, baseline, width, color=GOLD, label="Target-only EEG")
    axes[0].bar(positions + width / 2, transfer, width, color=TEAL, label="EMG-supervised EEG")
    axes[0].set_xticks(positions, [name.upper() if name == "openbmi" else name.title() for name in datasets])
    axes[0].set_ylim(45, max(65.0, float(np.max((baseline, transfer))) + 5.0))
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_title("Pure-EEG transfer by target dataset", loc="left", weight="bold", color=INK)
    axes[0].legend(frameon=False)
    for offset, values in ((-width / 2, baseline), (width / 2, transfer)):
        for index, value in enumerate(values):
            axes[0].text(index + offset, value + 0.5, f"{value:.1f}", ha="center", color=INK, weight="bold", fontsize=9)

    color_map = {"liu2024": CORAL, "openbmi": TEAL, "physionet": GOLD}
    records = participant_results["records"]
    for index, dataset in enumerate(datasets):
        deltas = 100.0 * np.asarray(
            [record["balanced_accuracy_delta"] for record in records if record["dataset_id"] == dataset]
        )
        jitter = np.linspace(-0.12, 0.12, len(deltas)) if len(deltas) > 1 else np.zeros(len(deltas))
        axes[1].scatter(
            np.full(len(deltas), index) + jitter,
            deltas,
            color=color_map.get(dataset, INK),
            alpha=0.80,
            edgecolor="white",
            linewidth=0.5,
            s=42,
        )
        if len(deltas):
            axes[1].plot([index - 0.18, index + 0.18], [np.mean(deltas)] * 2, color=INK, linewidth=3)
    axes[1].axhline(0.0, color=INK, linewidth=1.0)
    axes[1].set_xticks(positions, [name.upper() if name == "openbmi" else name.title() for name in datasets])
    axes[1].set_ylabel("Participant balanced-accuracy change (points)")
    axes[1].set_title("Paired participant transfer effects", loc="left", weight="bold", color=INK)
    for ax in axes:
        ax.grid(axis="y", color="#D5DEE2", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    low, high = participant_results["bootstrap_95_ci"]
    fig.suptitle(
        f"EMG-Confirmed Transfer Generalization | Mean participant change {100 * participant_results['mean_balanced_accuracy_delta']:+.1f} points "
        f"(95% bootstrap CI {100 * low:+.1f} to {100 * high:+.1f})",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.savefig(output / "02_emg_transfer_dataset_and_participants.png", dpi=220)
    plt.close(fig)


def _write_report(payload: dict[str, Any], output: Path) -> None:
    baseline = payload["evaluation"]["target_only"]
    transfer = payload["evaluation"]["emg_supervised_eeg"]
    delta = 100.0 * (transfer["balanced_accuracy"] - baseline["balanced_accuracy"])
    participant = payload["participant_analysis"]
    ci_low, ci_high = participant["bootstrap_95_ci"]
    lines = [
        "# EMG-Supervised EEG Transfer Experiment",
        "",
        "## Leakage Boundary",
        "",
        "- EMG is used only in the synchronized source training cohort.",
        "- Positive source EEG windows end at or before confirmed movement onset.",
        "- The source adapter may use synchronized EMG and/or kinematics to establish that boundary.",
        "- Target development, controller, and observed-test rows contain EEG features only.",
        "- The deployed student prediction API accepts EEG only.",
        "",
        "## Development-Observed Pure-EEG Result",
        "",
        "| Model | Balanced accuracy | Sensitivity | Specificity | AUROC |",
        "|---|---:|---:|---:|---:|",
        f"| Target-only EEG | {100 * baseline['balanced_accuracy']:.1f}% | {100 * baseline['sensitivity']:.1f}% | {100 * baseline['specificity']:.1f}% | {baseline['auroc']:.3f} |",
        f"| EMG-supervised EEG student | {100 * transfer['balanced_accuracy']:.1f}% | {100 * transfer['sensitivity']:.1f}% | {100 * transfer['specificity']:.1f}% | {transfer['auroc']:.3f} |",
        "",
        f"Balanced-accuracy change: **{delta:+.1f} percentage points**.",
        "",
        f"Across **{participant['participants']}** paired target participants, the equal-participant mean change was **{100 * participant['mean_balanced_accuracy_delta']:+.1f} points** (95% participant-bootstrap CI **{100 * ci_low:+.1f} to {100 * ci_high:+.1f}**); **{100 * participant['participants_improved_fraction']:.1f}%** of participants improved.",
        "",
        f"The transfer source weight was selected as **{payload['selected_source_weight']:.2f}** using only the controller-validation cohort. Target development training used equal aggregate weight per dataset and intent class; synchronized source training used equal aggregate weight per participant.",
        "",
        "## Dataset Results",
        "",
        "| Dataset | Model | Balanced accuracy | Sensitivity | Specificity |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset_id, comparison in payload["evaluation_by_dataset"].items():
        for key, label in (
            ("target_only", "Target-only EEG"),
            ("emg_supervised_eeg", "EMG-supervised EEG"),
        ):
            result = comparison[key]
            lines.append(
                f"| {dataset_id} | {label} | {100 * result['balanced_accuracy']:.1f}% | {100 * result['sensitivity']:.1f}% | {100 * result['specificity']:.1f}% |"
            )
    lines.extend([
        "",
        "This result measures transfer to EEG-only intent detection. It does not measure an EEG-EMG hybrid controller and does not alter the separate left/right direction head.",
        "It is exploratory because `development_observed` is not the still-locked external cohort.",
    ])
    (output / "EMG_SUPERVISED_EEG_TRANSFER_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_contract() -> None:
    print("Privileged-source EEG transfer experiment is ready for standardized caches.")
    print("Source NPZ keys:")
    print("  eeg_features, emg_features, labels, subject_keys")
    print("  premovement_end_seconds, eeg_feature_names, emg_feature_names, dataset_id")
    print("Target NPZ keys:")
    print("  eeg_features, labels, subject_keys, dataset_ids, roles, eeg_feature_names")
    print("Required roles: development, controller_validation, development_observed")
    print("Prepare caches with scripts/33_prepare_mrcp_eeg_emg_transfer.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer",
    )
    parser.add_argument(
        "--source-weight",
        type=float,
        help="Use one fixed transfer weight instead of the validation-selected grid.",
    )
    parser.add_argument(
        "--source-weight-grid",
        default="0.05,0.10,0.20,0.35",
        help="Comma-separated weights selected on controller validation only.",
    )
    parser.add_argument("--confidence-floor", type=float, default=0.60)
    parser.add_argument("--max-controller-false-activation", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print_contract()
        return
    if args.source_cache is None or args.target_cache is None:
        parser.error("--source-cache and --target-cache are required unless --check is used.")

    source = load_source_cache(args.source_cache)
    target = load_target_cache(args.target_cache)
    features = np.asarray(target["eeg_features"], dtype=np.float64)
    labels = np.asarray(target["labels"], dtype=np.int64)
    roles = np.asarray(target["roles"]).astype(str)
    dataset_ids = np.asarray(target["dataset_ids"]).astype(str)
    development = roles == "development"
    controller = roles == "controller_validation"
    observed = roles == "development_observed"
    target_weight = dataset_balanced_weights(
        labels[development],
        dataset_ids[development],
    )
    if args.source_weight is not None:
        source_weights = [args.source_weight]
    else:
        try:
            source_weights = sorted(
                {float(value.strip()) for value in args.source_weight_grid.split(",")}
            )
        except ValueError as error:
            parser.error(f"Invalid --source-weight-grid: {error}")
    if not source_weights or any(not 0.0 < value <= 1.0 for value in source_weights):
        parser.error("Every source weight must be in (0, 1].")

    candidates: list[dict[str, Any]] = []
    selected_model: EMGConfirmedEEGTransfer | None = None
    selected_key: tuple[float, float, float, float] | None = None
    selected_threshold = 0.5
    baseline_threshold = 0.5
    for source_weight in source_weights:
        candidate_model = EMGConfirmedEEGTransfer(
            seed=args.seed,
            source_weight=source_weight,
            confidence_floor=args.confidence_floor,
        ).fit(
            source,
            features[development],
            (labels[development] != 0).astype(np.int64),
            target["eeg_feature_names"],
            target_sample_weight=target_weight,
        )
        candidate_baseline = candidate_model.predict_target_only_probability(
            features[controller]
        )
        candidate_transfer = candidate_model.predict_intent_probability(
            features[controller]
        )
        candidate_baseline_threshold = select_intent_threshold(
            labels[controller],
            candidate_baseline,
            max_false_activation=args.max_controller_false_activation,
        )
        candidate_transfer_threshold = select_intent_threshold(
            labels[controller],
            candidate_transfer,
            max_false_activation=args.max_controller_false_activation,
        )
        controller_result = score(
            labels[controller],
            candidate_transfer,
            candidate_transfer_threshold,
        )
        candidates.append(
            {
                "source_weight": source_weight,
                "threshold": candidate_transfer_threshold,
                "metrics": controller_result,
            }
        )
        candidate_key = (
            float(controller_result["balanced_accuracy"]),
            float(controller_result["sensitivity"]),
            -float(controller_result["false_activation_rate_per_window"]),
            -source_weight,
        )
        if selected_key is None or candidate_key > selected_key:
            selected_key = candidate_key
            selected_model = candidate_model
            selected_threshold = candidate_transfer_threshold
            baseline_threshold = candidate_baseline_threshold
    if selected_model is None:
        raise RuntimeError("No valid transfer model candidate was fitted.")

    model = selected_model
    thresholds = {
        "target_only": baseline_threshold,
        "emg_supervised_eeg": selected_threshold,
    }
    baseline_probability = model.predict_target_only_probability(features[observed])
    transfer_probability = model.predict_intent_probability(features[observed])
    evaluation = {
        "target_only": score(
            labels[observed],
            baseline_probability,
            thresholds["target_only"],
        ),
        "emg_supervised_eeg": score(
            labels[observed],
            transfer_probability,
            thresholds["emg_supervised_eeg"],
        ),
    }
    observed_datasets = dataset_ids[observed]
    evaluation_by_dataset = {}
    for dataset_id in np.unique(observed_datasets):
        selected = observed_datasets == dataset_id
        evaluation_by_dataset[str(dataset_id)] = {
            "target_only": score(
                labels[observed][selected],
                baseline_probability[selected],
                thresholds["target_only"],
            ),
            "emg_supervised_eeg": score(
                labels[observed][selected],
                transfer_probability[selected],
                thresholds["emg_supervised_eeg"],
            ),
        }
    participant = participant_analysis(
        labels[observed],
        baseline_probability,
        transfer_probability,
        thresholds["target_only"],
        thresholds["emg_supervised_eeg"],
        np.asarray(target["subject_keys"]).astype(str)[observed],
        observed_datasets,
        args.seed,
    )
    payload = {
        "schema_version": 1,
        "experiment": "emg_confirmed_source_to_eeg_only_target",
        "source_dataset": source.dataset_id,
        "leakage_boundary": {
            "emg_used_at_target_inference": False,
            "positive_source_eeg_ends_at_or_before_confirmed_movement_onset": True,
            "direction_head_modified": False,
        },
        "fit_summary": asdict(model.summary_),
        "selected_source_weight": model.source_weight,
        "controller_selection": candidates,
        "thresholds": thresholds,
        "evaluation": evaluation,
        "evaluation_by_dataset": evaluation_by_dataset,
        "participant_analysis": participant,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "emg_supervised_eeg_transfer_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(model, args.output_dir / "emg_supervised_eeg_student.joblib")
    _plot(evaluation, args.output_dir)
    _plot_dataset_and_participants(evaluation_by_dataset, participant, args.output_dir)
    _write_report(payload, args.output_dir)
    print(f"[+] Saved EMG-supervised EEG transfer result to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
