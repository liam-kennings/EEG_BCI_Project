#!/usr/bin/env python3
"""Locked motor-execution to motor-imagery transfer benchmark with visuals."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from bci_pipeline.dataset import (  # noqa: E402
    EXECUTION_RUNS,
    IMAGERY_RUNS,
    UNILATERAL_RUNS,
    available_subjects,
    parse_int_list,
    resolve_data_dir,
)
from bci_pipeline.epochs import EpochConfig, RawEpochDataset, build_raw_epoch_dataset  # noqa: E402
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.models.eegnet import (  # noqa: E402
    EEGNetTrainingConfig,
    HierarchicalEEGNet,
    select_balanced_intent_threshold,
    select_safety_intent_threshold,
)
from bci_pipeline.models.fbcsp import HierarchicalFBCSP  # noqa: E402
from bci_pipeline.splits import load_partitions, three_way_subject_split  # noqa: E402


AUDIT_BANDS = {
    "mu_7_13": (("mu", 7.0, 13.0),),
    "low_beta_13_20": (("low_beta", 13.0, 20.0),),
    "high_beta_20_30": (("high_beta", 20.0, 30.0),),
    "mu_beta_7_30": (("mu_beta", 7.0, 30.0),),
}


def masks_for_protocol(dataset: RawEpochDataset, partitions: object) -> Dict[str, np.ndarray]:
    development = np.isin(dataset.subjects, partitions.development)
    controller = np.isin(dataset.subjects, partitions.controller_validation)
    final = np.isin(dataset.subjects, partitions.final_target)
    execution = np.isin(dataset.runs, EXECUTION_RUNS)
    imagery = np.isin(dataset.runs, IMAGERY_RUNS)
    return {
        "development_execution": development & execution,
        "development_imagery": development & imagery,
        "development_combined": development & (execution | imagery),
        "validation_execution": controller & (dataset.runs == 11),
        "validation_imagery": controller & (dataset.runs == 12),
        "test_execution": final & (dataset.runs == 11),
        "test_imagery": final & (dataset.runs == 12),
    }


def evaluate(
    model: object,
    validation_epochs: np.ndarray,
    validation_labels: np.ndarray,
    test_epochs: np.ndarray,
    test_labels: np.ndarray,
    test_subjects: np.ndarray,
    max_false_activation: float,
) -> Dict[str, object]:
    validation_intent, _ = model.predict_stage_probabilities(validation_epochs)
    balanced_threshold = select_balanced_intent_threshold(validation_labels, validation_intent)
    safety_threshold = select_safety_intent_threshold(
        validation_labels,
        validation_intent,
        max_false_activation,
    )
    test_intent, test_direction = model.predict_stage_probabilities(test_epochs)
    result: Dict[str, object] = {
        "balanced_threshold": balanced_threshold,
        "safety_threshold": safety_threshold,
        "balanced": hierarchical_metrics(
            test_labels,
            test_intent,
            test_direction,
            balanced_threshold,
        ),
        "safety": hierarchical_metrics(
            test_labels,
            test_intent,
            test_direction,
            safety_threshold,
        ),
        "per_subject": {},
    }
    for subject in np.unique(test_subjects):
        subject_mask = test_subjects == subject
        result["per_subject"][f"S{int(subject):03d}"] = {
            "trials": int(np.sum(subject_mask)),
            "balanced": hierarchical_metrics(
                test_labels[subject_mask],
                test_intent[subject_mask],
                test_direction[subject_mask],
                balanced_threshold,
            ),
            "safety": hierarchical_metrics(
                test_labels[subject_mask],
                test_intent[subject_mask],
                test_direction[subject_mask],
                safety_threshold,
            ),
        }
    return result


def print_result(model_name: str, protocol: str, result: Dict[str, object]) -> None:
    balanced, safety = result["balanced"], result["safety"]
    print(
        f"    {model_name:6s} {protocol:24s} | BA={balanced['balanced_accuracy'] * 100:5.2f}% "
        f"| intent={balanced['intent_balanced_accuracy'] * 100:5.2f}% "
        f"| direction={balanced['direction_balanced_accuracy'] * 100:5.2f}% "
        f"| FP={balanced['false_activation_rate'] * 100:5.1f}% "
        f"| safety FP={safety['false_activation_rate'] * 100:4.1f}% "
        f"miss={safety['missed_intent_rate'] * 100:5.1f}%"
    )


def paired_patient_comparisons(
    all_results: Dict[str, object],
    seed: int,
    bootstrap_samples: int = 10000,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    comparisons: Dict[str, object] = {}
    for model_name, protocols in all_results.items():
        baseline = protocols.get("imagery_to_imagery")
        if baseline is None:
            continue
        subjects = sorted(baseline["per_subject"])
        baseline_ba = np.asarray(
            [baseline["per_subject"][subject]["balanced"]["balanced_accuracy"] for subject in subjects]
        )
        comparisons[model_name] = {}
        for protocol, result in protocols.items():
            if protocol in {"imagery_to_imagery", "execution_to_execution"}:
                continue
            values = np.asarray(
                [result["per_subject"][subject]["balanced"]["balanced_accuracy"] for subject in subjects]
            )
            delta = values - baseline_ba
            bootstrap_indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
            bootstrap_means = np.mean(delta[bootstrap_indices], axis=1)
            if np.allclose(delta, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            safety_fp_delta = np.asarray(
                [
                    result["per_subject"][subject]["safety"]["false_activation_rate"]
                    - baseline["per_subject"][subject]["safety"]["false_activation_rate"]
                    for subject in subjects
                ]
            )
            safety_miss_delta = np.asarray(
                [
                    result["per_subject"][subject]["safety"]["missed_intent_rate"]
                    - baseline["per_subject"][subject]["safety"]["missed_intent_rate"]
                    for subject in subjects
                ]
            )
            comparisons[model_name][protocol] = {
                "patients": len(subjects),
                "patients_improved": int(np.sum(delta > 0.0)),
                "patients_unchanged": int(np.sum(np.isclose(delta, 0.0))),
                "mean_balanced_accuracy_delta": float(np.mean(delta)),
                "median_balanced_accuracy_delta": float(np.median(delta)),
                "mean_delta_bootstrap_95_ci": [
                    float(np.quantile(bootstrap_means, 0.025)),
                    float(np.quantile(bootstrap_means, 0.975)),
                ],
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p_value": float(p_value),
                "mean_safety_false_activation_delta": float(np.mean(safety_fp_delta)),
                "mean_safety_missed_intent_delta": float(np.mean(safety_miss_delta)),
            }
    return comparisons


def train_fbcsp_protocols(
    dataset: RawEpochDataset,
    masks: Dict[str, np.ndarray],
    max_false_activation: float,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, HierarchicalFBCSP]]:
    results: Dict[str, object] = {}
    models: Dict[str, HierarchicalFBCSP] = {}
    training_sets = {
        "execution": "development_execution",
        "imagery": "development_imagery",
        "combined": "development_combined",
    }
    for name, mask_name in training_sets.items():
        model = HierarchicalFBCSP(dataset.sfreq, random_seed=seed).fit(
            dataset.epochs[masks[mask_name]],
            dataset.labels[masks[mask_name]],
        )
        models[name] = model

    protocol_specs = {
        "execution_to_execution": ("execution", "validation_execution", "test_execution"),
        "execution_to_imagery": ("execution", "validation_imagery", "test_imagery"),
        "imagery_to_imagery": ("imagery", "validation_imagery", "test_imagery"),
        "combined_to_imagery": ("combined", "validation_imagery", "test_imagery"),
    }
    for protocol, (model_name, validation_name, test_name) in protocol_specs.items():
        result = evaluate(
            models[model_name],
            dataset.epochs[masks[validation_name]],
            dataset.labels[masks[validation_name]],
            dataset.epochs[masks[test_name]],
            dataset.labels[masks[test_name]],
            dataset.subjects[masks[test_name]],
            max_false_activation,
        )
        results[protocol] = result
        print_result("FBCSP", protocol, result)
    return results, models


def train_eegnet(
    dataset: RawEpochDataset,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: EEGNetTrainingConfig,
) -> HierarchicalEEGNet:
    return HierarchicalEEGNet(dataset.epochs.shape[1], config).fit(
        dataset.epochs[train_mask],
        dataset.labels[train_mask],
        dataset.epochs[validation_mask],
        dataset.labels[validation_mask],
    )


def train_eegnet_protocols(
    dataset: RawEpochDataset,
    masks: Dict[str, np.ndarray],
    max_false_activation: float,
    config: EEGNetTrainingConfig,
    fine_tune_epochs: int,
) -> Tuple[Dict[str, object], Dict[str, HierarchicalEEGNet]]:
    results: Dict[str, object] = {}
    models: Dict[str, HierarchicalEEGNet] = {}

    print("[+] EEGNet execution model")
    execution_model = train_eegnet(
        dataset,
        masks["development_execution"],
        masks["validation_execution"],
        config,
    )
    models["execution"] = execution_model
    for protocol, validation_name, test_name in (
        ("execution_to_execution", "validation_execution", "test_execution"),
        ("execution_to_imagery", "validation_imagery", "test_imagery"),
    ):
        result = evaluate(
            execution_model,
            dataset.epochs[masks[validation_name]],
            dataset.labels[masks[validation_name]],
            dataset.epochs[masks[test_name]],
            dataset.labels[masks[test_name]],
            dataset.subjects[masks[test_name]],
            max_false_activation,
        )
        results[protocol] = result
        print_result("EEGNet", protocol, result)

    print("[+] EEGNet imagery model")
    imagery_model = train_eegnet(
        dataset,
        masks["development_imagery"],
        masks["validation_imagery"],
        config,
    )
    models["imagery"] = imagery_model
    results["imagery_to_imagery"] = evaluate(
        imagery_model,
        dataset.epochs[masks["validation_imagery"]],
        dataset.labels[masks["validation_imagery"]],
        dataset.epochs[masks["test_imagery"]],
        dataset.labels[masks["test_imagery"]],
        dataset.subjects[masks["test_imagery"]],
        max_false_activation,
    )
    print_result("EEGNet", "imagery_to_imagery", results["imagery_to_imagery"])

    print("[+] EEGNet combined model")
    combined_model = train_eegnet(
        dataset,
        masks["development_combined"],
        masks["validation_imagery"],
        config,
    )
    models["combined"] = combined_model
    results["combined_to_imagery"] = evaluate(
        combined_model,
        dataset.epochs[masks["validation_imagery"]],
        dataset.labels[masks["validation_imagery"]],
        dataset.epochs[masks["test_imagery"]],
        dataset.labels[masks["test_imagery"]],
        dataset.subjects[masks["test_imagery"]],
        max_false_activation,
    )
    print_result("EEGNet", "combined_to_imagery", results["combined_to_imagery"])

    print("[+] EEGNet execution-pretrained imagery fine-tune")
    fine_tuned = copy.deepcopy(execution_model)
    fine_tuned.config = EEGNetTrainingConfig(
        epochs=fine_tune_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate * 0.2,
        weight_decay=config.weight_decay,
        patience=max(3, min(config.patience, fine_tune_epochs // 2)),
        dropout=config.dropout,
        random_seed=config.random_seed,
        device=str(fine_tuned.device),
    )
    fine_tuned.fit(
        dataset.epochs[masks["development_imagery"]],
        dataset.labels[masks["development_imagery"]],
        dataset.epochs[masks["validation_imagery"]],
        dataset.labels[masks["validation_imagery"]],
    )
    models["pretrained_finetuned"] = fine_tuned
    results["pretrained_finetuned_to_imagery"] = evaluate(
        fine_tuned,
        dataset.epochs[masks["validation_imagery"]],
        dataset.labels[masks["validation_imagery"]],
        dataset.epochs[masks["test_imagery"]],
        dataset.labels[masks["test_imagery"]],
        dataset.subjects[masks["test_imagery"]],
        max_false_activation,
    )
    print_result(
        "EEGNet",
        "pretrained_finetuned_to_imagery",
        results["pretrained_finetuned_to_imagery"],
    )
    return results, models


def run_fbcsp_audits(
    dataset: RawEpochDataset,
    masks: Dict[str, np.ndarray],
    max_false_activation: float,
    seed: int,
) -> Dict[str, object]:
    windows = {
        "full_0.5_3.5s": slice(None),
        "early_0.5_1.5s": slice(0, int(round(dataset.sfreq))),
        "late_1.5_3.5s": slice(int(round(dataset.sfreq)), None),
    }
    results: Dict[str, object] = {}
    for window_name, time_slice in windows.items():
        results[window_name] = {}
        for band_name, filter_bank in AUDIT_BANDS.items():
            model = HierarchicalFBCSP(
                dataset.sfreq,
                components_per_band=4,
                random_seed=seed,
                filter_bank=filter_bank,
            ).fit(
                dataset.epochs[masks["development_execution"]][:, :, time_slice],
                dataset.labels[masks["development_execution"]],
            )
            results[window_name][band_name] = evaluate(
                model,
                dataset.epochs[masks["validation_imagery"]][:, :, time_slice],
                dataset.labels[masks["validation_imagery"]],
                dataset.epochs[masks["test_imagery"]][:, :, time_slice],
                dataset.labels[masks["test_imagery"]],
                dataset.subjects[masks["test_imagery"]],
                max_false_activation,
            )
            metric = results[window_name][band_name]["balanced"]
            print(
                f"    audit {window_name:17s} {band_name:16s} | "
                f"BA={metric['balanced_accuracy'] * 100:.2f}% "
                f"FP={metric['false_activation_rate'] * 100:.1f}%"
            )
    return results


def metric_matrix(
    results: Dict[str, object],
    protocols: Iterable[str],
    operating_point: str,
) -> np.ndarray:
    rows = []
    for protocol in protocols:
        metrics = results[protocol][operating_point]
        rows.append(
            [
                metrics["balanced_accuracy"],
                metrics["intent_balanced_accuracy"],
                metrics["direction_balanced_accuracy"],
                1.0 - metrics["false_activation_rate"],
                1.0 - metrics["missed_intent_rate"],
            ]
        )
    return np.asarray(rows)


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column] * 100:.1f}", ha="center", va="center")


def plot_model_heatmap(model_name: str, results: Dict[str, object], output_dir: Path) -> None:
    protocols = list(results)
    labels = [name.replace("_", "\n") for name in protocols]
    columns = ["Balanced acc", "Intent BA", "Direction BA", "Rest specificity", "Intent sensitivity"]
    matrix = metric_matrix(results, protocols, "balanced")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.75 * len(protocols))))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    annotate_heatmap(ax, matrix)
    ax.set_xticks(range(len(columns)), columns, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(f"{model_name}: Cross-Task Final-Subject Metrics (%)")
    fig.colorbar(image, ax=ax, label="Rate")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name.lower()}_cross_task_heatmap.png", dpi=220)
    plt.close(fig)


def plot_safety_tradeoff(all_results: Dict[str, object], output_dir: Path) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(all_results), figsize=(14, 6), sharex=True, sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    palette = plt.get_cmap("tab10")
    for ax, (model_name, protocols) in zip(axes, all_results.items()):
        imagery_items = [item for item in protocols.items() if item[0] != "execution_to_execution"]
        for item_index, (protocol, result) in enumerate(imagery_items):
            color = palette(item_index)
            points = []
            for operating_point, marker in (("balanced", "o"), ("safety", "s")):
                metrics = result[operating_point]
                x = metrics["false_activation_rate"]
                y = 1.0 - metrics["missed_intent_rate"]
                points.append((x, y))
                ax.scatter(
                    x,
                    y,
                    color=color,
                    marker=marker,
                    s=65,
                    alpha=0.85,
                )
            label = protocol.replace("_to_imagery", "").replace(
                "pretrained_finetuned", "pretrain + fine-tune"
            )
            ax.plot(*zip(*points), color=color, alpha=0.7, label=label)
        ax.set_xlabel("False-activation rate")
        ax.set_title(model_name.upper())
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        protocol_legend = ax.legend(loc="upper left", frameon=False, fontsize=8)
        ax.add_artist(protocol_legend)
    axes[0].set_ylabel("Intent sensitivity (1 - missed intent)")
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray", label="Balanced threshold"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="gray", label="Safety threshold"),
    ]
    axes[-1].legend(handles=legend_items, loc="lower right", frameon=False)
    fig.suptitle("Imagery Safety Tradeoff on Unseen Final Subjects")
    fig.tight_layout()
    fig.savefig(output_dir / "safety_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_per_patient(all_results: Dict[str, object], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(all_results), figsize=(15, 6), sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    patient_count = 0
    for ax, (model_name, protocols) in zip(axes, all_results.items()):
        imagery_protocols = [name for name in protocols if name != "execution_to_execution"]
        subjects = sorted(next(iter(protocols.values()))["per_subject"])
        patient_count = len(subjects)
        baseline_name = "imagery_to_imagery"
        baseline = np.asarray(
            [protocols[baseline_name]["per_subject"][subject]["balanced"]["balanced_accuracy"] for subject in subjects]
        )
        for position, protocol in enumerate(imagery_protocols):
            values = np.asarray(
                [protocols[protocol]["per_subject"][subject]["balanced"]["balanced_accuracy"] for subject in subjects]
            )
            jitter = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(np.full(len(values), position) + jitter, values, s=18, alpha=0.55)
            ax.plot([position - 0.18, position + 0.18], [np.median(values)] * 2, color="black", lw=2)
            if protocol != baseline_name:
                improved = int(np.sum(values > baseline))
                ax.text(position, 0.99, f"{improved}/{len(values)}\nimproved", ha="center", va="top", fontsize=8)
        ax.set_xticks(range(len(imagery_protocols)), [name.replace("_to_imagery", "").replace("_", "\n") for name in imagery_protocols])
        ax.set_title(model_name.upper())
        ax.set_ylim(0.0, 1.03)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Per-patient balanced accuracy")
    fig.suptitle(f"Cross-Task Imagery Performance Across {patient_count} Unseen Subjects")
    fig.tight_layout()
    fig.savefig(output_dir / "per_patient_cross_task.png", dpi=220)
    plt.close(fig)


def plot_audit(audit: Dict[str, object], output_dir: Path) -> None:
    windows = list(audit)
    bands = list(next(iter(audit.values())))
    matrix = np.asarray(
        [
            [audit[window][band]["balanced"]["balanced_accuracy"] for band in bands]
            for window in windows
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    image = ax.imshow(matrix, vmin=0.25, vmax=max(0.6, float(np.max(matrix))), cmap="YlGnBu")
    annotate_heatmap(ax, matrix)
    ax.set_xticks(range(len(bands)), [name.replace("_", "\n") for name in bands])
    ax.set_yticks(range(len(windows)), [name.replace("_", "\n") for name in windows])
    ax.set_title("FBCSP Execution-to-Imagery Band/Time Audit (%)")
    fig.colorbar(image, ax=ax, label="Balanced accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "fbcsp_band_time_audit.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--models", default="fbcsp,eegnet")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--cache-path", default="artifacts/cross_task/execution_imagery_epochs.npz")
    parser.add_argument("--output-dir", default="artifacts/cross_task")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-audits", action="store_true")
    parser.add_argument("--plots-only", action="store_true", help="Regenerate visuals from the saved JSON")
    parser.add_argument("--eegnet-epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--eegnet-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_path = output_dir / "cross_task_metrics.json"
    if args.plots_only:
        report = json.loads(report_path.read_text(encoding="ascii"))
        report["paired_patient_comparisons"] = paired_patient_comparisons(
            report["results"], args.seed
        )
        for model_name, results in report["results"].items():
            plot_model_heatmap(model_name, results, output_dir)
        plot_safety_tradeoff(report["results"], output_dir)
        plot_per_patient(report["results"], output_dir)
        if "fbcsp_band_time_audit" in report:
            plot_audit(report["fbcsp_band_time_audit"], output_dir)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
        print(f"[+] Regenerated visuals: {output_dir.resolve()}")
        return

    requested_models = {name.strip().lower() for name in args.models.split(",") if name.strip()}
    if not requested_models or not requested_models.issubset({"fbcsp", "eegnet"}):
        raise ValueError("--models must contain fbcsp and/or eegnet.")
    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir, parse_int_list(args.subjects), args.max_subjects)
    if args.subjects or args.max_subjects:
        partitions = three_way_subject_split(subjects, random_seed=args.seed)
    else:
        partitions = load_partitions(Path(args.splits))
        expected = set(partitions.development + partitions.controller_validation + partitions.final_target)
        if set(subjects) != expected:
            raise ValueError("Saved partitions do not match the discovered cohort.")

    started = time.time()
    dataset = build_raw_epoch_dataset(
        data_dir,
        subjects,
        Path(args.cache_path),
        EpochConfig(runs=UNILATERAL_RUNS),
        args.rebuild_cache,
    )
    masks = masks_for_protocol(dataset, partitions)
    print("[+] Trial partitions")
    for name, mask in masks.items():
        print(f"    {name:24s}: {int(np.sum(mask))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "protocol": {
            "development_subjects": list(partitions.development),
            "controller_validation_subjects": list(partitions.controller_validation),
            "final_target_subjects": list(partitions.final_target),
            "execution_runs": list(EXECUTION_RUNS),
            "imagery_runs": list(IMAGERY_RUNS),
            "primary_final_test": "imagery Run 12",
            "execution_run_11_used_for_imagery_calibration": False,
        },
        "dataset": {
            "trials": int(len(dataset.labels)),
            "channels": dataset.channel_names,
            "sfreq": dataset.sfreq,
            "epoch_shape": list(dataset.epochs.shape),
        },
        "results": {},
    }
    if report_path.exists() and requested_models != {"fbcsp", "eegnet"}:
        previous = json.loads(report_path.read_text(encoding="ascii"))
        if previous.get("protocol") == report["protocol"]:
            report["results"].update(previous.get("results", {}))
            if "fbcsp_band_time_audit" in previous:
                report["fbcsp_band_time_audit"] = previous["fbcsp_band_time_audit"]
    saved_models: Dict[str, object] = {}

    if "fbcsp" in requested_models:
        print("[+] Cross-task FBCSP matrix")
        fbcsp_results, fbcsp_models = train_fbcsp_protocols(
            dataset,
            masks,
            args.max_false_activation,
            args.seed,
        )
        report["results"]["fbcsp"] = fbcsp_results
        saved_models["fbcsp"] = fbcsp_models
        joblib.dump(fbcsp_models, output_dir / "cross_task_fbcsp_models.joblib")
        if not args.skip_audits:
            print("[+] FBCSP execution-to-imagery band/time audit")
            report["fbcsp_band_time_audit"] = run_fbcsp_audits(
                dataset,
                masks,
                args.max_false_activation,
                args.seed,
            )

    if "eegnet" in requested_models:
        eegnet_config = EEGNetTrainingConfig(
            epochs=args.eegnet_epochs,
            batch_size=args.batch_size,
            patience=args.eegnet_patience,
            random_seed=args.seed,
        )
        eegnet_results, eegnet_models = train_eegnet_protocols(
            dataset,
            masks,
            args.max_false_activation,
            eegnet_config,
            args.fine_tune_epochs,
        )
        report["results"]["eegnet"] = eegnet_results
        for name, model in eegnet_models.items():
            torch.save(model.payload(), output_dir / f"eegnet_{name}.pt")
        report["eegnet_config"] = asdict(eegnet_config)

    for model_name, results in report["results"].items():
        plot_model_heatmap(model_name, results, output_dir)
    plot_safety_tradeoff(report["results"], output_dir)
    plot_per_patient(report["results"], output_dir)
    if "fbcsp_band_time_audit" in report:
        plot_audit(report["fbcsp_band_time_audit"], output_dir)

    report["paired_patient_comparisons"] = paired_patient_comparisons(
        report["results"], args.seed
    )
    report["runtime_minutes"] = (time.time() - started) / 60.0
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(f"[+] Saved report: {report_path.resolve()}")
    print(f"[+] Saved visuals: {output_dir.resolve()}")
    print(f"[+] Runtime: {report['runtime_minutes']:.2f} minutes")


if __name__ == "__main__":
    main()
