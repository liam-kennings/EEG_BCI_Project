#!/usr/bin/env python3
"""Combine imagery intent with execution-informed direction without retraining."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bci_pipeline.epochs import epoch_fingerprint, load_epoch_cache, EpochConfig  # noqa: E402
from bci_pipeline.dataset import UNILATERAL_RUNS  # noqa: E402
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.models.eegnet import (  # noqa: E402
    HierarchicalEEGNet,
    select_balanced_intent_threshold,
    select_safety_intent_threshold,
)
from bci_pipeline.splits import load_partitions  # noqa: E402


def load_model(path: Path) -> HierarchicalEEGNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["device"] = "cpu"
    return HierarchicalEEGNet.from_payload(payload)


def evaluate_hybrid(
    intent_model: HierarchicalEEGNet,
    direction_model: HierarchicalEEGNet,
    validation_epochs: np.ndarray,
    validation_labels: np.ndarray,
    test_epochs: np.ndarray,
    test_labels: np.ndarray,
    test_subjects: np.ndarray,
    max_false_activation: float,
) -> Dict[str, object]:
    validation_intent, _ = intent_model.predict_stage_probabilities(validation_epochs)
    balanced_threshold = select_balanced_intent_threshold(validation_labels, validation_intent)
    safety_threshold = select_safety_intent_threshold(
        validation_labels, validation_intent, max_false_activation
    )
    test_intent, _ = intent_model.predict_stage_probabilities(test_epochs)
    _, test_direction = direction_model.predict_stage_probabilities(test_epochs)
    result: Dict[str, object] = {
        "balanced_threshold": balanced_threshold,
        "safety_threshold": safety_threshold,
        "balanced": hierarchical_metrics(
            test_labels, test_intent, test_direction, balanced_threshold
        ),
        "safety": hierarchical_metrics(test_labels, test_intent, test_direction, safety_threshold),
        "per_subject": {},
    }
    for subject in np.unique(test_subjects):
        mask = test_subjects == subject
        result["per_subject"][f"S{int(subject):03d}"] = {
            "balanced": hierarchical_metrics(
                test_labels[mask], test_intent[mask], test_direction[mask], balanced_threshold
            ),
            "safety": hierarchical_metrics(
                test_labels[mask], test_intent[mask], test_direction[mask], safety_threshold
            ),
        }
    return result


def main() -> None:
    output_dir = PROJECT_ROOT / "artifacts" / "cross_task"
    report_path = output_dir / "cross_task_metrics.json"
    report = json.loads(report_path.read_text(encoding="ascii"))
    partitions = load_partitions(PROJECT_ROOT / "artifacts" / "dataset" / "splits.json")
    all_subjects = sorted(
        partitions.development + partitions.controller_validation + partitions.final_target
    )
    dataset = load_epoch_cache(
        output_dir / "execution_imagery_epochs.npz",
        epoch_fingerprint(EpochConfig(runs=UNILATERAL_RUNS), all_subjects),
    )
    if dataset is None:
        raise RuntimeError("The full execution/imagery epoch cache is unavailable or stale.")
    validation = np.isin(dataset.subjects, partitions.controller_validation) & (dataset.runs == 12)
    test = np.isin(dataset.subjects, partitions.final_target) & (dataset.runs == 12)

    imagery = load_model(output_dir / "eegnet_imagery.pt")
    execution = load_model(output_dir / "eegnet_execution.pt")
    fine_tuned = load_model(output_dir / "eegnet_pretrained_finetuned.pt")
    models = {
        "imagery_only": imagery,
        "imagery_intent_execution_direction": execution,
        "imagery_intent_pretrained_direction": fine_tuned,
    }
    results = {
        name: evaluate_hybrid(
            imagery,
            direction_model,
            dataset.epochs[validation],
            dataset.labels[validation],
            dataset.epochs[test],
            dataset.labels[test],
            dataset.subjects[test],
            0.10,
        )
        for name, direction_model in models.items()
    }
    report["hybrid_direction_results"] = results
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")

    names = list(results)
    balanced = [results[name]["balanced"]["balanced_accuracy"] for name in names]
    direction = [results[name]["balanced"]["direction_balanced_accuracy"] for name in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.34
    ax.bar(x - width / 2, balanced, width, label="Three-class balanced accuracy")
    ax.bar(x + width / 2, direction, width, label="Conditional direction accuracy")
    for index, value in enumerate(balanced):
        ax.text(index - width / 2, value + 0.01, f"{value * 100:.1f}", ha="center")
    for index, value in enumerate(direction):
        ax.text(index + width / 2, value + 0.01, f"{value * 100:.1f}", ha="center")
    ax.set_xticks(x, [name.replace("imagery_intent_", "").replace("_", "\n") for name in names])
    ax.set_ylim(0.0, 0.8)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Does Execution Help Only the Direction Decoder?")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_path = output_dir / "hybrid_direction_heads.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    for name, result in results.items():
        print(
            f"{name:38s} | BA={result['balanced']['balanced_accuracy'] * 100:.2f}% "
            f"| direction={result['balanced']['direction_balanced_accuracy'] * 100:.2f}% "
            f"| safety FP={result['safety']['false_activation_rate'] * 100:.2f}% "
            f"| safety miss={result['safety']['missed_intent_rate'] * 100:.2f}%"
        )
    print(f"[+] Saved visual: {figure_path.resolve()}")


if __name__ == "__main__":
    main()

