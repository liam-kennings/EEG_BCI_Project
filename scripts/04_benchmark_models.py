#!/usr/bin/env python3
"""Compare zero-shot FBCSP and EEGNet on locked unseen EEGMMI subjects."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bci_pipeline.dataset import available_subjects, parse_int_list, resolve_data_dir  # noqa: E402
from bci_pipeline.epochs import EpochConfig, build_raw_epoch_dataset  # noqa: E402
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.models.eegnet import (  # noqa: E402
    EEGNetTrainingConfig,
    HierarchicalEEGNet,
    select_balanced_intent_threshold,
    select_safety_intent_threshold,
)
from bci_pipeline.models.fbcsp import HierarchicalFBCSP  # noqa: E402
from bci_pipeline.splits import load_partitions, three_way_subject_split  # noqa: E402


def evaluate_model(
    name: str,
    model: object,
    validation_epochs: np.ndarray,
    validation_labels: np.ndarray,
    test_epochs: np.ndarray,
    test_labels: np.ndarray,
    test_subjects: np.ndarray,
    max_false_activation_rate: float,
) -> Dict[str, object]:
    validation_intent, _ = model.predict_stage_probabilities(validation_epochs)
    balanced_threshold = select_balanced_intent_threshold(validation_labels, validation_intent)
    safety_threshold = select_safety_intent_threshold(
        validation_labels,
        validation_intent,
        max_false_activation_rate,
    )
    test_intent, test_direction = model.predict_stage_probabilities(test_epochs)
    result = {
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
    print(
        f"[+] {name}: balanced={result['balanced']['balanced_accuracy'] * 100:.2f}% | "
        f"FP={result['balanced']['false_activation_rate'] * 100:.2f}% | "
        f"safety FP={result['safety']['false_activation_rate'] * 100:.2f}% | "
        f"safety miss={result['safety']['missed_intent_rate'] * 100:.2f}%"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--models", default="fbcsp,eegnet")
    parser.add_argument("--subjects", default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--cache-path", default="artifacts/benchmark/raw_epoch_cache.npz")
    parser.add_argument("--output-dir", default="artifacts/benchmark")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--eegnet-epochs", type=int, default=40)
    parser.add_argument("--eegnet-patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    requested_models = {name.strip().lower() for name in args.models.split(",") if name.strip()}
    if not requested_models or not requested_models.issubset({"fbcsp", "eegnet"}):
        raise ValueError("--models must contain fbcsp and/or eegnet.")
    if not 0.0 < args.max_false_activation < 1.0:
        raise ValueError("--max-false-activation must be between zero and one.")

    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir, parse_int_list(args.subjects), args.max_subjects)
    if args.subjects or args.max_subjects:
        partitions = three_way_subject_split(subjects, random_seed=args.seed)
    else:
        partitions = load_partitions(Path(args.splits))
        expected = set(partitions.development + partitions.controller_validation + partitions.final_target)
        if set(subjects) != expected:
            raise ValueError("The saved partitions do not match the discovered subject cohort.")

    dataset = build_raw_epoch_dataset(
        data_dir,
        subjects,
        Path(args.cache_path),
        EpochConfig(),
        args.rebuild_cache,
    )
    development_mask = np.isin(dataset.subjects, partitions.development)
    validation_mask = np.isin(dataset.subjects, partitions.controller_validation) & (dataset.runs == 12)
    test_mask = np.isin(dataset.subjects, partitions.final_target) & (dataset.runs == 12)
    if not np.any(development_mask) or not np.any(validation_mask) or not np.any(test_mask):
        raise RuntimeError("A benchmark partition contains no usable trials.")

    print(
        f"[+] Trials: development={np.sum(development_mask)}, "
        f"controller-validation={np.sum(validation_mask)}, final-target={np.sum(test_mask)}"
    )
    report: Dict[str, object] = {
        "protocol": {
            "claim": "zero-shot cross-subject, Run 12 final evaluation",
            "development_subjects": list(partitions.development),
            "controller_validation_subjects": list(partitions.controller_validation),
            "final_target_subjects": list(partitions.final_target),
            "development_uses_runs": [4, 8, 12],
            "validation_uses_runs": [12],
            "test_uses_runs": [12],
            "subject_overlap": False,
        },
        "dataset": {
            "trials": int(len(dataset.labels)),
            "channels": dataset.channel_names,
            "sfreq": dataset.sfreq,
        },
        "results": {},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "fbcsp" in requested_models:
        print("[+] Training hierarchical FBCSP...")
        fbcsp = HierarchicalFBCSP(dataset.sfreq, random_seed=args.seed).fit(
            dataset.epochs[development_mask],
            dataset.labels[development_mask],
        )
        report["results"]["fbcsp"] = evaluate_model(
            "FBCSP",
            fbcsp,
            dataset.epochs[validation_mask],
            dataset.labels[validation_mask],
            dataset.epochs[test_mask],
            dataset.labels[test_mask],
            dataset.subjects[test_mask],
            args.max_false_activation,
        )
        joblib.dump(fbcsp, output_dir / "fbcsp_zero_shot.joblib")

    if "eegnet" in requested_models:
        print("[+] Training hierarchical EEGNet...")
        eegnet_config = EEGNetTrainingConfig(
            epochs=args.eegnet_epochs,
            batch_size=args.batch_size,
            patience=args.eegnet_patience,
            random_seed=args.seed,
        )
        eegnet = HierarchicalEEGNet(dataset.epochs.shape[1], eegnet_config).fit(
            dataset.epochs[development_mask],
            dataset.labels[development_mask],
            dataset.epochs[validation_mask],
            dataset.labels[validation_mask],
        )
        report["results"]["eegnet"] = evaluate_model(
            "EEGNet",
            eegnet,
            dataset.epochs[validation_mask],
            dataset.labels[validation_mask],
            dataset.epochs[test_mask],
            dataset.labels[test_mask],
            dataset.subjects[test_mask],
            args.max_false_activation,
        )
        torch.save(eegnet.payload(), output_dir / "eegnet_zero_shot.pt")
        report["eegnet_training"] = {
            "config": asdict(eegnet_config),
            "history": eegnet.history_,
            "device": str(eegnet.device),
        }

    metrics_path = output_dir / "zero_shot_benchmark.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(f"[+] Saved benchmark: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
