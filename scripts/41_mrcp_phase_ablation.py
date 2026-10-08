#!/usr/bin/env python3
"""Development-only ablation of explicit MRCP phase features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from bci_pipeline.openbmi_controlled import StageProbabilities  # noqa: E402
from bci_pipeline.translational_dataset import load_translational_dataset  # noqa: E402
from bci_pipeline.translational_decoder import DomainAwareHierarchicalDecoder  # noqa: E402
from bci_pipeline.translational_features import (  # noqa: E402
    mrcp_features,
    mrcp_phase_features,
)
from bci_pipeline.translational_metrics import (  # noqa: E402
    score_operating_point,
    select_window_threshold,
)


DEFAULT_CACHE = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v3"
    / "translational_windows_v3.npz"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v4_neural_integrated"
    / "translational_v4_neural_models.joblib"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "mrcp_phase_ablation"


def _role_subjects(roles: dict[str, object], role: str) -> set[str]:
    return {
        str(subject)
        for dataset_roles in roles.values()
        for subject in dataset_roles.get(role, [])
    }


def _fit_predict(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    training: np.ndarray,
    evaluation: np.ndarray,
    seed: int,
) -> StageProbabilities:
    decoder = DomainAwareHierarchicalDecoder(seed=seed).fit(
        features[training], labels[training], domains[training]
    )
    return decoder.predict_stage_probabilities(
        features[evaluation], domains[evaluation]
    )


def run_ablation(cache_path: Path, model_path: Path, output_dir: Path) -> dict[str, object]:
    dataset = load_translational_dataset(cache_path)
    if dataset is None:
        raise FileNotFoundError(cache_path)
    bundle = joblib.load(model_path)
    roles = bundle["subject_roles"]
    development_subjects = _role_subjects(roles, "development")
    controller_subjects = _role_subjects(roles, "controller_validation")
    development = np.isin(dataset.subject_keys, list(development_subjects))
    controller = np.isin(dataset.subject_keys, list(controller_subjects))
    if not np.any(development) or not np.any(controller):
        raise RuntimeError("Development/controller split could not be reconstructed.")
    if np.any(np.isin(dataset.subject_keys[development], list(controller_subjects))):
        raise RuntimeError("Subject leakage detected between ablation roles.")

    base, base_names = mrcp_features(
        dataset.mrcp_epochs, dataset.sfreq, dataset.channel_names
    )
    phase, phase_names = mrcp_phase_features(
        dataset.mrcp_epochs, dataset.sfreq, dataset.channel_names
    )
    candidates = {
        "mrcp_amplitude": base,
        "mrcp_amplitude_plus_phase": np.column_stack((base, phase)).astype(np.float32),
    }
    development_indices = np.flatnonzero(development)
    unique_groups = np.unique(dataset.subject_keys[development])
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    results: dict[str, object] = {}
    for offset, (name, features) in enumerate(candidates.items()):
        oof_intent = np.zeros(np.sum(development), dtype=np.float64)
        oof_direction = np.zeros((np.sum(development), 2), dtype=np.float64)
        local_labels = dataset.labels[development]
        local_domains = dataset.dataset_ids[development]
        local_groups = dataset.subject_keys[development]
        for fold, (train_local, test_local) in enumerate(
            splitter.split(features[development], local_labels, local_groups)
        ):
            probability = _fit_predict(
                features[development],
                local_labels,
                local_domains,
                np.isin(np.arange(len(local_labels)), train_local),
                np.isin(np.arange(len(local_labels)), test_local),
                42 + offset * 100 + fold,
            )
            oof_intent[test_local] = probability.intent
            oof_direction[test_local] = probability.direction
        threshold = select_window_threshold(
            local_labels,
            oof_intent,
            np.ones(len(local_labels), dtype=bool),
            max_false_activation_rate=0.10,
        )
        evaluation_probability = _fit_predict(
            features,
            dataset.labels,
            dataset.dataset_ids,
            development,
            controller,
            900 + offset,
        )
        oof_metrics = score_operating_point(
            local_labels,
            StageProbabilities(oof_intent, oof_direction),
            np.ones(len(local_labels), dtype=bool),
            threshold,
        )
        validation_metrics = score_operating_point(
            dataset.labels[controller],
            evaluation_probability,
            np.ones(np.sum(controller), dtype=bool),
            threshold,
        )
        results[name] = {
            "feature_count": int(features.shape[1]),
            "threshold_selected_on_development_oof": float(threshold),
            "development_group_oof": oof_metrics,
            "controller_validation": validation_metrics,
        }

    baseline = results["mrcp_amplitude"]["controller_validation"]
    candidate = results["mrcp_amplitude_plus_phase"]["controller_validation"]
    ba_change = float(candidate["intent_balanced_accuracy"] - baseline["intent_balanced_accuracy"])
    auroc_change = float(candidate["intent_auroc"] - baseline["intent_auroc"])
    specificity_change = float(candidate["intent_specificity"] - baseline["intent_specificity"])
    advancement = (
        ba_change >= 0.01 and auroc_change > 0.0 and specificity_change >= -0.01
    )
    payload = {
        "status": "development_only_ablation_no_external_claim",
        "cache": str(cache_path.resolve()),
        "frozen_v4_model_unchanged": str(model_path.resolve()),
        "roles_used": ["development", "controller_validation"],
        "roles_not_opened": ["development_observed", "locked_external"],
        "features": {
            "amplitude_names": list(base_names),
            "phase_feature_count": len(phase_names),
            "phase_definition": (
                "Hilbert unit-phase cosine, sine, concentration, phase slope, and "
                "C3-C4 phase difference in predeclared 0.5 s MRCP bins"
            ),
        },
        "results": results,
        "candidate_minus_baseline": {
            "balanced_accuracy": ba_change,
            "auroc": auroc_change,
            "specificity": specificity_change,
        },
        "predeclared_advancement_rule": (
            ">=1 point controller-validation BA gain, AUROC gain >0, and specificity "
            "loss no worse than 1 point"
        ),
        "advancement_rule_passed": bool(advancement),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "mrcp_phase_ablation_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

    figure_path = output_dir / "01_mrcp_phase_ablation.png"
    metric_keys = ("intent_balanced_accuracy", "intent_sensitivity", "intent_specificity", "intent_auroc")
    labels = ("Balanced\naccuracy", "Sensitivity", "Specificity", "AUROC")
    x = np.arange(len(metric_keys))
    width = 0.34
    figure, axis = plt.subplots(figsize=(10.2, 5.3))
    figure.patch.set_facecolor("#f7f4ed")
    axis.set_facecolor("#ffffff")
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80)
    baseline_values = [100.0 * float(baseline[key]) for key in metric_keys]
    candidate_values = [100.0 * float(candidate[key]) for key in metric_keys]
    axis.bar(x - width / 2, baseline_values, width, color="#12355b", label="MRCP amplitude")
    axis.bar(x + width / 2, candidate_values, width, color="#2f8f83", label="Amplitude + phase")
    for positions, values in ((x - width / 2, baseline_values), (x + width / 2, candidate_values)):
        for position, value in zip(positions, values):
            axis.text(position, value + 1.0, f"{value:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Controller-validation performance (%)")
    figure.text(
        0.08,
        0.94,
        "Explicit MRCP Phase Feature Ablation",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=16,
    )
    figure.text(
        0.08,
        0.875,
        "Development subjects train; controller subjects select. "
        "Development-observed and external roles remain unopened.",
        ha="left",
        va="top",
        color="#627d98",
        fontsize=9,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="lower right")
    figure.savefig(figure_path, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)

    report_path = output_dir / "MRCP_PHASE_ABLATION_REPORT.md"
    decision = "passed the development advancement rule" if advancement else "did not pass the development advancement rule"
    report_path.write_text(
        "\n".join(
            (
                "# MRCP Phase Feature Ablation",
                "",
                "## Scope",
                "",
                "This is a development-only feature audit. It does not retrain or replace the frozen v4 model, and it does not inspect the `development_observed` or future `locked_external` cohorts.",
                "",
                "## Controller-Validation Result",
                "",
                f"- MRCP amplitude balanced accuracy: **{100.0 * float(baseline['intent_balanced_accuracy']):.1f}%**.",
                f"- MRCP amplitude + phase balanced accuracy: **{100.0 * float(candidate['intent_balanced_accuracy']):.1f}%**.",
                f"- AUROC change: **{100.0 * auroc_change:+.1f} points**.",
                f"- Specificity change: **{100.0 * specificity_change:+.1f} points**.",
                f"- Decision: the phase candidate **{decision}**.",
                "",
                "A passing result only licenses future locked testing; it is not evidence of external or clinical benefit.",
                "",
            )
        ),
        encoding="ascii",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ablation(args.cache_path, args.model_path, args.output_dir)
    change = result["candidate_minus_baseline"]
    print("\n=== Development-Only MRCP Phase Ablation ===")
    print(f"[+] Balanced-accuracy change: {100.0 * change['balanced_accuracy']:+.2f} points")
    print(f"[+] AUROC change: {100.0 * change['auroc']:+.2f} points")
    print(f"[+] Advancement rule passed: {result['advancement_rule_passed']}")
    print("[!] Frozen v4 model and development-observed/external cohorts were not changed or opened.")
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
