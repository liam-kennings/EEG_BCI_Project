#!/usr/bin/env python3
"""Evaluate the frozen V7 decoder on the unseen Cho2017 healthy cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from bci_pipeline.cho2017 import (  # noqa: E402
    CHO_ADAPTER_DEVELOPMENT_SUBJECTS,
    CHO_CALIBRATION_PER_CLASS,
    CHO_DATASET_ID,
    CHO_EXPECTED_SUBJECTS,
    CHO_LOCKED_SUBJECTS,
    build_cho2017_window_dataset,
    cho2017_snapshot,
)
from bci_pipeline.neural_integration import predict_mrcp_fbcnet_files  # noqa: E402
from bci_pipeline.openbmi_controlled import StageProbabilities  # noqa: E402
from bci_pipeline.torch_device import (  # noqa: E402
    TORCH_DEVICE_CHOICES,
    resolve_torch_device,
    torch_device_metadata,
)
from bci_pipeline.translational_dataset import load_translational_dataset  # noqa: E402
from bci_pipeline.translational_decoder import (  # noqa: E402
    patient_adapted_auxiliary_intent_probabilities,
)
from bci_pipeline.translational_features import (  # noqa: E402
    build_external_translational_features,
)


CONFIRMATION = "CONFIRM_CHO2017_V7_LOCKED_RUN"
DEFAULT_DATASET = PROJECT_ROOT / "files" / "external" / "cho2017" / "nm000245"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "cho2017_locked_external_v7"
DEFAULT_CONTRACT = DEFAULT_OUTPUT / "CHO2017_V7_LOCKED_CONTRACT.json"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v7_auxiliary_direction"
    / "translational_v7_auxiliary_direction_models.joblib"
)
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v5_patient_adaptive"
    / "translational_windows_v3.npz"
)
INK = "#17324D"
BLUE = "#176B87"
GREEN = "#218F7A"
ORANGE = "#E9A22B"
RED = "#D96832"
PALE = "#E8EEF1"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _neural_files(bundle: Mapping[str, object]) -> tuple[Path, ...]:
    records = bundle["neural_integration"]["ensembles"]["mrcp_fbcnet"]["models"]
    paths = []
    for record in records:
        candidate = Path(str(record["path"])).expanduser()
        if not candidate.is_file():
            candidate = (
                PROJECT_ROOT
                / "artifacts"
                / "gui_experiments"
                / "eeg_decoder_model_bakeoff"
                / "seeds"
                / f"seed_{int(record['seed'])}"
                / "mrcp_fbcnet.pt"
            )
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        paths.append(candidate.resolve())
    return tuple(paths)


def _development_mask(reference: object, roles: Mapping[str, object]) -> np.ndarray:
    subjects = [
        subject
        for dataset_roles in roles.values()
        for subject in dataset_roles["development"]
    ]
    mask = np.isin(reference.subject_keys, subjects)
    if not np.any(mask):
        raise RuntimeError("V7 development subjects do not match the reference cache.")
    return mask


def _prepare_contract(
    dataset_root: Path,
    contract_path: Path,
    model_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    if contract_path.exists():
        raise FileExistsError(
            f"Contract already exists at {contract_path}; refusing to silently replace it."
        )
    bundle = joblib.load(model_path)
    reference = load_translational_dataset(reference_path)
    if reference is None or reference.fingerprint != bundle["dataset_fingerprint"]:
        raise RuntimeError("The V7 reference cache does not match the frozen model bundle.")
    snapshot = cho2017_snapshot(dataset_root, CHO_LOCKED_SUBJECTS)
    neural_files = _neural_files(bundle)
    contract = {
        "status": "frozen_before_eeg_inference",
        "purpose": "external_healthy_domain_confirmation_of_v7_rest_vs_intent",
        "dataset": {
            "id": CHO_DATASET_ID,
            "root": str(dataset_root),
            "source_subjects": len(CHO_EXPECTED_SUBJECTS),
            "locked_subjects": len(CHO_LOCKED_SUBJECTS),
            "adapter_development_subjects": list(
                CHO_ADAPTER_DEVELOPMENT_SUBJECTS
            ),
            "snapshot_fingerprint": snapshot["fingerprint"],
            "event_totals": snapshot["event_totals"],
            "bdf_bytes": snapshot["bdf_bytes"],
            "source_mat_bytes": snapshot["source_mat_bytes"],
        },
        "frozen_decoder": {
            "model": str(model_path),
            "model_sha256": _sha256(model_path),
            "training_dataset_fingerprint": bundle["dataset_fingerprint"],
            "reference_cache": str(reference_path),
            "reference_cache_sha256": _sha256(reference_path),
            "neural_models": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in neural_files
            ],
        },
        "protocol": {
            "task": "binary_rest_vs_motor_imagery_intent",
            "zero_shot": "all_eligible_windows_no_cho2017_fitting",
            "intent_window_seconds_after_cue": [0.5, 2.5],
            "rest_source": "independent_continuous_rest_recording",
            "rest_windows": "2_s_windows_with_1_s_step_starting_at_2_s",
            "patient_calibration": {
                "design": "stratified_within_session_not_prospective",
                "windows": {"rest": 10, "left": 10, "right": 10},
                "budget_per_source_class": CHO_CALIBRATION_PER_CLASS,
                "blend_weight": 0.5,
                "auxiliary_weight": 0.25,
                "direction_output_exposed": False,
            },
            "thresholds": {
                "conservative": float(
                    bundle["calibrated_thresholds"]["conservative_fallback"]
                ),
                "sensitivity_targeted": float(
                    bundle["calibrated_thresholds"]["sensitivity_targeted_fallback"]
                ),
            },
            "primary_endpoint": "adapted_conservative_intent_balanced_accuracy",
            "safety_endpoint": "adapted_conservative_rest_specificity",
            "bootstrap_unit": "participant",
        },
        "claim_boundary": [
            "Cho2017 was not used for V7 training, model selection, probability calibration, or threshold selection.",
            "Subject 1 was used only for adapter/QC smoke testing and is excluded from locked model inference.",
            "The adapted analysis uses frozen stratified within-session calibration and is not a later-session prospective result.",
            "Rest comes from the independent continuous resting-state recording.",
            "Window specificity is not a temporally confirmed false-command rate per minute.",
            "No result may be used to retune V7 and remain labelled locked confirmation.",
        ],
    }
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, indent=2, default=_json_default) + "\n",
        encoding="ascii",
    )
    return contract


def _verify_contract(
    dataset_root: Path,
    contract_path: Path,
    model_path: Path,
    reference_path: Path,
) -> tuple[dict[str, object], dict[str, object], tuple[Path, ...]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_eeg_inference":
        raise RuntimeError("Cho2017 contract is not frozen before inference.")
    snapshot = cho2017_snapshot(dataset_root, CHO_LOCKED_SUBJECTS)
    if snapshot["fingerprint"] != contract["dataset"]["snapshot_fingerprint"]:
        raise RuntimeError("Cho2017 dataset snapshot changed after contract freeze.")
    if _sha256(model_path) != contract["frozen_decoder"]["model_sha256"]:
        raise RuntimeError("Frozen V7 model hash changed.")
    if _sha256(reference_path) != contract["frozen_decoder"]["reference_cache_sha256"]:
        raise RuntimeError("Frozen V7 reference cache hash changed.")
    bundle = joblib.load(model_path)
    if bundle["dataset_fingerprint"] != contract["frozen_decoder"]["training_dataset_fingerprint"]:
        raise RuntimeError("Frozen V7 training fingerprint changed.")
    neural_files = _neural_files(bundle)
    expected = {
        Path(record["path"]).name + ":" + str(Path(record["path"]).parent.name): record["sha256"]
        for record in contract["frozen_decoder"]["neural_models"]
    }
    for path in neural_files:
        key = path.name + ":" + path.parent.name
        if key not in expected or _sha256(path) != expected[key]:
            raise RuntimeError(f"Frozen neural model hash changed: {path}")
    return contract, bundle, neural_files


def _predict_population(
    dataset: object,
    features: object,
    bundle: Mapping[str, object],
    neural_files: tuple[Path, ...],
    device: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    domains = dataset.dataset_ids
    probabilities: dict[str, StageProbabilities] = {}
    neutral_direction = np.full((len(dataset.labels), 2), 0.5, dtype=np.float64)
    for branch in ("mrcp", "smr", "riemann"):
        intent = bundle["models"][branch].predict_intent_probabilities(
            {branch: features.branches[branch]},
            domains,
        )
        probabilities[branch] = StageProbabilities(intent, neutral_direction.copy())
    neural_intent, _, neural_metadata = predict_mrcp_fbcnet_files(
        dataset.mrcp_epochs,
        dataset.smr_epochs,
        dataset.channel_names,
        neural_files,
        batch_size=512,
        device=device,
    )
    probabilities["mrcp_fbcnet"] = StageProbabilities(
        neural_intent,
        neutral_direction.copy(),
    )
    fused = bundle["stage_specific_fusion"].predict(probabilities)
    calibrated = bundle["probability_calibrator"].predict(fused, domains)
    available = bundle["quality_gate"].available(
        features.branches["quality"],
        domains,
    )
    return calibrated, available, neural_metadata


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _score(
    labels: np.ndarray,
    probability: np.ndarray,
    available: np.ndarray,
    threshold: float,
    selected: np.ndarray | None = None,
) -> dict[str, float | int]:
    mask = np.ones(len(labels), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    y = (np.asarray(labels)[mask] != 0).astype(np.int64)
    p = np.asarray(probability)[mask]
    local_available = np.asarray(available)[mask]
    issued = local_available & (p >= threshold)
    sensitivity = _safe_ratio(np.sum(issued & (y == 1)), np.sum(y == 1))
    specificity = _safe_ratio(np.sum((~issued) & (y == 0)), np.sum(y == 0))
    return {
        "rows": int(len(y)),
        "intent_rows": int(np.sum(y == 1)),
        "rest_rows": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "intent_sensitivity": sensitivity,
        "rest_specificity": specificity,
        "intent_balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "false_activation_fraction": 1.0 - specificity,
        "missed_intent_fraction": 1.0 - sensitivity,
        "auroc": float(roc_auc_score(y, p)) if set(np.unique(y)) == {0, 1} else float("nan"),
        "available_fraction": float(np.mean(local_available)),
        "abstained_fraction": float(np.mean(~local_available)),
        "issued_command_fraction": float(np.mean(issued)),
        "issued_commands": int(np.sum(issued)),
    }


def _bootstrap_subjects(
    dataset: object,
    probability: np.ndarray,
    available: np.ndarray,
    threshold: float,
    selected: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    selected = np.asarray(selected, dtype=bool)
    subjects = np.unique(dataset.subject_keys[selected])
    indices_by_subject = {
        subject: np.flatnonzero(selected & (dataset.subject_keys == subject))
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    metric_names = (
        "intent_balanced_accuracy",
        "intent_sensitivity",
        "rest_specificity",
        "false_activation_fraction",
        "auroc",
    )
    samples = {name: [] for name in metric_names}
    for _ in range(iterations):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([indices_by_subject[subject] for subject in sampled])
        local = _score(
            dataset.labels[indices],
            probability[indices],
            available[indices],
            threshold,
        )
        for name in metric_names:
            samples[name].append(float(local[name]))
    return {
        name: {
            "low": float(np.nanpercentile(values, 2.5)),
            "high": float(np.nanpercentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def _subject_scores(
    dataset: object,
    probability: np.ndarray,
    available: np.ndarray,
    threshold: float,
    selected: np.ndarray,
) -> list[dict[str, object]]:
    records = []
    for subject in np.unique(dataset.subject_keys[selected]):
        local = selected & (dataset.subject_keys == subject)
        records.append(
            {"subject": str(subject), **_score(dataset.labels, probability, available, threshold, local)}
        )
    return records


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#AEBCC5",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def _plot_summary(path: Path, report: Mapping[str, object]) -> None:
    _style()
    names = ("Balanced accuracy", "Intent sensitivity", "Rest specificity", "AUROC")
    keys = ("intent_balanced_accuracy", "intent_sensitivity", "rest_specificity", "auroc")
    groups = (
        ("Zero-shot", report["zero_shot_evaluation"]["conservative"], BLUE),
        ("30-window adapted", report["adapted_evaluation"]["conservative"], GREEN),
    )
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    for index, (label, values, color) in enumerate(groups):
        heights = [100.0 * float(values[key]) for key in keys]
        bars = ax.bar(x + (index - 0.5) * width, heights, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10, fontweight="bold")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Locked external result (%)")
    ax.set_title("Cho2017 External V7 Evaluation", color=INK, fontweight="bold")
    ax.grid(axis="y", color=PALE)
    ax.legend(
        frameon=False,
        ncols=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_probability(
    path: Path,
    dataset: object,
    probability: np.ndarray,
    selected: np.ndarray,
    thresholds: Mapping[str, float],
) -> None:
    _style()
    labels = dataset.labels[selected] != 0
    values = probability[selected]
    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(values[~labels], bins=bins, density=True, alpha=0.58, color=BLUE, label="Continuous rest")
    ax.hist(values[labels], bins=bins, density=True, alpha=0.58, color=ORANGE, label="Motor imagery intent")
    ax.axvline(thresholds["conservative"], color=RED, linestyle="--", linewidth=2.5, label="Frozen conservative threshold")
    ax.axvline(thresholds["sensitivity_targeted"], color=GREEN, linestyle=":", linewidth=2.5, label="Frozen sensitivity-targeted threshold")
    ax.set_xlabel("Patient-adapted P(intent)")
    ax.set_ylabel("Density")
    ax.set_title("External Intent Evidence Distribution", color=INK, fontweight="bold")
    ax.grid(axis="y", color=PALE)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_subjects(path: Path, records: list[dict[str, object]]) -> None:
    _style()
    ordered = sorted(records, key=lambda value: float(value["intent_balanced_accuracy"]))
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(x, 100 * np.asarray([row["intent_sensitivity"] for row in ordered]), color=GREEN, marker="o", markersize=3, linewidth=1.7, label="Intent sensitivity")
    ax.plot(x, 100 * np.asarray([row["rest_specificity"] for row in ordered]), color=ORANGE, marker="s", markersize=3, linewidth=1.7, label="Rest specificity")
    ax.plot(x, 100 * np.asarray([row["intent_balanced_accuracy"] for row in ordered]), color=INK, linewidth=2.4, label="Balanced accuracy")
    ax.axhline(50, color="#788995", linestyle="--", linewidth=1.2, label="Chance BA")
    ax.set_xticks(x[::3], [row["subject"].split(":")[-1] for row in ordered][::3], rotation=45, ha="right")
    ax.set_ylim(0, 103)
    ax.set_ylabel("Result (%)")
    ax.set_xlabel("Participants ordered by balanced accuracy")
    ax.set_title("External Participant Variability", color=INK, fontweight="bold")
    ax.grid(axis="y", color=PALE)
    ax.legend(frameon=False, ncols=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(
    path: Path,
    dataset: object,
    zero_shot: np.ndarray,
    adapted: np.ndarray,
    selected: np.ndarray,
) -> None:
    _style()
    labels = (dataset.labels[selected] != 0).astype(np.int64)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for values, label, color in (
        (zero_shot[selected], "Zero-shot", BLUE),
        (adapted[selected], "30-window adapted", GREEN),
    ):
        false_positive, true_positive, _ = roc_curve(labels, values)
        area = roc_auc_score(labels, values)
        ax.plot(false_positive, true_positive, color=color, linewidth=2.5, label=f"{label} (AUROC {area:.3f})")
    ax.plot([0, 1], [0, 1], color="#8999A5", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Threshold-Independent External Discrimination", color=INK, fontweight="bold")
    ax.grid(color=PALE)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_predictions(
    path: Path,
    dataset: object,
    zero_shot: np.ndarray,
    adapted: np.ndarray,
    available: np.ndarray,
    adapted_mask: np.ndarray,
    thresholds: Mapping[str, float],
) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "subject", "trial_index", "calibration_row", "true_intent",
                "true_direction_numeric", "available", "zero_shot_probability",
                "adapted_probability", "adapted_evaluation_row",
                "conservative_prediction", "sensitivity_targeted_prediction",
            ]
        )
        for index in range(len(dataset.labels)):
            writer.writerow(
                [
                    dataset.subject_keys[index], int(dataset.trial_indices[index]),
                    int(dataset.recordings[index] == 0), int(dataset.labels[index] != 0),
                    int(dataset.labels[index]), int(available[index]),
                    float(zero_shot[index]), float(adapted[index]), int(adapted_mask[index]),
                    int(available[index] and adapted[index] >= thresholds["conservative"]),
                    int(available[index] and adapted[index] >= thresholds["sensitivity_targeted"]),
                ]
            )


def _report_markdown(report: Mapping[str, object]) -> str:
    primary = report["adapted_evaluation"]["conservative"]
    targeted = report["adapted_evaluation"]["sensitivity_targeted"]
    zero = report["zero_shot_evaluation"]["conservative"]
    ci = report["primary_participant_bootstrap_95ci"]
    return "\n".join(
        [
            "# Cho2017 Locked External V7 Evaluation",
            "",
            "> V7 was frozen before EEG inference. Cho2017 contributed no rows to",
            "> training, architecture selection, probability calibration, or threshold selection.",
            "",
            "## Cohort",
            "",
            f"- Healthy participants: **{report['cohort']['subjects']}**",
            "- Adapter-development participant excluded from inference: **S001**",
            f"- External EEG windows: **{report['cohort']['rows']:,}**",
            f"- Adapted evaluation windows: **{report['cohort']['adapted_evaluation_rows']:,}**",
            "- Source EEG: 64 native channels at 512 Hz; V7 uses seven shared motor channels at 100 Hz.",
            "- Intent: 0.5-2.5 s after the visual imagery cue.",
            "- Rest: independent continuous resting-state data, divided into 2-second windows with a 1-second step.",
            "",
            "## Frozen Results",
            "",
            "| Configuration | BA | Sensitivity | Specificity | False activation/rest window | AUROC | Abstained |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Zero-shot conservative | {zero['intent_balanced_accuracy']:.1%} | {zero['intent_sensitivity']:.1%} | {zero['rest_specificity']:.1%} | {zero['false_activation_fraction']:.1%} | {zero['auroc']:.3f} | {zero['abstained_fraction']:.1%} |",
            f"| 30-window adapted conservative | {primary['intent_balanced_accuracy']:.1%} | {primary['intent_sensitivity']:.1%} | {primary['rest_specificity']:.1%} | {primary['false_activation_fraction']:.1%} | {primary['auroc']:.3f} | {primary['abstained_fraction']:.1%} |",
            f"| 30-window adapted sensitivity-targeted | {targeted['intent_balanced_accuracy']:.1%} | {targeted['intent_sensitivity']:.1%} | {targeted['rest_specificity']:.1%} | {targeted['false_activation_fraction']:.1%} | {targeted['auroc']:.3f} | {targeted['abstained_fraction']:.1%} |",
            "",
            "Primary participant-bootstrap 95% confidence intervals:",
            f"BA **{ci['intent_balanced_accuracy']['low']:.1%}-{ci['intent_balanced_accuracy']['high']:.1%}**, "
            f"sensitivity **{ci['intent_sensitivity']['low']:.1%}-{ci['intent_sensitivity']['high']:.1%}**, "
            f"specificity **{ci['rest_specificity']['low']:.1%}-{ci['rest_specificity']['high']:.1%}**.",
            "",
            "## Interpretation Boundary",
            "",
            "- The zero-shot result is a fully external population-model test.",
            "- The adapted result uses a frozen 30-window stratified calibration subset from the same session; it is external few-shot adaptation, not prospective later-session validation.",
            "- The negative class is genuine continuous rest, but commands/minute remain unmeasured because temporal confirmation is not applied here.",
            "- Left/right labels are used only as internal calibration evidence; device-facing predictions remain REST or INTENT.",
            "- Retuning V7 from these results would make Cho2017 development data for any later experiment.",
            "",
            "## Outputs",
            "",
            "- `01_external_summary.png`",
            "- `02_adapted_probability_distribution.png`",
            "- `03_participant_variability.png`",
            "- `04_external_roc.png`",
            "- `cho2017_v7_locked_predictions.csv`",
            "- `cho2017_v7_locked_metrics.json`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--prepare-contract", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--execute-locked-evaluation", default="")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=TORCH_DEVICE_CHOICES, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    reference_path = args.reference_cache.expanduser().resolve()
    full_snapshot = cho2017_snapshot(dataset_root)
    snapshot = cho2017_snapshot(dataset_root, CHO_LOCKED_SUBJECTS)
    print(
        f"[+] Cho2017 inventory: {full_snapshot['subjects']} source subjects; "
        f"{snapshot['subjects']} locked subjects; "
        f"{sum(snapshot['event_totals'].values()):,} locked imagery events; "
        f"{snapshot['source_mat_bytes'] / 1024**3:.1f} GiB locked source MAT"
    )
    if args.inventory_only:
        print("[+] Inventory-only verification completed; no EEG arrays were opened.")
        return
    if args.prepare_contract:
        contract = _prepare_contract(
            dataset_root,
            contract_path,
            model_path,
            reference_path,
        )
        print(f"[+] Frozen pre-inference contract: {contract_path}")
        print(
            "[+] Conservative/sensitivity-targeted thresholds: "
            f"{contract['protocol']['thresholds']['conservative']:.3f}/"
            f"{contract['protocol']['thresholds']['sensitivity_targeted']:.3f}"
        )
        return
    if args.execute_locked_evaluation != CONFIRMATION:
        raise SystemExit(
            "Locked inference was not started. First run --prepare-contract, review "
            f"the contract, then pass --execute-locked-evaluation {CONFIRMATION}."
        )

    metrics_path = output / "cho2017_v7_locked_metrics.json"
    if metrics_path.exists():
        raise RuntimeError(
            f"A locked result already exists at {metrics_path}; refusing to overwrite it."
        )
    contract, bundle, neural_files = _verify_contract(
        dataset_root,
        contract_path,
        model_path,
        reference_path,
    )
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    receipt = {
        "status": "locked_inference_started",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "confirmation_token": CONFIRMATION,
        "started_unix_seconds": started,
    }
    (output / "locked_execution_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="ascii",
    )

    reference = load_translational_dataset(
        reference_path,
        bundle["dataset_fingerprint"],
    )
    if reference is None:
        raise RuntimeError("Frozen V7 reference cache failed fingerprint validation.")
    training_mask = _development_mask(reference, bundle["subject_roles"])
    dataset = build_cho2017_window_dataset(
        root=dataset_root,
        cache_path=output / "cho2017_v7_windows.npz",
        profile=bundle["profile"],
        subjects=CHO_LOCKED_SUBJECTS,
        rebuild_cache=args.rebuild_cache,
    )
    features = build_external_translational_features(
        dataset,
        reference,
        training_mask,
    )
    device = resolve_torch_device(args.device)
    zero_shot, available, neural_metadata = _predict_population(
        dataset,
        features,
        bundle,
        neural_files,
        device,
    )
    adapted, adapted_mask = patient_adapted_auxiliary_intent_probabilities(
        features.branches,
        dataset.labels,
        dataset.dataset_ids,
        dataset.subject_keys,
        dataset.recordings,
        np.ones(len(dataset.labels), dtype=bool),
        zero_shot,
        branch_names=("mrcp", "smr", "riemann"),
        budget_per_source_class=CHO_CALIBRATION_PER_CLASS,
        blend_weight=0.5,
        auxiliary_weight=0.25,
        seed=args.seed,
    )
    thresholds = {
        "conservative": float(contract["protocol"]["thresholds"]["conservative"]),
        "sensitivity_targeted": float(
            contract["protocol"]["thresholds"]["sensitivity_targeted"]
        ),
    }
    evaluation_mask = dataset.recordings != 0
    zero_shot_all = {
        name: _score(dataset.labels, zero_shot, available, threshold)
        for name, threshold in thresholds.items()
    }
    zero_shot_evaluation = {
        name: _score(
            dataset.labels,
            zero_shot,
            available,
            threshold,
            evaluation_mask,
        )
        for name, threshold in thresholds.items()
    }
    adapted_evaluation = {
        name: _score(
            dataset.labels,
            adapted,
            available,
            threshold,
            adapted_mask,
        )
        for name, threshold in thresholds.items()
    }
    primary_ci = _bootstrap_subjects(
        dataset,
        adapted,
        available,
        thresholds["conservative"],
        adapted_mask,
        args.bootstrap_iterations,
        args.seed,
    )
    participant_records = _subject_scores(
        dataset,
        adapted,
        available,
        thresholds["conservative"],
        adapted_mask,
    )
    adapted_subjects = set(dataset.subject_keys[adapted_mask])
    report = {
        "experiment": "cho2017_locked_external_v7",
        "status": "locked_external_completed",
        "contract_sha256": _sha256(contract_path),
        "cohort": {
            "subjects": int(len(np.unique(dataset.subject_keys))),
            "adapted_subjects": int(len(adapted_subjects)),
            "rows": int(len(dataset.labels)),
            "intent_rows": int(np.sum(dataset.labels != 0)),
            "rest_rows": int(np.sum(dataset.labels == 0)),
            "calibration_rows": int(np.sum(dataset.recordings == 0)),
            "adapted_evaluation_rows": int(np.sum(adapted_mask)),
        },
        "thresholds": thresholds,
        "zero_shot_all": zero_shot_all,
        "zero_shot_evaluation": zero_shot_evaluation,
        "adapted_evaluation": adapted_evaluation,
        "primary_participant_bootstrap_95ci": primary_ci,
        "participant_results": participant_records,
        "quality_control": {
            "recordings": list(dataset.recording_qc),
            "available_fraction": float(np.mean(available)),
        },
        "neural_inference": neural_metadata,
        "compute": torch_device_metadata(args.device, device),
        "runtime_minutes": (time.time() - started) / 60.0,
        "interpretation": {
            "zero_shot_is_fully_external": True,
            "adaptation_is_within_session": True,
            "prospective_later_session": False,
            "continuous_false_activations_per_minute_measured": False,
        },
    }
    metrics_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="ascii",
    )
    _save_predictions(
        output / "cho2017_v7_locked_predictions.csv",
        dataset,
        zero_shot,
        adapted,
        available,
        adapted_mask,
        thresholds,
    )
    _plot_summary(output / "01_external_summary.png", report)
    _plot_probability(
        output / "02_adapted_probability_distribution.png",
        dataset,
        adapted,
        adapted_mask,
        thresholds,
    )
    _plot_subjects(output / "03_participant_variability.png", participant_records)
    _plot_roc(
        output / "04_external_roc.png",
        dataset,
        zero_shot,
        adapted,
        adapted_mask,
    )
    report_path = output / "CHO2017_V7_LOCKED_EXTERNAL_REPORT.md"
    report_path.write_text(_report_markdown(report), encoding="ascii")

    primary = adapted_evaluation["conservative"]
    targeted = adapted_evaluation["sensitivity_targeted"]
    print("\n=== Cho2017 Locked External V7 Result ===")
    print(
        "Adapted conservative | "
        f"BA={primary['intent_balanced_accuracy']:.1%} | "
        f"sensitivity={primary['intent_sensitivity']:.1%} | "
        f"specificity={primary['rest_specificity']:.1%} | "
        f"AUROC={primary['auroc']:.3f}"
    )
    print(
        "Adapted sensitivity-targeted | "
        f"BA={targeted['intent_balanced_accuracy']:.1%} | "
        f"sensitivity={targeted['intent_sensitivity']:.1%} | "
        f"specificity={targeted['rest_specificity']:.1%}"
    )
    print(f"[+] Saved locked report and figures to {output}")


if __name__ == "__main__":
    main()
