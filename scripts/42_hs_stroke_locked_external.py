#!/usr/bin/env python3
"""Run the frozen v4 decoder once on the locked HS-Stroke v6 cohort."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from bci_pipeline.hs_stroke import (  # noqa: E402
    HS_DATASET_ID,
    HS_SUBJECTS,
    build_hs_locked_window_dataset,
    discover_hs_source_records,
)
from bci_pipeline.neural_integration import predict_mrcp_fbcnet_files  # noqa: E402
from bci_pipeline.openbmi_controlled import StageProbabilities  # noqa: E402
from bci_pipeline.translational_dataset import (  # noqa: E402
    load_translational_dataset,
    summarize_recording_qc,
)
from bci_pipeline.translational_features import (  # noqa: E402
    build_external_translational_features,
)


CONFIRMATION = "CONFIRM_HS_STROKE_V6_LOCKED_RUN"
DEFAULT_DATASET = PROJECT_ROOT / "30747806"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "hs_stroke_locked_external"
DEFAULT_CONTRACT = (
    DEFAULT_DATASET / "EEG_BCI_Project_HS_STROKE_LOCKED_CONTRACT.json"
)
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "prospective_cue_controller" / "frozen_model"
DEFAULT_MODEL = DEFAULT_MODEL_DIR / "translational_v4_neural_models.joblib"
DEFAULT_MODEL_MANIFEST = DEFAULT_MODEL_DIR / "frozen_model_manifest.json"
SOURCE_ARCHIVE_SHA256 = "bcfebde5f177db510c2dbc854f122d685e84a6f19f343a124ad0cdb7fd21e07e"
BLUE = "#176B87"
ORANGE = "#E07A2F"
GREEN = "#3A8D6D"
RED = "#C64B4B"
INK = "#17324D"
PALE = "#EAF1F5"


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


def _reference_cache() -> Path:
    candidates = (
        PROJECT_ROOT / "artifacts" / "gui_experiments" / "translational_v3" / "translational_windows_v3.npz",
        PROJECT_ROOT.parent / "artifacts" / "gui_experiments" / "translational_v3" / "translational_windows_v3.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("The exact frozen v4 translational window cache is unavailable.")


def _verify_lock(
    dataset_root: Path,
    contract_path: Path,
    model_path: Path,
    model_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if contract["status"] != "frozen_before_eeg_inference":
        raise RuntimeError("HS-Stroke contract is not in the frozen pre-inference state.")
    if contract["dataset"]["source_archive_sha256"] != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("Contract source archive hash changed.")
    source_archive = dataset_root / "sourcedata.zip"
    if not source_archive.is_file() or _sha256(source_archive) != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("HS-Stroke source archive does not match corrected v6.")
    actual_model_hash = _sha256(model_path)
    if actual_model_hash != manifest["model_sha256"]:
        raise RuntimeError("Frozen model hash verification failed.")
    for record in manifest["neural_models"]:
        path = model_path.parent / Path(str(record["path"])).name
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen neural model hash verification failed: {path}")
    if manifest["dataset_fingerprint"] != contract["frozen_decoder"]["training_dataset_fingerprint"]:
        raise RuntimeError("Frozen training fingerprint differs from the external contract.")
    return contract, manifest


def _inventory(dataset_root: Path) -> dict[str, object]:
    groups = discover_hs_source_records(dataset_root)
    sessions = {subject: 0 for subject in HS_SUBJECTS}
    runs = {subject: 0 for subject in HS_SUBJECTS}
    for (subject, _), paths in groups.items():
        sessions[subject] += 1
        runs[subject] += len(paths)
    return {
        "subjects": len(sessions),
        "sessions": int(sum(sessions.values())),
        "runs": int(sum(runs.values())),
        "sessions_by_subject": sessions,
        "runs_by_subject": runs,
    }


def _development_mask(reference: object, roles: dict[str, object]) -> np.ndarray:
    subjects = [
        subject
        for dataset_roles in roles.values()
        for subject in dataset_roles["development"]
    ]
    mask = np.isin(reference.subject_keys, subjects)
    if not np.any(mask):
        raise RuntimeError("Frozen development subjects do not match the reference cache.")
    return mask


def _predict(
    dataset: object,
    features: object,
    bundle: dict[str, object],
    neural_files: list[Path],
) -> tuple[StageProbabilities, np.ndarray, dict[str, object]]:
    domains = dataset.dataset_ids
    probabilities: dict[str, StageProbabilities] = {}
    for branch in ("mrcp", "smr", "riemann"):
        probabilities[branch] = bundle["models"][branch].predict_stage_probabilities(
            {branch: features.branches[branch]},
            domains,
        )
    intent, direction = bundle["models"]["fbcsp"].predict_stage_probabilities(
        dataset.smr_epochs
    )
    probabilities["fbcsp"] = StageProbabilities(intent, direction)
    neural_intent, _, neural_metadata = predict_mrcp_fbcnet_files(
        dataset.mrcp_epochs,
        dataset.smr_epochs,
        dataset.channel_names,
        neural_files,
        batch_size=512,
        device=torch.device("cpu"),
    )
    probabilities["mrcp_fbcnet"] = StageProbabilities(
        neural_intent,
        direction.copy(),
    )
    fused = bundle["stage_specific_fusion"].predict(probabilities)
    calibrated = bundle["probability_calibrator"].predict(fused, domains)
    available = bundle["quality_gate"].available(
        features.branches["quality"],
        domains,
    )
    return calibrated, available, neural_metadata


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _score(
    labels: np.ndarray,
    probabilities: StageProbabilities,
    available: np.ndarray,
    threshold: float,
    selected: np.ndarray | None = None,
) -> dict[str, float | int]:
    mask = np.ones(len(labels), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    y = labels[mask]
    local_available = available[mask]
    intent_probability = probabilities.intent[mask]
    direction_probability = probabilities.direction[mask]
    true_intent = y != 0
    issued = local_available & (intent_probability >= threshold)
    direction_head = np.argmax(direction_probability, axis=1) + 1
    prediction = np.zeros(len(y), dtype=np.int64)
    prediction[issued] = direction_head[issued]
    rest = ~true_intent
    issued_true = issued & true_intent
    available_true = local_available & true_intent
    direction_available = (
        balanced_accuracy_score(y[available_true], direction_head[available_true])
        if set(np.unique(y[available_true])) == {1, 2}
        else float("nan")
    )
    direction_issued = (
        balanced_accuracy_score(y[issued_true], direction_head[issued_true])
        if set(np.unique(y[issued_true])) == {1, 2}
        else _safe_ratio(np.sum(y[issued_true] == direction_head[issued_true]), np.sum(issued_true))
    )
    sensitivity = _safe_ratio(np.sum(issued_true), np.sum(true_intent))
    specificity = _safe_ratio(np.sum(rest & ~issued), np.sum(rest))
    return {
        "rows": int(len(y)),
        "intent_rows": int(np.sum(true_intent)),
        "rest_rows": int(np.sum(rest)),
        "available_fraction": float(np.mean(local_available)),
        "intent_sensitivity": sensitivity,
        "preparation_specificity": specificity,
        "intent_balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "false_activation_fraction": _safe_ratio(np.sum(rest & issued), np.sum(rest)),
        "missed_intent_fraction": 1.0 - sensitivity,
        "numeric_direction_balanced_accuracy_available": float(direction_available),
        "numeric_direction_balanced_accuracy_issued": float(direction_issued),
        "wrong_direction_command_fraction": _safe_ratio(
            np.sum(issued_true & (direction_head != y)),
            np.sum(issued_true),
        ),
        "three_class_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "issued_command_fraction": float(np.mean(issued)),
        "issued_commands": int(np.sum(issued)),
    }


def _bootstrap_subjects(
    dataset: object,
    probabilities: StageProbabilities,
    available: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    subjects = np.unique(dataset.subject_keys)
    rng = np.random.default_rng(seed)
    metric_names = (
        "intent_sensitivity",
        "preparation_specificity",
        "intent_balanced_accuracy",
        "false_activation_fraction",
        "numeric_direction_balanced_accuracy_available",
        "three_class_balanced_accuracy",
    )
    samples = {name: [] for name in metric_names}
    subject_indices = {
        subject: np.flatnonzero(dataset.subject_keys == subject)
        for subject in subjects
    }
    for _ in range(iterations):
        selected_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([subject_indices[subject] for subject in selected_subjects])
        local = _score(
            dataset.labels[indices],
            StageProbabilities(probabilities.intent[indices], probabilities.direction[indices]),
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


def _group_scores(
    dataset: object,
    probabilities: StageProbabilities,
    available: np.ndarray,
    threshold: float,
    groups: np.ndarray,
) -> list[dict[str, object]]:
    records = []
    for value in np.unique(groups):
        selected = groups == value
        records.append({"group": str(value), **_score(dataset.labels, probabilities, available, threshold, selected)})
    return records


def _prediction_array(
    probabilities: StageProbabilities,
    available: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    issued = available & (probabilities.intent >= threshold)
    prediction = np.zeros(len(issued), dtype=np.int64)
    prediction[issued] = np.argmax(probabilities.direction[issued], axis=1) + 1
    return prediction, issued


def _save_predictions(
    path: Path,
    dataset: object,
    probabilities: StageProbabilities,
    available: np.ndarray,
    thresholds: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: _prediction_array(probabilities, available, threshold)
        for name, threshold in thresholds.items()
    }
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "subject", "session", "run", "trial_index", "true_numeric_label",
                "negative_kind", "available", "intent_probability", "class1_probability",
                "class2_probability", "conservative_prediction", "conservative_issued",
                "sensitivity_targeted_prediction", "sensitivity_targeted_issued",
            ]
        )
        for index in range(len(dataset.labels)):
            writer.writerow(
                [
                    dataset.subject_keys[index], int(dataset.sessions[index]),
                    int(dataset.recordings[index]), int(dataset.trial_indices[index]),
                    int(dataset.labels[index]), dataset.negative_kinds[index],
                    int(available[index]), float(probabilities.intent[index]),
                    float(probabilities.direction[index, 0]),
                    float(probabilities.direction[index, 1]),
                    int(outputs["conservative"][0][index]),
                    int(outputs["conservative"][1][index]),
                    int(outputs["sensitivity_targeted"][0][index]),
                    int(outputs["sensitivity_targeted"][1][index]),
                ]
            )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#B6C4CC",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def _plot_summary(path: Path, report: dict[str, object]) -> None:
    _style()
    names = ("Sensitivity", "Specificity", "Intent BA", "Direction BA", "Availability")
    keys = (
        "intent_sensitivity", "preparation_specificity", "intent_balanced_accuracy",
        "numeric_direction_balanced_accuracy_available", "available_fraction",
    )
    x = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for offset, (operating, color, label) in enumerate(
        (("conservative", BLUE, "Conservative"), ("sensitivity_targeted", ORANGE, "Sensitivity-targeted"))
    ):
        values = [100.0 * report["operating_points"][operating][key] for key in keys]
        bars = ax.bar(x + (offset - 0.5) * width, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("Locked HS-Stroke External Evaluation")
    ax.grid(axis="y", color=PALE, linewidth=1)
    ax.legend(frameon=False, ncols=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_patients(path: Path, patient_records: list[dict[str, object]]) -> None:
    _style()
    subjects = [record["group"].split(":")[-1] for record in patient_records]
    sensitivity = 100 * np.asarray([record["intent_sensitivity"] for record in patient_records])
    specificity = 100 * np.asarray([record["preparation_specificity"] for record in patient_records])
    direction = 100 * np.asarray([record["numeric_direction_balanced_accuracy_available"] for record in patient_records])
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(x, sensitivity, marker="o", color=ORANGE, linewidth=2, label="Intent sensitivity")
    ax.plot(x, specificity, marker="s", color=BLUE, linewidth=2, label="Preparation specificity")
    ax.plot(x, direction, marker="^", color=GREEN, linewidth=2, label="Numeric direction BA")
    ax.axhline(50, color="#8999A5", linestyle="--", linewidth=1, label="Chance reference")
    ax.set_xticks(x, subjects, rotation=45, ha="right")
    ax.set_ylim(0, 103)
    ax.set_ylabel("Percent")
    ax.set_title("Patient-Level Variability at the Conservative Operating Point")
    ax.grid(axis="y", color=PALE)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_confusions(
    path: Path,
    labels: np.ndarray,
    probabilities: StageProbabilities,
    available: np.ndarray,
    thresholds: dict[str, float],
) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (name, threshold) in zip(axes, thresholds.items()):
        prediction, _ = _prediction_array(probabilities, available, threshold)
        matrix = confusion_matrix(labels, prediction, labels=[0, 1, 2])
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(3):
            for column in range(3):
                color = "white" if normalized[row, column] > 0.55 else INK
                ax.text(column, row, f"{matrix[row, column]:,}\n{normalized[row, column]:.1%}", ha="center", va="center", color=color, fontsize=9)
        ax.set_xticks(range(3), ("Rest", "Class 1", "Class 2"), rotation=25, ha="right")
        ax.set_yticks(range(3), ("Rest", "Class 1", "Class 2"))
        ax.set_xlabel("Issued prediction")
        ax.set_ylabel("True numeric class")
        ax.set_title(name.replace("_", " ").title())
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.04, label="Row-normalized fraction")
    fig.suptitle("Locked External Confusion Matrices", fontsize=14, color=INK)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.20, top=0.82, wspace=0.28)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_probability(path: Path, labels: np.ndarray, probability: np.ndarray, thresholds: dict[str, float]) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    bins = np.linspace(0, 1, 41)
    ax.hist(probability[labels == 0], bins=bins, density=True, alpha=0.55, color=BLUE, label="Preparation rest")
    ax.hist(probability[labels != 0], bins=bins, density=True, alpha=0.55, color=ORANGE, label="Motor imagery")
    ax.axvline(thresholds["conservative"], color=BLUE, linestyle="--", linewidth=2, label="Conservative threshold")
    ax.axvline(thresholds["sensitivity_targeted"], color=ORANGE, linestyle=":", linewidth=2, label="Sensitivity-targeted threshold")
    ax.set_xlabel("Frozen calibrated intent probability")
    ax.set_ylabel("Density")
    ax.set_title("Intent Evidence Distribution on the Locked Cohort")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=PALE)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _report_markdown(report: dict[str, object]) -> str:
    lines = [
        "# HS-Stroke Locked External Evaluation",
        "",
        "> This is the first evaluation of the frozen translational v4 decoder on the",
        "> corrected HS-Stroke v6 chronic-stroke cohort. No HS-Stroke rows were used",
        "> to fit preprocessing, features, models, calibration, thresholds, or timing.",
        "",
        "## Cohort",
        "",
        f"- Patients: **{report['cohort']['subjects']}**",
        f"- Sessions: **{report['cohort']['sessions']}**",
        f"- Trials represented: **{report['cohort']['intent_windows']:,}**",
        f"- Evaluation windows: **{report['cohort']['rows']:,}**",
        f"- Automatically excluded sessions: **{report['quality_control']['excluded_recordings_or_subjects']}**",
        "- Preparation window: 6.5-8.5 seconds",
        "- Imagery window: 9.5-11.5 seconds",
        "- Feedback: excluded",
        "",
        "## Frozen Results",
        "",
        "| Operating point | Sensitivity | Specificity | Intent BA | False activation/window | Missed intent | Direction BA* | Availability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("conservative", "Conservative"), ("sensitivity_targeted", "Sensitivity-targeted")):
        values = report["operating_points"][key]
        lines.append(
            f"| {label} | {values['intent_sensitivity']:.1%} | "
            f"{values['preparation_specificity']:.1%} | "
            f"{values['intent_balanced_accuracy']:.1%} | "
            f"{values['false_activation_fraction']:.1%} | "
            f"{values['missed_intent_fraction']:.1%} | "
            f"{values['numeric_direction_balanced_accuracy_available']:.1%} | "
            f"{values['available_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "*Direction is scored against numeric MAT classes 1 and 2.* The public",
            "release does not explicitly document which numeric value means left versus",
            "right, so semantic direction names remain provisional. Intent metrics are",
            "unaffected by that mapping.",
            "",
            "## Interpretation Limits",
            "",
            "- The independent sample size is 14 patients; sessions and trials are repeated observations.",
            "- Preparation is structured cue-locked rest, not natural idle, so the false-activation fraction cannot be reported as commands per minute.",
            "- Quality abstentions count as missed intent but as no issued command during preparation.",
            "- These results must not be used to retune the frozen model and then re-labelled confirmatory.",
            "- Patient-specific calibration is a separate secondary experiment using early sessions only, with later sessions re-locked.",
            "",
            "## Figures",
            "",
            "- `01_locked_external_summary.png`",
            "- `02_patient_variability.png`",
            "- `03_locked_confusion_matrices.png`",
            "- `04_intent_probability_distribution.png`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--execute-locked-evaluation", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    contract_path = args.contract.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    model_manifest_path = args.model_manifest.expanduser().resolve()
    contract, model_manifest = _verify_lock(
        dataset_root,
        contract_path,
        model_path,
        model_manifest_path,
    )
    inventory = _inventory(dataset_root)
    print(
        f"[+] HS-Stroke v6 inventory: {inventory['subjects']} subjects, "
        f"{inventory['sessions']} sessions, {inventory['runs']} run files"
    )
    if args.inventory_only:
        print("[+] Inventory-only verification completed; no EEG arrays were opened.")
        return
    if args.execute_locked_evaluation != CONFIRMATION:
        raise SystemExit(
            "Locked inference was not started. Pass "
            f"--execute-locked-evaluation {CONFIRMATION} after reviewing the contract."
        )
    metrics_path = output / "hs_stroke_locked_external_metrics.json"
    if metrics_path.exists():
        raise RuntimeError(
            f"A final locked result already exists at {metrics_path}; refusing to overwrite it."
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    execution = {
        "status": "locked_inference_started",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "model_sha256": _sha256(model_path),
        "source_archive_sha256": _sha256(dataset_root / "sourcedata.zip"),
        "inventory": inventory,
        "confirmation_token": CONFIRMATION,
    }
    (output / "locked_execution_receipt.json").write_text(
        json.dumps(execution, indent=2) + "\n",
        encoding="ascii",
    )

    bundle = joblib.load(model_path)
    if bundle["dataset_fingerprint"] != model_manifest["dataset_fingerprint"]:
        raise RuntimeError("Loaded model bundle fingerprint changed.")
    reference_path = _reference_cache()
    reference = load_translational_dataset(reference_path, bundle["dataset_fingerprint"])
    if reference is None:
        raise RuntimeError("Frozen reference cache failed fingerprint validation.")
    reference_training = _development_mask(reference, bundle["subject_roles"])
    cache_path = output / "hs_stroke_locked_windows.npz"
    dataset = build_hs_locked_window_dataset(
        root=dataset_root,
        cache_path=cache_path,
        profile=bundle["profile"],
    )
    features = build_external_translational_features(
        dataset,
        reference,
        reference_training,
    )
    neural_files = [
        model_path.parent / Path(str(record["path"])).name
        for record in model_manifest["neural_models"]
    ]
    probabilities, available, neural_metadata = _predict(
        dataset,
        features,
        bundle,
        neural_files,
    )
    thresholds = {
        "conservative": float(bundle["calibrated_thresholds"]["conservative_fallback"]),
        "sensitivity_targeted": float(bundle["calibrated_thresholds"]["sensitivity_targeted_fallback"]),
    }
    operating_points = {
        name: _score(dataset.labels, probabilities, available, threshold)
        for name, threshold in thresholds.items()
    }
    bootstrap = {
        name: _bootstrap_subjects(
            dataset,
            probabilities,
            available,
            threshold,
            iterations=int(contract["inference_rules"]["bootstrap_iterations"]),
            seed=4200 + offset,
        )
        for offset, (name, threshold) in enumerate(thresholds.items())
    }
    patient_records = _group_scores(
        dataset,
        probabilities,
        available,
        thresholds["conservative"],
        dataset.subject_keys,
    )
    session_groups = np.char.add(
        np.char.add(dataset.subject_keys.astype(str), ":ses-"),
        np.char.zfill(dataset.sessions.astype(str), 2),
    )
    session_records = _group_scores(
        dataset,
        probabilities,
        available,
        thresholds["conservative"],
        session_groups,
    )
    qc = summarize_recording_qc(dataset.recording_qc)
    report = {
        "schema_version": 1,
        "experiment": "hs_stroke_v6_locked_external_frozen_v4",
        "claim_role": "locked_external",
        "contract_sha256": _sha256(contract_path),
        "model_sha256": _sha256(model_path),
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "training_dataset_fingerprint": bundle["dataset_fingerprint"],
        "external_dataset_fingerprint": dataset.fingerprint,
        "cohort": {
            "subjects": int(len(np.unique(dataset.subject_keys))),
            "sessions": int(len(np.unique(session_groups))),
            "rows": int(len(dataset.labels)),
            "rest_windows": int(np.sum(dataset.labels == 0)),
            "intent_windows": int(np.sum(dataset.labels != 0)),
            "numeric_class1_windows": int(np.sum(dataset.labels == 1)),
            "numeric_class2_windows": int(np.sum(dataset.labels == 2)),
        },
        "preprocessing": bundle["profile"].fingerprint(),
        "feature_transform": features.metadata,
        "quality_control": qc,
        "thresholds": thresholds,
        "operating_points": operating_points,
        "subject_clustered_95_percent_confidence_intervals": bootstrap,
        "intent_auroc_descriptive_not_for_selection": float(
            roc_auc_score(dataset.labels != 0, probabilities.intent)
        ),
        "patient_metrics_conservative": patient_records,
        "session_metrics_conservative": session_records,
        "neural_ensemble": neural_metadata,
        "direction_semantics": "numeric classes 1/2; left/right mapping provisional",
        "continuous_no_control": "not_available_structured_preparation_only",
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    metrics_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="ascii",
    )
    _save_predictions(
        output / "hs_stroke_locked_predictions.csv",
        dataset,
        probabilities,
        available,
        thresholds,
    )
    _plot_summary(output / "01_locked_external_summary.png", report)
    _plot_patients(output / "02_patient_variability.png", patient_records)
    _plot_confusions(
        output / "03_locked_confusion_matrices.png",
        dataset.labels,
        probabilities,
        available,
        thresholds,
    )
    _plot_probability(
        output / "04_intent_probability_distribution.png",
        dataset.labels,
        probabilities.intent,
        thresholds,
    )
    (output / "HS_STROKE_LOCKED_EXTERNAL_REPORT.md").write_text(
        _report_markdown(report),
        encoding="ascii",
    )
    execution["status"] = "locked_inference_complete"
    execution["metrics"] = str(metrics_path)
    execution["metrics_sha256"] = _sha256(metrics_path)
    execution["runtime_minutes"] = report["runtime_minutes"]
    (output / "locked_execution_receipt.json").write_text(
        json.dumps(execution, indent=2, default=_json_default) + "\n",
        encoding="ascii",
    )
    conservative = operating_points["conservative"]
    targeted = operating_points["sensitivity_targeted"]
    print("\n=== HS-Stroke v6 Locked External Result ===")
    print(
        "Conservative | "
        f"BA={conservative['intent_balanced_accuracy']:.1%} | "
        f"sensitivity={conservative['intent_sensitivity']:.1%} | "
        f"specificity={conservative['preparation_specificity']:.1%} | "
        f"false activation/window={conservative['false_activation_fraction']:.1%}"
    )
    print(
        "Sensitivity-targeted | "
        f"BA={targeted['intent_balanced_accuracy']:.1%} | "
        f"sensitivity={targeted['intent_sensitivity']:.1%} | "
        f"specificity={targeted['preparation_specificity']:.1%} | "
        f"false activation/window={targeted['false_activation_fraction']:.1%}"
    )
    print(f"[+] Locked report: {output / 'HS_STROKE_LOCKED_EXTERNAL_REPORT.md'}")


if __name__ == "__main__":
    main()
