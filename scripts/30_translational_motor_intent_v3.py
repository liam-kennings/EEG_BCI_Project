#!/usr/bin/env python3
"""Run the quality-aware, domain-aware translational motor-intent v3 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.models.fbcsp import HierarchicalFBCSP  # noqa: E402
from bci_pipeline.neural_integration import (  # noqa: E402
    DEFAULT_NEURAL_SEEDS,
    predict_bakeoff_ensemble,
)
from bci_pipeline.openbmi_controlled import StageProbabilities  # noqa: E402
from bci_pipeline.quality_preprocessing import (  # noqa: E402
    BranchPreprocessingProfile,
)
from bci_pipeline.research_contract import (  # noqa: E402
    CONTROLLER_ROLE,
    DEVELOPMENT_OBSERVED_ROLE,
    DEVELOPMENT_ROLE,
    LOCKED_EXTERNAL_ROLE,
    validate_split_roles,
    write_research_contract,
)
from bci_pipeline.translational_dataset import (  # noqa: E402
    TranslationalWindowDataset,
    build_translational_window_dataset,
    summarize_recording_qc,
)
from bci_pipeline.translational_decoder import (  # noqa: E402
    AuxiliaryIntentBranchEnsemble,
    BranchEnsemble,
    DomainAwareShrinkageLDAIntentDecoder,
    DomainIntentProbabilityCalibrator,
    DomainProbabilityCalibrator,
    IntentBranchEnsemble,
    IntentSpecificFusion,
    QualityAvailabilityGate,
    StageSpecificFusion,
    patient_adapted_auxiliary_intent_probabilities,
    patient_adapted_intent_probabilities,
    patient_adapted_probabilities,
    population_intent_features,
    population_stage_features,
)
from bci_pipeline.translational_features import (  # noqa: E402
    SMR_FILTER_BANK,
    TranslationalFeatures,
    build_translational_features,
)
from bci_pipeline.translational_metrics import (  # noqa: E402
    continuous_false_activations,
    domain_threshold_vector,
    score_intent_operating_point,
    score_operating_point,
    select_continuous_threshold,
    select_sensitivity_threshold,
    select_window_threshold,
    subject_bootstrap_confidence_intervals,
    subject_bootstrap_intent_confidence_intervals,
)
from bci_pipeline.torch_device import TORCH_DEVICE_CHOICES  # noqa: E402


COLORS = {
    "navy": "#17324D",
    "blue": "#2878A6",
    "teal": "#21867A",
    "gold": "#E5A23C",
    "orange": "#D56A3A",
    "red": "#B94B4B",
    "gray": "#66737E",
    "pale": "#EEF2F3",
}
CORE_DATASETS = ("physionet", "openbmi", "liu2024")
GRAPH_ADAPTER_MIN_CONTROLLER_GAIN = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physionet-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument(
        "--openbmi-dir",
        default="files/external/openbmi/nm000338",
        help="OpenBMI nm000338 BIDS root.",
    )
    parser.add_argument("--liu-dir", default="Liu acute-stroke dataset")
    parser.add_argument(
        "--output-dir",
        default="artifacts/translational_motor_intent_v3_calibrated",
    )
    parser.add_argument(
        "--cache-path",
        default="artifacts/translational_motor_intent_v3_calibrated/"
        "translational_windows_v3.npz",
    )
    parser.add_argument("--hard-negative-manifest")
    parser.add_argument("--max-subjects-per-dataset", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-window-false-activation",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-continuous-false-activations-per-minute",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--target-intent-sensitivity",
        type=float,
        default=0.60,
    )
    parser.add_argument("--temporal-confirmations", type=int, default=2)
    parser.add_argument(
        "--temporal-refractory-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--quality-rejection-quantile", type=float, default=0.99)
    parser.add_argument("--domain-weight", type=float, default=0.5)
    parser.add_argument(
        "--neural-bakeoff-dir",
        help="Optional completed EEG decoder bakeoff folder for v4 integration.",
    )
    parser.add_argument(
        "--neural-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_NEURAL_SEEDS),
    )
    parser.add_argument(
        "--neural-architectures",
        nargs="+",
        choices=("mrcp_fbcnet", "shallow_convnet"),
        default=("mrcp_fbcnet",),
        help="Completed bakeoff seed ensembles supplied to controller-only fusion.",
    )
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument(
        "--neural-device",
        choices=TORCH_DEVICE_CHOICES,
        default="auto",
        help="PyTorch backend for optional MRCP + FBCNet inference.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--disable-csd", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--intent-only",
        action="store_true",
        help="Run v6 binary rest-versus-intent without a direction decoder.",
    )
    parser.add_argument(
        "--auxiliary-direction",
        action="store_true",
        help=(
            "Run v7 binary output with rest/left/right used only as an "
            "auxiliary training signal."
        ),
    )
    parser.add_argument(
        "--disable-v7-graph-lda",
        action="store_true",
        help=(
            "Reproduce pre-extension v7 without fixed-electrode graph features "
            "or the weighted shrinkage-LDA candidate."
        ),
    )
    args = parser.parse_args()
    if args.auxiliary_direction:
        args.intent_only = True
    args.v7_graph_lda = bool(
        args.auxiliary_direction and not args.disable_v7_graph_lda
    )
    return args


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#F8FAFA",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_general_model_bundle(
    output: Path,
    model_payload: dict[str, object],
    report: Mapping[str, object],
) -> dict[str, object]:
    bundle = output / "general_model_bundle"
    neural_dir = bundle / "neural_models"
    neural_dir.mkdir(parents=True, exist_ok=True)
    copied_models = []
    integration = report.get("neural_integration") or {}
    for architecture, metadata in integration.get("ensembles", {}).items():
        for record in metadata["models"]:
            source = Path(record["path"]).expanduser().resolve()
            destination = neural_dir / f"{architecture}_seed_{int(record['seed'])}.pt"
            shutil.copy2(source, destination)
            copied_models.append(
                {
                    "architecture": architecture,
                    "seed": int(record["seed"]),
                    "relative_path": str(destination.relative_to(bundle)),
                    "sha256": _sha256(destination),
                }
            )
    packaged_payload = dict(model_payload)
    packaged_payload["packaged_neural_models"] = copied_models
    controller_path = bundle / "general_population_controller.joblib"
    joblib.dump(packaged_payload, controller_path)
    patient = report["patient_calibration"]
    result = patient["development_observed_conservative"]
    intent_only = bool(report.get("intent_only", False))
    auxiliary_direction = bool(report.get("auxiliary_direction", False))
    manifest = {
        "schema_version": 1,
        "purpose": (
            "binary EEG intent model with auxiliary direction training"
            if auxiliary_direction
            else "binary rest-versus-intent EEG model with rapid patient calibration"
            if intent_only
            else "general EEG motor-intent model with rapid patient calibration"
        ),
        "claim_status": "development_observed_not_external_confirmation",
        "population_controller": controller_path.name,
        "population_controller_sha256": _sha256(controller_path),
        "neural_models": copied_models,
        "neural_architectures_evaluated": integration.get("architectures", []),
        "selected_intent_fusion": report["calibrated_stage_controller"]["fusion"][
            "intent_weights"
        ],
        "patient_calibration": {
            "protocol": patient["calibration_protocol"],
            "adapter_api": (
                "bci_pipeline.translational_decoder.RapidAuxiliaryIntentPatientAdapter"
                if auxiliary_direction
                else "bci_pipeline.translational_decoder.RapidIntentPatientAdapter"
                if intent_only
                else "bci_pipeline.translational_decoder.RapidPatientAdapter"
            ),
            "fit_contract": (
                "fit(branches, labels, domain, branch_names, blend_weight, auxiliary_weight, seed)"
                if auxiliary_direction
                else "fit(branches, labels, domain, branch_names, blend_weight, seed)"
            ),
            "predict_contract": "predict(branches, domain, population)",
            "adapter_inputs": patient["adapter_inputs"],
            "selected_by_dataset": patient["selected_configurations"],
            "labels_required": (
                ["rest", "left", "right"]
                if auxiliary_direction
                else ["rest", "intent"]
                if intent_only
                else ["rest", "left", "right"]
            ),
            "output_classes": ["rest", "intent"] if intent_only else ["rest", "left", "right"],
            "auxiliary_direction_output_exposed": (
                False if auxiliary_direction else None
            ),
            "calibration_rows_excluded_from_metrics": patient.get(
                "calibration_rows_excluded_from_metrics",
                False,
            ),
        },
        "development_observed_result": {
            "balanced_accuracy": result["intent_balanced_accuracy"],
            "sensitivity": result["intent_sensitivity"],
            "specificity": result["intent_specificity"],
        },
        "deployment_boundary": (
            "Temporal false activations per minute require continuous no-control "
            "validation before device commands are enabled."
        ),
    }
    if not intent_only:
        manifest["selected_direction_fusion"] = report[
            "calibrated_stage_controller"
        ]["fusion"]["direction_weights"]
        manifest["development_observed_result"][
            "conditional_direction_balanced_accuracy"
        ] = result["conditional_direction_balanced_accuracy"]
    manifest_path = bundle / "GENERAL_MODEL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    return {
        "directory": str(bundle),
        "controller": str(controller_path),
        "manifest": str(manifest_path),
        "neural_model_count": len(copied_models),
    }


def _three_way_keys(
    subjects: np.ndarray,
    seed: int,
) -> dict[str, list[str]]:
    unique = np.asarray(sorted(set(str(value) for value in subjects)))
    if len(unique) < 5:
        raise ValueError("At least five subjects are required for each split domain.")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    observed_count = max(1, int(round(0.20 * len(shuffled))))
    remaining = len(shuffled) - observed_count
    controller_count = max(1, int(round(0.20 * remaining)))
    observed = shuffled[:observed_count]
    controller = shuffled[observed_count : observed_count + controller_count]
    development = shuffled[observed_count + controller_count :]
    roles = {
        DEVELOPMENT_ROLE: sorted(development.tolist()),
        CONTROLLER_ROLE: sorted(controller.tolist()),
        DEVELOPMENT_OBSERVED_ROLE: sorted(observed.tolist()),
        LOCKED_EXTERNAL_ROLE: [],
    }
    validate_split_roles(roles)
    return roles


def _small_negative_only_roles(
    subjects: np.ndarray,
    seed: int,
) -> dict[str, list[str]]:
    """Keep small optional no-control sets usable without calling them final."""
    unique = np.asarray(sorted(set(str(value) for value in subjects)))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    roles = {
        DEVELOPMENT_ROLE: [],
        CONTROLLER_ROLE: [],
        DEVELOPMENT_OBSERVED_ROLE: [],
        LOCKED_EXTERNAL_ROLE: [],
    }
    if len(shuffled) == 1:
        roles[CONTROLLER_ROLE] = shuffled.tolist()
    elif len(shuffled) == 2:
        roles[CONTROLLER_ROLE] = shuffled[:1].tolist()
        roles[DEVELOPMENT_OBSERVED_ROLE] = shuffled[1:].tolist()
    else:
        roles[DEVELOPMENT_ROLE] = shuffled[:-2].tolist()
        roles[CONTROLLER_ROLE] = shuffled[-2:-1].tolist()
        roles[DEVELOPMENT_OBSERVED_ROLE] = shuffled[-1:].tolist()
    validate_split_roles(roles)
    return roles


def build_split_masks(
    dataset: TranslationalWindowDataset,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[str]]]]:
    masks = {
        DEVELOPMENT_ROLE: np.zeros(len(dataset.labels), dtype=bool),
        CONTROLLER_ROLE: np.zeros(len(dataset.labels), dtype=bool),
        DEVELOPMENT_OBSERVED_ROLE: np.zeros(len(dataset.labels), dtype=bool),
        LOCKED_EXTERNAL_ROLE: np.zeros(len(dataset.labels), dtype=bool),
    }
    partitions = {}
    for offset, dataset_id in enumerate(np.unique(dataset.dataset_ids)):
        domain = dataset.dataset_ids == dataset_id
        negative_only = set(np.unique(dataset.labels[domain])) == {0}
        roles = (
            _small_negative_only_roles(
                dataset.subject_keys[domain],
                seed + 100 * offset,
            )
            if negative_only
            and len(np.unique(dataset.subject_keys[domain])) < 5
            else _three_way_keys(
                dataset.subject_keys[domain],
                seed + 100 * offset,
            )
        )
        partitions[str(dataset_id)] = roles
        masks[DEVELOPMENT_ROLE] |= domain & np.isin(
            dataset.subject_keys,
            roles[DEVELOPMENT_ROLE],
        )
        masks[CONTROLLER_ROLE] |= (
            domain
            & np.isin(dataset.subject_keys, roles[CONTROLLER_ROLE])
            & (dataset.recordings == 1)
        )
        masks[DEVELOPMENT_OBSERVED_ROLE] |= (
            domain
            & np.isin(
                dataset.subject_keys,
                roles[DEVELOPMENT_OBSERVED_ROLE],
            )
            & (dataset.recordings == 1)
        )
    return masks, partitions


def calibration_mask(
    dataset: TranslationalWindowDataset,
    masks: Mapping[str, np.ndarray],
) -> np.ndarray:
    evaluation_subjects = np.unique(
        dataset.subject_keys[
            masks[CONTROLLER_ROLE] | masks[DEVELOPMENT_OBSERVED_ROLE]
        ]
    )
    return masks[DEVELOPMENT_ROLE] | (
        np.isin(dataset.subject_keys, evaluation_subjects)
        & (dataset.recordings == 0)
    )


def core_mask(dataset: TranslationalWindowDataset) -> np.ndarray:
    return np.isin(dataset.dataset_ids, CORE_DATASETS)


def hard_negative_mask(dataset: TranslationalWindowDataset) -> np.ndarray:
    return (dataset.labels == 0) & (~dataset.cue_locked)


def split_controller_subroles(
    dataset: TranslationalWindowDataset,
    controller_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, list[str]]]]:
    """Separate fusion/calibration subjects from policy-selection subjects."""
    calibration = np.zeros(len(dataset.labels), dtype=bool)
    policy = np.zeros(len(dataset.labels), dtype=bool)
    roles: dict[str, dict[str, list[str]]] = {}
    for offset, dataset_id in enumerate(CORE_DATASETS):
        local = controller_mask & (dataset.dataset_ids == dataset_id)
        subjects = np.unique(dataset.subject_keys[local])
        if len(subjects) < 2:
            raise ValueError(
                f"Nested controller selection needs two {dataset_id} subjects."
            )
        rng = np.random.default_rng(seed + 1000 * offset)
        shuffled = subjects.copy()
        rng.shuffle(shuffled)
        calibration_count = max(1, len(shuffled) // 2)
        calibration_subjects = shuffled[:calibration_count]
        policy_subjects = shuffled[calibration_count:]
        calibration |= local & np.isin(
            dataset.subject_keys,
            calibration_subjects,
        )
        policy |= local & np.isin(
            dataset.subject_keys,
            policy_subjects,
        )
        roles[dataset_id] = {
            "fusion_and_probability_calibration": sorted(
                calibration_subjects.tolist()
            ),
            "operating_point_selection": sorted(policy_subjects.tolist()),
        }
    return calibration, policy, roles


def _select_domain_thresholds(
    dataset: TranslationalWindowDataset,
    probabilities: StageProbabilities,
    available: np.ndarray,
    selection_mask: np.ndarray,
    *,
    max_false_activation: float,
    target_sensitivity: float,
) -> dict[str, object]:
    selected = selection_mask & core_mask(dataset)
    conservative_global = select_window_threshold(
        dataset.labels[selected],
        probabilities.intent[selected],
        available[selected],
        max_false_activation_rate=max_false_activation,
    )
    sensitivity_global = select_sensitivity_threshold(
        dataset.labels[selected],
        probabilities.intent[selected],
        available[selected],
        target_sensitivity=target_sensitivity,
    )
    conservative = {}
    sensitivity = {}
    for dataset_id in CORE_DATASETS:
        local = selected & (dataset.dataset_ids == dataset_id)
        conservative[dataset_id] = select_window_threshold(
            dataset.labels[local],
            probabilities.intent[local],
            available[local],
            max_false_activation_rate=max_false_activation,
        )
        sensitivity[dataset_id] = select_sensitivity_threshold(
            dataset.labels[local],
            probabilities.intent[local],
            available[local],
            target_sensitivity=target_sensitivity,
        )
    return {
        "conservative": conservative,
        "conservative_fallback": conservative_global,
        "sensitivity_targeted": sensitivity,
        "sensitivity_targeted_fallback": sensitivity_global,
    }


def _score_domain_thresholds(
    dataset: TranslationalWindowDataset,
    probabilities: StageProbabilities,
    available: np.ndarray,
    selected: np.ndarray,
    thresholds: Mapping[str, float],
    fallback: float,
    *,
    intent_only: bool = False,
) -> dict[str, object]:
    local_domains = dataset.dataset_ids[selected]
    local_thresholds = domain_threshold_vector(
        local_domains,
        thresholds,
        fallback,
    )
    result = (
        score_intent_operating_point(
            dataset.labels[selected],
            probabilities.intent[selected],
            available[selected],
            local_thresholds,
        )
        if intent_only
        else score_operating_point(
            dataset.labels[selected],
            StageProbabilities(
                probabilities.intent[selected],
                probabilities.direction[selected],
            ),
            available[selected],
            local_thresholds,
        )
    )
    result["thresholds_by_dataset"] = dict(thresholds)
    result["threshold_fallback"] = float(fallback)
    result["by_dataset"] = {}
    for dataset_id in CORE_DATASETS:
        local = selected & (dataset.dataset_ids == dataset_id)
        result["by_dataset"][dataset_id] = (
            score_intent_operating_point(
                dataset.labels[local],
                probabilities.intent[local],
                available[local],
                float(thresholds[dataset_id]),
            )
            if intent_only
            else score_operating_point(
                dataset.labels[local],
                StageProbabilities(
                    probabilities.intent[local],
                    probabilities.direction[local],
                ),
                available[local],
                float(thresholds[dataset_id]),
            )
        )
    return result


def _calibrated_stage_controller(
    dataset: TranslationalWindowDataset,
    probabilities: Mapping[str, StageProbabilities],
    available: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    StageProbabilities,
    object,
    object,
    dict[str, object],
]:
    calibration, policy, nested_roles = split_controller_subroles(
        dataset,
        masks[CONTROLLER_ROLE],
        args.seed,
    )
    if args.intent_only:
        fusion = IntentSpecificFusion.select(
            probabilities,
            dataset.labels,
            dataset.dataset_ids,
            calibration,
        )
        raw_intent = fusion.predict(probabilities)
        calibrator = DomainIntentProbabilityCalibrator(args.seed).fit(
            raw_intent[calibration],
            dataset.labels[calibration],
            dataset.dataset_ids[calibration],
        )
        calibrator.select_blends(
            raw_intent,
            dataset.labels,
            dataset.dataset_ids,
            policy,
        )
        calibrated_intent = calibrator.predict(
            raw_intent,
            dataset.dataset_ids,
        )
        calibrated = StageProbabilities(
            calibrated_intent,
            np.full((len(calibrated_intent), 2), 0.5),
        )
        fusion_report = {
            "intent_weights": fusion.intent_weights,
            "selection_scores": {
                "mean_domain_intent_auroc": fusion.selection_score,
            },
        }
        calibration_report = {
            "method": "per_dataset_binary_platt_with_identity_shrinkage",
            "intent_blends": calibrator.intent_blends_,
        }
    else:
        fusion = StageSpecificFusion.select(
            probabilities,
            dataset.labels,
            dataset.dataset_ids,
            calibration,
        )
        raw_stage = fusion.predict(probabilities)
        calibrator = DomainProbabilityCalibrator(args.seed).fit(
            StageProbabilities(
                raw_stage.intent[calibration],
                raw_stage.direction[calibration],
            ),
            dataset.labels[calibration],
            dataset.dataset_ids[calibration],
        )
        calibrator.select_blends(
            raw_stage,
            dataset.labels,
            dataset.dataset_ids,
            policy,
        )
        calibrated = calibrator.predict(raw_stage, dataset.dataset_ids)
        fusion_report = {
            "intent_weights": fusion.intent_weights,
            "direction_weights": fusion.direction_weights,
            "selection_scores": fusion.selection_scores,
        }
        calibration_report = {
            "method": "per_dataset_platt_with_identity_shrinkage",
            "intent_blends": calibrator.intent_blends_,
            "direction_blends": calibrator.direction_blends_,
        }
    thresholds = _select_domain_thresholds(
        dataset,
        calibrated,
        available,
        policy,
        max_false_activation=args.max_window_false_activation,
        target_sensitivity=args.target_intent_sensitivity,
    )
    report: dict[str, object] = {
        "nested_controller_roles": nested_roles,
        "fusion": fusion_report,
        "probability_calibration": calibration_report,
        "thresholds": thresholds,
        "operating_points": {},
    }
    for name, threshold_key, fallback_key in (
        (
            "conservative",
            "conservative",
            "conservative_fallback",
        ),
        (
            "sensitivity_targeted",
            "sensitivity_targeted",
            "sensitivity_targeted_fallback",
        ),
    ):
        report["operating_points"][name] = {
            "controller_policy": _score_domain_thresholds(
                dataset,
                calibrated,
                available,
                policy,
                thresholds[threshold_key],
                thresholds[fallback_key],
                intent_only=args.intent_only,
            ),
            DEVELOPMENT_OBSERVED_ROLE: _score_domain_thresholds(
                dataset,
                calibrated,
                available,
                masks[DEVELOPMENT_OBSERVED_ROLE] & core_mask(dataset),
                thresholds[threshold_key],
                thresholds[fallback_key],
                intent_only=args.intent_only,
            ),
        }

    hard_controller = masks[CONTROLLER_ROLE] & hard_negative_mask(dataset)
    intent_controller = (
        policy & core_mask(dataset) & (dataset.labels != 0)
    )
    if np.any(hard_controller):
        groups = np.char.add(
            np.char.add(dataset.subject_keys, ":"),
            dataset.sessions.astype(str),
        )
        temporal_threshold, temporal_controller = select_continuous_threshold(
            dataset.labels,
            calibrated.intent,
            available,
            intent_controller,
            hard_controller,
            groups,
            dataset.window_onsets_seconds,
            max_false_activations_per_minute=(
                args.max_continuous_false_activations_per_minute
            ),
            confirmations=args.temporal_confirmations,
            refractory_seconds=args.temporal_refractory_seconds,
        )
        observed_hard = (
            masks[DEVELOPMENT_OBSERVED_ROLE]
            & hard_negative_mask(dataset)
        )
        temporal_observed: dict[str, object] | str = (
            continuous_false_activations(
                calibrated.intent[observed_hard],
                available[observed_hard],
                groups[observed_hard],
                dataset.window_onsets_seconds[observed_hard],
                temporal_threshold,
                confirmations=args.temporal_confirmations,
                refractory_seconds=args.temporal_refractory_seconds,
            )
            if np.any(observed_hard)
            else "not_evaluated_no_observed_continuous_recording"
        )
        report["temporal_confirmation"] = {
            "status": "measured",
            "threshold": temporal_threshold,
            "controller": temporal_controller,
            DEVELOPMENT_OBSERVED_ROLE: temporal_observed,
        }
    else:
        report["temporal_confirmation"] = {
            "status": "prepared_not_evaluated",
            "reason": "no continuous no-control manifest",
            "confirmations": args.temporal_confirmations,
            "refractory_seconds": args.temporal_refractory_seconds,
        }
    return report, calibrated, fusion, calibrator, thresholds


def _fit_models(
    dataset: TranslationalWindowDataset,
    features: TranslationalFeatures,
    train_mask: np.ndarray,
    *,
    seed: int,
    domain_weight: float,
    intent_only: bool = False,
    auxiliary_direction: bool = False,
    v7_graph_lda: bool = False,
) -> tuple[
    dict[str, object],
    dict[str, StageProbabilities],
]:
    models: dict[str, object] = {}
    probabilities: dict[str, StageProbabilities] = {}
    branches = {
        name: values
        for name, values in features.branches.items()
        if name != "quality"
    }
    if intent_only:
        base_branch_names = ("mrcp", "smr", "riemann")
        enhanced_branch_names = base_branch_names + (
            ("gsp",) if v7_graph_lda else ()
        )
        if v7_graph_lda and "gsp" not in branches:
            raise RuntimeError("V7 graph/LDA integration requires the GSP branch.")
        for offset, branch_name in enumerate(enhanced_branch_names):
            model = IntentBranchEnsemble.fit(
                {branch_name: branches[branch_name][train_mask]},
                dataset.labels[train_mask],
                dataset.dataset_ids[train_mask],
                branch_names=(branch_name,),
                seed=seed + 10 * offset,
                domain_weight=domain_weight,
            )
            models[branch_name] = model
            intent = model.predict_intent_probabilities(
                {branch_name: branches[branch_name]},
                dataset.dataset_ids,
            )
            probabilities[branch_name] = StageProbabilities(
                intent,
                np.full((len(intent), 2), 0.5),
            )
        fusion = IntentBranchEnsemble.fit(
            {name: values[train_mask] for name, values in branches.items()},
            dataset.labels[train_mask],
            dataset.dataset_ids[train_mask],
            branch_names=base_branch_names,
            seed=seed + 100,
            domain_weight=domain_weight,
        )
        models["late_fusion"] = fusion
        intent = fusion.predict_intent_probabilities(
            branches,
            dataset.dataset_ids,
        )
        probabilities["late_fusion"] = StageProbabilities(
            intent,
            np.full((len(intent), 2), 0.5),
        )
        if v7_graph_lda:
            graph_fusion = IntentBranchEnsemble.fit(
                {
                    name: branches[name][train_mask]
                    for name in enhanced_branch_names
                },
                dataset.labels[train_mask],
                dataset.dataset_ids[train_mask],
                branch_names=enhanced_branch_names,
                seed=seed + 150,
                domain_weight=domain_weight,
            )
            models["graph_late_fusion"] = graph_fusion
            graph_intent = graph_fusion.predict_intent_probabilities(
                branches,
                dataset.dataset_ids,
            )
            probabilities["graph_late_fusion"] = StageProbabilities(
                graph_intent,
                np.full((len(graph_intent), 2), 0.5),
            )

            lda_features = np.concatenate(
                [branches[name] for name in enhanced_branch_names],
                axis=1,
            )
            shrinkage_lda = DomainAwareShrinkageLDAIntentDecoder(
                domain_weight=domain_weight,
            ).fit(
                lda_features[train_mask],
                dataset.labels[train_mask],
                dataset.dataset_ids[train_mask],
            )
            models["shrinkage_lda"] = shrinkage_lda
            lda_intent = shrinkage_lda.predict_intent_probabilities(
                lda_features,
                dataset.dataset_ids,
            )
            probabilities["shrinkage_lda"] = StageProbabilities(
                lda_intent,
                np.full((len(lda_intent), 2), 0.5),
            )
        if auxiliary_direction:
            auxiliary = AuxiliaryIntentBranchEnsemble.fit(
                {
                    name: values[train_mask]
                    for name, values in branches.items()
                },
                dataset.labels[train_mask],
                dataset.dataset_ids[train_mask],
                branch_names=("mrcp", "smr", "riemann"),
                seed=seed + 200,
                domain_weight=domain_weight,
            )
            models["auxiliary_direction_intent"] = auxiliary
            auxiliary_intent = auxiliary.predict_intent_probabilities(
                branches,
                dataset.dataset_ids,
            )
            probabilities["auxiliary_direction_intent"] = StageProbabilities(
                auxiliary_intent,
                np.full((len(auxiliary_intent), 2), 0.5),
            )
            if v7_graph_lda:
                graph_auxiliary = AuxiliaryIntentBranchEnsemble.fit(
                    {
                        name: branches[name][train_mask]
                        for name in enhanced_branch_names
                    },
                    dataset.labels[train_mask],
                    dataset.dataset_ids[train_mask],
                    branch_names=enhanced_branch_names,
                    seed=seed + 250,
                    domain_weight=domain_weight,
                )
                models["graph_auxiliary_direction_intent"] = graph_auxiliary
                graph_auxiliary_intent = (
                    graph_auxiliary.predict_intent_probabilities(
                        branches,
                        dataset.dataset_ids,
                    )
                )
                probabilities["graph_auxiliary_direction_intent"] = (
                    StageProbabilities(
                        graph_auxiliary_intent,
                        np.full((len(graph_auxiliary_intent), 2), 0.5),
                    )
                )
        return models, probabilities

    for offset, branch_name in enumerate(("mrcp", "smr", "riemann")):
        model = BranchEnsemble.fit(
            {branch_name: branches[branch_name][train_mask]},
            dataset.labels[train_mask],
            dataset.dataset_ids[train_mask],
            branch_names=(branch_name,),
            seed=seed + 10 * offset,
            domain_weight=domain_weight,
        )
        models[branch_name] = model
        probabilities[branch_name] = model.predict_stage_probabilities(
            {branch_name: branches[branch_name]},
            dataset.dataset_ids,
        )
    fusion = BranchEnsemble.fit(
        {
            name: values[train_mask]
            for name, values in branches.items()
        },
        dataset.labels[train_mask],
        dataset.dataset_ids[train_mask],
        branch_names=("mrcp", "smr", "riemann"),
        seed=seed + 100,
        domain_weight=domain_weight,
    )
    models["late_fusion"] = fusion
    probabilities["late_fusion"] = fusion.predict_stage_probabilities(
        branches,
        dataset.dataset_ids,
    )

    fbcsp = HierarchicalFBCSP(
        dataset.sfreq,
        components_per_band=4,
        random_seed=seed,
        filter_bank=SMR_FILTER_BANK,
    ).fit(
        dataset.smr_epochs[train_mask],
        dataset.labels[train_mask],
    )
    intent, direction = fbcsp.predict_stage_probabilities(dataset.smr_epochs)
    models["fbcsp"] = fbcsp
    probabilities["fbcsp"] = StageProbabilities(intent, direction)
    return models, probabilities


def _add_neural_intent_ensembles(
    dataset: TranslationalWindowDataset,
    models: dict[str, object],
    probabilities: dict[str, StageProbabilities],
    args: argparse.Namespace,
) -> dict[str, object] | None:
    if not args.neural_bakeoff_dir:
        return None
    records: dict[str, object] = {}
    for architecture in args.neural_architectures:
        intent, _, metadata = predict_bakeoff_ensemble(
            dataset.mrcp_epochs,
            dataset.smr_epochs,
            dataset.channel_names,
            args.neural_bakeoff_dir,
            architecture=architecture,
            seeds=args.neural_seeds,
            batch_size=args.neural_batch_size,
            device=args.neural_device,
        )
        probabilities[architecture] = StageProbabilities(
            intent=intent,
            direction=(
                probabilities["fbcsp"].direction.copy()
                if "fbcsp" in probabilities
                else np.full((len(intent), 2), 0.5)
            ),
        )
        models[architecture] = {
            "role": "five_seed_eeg_only_intent_proposal",
            "direction_source": (
                "existing_fbcsp_direction_head"
                if "fbcsp" in probabilities
                else "not_applicable_intent_only"
            ),
            "metadata": metadata,
        }
        records[architecture] = metadata
    return {
        "architectures": list(args.neural_architectures),
        "seeds": list(args.neural_seeds),
        "device": args.neural_device,
        "ensembles": records,
        "fusion_selection": "nested_controller_subjects_only",
        "peripheral_inputs_required": False,
    }


def _select_threshold(
    dataset: TranslationalWindowDataset,
    probabilities: StageProbabilities,
    available: np.ndarray,
    controller_mask: np.ndarray,
    *,
    max_window_false_activation: float,
    max_continuous_false_activations_per_minute: float,
) -> tuple[float, dict[str, object] | None, str]:
    hard = controller_mask & hard_negative_mask(dataset)
    intent = controller_mask & core_mask(dataset) & (dataset.labels != 0)
    if np.any(hard):
        groups = np.char.add(
            np.char.add(dataset.subject_keys, ":"),
            dataset.sessions.astype(str),
        )
        threshold, continuous = select_continuous_threshold(
            dataset.labels,
            probabilities.intent,
            available,
            intent,
            hard,
            groups,
            dataset.window_onsets_seconds,
            max_false_activations_per_minute=(
                max_continuous_false_activations_per_minute
            ),
        )
        return threshold, continuous, "measured_continuous_no_control"
    selected = controller_mask & core_mask(dataset)
    threshold = select_window_threshold(
        dataset.labels[selected],
        probabilities.intent[selected],
        available[selected],
        max_false_activation_rate=max_window_false_activation,
    )
    return threshold, None, "cue_locked_window_fallback"


def _score_methods(
    dataset: TranslationalWindowDataset,
    probabilities: Mapping[str, StageProbabilities],
    available: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, float]]:
    results: dict[str, object] = {}
    thresholds = {}
    for method, values in probabilities.items():
        threshold, continuous, basis = _select_threshold(
            dataset,
            values,
            available,
            masks[CONTROLLER_ROLE],
            max_window_false_activation=args.max_window_false_activation,
            max_continuous_false_activations_per_minute=(
                args.max_continuous_false_activations_per_minute
            ),
        )
        thresholds[method] = threshold
        method_results: dict[str, object] = {
            "threshold": threshold,
            "threshold_selection_basis": basis,
            "controller_continuous_no_control": continuous,
        }
        for role in (CONTROLLER_ROLE, DEVELOPMENT_OBSERVED_ROLE):
            selected = masks[role] & core_mask(dataset)
            method_results[role] = (
                score_intent_operating_point(
                    dataset.labels[selected],
                    values.intent[selected],
                    available[selected],
                    threshold,
                )
                if args.intent_only
                else score_operating_point(
                    dataset.labels[selected],
                    StageProbabilities(
                        values.intent[selected],
                        values.direction[selected],
                    ),
                    available[selected],
                    threshold,
                )
            )
            method_results[f"{role}_by_dataset"] = {
                dataset_id: (
                    score_intent_operating_point(
                        dataset.labels[
                            selected & (dataset.dataset_ids == dataset_id)
                        ],
                        values.intent[
                            selected & (dataset.dataset_ids == dataset_id)
                        ],
                        available[
                            selected & (dataset.dataset_ids == dataset_id)
                        ],
                        threshold,
                    )
                    if args.intent_only
                    else score_operating_point(
                        dataset.labels[
                            selected & (dataset.dataset_ids == dataset_id)
                        ],
                        StageProbabilities(
                            values.intent[
                                selected & (dataset.dataset_ids == dataset_id)
                            ],
                            values.direction[
                                selected & (dataset.dataset_ids == dataset_id)
                            ],
                        ),
                        available[
                            selected & (dataset.dataset_ids == dataset_id)
                        ],
                        threshold,
                    )
                )
                for dataset_id in CORE_DATASETS
                if np.any(selected & (dataset.dataset_ids == dataset_id))
            }
        observed_hard = masks[DEVELOPMENT_OBSERVED_ROLE] & hard_negative_mask(
            dataset
        )
        if np.any(observed_hard):
            groups = np.char.add(
                np.char.add(dataset.subject_keys, ":"),
                dataset.sessions.astype(str),
            )
            method_results["development_observed_continuous_no_control"] = (
                continuous_false_activations(
                    values.intent[observed_hard],
                    available[observed_hard],
                    groups[observed_hard],
                    dataset.window_onsets_seconds[observed_hard],
                    threshold,
                )
            )
        else:
            method_results["development_observed_continuous_no_control"] = (
                "not_evaluated_no_continuous_negative_recording"
            )
        results[method] = method_results
    return results, thresholds


def _select_patient_calibration(
    dataset: TranslationalWindowDataset,
    features: TranslationalFeatures,
    masks: Mapping[str, np.ndarray],
    global_probabilities: StageProbabilities,
    available: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    StageProbabilities,
    dict[str, dict[str, float]],
]:
    branches = {
        name: values
        for name, values in features.branches.items()
        if name != "quality"
    }
    branches["population_stage"] = population_stage_features(global_probabilities)
    patient_branch_names = ("population_stage", "mrcp", "smr", "riemann")
    final_probabilities = StageProbabilities(
        global_probabilities.intent.copy(),
        global_probabilities.direction.copy(),
    )
    report_by_dataset = {}
    conservative_thresholds = {}
    sensitivity_thresholds = {}
    selected_configurations = {}
    for dataset_id in CORE_DATASETS:
        controller = (
            masks[CONTROLLER_ROLE]
            & (dataset.dataset_ids == dataset_id)
        )
        configurations = []
        for budget in (2, 5, 10):
            for blend in (0.25, 0.5, 0.75):
                adapted = patient_adapted_probabilities(
                    branches,
                    dataset.labels,
                    dataset.dataset_ids,
                    dataset.subject_keys,
                    dataset.recordings,
                    controller,
                    global_probabilities,
                    branch_names=patient_branch_names,
                    budget_per_class=budget,
                    blend_weight=blend,
                    seed=args.seed,
                )
                conservative_threshold = select_window_threshold(
                    dataset.labels[controller],
                    adapted.intent[controller],
                    available[controller],
                    max_false_activation_rate=(
                        args.max_window_false_activation
                    ),
                )
                sensitivity_threshold = select_sensitivity_threshold(
                    dataset.labels[controller],
                    adapted.intent[controller],
                    available[controller],
                    target_sensitivity=args.target_intent_sensitivity,
                )
                conservative_metrics = score_operating_point(
                    dataset.labels[controller],
                    StageProbabilities(
                        adapted.intent[controller],
                        adapted.direction[controller],
                    ),
                    available[controller],
                    conservative_threshold,
                )
                configurations.append(
                    {
                        "budget_per_class": budget,
                        "blend_weight": blend,
                        "conservative_threshold": conservative_threshold,
                        "sensitivity_targeted_threshold": (
                            sensitivity_threshold
                        ),
                        "controller_conservative": conservative_metrics,
                    }
                )
        selected = max(
            configurations,
            key=lambda row: (
                float(
                    row["controller_conservative"]["intent_sensitivity"]
                ),
                float(
                    row["controller_conservative"][
                        "intent_balanced_accuracy"
                    ]
                ),
                -float(
                    row["controller_conservative"][
                        "false_activation_rate_per_window"
                    ]
                ),
            ),
        )
        adapted_controller = patient_adapted_probabilities(
            branches,
            dataset.labels,
            dataset.dataset_ids,
            dataset.subject_keys,
            dataset.recordings,
            controller,
            global_probabilities,
            branch_names=patient_branch_names,
            budget_per_class=int(selected["budget_per_class"]),
            blend_weight=float(selected["blend_weight"]),
            seed=args.seed,
        )
        final_probabilities.intent[controller] = adapted_controller.intent[controller]
        final_probabilities.direction[controller] = adapted_controller.direction[controller]
        observed = (
            masks[DEVELOPMENT_OBSERVED_ROLE]
            & (dataset.dataset_ids == dataset_id)
        )
        adapted_observed = patient_adapted_probabilities(
            branches,
            dataset.labels,
            dataset.dataset_ids,
            dataset.subject_keys,
            dataset.recordings,
            observed,
            global_probabilities,
            branch_names=patient_branch_names,
            budget_per_class=int(selected["budget_per_class"]),
            blend_weight=float(selected["blend_weight"]),
            seed=args.seed,
        )
        final_probabilities.intent[observed] = (
            adapted_observed.intent[observed]
        )
        final_probabilities.direction[observed] = (
            adapted_observed.direction[observed]
        )
        conservative_thresholds[dataset_id] = float(
            selected["conservative_threshold"]
        )
        sensitivity_thresholds[dataset_id] = float(
            selected["sensitivity_targeted_threshold"]
        )
        selected_configurations[dataset_id] = {
            "budget_per_class": int(selected["budget_per_class"]),
            "blend_weight": float(selected["blend_weight"]),
        }
        report_by_dataset[dataset_id] = {
            "selected_on_controller": selected,
            "all_controller_configurations": configurations,
            "development_observed_conservative": score_operating_point(
                dataset.labels[observed],
                StageProbabilities(
                    final_probabilities.intent[observed],
                    final_probabilities.direction[observed],
                ),
                available[observed],
                conservative_thresholds[dataset_id],
            ),
            "development_observed_sensitivity_targeted": (
                score_operating_point(
                    dataset.labels[observed],
                    StageProbabilities(
                        final_probabilities.intent[observed],
                        final_probabilities.direction[observed],
                    ),
                    available[observed],
                    sensitivity_thresholds[dataset_id],
                )
            ),
        }
    observed_all = (
        masks[DEVELOPMENT_OBSERVED_ROLE] & core_mask(dataset)
    )
    conservative_fallback = float(
        np.mean(list(conservative_thresholds.values()))
    )
    sensitivity_fallback = float(
        np.mean(list(sensitivity_thresholds.values()))
    )
    report = {
        "status": "core_stage_all_datasets",
        "adapter_inputs": list(patient_branch_names),
        "calibration_protocol": (
            "first labeled windows from recording 0; recording 1 evaluation-only"
        ),
        "candidate_budgets_per_class": [2, 5, 10],
        "selected_configurations": selected_configurations,
        "by_dataset": report_by_dataset,
        "development_observed_conservative": _score_domain_thresholds(
            dataset,
            final_probabilities,
            available,
            observed_all,
            conservative_thresholds,
            conservative_fallback,
        ),
        "development_observed_sensitivity_targeted": (
            _score_domain_thresholds(
                dataset,
                final_probabilities,
                available,
                observed_all,
                sensitivity_thresholds,
                sensitivity_fallback,
            )
        ),
    }
    return (
        report,
        final_probabilities,
        {
            "conservative": conservative_thresholds,
            "sensitivity_targeted": sensitivity_thresholds,
        },
    )


def _select_intent_only_patient_calibration(
    dataset: TranslationalWindowDataset,
    features: TranslationalFeatures,
    masks: Mapping[str, np.ndarray],
    global_probabilities: StageProbabilities,
    available: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    StageProbabilities,
    dict[str, dict[str, float]],
    np.ndarray,
]:
    branches = {
        name: values
        for name, values in features.branches.items()
        if name != "quality"
    }
    branches["population_intent"] = population_intent_features(
        global_probabilities.intent
    )
    auxiliary = bool(args.auxiliary_direction)
    base_patient_branch_names = (
        "population_intent",
        "mrcp",
        "smr",
        "riemann",
    )
    candidate_patient_branch_names = [base_patient_branch_names]
    if auxiliary and bool(getattr(args, "v7_graph_lda", False)):
        candidate_patient_branch_names.append(
            base_patient_branch_names + ("gsp",)
        )
    final_intent = global_probabilities.intent.copy()
    final_evaluation = np.zeros(len(dataset.labels), dtype=bool)
    report_by_dataset = {}
    conservative_thresholds = {}
    sensitivity_thresholds = {}
    selected_configurations = {}

    def adapt(
        subject_mask: np.ndarray,
        budget: int,
        blend: float,
        auxiliary_weight: float,
        branch_names: tuple[str, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        if auxiliary:
            return patient_adapted_auxiliary_intent_probabilities(
                branches,
                dataset.labels,
                dataset.dataset_ids,
                dataset.subject_keys,
                dataset.recordings,
                subject_mask,
                global_probabilities.intent,
                branch_names=branch_names,
                budget_per_source_class=budget,
                blend_weight=blend,
                auxiliary_weight=auxiliary_weight,
                seed=args.seed,
            )
        return patient_adapted_intent_probabilities(
            branches,
            dataset.labels,
            dataset.dataset_ids,
            dataset.subject_keys,
            dataset.recordings,
            subject_mask,
            global_probabilities.intent,
            branch_names=branch_names,
            budget_per_class=budget,
            blend_weight=blend,
            seed=args.seed,
        )

    for dataset_id in CORE_DATASETS:
        controller_subjects = (
            masks[CONTROLLER_ROLE]
            & (dataset.dataset_ids == dataset_id)
        )
        configurations = []
        for branch_names in candidate_patient_branch_names:
            for budget in (2, 5, 10):
                for blend in (0.25, 0.5, 0.75):
                    for auxiliary_weight in (
                        (0.0, 0.25, 0.5) if auxiliary else (0.0,)
                    ):
                        adapted, adapted_mask = adapt(
                            controller_subjects,
                            budget,
                            blend,
                            auxiliary_weight,
                            branch_names,
                        )
                        evaluation = controller_subjects & adapted_mask
                        conservative_threshold = select_window_threshold(
                            dataset.labels[evaluation],
                            adapted[evaluation],
                            available[evaluation],
                            max_false_activation_rate=(
                                args.max_window_false_activation
                            ),
                        )
                        sensitivity_threshold = select_sensitivity_threshold(
                            dataset.labels[evaluation],
                            adapted[evaluation],
                            available[evaluation],
                            target_sensitivity=args.target_intent_sensitivity,
                        )
                        configuration = {
                            "adapter_branch_names": list(branch_names),
                            "graph_branch_enabled": "gsp" in branch_names,
                            "total_calibration_windows": (
                                3 * budget if auxiliary else 2 * budget
                            ),
                            "blend_weight": blend,
                            "conservative_threshold": conservative_threshold,
                            "sensitivity_targeted_threshold": sensitivity_threshold,
                            "controller_conservative": score_intent_operating_point(
                                dataset.labels[evaluation],
                                adapted[evaluation],
                                available[evaluation],
                                conservative_threshold,
                            ),
                        }
                        if auxiliary:
                            configuration.update(
                                {
                                    "budget_per_source_class": budget,
                                    "auxiliary_weight": auxiliary_weight,
                                }
                            )
                        else:
                            configuration["budget_per_binary_class"] = budget
                        configurations.append(configuration)
        def ranking(row: Mapping[str, object]) -> tuple[float, float, float]:
            metrics = row["controller_conservative"]
            return (
                float(metrics["intent_sensitivity"]),
                float(metrics["intent_balanced_accuracy"]),
                -float(metrics["false_activation_rate_per_window"]),
            )
        selected = max(
            configurations,
            key=ranking,
        )
        graph_selection_guard = {"status": "not_applicable"}
        if selected["graph_branch_enabled"]:
            best_base = max(
                [
                    row
                    for row in configurations
                    if not row["graph_branch_enabled"]
                ],
                key=ranking,
            )
            sensitivity_gain = float(
                selected["controller_conservative"]["intent_sensitivity"]
                - best_base["controller_conservative"]["intent_sensitivity"]
            )
            balanced_accuracy_gain = float(
                selected["controller_conservative"][
                    "intent_balanced_accuracy"
                ]
                - best_base["controller_conservative"][
                    "intent_balanced_accuracy"
                ]
            )
            accepted = (
                sensitivity_gain >= GRAPH_ADAPTER_MIN_CONTROLLER_GAIN
                and balanced_accuracy_gain >= GRAPH_ADAPTER_MIN_CONTROLLER_GAIN
            )
            graph_selection_guard = {
                "status": "accepted" if accepted else "rejected_to_base",
                "minimum_gain": GRAPH_ADAPTER_MIN_CONTROLLER_GAIN,
                "controller_sensitivity_gain": sensitivity_gain,
                "controller_balanced_accuracy_gain": balanced_accuracy_gain,
            }
            if not accepted:
                selected = best_base
        budget = int(
            selected[
                "budget_per_source_class"
                if auxiliary
                else "budget_per_binary_class"
            ]
        )
        blend = float(selected["blend_weight"])
        auxiliary_weight = float(selected.get("auxiliary_weight", 0.0))
        selected_branch_names = tuple(selected["adapter_branch_names"])
        adapted_controller, controller_evaluation = adapt(
            controller_subjects,
            budget,
            blend,
            auxiliary_weight,
            selected_branch_names,
        )
        final_intent[controller_evaluation] = adapted_controller[
            controller_evaluation
        ]
        observed_subjects = (
            masks[DEVELOPMENT_OBSERVED_ROLE]
            & (dataset.dataset_ids == dataset_id)
        )
        adapted_observed, observed_evaluation = adapt(
            observed_subjects,
            budget,
            blend,
            auxiliary_weight,
            selected_branch_names,
        )
        final_intent[observed_evaluation] = adapted_observed[
            observed_evaluation
        ]
        final_evaluation |= observed_evaluation
        conservative_thresholds[dataset_id] = float(
            selected["conservative_threshold"]
        )
        sensitivity_thresholds[dataset_id] = float(
            selected["sensitivity_targeted_threshold"]
        )
        selected_configurations[dataset_id] = {
            (
                "budget_per_source_class"
                if auxiliary
                else "budget_per_binary_class"
            ): budget,
            "total_calibration_windows": 3 * budget if auxiliary else 2 * budget,
            "blend_weight": blend,
            "adapter_branch_names": list(selected_branch_names),
            "graph_branch_enabled": "gsp" in selected_branch_names,
            "graph_selection_guard": graph_selection_guard,
            **({"auxiliary_weight": auxiliary_weight} if auxiliary else {}),
        }
        report_by_dataset[dataset_id] = {
            "selected_on_controller": selected,
            "all_controller_configurations": configurations,
            "controller_evaluation_rows": int(np.sum(controller_evaluation)),
            "development_observed_evaluation_rows": int(
                np.sum(observed_evaluation)
            ),
            "development_observed_conservative": score_intent_operating_point(
                dataset.labels[observed_evaluation],
                final_intent[observed_evaluation],
                available[observed_evaluation],
                conservative_thresholds[dataset_id],
            ),
            "development_observed_sensitivity_targeted": (
                score_intent_operating_point(
                    dataset.labels[observed_evaluation],
                    final_intent[observed_evaluation],
                    available[observed_evaluation],
                    sensitivity_thresholds[dataset_id],
                )
            ),
        }
    conservative_fallback = float(
        np.mean(list(conservative_thresholds.values()))
    )
    sensitivity_fallback = float(
        np.mean(list(sensitivity_thresholds.values()))
    )
    final_probabilities = StageProbabilities(
        final_intent,
        np.full((len(final_intent), 2), 0.5),
    )
    population_matched = StageProbabilities(
        global_probabilities.intent,
        np.full((len(final_intent), 2), 0.5),
    )
    report = {
        "status": (
            "binary_intent_with_auxiliary_direction_all_datasets"
            if auxiliary
            else "strict_binary_intent_all_datasets"
        ),
        "classes": ["rest", "intent"],
        "auxiliary_training_labels": (
            ["rest", "left", "right"] if auxiliary else []
        ),
        "auxiliary_direction_output_exposed": False,
        "adapter_inputs": {
            dataset_id: configuration["adapter_branch_names"]
            for dataset_id, configuration in selected_configurations.items()
        },
        "calibration_protocol": (
            "recording 0 supplies labeled rest/left/right cue windows for an "
            "auxiliary direction signal; output remains rest/intent and only "
            "later recordings are evaluated"
            if auxiliary
            else "recording 0 supplies labeled rest/intent calibration windows; "
            "only later recordings are evaluated"
        ),
        "calibration_recording": 0,
        "evaluation_recordings": "recording != 0",
        "calibration_rows_excluded_from_metrics": True,
        (
            "candidate_budgets_per_source_class"
            if auxiliary
            else "candidate_budgets_per_binary_class"
        ): [2, 5, 10],
        "candidate_auxiliary_weights": [0.0, 0.25, 0.5] if auxiliary else [],
        "graph_adapter_minimum_controller_gain": (
            GRAPH_ADAPTER_MIN_CONTROLLER_GAIN
            if len(candidate_patient_branch_names) > 1
            else "not_applicable"
        ),
        "selected_configurations": selected_configurations,
        "by_dataset": report_by_dataset,
        "development_observed_evaluation_rows": int(np.sum(final_evaluation)),
        "population_matched_conservative": _score_domain_thresholds(
            dataset,
            population_matched,
            available,
            final_evaluation,
            conservative_thresholds,
            conservative_fallback,
            intent_only=True,
        ),
        "development_observed_conservative": _score_domain_thresholds(
            dataset,
            final_probabilities,
            available,
            final_evaluation,
            conservative_thresholds,
            conservative_fallback,
            intent_only=True,
        ),
        "development_observed_sensitivity_targeted": (
            _score_domain_thresholds(
                dataset,
                final_probabilities,
                available,
                final_evaluation,
                sensitivity_thresholds,
                sensitivity_fallback,
                intent_only=True,
            )
        ),
    }
    return (
        report,
        final_probabilities,
        {
            "conservative": conservative_thresholds,
            "sensitivity_targeted": sensitivity_thresholds,
        },
        final_evaluation,
    )


def _patient_temporal_confirmation(
    dataset: TranslationalWindowDataset,
    probabilities: StageProbabilities,
    available: np.ndarray,
    masks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, object]:
    hard_controller = masks[CONTROLLER_ROLE] & hard_negative_mask(dataset)
    if not np.any(hard_controller):
        return {
            "status": "prepared_not_evaluated",
            "reason": "no continuous no-control manifest",
            "confirmations": args.temporal_confirmations,
            "refractory_seconds": args.temporal_refractory_seconds,
            "probability_source": "patient_adapted_neural_integrated_intent",
        }
    groups = np.char.add(
        np.char.add(dataset.subject_keys, ":"),
        dataset.sessions.astype(str),
    )
    intent_controller = (
        masks[CONTROLLER_ROLE]
        & core_mask(dataset)
        & (dataset.labels != 0)
    )
    threshold, controller = select_continuous_threshold(
        dataset.labels,
        probabilities.intent,
        available,
        intent_controller,
        hard_controller,
        groups,
        dataset.window_onsets_seconds,
        max_false_activations_per_minute=(
            args.max_continuous_false_activations_per_minute
        ),
        confirmations=args.temporal_confirmations,
        refractory_seconds=args.temporal_refractory_seconds,
    )
    observed_hard = masks[DEVELOPMENT_OBSERVED_ROLE] & hard_negative_mask(dataset)
    observed: dict[str, object] | str = (
        continuous_false_activations(
            probabilities.intent[observed_hard],
            available[observed_hard],
            groups[observed_hard],
            dataset.window_onsets_seconds[observed_hard],
            threshold,
            confirmations=args.temporal_confirmations,
            refractory_seconds=args.temporal_refractory_seconds,
        )
        if np.any(observed_hard)
        else "not_evaluated_no_observed_continuous_recording"
    )
    return {
        "status": "measured",
        "threshold": threshold,
        "controller": controller,
        DEVELOPMENT_OBSERVED_ROLE: observed,
        "probability_source": "patient_adapted_neural_integrated_intent",
    }


def _bootstrap_operating_point(
    dataset: TranslationalWindowDataset,
    probabilities: StageProbabilities,
    available: np.ndarray,
    selected: np.ndarray,
    thresholds: Mapping[str, float],
    fallback: float,
    *,
    iterations: int,
    seed: int,
    intent_only: bool = False,
) -> dict[str, object]:
    local_thresholds = domain_threshold_vector(
        dataset.dataset_ids[selected],
        thresholds,
        fallback,
    )
    if intent_only:
        overall = subject_bootstrap_intent_confidence_intervals(
            dataset.labels[selected],
            probabilities.intent[selected],
            available[selected],
            local_thresholds,
            dataset.subject_keys[selected],
            iterations=iterations,
            seed=seed,
        )
    else:
        overall = subject_bootstrap_confidence_intervals(
            dataset.labels[selected],
            StageProbabilities(
                probabilities.intent[selected],
                probabilities.direction[selected],
            ),
            available[selected],
            local_thresholds,
            dataset.subject_keys[selected],
            iterations=iterations,
            seed=seed,
        )
    report: dict[str, object] = {"overall": overall, "by_dataset": {}}
    for offset, dataset_id in enumerate(CORE_DATASETS):
        local = selected & (dataset.dataset_ids == dataset_id)
        if intent_only:
            interval = subject_bootstrap_intent_confidence_intervals(
                dataset.labels[local],
                probabilities.intent[local],
                available[local],
                float(thresholds[dataset_id]),
                dataset.subject_keys[local],
                iterations=iterations,
                seed=seed + 100 * (offset + 1),
            )
        else:
            interval = subject_bootstrap_confidence_intervals(
                dataset.labels[local],
                StageProbabilities(
                    probabilities.intent[local],
                    probabilities.direction[local],
                ),
                available[local],
                float(thresholds[dataset_id]),
                dataset.subject_keys[local],
                iterations=iterations,
                seed=seed + 100 * (offset + 1),
            )
        report["by_dataset"][dataset_id] = interval
    return report


def _paired_patient_comparison(
    dataset: TranslationalWindowDataset,
    baseline: StageProbabilities,
    candidate: StageProbabilities,
    available: np.ndarray,
    selected: np.ndarray,
    baseline_thresholds: Mapping[str, float],
    candidate_thresholds: Mapping[str, float],
    *,
    seed: int,
    intent_only: bool = False,
) -> dict[str, object]:
    records = []
    for subject in np.unique(dataset.subject_keys[selected]):
        local = selected & (dataset.subject_keys == subject)
        domain = str(dataset.dataset_ids[local][0])
        if intent_only:
            baseline_result = score_intent_operating_point(
                dataset.labels[local],
                baseline.intent[local],
                available[local],
                float(baseline_thresholds[domain]),
            )
            candidate_result = score_intent_operating_point(
                dataset.labels[local],
                candidate.intent[local],
                available[local],
                float(candidate_thresholds[domain]),
            )
        else:
            baseline_result = score_operating_point(
                dataset.labels[local],
                StageProbabilities(
                    baseline.intent[local],
                    baseline.direction[local],
                ),
                available[local],
                float(baseline_thresholds[domain]),
            )
            candidate_result = score_operating_point(
                dataset.labels[local],
                StageProbabilities(
                    candidate.intent[local],
                    candidate.direction[local],
                ),
                available[local],
                float(candidate_thresholds[domain]),
            )
        records.append(
            {
                "subject": str(subject),
                "dataset": domain,
                "baseline_balanced_accuracy": float(
                    baseline_result["intent_balanced_accuracy"]
                ),
                "candidate_balanced_accuracy": float(
                    candidate_result["intent_balanced_accuracy"]
                ),
            }
        )
    deltas = np.asarray(
        [
            row["candidate_balanced_accuracy"] - row["baseline_balanced_accuracy"]
            for row in records
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(deltas, size=(5000, len(deltas)), replace=True).mean(axis=1)
    by_dataset = {}
    for dataset_id in CORE_DATASETS:
        local = np.asarray(
            [row["dataset"] == dataset_id for row in records],
            dtype=bool,
        )
        local_deltas = deltas[local]
        by_dataset[dataset_id] = {
            "participants": int(np.sum(local)),
            "mean_balanced_accuracy_change": float(np.mean(local_deltas)),
            "participants_improved_fraction": float(np.mean(local_deltas > 0.0)),
        }
    return {
        "participants": len(records),
        "mean_balanced_accuracy_change": float(np.mean(deltas)),
        "participant_bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "participants_improved_fraction": float(np.mean(deltas > 0.0)),
        "by_dataset": by_dataset,
        "records": records,
    }


def _plot_method_summary(
    path: Path,
    results: Mapping[str, object],
    calibrated_report: Mapping[str, object],
) -> None:
    methods = tuple(
        name
        for name in (
            "fbcsp",
            "riemann",
            "mrcp",
            "smr",
            "gsp",
            "late_fusion",
            "graph_late_fusion",
            "shrinkage_lda",
            "auxiliary_direction_intent",
            "graph_auxiliary_direction_intent",
            "shallow_convnet",
            "mrcp_fbcnet",
        )
        if name in results
    )
    method_labels = {
        "fbcsp": "FBCSP",
        "riemann": "Riemann",
        "mrcp": "MRCP",
        "smr": "SMR",
        "gsp": "Graph spectrum",
        "late_fusion": "Late fusion",
        "graph_late_fusion": "Graph-enhanced\nlate fusion",
        "shrinkage_lda": "Weighted\nshrinkage LDA",
        "auxiliary_direction_intent": "Auxiliary direction\ncollapsed to intent",
        "graph_auxiliary_direction_intent": "Graph + auxiliary\ncollapsed to intent",
        "shallow_convnet": "ShallowConvNet\n5-seed ensemble",
        "mrcp_fbcnet": "MRCP + FBCNet\n5-seed ensemble",
    }
    labels = tuple(method_labels[name] for name in methods) + (
        "Calibrated stage-specific",
    )
    metrics = (
        ("intent_balanced_accuracy", "Intent BA", COLORS["navy"]),
        ("intent_sensitivity", "Sensitivity", COLORS["teal"]),
        ("intent_specificity", "Specificity", COLORS["gold"]),
    )
    positions = np.arange(len(labels))
    width = 0.23
    horizontal = len(labels) >= 9
    figure, axis = plt.subplots(
        figsize=(11.5, max(7.2, 0.62 * len(labels)))
        if horizontal
        else (max(12, 1.15 * len(labels)), 6.6)
    )
    for offset, (metric, label, color) in enumerate(metrics):
        values = [
            100.0
            * float(
                results[method][DEVELOPMENT_OBSERVED_ROLE][metric]
            )
            for method in methods
        ]
        values.append(
            100.0
            * float(
                calibrated_report["operating_points"]["conservative"][
                    DEVELOPMENT_OBSERVED_ROLE
                ][metric]
            )
        )
        if horizontal:
            bars = axis.barh(
                positions + (offset - 1) * width,
                values,
                width,
                color=color,
                label=label,
            )
        else:
            bars = axis.bar(
                positions + (offset - 1) * width,
                values,
                width,
                color=color,
                label=label,
            )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    if horizontal:
        axis.set_yticks(
            positions,
            [label.replace("\n", " ") for label in labels],
        )
        axis.set_xlim(0, 104)
        axis.set_xlabel("Development-observed result (%)")
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    else:
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Development-observed result (%)")
        axis.grid(axis="y", alpha=0.2)
    axis.set_title(
        "Decoder and Calibrated Controller Comparison",
        fontsize=16,
        fontweight="bold",
        pad=34,
    )
    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        fontsize=10.5,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _diagnostic_sample(
    labels: np.ndarray,
    domains: np.ndarray,
    mask: np.ndarray,
    *,
    rows_per_group: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    binary = np.asarray(labels) != 0
    domains = np.asarray(domains).astype(str)
    for domain in CORE_DATASETS:
        for intent in (False, True):
            candidates = np.flatnonzero(mask & (domains == domain) & (binary == intent))
            if len(candidates) > rows_per_group:
                candidates = rng.choice(
                    candidates,
                    size=rows_per_group,
                    replace=False,
                )
            selected.append(np.asarray(candidates, dtype=np.int64))
    return np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)


def _plot_graph_signal_audit(
    path: Path,
    dataset: TranslationalWindowDataset,
    features: TranslationalFeatures,
    observed: np.ndarray,
) -> None:
    metadata = features.metadata.get("gsp")
    if not metadata or "gsp" not in features.branches:
        return
    adjacency = np.asarray(metadata["adjacency"], dtype=np.float64)
    coordinates = np.asarray(metadata["coordinates_m"], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.diag(np.sum(adjacency, axis=1)) - adjacency
    )
    channel_names = tuple(metadata["channel_names"])

    figure = plt.figure(figsize=(18.0, 6.4))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.25, 1.2),
        left=0.035,
        right=0.96,
        bottom=0.19,
        top=0.82,
        wspace=0.52,
    )

    graph_axis = figure.add_subplot(grid[0, 0])
    maximum = float(np.max(adjacency))
    for first in range(len(channel_names)):
        for second in range(first + 1, len(channel_names)):
            weight = float(adjacency[first, second])
            if weight <= 0.0:
                continue
            graph_axis.plot(
                coordinates[[first, second], 0],
                coordinates[[first, second], 1],
                color=COLORS["gray"],
                linewidth=1.0 + 4.0 * weight / maximum,
                alpha=0.25 + 0.55 * weight / maximum,
                zorder=1,
            )
    graph_axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=900,
        color=COLORS["pale"],
        edgecolor=COLORS["navy"],
        linewidth=2.2,
        zorder=2,
    )
    for index, name in enumerate(channel_names):
        graph_axis.text(
            coordinates[index, 0],
            coordinates[index, 1],
            name.replace("CZ", "Cz"),
            ha="center",
            va="center",
            color=COLORS["navy"],
            fontsize=12,
            fontweight="bold",
            zorder=3,
        )
    padding = 0.014
    graph_axis.set_xlim(
        float(np.min(coordinates[:, 0])) - padding,
        float(np.max(coordinates[:, 0])) + padding,
    )
    graph_axis.set_ylim(
        float(np.min(coordinates[:, 1])) - padding,
        float(np.max(coordinates[:, 1])) + padding,
    )
    graph_axis.set_title("A. Fixed Electrode Graph", fontsize=15)
    graph_axis.set_aspect("equal")
    graph_axis.axis("off")

    spectrum_axis = figure.add_subplot(grid[0, 1])
    image = spectrum_axis.imshow(
        eigenvectors,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-float(np.max(np.abs(eigenvectors))),
        vmax=float(np.max(np.abs(eigenvectors))),
    )
    spectrum_axis.set_yticks(np.arange(len(channel_names)), channel_names)
    spectrum_axis.set_xticks(
        np.arange(len(eigenvalues)),
        [f"g{index}\n{value:.2f}" for index, value in enumerate(eigenvalues)],
    )
    spectrum_axis.set_xlabel("Graph mode / eigenvalue")
    spectrum_axis.set_ylabel("Electrode")
    spectrum_axis.set_title("B. Graph Fourier Basis", fontsize=14)
    figure.colorbar(
        image,
        ax=spectrum_axis,
        fraction=0.05,
        pad=0.025,
        shrink=0.86,
        label="Mode weight",
    )

    delta_axis = figure.add_subplot(grid[0, 2])
    graph_values = features.branches["gsp"]
    graph_names = features.names["gsp"]
    groups = ("low", "mid", "high")
    differences = np.zeros((len(CORE_DATASETS), len(groups)), dtype=np.float64)
    for domain_index, domain in enumerate(CORE_DATASETS):
        local = observed & (dataset.dataset_ids == domain)
        rest = local & (dataset.labels == 0)
        intent = local & (dataset.labels != 0)
        for group_index, group in enumerate(groups):
            columns = [
                index
                for index, name in enumerate(graph_names)
                if name.endswith(f"{group}_energy_fraction")
            ]
            differences[domain_index, group_index] = 100.0 * (
                float(np.mean(graph_values[intent][:, columns]))
                - float(np.mean(graph_values[rest][:, columns]))
            )
    limit = max(float(np.max(np.abs(differences))), 0.5)
    heatmap = delta_axis.imshow(
        differences,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    delta_axis.set_xticks(np.arange(3), ("Low", "Mid", "High"))
    delta_axis.set_yticks(
        np.arange(3),
        ("PhysioNet", "OpenBMI", "Liu stroke"),
    )
    for row in range(differences.shape[0]):
        for column in range(differences.shape[1]):
            delta_axis.text(
                column,
                row,
                f"{differences[row, column]:+.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
                fontweight="bold",
            )
    delta_axis.set_xlabel("Graph-frequency energy band")
    delta_axis.set_title("C. Intent Minus Rest Energy", fontsize=14)
    figure.colorbar(
        heatmap,
        ax=delta_axis,
        fraction=0.05,
        pad=0.025,
        shrink=0.86,
        label="Mean energy change (percentage points)",
    )
    figure.suptitle(
        "V7 Fixed-Geometry Graph Signal Audit",
        fontsize=18,
        fontweight="bold",
        color=COLORS["navy"],
        y=0.96,
    )
    figure.text(
        0.5,
        0.04,
        "Edges use fixed 10-10 geometry; no held-out EEG or labels define the graph.",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_representation_diagnostics(
    path: Path,
    dataset: TranslationalWindowDataset,
    features: TranslationalFeatures,
    probabilities: Mapping[str, StageProbabilities],
    masks: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> None:
    if "gsp" not in features.branches or "shrinkage_lda" not in probabilities:
        return
    development = masks[DEVELOPMENT_ROLE] & core_mask(dataset)
    observed = masks[DEVELOPMENT_OBSERVED_ROLE] & core_mask(dataset)
    development_rows = _diagnostic_sample(
        dataset.labels,
        dataset.dataset_ids,
        development,
        rows_per_group=1000,
        seed=seed,
    )
    observed_rows = _diagnostic_sample(
        dataset.labels,
        dataset.dataset_ids,
        observed,
        rows_per_group=350,
        seed=seed + 1,
    )
    values = features.branches["gsp"]
    scaler = StandardScaler().fit(values[development_rows])
    development_scaled = scaler.transform(values[development_rows])
    observed_scaled = scaler.transform(values[observed_rows])
    component_count = min(10, observed_scaled.shape[1])
    pca = PCA(n_components=component_count, random_state=seed).fit(
        development_scaled
    )
    projection = pca.transform(observed_scaled)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=seed).fit(
        development_scaled
    )
    clusters = kmeans.predict(scaler.transform(values[observed]))

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    scatter_axis = axes[0, 0]
    markers = {"physionet": "o", "openbmi": "s", "liu2024": "^"}
    domain_labels = {
        "physionet": "PhysioNet",
        "openbmi": "OpenBMI",
        "liu2024": "Liu stroke",
    }
    sampled_labels = dataset.labels[observed_rows] != 0
    sampled_domains = dataset.dataset_ids[observed_rows]
    for domain in CORE_DATASETS:
        for intent, class_label, color in (
            (False, "REST", COLORS["gray"]),
            (True, "INTENT", COLORS["teal"]),
        ):
            selected = (sampled_domains == domain) & (sampled_labels == intent)
            scatter_axis.scatter(
                projection[selected, 0],
                projection[selected, 1],
                s=16,
                marker=markers[domain],
                color=color,
                alpha=0.38,
                linewidths=0,
                label=f"{domain_labels[domain]} {class_label}",
            )
    scatter_axis.axhline(0.0, color="#B8C1C7", linewidth=0.8)
    scatter_axis.axvline(0.0, color="#B8C1C7", linewidth=0.8)
    scatter_axis.set_xlabel("Development-fitted graph PC1")
    scatter_axis.set_ylabel("Development-fitted graph PC2")
    scatter_axis.set_title("A. Later-Recording PCA Projection", fontsize=14)
    scatter_axis.legend(frameon=False, fontsize=8, ncol=2)

    variance_axis = axes[0, 1]
    explained = 100.0 * pca.explained_variance_ratio_
    bars = variance_axis.bar(
        np.arange(1, len(explained) + 1),
        explained,
        color=COLORS["blue"],
    )
    variance_axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
    variance_axis.set_xlabel("Graph-feature principal component")
    variance_axis.set_ylabel("Development variance explained (%)")
    variance_axis.set_title("B. PCA Variance Spectrum", fontsize=14)
    variance_axis.grid(axis="y", alpha=0.2)

    probability_axis = axes[1, 0]
    probability_values = probabilities["shrinkage_lda"].intent
    boxes = []
    box_labels = []
    box_colors = []
    for domain in CORE_DATASETS:
        local = observed & (dataset.dataset_ids == domain)
        for intent, suffix, color in (
            (False, "R", COLORS["gray"]),
            (True, "I", COLORS["teal"]),
        ):
            boxes.append(probability_values[local & ((dataset.labels != 0) == intent)])
            box_labels.append(f"{domain_labels[domain]}\n{suffix}")
            box_colors.append(color)
    artists = probability_axis.boxplot(
        boxes,
        patch_artist=True,
        tick_labels=box_labels,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.8},
    )
    for patch, color in zip(artists["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    probability_axis.set_ylim(0.0, 1.0)
    probability_axis.set_ylabel("Shrinkage-LDA P(INTENT)")
    probability_axis.set_title("C. LDA Score Separation", fontsize=14)
    probability_axis.grid(axis="y", alpha=0.2)

    cluster_axis = axes[1, 1]
    observed_labels = dataset.labels[observed] != 0
    observed_domains = dataset.dataset_ids[observed]
    composition = np.zeros((3, 6), dtype=np.float64)
    column = 0
    for domain in CORE_DATASETS:
        for intent in (False, True):
            group = (observed_domains == domain) & (observed_labels == intent)
            for cluster in range(3):
                composition[cluster, column] = np.mean(clusters[group] == cluster)
            column += 1
    cluster_image = cluster_axis.imshow(
        100.0 * composition,
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        aspect="auto",
    )
    cluster_axis.set_xticks(
        np.arange(6),
        ("P-R", "P-I", "O-R", "O-I", "L-R", "L-I"),
    )
    cluster_axis.set_yticks(np.arange(3), ("Cluster 1", "Cluster 2", "Cluster 3"))
    for row in range(3):
        for col in range(6):
            cluster_axis.text(
                col,
                row,
                f"{100.0 * composition[row, col]:.0f}%",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )
    cluster_axis.set_title("D. Unsupervised K-Means Diagnostic", fontsize=14)
    figure.colorbar(
        cluster_image,
        ax=cluster_axis,
        fraction=0.046,
        pad=0.04,
        label="Rows assigned to cluster (%)",
    )

    figure.suptitle(
        "V7 Graph Representation Diagnostics",
        fontsize=19,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.text(
        0.5,
        0.01,
        "PCA, scaling, and K-means were fitted on development participants only; later recordings are displayed. K-means is diagnostic, not a decoder.",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_safety_frontier(
    path: Path,
    dataset: TranslationalWindowDataset,
    values: StageProbabilities,
    available: np.ndarray,
    controller: np.ndarray,
    operating_points: Mapping[str, object],
    target_sensitivity: float = 0.60,
) -> None:
    selected = controller & core_mask(dataset)
    true_intent = dataset.labels[selected] != 0
    figure, axis = plt.subplots(figsize=(8.8, 6.2))
    thresholds = np.linspace(0.05, 0.95, 181)
    false_activation = []
    sensitivity = []
    for threshold in thresholds:
        predicted = available[selected] & (values.intent[selected] >= threshold)
        false_activation.append(100.0 * np.mean(predicted[~true_intent]))
        sensitivity.append(100.0 * np.mean(predicted[true_intent]))
    axis.plot(false_activation, sensitivity, color=COLORS["navy"], linewidth=2.4)
    for name, color, marker in (
        ("conservative", COLORS["teal"], "o"),
        ("sensitivity_targeted", COLORS["orange"], "s"),
    ):
        metrics = operating_points[name]["controller_policy"]
        axis.scatter(
            100.0 * float(metrics["false_activation_rate_per_window"]),
            100.0 * float(metrics["intent_sensitivity"]),
            s=80,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
            label=name.replace("_", " ").title(),
        )
    axis.axvline(10.0, color=COLORS["teal"], linestyle="--", label="10% window fallback")
    axis.axhline(
        100.0 * target_sensitivity,
        color=COLORS["gold"],
        linestyle="--",
        label=f"{100 * target_sensitivity:.0f}% sensitivity target",
    )
    axis.set(
        xlabel="False activation per cue-locked rest window (%)",
        ylabel="Detected motor-intent windows (%)",
        title="Controller Safety-Sensitivity Frontier",
        xlim=(0, min(100, max(30, max(false_activation)))),
        ylim=(0, 100),
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_calibration(
    path: Path,
    report: Mapping[str, object],
) -> None:
    budgets = (2, 5, 10)
    intent_only = report.get("classes") == ["rest", "intent"]
    auxiliary = bool(report.get("auxiliary_training_labels"))
    budget_key = (
        "budget_per_source_class"
        if auxiliary
        else "budget_per_binary_class"
        if intent_only
        else "budget_per_class"
    )
    figure, axis = plt.subplots(figsize=(8.8, 5.8))
    for dataset_id, color in zip(
        CORE_DATASETS,
        (COLORS["navy"], COLORS["blue"], COLORS["orange"]),
    ):
        rows = report["by_dataset"][dataset_id][
            "all_controller_configurations"
        ]
        values = [
            100.0
            * max(
                float(
                    row["controller_conservative"][
                        "intent_balanced_accuracy"
                    ]
                )
                for row in rows
                if int(row[budget_key]) == budget
            )
            for budget in budgets
        ]
        axis.plot(
            budgets,
            values,
            marker="o",
            linewidth=2,
            color=color,
            label=dataset_id,
        )
    axis.set(
        xlabel=(
            "Calibration windows per source class (3 x total)"
            if auxiliary
            else
            "Calibration windows per binary class (2 x total)"
            if intent_only
            else "Calibration trials per class"
        ),
        ylabel="Controller intent balanced accuracy (%)",
        title=(
            "Auxiliary-Direction Patient-Calibration Burden"
            if auxiliary
            else
            "Intent-Only Patient-Calibration Burden"
            if intent_only
            else "Patient-Calibration Burden by Dataset"
        ),
        ylim=(0, 100),
    )
    axis.set_xticks(budgets)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_domain_calibration(
    path: Path,
    dataset: TranslationalWindowDataset,
    raw: StageProbabilities,
    calibrated: StageProbabilities,
    mask: np.ndarray,
) -> None:
    figure, axes = plt.subplots(
        1,
        len(CORE_DATASETS),
        figsize=(12, 4.2),
        sharex=True,
        sharey=True,
    )
    bins = np.linspace(0.0, 1.0, 9)
    for axis, dataset_id in zip(axes, CORE_DATASETS):
        selected = mask & (dataset.dataset_ids == dataset_id)
        labels = dataset.labels[selected] != 0
        for values, label, color in (
            (raw.intent[selected], "Before", COLORS["gray"]),
            (calibrated.intent[selected], "After", COLORS["teal"]),
        ):
            means = []
            observed = []
            for low, high in zip(bins[:-1], bins[1:]):
                local = (values >= low) & (
                    values <= high if high == 1.0 else values < high
                )
                if np.any(local):
                    means.append(float(np.mean(values[local])))
                    observed.append(float(np.mean(labels[local])))
            axis.plot(
                means,
                observed,
                marker="o",
                linewidth=1.8,
                color=color,
                label=label,
            )
        axis.plot((0, 1), (0, 1), linestyle="--", color="#AAB3B8")
        axis.set_title(dataset_id)
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed intent fraction")
    axes[-1].legend(frameon=False)
    figure.suptitle(
        "Per-Dataset Intent Probability Calibration",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_operating_point_comparison(
    path: Path,
    calibrated_report: Mapping[str, object],
    patient_report: Mapping[str, object],
) -> None:
    sources = (
        (
            "Population\nconservative",
            calibrated_report["operating_points"]["conservative"][
                DEVELOPMENT_OBSERVED_ROLE
            ],
        ),
        (
            "Population\nsensitivity-targeted",
            calibrated_report["operating_points"][
                "sensitivity_targeted"
            ][DEVELOPMENT_OBSERVED_ROLE],
        ),
        (
            "Patient-adapted\nconservative",
            patient_report["development_observed_conservative"],
        ),
        (
            "Patient-adapted\nsensitivity-targeted",
            patient_report[
                "development_observed_sensitivity_targeted"
            ],
        ),
    )
    x = np.arange(len(sources))
    width = 0.34
    sensitivity = [
        100.0 * float(metrics["intent_sensitivity"])
        for _, metrics in sources
    ]
    specificity = [
        100.0 * float(metrics["intent_specificity"])
        for _, metrics in sources
    ]
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    first = axis.bar(
        x - width / 2,
        sensitivity,
        width,
        color=COLORS["teal"],
        label="Sensitivity",
    )
    second = axis.bar(
        x + width / 2,
        specificity,
        width,
        color=COLORS["gold"],
        label="Specificity",
    )
    axis.bar_label(first, fmt="%.1f%%", padding=2)
    axis.bar_label(second, fmt="%.1f%%", padding=2)
    axis.set_xticks(x, [label for label, _ in sources])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Development-observed result (%)")
    axis.set_title("Operating-Point and Calibration Tradeoffs")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_neural_integration_summary(
    path: Path,
    report: Mapping[str, object],
) -> None:
    methods = report["methods"]
    calibrated = report["calibrated_stage_controller"]["operating_points"][
        "conservative"
    ][DEVELOPMENT_OBSERVED_ROLE]
    patient = report["patient_calibration"]["development_observed_conservative"]
    baseline_patient = report["neural_integration_comparison"][
        "physiology_only_patient_adapted_conservative"
    ]
    configurations = [("Physiology-only\nadapted", baseline_patient)]
    if "shallow_convnet" in methods:
        configurations.append(
            (
                "ShallowConvNet",
                methods["shallow_convnet"][DEVELOPMENT_OBSERVED_ROLE],
            )
        )
    configurations.extend(
        (
            (
                "MRCP + FBCNet",
                methods["mrcp_fbcnet"][DEVELOPMENT_OBSERVED_ROLE],
            ),
            ("Population\ncalibrated", calibrated),
            ("Patient\nadapted", patient),
        )
    )
    metric_rows = (
        ("intent_balanced_accuracy", "Balanced accuracy", COLORS["navy"]),
        ("intent_sensitivity", "Sensitivity", COLORS["teal"]),
        ("intent_specificity", "Specificity", COLORS["gold"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 6.2), constrained_layout=True)
    x = np.arange(len(configurations))
    width = 0.24
    for offset, (metric, label, color) in enumerate(metric_rows):
        values = [100.0 * float(result[metric]) for _, result in configurations]
        bars = axes[0].bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_xticks(x, [name for name, _ in configurations])
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Development-observed windows (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Integration Improves Intent Recovery", loc="left")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    domain_sources = [
        (
            "Physiology-only + patient adaptation",
            baseline_patient["by_dataset"],
            COLORS["gray"],
        ),
    ]
    if "shallow_convnet" in methods:
        domain_sources.append(
            (
                "ShallowConvNet",
                methods["shallow_convnet"][
                    f"{DEVELOPMENT_OBSERVED_ROLE}_by_dataset"
                ],
                COLORS["blue"],
            )
        )
    domain_sources.extend(
        (
            (
                "MRCP + FBCNet",
                methods["mrcp_fbcnet"][
                    f"{DEVELOPMENT_OBSERVED_ROLE}_by_dataset"
                ],
                COLORS["navy"],
            ),
            ("Integrated calibrated", calibrated["by_dataset"], COLORS["teal"]),
            (
                "Integrated + patient adaptation",
                {
                    dataset_id: report["patient_calibration"]["by_dataset"][dataset_id][
                        "development_observed_conservative"
                    ]
                    for dataset_id in CORE_DATASETS
                },
                COLORS["orange"],
            ),
        )
    )
    domain_x = np.arange(len(CORE_DATASETS))
    domain_width = 0.8 / len(domain_sources)
    for offset, (label, values, color) in enumerate(domain_sources):
        scores = [
            100.0 * float(values[dataset_id]["intent_balanced_accuracy"])
            for dataset_id in CORE_DATASETS
        ]
        axes[1].bar(
            domain_x + (offset - (len(domain_sources) - 1) / 2) * domain_width,
            scores,
            domain_width,
            label=label,
            color=color,
        )
    axes[1].axhline(50.0, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[1].set_xticks(domain_x, CORE_DATASETS)
    axes[1].set_ylabel("Intent balanced accuracy (%)")
    axes[1].set_ylim(40, 80)
    axes[1].set_title("Cross-Dataset Effect", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Neural Ensemble Fusion with Rapid Patient Adaptation",
        fontsize=16,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_physiology(
    path: Path,
    dataset: TranslationalWindowDataset,
    mask: np.ndarray,
) -> None:
    channels = {name.upper(): index for index, name in enumerate(dataset.channel_names)}
    central = [channels[name] for name in ("C3", "CZ", "C4") if name in channels]
    times = np.arange(dataset.mrcp_epochs.shape[-1]) / dataset.sfreq
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for dataset_id, color in zip(CORE_DATASETS, (COLORS["navy"], COLORS["blue"], COLORS["orange"])):
        local = mask & (dataset.dataset_ids == dataset_id)
        intent = local & (dataset.labels != 0)
        rest = local & (dataset.labels == 0)
        difference = (
            np.mean(dataset.mrcp_epochs[intent][:, central], axis=(0, 1))
            - np.mean(dataset.mrcp_epochs[rest][:, central], axis=(0, 1))
        ) * 1e6
        axes[0].plot(times, difference, linewidth=2, color=color, label=dataset_id)
        frequencies, intent_psd = welch(
            dataset.smr_epochs[intent][:, central],
            fs=dataset.sfreq,
            nperseg=dataset.smr_epochs.shape[-1],
            axis=-1,
        )
        _, rest_psd = welch(
            dataset.smr_epochs[rest][:, central],
            fs=dataset.sfreq,
            nperseg=dataset.smr_epochs.shape[-1],
            axis=-1,
        )
        erd = 10.0 * np.log10(
            np.clip(np.mean(intent_psd, axis=(0, 1)), 1e-30, None)
            / np.clip(np.mean(rest_psd, axis=(0, 1)), 1e-30, None)
        )
        shown = (frequencies >= 7) & (frequencies <= 35)
        axes[1].plot(frequencies[shown], erd[shown], linewidth=2, color=color, label=dataset_id)
    axes[0].axhline(0, color=COLORS["gray"], linewidth=0.8)
    axes[0].set(
        title="Central MRCP: Intent - Rest",
        xlabel="Time after analysis-window onset (s)",
        ylabel="Amplitude difference (uV)",
    )
    axes[1].axhline(0, color=COLORS["gray"], linewidth=0.8)
    axes[1].axvspan(8, 13, color=COLORS["gold"], alpha=0.13, label="Mu")
    axes[1].axvspan(13, 30, color=COLORS["teal"], alpha=0.10, label="Beta")
    axes[1].set(
        title="Central SMR: Intent / Rest",
        xlabel="Frequency (Hz)",
        ylabel="Power change (dB)",
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    figure.suptitle("Development-Observed Physiology Audit", fontsize=15, fontweight="bold")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_quality(
    path: Path,
    dataset: TranslationalWindowDataset,
    available: np.ndarray,
) -> None:
    datasets = np.unique(dataset.dataset_ids)
    retained = [
        100.0 * float(np.mean(available[dataset.dataset_ids == value]))
        for value in datasets
    ]
    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    bars = axis.bar(
        datasets,
        retained,
        color=(COLORS["navy"], COLORS["blue"], COLORS["orange"], COLORS["gray"])[: len(datasets)],
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=3, fontweight="bold")
    axis.set(
        title="Quality-Gate Availability by Dataset",
        ylabel="Windows available to decoder (%)",
        ylim=(0, 105),
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_v5_v6_comparison(
    path: Path,
    v5: Mapping[str, object],
    v6: Mapping[str, object],
) -> None:
    sources = (
        ("V5 directional\nconservative", v5["conservative"]),
        ("V6 intent-only\nconservative", v6["conservative"]),
        ("V5 directional\ntargeted", v5["sensitivity_targeted"]),
        ("V6 intent-only\ntargeted", v6["sensitivity_targeted"]),
    )
    metrics = (
        ("intent_balanced_accuracy", "Balanced accuracy", COLORS["navy"]),
        ("intent_sensitivity", "Sensitivity", COLORS["teal"]),
        ("intent_specificity", "Specificity", COLORS["gold"]),
    )
    x = np.arange(len(sources))
    width = 0.23
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    for offset, (metric, label, color) in enumerate(metrics):
        values = [100.0 * float(record[metric]) for _, record in sources]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_xticks(x, [label for label, _ in sources])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Development-observed result (%)")
    axis.set_title("Directional versus Intent-Only Patient Adaptation")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _plot_v5_v6_v7_comparison(
    path: Path,
    v5: Mapping[str, object],
    v6: Mapping[str, object],
    v7: Mapping[str, object],
) -> None:
    sources = (
        ("V5\ndirectional", v5["conservative"]),
        ("V6\nintent-only", v6["conservative"]),
        ("V7\nauxiliary", v7["conservative"]),
        ("V5\ntargeted", v5["sensitivity_targeted"]),
        ("V6\ntargeted", v6["sensitivity_targeted"]),
        ("V7\ntargeted", v7["sensitivity_targeted"]),
    )
    metrics = (
        ("intent_balanced_accuracy", "Balanced accuracy", COLORS["navy"]),
        ("intent_sensitivity", "Sensitivity", COLORS["teal"]),
        ("intent_specificity", "Specificity", COLORS["gold"]),
    )
    x = np.arange(len(sources))
    width = 0.23
    figure, axis = plt.subplots(figsize=(13.2, 6.2))
    for offset, (metric, label, color) in enumerate(metrics):
        values = [100.0 * float(record[metric]) for _, record in sources]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            color=color,
            label=label,
        )
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.axvline(2.5, color=COLORS["gray"], linewidth=1, alpha=0.5)
    axis.set_xticks(x, [label for label, _ in sources])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Development-observed result (%)")
    axis.set_title("Patient-Adaptation Model Version Comparison")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _load_v5_reference(output: Path) -> dict[str, object] | None:
    path = (
        output.parent
        / "translational_v5_patient_adaptive"
        / "translational_v5_patient_adaptive_metrics.json"
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="ascii"))
    patient = payload["patient_calibration"]
    return {
        "source": str(path),
        "conservative": patient["development_observed_conservative"],
        "sensitivity_targeted": patient[
            "development_observed_sensitivity_targeted"
        ],
        "calibration_configurations": patient["selected_configurations"],
    }


def _load_v6_reference(output: Path) -> dict[str, object] | None:
    path = (
        output.parent
        / "translational_v6_intent_only"
        / "translational_v6_intent_only_metrics.json"
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="ascii"))
    patient = payload["patient_calibration"]
    return {
        "source": str(path),
        "conservative": patient["development_observed_conservative"],
        "sensitivity_targeted": patient[
            "development_observed_sensitivity_targeted"
        ],
        "calibration_configurations": patient["selected_configurations"],
    }


def write_report(path: Path, report: Mapping[str, object]) -> None:
    fusion = report["methods"]["late_fusion"][DEVELOPMENT_OBSERVED_ROLE]
    neural_enabled = "mrcp_fbcnet" in report["methods"]
    multi_neural_enabled = "shallow_convnet" in report["methods"]
    neural = (
        report["methods"]["mrcp_fbcnet"][DEVELOPMENT_OBSERVED_ROLE]
        if neural_enabled
        else None
    )
    shallow = (
        report["methods"]["shallow_convnet"][DEVELOPMENT_OBSERVED_ROLE]
        if multi_neural_enabled
        else None
    )
    neural_comparison = report.get("neural_integration_comparison")
    paired = (
        neural_comparison["paired_patient_analysis"]
        if neural_comparison is not None
        else None
    )
    metrics_filename = (
        "translational_v5_patient_adaptive_metrics.json"
        if multi_neural_enabled
        else "translational_v4_neural_metrics.json"
        if neural_enabled
        else "translational_v3_metrics.json"
    )
    calibrated = report["calibrated_stage_controller"]
    conservative = calibrated["operating_points"]["conservative"][
        DEVELOPMENT_OBSERVED_ROLE
    ]
    sensitivity_targeted = calibrated["operating_points"][
        "sensitivity_targeted"
    ][DEVELOPMENT_OBSERVED_ROLE]
    patient = report["patient_calibration"]
    patient_conservative = patient["development_observed_conservative"]
    patient_sensitivity = patient[
        "development_observed_sensitivity_targeted"
    ]
    quality_control = report["dataset"]["quality_control"]
    confidence = report["bootstrap_confidence_intervals"]["population"][
        "conservative"
    ]["overall"]
    ba_interval = confidence["intent_balanced_accuracy"]
    sensitivity_interval = confidence["intent_sensitivity"]
    legacy_ba = 100 * float(fusion["intent_balanced_accuracy"])
    liu_sensitive_specificity = 100 * float(
        sensitivity_targeted["by_dataset"]["liu2024"][
            "intent_specificity"
        ]
    )
    dataset_rows = []
    sensitivity_dataset_rows = []
    for dataset_id in CORE_DATASETS:
        population = conservative["by_dataset"][dataset_id]
        population_sensitive = sensitivity_targeted["by_dataset"][dataset_id]
        adapted = patient["by_dataset"][dataset_id][
            "development_observed_conservative"
        ]
        dataset_rows.append(
            "| "
            f"{dataset_id} | "
            f"{100 * float(population['intent_balanced_accuracy']):.1f}% | "
            f"{100 * float(population['intent_sensitivity']):.1f}% | "
            f"{100 * float(population['intent_specificity']):.1f}% | "
            f"{100 * float(adapted['intent_balanced_accuracy']):.1f}% | "
            f"{100 * float(adapted['intent_sensitivity']):.1f}% | "
            f"{100 * float(adapted['intent_specificity']):.1f}% |"
        )
        sensitivity_dataset_rows.append(
            "| "
            f"{dataset_id} | "
            f"{100 * float(population_sensitive['intent_sensitivity']):.1f}% | "
            f"{100 * float(population_sensitive['intent_specificity']):.1f}% | "
            f"{100 * float(population_sensitive['intent_balanced_accuracy']):.1f}% |"
        )
    intent_blends = {
        key: float(value)
        for key, value in calibrated["probability_calibration"][
            "intent_blends"
        ].items()
    }
    lines = [
        (
            "# Patient-Adaptive Neural-Fusion Motor-Intent Pipeline v5"
            if multi_neural_enabled
            else "# Neural-Integrated Translational Motor-Intent Pipeline v4"
            if neural_enabled
            else "# Calibrated Translational Motor-Intent Pipeline v3"
        ),
        "",
        "## Status",
        "",
        "- The previous final cohort is now labeled `development_observed`.",
        "- No confirmatory accuracy is reported because `locked_external` is empty.",
        "- Fusion and Platt-model fitting use controller subjects disjoint from "
        "the controller subjects used to select calibration shrinkage and "
        "operating points.",
        "- Thresholds are dataset-specific and selected without using the "
        "development-observed cohort.",
        *(
            [
                "- Neural architecture selection previously observed the development-observed cohort; this integrated result is exploratory and cannot be relabeled as external validation.",
                "- Each neural architecture is averaged across five seeds before controller-only fusion, calibration, patient adaptation, quality rejection, and temporal confirmation.",
            ]
            if neural_enabled
            else []
        ),
        "",
        "## Development-Observed Results",
        "",
        f"- Legacy late-fusion BA: **{legacy_ba:.1f}%**.",
        *(
            [
                "- Raw ShallowConvNet seed-ensemble result: "
                f"**{100 * float(shallow['intent_balanced_accuracy']):.1f}% BA**, "
                f"**{100 * float(shallow['intent_sensitivity']):.1f}% sensitivity**, "
                f"**{100 * float(shallow['intent_specificity']):.1f}% specificity**."
            ]
            if multi_neural_enabled
            else []
        ),
        *(
            [
                "- Raw MRCP + FBCNet seed-ensemble result: "
                f"**{100 * float(neural['intent_balanced_accuracy']):.1f}% BA**, "
                f"**{100 * float(neural['intent_sensitivity']):.1f}% sensitivity**, "
                f"**{100 * float(neural['intent_specificity']):.1f}% specificity**."
            ]
            if neural_enabled
            else []
        ),
        "- Calibrated conservative BA: "
        f"**{100 * float(conservative['intent_balanced_accuracy']):.1f}%** "
        f"(subject-bootstrap 95% CI "
        f"{100 * ba_interval['lower_95']:.1f}-"
        f"{100 * ba_interval['upper_95']:.1f}%).",
        "- Calibrated conservative sensitivity: "
        f"**{100 * float(conservative['intent_sensitivity']):.1f}%** "
        f"(95% CI {100 * sensitivity_interval['lower_95']:.1f}-"
        f"{100 * sensitivity_interval['upper_95']:.1f}%).",
        "- Calibrated conservative specificity: "
        f"**{100 * float(conservative['intent_specificity']):.1f}%**.",
        "- Sensitivity-targeted population result: "
        f"**{100 * float(sensitivity_targeted['intent_sensitivity']):.1f}% "
        "sensitivity**, "
        f"**{100 * float(sensitivity_targeted['intent_specificity']):.1f}% "
        "specificity**.",
        "- Patient-adapted conservative result: "
        f"**{100 * float(patient_conservative['intent_balanced_accuracy']):.1f}% "
        "BA**, "
        f"**{100 * float(patient_conservative['intent_sensitivity']):.1f}% "
        "sensitivity**, "
        f"**{100 * float(patient_conservative['intent_specificity']):.1f}% "
        "specificity**.",
        *(
            [
                "- Paired equal-participant change versus the physiology-only patient-adapted pipeline: "
                f"**{100 * float(paired['mean_balanced_accuracy_change']):+.2f} points** "
                f"(95% participant-bootstrap interval **{100 * float(paired['participant_bootstrap_95_ci'][0]):+.2f} to "
                f"{100 * float(paired['participant_bootstrap_95_ci'][1]):+.2f}**); "
                f"**{100 * float(paired['participants_improved_fraction']):.1f}%** of participants improved."
            ]
            if paired is not None
            else []
        ),
        "- Patient-adapted sensitivity-targeted result: "
        f"**{100 * float(patient_sensitivity['intent_sensitivity']):.1f}% "
        "sensitivity**, "
        f"**{100 * float(patient_sensitivity['intent_specificity']):.1f}% "
        "specificity**.",
        "",
        (
            "The patient-adapted conservative operating point is the preferred integrated candidate, but it remains development-observed until a new unseen cohort confirms the gain. "
            if neural_enabled
            else "The patient-adapted conservative operating point is the current preferred cue-locked configuration because it improves intent recovery while retaining high specificity. "
        )
        + "The sensitivity-targeted population "
        "point is exploratory, not a device-command policy: its Liu specificity "
        f"is only **{liu_sensitive_specificity:.1f}%**.",
        *(
            [
                "",
                "PhysioNet patient-adapted specificity is "
                f"**{100 * float(patient['by_dataset']['physionet']['development_observed_conservative']['intent_specificity']):.1f}%**, "
                "so its observed false-activation rate exceeds the 10% controller target. This domain requires recalibration or stronger temporal confirmation before prospective use."
            ]
            if neural_enabled
            else []
        ),
        "",
        "All values are exploratory. They are not a new locked final result, and "
        "cue-locked specificity is not a continuous false-activation rate.",
        "",
        "## Per-Dataset Results",
        "",
        "| Dataset | Population BA | Population sensitivity | Population "
        "specificity | Adapted BA | Adapted sensitivity | Adapted specificity |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *dataset_rows,
        "",
        "### Sensitivity-Targeted Population Check",
        "",
        "| Dataset | Sensitivity | Specificity | Balanced accuracy |",
        "|---|---:|---:|---:|",
        *sensitivity_dataset_rows,
        "",
        "Subject-bootstrap intervals for the pooled and per-dataset endpoints "
        f"are stored in `{metrics_filename}`.",
        "",
        "## Controller Design",
        "",
        f"- Intent fusion weights: `{calibrated['fusion']['intent_weights']}`.",
        f"- Direction fusion weights: `{calibrated['fusion']['direction_weights']}`.",
        "- Probability calibration: per-dataset Platt scaling with "
        "controller-selected identity shrinkage.",
        f"- Intent calibration blends: `{intent_blends}`.",
        "- Patient calibration: early recording trials provide labeled rest, left, and right examples; later recording trials remain evaluation-only.",
        f"- Patient-adapter inputs: `{report['patient_calibration'].get('adapter_inputs', [])}`.",
        f"- Controller-selected calibration configurations: `{report['patient_calibration']['selected_configurations']}`.",
        *(
            [
                f"- Neural ensemble seeds: `{report['neural_integration']['seeds']}`.",
                f"- Neural ensemble architectures: `{report['neural_integration']['architectures']}`.",
                "- Neural inference remains EEG-only; FBCSP/Riemann physiology retains responsibility for conditional direction evidence.",
            ]
            if neural_enabled
            else []
        ),
        *(
            [
                "- Portable general-model bundle: "
                f"`{report['general_model_bundle']['directory']}` "
                f"with {report['general_model_bundle']['neural_model_count']} "
                "fingerprinted neural seed models and the rapid-calibration recipe."
            ]
            if report.get("general_model_bundle")
            else []
        ),
        "",
        "## Dataset Quality Control",
        "",
        f"- Accepted recordings: **{quality_control['accepted_recordings']}**.",
        "- Excluded recordings or subjects: "
        f"**{quality_control['excluded_recordings_or_subjects']}**.",
        "- Excluded subjects: "
        f"`{', '.join(quality_control['excluded_subjects']) or 'none'}`.",
        "",
        f"Full exclusion reasons are stored in `{metrics_filename}` "
        "and beside the window cache in `translational_windows_v3.qc.json`.",
        "",
        "## Implemented Changes",
        "",
        "1. Stage-specific intent and direction fusion.",
        "2. Per-dataset probability calibration and operating thresholds.",
        "3. Conservative and sensitivity-targeted operating points.",
        "4. Patient calibration for all three datasets.",
        "5. Subject-level bootstrap confidence intervals.",
        "6. Temporal confirmation and refractory logic prepared for continuous "
        "no-control recordings.",
        "7. Legacy late-fusion results retained as a fixed reference.",
        *(
            [
                "8. Five-seed neural intent proposals integrated through controller-only fusion and calibration.",
                "9. Population score, MRCP, SMR, and Riemann features used for rapid patient adaptation.",
                "10. Patient-adapted probabilities passed through the existing quality and temporal safety layers.",
            ]
            if neural_enabled
            else []
        ),
        "",
        "## Continuous No-Control Boundary",
        "",
        str(report["continuous_no_control_status"]),
        "",
        "Temporal confirmation status: "
        f"`{patient.get('temporal_confirmation', calibrated['temporal_confirmation'])['status']}`.",
        "",
        "A false-activations-per-minute claim is emitted only when a hard-negative "
        "manifest supplies explicitly continuous no-control recordings. Otherwise "
        "the script reports cue-locked window false activation and marks continuous "
        "safety as not evaluated.",
        "",
        "## Figures",
        "",
        "- `01_decoder_comparison.png`",
        "- `02_safety_sensitivity_frontier.png`",
        "- `03_patient_calibration_burden.png`",
        "- `04_physiology_audit.png`",
        "- `05_quality_gate_availability.png`",
        "- `06_domain_probability_calibration.png`",
        "- `07_operating_point_comparison.png`",
        *( ["- `08_neural_integration_summary.png`"] if neural_enabled else [] ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_intent_only_report(
    path: Path,
    report: Mapping[str, object],
) -> None:
    auxiliary = bool(report.get("auxiliary_direction", False))
    calibrated = report["calibrated_stage_controller"]
    population = calibrated["operating_points"]["conservative"][
        DEVELOPMENT_OBSERVED_ROLE
    ]
    population_sensitive = calibrated["operating_points"][
        "sensitivity_targeted"
    ][DEVELOPMENT_OBSERVED_ROLE]
    patient = report["patient_calibration"]
    matched = patient["population_matched_conservative"]
    adapted = patient["development_observed_conservative"]
    adapted_sensitive = patient["development_observed_sensitivity_targeted"]
    bootstrap = report["bootstrap_confidence_intervals"]["patient_adapted"][
        "conservative"
    ]["overall"]
    ba_interval = bootstrap["intent_balanced_accuracy"]
    sensitivity_interval = bootstrap["intent_sensitivity"]
    neural = report["methods"].get("mrcp_fbcnet", {}).get(
        DEVELOPMENT_OBSERVED_ROLE
    )
    shallow = report["methods"].get("shallow_convnet", {}).get(
        DEVELOPMENT_OBSERVED_ROLE
    )
    auxiliary_method = report["methods"].get(
        "auxiliary_direction_intent",
        {},
    ).get(DEVELOPMENT_OBSERVED_ROLE)
    graph_method = report["methods"].get("gsp", {}).get(
        DEVELOPMENT_OBSERVED_ROLE
    )
    lda_method = report["methods"].get("shrinkage_lda", {}).get(
        DEVELOPMENT_OBSERVED_ROLE
    )
    v7_extensions = report.get("v7_extensions") or {}
    paired = report["neural_integration_comparison"][
        "paired_patient_analysis"
    ]
    v5_reference = report.get("v5_directional_reference")
    v6_reference = report.get("v6_intent_only_reference")
    intent_blends = {
        key: float(value)
        for key, value in calibrated["probability_calibration"][
            "intent_blends"
        ].items()
    }
    rows = []
    for dataset_id in CORE_DATASETS:
        baseline = matched["by_dataset"][dataset_id]
        candidate = adapted["by_dataset"][dataset_id]
        config = patient["selected_configurations"][dataset_id]
        rows.append(
            "| "
            f"{dataset_id} | "
            f"{config['total_calibration_windows']} | "
            f"{100 * float(baseline['intent_balanced_accuracy']):.1f}% | "
            f"{100 * float(candidate['intent_balanced_accuracy']):.1f}% | "
            f"{100 * float(candidate['intent_sensitivity']):.1f}% | "
            f"{100 * float(candidate['intent_specificity']):.1f}% |"
        )
    lines = [
        (
            "# Auxiliary-Direction Binary Motor-Intent Pipeline v7"
            if auxiliary
            else "# Binary Rest-versus-Intent Patient-Adaptive Pipeline v6"
        ),
        "",
        "## Status",
        "",
        (
            "- The external output remains binary `rest` versus `intent`; left/right cue labels are used only by an internal auxiliary model."
            if auxiliary
            else "- Left and right imagery labels are merged into one `intent` class."
        ),
        (
            "- Auxiliary direction is collapsed to `P(left) + P(right)` and never exposed as a device command."
            if auxiliary
            else "- No direction decoder, direction calibration model, or direction metric is fitted."
        ),
        "- Recording 0 supplies patient calibration; all calibration rows are excluded from adapted evaluation.",
        "- Later recordings from the same participant are evaluation-only.",
        "- Fusion, calibration, adapter configuration, and thresholds use controller participants only.",
        *(
            [
                "- V7 graph/LDA extension: fixed 10-10 electrode geometry supplies graph features; no EEG-derived edge is fitted.",
                "- Weighted shrinkage LDA and graph models remain controller-selected candidates rather than mandatory fusion components.",
            ]
            if v7_extensions.get("enabled")
            else []
        ),
        "- The result remains development-observed because `locked_external` is empty.",
        "",
        "## Development-Observed Results",
        "",
        "- General population conservative: "
        f"**{100 * float(population['intent_balanced_accuracy']):.1f}% BA**, "
        f"**{100 * float(population['intent_sensitivity']):.1f}% sensitivity**, "
        f"**{100 * float(population['intent_specificity']):.1f}% specificity**.",
        "- General population sensitivity-targeted: "
        f"**{100 * float(population_sensitive['intent_balanced_accuracy']):.1f}% BA**, "
        f"**{100 * float(population_sensitive['intent_sensitivity']):.1f}% sensitivity**, "
        f"**{100 * float(population_sensitive['intent_specificity']):.1f}% specificity**.",
        "- Population model on the same later-recording rows and patient thresholds: "
        f"**{100 * float(matched['intent_balanced_accuracy']):.1f}% BA**.",
        "- Binary patient-adapted conservative: "
        f"**{100 * float(adapted['intent_balanced_accuracy']):.1f}% BA** "
        f"(95% CI {100 * ba_interval['lower_95']:.1f}-"
        f"{100 * ba_interval['upper_95']:.1f}%), "
        f"**{100 * float(adapted['intent_sensitivity']):.1f}% sensitivity** "
        f"(95% CI {100 * sensitivity_interval['lower_95']:.1f}-"
        f"{100 * sensitivity_interval['upper_95']:.1f}%), and "
        f"**{100 * float(adapted['intent_specificity']):.1f}% specificity**.",
        "- Binary patient-adapted sensitivity-targeted: "
        f"**{100 * float(adapted_sensitive['intent_balanced_accuracy']):.1f}% BA**, "
        f"**{100 * float(adapted_sensitive['intent_sensitivity']):.1f}% sensitivity**, "
        f"**{100 * float(adapted_sensitive['intent_specificity']):.1f}% specificity**.",
        *(
            [
                "- Raw auxiliary-direction intent model: "
                f"**{100 * float(auxiliary_method['intent_balanced_accuracy']):.1f}% BA**.",
            ]
            if auxiliary_method is not None
            else []
        ),
        *(
            [
                "- Raw fixed-geometry graph model: "
                f"**{100 * float(graph_method['intent_balanced_accuracy']):.1f}% BA**.",
            ]
            if graph_method is not None
            else []
        ),
        *(
            [
                "- Raw weighted shrinkage-LDA model: "
                f"**{100 * float(lda_method['intent_balanced_accuracy']):.1f}% BA**.",
            ]
            if lda_method is not None
            else []
        ),
        *(
            [
                "- Raw ShallowConvNet five-seed ensemble: "
                f"**{100 * float(shallow['intent_balanced_accuracy']):.1f}% BA**.",
            ]
            if shallow is not None
            else []
        ),
        *(
            [
                "- Raw MRCP + FBCNet five-seed ensemble: "
                f"**{100 * float(neural['intent_balanced_accuracy']):.1f}% BA**.",
            ]
            if neural is not None
            else []
        ),
        "- Neural integration changed equal-participant adapted BA by "
        f"**{100 * float(paired['mean_balanced_accuracy_change']):+.2f} points** "
        "versus physiology-only adaptation.",
        "",
        "The matched later-recording comparison isolates adaptation benefit using the same evaluation rows and operating thresholds.",
        *(
            [
                "",
                "## V5/V6 Reference Comparison",
                "",
                "- Conservative BA: "
                f"v5 {100 * float(v5_reference['conservative']['intent_balanced_accuracy']):.1f}%, "
                f"v6 {100 * float(v6_reference['conservative']['intent_balanced_accuracy']):.1f}%, "
                f"v7 {100 * float(adapted['intent_balanced_accuracy']):.1f}%.",
                "- Conservative sensitivity/specificity: "
                f"v5 {100 * float(v5_reference['conservative']['intent_sensitivity']):.1f}%/"
                f"{100 * float(v5_reference['conservative']['intent_specificity']):.1f}%, "
                f"v6 {100 * float(v6_reference['conservative']['intent_sensitivity']):.1f}%/"
                f"{100 * float(v6_reference['conservative']['intent_specificity']):.1f}%, "
                f"v7 {100 * float(adapted['intent_sensitivity']):.1f}%/"
                f"{100 * float(adapted['intent_specificity']):.1f}%.",
                "- Sensitivity-targeted BA: "
                f"v5 {100 * float(v5_reference['sensitivity_targeted']['intent_balanced_accuracy']):.1f}%, "
                f"v6 {100 * float(v6_reference['sensitivity_targeted']['intent_balanced_accuracy']):.1f}%, "
                f"v7 {100 * float(adapted_sensitive['intent_balanced_accuracy']):.1f}%.",
                "",
                "V7 is retained only if controller-selected auxiliary weights and later-recording results justify the added calibration labels.",
            ]
            if auxiliary and v5_reference is not None and v6_reference is not None
            else []
        ),
        *(
            [
                "",
                "## V5 Directional Comparison",
                "",
                "- Conservative v5 to v6: "
                f"BA {100 * float(v5_reference['conservative']['intent_balanced_accuracy']):.1f}% to {100 * float(adapted['intent_balanced_accuracy']):.1f}%, "
                f"sensitivity {100 * float(v5_reference['conservative']['intent_sensitivity']):.1f}% to {100 * float(adapted['intent_sensitivity']):.1f}%, and "
                f"specificity {100 * float(v5_reference['conservative']['intent_specificity']):.1f}% to {100 * float(adapted['intent_specificity']):.1f}%.",
                "- Sensitivity-targeted v5 to v6: "
                f"BA {100 * float(v5_reference['sensitivity_targeted']['intent_balanced_accuracy']):.1f}% to {100 * float(adapted_sensitive['intent_balanced_accuracy']):.1f}%, "
                f"sensitivity {100 * float(v5_reference['sensitivity_targeted']['intent_sensitivity']):.1f}% to {100 * float(adapted_sensitive['intent_sensitivity']):.1f}%, and "
                f"specificity {100 * float(v5_reference['sensitivity_targeted']['intent_specificity']):.1f}% to {100 * float(adapted_sensitive['intent_specificity']):.1f}%.",
                "",
                "Both pipelines evaluate rest versus intent at the headline endpoint. V6 removes direction-derived adapter inputs and uses a uniform 20-window binary calibration contract.",
            ]
            if not auxiliary and v5_reference is not None
            else []
        ),
        "",
        "## Per-Dataset Results",
        "",
        "| Dataset | Calibration windows | Matched population BA | Adapted BA | Adapted sensitivity | Adapted specificity |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Binary Controller",
        "",
        f"- Classes: `{patient['classes']}`.",
        *(
            [
                f"- Auxiliary training labels: `{patient['auxiliary_training_labels']}`.",
                "- Auxiliary direction output exposed: `False`.",
            ]
            if auxiliary
            else []
        ),
        f"- Selected intent fusion: `{calibrated['fusion']['intent_weights']}`.",
        f"- Intent calibration blends: `{intent_blends}`.",
        f"- Patient-adapter inputs: `{patient['adapter_inputs']}`.",
        *(
            [
                f"- V7 graph/LDA extension metadata: `{v7_extensions}`.",
            ]
            if v7_extensions.get("enabled")
            else []
        ),
        f"- Calibration configurations: `{patient['selected_configurations']}`.",
        f"- Adapted evaluation rows: **{patient['development_observed_evaluation_rows']:,}**.",
        "- Known visual or auditory cue supplies the requested movement; EEG only confirms rest versus intent.",
        "- Three-window confirmation can gate commands, but it is not reflected in cue-locked metrics.",
        "",
        "## Safety Boundary",
        "",
        str(report["continuous_no_control_status"]),
        "",
        "Temporal confirmation: "
        f"`{patient['temporal_confirmation']}`.",
        "",
        "Specificity is measured per cue-locked rest window. False activations per minute and command latency require continuous no-control recordings.",
        "",
        "## Protocol Note",
        "",
        (
            "V7 uses recording-0 cue labels only for calibration and excludes those rows from evaluation. Its binary endpoint is directly comparable with v5 and v6 on recording 1."
            if auxiliary
            else "V6 makes the patient-adaptation boundary explicit and regression-tested: recording-0 calibration windows are never counted in adapted metrics. The v5 split masks also selected recording 1 for evaluation, so the v5 result remains a valid directional benchmark."
        ),
        "",
        "## Outputs",
        "",
        (
            "- `translational_v7_auxiliary_direction_metrics.json`"
            if auxiliary
            else "- `translational_v6_intent_only_metrics.json`"
        ),
        (
            "- `translational_v7_auxiliary_direction_models.joblib`"
            if auxiliary
            else "- `translational_v6_intent_only_models.joblib`"
        ),
        "- `general_model_bundle/`",
        "- `01_decoder_comparison.png`",
        "- `02_safety_sensitivity_frontier.png`",
        "- `03_patient_calibration_burden.png`",
        "- `04_physiology_audit.png`",
        "- `05_quality_gate_availability.png`",
        "- `06_domain_probability_calibration.png`",
        "- `07_operating_point_comparison.png`",
        "- `08_neural_integration_summary.png`",
        *(
            [
                "- `09_v5_v6_v7_comparison.png`"
                if auxiliary
                else "- `09_v5_v6_intent_comparison.png`"
            ]
            if v5_reference is not None
            else []
        ),
        *(
            [
                "- `10_v7_graph_signal_audit.png`",
                "- `11_v7_graph_representation_diagnostics.png`",
            ]
            if v7_extensions.get("enabled")
            else []
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = parse_args()
    configure_plots()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_research_contract(output)
    started = time.time()
    profile = BranchPreprocessingProfile(
        phase="forward" if args.causal else "zero",
        compute_csd=not args.disable_csd,
    )
    dataset = build_translational_window_dataset(
        physionet_dir=args.physionet_dir,
        openbmi_dir=args.openbmi_dir,
        liu_dir=args.liu_dir,
        cache_path=Path(args.cache_path),
        max_subjects_per_dataset=args.max_subjects_per_dataset,
        hard_negative_manifest=args.hard_negative_manifest,
        profile=profile,
        rebuild_cache=args.rebuild_cache,
    )
    quality_control = summarize_recording_qc(dataset.recording_qc)
    excluded_count = int(
        quality_control["excluded_recordings_or_subjects"]
    )
    if excluded_count:
        print(
            f"[!] QC excluded {excluded_count} recording(s) or subject(s). "
            "See the cache-side *.qc.json report for details."
        )
    if args.build_only:
        print(f"[+] Build-only complete: {len(dataset.labels):,} windows")
        return
    for dataset_id in CORE_DATASETS:
        usable_subjects = len(
            np.unique(
                dataset.subject_keys[dataset.dataset_ids == dataset_id]
            )
        )
        if usable_subjects < 5:
            raise RuntimeError(
                f"Dataset {dataset_id} has only {usable_subjects} complete "
                "subjects after QC; at least five are required for disjoint "
                "development, controller-validation, and observed-test roles. "
                "Review the cache-side *.qc.json report."
            )

    masks, partitions = build_split_masks(dataset, args.seed)
    split_path = output / "subject_roles.json"
    split_path.write_text(
        json.dumps(partitions, indent=2) + "\n",
        encoding="ascii",
    )
    references = calibration_mask(dataset, masks)
    features = build_translational_features(
        dataset,
        masks[DEVELOPMENT_ROLE],
        references,
        include_graph=args.v7_graph_lda,
    )
    gate = QualityAvailabilityGate(
        args.quality_rejection_quantile
    ).fit(
        features.branches["quality"][masks[DEVELOPMENT_ROLE]],
        dataset.dataset_ids[masks[DEVELOPMENT_ROLE]],
    )
    available = gate.available(
        features.branches["quality"],
        dataset.dataset_ids,
    )
    models, probabilities = _fit_models(
        dataset,
        features,
        masks[DEVELOPMENT_ROLE],
        seed=args.seed,
        domain_weight=args.domain_weight,
        intent_only=args.intent_only,
        auxiliary_direction=args.auxiliary_direction,
        v7_graph_lda=args.v7_graph_lda,
    )
    neural_integration = _add_neural_intent_ensembles(
        dataset,
        models,
        probabilities,
        args,
    )
    neural_enabled = neural_integration is not None
    multi_neural_enabled = (
        neural_enabled and "shallow_convnet" in probabilities
    )
    results, thresholds = _score_methods(
        dataset,
        probabilities,
        available,
        masks,
        args,
    )
    (
        calibrated_report,
        calibrated_probabilities,
        stage_fusion,
        probability_calibrator,
        calibrated_thresholds,
    ) = _calibrated_stage_controller(
        dataset,
        probabilities,
        available,
        masks,
        args,
    )
    if args.intent_only:
        (
            patient_report,
            patient_probabilities,
            patient_thresholds,
            patient_evaluation,
        ) = _select_intent_only_patient_calibration(
            dataset,
            features,
            masks,
            calibrated_probabilities,
            available,
            args,
        )
    else:
        (
            patient_report,
            patient_probabilities,
            patient_thresholds,
        ) = _select_patient_calibration(
            dataset,
            features,
            masks,
            calibrated_probabilities,
            available,
            args,
        )
        patient_evaluation = (
            masks[DEVELOPMENT_OBSERVED_ROLE] & core_mask(dataset)
        )
    patient_report["temporal_confirmation"] = _patient_temporal_confirmation(
        dataset,
        patient_probabilities,
        available,
        masks,
        args,
    )
    observed = masks[DEVELOPMENT_OBSERVED_ROLE] & core_mask(dataset)
    neural_comparison = None
    if neural_enabled:
        physiology_probabilities = {
            name: values
            for name, values in probabilities.items()
            if name not in set(args.neural_architectures)
        }
        (
            baseline_calibrated_report,
            baseline_calibrated_probabilities,
            _,
            _,
            _,
        ) = _calibrated_stage_controller(
            dataset,
            physiology_probabilities,
            available,
            masks,
            args,
        )
        if args.intent_only:
            (
                baseline_patient_report,
                baseline_patient_probabilities,
                baseline_patient_thresholds,
                baseline_evaluation,
            ) = _select_intent_only_patient_calibration(
                dataset,
                features,
                masks,
                baseline_calibrated_probabilities,
                available,
                args,
            )
        else:
            (
                baseline_patient_report,
                baseline_patient_probabilities,
                baseline_patient_thresholds,
            ) = _select_patient_calibration(
                dataset,
                features,
                masks,
                baseline_calibrated_probabilities,
                available,
                args,
            )
            baseline_evaluation = observed
        paired = _paired_patient_comparison(
            dataset,
            baseline_patient_probabilities,
            patient_probabilities,
            available,
            patient_evaluation & baseline_evaluation,
            baseline_patient_thresholds["conservative"],
            patient_thresholds["conservative"],
            seed=args.seed + 30000,
            intent_only=args.intent_only,
        )
        neural_comparison = {
            "physiology_only_calibrated_conservative": (
                baseline_calibrated_report["operating_points"]["conservative"][
                    DEVELOPMENT_OBSERVED_ROLE
                ]
            ),
            "physiology_only_patient_adapted_conservative": (
                baseline_patient_report["development_observed_conservative"]
            ),
            "neural_integrated_calibrated_conservative": (
                calibrated_report["operating_points"]["conservative"][
                    DEVELOPMENT_OBSERVED_ROLE
                ]
            ),
            "neural_integrated_patient_adapted_conservative": (
                patient_report["development_observed_conservative"]
            ),
            "paired_patient_analysis": paired,
        }
    bootstrap = {
        "iterations": args.bootstrap_iterations,
        "population": {},
        "patient_adapted": {},
    }
    for offset, operating_point in enumerate(
        ("conservative", "sensitivity_targeted")
    ):
        population_mapping = calibrated_thresholds[operating_point]
        population_fallback = float(
            calibrated_thresholds[
                f"{operating_point}_fallback"
            ]
        )
        bootstrap["population"][operating_point] = (
            _bootstrap_operating_point(
                dataset,
                calibrated_probabilities,
                available,
                observed,
                population_mapping,
                population_fallback,
                iterations=args.bootstrap_iterations,
                seed=args.seed + 10000 + offset,
                intent_only=args.intent_only,
            )
        )
        patient_mapping = patient_thresholds[operating_point]
        patient_fallback = float(np.mean(list(patient_mapping.values())))
        bootstrap["patient_adapted"][operating_point] = (
            _bootstrap_operating_point(
                dataset,
                patient_probabilities,
                available,
                patient_evaluation,
                patient_mapping,
                patient_fallback,
                iterations=args.bootstrap_iterations,
                seed=args.seed + 20000 + offset,
                intent_only=args.intent_only,
            )
        )
    continuous_status = (
        "Measured on explicit continuous no-control recordings."
        if np.any(hard_negative_mask(dataset))
        else (
            "Not evaluated: no hard-negative manifest was supplied. "
            "Cue-locked window specificity is not converted into a deployment "
            "false-activation rate."
        )
    )
    v5_reference = _load_v5_reference(output) if args.intent_only else None
    v6_reference = (
        _load_v6_reference(output) if args.auxiliary_direction else None
    )
    report = {
        "schema_version": (
            9
            if args.v7_graph_lda
            else 8
            if args.auxiliary_direction
            else 7
            if args.intent_only
            else 6
            if multi_neural_enabled
            else 5
            if neural_enabled
            else 4
        ),
        "experiment": (
            "translational_motor_intent_v7_graph_lda_auxiliary_direction_binary_output"
            if args.v7_graph_lda
            else "translational_motor_intent_v7_auxiliary_direction_binary_output"
            if args.auxiliary_direction
            else "translational_motor_intent_v6_binary_patient_adaptive"
            if args.intent_only
            else "translational_motor_intent_v5_patient_adaptive_neural_fusion"
            if multi_neural_enabled
            else "translational_motor_intent_v4_neural_integrated"
            if neural_enabled
            else "translational_motor_intent_v3_calibrated"
        ),
        "intent_only": bool(args.intent_only),
        "auxiliary_direction": bool(args.auxiliary_direction),
        "v7_extensions": (
            {
                "enabled": True,
                "scope": "v7_auxiliary_direction_only",
                "fixed_graph_feature_count": int(
                    features.branches["gsp"].shape[1]
                ),
                "graph_edge_source": "fixed_standard_1005_geometry",
                "graph_edges_learned_from_eeg": False,
                "shrinkage_lda": "weighted_OAS_shared_covariance",
                "population_feature_branches": [
                    "mrcp",
                    "smr",
                    "riemann",
                    "gsp",
                ],
                "controller_selected_intent_weights": dict(
                    stage_fusion.intent_weights
                ),
                "patient_adapter_graph_candidate_available": True,
                "patient_adapter_graph_selected_by_dataset": {
                    dataset_id: bool(configuration["graph_branch_enabled"])
                    for dataset_id, configuration in patient_report[
                        "selected_configurations"
                    ].items()
                },
                "diagnostic_only_methods": ["PCA", "k-means"],
            }
            if args.v7_graph_lda
            else {"enabled": False}
        ),
        "confirmatory_status": {
            "current_test_role": DEVELOPMENT_OBSERVED_ROLE,
            "locked_external_available": False,
            "claim_bearing_result_available": False,
        },
        "preprocessing": profile.fingerprint(),
        "dataset": {
            "rows": len(dataset.labels),
            "subjects": int(len(np.unique(dataset.subject_keys))),
            "datasets": {
                dataset_id: {
                    "subjects": int(
                        len(
                            np.unique(
                                dataset.subject_keys[
                                    dataset.dataset_ids == dataset_id
                                ]
                            )
                        )
                    ),
                    "rows": int(np.sum(dataset.dataset_ids == dataset_id)),
                    "negative_kinds": sorted(
                        set(
                            dataset.negative_kinds[
                                (dataset.dataset_ids == dataset_id)
                                & (dataset.labels == 0)
                            ].tolist()
                        )
                    ),
                }
                for dataset_id in np.unique(dataset.dataset_ids)
            },
            "fingerprint": dataset.fingerprint,
            "quality_control": quality_control,
        },
        "subject_roles": partitions,
        "feature_metadata": features.metadata,
        "quality_gate": {
            "rejection_quantile": args.quality_rejection_quantile,
            "threshold": gate.threshold_,
            "available_fraction": float(np.mean(available)),
        },
        "methods": results,
        "thresholds": thresholds,
        "calibrated_stage_controller": calibrated_report,
        "patient_calibration": patient_report,
        "neural_integration": neural_integration,
        "neural_integration_comparison": neural_comparison,
        "v5_directional_reference": v5_reference,
        "v6_intent_only_reference": v6_reference,
        "bootstrap_confidence_intervals": bootstrap,
        "continuous_no_control_status": continuous_status,
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    metrics_path = output / (
        "translational_v7_auxiliary_direction_metrics.json"
        if args.auxiliary_direction
        else "translational_v6_intent_only_metrics.json"
        if args.intent_only
        else "translational_v5_patient_adaptive_metrics.json"
        if multi_neural_enabled
        else "translational_v4_neural_metrics.json"
        if neural_enabled
        else "translational_v3_metrics.json"
    )
    model_payload = {
        "intent_only": bool(args.intent_only),
        "auxiliary_direction": bool(args.auxiliary_direction),
        "v7_extensions": report["v7_extensions"],
        "class_contract": ["rest", "intent"] if args.intent_only else ["rest", "left", "right"],
        "auxiliary_training_labels": (
            ["rest", "left", "right"] if args.auxiliary_direction else []
        ),
        "models": models,
        "quality_gate": gate,
        "thresholds": thresholds,
        "stage_specific_fusion": stage_fusion,
        "probability_calibrator": probability_calibrator,
        "calibrated_thresholds": calibrated_thresholds,
        "patient_calibration_thresholds": patient_thresholds,
        "patient_calibration_configurations": patient_report[
            "selected_configurations"
        ],
        "patient_adapter_inputs": patient_report["adapter_inputs"],
        "subject_roles": partitions,
        "dataset_fingerprint": dataset.fingerprint,
        "channel_names": dataset.channel_names,
        "feature_names": features.names,
        "feature_metadata": features.metadata,
        "profile": profile,
        "neural_integration": neural_integration,
    }
    model_path = output / (
        "translational_v7_auxiliary_direction_models.joblib"
        if args.auxiliary_direction
        else "translational_v6_intent_only_models.joblib"
        if args.intent_only
        else "translational_v5_patient_adaptive_models.joblib"
        if multi_neural_enabled
        else "translational_v4_neural_models.joblib"
        if neural_enabled
        else "translational_v3_models.joblib"
    )
    joblib.dump(model_payload, model_path)
    if multi_neural_enabled:
        report["general_model_bundle"] = _write_general_model_bundle(
            output,
            model_payload,
            report,
        )
    metrics_path.write_text(
        json.dumps(report, indent=2, default=json_default) + "\n",
        encoding="ascii",
    )
    _plot_method_summary(
        output / "01_decoder_comparison.png",
        results,
        calibrated_report,
    )
    _plot_safety_frontier(
        output / "02_safety_sensitivity_frontier.png",
        dataset,
        calibrated_probabilities,
        available,
        np.isin(
            dataset.subject_keys,
            [
                subject
                for roles in calibrated_report[
                    "nested_controller_roles"
                ].values()
                for subject in roles["operating_point_selection"]
            ],
        )
        & masks[CONTROLLER_ROLE],
        calibrated_report["operating_points"],
        target_sensitivity=args.target_intent_sensitivity,
    )
    _plot_calibration(
        output / "03_patient_calibration_burden.png",
        patient_report,
    )
    _plot_physiology(
        output / "04_physiology_audit.png",
        dataset,
        masks[DEVELOPMENT_OBSERVED_ROLE] & core_mask(dataset),
    )
    _plot_quality(
        output / "05_quality_gate_availability.png",
        dataset,
        available,
    )
    raw_fusion = stage_fusion.predict(probabilities)
    if args.intent_only:
        raw_fusion = StageProbabilities(
            raw_fusion,
            np.full((len(raw_fusion), 2), 0.5),
        )
    _plot_domain_calibration(
        output / "06_domain_probability_calibration.png",
        dataset,
        raw_fusion,
        calibrated_probabilities,
        observed,
    )
    _plot_operating_point_comparison(
        output / "07_operating_point_comparison.png",
        calibrated_report,
        patient_report,
    )
    if neural_enabled:
        _plot_neural_integration_summary(
            output / "08_neural_integration_summary.png",
            report,
        )
    if (
        args.auxiliary_direction
        and v5_reference is not None
        and v6_reference is not None
    ):
        _plot_v5_v6_v7_comparison(
            output / "09_v5_v6_v7_comparison.png",
            v5_reference,
            v6_reference,
            {
                "conservative": patient_report[
                    "development_observed_conservative"
                ],
                "sensitivity_targeted": patient_report[
                    "development_observed_sensitivity_targeted"
                ],
            },
        )
    elif v5_reference is not None:
        _plot_v5_v6_comparison(
            output / "09_v5_v6_intent_comparison.png",
            v5_reference,
            {
                "conservative": patient_report[
                    "development_observed_conservative"
                ],
                "sensitivity_targeted": patient_report[
                    "development_observed_sensitivity_targeted"
                ],
            },
        )
    if args.v7_graph_lda:
        _plot_graph_signal_audit(
            output / "10_v7_graph_signal_audit.png",
            dataset,
            features,
            observed,
        )
        _plot_representation_diagnostics(
            output / "11_v7_graph_representation_diagnostics.png",
            dataset,
            features,
            probabilities,
            masks,
            seed=args.seed,
        )
    report_path = output / (
        "TRANSLATIONAL_V7_AUXILIARY_DIRECTION_REPORT.md"
        if args.auxiliary_direction
        else "TRANSLATIONAL_V6_INTENT_ONLY_REPORT.md"
        if args.intent_only
        else "TRANSLATIONAL_V5_PATIENT_ADAPTIVE_REPORT.md"
        if multi_neural_enabled
        else "TRANSLATIONAL_V4_NEURAL_REPORT.md"
        if neural_enabled
        else "TRANSLATIONAL_V3_REPORT.md"
    )
    (write_intent_only_report if args.intent_only else write_report)(
        report_path,
        report,
    )

    fusion = results["late_fusion"][DEVELOPMENT_OBSERVED_ROLE]
    calibrated = calibrated_report["operating_points"]["conservative"][
        DEVELOPMENT_OBSERVED_ROLE
    ]
    sensitivity_targeted = calibrated_report["operating_points"][
        "sensitivity_targeted"
    ][DEVELOPMENT_OBSERVED_ROLE]
    print(
        "\n=== Translational v7 Auxiliary-Direction Binary-Output Result ==="
        if args.auxiliary_direction
        else "\n=== Translational v6 Intent-Only Patient-Adaptive Result ==="
        if args.intent_only
        else "\n=== Translational v5 Patient-Adaptive Neural-Fusion Result ==="
        if multi_neural_enabled
        else "\n=== Translational v4 Neural-Integrated Development-Observed Result ==="
        if neural_enabled
        else "\n=== Translational v3 Development-Observed Result ==="
    )
    print(
        f"Late fusion | BA={100 * fusion['intent_balanced_accuracy']:.1f}% | "
        f"sensitivity={100 * fusion['intent_sensitivity']:.1f}% | "
        f"specificity={100 * fusion['intent_specificity']:.1f}% | "
        f"abstained={100 * fusion['abstained_fraction']:.1f}%"
    )
    print(
        "Calibrated conservative | "
        f"BA={100 * calibrated['intent_balanced_accuracy']:.1f}% | "
        f"sensitivity={100 * calibrated['intent_sensitivity']:.1f}% | "
        f"specificity={100 * calibrated['intent_specificity']:.1f}%"
    )
    print(
        "Calibrated sensitivity-targeted | "
        f"BA={100 * sensitivity_targeted['intent_balanced_accuracy']:.1f}% | "
        f"sensitivity={100 * sensitivity_targeted['intent_sensitivity']:.1f}% | "
        f"specificity={100 * sensitivity_targeted['intent_specificity']:.1f}%"
    )
    if neural_enabled:
        neural = results["mrcp_fbcnet"][DEVELOPMENT_OBSERVED_ROLE]
        patient = patient_report["development_observed_conservative"]
        if multi_neural_enabled:
            shallow = results["shallow_convnet"][DEVELOPMENT_OBSERVED_ROLE]
            print(
                "ShallowConvNet seed ensemble | "
                f"BA={100 * shallow['intent_balanced_accuracy']:.1f}% | "
                f"sensitivity={100 * shallow['intent_sensitivity']:.1f}% | "
                f"specificity={100 * shallow['intent_specificity']:.1f}%"
            )
        print(
            "MRCP + FBCNet seed ensemble | "
            f"BA={100 * neural['intent_balanced_accuracy']:.1f}% | "
            f"sensitivity={100 * neural['intent_sensitivity']:.1f}% | "
            f"specificity={100 * neural['intent_specificity']:.1f}%"
        )
        print(
            "Patient-adapted neural-integrated | "
            f"BA={100 * patient['intent_balanced_accuracy']:.1f}% | "
            f"sensitivity={100 * patient['intent_sensitivity']:.1f}% | "
            f"specificity={100 * patient['intent_specificity']:.1f}%"
        )
    print("[!] Confirmatory result unavailable: locked_external is empty.")
    figure_count = len(list(output.glob("*.png")))
    print(f"[+] Saved report, models, metrics, and {figure_count} figures to {output}")


if __name__ == "__main__":
    main()
