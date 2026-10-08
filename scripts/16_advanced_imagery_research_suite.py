#!/usr/bin/env python3
"""Run and consolidate advanced, leakage-safe EEG motor-imagery experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


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
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from pyriemann.utils.base import invsqrtm  # noqa: E402
from scipy.signal import butter, sosfiltfilt, welch  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.dataset import (  # noqa: E402
    BILATERAL_IMAGERY_RUNS,
    available_subjects,
    normalize_channel_name,
    resolve_data_dir,
)
from bci_pipeline.epochs import EpochConfig, RawEpochDataset, build_raw_epoch_dataset  # noqa: E402
from bci_pipeline.metrics import hierarchical_metrics  # noqa: E402
from bci_pipeline.models.eegnet import (  # noqa: E402
    select_balanced_intent_threshold,
    select_safety_intent_threshold,
)
from bci_pipeline.models.fbcsp import HierarchicalFBCSP  # noqa: E402
from bci_pipeline.representations.riemann import MOTOR_CHANNELS, TEMPORAL_CHANNELS  # noqa: E402
from bci_pipeline.splits import SubjectPartitions, load_partitions, three_way_subject_split  # noqa: E402


COLORS = {
    "baseline": "#778899",
    "erd": "#D05A3A",
    "riemann": "#2A9D8F",
    "alignment": "#1D4E89",
    "adaptation": "#E9A23B",
    "safety": "#8E5572",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFE",
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities >= threshold
    rest = labels == 0
    intent = labels == 1
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_activation_rate": float(np.mean(predictions[rest])) if np.any(rest) else 0.0,
        "missed_intent_rate": float(np.mean(~predictions[intent])) if np.any(intent) else 0.0,
        "intent_sensitivity": float(np.mean(predictions[intent])) if np.any(intent) else 0.0,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "trials": int(len(labels)),
    }


def select_binary_thresholds(
    labels: np.ndarray, probabilities: np.ndarray, safety_false_activation: float = 0.10
) -> Tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = np.asarray(
        [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    )
    tied = candidates[np.isclose(scores, np.max(scores))]
    balanced = float(tied[np.argmin(np.abs(tied - 0.5))])
    rest = probabilities[labels == 0]
    safety = float(
        np.nextafter(
            np.quantile(rest, 1.0 - safety_false_activation, method="higher"), np.inf
        )
    )
    return balanced, safety


def make_intent_pipeline(feature_count: int, seed: int, selected_count: int = 60) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selection", SelectKBest(f_classif, k=min(selected_count, feature_count))),
            (
                "model",
                LogisticRegression(
                    C=0.2,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_binary_model(
    model: Pipeline,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    max_false_activation: float,
) -> Dict[str, object]:
    validation_probability = model.predict_proba(validation_x)[:, 1]
    balanced_threshold, safety_threshold = select_binary_thresholds(
        validation_y, validation_probability, max_false_activation
    )
    test_probability = model.predict_proba(test_x)[:, 1]
    return {
        "balanced": binary_metrics(test_y, test_probability, balanced_threshold),
        "safety": binary_metrics(test_y, test_probability, safety_threshold),
        "balanced_threshold": balanced_threshold,
        "safety_threshold": safety_threshold,
    }


def load_morlet_features(path: Path) -> Dict[str, object]:
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Morlet feature table has no header.")
        rows = list(reader)
        feature_names = [name for name in reader.fieldnames if name.endswith("_erd_db")]
    return {
        "features": np.asarray(
            [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
        ),
        "labels": np.asarray([int(row["label"]) != 0 for row in rows], dtype=np.int64),
        "subjects": np.asarray([int(row["subject"]) for row in rows], dtype=np.int64),
        "runs": np.asarray([int(row["run"]) for row in rows], dtype=np.int64),
        "trial_indices": np.asarray([int(row["trial_index"]) for row in rows], dtype=np.int64),
        "max_peak_to_peak_uv": np.asarray(
            [float(row["max_peak_to_peak_uv"]) for row in rows], dtype=np.float64
        ),
        "feature_names": feature_names,
    }


def locked_masks(
    subjects: np.ndarray,
    runs: np.ndarray,
    partitions: SubjectPartitions,
    final_run: int,
) -> Dict[str, np.ndarray]:
    return {
        "development": np.isin(subjects, partitions.development),
        "validation": np.isin(subjects, partitions.controller_validation) & (runs == final_run),
        "test": np.isin(subjects, partitions.final_target) & (runs == final_run),
    }


def hard_negative_experiment(
    table: Mapping[str, object],
    partitions: SubjectPartitions,
    max_false_activation: float,
    seed: int,
) -> Dict[str, object]:
    x = np.asarray(table["features"])
    y = np.asarray(table["labels"])
    subjects = np.asarray(table["subjects"])
    runs = np.asarray(table["runs"])
    masks = locked_masks(subjects, runs, partitions, 12)
    train_indices = np.flatnonzero(masks["development"])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_indices), dtype=np.float64)
    for fit_local, score_local in splitter.split(
        x[train_indices], y[train_indices], subjects[train_indices]
    ):
        model = make_intent_pipeline(x.shape[1], seed)
        model.fit(x[train_indices[fit_local]], y[train_indices[fit_local]])
        oof[score_local] = model.predict_proba(x[train_indices[score_local]])[:, 1]

    candidates: List[Dict[str, object]] = []
    for fraction in (0.10, 0.20, 0.30):
        rest_local = y[train_indices] == 0
        limit = np.quantile(oof[rest_local], 1.0 - fraction)
        hard_local = rest_local & (oof >= limit)
        for weight in (2.0, 4.0, 6.0):
            sample_weight = np.ones(len(train_indices), dtype=np.float64)
            sample_weight[hard_local] = weight
            model = make_intent_pipeline(x.shape[1], seed)
            model.fit(
                x[train_indices], y[train_indices], model__sample_weight=sample_weight
            )
            validation_probability = model.predict_proba(x[masks["validation"]])[:, 1]
            balanced_threshold, safety_threshold = select_binary_thresholds(
                y[masks["validation"]], validation_probability, max_false_activation
            )
            validation_metrics = binary_metrics(
                y[masks["validation"]], validation_probability, balanced_threshold
            )
            candidates.append(
                {
                    "hard_rest_fraction": fraction,
                    "hard_rest_weight": weight,
                    "model": model,
                    "balanced_threshold": balanced_threshold,
                    "safety_threshold": safety_threshold,
                    "validation": validation_metrics,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["validation"]["balanced_accuracy"],
            -row["validation"]["false_activation_rate"],
        ),
    )
    final_probability = selected["model"].predict_proba(x[masks["test"]])[:, 1]
    return {
        "selected_hard_rest_fraction": selected["hard_rest_fraction"],
        "selected_hard_rest_weight": selected["hard_rest_weight"],
        "controller_validation": selected["validation"],
        "balanced": binary_metrics(
            y[masks["test"]], final_probability, selected["balanced_threshold"]
        ),
        "safety": binary_metrics(
            y[masks["test"]], final_probability, selected["safety_threshold"]
        ),
        "candidate_count": len(candidates),
    }


def artifact_threshold_experiment(
    table: Mapping[str, object],
    partitions: SubjectPartitions,
    max_false_activation: float,
    seed: int,
) -> Dict[str, object]:
    x = np.asarray(table["features"])
    y = np.asarray(table["labels"])
    subjects = np.asarray(table["subjects"])
    runs = np.asarray(table["runs"])
    amplitudes = np.asarray(table["max_peak_to_peak_uv"])
    base_masks = locked_masks(subjects, runs, partitions, 12)
    results: Dict[str, object] = {}
    for cutoff in (100.0, 150.0, 200.0, 250.0):
        quality = amplitudes <= cutoff
        masks = {name: mask & quality for name, mask in base_masks.items()}
        model = make_intent_pipeline(x.shape[1], seed).fit(
            x[masks["development"]], y[masks["development"]]
        )
        result = evaluate_binary_model(
            model,
            x[masks["validation"]],
            y[masks["validation"]],
            x[masks["test"]],
            y[masks["test"]],
            max_false_activation,
        )
        result["retained_development_percent"] = float(
            100.0 * np.sum(masks["development"]) / np.sum(base_masks["development"])
        )
        result["retained_final_percent"] = float(
            100.0 * np.sum(masks["test"]) / np.sum(base_masks["test"])
        )
        results[str(int(cutoff))] = result
    return results


def load_raw_cache(path: Path) -> RawEpochDataset:
    with np.load(path, allow_pickle=False) as cached:
        return RawEpochDataset(
            epochs=cached["epochs"],
            baselines=cached["baselines"],
            labels=cached["labels"],
            subjects=cached["subjects"],
            runs=cached["runs"],
            trial_indices=cached["trial_indices"],
            channel_names=[str(name) for name in cached["channel_names"]],
            sfreq=float(cached["sfreq"].item()),
        )


def channel_positions(names: Sequence[str], requested: Sequence[str]) -> List[int]:
    lookup = {normalize_channel_name(name): index for index, name in enumerate(names)}
    return [lookup[normalize_channel_name(name)] for name in requested]


def estimate_individual_mu_peaks(dataset: RawEpochDataset) -> Dict[int, float]:
    calibration_runs = np.isin(dataset.runs, (4, 8))
    central = channel_positions(dataset.channel_names, ("C3", "Cz", "C4"))
    peaks: Dict[int, float] = {}
    for subject in np.unique(dataset.subjects):
        selected = calibration_runs & (dataset.subjects == subject)
        frequencies, spectrum = welch(
            dataset.baselines[selected][:, central],
            fs=dataset.sfreq,
            nperseg=int(dataset.sfreq),
            axis=-1,
        )
        mu = (frequencies >= 7.0) & (frequencies <= 13.0)
        average = np.mean(spectrum, axis=(0, 1))
        peaks[int(subject)] = float(frequencies[mu][np.argmax(average[mu])])
    return peaks


def erd_feature_block(
    epochs: np.ndarray,
    baselines: np.ndarray,
    sfreq: float,
    bands: Sequence[Tuple[float, float]],
    batch_size: int = 256,
) -> np.ndarray:
    phase_edges = ((0, 160), (160, 320), (320, 480))
    output = np.empty((len(epochs), epochs.shape[1], len(bands), len(phase_edges)), dtype=np.float32)
    for band_index, (low, high) in enumerate(bands):
        sos = butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos")
        for start in range(0, len(epochs), batch_size):
            stop = min(start + batch_size, len(epochs))
            task = sosfiltfilt(sos, epochs[start:stop], axis=-1)
            baseline = sosfiltfilt(sos, baselines[start:stop], axis=-1)
            baseline_power = np.mean(baseline**2, axis=-1)
            for phase_index, (phase_start, phase_stop) in enumerate(phase_edges):
                task_power = np.mean(task[:, :, phase_start:phase_stop] ** 2, axis=-1)
                output[start:stop, :, band_index, phase_index] = 10.0 * np.log10(
                    (task_power + 1e-20) / (baseline_power + 1e-20)
                )
    return output.reshape(len(epochs), -1)


def individualized_erd_features(
    dataset: RawEpochDataset,
    cache_path: Path,
    rebuild: bool,
) -> Dict[str, object]:
    imagery = np.isin(dataset.runs, (4, 8, 12))
    epochs = dataset.epochs[imagery]
    baselines = dataset.baselines[imagery]
    subjects = dataset.subjects[imagery]
    runs = dataset.runs[imagery]
    labels = (dataset.labels[imagery] != 0).astype(np.int64)
    trials = dataset.trial_indices[imagery]
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            return {
                "fixed": cached["fixed"],
                "individualized": cached["individualized"],
                "labels": cached["labels"],
                "subjects": cached["subjects"],
                "runs": cached["runs"],
                "trial_indices": cached["trial_indices"],
                "peaks": json.loads(str(cached["peaks"].item())),
            }

    print("[+] Extracting fixed and individualized mu/beta ERD features...")
    peaks = estimate_individual_mu_peaks(dataset)
    fixed = erd_feature_block(
        epochs, baselines, dataset.sfreq, ((8.0, 13.0), (13.0, 20.0), (20.0, 30.0))
    )
    individualized = np.empty_like(fixed)
    for peak in sorted(set(peaks.values())):
        selected = np.asarray([peaks[int(subject)] == peak for subject in subjects])
        low, high = max(6.0, peak - 2.0), min(15.0, peak + 2.0)
        individualized[selected] = erd_feature_block(
            epochs[selected],
            baselines[selected],
            dataset.sfreq,
            ((low, high), (13.0, 20.0), (20.0, 30.0)),
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        fixed=fixed,
        individualized=individualized,
        labels=labels,
        subjects=subjects,
        runs=runs,
        trial_indices=trials,
        peaks=np.asarray(json.dumps({str(key): value for key, value in peaks.items()})),
    )
    return {
        "fixed": fixed,
        "individualized": individualized,
        "labels": labels,
        "subjects": subjects,
        "runs": runs,
        "trial_indices": trials,
        "peaks": {str(key): value for key, value in peaks.items()},
    }


def compare_individualized_erd(
    feature_data: Mapping[str, object],
    partitions: SubjectPartitions,
    max_false_activation: float,
    seed: int,
) -> Dict[str, object]:
    labels = np.asarray(feature_data["labels"])
    subjects = np.asarray(feature_data["subjects"])
    runs = np.asarray(feature_data["runs"])
    masks = locked_masks(subjects, runs, partitions, 12)
    results: Dict[str, object] = {}
    for name in ("fixed", "individualized"):
        features = np.asarray(feature_data[name])
        model = make_intent_pipeline(features.shape[1], seed).fit(
            features[masks["development"]], labels[masks["development"]]
        )
        results[name] = evaluate_binary_model(
            model,
            features[masks["validation"]],
            labels[masks["validation"]],
            features[masks["test"]],
            labels[masks["test"]],
            max_false_activation,
        )
    peak_values = np.asarray(list(feature_data["peaks"].values()), dtype=float)
    results["peak_summary"] = {
        "subjects": int(len(peak_values)),
        "median_hz": float(np.median(peak_values)),
        "minimum_hz": float(np.min(peak_values)),
        "maximum_hz": float(np.max(peak_values)),
    }
    return results


def normalize_covariances(covariances: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    traces = np.trace(symmetric, axis1=-2, axis2=-1)
    return symmetric / np.clip(traces[..., None, None], 1e-12, None)


def covariance_channel_subset(
    covariances: np.ndarray, channel_names: Sequence[str], selected_names: Sequence[str]
) -> np.ndarray:
    indices = channel_positions(channel_names, selected_names)
    subset = covariances[:, :, indices][:, :, :, indices]
    return normalize_covariances(subset.astype(np.float64))


def alignment_reference(covariances: np.ndarray) -> np.ndarray:
    reference = np.mean(covariances.astype(np.float64), axis=0)
    reference = 0.5 * (reference + reference.T)
    regularizer = 1e-8 * max(float(np.trace(reference)) / reference.shape[0], 1e-12)
    return reference + regularizer * np.eye(reference.shape[0])


def align_covariances(covariances: np.ndarray, references: Sequence[np.ndarray]) -> np.ndarray:
    aligned = np.empty_like(covariances, dtype=np.float64)
    for view_index, reference in enumerate(references):
        whitening = invsqrtm(reference)
        values = whitening @ covariances[:, view_index] @ whitening
        aligned[:, view_index] = 0.5 * (values + np.swapaxes(values, -1, -2))
    return normalize_covariances(aligned)


def align_development_by_subject(
    covariances: np.ndarray, subjects: np.ndarray
) -> np.ndarray:
    aligned = np.empty_like(covariances, dtype=np.float64)
    for subject in np.unique(subjects):
        selected = subjects == subject
        references = [
            alignment_reference(covariances[selected, view_index])
            for view_index in range(covariances.shape[1])
        ]
        aligned[selected] = align_covariances(covariances[selected], references)
    return aligned


def align_targets_from_calibration(
    covariances: np.ndarray,
    subjects: np.ndarray,
    runs: np.ndarray,
    target_subjects: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    output_indices: List[int] = []
    aligned_rows: List[np.ndarray] = []
    for subject in target_subjects:
        calibration = (subjects == subject) & np.isin(runs, (4, 8))
        evaluation = (subjects == subject) & (runs == 12)
        if not np.any(calibration) or not np.any(evaluation):
            continue
        references = [
            alignment_reference(covariances[calibration, view_index])
            for view_index in range(covariances.shape[1])
        ]
        indices = np.flatnonzero(evaluation)
        output_indices.extend(indices.tolist())
        aligned_rows.append(align_covariances(covariances[indices], references))
    order = np.argsort(output_indices)
    return np.asarray(output_indices, dtype=np.int64)[order], np.concatenate(aligned_rows)[order]


def fit_tangent_transformer(
    train_covariances: np.ndarray,
) -> Tuple[List[TangentSpace], np.ndarray]:
    transformers: List[TangentSpace] = []
    blocks: List[np.ndarray] = []
    for view_index in range(train_covariances.shape[1]):
        transformer = TangentSpace(metric="riemann")
        blocks.append(transformer.fit_transform(train_covariances[:, view_index]))
        transformers.append(transformer)
    return transformers, np.concatenate(blocks, axis=1)


def transform_tangent(
    covariances: np.ndarray, transformers: Sequence[TangentSpace]
) -> np.ndarray:
    return np.concatenate(
        [
            transformer.transform(covariances[:, view_index])
            for view_index, transformer in enumerate(transformers)
        ],
        axis=1,
    )


def riemannian_experiments(
    cache_path: Path,
    partitions: SubjectPartitions,
    max_false_activation: float,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    with np.load(cache_path, allow_pickle=False) as cached:
        covariance_all = cached["covariances"]
        labels = cached["labels"]
        subjects = cached["subjects"]
        runs = cached["runs"]
        trials = cached["trial_indices"]
        names = [str(name) for name in cached["channel_names"]]
    binary_labels = (labels != 0).astype(np.int64)
    masks = locked_masks(subjects, runs, partitions, 12)
    results: Dict[str, object] = {}
    adaptation_payload: Dict[str, object] = {}
    for montage_name, montage in (("nine_channel", TEMPORAL_CHANNELS), ("fifteen_channel", MOTOR_CHANNELS)):
        covariances = covariance_channel_subset(covariance_all, names, montage)
        train_cov = covariances[masks["development"]]
        train_y = binary_labels[masks["development"]]
        transformers, train_features = fit_tangent_transformer(train_cov)
        validation_features = transform_tangent(covariances[masks["validation"]], transformers)
        test_features = transform_tangent(covariances[masks["test"]], transformers)
        model = make_intent_pipeline(train_features.shape[1], seed, selected_count=120).fit(
            train_features, train_y
        )
        unaligned_result = evaluate_binary_model(
            model,
            validation_features,
            binary_labels[masks["validation"]],
            test_features,
            binary_labels[masks["test"]],
            max_false_activation,
        )

        development_indices = np.flatnonzero(masks["development"])
        aligned_development = align_development_by_subject(
            covariances[development_indices], subjects[development_indices]
        )
        validation_indices, aligned_validation = align_targets_from_calibration(
            covariances, subjects, runs, partitions.controller_validation
        )
        test_indices, aligned_test = align_targets_from_calibration(
            covariances, subjects, runs, partitions.final_target
        )
        aligned_transformers, aligned_train_features = fit_tangent_transformer(aligned_development)
        aligned_validation_features = transform_tangent(aligned_validation, aligned_transformers)
        aligned_test_features = transform_tangent(aligned_test, aligned_transformers)
        aligned_model = make_intent_pipeline(
            aligned_train_features.shape[1], seed, selected_count=120
        ).fit(aligned_train_features, train_y)
        aligned_result = evaluate_binary_model(
            aligned_model,
            aligned_validation_features,
            binary_labels[validation_indices],
            aligned_test_features,
            binary_labels[test_indices],
            max_false_activation,
        )
        results[montage_name] = {
            "unaligned_tangent": unaligned_result,
            "euclidean_aligned_tangent": aligned_result,
            "feature_count": int(aligned_train_features.shape[1]),
            "channels": list(montage),
        }
        if montage_name == "fifteen_channel":
            adaptation_payload = {
                "covariances": covariances,
                "labels": labels,
                "binary_labels": binary_labels,
                "subjects": subjects,
                "runs": runs,
                "trials": trials,
                "transformers": aligned_transformers,
                "model": aligned_model,
                "development_features": aligned_train_features,
                "validation_indices": validation_indices,
                "validation_features": aligned_validation_features,
                "test_indices": test_indices,
                "test_features": aligned_test_features,
            }
    return results, adaptation_payload


def aligned_subject_features(
    payload: Mapping[str, object], subject: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    covariances = np.asarray(payload["covariances"])
    subjects = np.asarray(payload["subjects"])
    runs = np.asarray(payload["runs"])
    trials = np.asarray(payload["trials"])
    labels = np.asarray(payload["labels"])
    calibration = (subjects == subject) & np.isin(runs, (4, 8))
    evaluation = (subjects == subject) & (runs == 12)
    references = [
        alignment_reference(covariances[calibration, view_index])
        for view_index in range(covariances.shape[1])
    ]
    calibration_cov = align_covariances(covariances[calibration], references)
    evaluation_cov = align_covariances(covariances[evaluation], references)
    calibration_features = transform_tangent(calibration_cov, payload["transformers"])
    evaluation_features = transform_tangent(evaluation_cov, payload["transformers"])
    return (
        calibration_features,
        labels[calibration],
        runs[calibration],
        trials[calibration],
        evaluation_features,
    )


def calibration_selection(
    labels: np.ndarray, runs: np.ndarray, trials: np.ndarray, budget: int
) -> np.ndarray:
    selected: List[int] = []
    for label in (0, 1, 2):
        candidates = np.flatnonzero(labels == label)
        order = np.lexsort((trials[candidates], runs[candidates]))
        if len(candidates) < budget:
            return np.asarray([], dtype=np.int64)
        selected.extend(candidates[order][:budget].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def adaptation_probabilities(
    payload: Mapping[str, object], target_subjects: Iterable[int], budget: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_labels: List[np.ndarray] = []
    global_rows: List[np.ndarray] = []
    local_rows: List[np.ndarray] = []
    subject_rows: List[np.ndarray] = []
    global_model = payload["model"]
    subjects = np.asarray(payload["subjects"])
    labels = np.asarray(payload["labels"])
    runs_all = np.asarray(payload["runs"])
    for subject in target_subjects:
        calibration_x, calibration_y, calibration_runs, calibration_trials, evaluation_x = (
            aligned_subject_features(payload, int(subject))
        )
        chosen = calibration_selection(
            calibration_y, calibration_runs, calibration_trials, budget
        )
        evaluation = (subjects == subject) & (runs_all == 12)
        if not len(chosen) or not np.any(evaluation):
            continue
        transformed_calibration = global_model[:-1].transform(calibration_x[chosen])
        transformed_evaluation = global_model[:-1].transform(evaluation_x)
        local_model = LogisticRegression(
            C=0.05,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed + int(subject),
        ).fit(transformed_calibration, (calibration_y[chosen] != 0).astype(np.int64))
        all_labels.append((labels[evaluation] != 0).astype(np.int64))
        global_rows.append(global_model.predict_proba(evaluation_x)[:, 1])
        local_rows.append(local_model.predict_proba(transformed_evaluation)[:, 1])
        subject_rows.append(np.full(np.sum(evaluation), int(subject), dtype=np.int64))
    return (
        np.concatenate(all_labels),
        np.concatenate(global_rows),
        np.concatenate(local_rows),
        np.concatenate(subject_rows),
    )


def patient_calibration_experiment(
    payload: Mapping[str, object],
    partitions: SubjectPartitions,
    budgets: Sequence[int],
    max_false_activation: float,
    seed: int,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    for budget in budgets:
        validation_y, validation_global, validation_local, _ = adaptation_probabilities(
            payload, partitions.controller_validation, budget, seed
        )
        candidates: List[Tuple[float, float, Dict[str, object]]] = []
        for weight in (0.0, 0.25, 0.50, 0.75):
            probability = (1.0 - weight) * validation_global + weight * validation_local
            for threshold in np.linspace(0.20, 0.80, 121):
                candidate_metrics = binary_metrics(validation_y, probability, float(threshold))
                candidates.append((weight, float(threshold), candidate_metrics))
        selected_weight, balanced_threshold, controller_metrics = max(
            candidates,
            key=lambda row: (
                row[2]["balanced_accuracy"],
                -row[2]["false_activation_rate"],
                -abs(row[1] - 0.5),
            ),
        )
        validation_probability = (
            (1.0 - selected_weight) * validation_global + selected_weight * validation_local
        )
        safety_threshold = float(
            np.nextafter(
                np.quantile(
                    validation_probability[validation_y == 0],
                    1.0 - max_false_activation,
                    method="higher",
                ),
                np.inf,
            )
        )
        test_y, test_global, test_local, test_subjects = adaptation_probabilities(
            payload, partitions.final_target, budget, seed
        )
        test_probability = (1.0 - selected_weight) * test_global + selected_weight * test_local
        per_subject: Dict[str, Dict[str, object]] = {}
        for subject in np.unique(test_subjects):
            selected = test_subjects == subject
            per_subject[f"S{int(subject):03d}"] = binary_metrics(
                test_y[selected], test_probability[selected], balanced_threshold
            )
        results[str(budget)] = {
            "calibration_trials_per_original_class": int(budget),
            "total_labeled_calibration_trials_per_subject": int(3 * budget),
            "selected_local_model_weight": selected_weight,
            "balanced_threshold": balanced_threshold,
            "safety_threshold": safety_threshold,
            "controller_validation": controller_metrics,
            "global": binary_metrics(test_y, test_global, balanced_threshold),
            "adapted_balanced": binary_metrics(test_y, test_probability, balanced_threshold),
            "adapted_safety": binary_metrics(test_y, test_probability, safety_threshold),
            "evaluated_subjects": int(len(np.unique(test_subjects))),
            "per_subject": per_subject,
        }
    return results


def bilateral_montage_experiment(
    data_dir: Path,
    subjects: Sequence[int],
    partitions: SubjectPartitions,
    cache_path: Path,
    rebuild_cache: bool,
    max_false_activation: float,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, HierarchicalFBCSP]]:
    dataset = build_raw_epoch_dataset(
        data_dir,
        subjects,
        cache_path,
        EpochConfig(runs=BILATERAL_IMAGERY_RUNS, channels=MOTOR_CHANNELS),
        rebuild_cache,
    )
    represented = set(np.unique(dataset.subjects).astype(int).tolist())
    development_subjects = [value for value in partitions.development if value in represented]
    validation_subjects = [value for value in partitions.controller_validation if value in represented]
    test_subjects = [value for value in partitions.final_target if value in represented]
    masks = {
        "development": np.isin(dataset.subjects, development_subjects),
        "validation": np.isin(dataset.subjects, validation_subjects) & (dataset.runs == 14),
        "test": np.isin(dataset.subjects, test_subjects) & (dataset.runs == 14),
    }
    montage_indices = {
        "nine_channel": channel_positions(dataset.channel_names, TEMPORAL_CHANNELS),
        "fifteen_channel": channel_positions(dataset.channel_names, MOTOR_CHANNELS),
    }
    results: Dict[str, object] = {}
    models: Dict[str, HierarchicalFBCSP] = {}
    for name, indices in montage_indices.items():
        epochs = dataset.epochs[:, indices]
        model = HierarchicalFBCSP(dataset.sfreq, random_seed=seed).fit(
            epochs[masks["development"]], dataset.labels[masks["development"]]
        )
        validation_intent, _ = model.predict_stage_probabilities(epochs[masks["validation"]])
        balanced_threshold = select_balanced_intent_threshold(
            dataset.labels[masks["validation"]], validation_intent
        )
        safety_threshold = select_safety_intent_threshold(
            dataset.labels[masks["validation"]], validation_intent, max_false_activation
        )
        test_intent, test_direction = model.predict_stage_probabilities(epochs[masks["test"]])
        results[name] = {
            "channels": [dataset.channel_names[index] for index in indices],
            "train_trials": int(np.sum(masks["development"])),
            "validation_trials": int(np.sum(masks["validation"])),
            "test_trials": int(np.sum(masks["test"])),
            "balanced": hierarchical_metrics(
                dataset.labels[masks["test"]],
                test_intent,
                test_direction,
                balanced_threshold,
                {0: "rest", 1: "both_fists", 2: "both_feet"},
            ),
            "safety": hierarchical_metrics(
                dataset.labels[masks["test"]],
                test_intent,
                test_direction,
                safety_threshold,
                {0: "rest", 1: "both_fists", 2: "both_feet"},
            ),
            "balanced_threshold": balanced_threshold,
            "safety_threshold": safety_threshold,
        }
        models[name] = model
    return results, models


def existing_results(project_root: Path) -> Dict[str, object]:
    return {
        "dual_benchmark": read_json(
            project_root / "artifacts/dual_task_benchmark/dual_task_metrics.json"
        ),
        "morlet": read_json(
            project_root / "artifacts/morlet_intent_models/morlet_intent_model_metrics.json"
        ),
        "cue_audit": read_json(
            project_root / "artifacts/cue_intent_audit/cue_intent_correlation_metrics.json"
        ),
        "cross_task": read_json(
            project_root / "artifacts/cross_task/cross_task_metrics.json"
        ),
        "temporal": read_json(
            project_root
            / "artifacts/rest_aware_sequential_controller/rest_aware_controller_metrics.json"
        ),
        "ica": read_json(
            project_root / "artifacts/ica_diagnostic/S001_R04/ica_diagnostic.json"
        ),
    }


def intent_summary_rows(report: Mapping[str, object]) -> List[Dict[str, object]]:
    existing = report["existing_results"]
    fbcsp = existing["dual_benchmark"]["results"]["unilateral_imagery"]["balanced"]
    morlet = existing["morlet"]["models"]["logistic_regression"]["balanced"]
    eegnet = existing["cross_task"]["results"]["eegnet"]["imagery_to_imagery"]["balanced"]
    pretrained = existing["cross_task"]["results"]["eegnet"][
        "pretrained_finetuned_to_imagery"
    ]["balanced"]
    hard_negative = report["hard_negative"]["balanced"]
    individualized = report["individualized_erd"]["individualized"]["balanced"]
    riemann_unaligned = report["riemannian"]["fifteen_channel"]["unaligned_tangent"][
        "balanced"
    ]
    riemann_aligned = report["riemannian"]["fifteen_channel"][
        "euclidean_aligned_tangent"
    ]["balanced"]
    calibration_budgets = report["patient_calibration"]
    maximum_budget = max(calibration_budgets, key=int)
    calibrated = calibration_budgets[maximum_budget]["adapted_balanced"]

    def hierarchical_row(name: str, source: Mapping[str, object], category: str) -> Dict[str, object]:
        return {
            "name": name,
            "category": category,
            "balanced_accuracy": source["intent_balanced_accuracy"],
            "false_activation_rate": source["false_activation_rate"],
            "missed_intent_rate": source["missed_intent_rate"],
            "intent_sensitivity": 1.0 - source["missed_intent_rate"],
            "roc_auc": None,
        }

    def binary_row(name: str, source: Mapping[str, object], category: str) -> Dict[str, object]:
        return {
            "name": name,
            "category": category,
            "balanced_accuracy": source["balanced_accuracy"],
            "false_activation_rate": source["false_activation_rate"],
            "missed_intent_rate": source["missed_intent_rate"],
            "intent_sensitivity": source["intent_sensitivity"],
            "roc_auc": source.get("roc_auc"),
        }

    return [
        hierarchical_row("FBCSP", fbcsp, "baseline"),
        binary_row("Morlet ERD", morlet, "erd"),
        binary_row("Hard-negative ERD", hard_negative, "erd"),
        binary_row("Individual-mu ERD", individualized, "erd"),
        binary_row("Riemannian TS", riemann_unaligned, "riemann"),
        binary_row("EA + Riemannian", riemann_aligned, "alignment"),
        binary_row(f"EA + {maximum_budget}/class calibration", calibrated, "adaptation"),
        hierarchical_row("EEGNet", eegnet, "baseline"),
        hierarchical_row("Execution-pretrained EEGNet", pretrained, "baseline"),
    ]


def plot_model_comparison(report: Mapping[str, object], output_dir: Path) -> None:
    rows = intent_summary_rows(report)
    labels = [row["name"] for row in rows]
    colors = [COLORS[row["category"]] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), constrained_layout=True)
    specs = (
        ("balanced_accuracy", "Intent Balanced Accuracy"),
        ("false_activation_rate", "False Activation During Rest"),
        ("missed_intent_rate", "Missed Motor Intent"),
    )
    positions = np.arange(len(rows))
    for axis, (key, title) in zip(axes, specs):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(positions, values, color=colors)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value * 100:.1f}%",
                ha="center",
                fontsize=8,
                weight="bold",
            )
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Final unseen-subject rate")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "Advanced Motor-Imagery Intent Models on the Locked Final Subjects",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(output_dir / "advanced_intent_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_safety_plane(report: Mapping[str, object], output_dir: Path) -> None:
    rows = intent_summary_rows(report)
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    annotation_offsets = {
        "Morlet ERD": (8, 12),
        "Hard-negative ERD": (8, -10),
        "Individual-mu ERD": (8, 1),
        "EA + Riemannian": (8, 8),
        "EA + 10/class calibration": (8, 8),
        "Execution-pretrained EEGNet": (8, 8),
    }
    for row in rows:
        ax.scatter(
            row["false_activation_rate"],
            row["intent_sensitivity"],
            s=85,
            color=COLORS[row["category"]],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(
            row["name"],
            (row["false_activation_rate"], row["intent_sensitivity"]),
            xytext=annotation_offsets.get(row["name"], (6, 6)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvspan(
        0.0,
        0.10,
        color="#DDEFE9",
        alpha=0.65,
        label=r"Target false activation $\leq$10%",
    )
    ax.set_xlim(-0.02, 0.82)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False-activation rate during rest")
    ax.set_ylabel("Intent sensitivity")
    ax.set_title("Safety-Responsiveness Operating Points")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "advanced_safety_responsiveness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(report: Mapping[str, object], output_dir: Path) -> None:
    results = report["patient_calibration"]
    budgets = sorted(int(value) for value in results)
    balanced = [results[str(value)]["adapted_balanced"]["balanced_accuracy"] for value in budgets]
    false_activation = [
        results[str(value)]["adapted_balanced"]["false_activation_rate"] for value in budgets
    ]
    missed = [results[str(value)]["adapted_balanced"]["missed_intent_rate"] for value in budgets]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    axes[0].plot(budgets, balanced, marker="o", linewidth=2.3, color=COLORS["adaptation"])
    for x, value in zip(budgets, balanced):
        axes[0].text(x, value + 0.018, f"{value * 100:.1f}%", ha="center", fontsize=8)
    axes[0].set_ylim(0.35, 0.85)
    axes[0].set_xlabel("Labeled calibration trials per rest/left/right class")
    axes[0].set_ylabel("Intent balanced accuracy")
    axes[0].set_title("Patient Calibration Benefit")
    axes[0].grid(alpha=0.22)
    axes[1].plot(budgets, false_activation, marker="o", linewidth=2.2, label="False activation")
    axes[1].plot(budgets, missed, marker="s", linewidth=2.2, label="Missed intent")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Labeled calibration trials per rest/left/right class")
    axes[1].set_ylabel("Final unseen-subject error rate")
    axes[1].set_title("Calibration Safety Tradeoff")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("Few-Shot Patient Adaptation Using Runs 4/8; Evaluation on Run 12", fontsize=14, weight="bold")
    fig.savefig(output_dir / "patient_calibration_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_design(report: Mapping[str, object], output_dir: Path) -> None:
    individualized = report["individualized_erd"]
    riemannian = report["riemannian"]
    bilateral = report["bilateral_montage"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    erd_labels = ["Fixed\n8-13 Hz", "Individual\npeak +/-2 Hz"]
    erd_values = [
        individualized["fixed"]["balanced"]["balanced_accuracy"],
        individualized["individualized"]["balanced"]["balanced_accuracy"],
    ]
    bars = axes[0].bar(erd_labels, erd_values, color=[COLORS["baseline"], COLORS["erd"]])
    axes[0].set_title("Individualized Mu Band")
    axes[0].set_ylabel("Intent balanced accuracy")
    axes[0].set_ylim(0.35, 0.80)
    for bar, value in zip(bars, erd_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value * 100:.1f}%", ha="center", weight="bold")

    riemann_labels = ["9-channel", "15-channel"]
    riemann_values = [
        riemannian["nine_channel"]["euclidean_aligned_tangent"]["balanced"]["balanced_accuracy"],
        riemannian["fifteen_channel"]["euclidean_aligned_tangent"]["balanced"]["balanced_accuracy"],
    ]
    bars = axes[1].bar(riemann_labels, riemann_values, color=[COLORS["riemann"], COLORS["alignment"]])
    axes[1].set_title("Unilateral Imagery Montage")
    axes[1].set_ylabel("Intent balanced accuracy")
    axes[1].set_ylim(0.35, 0.80)
    for bar, value in zip(bars, riemann_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value * 100:.1f}%", ha="center", weight="bold")

    bilateral_values = [
        bilateral["nine_channel"]["balanced"]["intent_balanced_accuracy"],
        bilateral["fifteen_channel"]["balanced"]["intent_balanced_accuracy"],
    ]
    bars = axes[2].bar(riemann_labels, bilateral_values, color=[COLORS["riemann"], COLORS["alignment"]])
    axes[2].set_title("Bilateral Imagery Montage")
    axes[2].set_ylabel("Intent balanced accuracy")
    axes[2].set_ylim(0.35, 0.80)
    for bar, value in zip(bars, bilateral_values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value * 100:.1f}%", ha="center", weight="bold")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
    peak = individualized["peak_summary"]
    fig.suptitle(
        f"Feature-Design Ablations on Final Subjects | Individual mu median {peak['median_hz']:.1f} Hz",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output_dir / "feature_design_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_artifact_audit(report: Mapping[str, object], output_dir: Path) -> None:
    results = report["artifact_thresholds"]
    cutoffs = sorted(int(value) for value in results)
    retained = [results[str(value)]["retained_final_percent"] for value in cutoffs]
    balanced = [results[str(value)]["balanced"]["balanced_accuracy"] for value in cutoffs]
    false_activation = [
        results[str(value)]["balanced"]["false_activation_rate"] for value in cutoffs
    ]
    fig, axis = plt.subplots(figsize=(8.6, 5.1))
    axis.plot(cutoffs, np.asarray(balanced) * 100, marker="o", linewidth=2.2, label="Intent balanced accuracy")
    axis.plot(cutoffs, np.asarray(false_activation) * 100, marker="s", linewidth=2.2, label="False activation")
    axis.set_xlabel("Maximum retained peak-to-peak amplitude (microvolts)")
    axis.set_ylabel("Final unseen-subject rate (%)")
    axis.set_title("Artifact Rejection: Signal Quality Versus Trial Retention")
    axis.grid(alpha=0.22)
    retention_axis = axis.twinx()
    retention_axis.plot(cutoffs, retained, color="#6C757D", marker="^", linestyle="--", linewidth=1.8, label="Trials retained")
    retention_axis.set_ylabel("Final trials retained (%)")
    lines, labels = axis.get_legend_handles_labels()
    lines_2, labels_2 = retention_axis.get_legend_handles_labels()
    axis.legend(lines + lines_2, labels + labels_2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "artifact_rejection_audit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cue_audit(report: Mapping[str, object], output_dir: Path) -> None:
    audit = report["existing_results"]["cue_audit"]
    confound = audit["left_right_confound_audit"]
    labels = ["Early visual\n80-300 ms", "Late motor ERD\n0.5-3.5 s"]
    values = [confound["visual_early"]["final_balanced_accuracy"], confound["motor_late"]["final_balanced_accuracy"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    bars = axes[0].bar(labels, values, color=["#CC8B65", COLORS["alignment"]])
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="Chance")
    axes[0].set_ylim(0.35, 0.70)
    axes[0].set_ylabel("Left/right balanced accuracy")
    axes[0].set_title("Cue-Confound Validation")
    axes[0].legend(frameon=False)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value * 100:.1f}%", ha="center", weight="bold")

    subject_r = np.asarray([row["r"] for row in audit["correlation"]["per_subject"]])
    axes[1].hist(subject_r, bins=np.linspace(-0.65, 0.85, 16), color=COLORS["riemann"], edgecolor="white")
    axes[1].axvline(audit["correlation"]["mean_r"], color="#A33A2B", linewidth=2, label=f"Mean r = {audit['correlation']['mean_r']:.3f}")
    axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Within-subject cue-to-ERD correlation")
    axes[1].set_ylabel("Subjects")
    axes[1].set_title("Cue-to-Motor Association")
    axes[1].legend(frameon=False)
    fig.suptitle("Visual Cue Information Is Separated From Later Motor-Imagery Features", fontsize=14, weight="bold")
    fig.savefig(output_dir / "cue_confound_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_controller(report: Mapping[str, object], output_dir: Path) -> None:
    temporal = report["existing_results"]["temporal"]["final_target"]
    names = ["Single window", "Responsive", "3-of-3 safety"]
    keys = ["single_window_no_ood", "responsive_rest_aware_policy", "safety_rest_aware_policy"]
    metrics = [temporal[key]["mean_patient_metrics"] for key in keys]
    false_per_minute = [row["false_activations_per_minute"] for row in metrics]
    sensitivity = [1.0 - row["missed_intent_rate"] for row in metrics]
    latency = [row["median_true_intent_latency_seconds"] for row in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), constrained_layout=True)
    bars = axes[0].bar(names, false_per_minute, color=[COLORS["baseline"], COLORS["adaptation"], COLORS["safety"]])
    axes[0].set_ylabel("False activations per minute")
    axes[0].set_title("Command Safety")
    axes[0].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, false_per_minute):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center", weight="bold")
    for name, x, y, delay in zip(names, false_per_minute, sensitivity, latency):
        axes[1].scatter(x, y, s=95)
        axes[1].annotate(f"{name}\n{delay:.1f} s latency", (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("False activations per minute")
    axes[1].set_ylabel("Intent sensitivity")
    axes[1].set_ylim(0, 0.8)
    axes[1].set_title("Safety, Detection, and Latency")
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("Temporal Confirmation Reduces False Commands but Increases Missed Intent", fontsize=14, weight="bold")
    fig.savefig(output_dir / "temporal_confirmation_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_markdown_report(report: Mapping[str, object], output_path: Path) -> None:
    rows = intent_summary_rows(report)
    best = max(rows, key=lambda row: row["balanced_accuracy"])
    safety_best = min(rows, key=lambda row: row["false_activation_rate"])
    hard = report["hard_negative"]
    individual = report["individualized_erd"]
    riemann = report["riemannian"]["fifteen_channel"]
    calibration = report["patient_calibration"]
    bilateral = report["bilateral_montage"]
    cue = report["existing_results"]["cue_audit"]
    temporal = report["existing_results"]["temporal"]["final_target"]
    artifact = report["artifact_thresholds"]

    table_lines = [
        "| Model | Intent balanced accuracy | False activation | Missed intent | Intent sensitivity |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['name']} | {percentage(row['balanced_accuracy'])} | "
            f"{percentage(row['false_activation_rate'])} | {percentage(row['missed_intent_rate'])} | "
            f"{percentage(row['intent_sensitivity'])} |"
        )
    calibration_lines = [
        "| Trials per rest/left/right class | Total labels per patient | Adapted balanced accuracy | False activation | Missed intent | Local weight |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in sorted(calibration, key=int):
        item = calibration[budget]
        metrics = item["adapted_balanced"]
        calibration_lines.append(
            f"| {budget} | {item['total_labeled_calibration_trials_per_subject']} | "
            f"{percentage(metrics['balanced_accuracy'])} | {percentage(metrics['false_activation_rate'])} | "
            f"{percentage(metrics['missed_intent_rate'])} | {item['selected_local_model_weight']:.2f} |"
        )
    artifact_lines = [
        "| Peak-to-peak cutoff | Final trials retained | Balanced accuracy | False activation | Missed intent |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cutoff in sorted(artifact, key=int):
        item = artifact[cutoff]
        metrics = item["balanced"]
        artifact_lines.append(
            f"| {cutoff} microvolts | {item['retained_final_percent']:.1f}% | "
            f"{percentage(metrics['balanced_accuracy'])} | {percentage(metrics['false_activation_rate'])} | "
            f"{percentage(metrics['missed_intent_rate'])} |"
        )
    temporal_lines = [
        "| Controller policy | False activations/min | Intent sensitivity | Missed intent | Median latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Single window", "single_window_no_ood"),
        ("Responsive rest-aware", "responsive_rest_aware_policy"),
        ("3-of-3 safety", "safety_rest_aware_policy"),
    ):
        metrics = temporal[key]["mean_patient_metrics"]
        temporal_lines.append(
            f"| {label} | {metrics['false_activations_per_minute']:.2f} | "
            f"{percentage(1.0 - metrics['missed_intent_rate'])} | "
            f"{percentage(metrics['missed_intent_rate'])} | "
            f"{metrics['median_true_intent_latency_seconds']:.1f} s |"
        )

    markdown = f"""# Advanced Motor-Imagery Pipeline Results

## Scope and protocol

This suite evaluates motor-imagery intent detection using the fixed, subject-disjoint split in `artifacts/dataset/splits.json`. Development subjects fit model parameters, controller-validation subjects select hyperparameters and thresholds, and the locked final subjects supply the percentages below. Calibration uses Runs 4 and 8; final evaluation uses Run 12. No Run 12 final-subject label is used for fitting or threshold selection.

The primary target is **rest versus motor imagery intent**. Left/right direction remains a separate second-stage problem. Every model has a balanced operating point and a more conservative safety operating point selected without examining final-subject labels.

## Final unseen-subject comparison

{chr(10).join(table_lines)}

The highest measured balanced intent accuracy in this comparison was **{percentage(best['balanced_accuracy'])}** from **{best['name']}**. The lowest balanced-point false activation was **{percentage(safety_best['false_activation_rate'])}** from **{safety_best['name']}**. These are different objectives; neither number should be reported without its paired missed-intent rate.

Hard-negative training selected the top **{percentage(hard['selected_hard_rest_fraction'])}** most confusing development-rest trials at weight **{hard['selected_hard_rest_weight']:.1f}**. Its final balanced point reached **{percentage(hard['balanced']['balanced_accuracy'])}** balanced accuracy, **{percentage(hard['balanced']['false_activation_rate'])}** false activation, and **{percentage(hard['balanced']['missed_intent_rate'])}** missed intent.

## Feature and preprocessing ablations

Fixed 8-13 Hz ERD reached **{percentage(individual['fixed']['balanced']['balanced_accuracy'])}**, while individualized peak-centered mu ERD reached **{percentage(individual['individualized']['balanced']['balanced_accuracy'])}**. Estimated individual mu peaks had a median of **{individual['peak_summary']['median_hz']:.1f} Hz** and ranged from **{individual['peak_summary']['minimum_hz']:.1f} to {individual['peak_summary']['maximum_hz']:.1f} Hz**.

The 15-channel Riemannian tangent-space detector changed from **{percentage(riemann['unaligned_tangent']['balanced']['balanced_accuracy'])}** without alignment to **{percentage(riemann['euclidean_aligned_tangent']['balanced']['balanced_accuracy'])}** after subject-wise Euclidean alignment. This result tests cross-subject covariance normalization rather than cue-side classification.

For bilateral imagery, the 9-channel FBCSP intent gate reached **{percentage(bilateral['nine_channel']['balanced']['intent_balanced_accuracy'])}**, compared with **{percentage(bilateral['fifteen_channel']['balanced']['intent_balanced_accuracy'])}** using the expanded sensorimotor montage.

### Artifact threshold audit

{chr(10).join(artifact_lines)}

The strict 100 microvolt audit point reached **{percentage(artifact['100']['balanced']['balanced_accuracy'])}** on the **{artifact['100']['retained_final_percent']:.1f}%** of final trials it retained. This is not directly comparable to a full-test result because 30.6% of the trials were rejected. Amplitude rejection is therefore treated as an audit, not automatic proof of artifact removal.

## Patient calibration

{chr(10).join(calibration_lines)}

These calibration results blend a global aligned-Riemannian model with a small patient-specific logistic model. Calibration labels come only from Runs 4 and 8; each patient is evaluated on later Run 12 trials.

## Cue validity and command safety

The early visual-window left/right classifier scored **{percentage(cue['left_right_confound_audit']['visual_early']['final_balanced_accuracy'])}**, while later motor ERD scored **{percentage(cue['left_right_confound_audit']['motor_late']['final_balanced_accuracy'])}**. The mean within-subject cue-to-ERD correlation was **r={cue['correlation']['mean_r']:.3f}** (two-sided p={cue['correlation']['two_sided_p']:.3g}). This audit supports using the late sensorimotor window, while also showing that cue timing alone is not a substitute for motor-intent evidence.

{chr(10).join(temporal_lines)}

Temporal confirmation lowers command frequency, but the present 3-of-3 rule pays for that safety with more missed intent and longer latency. The practical next controller should use hysteresis/evidence accumulation and report false activations per minute, misses, and latency together.

## Figures

- `advanced_intent_model_comparison.png`: final intent accuracy, false activation, and missed intent across models.
- `advanced_safety_responsiveness.png`: intent sensitivity versus rest false activation.
- `feature_design_ablation.png`: individualized mu and 9- versus 15-channel comparisons.
- `artifact_rejection_audit.png`: artifact cutoff, retained data, and performance.
- `patient_calibration_results.png`: calibration-budget response.
- `cue_confound_validation.png`: early visual versus late motor information.
- `temporal_confirmation_tradeoff.png`: false commands, sensitivity, and latency.

## Interpretation

The results should be presented as a benchmark of a difficult cross-subject problem, not as clinical-device validation. Cue-locked EDF annotations do not provide a separate physiological onset label, and the dataset contains healthy participants rather than stroke rehabilitation patients. Before deployment, the strongest offline model still needs a continuous no-control-state evaluation, session-to-session validation, and patient-specific calibration on the intended clinical population.
"""
    output_path.write_text(markdown, encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "files/eegmmidb/1.0.0",
    )
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts/dataset/splits.json")
    parser.add_argument("--raw-cache", type=Path, default=PROJECT_ROOT / "artifacts/cross_task/all_task_epochs.npz")
    parser.add_argument("--covariance-cache", type=Path, default=PROJECT_ROOT / "artifacts/patient_adaptive/covariance_cache.npz")
    parser.add_argument("--morlet-features", type=Path, default=PROJECT_ROOT / "artifacts/morlet_feature_audit/morlet_erd_trial_features.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/advanced_imagery")
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--calibration-budgets", default="1,3,5,10")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-bilateral-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        partitions = three_way_subject_split(subjects, seed=args.seed)
    else:
        partitions = load_partitions(args.splits)
    budgets = [int(value) for value in args.calibration_budgets.split(",") if value.strip()]
    print("\n=== Advanced Motor-Imagery Research Suite ===")
    print(f"Data: {data_dir}")
    print(
        f"Subjects: {len(subjects)} available | {len(partitions.development)} development | "
        f"{len(partitions.controller_validation)} controller | {len(partitions.final_target)} final"
    )

    morlet_table = load_morlet_features(args.morlet_features)
    print("[+] Evaluating hard-negative rest training and artifact thresholds...")
    hard_negative = hard_negative_experiment(
        morlet_table, partitions, args.max_false_activation, args.seed
    )
    artifact_thresholds = artifact_threshold_experiment(
        morlet_table, partitions, args.max_false_activation, args.seed
    )

    print("[+] Evaluating fixed versus individualized mu-band ERD...")
    raw_dataset = load_raw_cache(args.raw_cache)
    individualized_features = individualized_erd_features(
        raw_dataset,
        args.output_dir / "individualized_erd_features.npz",
        args.rebuild_feature_cache,
    )
    individualized_erd = compare_individualized_erd(
        individualized_features, partitions, args.max_false_activation, args.seed
    )

    print("[+] Evaluating Riemannian tangent space and Euclidean alignment...")
    riemannian, adaptation_payload = riemannian_experiments(
        args.covariance_cache, partitions, args.max_false_activation, args.seed
    )
    print("[+] Evaluating patient-specific calibration budgets...")
    patient_calibration = patient_calibration_experiment(
        adaptation_payload,
        partitions,
        budgets,
        args.max_false_activation,
        args.seed,
    )

    print("[+] Evaluating 9- versus 15-channel bilateral imagery montages...")
    bilateral_montage, bilateral_models = bilateral_montage_experiment(
        data_dir,
        subjects,
        partitions,
        args.output_dir / "bilateral_15ch_epochs.npz",
        args.rebuild_bilateral_cache,
        args.max_false_activation,
        args.seed,
    )
    report: Dict[str, object] = {
        "protocol": {
            "task": "advanced rest-versus-motor-imagery intent benchmark",
            "development_subjects": list(partitions.development),
            "controller_validation_subjects": list(partitions.controller_validation),
            "final_target_subjects": list(partitions.final_target),
            "calibration_runs": [4, 8],
            "final_evaluation_run": 12,
            "max_controller_false_activation": args.max_false_activation,
            "calibration_budgets_per_original_class": budgets,
            "threshold_policy": "controller subjects only; final subject labels remain locked",
        },
        "dataset": {
            "data_dir": str(data_dir),
            "available_subjects": len(subjects),
            "morlet_trials": int(len(morlet_table["labels"])),
            "raw_trials": int(len(raw_dataset.labels)),
        },
        "hard_negative": hard_negative,
        "artifact_thresholds": artifact_thresholds,
        "individualized_erd": individualized_erd,
        "riemannian": riemannian,
        "patient_calibration": patient_calibration,
        "bilateral_montage": bilateral_montage,
        "existing_results": existing_results(PROJECT_ROOT),
        "runtime_minutes": (time.perf_counter() - started) / 60.0,
    }
    metrics_path = args.output_dir / "advanced_imagery_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="ascii")
    joblib.dump(bilateral_models, args.output_dir / "bilateral_montage_models.joblib")
    joblib.dump(
        {
            "channels": list(MOTOR_CHANNELS),
            "calibration_runs": [4, 8],
            "evaluation_run": 12,
            "tangent_transformers": adaptation_payload["transformers"],
            "intent_pipeline": adaptation_payload["model"],
            "note": "Compute target-subject Euclidean alignment references from calibration runs before tangent transformation.",
        },
        args.output_dir / "aligned_riemannian_intent_model.joblib",
    )
    plot_model_comparison(report, args.output_dir)
    plot_safety_plane(report, args.output_dir)
    plot_calibration(report, args.output_dir)
    plot_feature_design(report, args.output_dir)
    plot_artifact_audit(report, args.output_dir)
    plot_cue_audit(report, args.output_dir)
    plot_temporal_controller(report, args.output_dir)
    write_markdown_report(report, args.output_dir / "ADVANCED_IMAGERY_RESULTS.md")

    best = max(intent_summary_rows(report), key=lambda row: row["balanced_accuracy"])
    print("\n=== Final Locked-Subject Highlights ===")
    print(
        f"Best balanced intent model: {best['name']} | "
        f"balanced accuracy={percentage(best['balanced_accuracy'])} | "
        f"false activation={percentage(best['false_activation_rate'])} | "
        f"missed intent={percentage(best['missed_intent_rate'])}"
    )
    print(f"[+] Saved metrics: {metrics_path}")
    print(f"[+] Saved report and seven figures: {args.output_dir}")
    print(f"[+] Runtime: {(time.perf_counter() - started) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
