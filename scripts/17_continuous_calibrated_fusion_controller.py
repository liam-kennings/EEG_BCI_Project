#!/usr/bin/env python3
"""Continuous patient-calibrated ERD/Riemannian motor-imagery controller."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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
import mne  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy.signal import butter, sosfilt  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bci_pipeline.continuous import (  # noqa: E402
    ContinuousStream,
    EvidencePolicy,
    aggregate_metrics,
    command_metrics,
    simulate_stream,
)
from bci_pipeline.dataset import (  # noqa: E402
    IMAGERY_RUNS,
    available_subjects,
    find_run_file,
    map_imagery_annotation,
    normalize_channel_name,
    resolve_data_dir,
)
from bci_pipeline.preprocessing import CAUSAL_PROFILE, preprocess_run  # noqa: E402
from bci_pipeline.representations.riemann import FILTER_BANK, TEMPORAL_CHANNELS  # noqa: E402
from bci_pipeline.splits import SubjectPartitions, load_partitions  # noqa: E402


mne.set_log_level("WARNING")
ARTIFACT_UV = 150.0
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.25
POWER_BANDS = (("mu", 8.0, 13.0), ("beta_low", 13.0, 20.0), ("beta_high", 20.0, 30.0))
COLORS = {"baseline": "#7B8DA1", "evidence": "#D97642", "fusion": "#1F7A8C", "safe": "#2A9D6F", "artifact": "#A23B72"}


@dataclass
class ContinuousFeatureDataset:
    erd_features: np.ndarray
    riemann_features: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    times: np.ndarray
    segments: np.ndarray
    segment_onsets: np.ndarray
    window_indices: np.ndarray
    peak_to_peak_uv: np.ndarray
    channel_names: List[str]


@dataclass
class DecoderBundle:
    erd_intent: Pipeline
    riemann_intent: Pipeline
    erd_direction: Pipeline
    riemann_direction: Pipeline


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


def cache_fingerprint(subjects: Sequence[int]) -> str:
    return json.dumps(
        {
            "version": 1,
            "subjects": list(subjects),
            "runs": list(IMAGERY_RUNS),
            "channels": list(TEMPORAL_CHANNELS),
            "preprocessing": CAUSAL_PROFILE.fingerprint(),
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "power_bands": POWER_BANDS,
            "covariance_bands": FILTER_BANK,
            "alignment_artifact_uv": ARTIFACT_UV,
        },
        sort_keys=True,
    )


def empirical_shrinkage_covariance(windows: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    centered = windows - np.mean(windows, axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", centered, centered, optimize=True)
    covariance /= max(windows.shape[-1] - 1, 1)
    trace = np.trace(covariance, axis1=1, axis2=2) / covariance.shape[1]
    identity = np.eye(covariance.shape[1])[None]
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace[:, None, None] * identity
    covariance /= np.clip(
        np.trace(covariance, axis1=1, axis2=2)[:, None, None], 1e-15, None
    )
    return covariance.astype(np.float32)


def inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = max(float(np.max(values)) * 1e-8, 1e-12)
    return (vectors * np.clip(values, floor, None) ** -0.5) @ vectors.T


def log_vectorize(covariances: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(
        0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    )
    log_values = np.log(np.clip(values, 1e-10, None))
    log_matrices = np.einsum(
        "nik,nk,njk->nij", vectors, log_values, vectors, optimize=True
    )
    row, column = np.triu_indices(covariances.shape[-1])
    output = log_matrices[:, row, column]
    output[:, row != column] *= math.sqrt(2.0)
    return output.astype(np.float32)


def annotation_labels(
    raw: mne.io.BaseRaw,
    window_end_seconds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        (
            float(annotation["onset"]),
            float(annotation["duration"]),
            map_imagery_annotation(str(annotation["description"])),
        )
        for annotation in raw.annotations
    ]
    rows = [row for row in rows if row[2] is not None]
    onsets = np.asarray([row[0] for row in rows], dtype=float)
    durations = np.asarray([row[1] for row in rows], dtype=float)
    annotation_values = np.asarray([row[2] for row in rows], dtype=np.int64)
    lookup_times = window_end_seconds - 1.0 / float(raw.info["sfreq"])
    segment_indices = np.searchsorted(onsets, lookup_times, side="right") - 1
    valid = segment_indices >= 0
    clipped = np.clip(segment_indices, 0, len(rows) - 1)
    valid &= lookup_times <= onsets[clipped] + durations[clipped] + 1e-6
    return annotation_values[clipped], clipped.astype(np.int64), onsets[clipped], valid


def extract_run_windows(
    raw: mne.io.BaseRaw,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    processed = preprocess_run(raw, CAUSAL_PROFILE, TEMPORAL_CHANNELS)
    signal = processed.get_data()
    sfreq = float(processed.info["sfreq"])
    window_samples = int(round(WINDOW_SECONDS * sfreq))
    step_samples = int(round(STEP_SECONDS * sfreq))
    starts = np.arange(0, signal.shape[1] - window_samples + 1, step_samples)
    window_end_seconds = (starts + window_samples) / sfreq
    labels, segments, segment_onsets, valid = annotation_labels(processed, window_end_seconds)
    starts = starts[valid]
    window_end_seconds = window_end_seconds[valid]
    labels, segments, segment_onsets = labels[valid], segments[valid], segment_onsets[valid]
    broad_windows = np.stack([signal[:, start : start + window_samples] for start in starts])
    peak_to_peak_uv = np.max(np.ptp(broad_windows, axis=-1), axis=-1) * 1e6

    power_rows: List[np.ndarray] = []
    for _, low_hz, high_hz in POWER_BANDS:
        filtered = sosfilt(
            butter(4, (low_hz, high_hz), btype="bandpass", fs=sfreq, output="sos"),
            signal,
            axis=-1,
        )
        windows = np.stack([filtered[:, start : start + window_samples] for start in starts])
        power_rows.append(np.log(np.mean(windows**2, axis=-1) + 1e-20))

    covariance_rows: List[np.ndarray] = []
    for _, low_hz, high_hz in FILTER_BANK:
        filtered = sosfilt(
            butter(4, (low_hz, high_hz), btype="bandpass", fs=sfreq, output="sos"),
            signal,
            axis=-1,
        )
        windows = np.stack([filtered[:, start : start + window_samples] for start in starts])
        covariance_rows.append(empirical_shrinkage_covariance(windows))
    return (
        np.stack(power_rows, axis=1).astype(np.float32),
        np.stack(covariance_rows, axis=1),
        labels,
        segments,
        segment_onsets,
        window_end_seconds.astype(np.float32),
        peak_to_peak_uv.astype(np.float32),
    )


def subject_features(
    data_dir: Path,
    subject: int,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    rows: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "power",
            "covariances",
            "labels",
            "runs",
            "segments",
            "segment_onsets",
            "times",
            "peak_to_peak_uv",
            "window_indices",
        )
    }
    channel_names: List[str] = []
    for run in IMAGERY_RUNS:
        path = find_run_file(data_dir, subject, run)
        if path is None:
            raise FileNotFoundError(f"Missing S{subject:03d}R{run:02d}.edf")
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if not channel_names:
            lookup = {
                normalize_channel_name(name): name for name in raw.copy().pick("eeg").ch_names
            }
            channel_names = [lookup[normalize_channel_name(name)] for name in TEMPORAL_CHANNELS]
        power, covariances, labels, segments, onsets, times, amplitudes = extract_run_windows(raw)
        count = len(labels)
        rows["power"].append(power)
        rows["covariances"].append(covariances)
        rows["labels"].append(labels)
        rows["runs"].append(np.full(count, run, dtype=np.int64))
        rows["segments"].append(segments)
        rows["segment_onsets"].append(onsets.astype(np.float32))
        rows["times"].append(times)
        rows["peak_to_peak_uv"].append(amplitudes)
        rows["window_indices"].append(np.arange(count, dtype=np.int64))

    combined = {key: np.concatenate(values) for key, values in rows.items()}
    clean_calibration = (
        np.isin(combined["runs"], (4, 8))
        & (combined["peak_to_peak_uv"] <= ARTIFACT_UV)
    )
    rest_calibration = clean_calibration & (combined["labels"] == 0)
    if not np.any(rest_calibration):
        raise RuntimeError(f"Subject {subject} has no clean calibration-rest windows.")

    rest_log_power = np.median(combined["power"][rest_calibration], axis=0)
    erd = (10.0 / math.log(10.0)) * (combined["power"] - rest_log_power)
    channel_lookup = {
        normalize_channel_name(name): index for index, name in enumerate(channel_names)
    }
    c3, c4 = channel_lookup["C3"], channel_lookup["C4"]
    lateralization = (erd[:, :, c3] - erd[:, :, c4])[:, :, None]
    erd_features = np.concatenate((erd.reshape(len(erd), -1), lateralization.reshape(len(erd), -1)), axis=1)

    covariance_blocks: List[np.ndarray] = []
    for band in range(combined["covariances"].shape[1]):
        reference = np.mean(
            combined["covariances"][clean_calibration, band].astype(np.float64), axis=0
        )
        whitening = inverse_square_root(reference)
        covariance = combined["covariances"][:, band].astype(np.float64)
        aligned = np.einsum(
            "ij,njk,kl->nil", whitening, covariance, whitening, optimize=True
        )
        trace = np.trace(aligned, axis1=1, axis2=2)
        aligned /= np.clip(trace[:, None, None], 1e-12, None)
        covariance_blocks.append(log_vectorize(aligned))
    combined["erd_features"] = erd_features.astype(np.float32)
    combined["riemann_features"] = np.concatenate(covariance_blocks, axis=1)
    return combined, channel_names


def save_feature_cache(
    path: Path,
    fingerprint: str,
    dataset: ContinuousFeatureDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprint=np.asarray(fingerprint),
        erd_features=dataset.erd_features,
        riemann_features=dataset.riemann_features,
        labels=dataset.labels,
        subjects=dataset.subjects,
        runs=dataset.runs,
        times=dataset.times,
        segments=dataset.segments,
        segment_onsets=dataset.segment_onsets,
        window_indices=dataset.window_indices,
        peak_to_peak_uv=dataset.peak_to_peak_uv,
        channel_names=np.asarray(dataset.channel_names),
    )


def load_feature_cache(path: Path, fingerprint: str) -> ContinuousFeatureDataset | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["fingerprint"].item()) != fingerprint:
            return None
        return ContinuousFeatureDataset(
            erd_features=cached["erd_features"],
            riemann_features=cached["riemann_features"],
            labels=cached["labels"],
            subjects=cached["subjects"],
            runs=cached["runs"],
            times=cached["times"],
            segments=cached["segments"],
            segment_onsets=cached["segment_onsets"],
            window_indices=cached["window_indices"],
            peak_to_peak_uv=cached["peak_to_peak_uv"],
            channel_names=[str(value) for value in cached["channel_names"]],
        )


def build_feature_dataset(
    data_dir: Path,
    subjects: Sequence[int],
    cache_path: Path,
    rebuild: bool,
) -> ContinuousFeatureDataset:
    fingerprint = cache_fingerprint(subjects)
    if not rebuild:
        cached = load_feature_cache(cache_path, fingerprint)
        if cached is not None:
            print(f"[+] Loaded {len(cached.labels)} continuous windows from {cache_path}")
            return cached

    output: Dict[str, List[np.ndarray]] = {
        key: []
        for key in (
            "erd_features",
            "riemann_features",
            "labels",
            "subjects",
            "runs",
            "times",
            "segments",
            "segment_onsets",
            "window_indices",
            "peak_to_peak_uv",
        )
    }
    represented: List[int] = []
    channel_names: List[str] = []
    print(f"[+] Extracting continuous causal features for {len(subjects)} subjects...")
    for position, subject in enumerate(subjects, start=1):
        try:
            rows, current_names = subject_features(data_dir, subject)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"[!] Skipping S{subject:03d}: {error}")
            continue
        represented.append(subject)
        channel_names = current_names
        count = len(rows["labels"])
        for key in output:
            if key == "subjects":
                output[key].append(np.full(count, subject, dtype=np.int64))
            else:
                output[key].append(rows[key])
        if position % 10 == 0 or position == len(subjects):
            print(
                f"    processed {position:3d}/{len(subjects)} subjects; "
                f"kept {sum(len(values) for values in output['labels'])} windows"
            )
    dataset = ContinuousFeatureDataset(
        erd_features=np.concatenate(output["erd_features"]),
        riemann_features=np.concatenate(output["riemann_features"]),
        labels=np.concatenate(output["labels"]),
        subjects=np.concatenate(output["subjects"]),
        runs=np.concatenate(output["runs"]),
        times=np.concatenate(output["times"]),
        segments=np.concatenate(output["segments"]),
        segment_onsets=np.concatenate(output["segment_onsets"]),
        window_indices=np.concatenate(output["window_indices"]),
        peak_to_peak_uv=np.concatenate(output["peak_to_peak_uv"]),
        channel_names=channel_names,
    )
    save_feature_cache(cache_path, fingerprint, dataset)
    print(
        f"[+] Saved {len(dataset.labels)} windows from {len(represented)} subjects to {cache_path}"
    )
    return dataset


def make_decoder(feature_count: int, seed: int, selected_features: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("selection", SelectKBest(f_classif, k=min(selected_features, feature_count))),
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


def fit_global_decoders(
    dataset: ContinuousFeatureDataset,
    partitions: SubjectPartitions,
    seed: int,
) -> DecoderBundle:
    development = np.isin(dataset.subjects, partitions.development)
    clean = dataset.peak_to_peak_uv <= ARTIFACT_UV
    decorrelated = dataset.window_indices % int(round(WINDOW_SECONDS / STEP_SECONDS)) == 0
    training = development & clean & decorrelated
    intent_labels = (dataset.labels[training] != 0).astype(np.int64)
    movement = training & (dataset.labels != 0)
    print(
        f"[+] Fitting global decoders on {np.sum(training)} decorrelated clean windows "
        f"({np.sum(movement)} movement windows)..."
    )
    return DecoderBundle(
        erd_intent=make_decoder(dataset.erd_features.shape[1], seed, 30).fit(
            dataset.erd_features[training], intent_labels
        ),
        riemann_intent=make_decoder(dataset.riemann_features.shape[1], seed, 100).fit(
            dataset.riemann_features[training], intent_labels
        ),
        erd_direction=make_decoder(dataset.erd_features.shape[1], seed + 1, 30).fit(
            dataset.erd_features[movement], dataset.labels[movement]
        ),
        riemann_direction=make_decoder(
            dataset.riemann_features.shape[1], seed + 1, 100
        ).fit(dataset.riemann_features[movement], dataset.labels[movement]),
    )


def calibration_mask(
    dataset: ContinuousFeatureDataset,
    subject: int,
    budget_per_class: int,
) -> np.ndarray:
    selected = np.zeros(len(dataset.labels), dtype=bool)
    subject_calibration = (dataset.subjects == subject) & np.isin(dataset.runs, (4, 8))
    for label in (0, 1, 2):
        candidate = subject_calibration & (dataset.labels == label)
        keys = dataset.runs[candidate] * 100 + dataset.segments[candidate]
        ordered_keys = []
        for key in keys:
            if int(key) not in ordered_keys:
                ordered_keys.append(int(key))
        if len(ordered_keys) < budget_per_class:
            return np.zeros(len(dataset.labels), dtype=bool)
        chosen = set(ordered_keys[:budget_per_class])
        selected |= candidate & np.isin(
            dataset.runs * 100 + dataset.segments, list(chosen)
        )
    return selected


def class_probability(model: Pipeline, features: np.ndarray, label: int) -> np.ndarray:
    classes = model.named_steps["model"].classes_.tolist()
    return model.predict_proba(features)[:, classes.index(label)]


def local_probability(
    global_model: Pipeline,
    features: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    target_label: int,
    seed: int,
) -> np.ndarray:
    transformed_calibration = global_model[:-1].transform(features[calibration])
    transformed_all = global_model[:-1].transform(features)
    local = LogisticRegression(
        C=0.05,
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=seed,
    ).fit(transformed_calibration, labels[calibration])
    classes = local.classes_.tolist()
    return local.predict_proba(transformed_all)[:, classes.index(target_label)]


def score_subject(
    dataset: ContinuousFeatureDataset,
    decoders: DecoderBundle,
    subject: int,
    budget_per_class: int,
    seed: int,
) -> ContinuousStream | None:
    selected = dataset.subjects == subject
    if not np.any(selected):
        return None
    local_calibration_global = calibration_mask(dataset, subject, budget_per_class)
    if not np.any(local_calibration_global):
        return None
    calibration = local_calibration_global[selected]
    clean_calibration = calibration & (dataset.peak_to_peak_uv[selected] <= ARTIFACT_UV)
    labels = dataset.labels[selected]
    intent_labels = (labels != 0).astype(np.int64)
    erd = dataset.erd_features[selected]
    riemann = dataset.riemann_features[selected]

    erd_global_intent = class_probability(decoders.erd_intent, erd, 1)
    riemann_global_intent = class_probability(decoders.riemann_intent, riemann, 1)
    erd_local_intent = local_probability(
        decoders.erd_intent,
        erd,
        intent_labels,
        clean_calibration,
        1,
        seed + subject,
    )
    riemann_local_intent = local_probability(
        decoders.riemann_intent,
        riemann,
        intent_labels,
        clean_calibration,
        1,
        seed + subject,
    )
    movement_calibration = clean_calibration & (labels != 0)
    erd_global_direction = class_probability(decoders.erd_direction, erd, 1)
    riemann_global_direction = class_probability(decoders.riemann_direction, riemann, 1)
    erd_local_direction = local_probability(
        decoders.erd_direction,
        erd,
        labels,
        movement_calibration,
        1,
        seed + 1000 + subject,
    )
    riemann_local_direction = local_probability(
        decoders.riemann_direction,
        riemann,
        labels,
        movement_calibration,
        1,
        seed + 1000 + subject,
    )
    order = np.lexsort((dataset.times[selected], dataset.runs[selected]))
    return ContinuousStream(
        subject=subject,
        runs=dataset.runs[selected][order],
        times=dataset.times[selected][order],
        labels=labels[order],
        segments=dataset.segments[selected][order],
        segment_onsets=dataset.segment_onsets[selected][order],
        erd_global_intent=erd_global_intent[order],
        erd_local_intent=erd_local_intent[order],
        riemann_global_intent=riemann_global_intent[order],
        riemann_local_intent=riemann_local_intent[order],
        erd_global_direction=erd_global_direction[order],
        erd_local_direction=erd_local_direction[order],
        riemann_global_direction=riemann_global_direction[order],
        riemann_local_direction=riemann_local_direction[order],
        artifact_rejected=(dataset.peak_to_peak_uv[selected][order] > ARTIFACT_UV),
        calibration_mask=calibration[order],
        step_seconds=STEP_SECONDS,
    )


def collect_streams(
    dataset: ContinuousFeatureDataset,
    decoders: DecoderBundle,
    subjects: Iterable[int],
    budget_per_class: int,
    seed: int,
) -> Dict[int, ContinuousStream]:
    streams: Dict[int, ContinuousStream] = {}
    for subject in subjects:
        stream = score_subject(dataset, decoders, int(subject), budget_per_class, seed)
        if stream is not None:
            streams[int(subject)] = stream
    return streams


def evaluate_policy(
    streams: Mapping[int, ContinuousStream],
    policy: EvidencePolicy,
    artifact_abstention: bool,
) -> Dict[str, object]:
    per_subject: Dict[str, Dict[str, object]] = {}
    for subject, stream in streams.items():
        simulation = simulate_stream(stream, policy, artifact_abstention)
        metrics = command_metrics(stream, simulation)
        if not artifact_abstention:
            metrics["artifact_abstention_rate"] = 0.0
        per_subject[f"S{subject:03d}"] = metrics
    return {
        "policy": asdict(policy),
        "artifact_abstention": artifact_abstention,
        "mean_patient_metrics": aggregate_metrics(per_subject),
        "per_subject": per_subject,
    }


def policy_rank(
    metrics: Mapping[str, object],
    target_fa: float,
    target_sensitivity: float,
    target_latency: float,
) -> Tuple[float, ...]:
    latency = metrics["median_latency_seconds"]
    latency_value = float(latency) if latency is not None else 99.0
    penalty = (
        max(0.0, float(metrics["false_activations_per_minute"]) - target_fa) / target_fa
        + max(0.0, target_sensitivity - float(metrics["intent_sensitivity"]))
        / target_sensitivity
        + max(0.0, latency_value - target_latency) / target_latency
    )
    return (
        -penalty,
        float(metrics["segment_balanced_accuracy"]),
        float(metrics["intent_sensitivity"]),
        -float(metrics["false_activations_per_minute"]),
        -float(metrics["wrong_direction_rate"]),
    )


def tune_policy(
    streams: Mapping[int, ContinuousStream],
    target_fa: float,
    target_sensitivity: float,
    target_latency: float,
) -> Tuple[Dict[str, EvidencePolicy], List[Dict[str, object]]]:
    frontier: List[Dict[str, object]] = []
    best: Tuple[Tuple[float, ...], EvidencePolicy] | None = None
    for erd_weight in (0.35, 0.65):
        for local_weight in (0.25, 0.50, 0.75):
            for rest_quantile in (0.80, 0.90, 0.95):
                for decay in (0.75, 0.90):
                    for activation in (1.0, 2.0, 3.0):
                        for release_ratio in (0.25, 0.50):
                            for cooldown in (0.5, 1.0):
                                policy = EvidencePolicy(
                                    erd_weight,
                                    local_weight,
                                    rest_quantile,
                                    decay,
                                    activation,
                                    release_ratio,
                                    3.0,
                                    cooldown,
                                )
                                result = evaluate_policy(streams, policy, True)
                                metrics = result["mean_patient_metrics"]
                                frontier.append({"policy": asdict(policy), **metrics})
                                rank = policy_rank(
                                    metrics, target_fa, target_sensitivity, target_latency
                                )
                                if best is None or rank > best[0]:
                                    best = (rank, policy)
    if best is None:
        raise RuntimeError("Policy search produced no candidates.")
    balanced_row = max(
        frontier,
        key=lambda row: (
            row["segment_balanced_accuracy"],
            row["intent_sensitivity"],
            -row["false_activations_per_minute"],
        ),
    )
    sensitivity_pool = [
        row
        for row in frontier
        if row["intent_sensitivity"] >= target_sensitivity
        and row["median_latency_seconds"] is not None
        and row["median_latency_seconds"] <= target_latency
    ]
    sensitivity_row = min(
        sensitivity_pool or frontier,
        key=lambda row: (
            max(0.0, target_sensitivity - row["intent_sensitivity"]),
            row["false_activations_per_minute"],
            -row["segment_balanced_accuracy"],
        ),
    )
    return {
        "safety": best[1],
        "balanced": EvidencePolicy(**balanced_row["policy"]),
        "sensitivity_target": EvidencePolicy(**sensitivity_row["policy"]),
    }, frontier


def single_window_policy(selected: EvidencePolicy) -> EvidencePolicy:
    return EvidencePolicy(
        erd_weight=selected.erd_weight,
        local_weight=selected.local_weight,
        rest_quantile=selected.rest_quantile,
        decay=0.0,
        activation_threshold=0.01,
        release_ratio=0.0,
        max_hold_seconds=STEP_SECONDS,
        cooldown_seconds=STEP_SECONDS,
    )


def plot_overview(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(14.2, 3.8))
    axis.set_axis_off()
    blocks = (
        ("Causal stream", "1 s windows\nevery 250 ms"),
        ("150 uV gate", "Artifact windows\nproduce no evidence"),
        ("Dual features", "Fixed-band ERD +\naligned covariance"),
        ("Patient adapter", "10 segments/class\nfrom Runs 4/8"),
        ("Evidence state", "Leaky accumulator +\nhysteresis/cooldown"),
        ("Command", "Intent then\nleft/right direction"),
    )
    positions = np.linspace(0.025, 0.825, len(blocks))
    for index, ((title, detail), x_value) in enumerate(zip(blocks, positions)):
        patch = FancyBboxPatch(
            (x_value, 0.33),
            0.14,
            0.34,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor=COLORS["fusion"],
            facecolor="#F2FAFC",
        )
        axis.add_patch(patch)
        axis.text(x_value + 0.07, 0.55, title, ha="center", weight="bold")
        axis.text(x_value + 0.07, 0.40, detail, ha="center", fontsize=8.5, color="#415466")
        if index < len(blocks) - 1:
            axis.annotate(
                "",
                xy=(x_value + 0.163, 0.50),
                xytext=(x_value + 0.143, 0.50),
                arrowprops={"arrowstyle": "->", "color": "#718096", "lw": 1.4},
            )
    axis.text(
        0.5,
        0.90,
        "Continuous Patient-Calibrated Motor-Imagery Command Pipeline",
        ha="center",
        fontsize=16,
        weight="bold",
    )
    axis.text(
        0.5,
        0.10,
        "All policy parameters are selected on controller subjects; final Run 12 streams remain locked.",
        ha="center",
        color="#415466",
    )
    fig.savefig(output_dir / "continuous_pipeline_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(
    output_dir: Path,
    frontier: Sequence[Mapping[str, object]],
    selected: Mapping[str, EvidencePolicy],
    target_fa: float,
    target_sensitivity: float,
) -> None:
    fig, axis = plt.subplots(figsize=(8.3, 6.0))
    scatter = axis.scatter(
        [row["false_activations_per_minute"] for row in frontier],
        [100 * row["intent_sensitivity"] for row in frontier],
        c=[row["median_latency_seconds"] or 5.0 for row in frontier],
        cmap="viridis_r",
        s=24,
        alpha=0.42,
    )
    marker_settings = {
        "safety": ("*", "#D1495B", "Safety selected"),
        "balanced": ("D", "#F4A261", "Balanced selected"),
        "sensitivity_target": ("s", "#2A9D6F", "Sensitivity selected"),
    }
    for name, policy in selected.items():
        selected_row = next(row for row in frontier if row["policy"] == asdict(policy))
        marker, color, label = marker_settings[name]
        axis.scatter(
            selected_row["false_activations_per_minute"],
            100 * selected_row["intent_sensitivity"],
            color=color,
            s=145,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=4,
        )
    axis.axvline(target_fa, linestyle="--", color="#596A7B", label=f"Target <= {target_fa:.1f} FA/min")
    axis.axhline(100 * target_sensitivity, linestyle=":", color="#596A7B", label=f"Target >= {target_sensitivity:.0%} sensitivity")
    axis.set_xlabel("Controller-subject false activations per rest minute")
    axis.set_ylabel("Controller-subject intent-segment sensitivity (%)")
    axis.set_title("Controller-Only Policy Selection Frontier")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Median detection latency (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "continuous_controller_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(
    output_dir: Path,
    comparisons: Mapping[str, Mapping[str, object]],
) -> None:
    names = list(comparisons)
    metrics = [row["mean_patient_metrics"] for row in comparisons.values()]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), constrained_layout=True)
    specifications = (
        ("false_activations_per_minute", "False Activations / Rest Minute", None),
        ("intent_sensitivity", "Intent-Segment Sensitivity", 100.0),
        ("median_latency_seconds", "Median Detection Latency", None),
    )
    colors = [
        COLORS["baseline"],
        COLORS["evidence"],
        "#F4A261",
        COLORS["safe"],
        COLORS["artifact"],
    ]
    for axis, (key, title, multiplier) in zip(axes, specifications):
        values = [float(row[key]) * (multiplier or 1.0) for row in metrics]
        bars = axis.bar(names, values, color=colors)
        for bar, value in zip(bars, values):
            suffix = "%" if multiplier else ""
            axis.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.035, f"{value:.1f}{suffix}", ha="center", weight="bold")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean final-subject event rate")
    axes[1].set_ylabel("Final intent segments detected (%)")
    axes[2].set_ylabel("Seconds from cue onset")
    fig.suptitle("Continuous Run 12 Controller: Ablation and Final Performance", fontsize=15, weight="bold")
    fig.savefig(output_dir / "continuous_final_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_tradeoff(output_dir: Path, result: Mapping[str, object]) -> None:
    rows = result["per_subject"]
    fig, axis = plt.subplots(figsize=(8.0, 5.8))
    x_values = [row["false_activations_per_minute"] for row in rows.values()]
    y_values = [100 * row["intent_sensitivity"] for row in rows.values()]
    sizes = [50 + 140 * row["artifact_abstention_rate"] for row in rows.values()]
    axis.scatter(x_values, y_values, s=sizes, color=COLORS["fusion"], alpha=0.80, edgecolor="white")
    ordered_names = list(rows)
    notable = set(
        sorted(ordered_names, key=lambda name: rows[name]["intent_sensitivity"], reverse=True)[:4]
        + sorted(ordered_names, key=lambda name: rows[name]["false_activations_per_minute"])[:4]
    )
    for name, x_value, y_value in zip(ordered_names, x_values, y_values):
        if name in notable:
            axis.annotate(
                name,
                (x_value, y_value),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axvline(1.0, linestyle="--", color="#596A7B", label="1 FA/min target")
    axis.axhline(60.0, linestyle=":", color="#596A7B", label="60% sensitivity target")
    axis.set_xlabel("False activations per rest minute")
    axis.set_ylabel("Intent-segment sensitivity (%)")
    axis.set_title("Final Patient Variability and Safety Tradeoff")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "continuous_final_subject_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_example_timeline(
    output_dir: Path,
    stream: ContinuousStream,
    policy: EvidencePolicy,
) -> None:
    simulation = simulate_stream(stream, policy, True)
    selected = stream.runs == 12
    times = stream.times[selected]
    labels = stream.labels[selected]
    probability = simulation["intent_probability"][selected]
    evidence = simulation["evidence_trace"][selected]
    active = simulation["active_trace"][selected]
    rejected = stream.artifact_rejected[selected]
    reference = simulation["rest_reference_probability"]
    fig, axes = plt.subplots(3, 1, figsize=(14.0, 7.7), sharex=True, constrained_layout=True)
    axes[0].step(times, labels, where="post", color="#364152", linewidth=1.5)
    axes[0].set_yticks([0, 1, 2], ["Rest", "Left", "Right"])
    axes[0].set_ylabel("Protocol state")
    axes[0].set_title(f"Example Final Continuous Stream: S{stream.subject:03d}, Run 12")
    axes[1].plot(times, probability, color=COLORS["fusion"], linewidth=1.3, label="Fused intent probability")
    axes[1].axhline(reference, color="#596A7B", linestyle="--", label="Patient rest reference")
    axes[1].scatter(times[rejected], probability[rejected], color=COLORS["artifact"], marker="x", s=22, label="Artifact abstention")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Probability")
    axes[1].legend(frameon=False, ncol=3, fontsize=8)
    axes[2].plot(times, evidence, color=COLORS["evidence"], linewidth=1.4, label="Accumulated evidence")
    axes[2].axhline(policy.activation_threshold, color="#D1495B", linestyle="--", label="Activation threshold")
    axes[2].fill_between(times, 0, np.max(evidence) if len(evidence) else 1, where=active, step="post", color="#CDEBDD", alpha=0.65, label="Command active")
    axes[2].set_ylabel("Evidence")
    axes[2].set_xlabel("Run time (seconds)")
    axes[2].legend(frameon=False, ncol=3, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.savefig(output_dir / "continuous_example_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(path: Path, report: Mapping[str, object]) -> None:
    selected = report["final_target"]["safety_evidence_plus_artifact"]["mean_patient_metrics"]
    balanced = report["final_target"]["balanced_evidence_plus_artifact"]["mean_patient_metrics"]
    responsive = report["final_target"]["sensitivity_evidence_plus_artifact"]["mean_patient_metrics"]
    baseline = report["final_target"]["single_window_plus_artifact"]["mean_patient_metrics"]
    no_artifact = report["final_target"]["evidence_without_artifact"]["mean_patient_metrics"]
    policies = report["selected_policies"]
    policy = policies["safety"]
    targets = report["protocol"]["selection_targets"]
    target_met = (
        selected["false_activations_per_minute"] <= targets["false_activations_per_minute"]
        and selected["intent_sensitivity"] >= targets["intent_sensitivity"]
        and selected["median_latency_seconds"] <= targets["median_latency_seconds"]
    )
    text = f"""# Continuous Calibrated Fusion Controller Results

## Protocol

This experiment processes complete unilateral motor-imagery Runs 4, 8, and 12 using causal 4-35 Hz filtering, one-second windows, and a 250 ms update interval. The fixed subject split remains unchanged: global decoders use development subjects, controller subjects select all fusion and command-policy parameters, and final subjects are evaluated only after selection.

Each target subject contributes 10 labeled calibration segments per rest/left/right class from Runs 4 and 8. Run 12 is evaluated as a chronological stream. Windows above 150 microvolts peak-to-peak abstain rather than voting for rest or intent.

## Safety-selected controller policy

- ERD probability weight: `{policy['erd_weight']:.2f}`
- Patient-local model weight: `{policy['local_weight']:.2f}`
- Calibration-rest quantile: `{policy['rest_quantile']:.2f}`
- Evidence decay: `{policy['decay']:.2f}`
- Activation threshold: `{policy['activation_threshold']:.2f}`
- Release ratio: `{policy['release_ratio']:.2f}`
- Maximum command hold: `{policy['max_hold_seconds']:.1f} s`
- Re-activation cooldown: `{policy['cooldown_seconds']:.1f} s`

## Final unseen-subject results

| Controller | False activations/rest min | Intent sensitivity | Missed intent | Segment balanced accuracy | Median latency | Wrong direction | Artifact abstention |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-window fusion + artifact gate | {baseline['false_activations_per_minute']:.2f} | {percent(baseline['intent_sensitivity'])} | {percent(baseline['missed_intent_rate'])} | {percent(baseline['segment_balanced_accuracy'])} | {baseline['median_latency_seconds']:.2f} s | {percent(baseline['wrong_direction_rate'])} | {percent(baseline['artifact_abstention_rate'])} |
| Safety evidence without artifact gate | {no_artifact['false_activations_per_minute']:.2f} | {percent(no_artifact['intent_sensitivity'])} | {percent(no_artifact['missed_intent_rate'])} | {percent(no_artifact['segment_balanced_accuracy'])} | {no_artifact['median_latency_seconds']:.2f} s | {percent(no_artifact['wrong_direction_among_detected'])} | 0.0% |
| Safety evidence + 150 uV abstention | {selected['false_activations_per_minute']:.2f} | {percent(selected['intent_sensitivity'])} | {percent(selected['missed_intent_rate'])} | {percent(selected['segment_balanced_accuracy'])} | {selected['median_latency_seconds']:.2f} s | {percent(selected['wrong_direction_among_detected'])} | {percent(selected['artifact_abstention_rate'])} |
| Balanced evidence + 150 uV abstention | {balanced['false_activations_per_minute']:.2f} | {percent(balanced['intent_sensitivity'])} | {percent(balanced['missed_intent_rate'])} | {percent(balanced['segment_balanced_accuracy'])} | {balanced['median_latency_seconds']:.2f} s | {percent(balanced['wrong_direction_among_detected'])} | {percent(balanced['artifact_abstention_rate'])} |
| Sensitivity-target evidence + 150 uV abstention | {responsive['false_activations_per_minute']:.2f} | {percent(responsive['intent_sensitivity'])} | {percent(responsive['missed_intent_rate'])} | {percent(responsive['segment_balanced_accuracy'])} | {responsive['median_latency_seconds']:.2f} s | {percent(responsive['wrong_direction_among_detected'])} | {percent(responsive['artifact_abstention_rate'])} |

The safety result covers **{selected['subjects']} unseen subjects**, **{selected['total_rest_minutes']:.1f} rest minutes**, and **{selected['total_intent_segments']} intent segments**. It emitted **{selected['total_false_activation_events']} false command events** and detected **{selected['total_detected_intent_segments']} intent segments**.

The joint design target was **{targets['false_activations_per_minute']:.1f} or fewer false activations per minute**, **at least {percent(targets['intent_sensitivity'])} sensitivity**, and **no more than {targets['median_latency_seconds']:.1f} seconds median latency**. Joint target met on final subjects: **{'yes' if target_met else 'no'}**.

## Interpretation

This is a stronger pseudo-online test than cue-isolated trial classification because controller state persists through complete runs and false activation is counted as command-onset events per rest minute. The three operating points show whether EEG-only evidence can simultaneously satisfy safety and responsiveness; if the joint target is not met, the result supports adding a separate user-arming or confirmation mechanism rather than hiding the tradeoff behind one threshold. It is still not clinical validation: EEGMMIDB contains healthy participants, visual cues define intended state, and each subject's Run 12 occurs in the same recording protocol as calibration.

## Figures

- `continuous_pipeline_overview.png`: complete signal-to-command pipeline.
- `continuous_controller_frontier.png`: controller-only safety, sensitivity, and latency search.
- `continuous_final_comparison.png`: single-window, no-artifact, and final controller ablation.
- `continuous_final_subject_tradeoff.png`: final patient variability.
- `continuous_example_timeline.png`: chronological state, probability, artifact, evidence, and command trace.
"""
    path.write_text(text, encoding="ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "files/eegmmidb/1.0.0")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "artifacts/dataset/splits.json")
    parser.add_argument("--cache-path", type=Path, default=PROJECT_ROOT / "artifacts/continuous_fusion/continuous_features.npz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/continuous_fusion")
    parser.add_argument("--calibration-segments-per-class", type=int, default=10)
    parser.add_argument("--target-fa-per-minute", type=float, default=1.0)
    parser.add_argument("--target-sensitivity", type=float, default=0.60)
    parser.add_argument("--target-latency", type=float, default=2.5)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    configure_plot_style()
    data_dir = resolve_data_dir(args.data_dir)
    subjects = available_subjects(data_dir, max_subjects=args.max_subjects)
    partitions = load_partitions(args.splits)
    if args.max_subjects is not None:
        represented = set(subjects)
        partitions = SubjectPartitions(
            development=tuple(value for value in partitions.development if value in represented),
            controller_validation=tuple(value for value in partitions.controller_validation if value in represented),
            final_target=tuple(value for value in partitions.final_target if value in represented),
            random_seed=partitions.random_seed,
        )
        partitions.validate()

    print("\n=== Continuous Calibrated ERD/Riemannian Fusion Controller ===")
    print(f"Data: {data_dir}")
    print("Protocol: development fit -> controller policy selection -> locked final Run 12")
    dataset = build_feature_dataset(data_dir, subjects, args.cache_path, args.rebuild_cache)
    represented = set(np.unique(dataset.subjects).astype(int))
    missing_final = sorted(set(partitions.final_target) - represented)
    if missing_final:
        raise RuntimeError(f"Locked final subjects are absent from the continuous cache: {missing_final}")
    decoders = fit_global_decoders(dataset, partitions, args.seed)
    print("[+] Fitting patient adapters and scoring controller/final streams...")
    controller_streams = collect_streams(
        dataset,
        decoders,
        partitions.controller_validation,
        args.calibration_segments_per_class,
        args.seed,
    )
    final_streams = collect_streams(
        dataset,
        decoders,
        partitions.final_target,
        args.calibration_segments_per_class,
        args.seed,
    )
    if len(controller_streams) != len(partitions.controller_validation):
        raise RuntimeError("At least one controller subject lacks complete calibration segments.")
    if len(final_streams) != len(partitions.final_target):
        raise RuntimeError("At least one final subject lacks complete calibration segments.")

    print(f"[+] Searching evidence policies on {len(controller_streams)} controller subjects...")
    selected_policies, frontier = tune_policy(
        controller_streams,
        args.target_fa_per_minute,
        args.target_sensitivity,
        args.target_latency,
    )
    controller_results = {
        name: evaluate_policy(controller_streams, policy, True)
        for name, policy in selected_policies.items()
    }
    final_results = {
        name: evaluate_policy(final_streams, policy, True)
        for name, policy in selected_policies.items()
    }
    safety_policy = selected_policies["safety"]
    final_no_artifact = evaluate_policy(final_streams, safety_policy, False)
    final_single = evaluate_policy(final_streams, single_window_policy(safety_policy), True)
    comparisons = {
        "Single window": final_single,
        "Safety/no artifact": final_no_artifact,
        "Safety": final_results["safety"],
        "Balanced": final_results["balanced"],
        "Sensitivity": final_results["sensitivity_target"],
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {
        "protocol": {
            "task": "continuous rest/left/right unilateral motor-imagery command simulation",
            "development_subjects": list(partitions.development),
            "controller_validation_subjects": list(partitions.controller_validation),
            "final_target_subjects": list(partitions.final_target),
            "calibration_runs": [4, 8],
            "evaluation_run": 12,
            "calibration_segments_per_class": args.calibration_segments_per_class,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "artifact_abstention_uv": ARTIFACT_UV,
            "causal_preprocessing": CAUSAL_PROFILE.fingerprint(),
            "selection_targets": {
                "false_activations_per_minute": args.target_fa_per_minute,
                "intent_sensitivity": args.target_sensitivity,
                "median_latency_seconds": args.target_latency,
            },
        },
        "dataset": {
            "windows": int(len(dataset.labels)),
            "subjects": int(len(np.unique(dataset.subjects))),
            "erd_features": int(dataset.erd_features.shape[1]),
            "riemann_features": int(dataset.riemann_features.shape[1]),
            "channels": dataset.channel_names,
        },
        "selected_policies": {
            name: asdict(policy) for name, policy in selected_policies.items()
        },
        "controller_validation": controller_results,
        "final_target": {
            "single_window_plus_artifact": final_single,
            "evidence_without_artifact": final_no_artifact,
            "safety_evidence_plus_artifact": final_results["safety"],
            "balanced_evidence_plus_artifact": final_results["balanced"],
            "sensitivity_evidence_plus_artifact": final_results["sensitivity_target"],
        },
        "frontier": frontier,
        "runtime_minutes": (time.perf_counter() - started) / 60.0,
    }
    metrics_path = output_dir / "continuous_fusion_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="ascii")
    joblib.dump(
        {
            "decoders": {
                "erd_intent": decoders.erd_intent,
                "riemann_intent": decoders.riemann_intent,
                "erd_direction": decoders.erd_direction,
                "riemann_direction": decoders.riemann_direction,
            },
            "selected_policies": selected_policies,
            "channels": dataset.channel_names,
            "power_bands": POWER_BANDS,
            "covariance_bands": FILTER_BANK,
            "artifact_abstention_uv": ARTIFACT_UV,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
        },
        output_dir / "continuous_fusion_model.joblib",
    )
    plot_overview(output_dir)
    plot_frontier(
        output_dir,
        frontier,
        selected_policies,
        args.target_fa_per_minute,
        args.target_sensitivity,
    )
    plot_final_comparison(output_dir, comparisons)
    plot_subject_tradeoff(output_dir, final_results["balanced"])
    example_subject = sorted(final_streams)[0]
    plot_example_timeline(
        output_dir, final_streams[example_subject], selected_policies["balanced"]
    )
    write_report(output_dir / "CONTINUOUS_FUSION_RESULTS.md", report)

    safety_metrics = final_results["safety"]["mean_patient_metrics"]
    balanced_metrics = final_results["balanced"]["mean_patient_metrics"]
    sensitivity_metrics = final_results["sensitivity_target"]["mean_patient_metrics"]
    print("\n=== Final Locked Continuous-Run Results ===")
    for name, metrics in (
        ("safety", safety_metrics),
        ("balanced", balanced_metrics),
        ("sensitivity", sensitivity_metrics),
    ):
        print(
            f"{name:11s} | FA/min={metrics['false_activations_per_minute']:.2f} | "
            f"sensitivity={percent(metrics['intent_sensitivity'])} | "
            f"missed={percent(metrics['missed_intent_rate'])} | "
            f"latency={metrics['median_latency_seconds']:.2f}s | "
            f"wrong direction/detected={percent(metrics['wrong_direction_among_detected'])}"
        )
    print(f"[+] Saved metrics, model, report, and five figures: {output_dir}")
    print(f"[+] Runtime: {(time.perf_counter() - started) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
