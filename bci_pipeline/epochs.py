"""Shared one-cue-per-trial raw epoch cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import mne
import numpy as np

from .dataset import IMAGERY_RUNS, find_run_file, map_task_annotation
from .preprocessing import OFFLINE_PROFILE, PreprocessingProfile, is_valid_epoch, preprocess_run
from .representations.riemann import TEMPORAL_CHANNELS


@dataclass(frozen=True)
class EpochConfig:
    baseline_seconds: float = 1.0
    epoch_start_seconds: float = 0.5
    epoch_duration_seconds: float = 3.0
    artifact_peak_to_peak_uv: float = 250.0
    channels: tuple[str, ...] = TEMPORAL_CHANNELS
    runs: tuple[int, ...] = IMAGERY_RUNS
    preprocessing: PreprocessingProfile = OFFLINE_PROFILE


@dataclass
class RawEpochDataset:
    epochs: np.ndarray
    baselines: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    trial_indices: np.ndarray
    channel_names: List[str]
    sfreq: float


def epoch_fingerprint(config: EpochConfig, subjects: Sequence[int]) -> str:
    payload = {
        "schema_version": 1,
        "subjects": list(subjects),
        "runs": list(config.runs),
        "channels": list(config.channels),
        "baseline_seconds": config.baseline_seconds,
        "epoch_start_seconds": config.epoch_start_seconds,
        "epoch_duration_seconds": config.epoch_duration_seconds,
        "artifact_peak_to_peak_uv": config.artifact_peak_to_peak_uv,
        "preprocessing": config.preprocessing.fingerprint(),
    }
    return json.dumps(payload, sort_keys=True)


def load_epoch_cache(path: Path, fingerprint: str) -> Optional[RawEpochDataset]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["fingerprint"].item()) != fingerprint:
            return None
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


def save_epoch_cache(path: Path, fingerprint: str, dataset: RawEpochDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fingerprint=np.asarray(fingerprint),
        epochs=dataset.epochs,
        baselines=dataset.baselines,
        labels=dataset.labels,
        subjects=dataset.subjects,
        runs=dataset.runs,
        trial_indices=dataset.trial_indices,
        channel_names=np.asarray(dataset.channel_names),
        sfreq=np.asarray(dataset.sfreq),
    )


def build_raw_epoch_dataset(
    data_dir: Path,
    subjects: Sequence[int],
    cache_path: Path,
    config: EpochConfig = EpochConfig(),
    rebuild_cache: bool = False,
) -> RawEpochDataset:
    fingerprint = epoch_fingerprint(config, subjects)
    if not rebuild_cache:
        cached = load_epoch_cache(cache_path, fingerprint)
        if cached is not None:
            print(f"[+] Loaded {len(cached.labels)} raw trials from {cache_path}")
            return cached

    epoch_rows: List[np.ndarray] = []
    baseline_rows: List[np.ndarray] = []
    labels: List[int] = []
    groups: List[int] = []
    runs: List[int] = []
    trial_indices: List[int] = []
    channel_names: Optional[List[str]] = None
    rejected = 0

    print(f"[+] Extracting raw trials from {len(subjects)} subjects...")
    for position, subject in enumerate(subjects, start=1):
        subject_trial = 0
        for run_id in config.runs:
            file_path = find_run_file(data_dir, subject, run_id)
            if file_path is None:
                raise FileNotFoundError(f"Missing S{subject:03d}R{run_id:02d}.edf")
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            processed = preprocess_run(raw, config.preprocessing, config.channels)
            if channel_names is None:
                channel_names = list(processed.ch_names)
            signal = processed.get_data()
            sfreq = float(processed.info["sfreq"])
            baseline_samples = int(round(config.baseline_seconds * sfreq))
            epoch_offset = int(round(config.epoch_start_seconds * sfreq))
            epoch_samples = int(round(config.epoch_duration_seconds * sfreq))

            for annotation in processed.annotations:
                label = map_task_annotation(str(annotation["description"]))
                if label is None:
                    continue
                onset = int(round(float(annotation["onset"]) * sfreq))
                baseline_start = onset - baseline_samples
                epoch_start = onset + epoch_offset
                epoch_stop = epoch_start + epoch_samples
                if baseline_start < 0 or epoch_stop > signal.shape[1]:
                    continue
                baseline = signal[:, baseline_start:onset].copy()
                epoch = signal[:, epoch_start:epoch_stop].copy()
                if not is_valid_epoch(
                    baseline,
                    epoch,
                    peak_to_peak_uv=config.artifact_peak_to_peak_uv,
                ):
                    rejected += 1
                    continue
                # Remove each trial's pre-cue DC level without leaking task data.
                epoch -= np.mean(baseline, axis=1, keepdims=True)
                baseline -= np.mean(baseline, axis=1, keepdims=True)
                epoch_rows.append(epoch.astype(np.float32))
                baseline_rows.append(baseline.astype(np.float32))
                labels.append(label)
                groups.append(subject)
                runs.append(run_id)
                trial_indices.append(subject_trial)
                subject_trial += 1

        if position % 10 == 0 or position == len(subjects):
            print(f"    processed {position:3d}/{len(subjects)} subjects; kept {len(labels)} trials")

    if not epoch_rows or channel_names is None:
        raise RuntimeError("No valid raw epochs were extracted.")
    dataset = RawEpochDataset(
        epochs=np.stack(epoch_rows),
        baselines=np.stack(baseline_rows),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(groups, dtype=np.int64),
        runs=np.asarray(runs, dtype=np.int64),
        trial_indices=np.asarray(trial_indices, dtype=np.int64),
        channel_names=channel_names,
        sfreq=config.preprocessing.sfreq,
    )
    print(f"[+] Kept {len(labels)} trials; rejected {rejected}")
    save_epoch_cache(cache_path, fingerprint, dataset)
    return dataset
