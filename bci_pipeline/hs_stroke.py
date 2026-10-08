"""Locked external-cohort adapter for the corrected HS-Stroke v6 release."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import mne
import numpy as np
from scipy.io import loadmat

from .quality_preprocessing import (
    BranchPreprocessingProfile,
    RecordingQualityError,
    preprocess_native_branches,
)
from .translational_dataset import (
    TranslationalWindowDataset,
    _append_window,
    _empty_rows,
    _record_exclusion,
    _record_qc,
    load_translational_dataset,
    save_translational_dataset,
)


HS_DATASET_ID = "hs_stroke_2026"
HS_NATIVE_SFREQ = 256.0
HS_TRIAL_SECONDS = 17.0
HS_REST_ONSET_SECONDS = 6.5
HS_IMAGERY_ONSET_SECONDS = 9.5
HS_EXPECTED_CHANNELS = (
    "FC3",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP4",
)
HS_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 15))
_PATH_PATTERN = re.compile(
    r"(?P<subject>sub-\d{2})_ses-(?P<session>\d+)_task-motorimagery_"
    r"run-(?P<run>\d+)_eeg\.mat$"
)


def resolve_hs_stroke_root(value: str | Path) -> Path:
    """Resolve a release root or its sourcedata directory."""
    root = Path(value).expanduser().resolve()
    candidates = (root / "sourcedata", root)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("sub-*/ses-*/eeg/*_eeg.mat")):
            return candidate
    raise FileNotFoundError(f"Could not find extracted HS-Stroke sourcedata under {root}")


def discover_hs_source_records(
    root: str | Path,
    subjects: Sequence[str] = HS_SUBJECTS,
) -> dict[tuple[str, int], tuple[Path, ...]]:
    """Return deterministic subject/session groups of raw run MAT files."""
    sourcedata = resolve_hs_stroke_root(root)
    selected = {str(value) for value in subjects}
    grouped: dict[tuple[str, int], list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(sourcedata.glob("sub-*/ses-*/eeg/*_eeg.mat")):
        match = _PATH_PATTERN.match(path.name)
        if match is None or match.group("subject") not in selected:
            continue
        grouped[(match.group("subject"), int(match.group("session")))].append(
            (int(match.group("run")), path)
        )
    missing = sorted(selected.difference(subject for subject, _ in grouped))
    if missing:
        raise FileNotFoundError(f"HS-Stroke subjects have no source records: {missing}")
    return {
        key: tuple(path for _, path in sorted(values))
        for key, values in sorted(grouped.items())
    }


def _participants(root: Path) -> dict[str, dict[str, str]]:
    release_root = root.parent if root.name == "sourcedata" else root
    path = release_root / "participants.tsv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            row["participant_id"]: row
            for row in csv.DictReader(stream, delimiter="\t")
        }


def _record_number(path: Path) -> int:
    match = _PATH_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected HS-Stroke filename: {path.name}")
    return int(match.group("run"))


def _load_session_trials(
    paths: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    run_ids: list[np.ndarray] = []
    local_ids: list[np.ndarray] = []
    expected_samples = int(round(HS_NATIVE_SFREQ * HS_TRIAL_SECONDS))
    for path in paths:
        mat = loadmat(
            path,
            variable_names=["EEGdata", "EEGdatalabel"],
            verify_compressed_data_integrity=False,
        )
        data = np.asarray(mat["EEGdata"], dtype=np.float64)
        local_labels = np.asarray(mat["EEGdatalabel"]).reshape(-1).astype(np.int64)
        if data.ndim != 3 or data.shape[:2] != (
            len(HS_EXPECTED_CHANNELS),
            expected_samples,
        ):
            raise ValueError(f"Unexpected EEGdata shape {data.shape} in {path}")
        if data.shape[2] != len(local_labels):
            raise ValueError(f"Trial/label mismatch in {path}")
        if not set(np.unique(local_labels)).issubset({1, 2}):
            raise ValueError(f"Unexpected HS-Stroke labels in {path}")
        trials.append(data)
        labels.append(local_labels)
        run_ids.append(np.full(len(local_labels), _record_number(path), dtype=np.int16))
        local_ids.append(np.arange(len(local_labels), dtype=np.int16))
    return (
        np.concatenate(trials, axis=2),
        np.concatenate(labels),
        np.concatenate(run_ids),
        np.concatenate(local_ids),
    )


def _session_raw(trials: np.ndarray) -> mne.io.RawArray:
    channels, samples, count = trials.shape
    continuous = np.transpose(trials, (0, 2, 1)).reshape(channels, count * samples)
    info = mne.create_info(
        ch_names=list(HS_EXPECTED_CHANNELS),
        sfreq=HS_NATIVE_SFREQ,
        ch_types=["eeg"] * channels,
    )
    return mne.io.RawArray(continuous, info, verbose=False)


def _fingerprint(
    sourcedata: Path,
    groups: Mapping[tuple[str, int], Sequence[Path]],
    profile: BranchPreprocessingProfile,
) -> str:
    payload = {
        "schema_version": 1,
        "dataset": HS_DATASET_ID,
        "release": "30747806_v6",
        "subjects": list(HS_SUBJECTS),
        "profile": profile.fingerprint(),
        "rest_window_seconds": [HS_REST_ONSET_SECONDS, HS_REST_ONSET_SECONDS + 2.0],
        "imagery_window_seconds": [
            HS_IMAGERY_ONSET_SECONDS,
            HS_IMAGERY_ONSET_SECONDS + 2.0,
        ],
        "records": [
            {
                "path": str(path.relative_to(sourcedata)),
                "bytes": path.stat().st_size,
            }
            for paths in groups.values()
            for path in paths
        ],
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("ascii")).hexdigest()


def build_hs_locked_window_dataset(
    *,
    root: str | Path,
    cache_path: Path,
    profile: BranchPreprocessingProfile,
    rebuild_cache: bool = False,
) -> TranslationalWindowDataset:
    """Build the predeclared rest/imagery windows for all locked patients."""
    sourcedata = resolve_hs_stroke_root(root)
    groups = discover_hs_source_records(sourcedata)
    fingerprint = _fingerprint(sourcedata, groups, profile)
    if not rebuild_cache:
        cached = load_translational_dataset(cache_path, fingerprint)
        if cached is not None:
            print(f"[+] Loaded {len(cached.labels):,} locked HS-Stroke windows from {cache_path}")
            return cached

    participant_rows = _participants(sourcedata)
    rows = _empty_rows()
    trial_samples = int(round(HS_NATIVE_SFREQ * HS_TRIAL_SECONDS))
    trial_seconds_after_resampling = trial_samples / HS_NATIVE_SFREQ
    for group_index, ((subject, session), paths) in enumerate(groups.items(), start=1):
        print(
            f"[+] HS locked adapter {group_index:03d}/{len(groups):03d}: "
            f"{subject} session {session:02d} ({len(paths)} runs)"
        )
        subject_key = f"{HS_DATASET_ID}:{subject}"
        paralysis_side = participant_rows.get(subject, {}).get(
            "paralysis_side",
            "unknown",
        ).lower()
        try:
            trials, labels, run_ids, local_ids = _load_session_trials(paths)
            raw = _session_raw(trials)
            branches = preprocess_native_branches(raw, profile)
        except (RecordingQualityError, RuntimeError, ValueError) as error:
            _record_exclusion(
                rows,
                error,
                dataset=HS_DATASET_ID,
                subject=subject_key,
                session=session,
                run="all",
                paths=[str(path) for path in paths],
            )
            continue
        _record_qc(
            rows,
            branches,
            dataset=HS_DATASET_ID,
            subject=subject_key,
            session=session,
            run="all",
            paths=[str(path) for path in paths],
            trials=int(len(labels)),
        )
        for trial_offset, (label, run_id, local_id) in enumerate(
            zip(labels, run_ids, local_ids)
        ):
            base = trial_offset * trial_seconds_after_resampling
            trial_index = session * 1_000_000 + int(run_id) * 1_000 + int(local_id)
            _append_window(
                rows,
                branches,
                onset_seconds=base + HS_REST_ONSET_SECONDS,
                label=0,
                dataset_id=HS_DATASET_ID,
                subject_key=subject_key,
                session=session,
                recording=int(run_id),
                trial_index=trial_index,
                negative_kind="preparation_rest",
                cue_locked=True,
                paralysis_side=paralysis_side,
            )
            _append_window(
                rows,
                branches,
                onset_seconds=base + HS_IMAGERY_ONSET_SECONDS,
                label=int(label),
                dataset_id=HS_DATASET_ID,
                subject_key=subject_key,
                session=session,
                recording=int(run_id),
                trial_index=trial_index,
                negative_kind="not_applicable",
                cue_locked=True,
                paralysis_side=paralysis_side,
            )
        del raw, branches, trials

    if not rows.labels or rows.channel_names is None or rows.quality_names is None:
        raise RuntimeError("No valid locked HS-Stroke windows were extracted.")
    dataset = TranslationalWindowDataset(
        mrcp_epochs=np.stack(rows.mrcp),
        smr_epochs=np.stack(rows.smr),
        quality_features=np.stack(rows.quality),
        labels=np.asarray(rows.labels, dtype=np.int64),
        dataset_ids=np.asarray(rows.dataset_ids, dtype="<U32"),
        subject_keys=np.asarray(rows.subject_keys, dtype="<U64"),
        sessions=np.asarray(rows.sessions, dtype=np.int16),
        recordings=np.asarray(rows.recordings, dtype=np.int16),
        trial_indices=np.asarray(rows.trial_indices, dtype=np.int64),
        negative_kinds=np.asarray(rows.negative_kinds, dtype="<U32"),
        cue_locked=np.asarray(rows.cue_locked, dtype=bool),
        spatial_methods=np.asarray(rows.spatial_methods, dtype="<U8"),
        window_onsets_seconds=np.asarray(rows.onsets, dtype=np.float64),
        paralysis_sides=np.asarray(rows.paralysis_sides, dtype="<U24"),
        nihss=np.asarray(rows.nihss, dtype=np.float32),
        channel_names=rows.channel_names,
        quality_feature_names=rows.quality_names,
        sfreq=profile.sfreq,
        fingerprint=fingerprint,
        recording_qc=tuple(rows.recording_qc),
    )
    dataset.validate()
    if set(np.unique(dataset.subject_keys)) != {
        f"{HS_DATASET_ID}:{subject}" for subject in HS_SUBJECTS
    }:
        raise RuntimeError("At least one locked HS-Stroke subject has no valid windows.")
    save_translational_dataset(cache_path, dataset)
    print(f"[+] Saved {len(dataset.labels):,} locked HS-Stroke windows to {cache_path}")
    return dataset
