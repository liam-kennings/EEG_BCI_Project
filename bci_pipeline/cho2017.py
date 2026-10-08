"""Cho2017 source-data adapter for locked external V7 evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

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
    _merge_rows,
    _record_exclusion,
    _record_qc,
    load_translational_dataset,
    save_translational_dataset,
)


CHO_DATASET_ID = "cho2017"
CHO_RELEASE = "nm000245_v1.0.1"
CHO_INTENT_OFFSET_SECONDS = 0.5
CHO_REST_START_SECONDS = 2.0
CHO_REST_STEP_SECONDS = 1.0
CHO_CALIBRATION_PER_CLASS = 10
CHO_EXPECTED_SUBJECTS = tuple(range(1, 53))
CHO_ADAPTER_DEVELOPMENT_SUBJECTS = (1,)
CHO_LOCKED_SUBJECTS = tuple(range(2, 53))
CHO_EEG_CHANNELS = (
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3",
    "FC1", "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1",
    "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz",
    "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4",
    "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4",
    "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8",
    "P10", "PO8", "PO4", "O2",
)
CHO_EMG_CHANNELS = ("EMG1", "EMG2", "EMG3", "EMG4")


@dataclass(frozen=True)
class ChoRecording:
    """One participant's BIDS EEG recording and event table."""

    subject: int
    bdf_path: Path
    events_path: Path


@dataclass(frozen=True)
class ChoSourceRecord:
    """One participant's original MATLAB source file."""

    subject: int
    mat_path: Path


@dataclass(frozen=True)
class ChoEvent:
    """A left- or right-hand imagery event from the BIDS table."""

    onset: float
    duration: float
    label: int
    trial_type: str


def resolve_cho2017_root(value: str | Path) -> Path:
    """Resolve the extracted NEMAR BIDS root."""
    root = Path(value).expanduser().resolve()
    candidates = (root, root / "nm000245")
    for candidate in candidates:
        if (
            (candidate / "dataset_description.json").is_file()
            and (candidate / "participants.tsv").is_file()
            and any(candidate.glob("sub-*/ses-0/eeg/*_eeg.bdf"))
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find the extracted Cho2017 BIDS dataset under {root}."
    )


def discover_cho2017_recordings(
    root: str | Path,
    subjects: Sequence[int] | None = None,
) -> tuple[ChoRecording, ...]:
    """Discover one complete BIDS imagery recording per selected subject."""
    dataset_root = resolve_cho2017_root(root)
    selected = set(
        CHO_EXPECTED_SUBJECTS if subjects is None else (int(value) for value in subjects)
    )
    recordings = []
    for subject in sorted(selected):
        eeg_dir = dataset_root / f"sub-{subject}" / "ses-0" / "eeg"
        bdf_paths = sorted(eeg_dir.glob("*_task-imagery*_eeg.bdf"))
        event_paths = sorted(eeg_dir.glob("*_task-imagery*_events.tsv"))
        if len(bdf_paths) != 1 or len(event_paths) != 1:
            raise FileNotFoundError(
                f"Expected one BDF and one events table for Cho2017 subject {subject}; "
                f"found {len(bdf_paths)} and {len(event_paths)}."
            )
        recordings.append(ChoRecording(subject, bdf_paths[0], event_paths[0]))
    if not recordings:
        raise FileNotFoundError("No Cho2017 BIDS recordings were discovered.")
    return tuple(recordings)


def discover_cho2017_source_records(
    root: str | Path,
    subjects: Sequence[int] | None = None,
) -> tuple[ChoSourceRecord, ...]:
    """Discover the original MAT files containing imagery and continuous rest."""
    dataset_root = resolve_cho2017_root(root)
    selected = set(
        CHO_EXPECTED_SUBJECTS if subjects is None else (int(value) for value in subjects)
    )
    records = []
    for subject in sorted(selected):
        path = dataset_root / "sourcedata" / f"s{subject:02d}.mat"
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(ChoSourceRecord(subject, path))
    if not records:
        raise FileNotFoundError("No Cho2017 source MAT files were discovered.")
    return tuple(records)


def load_cho2017_events(path: str | Path) -> tuple[ChoEvent, ...]:
    """Load and validate left/right imagery events from a BIDS TSV."""
    label_ids = {"left_hand": 1, "right_hand": 2}
    events = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            trial_type = str(row.get("trial_type", "")).strip().lower()
            if trial_type not in label_ids:
                continue
            events.append(
                ChoEvent(
                    onset=float(row["onset"]),
                    duration=float(row["duration"]),
                    label=label_ids[trial_type],
                    trial_type=trial_type,
                )
            )
    if not events:
        raise ValueError(f"No left/right imagery events found in {path}.")
    onsets = np.asarray([event.onset for event in events])
    if np.any(np.diff(onsets) <= 0):
        raise ValueError(f"Cho2017 events are not strictly chronological in {path}.")
    if set(event.label for event in events) != {1, 2}:
        raise ValueError(f"Cho2017 event table is missing an imagery class: {path}.")
    if any(abs(event.duration - 3.0) > 1e-3 for event in events):
        raise ValueError(f"Cho2017 contains an unexpected imagery duration in {path}.")
    return tuple(events)


def calibration_event_indices(
    events: Sequence[ChoEvent],
    budget_per_class: int = CHO_CALIBRATION_PER_CLASS,
) -> tuple[set[int], set[int]]:
    """Return frozen intent and independent-rest indices for calibration."""
    if budget_per_class <= 0:
        raise ValueError("Calibration budget must be positive.")
    intent_indices: set[int] = set()
    for label in (1, 2):
        candidates = [
            index for index, event in enumerate(events) if event.label == label
        ]
        if len(candidates) < budget_per_class:
            raise ValueError(f"Cho2017 has fewer than {budget_per_class} class-{label} trials.")
        intent_indices.update(candidates[:budget_per_class])
    return intent_indices, set(range(budget_per_class))


def _event_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cho2017_snapshot(
    root: str | Path,
    subjects: Sequence[int] | None = None,
) -> dict[str, object]:
    """Return a deterministic, inference-free dataset inventory."""
    dataset_root = resolve_cho2017_root(root)
    recordings = discover_cho2017_recordings(dataset_root, subjects)
    source_records = {
        record.subject: record
        for record in discover_cho2017_source_records(dataset_root, subjects)
    }
    provenance_path = dataset_root / "sourcedata" / "sourcedata_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_by_name = {
        str(record["file"]): record for record in provenance["files"]
    }
    records = []
    event_totals = {"left_hand": 0, "right_hand": 0}
    for recording in recordings:
        events = load_cho2017_events(recording.events_path)
        source = source_records[recording.subject]
        source_manifest = provenance_by_name.get(source.mat_path.name)
        if source_manifest is None:
            raise ValueError(f"Missing source provenance for {source.mat_path.name}.")
        if source.mat_path.stat().st_size != int(source_manifest["bytes"]):
            raise ValueError(f"Source MAT size does not match provenance: {source.mat_path}")
        for event in events:
            event_totals[event.trial_type] += 1
        records.append(
            {
                "subject": recording.subject,
                "bdf": str(recording.bdf_path.relative_to(dataset_root)),
                "bdf_bytes": recording.bdf_path.stat().st_size,
                "events": str(recording.events_path.relative_to(dataset_root)),
                "events_bytes": recording.events_path.stat().st_size,
                "events_sha256": _event_sha256(recording.events_path),
                "event_count": len(events),
                "source_mat": str(source.mat_path.relative_to(dataset_root)),
                "source_mat_bytes": source.mat_path.stat().st_size,
                "source_mat_sha256": str(source_manifest["sha256"]),
            }
        )
    payload = {
        "dataset": CHO_DATASET_ID,
        "release": CHO_RELEASE,
        "subjects": len(recordings),
        "event_totals": event_totals,
        "bdf_bytes": int(sum(record["bdf_bytes"] for record in records)),
        "source_mat_bytes": int(
            sum(record["source_mat_bytes"] for record in records)
        ),
        "source_provenance_sha256": _event_sha256(provenance_path),
        "records": records,
    }
    payload["fingerprint"] = sha256(
        json.dumps(payload, sort_keys=True).encode("ascii")
    ).hexdigest()
    return payload


def _cache_fingerprint(
    snapshot: dict[str, object],
    profile: BranchPreprocessingProfile,
) -> str:
    payload = {
        "schema_version": 2,
        "snapshot_fingerprint": snapshot["fingerprint"],
        "profile": profile.fingerprint(),
        "intent_window_seconds": [
            CHO_INTENT_OFFSET_SECONDS,
            CHO_INTENT_OFFSET_SECONDS + 2.0,
        ],
        "rest_source": "independent_continuous_rest_recording",
        "rest_start_seconds": CHO_REST_START_SECONDS,
        "rest_step_seconds": CHO_REST_STEP_SECONDS,
        "calibration_per_class": CHO_CALIBRATION_PER_CLASS,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("ascii")).hexdigest()


def _source_raw(values: np.ndarray, sfreq: float) -> mne.io.RawArray:
    """Create an EEG-only RawArray from microvolt-valued source data."""
    signal = np.asarray(values[: len(CHO_EEG_CHANNELS)], dtype=np.float64)
    if signal.shape[0] != len(CHO_EEG_CHANNELS):
        raise ValueError(f"Unexpected Cho2017 source shape: {values.shape}")
    signal = (signal - np.mean(signal, axis=1, keepdims=True)) * 1e-6
    info = mne.create_info(
        ch_names=list(CHO_EEG_CHANNELS),
        sfreq=float(sfreq),
        ch_types=["eeg"] * len(CHO_EEG_CHANNELS),
    )
    raw = mne.io.RawArray(signal, info, verbose=False)
    raw.set_montage("standard_1005", on_missing="raise")
    return raw


def _source_event_onsets(event_vector: np.ndarray, sfreq: float) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(event_vector).reshape(-1) > 0)
    if not len(indices):
        raise ValueError("Cho2017 source file contains no imagery event markers.")
    return indices.astype(np.float64) / float(sfreq)


def _bad_trial_indices(source: dict[str, object], label: int) -> set[int]:
    """Return the union of the source file's one-based bad-trial lists."""
    bad = source.get("bad_trial_indices", {})
    if not isinstance(bad, dict):
        return set()
    result: set[int] = set()
    for key in ("bad_trial_idx_voltage", "bad_trial_idx_mi"):
        sides = bad.get(key)
        if sides is None:
            continue
        values = np.asarray(sides, dtype=object).reshape(-1)
        if len(values) < label:
            continue
        local = np.asarray(values[label - 1]).reshape(-1)
        for value in local:
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                continue
            if numeric >= 1:
                result.add(numeric - 1)
    return result


def build_cho2017_window_dataset(
    *,
    root: str | Path,
    cache_path: Path,
    profile: BranchPreprocessingProfile,
    subjects: Sequence[int] | None = None,
    rebuild_cache: bool = False,
) -> TranslationalWindowDataset:
    """Build imagery and independent continuous-rest windows for Cho2017."""
    dataset_root = resolve_cho2017_root(root)
    records = discover_cho2017_source_records(dataset_root, subjects)
    snapshot = cho2017_snapshot(dataset_root, [value.subject for value in records])
    fingerprint = _cache_fingerprint(snapshot, profile)
    if not rebuild_cache:
        cached = load_translational_dataset(cache_path, fingerprint)
        if cached is not None:
            print(f"[+] Loaded {len(cached.labels):,} Cho2017 windows from {cache_path}")
            return cached

    rows = _empty_rows()
    for position, record in enumerate(records, start=1):
        subject_key = f"{CHO_DATASET_ID}:S{record.subject:03d}"
        subject_rows = _empty_rows()
        print(
            f"[+] Cho2017 adapter {position:02d}/{len(records):02d}: "
            f"{subject_key}"
        )
        try:
            source = loadmat(
                record.mat_path,
                simplify_cells=True,
                variable_names=["eeg"],
                verify_compressed_data_integrity=False,
            )["eeg"]
            sfreq = float(source["srate"])
            event_onsets = _source_event_onsets(source["imagery_event"], sfreq)
            for label, field in ((1, "imagery_left"), (2, "imagery_right")):
                raw = _source_raw(source[field], sfreq)
                branches = preprocess_native_branches(raw, profile)
                _record_qc(
                    subject_rows,
                    branches,
                    dataset=CHO_DATASET_ID,
                    subject=subject_key,
                    session=0,
                    run=f"imagery_class_{label}",
                    path=str(record.mat_path),
                    source=field,
                )
                bad_trials = _bad_trial_indices(source, label)
                calibration_kept = 0
                for trial_index, onset in enumerate(event_onsets):
                    if trial_index in bad_trials:
                        continue
                    is_calibration = calibration_kept < CHO_CALIBRATION_PER_CLASS
                    kept = _append_window(
                        subject_rows,
                        branches,
                        onset_seconds=float(onset) + CHO_INTENT_OFFSET_SECONDS,
                        label=label,
                        dataset_id=CHO_DATASET_ID,
                        subject_key=subject_key,
                        session=0,
                        recording=0 if is_calibration else 1,
                        trial_index=label * 1_000_000 + trial_index,
                        negative_kind="not_applicable",
                        cue_locked=True,
                    )
                    if kept and is_calibration:
                        calibration_kept += 1
                if calibration_kept != CHO_CALIBRATION_PER_CLASS:
                    raise RuntimeError(
                        f"{subject_key} class {label} has only "
                        f"{calibration_kept} QC-passing calibration windows."
                    )
                del raw, branches

            rest_raw = _source_raw(source["rest"], sfreq)
            rest_branches = preprocess_native_branches(rest_raw, profile)
            _record_qc(
                subject_rows,
                rest_branches,
                dataset=CHO_DATASET_ID,
                subject=subject_key,
                session=0,
                run="continuous_rest",
                path=str(record.mat_path),
                source="rest",
            )
            rest_duration = float(rest_raw.times[-1])
            rest_onsets = np.arange(
                CHO_REST_START_SECONDS,
                rest_duration - 2.0 + 1e-9,
                CHO_REST_STEP_SECONDS,
            )
            rest_calibration_kept = 0
            for rest_index, onset in enumerate(rest_onsets):
                is_calibration = (
                    rest_calibration_kept < CHO_CALIBRATION_PER_CLASS
                )
                kept = _append_window(
                    subject_rows,
                    rest_branches,
                    onset_seconds=float(onset),
                    label=0,
                    dataset_id=CHO_DATASET_ID,
                    subject_key=subject_key,
                    session=0,
                    recording=0 if is_calibration else 1,
                    trial_index=rest_index,
                    negative_kind="continuous_no_control",
                    cue_locked=False,
                )
                if kept and is_calibration:
                    rest_calibration_kept += 1
            if rest_calibration_kept != CHO_CALIBRATION_PER_CLASS:
                raise RuntimeError(
                    f"{subject_key} has only {rest_calibration_kept} "
                    "QC-passing rest calibration windows."
                )
            del rest_raw, rest_branches

            local_labels = set(subject_rows.labels)
            local_recordings = set(subject_rows.recordings)
            calibration_labels = {
                label
                for label, recording_id in zip(
                    subject_rows.labels,
                    subject_rows.recordings,
                )
                if recording_id == 0
            }
            if (
                local_labels != {0, 1, 2}
                or local_recordings != {0, 1}
                or calibration_labels != {0, 1, 2}
            ):
                raise RuntimeError(
                    f"{subject_key} is incomplete after window QC: "
                    f"labels={sorted(local_labels)}, "
                    f"recordings={sorted(local_recordings)}."
                )
        except (OSError, RuntimeError, ValueError, RecordingQualityError) as error:
            _record_exclusion(
                rows,
                error,
                dataset=CHO_DATASET_ID,
                subject=subject_key,
                session=0,
                run=0,
                path=str(record.mat_path),
                exclusion_scope="subject",
            )
            print(f"    [!] Excluded {subject_key}: {error}")
            continue
        _merge_rows(rows, subject_rows)
        del source

    if not rows.labels or rows.channel_names is None or rows.quality_names is None:
        raise RuntimeError("No valid Cho2017 windows were extracted.")
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
    save_translational_dataset(cache_path, dataset)
    print(f"[+] Saved {len(dataset.labels):,} Cho2017 windows to {cache_path}")
    return dataset
