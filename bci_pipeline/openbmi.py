"""OpenBMI/NEMAR BIDS adapter for motor-imagery experiments."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np

from . import PROJECT_ROOT
from .epochs import RawEpochDataset, load_epoch_cache, save_epoch_cache
from .preprocessing import OFFLINE_PROFILE, PreprocessingProfile, is_valid_epoch, preprocess_run
from .representations.riemann import TEMPORAL_CHANNELS


DATASET_KEY = "openbmi"
EXPECTED_SUBJECTS = 54
EXPECTED_SESSIONS = (1, 2)
EXPECTED_RECORDINGS_PER_SUBJECT = 4
EXPECTED_EVENTS_PER_RECORDING = 100
EXPECTED_EEG_CHANNELS = 62
EXPECTED_EMG_CHANNELS = 4
NATIVE_SAMPLING_HZ = 1000.0
LABEL_IDS = {"left_hand": 1, "right_hand": 2}
OPENBMI_PAPER_MOTOR_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
)
OPENBMI_OCCIPITAL_CHANNELS = ("P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2")
RECORDING_PATTERN = re.compile(
    r"sub-(?P<subject>\d+)_ses-(?P<session>[12])_task-imagery_"
    r"run-(?P<run>[14])_recording-(?P<recording>train|test)_eeg\.bdf$"
)


@dataclass(frozen=True)
class OpenBMIRecording:
    subject: int
    session: int
    recording: str
    native_run: int
    pipeline_run: int
    bdf_path: Path
    events_path: Path
    channels_path: Path


@dataclass(frozen=True)
class OpenBMITrial:
    dataset: str
    subject: str
    session: str
    run: str
    recording: str
    trial_index: int
    cue_onset_s: float
    intent_start_s: float
    intent_end_s: float
    rest_start_s: float
    rest_end_s: float
    native_label: str
    native_value: str
    canonical_label: str
    timing_source: str
    recording_path: str
    quality_flags: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OpenBMIRecordingSummary:
    subject: str
    session: str
    recording: str
    path: str
    signal_available: bool
    sampling_hz: float
    channels_total: int
    eeg_channels: int
    emg_channels: int
    trials: int
    left_trials: int
    right_trials: int
    quality_flags: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OpenBMIEpochConfig:
    baseline_seconds: float = 1.0
    epoch_start_seconds: float = 0.5
    epoch_duration_seconds: float = 3.0
    rest_start_seconds: float = 5.0
    artifact_peak_to_peak_uv: float = 250.0
    channels: tuple[str, ...] = TEMPORAL_CHANNELS
    preprocessing: PreprocessingProfile = OFFLINE_PROFILE


def resolve_openbmi_root(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    candidates = (
        candidate,
        PROJECT_ROOT / candidate,
        candidate / "nm000338",
        PROJECT_ROOT / candidate / "nm000338",
    )
    for root in candidates:
        root = root.resolve()
        if (
            (root / "dataset_description.json").is_file()
            and (root / "participants.tsv").is_file()
            and any(root.glob("sub-*"))
        ):
            description = json.loads(
                (root / "dataset_description.json").read_text(encoding="utf-8")
            )
            if str(description.get("DatasetDOI", "")).endswith("nm000338"):
                return root
    raise FileNotFoundError(
        "Could not find the OpenBMI nm000338 BIDS root containing "
        "dataset_description.json, participants.tsv, and sub-* folders."
    )


def metadata_subjects(root: str | Path) -> list[int]:
    dataset_root = resolve_openbmi_root(root)
    return sorted(
        int(path.name.removeprefix("sub-"))
        for path in dataset_root.glob("sub-*")
        if path.is_dir() and path.name.removeprefix("sub-").isdigit()
    )


def _pipeline_run(session: int, recording: str) -> int:
    # Preserve the existing calibration/evaluation convention used downstream.
    if session == 1:
        return 4 if recording == "train" else 8
    return 12


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.name.replace("_eeg.bdf", suffix))


def discover_openbmi_recordings(
    root: str | Path,
    *,
    include_missing: bool = False,
) -> list[OpenBMIRecording]:
    dataset_root = resolve_openbmi_root(root)
    records: list[OpenBMIRecording] = []
    for subject in metadata_subjects(dataset_root):
        for session in EXPECTED_SESSIONS:
            for recording, native_run in (("train", 1), ("test", 4)):
                stem = (
                    f"sub-{subject}_ses-{session}_task-imagery_run-{native_run}_"
                    f"recording-{recording}_eeg.bdf"
                )
                bdf = dataset_root / f"sub-{subject}" / f"ses-{session}" / "eeg" / stem
                events = _sidecar(bdf, "_events.tsv")
                channels = _sidecar(bdf, "_channels.tsv")
                if include_missing or bdf.is_file():
                    records.append(
                        OpenBMIRecording(
                            subject=subject,
                            session=session,
                            recording=recording,
                            native_run=native_run,
                            pipeline_run=_pipeline_run(session, recording),
                            bdf_path=bdf,
                            events_path=events,
                            channels_path=channels,
                        )
                    )
    return records


def available_openbmi_subjects(
    root: str | Path,
    *,
    complete_only: bool = True,
) -> list[int]:
    records = discover_openbmi_recordings(root)
    counts: dict[int, int] = {}
    for record in records:
        counts[record.subject] = counts.get(record.subject, 0) + 1
    if complete_only:
        return sorted(
            subject
            for subject, count in counts.items()
            if count == EXPECTED_RECORDINGS_PER_SUBJECT
        )
    return sorted(counts)


def load_openbmi_events(recording: OpenBMIRecording) -> list[dict[str, object]]:
    if not recording.events_path.is_file():
        raise FileNotFoundError(f"Missing event sidecar: {recording.events_path}")
    with recording.events_path.open(newline="", encoding="utf-8-sig") as handle:
        native_rows = list(csv.DictReader(handle, delimiter="\t"))
    rows: list[dict[str, object]] = []
    for index, row in enumerate(native_rows):
        trial_type = str(row["trial_type"]).strip().lower()
        if trial_type not in LABEL_IDS:
            raise ValueError(
                f"Unsupported OpenBMI trial_type '{trial_type}' in "
                f"{recording.events_path}"
            )
        rows.append(
            {
                "trial_index": index,
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "trial_type": trial_type,
                "value": str(row.get("value", "")),
                "sample": int(row["sample"]),
            }
        )
    if len(rows) != EXPECTED_EVENTS_PER_RECORDING:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS_PER_RECORDING} events in "
            f"{recording.events_path}, found {len(rows)}."
        )
    return rows


def _channel_counts(path: Path) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing channel sidecar: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    types = [str(row["type"]).strip().upper() for row in rows]
    return len(rows), types.count("EEG"), types.count("EMG")


def read_openbmi_raw(
    recording: OpenBMIRecording,
    *,
    preload: bool = False,
) -> mne.io.BaseRaw:
    if not recording.bdf_path.is_file():
        raise FileNotFoundError(f"Missing OpenBMI BDF: {recording.bdf_path}")
    raw = mne.io.read_raw_bdf(recording.bdf_path, preload=preload, verbose=False)
    return mark_openbmi_channel_types(raw)


def mark_openbmi_channel_types(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Mark the four auxiliary muscle channels excluded from EEG processing."""
    adjusted = raw.copy()
    emg_types = {
        name: "emg"
        for name in adjusted.ch_names
        if name.upper().startswith("EMG")
    }
    if emg_types:
        adjusted.set_channel_types(emg_types, verbose=False)
    return adjusted


def build_recording_trials(recording: OpenBMIRecording) -> list[OpenBMITrial]:
    events = load_openbmi_events(recording)
    trials: list[OpenBMITrial] = []
    for index, event in enumerate(events):
        onset = float(event["onset"])
        intent_end = onset + float(event["duration"])
        rest_start = onset + 5.0
        rest_end = rest_start + 4.0
        next_onset = (
            float(events[index + 1]["onset"])
            if index + 1 < len(events)
            else None
        )
        flags: list[str] = []
        if next_onset is not None and rest_end > next_onset - 0.5:
            flags.append("short_intertrial_rest")
            rest_end = next_onset - 0.5
        trial_type = str(event["trial_type"])
        trials.append(
            OpenBMITrial(
                dataset=DATASET_KEY,
                subject=f"openbmi:sub-{recording.subject:02d}",
                session=f"ses-{recording.session}",
                run=f"run-{recording.native_run}",
                recording=recording.recording,
                trial_index=index,
                cue_onset_s=onset,
                intent_start_s=onset,
                intent_end_s=intent_end,
                rest_start_s=rest_start,
                rest_end_s=rest_end,
                native_label=trial_type,
                native_value=str(event["value"]),
                canonical_label=trial_type,
                timing_source="bids_events_tsv",
                recording_path=str(recording.bdf_path),
                quality_flags=";".join(flags),
            )
        )
    return trials


def inspect_openbmi_recording(
    recording: OpenBMIRecording,
) -> tuple[OpenBMIRecordingSummary, list[OpenBMITrial]]:
    total, eeg_count, emg_count = _channel_counts(recording.channels_path)
    trials = build_recording_trials(recording)
    flags: list[str] = []
    if total != EXPECTED_EEG_CHANNELS + EXPECTED_EMG_CHANNELS:
        flags.append("unexpected_channel_count")
    if eeg_count != EXPECTED_EEG_CHANNELS:
        flags.append("unexpected_eeg_count")
    if emg_count != EXPECTED_EMG_CHANNELS:
        flags.append("unexpected_emg_count")
    if any(trial.quality_flags for trial in trials):
        flags.append("short_intertrial_rest")
    left = sum(trial.canonical_label == "left_hand" for trial in trials)
    right = sum(trial.canonical_label == "right_hand" for trial in trials)
    return (
        OpenBMIRecordingSummary(
            subject=f"sub-{recording.subject}",
            session=f"ses-{recording.session}",
            recording=recording.recording,
            path=str(recording.bdf_path),
            signal_available=recording.bdf_path.is_file(),
            sampling_hz=NATIVE_SAMPLING_HZ,
            channels_total=total,
            eeg_channels=eeg_count,
            emg_channels=emg_count,
            trials=len(trials),
            left_trials=left,
            right_trials=right,
            quality_flags=";".join(flags),
        ),
        trials,
    )


def iter_openbmi_dataset(
    root: str | Path,
    subjects: Sequence[int] | None = None,
) -> Iterable[tuple[OpenBMIRecordingSummary, list[OpenBMITrial]]]:
    selected = (
        set(subjects)
        if subjects is not None
        else set(available_openbmi_subjects(root, complete_only=True))
    )
    for recording in discover_openbmi_recordings(root):
        if recording.subject in selected:
            yield inspect_openbmi_recording(recording)


def openbmi_epoch_fingerprint(
    subjects: Sequence[int],
    config: OpenBMIEpochConfig,
) -> str:
    payload = {
        "schema_version": 1,
        "dataset": DATASET_KEY,
        "subjects": sorted(int(subject) for subject in subjects),
        "session_policy": {
            "session_1_train": 4,
            "session_1_test": 8,
            "session_2_train_and_test": 12,
        },
        "label_mapping": {"rest": 0, **LABEL_IDS},
        "rest_definition": "5.0_to_9.0_seconds_after_imagery_onset",
        "baseline_seconds": config.baseline_seconds,
        "epoch_start_seconds": config.epoch_start_seconds,
        "epoch_duration_seconds": config.epoch_duration_seconds,
        "artifact_peak_to_peak_uv": config.artifact_peak_to_peak_uv,
        "channels": list(config.channels),
        "preprocessing": config.preprocessing.fingerprint(),
    }
    return json.dumps(payload, sort_keys=True)


def build_openbmi_raw_epoch_dataset(
    root: str | Path,
    subjects: Sequence[int],
    cache_path: Path,
    *,
    config: OpenBMIEpochConfig = OpenBMIEpochConfig(),
    rebuild_cache: bool = False,
) -> RawEpochDataset:
    dataset_root = resolve_openbmi_root(root)
    selected = sorted(set(int(subject) for subject in subjects))
    complete = set(available_openbmi_subjects(dataset_root, complete_only=True))
    missing = sorted(set(selected) - complete)
    if missing:
        raise FileNotFoundError(
            f"OpenBMI subjects without all four BDF recordings: {missing}"
        )
    fingerprint = openbmi_epoch_fingerprint(selected, config)
    if not rebuild_cache:
        cached = load_epoch_cache(cache_path, fingerprint)
        if cached is not None:
            print(f"[+] Loaded {len(cached.labels):,} OpenBMI windows from {cache_path}")
            return cached

    epoch_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    labels: list[int] = []
    subject_rows: list[int] = []
    run_rows: list[int] = []
    trial_rows: list[int] = []
    channel_names: list[str] | None = None
    rejected = 0
    subject_trial = {subject: 0 for subject in selected}
    records = [
        record
        for record in discover_openbmi_recordings(dataset_root)
        if record.subject in set(selected)
    ]
    print(
        f"[+] Extracting OpenBMI raw windows from {len(selected)} subjects "
        f"({len(records)} recordings)..."
    )
    for position, record in enumerate(records, start=1):
        raw = read_openbmi_raw(record, preload=False)
        processed = preprocess_run(raw, config.preprocessing, config.channels)
        if channel_names is None:
            channel_names = list(processed.ch_names)
        signal = processed.get_data()
        sfreq = float(processed.info["sfreq"])
        baseline_samples = int(round(config.baseline_seconds * sfreq))
        epoch_samples = int(round(config.epoch_duration_seconds * sfreq))
        epoch_offset = int(round(config.epoch_start_seconds * sfreq))

        for event in load_openbmi_events(record):
            onset = int(round(float(event["onset"]) * sfreq))
            intent_start = onset + epoch_offset
            rest_anchor = onset + int(round(config.rest_start_seconds * sfreq))
            rest_start = rest_anchor + epoch_offset
            windows = (
                (
                    LABEL_IDS[str(event["trial_type"])],
                    onset - baseline_samples,
                    onset,
                    intent_start,
                ),
                (
                    0,
                    rest_anchor - baseline_samples,
                    rest_anchor,
                    rest_start,
                ),
            )
            for label, baseline_start, baseline_stop, start in windows:
                stop = start + epoch_samples
                if baseline_start < 0 or stop > signal.shape[1]:
                    rejected += 1
                    continue
                baseline = signal[:, baseline_start:baseline_stop].copy()
                epoch = signal[:, start:stop].copy()
                if not is_valid_epoch(
                    baseline,
                    epoch,
                    peak_to_peak_uv=config.artifact_peak_to_peak_uv,
                ):
                    rejected += 1
                    continue
                baseline_mean = np.mean(baseline, axis=1, keepdims=True)
                baseline -= baseline_mean
                epoch -= baseline_mean
                epoch_rows.append(epoch.astype(np.float32))
                baseline_rows.append(baseline.astype(np.float32))
                labels.append(label)
                subject_rows.append(record.subject)
                run_rows.append(record.pipeline_run)
                trial_rows.append(subject_trial[record.subject])
                subject_trial[record.subject] += 1
        del raw, processed, signal
        if position % 4 == 0 or position == len(records):
            print(
                f"    processed {position:3d}/{len(records)} recordings; "
                f"kept {len(labels):,} windows"
            )

    if not epoch_rows or channel_names is None:
        raise RuntimeError("No valid OpenBMI windows were extracted.")
    dataset = RawEpochDataset(
        epochs=np.stack(epoch_rows),
        baselines=np.stack(baseline_rows),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subject_rows, dtype=np.int64),
        runs=np.asarray(run_rows, dtype=np.int64),
        trial_indices=np.asarray(trial_rows, dtype=np.int64),
        channel_names=channel_names,
        sfreq=config.preprocessing.sfreq,
    )
    save_epoch_cache(cache_path, fingerprint, dataset)
    print(f"[+] Kept {len(labels):,} windows; rejected {rejected:,}")
    return dataset
