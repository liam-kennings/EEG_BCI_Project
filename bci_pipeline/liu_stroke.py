"""Adapter and structural validation for the Liu acute-stroke MI dataset."""

from __future__ import annotations

import csv
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np

from . import PROJECT_ROOT


DATASET_KEY = "liu2024"
SAMPLING_HZ = 500.0
EXPECTED_SUBJECTS = 50
EXPECTED_TRIALS_PER_SUBJECT = 40
EXPECTED_EEG_CHANNELS = 29
EXPECTED_EOG_CHANNELS = 2
STAGE_NAMES = {1: "instruction", 2: "motor_imagery", 3: "break"}
CANONICAL_LABELS = {1: "left_hand", 2: "right_hand"}


@dataclass(frozen=True)
class LiuTrial:
    """One protocol trial with subject-specific trigger timing."""

    dataset: str
    subject: str
    session: str
    run: str
    trial_index: int
    cue_onset_s: float
    intent_start_s: float
    intent_end_s: float
    rest_start_s: float
    rest_end_s: float
    native_label: int
    canonical_label: str
    timing_source: str
    recording_path: str
    paralysis_side: str
    handedness: str
    nihss: int
    mbi: int
    mrs: int
    quality_flags: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LiuRecordingSummary:
    subject: str
    path: str
    sampling_hz: float
    duration_seconds: float
    channels_total: int
    eeg_channels: int
    eog_channels: int
    stim_channels: int
    trials: int
    left_trials: int
    right_trials: int
    timing_source: str
    trigger_fallbacks: int
    ignored_trigger_pulses: int
    quality_flags: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_liu_root(value: str | Path) -> Path:
    """Resolve either the dataset root or one of its internal folders."""
    candidate = Path(value).expanduser()
    candidates = (
        candidate,
        PROJECT_ROOT / candidate,
        candidate.parent if candidate.name in {"edffile", "sourcedata"} else candidate,
    )
    for root in candidates:
        root = root.resolve()
        if (
            (root / "participants.tsv").is_file()
            and (root / "task-motor-imagery_events.tsv").is_file()
            and (root / "edffile").is_dir()
        ):
            return root
    raise FileNotFoundError(
        "Could not find a Liu dataset root containing participants.tsv, "
        "task-motor-imagery_events.tsv, and edffile/."
    )


def available_liu_subjects(root: str | Path) -> list[int]:
    dataset_root = resolve_liu_root(root)
    return sorted(
        int(path.name.removeprefix("sub-"))
        for path in (dataset_root / "edffile").glob("sub-[0-9][0-9]")
        if path.is_dir()
    )


def liu_edf_path(root: str | Path, subject: int) -> Path:
    dataset_root = resolve_liu_root(root)
    subject_name = f"sub-{int(subject):02d}"
    path = (
        dataset_root
        / "edffile"
        / subject_name
        / "eeg"
        / f"{subject_name}_task-motor-imagery_eeg.edf"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing Liu recording: {path}")
    return path


def load_participants(root: str | Path) -> dict[str, dict[str, str]]:
    path = resolve_liu_root(root) / "participants.tsv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["Participant_ID"]: dict(row)
            for row in csv.DictReader(handle, delimiter="\t")
        }


def load_protocol_labels(root: str | Path) -> list[int]:
    """Read one left/right label per trial from the shared event template."""
    path = resolve_liu_root(root) / "task-motor-imagery_events.tsv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != EXPECTED_TRIALS_PER_SUBJECT * 3:
        raise ValueError(f"Expected 120 protocol stages, found {len(rows)}.")

    labels: list[int] = []
    for trial_index in range(EXPECTED_TRIALS_PER_SUBJECT):
        stages = rows[trial_index * 3 : (trial_index + 1) * 3]
        stage_values = [int(row["value"]) for row in stages]
        trial_labels = {int(row["trial_type"]) for row in stages}
        if stage_values != [1, 2, 3] or len(trial_labels) != 1:
            raise ValueError(
                f"Malformed protocol stages for trial {trial_index + 1}: "
                f"values={stage_values}, labels={sorted(trial_labels)}"
            )
        label = trial_labels.pop()
        if label not in CANONICAL_LABELS:
            raise ValueError(f"Unsupported Liu trial label: {label}")
        labels.append(label)
    return labels


def _rename_and_type_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Correct channel roles encoded incompletely in the distributed EDF."""
    adjusted = raw.copy()
    blank_channels = [name for name in adjusted.ch_names if not name.strip()]
    if len(blank_channels) == 1:
        adjusted.rename_channels({blank_channels[0]: "STIM"})

    channel_types: dict[str, str] = {}
    for name in ("HEOL", "HEOR", "VEOR"):
        if name in adjusted.ch_names:
            channel_types[name] = "eog"
    if "CPz" in adjusted.ch_names:
        # CPz was the online reference and is intentionally flat.
        channel_types["CPz"] = "misc"
    if "STIM" in adjusted.ch_names:
        channel_types["STIM"] = "stim"
    if channel_types:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The unit for channel.*has changed",
                category=RuntimeWarning,
            )
            adjusted.set_channel_types(channel_types, verbose=False)
    return adjusted


def read_liu_raw(
    root: str | Path,
    subject: int,
    *,
    preload: bool = False,
) -> mne.io.BaseRaw:
    """Read a Liu EDF while preserving EOG, reference, and trigger semantics."""
    path = liu_edf_path(root, subject)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Physical range is not defined in following channels:",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
    return _rename_and_type_channels(raw)


@dataclass(frozen=True)
class LiuTriggerAlignment:
    events: tuple[tuple[float, int], ...]
    fallback_indices: tuple[int, ...]
    observed_pulses: int
    ignored_pulses: int


def align_trigger_events(
    raw: mne.io.BaseRaw,
    *,
    tolerance_seconds: float = 0.75,
) -> LiuTriggerAlignment:
    """Reconcile EDF pulses with the fixed 8-second, three-stage protocol."""
    if "STIM" not in raw.ch_names:
        raise ValueError("Liu recording does not contain the expected STIM channel.")
    values = raw.get_data(picks=["STIM"])[0]
    # The EDF physical-unit conversion represents integer trigger levels in uV.
    levels = np.rint(values * 1e6).astype(np.int64)
    active = levels != 0
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    observed = [
        (float(sample / raw.info["sfreq"]), int(levels[sample]))
        for sample in starts
    ]

    aligned: list[tuple[float, int]] = []
    fallback_indices: list[int] = []
    used: set[int] = set()
    stage_offsets = ((0.0, 1), (2.0, 2), (6.0, 3))
    for trial_index in range(EXPECTED_TRIALS_PER_SUBJECT):
        for offset, expected_stage in stage_offsets:
            expected_onset = trial_index * 8.0 + offset
            candidates = [
                (abs(onset - expected_onset), index, onset)
                for index, (onset, stage) in enumerate(observed)
                if index not in used
                and stage == expected_stage
                and abs(onset - expected_onset) <= tolerance_seconds
            ]
            if candidates:
                _, observed_index, onset = min(candidates)
                used.add(observed_index)
                aligned.append((onset, expected_stage))
            else:
                fallback_indices.append(len(aligned))
                aligned.append((expected_onset, expected_stage))
    return LiuTriggerAlignment(
        events=tuple(aligned),
        fallback_indices=tuple(fallback_indices),
        observed_pulses=len(observed),
        ignored_pulses=len(observed) - len(used),
    )


def trigger_events(raw: mne.io.BaseRaw) -> list[tuple[float, int]]:
    """Return protocol-aligned stage events, repairing isolated pulse defects."""
    return list(align_trigger_events(raw).events)


def build_subject_trials(
    root: str | Path,
    subject: int,
    *,
    raw: mne.io.BaseRaw | None = None,
    participant: Mapping[str, str] | None = None,
    alignment: LiuTriggerAlignment | None = None,
) -> list[LiuTrial]:
    """Build the harmonized trial table for one patient."""
    dataset_root = resolve_liu_root(root)
    recording = raw if raw is not None else read_liu_raw(dataset_root, subject)
    labels = load_protocol_labels(dataset_root)
    trigger_alignment = (
        alignment if alignment is not None else align_trigger_events(recording)
    )
    events = list(trigger_alignment.events)
    fallback_indices = set(trigger_alignment.fallback_indices)
    metadata = (
        dict(participant)
        if participant is not None
        else load_participants(dataset_root)[f"sub-{subject:02d}"]
    )
    duration = float(recording.n_times / recording.info["sfreq"])
    path = liu_edf_path(dataset_root, subject)
    trials: list[LiuTrial] = []
    for trial_index, label in enumerate(labels):
        stage_events = events[trial_index * 3 : (trial_index + 1) * 3]
        cue_onset, cue_stage = stage_events[0]
        intent_start, intent_stage = stage_events[1]
        rest_start, rest_stage = stage_events[2]
        if (cue_stage, intent_stage, rest_stage) != (1, 2, 3):
            raise ValueError(f"Invalid trigger sequence for trial {trial_index + 1}.")
        rest_end = (
            events[(trial_index + 1) * 3][0]
            if trial_index + 1 < len(labels)
            else duration
        )
        used_fallback = any(
            trial_index * 3 + offset in fallback_indices for offset in range(3)
        )
        trials.append(
            LiuTrial(
                dataset=DATASET_KEY,
                subject=f"liu2024:sub-{subject:02d}",
                session="ses-01",
                run="run-01",
                trial_index=trial_index,
                cue_onset_s=cue_onset,
                intent_start_s=intent_start,
                intent_end_s=rest_start,
                rest_start_s=rest_start,
                rest_end_s=rest_end,
                native_label=label,
                canonical_label=CANONICAL_LABELS[label],
                timing_source=(
                    "protocol_fallback" if used_fallback else "edf_stim_channel"
                ),
                recording_path=str(path),
                paralysis_side=metadata["ParalysisSide"],
                handedness=metadata["Handedness"],
                nihss=int(metadata["NIHSS"]),
                mbi=int(metadata["MBI"]),
                mrs=int(metadata["mRS"]),
                quality_flags=(
                    "trigger_timing_fallback" if used_fallback else ""
                ),
            )
        )
    return trials


def inspect_liu_recording(
    root: str | Path,
    subject: int,
    *,
    participant: Mapping[str, str] | None = None,
) -> tuple[LiuRecordingSummary, list[LiuTrial]]:
    raw = read_liu_raw(root, subject)
    alignment = align_trigger_events(raw)
    trials = build_subject_trials(
        root,
        subject,
        raw=raw,
        participant=participant,
        alignment=alignment,
    )
    sfreq = float(raw.info["sfreq"])
    flags: list[str] = []
    if not np.isclose(sfreq, SAMPLING_HZ):
        flags.append("unexpected_sampling_rate")
    if len(raw.ch_names) != 33:
        flags.append("unexpected_channel_count")
    if len(mne.pick_types(raw.info, eeg=True, exclude=[])) != EXPECTED_EEG_CHANNELS:
        flags.append("unexpected_eeg_count")
    if len(mne.pick_types(raw.info, eog=True, exclude=[])) != EXPECTED_EOG_CHANNELS:
        flags.append("unexpected_eog_count")
    if float(raw.n_times / sfreq) < trials[-1].rest_end_s:
        flags.append("event_outside_recording")
    if alignment.fallback_indices:
        flags.append("trigger_timing_fallback")
    if alignment.ignored_pulses:
        flags.append("extra_trigger_pulses")
    quality_flags = ";".join(flags)
    summary = LiuRecordingSummary(
        subject=f"sub-{subject:02d}",
        path=str(liu_edf_path(root, subject)),
        sampling_hz=sfreq,
        duration_seconds=float(raw.n_times / sfreq),
        channels_total=len(raw.ch_names),
        eeg_channels=len(mne.pick_types(raw.info, eeg=True, exclude=[])),
        eog_channels=len(mne.pick_types(raw.info, eog=True, exclude=[])),
        stim_channels=len(mne.pick_types(raw.info, stim=True, exclude=[])),
        trials=len(trials),
        left_trials=sum(trial.native_label == 1 for trial in trials),
        right_trials=sum(trial.native_label == 2 for trial in trials),
        timing_source=(
            "edf_stim_channel+protocol_fallback"
            if alignment.fallback_indices
            else "edf_stim_channel"
        ),
        trigger_fallbacks=len(alignment.fallback_indices),
        ignored_trigger_pulses=alignment.ignored_pulses,
        quality_flags=quality_flags,
    )
    return summary, trials


def iter_liu_dataset(
    root: str | Path,
    subjects: Sequence[int] | None = None,
) -> Iterable[tuple[LiuRecordingSummary, list[LiuTrial]]]:
    dataset_root = resolve_liu_root(root)
    selected = list(subjects) if subjects is not None else available_liu_subjects(dataset_root)
    participants = load_participants(dataset_root)
    for subject in selected:
        participant_id = f"sub-{int(subject):02d}"
        if participant_id not in participants:
            raise ValueError(f"Missing participant metadata for {participant_id}.")
        yield inspect_liu_recording(
            dataset_root,
            int(subject),
            participant=participants[participant_id],
        )
