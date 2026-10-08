"""Native-recording builder for the translational motor-intent v3 dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import mne
import numpy as np

from .dataset import (
    IMAGERY_RUNS,
    available_subjects,
    find_run_file,
    map_task_annotation,
    resolve_data_dir,
)
from .liu_stroke import (
    available_liu_subjects,
    build_subject_trials,
    load_participants,
    read_liu_raw,
    resolve_liu_root,
)
from .openbmi import (
    LABEL_IDS,
    available_openbmi_subjects,
    discover_openbmi_recordings,
    load_openbmi_events,
    read_openbmi_raw,
    resolve_openbmi_root,
)
from .quality_preprocessing import (
    BranchPreprocessingProfile,
    RecordingQualityError,
    extract_branch_window,
    preprocess_native_branches,
    window_quality_features,
)
from .research_contract import event_contract


TRANSLATIONAL_DATASET_SCHEMA_VERSION = 2
WINDOW_DURATION_SECONDS = 2.0
WINDOW_OFFSET_SECONDS = 0.5
LIU_INTENT_OFFSET_SECONDS = 1.0
HARD_REJECT_PEAK_TO_PEAK_UV = 1000.0


class SubjectCompletenessError(RuntimeError):
    """Raised when a subject cannot support calibration and evaluation."""


@dataclass
class TranslationalWindowDataset:
    """Common motor-region windows produced after native-montage processing."""

    mrcp_epochs: np.ndarray
    smr_epochs: np.ndarray
    quality_features: np.ndarray
    labels: np.ndarray
    dataset_ids: np.ndarray
    subject_keys: np.ndarray
    sessions: np.ndarray
    recordings: np.ndarray
    trial_indices: np.ndarray
    negative_kinds: np.ndarray
    cue_locked: np.ndarray
    spatial_methods: np.ndarray
    window_onsets_seconds: np.ndarray
    paralysis_sides: np.ndarray
    nihss: np.ndarray
    channel_names: tuple[str, ...]
    quality_feature_names: tuple[str, ...]
    sfreq: float
    fingerprint: str
    recording_qc: tuple[dict[str, object], ...]

    def validate(self) -> None:
        count = len(self.labels)
        arrays = (
            self.mrcp_epochs,
            self.smr_epochs,
            self.quality_features,
            self.dataset_ids,
            self.subject_keys,
            self.sessions,
            self.recordings,
            self.trial_indices,
            self.negative_kinds,
            self.cue_locked,
            self.spatial_methods,
            self.window_onsets_seconds,
            self.paralysis_sides,
            self.nihss,
        )
        if count == 0 or any(len(values) != count for values in arrays):
            raise ValueError("Translational dataset arrays are empty or not row-aligned.")
        expected_samples = int(round(self.sfreq * WINDOW_DURATION_SECONDS))
        expected_shape = (count, len(self.channel_names), expected_samples)
        if self.mrcp_epochs.shape != expected_shape:
            raise ValueError("MRCP epochs have the wrong shape.")
        if self.smr_epochs.shape != expected_shape:
            raise ValueError("SMR epochs have the wrong shape.")
        if self.quality_features.shape != (
            count,
            len(self.quality_feature_names),
        ):
            raise ValueError("Quality features do not match their names.")
        if not np.all(np.isfinite(self.mrcp_epochs)):
            raise ValueError("MRCP epochs contain non-finite values.")
        if not np.all(np.isfinite(self.smr_epochs)):
            raise ValueError("SMR epochs contain non-finite values.")
        if not np.all(np.isfinite(self.quality_features)):
            raise ValueError("Quality features contain non-finite values.")
        if not set(np.unique(self.labels)).issubset({0, 1, 2}):
            raise ValueError("Labels must be rest, left hand, or right hand.")
        for dataset_id in np.unique(self.dataset_ids):
            selected = self.dataset_ids == dataset_id
            local = set(np.unique(self.labels[selected]))
            continuous_negative_only = local == {0} and np.all(
                self.negative_kinds[selected] == "continuous_no_control"
            )
            if local != {0, 1, 2} and not continuous_negative_only:
                raise ValueError(f"Dataset {dataset_id} is missing a required class.")


@dataclass
class _Rows:
    mrcp: list[np.ndarray]
    smr: list[np.ndarray]
    quality: list[np.ndarray]
    labels: list[int]
    dataset_ids: list[str]
    subject_keys: list[str]
    sessions: list[int]
    recordings: list[int]
    trial_indices: list[int]
    negative_kinds: list[str]
    cue_locked: list[bool]
    spatial_methods: list[str]
    onsets: list[float]
    paralysis_sides: list[str]
    nihss: list[float]
    recording_qc: list[dict[str, object]]
    quality_names: tuple[str, ...] | None = None
    channel_names: tuple[str, ...] | None = None


def _empty_rows() -> _Rows:
    return _Rows(
        mrcp=[],
        smr=[],
        quality=[],
        labels=[],
        dataset_ids=[],
        subject_keys=[],
        sessions=[],
        recordings=[],
        trial_indices=[],
        negative_kinds=[],
        cue_locked=[],
        spatial_methods=[],
        onsets=[],
        paralysis_sides=[],
        nihss=[],
        recording_qc=[],
    )


def _fingerprint(
    sources: Mapping[str, Sequence[int]],
    profile: BranchPreprocessingProfile,
    hard_negative_manifest: Path | None,
) -> str:
    payload = {
        "schema_version": TRANSLATIONAL_DATASET_SCHEMA_VERSION,
        "sources": {
            key: [int(value) for value in values]
            for key, values in sources.items()
        },
        "profile": profile.fingerprint(),
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "window_offset_seconds": WINDOW_OFFSET_SECONDS,
        "liu_intent_offset_seconds": LIU_INTENT_OFFSET_SECONDS,
        "hard_reject_peak_to_peak_uv": HARD_REJECT_PEAK_TO_PEAK_UV,
        "hard_negative_manifest": (
            str(hard_negative_manifest.resolve())
            if hard_negative_manifest is not None
            else None
        ),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("ascii")
    return sha256(encoded).hexdigest()


def summarize_recording_qc(
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize accepted recordings and subject-level exclusions."""
    accepted = [
        dict(entry)
        for entry in entries
        if entry.get("status", "accepted") == "accepted"
    ]
    excluded = [
        dict(entry)
        for entry in entries
        if entry.get("status") == "rejected"
    ]
    return {
        "accepted_recordings": len(accepted),
        "excluded_recordings_or_subjects": len(excluded),
        "excluded_subjects": sorted(
            {
                str(entry["subject"])
                for entry in excluded
                if entry.get("subject")
            }
        ),
        "exclusions": excluded,
    }


def save_translational_dataset(
    path: Path,
    dataset: TranslationalWindowDataset,
) -> None:
    dataset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mrcp_epochs=dataset.mrcp_epochs,
        smr_epochs=dataset.smr_epochs,
        quality_features=dataset.quality_features,
        labels=dataset.labels,
        dataset_ids=dataset.dataset_ids,
        subject_keys=dataset.subject_keys,
        sessions=dataset.sessions,
        recordings=dataset.recordings,
        trial_indices=dataset.trial_indices,
        negative_kinds=dataset.negative_kinds,
        cue_locked=dataset.cue_locked,
        spatial_methods=dataset.spatial_methods,
        window_onsets_seconds=dataset.window_onsets_seconds,
        paralysis_sides=dataset.paralysis_sides,
        nihss=dataset.nihss,
        channel_names=np.asarray(dataset.channel_names),
        quality_feature_names=np.asarray(dataset.quality_feature_names),
        sfreq=np.asarray(dataset.sfreq),
        fingerprint=np.asarray(dataset.fingerprint),
        recording_qc=np.asarray(json.dumps(dataset.recording_qc)),
    )
    qc_path = path.with_name(f"{path.stem}.qc.json")
    qc_path.write_text(
        json.dumps(
            {
                "summary": summarize_recording_qc(dataset.recording_qc),
                "recordings": dataset.recording_qc,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )


def load_translational_dataset(
    path: Path,
    expected_fingerprint: str | None = None,
) -> TranslationalWindowDataset | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        fingerprint = str(cached["fingerprint"].item())
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            return None
        dataset = TranslationalWindowDataset(
            mrcp_epochs=cached["mrcp_epochs"],
            smr_epochs=cached["smr_epochs"],
            quality_features=cached["quality_features"],
            labels=cached["labels"],
            dataset_ids=cached["dataset_ids"],
            subject_keys=cached["subject_keys"],
            sessions=cached["sessions"],
            recordings=cached["recordings"],
            trial_indices=cached["trial_indices"],
            negative_kinds=cached["negative_kinds"],
            cue_locked=cached["cue_locked"],
            spatial_methods=cached["spatial_methods"],
            window_onsets_seconds=cached["window_onsets_seconds"],
            paralysis_sides=cached["paralysis_sides"],
            nihss=cached["nihss"],
            channel_names=tuple(str(value) for value in cached["channel_names"]),
            quality_feature_names=tuple(
                str(value) for value in cached["quality_feature_names"]
            ),
            sfreq=float(cached["sfreq"].item()),
            fingerprint=fingerprint,
            recording_qc=tuple(
                json.loads(str(cached["recording_qc"].item()))
            ),
        )
    dataset.validate()
    return dataset


def _append_window(
    rows: _Rows,
    branches: object,
    *,
    onset_seconds: float,
    label: int,
    dataset_id: str,
    subject_key: str,
    session: int,
    recording: int,
    trial_index: int,
    negative_kind: str,
    cue_locked: bool,
    paralysis_side: str = "not_applicable",
    nihss: float = np.nan,
) -> bool:
    try:
        windows = extract_branch_window(
            branches,
            onset_seconds,
            WINDOW_DURATION_SECONDS,
        )
    except ValueError:
        return False
    mrcp = np.asarray(windows["mrcp"])
    smr_car = np.asarray(windows["smr_car"])
    smr_csd = windows["smr_csd"]
    quality_window = np.asarray(windows["quality"])
    auxiliary = windows["auxiliary"]
    if (
        not np.all(np.isfinite(mrcp))
        or not np.all(np.isfinite(smr_car))
        or not np.all(np.isfinite(quality_window))
    ):
        return False
    if np.max(np.ptp(quality_window, axis=-1)) * 1e6 > HARD_REJECT_PEAK_TO_PEAK_UV:
        return False
    quality, quality_names = window_quality_features(
        quality_window,
        None if auxiliary is None else np.asarray(auxiliary),
    )
    quality = np.concatenate(
        (
            quality,
            np.asarray(
                [
                    float(branches.recording_qc["bad_channel_fraction"]),
                    float(smr_csd is not None),
                ],
                dtype=np.float32,
            ),
        )
    )
    names = (
        *quality_names,
        "recording_bad_channel_fraction",
        "csd_available",
    )
    if rows.quality_names is None:
        rows.quality_names = tuple(names)
        rows.channel_names = tuple(branches.selected_channels)
    elif tuple(names) != rows.quality_names:
        raise ValueError("Quality feature names changed within a dataset build.")

    rows.mrcp.append(mrcp.astype(np.float32))
    rows.smr.append(
        np.asarray(smr_csd if smr_csd is not None else smr_car).astype(np.float32)
    )
    rows.quality.append(quality)
    rows.labels.append(int(label))
    rows.dataset_ids.append(dataset_id)
    rows.subject_keys.append(subject_key)
    rows.sessions.append(int(session))
    rows.recordings.append(int(recording))
    rows.trial_indices.append(int(trial_index))
    rows.negative_kinds.append("not_applicable" if label else negative_kind)
    rows.cue_locked.append(bool(cue_locked))
    rows.spatial_methods.append("csd" if smr_csd is not None else "car")
    rows.onsets.append(float(onset_seconds))
    rows.paralysis_sides.append(paralysis_side)
    rows.nihss.append(float(nihss))
    return True


def _record_qc(
    rows: _Rows,
    branches: object,
    **metadata: object,
) -> None:
    rows.recording_qc.append(
        {
            **metadata,
            "status": "accepted",
            **branches.recording_qc,
        }
    )


def _record_exclusion(
    rows: _Rows,
    error: Exception,
    **metadata: object,
) -> None:
    quality = (
        error.recording_qc
        if isinstance(error, RecordingQualityError)
        else {}
    )
    rows.recording_qc.append(
        {
            **metadata,
            "status": "rejected",
            "error_type": type(error).__name__,
            "reason": str(error),
            **quality,
        }
    )


def _merge_rows(target: _Rows, source: _Rows) -> None:
    if not source.labels:
        return
    if target.quality_names is None:
        target.quality_names = source.quality_names
        target.channel_names = source.channel_names
    elif (
        target.quality_names != source.quality_names
        or target.channel_names != source.channel_names
    ):
        raise ValueError("Feature geometry changed while combining subjects.")
    for field in (
        "mrcp",
        "smr",
        "quality",
        "labels",
        "dataset_ids",
        "subject_keys",
        "sessions",
        "recordings",
        "trial_indices",
        "negative_kinds",
        "cue_locked",
        "spatial_methods",
        "onsets",
        "paralysis_sides",
        "nihss",
        "recording_qc",
    ):
        getattr(target, field).extend(getattr(source, field))


def _validate_complete_subject(rows: _Rows, subject_key: str) -> None:
    labels = set(rows.labels)
    recordings = set(rows.recordings)
    if labels != {0, 1, 2} or not {0, 1}.issubset(recordings):
        raise SubjectCompletenessError(
            f"{subject_key} is incomplete after window QC: "
            f"labels={sorted(labels)}, recordings={sorted(recordings)}."
        )


def _build_physionet(
    rows: _Rows,
    root: Path,
    subjects: Sequence[int],
    profile: BranchPreprocessingProfile,
) -> None:
    semantics = event_contract("physionet")
    print(f"[+] Building native PhysioNet branches for {len(subjects)} subjects...")
    for position, subject in enumerate(subjects, start=1):
        subject_key = f"physionet:S{subject:03d}"
        subject_rows = _empty_rows()
        current_path = root
        try:
            for run in IMAGERY_RUNS:
                path = find_run_file(root, int(subject), run)
                if path is None:
                    raise FileNotFoundError(
                        f"Missing S{subject:03d}R{run:02d}.edf"
                    )
                current_path = path
                raw = mne.io.read_raw_edf(
                    path,
                    preload=False,
                    verbose=False,
                )
                branches = preprocess_native_branches(raw, profile)
                _record_qc(
                    subject_rows,
                    branches,
                    dataset="physionet",
                    subject=subject_key,
                    session=1,
                    run=run,
                    path=str(path),
                )
                trial_index = 0
                for annotation in raw.annotations:
                    label = map_task_annotation(
                        str(annotation["description"])
                    )
                    if label is None:
                        continue
                    _append_window(
                        subject_rows,
                        branches,
                        onset_seconds=(
                            float(annotation["onset"])
                            + WINDOW_OFFSET_SECONDS
                        ),
                        label=label,
                        dataset_id="physionet",
                        subject_key=subject_key,
                        session=1,
                        recording=0 if run in (4, 8) else 1,
                        trial_index=run * 1000 + trial_index,
                        negative_kind=semantics.negative_subtype,
                        cue_locked=semantics.cue_locked,
                    )
                    trial_index += 1
                del raw, branches
            _validate_complete_subject(subject_rows, subject_key)
        except (
            FileNotFoundError,
            OSError,
            RecordingQualityError,
            SubjectCompletenessError,
        ) as error:
            _record_exclusion(
                rows,
                error,
                dataset="physionet",
                subject=subject_key,
                path=str(current_path),
                exclusion_scope="subject",
            )
            print(f"    [!] Excluded {subject_key}: {error}")
        else:
            _merge_rows(rows, subject_rows)
        if position % 10 == 0 or position == len(subjects):
            print(
                f"    PhysioNet {position:3d}/{len(subjects)} subjects; "
                f"{len(rows.labels):,} total windows"
            )


def _build_openbmi(
    rows: _Rows,
    root: Path,
    subjects: Sequence[int],
    profile: BranchPreprocessingProfile,
) -> None:
    semantics = event_contract("openbmi")
    selected = set(int(subject) for subject in subjects)
    recordings = [
        value
        for value in discover_openbmi_recordings(root)
        if value.subject in selected
    ]
    print(
        f"[+] Building native OpenBMI branches for {len(subjects)} subjects "
        f"({len(recordings)} recordings)..."
    )
    for position, subject in enumerate(subjects, start=1):
        subject_key = f"openbmi:S{subject:03d}"
        subject_rows = _empty_rows()
        subject_recordings = [
            recording
            for recording in recordings
            if recording.subject == int(subject)
        ]
        current_path = root
        try:
            if not subject_recordings:
                raise FileNotFoundError(
                    f"No complete OpenBMI recordings for subject {subject}."
                )
            for recording in subject_recordings:
                current_path = recording.bdf_path
                raw = read_openbmi_raw(recording, preload=False)
                branches = preprocess_native_branches(raw, profile)
                _record_qc(
                    subject_rows,
                    branches,
                    dataset="openbmi",
                    subject=subject_key,
                    session=recording.session,
                    run=recording.pipeline_run,
                    path=str(recording.bdf_path),
                )
                for event in load_openbmi_events(recording):
                    onset = float(event["onset"])
                    label = LABEL_IDS[str(event["trial_type"])]
                    base_index = (
                        recording.session * 100000
                        + recording.native_run * 10000
                        + int(event["trial_index"]) * 2
                    )
                    _append_window(
                        subject_rows,
                        branches,
                        onset_seconds=onset + WINDOW_OFFSET_SECONDS,
                        label=label,
                        dataset_id="openbmi",
                        subject_key=subject_key,
                        session=recording.session,
                        recording=(
                            0 if recording.pipeline_run == 4 else 1
                        ),
                        trial_index=base_index,
                        negative_kind=semantics.negative_subtype,
                        cue_locked=semantics.cue_locked,
                    )
                    _append_window(
                        subject_rows,
                        branches,
                        onset_seconds=(
                            onset + 5.0 + WINDOW_OFFSET_SECONDS
                        ),
                        label=0,
                        dataset_id="openbmi",
                        subject_key=subject_key,
                        session=recording.session,
                        recording=(
                            0 if recording.pipeline_run == 4 else 1
                        ),
                        trial_index=base_index + 1,
                        negative_kind=semantics.negative_subtype,
                        cue_locked=semantics.cue_locked,
                    )
                del raw, branches
            _validate_complete_subject(subject_rows, subject_key)
        except (
            FileNotFoundError,
            OSError,
            RecordingQualityError,
            SubjectCompletenessError,
        ) as error:
            _record_exclusion(
                rows,
                error,
                dataset="openbmi",
                subject=subject_key,
                path=str(current_path),
                exclusion_scope="subject",
            )
            print(f"    [!] Excluded {subject_key}: {error}")
        else:
            _merge_rows(rows, subject_rows)
        if position % 10 == 0 or position == len(subjects):
            print(
                f"    OpenBMI {position:3d}/{len(subjects)} subjects; "
                f"{len(rows.labels):,} total windows"
            )


def _build_liu(
    rows: _Rows,
    root: Path,
    subjects: Sequence[int],
    profile: BranchPreprocessingProfile,
) -> None:
    semantics = event_contract("liu2024")
    participants = load_participants(root)
    print(f"[+] Building native Liu branches for {len(subjects)} patients...")
    for position, subject in enumerate(subjects, start=1):
        subject_key = f"liu2024:S{subject:03d}"
        raw = None
        recording_path = str(root)
        subject_rows = _empty_rows()
        try:
            participant_key = f"sub-{subject:02d}"
            if participant_key not in participants:
                raise FileNotFoundError(
                    f"Missing participant metadata for {participant_key}."
                )
            participant = participants[participant_key]
            raw = read_liu_raw(root, int(subject), preload=False)
            trials = build_subject_trials(
                root,
                int(subject),
                raw=raw,
                participant=participant,
            )
            if not trials:
                raise SubjectCompletenessError(
                    f"{subject_key} contains no usable trials."
                )
            recording_path = str(trials[0].recording_path)
            branches = preprocess_native_branches(raw, profile)
            _record_qc(
                subject_rows,
                branches,
                dataset="liu2024",
                subject=subject_key,
                session=1,
                run=1,
                path=recording_path,
            )
            for trial in trials:
                recording = int(trial.trial_index >= 20)
                _append_window(
                    subject_rows,
                    branches,
                    onset_seconds=(
                        trial.intent_start_s + LIU_INTENT_OFFSET_SECONDS
                    ),
                    label=trial.native_label,
                    dataset_id="liu2024",
                    subject_key=subject_key,
                    session=1,
                    recording=recording,
                    trial_index=trial.trial_index * 2,
                    negative_kind=semantics.negative_subtype,
                    cue_locked=semantics.cue_locked,
                    paralysis_side=trial.paralysis_side.strip().lower(),
                    nihss=float(trial.nihss),
                )
                _append_window(
                    subject_rows,
                    branches,
                    onset_seconds=trial.rest_start_s,
                    label=0,
                    dataset_id="liu2024",
                    subject_key=subject_key,
                    session=1,
                    recording=recording,
                    trial_index=trial.trial_index * 2 + 1,
                    negative_kind=semantics.negative_subtype,
                    cue_locked=semantics.cue_locked,
                    paralysis_side=trial.paralysis_side.strip().lower(),
                    nihss=float(trial.nihss),
                )
            _validate_complete_subject(subject_rows, subject_key)
        except (
            FileNotFoundError,
            OSError,
            RecordingQualityError,
            SubjectCompletenessError,
        ) as error:
            _record_exclusion(
                rows,
                error,
                dataset="liu2024",
                subject=subject_key,
                path=recording_path,
                exclusion_scope="subject",
            )
            print(f"    [!] Excluded {subject_key}: {error}")
        else:
            _merge_rows(rows, subject_rows)
        if raw is not None:
            del raw
        if position % 10 == 0 or position == len(subjects):
            print(
                f"    Liu {position:2d}/{len(subjects)} patients; "
                f"{len(rows.labels):,} total windows"
            )


def _read_manifest_recording(entry: Mapping[str, object]) -> mne.io.BaseRaw:
    path = Path(str(entry["path"])).expanduser().resolve()
    extension = (
        ".fif.gz"
        if path.name.lower().endswith(".fif.gz")
        else path.suffix.lower()
    )
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".gdf": mne.io.read_raw_gdf,
        ".fif": mne.io.read_raw_fif,
        ".fif.gz": mne.io.read_raw_fif,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".cnt": mne.io.read_raw_cnt,
    }
    if extension not in readers:
        raise ValueError(
            f"Hard-negative manifest format is not supported: {extension}"
        )
    return readers[extension](path, preload=False, verbose=False)


def _build_hard_negatives(
    rows: _Rows,
    manifest_path: Path,
    profile: BranchPreprocessingProfile,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("recordings", [])
    if not entries:
        raise ValueError("Hard-negative manifest contains no recordings.")
    semantics = event_contract("hard_negative")
    print(f"[+] Building continuous hard negatives from {len(entries)} recordings...")
    for entry_index, entry in enumerate(entries):
        raw = _read_manifest_recording(entry)
        branches = preprocess_native_branches(raw, profile)
        duration = float(branches.mrcp.n_times / branches.mrcp.info["sfreq"])
        start = max(0.0, float(entry.get("start_seconds", 0.0)))
        stop = min(duration, float(entry.get("stop_seconds", duration)))
        step = float(entry.get("step_seconds", 1.0))
        subject_key = str(entry["subject_key"])
        dataset_id = str(entry.get("dataset_id", "hard_negative"))
        session = int(entry.get("session", 1))
        recording = int(entry.get("recording", 1))
        _record_qc(
            rows,
            branches,
            dataset=dataset_id,
            subject=subject_key,
            session=session,
            run=recording,
            path=str(entry["path"]),
            negative_only=True,
        )
        onset = start
        local_index = 0
        while onset + WINDOW_DURATION_SECONDS <= stop + 1e-9:
            _append_window(
                rows,
                branches,
                onset_seconds=onset,
                label=0,
                dataset_id=dataset_id,
                subject_key=subject_key,
                session=session,
                recording=recording,
                trial_index=entry_index * 100000 + local_index,
                negative_kind=str(
                    entry.get("negative_kind", semantics.negative_subtype)
                ),
                cue_locked=False,
            )
            onset += step
            local_index += 1
        del raw, branches


def build_translational_window_dataset(
    *,
    physionet_dir: str | Path,
    openbmi_dir: str | Path,
    liu_dir: str | Path,
    cache_path: Path,
    physionet_subjects: Sequence[int] | None = None,
    openbmi_subjects: Sequence[int] | None = None,
    liu_subjects: Sequence[int] | None = None,
    max_subjects_per_dataset: int | None = None,
    hard_negative_manifest: str | Path | None = None,
    profile: BranchPreprocessingProfile = BranchPreprocessingProfile(),
    rebuild_cache: bool = False,
) -> TranslationalWindowDataset:
    """Build all three datasets with native QC before common channel selection."""
    physionet_root = resolve_data_dir(physionet_dir)
    openbmi_root = resolve_openbmi_root(openbmi_dir)
    liu_root = resolve_liu_root(liu_dir)
    selected_physionet = (
        available_subjects(physionet_root)
        if physionet_subjects is None
        else sorted(set(int(value) for value in physionet_subjects))
    )
    selected_openbmi = (
        available_openbmi_subjects(openbmi_root, complete_only=True)
        if openbmi_subjects is None
        else sorted(set(int(value) for value in openbmi_subjects))
    )
    selected_liu = (
        available_liu_subjects(liu_root)
        if liu_subjects is None
        else sorted(set(int(value) for value in liu_subjects))
    )
    if max_subjects_per_dataset is not None:
        selected_physionet = selected_physionet[:max_subjects_per_dataset]
        selected_openbmi = selected_openbmi[:max_subjects_per_dataset]
        selected_liu = selected_liu[:max_subjects_per_dataset]
    manifest = (
        Path(hard_negative_manifest).expanduser().resolve()
        if hard_negative_manifest is not None
        else None
    )
    fingerprint = _fingerprint(
        {
            "physionet": selected_physionet,
            "openbmi": selected_openbmi,
            "liu2024": selected_liu,
        },
        profile,
        manifest,
    )
    if not rebuild_cache:
        cached = load_translational_dataset(cache_path, fingerprint)
        if cached is not None:
            print(
                f"[+] Loaded {len(cached.labels):,} translational windows "
                f"from {cache_path}"
            )
            return cached

    rows = _empty_rows()
    _build_physionet(rows, physionet_root, selected_physionet, profile)
    _build_openbmi(rows, openbmi_root, selected_openbmi, profile)
    _build_liu(rows, liu_root, selected_liu, profile)
    if manifest is not None:
        _build_hard_negatives(rows, manifest, profile)
    if not rows.labels or rows.channel_names is None or rows.quality_names is None:
        raise RuntimeError("No valid translational windows were extracted.")

    dataset = TranslationalWindowDataset(
        mrcp_epochs=np.stack(rows.mrcp),
        smr_epochs=np.stack(rows.smr),
        quality_features=np.stack(rows.quality),
        labels=np.asarray(rows.labels, dtype=np.int64),
        dataset_ids=np.asarray(rows.dataset_ids, dtype="<U32"),
        subject_keys=np.asarray(rows.subject_keys, dtype="<U64"),
        sessions=np.asarray(rows.sessions, dtype=np.int16),
        recordings=np.asarray(rows.recordings, dtype=np.int8),
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
    print(f"[+] Saved {len(dataset.labels):,} windows to {cache_path}")
    return dataset
