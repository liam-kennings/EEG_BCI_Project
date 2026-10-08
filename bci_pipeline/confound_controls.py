"""Window-definition controls for the rest-versus-intent contrast.

In every V7 dataset the intent window is taken while a visual cue is on screen
and the rest window is taken right after a task or stimulus ends. A decoder can
then separate "cue present" from "task just ended" without any motor intent.
This module re-extracts windows from the native recordings under alternative
definitions so the contrast can be tested directly:

* ``intent@<offset>``: 2-s window starting ``offset`` seconds after the cue
  (PhysioNet T1/T2 onset, OpenBMI arrow onset, Liu motor-imagery stage onset).
* ``rest_original``: the V7 negative (PhysioNet T0 + 0.5 s, OpenBMI cue + 5.5 s,
  Liu break onset), i.e. immediately after the task ends.
* ``rest_precue``: the 2 s before the cue. PhysioNet: end of the inter-trial
  rest; OpenBMI: inside the 3-s fixation cross. Liu has no pre-trial rest (the
  8-s trials are contiguous), so this is the instruction stage, where the hand
  cue is shown but imagery has not begun. It is not cue-free.

Each window is reduced to compact features so many definitions and two filter
phases (zero-phase and causal) fit in memory. Preprocessing reuses the V7
native-montage pipeline unchanged apart from the filter phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np
from scipy.signal import welch
from sklearn.covariance import oas

from .dataset import IMAGERY_RUNS, find_run_file, map_task_annotation
from .liu_stroke import build_subject_trials, load_participants, read_liu_raw
from .openbmi import (
    LABEL_IDS,
    discover_openbmi_recordings,
    load_openbmi_events,
    read_openbmi_raw,
)
from .quality_preprocessing import (
    BranchPreprocessingProfile,
    RecordingQualityError,
    extract_branch_window,
    preprocess_native_branches,
    window_quality_features,
)
from .translational_dataset import HARD_REJECT_PEAK_TO_PEAK_UV, WINDOW_DURATION_SECONDS

INTENT_OFFSETS_SECONDS = (0.0, 0.5, 1.0, 1.5, 2.0)
PRECUE_SECONDS = 2.0
OPENBMI_REST_OFFSET_SECONDS = 5.5
PHYSIONET_REST_OFFSET_SECONDS = 0.5
BANDS_HZ = {"mu": (8.0, 13.0), "beta_low": (13.0, 20.0), "beta_high": (20.0, 30.0)}
MRCP_BINS = 4
PHASES = ("zero", "forward")


@dataclass
class _FeatureRows:
    kinds: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    offsets: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    recordings: list[int] = field(default_factory=list)
    trials: list[int] = field(default_factory=list)
    amplitude: list[np.ndarray] = field(default_factory=list)
    auxiliary: list[np.ndarray] = field(default_factory=list)
    bandpower: list[np.ndarray] = field(default_factory=list)
    covariance: list[np.ndarray] = field(default_factory=list)
    mrcp: list[np.ndarray] = field(default_factory=list)

    def extend(self, other: "_FeatureRows") -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name).extend(getattr(other, name))


@dataclass(frozen=True)
class _WindowRequest:
    kind: str
    onset_seconds: float
    label: int
    offset: float
    trial: int
    recording: int


def window_features(windows: Mapping[str, np.ndarray | None], sfreq: float) -> dict[str, np.ndarray] | None:
    """Reduce one extracted branch window to the features used by the controls."""
    mrcp = np.asarray(windows["mrcp"], dtype=np.float64)
    smr = np.asarray(
        windows["smr_csd"] if windows["smr_csd"] is not None else windows["smr_car"],
        dtype=np.float64,
    )
    quality = np.asarray(windows["quality"], dtype=np.float64)
    if not (np.all(np.isfinite(mrcp)) and np.all(np.isfinite(smr)) and np.all(np.isfinite(quality))):
        return None
    if np.max(np.ptp(quality, axis=-1)) * 1e6 > HARD_REJECT_PEAK_TO_PEAK_UV:
        return None
    auxiliary_window = windows["auxiliary"]
    quality_values, _ = window_quality_features(
        quality,
        None if auxiliary_window is None else np.asarray(auxiliary_window),
    )
    frequencies, power = welch(smr, fs=sfreq, nperseg=int(sfreq), axis=-1)
    log_power = np.log(power + 1e-30)
    bandpower = np.concatenate(
        [
            log_power[:, (frequencies >= low) & (frequencies < high)].mean(axis=1)
            for low, high in BANDS_HZ.values()
        ]
    )
    scaled = smr / (np.std(smr) + 1e-30)
    covariance, _ = oas(scaled.T, assume_centered=False)
    bins = np.array_split(mrcp * 1e6, MRCP_BINS, axis=1)
    mrcp_bins = np.stack([chunk.mean(axis=1) for chunk in bins], axis=1).ravel()
    return {
        "amplitude": quality_values[:6].astype(np.float32),
        "auxiliary": quality_values[6:9].astype(np.float32),
        "bandpower": bandpower.astype(np.float32),
        "covariance": covariance.astype(np.float32),
        "mrcp": mrcp_bins.astype(np.float32),
    }


def _extract(
    rows: _FeatureRows,
    branches: object,
    requests: Iterable[_WindowRequest],
    *,
    dataset: str,
    subject: str,
    phase: str,
) -> None:
    sfreq = float(branches.mrcp.info["sfreq"])
    for request in requests:
        try:
            windows = extract_branch_window(branches, request.onset_seconds, WINDOW_DURATION_SECONDS)
        except ValueError:
            continue
        features = window_features(windows, sfreq)
        if features is None:
            continue
        rows.kinds.append(request.kind)
        rows.phases.append(phase)
        rows.offsets.append(request.offset)
        rows.labels.append(request.label)
        rows.datasets.append(dataset)
        rows.subjects.append(subject)
        rows.recordings.append(request.recording)
        rows.trials.append(request.trial)
        for name, values in features.items():
            getattr(rows, name).append(values)


def _cue_requests(cue: float, label: int, trial: int, recording: int) -> list[_WindowRequest]:
    requests = [
        _WindowRequest(f"intent@{offset:.1f}", cue + offset, label, offset, trial, recording)
        for offset in INTENT_OFFSETS_SECONDS
    ]
    requests.append(
        _WindowRequest("rest_precue", cue - PRECUE_SECONDS, 0, -PRECUE_SECONDS, trial, recording)
    )
    return requests


def physionet_subject_rows(root: Path, subject: int, phases: Sequence[str] = PHASES) -> _FeatureRows:
    rows = _FeatureRows()
    key = f"physionet:S{subject:03d}"
    for run in IMAGERY_RUNS:
        path = find_run_file(root, int(subject), run)
        if path is None:
            raise FileNotFoundError(f"Missing S{subject:03d}R{run:02d}.edf")
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        recording = 0 if run in (4, 8) else 1
        requests: list[_WindowRequest] = []
        for index, annotation in enumerate(raw.annotations):
            label = map_task_annotation(str(annotation["description"]))
            onset = float(annotation["onset"])
            trial = run * 1000 + index
            if label == 0:
                requests.append(
                    _WindowRequest(
                        "rest_original",
                        onset + PHYSIONET_REST_OFFSET_SECONDS,
                        0,
                        PHYSIONET_REST_OFFSET_SECONDS,
                        trial,
                        recording,
                    )
                )
            elif label in (1, 2):
                requests.extend(_cue_requests(onset, label, trial, recording))
        for phase in phases:
            branches = preprocess_native_branches(raw, BranchPreprocessingProfile(phase=phase))
            _extract(rows, branches, requests, dataset="physionet", subject=key, phase=phase)
            del branches
    return rows


def openbmi_subject_rows(root: Path, subject: int, phases: Sequence[str] = PHASES) -> _FeatureRows:
    rows = _FeatureRows()
    key = f"openbmi:S{subject:03d}"
    recordings = [value for value in discover_openbmi_recordings(root) if value.subject == int(subject)]
    if not recordings:
        raise FileNotFoundError(f"No complete OpenBMI recordings for subject {subject}.")
    for recording in recordings:
        raw = read_openbmi_raw(recording, preload=False)
        role = 0 if recording.pipeline_run == 4 else 1
        requests: list[_WindowRequest] = []
        for event in load_openbmi_events(recording):
            cue = float(event["onset"])
            label = LABEL_IDS[str(event["trial_type"])]
            trial = recording.session * 100000 + recording.native_run * 10000 + int(event["trial_index"])
            requests.extend(_cue_requests(cue, label, trial, role))
            requests.append(
                _WindowRequest(
                    "rest_original",
                    cue + OPENBMI_REST_OFFSET_SECONDS,
                    0,
                    OPENBMI_REST_OFFSET_SECONDS,
                    trial,
                    role,
                )
            )
        for phase in phases:
            branches = preprocess_native_branches(raw, BranchPreprocessingProfile(phase=phase))
            _extract(rows, branches, requests, dataset="openbmi", subject=key, phase=phase)
            del branches
        del raw
    return rows


def liu_subject_rows(root: Path, subject: int, phases: Sequence[str] = PHASES) -> _FeatureRows:
    rows = _FeatureRows()
    key = f"liu2024:S{subject:03d}"
    participants = load_participants(root)
    participant = participants[f"sub-{subject:02d}"]
    raw = read_liu_raw(root, int(subject), preload=False)
    trials = build_subject_trials(root, int(subject), raw=raw, participant=participant)
    if not trials:
        raise FileNotFoundError(f"{key} contains no usable trials.")
    requests: list[_WindowRequest] = []
    for trial in trials:
        recording = int(trial.trial_index >= 20)
        cue = float(trial.intent_start_s)  # motor-imagery stage onset
        requests.extend(_cue_requests(cue, int(trial.native_label), trial.trial_index, recording))
        requests.append(
            _WindowRequest(
                "rest_original",
                float(trial.rest_start_s),
                0,
                float(trial.rest_start_s) - cue,
                trial.trial_index,
                recording,
            )
        )
    for phase in phases:
        branches = preprocess_native_branches(raw, BranchPreprocessingProfile(phase=phase))
        _extract(rows, branches, requests, dataset="liu2024", subject=key, phase=phase)
        del branches
    return rows


def subject_rows(dataset: str, root: Path, subject: int, phases: Sequence[str] = PHASES) -> tuple[_FeatureRows, str | None]:
    """Build one subject; return empty rows and the reason if it is excluded."""
    builders = {
        "physionet": physionet_subject_rows,
        "openbmi": openbmi_subject_rows,
        "liu2024": liu_subject_rows,
    }
    try:
        return builders[dataset](root, subject, phases), None
    except (FileNotFoundError, OSError, KeyError, ValueError, RecordingQualityError) as error:
        return _FeatureRows(), f"{dataset}:S{subject:03d}: {type(error).__name__}: {error}"


def rows_to_arrays(rows: _FeatureRows) -> dict[str, np.ndarray]:
    return {
        "kinds": np.asarray(rows.kinds),
        "phases": np.asarray(rows.phases),
        "offsets": np.asarray(rows.offsets, dtype=np.float32),
        "labels": np.asarray(rows.labels, dtype=np.int8),
        "datasets": np.asarray(rows.datasets),
        "subjects": np.asarray(rows.subjects),
        "recordings": np.asarray(rows.recordings, dtype=np.int8),
        "trials": np.asarray(rows.trials, dtype=np.int64),
        "amplitude": np.stack(rows.amplitude),
        "auxiliary": np.stack(rows.auxiliary),
        "bandpower": np.stack(rows.bandpower),
        "covariance": np.stack(rows.covariance),
        "mrcp": np.stack(rows.mrcp),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

ORIGINAL_INTENT_KIND = {"physionet": "intent@0.5", "openbmi": "intent@0.5", "liu2024": "intent@1.0"}
LATE_INTENT_KIND = {"physionet": "intent@1.5", "openbmi": "intent@1.5", "liu2024": "intent@2.0"}

CONDITIONS: dict[str, dict[str, object]] = {
    "original": {
        "phase": "zero",
        "positive": ORIGINAL_INTENT_KIND,
        "negative": "rest_original",
        "description": "V7 windows: cue-period intent vs post-task rest, zero-phase filters",
    },
    "original_causal": {
        "phase": "forward",
        "positive": ORIGINAL_INTENT_KIND,
        "negative": "rest_original",
        "description": "V7 windows with causal (forward) filters",
    },
    "precue_rest": {
        "phase": "zero",
        "positive": ORIGINAL_INTENT_KIND,
        "negative": "rest_precue",
        "description": "Same intent windows; rest taken from the 2 s before the cue",
    },
    "precue_rest_late_intent": {
        "phase": "zero",
        "positive": LATE_INTENT_KIND,
        "negative": "rest_precue",
        "description": "Pre-cue rest; intent window moved 1 s later to avoid cue-evoked responses",
    },
    "precue_rest_late_intent_causal": {
        "phase": "forward",
        "positive": LATE_INTENT_KIND,
        "negative": "rest_precue",
        "description": "Strictest: pre-cue rest, late intent window, causal filters",
    },
    "rest_vs_rest_control": {
        "phase": "zero",
        "positive": {"physionet": "rest_original", "openbmi": "rest_original", "liu2024": "rest_original"},
        "negative": "rest_precue",
        "description": (
            "Negative control: post-task rest (scored as 'positive') vs pre-cue rest. "
            "Neither contains motor imagery; separability measures trial-phase signal."
        ),
    },
}

MODELS = (
    "amplitude",
    "auxiliary_eog_emg",
    "bandpower",
    "bandpower_baseline_normalized",
    "riemann_recentered",
    "mrcp",
    "combined",
)


def condition_selection(arrays: Mapping[str, np.ndarray], condition: str, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (selected rows, binary target) for one dataset under one condition."""
    spec = CONDITIONS[condition]
    base = (arrays["datasets"] == dataset) & (arrays["phases"] == spec["phase"])
    positive = base & (arrays["kinds"] == spec["positive"][dataset])
    negative = base & (arrays["kinds"] == spec["negative"])
    selected = positive | negative
    return selected, positive[selected].astype(int)


def _subject_baselines(arrays: Mapping[str, np.ndarray], phase: str) -> dict[str, dict[str, np.ndarray]]:
    """Label-free per-subject references from recording-0 pre-cue windows."""
    from pyriemann.utils.mean import mean_riemann

    references: dict[str, dict[str, np.ndarray]] = {}
    base = (arrays["phases"] == phase) & (arrays["recordings"] == 0) & (arrays["kinds"] == "rest_precue")
    for subject in np.unique(arrays["subjects"][arrays["phases"] == phase]):
        rows = base & (arrays["subjects"] == subject)
        if not np.any(rows):
            rows = (arrays["phases"] == phase) & (arrays["recordings"] == 0) & (arrays["subjects"] == subject)
        covariances = arrays["covariance"][rows].astype(np.float64)
        references[str(subject)] = {
            "bandpower": arrays["bandpower"][rows].mean(axis=0),
            "covariance": mean_riemann(covariances, maxiter=20),
        }
    return references


def model_features(
    arrays: Mapping[str, np.ndarray],
    rows: np.ndarray,
    model: str,
    baselines: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.tangentspace import tangent_space

    subjects = arrays["subjects"][rows]
    if model == "amplitude":
        return arrays["amplitude"][rows].astype(np.float64)
    if model == "auxiliary_eog_emg":
        return arrays["auxiliary"][rows][:, 1:].astype(np.float64)
    if model == "bandpower":
        return arrays["bandpower"][rows].astype(np.float64)
    if model == "bandpower_baseline_normalized":
        reference = np.stack([baselines[str(s)]["bandpower"] for s in subjects])
        return arrays["bandpower"][rows].astype(np.float64) - reference
    if model == "riemann_recentered":
        covariances = arrays["covariance"][rows].astype(np.float64)
        output = np.empty((len(covariances), covariances.shape[1] * (covariances.shape[1] + 1) // 2))
        for subject in np.unique(subjects):
            local = subjects == subject
            whitening = invsqrtm(baselines[str(subject)]["covariance"])
            aligned = whitening @ covariances[local] @ whitening
            output[local] = tangent_space(aligned, np.eye(aligned.shape[1]), metric="riemann")
        return output
    if model == "mrcp":
        return arrays["mrcp"][rows].astype(np.float64)
    if model == "combined":
        return np.concatenate(
            [
                model_features(arrays, rows, "bandpower_baseline_normalized", baselines),
                model_features(arrays, rows, "riemann_recentered", baselines),
                model_features(arrays, rows, "mrcp", baselines),
            ],
            axis=1,
        )
    raise ValueError(f"Unknown model {model}")


def _classifier(regularization: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=regularization, class_weight="balanced", max_iter=4000),
    )


def _participant_auroc(target: np.ndarray, scores: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    values = {}
    for subject in np.unique(subjects):
        local = subjects == subject
        if len(np.unique(target[local])) == 2:
            values[str(subject)] = float(roc_auc_score(target[local], scores[local]))
    return values


def _mean_interval(values: Sequence[float], rng: np.random.Generator, iterations: int) -> list[float]:
    data = np.asarray(list(values), dtype=np.float64)
    means = [np.mean(data[rng.integers(0, len(data), len(data))]) for _ in range(iterations)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate(
    arrays: Mapping[str, np.ndarray],
    roles: Mapping[str, str],
    condition: str,
    model: str,
    dataset: str,
    baselines: Mapping[str, Mapping[str, np.ndarray]],
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, object] | None:
    """Population model: train on development + controller participants, score
    development_observed participants on later recordings (the V7 evaluation rows)."""
    from sklearn.metrics import roc_auc_score

    from .translational_metrics import participant_level_summary

    if model == "auxiliary_eog_emg" and dataset == "physionet":
        return None
    selected, target = condition_selection(arrays, condition, dataset)
    indices = np.flatnonzero(selected)
    subject_role = np.asarray([roles.get(str(s), "none") for s in arrays["subjects"][indices]])
    train = np.isin(subject_role, ["development", "controller_validation"])
    test = (subject_role == "development_observed") & (arrays["recordings"][indices] != 0)
    if not np.any(train) or not np.any(test):
        return None
    features = model_features(arrays, indices, model, baselines)
    classifier = _classifier().fit(features[train], target[train])
    scores = classifier.predict_proba(features[test])[:, 1]
    subjects = arrays["subjects"][indices][test]
    per_subject = _participant_auroc(target[test], scores, subjects)
    rng = np.random.default_rng(seed)
    summary = participant_level_summary(
        target[test], scores, np.ones(len(scores), dtype=bool), 0.5,
        subjects, np.full(len(scores), dataset), iterations=iterations, seed=seed,
    )
    return {
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "train_rows": int(np.sum(train)),
        "test_rows": int(np.sum(test)),
        "participants": len(per_subject),
        "participant_mean_auroc": float(np.mean(list(per_subject.values()))),
        "participant_mean_auroc_95_ci": _mean_interval(per_subject.values(), rng, iterations),
        "pooled_auroc": float(roc_auc_score(target[test], scores)),
        "participant_mean_balanced_accuracy": summary["mean_balanced_accuracy"],
        "participants_above_chance": summary["participants_above_chance"],
        "per_participant_auroc": per_subject,
    }


def evaluate_within_subject(
    arrays: Mapping[str, np.ndarray],
    roles: Mapping[str, str],
    condition: str,
    model: str,
    dataset: str,
    baselines: Mapping[str, Mapping[str, np.ndarray]],
    *,
    iterations: int = 1000,
    seed: int = 0,
) -> dict[str, object] | None:
    """Patient calibration analogue: fit on a participant's recording 0, score later recordings."""
    if model == "auxiliary_eog_emg" and dataset == "physionet":
        return None
    selected, target = condition_selection(arrays, condition, dataset)
    indices = np.flatnonzero(selected)
    subjects = arrays["subjects"][indices]
    recordings = arrays["recordings"][indices]
    features = model_features(arrays, indices, model, baselines)
    per_subject: dict[str, float] = {}
    from sklearn.metrics import roc_auc_score

    for subject in np.unique(subjects):
        if roles.get(str(subject)) != "development_observed":
            continue
        calibration = (subjects == subject) & (recordings == 0)
        evaluation = (subjects == subject) & (recordings != 0)
        if len(np.unique(target[calibration])) < 2 or len(np.unique(target[evaluation])) < 2:
            continue
        classifier = _classifier(regularization=0.1).fit(features[calibration], target[calibration])
        scores = classifier.predict_proba(features[evaluation])[:, 1]
        per_subject[str(subject)] = float(roc_auc_score(target[evaluation], scores))
    if not per_subject:
        return None
    rng = np.random.default_rng(seed)
    return {
        "condition": condition,
        "model": model,
        "dataset": dataset,
        "participants": len(per_subject),
        "participant_mean_auroc": float(np.mean(list(per_subject.values()))),
        "participant_mean_auroc_95_ci": _mean_interval(per_subject.values(), rng, iterations),
        "per_participant_auroc": per_subject,
    }


def offset_sweep(
    arrays: Mapping[str, np.ndarray],
    roles: Mapping[str, str],
    dataset: str,
    model: str,
    negative: str,
    baselines: Mapping[str, Mapping[str, np.ndarray]],
    *,
    phase: str = "zero",
    seed: int = 0,
) -> list[dict[str, float]]:
    """Participant-mean AUROC as the intent window moves later after the cue."""
    results = []
    for offset in INTENT_OFFSETS_SECONDS:
        name = f"_sweep_{offset}"
        CONDITIONS[name] = {
            "phase": phase,
            "positive": {key: f"intent@{offset:.1f}" for key in ORIGINAL_INTENT_KIND},
            "negative": negative,
            "description": "offset sweep",
        }
        try:
            result = evaluate(arrays, roles, name, model, dataset, baselines, iterations=200, seed=seed)
        finally:
            del CONDITIONS[name]
        if result is not None:
            results.append(
                {
                    "offset_seconds": float(offset),
                    "participant_mean_auroc": result["participant_mean_auroc"],
                    "ci": result["participant_mean_auroc_95_ci"],
                }
            )
    return results
