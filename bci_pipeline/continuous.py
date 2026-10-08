"""Continuous-window evidence accumulation and command-event metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EvidencePolicy:
    """Patient-calibrated fusion and hysteretic command-controller settings."""

    erd_weight: float
    local_weight: float
    rest_quantile: float
    decay: float
    activation_threshold: float
    release_ratio: float
    max_hold_seconds: float
    cooldown_seconds: float


@dataclass
class ContinuousStream:
    """Chronological windows and decoder scores for one subject."""

    subject: int
    runs: np.ndarray
    times: np.ndarray
    labels: np.ndarray
    segments: np.ndarray
    segment_onsets: np.ndarray
    erd_global_intent: np.ndarray
    erd_local_intent: np.ndarray
    riemann_global_intent: np.ndarray
    riemann_local_intent: np.ndarray
    erd_global_direction: np.ndarray
    erd_local_direction: np.ndarray
    riemann_global_direction: np.ndarray
    riemann_local_direction: np.ndarray
    artifact_rejected: np.ndarray
    calibration_mask: np.ndarray
    step_seconds: float


def fuse_probabilities(
    global_values: np.ndarray,
    local_values: np.ndarray,
    local_weight: float,
) -> np.ndarray:
    return (1.0 - local_weight) * global_values + local_weight * local_values


def stream_probabilities(
    stream: ContinuousStream,
    policy: EvidencePolicy,
) -> Tuple[np.ndarray, np.ndarray, float]:
    erd_intent = fuse_probabilities(
        stream.erd_global_intent, stream.erd_local_intent, policy.local_weight
    )
    riemann_intent = fuse_probabilities(
        stream.riemann_global_intent,
        stream.riemann_local_intent,
        policy.local_weight,
    )
    intent = policy.erd_weight * erd_intent + (1.0 - policy.erd_weight) * riemann_intent

    erd_direction = fuse_probabilities(
        stream.erd_global_direction,
        stream.erd_local_direction,
        policy.local_weight,
    )
    riemann_direction = fuse_probabilities(
        stream.riemann_global_direction,
        stream.riemann_local_direction,
        policy.local_weight,
    )
    direction = (
        policy.erd_weight * erd_direction
        + (1.0 - policy.erd_weight) * riemann_direction
    )
    calibration_rest = (
        stream.calibration_mask
        & (stream.labels == 0)
        & ~stream.artifact_rejected
    )
    if not np.any(calibration_rest):
        raise ValueError(f"Subject {stream.subject} has no clean calibration-rest windows.")
    rest_reference = float(
        np.quantile(intent[calibration_rest], policy.rest_quantile, method="higher")
    )
    return intent, direction, rest_reference


def simulate_stream(
    stream: ContinuousStream,
    policy: EvidencePolicy,
    artifact_abstention: bool = True,
) -> Dict[str, object]:
    """Run a causal leaky accumulator and emit command-onset events."""

    intent, direction, rest_reference = stream_probabilities(stream, policy)
    rest_reference = float(np.clip(rest_reference, 1e-4, 1.0 - 1e-4))
    reference_log_odds = math.log(rest_reference / (1.0 - rest_reference))
    evidence = 0.0
    active = False
    active_since = -np.inf
    cooldown_until = -np.inf
    current_run = None
    events: List[Dict[str, object]] = []
    evidence_trace = np.zeros(len(stream.labels), dtype=np.float32)
    active_trace = np.zeros(len(stream.labels), dtype=bool)
    release_threshold = policy.activation_threshold * policy.release_ratio

    for index in range(len(stream.labels)):
        run = int(stream.runs[index])
        time_value = float(stream.times[index])
        if current_run != run:
            current_run = run
            evidence = 0.0
            active = False
            active_since = -np.inf
            cooldown_until = -np.inf

        rejected = bool(stream.artifact_rejected[index]) and artifact_abstention
        if rejected:
            evidence *= policy.decay
        else:
            probability = float(np.clip(intent[index], 1e-4, 1.0 - 1e-4))
            log_odds = math.log(probability / (1.0 - probability))
            increment = float(np.clip(log_odds - reference_log_odds, -2.0, 2.0))
            evidence = max(0.0, policy.decay * evidence + increment)

        if active and (
            evidence <= release_threshold
            or time_value - active_since >= policy.max_hold_seconds
        ):
            active = False
            cooldown_until = time_value + policy.cooldown_seconds

        if (
            not active
            and not rejected
            and time_value >= cooldown_until
            and evidence >= policy.activation_threshold
        ):
            active = True
            active_since = time_value
            prediction = 1 if direction[index] >= 0.5 else 2
            events.append(
                {
                    "subject": int(stream.subject),
                    "run": run,
                    "time_seconds": time_value,
                    "label": int(stream.labels[index]),
                    "prediction": prediction,
                    "segment": int(stream.segments[index]),
                    "segment_onset_seconds": float(stream.segment_onsets[index]),
                    "intent_probability": float(intent[index]),
                    "evidence": float(evidence),
                    "artifact_rejected": rejected,
                }
            )

        evidence_trace[index] = evidence
        active_trace[index] = active

    return {
        "policy": asdict(policy),
        "rest_reference_probability": rest_reference,
        "events": events,
        "intent_probability": intent,
        "direction_probability": direction,
        "evidence_trace": evidence_trace,
        "active_trace": active_trace,
    }


def command_metrics(
    stream: ContinuousStream,
    simulation: Mapping[str, object],
    evaluation_run: int = 12,
) -> Dict[str, object]:
    """Score command events per rest minute and detections per intent segment."""

    evaluation = stream.runs == evaluation_run
    labels = stream.labels[evaluation]
    segments = stream.segments[evaluation]
    runs = stream.runs[evaluation]
    rejected = stream.artifact_rejected[evaluation]
    if not len(labels):
        raise ValueError(f"Subject {stream.subject} has no Run {evaluation_run} windows.")

    events = [
        event for event in simulation["events"] if int(event["run"]) == evaluation_run
    ]
    rest_segments = {
        (int(run), int(segment))
        for run, segment, label in zip(runs, segments, labels)
        if label == 0
    }
    intent_segments = {
        (int(run), int(segment)): int(label)
        for run, segment, label in zip(runs, segments, labels)
        if label != 0
    }
    false_segments = {
        (int(event["run"]), int(event["segment"]))
        for event in events
        if int(event["label"]) == 0
    }
    first_true_event: Dict[Tuple[int, int], Mapping[str, object]] = {}
    for event in events:
        key = (int(event["run"]), int(event["segment"]))
        if int(event["label"]) != 0 and key not in first_true_event:
            first_true_event[key] = event

    detected = set(first_true_event) & set(intent_segments)
    latencies = [
        float(first_true_event[key]["time_seconds"])
        - float(first_true_event[key]["segment_onset_seconds"])
        for key in detected
    ]
    wrong_direction = sum(
        int(first_true_event[key]["prediction"]) != intent_segments[key]
        for key in detected
    )
    false_events = sum(int(event["label"]) == 0 for event in events)
    rest_minutes = max(float(np.sum(labels == 0) * stream.step_seconds / 60.0), 1e-9)
    rest_specificity = 1.0 - len(false_segments) / max(len(rest_segments), 1)
    sensitivity = len(detected) / max(len(intent_segments), 1)
    return {
        "subject": int(stream.subject),
        "evaluation_run": int(evaluation_run),
        "rest_minutes": rest_minutes,
        "rest_segments": int(len(rest_segments)),
        "intent_segments": int(len(intent_segments)),
        "command_events": int(len(events)),
        "false_activation_events": int(false_events),
        "false_activations_per_minute": float(false_events / rest_minutes),
        "rest_segment_false_activation_rate": float(1.0 - rest_specificity),
        "intent_sensitivity": float(sensitivity),
        "missed_intent_rate": float(1.0 - sensitivity),
        "segment_balanced_accuracy": float((rest_specificity + sensitivity) / 2.0),
        "wrong_direction_rate": float(wrong_direction / max(len(intent_segments), 1)),
        "wrong_direction_among_detected": float(wrong_direction / max(len(detected), 1)),
        "median_latency_seconds": float(np.median(latencies)) if latencies else None,
        "artifact_abstention_rate": float(np.mean(rejected)),
        "detected_intent_segments": int(len(detected)),
        "latencies_seconds": latencies,
    }


def aggregate_metrics(per_subject: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """Average patient-level rates so each held-out subject has equal weight."""

    keys = (
        "false_activations_per_minute",
        "rest_segment_false_activation_rate",
        "intent_sensitivity",
        "missed_intent_rate",
        "segment_balanced_accuracy",
        "wrong_direction_rate",
        "wrong_direction_among_detected",
        "artifact_abstention_rate",
    )
    result = {
        key: float(np.mean([float(row[key]) for row in per_subject.values()]))
        for key in keys
    }
    latencies = [
        float(value)
        for row in per_subject.values()
        for value in row["latencies_seconds"]
    ]
    result.update(
        {
            "subjects": int(len(per_subject)),
            "total_rest_minutes": float(
                np.sum([float(row["rest_minutes"]) for row in per_subject.values()])
            ),
            "total_false_activation_events": int(
                np.sum([int(row["false_activation_events"]) for row in per_subject.values()])
            ),
            "total_intent_segments": int(
                np.sum([int(row["intent_segments"]) for row in per_subject.values()])
            ),
            "total_detected_intent_segments": int(
                np.sum([int(row["detected_intent_segments"]) for row in per_subject.values()])
            ),
            "median_latency_seconds": float(np.median(latencies)) if latencies else None,
        }
    )
    return result

