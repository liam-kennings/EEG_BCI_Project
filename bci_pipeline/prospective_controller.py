"""Frozen-model provenance and cue-constrained prospective EEG control logic."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from .dataset import normalize_channel_name


PROSPECTIVE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a model or provenance file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "fingerprint"):
        return _json_safe(value.fingerprint())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CueControllerConfig:
    """Safety policy applied after the frozen EEG decoder and OOD gate."""

    intent_threshold: float = 0.61
    confirmation_windows: int = 2
    arming_seconds: float = 4.0
    refractory_seconds: float = 2.0
    minimum_direction_probability: float = 0.55
    reset_on_unavailable: bool = True

    def validate(self) -> None:
        if not 0.0 < self.intent_threshold < 1.0:
            raise ValueError("intent_threshold must be between zero and one.")
        if self.confirmation_windows < 1:
            raise ValueError("confirmation_windows must be positive.")
        if self.arming_seconds <= 0 or self.refractory_seconds < 0:
            raise ValueError("Controller timing values are invalid.")
        if not 0.5 <= self.minimum_direction_probability <= 1.0:
            raise ValueError("minimum_direction_probability must be at least 0.5.")


@dataclass(frozen=True)
class ControllerEvent:
    timestamp_seconds: float
    kind: str
    state: str
    cue_id: str = ""
    expected_label: int = -1
    decoded_label: int = -1
    intent_probability: float = 0.0
    direction_probability: float = 0.0
    latency_seconds: float = -1.0
    reason: str = ""


class CueConstrainedController:
    """Arm from a cue, but issue a command only from confirmed EEG evidence."""

    def __init__(self, config: CueControllerConfig = CueControllerConfig()) -> None:
        config.validate()
        self.config = config
        self.state = "idle"
        self.cue_id = ""
        self.expected_label = -1
        self.cue_onset = -np.inf
        self.armed_until = -np.inf
        self.refractory_until = -np.inf
        self.confirmations = 0
        self.confirmed_direction = -1
        self.events: list[ControllerEvent] = []

    def _emit(self, timestamp: float, kind: str, **values: Any) -> ControllerEvent:
        event = ControllerEvent(
            timestamp_seconds=float(timestamp),
            kind=kind,
            state=self.state,
            cue_id=self.cue_id,
            expected_label=self.expected_label,
            **values,
        )
        self.events.append(event)
        return event

    def _clear_evidence(self) -> None:
        self.confirmations = 0
        self.confirmed_direction = -1

    def arm(
        self,
        cue_id: str,
        timestamp_seconds: float,
        *,
        expected_label: int = -1,
    ) -> ControllerEvent:
        """Open one decision opportunity without supplying the decoded command."""
        timestamp = float(timestamp_seconds)
        if expected_label not in (-1, 0, 1, 2):
            raise ValueError("expected_label must be unknown, rest, left, or right.")
        if timestamp < self.refractory_until:
            return self._emit(
                timestamp,
                "cue_suppressed_refractory",
                reason="cue_arrived_before_refractory_period_ended",
            )
        if self.state == "armed":
            self._emit(
                timestamp,
                "no_decision",
                latency_seconds=max(0.0, timestamp - self.cue_onset),
                reason="new_cue_replaced_open_decision_window",
            )
        self.state = "armed"
        self.cue_id = str(cue_id)
        self.expected_label = int(expected_label)
        self.cue_onset = timestamp
        self.armed_until = timestamp + self.config.arming_seconds
        self._clear_evidence()
        return self._emit(timestamp, "armed")

    def _expire_if_needed(self, timestamp: float) -> ControllerEvent | None:
        if self.state == "refractory" and timestamp >= self.refractory_until:
            self.state = "idle"
            self.cue_id = ""
            self.expected_label = -1
            self._clear_evidence()
        if self.state == "armed" and timestamp > self.armed_until:
            event = self._emit(
                timestamp,
                "no_decision",
                latency_seconds=self.config.arming_seconds,
                reason="arming_window_expired_without_confirmed_eeg",
            )
            self.state = "idle"
            self.cue_id = ""
            self.expected_label = -1
            self._clear_evidence()
            return event
        return None

    def step(
        self,
        timestamp_seconds: float,
        intent_probability: float,
        direction_probabilities: Sequence[float],
        *,
        available: bool = True,
        contamination_veto: bool = False,
    ) -> tuple[ControllerEvent, ...]:
        """Consume one decoder update and return any state or command events."""
        timestamp = float(timestamp_seconds)
        intent = float(intent_probability)
        direction = np.asarray(direction_probabilities, dtype=np.float64)
        if not np.isfinite(intent) or direction.shape != (2,) or not np.all(
            np.isfinite(direction)
        ):
            raise ValueError("Decoder probabilities must be finite and two-class.")
        if not 0.0 <= intent <= 1.0 or np.any(direction < 0.0):
            raise ValueError("Decoder probabilities are outside their valid range.")
        direction_sum = float(np.sum(direction))
        if direction_sum <= 0:
            raise ValueError("Direction probabilities must have positive mass.")
        direction /= direction_sum

        emitted: list[ControllerEvent] = []
        expired = self._expire_if_needed(timestamp)
        if expired is not None:
            emitted.append(expired)
        active_candidate = intent >= self.config.intent_threshold
        if self.state != "armed":
            if active_candidate and available and not contamination_veto:
                emitted.append(
                    self._emit(
                        timestamp,
                        "unarmed_candidate",
                        decoded_label=int(np.argmax(direction)) + 1,
                        intent_probability=intent,
                        direction_probability=float(np.max(direction)),
                        reason="EEG evidence was suppressed because no cue was armed",
                    )
                )
            return tuple(emitted)

        if contamination_veto:
            self._clear_evidence()
            emitted.append(
                self._emit(
                    timestamp,
                    "contamination_veto",
                    intent_probability=intent,
                    reason="full_montage_artifact_rule",
                )
            )
            return tuple(emitted)
        if not available:
            if self.config.reset_on_unavailable:
                self._clear_evidence()
            emitted.append(
                self._emit(
                    timestamp,
                    "ood_abstention",
                    intent_probability=intent,
                    reason="frozen_quality_gate_unavailable",
                )
            )
            return tuple(emitted)

        decoded = int(np.argmax(direction)) + 1
        confidence = float(np.max(direction))
        accepted = (
            active_candidate
            and confidence >= self.config.minimum_direction_probability
        )
        if not accepted:
            self._clear_evidence()
            return tuple(emitted)
        if decoded == self.confirmed_direction:
            self.confirmations += 1
        else:
            self.confirmed_direction = decoded
            self.confirmations = 1
        if self.confirmations < self.config.confirmation_windows:
            emitted.append(
                self._emit(
                    timestamp,
                    "evidence_accumulating",
                    decoded_label=decoded,
                    intent_probability=intent,
                    direction_probability=confidence,
                )
            )
            return tuple(emitted)

        latency = timestamp - self.cue_onset
        command = self._emit(
            timestamp,
            "confirmed_command",
            decoded_label=decoded,
            intent_probability=intent,
            direction_probability=confidence,
            latency_seconds=latency,
        )
        emitted.append(command)
        self.state = "refractory"
        self.refractory_until = timestamp + self.config.refractory_seconds
        self._clear_evidence()
        return tuple(emitted)

    def finalize(self, timestamp_seconds: float) -> tuple[ControllerEvent, ...]:
        """Close an unfinished session while preserving no-decision accounting."""
        if self.state != "armed":
            return ()
        event = self._emit(
            float(timestamp_seconds),
            "no_decision",
            latency_seconds=max(0.0, float(timestamp_seconds) - self.cue_onset),
            reason="session_ended_before_confirmed_eeg",
        )
        self.state = "idle"
        self.cue_id = ""
        self.expected_label = -1
        self._clear_evidence()
        return (event,)


class FullMontageContaminationVeto:
    """Robust EOG/EMG-like burst veto fitted only on calibration/rest windows."""

    feature_names = (
        "global_peak_to_peak_uv",
        "global_derivative_rms_uv_per_s",
        "frontal_delta_fraction",
        "peripheral_high_frequency_fraction",
        "channel_amplitude_outlier_fraction",
    )

    def __init__(self, robust_z: float = 8.0, quantile: float = 0.995) -> None:
        if robust_z <= 0 or not 0.5 < quantile < 1.0:
            raise ValueError("Invalid contamination-veto thresholds.")
        self.robust_z = float(robust_z)
        self.quantile = float(quantile)

    @staticmethod
    def features(
        windows: np.ndarray,
        sfreq: float,
        channel_names: Sequence[str],
    ) -> np.ndarray:
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != len(channel_names):
            raise ValueError("Full-montage windows must be trials x channels x samples.")
        if sfreq <= 0:
            raise ValueError("Sampling frequency must be positive.")
        uv = values * 1e6
        peak_to_peak = np.ptp(uv, axis=-1)
        global_peak = np.quantile(peak_to_peak, 0.95, axis=1)
        derivative_rms = np.sqrt(np.mean(np.diff(uv, axis=-1) ** 2, axis=(1, 2))) * sfreq

        normalized = [normalize_channel_name(name) for name in channel_names]
        frontal = np.asarray(
            [name.startswith(("FP", "AF", "F")) and not name.startswith("FC") for name in normalized]
        )
        peripheral = np.asarray(
            [name.startswith(("FP", "AF", "F", "T", "FT", "TP")) for name in normalized]
        )
        if not np.any(frontal):
            frontal = np.ones(len(normalized), dtype=bool)
        if not np.any(peripheral):
            peripheral = np.ones(len(normalized), dtype=bool)

        frequencies, spectrum = welch(
            values,
            fs=sfreq,
            nperseg=min(values.shape[-1], max(32, int(round(2.0 * sfreq)))),
            axis=-1,
        )

        def band_power(low: float, high: float) -> np.ndarray:
            selected = (frequencies >= low) & (frequencies < high)
            if np.sum(selected) < 2:
                return np.zeros(spectrum.shape[:2], dtype=np.float64)
            return np.trapezoid(spectrum[..., selected], frequencies[selected], axis=-1)

        delta = band_power(0.5, 4.0)
        broad = band_power(0.5, min(30.0, sfreq * 0.45))
        high = band_power(30.0, min(45.0, sfreq * 0.45))
        lower = band_power(8.0, min(30.0, sfreq * 0.45))
        frontal_delta = np.median(
            delta[:, frontal] / np.clip(broad[:, frontal], 1e-30, None),
            axis=1,
        )
        peripheral_high = np.median(
            high[:, peripheral] / np.clip(lower[:, peripheral], 1e-30, None),
            axis=1,
        )
        channel_median = np.median(peak_to_peak, axis=1, keepdims=True)
        channel_mad = 1.4826 * np.median(
            np.abs(peak_to_peak - channel_median), axis=1, keepdims=True
        )
        channel_outliers = np.mean(
            peak_to_peak > channel_median + 6.0 * np.clip(channel_mad, 1e-6, None),
            axis=1,
        )
        result = np.column_stack(
            (global_peak, derivative_rms, frontal_delta, peripheral_high, channel_outliers)
        )
        return np.nan_to_num(result, nan=0.0, posinf=1e12, neginf=0.0)

    def fit(
        self,
        windows: np.ndarray,
        sfreq: float,
        channel_names: Sequence[str],
    ) -> "FullMontageContaminationVeto":
        values = self.features(windows, sfreq, channel_names)
        if len(values) < 10:
            raise ValueError("At least ten clean calibration windows are required.")
        self.median_ = np.median(values, axis=0)
        self.scale_ = 1.4826 * np.median(
            np.abs(values - self.median_), axis=0
        )
        self.scale_ = np.clip(self.scale_, 1e-9, None)
        robust_limit = self.median_ + self.robust_z * self.scale_
        empirical_limit = np.quantile(values, self.quantile, axis=0)
        self.thresholds_ = np.maximum(robust_limit, empirical_limit)
        return self

    def predict(
        self,
        windows: np.ndarray,
        sfreq: float,
        channel_names: Sequence[str],
    ) -> np.ndarray:
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("The contamination veto must be fitted first.")
        values = self.features(windows, sfreq, channel_names)
        return np.any(values > self.thresholds_, axis=1)


def controller_metrics(
    events: Sequence[ControllerEvent],
    *,
    session_duration_seconds: float,
) -> dict[str, float | int | str]:
    """Score command opportunities without hiding abstention or unarmed activity."""
    commands = [event for event in events if event.kind == "confirmed_command"]
    no_decisions = [event for event in events if event.kind == "no_decision"]
    armed = [event for event in events if event.kind == "armed"]
    unarmed = [event for event in events if event.kind == "unarmed_candidate"]
    motor_cues = [event for event in armed if event.expected_label in (1, 2)]
    rest_cues = [event for event in armed if event.expected_label == 0]
    command_by_cue = {event.cue_id: event for event in commands}
    correct_motor = sum(
        cue.cue_id in command_by_cue
        and command_by_cue[cue.cue_id].decoded_label == cue.expected_label
        for cue in motor_cues
    )
    wrong_motor = sum(
        cue.cue_id in command_by_cue
        and command_by_cue[cue.cue_id].decoded_label != cue.expected_label
        for cue in motor_cues
    )
    false_rest = sum(cue.cue_id in command_by_cue for cue in rest_cues)
    motor_no_decisions = sum(
        cue.cue_id not in command_by_cue for cue in motor_cues
    )
    rest_correct_rejections = sum(
        cue.cue_id not in command_by_cue for cue in rest_cues
    )
    latencies = [
        command_by_cue[cue.cue_id].latency_seconds
        for cue in motor_cues
        if cue.cue_id in command_by_cue
    ]
    minutes = max(float(session_duration_seconds) / 60.0, 1e-12)
    return {
        "command_opportunities": len(armed),
        "motor_cues": len(motor_cues),
        "rest_or_no_go_cues": len(rest_cues),
        "issued_commands": len(commands),
        "correct_motor_commands": int(correct_motor),
        "wrong_direction_commands": int(wrong_motor),
        "false_commands_on_rest_or_no_go": int(false_rest),
        "correct_rest_or_no_go_rejections": int(rest_correct_rejections),
        "motor_command_sensitivity": (
            float(correct_motor / len(motor_cues)) if motor_cues else "not_evaluable"
        ),
        "rest_cue_specificity": (
            float(1.0 - false_rest / len(rest_cues)) if rest_cues else "not_evaluable"
        ),
        "no_decisions": len(no_decisions),
        "motor_no_decisions": int(motor_no_decisions),
        "motor_no_decision_fraction": (
            float(motor_no_decisions / len(motor_cues))
            if motor_cues
            else "not_evaluable"
        ),
        "no_decision_fraction": (
            float(len(no_decisions) / len(armed)) if armed else "not_evaluable"
        ),
        "median_command_latency_seconds": (
            float(np.median(latencies)) if latencies else "not_evaluable"
        ),
        "p95_command_latency_seconds": (
            float(np.quantile(latencies, 0.95)) if latencies else "not_evaluable"
        ),
        "issued_false_commands_per_minute": float(false_rest / minutes),
        "latent_unarmed_candidates": len(unarmed),
        "latent_unarmed_candidates_per_minute": float(len(unarmed) / minutes),
        "ood_abstentions": sum(event.kind == "ood_abstention" for event in events),
        "contamination_vetoes": sum(
            event.kind == "contamination_veto" for event in events
        ),
        "session_minutes": minutes,
    }


def freeze_v4_bundle(
    model_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy and fingerprint the selected v4 model and all neural seed weights."""
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = joblib.load(source)
    required = {
        "models",
        "quality_gate",
        "stage_specific_fusion",
        "probability_calibrator",
        "calibrated_thresholds",
        "subject_roles",
        "dataset_fingerprint",
        "profile",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Model bundle is missing required fields: {sorted(missing)}")
    destination = Path(output_dir).expanduser().resolve() / "frozen_model"
    destination.mkdir(parents=True, exist_ok=True)
    frozen_bundle = destination / "translational_v4_neural_models.joblib"
    shutil.copy2(source, frozen_bundle)

    neural_records = []
    for record in payload.get("neural_integration", {}).get("models", []):
        neural_source = Path(str(record["path"])).expanduser().resolve()
        if not neural_source.is_file():
            raise FileNotFoundError(neural_source)
        neural_destination = destination / f"mrcp_fbcnet_seed_{int(record['seed'])}.pt"
        shutil.copy2(neural_source, neural_destination)
        neural_records.append(
            {
                "seed": int(record["seed"]),
                "path": str(neural_destination),
                "sha256": sha256_file(neural_destination),
                "bytes": neural_destination.stat().st_size,
            }
        )
    manifest = {
        "schema_version": PROSPECTIVE_SCHEMA_VERSION,
        "status": "frozen_for_future_locked_external_evaluation",
        "claim_status": "development_observed_not_externally_confirmed",
        "source_model": str(source),
        "frozen_model": str(frozen_bundle),
        "model_sha256": sha256_file(frozen_bundle),
        "model_bytes": frozen_bundle.stat().st_size,
        "dataset_fingerprint": str(payload["dataset_fingerprint"]),
        "preprocessing_profile": _json_safe(payload["profile"]),
        "stage_specific_fusion": _json_safe(payload["stage_specific_fusion"].__dict__),
        "calibrated_thresholds": _json_safe(payload["calibrated_thresholds"]),
        "patient_calibration_thresholds": _json_safe(
            payload.get("patient_calibration_thresholds", {})
        ),
        "neural_models": neural_records,
        "locked_external_subjects": [],
        "tamper_check": (
            "Recompute every SHA-256 before external inference. Any mismatch invalidates "
            "the frozen evaluation contract."
        ),
    }
    manifest_path = destination / "frozen_model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    return manifest


def verify_frozen_bundle(manifest_path: str | Path) -> bool:
    """Raise on any frozen model hash mismatch."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="ascii"))
    if sha256_file(manifest["frozen_model"]) != manifest["model_sha256"]:
        raise RuntimeError("Frozen v4 model hash mismatch.")
    for record in manifest.get("neural_models", []):
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError(
                f"Frozen neural model hash mismatch for seed {record['seed']}."
            )
    return True


def _demo_session(config: CueControllerConfig) -> tuple[list[dict[str, Any]], list[ControllerEvent]]:
    controller = CueConstrainedController(config)
    times = np.arange(0.0, 42.5, 0.5)
    cues = {5.0: ("left_01", 1), 15.0: ("right_01", 2), 25.0: ("nogo_01", 0), 35.0: ("left_missed", 1)}
    rows = []
    for timestamp in times:
        if timestamp in cues:
            cue_id, expected = cues[timestamp]
            controller.arm(cue_id, timestamp, expected_label=expected)
        intent = 0.12
        direction = (0.52, 0.48)
        veto = False
        if timestamp in (6.0, 6.5):
            intent, direction = 0.82, (0.78, 0.22)
        if timestamp in (11.0, 11.5):
            intent, direction = 0.86, (0.80, 0.20)
        if timestamp == 16.0:
            intent, direction, veto = 0.95, (0.15, 0.85), True
        if timestamp in (17.0, 17.5):
            intent, direction = 0.84, (0.20, 0.80)
        if timestamp == 26.0:
            intent, direction = 0.85, (0.75, 0.25)
        controller.step(
            timestamp,
            intent,
            direction,
            available=True,
            contamination_veto=veto,
        )
        rows.append(
            {
                "timestamp_seconds": float(timestamp),
                "intent_probability": intent,
                "left_probability": direction[0],
                "right_probability": direction[1],
                "contamination_veto": int(veto),
            }
        )
    controller.finalize(float(times[-1]))
    return rows, controller.events


def prepare_prospective_package(
    output_dir: str | Path,
    model_path: str | Path,
    *,
    acquisition: Mapping[str, Any] | None = None,
    config: CueControllerConfig | None = None,
) -> dict[str, Path]:
    """Create the frozen controller, external-test protocol, templates, and demo."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = freeze_v4_bundle(model_path, destination)
    thresholds = manifest["calibrated_thresholds"]
    fallback = float(thresholds.get("conservative_fallback", 0.61))
    selected = config or CueControllerConfig(intent_threshold=fallback)
    selected.validate()

    config_path = destination / "prospective_controller_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": PROSPECTIVE_SCHEMA_VERSION,
                "controller": asdict(selected),
                "cue_role": "arming_only_not_a_command_or_direction_label_at_inference",
                "decoder_role": "EEG supplies intent and left/right command",
                "quality_layers": [
                    "frozen_v4_OOD_availability_gate",
                    "prospective_full_montage_EOG_EMG_contamination_veto",
                ],
                "device_output_enabled": False,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    protocol = {
        "schema_version": PROSPECTIVE_SCHEMA_VERSION,
        "status": "prepared_not_collected",
        "model_manifest": str(
            destination / "frozen_model" / "frozen_model_manifest.json"
        ),
        "acquisition": _json_safe(acquisition or {}),
        "required_blocks": [
            "short_labeled_calibration_with_left_right_rest_and_no_go",
            "locked_later_cued_evaluation_block",
            "continuous_natural_idle_no_control_block",
            "artifact_challenge_block_blinks_jaw_swallow_posture",
            "periodic_sham_arming_opportunities_during_no_control",
        ],
        "required_synchronization": [
            "hardware_event_trigger_or_photodiode",
            "vertical_and_horizontal_EOG",
            "bilateral_task_muscle_EMG",
            "force_or_grip_and_limb_IMU",
        ],
        "primary_endpoints": [
            "issued_false_commands_per_minute_on_continuous_no_control",
            "motor_command_sensitivity",
            "no_decision_fraction",
            "wrong_direction_command_fraction",
            "median_and_p95_command_latency_seconds",
        ],
        "locked_external_rule": (
            "Do not refit preprocessing, features, model weights, fusion, calibration, "
            "thresholds, or controller timing after opening the locked evaluation block."
        ),
        "patient_calibration_rule": (
            "Calibration may estimate subject normalization, contamination limits, and a "
            "predeclared threshold offset; later blocks remain untouched."
        ),
        "clinical_status": "research_only_no_autonomous_device_commands",
    }
    protocol_path = destination / "locked_external_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="ascii")

    hard_negative_path = destination / "hard_negative_manifest.template.json"
    hard_negative_path.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "path": "/absolute/path/to/continuous_no_control.edf",
                        "dataset_id": "hard_negative",
                        "subject_key": "prospective:P001",
                        "session": 1,
                        "recording": 1,
                        "negative_kind": "continuous_no_control",
                        "start_seconds": 0.0,
                        "step_seconds": 0.5,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    replay_path = destination / "probability_replay_template.csv"
    with replay_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "timestamp_seconds",
                "cue_id",
                "cue_onset",
                "expected_label",
                "intent_probability",
                "left_probability",
                "right_probability",
                "available",
                "contamination_veto",
            )
        )
        writer.writerow((0.0, "left_001", 1, 1, 0.20, 0.55, 0.45, 1, 0))

    demo_rows, demo_events = _demo_session(selected)
    demo_metrics = controller_metrics(demo_events, session_duration_seconds=42.5)
    demo_path = destination / "functional_demo_metrics.json"
    demo_path.write_text(
        json.dumps(
            {
                "status": "synthetic_functional_test_not_a_scientific_result",
                "metrics": demo_metrics,
                "events": [asdict(event) for event in demo_events],
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )

    timeline_path = destination / "01_cue_constrained_controller_demo.png"
    times = np.asarray([row["timestamp_seconds"] for row in demo_rows])
    probabilities = np.asarray([row["intent_probability"] for row in demo_rows])
    veto = np.asarray([row["contamination_veto"] for row in demo_rows], dtype=bool)
    figure, axis = plt.subplots(figsize=(12.0, 4.8), constrained_layout=True)
    figure.patch.set_facecolor("#f7f4ed")
    axis.set_facecolor("#ffffff")
    axis.plot(times, probabilities, color="#12355b", linewidth=2.2, label="EEG intent probability")
    axis.axhline(selected.intent_threshold, color="#b23a48", linestyle="--", linewidth=1.8, label="Frozen threshold")
    for onset, (cue_id, expected) in {5.0: ("L", 1), 15.0: ("R", 2), 25.0: ("No-go", 0), 35.0: ("L missed", 1)}.items():
        axis.axvspan(onset, onset + selected.arming_seconds, color="#2f8f83", alpha=0.10)
        axis.text(onset + 0.08, 0.96, cue_id, va="top", fontsize=9, color="#2f8f83")
    axis.scatter(times[veto], probabilities[veto], marker="x", s=90, linewidths=2.2, color="#e9a23b", label="Contamination veto")
    commands = [event for event in demo_events if event.kind == "confirmed_command"]
    axis.scatter(
        [event.timestamp_seconds for event in commands],
        [event.intent_probability for event in commands],
        marker="D",
        s=65,
        color="#2f8f83",
        label="Confirmed EEG command",
        zorder=5,
    )
    axis.set(xlabel="Session time (s)", ylabel="Intent probability", ylim=(0.0, 1.02))
    axis.set_title("Cue-Constrained EEG Controller: Synthetic Functional Demonstration", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=4, loc="lower center")
    figure.savefig(timeline_path, dpi=220, facecolor=figure.get_facecolor())
    plt.close(figure)

    report_path = destination / "PROSPECTIVE_CONTROLLER_REPORT.md"
    report_path.write_text(
        "\n".join(
            (
                "# Prospective Cue-Constrained EEG Controller",
                "",
                "## Status",
                "",
                "The v4 decoder and all five MRCP-FBCNet seed weights are copied and SHA-256 fingerprinted. The current result remains `development_observed`; no locked external cohort has been evaluated.",
                "",
                "The controller is research-only and device output is disabled. The included timeline and metrics are a synthetic functional test, not participant evidence.",
                "",
                "## Decision Path",
                "",
                "`cue arms a 4 s opportunity -> frozen EEG decoder -> frozen OOD availability -> full-montage contamination veto -> two direction-consistent EEG confirmations -> command or explicit no-decision -> refractory period`",
                "",
                "The cue never supplies the command direction. Unarmed EEG detections are logged as latent candidates but cannot issue a command.",
                "",
                "## Required Confirmation",
                "",
                "1. Fit only subject normalization and contamination limits on the labeled calibration block.",
                "2. Keep the later cue-locked block and continuous no-control block untouched.",
                "3. Verify the frozen hashes immediately before inference.",
                "4. Report false commands/min, sensitivity, no-decisions, wrong direction, latency, OOD abstention, and contamination veto rate together.",
                "5. Do not claim clinical utility until a prospective safety protocol and supervised hardware stop have been validated.",
                "",
                "## Prepared Files",
                "",
                "- `frozen_model/frozen_model_manifest.json`: immutable model provenance and hashes.",
                "- `prospective_controller_config.json`: predeclared command policy.",
                "- `locked_external_protocol.json`: split, acquisition, and endpoint contract.",
                "- `hard_negative_manifest.template.json`: continuous no-control registration template.",
                "- `probability_replay_template.csv`: decoder-output replay interface.",
                "- `functional_demo_metrics.json`: synthetic state-machine verification.",
                "- `01_cue_constrained_controller_demo.png`: presentation-ready functional diagram.",
                "",
            )
        ),
        encoding="ascii",
    )
    return {
        "manifest": destination / "frozen_model" / "frozen_model_manifest.json",
        "config": config_path,
        "protocol": protocol_path,
        "hard_negative_template": hard_negative_path,
        "replay_template": replay_path,
        "demo_metrics": demo_path,
        "timeline": timeline_path,
        "report": report_path,
    }
