#!/usr/bin/env python3
"""Freeze v4 and prepare or replay the cue-constrained prospective controller."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

from bci_pipeline.prospective_controller import (  # noqa: E402
    CueConstrainedController,
    CueControllerConfig,
    controller_metrics,
    prepare_prospective_package,
    verify_frozen_bundle,
)


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v4_neural_integrated"
    / "translational_v4_neural_models.joblib"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "prospective_cue_controller"


def _boolean(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def replay_probabilities(
    path: Path,
    output_dir: Path,
    config: CueControllerConfig,
) -> tuple[Path, Path]:
    """Replay a causal decoder-output table through the frozen command policy."""
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError("Probability replay CSV is empty.")
    required = {
        "timestamp_seconds",
        "intent_probability",
        "left_probability",
        "right_probability",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Probability replay CSV is missing: {sorted(missing)}")
    controller = CueConstrainedController(config)
    first_time = float(rows[0]["timestamp_seconds"])
    last_time = first_time
    for row in rows:
        timestamp = float(row["timestamp_seconds"])
        if timestamp < last_time:
            raise ValueError("Probability replay timestamps must be monotonic.")
        last_time = timestamp
        if _boolean(row.get("cue_onset", "0")):
            controller.arm(
                row.get("cue_id", f"cue_{timestamp:g}"),
                timestamp,
                expected_label=int(row.get("expected_label", "-1") or -1),
            )
        controller.step(
            timestamp,
            float(row["intent_probability"]),
            (float(row["left_probability"]), float(row["right_probability"])),
            available=_boolean(row.get("available", "1")),
            contamination_veto=_boolean(row.get("contamination_veto", "0")),
        )
    controller.finalize(last_time)
    duration = max(0.0, last_time - first_time)
    metrics = controller_metrics(controller.events, session_duration_seconds=duration)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "probability_replay_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "source": str(path.resolve()),
                "controller": asdict(config),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    events_path = output_dir / "probability_replay_events.csv"
    with events_path.open("w", newline="", encoding="ascii") as handle:
        fieldnames = list(asdict(controller.events[0]).keys()) if controller.events else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(asdict(event) for event in controller.events)
    return metrics_path, events_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intent-threshold", type=float)
    parser.add_argument("--confirmation-windows", type=int, default=2)
    parser.add_argument("--arming-seconds", type=float, default=4.0)
    parser.add_argument("--refractory-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-direction-probability", type=float, default=0.55)
    parser.add_argument("--replay-probabilities", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir / "frozen_model" / "frozen_model_manifest.json"
    if args.verify_only:
        verify_frozen_bundle(manifest_path)
        print(f"[+] Frozen model hashes verified: {manifest_path}")
        return
    config = None
    if args.intent_threshold is not None:
        config = CueControllerConfig(
            intent_threshold=args.intent_threshold,
            confirmation_windows=args.confirmation_windows,
            arming_seconds=args.arming_seconds,
            refractory_seconds=args.refractory_seconds,
            minimum_direction_probability=args.minimum_direction_probability,
        )
    paths = prepare_prospective_package(
        args.output_dir,
        args.model_path,
        config=config,
    )
    verify_frozen_bundle(paths["manifest"])
    print("\n=== Prospective Cue-Constrained Controller Preparation ===")
    print(f"[+] Frozen model manifest: {paths['manifest']}")
    print(f"[+] Controller protocol: {paths['protocol']}")
    print(f"[+] Functional demonstration: {paths['timeline']}")
    print("[!] External result unavailable: no locked prospective cohort has been collected.")
    print("[!] Device output remains disabled.")
    if args.replay_probabilities:
        selected_config = config
        if selected_config is None:
            payload = json.loads(paths["config"].read_text(encoding="ascii"))
            selected_config = CueControllerConfig(**payload["controller"])
        metrics, events = replay_probabilities(
            args.replay_probabilities.expanduser().resolve(),
            args.output_dir,
            selected_config,
        )
        print(f"[+] Replay metrics: {metrics}")
        print(f"[+] Replay events: {events}")


if __name__ == "__main__":
    main()
