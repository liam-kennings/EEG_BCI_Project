#!/usr/bin/env python3
"""Run the V7-H EEG-fNIRS REST-versus-INTENT benchmark on MIND."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

from bci_pipeline.mind_dataset import (
    MINDHybridFeatureDataset,
    build_mind_hybrid_feature_dataset,
    inspect_mind_dataset,
    synthetic_mind_feature_dataset,
)
from bci_pipeline.v7h import V7H_MODES, evaluate_v7h


COLORS = {
    "navy": "#173B5E",
    "blue": "#2B83A9",
    "teal": "#248F83",
    "gold": "#E9A12B",
    "orange": "#D96535",
    "paper": "#F7F4EC",
    "ink": "#172B3A",
    "muted": "#61717D",
}


MODE_LABELS = {
    "eeg_only": "EEG only",
    "fnirs_only": "fNIRS only",
    "late_fusion_general": "Population fusion",
    "late_fusion_patient_adapted": "Patient-adapted V7-H",
    "sequential_confirmation": "Sequential confirmation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mind-dir", type=Path, default=PROJECT_ROOT / "MIND")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "v7h_mind",
    )
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-specificity", type=float, default=0.93)
    parser.add_argument("--calibration-rest", type=int, default=10)
    parser.add_argument("--calibration-intent", type=int, default=20)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a deterministic end-to-end smoke test without downloaded data.",
    )
    return parser.parse_args()


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
        }
    )


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _build_dataset(args: argparse.Namespace, cache: Path) -> MINDHybridFeatureDataset:
    if cache.is_file() and not args.rebuild_cache:
        print(f"[+] Loading V7-H feature cache: {cache}")
        return MINDHybridFeatureDataset.load(cache)
    if args.synthetic:
        print("[+] Building deterministic synthetic EEG-fNIRS smoke-test data...")
        dataset = synthetic_mind_feature_dataset(subjects=18, seed=args.seed)
    else:
        summary = inspect_mind_dataset(args.mind_dir)
        print(summary.to_text())
        dataset = build_mind_hybrid_feature_dataset(
            summary.root,
            max_subjects=args.max_subjects,
        )
    dataset.save(cache)
    print(f"[+] Saved {len(dataset.labels):,} aligned rows to {cache}")
    return dataset


def _metric_figure(report: dict[str, object], output: Path) -> None:
    metrics = report["metrics"]
    modes = list(V7H_MODES)
    x = np.arange(len(modes))
    width = 0.23
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    for offset, key, color, label in (
        (-width, "balanced_accuracy", COLORS["navy"], "Balanced accuracy"),
        (0.0, "sensitivity", COLORS["teal"], "Intent sensitivity"),
        (width, "specificity", COLORS["gold"], "Rest specificity"),
    ):
        values = [100.0 * float(metrics[mode][key]) for mode in modes]
        bars = ax.bar(x + offset, values, width, color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=9, fontweight="bold", padding=2)
    ax.axhline(93.0, color=COLORS["orange"], linewidth=1.5, linestyle="--")
    ax.text(
        len(modes) - 0.45,
        94.1,
        "93% target specificity",
        ha="right",
        color=COLORS["orange"],
        fontweight="bold",
    )
    ax.set_xticks(x, [MODE_LABELS[mode] for mode in modes])
    ax.set_ylabel("Locked internal test result (%)")
    ax.set_ylim(0, 105)
    ax.set_title("V7-H Modality and Controller Comparison", loc="left")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "01_v7h_model_comparison.png")
    plt.close(fig)


def _roc_figure(arrays: dict[str, np.ndarray], report: dict[str, object], output: Path) -> None:
    labels = arrays["labels"]
    curves = (
        ("eeg_probability", "EEG only", COLORS["blue"]),
        ("fnirs_probability", "fNIRS only", COLORS["gold"]),
        ("fusion_general_probability", "Population fusion", COLORS["muted"]),
        (
            "fusion_adapted_probability",
            "Patient-adapted V7-H",
            COLORS["teal"],
        ),
    )
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    mode_lookup = {
        "eeg_probability": "eeg_only",
        "fnirs_probability": "fnirs_only",
        "fusion_general_probability": "late_fusion_general",
        "fusion_adapted_probability": "late_fusion_patient_adapted",
    }
    for key, label, color in curves:
        false_positive, true_positive, _ = roc_curve(labels, arrays[key])
        auc = report["metrics"][mode_lookup[key]]["auroc"]
        ax.plot(
            false_positive,
            true_positive,
            linewidth=2.6,
            color=color,
            label=f"{label} (AUROC {auc:.3f})",
        )
    ax.plot((0, 1), (0, 1), color="#9AA5AD", linestyle="--", linewidth=1.2)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Threshold-Independent Intent Discrimination", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output / "02_v7h_roc.png")
    plt.close(fig)


def _frontier_figure(arrays: dict[str, np.ndarray], output: Path) -> None:
    labels = arrays["labels"]
    probability = arrays["fusion_adapted_probability"]
    sensitivity = []
    specificity = []
    thresholds = np.linspace(0.02, 0.98, 193)
    for threshold in thresholds:
        prediction = probability >= threshold
        sensitivity.append(np.mean(prediction[labels == 1]))
        specificity.append(np.mean(~prediction[labels == 0]))
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    points = ax.scatter(
        100.0 * np.asarray(specificity),
        100.0 * np.asarray(sensitivity),
        c=thresholds,
        cmap="viridis",
        s=30,
    )
    ax.axvline(93.0, color=COLORS["orange"], linestyle="--", linewidth=2)
    ax.text(92.3, 8, "Rest-safety target", rotation=90, ha="right", color=COLORS["orange"], fontweight="bold")
    ax.set_xlabel("Rest specificity (%)")
    ax.set_ylabel("Intent sensitivity (%)")
    ax.set_title("V7-H Safety-Sensitivity Operating Frontier", loc="left")
    ax.set_xlim(45, 101)
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.22)
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Decision threshold")
    fig.tight_layout()
    fig.savefig(output / "03_v7h_safety_sensitivity_frontier.png")
    plt.close(fig)


def _architecture_figure(report: dict[str, object], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    stages = (
        (0.3, 3.1, "EEG", "MRCP + SMR\n+ covariance", COLORS["blue"]),
        (0.3, 0.8, "fNIRS", "HbO/HbR\n amplitude + slope", COLORS["gold"]),
        (3.2, 2.0, "POPULATION", "Cross-fitted\ncalibrated fusion", COLORS["navy"]),
        (6.0, 2.0, "PATIENT", "30-window\ncalibration", COLORS["teal"]),
        (8.8, 2.0, "QUALITY GATE", "Hybrid / fallback\n/ abstain", COLORS["orange"]),
        (11.6, 2.0, "OUTPUT", "REST or\nCONFIRM INTENT", COLORS["navy"]),
    )
    for x, y, heading, body, color in stages:
        box = FancyBboxPatch(
            (x, y),
            2.1,
            1.35,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            linewidth=2.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x + 1.05, y + 0.95, heading, ha="center", color=color, fontweight="bold", fontsize=11)
        ax.text(x + 1.05, y + 0.42, body, ha="center", color=COLORS["ink"], fontweight="bold", fontsize=11)
    for start, end in (((2.4, 3.78), (3.2, 2.9)), ((2.4, 1.48), (3.2, 2.5)), ((5.3, 2.68), (6.0, 2.68)), ((8.1, 2.68), (8.8, 2.68)), ((10.9, 2.68), (11.6, 2.68))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2.2, "color": COLORS["muted"]})
    ax.set_title("V7-H: Fast EEG Proposal with Slow Hemodynamic Confirmation", loc="left", fontsize=19, fontweight="bold", color=COLORS["ink"])
    ax.text(
        13.8,
        0.25,
        f"Quality threshold learned from development data: {float(report['quality_threshold']):.2f}",
        ha="right",
        color=COLORS["muted"],
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output / "04_v7h_architecture.png")
    plt.close(fig)


def _write_predictions(arrays: dict[str, np.ndarray], output: Path) -> Path:
    frame = pd.DataFrame({key: value for key, value in arrays.items()})
    path = output / "v7h_mind_locked_internal_predictions.csv"
    frame.to_csv(path, index=False)
    return path


def _write_report(report: dict[str, object], output: Path, source: str) -> Path:
    metrics = report["metrics"]
    rows = []
    for mode in V7H_MODES:
        score = metrics[mode]
        rows.append(
            "| "
            + MODE_LABELS[mode]
            + " | "
            + f"{100.0 * float(score['balanced_accuracy']):.1f}% | "
            + f"{100.0 * float(score['sensitivity']):.1f}% | "
            + f"{100.0 * float(score['specificity']):.1f}% | "
            + f"{float(score['auroc']):.3f} | "
            + f"{100.0 * float(score['abstained_fraction']):.1f}% |"
        )
    text = f"""# V7-H MIND EEG-fNIRS Benchmark

## Status

- Source: {source}
- Endpoint: binary REST versus INTENT
- Validation: participant-isolated development, controller validation, and locked internal public-dataset test
- Patient calibration: recording 0 only; later recordings evaluated
- External stroke validation: not evaluated

## Locked Internal Result

| Mode | Balanced accuracy | Sensitivity | Specificity | AUROC | Abstained |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The primary hybrid comparison is patient-adapted V7-H versus the EEG-only
branch. Sequential confirmation requires both an EEG proposal and acceptable
fNIRS confirmation. Poor derivative-level fNIRS quality causes abstention in
that mode and EEG-only fallback in the patient-adapted late-fusion mode.

## Research Contract

- Thresholds were selected on controller-validation participants only.
- Locked internal test participants were not used for population training or
  operating-point selection.
- Each test participant used at most {report['calibration_budget']['rest_windows']}
  REST and {report['calibration_budget']['intent_windows']} INTENT windows from
  recording 0 for calibration.
- Directional labels are retained as metadata but the exposed endpoint is only
  REST or INTENT.
- Continuous false activations per minute remain unmeasured.

## Outputs

- `v7h_mind_metrics.json`
- `v7h_mind_model.joblib`
- `v7h_mind_locked_internal_predictions.csv`
- `01_v7h_model_comparison.png`
- `02_v7h_roc.png`
- `03_v7h_safety_sensitivity_frontier.png`
- `04_v7h_architecture.png`

## Caveats

{chr(10).join('- ' + item for item in report['caveats'])}
"""
    path = output / "V7H_MIND_REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.target_specificity < 1.0:
        raise ValueError("--target-specificity must be in [0.5, 1.0).")
    configure_plots()
    started = time.perf_counter()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = (
        args.cache_path.expanduser().resolve()
        if args.cache_path is not None
        else output / "mind_v7h_features.npz"
    )
    dataset = _build_dataset(args, cache)
    model, report, arrays = evaluate_v7h(
        dataset,
        seed=args.seed,
        target_specificity=args.target_specificity,
        calibration_rest=args.calibration_rest,
        calibration_intent=args.calibration_intent,
    )
    source = "synthetic smoke test" if args.synthetic else str(args.mind_dir.expanduser().resolve())
    report["source"] = source
    report["runtime_minutes"] = (time.perf_counter() - started) / 60.0
    metrics_path = output / "v7h_mind_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "schema_version": 1,
            "model": model,
            "roles": report["roles"],
            "thresholds": report["thresholds"],
            "quality_threshold": report["quality_threshold"],
            "endpoint": "binary_rest_vs_intent",
        },
        output / "v7h_mind_model.joblib",
    )
    predictions_path = _write_predictions(arrays, output)
    _metric_figure(report, output)
    _roc_figure(arrays, report, output)
    _frontier_figure(arrays, output)
    _architecture_figure(report, output)
    report_path = _write_report(report, output, source)

    print("\n=== V7-H MIND EEG-fNIRS Locked Internal Result ===")
    for mode in V7H_MODES:
        score = report["metrics"][mode]
        print(
            f"{MODE_LABELS[mode]:28s} | "
            f"BA={100.0 * score['balanced_accuracy']:.1f}% | "
            f"sensitivity={100.0 * score['sensitivity']:.1f}% | "
            f"specificity={100.0 * score['specificity']:.1f}% | "
            f"AUROC={score['auroc']:.3f}"
        )
    print("[!] This is a healthy-cohort locked internal test, not stroke validation.")
    print(f"[+] Metrics: {metrics_path}")
    print(f"[+] Predictions: {predictions_path}")
    print(f"[COMPLETE] Report: {report_path}")


if __name__ == "__main__":
    main()
