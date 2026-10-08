#!/usr/bin/env python3
"""Test whether the V7 rest-vs-intent contrast survives cue and trial-phase controls.

Rebuilds windows from the native PhysioNet, OpenBMI, and Liu recordings under
alternative definitions (pre-cue rest, later intent windows, causal filters),
then scores simple transparent decoders with the same participant roles and
evaluation rows as V7 (development_observed participants, later recordings).
Negative controls: an amplitude-only model, an EOG/EMG-only model, and a
rest-vs-rest contrast that contains no motor imagery at all.

This is an audit experiment. It does not refit or modify V7.

Usage:
    .venv/bin/python scripts/54_confound_control_benchmark.py            # build if needed, evaluate
    .venv/bin/python scripts/54_confound_control_benchmark.py --rebuild  # force feature rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bci_pipeline.confound_controls import (  # noqa: E402
    CONDITIONS,
    MODELS,
    PHASES,
    _FeatureRows,
    _subject_baselines,
    evaluate,
    evaluate_within_subject,
    offset_sweep,
    rows_to_arrays,
    subject_rows,
)

DATASETS = ("physionet", "openbmi", "liu2024")
DEFAULT_ROOTS = {
    "physionet": PROJECT_ROOT / "files/eegmmidb/1.0.0",
    "openbmi": PROJECT_ROOT / "files/external/openbmi/nm000338",
    "liu2024": PROJECT_ROOT / "Liu acute-stroke dataset",
}
DEFAULT_ROLES = (
    PROJECT_ROOT
    / "artifacts/gui_experiments/translational_v7_graph_lda/subject_roles.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/confound_controls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--physionet-dir", type=Path, default=DEFAULT_ROOTS["physionet"])
    parser.add_argument("--openbmi-dir", type=Path, default=DEFAULT_ROOTS["openbmi"])
    parser.add_argument("--liu-dir", type=Path, default=DEFAULT_ROOTS["liu2024"])
    parser.add_argument("--subject-roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel subjects during the build.")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the report and figures from the saved metrics JSON.",
    )
    return parser.parse_args()


def load_roles(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    return {subject: role for dataset in payload.values() for role, subjects in dataset.items() for subject in subjects}


def build_cache(args: argparse.Namespace, roles: dict[str, str], cache: Path) -> dict[str, np.ndarray]:
    from joblib import Parallel, delayed

    roots = {"physionet": args.physionet_dir, "openbmi": args.openbmi_dir, "liu2024": args.liu_dir}
    jobs = [
        (dataset, roots[dataset], int(subject.split(":S")[1]))
        for subject in sorted(roles)
        for dataset in [subject.split(":")[0]]
    ]
    print(f"[+] Building confound-control features for {len(jobs)} participants ({PHASES})...")
    started = time.time()
    outputs = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(subject_rows)(dataset, root, subject) for dataset, root, subject in jobs
    )
    rows = _FeatureRows()
    excluded = []
    for subject_output, error in outputs:
        rows.extend(subject_output)
        if error:
            excluded.append(error)
    arrays = rows_to_arrays(rows)
    arrays["excluded"] = np.asarray(excluded)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    print(f"[+] {len(arrays['kinds']):,} windows in {(time.time() - started) / 60:.1f} min; {len(excluded)} excluded")
    for error in excluded:
        print(f"    [!] {error}")
    return arrays


# --------------------------------------------------------------------------
# Report and figures
# --------------------------------------------------------------------------

CONDITION_LABELS = {
    "original": "V7 windows",
    "original_causal": "V7 windows,\ncausal filters",
    "precue_rest": "Pre-cue rest",
    "precue_rest_late_intent": "Pre-cue rest,\nlate intent",
    "precue_rest_late_intent_causal": "Pre-cue rest,\nlate intent,\ncausal",
    "rest_vs_rest_control": "Control:\nrest vs rest",
}
DATASET_LABELS = {"physionet": "PhysioNet (healthy)", "openbmi": "OpenBMI (healthy)", "liu2024": "Liu (acute stroke)"}
MODEL_LABELS = {
    "amplitude": "Amplitude only",
    "auxiliary_eog_emg": "EOG/EMG only",
    "bandpower": "Mu/beta bandpower",
    "bandpower_baseline_normalized": "Bandpower, baseline-normalized",
    "riemann_recentered": "Riemannian, re-centered",
    "mrcp": "MRCP (0.1-4 Hz)",
    "combined": "Combined physiology",
}
# Reference categorical palette, first three slots (validated all-pairs).
SERIES = {
    "combined": ("#2a78d6", "o"),
    "amplitude": ("#eb6834", "s"),
    "auxiliary_eog_emg": ("#1baf7a", "^"),
}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e4e3df", "surface": "#fcfcfb"}


def _lookup(results, condition, model, dataset):
    for result in results:
        if result["condition"] == condition and result["model"] == model and result["dataset"] == dataset:
            return result
    return None


def _style_axis(axis) -> None:
    axis.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK["muted"])
    axis.tick_params(colors=INK["secondary"], labelsize=9)
    axis.grid(axis="y", color=INK["grid"], linewidth=0.8)
    axis.set_axisbelow(True)


def plot_conditions(payload, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(CONDITION_LABELS)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, facecolor=INK["surface"])
    offsets = {"combined": -0.18, "amplitude": 0.0, "auxiliary_eog_emg": 0.18}
    for axis, dataset in zip(axes, DATASET_LABELS):
        _style_axis(axis)
        axis.axhline(0.5, color=INK["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        for model, (color, marker) in SERIES.items():
            xs, ys, lows, highs = [], [], [], []
            for position, condition in enumerate(conditions):
                result = _lookup(payload["population"], condition, model, dataset)
                if result is None:
                    continue
                xs.append(position + offsets[model])
                ys.append(result["participant_mean_auroc"])
                lows.append(result["participant_mean_auroc"] - result["participant_mean_auroc_95_ci"][0])
                highs.append(result["participant_mean_auroc_95_ci"][1] - result["participant_mean_auroc"])
            if not xs:
                continue
            axis.errorbar(
                xs, ys, yerr=[lows, highs], fmt=marker, color=color, markersize=7,
                markeredgecolor=INK["surface"], markeredgewidth=1.5, elinewidth=1.6,
                capsize=0, label=MODEL_LABELS[model], zorder=3,
            )
        axis.set_xticks(range(len(conditions)))
        axis.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=8.5, color=INK["secondary"])
        axis.set_title(DATASET_LABELS[dataset], fontsize=11.5, color=INK["primary"], loc="left", pad=10)
        axis.text(len(conditions) - 0.55, 0.505, "chance", fontsize=8.5, color=INK["muted"], ha="right", va="bottom")
    axes[0].set_ylabel("Participant-mean AUROC\n(later recordings, 95% CI)", color=INK["secondary"], fontsize=10)
    axes[0].set_ylim(0.35, 1.0)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 1.0))
    figure.suptitle(
        "Rest-vs-intent separability under cue and trial-phase controls",
        x=0.01, ha="left", fontsize=13, color=INK["primary"], y=0.99,
    )
    figure.text(
        0.01, 0.925,
        "Population models trained on development + controller participants; V7 evaluation participants scored. "
        "EOG/EMG unavailable for PhysioNet.",
        fontsize=9, color=INK["secondary"],
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=160, facecolor=INK["surface"])
    plt.close(figure)


def plot_offset_sweep(payload, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        "rest_original": ("#eb6834", "s", "vs post-task rest (V7)"),
        "rest_precue": ("#2a78d6", "o", "vs pre-cue rest"),
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, facecolor=INK["surface"])
    for axis, dataset in zip(axes, DATASET_LABELS):
        _style_axis(axis)
        axis.axhline(0.5, color=INK["muted"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        for negative, (color, marker, label) in series.items():
            sweep = payload["offset_sweeps"].get(f"{dataset}|combined|{negative}", [])
            if not sweep:
                continue
            xs = [point["offset_seconds"] for point in sweep]
            ys = [point["participant_mean_auroc"] for point in sweep]
            low = [point["ci"][0] for point in sweep]
            high = [point["ci"][1] for point in sweep]
            axis.fill_between(xs, low, high, color=color, alpha=0.12, linewidth=0)
            axis.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                      markeredgecolor=INK["surface"], markeredgewidth=1.5, label=label, zorder=3)
            axis.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                          fontsize=8.5, color=INK["secondary"], va="center")
        axis.set_title(DATASET_LABELS[dataset], fontsize=11.5, color=INK["primary"], loc="left", pad=10)
        axis.set_xlabel("Intent window start (s after cue; 2-s window)", fontsize=9.5, color=INK["secondary"])
        axis.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
        axis.set_xlim(-0.15, 2.35)
    axes[0].set_ylabel("Participant-mean AUROC\n(combined physiology)", fontsize=10, color=INK["secondary"])
    axes[0].set_ylim(0.35, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.99, 1.0))
    figure.suptitle("How separability changes as the intent window moves away from the cue",
                    x=0.01, ha="left", fontsize=13, color=INK["primary"], y=0.99)
    figure.text(0.01, 0.9, "Liu offsets are relative to the motor-imagery stage onset (V7 used 1.0 s); "
                "Liu 'pre-cue' is the instruction stage. Shaded: 95% CI over participants.",
                fontsize=9, color=INK["secondary"])
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=160, facecolor=INK["surface"])
    plt.close(figure)


def _cell(result) -> str:
    if result is None:
        return "-"
    low, high = result["participant_mean_auroc_95_ci"]
    return f"{result['participant_mean_auroc']:.3f} ({low:.2f}-{high:.2f})"


def write_report(payload, path: Path) -> None:
    conditions = list(CONDITION_LABELS)
    lines = [
        "# Confound-Control Benchmark for the Rest-vs-Intent Contrast",
        "",
        "Generated by `scripts/54_confound_control_benchmark.py`. Audit experiment; V7 is not refit or modified.",
        "",
        "## Protocol",
        "",
        f"- Training rows: {payload['training_rows']}.",
        f"- Evaluation rows: {payload['evaluation_rows']}.",
        "- Population models: standardized features + class-balanced logistic regression, fitted per dataset.",
        "- Within-subject models: fitted on each evaluation participant's recording 0, scored on later recordings.",
        "- Primary metric: AUROC computed within each participant, then averaged (threshold-free; removes between-participant offsets). 95% CI from a participant bootstrap.",
        "- Baseline normalization and Riemannian re-centering use each participant's recording-0 pre-cue windows (label-free).",
        "",
        "| Condition | Definition |",
        "|---|---|",
        *[f"| `{name}` | {payload['conditions'][name]['description']} |" for name in conditions],
        "",
        "Liu has no cue-free pre-trial rest (contiguous 8-s trials), so its `rest_precue` is the 2-s instruction stage.",
        "",
        "## Population models: participant-mean AUROC (95% CI)",
        "",
    ]
    for dataset, label in DATASET_LABELS.items():
        lines += [f"### {label}", "", "| Model | " + " | ".join(CONDITION_LABELS[c].replace(chr(10), " ") for c in conditions) + " |",
                  "|---|" + "---:|" * len(conditions)]
        for model in payload["models"]:
            cells = [_cell(_lookup(payload["population"], c, model, dataset)) for c in conditions]
            if all(cell == "-" for cell in cells):
                continue
            lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Within-subject calibration (recording 0 -> later recordings): participant-mean AUROC", "",
              "| Dataset | Model | " + " | ".join(CONDITION_LABELS[c].replace(chr(10), " ") for c in conditions) + " |",
              "|---|---|" + "---:|" * len(conditions)]
    for dataset, label in DATASET_LABELS.items():
        for model in ("combined", "amplitude"):
            cells = [_cell(_lookup(payload["within_subject_calibration"], c, model, dataset)) for c in conditions]
            lines.append(f"| {label} | {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    lines += ["", "## Intent-window offset sweep (combined physiology, population model)", "",
              "| Dataset | Negative | " + " | ".join(f"{o:.1f} s" for o in (0.0, 0.5, 1.0, 1.5, 2.0)) + " |",
              "|---|---|" + "---:|" * 5]
    for dataset, label in DATASET_LABELS.items():
        for negative in ("rest_original", "rest_precue"):
            sweep = payload["offset_sweeps"].get(f"{dataset}|combined|{negative}", [])
            lines.append(f"| {label} | `{negative}` | " + " | ".join(f"{p['participant_mean_auroc']:.3f}" for p in sweep) + " |")
    lines += ["", "## Participant-level balanced accuracy at threshold 0.5 (combined physiology)", "",
              "| Dataset | " + " | ".join(CONDITION_LABELS[c].replace(chr(10), " ") for c in conditions) + " |",
              "|---|" + "---:|" * len(conditions)]
    for dataset, label in DATASET_LABELS.items():
        cells = []
        for c in conditions:
            r = _lookup(payload["population"], c, "combined", dataset)
            cells.append("-" if r is None else f"{100 * r['participant_mean_balanced_accuracy']:.1f}% ({r['participants_above_chance']}/{r['participants']} above chance)")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "## Figures", "", "- `01_confound_conditions.png`", "- `02_intent_offset_sweep.png`", "",
              "Interpretation is in `docs/final/CONFOUND_CONTROL_FINDINGS.md`.", ""]
    if payload.get("excluded"):
        lines += ["## Excluded participants", "", *[f"- {e}" for e in payload["excluded"]], ""]
    path.write_text("\n".join(lines), encoding="ascii")


def write_outputs(output: Path) -> None:
    payload = json.loads((output / "confound_control_metrics.json").read_text(encoding="ascii"))
    plot_conditions(payload, output / "01_confound_conditions.png")
    plot_offset_sweep(payload, output / "02_intent_offset_sweep.png")
    write_report(payload, output / "CONFOUND_CONTROL_REPORT.md")
    print(f"[+] Wrote report and figures to {output}")


def main() -> None:
    args = parse_args()
    roles = load_roles(args.subject_roles)
    output = args.output_dir
    if args.report_only:
        write_outputs(output)
        return
    cache = output / "confound_window_features.npz"
    if cache.is_file() and not args.rebuild:
        loaded = np.load(cache, allow_pickle=False)
        arrays = {key: loaded[key] for key in loaded.files}
        print(f"[+] Loaded {len(arrays['kinds']):,} cached windows from {cache}")
    else:
        arrays = build_cache(args, roles, cache)

    started = time.time()
    baselines = {phase: _subject_baselines(arrays, phase) for phase in PHASES}
    population, within = [], []
    for condition, spec in CONDITIONS.items():
        phase_baselines = baselines[str(spec["phase"])]
        for dataset in DATASETS:
            for model in MODELS:
                result = evaluate(
                    arrays, roles, condition, model, dataset, phase_baselines,
                    iterations=args.bootstrap_iterations, seed=args.seed,
                )
                if result is not None:
                    population.append(result)
                if model in ("amplitude", "combined"):
                    result = evaluate_within_subject(
                        arrays, roles, condition, model, dataset, phase_baselines,
                        iterations=args.bootstrap_iterations, seed=args.seed,
                    )
                    if result is not None:
                        within.append(result)
        print(f"    evaluated {condition}")
    sweeps = {
        f"{dataset}|{model}|{negative}": offset_sweep(
            arrays, roles, dataset, model, negative, baselines["zero"], seed=args.seed
        )
        for dataset in DATASETS
        for model in ("amplitude", "combined")
        for negative in ("rest_original", "rest_precue")
    }
    counts = {
        dataset: {
            "participants": int(len(np.unique(arrays["subjects"][arrays["datasets"] == dataset]))),
            "windows_per_phase": int(np.sum((arrays["datasets"] == dataset) & (arrays["phases"] == "zero"))),
        }
        for dataset in DATASETS
    }
    payload = {
        "schema_version": 1,
        "purpose": "Audit of cue/trial-phase confounds in the V7 rest-vs-intent contrast",
        "evaluation_rows": "development_observed participants (V7 roles), recordings after recording 0",
        "training_rows": "development + controller_validation participants, all recordings",
        "conditions": {name: {k: v for k, v in spec.items()} for name, spec in CONDITIONS.items()},
        "models": list(MODELS),
        "dataset_counts": counts,
        "excluded": arrays["excluded"].tolist(),
        "population": population,
        "within_subject_calibration": within,
        "offset_sweeps": sweeps,
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (output / "confound_control_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii"
    )
    print(f"[+] Wrote {output / 'confound_control_metrics.json'}")
    write_outputs(output)


if __name__ == "__main__":
    main()
