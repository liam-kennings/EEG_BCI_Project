#!/usr/bin/env python3
"""Run and aggregate the five-seed WAY EEG-only SSL stability experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bci_pipeline.seed_stability import SUMMARY_METRICS, summarize_way_ssl_seeds  # noqa: E402


INK = "#17324D"
TEAL = "#087E8B"
BLUE = "#3F6C8A"
GOLD = "#D99B24"
CORAL = "#C64B42"
DOMAIN_LABELS = {"liu2024": "Liu stroke", "openbmi": "OpenBMI", "physionet": "PhysioNet"}


def _seed_command(args: argparse.Namespace, seed: int, output: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "37_way_eeg_only_ssl_transfer.py"),
        "--way-dir",
        str(args.way_dir),
        "--way-cache",
        str(args.way_cache),
        "--target-raw-cache",
        str(args.target_raw_cache),
        "--target-feature-cache",
        str(args.target_feature_cache),
        "--comparison-model",
        str(args.comparison_model),
        "--output-dir",
        str(output),
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--fine-tune-epochs",
        str(args.fine_tune_epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-controller-false-activation",
        str(args.max_controller_false_activation),
        "--bootstrap-iterations",
        str(args.per_seed_bootstrap_iterations),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]


def _plot_seed_pairs(summary: dict[str, Any], output: Path) -> None:
    records = summary["per_seed"]
    seeds = [str(record["seed"]) for record in records]
    random_ba = 100.0 * np.asarray([record["random_cnn"]["balanced_accuracy"] for record in records])
    ssl_ba = 100.0 * np.asarray([record["way_ssl"]["balanced_accuracy"] for record in records])
    y = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(10.5, 6.4), constrained_layout=True)
    for index in range(len(records)):
        ax.plot([random_ba[index], ssl_ba[index]], [index, index], color="#B8C6CC", linewidth=3)
    ax.scatter(random_ba, y, color=BLUE, s=95, label="Random CNN", zorder=3)
    ax.scatter(ssl_ba, y, color=TEAL, s=95, label="WAY EEG-only SSL", zorder=3)
    for index, (left, right) in enumerate(zip(random_ba, ssl_ba, strict=True)):
        ax.text(max(left, right) + 0.35, index, f"{right - left:+.2f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, [f"Seed {seed}" for seed in seeds])
    ax.invert_yaxis()
    ax.set_xlabel("Development-observed balanced accuracy (%)")
    ax.set_title("WAY EEG-Only SSL Is Compared Within Every Neural Seed", loc="left", weight="bold", color=INK)
    ax.legend(frameon=False, ncols=2, loc="lower right")
    ax.grid(axis="x", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "01_seed_paired_balanced_accuracy.png", dpi=220)
    plt.close(fig)


def _plot_aggregate(summary: dict[str, Any], output: Path) -> None:
    labels = ("Balanced accuracy", "Sensitivity", "Specificity", "AUROC", "AUPRC")
    x = np.arange(len(SUMMARY_METRICS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12.5, 6.5), constrained_layout=True)
    for offset, model, label, color in (
        (-width / 2, "random_cnn", "Random CNN", BLUE),
        (width / 2, "way_ssl", "WAY EEG-only SSL", TEAL),
    ):
        means = 100.0 * np.asarray([summary["aggregate"][model][metric]["mean"] for metric in SUMMARY_METRICS])
        errors = 100.0 * np.asarray(
            [summary["aggregate"][model][metric]["standard_deviation"] for metric in SUMMARY_METRICS]
        )
        bars = ax.bar(x + offset, means, width, yerr=errors, capsize=4, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=5, fontsize=9, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Five-seed mean (%) with SD")
    ax.set_title("Repeated-Seed Performance Summary", loc="left", weight="bold", color=INK)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "02_seed_aggregate_metrics.png", dpi=220)
    plt.close(fig)


def _plot_domains(summary: dict[str, Any], output: Path) -> None:
    domains = tuple(summary["domain_balanced_accuracy_delta"])
    seeds = summary["seeds"]
    matrix = 100.0 * np.asarray(
        [summary["domain_balanced_accuracy_delta"][domain]["per_seed"] for domain in domains]
    )
    means = np.mean(matrix, axis=1, keepdims=True)
    displayed = np.concatenate([matrix, means], axis=1)
    limit = max(5.0, float(np.max(np.abs(displayed))))
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    image = ax.imshow(displayed, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(displayed.shape[0]):
        for column in range(displayed.shape[1]):
            value = displayed[row, column]
            color = "white" if abs(value) > 0.55 * limit else INK
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center", color=color, weight="bold")
    ax.set_xticks(np.arange(len(seeds) + 1), [*[f"Seed {seed}" for seed in seeds], "Mean"])
    ax.set_yticks(np.arange(len(domains)), [DOMAIN_LABELS.get(domain, domain) for domain in domains])
    ax.set_title("SSL Balanced-Accuracy Change by Target Dataset", loc="left", weight="bold", color=INK)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("SSL minus random CNN (percentage points)")
    fig.savefig(output / "03_seed_domain_stability.png", dpi=220)
    plt.close(fig)


def _plot_original_eeg_only(summary: dict[str, Any], output: Path) -> None:
    records = summary["per_seed"]
    seeds = np.asarray(summary["seeds"])
    overall = 100.0 * np.asarray(
        [record["random_cnn"]["balanced_accuracy"] for record in records]
    )
    domains = tuple(summary["domain_model_summary"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)

    axes[0].plot(seeds, overall, color=BLUE, marker="o", markersize=8, linewidth=2.5)
    mean = float(np.mean(overall))
    standard_deviation = float(np.std(overall, ddof=1))
    axes[0].axhline(mean, color=GOLD, linewidth=2.5, label=f"Mean {mean:.2f}%")
    axes[0].fill_between(
        [seeds.min() - 0.2, seeds.max() + 0.2],
        mean - standard_deviation,
        mean + standard_deviation,
        color=GOLD,
        alpha=0.15,
        label=f"SD {standard_deviation:.2f}",
    )
    for seed, value in zip(seeds, overall, strict=True):
        axes[0].text(seed, value + 0.25, f"{value:.2f}", ha="center", color=INK, fontsize=9)
    axes[0].set_xticks(seeds, [f"Seed {seed}" for seed in seeds])
    axes[0].set_ylim(min(overall) - 1.0, max(overall) + 1.4)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_title("Paired random-initialization control", loc="left", weight="bold", color=INK)
    axes[0].legend(frameon=False)

    x = np.arange(len(domains))
    colors = {"liu2024": CORAL, "openbmi": TEAL, "physionet": GOLD}
    for index, domain in enumerate(domains):
        domain_values = 100.0 * np.asarray(
            summary["domain_model_summary"][domain]["random_cnn"]["balanced_accuracy"][
                "per_seed"
            ]
        )
        jitter = np.linspace(-0.12, 0.12, len(domain_values))
        axes[1].scatter(
            np.full(len(domain_values), index) + jitter,
            domain_values,
            color=colors.get(domain, BLUE),
            edgecolor="white",
            linewidth=0.7,
            s=70,
            alpha=0.9,
        )
        domain_mean = float(np.mean(domain_values))
        axes[1].plot(
            [index - 0.2, index + 0.2],
            [domain_mean, domain_mean],
            color=INK,
            linewidth=3,
        )
        axes[1].text(index, domain_mean + 0.7, f"{domain_mean:.2f}%", ha="center", color=INK)
    axes[1].set_xticks(x, [DOMAIN_LABELS.get(domain, domain) for domain in domains])
    axes[1].set_ylabel("Balanced accuracy (%)")
    axes[1].set_title("Original-only performance by dataset", loc="left", weight="bold", color=INK)

    for axis in axes:
        axis.grid(axis="y", color="#D5DEE2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Five-Seed Paired Neural-Control Stability",
        fontsize=17,
        weight="bold",
        color=INK,
    )
    fig.savefig(output / "04_original_eeg_only_seed_stability.png", dpi=220)
    plt.close(fig)


def _write_original_eeg_only_report(summary: dict[str, Any], output: Path) -> None:
    overall = summary["aggregate"]["random_cnn"]
    ba = overall["balanced_accuracy"]
    low, high = ba["t_95_ci"]
    lines = [
        "# Target-Label-Only Paired CNN Five-Seed Result",
        "",
        "## Contract",
        "",
        "- Gradient training and labels use only PhysioNet EEGMMIDB, OpenBMI, and Liu EEG development participants.",
        "- The decoder starts from random weights; WAY labels, EMG, kinematics, force, and peripheral supervision are not used.",
        "- For the original paired SSL ablation, branch-normalization scales are shared with the unlabeled WAY source condition. This is a matched transfer control, not a strict target-only preprocessing baseline.",
        "- Controller-validation participants select dataset-specific thresholds; development-observed participants provide the reported result.",
        "- Seeds 42-46 change neural initialization, dropout, and batch randomness while all data and training settings remain fixed.",
        "",
        "## Overall Result",
        "",
        f"Across five seeds, the paired random-initialization control reached **{100 * ba['mean']:.2f}% balanced accuracy** (SD **{100 * ba['standard_deviation']:.2f}**, 95% seed-level t interval **{100 * low:.2f} to {100 * high:.2f}**), **{100 * overall['sensitivity']['mean']:.2f}% sensitivity**, **{100 * overall['specificity']['mean']:.2f}% specificity**, **{overall['auroc']['mean']:.3f} AUROC**, and **{overall['auprc']['mean']:.3f} AUPRC**.",
        "",
        "| Seed | Balanced accuracy | Sensitivity | Specificity | AUROC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for record in summary["per_seed"]:
        result = record["random_cnn"]
        lines.append(
            f"| {record['seed']} | {100 * result['balanced_accuracy']:.2f}% | "
            f"{100 * result['sensitivity']:.2f}% | {100 * result['specificity']:.2f}% | "
            f"{result['auroc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Stability",
            "",
            "| Dataset | Mean BA | SD | Mean sensitivity | Mean specificity | Mean AUROC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for domain, models in summary["domain_model_summary"].items():
        result = models["random_cnn"]
        lines.append(
            f"| {DOMAIN_LABELS.get(domain, domain)} | "
            f"{100 * result['balanced_accuracy']['mean']:.2f}% | "
            f"{100 * result['balanced_accuracy']['standard_deviation']:.2f} | "
            f"{100 * result['sensitivity']['mean']:.2f}% | "
            f"{100 * result['specificity']['mean']:.2f}% | "
            f"{result['auroc']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The paired random-initialization control is reproducible but remains close to chance at the selected low-false-activation operating point. WAY EEG-only pretraining raises mean balanced accuracy from **53.49%** to **56.29%** relative to this normalization-matched control. A later strict target-development-only normalization audit reached **60.63%** with the same Conv1D family, showing that source-derived scaling was a material confound. The Liu result remains unstable, reinforcing the need for stroke-specific calibration and domain adaptation.",
        ]
    )
    (output / "ORIGINAL_EEG_ONLY_FIVE_SEED_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_report(summary: dict[str, Any], output: Path) -> None:
    delta = summary["paired_ssl_minus_random"]["balanced_accuracy"]
    low, high = delta["t_95_ci"]
    participant = summary["participant_analysis"]
    participant_low, participant_high = participant["participant_bootstrap_95_ci"]
    criteria = summary["predeclared_criteria"]
    lines = [
        "# WAY EEG-Only SSL Five-Seed Stability",
        "",
        "## Fixed Contract",
        "",
        f"Neural seeds **{', '.join(str(seed) for seed in summary['seeds'])}** change only initialization, masking, dropout, and batch randomness. Source windows, target rows, participant roles, architecture, epochs, sample weights, controller thresholding, and development-observed evaluation are fixed.",
        "",
        "## Seed Results",
        "",
        "| Seed | Random CNN BA | WAY SSL BA | BA change | Sensitivity change | Specificity change |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summary["per_seed"]:
        lines.append(
            f"| {record['seed']} | {100 * record['random_cnn']['balanced_accuracy']:.2f}% | "
            f"{100 * record['way_ssl']['balanced_accuracy']:.2f}% | "
            f"{100 * record['ssl_minus_random']['balanced_accuracy']:+.2f} points | "
            f"{100 * record['ssl_minus_random']['sensitivity']:+.2f} points | "
            f"{100 * record['ssl_minus_random']['specificity']:+.2f} points |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Result",
            "",
            f"The paired SSL effect was **{100 * delta['mean']:+.2f} balanced-accuracy points** across seeds (SD **{100 * delta['standard_deviation']:.2f}**, 95% t interval **{100 * low:+.2f} to {100 * high:+.2f}**).",
            f"After averaging each participant across all five seeds, the equal-participant effect was **{100 * participant['mean_seed_averaged_balanced_accuracy_delta']:+.2f} points** (participant-bootstrap 95% CI **{100 * participant_low:+.2f} to {100 * participant_high:+.2f}**); **{100 * participant['participants_improved_fraction']:.1f}%** improved.",
            f"The five-seed SSL mean was **{100 * summary['aggregate']['way_ssl']['balanced_accuracy']['mean']:.2f}% BA**, **{100 * summary['aggregate']['way_ssl']['sensitivity']['mean']:.2f}% sensitivity**, and **{100 * summary['aggregate']['way_ssl']['specificity']['mean']:.2f}% specificity**. Relative to the fixed statistical target-only baseline, this is **{100 * (summary['aggregate']['way_ssl']['balanced_accuracy']['mean'] - summary['fixed_references']['classical_target_only']['balanced_accuracy']):+.2f} BA points**, **{100 * (summary['aggregate']['way_ssl']['sensitivity']['mean'] - summary['fixed_references']['classical_target_only']['sensitivity']):+.2f} sensitivity points**, and **{100 * (summary['aggregate']['way_ssl']['specificity']['mean'] - summary['fixed_references']['classical_target_only']['specificity']):+.2f} specificity points**.",
            "",
            "| Dataset | Mean BA change | Positive seeds |",
            "|---|---:|---:|",
        ]
    )
    for domain, result in summary["domain_balanced_accuracy_delta"].items():
        lines.append(
            f"| {DOMAIN_LABELS.get(domain, domain)} | "
            f"{100 * result['balanced_accuracy_delta']['mean']:+.2f} points | "
            f"{100 * result['positive_seed_fraction']:.0f}% |"
        )
    lines.extend(
        [
            "",
            "## Predeclared Check",
            "",
            f"Overall criteria met: **{'yes' if criteria['all_predeclared_criteria_met'] else 'no'}**.",
            "",
            f"- Positive BA effect in at least four of five seeds: **{'yes' if criteria['primary_stability']['at_least_four_of_five_positive'] else 'no'}**.",
            f"- Positive mean BA effect with seed-level 95% t interval above zero: **{'yes' if criteria['primary_stability']['positive_mean_balanced_accuracy_delta'] and criteria['primary_stability']['seed_t_interval_excludes_zero'] else 'no'}**.",
            f"- Mean specificity erosion no worse than one point: **{'yes' if criteria['safety']['mean_specificity_erosion_within_one_point'] else 'no'}**.",
            f"- Positive mean effect in both OpenBMI and PhysioNet: **{'yes' if all(criteria['cross_domain_replication'].values()) else 'no'}**.",
            "",
            "## Interpretation Boundary",
            "",
            "This confirms or rejects robustness to neural-training randomness only. The same development-observed participants are reused for every seed, so this is not a new external patient validation and does not replace continuous no-control testing.",
        ]
    )
    (output / "WAY_EEG_SSL_SEED_STABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--way-dir", type=Path, default=PROJECT_ROOT / "WAY-EEG-GAL")
    parser.add_argument("--way-cache", type=Path, default=PROJECT_ROOT / "artifacts" / "way_eeg_only_ssl" / "way_unlabeled_eeg_windows.npz")
    parser.add_argument("--target-raw-cache", type=Path, default=PROJECT_ROOT.parent / "artifacts" / "gui_experiments" / "translational_v3" / "translational_windows_v3.npz")
    parser.add_argument("--target-feature-cache", type=Path, default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer" / "pure_eeg_target_features.npz")
    parser.add_argument("--comparison-model", type=Path, default=PROJECT_ROOT / "artifacts" / "multisource_privileged_ensemble" / "multisource_eeg_intent_ensemble.joblib")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "way_eeg_only_ssl_seed_stability")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="PyTorch backend forwarded to every seed run.",
    )
    parser.add_argument("--max-controller-false-activation", type=float, default=0.10)
    parser.add_argument("--per-seed-bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--participant-bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--force", action="store_true", help="Rerun seeds with completed metrics.")
    parser.add_argument("--aggregate-only", action="store_true", help="Do not launch missing seed runs.")
    args = parser.parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        parser.error("The confirmation experiment requires exactly five unique seeds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_root = args.output_dir / "seeds"
    seed_root.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_output = seed_root / f"seed_{seed}"
        metrics_path = seed_output / "way_eeg_only_ssl_metrics.json"
        if args.force or not metrics_path.is_file():
            if args.aggregate_only:
                raise FileNotFoundError(f"Missing seed result: {metrics_path}")
            seed_output.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Neural seed {seed} ===", flush=True)
            subprocess.run(_seed_command(args, seed, seed_output), cwd=PROJECT_ROOT, check=True)
        else:
            print(f"[+] Reusing completed seed {seed}: {metrics_path}", flush=True)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        recorded_seed = payload.get("neural_seed", payload.get("training", {}).get("neural_seed"))
        if recorded_seed is not None and int(recorded_seed) != seed:
            raise ValueError(f"Seed directory {seed} contains metrics for seed {recorded_seed}.")
        payloads.append(payload)

    summary = summarize_way_ssl_seeds(
        payloads,
        args.seeds,
        participant_bootstrap_iterations=args.participant_bootstrap_iterations,
    )
    summary["configuration"] = {
        "pretrain_epochs": args.pretrain_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "max_controller_false_activation": args.max_controller_false_activation,
        "way_cache": str(args.way_cache.resolve()),
        "target_raw_cache": str(args.target_raw_cache.resolve()),
        "target_feature_cache": str(args.target_feature_cache.resolve()),
    }
    (args.output_dir / "way_eeg_ssl_seed_stability_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_seed_pairs(summary, args.output_dir)
    _plot_aggregate(summary, args.output_dir)
    _plot_domains(summary, args.output_dir)
    _plot_original_eeg_only(summary, args.output_dir)
    _write_report(summary, args.output_dir)
    _write_original_eeg_only_report(summary, args.output_dir)
    delta = summary["paired_ssl_minus_random"]["balanced_accuracy"]
    low, high = delta["t_95_ci"]
    print("\n=== Five-Seed WAY EEG-Only SSL Stability ===")
    print(
        f"Paired BA change={100 * delta['mean']:+.2f} points | "
        f"SD={100 * delta['standard_deviation']:.2f} | "
        f"95% t interval={100 * low:+.2f} to {100 * high:+.2f}"
    )
    print(
        "Predeclared criteria: "
        + ("MET" if summary["predeclared_criteria"]["all_predeclared_criteria_met"] else "NOT MET")
    )
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
