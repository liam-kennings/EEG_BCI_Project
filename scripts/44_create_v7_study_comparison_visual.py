#!/usr/bin/env python3
"""Create a poster-ready comparison of V7 with representative BCI studies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "study_comparison"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#17324D",
    "muted": "#607080",
    "line": "#CBD3D7",
    "navy": "#173A59",
    "blue": "#2D7FA8",
    "teal": "#248B7F",
    "gold": "#E9A437",
    "coral": "#D96B3B",
    "soft_blue": "#E8F2F7",
    "soft_teal": "#E5F3EF",
    "soft_gold": "#FBF0D6",
    "soft_coral": "#F8E8DF",
    "soft_gray": "#EEF1F2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "DejaVu Sans"],
            "font.size": 11,
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.25,
    radius: float = 0.018,
    zorder: float = 1,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def badge(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    text: str,
    *,
    color: str,
    soft: str,
) -> None:
    card(
        ax,
        x,
        y,
        width,
        0.026,
        facecolor=soft,
        edgecolor=color,
        linewidth=1.0,
        radius=0.011,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + 0.013,
        text,
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        zorder=5,
    )


def comparison_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    tag: str,
    metric: str,
    metric_note: str,
    study: str,
    task: str,
    design: str,
    color: str,
    soft: str,
) -> None:
    compact = height < 0.19
    card(
        ax,
        x,
        y,
        width,
        height,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=1.35,
        radius=0.021,
    )
    badge(ax, x + 0.016, y + height - 0.044, width - 0.032, tag, color=color, soft=soft)
    ax.text(
        x + 0.02,
        y + height - 0.077,
        metric,
        transform=ax.transAxes,
        fontsize=16.8,
        fontweight="bold",
        color=color,
        va="top",
    )
    ax.text(
        x + width - 0.02,
        y + height - 0.073,
        metric_note,
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["muted"],
        ha="right",
        va="top",
    )
    ax.text(
        x + 0.02,
        y + height - (0.112 if compact else 0.116),
        study,
        transform=ax.transAxes,
        fontsize=9.3,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    ax.text(
        x + 0.02,
        y + height - (0.134 if compact else 0.149),
        task,
        transform=ax.transAxes,
        fontsize=7.9,
        color=COLORS["ink"],
        va="top",
        linespacing=1.25,
    )
    ax.text(
        x + 0.02,
        y + (0.009 if compact else 0.018),
        design,
        transform=ax.transAxes,
        fontsize=7.0 if compact else 7.5,
        color=COLORS["muted"],
        va="bottom",
        linespacing=1.25,
    )


def create_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(10.5, 10.5), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    card(
        ax,
        0.045,
        0.91,
        0.145,
        0.052,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        radius=0.026,
    )
    ax.text(
        0.1175,
        0.936,
        "CONTEXT",
        transform=ax.transAxes,
        color="white",
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0.218,
        0.957,
        "How Do Our Results Compare?",
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.22,
        0.908,
        "Published percentages are shown by endpoint and validation design, not as one accuracy leaderboard",
        transform=ax.transAxes,
        fontsize=10.7,
        color=COLORS["muted"],
        va="top",
    )

    # Current V7 result.
    card(
        ax,
        0.04,
        0.712,
        0.92,
        0.157,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.026,
    )
    ax.add_patch(
        Circle(
            (0.145, 0.7905),
            0.061,
            transform=ax.transAxes,
            facecolor=COLORS["teal"],
            edgecolor="white",
            linewidth=2.0,
            zorder=4,
        )
    )
    ax.text(
        0.145,
        0.803,
        "73.6%",
        transform=ax.transAxes,
        fontsize=18.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        0.145,
        0.775,
        "BALANCED\nACCURACY",
        transform=ax.transAxes,
        fontsize=6.8,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        linespacing=1.1,
        zorder=5,
    )
    ax.text(
        0.225,
        0.835,
        "V7 PATIENT-ADAPTED EEG INTENT GATE",
        transform=ax.transAxes,
        fontsize=12.2,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.225,
        0.805,
        "REST vs INTENT  |  EEG only  |  3 datasets  |  187 represented participants",
        transform=ax.transAxes,
        fontsize=8.7,
        color="#DDE7EC",
        va="center",
    )
    chips = [
        (0.225, "53.5%", "intent sensitivity"),
        (0.405, "93.7%", "rest specificity"),
        (0.585, "0.896", "AUROC"),
    ]
    for x, value, label in chips:
        card(
            ax,
            x,
            0.738,
            0.158,
            0.046,
            facecolor="#234864",
            edgecolor="#52738A",
            linewidth=0.9,
            radius=0.013,
            zorder=3,
        )
        ax.text(
            x + 0.079,
            0.768,
            value,
            transform=ax.transAxes,
            fontsize=10.1,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            x + 0.079,
            0.748,
            label,
            transform=ax.transAxes,
            fontsize=6.7,
            color="#C8D6DE",
            ha="center",
            va="center",
            zorder=5,
        )
    card(
        ax,
        0.765,
        0.738,
        0.165,
        0.095,
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        radius=0.016,
        zorder=3,
    )
    ax.text(
        0.8475,
        0.81,
        "EVALUATION",
        transform=ax.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["gold"],
        ha="center",
    )
    ax.text(
        0.8475,
        0.782,
        "30-window calibration\nlater recordings\n95% CI: 67.5-78.7%",
        transform=ax.transAxes,
        fontsize=7.4,
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.3,
    )
    ax.text(
        0.8475,
        0.749,
        "development observed",
        transform=ax.transAxes,
        fontsize=6.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
    )

    ax.text(
        0.04,
        0.675,
        "CLOSEST PUBLISHED COMPARISONS: MOTOR INTENT / IMAGERY vs REST",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["muted"],
        va="center",
    )

    comparison_card(
        ax,
        x=0.04,
        y=0.435,
        width=0.29,
        height=0.21,
        tag="CLOSEST HISTORICAL ENDPOINT",
        metric="74%",
        metric_note="mean accuracy",
        study="Ang et al., 2011 [1]",
        task="Affected-hand motor imagery versus rest",
        design="46 stroke patients | offline\npatient-specific calibration",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )
    comparison_card(
        ax,
        x=0.355,
        y=0.435,
        width=0.29,
        height=0.21,
        tag="HEALTHY-TO-STROKE TRANSFER",
        metric="71.15%",
        metric_note="mean accuracy",
        study="Nagarajan et al., 2024 [2]",
        task="Stroke motor imagery versus rest after transfer",
        design="71 stroke patients | healthy pretraining\n+ target-domain adaptation",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )
    comparison_card(
        ax,
        x=0.67,
        y=0.435,
        width=0.29,
        height=0.21,
        tag="MULTIMODAL ATTEMPT DETECTION",
        metric="69.7%",
        metric_note="mean accuracy",
        study="Lopez-Larraz et al., 2025 [3]",
        task="Paralyzed-hand attempt versus rest using EEG + EMG",
        design="34 analyzed stroke patients\nwithin-patient block validation",
        color=COLORS["coral"],
        soft=COLORS["soft_coral"],
    )

    ax.text(
        0.04,
        0.397,
        "RELATED CLINICAL CONTEXT: DIFFERENT ENDPOINTS AND METRICS",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["muted"],
        va="center",
    )

    comparison_card(
        ax,
        x=0.04,
        y=0.194,
        width=0.445,
        height=0.175,
        tag="DIRECTION-ONLY BENCHMARK",
        metric="up to 82.65%",
        metric_note="accuracy",
        study="Lu et al., HS-Stroke, 2026 [4]",
        task="Left versus right motor imagery",
        design="14 patients | 278 sessions | cross-trial; no rest rejection",
        color=COLORS["gold"],
        soft=COLORS["soft_gold"],
    )
    comparison_card(
        ax,
        x=0.515,
        y=0.194,
        width=0.445,
        height=0.175,
        tag="ONLINE SAFETY BENCHMARK",
        metric="84.88% TPR",
        metric_note="0.52 FP/min",
        study="Niazi et al., 2022 [5]",
        task="Cue-gated movement-attempt onset detection",
        design="9 patients | online within patient; continuous safety endpoint",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )

    card(
        ax,
        0.04,
        0.055,
        0.92,
        0.105,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.022,
    )
    ax.text(
        0.065,
        0.134,
        "INTERPRETATION",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["teal"],
        va="center",
    )
    ax.text(
        0.205,
        0.134,
        "V7 is competitive with published cue-locked intent detection under a broader multi-dataset, patient-adapted design.",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.065,
        0.096,
        "STRENGTH",
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["gold"],
        va="center",
    )
    ax.text(
        0.14,
        0.096,
        "187 represented participants + explicit rest specificity",
        transform=ax.transAxes,
        fontsize=7.7,
        color="#DDE7EC",
        va="center",
    )
    ax.text(
        0.51,
        0.096,
        "NEXT PROOF",
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["coral"],
        va="center",
    )
    ax.text(
        0.60,
        0.096,
        "locked external cohort + false activations/min + latency",
        transform=ax.transAxes,
        fontsize=7.7,
        color="#DDE7EC",
        va="center",
    )

    ax.text(
        0.5,
        0.026,
        "BA, accuracy, TPR, and FP/min are not interchangeable. References and comparison limits accompany this figure.",
        transform=ax.transAxes,
        fontsize=7.0,
        color=COLORS["muted"],
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "08_v7_results_vs_current_studies"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi if suffix == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def create_vertical_comparison_visual(output_dir: Path, dpi: int) -> None:
    """Stack the original comparison-card design vertically."""
    fig = plt.figure(figsize=(8.2, 18.8), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.text(
        0.07,
        0.97,
        "CLOSEST PUBLISHED COMPARISONS",
        transform=ax.transAxes,
        fontsize=20.5,
        fontweight="bold",
        color=COLORS["muted"],
        va="center",
    )
    ax.text(
        0.07,
        0.938,
        "MOTOR INTENT / IMAGERY vs REST",
        transform=ax.transAxes,
        fontsize=14.0,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )

    def square_study_card(
        *,
        y: float,
        tag: str,
        metric: str,
        study: str,
        task: str,
        design: str,
        color: str,
        soft: str,
    ) -> None:
        x = 0.07
        width = 0.86
        height = 0.267
        card(
            ax,
            x,
            y,
            width,
            height,
            facecolor=COLORS["paper"],
            edgecolor=color,
            linewidth=1.45,
            radius=0.018,
        )
        card(
            ax,
            x + 0.025,
            y + height - 0.058,
            width - 0.05,
            0.038,
            facecolor=soft,
            edgecolor=color,
            linewidth=1.1,
            radius=0.012,
            zorder=3,
        )
        ax.text(
            x + width / 2,
            y + height - 0.039,
            tag,
            transform=ax.transAxes,
            fontsize=12.2,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            zorder=4,
        )
        ax.text(
            x + 0.035,
            y + height - 0.105,
            metric,
            transform=ax.transAxes,
            fontsize=33,
            fontweight="bold",
            color=color,
            va="center",
        )
        ax.text(
            x + width - 0.035,
            y + height - 0.105,
            "mean accuracy",
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
            color=COLORS["muted"],
            ha="right",
            va="center",
        )
        ax.text(
            x + 0.035,
            y + height - 0.154,
            study,
            transform=ax.transAxes,
            fontsize=16.0,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
        )
        ax.text(
            x + 0.035,
            y + height - 0.202,
            task,
            transform=ax.transAxes,
            fontsize=13.0,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
        )
        ax.text(
            x + 0.035,
            y + 0.025,
            design,
            transform=ax.transAxes,
            fontsize=12.0,
            fontweight="bold",
            color=COLORS["muted"],
            va="bottom",
            linespacing=1.3,
        )

    square_study_card(
        y=0.615,
        tag="CLOSEST HISTORICAL ENDPOINT",
        metric="74%",
        study="Ang et al., 2011 [1]",
        task="Affected-hand motor imagery versus rest",
        design="46 stroke patients | offline\nPatient-specific calibration",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )
    square_study_card(
        y=0.325,
        tag="HEALTHY-TO-STROKE TRANSFER",
        metric="71.15%",
        study="Nagarajan et al., 2024 [2]",
        task="Stroke motor imagery versus rest after transfer",
        design="71 stroke patients | healthy pretraining\nTarget-domain adaptation",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )
    square_study_card(
        y=0.035,
        tag="MULTIMODAL ATTEMPT DETECTION",
        metric="69.7%",
        study="Lopez-Larraz et al., 2025 [3]",
        task="Paralyzed-hand attempt versus rest using EEG + EMG",
        design="34 analyzed stroke patients\nWithin-patient block validation",
        color=COLORS["coral"],
        soft=COLORS["soft_coral"],
    )

    ax.text(
        0.5,
        0.012,
        "Context, not a leaderboard: metrics, modalities, calibration, and validation designs differ.",
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["muted"],
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "08b_v7_closest_studies_vertical"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi if suffix == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    create_visual(args.output_dir, args.dpi)
    create_vertical_comparison_visual(args.output_dir, args.dpi)
    print(f"[+] Saved study-comparison poster visuals to {args.output_dir}")


if __name__ == "__main__":
    main()
