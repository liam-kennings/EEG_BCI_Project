#!/usr/bin/env python3
"""Create one complete poster-ready visual for the V7 EEG BCI pipeline."""

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
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "complete_pipeline"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#17324D",
    "muted": "#34424E",
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
            "font.size": 12,
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


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    width: float = 1.8,
    mutation_scale: float = 13,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            color=color,
            linewidth=width,
            mutation_scale=mutation_scale,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            zorder=8,
        )
    )


def stage_header(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    number: str,
    title: str,
    color: str,
) -> None:
    ax.add_patch(
        Circle(
            (x + 0.021, y),
            0.0195,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            zorder=5,
        )
    )
    ax.text(
        x + 0.021,
        y,
        number,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        x + 0.05,
        y,
        title,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
        zorder=6,
    )


def pill(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    text: str,
    color: str,
    soft: str,
    fontsize: float = 10.5,
) -> None:
    card(
        ax,
        x,
        y,
        width,
        0.032,
        facecolor=soft,
        edgecolor=color,
        linewidth=1.0,
        radius=0.011,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + 0.016,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        zorder=5,
    )


def waveform(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 220)
    signal = (
        0.55 * np.sin(2 * np.pi * 8.0 * t)
        + 0.28 * np.sin(2 * np.pi * 14.0 * t + 0.5)
        + rng.normal(0.0, 0.17, t.size)
    )
    envelope = 1.0 - 0.38 * np.exp(-((t - 0.68) / 0.17) ** 2)
    signal *= envelope
    ax.plot(
        x + width * t,
        y + 0.5 * height + 0.33 * height * signal,
        transform=ax.transAxes,
        color=color,
        linewidth=1.15,
        clip_on=False,
        zorder=6,
    )


def create_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(16.0, 9.0), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    card(
        ax,
        0.035,
        0.902,
        0.105,
        0.05,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        radius=0.022,
    )
    ax.text(
        0.0875,
        0.927,
        "V7 PIPELINE",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )
    ax.text(
        0.165,
        0.951,
        "General EEG Model to Rapid Patient-Adapted Motor-Intent Detection",
        transform=ax.transAxes,
        fontsize=23.0,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.167,
        0.903,
        "Population learning and patient calibration remain separated; the device-facing output is strictly REST or INTENT",
        transform=ax.transAxes,
        fontsize=11.5,
        color=COLORS["muted"],
        va="top",
    )

    pill(
        ax,
        x=0.035,
        y=0.824,
        width=0.172,
        text="A  POPULATION DEVELOPMENT",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
        fontsize=10.8,
    )
    ax.plot(
        [0.218, 0.965],
        [0.84, 0.84],
        transform=ax.transAxes,
        color=COLORS["line"],
        linewidth=1.3,
        zorder=1,
    )

    top_y = 0.49
    top_h = 0.285
    top_cards = [
        (0.035, 0.19, COLORS["blue"]),
        (0.255, 0.19, COLORS["teal"]),
        (0.475, 0.245, COLORS["gold"]),
        (0.75, 0.215, COLORS["coral"]),
    ]
    for x, width, color in top_cards:
        card(
            ax,
            x,
            top_y,
            width,
            top_h,
            facecolor=COLORS["paper"],
            edgecolor=color,
            linewidth=1.35,
            radius=0.022,
        )

    stage_header(
        ax,
        x=0.05,
        y=0.742,
        number="01",
        title="MULTI-DATASET EEG",
        color=COLORS["blue"],
    )
    waveform(
        ax,
        x=0.055,
        y=0.675,
        width=0.15,
        height=0.045,
        color=COLORS["blue"],
        seed=8,
    )
    datasets = [
        ("PhysioNet", "109", COLORS["blue"]),
        ("OpenBMI", "29", COLORS["teal"]),
        ("Liu stroke", "49", COLORS["coral"]),
    ]
    for index, (name, count, color) in enumerate(datasets):
        y = 0.64 - index * 0.045
        ax.add_patch(
            Circle(
                (0.065, y),
                0.016,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=4,
            )
        )
        ax.text(
            0.065,
            y,
            count,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            0.085,
            y,
            name,
            transform=ax.transAxes,
            fontsize=11.0,
            fontweight="bold",
            va="center",
        )
    ax.text(
        0.055,
        0.496,
        "36,903 windows\n187 represented participants",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["muted"],
        linespacing=1.1,
        va="bottom",
    )

    stage_header(
        ax,
        x=0.27,
        y=0.742,
        number="02",
        title="NATIVE QC +\nHARMONIZATION",
        color=COLORS["teal"],
    )
    qc_steps = [
        "Detect and repair bad channels",
        "Common-average reference (CAR)",
        "Resample to 100 Hz",
        "Select 7 shared motor channels",
        "Preserve dataset identity",
    ]
    for index, label in enumerate(qc_steps):
        y = 0.686 - index * 0.038
        ax.add_patch(
            Circle(
                (0.282, y),
                0.007,
                transform=ax.transAxes,
                facecolor=COLORS["teal"],
                edgecolor=COLORS["teal"],
                zorder=4,
            )
        )
        ax.text(
            0.295,
            y,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold" if index in (0, 3) else "normal",
            color=COLORS["ink"],
            va="center",
        )
    ax.text(
        0.275,
        0.512,
        "FC3  FC4  C3  Cz  C4  CP3  CP4",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["teal"],
    )

    stage_header(
        ax,
        x=0.49,
        y=0.742,
        number="03",
        title="PARALLEL EEG REPRESENTATIONS",
        color=COLORS["gold"],
    )
    branches = [
        (
            0.665,
            "MRCP  0.1-4 Hz",
            "slow preparation + 5-seed FBCNet",
            COLORS["blue"],
            COLORS["soft_blue"],
        ),
        (
            0.605,
            "SMR  7-35 Hz",
            "mu/beta desynchronization statistics",
            COLORS["teal"],
            COLORS["soft_teal"],
        ),
        (
            0.545,
            "RIEMANNIAN COVARIANCE",
            "four-band sensorimotor spatial structure",
            COLORS["gold"],
            COLORS["soft_gold"],
        ),
    ]
    for y, title, note, color, soft in branches:
        card(
            ax,
            0.495,
            y,
            0.205,
            0.043,
            facecolor=soft,
            edgecolor=color,
            linewidth=1.0,
            radius=0.012,
            zorder=3,
        )
        ax.text(
            0.51,
            y + 0.028,
            title,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            va="center",
        )
        ax.text(
            0.51,
            y + 0.012,
            note,
            transform=ax.transAxes,
            fontsize=10.5,
            color=COLORS["muted"],
            va="center",
        )
    ax.text(
        0.495,
        0.512,
        "Quality / OOD evidence remains available for rejection",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
    )

    stage_header(
        ax,
        x=0.765,
        y=0.742,
        number="04",
        title="GENERAL INTENT MODEL",
        color=COLORS["coral"],
    )
    ax.text(
        0.777,
        0.695,
        "Controller-selected late fusion",
        transform=ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
    )
    weights = [
        ("MRCP + FBCNet", "60%", COLORS["blue"], COLORS["soft_blue"]),
        ("Riemannian", "25%", COLORS["gold"], COLORS["soft_gold"]),
        ("SMR statistics", "15%", COLORS["teal"], COLORS["soft_teal"]),
    ]
    for index, (label, value, color, soft) in enumerate(weights):
        y = 0.64 - index * 0.047
        card(
            ax,
            0.777,
            y,
            0.16,
            0.029,
            facecolor=soft,
            edgecolor=color,
            linewidth=0.9,
            radius=0.01,
            zorder=3,
        )
        ax.text(
            0.789,
            y + 0.0145,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
        )
        ax.text(
            0.925,
            y + 0.0145,
            value,
            transform=ax.transAxes,
            fontsize=10.8,
            fontweight="bold",
            color=color,
            ha="right",
            va="center",
        )
    ax.text(
        0.777,
        0.498,
        "Per-dataset Platt calibration\n+ operating threshold",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["muted"],
        linespacing=1.1,
        va="bottom",
    )

    arrow(ax, (0.226, 0.632), (0.253, 0.632))
    arrow(ax, (0.446, 0.632), (0.473, 0.632))
    arrow(ax, (0.721, 0.632), (0.748, 0.632))

    pill(
        ax,
        x=0.035,
        y=0.425,
        width=0.19,
        text="B  RAPID PATIENT ADAPTATION",
        color=COLORS["coral"],
        soft=COLORS["soft_coral"],
        fontsize=10.8,
    )
    ax.plot(
        [0.237, 0.965],
        [0.441, 0.441],
        transform=ax.transAxes,
        color=COLORS["line"],
        linewidth=1.3,
        zorder=1,
    )

    bottom_y = 0.135
    bottom_h = 0.24
    bottom_cards = [
        (0.035, 0.235, COLORS["coral"]),
        (0.315, 0.235, COLORS["teal"]),
        (0.595, 0.17, COLORS["blue"]),
        (0.81, 0.155, COLORS["gold"]),
    ]
    for x, width, color in bottom_cards:
        card(
            ax,
            x,
            bottom_y,
            width,
            bottom_h,
            facecolor=COLORS["paper"],
            edgecolor=color,
            linewidth=1.35,
            radius=0.022,
        )

    stage_header(
        ax,
        x=0.05,
        y=0.343,
        number="05",
        title="30-WINDOW CALIBRATION",
        color=COLORS["coral"],
    )
    ax.text(
        0.055,
        0.3,
        "Recording 0 only",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["coral"],
    )
    calibration = [
        ("10 REST", COLORS["muted"], COLORS["soft_gray"]),
        ("10 LEFT", COLORS["blue"], COLORS["soft_blue"]),
        ("10 RIGHT", COLORS["teal"], COLORS["soft_teal"]),
    ]
    for index, (label, color, soft) in enumerate(calibration):
        pill(
            ax,
            x=0.055 + index * 0.067,
            y=0.252,
            width=0.057,
            text=label,
            color=color,
            soft=soft,
            fontsize=10.5,
        )
    ax.text(
        0.055,
        0.218,
        "Auxiliary direction labels are internal only",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["muted"],
    )
    ax.text(
        0.055,
        0.176,
        "P(intent) = P(left) + P(right)",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["ink"],
    )

    stage_header(
        ax,
        x=0.33,
        y=0.343,
        number="06",
        title="PATIENT-ADAPTED DECODER",
        color=COLORS["teal"],
    )
    adapter_steps = [
        "General population probability",
        "+ patient-specific calibration evidence",
        "+ dataset-specific threshold",
    ]
    for index, label in enumerate(adapter_steps):
        y = 0.293 - index * 0.036
        ax.text(
            0.338,
            y,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold" if index == 0 else "normal",
            color=COLORS["ink"],
            va="center",
        )
    pill(
        ax,
        x=0.338,
        y=0.169,
        width=0.184,
        text="PERSONALIZED P(INTENT)",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
        fontsize=10.5,
    )

    stage_header(
        ax,
        x=0.61,
        y=0.343,
        number="07",
        title="BINARY SAFETY GATE",
        color=COLORS["blue"],
    )
    pill(
        ax,
        x=0.615,
        y=0.275,
        width=0.13,
        text="REST = NO COMMAND",
        color=COLORS["muted"],
        soft=COLORS["soft_gray"],
        fontsize=10.5,
    )
    pill(
        ax,
        x=0.615,
        y=0.222,
        width=0.13,
        text="INTENT = CONFIRM",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
        fontsize=10.5,
    )
    ax.text(
        0.68,
        0.181,
        "Known cue supplies movement",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
    )
    ax.text(
        0.68,
        0.158,
        "3-window confirmation prepared",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["muted"],
        ha="center",
    )

    stage_header(
        ax,
        x=0.825,
        y=0.343,
        number="08",
        title="LATER RECORDINGS",
        color=COLORS["gold"],
    )
    ax.text(
        0.832,
        0.306,
        "73.6%",
        transform=ax.transAxes,
        fontsize=22.0,
        fontweight="bold",
        color=COLORS["teal"],
        va="center",
    )
    ax.text(
        0.832,
        0.278,
        "balanced accuracy",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["muted"],
        va="center",
        ha="left",
    )
    result_rows = [
        ("Sensitivity", "53.5%"),
        ("Specificity", "93.7%"),
        ("AUROC", "0.896"),
    ]
    for index, (label, value) in enumerate(result_rows):
        y = 0.246 - index * 0.030
        ax.text(
            0.832,
            y,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            color=COLORS["muted"],
            va="center",
        )
        ax.text(
            0.942,
            y,
            value,
            transform=ax.transAxes,
            fontsize=10.8,
            fontweight="bold",
            color=COLORS["ink"],
            ha="right",
            va="center",
        )
    ax.text(
        0.832,
        0.138,
        "Development-observed\nCue-locked",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        linespacing=1.05,
        va="bottom",
    )

    arrow(ax, (0.271, 0.255), (0.313, 0.255))
    arrow(ax, (0.551, 0.255), (0.593, 0.255))
    arrow(ax, (0.766, 0.255), (0.808, 0.255))
    arrow(
        ax,
        (0.855, 0.488),
        (0.535, 0.383),
        color=COLORS["coral"],
        width=2.0,
        connectionstyle="arc3,rad=0.17",
    )
    ax.text(
        0.714,
        0.405,
        "general model initializes patient decoder",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
        rotation=-7,
    )

    card(
        ax,
        0.035,
        0.035,
        0.93,
        0.062,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.018,
    )
    ax.text(
        0.055,
        0.066,
        "VALIDATION BOUNDARY",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["teal"],
        va="center",
    )
    ax.text(
        0.185,
        0.066,
        "Develop model on population roles  |  Calibrate on recording 0  |  Evaluate on later recordings only",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.785,
        0.066,
        "LOCKED EXTERNAL: NOT YET AVAILABLE",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "09_v7_complete_pipeline"
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
    print(f"[+] Saved complete V7 pipeline visual to {args.output_dir}")


if __name__ == "__main__":
    main()
