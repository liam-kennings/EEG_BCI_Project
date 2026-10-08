#!/usr/bin/env python3
"""Create a poster-ready roadmap for prospective EEG BCI development."""

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
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "next_steps"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#111111",
    "navy": "#173A59",
    "blue": "#2D7FA8",
    "teal": "#248B7F",
    "gold": "#E9A437",
    "coral": "#D96B3B",
    "line": "#C8D1D5",
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
            "font.size": 14,
            "font.weight": "bold",
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 2.0,
    radius: float = 0.018,
    zorder: float = 2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def flow_arrow(ax: plt.Axes, y_start: float, y_end: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (0.50, y_start),
            (0.50, y_end),
            transform=ax.transAxes,
            arrowstyle="-|>",
            color=COLORS["navy"],
            linewidth=3.0,
            mutation_scale=22,
            zorder=20,
        )
    )


def stage_card(
    ax: plt.Axes,
    *,
    y: float,
    height: float,
    number: str,
    title: str,
    color: str,
    soft_color: str,
) -> None:
    rounded_box(
        ax,
        0.045,
        y,
        0.91,
        height,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=2.5,
        radius=0.022,
    )
    rounded_box(
        ax,
        0.065,
        y + height - 0.066,
        0.87,
        0.055,
        facecolor=soft_color,
        edgecolor=color,
        linewidth=1.7,
        radius=0.014,
        zorder=4,
    )
    ax.add_patch(
        Circle(
            (0.105, y + height - 0.0385),
            0.027,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=6,
        )
    )
    ax.text(
        0.105,
        y + height - 0.0385,
        number,
        transform=ax.transAxes,
        color="white",
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        0.15,
        y + height - 0.0385,
        title,
        transform=ax.transAxes,
        color=COLORS["ink"],
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=7,
    )


def draw_protocol_icon(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    rounded_box(
        ax,
        cx - 0.080,
        cy - 0.054,
        0.160,
        0.108,
        facecolor=COLORS["soft_blue"],
        edgecolor=color,
        linewidth=2.0,
        radius=0.012,
        zorder=5,
    )
    ax.text(
        cx,
        cy + 0.030,
        "CUE",
        transform=ax.transAxes,
        fontsize=13,
        color=color,
        ha="center",
        va="center",
        zorder=7,
    )
    ax.plot(
        [cx - 0.050, cx - 0.012, cx - 0.012, cx + 0.050],
        [cy - 0.025, cy - 0.025, cy + 0.008, cy + 0.008],
        transform=ax.transAxes,
        color=COLORS["ink"],
        linewidth=2.5,
        zorder=7,
    )
    ax.text(
        cx - 0.048,
        cy - 0.044,
        "REST",
        transform=ax.transAxes,
        fontsize=9.5,
        color=COLORS["ink"],
        ha="center",
        zorder=7,
    )
    ax.text(
        cx + 0.032,
        cy + 0.017,
        "INTENT",
        transform=ax.transAxes,
        fontsize=9.5,
        color=COLORS["ink"],
        ha="center",
        zorder=7,
    )


def draw_dataset_icon(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    width, height = 0.100, 0.078
    ax.add_patch(
        Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            transform=ax.transAxes,
            facecolor=COLORS["soft_teal"],
            edgecolor=color,
            linewidth=2.0,
            zorder=5,
        )
    )
    for offset in (-height / 2, 0.0, height / 2):
        ax.add_patch(
            Ellipse(
                (cx, cy + offset),
                width,
                0.026,
                transform=ax.transAxes,
                facecolor=COLORS["soft_teal"] if offset else "none",
                edgecolor=color,
                linewidth=2.0,
                zorder=6,
            )
        )
    for dx in (-0.062, 0.062):
        ax.add_patch(
            Circle(
                (cx + dx, cy + 0.016),
                0.015,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=6,
            )
        )
        ax.add_patch(
            Arc(
                (cx + dx, cy - 0.022),
                0.050,
                0.062,
                theta1=15,
                theta2=165,
                transform=ax.transAxes,
                color=color,
                linewidth=4.0,
                zorder=6,
            )
        )


def signal_wave(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    color: str,
    frequency: float,
    amplitude: float,
) -> None:
    t = np.linspace(0.0, 1.0, 100)
    signal = np.sin(2.0 * np.pi * frequency * t) * (0.7 + 0.3 * np.sin(2 * np.pi * t))
    ax.plot(
        x + width * t,
        y + amplitude * signal,
        transform=ax.transAxes,
        color=color,
        linewidth=2.0,
        zorder=8,
    )


def draw_multimodal_icon(ax: plt.Axes, cx: float, cy: float) -> None:
    ax.add_patch(
        Circle(
            (cx, cy),
            0.043,
            transform=ax.transAxes,
            facecolor=COLORS["soft_blue"],
            edgecolor=COLORS["blue"],
            linewidth=2.2,
            zorder=6,
        )
    )
    ax.text(
        cx,
        cy,
        "EEG",
        transform=ax.transAxes,
        fontsize=12.5,
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=8,
    )
    labels = [
        (cx - 0.073, cy + 0.040, "EMG", COLORS["coral"]),
        (cx + 0.073, cy + 0.040, "fNIRS", COLORS["teal"]),
        (cx, cy - 0.060, "IMU", COLORS["gold"]),
    ]
    for lx, ly, label, color in labels:
        ax.plot(
            [cx, lx],
            [cy, ly],
            transform=ax.transAxes,
            color=color,
            linewidth=2.0,
            zorder=5,
        )
        rounded_box(
            ax,
            lx - 0.038,
            ly - 0.016,
            0.076,
            0.032,
            facecolor=COLORS["paper"],
            edgecolor=color,
            linewidth=1.8,
            radius=0.010,
            zorder=7,
        )
        ax.text(
            lx,
            ly,
            label,
            transform=ax.transAxes,
            fontsize=9.2,
            color=COLORS["ink"],
            ha="center",
            va="center",
            zorder=9,
        )


def draw_closed_loop_icon(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    shield = np.array(
        [
            [cx, cy + 0.052],
            [cx + 0.046, cy + 0.033],
            [cx + 0.037, cy - 0.027],
            [cx, cy - 0.058],
            [cx - 0.037, cy - 0.027],
            [cx - 0.046, cy + 0.033],
        ]
    )
    ax.add_patch(
        Polygon(
            shield,
            closed=True,
            transform=ax.transAxes,
            facecolor=COLORS["soft_coral"],
            edgecolor=color,
            linewidth=2.2,
            zorder=6,
        )
    )
    ax.plot(
        [cx - 0.021, cx - 0.004, cx + 0.026],
        [cy - 0.002, cy - 0.020, cy + 0.019],
        transform=ax.transAxes,
        color=COLORS["ink"],
        linewidth=4.0,
        solid_capstyle="round",
        zorder=8,
    )
    ax.add_patch(
        FancyArrowPatch(
            (cx + 0.049, cy),
            (cx + 0.100, cy),
            transform=ax.transAxes,
            arrowstyle="-|>",
            color=COLORS["ink"],
            linewidth=2.5,
            mutation_scale=17,
            zorder=8,
        )
    )
    rounded_box(
        ax,
        cx + 0.100,
        cy - 0.031,
        0.058,
        0.062,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=2.0,
        radius=0.010,
        zorder=6,
    )
    signal_wave(ax, cx + 0.108, cy, 0.042, color, 3.5, 0.010)


def metric_pill(
    ax: plt.Axes, x: float, y: float, width: float, text: str, color: str
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        0.035,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=1.4,
        radius=0.009,
        zorder=7,
    )
    ax.text(
        x + width / 2,
        y + 0.0175,
        text,
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=9,
    )


def create_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(8.6, 10.2), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.text(
        0.50,
        0.965,
        "ROADMAP TO PROSPECTIVE VALIDATION",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="top",
    )
    ax.text(
        0.50,
        0.925,
        "Move from cue-locked analysis to a safe, personalized, real-time BCI",
        transform=ax.transAxes,
        fontsize=14.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="top",
    )

    stages = [
        (0.725, 0.170, "1", "DEFINE THE EXPERIMENTAL PROTOCOL", COLORS["blue"], COLORS["soft_blue"]),
        (0.535, 0.170, "2", "BUILD A PROSPECTIVE PATIENT DATASET", COLORS["teal"], COLORS["soft_teal"]),
        (0.345, 0.170, "3", "EXPAND COMPLEMENTARY SENSING", COLORS["gold"], COLORS["soft_gold"]),
        (0.155, 0.170, "4", "VALIDATE CLOSED-LOOP SAFETY", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for y, height, number, title, color, soft in stages:
        stage_card(
            ax,
            y=y,
            height=height,
            number=number,
            title=title,
            color=color,
            soft_color=soft,
        )

    draw_protocol_icon(ax, 0.185, 0.772, COLORS["blue"])
    ax.text(
        0.335,
        0.797,
        "Standardize when motor intent should occur.",
        transform=ax.transAxes,
        fontsize=15.5,
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.335,
        0.752,
        "Randomized visual/audio cues + rest/no-control periods\nRepeated sessions in healthy and stroke participants",
        transform=ax.transAxes,
        fontsize=13.2,
        color=COLORS["ink"],
        ha="left",
        va="center",
        linespacing=1.35,
    )

    draw_dataset_icon(ax, 0.185, 0.582, COLORS["teal"])
    ax.text(
        0.335,
        0.607,
        "Use early calibration; lock later sessions for testing.",
        transform=ax.transAxes,
        fontsize=14.3,
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.335,
        0.562,
        "Store raw EEG, cue timing, artifacts, function scores,\nand recovery outcomes across sessions",
        transform=ax.transAxes,
        fontsize=13.2,
        color=COLORS["ink"],
        ha="left",
        va="center",
        linespacing=1.35,
    )

    draw_multimodal_icon(ax, 0.185, 0.392)
    ax.text(
        0.335,
        0.417,
        "Keep EEG central; use other signals to verify intent.",
        transform=ax.transAxes,
        fontsize=14.2,
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.335,
        0.372,
        "EMG: attempt ground truth  |  fNIRS: cortical evidence\nIMU / force: movement and device-response verification",
        transform=ax.transAxes,
        fontsize=11.8,
        color=COLORS["ink"],
        ha="left",
        va="center",
        linespacing=1.35,
    )

    draw_closed_loop_icon(ax, 0.170, 0.205, COLORS["coral"])
    ax.text(
        0.335,
        0.230,
        "Test confirmation, abstention, and device control.",
        transform=ax.transAxes,
        fontsize=14.2,
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    ax.text(
        0.335,
        0.198,
        "Measure real-world safety and usability:",
        transform=ax.transAxes,
        fontsize=11.8,
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    metric_pill(ax, 0.335, 0.165, 0.120, "FA / MIN", COLORS["coral"])
    metric_pill(ax, 0.465, 0.165, 0.145, "MISSED INTENT", COLORS["teal"])
    metric_pill(ax, 0.620, 0.165, 0.105, "LATENCY", COLORS["blue"])
    metric_pill(ax, 0.735, 0.165, 0.175, "CALIBRATION TIME", COLORS["gold"])

    flow_arrow(ax, 0.718, 0.710)
    flow_arrow(ax, 0.528, 0.520)
    flow_arrow(ax, 0.338, 0.330)

    rounded_box(
        ax,
        0.045,
        0.035,
        0.91,
        0.085,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        linewidth=2.0,
        radius=0.020,
        zorder=5,
    )
    ax.text(
        0.50,
        0.089,
        "TRANSLATIONAL GOAL",
        transform=ax.transAxes,
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["gold"],
        ha="center",
        va="center",
        zorder=7,
    )
    ax.text(
        0.50,
        0.058,
        "Rapidly personalize a general EEG model,\nthen confirm intent safely before assistive action",
        transform=ax.transAxes,
        fontsize=11.8,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        linespacing=1.1,
        zorder=7,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "10_v7_next_steps_roadmap"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi if suffix == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.10,
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    create_visual(args.output_dir.expanduser().resolve(), args.dpi)
    print(f"Saved next-steps roadmap to {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
