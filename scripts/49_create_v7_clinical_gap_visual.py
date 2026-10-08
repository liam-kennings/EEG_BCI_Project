#!/usr/bin/env python3
"""Create a poster-ready current-methods to project-need visual for V7."""

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
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "clinical_gap"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#111111",
    "navy": "#173A59",
    "blue": "#2D7FA8",
    "teal": "#248B7F",
    "gold": "#E9A437",
    "coral": "#D96B3B",
    "red": "#B94A48",
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


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["navy"],
    width: float = 2.2,
    scale: float = 17,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            color=color,
            linewidth=width,
            mutation_scale=scale,
            zorder=20,
        )
    )


def section_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    color: str,
    soft: str,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=2.4,
        radius=0.022,
    )
    rounded_box(
        ax,
        x + 0.018,
        y + height - 0.060,
        width - 0.036,
        0.047,
        facecolor=soft,
        edgecolor=color,
        linewidth=1.5,
        radius=0.012,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height - 0.0365,
        title,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=6,
    )


def draw_cue(ax: plt.Axes, cx: float, cy: float) -> None:
    rounded_box(
        ax,
        cx - 0.050,
        cy - 0.044,
        0.100,
        0.088,
        facecolor=COLORS["soft_blue"],
        edgecolor=COLORS["blue"],
        linewidth=1.7,
        radius=0.010,
        zorder=5,
    )
    ax.add_patch(
        Polygon(
            [
                (cx - 0.014, cy - 0.022),
                (cx - 0.014, cy + 0.022),
                (cx + 0.025, cy),
            ],
            closed=True,
            transform=ax.transAxes,
            facecolor=COLORS["blue"],
            edgecolor=COLORS["blue"],
            zorder=7,
        )
    )


def draw_eeg(ax: plt.Axes, x: float, y: float, width: float) -> None:
    t = np.linspace(0.0, 1.0, 180)
    wave = (
        0.52 * np.sin(2 * np.pi * 8 * t)
        + 0.24 * np.sin(2 * np.pi * 14 * t + 0.6)
        + 0.10 * np.sin(2 * np.pi * 23 * t)
    )
    ax.plot(
        x + width * t,
        y + 0.023 * wave,
        transform=ax.transAxes,
        color=COLORS["teal"],
        linewidth=2.0,
        zorder=7,
    )


def draw_classifier(ax: plt.Axes, cx: float, cy: float) -> None:
    nodes = [
        (cx - 0.035, cy + 0.032),
        (cx - 0.035, cy),
        (cx - 0.035, cy - 0.032),
        (cx + 0.005, cy + 0.020),
        (cx + 0.005, cy - 0.020),
        (cx + 0.045, cy),
    ]
    edges = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)]
    for a, b in edges:
        ax.plot(
            [nodes[a][0], nodes[b][0]],
            [nodes[a][1], nodes[b][1]],
            transform=ax.transAxes,
            color="#8A969D",
            linewidth=1.0,
            zorder=5,
        )
    for index, (nx, ny) in enumerate(nodes):
        ax.add_patch(
            Circle(
                (nx, ny),
                0.009,
                transform=ax.transAxes,
                facecolor=COLORS["coral"] if index == 5 else COLORS["paper"],
                edgecolor=COLORS["navy"],
                linewidth=1.4,
                zorder=7,
            )
        )


def method_step(
    ax: plt.Axes,
    *,
    cx: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    ax.text(
        cx,
        0.722,
        title,
        transform=ax.transAxes,
        fontsize=13.2,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
    )
    ax.text(
        cx,
        0.682,
        subtitle,
        transform=ax.transAxes,
        fontsize=10.7,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.1,
    )


def problem_box(
    ax: plt.Axes,
    *,
    x: float,
    title: str,
    body: str,
    color: str,
    soft: str,
) -> None:
    rounded_box(
        ax,
        x,
        0.378,
        0.275,
        0.160,
        facecolor=soft,
        edgecolor=color,
        linewidth=1.8,
        radius=0.015,
        zorder=5,
    )
    ax.add_patch(
        Circle(
            (x + 0.1375, 0.507),
            0.020,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            zorder=7,
        )
    )
    ax.text(
        x + 0.1375,
        0.507,
        "!",
        transform=ax.transAxes,
        fontsize=14,
        color="white",
        ha="center",
        va="center",
        zorder=8,
    )
    ax.text(
        x + 0.1375,
        0.466,
        title,
        transform=ax.transAxes,
        fontsize=13.6,
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=8,
    )
    ax.text(
        x + 0.1375,
        0.414,
        body,
        transform=ax.transAxes,
        fontsize=10.4,
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.2,
        zorder=8,
    )


def response_step(
    ax: plt.Axes,
    *,
    x: float,
    number: str,
    title: str,
    body: str,
    color: str,
    soft: str,
) -> None:
    rounded_box(
        ax,
        x,
        0.087,
        0.245,
        0.108,
        facecolor=soft,
        edgecolor=color,
        linewidth=1.8,
        radius=0.015,
        zorder=5,
    )
    ax.add_patch(
        Circle(
            (x + 0.034, 0.166),
            0.020,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            zorder=7,
        )
    )
    ax.text(
        x + 0.034,
        0.166,
        number,
        transform=ax.transAxes,
        fontsize=12,
        color="white",
        ha="center",
        va="center",
        zorder=8,
    )
    ax.text(
        x + 0.1225,
        0.168,
        title,
        transform=ax.transAxes,
        fontsize=12.8,
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=8,
    )
    ax.text(
        x + 0.1225,
        0.123,
        body,
        transform=ax.transAxes,
        fontsize=10.3,
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.2,
        zorder=8,
    )


def create_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(9.0, 9.2), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.text(
        0.50,
        0.970,
        "THE CLINICAL TRANSLATION GAP",
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="top",
    )
    ax.text(
        0.50,
        0.928,
        "Why conventional offline EEG decoding is not yet reliable device control",
        transform=ax.transAxes,
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="top",
    )

    section_card(
        ax,
        x=0.045,
        y=0.625,
        width=0.91,
        height=0.245,
        title="COMMON RESEARCH APPROACH",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )
    draw_cue(ax, 0.160, 0.785)
    draw_eeg(ax, 0.315, 0.785, 0.130)
    draw_classifier(ax, 0.570, 0.785)
    rounded_box(
        ax,
        0.735,
        0.744,
        0.150,
        0.082,
        facecolor=COLORS["soft_gray"],
        edgecolor=COLORS["navy"],
        linewidth=1.7,
        radius=0.012,
        zorder=5,
    )
    for index, (label, color) in enumerate(
        (("REST", COLORS["navy"]), ("LEFT", COLORS["blue"]), ("RIGHT", COLORS["teal"]))
    ):
        ax.text(
            0.810,
            0.804 - index * 0.020,
            label,
            transform=ax.transAxes,
            fontsize=9.2,
            color=color,
            ha="center",
            va="center",
            zorder=7,
        )
    arrow(ax, (0.220, 0.785), (0.300, 0.785))
    arrow(ax, (0.455, 0.785), (0.500, 0.785))
    arrow(ax, (0.635, 0.785), (0.720, 0.785))
    method_step(
        ax,
        cx=0.160,
        title="CONTROLLED CUES",
        subtitle="Known onset\nand task",
        color=COLORS["blue"],
    )
    method_step(
        ax,
        cx=0.380,
        title="CUE-LOCKED EEG",
        subtitle="Clean offline\ntrial windows",
        color=COLORS["teal"],
    )
    method_step(
        ax,
        cx=0.570,
        title="CLASSIFIER",
        subtitle="Often patient-specific\nor cohort-trained",
        color=COLORS["coral"],
    )
    method_step(
        ax,
        cx=0.810,
        title="FORCED DECISION",
        subtitle="Average accuracy\nas primary metric",
        color=COLORS["navy"],
    )

    section_card(
        ax,
        x=0.045,
        y=0.325,
        width=0.91,
        height=0.260,
        title="WHY THIS BREAKS DOWN IN REHABILITATION",
        color=COLORS["coral"],
        soft=COLORS["soft_coral"],
    )
    problem_box(
        ax,
        x=0.075,
        title="WEAK SIGNAL",
        body="Intent can be faint\nand masked by\nnoise or artifacts.",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )
    problem_box(
        ax,
        x=0.3625,
        title="DOMAIN SHIFT",
        body="Patterns change across\npeople, sessions, devices,\nand stroke status.",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )
    problem_box(
        ax,
        x=0.650,
        title="SAFETY GAP",
        body="Forced predictions can\nturn rest into a command;\naccuracy can hide false activations.",
        color=COLORS["gold"],
        soft=COLORS["soft_gold"],
    )

    section_card(
        ax,
        x=0.045,
        y=0.070,
        width=0.91,
        height=0.215,
        title="THIS PROJECT: FROM GENERAL MODEL TO SAFE CONFIRMATION",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )
    response_step(
        ax,
        x=0.075,
        number="1",
        title="GENERALIZE",
        body="Learn shared structure\nacross multiple EEG\ndatasets.",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )
    response_step(
        ax,
        x=0.3775,
        number="2",
        title="PERSONALIZE",
        body="Adapt the population\nmodel with a brief\npatient calibration.",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )
    response_step(
        ax,
        x=0.680,
        number="3",
        title="PROTECT",
        body="Confirm REST or INTENT;\nreject weak evidence\nbefore device action.",
        color=COLORS["gold"],
        soft=COLORS["soft_gold"],
    )
    arrow(ax, (0.325, 0.141), (0.365, 0.141), color=COLORS["navy"])
    arrow(ax, (0.628, 0.141), (0.668, 0.141), color=COLORS["navy"])

    rounded_box(
        ax,
        0.075,
        0.015,
        0.85,
        0.040,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        linewidth=1.8,
        radius=0.013,
        zorder=5,
    )
    ax.text(
        0.50,
        0.035,
        "PROJECT AIM: RECOVER WEAK INTENT WITHOUT SACRIFICING SAFE REST REJECTION",
        transform=ax.transAxes,
        fontsize=11.3,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=7,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "11_v7_clinical_translation_gap"
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
    output_dir = args.output_dir.expanduser().resolve()
    create_visual(output_dir, args.dpi)
    print(f"Saved clinical-gap visual to {output_dir}")


if __name__ == "__main__":
    main()
