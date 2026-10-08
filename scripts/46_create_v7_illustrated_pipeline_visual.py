#!/usr/bin/env python3
"""Create an illustrated, poster-ready overview of the V7 EEG BCI pipeline."""

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
    Rectangle,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "illustrated_pipeline"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#111111",
    "slate": "#292929",
    "line": "#C6D0D5",
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


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.2,
    radius: float = 0.012,
    zorder: float = 2,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.005,rounding_size={radius}",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["slate"],
    width: float = 1.8,
    mutation_scale: float = 14,
    connectionstyle: str = "arc3",
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
            zorder=20,
        )
    )


def stage_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    number: str,
    title: str,
    color: str,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=1.4,
        radius=0.016,
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + height - 0.064),
            width,
            0.064,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + height - 0.035),
            width,
            0.035,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=3,
        )
    )
    ax.text(
        x + 0.012,
        y + height - 0.031,
        number,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="white",
        ha="left",
        va="center",
        zorder=5,
    )
    ax.text(
        x + width / 2 + 0.006,
        y + height - 0.031,
        title,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        linespacing=1.0,
        zorder=5,
    )


def pill(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    soft: str,
    fontsize: float = 10.5,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=soft,
        edgecolor=color,
        linewidth=1.0,
        radius=0.009,
        zorder=5,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        linespacing=1.0,
        zorder=6,
    )


def draw_database(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    width, height = 0.052, 0.075
    ax.add_patch(
        Rectangle(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            transform=ax.transAxes,
            facecolor=COLORS["soft_blue"],
            edgecolor=color,
            linewidth=1.6,
            zorder=5,
        )
    )
    for offset in (-height / 2, 0.0, height / 2):
        ax.add_patch(
            Ellipse(
                (cx, cy + offset),
                width,
                0.022,
                transform=ax.transAxes,
                facecolor=COLORS["soft_blue"] if offset else "none",
                edgecolor=color,
                linewidth=1.6,
                zorder=6,
            )
        )


def draw_head(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    ax.add_patch(
        Circle(
            (cx, cy),
            0.037,
            transform=ax.transAxes,
            facecolor=COLORS["paper"],
            edgecolor=color,
            linewidth=1.6,
            zorder=5,
        )
    )
    ax.add_patch(
        Arc(
            (cx, cy - 0.004),
            0.064,
            0.055,
            theta1=205,
            theta2=335,
            transform=ax.transAxes,
            color=color,
            linewidth=1.2,
            zorder=6,
        )
    )
    points = [
        (-0.020, 0.014),
        (0.000, 0.020),
        (0.020, 0.014),
        (-0.024, -0.005),
        (0.000, 0.000),
        (0.024, -0.005),
        (-0.014, -0.023),
        (0.014, -0.023),
    ]
    for dx, dy in points:
        ax.add_patch(
            Circle(
                (cx + dx, cy + dy),
                0.0042,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=7,
            )
        )


def draw_timeline(ax: plt.Axes, x: float, y: float, width: float) -> None:
    ax.plot(
        [x, x + width],
        [y, y],
        transform=ax.transAxes,
        color=COLORS["ink"],
        linewidth=1.5,
        zorder=5,
    )
    flow_arrow(ax, (x + width - 0.012, y), (x + width + 0.006, y), width=1.3)
    cue_x = x + 0.025
    ax.plot(
        [cue_x, cue_x],
        [y - 0.018, y + 0.052],
        transform=ax.transAxes,
        color=COLORS["coral"],
        linewidth=1.3,
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(
        Rectangle(
            (cue_x, y + 0.005),
            width - 0.035,
            0.034,
            transform=ax.transAxes,
            facecolor=COLORS["soft_gold"],
            edgecolor=COLORS["gold"],
            linewidth=1.0,
            zorder=4,
        )
    )
    ax.text(
        cue_x,
        y + 0.058,
        "CUE",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="bottom",
    )
    ax.text(
        cue_x,
        y - 0.024,
        "0 s",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["slate"],
        ha="center",
        va="top",
    )
    ax.text(
        x + width - 0.006,
        y - 0.024,
        "3 s",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["slate"],
        ha="right",
        va="top",
    )


def draw_signal(ax: plt.Axes, x: float, y: float, width: float, height: float, color: str) -> None:
    t = np.linspace(0.0, 1.0, 120)
    signal = 0.55 * np.sin(2 * np.pi * 7 * t) + 0.25 * np.sin(2 * np.pi * 13 * t + 0.7)
    envelope = 1.0 - 0.42 * np.exp(-((t - 0.70) / 0.20) ** 2)
    ax.plot(
        x + width * t,
        y + height / 2 + 0.32 * height * signal * envelope,
        transform=ax.transAxes,
        color=color,
        linewidth=1.3,
        zorder=8,
    )


def draw_covariance(ax: plt.Axes, x: float, y: float, size: float) -> None:
    values = np.array(
        [
            [0.95, 0.58, 0.35, 0.20],
            [0.58, 0.90, 0.52, 0.28],
            [0.35, 0.52, 0.88, 0.62],
            [0.20, 0.28, 0.62, 0.92],
        ]
    )
    cell = size / 4
    for row in range(4):
        for col in range(4):
            alpha = 0.18 + 0.72 * values[row, col]
            ax.add_patch(
                Rectangle(
                    (x + col * cell, y + (3 - row) * cell),
                    cell,
                    cell,
                    transform=ax.transAxes,
                    facecolor=COLORS["gold"],
                    edgecolor=COLORS["paper"],
                    linewidth=0.5,
                    alpha=alpha,
                    zorder=8,
                )
            )


def draw_network(ax: plt.Axes, cx: float, cy: float) -> None:
    layers = [
        [(cx - 0.036, cy + 0.028), (cx - 0.036, cy), (cx - 0.036, cy - 0.028)],
        [(cx, cy + 0.042), (cx, cy + 0.014), (cx, cy - 0.014), (cx, cy - 0.042)],
        [(cx + 0.036, cy + 0.022), (cx + 0.036, cy - 0.022)],
    ]
    for left, right in zip(layers[:-1], layers[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    transform=ax.transAxes,
                    color=COLORS["line"],
                    linewidth=0.8,
                    zorder=4,
                )
    layer_colors = (COLORS["blue"], COLORS["coral"], COLORS["teal"])
    for layer, color in zip(layers, layer_colors):
        for x, y in layer:
            ax.add_patch(
                Circle(
                    (x, y),
                    0.006,
                    transform=ax.transAxes,
                    facecolor=COLORS["paper"],
                    edgecolor=color,
                    linewidth=1.3,
                    zorder=6,
                )
            )


def draw_person(ax: plt.Axes, cx: float, cy: float, color: str) -> None:
    ax.add_patch(
        Circle(
            (cx, cy + 0.025),
            0.018,
            transform=ax.transAxes,
            facecolor=COLORS["soft_teal"],
            edgecolor=color,
            linewidth=1.4,
            zorder=6,
        )
    )
    ax.add_patch(
        Arc(
            (cx, cy - 0.018),
            0.082,
            0.075,
            theta1=15,
            theta2=165,
            transform=ax.transAxes,
            color=color,
            linewidth=2.2,
            zorder=6,
        )
    )
    ax.add_patch(
        Circle(
            (cx + 0.040, cy + 0.024),
            0.012,
            transform=ax.transAxes,
            facecolor=COLORS["soft_gold"],
            edgecolor=COLORS["gold"],
            linewidth=1.1,
            zorder=7,
        )
    )
    ax.text(
        cx + 0.040,
        cy + 0.024,
        "30",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=8,
    )


def output_panel(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    color: str,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=COLORS["paper"],
        edgecolor=color,
        linewidth=1.1,
        radius=0.012,
    )
    ax.text(
        x + 0.012,
        y + height - 0.024,
        title,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        zorder=5,
    )


def create_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(16.0, 7.2), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.text(
        0.5,
        0.965,
        "V7 Patient-Adaptive EEG Motor-Intent Pipeline",
        transform=ax.transAxes,
        fontsize=23.0,
        fontweight="bold",
        color=COLORS["navy"],
        ha="center",
        va="top",
    )
    ax.text(
        0.5,
        0.914,
        "Population learning, brief patient calibration, and a conservative REST / INTENT decision remain separated",
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="top",
    )

    widths = [0.112, 0.126, 0.126, 0.164, 0.124, 0.124, 0.126]
    gap = 0.008
    xs = [0.025]
    for width in widths[:-1]:
        xs.append(xs[-1] + width + gap)

    stage_y, stage_h = 0.285, 0.575
    stage_data = [
        ("01", "DATASETS", COLORS["blue"]),
        ("02", "NATIVE QC", COLORS["teal"]),
        ("03", "WINDOWING", COLORS["gold"]),
        ("04", "EEG\nREPRESENTATIONS", COLORS["blue"]),
        ("05", "GENERAL\nMODEL", COLORS["coral"]),
        ("06", "PATIENT\nCALIBRATION", COLORS["teal"]),
        ("07", "INTENT GATE", COLORS["gold"]),
    ]
    for x, width, (number, title, color) in zip(xs, widths, stage_data):
        stage_card(
            ax,
            x=x,
            y=stage_y,
            width=width,
            height=stage_h,
            number=number,
            title=title,
            color=color,
        )

    for index in range(len(xs) - 1):
        flow_arrow(
            ax,
            (xs[index] + widths[index] - 0.002, 0.567),
            (xs[index + 1] + 0.003, 0.567),
        )

    # 01 - datasets
    draw_database(ax, xs[0] + widths[0] / 2, 0.735, COLORS["blue"])
    datasets = [
        ("109", "PhysioNet", COLORS["blue"]),
        ("29", "OpenBMI", COLORS["teal"]),
        ("49", "Liu stroke", COLORS["coral"]),
    ]
    for index, (count, name, color) in enumerate(datasets):
        y = 0.655 - index * 0.055
        ax.add_patch(
            Circle(
                (xs[0] + 0.027, y),
                0.015,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=5,
            )
        )
        ax.text(
            xs[0] + 0.027,
            y,
            count,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            xs[0] + 0.048,
            y,
            name,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )
    pill(
        ax,
        x=xs[0] + 0.011,
        y=0.414,
        width=widths[0] - 0.022,
        height=0.047,
        text="187 PARTICIPANTS",
        color=COLORS["blue"],
        soft=COLORS["soft_blue"],
    )
    ax.text(
        xs[0] + widths[0] / 2,
        0.367,
        "36,903 cue-locked\nEEG windows",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
        linespacing=1.12,
    )

    # 02 - native QC and harmonization
    draw_head(ax, xs[1] + widths[1] / 2, 0.735, COLORS["teal"])
    qc_steps = [
        "Repair bad channels",
        "CAR re-reference",
        "Resample to 100 Hz",
        "Select 7 motor channels",
        "Preserve dataset ID",
    ]
    for index, label in enumerate(qc_steps):
        y = 0.655 - index * 0.052
        ax.add_patch(
            Circle(
                (xs[1] + 0.016, y),
                0.0045,
                transform=ax.transAxes,
                facecolor=COLORS["teal"],
                edgecolor=COLORS["teal"],
                zorder=5,
            )
        )
        ax.text(
            xs[1] + 0.027,
            y,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold" if index in (0, 3) else "normal",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )
    ax.text(
        xs[1] + widths[1] / 2,
        0.350,
        "FC3  FC4  C3  Cz\nC4  CP3  CP4",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["teal"],
        ha="center",
        va="center",
        linespacing=1.15,
    )

    # 03 - cue-locked windows and roles
    draw_timeline(ax, xs[2] + 0.018, 0.690, widths[2] - 0.042)
    ax.text(
        xs[2] + widths[2] / 2,
        0.615,
        "3 s cue-locked window",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
    )
    labels = [
        ("REST", COLORS["slate"], COLORS["soft_gray"]),
        ("LEFT", COLORS["blue"], COLORS["soft_blue"]),
        ("RIGHT", COLORS["teal"], COLORS["soft_teal"]),
    ]
    for index, (label, color, soft) in enumerate(labels):
        pill(
            ax,
            x=xs[2] + 0.010 + index * 0.039,
            y=0.530,
            width=0.034,
            height=0.041,
            text=label,
            color=color,
            soft=soft,
            fontsize=10.5,
        )
    ax.text(
        xs[2] + widths[2] / 2,
        0.460,
        "Recording 0 reserved\nfor patient calibration",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
        linespacing=1.14,
    )
    ax.text(
        xs[2] + widths[2] / 2,
        0.370,
        "Later recordings held out\nfor evaluation",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["slate"],
        ha="center",
        va="center",
        linespacing=1.14,
    )

    # 04 - parallel EEG representations
    branch_x = xs[3] + 0.012
    branch_w = widths[3] - 0.024
    branches = [
        (0.662, "MRCP  0.1-4 Hz", "Slow cortical\npreparation", COLORS["blue"], COLORS["soft_blue"]),
        (0.535, "SMR  7-35 Hz", "Mu/beta\ndesynchronization", COLORS["teal"], COLORS["soft_teal"]),
        (0.408, "RIEMANNIAN", "Four-band spatial\nstructure", COLORS["gold"], COLORS["soft_gold"]),
    ]
    for index, (y, title, note, color, soft) in enumerate(branches):
        rounded_box(
            ax,
            branch_x,
            y,
            branch_w,
            0.095,
            facecolor=soft,
            edgecolor=color,
            linewidth=1.0,
            radius=0.010,
            zorder=5,
        )
        if index == 0:
            t = np.linspace(0, 1, 80)
            signal = -0.8 * np.exp(-((t - 0.72) / 0.20) ** 2) + 0.08 * np.sin(2 * np.pi * 3 * t)
            ax.plot(
                branch_x + 0.010 + 0.044 * t,
                y + 0.047 + 0.026 * signal,
                transform=ax.transAxes,
                color=color,
                linewidth=1.3,
                zorder=8,
            )
        elif index == 1:
            draw_signal(ax, branch_x + 0.010, y + 0.020, 0.044, 0.052, color)
        else:
            draw_covariance(ax, branch_x + 0.012, y + 0.022, 0.046)
        ax.text(
            branch_x + 0.064,
            y + 0.060,
            title,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            zorder=9,
        )
        ax.text(
            branch_x + 0.064,
            y + 0.031,
            note,
            transform=ax.transAxes,
            fontsize=10.5,
            color=COLORS["slate"],
            ha="left",
            va="center",
            linespacing=1.0,
            zorder=9,
        )
    ax.text(
        xs[3] + widths[3] / 2,
        0.337,
        "Quality / OOD evidence retained",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )

    # 05 - population decoder and late fusion
    draw_network(ax, xs[4] + widths[4] / 2, 0.727)
    ax.text(
        xs[4] + widths[4] / 2,
        0.658,
        "Controller-selected\nlate fusion",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.0,
        zorder=9,
    )
    weights = [
        ("MRCP + FBCNet", "60%", COLORS["blue"], COLORS["soft_blue"]),
        ("Riemannian", "25%", COLORS["gold"], COLORS["soft_gold"]),
        ("SMR statistics", "15%", COLORS["teal"], COLORS["soft_teal"]),
    ]
    for index, (label, value, color, soft) in enumerate(weights):
        y = 0.590 - index * 0.067
        rounded_box(
            ax,
            xs[4] + 0.011,
            y,
            widths[4] - 0.022,
            0.045,
            facecolor=soft,
            edgecolor=color,
            linewidth=0.9,
            radius=0.008,
            zorder=5,
        )
        ax.text(
            xs[4] + widths[4] / 2,
            y + 0.0225,
            f"{label}  |  {value}",
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            ha="center",
            va="center",
            zorder=9,
        )
    ax.text(
        xs[4] + widths[4] / 2,
        0.345,
        "Per-dataset probability\ncalibration + threshold",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
        linespacing=1.12,
    )

    # 06 - rapid patient calibration
    draw_person(ax, xs[5] + widths[5] / 2, 0.720, COLORS["teal"])
    ax.text(
        xs[5] + widths[5] / 2,
        0.650,
        "30 calibration windows",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )
    for index, (label, color, soft) in enumerate(labels):
        pill(
            ax,
            x=xs[5] + 0.009 + index * 0.038,
            y=0.564,
            width=0.033,
            height=0.043,
            text=f"10\n{label}",
            color=color,
            soft=soft,
            fontsize=10.5,
        )
    ax.text(
        xs[5] + widths[5] / 2,
        0.502,
        "Direction labels are\ninternal evidence only",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["slate"],
        ha="center",
        va="center",
        linespacing=1.12,
    )
    ax.text(
        xs[5] + widths[5] / 2,
        0.425,
        "P(intent) = P(left) + P(right)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
    )
    pill(
        ax,
        x=xs[5] + 0.012,
        y=0.338,
        width=widths[5] - 0.024,
        height=0.060,
        text="PERSONALIZED\nP(INTENT)",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )

    # 07 - device-facing binary gate
    ax.text(
        xs[6] + widths[6] / 2,
        0.746,
        "CONSERVATIVE THRESHOLD",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
    )
    pill(
        ax,
        x=xs[6] + 0.014,
        y=0.652,
        width=widths[6] - 0.028,
        height=0.056,
        text="REST\nNO COMMAND",
        color=COLORS["slate"],
        soft=COLORS["soft_gray"],
    )
    flow_arrow(
        ax,
        (xs[6] + widths[6] / 2, 0.641),
        (xs[6] + widths[6] / 2, 0.606),
        color=COLORS["gold"],
        width=1.5,
    )
    pill(
        ax,
        x=xs[6] + 0.014,
        y=0.536,
        width=widths[6] - 0.028,
        height=0.056,
        text="INTENT\nCONFIRM",
        color=COLORS["teal"],
        soft=COLORS["soft_teal"],
    )
    ax.text(
        xs[6] + widths[6] / 2,
        0.452,
        "Cue defines the requested\nmovement direction",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
        linespacing=1.12,
    )
    ax.text(
        xs[6] + widths[6] / 2,
        0.365,
        "Temporal confirmation planned\nnot yet evaluated",
        transform=ax.transAxes,
        fontsize=10.5,
        color=COLORS["slate"],
        ha="center",
        va="center",
        linespacing=1.12,
    )

    # V7 result strip
    ax.text(
        0.025,
        0.252,
        "V7 EVALUATION OUTPUTS",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="center",
    )
    ax.plot(
        [0.190, 0.975],
        [0.252, 0.252],
        transform=ax.transAxes,
        color=COLORS["line"],
        linewidth=1.3,
    )

    panel_y, panel_h = 0.040, 0.185
    panel_specs = [
        (0.025, 0.355, "DOMAIN PERFORMANCE", COLORS["blue"]),
        (0.390, 0.190, "PATIENT ADAPTATION", COLORS["teal"]),
        (0.590, 0.205, "CONSERVATIVE OPERATING POINT", COLORS["gold"]),
        (0.805, 0.170, "VALIDATION STATUS", COLORS["coral"]),
    ]
    for x, width, title, color in panel_specs:
        output_panel(
            ax,
            x=x,
            y=panel_y,
            width=width,
            height=panel_h,
            title=title,
            color=color,
        )

    domain_values = {
        "PhysioNet": (63.8, 40.7, 86.9),
        "OpenBMI": (76.3, 57.3, 95.3),
        "Liu stroke": (65.0, 40.5, 89.5),
    }
    chart_x0, chart_y0 = 0.045, 0.077
    chart_w, chart_h = 0.315, 0.085
    bar_colors = (COLORS["navy"], COLORS["teal"], COLORS["gold"])
    for group_index, (name, values) in enumerate(domain_values.items()):
        group_x = chart_x0 + group_index * chart_w / 3 + 0.006
        for bar_index, (value, color) in enumerate(zip(values, bar_colors)):
            bar_w = 0.018
            x = group_x + bar_index * 0.022
            height = chart_h * value / 100
            ax.add_patch(
                Rectangle(
                    (x, chart_y0),
                    bar_w,
                    height,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=0,
                    zorder=5,
                )
            )
            ax.text(
                x + bar_w / 2,
                chart_y0 + height + 0.008,
                f"{value:g}",
                transform=ax.transAxes,
                fontsize=10.5,
                fontweight="bold",
                color=COLORS["ink"],
                ha="center",
                va="bottom",
            )
        ax.text(
            group_x + 0.031,
            0.058,
            name,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["slate"],
            ha="center",
            va="center",
        )
    legend = [
        ("BA", COLORS["navy"]),
        ("Sensitivity", COLORS["teal"]),
        ("Specificity", COLORS["gold"]),
    ]
    for index, (label, color) in enumerate(legend):
        x = 0.170 + index * 0.070
        ax.add_patch(
            Circle(
                (x, 0.193),
                0.004,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
            )
        )
        ax.text(
            x + 0.008,
            0.193,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            color=COLORS["slate"],
            ha="left",
            va="center",
        )

    ax.text(
        0.415,
        0.135,
        "69.1%",
        transform=ax.transAxes,
        fontsize=18.0,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
    )
    flow_arrow(ax, (0.455, 0.135), (0.510, 0.135), color=COLORS["teal"], width=2.2)
    ax.text(
        0.550,
        0.135,
        "73.6%",
        transform=ax.transAxes,
        fontsize=18.0,
        fontweight="bold",
        color=COLORS["teal"],
        ha="center",
        va="center",
    )
    ax.text(
        0.485,
        0.078,
        "+4.5 balanced-accuracy points\non matched later recordings",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
        linespacing=1.12,
    )

    ax.text(
        0.615,
        0.150,
        "93.7%",
        transform=ax.transAxes,
        fontsize=18.0,
        fontweight="bold",
        color=COLORS["gold"],
        ha="center",
        va="center",
    )
    ax.text(
        0.615,
        0.098,
        "REST SPECIFICITY",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
    )
    ax.text(
        0.760,
        0.150,
        "53.5%",
        transform=ax.transAxes,
        fontsize=18.0,
        fontweight="bold",
        color=COLORS["teal"],
        ha="center",
        va="center",
    )
    ax.text(
        0.760,
        0.098,
        "INTENT SENSITIVITY",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="center",
        va="center",
    )
    ax.text(
        0.692,
        0.058,
        "46.5% of intent windows missed",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )

    status_rows = [
        ("LATER RECORDINGS", COLORS["teal"]),
        ("CUE-LOCKED", COLORS["blue"]),
        ("DEVELOPMENT-OBSERVED", COLORS["coral"]),
    ]
    for index, (label, color) in enumerate(status_rows):
        y = 0.162 - index * 0.044
        ax.add_patch(
            Circle(
                (0.825, y),
                0.006,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=5,
            )
        )
        ax.text(
            0.840,
            y,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )
    ax.text(
        0.890,
        0.056,
        "Locked external testing still required",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "10_v7_illustrated_pipeline"
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
    create_visual(args.output_dir, args.dpi)
    print(f"[+] Saved illustrated V7 pipeline to {args.output_dir}")


if __name__ == "__main__":
    main()
