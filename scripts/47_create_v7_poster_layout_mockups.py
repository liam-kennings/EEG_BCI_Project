#!/usr/bin/env python3
"""Create three 48 x 36 poster-layout mockups for the V7 EEG BCI project."""

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
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "layout_mockups"
PIPELINE_IMAGE = (
    PROJECT_ROOT
    / "artifacts"
    / "poster_v7"
    / "illustrated_pipeline"
    / "10_v7_illustrated_pipeline.png"
)
RESULTS_IMAGE = (
    PROJECT_ROOT
    / "artifacts"
    / "poster_v7"
    / "layout_mockups"
    / "highres_assets"
    / "03_v7_poster_results_dashboard.png"
)

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#111111",
    "slate": "#292929",
    "muted": "#555555",
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

FIG_W = 16.0
FIG_H = 12.0


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
    facecolor: str = COLORS["paper"],
    edgecolor: str = COLORS["line"],
    linewidth: float = 1.3,
    radius: float = 0.012,
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


def flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["slate"],
    width: float = 1.8,
    mutation_scale: float = 14,
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
            zorder=12,
        )
    )


def poster_canvas(layout_tag: str, layout_color: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.add_patch(
        Rectangle(
            (0, 0.902),
            1,
            0.098,
            transform=ax.transAxes,
            facecolor="#CDEEF7",
            edgecolor="none",
            zorder=0,
        )
    )
    ax.text(
        0.020,
        0.967,
        "URI\nCOLLEGE OF ENGINEERING",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="top",
        linespacing=1.0,
    )
    ax.text(
        0.500,
        0.970,
        "Improving Weak Motor-Intent Detection in EEG-Based Rehabilitation BCIs",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color="#101820",
        ha="center",
        va="top",
    )
    ax.text(
        0.500,
        0.925,
        "Liam Kennings  |  Electrical Engineering PhD  |  University of Rhode Island",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
    )
    rounded_box(
        ax,
        0.875,
        0.923,
        0.105,
        0.047,
        facecolor=layout_color,
        edgecolor=layout_color,
        linewidth=0,
        radius=0.012,
        zorder=4,
    )
    ax.text(
        0.9275,
        0.946,
        layout_tag,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=5,
    )
    return fig, ax


def section_header(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    title: str,
    *,
    color: str = COLORS["navy"],
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            0.038,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=5,
        )
    )
    ax.text(
        x + 0.012,
        y + 0.019,
        title,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="white",
        ha="left",
        va="center",
        zorder=6,
    )


def panel_title(
    ax: plt.Axes,
    x: float,
    y: float,
    title: str,
    *,
    color: str = COLORS["ink"],
    fontsize: float = 14,
) -> None:
    ax.text(
        x,
        y,
        title,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="left",
        va="top",
    )


def place_image_contain(
    fig: plt.Figure,
    path: Path,
    box: tuple[float, float, float, float],
    *,
    border_color: str = COLORS["line"],
    border: bool = True,
) -> None:
    x, y, width, height = box
    image = plt.imread(path)
    image_aspect = image.shape[1] / image.shape[0]
    fig_width = fig.get_figwidth()
    fig_height = fig.get_figheight()
    box_aspect = (width * fig_width) / (height * fig_height)
    if image_aspect > box_aspect:
        actual_width = width
        actual_height = width * fig_width / image_aspect / fig_height
        actual_x = x
        actual_y = y + (height - actual_height) / 2
    else:
        actual_height = height
        actual_width = height * fig_height * image_aspect / fig_width
        actual_x = x + (width - actual_width) / 2
        actual_y = y
    image_ax = fig.add_axes((actual_x, actual_y, actual_width, actual_height), zorder=4)
    image_ax.imshow(image)
    image_ax.set_axis_off()
    if border:
        for spine in image_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.1)


def place_image_top_crop(
    fig: plt.Figure,
    path: Path,
    box: tuple[float, float, float, float],
    *,
    keep_fraction: float = 0.76,
) -> None:
    """Place the upper process portion of a figure without its result strip."""
    image = plt.imread(path)
    crop_end = max(1, min(image.shape[0], int(image.shape[0] * keep_fraction)))
    cropped = image[:crop_end]
    x, y, width, height = box
    image_aspect = cropped.shape[1] / cropped.shape[0]
    box_aspect = (width * fig.get_figwidth()) / (height * fig.get_figheight())
    if image_aspect > box_aspect:
        actual_width = width
        actual_height = width * fig.get_figwidth() / image_aspect / fig.get_figheight()
        actual_x = x
        actual_y = y + (height - actual_height) / 2
    else:
        actual_height = height
        actual_width = height * fig.get_figheight() * image_aspect / fig.get_figwidth()
        actual_x = x + (width - actual_width) / 2
        actual_y = y
    image_ax = fig.add_axes((actual_x, actual_y, actual_width, actual_height), zorder=4)
    image_ax.imshow(cropped)
    image_ax.set_axis_off()


def metric_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    label: str,
    value: str,
    note: str,
    color: str,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        0.092,
        facecolor=COLORS["paper"],
        edgecolor=color,
        radius=0.009,
    )
    ax.text(
        x + 0.012,
        y + 0.070,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["slate"],
        ha="left",
        va="center",
    )
    ax.text(
        x + 0.012,
        y + 0.039,
        value,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
    )
    ax.text(
        x + width - 0.010,
        y + 0.020,
        note,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["muted"],
        ha="right",
        va="center",
    )


def dataset_strip(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["blue"], radius=0.012)
    panel_title(ax, x + 0.014, y + height - 0.018, "DATA FOUNDATION", color=COLORS["blue"])
    metrics = [("187", "participants"), ("36,903", "EEG windows"), ("3", "recording domains")]
    for index, (value, label) in enumerate(metrics):
        cx = x + width * (0.17 + index * 0.33)
        ax.text(
            cx,
            y + height * 0.56,
            value,
            transform=ax.transAxes,
            fontsize=19,
            fontweight="bold",
            color=(COLORS["blue"], COLORS["teal"], COLORS["coral"])[index],
            ha="center",
            va="center",
        )
        ax.text(
            cx,
            y + height * 0.34,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["slate"],
            ha="center",
            va="center",
        )
    ax.text(
        x + width / 2,
        y + 0.022,
        "PhysioNet healthy  |  OpenBMI healthy  |  Liu acute stroke",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
    )


def results_summary(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["gold"], radius=0.012)
    panel_title(
        ax,
        x + 0.014,
        y + height - 0.018,
        "PATIENT-ADAPTED RESULT ON LATER RECORDINGS",
        color=COLORS["navy"],
    )
    labels = [
        ("73.6%", "BALANCED ACCURACY", COLORS["navy"]),
        ("53.5%", "INTENT SENSITIVITY", COLORS["teal"]),
        ("93.7%", "REST SPECIFICITY", COLORS["gold"]),
        ("0.896", "AUROC", COLORS["coral"]),
    ]
    for index, (value, label, color) in enumerate(labels):
        cx = x + width * (0.13 + index * 0.245)
        ax.text(
            cx,
            y + height * 0.56,
            value,
            transform=ax.transAxes,
            fontsize=21,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            cx,
            y + height * 0.35,
            label,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["slate"],
            ha="center",
            va="center",
        )
    ax.text(
        x + width / 2,
        y + 0.030,
        "Strong rest rejection reduced false activations, but 46.5% of intent windows were missed.",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )


def refined_results_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Draw the final poster results at native size instead of embedding a dashboard."""
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["gold"], radius=0.012)
    gap = 0.010
    card_width = (width - 0.024 - 3 * gap) / 4
    card_x = x + 0.012
    card_y = y + height - 0.105
    cards = [
        ("BALANCED ACCURACY", "73.6%", "95% CI 67.5-78.7", COLORS["navy"]),
        ("INTENT SENSITIVITY", "53.5%", "46.5% missed", COLORS["teal"]),
        ("REST SPECIFICITY", "93.7%", "6.3% rest FPR", COLORS["gold"]),
        ("INTENT AUROC", "0.896", "threshold-independent", COLORS["coral"]),
    ]
    for index, (label, value, note, color) in enumerate(cards):
        metric_card(
            ax,
            card_x + index * (card_width + gap),
            card_y,
            card_width,
            label,
            value,
            note,
            color,
        )

    chart_y = y + 0.030
    chart_h = height - 0.180
    chart_w = (width - 0.055) / 2
    progression_ax = fig.add_axes((x + 0.015, chart_y, chart_w, chart_h), zorder=5)
    domain_ax = fig.add_axes((x + 0.040 + chart_w, chart_y, chart_w, chart_h), zorder=5)

    progression = np.array(
        [
            [73.3, 52.9, 93.8],
            [71.9, 49.1, 94.7],
            [73.6, 53.5, 93.7],
        ]
    )
    domain = np.array(
        [
            [63.8, 40.7, 86.9],
            [76.3, 57.3, 95.3],
            [65.0, 40.5, 89.5],
        ]
    )
    colors = [COLORS["navy"], COLORS["teal"], COLORS["gold"]]
    legend_labels = ["Balanced accuracy", "Sensitivity", "Specificity"]

    def draw_grouped_bars(
        chart: plt.Axes,
        values: np.ndarray,
        labels: list[str],
        title: str,
        subtitle: str,
        *,
        show_ylabel: bool,
        show_legend: bool,
    ) -> None:
        indices = np.arange(len(labels))
        bar_width = 0.22
        chart.set_facecolor(COLORS["paper"])
        for series_index, (color, legend_label) in enumerate(zip(colors, legend_labels)):
            positions = indices + (series_index - 1) * bar_width
            bars = chart.bar(
                positions,
                values[:, series_index],
                width=bar_width,
                color=color,
                label=legend_label,
                zorder=3,
            )
            for bar, value in zip(bars, values[:, series_index]):
                chart.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.3,
                    f"{value:g}",
                    fontsize=7.5,
                    fontweight="bold",
                    color=COLORS["ink"],
                    ha="center",
                    va="bottom",
                )
        chart.set_title(
            f"{title}\n{subtitle}",
            fontsize=10.2,
            fontweight="bold",
            color=COLORS["ink"],
            loc="left",
            pad=8,
        )
        chart.set_ylim(0, 108)
        chart.set_yticks([0, 25, 50, 75, 100])
        chart.set_xticks(indices)
        chart.set_xticklabels(labels, fontsize=8.3, fontweight="bold")
        chart.tick_params(axis="y", labelsize=8.0, colors=COLORS["ink"])
        chart.grid(axis="y", color=COLORS["line"], linewidth=0.7, alpha=0.75, zorder=0)
        chart.spines[["top", "right"]].set_visible(False)
        chart.spines[["left", "bottom"]].set_color(COLORS["line"])
        if show_ylabel:
            chart.set_ylabel("Result (%)", fontsize=8.7, fontweight="bold", color=COLORS["ink"])
        else:
            chart.set_yticklabels([])
        if show_legend:
            chart.legend(
                loc="upper center",
                bbox_to_anchor=(0.58, 1.01),
                ncol=3,
                frameon=False,
                fontsize=7.4,
            )

    draw_grouped_bars(
        progression_ax,
        progression,
        ["V5\ndirectional", "V6\nbinary", "V7\nauxiliary"],
        "MODEL PROGRESSION",
        "Conservative operating point",
        show_ylabel=True,
        show_legend=True,
    )
    draw_grouped_bars(
        domain_ax,
        domain,
        ["PhysioNet", "OpenBMI", "Liu stroke"],
        "PERFORMANCE REMAINS DOMAIN DEPENDENT",
        "V7 patient-adapted result",
        show_ylabel=False,
        show_legend=False,
    )


def story_results_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    """Compare V7 with published percentages while preserving endpoint caveats."""
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["gold"], radius=0.012)
    chart_width = width * 0.675
    chart = fig.add_axes(
        (x + 0.024, y + 0.044, chart_width - 0.040, height - 0.086),
        zorder=5,
    )
    values = [73.6, 74.0, 71.15, 69.7]
    positions = np.arange(len(values))
    bar_colors = [COLORS["teal"], "#92999F", "#B0B5B9", "#C7CBCE"]
    edge_colors = [COLORS["teal"], "#6F767C", "#858C91", "#9CA2A7"]
    bars = chart.bar(
        positions,
        values,
        width=0.64,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.0,
        zorder=3,
    )
    for index, (bar, value) in enumerate(zip(bars, values)):
        chart.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:g}%",
            fontsize=10.0 if index == 0 else 9.2,
            fontweight="bold",
            color=COLORS["teal"] if index == 0 else COLORS["ink"],
            ha="center",
            va="bottom",
        )
    chart.axhline(50, color=COLORS["coral"], linewidth=1.0, linestyle="--", zorder=2)
    chart.text(
        3.48,
        51.8,
        "50% reference",
        fontsize=7.3,
        fontweight="bold",
        color=COLORS["coral"],
        ha="right",
        va="bottom",
    )
    chart.set_facecolor(COLORS["paper"])
    chart.set_title(
        "V7 RESULT IN PUBLISHED CONTEXT",
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["ink"],
        loc="left",
        pad=8,
    )
    chart.set_ylim(0, 104)
    chart.set_yticks([0, 25, 50, 75, 100])
    chart.set_ylabel("Reported result (%)", fontsize=8.8, fontweight="bold", color=COLORS["ink"])
    chart.set_xticks(positions)
    chart.set_xticklabels(
        [
            "THIS PROJECT\nV7 | balanced accuracy\n187 represented participants",
            "Ang et al., 2011 [5]\nmean accuracy\nstroke MI vs rest",
            "Nagarajan et al., 2024 [6]\nmean accuracy\nhealthy-to-stroke transfer",
            "Lopez-Larraz et al., 2025 [7]\nmean accuracy\nEEG + EMG attempt vs rest",
        ],
        fontsize=7.4,
        fontweight="bold",
        linespacing=1.12,
    )
    chart.tick_params(axis="y", labelsize=8.0, colors=COLORS["ink"])
    chart.grid(axis="y", color=COLORS["line"], linewidth=0.7, alpha=0.75, zorder=0)
    chart.spines[["top", "right"]].set_visible(False)
    chart.spines[["left", "bottom"]].set_color(COLORS["line"])

    profile_x = x + chart_width + 0.006
    profile_width = width - chart_width - 0.020
    rounded_box(
        ax,
        profile_x,
        y + 0.016,
        profile_width,
        height - 0.032,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["navy"],
        radius=0.010,
    )
    ax.text(
        profile_x + 0.014,
        y + height - 0.036,
        "V7 SAFETY PROFILE",
        transform=ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    profile_metrics = [
        ("53.5%", "intent sensitivity", "46.5% missed", COLORS["teal"], COLORS["soft_teal"]),
        ("93.7%", "rest specificity", "6.3% rest FPR", COLORS["gold"], COLORS["soft_gold"]),
        ("0.896", "intent AUROC", "threshold-independent", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for index, (value, label, note, color, soft) in enumerate(profile_metrics):
        row_y = y + height - 0.098 - index * 0.052
        rounded_box(
            ax,
            profile_x + 0.012,
            row_y,
            profile_width - 0.024,
            0.042,
            facecolor=soft,
            edgecolor=color,
            radius=0.007,
        )
        ax.text(
            profile_x + 0.025,
            row_y + 0.026,
            value,
            transform=ax.transAxes,
            fontsize=12.4,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
        ax.text(
            profile_x + 0.096,
            row_y + 0.027,
            label,
            transform=ax.transAxes,
            fontsize=8.4,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )
        ax.text(
            profile_x + 0.096,
            row_y + 0.012,
            note,
            transform=ax.transAxes,
            fontsize=7.4,
            color=COLORS["ink"],
            ha="left",
            va="center",
        )
    ax.text(
        profile_x + 0.014,
        y + 0.026,
        "CONTEXT, NOT A LEADERBOARD\nMetrics, modalities, cohorts, calibration,\nand validation designs differ.",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="left",
        va="bottom",
        linespacing=1.15,
    )


def next_research_phase_panel(
    ax: plt.Axes, x: float, y: float, width: float, height: float
) -> None:
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["teal"], radius=0.010)
    steps = [
        (
            "1",
            "EXPERIMENTAL PROTOCOL",
            "Timed cues + long no-control periods\nMeasure false activations/min and latency",
            COLORS["blue"],
            COLORS["soft_blue"],
        ),
        (
            "2",
            "CREATE A NEW DATASET",
            "Healthy + stroke repeated sessions\nSynchronized EEG and task markers",
            COLORS["teal"],
            COLORS["soft_teal"],
        ),
        (
            "3",
            "RETRAIN + VALIDATE",
            "Locked prospective cohorts\nCalibration and temporal confirmation",
            COLORS["gold"],
            COLORS["soft_gold"],
        ),
        (
            "4",
            "DEVICE + SENSOR TESTING",
            "Rehabilitation/prosthetic integration\nExplore EEG + EMG + IMU fusion",
            COLORS["coral"],
            COLORS["soft_coral"],
        ),
    ]
    gap = 0.012
    card_width = (width - 0.030 - 3 * gap) / 4
    for index, (number, title, body, color, soft) in enumerate(steps):
        card_x = x + 0.015 + index * (card_width + gap)
        rounded_box(
            ax,
            card_x,
            y + 0.010,
            card_width,
            height - 0.020,
            facecolor=soft,
            edgecolor=color,
            radius=0.008,
        )
        ax.add_patch(
            Circle(
                (card_x + 0.022, y + height / 2),
                0.014,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=6,
            )
        )
        ax.text(
            card_x + 0.022,
            y + height / 2,
            number,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=7,
        )
        ax.text(
            card_x + 0.045,
            y + height * 0.64,
            title,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )
        ax.text(
            card_x + 0.045,
            y + height * 0.35,
            body,
            transform=ax.transAxes,
            fontsize=7.6,
            color=COLORS["ink"],
            ha="left",
            va="center",
            linespacing=1.12,
        )


def compact_references_block(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle(
            (0.018, 0.070),
            0.964,
            0.027,
            transform=ax.transAxes,
            facecolor=COLORS["navy"],
            edgecolor=COLORS["navy"],
            linewidth=0,
            zorder=5,
        )
    )
    ax.text(
        0.030,
        0.0835,
        "SELECTED REFERENCES + VALIDATION NOTE",
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        color="white",
        ha="left",
        va="center",
        zorder=6,
    )
    rounded_box(ax, 0.018, 0.012, 0.964, 0.052, edgecolor=COLORS["navy"], radius=0.006)
    left = (
        "[1] Schalk G. EEGMMIDB. PhysioNet; 2009.  [2] Lee MH et al. OpenBMI. GigaScience. 2019.\n"
        "[3] Liu H et al. Acute-stroke MI EEG. Scientific Data. 2024.  [4] Pfurtscheller G et al. ERD/ERS. 1999."
    )
    right = (
        "[5] Ang et al. Stroke MI vs rest. 2011.  [6] Nagarajan et al. Healthy-to-stroke transfer. 2024.\n"
        "[7] Lopez-Larraz et al. EEG+EMG attempt detection. 2025.  Development-observed and cue-locked."
    )
    ax.text(
        0.030,
        0.048,
        left,
        transform=ax.transAxes,
        fontsize=7.0,
        color=COLORS["ink"],
        ha="left",
        va="top",
        linespacing=1.28,
    )
    ax.text(
        0.510,
        0.048,
        right,
        transform=ax.transAxes,
        fontsize=7.0,
        color=COLORS["ink"],
        ha="left",
        va="top",
        linespacing=1.28,
    )


def method_strip(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    stages = [
        ("01", "DATASETS", "187 people", COLORS["blue"], COLORS["soft_blue"]),
        ("02", "NATIVE QC", "CAR + 100 Hz", COLORS["teal"], COLORS["soft_teal"]),
        ("03", "WINDOWS", "Cue-locked 3 s", COLORS["gold"], COLORS["soft_gold"]),
        ("04", "PHYSIOLOGY", "MRCP / SMR / Riemann", COLORS["blue"], COLORS["soft_blue"]),
        ("05", "LATE FUSION", "60 / 25 / 15", COLORS["coral"], COLORS["soft_coral"]),
        ("06", "CALIBRATE", "30 windows", COLORS["teal"], COLORS["soft_teal"]),
        ("07", "INTENT GATE", "REST or INTENT", COLORS["gold"], COLORS["soft_gold"]),
    ]
    gap = width * 0.010
    card_width = (width - gap * 6) / 7
    for index, (number, title, body, color, soft) in enumerate(stages):
        stage_x = x + index * (card_width + gap)
        rounded_box(
            ax,
            stage_x,
            y,
            card_width,
            height,
            facecolor=soft,
            edgecolor=color,
            radius=0.009,
        )
        ax.add_patch(
            Circle(
                (stage_x + card_width / 2, y + height * 0.73),
                0.014,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=5,
            )
        )
        ax.text(
            stage_x + card_width / 2,
            y + height * 0.73,
            number,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            stage_x + card_width / 2,
            y + height * 0.48,
            title,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            stage_x + card_width / 2,
            y + height * 0.23,
            body,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["ink"],
            ha="center",
            va="center",
        )
        if index < 6:
            flow_arrow(
                ax,
                (stage_x + card_width - 0.002, y + height / 2),
                (stage_x + card_width + gap + 0.002, y + height / 2),
                color=COLORS["slate"],
                width=1.2,
                mutation_scale=9,
            )


def next_steps_band(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        radius=0.012,
    )
    ax.text(
        x + 0.018,
        y + height / 2,
        "NEXT STEPS",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=COLORS["gold"],
        ha="left",
        va="center",
    )
    steps = [
        "Continuous\nno-control test",
        "Locked stroke\ncohort",
        "Prospective\ncalibration",
        "Device\nintegration",
    ]
    for index, step in enumerate(steps):
        cx = x + width * (0.265 + index * 0.185)
        ax.add_patch(
            Circle(
                (cx, y + height * 0.66),
                0.014,
                transform=ax.transAxes,
                facecolor=(COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["coral"])[index],
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
        )
        ax.text(
            cx,
            y + height * 0.66,
            str(index + 1),
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            cx,
            y + height * 0.24,
            step,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            linespacing=1.0,
        )
        if index < len(steps) - 1:
            flow_arrow(
                ax,
                (cx + width * 0.035, y + height * 0.66),
                (cx + width * 0.150, y + height * 0.66),
                color=COLORS["line"],
                width=1.2,
                mutation_scale=10,
            )


def references_footer(ax: plt.Axes) -> None:
    ax.text(
        0.020,
        0.012,
        "Development-observed, cue-locked benchmark. Published values provide context only; endpoints and validation designs differ.",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["muted"],
        ha="left",
        va="bottom",
    )


def project_question_panel(
    ax: plt.Axes, x: float, y: float, width: float, height: float
) -> None:
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["blue"], radius=0.012)
    ax.text(
        x + width / 2,
        y + height - 0.030,
        "CAN EEG CONFIRM MOTOR INTENT\nWHEN PHYSICAL OUTPUT IS LIMITED?",
        transform=ax.transAxes,
        fontsize=14.2,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="top",
        linespacing=1.03,
    )

    center_x = x + 0.071
    center_y = y + 0.071
    ax.add_patch(
        Ellipse(
            (center_x, center_y),
            0.092,
            0.122,
            transform=ax.transAxes,
            facecolor=COLORS["soft_blue"],
            edgecolor=COLORS["blue"],
            linewidth=1.4,
            zorder=4,
        )
    )
    t = np.linspace(0, 1, 90)
    signal = 0.55 * np.sin(2 * np.pi * 4 * t) + 0.25 * np.sin(2 * np.pi * 8 * t)
    ax.plot(
        center_x - 0.034 + 0.068 * t,
        center_y + 0.018 * signal,
        transform=ax.transAxes,
        color=COLORS["teal"],
        linewidth=1.8,
        zorder=6,
    )
    ax.text(
        center_x,
        center_y - 0.040,
        "EEG",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        zorder=7,
    )

    statements = [
        ("Intent can precede\nvisible movement", COLORS["blue"]),
        ("EEG does not require\nreliable muscle output", COLORS["teal"]),
        ("False commands make\nREST rejection essential", COLORS["coral"]),
    ]
    for index, (statement, color) in enumerate(statements):
        statement_y = y + 0.112 - index * 0.043
        ax.add_patch(
            Rectangle(
                (x + 0.126, statement_y - 0.015),
                0.006,
                0.030,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                zorder=5,
            )
        )
        ax.text(
            x + 0.142,
            statement_y,
            statement,
            transform=ax.transAxes,
            fontsize=9.4,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
            linespacing=1.05,
        )


def contribution_panel(
    ax: plt.Axes, x: float, y: float, width: float, height: float
) -> None:
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["teal"], radius=0.012)
    contributions = [
        "Healthy + stroke population model",
        "MRCP, SMR, and Riemannian fusion",
        "Rapid 30-window patient calibration",
        "Device-facing REST / INTENT gate",
    ]
    for index, statement in enumerate(contributions):
        row_y = y + height - 0.030 - index * 0.035
        ax.add_patch(
            Circle(
                (x + 0.031, row_y),
                0.012,
                transform=ax.transAxes,
                facecolor=COLORS["soft_teal"],
                edgecolor=COLORS["teal"],
                linewidth=1.1,
                zorder=5,
            )
        )
        ax.text(
            x + 0.031,
            row_y,
            f"{index + 1}",
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["teal"],
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            x + 0.053,
            row_y,
            statement,
            transform=ax.transAxes,
            fontsize=10.2,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )


def interpretation_panel(
    ax: plt.Axes, x: float, y: float, width: float, height: float
) -> None:
    rounded_box(ax, x, y, width, height, edgecolor=COLORS["gold"], radius=0.012)
    ax.text(
        x + 0.016,
        y + height - 0.024,
        "CURRENT ANSWER",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["gold"],
        ha="left",
        va="center",
    )
    ax.text(
        x + 0.016,
        y + height - 0.056,
        "Conservative cue confirmation is promising,\nbut weak-intent recovery remains the bottleneck.",
        transform=ax.transAxes,
        fontsize=11.2,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
        linespacing=1.12,
    )

    summaries = [
        (x + 0.016, "STRENGTH", "Rest rejection", COLORS["teal"], COLORS["soft_teal"]),
        (x + 0.158, "BOTTLENECK", "Intent recovery", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for box_x, label, body, color, soft in summaries:
        rounded_box(
            ax,
            box_x,
            y + 0.075,
            0.126,
            0.046,
            facecolor=soft,
            edgecolor=color,
            radius=0.008,
        )
        ax.text(
            box_x + 0.063,
            y + 0.103,
            label,
            transform=ax.transAxes,
            fontsize=8.2,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            box_x + 0.063,
            y + 0.085,
            body,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["ink"],
            ha="center",
            va="center",
        )

    ax.text(
        x + 0.016,
        y + 0.058,
        "NEXT VALIDATION",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="center",
    )
    steps = ["Continuous\nno-control", "Locked stroke\ncohort", "Prospective\ncalibration"]
    centers = [x + 0.054, x + 0.150, x + 0.246]
    for index, (center, step) in enumerate(zip(centers, steps)):
        ax.add_patch(
            Circle(
                (center, y + 0.034),
                0.012,
                transform=ax.transAxes,
                facecolor=(COLORS["blue"], COLORS["teal"], COLORS["coral"])[index],
                edgecolor="none",
                zorder=5,
            )
        )
        ax.text(
            center,
            y + 0.034,
            str(index + 1),
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=6,
        )
        ax.text(
            center,
            y + 0.010,
            step,
            transform=ax.transAxes,
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["ink"],
            ha="center",
            va="center",
            linespacing=1.0,
        )
        if index < 2:
            flow_arrow(
                ax,
                (center + 0.018, y + 0.034),
                (centers[index + 1] - 0.018, y + 0.034),
                color=COLORS["muted"],
                width=1.0,
                mutation_scale=8,
            )


def references_block(ax: plt.Axes) -> None:
    section_header(ax, 0.018, 0.100, 0.964, "SELECTED REFERENCES + VALIDATION NOTE")
    rounded_box(ax, 0.018, 0.014, 0.964, 0.080, edgecolor=COLORS["navy"], radius=0.008)
    left = (
        "[1] Schalk G. EEG Motor Movement/Imagery Dataset. PhysioNet; 2009. doi:10.13026/C28G6P\n"
        "[2] Lee MH et al. EEG dataset and OpenBMI toolbox. GigaScience. 2019;8:giz002.\n"
        "[3] Liu H et al. Acute-stroke motor-imagery EEG dataset. Scientific Data. 2024;11:131.\n"
        "[4] Pfurtscheller G, Lopes da Silva FH. ERD/ERS. Clinical Neurophysiology. 1999."
    )
    right = (
        "[5] Barachant A et al. Riemannian-geometry BCI classification. IEEE TBME. 2012.\n"
        "[6] Mane R et al. FBCNet: multi-view convolution for BCI. 2021.\n"
        "[7] Bhattacharyya S et al. Cue-constrained versus asynchronous MI after stroke. 2022.\n"
        "Development-observed and cue-locked; locked external and continuous testing remain required."
    )
    ax.text(
        0.032,
        0.080,
        left,
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["ink"],
        ha="left",
        va="top",
        linespacing=1.34,
    )
    ax.text(
        0.510,
        0.080,
        right,
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["ink"],
        ha="left",
        va="top",
        linespacing=1.34,
    )
def layout_narrative(output_dir: Path, dpi: int) -> Path:
    fig, ax = poster_canvas("V7  |  PATIENT-ADAPTIVE", COLORS["blue"])

    section_header(ax, 0.018, 0.855, 0.300, "1. WHY THIS PROJECT?")
    project_question_panel(ax, 0.018, 0.650, 0.300, 0.199)
    rounded_box(
        ax,
        0.018,
        0.570,
        0.300,
        0.070,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.009,
    )
    ax.text(
        0.035,
        0.619,
        "PROBLEM TO ADDRESS",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="left",
        va="center",
    )
    ax.text(
        0.035,
        0.590,
        "Weak, variable EEG must be separated from rest\nwithout creating unsafe false commands.",
        transform=ax.transAxes,
        fontsize=9.8,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
        linespacing=1.10,
    )

    section_header(ax, 0.335, 0.855, 0.647, "2. GENERAL APPROACH + V7 PIPELINE", color=COLORS["navy"])
    rounded_box(ax, 0.335, 0.570, 0.647, 0.279, edgecolor=COLORS["navy"], radius=0.012)
    place_image_top_crop(
        fig,
        PIPELINE_IMAGE,
        (0.345, 0.580, 0.627, 0.259),
        keep_fraction=0.73,
    )

    section_header(ax, 0.018, 0.520, 0.964, "3. EXPERIMENTAL RESULTS + PUBLISHED CONTEXT", color=COLORS["gold"])
    story_results_panel(fig, ax, 0.018, 0.245, 0.964, 0.269)

    section_header(ax, 0.018, 0.190, 0.964, "4. NEXT RESEARCH PHASE", color=COLORS["teal"])
    next_research_phase_panel(ax, 0.018, 0.105, 0.964, 0.079)

    compact_references_block(ax)
    output = output_dir / "layout_a_story_flow_final_editing_base.png"
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def layout_evidence(output_dir: Path, dpi: int) -> Path:
    fig, ax = poster_canvas("LAYOUT B  |  EVIDENCE", COLORS["gold"])

    ax.text(
        0.020,
        0.870,
        "Patient adaptation improved intent recovery while preserving high rest rejection",
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=COLORS["navy"],
        ha="left",
        va="center",
    )
    metric_card(ax, 0.020, 0.764, 0.205, "BALANCED ACCURACY", "73.6%", "95% CI 67.5-78.7", COLORS["navy"])
    metric_card(ax, 0.235, 0.764, 0.205, "INTENT SENSITIVITY", "53.5%", "46.5% missed", COLORS["teal"])
    metric_card(ax, 0.450, 0.764, 0.205, "REST SPECIFICITY", "93.7%", "6.3% rest FPR", COLORS["gold"])
    metric_card(ax, 0.665, 0.764, 0.155, "AUROC", "0.896", "threshold-independent", COLORS["coral"])
    rounded_box(ax, 0.830, 0.764, 0.150, 0.092, facecolor=COLORS["soft_coral"], edgecolor=COLORS["coral"], radius=0.009)
    ax.text(
        0.905,
        0.810,
        "KEY LIMITATION\nNearly half of intent\nwindows were missed",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
        linespacing=1.05,
    )

    section_header(ax, 0.020, 0.715, 0.705, "PRIMARY EVIDENCE", color=COLORS["navy"])
    rounded_box(ax, 0.020, 0.335, 0.705, 0.374, edgecolor=COLORS["navy"], radius=0.012)
    place_image_contain(fig, RESULTS_IMAGE, (0.030, 0.345, 0.685, 0.354), border=False)

    section_header(ax, 0.740, 0.715, 0.242, "INTERPRETATION", color=COLORS["teal"])
    rounded_box(ax, 0.740, 0.535, 0.242, 0.174, edgecolor=COLORS["teal"], radius=0.012)
    interpretation = [
        ("+4.5 points", "after 30-window calibration", COLORS["teal"]),
        ("Domain dependent", "OpenBMI > Liu / PhysioNet", COLORS["blue"]),
        ("Conservative gate", "prioritizes rest rejection", COLORS["gold"]),
    ]
    for index, (headline, note, color) in enumerate(interpretation):
        y = 0.674 - index * 0.052
        ax.text(
            0.760,
            y,
            headline,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
        ax.text(
            0.760,
            y - 0.021,
            note,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["slate"],
            ha="left",
            va="center",
        )

    rounded_box(ax, 0.740, 0.335, 0.242, 0.184, edgecolor=COLORS["coral"], radius=0.012)
    panel_title(ax, 0.757, 0.500, "PUBLISHED CONTEXT", color=COLORS["coral"])
    ax.text(
        0.757,
        0.462,
        "74%  Ang et al., 2011\n71.15%  Nagarajan et al., 2024\n69.7%  Lopez-Larraz et al., 2025",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
        linespacing=1.55,
    )
    ax.text(
        0.757,
        0.355,
        "Context only: populations, endpoints,\nand validation designs differ.",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="left",
        va="bottom",
        linespacing=1.1,
    )

    section_header(ax, 0.020, 0.286, 0.705, "METHOD IN ONE VIEW", color=COLORS["blue"])
    rounded_box(ax, 0.020, 0.085, 0.705, 0.195, edgecolor=COLORS["blue"], radius=0.012)
    method_strip(ax, 0.035, 0.110, 0.675, 0.140)

    section_header(ax, 0.740, 0.286, 0.242, "NEXT DECISION", color=COLORS["coral"])
    rounded_box(ax, 0.740, 0.085, 0.242, 0.195, edgecolor=COLORS["coral"], radius=0.012)
    next_items = [
        "1  Validate on locked HS-Stroke cohort",
        "2  Measure false activations per minute",
        "3  Test cue-guided calibration prospectively",
        "4  Integrate temporal confirmation",
    ]
    for index, item in enumerate(next_items):
        ax.text(
            0.760,
            0.235 - index * 0.039,
            item,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["ink"],
            ha="left",
            va="center",
        )

    references_footer(ax)
    output = output_dir / "layout_b_evidence_first.png"
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def lane_stage(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    number: str,
    title: str,
    body: str,
    color: str,
    soft: str,
) -> None:
    rounded_box(ax, x, y, width, height, facecolor=soft, edgecolor=color, radius=0.012)
    ax.add_patch(
        Circle(
            (x + 0.025, y + height - 0.032),
            0.015,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            zorder=5,
        )
    )
    ax.text(
        x + 0.025,
        y + height - 0.032,
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
        x + 0.050,
        y + height - 0.030,
        title,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
    )
    ax.text(
        x + width / 2,
        y + height * 0.40,
        body,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.15,
    )


def layout_translational(output_dir: Path, dpi: int) -> Path:
    fig, ax = poster_canvas("LAYOUT C  |  JOURNEY", COLORS["coral"])

    ax.text(
        0.500,
        0.866,
        "From population EEG to a rapidly calibrated rehabilitation intent gate",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=COLORS["navy"],
        ha="center",
        va="center",
    )
    top_cards = [
        (0.020, "UNMET NEED", "Motor intent may remain when\nmovement and EMG are weak", COLORS["blue"], COLORS["soft_blue"]),
        (0.350, "EEG OPPORTUNITY", "Capture cortical evidence without\nrequiring reliable muscle output", COLORS["teal"], COLORS["soft_teal"]),
        (0.680, "PROJECT GOAL", "Personalize REST / INTENT detection\nwhile limiting false commands", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for x, title, body, color, soft in top_cards:
        rounded_box(ax, x, 0.710, 0.300, 0.125, facecolor=soft, edgecolor=color, radius=0.014)
        ax.text(
            x + 0.018,
            0.807,
            title,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
        )
        ax.text(
            x + 0.150,
            0.753,
            body,
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
            color=COLORS["ink"],
            ha="center",
            va="center",
            linespacing=1.15,
        )

    section_header(ax, 0.020, 0.654, 0.960, "A  POPULATION DEVELOPMENT", color=COLORS["blue"])
    pop_y, pop_h = 0.470, 0.170
    pop_stages = [
        (0.020, 0.210, "01", "MULTI-DATASET EEG", "187 participants\n36,903 windows\nHealthy + stroke", COLORS["blue"], COLORS["soft_blue"]),
        (0.265, 0.210, "02", "NATIVE QC", "Repair channels\nCAR + 100 Hz\n7 motor channels", COLORS["teal"], COLORS["soft_teal"]),
        (0.510, 0.210, "03", "PHYSIOLOGY", "MRCP 0.1-4 Hz\nSMR 7-35 Hz\nRiemannian covariance", COLORS["gold"], COLORS["soft_gold"]),
        (0.755, 0.225, "04", "GENERAL MODEL", "Calibrated late fusion\n60% / 25% / 15%\nDataset-aware threshold", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for x, width, number, title, body, color, soft in pop_stages:
        lane_stage(ax, x, pop_y, width, pop_h, number, title, body, color, soft)
    for start_x, end_x in ((0.230, 0.265), (0.475, 0.510), (0.720, 0.755)):
        flow_arrow(ax, (start_x, 0.555), (end_x, 0.555), color=COLORS["blue"])

    section_header(ax, 0.020, 0.414, 0.960, "B  RAPID PATIENT ADAPTATION", color=COLORS["coral"])
    patient_y, patient_h = 0.225, 0.175
    patient_stages = [
        (0.020, 0.175, "05", "KNOWN CUE", "Rehabilitation task\ndefines requested\nmovement", COLORS["coral"], COLORS["soft_coral"]),
        (0.225, 0.175, "06", "30 WINDOWS", "10 REST\n10 LEFT\n10 RIGHT", COLORS["gold"], COLORS["soft_gold"]),
        (0.430, 0.175, "07", "PERSONALIZE", "General probability\n+ patient evidence\n+ threshold", COLORS["teal"], COLORS["soft_teal"]),
        (0.635, 0.175, "08", "INTENT GATE", "REST = no command\nINTENT = confirm\nDirection stays internal", COLORS["blue"], COLORS["soft_blue"]),
        (0.840, 0.140, "09", "LATER DATA", "73.6% BA\n53.5% sensitivity\n93.7% specificity", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for x, width, number, title, body, color, soft in patient_stages:
        lane_stage(ax, x, patient_y, width, patient_h, number, title, body, color, soft)
    for start_x, end_x in ((0.195, 0.225), (0.400, 0.430), (0.605, 0.635), (0.810, 0.840)):
        flow_arrow(ax, (start_x, 0.312), (end_x, 0.312), color=COLORS["coral"])
    flow_arrow(ax, (0.865, 0.470), (0.700, 0.407), color=COLORS["coral"], width=2.0)
    ax.text(
        0.790,
        0.440,
        "GENERAL MODEL INITIALIZES\nPATIENT DECODER",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
        linespacing=1.05,
    )

    next_steps_band(ax, 0.020, 0.115, 0.960, 0.075)
    applications = [
        ("REHABILITATION", "intent-driven feedback", COLORS["blue"]),
        ("ROBOTIC ASSISTANCE", "orthoses + exoskeletons", COLORS["teal"]),
        ("PROSTHETIC CONTROL", "intent-gated commands", COLORS["gold"]),
        ("STIMULATION", "future FES triggering", COLORS["coral"]),
    ]
    for index, (title, note, color) in enumerate(applications):
        x = 0.020 + index * 0.245
        rounded_box(ax, x, 0.035, 0.225, 0.060, facecolor=COLORS["paper"], edgecolor=color, radius=0.010)
        ax.text(
            x + 0.1125,
            0.070,
            title,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.1125,
            0.048,
            note,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=COLORS["slate"],
            ha="center",
            va="center",
        )

    output = output_dir / "layout_c_translational_journey.png"
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def comparison_sheet(outputs: list[Path], output_dir: Path, dpi: int) -> Path:
    fig = plt.figure(figsize=(18.0, 6.8), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.text(
        0.5,
        0.965,
        "V7 Poster Layout Directions",
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        color=COLORS["navy"],
        ha="center",
        va="top",
    )
    labels = [
        ("A  NARRATIVE FIRST", "Best for a broad poster audience", COLORS["blue"]),
        ("B  EVIDENCE FIRST", "Best for technical judging", COLORS["gold"]),
        ("C  TRANSLATIONAL JOURNEY", "Best for clinical and PhD framing", COLORS["coral"]),
    ]
    for index, (path, (label, note, color)) in enumerate(zip(outputs, labels)):
        x = 0.025 + index * 0.325
        rounded_box(ax, x, 0.080, 0.300, 0.790, facecolor=COLORS["paper"], edgecolor=color, radius=0.012)
        place_image_contain(fig, path, (x + 0.010, 0.145, 0.280, 0.700), border=False)
        ax.text(
            x + 0.150,
            0.110,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.150,
            0.086,
            note,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=COLORS["slate"],
            ha="center",
            va="center",
        )
    output = output_dir / "poster_layout_options_comparison.png"
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".svg"), facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        layout_narrative(args.output_dir, args.dpi),
        layout_evidence(args.output_dir, args.dpi),
        layout_translational(args.output_dir, args.dpi),
    ]
    comparison_sheet(outputs, args.output_dir, args.dpi)
    print(f"[+] Saved three poster mockups and comparison sheet to {args.output_dir}")


if __name__ == "__main__":
    main()
