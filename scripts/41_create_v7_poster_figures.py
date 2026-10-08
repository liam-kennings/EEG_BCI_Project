#!/usr/bin/env python3
"""Create presentation-ready poster figures for the Translational V7 pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


DEFAULT_METRICS = (
    PROJECT_ROOT
    / "artifacts"
    / "gui_experiments"
    / "translational_v7_auxiliary_direction"
    / "translational_v7_auxiliary_direction_metrics.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#111111",
    "muted": "#555555",
    "line": "#CBD3D7",
    "navy": "#173A59",
    "blue": "#2D7FA8",
    "teal": "#248B7F",
    "green": "#4E9B73",
    "gold": "#E9A437",
    "coral": "#D96B3B",
    "red": "#B94A48",
    "soft_blue": "#E8F2F7",
    "soft_teal": "#E5F3EF",
    "soft_gold": "#FBF0D6",
    "soft_coral": "#F8E8DF",
    "soft_gray": "#EEF1F2",
}

DATASET_COLORS = {
    "physionet": COLORS["blue"],
    "openbmi": COLORS["teal"],
    "liu2024": COLORS["coral"],
}

DISPLAY_NAMES = {
    "physionet": "PhysioNet",
    "openbmi": "OpenBMI",
    "liu2024": "Liu stroke",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Avenir Next", "DejaVu Sans"],
            "font.size": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["line"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_metrics(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"V7 metrics not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if int(report.get("schema_version", 0)) < 8:
        raise ValueError("Poster figures require the schema-8 V7 metrics file.")
    return report


def pct(value: float) -> float:
    return 100.0 * float(value)


def card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = COLORS["paper"],
    edgecolor: str = COLORS["line"],
    linewidth: float = 1.3,
    radius: float = 0.018,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
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
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    width: float = 1.8,
    mutation_scale: float = 14,
    connectionstyle: str = "arc3,rad=0",
    zorder: float = 3,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        color=color,
        linewidth=width,
        mutation_scale=mutation_scale,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)


def section_header(
    ax: plt.Axes,
    x: float,
    y: float,
    number: str,
    title: str,
    color: str,
) -> None:
    ax.text(
        x,
        y,
        number,
        transform=ax.transAxes,
        color=color,
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    ax.text(
        x + 0.033,
        y,
        title.upper(),
        transform=ax.transAxes,
        color=COLORS["ink"],
        fontsize=13,
        fontweight="bold",
        va="center",
    )


def add_waveform(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    *,
    cycles: float = 4.0,
    linewidth: float = 2.2,
) -> None:
    t = np.linspace(0.0, 1.0, 100)
    envelope = 0.6 + 0.4 * np.sin(np.pi * t) ** 2
    signal = np.sin(2.0 * np.pi * cycles * t) * envelope
    ax.plot(
        x + width * t,
        y + 0.5 * height + 0.43 * height * signal,
        transform=ax.transAxes,
        color=color,
        linewidth=linewidth,
        clip_on=False,
        zorder=5,
    )


def add_progress_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    value: float,
    color: str,
    label: str,
) -> None:
    ax.text(
        x,
        y + 0.025,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        x + width,
        y + 0.025,
        f"{value:.1f}%",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=color,
    )
    card(
        ax,
        x,
        y,
        width,
        0.014,
        facecolor=COLORS["soft_gray"],
        edgecolor=COLORS["soft_gray"],
        radius=0.007,
        linewidth=0,
    )
    card(
        ax,
        x,
        y,
        width * value / 100.0,
        0.014,
        facecolor=color,
        edgecolor=color,
        radius=0.007,
        linewidth=0,
        zorder=2,
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=dpi if suffix == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def create_graphical_abstract(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    dataset = report["dataset"]
    patient = report["patient_calibration"]["development_observed_conservative"]
    intervals = report["bootstrap_confidence_intervals"]["patient_adapted"][
        "conservative"
    ]["overall"]
    configurations = report["patient_calibration"]["selected_configurations"]

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    ax.text(
        0.038,
        0.956,
        "Patient-Adaptive EEG Detection of Weak Motor Intent",
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="top",
        color=COLORS["ink"],
    )
    ax.text(
        0.04,
        0.902,
        "A multi-dataset population model uses a short calibration to confirm REST versus INTENT after a known rehabilitation cue",
        transform=ax.transAxes,
        fontsize=14,
        color=COLORS["muted"],
        va="top",
    )
    card(
        ax,
        0.795,
        0.914,
        0.165,
        0.043,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.015,
    )
    ax.text(
        0.8775,
        0.935,
        "V7  |  DEVELOPMENT-OBSERVED",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["coral"],
    )

    left = (0.035, 0.15, 0.255, 0.70)
    middle = (0.31, 0.15, 0.40, 0.70)
    right = (0.73, 0.15, 0.235, 0.70)
    for x, y, width, height in (left, middle, right):
        card(ax, x, y, width, height, linewidth=1.0, radius=0.022)

    section_header(ax, left[0] + 0.02, 0.817, "01", "Three EEG datasets", COLORS["blue"])
    dataset_rows = [
        ("physionet", "Healthy motor imagery", "protocol rest"),
        ("openbmi", "Healthy motor imagery", "post-task rest"),
        ("liu2024", "Acute stroke motor imagery", "protocol break"),
    ]
    y_positions = [0.665, 0.505, 0.345]
    for (key, cohort, rest_kind), y in zip(dataset_rows, y_positions):
        values = dataset["datasets"][key]
        color = DATASET_COLORS[key]
        soft = {
            "physionet": COLORS["soft_blue"],
            "openbmi": COLORS["soft_teal"],
            "liu2024": COLORS["soft_coral"],
        }[key]
        card(
            ax,
            left[0] + 0.018,
            y,
            left[2] - 0.036,
            0.125,
            facecolor=soft,
            edgecolor=color,
            radius=0.015,
            linewidth=1.2,
        )
        circle = Circle(
            (left[0] + 0.057, y + 0.064),
            0.032,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(
            left[0] + 0.057,
            y + 0.064,
            str(values["subjects"]),
            transform=ax.transAxes,
            color="white",
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            left[0] + 0.099,
            y + 0.091,
            DISPLAY_NAMES[key],
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="center",
        )
        ax.text(
            left[0] + 0.099,
            y + 0.061,
            f"{cohort}  |  {values['rows']:,} windows",
            transform=ax.transAxes,
            fontsize=9.4,
            color=COLORS["muted"],
            va="center",
        )
        ax.text(
            left[0] + 0.099,
            y + 0.032,
            f"Negative state: {rest_kind}",
            transform=ax.transAxes,
            fontsize=9.0,
            color=color,
            va="center",
        )

    card(
        ax,
        left[0] + 0.018,
        0.19,
        left[2] - 0.036,
        0.105,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.015,
    )
    ax.text(
        left[0] + 0.072,
        0.253,
        f"{dataset['subjects']}",
        transform=ax.transAxes,
        color="white",
        fontsize=22,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        left[0] + 0.072,
        0.215,
        "participants",
        transform=ax.transAxes,
        color="#DCE7EC",
        fontsize=10,
        ha="center",
    )
    ax.text(
        left[0] + 0.175,
        0.253,
        f"{dataset['rows']:,}",
        transform=ax.transAxes,
        color="white",
        fontsize=22,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        left[0] + 0.175,
        0.215,
        "EEG windows",
        transform=ax.transAxes,
        color="#DCE7EC",
        fontsize=10,
        ha="center",
    )

    section_header(ax, middle[0] + 0.02, 0.817, "02", "Translational V7", COLORS["teal"])
    card(
        ax,
        middle[0] + 0.025,
        0.708,
        middle[2] - 0.05,
        0.075,
        facecolor=COLORS["soft_gray"],
        edgecolor=COLORS["line"],
        radius=0.014,
    )
    add_waveform(ax, middle[0] + 0.045, 0.725, 0.075, 0.036, COLORS["navy"])
    ax.text(
        middle[0] + 0.14,
        0.753,
        "Native-montage quality control",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="center",
    )
    ax.text(
        middle[0] + 0.14,
        0.724,
        "repair bad channels  >  CAR  >  100 Hz  >  7 motor channels",
        transform=ax.transAxes,
        fontsize=9.5,
        color=COLORS["muted"],
        va="center",
    )
    arrow(ax, (middle[0] + 0.20, 0.705), (middle[0] + 0.20, 0.663), color=COLORS["teal"])

    branch_y = 0.515
    branch_w = 0.165
    branch_h = 0.14
    card(
        ax,
        middle[0] + 0.025,
        branch_y,
        branch_w,
        branch_h,
        facecolor=COLORS["soft_blue"],
        edgecolor=COLORS["blue"],
        radius=0.017,
    )
    ax.text(
        middle[0] + 0.045,
        branch_y + 0.108,
        "MRCP  |  0.1-4 Hz",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=COLORS["blue"],
    )
    add_waveform(ax, middle[0] + 0.045, branch_y + 0.044, 0.105, 0.04, COLORS["blue"], cycles=1.2)
    ax.text(
        middle[0] + 0.045,
        branch_y + 0.02,
        "slow preparation + onset",
        transform=ax.transAxes,
        fontsize=9.4,
        color=COLORS["muted"],
    )

    card(
        ax,
        middle[0] + 0.21,
        branch_y,
        branch_w,
        branch_h,
        facecolor=COLORS["soft_teal"],
        edgecolor=COLORS["teal"],
        radius=0.017,
    )
    ax.text(
        middle[0] + 0.23,
        branch_y + 0.108,
        "SMR  |  7-35 Hz",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=COLORS["teal"],
    )
    add_waveform(ax, middle[0] + 0.23, branch_y + 0.044, 0.105, 0.04, COLORS["teal"], cycles=5.0)
    ax.text(
        middle[0] + 0.23,
        branch_y + 0.02,
        "mu/beta + Riemannian",
        transform=ax.transAxes,
        fontsize=9.4,
        color=COLORS["muted"],
    )
    arrow(ax, (middle[0] + 0.107, branch_y), (middle[0] + 0.19, 0.477), color=COLORS["teal"])
    arrow(ax, (middle[0] + 0.292, branch_y), (middle[0] + 0.21, 0.477), color=COLORS["teal"])

    card(
        ax,
        middle[0] + 0.055,
        0.397,
        middle[2] - 0.11,
        0.078,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.015,
    )
    ax.text(
        middle[0] + middle[2] / 2,
        0.449,
        "CALIBRATED LATE FUSION",
        transform=ax.transAxes,
        color="white",
        fontsize=11.5,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        middle[0] + middle[2] / 2,
        0.417,
        "MRCP + FBCNet 60%   |   Riemannian 25%   |   SMR 15%",
        transform=ax.transAxes,
        color="#DCE7EC",
        fontsize=9.0,
        ha="center",
    )
    arrow(ax, (middle[0] + 0.20, 0.394), (middle[0] + 0.20, 0.354), color=COLORS["coral"])

    card(
        ax,
        middle[0] + 0.035,
        0.247,
        middle[2] - 0.07,
        0.103,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.017,
    )
    ax.text(
        middle[0] + 0.055,
        0.318,
        "Rapid patient calibration",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["coral"],
    )
    ax.text(
        middle[0] + 0.055,
        0.286,
        "30 windows: 10 REST + 10 LEFT + 10 RIGHT",
        transform=ax.transAxes,
        fontsize=9.6,
        color=COLORS["ink"],
    )
    ax.text(
        middle[0] + 0.055,
        0.259,
        "Direction is collapsed to P(intent); the controller can assign weight 0",
        transform=ax.transAxes,
        fontsize=8.9,
        color=COLORS["muted"],
    )
    arrow(ax, (middle[0] + 0.20, 0.244), (middle[0] + 0.20, 0.216), color=COLORS["teal"])

    card(
        ax,
        middle[0] + 0.07,
        0.168,
        0.105,
        0.047,
        facecolor=COLORS["soft_gray"],
        edgecolor=COLORS["muted"],
        radius=0.018,
    )
    card(
        ax,
        middle[0] + 0.225,
        0.168,
        0.105,
        0.047,
        facecolor=COLORS["soft_teal"],
        edgecolor=COLORS["teal"],
        radius=0.018,
    )
    ax.text(
        middle[0] + 0.1225,
        0.1915,
        "REST",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["muted"],
    )
    ax.text(
        middle[0] + 0.2775,
        0.1915,
        "INTENT",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["teal"],
    )
    section_header(ax, right[0] + 0.02, 0.817, "03", "Later-recording result", COLORS["gold"])
    ax.text(
        right[0] + right[2] / 2,
        0.71,
        f"{pct(patient['intent_balanced_accuracy']):.1f}%",
        transform=ax.transAxes,
        fontsize=38,
        fontweight="bold",
        ha="center",
        color=COLORS["navy"],
    )
    ax.text(
        right[0] + right[2] / 2,
        0.668,
        "BALANCED ACCURACY",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="center",
        color=COLORS["navy"],
    )
    ba_ci = intervals["intent_balanced_accuracy"]
    ax.text(
        right[0] + right[2] / 2,
        0.638,
        f"subject-bootstrap 95% CI {pct(ba_ci['lower_95']):.1f}-{pct(ba_ci['upper_95']):.1f}%",
        transform=ax.transAxes,
        fontsize=8.8,
        ha="center",
        color=COLORS["muted"],
    )
    add_progress_bar(
        ax,
        right[0] + 0.025,
        0.572,
        right[2] - 0.05,
        pct(patient["intent_sensitivity"]),
        COLORS["teal"],
        "Intent sensitivity",
    )
    add_progress_bar(
        ax,
        right[0] + 0.025,
        0.505,
        right[2] - 0.05,
        pct(patient["intent_specificity"]),
        COLORS["gold"],
        "Rest specificity",
    )

    ax.text(
        right[0] + 0.025,
        0.458,
        "RESULT BY DATASET",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=COLORS["muted"],
    )
    domain_y = [0.413, 0.357, 0.301]
    for key, y in zip(("physionet", "openbmi", "liu2024"), domain_y):
        value = pct(patient["by_dataset"][key]["intent_balanced_accuracy"])
        color = DATASET_COLORS[key]
        ax.text(
            right[0] + 0.025,
            y,
            DISPLAY_NAMES[key],
            transform=ax.transAxes,
            fontsize=9.3,
            fontweight="bold",
            va="center",
        )
        card(
            ax,
            right[0] + 0.092,
            y - 0.009,
            0.085,
            0.018,
            facecolor=COLORS["soft_gray"],
            edgecolor=COLORS["soft_gray"],
            radius=0.008,
            linewidth=0,
        )
        card(
            ax,
            right[0] + 0.092,
            y - 0.009,
            0.085 * value / 100.0,
            0.018,
            facecolor=color,
            edgecolor=color,
            radius=0.008,
            linewidth=0,
            zorder=3,
        )
        ax.text(
            right[0] + 0.187,
            y,
            f"{value:.1f}%",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=color,
            va="center",
        )

    card(
        ax,
        right[0] + 0.02,
        0.185,
        right[2] - 0.04,
        0.083,
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        radius=0.015,
    )
    ax.text(
        right[0] + 0.035,
        0.245,
        "AUXILIARY DIRECTION WEIGHT",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["ink"],
    )
    weight_text = "   ".join(
        f"{DISPLAY_NAMES[key]} {configurations[key]['auxiliary_weight']:.2f}"
        for key in ("physionet", "openbmi", "liu2024")
    )
    ax.text(
        right[0] + right[2] / 2,
        0.208,
        weight_text,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="center",
        color=COLORS["coral"],
    )

    card(
        ax,
        0.035,
        0.045,
        0.93,
        0.065,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.018,
    )
    ax.text(
        0.055,
        0.077,
        "TRANSLATIONAL BOUNDARY",
        transform=ax.transAxes,
        color=COLORS["gold"],
        fontsize=10.5,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.205,
        0.077,
        "3-window confirmation is prepared, but continuous false activations/min and locked external validation are not yet evaluated.",
        transform=ax.transAxes,
        color="white",
        fontsize=10.3,
        va="center",
    )

    save_figure(fig, output_dir, "01_v7_poster_graphical_abstract", dpi)


def create_pipeline_figure(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    dataset = report["dataset"]
    configs = report["patient_calibration"]["selected_configurations"]

    fig = plt.figure(figsize=(19.2, 7.2), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.text(
        0.035,
        0.94,
        "Translational V7: From Heterogeneous EEG to a Binary Intent Proposal",
        transform=ax.transAxes,
        fontsize=26,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.037,
        0.84,
        "Dataset identity remains explicit throughout quality control, fusion, probability calibration, and rapid patient adaptation",
        transform=ax.transAxes,
        fontsize=13,
        color=COLORS["muted"],
    )

    x_positions = [0.035, 0.235, 0.43, 0.66, 0.825]
    widths = [0.17, 0.16, 0.20, 0.135, 0.14]
    y = 0.40
    h = 0.35

    card(ax, x_positions[0], y, widths[0], h, radius=0.022)
    section_header(ax, x_positions[0] + 0.016, y + h - 0.045, "01", "Input domains", COLORS["blue"])
    dataset_y = [y + 0.22, y + 0.145, y + 0.07]
    for key, item_y in zip(("physionet", "openbmi", "liu2024"), dataset_y):
        values = dataset["datasets"][key]
        color = DATASET_COLORS[key]
        circle = Circle(
            (x_positions[0] + 0.032, item_y),
            0.017,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(circle)
        ax.text(
            x_positions[0] + 0.032,
            item_y,
            str(values["subjects"]),
            transform=ax.transAxes,
            fontsize=7.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        ax.text(
            x_positions[0] + 0.057,
            item_y + 0.012,
            DISPLAY_NAMES[key],
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x_positions[0] + 0.057,
            item_y - 0.014,
            f"{values['rows']:,} windows",
            transform=ax.transAxes,
            fontsize=8.5,
            color=COLORS["muted"],
            va="center",
        )

    card(ax, x_positions[1], y, widths[1], h, radius=0.022)
    section_header(ax, x_positions[1] + 0.016, y + h - 0.045, "02", "Native QC", COLORS["teal"])
    qc_steps = [
        "Detect bad channels",
        "Repair before reference",
        "Common-average reference",
        "Resample to 100 Hz",
        "Select 7 motor channels",
    ]
    for index, text in enumerate(qc_steps):
        yy = y + 0.235 - index * 0.045
        circle = Circle(
            (x_positions[1] + 0.028, yy),
            0.008,
            transform=ax.transAxes,
            facecolor=COLORS["teal"],
            edgecolor="none",
        )
        ax.add_patch(circle)
        ax.text(
            x_positions[1] + 0.045,
            yy,
            text,
            transform=ax.transAxes,
            fontsize=9.1,
            va="center",
        )

    card(ax, x_positions[2], y, widths[2], h, radius=0.022)
    section_header(ax, x_positions[2] + 0.016, y + h - 0.045, "03", "Physiology", COLORS["navy"])
    card(
        ax,
        x_positions[2] + 0.018,
        y + 0.17,
        widths[2] - 0.036,
        0.082,
        facecolor=COLORS["soft_blue"],
        edgecolor=COLORS["blue"],
        radius=0.012,
    )
    ax.text(
        x_positions[2] + 0.033,
        y + 0.225,
        "MRCP 0.1-4 Hz",
        transform=ax.transAxes,
        color=COLORS["blue"],
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        x_positions[2] + 0.033,
        y + 0.192,
        "slow trajectory + 5-seed FBCNet",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=8.8,
    )
    card(
        ax,
        x_positions[2] + 0.018,
        y + 0.065,
        widths[2] - 0.036,
        0.082,
        facecolor=COLORS["soft_teal"],
        edgecolor=COLORS["teal"],
        radius=0.012,
    )
    ax.text(
        x_positions[2] + 0.033,
        y + 0.12,
        "SMR 7-35 Hz",
        transform=ax.transAxes,
        color=COLORS["teal"],
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        x_positions[2] + 0.033,
        y + 0.087,
        "mu/beta + Riemannian covariance",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=8.8,
    )

    card(ax, x_positions[3], y, widths[3], h, radius=0.022)
    section_header(ax, x_positions[3] + 0.015, y + h - 0.045, "04", "Fusion", COLORS["gold"])
    fusion_rows = [
        ("MRCP + FBCNet", "60%", COLORS["navy"]),
        ("Riemannian", "25%", COLORS["blue"]),
        ("SMR", "15%", COLORS["teal"]),
    ]
    for index, (label, value, color) in enumerate(fusion_rows):
        yy = y + 0.225 - index * 0.06
        ax.text(
            x_positions[3] + 0.018,
            yy,
            label,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            va="center",
        )
        ax.text(
            x_positions[3] + widths[3] - 0.018,
            yy,
            value,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            ha="right",
            va="center",
        )
    ax.text(
        x_positions[3] + widths[3] / 2,
        y + 0.045,
        "Per-domain\nPlatt calibration",
        transform=ax.transAxes,
        fontsize=9.0,
        color=COLORS["muted"],
        ha="center",
        va="center",
    )

    card(ax, x_positions[4], y, widths[4], h, radius=0.022)
    section_header(ax, x_positions[4] + 0.015, y + h - 0.045, "05", "Binary gate", COLORS["coral"])
    card(
        ax,
        x_positions[4] + 0.025,
        y + 0.18,
        widths[4] - 0.05,
        0.055,
        facecolor=COLORS["soft_gray"],
        edgecolor=COLORS["muted"],
        radius=0.018,
    )
    card(
        ax,
        x_positions[4] + 0.025,
        y + 0.095,
        widths[4] - 0.05,
        0.055,
        facecolor=COLORS["soft_teal"],
        edgecolor=COLORS["teal"],
        radius=0.018,
    )
    ax.text(
        x_positions[4] + widths[4] / 2,
        y + 0.207,
        "REST",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["muted"],
    )
    ax.text(
        x_positions[4] + widths[4] / 2,
        y + 0.122,
        "INTENT",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["teal"],
    )
    ax.text(
        x_positions[4] + widths[4] / 2,
        y + 0.045,
        "No direction\noutput",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["coral"],
        fontweight="bold",
        ha="center",
        va="center",
    )

    for index in range(len(x_positions) - 1):
        arrow(
            ax,
            (x_positions[index] + widths[index] + 0.006, y + h / 2),
            (x_positions[index + 1] - 0.006, y + h / 2),
            color=COLORS["muted"],
            mutation_scale=15,
        )

    card(
        ax,
        0.235,
        0.14,
        0.59,
        0.16,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.02,
    )
    ax.text(
        0.255,
        0.26,
        "RAPID CALIBRATION FROM RECORDING 0",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["coral"],
    )
    ax.text(
        0.255,
        0.218,
        "10 REST + 10 LEFT + 10 RIGHT windows  >  internal P(left) + P(right)  >  binary P(intent)",
        transform=ax.transAxes,
        fontsize=10.2,
        color=COLORS["ink"],
    )
    ax.text(
        0.255,
        0.176,
        "Later recordings only are scored",
        transform=ax.transAxes,
        fontsize=9.2,
        color=COLORS["muted"],
    )
    for index, key in enumerate(("physionet", "openbmi", "liu2024")):
        xx = 0.54 + index * 0.09
        color = DATASET_COLORS[key]
        ax.text(
            xx,
            0.176,
            f"{DISPLAY_NAMES[key]}  w={configs[key]['auxiliary_weight']:.2f}",
            transform=ax.transAxes,
            fontsize=8.8,
            color=color,
            fontweight="bold",
            ha="center",
        )

    card(
        ax,
        0.035,
        0.045,
        0.93,
        0.055,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.016,
    )
    ax.text(
        0.5,
        0.072,
        "Known rehabilitation cue selects the requested movement  |  EEG confirms whether credible intent followed",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    save_figure(fig, output_dir, "02_v7_poster_dataset_pipeline", dpi)


def create_results_dashboard(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    patient = report["patient_calibration"]["development_observed_conservative"]
    patient_targeted = report["patient_calibration"][
        "development_observed_sensitivity_targeted"
    ]
    population = report["calibrated_stage_controller"]["operating_points"][
        "conservative"
    ]["development_observed"]
    configs = report["patient_calibration"]["selected_configurations"]
    v5 = report["v5_directional_reference"]["conservative"]
    v6 = report["v6_intent_only_reference"]["conservative"]

    fig = plt.figure(figsize=(17.5, 10), facecolor=COLORS["background"])
    title_ax = fig.add_axes((0.04, 0.87, 0.92, 0.10))
    title_ax.set_axis_off()
    title_ax.text(
        0,
        0.88,
        "V7 Results: Better Intent Recovery with High Rest Rejection",
        fontsize=28,
        fontweight="bold",
        va="top",
    )
    title_ax.text(
        0,
        0.22,
        "Patient-adapted, binary REST versus INTENT evaluation on later recordings",
        fontsize=15,
        color=COLORS["ink"],
        fontweight="bold",
        va="top",
    )
    title_ax.text(
        1,
        0.22,
        "DEVELOPMENT-OBSERVED  |  4,645 EVALUATION WINDOWS",
        fontsize=12,
        color=COLORS["coral"],
        fontweight="bold",
        ha="right",
        va="top",
    )

    metric_ax = fig.add_axes((0.04, 0.68, 0.92, 0.17))
    metric_ax.set_axis_off()
    metric_cards = [
        (
            "Balanced accuracy",
            pct(patient["intent_balanced_accuracy"]),
            COLORS["navy"],
            "95% CI 67.5-78.7%",
            False,
        ),
        (
            "Intent sensitivity",
            pct(patient["intent_sensitivity"]),
            COLORS["teal"],
            f"MISSED INTENT: {pct(patient['missed_intent_rate']):.1f}%",
            False,
        ),
        (
            "Safety result: rest specificity",
            pct(patient["intent_specificity"]),
            COLORS["gold"],
            f"ONLY {pct(patient['false_activation_rate_per_window']):.1f}% FALSE ACTIVATION / REST WINDOW",
            True,
        ),
        (
            "Intent AUROC",
            pct(patient["intent_auroc"]),
            COLORS["coral"],
            "THRESHOLD-INDEPENDENT DISCRIMINATION",
            False,
        ),
    ]
    for index, (label, value, color, note, safety_highlight) in enumerate(metric_cards):
        x = 0.005 + index * 0.25
        card(
            metric_ax,
            x,
            0.06,
            0.225,
            0.84,
            facecolor=COLORS["soft_gold"] if safety_highlight else COLORS["paper"],
            edgecolor=color,
            linewidth=3.0 if safety_highlight else 1.8,
            radius=0.025,
        )
        metric_ax.text(
            x + 0.018,
            0.76,
            label.upper(),
            transform=metric_ax.transAxes,
            fontsize=12.4 if safety_highlight else 12,
            color=COLORS["ink"],
            fontweight="bold",
        )
        display_value = value if label != "Intent AUROC" else value / 100.0
        suffix = "%" if label != "Intent AUROC" else ""
        precision = ".1f" if label != "Intent AUROC" else ".3f"
        metric_ax.text(
            x + 0.018,
            0.39,
            f"{display_value:{precision}}{suffix}",
            transform=metric_ax.transAxes,
            fontsize=31 if safety_highlight else 29,
            color=color,
            fontweight="bold",
        )
        metric_ax.text(
            x + 0.018,
            0.16,
            note,
            transform=metric_ax.transAxes,
            fontsize=11.5 if safety_highlight else 10.8,
            color=COLORS["ink"],
            fontweight="bold",
        )

    version_ax = fig.add_axes((0.06, 0.175, 0.41, 0.425), facecolor=COLORS["paper"])
    domain_ax = fig.add_axes((0.54, 0.175, 0.41, 0.425), facecolor=COLORS["paper"])

    versions = ["V5\ndirectional", "V6\nbinary", "V7\nauxiliary"]
    version_metrics = [v5, v6, patient]
    x = np.arange(len(versions))
    width = 0.22
    for offset, key, label, color in (
        (-width, "intent_balanced_accuracy", "Balanced accuracy", COLORS["navy"]),
        (0.0, "intent_sensitivity", "Sensitivity", COLORS["teal"]),
        (width, "intent_specificity", "Specificity", COLORS["gold"]),
    ):
        values = [pct(item[key]) for item in version_metrics]
        bars = version_ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            version_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color=COLORS["ink"],
            )
    version_ax.set_title("Model Progression", fontsize=19, loc="left", pad=24)
    version_ax.text(
        0,
        1.025,
        "Conservative operating point",
        transform=version_ax.transAxes,
        fontsize=12.5,
        color=COLORS["ink"],
        fontweight="bold",
    )
    version_ax.set_xticks(x, versions)
    version_ax.set_ylim(0, 115)
    version_ax.set_yticks(np.arange(0, 101, 20))
    version_ax.set_ylabel(
        "Development-observed result (%)", fontsize=13.5, fontweight="bold"
    )
    version_ax.tick_params(axis="both", labelsize=12.5, colors=COLORS["ink"])
    for label in version_ax.get_xticklabels() + version_ax.get_yticklabels():
        label.set_fontweight("bold")
    version_ax.grid(axis="y", color=COLORS["line"], alpha=0.55, linewidth=0.8)
    version_ax.spines[["top", "right"]].set_visible(False)
    version_ax.legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        prop={"size": 11.5, "weight": "bold"},
        bbox_to_anchor=(0.5, 1.0),
    )
    domain_keys = ["physionet", "openbmi", "liu2024"]
    domains = [
        f"{DISPLAY_NAMES[key]}\naux w={configs[key]['auxiliary_weight']:.2f}"
        for key in domain_keys
    ]
    x2 = np.arange(len(domains))
    for offset, key, label, color in (
        (-width, "intent_balanced_accuracy", "Balanced accuracy", COLORS["navy"]),
        (0.0, "intent_sensitivity", "Sensitivity", COLORS["teal"]),
        (width, "intent_specificity", "Specificity", COLORS["gold"]),
    ):
        values = [pct(patient["by_dataset"][domain][key]) for domain in domain_keys]
        bars = domain_ax.bar(x2 + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            domain_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color=COLORS["ink"],
            )
    domain_ax.set_title(
        "Performance Remains Domain Dependent", fontsize=19, loc="left", pad=24
    )
    domain_ax.text(
        0,
        1.025,
        "V7 patient-adapted conservative result",
        transform=domain_ax.transAxes,
        fontsize=12.5,
        color=COLORS["ink"],
        fontweight="bold",
    )
    domain_ax.set_xticks(x2, domains)
    domain_ax.set_ylim(0, 115)
    domain_ax.set_yticks(np.arange(0, 101, 20))
    domain_ax.set_ylabel("Result (%)", fontsize=13.5, fontweight="bold")
    domain_ax.tick_params(axis="both", labelsize=12.5, colors=COLORS["ink"])
    for label in domain_ax.get_xticklabels() + domain_ax.get_yticklabels():
        label.set_fontweight("bold")
    domain_ax.grid(axis="y", color=COLORS["line"], alpha=0.55, linewidth=0.8)
    domain_ax.spines[["top", "right"]].set_visible(False)
    footer_ax = fig.add_axes((0.04, 0.018, 0.92, 0.105))
    footer_ax.set_axis_off()
    card(
        footer_ax,
        0,
        0.05,
        1,
        0.90,
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        linewidth=2.6,
        radius=0.025,
    )
    footer_ax.text(
        0.02,
        0.68,
        "LOW FALSE ACTIVATION",
        transform=footer_ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    footer_ax.text(
        0.245,
        0.68,
        f"{pct(patient['intent_specificity']):.1f}% REST SPECIFICITY  |  ONLY {pct(patient['false_activation_rate_per_window']):.1f}% FALSE ACTIVATION PER ANALYZED REST WINDOW",
        transform=footer_ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    footer_ax.text(
        0.02,
        0.28,
        f"Conservative sensitivity {pct(patient['intent_sensitivity']):.1f}%  |  Sensitivity-targeted option {pct(patient_targeted['intent_sensitivity']):.1f}% sensitivity / {pct(patient_targeted['intent_specificity']):.1f}% specificity",
        transform=footer_ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
    )
    footer_ax.text(
        0.98,
        0.28,
        "Cue-locked result; continuous false activations/min not yet measured",
        transform=footer_ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
        color=COLORS["ink"],
        va="center",
        ha="right",
    )

    save_figure(fig, output_dir, "03_v7_poster_results_dashboard", dpi)


def write_readme(output_dir: Path) -> None:
    readme = """# Translational V7 Poster Figures

These figures are generated from the schema-8 V7 metrics file by:

```bash
python scripts/41_create_v7_poster_figures.py
```

## Recommended use

- `01_v7_poster_graphical_abstract`: Main poster figure. It combines the three
  datasets, native-montage processing, V7 fusion and patient calibration,
  binary output, headline metrics, domain results, and validation boundary.
- `02_v7_poster_dataset_pipeline`: Methods figure. It provides a larger,
  presentation-readable dataset-to-output workflow.
- `03_v7_poster_results_dashboard`: Results figure. It emphasizes the V5/V6/V7
  progression, domain variation, auxiliary weights, and operating-point
  tradeoff.

Each figure is exported as a 300-DPI PNG and as vector SVG and PDF files.

## Interpretation boundary

All values are development-observed and cue-locked. Specificity is measured per
rest window, not as continuous false activations per minute. The locked external
cohort and continuous no-control evaluation remain unavailable.
"""
    (output_dir / "POSTER_FIGURES.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    report = load_metrics(args.metrics.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    create_graphical_abstract(report, output_dir, args.dpi)
    create_pipeline_figure(report, output_dir, args.dpi)
    create_results_dashboard(report, output_dir, args.dpi)
    write_readme(output_dir)
    print(f"Saved poster figures to {output_dir}")


if __name__ == "__main__":
    main()
