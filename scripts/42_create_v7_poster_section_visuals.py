#!/usr/bin/env python3
"""Create standalone poster visuals for V7 poster Sections 3 through 7."""

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
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "sections"

COLORS = {
    "background": "#F4F1E8",
    "paper": "#FFFDF8",
    "ink": "#17324D",
    "muted": "#607080",
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

DATASET_SOFT = {
    "physionet": COLORS["soft_blue"],
    "openbmi": COLORS["soft_teal"],
    "liu2024": COLORS["soft_coral"],
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
            "font.size": 11,
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
        raise ValueError("Section visuals require the schema-8 V7 metrics file.")
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
    linewidth: float = 1.2,
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


def base_figure(
    section: str,
    title: str,
    subtitle: str,
    accent: str,
    *,
    figsize: tuple[float, float] = (13.0, 8.5),
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=figsize, facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    card(
        ax,
        0.038,
        0.898,
        0.075,
        0.054,
        facecolor=accent,
        edgecolor=accent,
        radius=0.018,
    )
    if len(title) > 65:
        title_size = 15.5
    elif len(title) > 52:
        title_size = 18.0
    elif len(title) > 48:
        title_size = 19.5
    else:
        title_size = 23.0
    ax.text(
        0.0755,
        0.925,
        section,
        transform=ax.transAxes,
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0.13,
        0.94,
        title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.132,
        0.89,
        subtitle,
        transform=ax.transAxes,
        fontsize=11.5,
        color=COLORS["muted"],
        va="top",
    )
    card(
        ax,
        0.785,
        0.912,
        0.175,
        0.035,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.014,
    )
    ax.text(
        0.8725,
        0.9295,
        "V7  |  DEVELOPMENT-OBSERVED",
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )
    return fig, ax


def footer(ax: plt.Axes, label: str, text: str, color: str) -> None:
    card(
        ax,
        0.04,
        0.035,
        0.92,
        0.063,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.018,
    )
    label_color = COLORS["blue"] if color in {COLORS["ink"], COLORS["navy"]} else color
    ax.text(
        0.06,
        0.066,
        label.upper(),
        transform=ax.transAxes,
        fontsize=9.3,
        fontweight="bold",
        color=label_color,
        va="center",
    )
    ax.text(
        0.21,
        0.066,
        text,
        transform=ax.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color="white",
        va="center",
    )


def waveform(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    *,
    cycles: float,
    trend: float = 0.0,
) -> None:
    t = np.linspace(0.0, 1.0, 120)
    signal = 0.72 * np.sin(2.0 * np.pi * cycles * t) + trend * (t - 0.5)
    ax.plot(
        x + width * t,
        y + 0.5 * height + 0.42 * height * signal,
        transform=ax.transAxes,
        color=color,
        linewidth=2.2,
        clip_on=False,
        zorder=5,
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi if suffix == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def create_section_3_datasets(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    dataset = report["dataset"]
    fig, ax = base_figure(
        "03",
        "Datasets: Healthy Development and Stroke Translation",
        "Three acquisition domains increase diversity while preserving their distinct negative-state semantics",
        COLORS["blue"],
    )

    card(
        ax,
        0.045,
        0.72,
        0.91,
        0.11,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.02,
    )
    summary = [
        (f"{dataset['subjects']}", "participants"),
        (f"{dataset['rows']:,}", "cue-locked EEG windows"),
        ("3", "recording domains"),
    ]
    for index, (value, label) in enumerate(summary):
        x = 0.19 + index * 0.31
        ax.text(
            x,
            0.778,
            value,
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        ax.text(
            x,
            0.744,
            label,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#DCE7EC",
            ha="center",
            va="center",
        )

    cohort_info = {
        "physionet": ("HEALTHY", "Motor imagery", "Protocol rest"),
        "openbmi": ("HEALTHY", "Motor imagery", "Post-task rest"),
        "liu2024": ("STROKE", "Acute-stroke imagery", "Protocol break/rest"),
    }
    x_positions = [0.045, 0.355, 0.665]
    for key, x in zip(("physionet", "openbmi", "liu2024"), x_positions):
        values = dataset["datasets"][key]
        cohort, task, negative = cohort_info[key]
        color = DATASET_COLORS[key]
        card(
            ax,
            x,
            0.25,
            0.285,
            0.41,
            facecolor=DATASET_SOFT[key],
            edgecolor=color,
            linewidth=1.5,
            radius=0.024,
        )
        card(
            ax,
            x + 0.018,
            0.598,
            0.083,
            0.034,
            facecolor=color,
            edgecolor=color,
            radius=0.012,
        )
        ax.text(
            x + 0.0595,
            0.615,
            cohort,
            transform=ax.transAxes,
            fontsize=8.3,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.022,
            0.552,
            DISPLAY_NAMES[key],
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            x + 0.022,
            0.514,
            task,
            transform=ax.transAxes,
            fontsize=10,
            color=COLORS["muted"],
        )
        ax.text(
            x + 0.022,
            0.43,
            str(values["subjects"]),
            transform=ax.transAxes,
            fontsize=28,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + 0.022,
            0.395,
            "participants",
            transform=ax.transAxes,
            fontsize=9.5,
            color=COLORS["muted"],
        )
        ax.text(
            x + 0.145,
            0.43,
            f"{values['rows']:,}",
            transform=ax.transAxes,
            fontsize=22,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            x + 0.145,
            0.395,
            "windows",
            transform=ax.transAxes,
            fontsize=9.5,
            color=COLORS["muted"],
        )
        ax.plot(
            [x + 0.022, x + 0.263],
            [0.355, 0.355],
            transform=ax.transAxes,
            color=color,
            alpha=0.35,
            linewidth=1.2,
        )
        ax.text(
            x + 0.022,
            0.315,
            "NEGATIVE STATE",
            transform=ax.transAxes,
            fontsize=8.3,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + 0.022,
            0.278,
            negative,
            transform=ax.transAxes,
            fontsize=10.2,
            fontweight="bold",
            color=COLORS["ink"],
        )

    card(
        ax,
        0.18,
        0.14,
        0.64,
        0.065,
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        radius=0.018,
    )
    ax.text(
        0.5,
        0.172,
        "Dataset identity is retained during training, probability calibration, and threshold selection",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
    )
    footer(
        ax,
        "Interpretation",
        "Protocol rest, post-task rest, and stroke break/rest are not equivalent to unrestricted real-world idle.",
        COLORS["blue"],
    )
    save_figure(fig, output_dir, "03_datasets_and_cohorts", dpi)


def create_section_4_signal_processing(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    preprocessing = report["preprocessing"]
    fig, ax = base_figure(
        "04",
        "Signal Processing and Physiological Representations",
        "Quality control is completed in the native montage before reduction to shared motor-region channels",
        COLORS["teal"],
    )

    stages = [
        ("RAW EEG", "native montage", COLORS["soft_gray"], COLORS["muted"]),
        ("RECORDING QC", "bad channels + noise", COLORS["soft_gold"], COLORS["gold"]),
        ("REPAIR", "before referencing", COLORS["soft_coral"], COLORS["coral"]),
        ("CAR", "common average", COLORS["soft_blue"], COLORS["blue"]),
        ("100 Hz", "7 motor channels", COLORS["soft_teal"], COLORS["teal"]),
    ]
    x_positions = [0.045, 0.225, 0.405, 0.585, 0.765]
    for index, ((title, note, soft, color), x) in enumerate(zip(stages, x_positions)):
        card(
            ax,
            x,
            0.665,
            0.145,
            0.105,
            facecolor=soft,
            edgecolor=color,
            radius=0.017,
            linewidth=1.3,
        )
        ax.text(
            x + 0.0725,
            0.73,
            title,
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
            color=color,
            ha="center",
        )
        ax.text(
            x + 0.0725,
            0.695,
            note,
            transform=ax.transAxes,
            fontsize=8.8,
            color=COLORS["muted"],
            ha="center",
        )
        if index < len(stages) - 1:
            arrow(
                ax,
                (x + 0.151, 0.717),
                (x_positions[index + 1] - 0.006, 0.717),
                color=COLORS["muted"],
            )

    ax.text(
        0.5,
        0.625,
        "FC3  |  FC4  |  C3  |  Cz  |  C4  |  CP3  |  CP4",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
    )
    arrow(ax, (0.837, 0.658), (0.837, 0.585), color=COLORS["teal"])

    branches = [
        (
            "MRCP",
            "0.1-4 Hz",
            "Slow preparation and onset\ntrajectory, slope, and area",
            COLORS["soft_blue"],
            COLORS["blue"],
            1.15,
        ),
        (
            "SMR",
            "7-35 Hz",
            "Mu/beta power and individualized\nsensorimotor rhythm",
            COLORS["soft_teal"],
            COLORS["teal"],
            5.2,
        ),
        (
            "RIEMANNIAN",
            "4 SMR bands",
            "Spatial covariance across seven\nmotor-region channels",
            COLORS["soft_gold"],
            COLORS["gold"],
            3.0,
        ),
        (
            "QUALITY / OOD",
            "1-45 Hz",
            "Availability, artifact, and\ndistribution-shift evidence",
            COLORS["soft_coral"],
            COLORS["coral"],
            7.0,
        ),
    ]
    branch_x = [0.045, 0.285, 0.525, 0.765]
    for (title, band, note, soft, color, cycles), x in zip(branches, branch_x):
        card(
            ax,
            x,
            0.25,
            0.19,
            0.29,
            facecolor=soft,
            edgecolor=color,
            radius=0.022,
            linewidth=1.4,
        )
        title_size = 11.2 if title in {"RIEMANNIAN", "QUALITY / OOD"} else 14.0
        ax.text(
            x + 0.02,
            0.485,
            title,
            transform=ax.transAxes,
            fontsize=title_size,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + 0.17,
            0.485,
            band,
            transform=ax.transAxes,
            fontsize=9.8,
            fontweight="bold",
            color=COLORS["muted"],
            ha="right",
        )
        waveform(ax, x + 0.025, 0.38, 0.14, 0.06, color, cycles=cycles, trend=-0.5 if title == "MRCP" else 0.0)
        ax.text(
            x + 0.095,
            0.305,
            note,
            transform=ax.transAxes,
            fontsize=7.7,
            color=COLORS["ink"],
            va="center",
            ha="center",
            linespacing=1.2,
        )
        if title == "QUALITY / OOD":
            ax.text(
                x + 0.02,
                0.272,
                "Gate only; not intent evidence",
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )

    card(
        ax,
        0.22,
        0.14,
        0.56,
        0.063,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["line"],
        radius=0.016,
    )
    ax.text(
        0.5,
        0.171,
        f"Reference: {preprocessing['reference']}   |   CSD available when native montage supports it",
        transform=ax.transAxes,
        fontsize=9.6,
        color=COLORS["muted"],
        ha="center",
        va="center",
    )
    footer(
        ax,
        "Design rule",
        "Full-montage cleaning precedes channel reduction; quality features control abstention rather than class evidence.",
        COLORS["teal"],
    )
    save_figure(fig, output_dir, "04_signal_processing_and_features", dpi)


def create_section_5_decoder_calibration(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    configs = report["patient_calibration"]["selected_configurations"]
    fig, ax = base_figure(
        "05",
        "V7 Decoder and Rapid Patient Calibration",
        "A population fusion is individualized with early labeled EEG while the external output remains strictly binary",
        COLORS["coral"],
    )

    card(
        ax,
        0.045,
        0.60,
        0.91,
        0.19,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["line"],
        radius=0.022,
    )
    ax.text(
        0.068,
        0.747,
        "POPULATION INTENT FUSION",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["muted"],
    )
    fusion = [
        ("MRCP + FBCNet", 0.60, COLORS["navy"]),
        ("Riemannian", 0.25, COLORS["blue"]),
        ("SMR", 0.15, COLORS["teal"]),
    ]
    start_x = 0.068
    total_width = 0.56
    running = start_x
    for label, weight, color in fusion:
        width = total_width * weight
        card(
            ax,
            running,
            0.655,
            width,
            0.06,
            facecolor=color,
            edgecolor=color,
            radius=0.008,
            linewidth=0,
        )
        ax.text(
            running + width / 2,
            0.685,
            f"{label}  {int(weight * 100)}%",
            transform=ax.transAxes,
            fontsize=8.8 if weight < 0.3 else 9.4,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        running += width
    card(
        ax,
        0.67,
        0.64,
        0.25,
        0.09,
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        radius=0.016,
    )
    ax.text(
        0.795,
        0.695,
        "PER-DOMAIN CALIBRATION",
        transform=ax.transAxes,
        fontsize=9.7,
        fontweight="bold",
        color=COLORS["gold"],
        ha="center",
    )
    ax.text(
        0.795,
        0.662,
        "Platt scaling + thresholds",
        transform=ax.transAxes,
        fontsize=9.0,
        color=COLORS["muted"],
        ha="center",
    )

    arrow(ax, (0.5, 0.592), (0.5, 0.548), color=COLORS["coral"])
    card(
        ax,
        0.045,
        0.325,
        0.91,
        0.215,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.022,
        linewidth=1.4,
    )
    ax.text(
        0.068,
        0.493,
        "RECORDING 0: 30-WINDOW PATIENT CALIBRATION",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["coral"],
    )
    classes = [
        ("10 REST", COLORS["muted"], COLORS["soft_gray"]),
        ("10 LEFT", COLORS["blue"], COLORS["soft_blue"]),
        ("10 RIGHT", COLORS["teal"], COLORS["soft_teal"]),
    ]
    for index, (label, color, soft) in enumerate(classes):
        x = 0.075 + index * 0.16
        card(
            ax,
            x,
            0.405,
            0.135,
            0.054,
            facecolor=soft,
            edgecolor=color,
            radius=0.016,
        )
        ax.text(
            x + 0.0675,
            0.432,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
    arrow(ax, (0.545, 0.432), (0.615, 0.432), color=COLORS["coral"])
    card(
        ax,
        0.625,
        0.385,
        0.285,
        0.095,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["coral"],
        radius=0.018,
    )
    ax.text(
        0.7675,
        0.445,
        "P(intent) = P(left) + P(right)",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
    )
    ax.text(
        0.7675,
        0.407,
        "Direction is internal calibration evidence only",
        transform=ax.transAxes,
        fontsize=8.8,
        color=COLORS["muted"],
        ha="center",
    )
    ax.text(
        0.068,
        0.357,
        "Controller-selected auxiliary direction weight",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["muted"],
    )
    for index, key in enumerate(("physionet", "openbmi", "liu2024")):
        x = 0.34 + index * 0.19
        ax.text(
            x,
            0.357,
            f"{DISPLAY_NAMES[key]}  {configs[key]['auxiliary_weight']:.2f}",
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color=DATASET_COLORS[key],
            ha="center",
        )

    arrow(ax, (0.5, 0.316), (0.5, 0.272), color=COLORS["teal"])
    card(
        ax,
        0.19,
        0.15,
        0.24,
        0.105,
        facecolor=COLORS["soft_gray"],
        edgecolor=COLORS["muted"],
        radius=0.024,
    )
    card(
        ax,
        0.57,
        0.15,
        0.24,
        0.105,
        facecolor=COLORS["soft_teal"],
        edgecolor=COLORS["teal"],
        radius=0.024,
    )
    ax.text(
        0.31,
        0.207,
        "REST",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        color=COLORS["muted"],
        ha="center",
        va="center",
    )
    ax.text(
        0.69,
        0.207,
        "INTENT",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        color=COLORS["teal"],
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        0.12,
        "Binary output only; the known rehabilitation cue supplies the requested movement",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
    )
    footer(
        ax,
        "Key finding",
        "Direction evidence helped healthy-domain calibration but was assigned zero weight for Liu stroke data.",
        COLORS["coral"],
    )
    save_figure(fig, output_dir, "05_v7_decoder_and_patient_calibration", dpi)


def create_section_6_evaluation(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    roles = report["subject_roles"]
    patient = report["patient_calibration"]
    fig, ax = base_figure(
        "06",
        "Evaluation Protocol and Leakage Prevention",
        "Participant roles select different parts of the pipeline; early calibration rows are never scored",
        COLORS["gold"],
    )

    role_names = [
        ("DEVELOPMENT", "Fit features and models", COLORS["blue"], COLORS["soft_blue"]),
        ("CONTROLLER", "Select fusion and policy", COLORS["teal"], COLORS["soft_teal"]),
        ("OBSERVED", "Exploratory result", COLORS["gold"], COLORS["soft_gold"]),
        ("LOCKED EXTERNAL", "Confirmatory claim", COLORS["coral"], COLORS["soft_coral"]),
    ]
    x_positions = [0.04, 0.285, 0.53, 0.775]
    total_by_role = {
        "DEVELOPMENT": sum(len(values["development"]) for values in roles.values()),
        "CONTROLLER": sum(
            len(values["controller_validation"]) for values in roles.values()
        ),
        "OBSERVED": sum(
            len(values["development_observed"]) for values in roles.values()
        ),
        "LOCKED EXTERNAL": sum(
            len(values["locked_external"]) for values in roles.values()
        ),
    }
    for index, ((name, purpose, color, soft), x) in enumerate(zip(role_names, x_positions)):
        card(
            ax,
            x,
            0.64,
            0.19,
            0.16,
            facecolor=soft,
            edgecolor=color,
            radius=0.021,
            linewidth=1.4,
        )
        ax.text(
            x + 0.095,
            0.758,
            name,
            transform=ax.transAxes,
            fontsize=9.2,
            fontweight="bold",
            color=color,
            ha="center",
        )
        ax.text(
            x + 0.095,
            0.704,
            str(total_by_role[name]),
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            color=COLORS["ink"],
            ha="center",
        )
        ax.text(
            x + 0.095,
            0.666,
            purpose,
            transform=ax.transAxes,
            fontsize=8.7,
            color=COLORS["muted"],
            ha="center",
        )
        if index < len(role_names) - 1:
            arrow(
                ax,
                (x + 0.198, 0.72),
                (x_positions[index + 1] - 0.007, 0.72),
                color=COLORS["muted"],
            )

    card(
        ax,
        0.06,
        0.365,
        0.88,
        0.20,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["line"],
        radius=0.022,
    )
    ax.text(
        0.085,
        0.522,
        "PATIENT-LEVEL RECORDING BOUNDARY",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["muted"],
    )
    card(
        ax,
        0.09,
        0.405,
        0.31,
        0.085,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.018,
    )
    ax.text(
        0.245,
        0.461,
        "RECORDING 0: CALIBRATION",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
    )
    ax.text(
        0.245,
        0.426,
        "30 labeled windows; excluded from metrics",
        transform=ax.transAxes,
        fontsize=9.1,
        color=COLORS["muted"],
        ha="center",
    )
    arrow(ax, (0.415, 0.448), (0.57, 0.448), color=COLORS["ink"], width=2.2)
    card(
        ax,
        0.585,
        0.405,
        0.31,
        0.085,
        facecolor=COLORS["soft_teal"],
        edgecolor=COLORS["teal"],
        radius=0.018,
    )
    ax.text(
        0.74,
        0.461,
        "LATER RECORDINGS: EVALUATION",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["teal"],
        ha="center",
    )
    ax.text(
        0.74,
        0.426,
        f"{patient['development_observed_evaluation_rows']:,} scored windows",
        transform=ax.transAxes,
        fontsize=9.1,
        color=COLORS["muted"],
        ha="center",
    )

    metric_cards = [
        ("BA", "(sensitivity + specificity) / 2", COLORS["navy"]),
        ("SENSITIVITY", "intent windows detected", COLORS["teal"]),
        ("SPECIFICITY", "rest windows rejected", COLORS["gold"]),
        ("AUROC", "ranking across thresholds", COLORS["coral"]),
    ]
    for index, (metric, definition, color) in enumerate(metric_cards):
        x = 0.055 + index * 0.235
        card(
            ax,
            x,
            0.16,
            0.205,
            0.13,
            facecolor=COLORS["paper"],
            edgecolor=color,
            radius=0.018,
        )
        ax.text(
            x + 0.1025,
            0.242,
            metric,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color=color,
            ha="center",
        )
        ax.text(
            x + 0.1025,
            0.195,
            definition,
            transform=ax.transAxes,
            fontsize=8.6,
            color=COLORS["muted"],
            ha="center",
        )

    ax.text(
        0.5,
        0.12,
        "Conservative policy prioritizes rest rejection  |  Sensitivity-targeted policy accepts more rest errors",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
    )
    footer(
        ax,
        "Validation boundary",
        "The locked external role is empty; V7 is exploratory and cannot be labeled a confirmatory external result.",
        COLORS["gold"],
    )
    save_figure(fig, output_dir, "06_evaluation_protocol_and_leakage_control", dpi)


def create_section_7_results(
    report: Mapping[str, Any], output_dir: Path, dpi: int
) -> None:
    patient = report["patient_calibration"]["development_observed_conservative"]
    targeted = report["patient_calibration"][
        "development_observed_sensitivity_targeted"
    ]
    population = report["patient_calibration"]["population_matched_conservative"]
    v5 = report["v5_directional_reference"]["conservative"]
    v6 = report["v6_intent_only_reference"]["conservative"]
    fig, ax = base_figure(
        "07",
        "Results: Intent Recovery, Rest Rejection, and Domain Variation",
        "Patient-adapted conservative performance on later recordings",
        COLORS["navy"],
    )

    kpis = [
        ("BALANCED ACCURACY", pct(patient["intent_balanced_accuracy"]), COLORS["navy"], "95% CI 67.5-78.7%"),
        ("SENSITIVITY", pct(patient["intent_sensitivity"]), COLORS["teal"], "missed intent 46.5%"),
        ("SPECIFICITY", pct(patient["intent_specificity"]), COLORS["gold"], "false activation/window 6.3%"),
        ("AUROC", float(patient["intent_auroc"]), COLORS["coral"], "threshold-independent"),
    ]
    for index, (label, value, color, note) in enumerate(kpis):
        x = 0.04 + index * 0.235
        card(
            ax,
            x,
            0.69,
            0.205,
            0.14,
            facecolor=COLORS["paper"],
            edgecolor=color,
            radius=0.019,
            linewidth=1.4,
        )
        ax.text(
            x + 0.018,
            0.793,
            label,
            transform=ax.transAxes,
            fontsize=8.7,
            fontweight="bold",
            color=COLORS["muted"],
        )
        display = f"{value:.1f}%" if label != "AUROC" else f"{value:.3f}"
        ax.text(
            x + 0.018,
            0.738,
            display,
            transform=ax.transAxes,
            fontsize=23,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + 0.018,
            0.706,
            note,
            transform=ax.transAxes,
            fontsize=7.9,
            color=COLORS["muted"],
        )

    domain_ax = fig.add_axes((0.065, 0.18, 0.54, 0.42), facecolor=COLORS["paper"])
    domain_keys = ["physionet", "openbmi", "liu2024"]
    labels = [DISPLAY_NAMES[key] for key in domain_keys]
    x = np.arange(len(labels))
    width = 0.22
    for offset, key, label, color in (
        (-width, "intent_balanced_accuracy", "Balanced accuracy", COLORS["navy"]),
        (0.0, "intent_sensitivity", "Sensitivity", COLORS["teal"]),
        (width, "intent_specificity", "Specificity", COLORS["gold"]),
    ):
        values = [pct(patient["by_dataset"][domain][key]) for domain in domain_keys]
        bars = domain_ax.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            domain_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                fontsize=8.2,
                fontweight="bold",
                color=color,
                ha="center",
                va="bottom",
            )
    domain_ax.set_title("Performance Remains Domain Dependent", loc="left", fontsize=14, pad=13)
    domain_ax.set_xticks(x, labels)
    domain_ax.set_ylim(0, 114)
    domain_ax.set_ylabel("Development-observed result (%)")
    domain_ax.grid(axis="y", color=COLORS["line"], alpha=0.55, linewidth=0.8)
    domain_ax.spines[["top", "right"]].set_visible(False)
    domain_ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=8.3)

    card(
        ax,
        0.64,
        0.43,
        0.315,
        0.18,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["line"],
        radius=0.021,
    )
    ax.text(
        0.665,
        0.57,
        "PATIENT ADAPTATION",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["muted"],
    )
    pop_ba = pct(population["intent_balanced_accuracy"])
    adapted_ba = pct(patient["intent_balanced_accuracy"])
    ax.text(
        0.68,
        0.5,
        f"{pop_ba:.1f}%",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=COLORS["muted"],
        ha="center",
    )
    arrow(ax, (0.73, 0.51), (0.82, 0.51), color=COLORS["teal"], width=2.4)
    ax.text(
        0.865,
        0.5,
        f"{adapted_ba:.1f}%",
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
        color=COLORS["teal"],
        ha="center",
    )
    ax.text(
        0.68,
        0.46,
        "matched population",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS["muted"],
        ha="center",
    )
    ax.text(
        0.865,
        0.46,
        "patient adapted",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS["teal"],
        ha="center",
    )

    card(
        ax,
        0.64,
        0.18,
        0.315,
        0.21,
        facecolor=COLORS["soft_gold"],
        edgecolor=COLORS["gold"],
        radius=0.021,
    )
    ax.text(
        0.665,
        0.35,
        "TRADEOFF AND VERSION CHECK",
        transform=ax.transAxes,
        fontsize=9.4,
        fontweight="bold",
        color=COLORS["gold"],
    )
    rows = [
        ("V5 conservative BA", pct(v5["intent_balanced_accuracy"]), COLORS["blue"]),
        ("V6 conservative BA", pct(v6["intent_balanced_accuracy"]), COLORS["muted"]),
        ("V7 conservative BA", pct(patient["intent_balanced_accuracy"]), COLORS["navy"]),
    ]
    for index, (label, value, color) in enumerate(rows):
        y = 0.309 - index * 0.043
        ax.text(
            0.665,
            y,
            label,
            transform=ax.transAxes,
            fontsize=8.6,
            color=COLORS["ink"],
            va="center",
        )
        ax.text(
            0.925,
            y,
            f"{value:.1f}%",
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            color=color,
            ha="right",
            va="center",
        )
    ax.plot(
        [0.665, 0.925],
        [0.207, 0.207],
        transform=ax.transAxes,
        color=COLORS["gold"],
        alpha=0.45,
    )
    ax.text(
        0.665,
        0.187,
        f"Sensitivity-targeted: {pct(targeted['intent_sensitivity']):.1f}% sens / {pct(targeted['intent_specificity']):.1f}% spec",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["coral"],
    )

    ax.text(
        0.5,
        0.12,
        "V7 is the strongest conservative development candidate, but its advantage over V5 is only 0.3 percentage points",
        transform=ax.transAxes,
        fontsize=9.7,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
    )
    footer(
        ax,
        "Safety boundary",
        "Specificity is per cue-locked rest window; continuous false activations/min and command latency are not evaluated.",
        COLORS["navy"],
    )
    save_figure(fig, output_dir, "07_v7_results_and_domain_variation", dpi)


def write_readme(output_dir: Path) -> None:
    readme = """# V7 Poster Section Visuals

Standalone visuals for Sections 3 through 7 of
`docs/final/EEG_BCI_V7_POSTER_CONTENT.md`.

- `03_datasets_and_cohorts`: cohort composition, retained windows, and negative-state semantics.
- `04_signal_processing_and_features`: native-montage QC and parallel physiological branches.
- `05_v7_decoder_and_patient_calibration`: population fusion, 30-window calibration, auxiliary direction, and binary output.
- `06_evaluation_protocol_and_leakage_control`: subject roles, recording boundary, and primary metrics.
- `07_v7_results_and_domain_variation`: headline metrics, domain results, adaptation effect, and version comparison.

Each visual is exported as a 300-DPI PNG and vector SVG/PDF. Regenerate with:

```bash
python scripts/42_create_v7_poster_section_visuals.py
```

All results are development observed and cue locked.
"""
    (output_dir / "SECTION_VISUALS.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    report = load_metrics(args.metrics.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    create_section_3_datasets(report, output_dir, args.dpi)
    create_section_4_signal_processing(report, output_dir, args.dpi)
    create_section_5_decoder_calibration(report, output_dir, args.dpi)
    create_section_6_evaluation(report, output_dir, args.dpi)
    create_section_7_results(report, output_dir, args.dpi)
    write_readme(output_dir)
    print(f"Saved section visuals to {output_dir}")


if __name__ == "__main__":
    main()
