#!/usr/bin/env python3
"""Create a poster-ready visual explaining the clinical motivation for V7."""

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


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "poster_v7" / "why_project"

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
    linewidth: float = 1.3,
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
    color: str,
    width: float = 2.0,
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
            zorder=5,
        )
    )


def waveform(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    color: str,
    amplitude: float,
    phase: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 220)
    noise = rng.normal(0.0, 0.18, t.size)
    signal = (
        0.56 * np.sin(2.0 * np.pi * 8.0 * t + phase)
        + 0.34 * np.sin(2.0 * np.pi * 13.0 * t + 0.5 * phase)
        + noise
    )
    envelope = 1.0 - amplitude * np.exp(-((t - 0.66) / 0.18) ** 2)
    signal *= envelope
    ax.plot(
        x + width * t,
        y + 0.5 * height + 0.34 * height * signal,
        transform=ax.transAxes,
        color=color,
        linewidth=1.6,
        clip_on=False,
        zorder=6,
    )


def save_figure(fig: plt.Figure, output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "00_why_this_project"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi if suffix == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12,
        )
    plt.close(fig)


def create_landscape_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(13.0, 8.5), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    card(
        ax,
        0.038,
        0.898,
        0.075,
        0.054,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        radius=0.018,
    )
    ax.text(
        0.0755,
        0.925,
        "WHY",
        transform=ax.transAxes,
        color="white",
        fontsize=12.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0.13,
        0.94,
        "Why This Project?",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.132,
        0.89,
        "A rehabilitation BCI must detect a faint cortical signal without activating during rest",
        transform=ax.transAxes,
        fontsize=12,
        color=COLORS["muted"],
        va="top",
    )
    card(
        ax,
        0.78,
        0.912,
        0.18,
        0.035,
        facecolor=COLORS["soft_coral"],
        edgecolor=COLORS["coral"],
        radius=0.014,
    )
    ax.text(
        0.87,
        0.9295,
        "CLINICAL MOTIVATION",
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["coral"],
        ha="center",
        va="center",
    )

    card(
        ax,
        0.12,
        0.765,
        0.76,
        0.075,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.019,
    )
    ax.text(
        0.5,
        0.802,
        "Can a general EEG decoder be rapidly adapted to each patient while preserving safe rest rejection?",
        transform=ax.transAxes,
        fontsize=12.2,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    reasons = [
        (
            0.045,
            "01",
            "WHY EEG?",
            "Usable muscle activity may be\nabsent, inconsistent, or delayed.",
            "EEG can preserve cortical evidence of\nattempted or imagined movement before\na reliable peripheral response is available.",
            COLORS["blue"],
            COLORS["soft_blue"],
        ),
        (
            0.355,
            "02",
            "WHY ADAPT?",
            "Motor intent is a weak and\npatient-specific EEG 'whisper.'",
            "Signal strength, timing, anatomy, and\nrecording conditions vary across people,\nsessions, devices, and stroke status.",
            COLORS["teal"],
            COLORS["soft_teal"],
        ),
        (
            0.665,
            "03",
            "WHY A BINARY GATE?",
            "A false command during rest can\nreduce safety, usability, and trust.",
            "The known rehabilitation cue supplies\nthe requested movement; EEG only confirms\nwhether credible motor intent followed.",
            COLORS["coral"],
            COLORS["soft_coral"],
        ),
    ]
    for x, number, title, problem, response, color, soft in reasons:
        card(
            ax,
            x,
            0.405,
            0.285,
            0.31,
            facecolor=soft,
            edgecolor=color,
            linewidth=1.5,
            radius=0.024,
        )
        card(
            ax,
            x + 0.018,
            0.658,
            0.058,
            0.032,
            facecolor=color,
            edgecolor=color,
            radius=0.012,
        )
        ax.text(
            x + 0.047,
            0.674,
            number,
            transform=ax.transAxes,
            fontsize=8.8,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.092,
            0.674,
            title,
            transform=ax.transAxes,
            fontsize=11.5 if number != "03" else 10.8,
            fontweight="bold",
            color=color,
            va="center",
        )

        if number == "01":
            waveform(
                ax,
                x + 0.04,
                0.572,
                0.205,
                0.045,
                color=color,
                amplitude=0.0,
                phase=0.2,
                seed=4,
            )
            ax.text(
                x + 0.255,
                0.594,
                "EEG",
                transform=ax.transAxes,
                fontsize=8.2,
                fontweight="bold",
                color=color,
                ha="right",
            )
        elif number == "02":
            waveform(
                ax,
                x + 0.04,
                0.585,
                0.205,
                0.035,
                color=COLORS["muted"],
                amplitude=0.0,
                phase=0.0,
                seed=9,
            )
            waveform(
                ax,
                x + 0.04,
                0.55,
                0.205,
                0.035,
                color=color,
                amplitude=0.42,
                phase=0.25,
                seed=9,
            )
            ax.text(
                x + 0.255,
                0.605,
                "REST",
                transform=ax.transAxes,
                fontsize=7.7,
                fontweight="bold",
                color=COLORS["muted"],
                ha="right",
            )
            ax.text(
                x + 0.255,
                0.56,
                "INTENT",
                transform=ax.transAxes,
                fontsize=7.7,
                fontweight="bold",
                color=color,
                ha="right",
            )
        else:
            card(
                ax,
                x + 0.055,
                0.558,
                0.175,
                0.055,
                facecolor=COLORS["paper"],
                edgecolor=color,
                radius=0.016,
            )
            ax.text(
                x + 0.1425,
                0.585,
                "REST = NO COMMAND",
                transform=ax.transAxes,
                fontsize=10.5,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
            )

        ax.text(
            x + 0.025,
            0.525,
            problem,
            transform=ax.transAxes,
            fontsize=10.4,
            fontweight="bold",
            color=COLORS["ink"],
            va="top",
            linespacing=1.35,
        )
        ax.text(
            x + 0.025,
            0.455,
            response,
            transform=ax.transAxes,
            fontsize=8.8,
            color=COLORS["muted"],
            va="top",
            linespacing=1.35,
        )

    for x in (0.1875, 0.4975, 0.8075):
        arrow(ax, (x, 0.395), (x, 0.355), color=COLORS["muted"])

    card(
        ax,
        0.045,
        0.175,
        0.91,
        0.165,
        facecolor=COLORS["paper"],
        edgecolor=COLORS["line"],
        radius=0.022,
    )
    ax.text(
        0.065,
        0.305,
        "PROJECT DESIGN RESPONSE",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["muted"],
    )
    response_steps = [
        (0.075, 0.17, "KNOWN CUE", "Which movement?", COLORS["soft_blue"], COLORS["blue"]),
        (0.295, 0.17, "RAPID CALIBRATION", "30 labeled windows", COLORS["soft_coral"], COLORS["coral"]),
        (0.515, 0.17, "EEG EVIDENCE", "MRCP + SMR + covariance", COLORS["soft_gold"], COLORS["gold"]),
        (0.735, 0.17, "BINARY GATE", "REST  or  INTENT", COLORS["soft_teal"], COLORS["teal"]),
    ]
    for index, (x, width, title, note, soft, color) in enumerate(response_steps):
        card(
            ax,
            x,
            0.215,
            width,
            0.065,
            facecolor=soft,
            edgecolor=color,
            radius=0.015,
        )
        ax.text(
            x + width / 2,
            0.258,
            title,
            transform=ax.transAxes,
            fontsize=9.2,
            fontweight="bold",
            color=color,
            ha="center",
        )
        ax.text(
            x + width / 2,
            0.231,
            note,
            transform=ax.transAxes,
            fontsize=8.0,
            color=COLORS["muted"],
            ha="center",
        )
        if index < len(response_steps) - 1:
            arrow(
                ax,
                (x + width + 0.006, 0.248),
                (response_steps[index + 1][0] - 0.008, 0.248),
                color=COLORS["muted"],
                width=1.6,
                mutation_scale=12,
            )

    card(
        ax,
        0.04,
        0.035,
        0.92,
        0.075,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.018,
    )
    ax.text(
        0.06,
        0.073,
        "PROJECT GOAL",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["teal"],
        va="center",
    )
    ax.text(
        0.2,
        0.073,
        "A trustworthy EEG proposal signal that confirms whether motor intent followed a rehabilitation cue.",
        transform=ax.transAxes,
        fontsize=10.1,
        fontweight="bold",
        color="white",
        va="center",
    )

    save_figure(fig, output_dir, dpi)


def create_visual(output_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(10.5, 10.5), facecolor=COLORS["background"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()

    card(
        ax,
        0.045,
        0.91,
        0.14,
        0.052,
        facecolor=COLORS["navy"],
        edgecolor=COLORS["navy"],
        radius=0.026,
    )
    ax.text(
        0.115,
        0.936,
        "WHY",
        transform=ax.transAxes,
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        0.215,
        0.957,
        "Why This Project?",
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.218,
        0.908,
        "Detect the intention to move when movement itself is weak, delayed, or absent",
        transform=ax.transAxes,
        fontsize=11.5,
        color=COLORS["muted"],
        va="top",
    )

    card(
        ax,
        0.12,
        0.82,
        0.76,
        0.055,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.025,
    )
    ax.text(
        0.5,
        0.848,
        "Can EEG provide a safe, rapidly personalized signal for cue-guided assistance?",
        transform=ax.transAxes,
        fontsize=11.8,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    satellites = [
        (
            0.045,
            0.595,
            "ACCESS",
            "EEG may preserve cortical evidence\nwhen usable EMG or movement is limited.",
            COLORS["blue"],
            COLORS["soft_blue"],
        ),
        (
            0.695,
            0.595,
            "SAFETY",
            "Rest must not become an unintended\ndevice command.",
            COLORS["coral"],
            COLORS["soft_coral"],
        ),
        (
            0.045,
            0.375,
            "VARIABILITY",
            "Intent patterns differ across people,\nsessions, devices, and stroke status.",
            COLORS["teal"],
            COLORS["soft_teal"],
        ),
        (
            0.695,
            0.375,
            "PERSONALIZATION",
            "A short calibration must adapt the\npopulation model to the individual.",
            COLORS["gold"],
            COLORS["soft_gold"],
        ),
    ]
    for x, y, title, note, color, soft in satellites:
        card(
            ax,
            x,
            y,
            0.26,
            0.14,
            facecolor=soft,
            edgecolor=color,
            linewidth=1.6,
            radius=0.055,
        )
        ax.add_patch(
            Circle(
                (x + 0.05, y + 0.098),
                0.028,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                zorder=4,
            )
        )
        symbol = {
            "ACCESS": "EEG",
            "SAFETY": "REST",
            "VARIABILITY": "1:n",
            "PERSONALIZATION": "1:1",
        }[title]
        ax.text(
            x + 0.05,
            y + 0.098,
            symbol,
            transform=ax.transAxes,
            fontsize=7.2 if title != "SAFETY" else 6.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            zorder=5,
        )
        ax.text(
            x + 0.09,
            y + 0.105,
            title,
            transform=ax.transAxes,
            fontsize=10.4 if title != "PERSONALIZATION" else 9.4,
            fontweight="bold",
            color=color,
            va="center",
        )
        ax.text(
            x + 0.03,
            y + 0.045,
            note,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
            linespacing=1.35,
        )

    arrow(ax, (0.305, 0.655), (0.36, 0.61), color=COLORS["blue"], width=1.8)
    arrow(ax, (0.695, 0.655), (0.64, 0.61), color=COLORS["coral"], width=1.8)
    arrow(ax, (0.305, 0.435), (0.36, 0.49), color=COLORS["teal"], width=1.8)
    arrow(ax, (0.695, 0.435), (0.64, 0.49), color=COLORS["gold"], width=1.8)

    ax.add_patch(
        Circle(
            (0.5, 0.555),
            0.165,
            transform=ax.transAxes,
            facecolor=COLORS["soft_blue"],
            edgecolor=COLORS["navy"],
            linewidth=2.4,
            zorder=3,
        )
    )
    ax.add_patch(
        Circle(
            (0.5, 0.555),
            0.135,
            transform=ax.transAxes,
            facecolor=COLORS["paper"],
            edgecolor=COLORS["line"],
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.text(
        0.5,
        0.635,
        "MOTOR INTENT",
        transform=ax.transAxes,
        fontsize=14.5,
        fontweight="bold",
        color=COLORS["navy"],
        ha="center",
        va="center",
        zorder=6,
    )
    t = np.linspace(0.0, 1.0, 180)
    signal = 0.58 * np.sin(2 * np.pi * 7 * t) + 0.25 * np.sin(2 * np.pi * 13 * t)
    envelope = 1.0 - 0.42 * np.exp(-((t - 0.67) / 0.18) ** 2)
    ax.plot(
        0.405 + 0.19 * t,
        0.565 + 0.025 * signal * envelope,
        transform=ax.transAxes,
        color=COLORS["teal"],
        linewidth=2.2,
        zorder=7,
    )
    ax.text(
        0.5,
        0.485,
        "may exist before visible movement\nor reliable muscle output",
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["muted"],
        ha="center",
        va="center",
        linespacing=1.3,
        zorder=6,
    )

    arrow(ax, (0.5, 0.365), (0.5, 0.325), color=COLORS["navy"], width=2.2)
    card(
        ax,
        0.19,
        0.265,
        0.62,
        0.055,
        facecolor=COLORS["ink"],
        edgecolor=COLORS["ink"],
        radius=0.026,
    )
    ax.text(
        0.5,
        0.293,
        "KNOWN REHABILITATION CUE  +  PATIENT EEG  ->  REST or INTENT",
        transform=ax.transAxes,
        fontsize=10.3,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
    )

    ax.text(
        0.5,
        0.225,
        "POTENTIAL APPLICATIONS",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["muted"],
        ha="center",
    )
    applications = [
        ("REHABILITATION", "feedback and training", COLORS["blue"], COLORS["soft_blue"]),
        ("ROBOTIC ASSISTANCE", "orthoses and exoskeletons", COLORS["teal"], COLORS["soft_teal"]),
        ("PROSTHETIC CONTROL", "intent confirmation", COLORS["gold"], COLORS["soft_gold"]),
        ("STIMULATION", "future FES triggering", COLORS["coral"], COLORS["soft_coral"]),
    ]
    for index, (title, note, color, soft) in enumerate(applications):
        x = 0.045 + index * 0.237
        card(
            ax,
            x,
            0.095,
            0.205,
            0.095,
            facecolor=soft,
            edgecolor=color,
            linewidth=1.4,
            radius=0.045,
        )
        ax.text(
            x + 0.1025,
            0.153,
            title,
            transform=ax.transAxes,
            fontsize=8.8 if index != 1 else 8.0,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
        ax.text(
            x + 0.1025,
            0.12,
            note,
            transform=ax.transAxes,
            fontsize=7.7,
            color=COLORS["muted"],
            ha="center",
            va="center",
        )

    ax.text(
        0.5,
        0.048,
        "PROJECT GOAL: confirm that credible motor intent followed the cue before assistance is enabled",
        transform=ax.transAxes,
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
    )

    save_figure(fig, output_dir, dpi)


def main() -> None:
    args = parse_args()
    configure_style()
    create_visual(args.output_dir, args.dpi)
    print(f"Saved project-motivation visual to {args.output_dir}")


if __name__ == "__main__":
    main()
