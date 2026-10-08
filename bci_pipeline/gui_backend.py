"""Reusable file inspection and experiment-launch services for the BCI GUI."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from .dataset import normalize_channel_name
from .feature_atlas import recording_quality
from .openbmi import (
    EXPECTED_RECORDINGS_PER_SUBJECT,
    available_openbmi_subjects,
    discover_openbmi_recordings,
    metadata_subjects,
    resolve_openbmi_root,
)
from .preprocessing import OFFLINE_PROFILE, PreprocessingProfile
from .prospective_controller import prepare_prospective_package
from .quality_preprocessing import (
    BranchPreprocessingProfile,
    preprocess_native_branches,
)
from .research_contract import (
    ProspectiveAcquisitionTemplate,
    write_research_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDARD_RECORDING_EXTENSIONS = (
    ".edf",
    ".bdf",
    ".gdf",
    ".fif",
    ".fif.gz",
    ".vhdr",
    ".set",
    ".cnt",
)
ARRAY_RECORDING_EXTENSIONS = (".csv", ".tsv", ".npy", ".npz")
SUPPORTED_RECORDING_EXTENSIONS = (
    *STANDARD_RECORDING_EXTENSIONS,
    *ARRAY_RECORDING_EXTENSIONS,
)
CENTRAL_CHANNEL_PRIORITY = (
    "C3",
    "CZ",
    "C4",
    "FC3",
    "FCZ",
    "FC4",
    "CP3",
    "CPZ",
    "CP4",
)


@dataclass(frozen=True)
class RecordingSummary:
    path: str
    file_format: str
    channels: int
    eeg_channels: int
    sampling_hz: float
    duration_seconds: float
    samples: int
    annotations: int
    annotation_labels: tuple[str, ...]
    channel_names: tuple[str, ...]

    def to_text(self) -> str:
        labels = ", ".join(self.annotation_labels) if self.annotation_labels else "none"
        return "\n".join(
            (
                f"File: {self.path}",
                f"Format: {self.file_format}",
                f"Channels: {self.channels} total / {self.eeg_channels} EEG",
                f"Sampling: {self.sampling_hz:.2f} Hz",
                f"Duration: {self.duration_seconds:.2f} s ({self.samples:,} samples)",
                f"Annotations: {self.annotations} ({labels})",
                f"Channel names: {', '.join(self.channel_names)}",
            )
        )


@dataclass(frozen=True)
class RecordingAudit:
    report_path: Path
    figure_path: Path
    summary: RecordingSummary
    quality: dict[str, object]
    band_metrics: dict[str, float]
    branch_metrics: dict[str, float | str]
    plotted_channels: tuple[str, ...]
    preprocessing_profile: dict[str, object]


@dataclass(frozen=True)
class DatasetFolderSummary:
    path: str
    subjects: int
    edf_files: int
    complete_subjects: int
    looks_like_eegmmidb: bool

    def to_text(self) -> str:
        status = "valid EEGMMIDB layout" if self.looks_like_eegmmidb else "layout incomplete"
        return (
            f"{self.path}\n"
            f"{self.subjects} subject folders | {self.edf_files} EDF files | "
            f"{self.complete_subjects} subjects with all 14 runs | {status}"
        )


@dataclass(frozen=True)
class OpenBMIFolderSummary:
    path: str
    metadata_subjects: int
    complete_subjects: int
    bdf_files: int
    expected_bdf_files: int
    adapter_ready: bool

    def to_text(self) -> str:
        status = "OpenBMI adapter ready" if self.adapter_ready else "OpenBMI signal data incomplete"
        return (
            f"{self.path}\n"
            f"{self.complete_subjects}/{self.metadata_subjects} subjects with all four BDFs | "
            f"{self.bdf_files}/{self.expected_bdf_files} BDF recordings | {status}"
        )


@dataclass(frozen=True)
class PipelineStep:
    key: str
    name: str
    command: tuple[str, ...]
    expected_output: Path


def _path_extension(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".fif.gz"):
        return ".fif.gz"
    return path.suffix.lower()


def _unit_scale(unit: str) -> float:
    normalized = unit.strip().lower().replace("micro", "u").replace("µ", "u")
    scales = {"v": 1.0, "mv": 1e-3, "uv": 1e-6}
    if normalized not in scales:
        raise ValueError(f"Unsupported voltage unit: {unit}")
    return scales[normalized]


def _coerce_channel_names(
    channel_names: Sequence[str] | None,
    channel_count: int,
) -> list[str]:
    if channel_names is None:
        return [f"EEG{index + 1:03d}" for index in range(channel_count)]
    names = [str(name).strip() for name in channel_names]
    if len(names) != channel_count:
        raise ValueError(
            f"Expected {channel_count} channel names, received {len(names)}."
        )
    if len(set(names)) != len(names):
        raise ValueError("Channel names must be unique.")
    return names


def _orient_array(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("Uploaded NumPy data must be a one- or two-dimensional array.")
    if data.shape[0] > data.shape[1] and data.shape[1] <= 512:
        data = data.T
    if data.shape[1] < 2:
        raise ValueError("Recording must contain at least two time samples.")
    return data


def _raw_from_array(
    data: np.ndarray,
    sfreq: float | None,
    channel_names: Sequence[str] | None,
    unit: str,
) -> mne.io.RawArray:
    if sfreq is None or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("A positive sampling frequency is required for array files.")
    oriented = _orient_array(data) * _unit_scale(unit)
    names = _coerce_channel_names(channel_names, oriented.shape[0])
    info = mne.create_info(names, float(sfreq), ch_types="eeg")
    return mne.io.RawArray(oriented, info, verbose=False)


def _load_delimited(
    path: Path,
    sfreq: float | None,
    channel_names: Sequence[str] | None,
    unit: str,
) -> mne.io.RawArray:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.reader(handle, delimiter=delimiter)
            if row and any(cell.strip() for cell in row)
        ]
    if not rows:
        raise ValueError("The uploaded table is empty.")

    def numeric(row: Sequence[str]) -> bool:
        try:
            [float(value) for value in row]
        except ValueError:
            return False
        return True

    header = None if numeric(rows[0]) else [cell.strip() for cell in rows.pop(0)]
    try:
        matrix = np.asarray([[float(value) for value in row] for row in rows])
    except ValueError as error:
        raise ValueError("CSV/TSV data must contain only numeric samples.") from error
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("CSV/TSV recording must contain at least two sample rows.")

    inferred_sfreq = sfreq
    inferred_names = list(channel_names) if channel_names is not None else header
    first_name = header[0].lower() if header else ""
    if first_name in {"time", "time_s", "seconds", "timestamp"}:
        time_values = matrix[:, 0]
        differences = np.diff(time_values)
        if np.all(np.isfinite(differences)) and np.all(differences > 0):
            inferred_sfreq = inferred_sfreq or float(1.0 / np.median(differences))
        matrix = matrix[:, 1:]
        if channel_names is None and inferred_names is not None:
            inferred_names = inferred_names[1:]
    return _raw_from_array(
        matrix.T,
        inferred_sfreq,
        inferred_names,
        unit,
    )


def load_recording(
    path: str | Path,
    *,
    sfreq: float | None = None,
    channel_names: Sequence[str] | None = None,
    unit: str = "uV",
) -> mne.io.BaseRaw:
    """Load a standard EEG file or a simple channel-by-time array."""
    recording_path = Path(path).expanduser().resolve()
    if not recording_path.is_file():
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    extension = _path_extension(recording_path)
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".gdf": mne.io.read_raw_gdf,
        ".fif": mne.io.read_raw_fif,
        ".fif.gz": mne.io.read_raw_fif,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".cnt": mne.io.read_raw_cnt,
    }
    if extension in readers:
        return readers[extension](recording_path, preload=False, verbose=False)
    if extension in {".csv", ".tsv"}:
        return _load_delimited(recording_path, sfreq, channel_names, unit)
    if extension == ".npy":
        return _raw_from_array(
            np.load(recording_path, allow_pickle=False),
            sfreq,
            channel_names,
            unit,
        )
    if extension == ".npz":
        with np.load(recording_path, allow_pickle=False) as archive:
            data_key = next(
                (
                    key
                    for key in ("data", "raw", "eeg", "X")
                    if key in archive.files
                ),
                None,
            )
            if data_key is None:
                raise ValueError("NPZ recording needs a data, raw, eeg, or X array.")
            archive_sfreq = sfreq
            for key in ("sfreq", "sampling_hz", "sampling_rate"):
                if archive_sfreq is None and key in archive.files:
                    archive_sfreq = float(np.asarray(archive[key]).squeeze())
            archive_names = channel_names
            for key in ("ch_names", "channel_names", "channels"):
                if archive_names is None and key in archive.files:
                    archive_names = [str(value) for value in np.asarray(archive[key])]
                    break
            data = np.asarray(archive[data_key])
        return _raw_from_array(data, archive_sfreq, archive_names, unit)
    raise ValueError(
        f"Unsupported recording format '{extension}'. Supported: "
        f"{', '.join(SUPPORTED_RECORDING_EXTENSIONS)}"
    )


def summarize_recording(
    raw: mne.io.BaseRaw,
    path: str | Path = "in-memory recording",
) -> RecordingSummary:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    labels = tuple(sorted({str(label) for label in raw.annotations.description}))
    return RecordingSummary(
        path=str(path),
        file_format=_path_extension(Path(path)) or "memory",
        channels=len(raw.ch_names),
        eeg_channels=len(eeg_picks),
        sampling_hz=float(raw.info["sfreq"]),
        duration_seconds=float(raw.n_times / raw.info["sfreq"]),
        samples=int(raw.n_times),
        annotations=len(raw.annotations),
        annotation_labels=labels,
        channel_names=tuple(raw.ch_names),
    )


def _selected_plot_channels(channel_names: Sequence[str], limit: int = 6) -> list[str]:
    lookup = {normalize_channel_name(name): name for name in channel_names}
    selected = [
        lookup[name]
        for name in CENTRAL_CHANNEL_PRIORITY
        if name in lookup
    ]
    if len(selected) < min(limit, len(channel_names)):
        for name in channel_names:
            if name not in selected:
                selected.append(name)
            if len(selected) == limit:
                break
    return selected[:limit]


def _mean_band_power(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    low: float,
    high: float,
) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if not np.any(mask):
        return 0.0
    return float(
        np.mean(np.trapezoid(spectrum[:, mask], frequencies[mask], axis=1))
    )


def _plot_stacked_trace(
    axis: plt.Axes,
    data_uv: np.ndarray,
    times: np.ndarray,
    names: Sequence[str],
    title: str,
) -> None:
    centered = data_uv - np.median(data_uv, axis=1, keepdims=True)
    scale = float(np.median(np.std(centered, axis=1)))
    spacing = max(5.0 * scale, 20.0)
    offsets = np.arange(len(names) - 1, -1, -1) * spacing
    for signal, offset, name in zip(centered, offsets, names):
        axis.plot(times, signal + offset, linewidth=0.8, color="#12355b")
        axis.text(
            times[0],
            offset,
            f" {name}",
            va="center",
            ha="left",
            fontsize=8,
            color="#102a43",
        )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Amplitude + channel offset (uV)")
    axis.set_yticks([])
    axis.grid(axis="x", alpha=0.2)


def audit_recording(
    raw: mne.io.BaseRaw,
    output_dir: str | Path,
    *,
    source_path: str | Path = "in-memory recording",
    profile: PreprocessingProfile = OFFLINE_PROFILE,
    plot_seconds: float = 12.0,
) -> RecordingAudit:
    """Run gross QC, preprocessing, spectral summaries, and a visual audit."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw_eeg = raw.copy().pick("eeg").load_data()
    if not raw_eeg.ch_names:
        raise ValueError("The recording does not contain channels marked as EEG.")
    summary = summarize_recording(raw, source_path)
    quality = recording_quality(
        raw_eeg.get_data(),
        float(raw_eeg.info["sfreq"]),
        raw_eeg.ch_names,
    )
    plotted_channels = _selected_plot_channels(raw_eeg.ch_names)
    branch_profile = BranchPreprocessingProfile(
        sfreq=min(100.0, float(raw_eeg.info["sfreq"])),
        quality_band_hz=(
            1.0,
            min(45.0, float(raw_eeg.info["sfreq"]) * 0.45),
        ),
        selected_motor_channels=tuple(plotted_channels),
        compute_csd=len(raw_eeg.ch_names) >= 16,
    )
    branches = preprocess_native_branches(raw, branch_profile)
    processed = branches.smr_car
    raw_plot = raw_eeg.copy().pick(plotted_channels)
    processed_plot = processed.copy()
    mrcp_plot = branches.mrcp.copy()

    duration = min(
        float(plot_seconds),
        float(raw_plot.n_times / raw_plot.info["sfreq"]),
        float(processed_plot.n_times / processed_plot.info["sfreq"]),
    )
    raw_samples = max(2, int(round(duration * raw_plot.info["sfreq"])))
    processed_samples = max(2, int(round(duration * processed_plot.info["sfreq"])))
    mrcp_samples = max(2, int(round(duration * mrcp_plot.info["sfreq"])))
    raw_data = raw_plot.get_data(stop=raw_samples) * 1e6
    processed_data = processed_plot.get_data(stop=processed_samples) * 1e6
    mrcp_data = mrcp_plot.get_data(stop=mrcp_samples) * 1e6
    raw_times = np.arange(raw_data.shape[1]) / float(raw_plot.info["sfreq"])
    processed_times = (
        np.arange(processed_data.shape[1]) / float(processed_plot.info["sfreq"])
    )
    mrcp_times = np.arange(mrcp_data.shape[1]) / float(mrcp_plot.info["sfreq"])

    frequencies, spectrum = welch(
        processed_plot.get_data(),
        fs=float(processed_plot.info["sfreq"]),
        nperseg=min(
            int(round(2.0 * processed_plot.info["sfreq"])),
            processed_plot.n_times,
        ),
        axis=-1,
    )
    total = _mean_band_power(spectrum, frequencies, 4.0, 35.0)
    mu = _mean_band_power(spectrum, frequencies, 8.0, 13.0)
    beta = _mean_band_power(spectrum, frequencies, 13.0, 30.0)
    band_metrics = {
        "mean_mu_power_uv2": mu * 1e12,
        "mean_beta_power_uv2": beta * 1e12,
        "mu_fraction_4_35": mu / max(total, 1e-30),
        "beta_fraction_4_35": beta / max(total, 1e-30),
    }
    branch_metrics: dict[str, float | str] = {
        "bad_channel_fraction": float(
            branches.recording_qc["bad_channel_fraction"]
        ),
        "bad_channel_count": float(branches.recording_qc["bad_channel_count"]),
        "csd_status": str(branches.recording_qc["csd_status"]),
        "auxiliary_channels": float(
            0 if branches.auxiliary is None else len(branches.auxiliary.ch_names)
        ),
        "mrcp_low_hz": branch_profile.mrcp_band_hz[0],
        "mrcp_high_hz": branch_profile.mrcp_band_hz[1],
        "smr_low_hz": branch_profile.smr_band_hz[0],
        "smr_high_hz": branch_profile.smr_band_hz[1],
    }

    figure, axes = plt.subplots(3, 2, figsize=(13.5, 11.0), constrained_layout=True)
    figure.patch.set_facecolor("#f7f4ed")
    _plot_stacked_trace(
        axes[0, 0],
        raw_data,
        raw_times,
        plotted_channels,
        "Raw EEG",
    )
    _plot_stacked_trace(
        axes[1, 0],
        processed_data,
        processed_times,
        plotted_channels,
        "SMR Branch (7-35 Hz, CAR after channel repair)",
    )
    _plot_stacked_trace(
        axes[2, 0],
        mrcp_data,
        mrcp_times,
        plotted_channels,
        "MRCP Branch (0.1-4 Hz)",
    )
    psd_db = 10.0 * np.log10(np.clip(np.mean(spectrum, axis=0) * 1e12, 1e-20, None))
    shown = (frequencies >= 1.0) & (frequencies <= min(45.0, frequencies[-1]))
    axes[0, 1].plot(frequencies[shown], psd_db[shown], color="#12355b", linewidth=1.8)
    axes[0, 1].axvspan(8.0, 13.0, color="#e9a23b", alpha=0.24, label="Mu 8-13 Hz")
    axes[0, 1].axvspan(13.0, 30.0, color="#2f8f83", alpha=0.20, label="Beta 13-30 Hz")
    axes[0, 1].set(
        title="Processed Central-channel Spectrum",
        xlabel="Frequency (Hz)",
        ylabel="Power spectral density (dB uV^2/Hz)",
    )
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(alpha=0.2)

    labels = ("Mu\n8-13 Hz", "Beta\n13-30 Hz")
    percentages = (
        100.0 * band_metrics["mu_fraction_4_35"],
        100.0 * band_metrics["beta_fraction_4_35"],
    )
    bars = axes[1, 1].bar(
        labels,
        percentages,
        color=("#e9a23b", "#2f8f83"),
        width=0.58,
    )
    axes[1, 1].bar_label(bars, fmt="%.1f%%", padding=3, fontweight="bold")
    axes[1, 1].set(
        title="Share of 4-35 Hz Power",
        ylabel="Relative power (%)",
    )
    axes[1, 1].set_ylim(0.0, max(50.0, max(percentages) * 1.25))
    axes[1, 1].grid(axis="y", alpha=0.2)
    qc_labels = ("Available", "Interpolated", "Auxiliary")
    qc_values = (
        100.0 * (1.0 - float(branches.recording_qc["bad_channel_fraction"])),
        100.0 * float(branches.recording_qc["bad_channel_fraction"]),
        float(branch_metrics["auxiliary_channels"]),
    )
    qc_bars = axes[2, 1].bar(
        qc_labels,
        qc_values,
        color=("#2f8f83", "#e9a23b", "#12355b"),
        width=0.58,
    )
    axes[2, 1].bar_label(qc_bars, fmt="%.1f", padding=3, fontweight="bold")
    axes[2, 1].set(
        title="Recording Quality and Auxiliary Evidence",
        ylabel="EEG channels (%) / auxiliary count",
    )
    axes[2, 1].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Motor-Intent EEG Recording Audit",
        fontsize=17,
        fontweight="bold",
        color="#102a43",
    )

    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(source_path).stem) or "recording"
    figure_path = destination / f"{safe_stem}_signal_audit.png"
    report_path = destination / f"{safe_stem}_signal_audit.json"
    figure.savefig(figure_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)

    payload = {
        "source": str(source_path),
        "summary": asdict(summary),
        "quality": quality,
        "band_metrics": band_metrics,
        "branch_metrics": branch_metrics,
        "plotted_channels": plotted_channels,
        "preprocessing_profile": branch_profile.fingerprint(),
        "interpretation": (
            "This audit verifies signal loading, gross channel quality, preprocessing, "
            "and motor-band content. It is not a motor-intent prediction."
        ),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return RecordingAudit(
        report_path=report_path,
        figure_path=figure_path,
        summary=summary,
        quality=quality,
        band_metrics=band_metrics,
        branch_metrics=branch_metrics,
        plotted_channels=tuple(plotted_channels),
        preprocessing_profile=branch_profile.fingerprint(),
    )


def inspect_eegmmidb_folder(path: str | Path) -> DatasetFolderSummary:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {root}")
    subject_dirs = sorted(
        candidate
        for candidate in root.iterdir()
        if candidate.is_dir() and re.fullmatch(r"S\d{3}", candidate.name)
    )
    run_counts = [
        len(list(subject.rglob(f"{subject.name}R*.edf")))
        for subject in subject_dirs
    ]
    return DatasetFolderSummary(
        path=str(root),
        subjects=len(subject_dirs),
        edf_files=int(sum(run_counts)),
        complete_subjects=int(sum(count >= 14 for count in run_counts)),
        looks_like_eegmmidb=bool(subject_dirs) and all(count >= 3 for count in run_counts),
    )


def inspect_openbmi_folder(path: str | Path) -> OpenBMIFolderSummary:
    root = resolve_openbmi_root(path)
    subjects = metadata_subjects(root)
    complete = available_openbmi_subjects(root, complete_only=True)
    recordings = discover_openbmi_recordings(root)
    expected = len(subjects) * EXPECTED_RECORDINGS_PER_SUBJECT
    return OpenBMIFolderSummary(
        path=str(root),
        metadata_subjects=len(subjects),
        complete_subjects=len(complete),
        bdf_files=len(recordings),
        expected_bdf_files=expected,
        adapter_ready=bool(complete),
    )


def build_openbmi_adapter_step(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> PipelineStep:
    data = resolve_openbmi_root(data_dir)
    output = Path(output_root).expanduser().resolve() / "openbmi_adapter"
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "24_prepare_openbmi_dataset.py"),
        "--data-dir",
        str(data),
        "--output-dir",
        str(output),
        "--build-epoch-cache",
    ]
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="openbmi_adapter",
        name="OpenBMI BIDS harmonization and raw-window cache",
        command=tuple(command),
        expected_output=output / "OPENBMI_IMPORT_REPORT.md",
    )


def build_pipeline_steps(
    data_dir: str | Path,
    output_root: str | Path,
    *,
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> tuple[PipelineStep, ...]:
    """Build the current feature-audit and controlled-ablation commands."""
    data = Path(data_dir).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    python = str(python_executable or sys.executable)
    atlas_dir = output / "feature_atlas"
    atlas_path = atlas_dir / "motor_imagery_feature_atlas.npz"
    extended_dir = output / "extended_features"
    extended_path = extended_dir / "extended_motor_imagery_features.npz"
    ablation_dir = output / "controlled_ablation"
    split_path = PROJECT_ROOT / "artifacts" / "dataset" / "splits.json"
    covariance_cache = (
        PROJECT_ROOT
        / "artifacts"
        / "multibranch_benchmark"
        / "aligned_covariance_features.npz"
    )

    atlas_command = [
        python,
        str(PROJECT_ROOT / "scripts" / "18_build_motor_imagery_feature_atlas.py"),
        "--data-dir",
        str(data),
        "--cache-path",
        str(atlas_path),
        "--output-dir",
        str(atlas_dir),
        "--split-path",
        str(split_path),
    ]
    extended_command = [
        python,
        str(PROJECT_ROOT / "scripts" / "20_build_extended_motor_imagery_features.py"),
        "--data-dir",
        str(data),
        "--atlas-cache",
        str(atlas_path),
        "--split-file",
        str(split_path),
        "--output-dir",
        str(extended_dir),
    ]
    ablation_command = [
        python,
        str(PROJECT_ROOT / "scripts" / "21_controlled_extended_feature_ablation.py"),
        "--atlas-path",
        str(atlas_path),
        "--extended-path",
        str(extended_path),
        "--split-path",
        str(split_path),
        "--covariance-cache",
        str(covariance_cache),
        "--output-dir",
        str(ablation_dir),
    ]
    if rebuild:
        atlas_command.append("--rebuild-cache")
        extended_command.append("--rebuild-cache")
        ablation_command.append("--rebuild-covariance-cache")
    return (
        PipelineStep(
            key="atlas",
            name="Base physiology feature atlas",
            command=tuple(atlas_command),
            expected_output=atlas_dir / "FEATURE_ATLAS_RESULTS.md",
        ),
        PipelineStep(
            key="extended",
            name="Extended feature audit",
            command=tuple(extended_command),
            expected_output=extended_dir / "EXTENDED_FEATURE_AUDIT.md",
        ),
        PipelineStep(
            key="ablation",
            name="Controlled feature ablation",
            command=tuple(ablation_command),
            expected_output=ablation_dir / "CONTROLLED_ABLATION_RESULTS.md",
        ),
    )


def build_translational_v3_step(
    physionet_dir: str | Path,
    openbmi_dir: str | Path,
    liu_dir: str | Path,
    output_root: str | Path,
    *,
    hard_negative_manifest: str | Path | None = None,
    target_intent_sensitivity: float = 0.60,
    temporal_confirmations: int = 2,
    temporal_refractory_seconds: float = 2.0,
    bootstrap_iterations: int = 1000,
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the calibrated native-montage translational v3 command."""
    root = Path(output_root).expanduser().resolve()
    output = root / "translational_v3_calibrated"
    legacy_cache = root / "translational_v3" / "translational_windows_v3.npz"
    cache_path = (
        legacy_cache
        if legacy_cache.is_file() and not rebuild
        else output / "translational_windows_v3.npz"
    )
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "30_translational_motor_intent_v3.py"),
        "--physionet-dir",
        str(Path(physionet_dir).expanduser().resolve()),
        "--openbmi-dir",
        str(Path(openbmi_dir).expanduser().resolve()),
        "--liu-dir",
        str(Path(liu_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--cache-path",
        str(cache_path),
        "--target-intent-sensitivity",
        str(float(target_intent_sensitivity)),
        "--temporal-confirmations",
        str(int(temporal_confirmations)),
        "--temporal-refractory-seconds",
        str(float(temporal_refractory_seconds)),
        "--bootstrap-iterations",
        str(int(bootstrap_iterations)),
    ]
    if hard_negative_manifest:
        command.extend(
            (
                "--hard-negative-manifest",
                str(Path(hard_negative_manifest).expanduser().resolve()),
            )
        )
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="translational_v3_calibrated",
        name="Translational v3 calibrated safety benchmark",
        command=tuple(command),
        expected_output=output / "TRANSLATIONAL_V3_REPORT.md",
    )


def build_translational_v4_neural_step(
    physionet_dir: str | Path,
    openbmi_dir: str | Path,
    liu_dir: str | Path,
    output_root: str | Path,
    *,
    hard_negative_manifest: str | Path | None = None,
    target_intent_sensitivity: float = 0.60,
    temporal_confirmations: int = 2,
    temporal_refractory_seconds: float = 2.0,
    bootstrap_iterations: int = 1000,
    rebuild: bool = False,
    neural_device: str = "auto",
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the neural-integrated translational v4 command."""
    root = Path(output_root).expanduser().resolve()
    output = root / "translational_v4_neural_integrated"
    legacy_cache = root / "translational_v3" / "translational_windows_v3.npz"
    cache_path = (
        legacy_cache
        if legacy_cache.is_file() and not rebuild
        else output / "translational_windows_v3.npz"
    )
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "30_translational_motor_intent_v3.py"),
        "--physionet-dir",
        str(Path(physionet_dir).expanduser().resolve()),
        "--openbmi-dir",
        str(Path(openbmi_dir).expanduser().resolve()),
        "--liu-dir",
        str(Path(liu_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--cache-path",
        str(cache_path),
        "--neural-bakeoff-dir",
        str(PROJECT_ROOT / "artifacts" / "eeg_decoder_model_bakeoff"),
        "--neural-seeds",
        "42",
        "43",
        "44",
        "45",
        "46",
        "--neural-device",
        str(neural_device),
        "--target-intent-sensitivity",
        str(float(target_intent_sensitivity)),
        "--temporal-confirmations",
        str(int(temporal_confirmations)),
        "--temporal-refractory-seconds",
        str(float(temporal_refractory_seconds)),
        "--bootstrap-iterations",
        str(int(bootstrap_iterations)),
    ]
    if hard_negative_manifest:
        command.extend(
            (
                "--hard-negative-manifest",
                str(Path(hard_negative_manifest).expanduser().resolve()),
            )
        )
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="translational_v4_neural_integrated",
        name="Translational v4 MRCP + FBCNet integration",
        command=tuple(command),
        expected_output=output / "TRANSLATIONAL_V4_NEURAL_REPORT.md",
    )


def build_translational_v5_patient_adaptive_step(
    physionet_dir: str | Path,
    openbmi_dir: str | Path,
    liu_dir: str | Path,
    output_root: str | Path,
    *,
    hard_negative_manifest: str | Path | None = None,
    target_intent_sensitivity: float = 0.60,
    temporal_confirmations: int = 2,
    temporal_refractory_seconds: float = 2.0,
    bootstrap_iterations: int = 1000,
    rebuild: bool = False,
    neural_device: str = "auto",
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build v5 dual-neural fusion with rapid per-patient calibration."""
    root = Path(output_root).expanduser().resolve()
    output = root / "translational_v5_patient_adaptive"
    legacy_cache = root / "translational_v3" / "translational_windows_v3.npz"
    cache_path = (
        legacy_cache
        if legacy_cache.is_file() and not rebuild
        else output / "translational_windows_v3.npz"
    )
    bakeoff = root / "eeg_decoder_model_bakeoff"
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "30_translational_motor_intent_v3.py"),
        "--physionet-dir",
        str(Path(physionet_dir).expanduser().resolve()),
        "--openbmi-dir",
        str(Path(openbmi_dir).expanduser().resolve()),
        "--liu-dir",
        str(Path(liu_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--cache-path",
        str(cache_path),
        "--neural-bakeoff-dir",
        str(bakeoff),
        "--neural-seeds",
        "42",
        "43",
        "44",
        "45",
        "46",
        "--neural-architectures",
        "mrcp_fbcnet",
        "shallow_convnet",
        "--neural-device",
        str(neural_device),
        "--target-intent-sensitivity",
        str(float(target_intent_sensitivity)),
        "--temporal-confirmations",
        str(int(temporal_confirmations)),
        "--temporal-refractory-seconds",
        str(float(temporal_refractory_seconds)),
        "--bootstrap-iterations",
        str(int(bootstrap_iterations)),
    ]
    if hard_negative_manifest:
        command.extend(
            (
                "--hard-negative-manifest",
                str(Path(hard_negative_manifest).expanduser().resolve()),
            )
        )
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="translational_v5_patient_adaptive",
        name="Translational v5 dual-neural patient adaptation",
        command=tuple(command),
        expected_output=output / "TRANSLATIONAL_V5_PATIENT_ADAPTIVE_REPORT.md",
    )


def build_translational_v6_intent_only_step(
    physionet_dir: str | Path,
    openbmi_dir: str | Path,
    liu_dir: str | Path,
    output_root: str | Path,
    *,
    hard_negative_manifest: str | Path | None = None,
    target_intent_sensitivity: float = 0.70,
    temporal_confirmations: int = 3,
    temporal_refractory_seconds: float = 2.0,
    bootstrap_iterations: int = 1000,
    rebuild: bool = False,
    neural_device: str = "auto",
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build v6 binary rest-versus-intent patient adaptation."""
    root = Path(output_root).expanduser().resolve()
    output = root / "translational_v6_intent_only"
    reusable_caches = (
        root / "translational_v5_patient_adaptive" / "translational_windows_v3.npz",
        root / "translational_v3" / "translational_windows_v3.npz",
    )
    cache_path = output / "translational_windows_v3.npz"
    if not rebuild:
        cache_path = next(
            (candidate for candidate in reusable_caches if candidate.is_file()),
            cache_path,
        )
    bakeoff = root / "eeg_decoder_model_bakeoff"
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "30_translational_motor_intent_v3.py"),
        "--physionet-dir",
        str(Path(physionet_dir).expanduser().resolve()),
        "--openbmi-dir",
        str(Path(openbmi_dir).expanduser().resolve()),
        "--liu-dir",
        str(Path(liu_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--cache-path",
        str(cache_path),
        "--neural-bakeoff-dir",
        str(bakeoff),
        "--neural-seeds",
        "42",
        "43",
        "44",
        "45",
        "46",
        "--neural-architectures",
        "mrcp_fbcnet",
        "shallow_convnet",
        "--neural-device",
        str(neural_device),
        "--target-intent-sensitivity",
        str(float(target_intent_sensitivity)),
        "--temporal-confirmations",
        str(int(temporal_confirmations)),
        "--temporal-refractory-seconds",
        str(float(temporal_refractory_seconds)),
        "--bootstrap-iterations",
        str(int(bootstrap_iterations)),
        "--intent-only",
    ]
    if hard_negative_manifest:
        command.extend(
            (
                "--hard-negative-manifest",
                str(Path(hard_negative_manifest).expanduser().resolve()),
            )
        )
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="translational_v6_intent_only",
        name="Translational v6 binary patient adaptation",
        command=tuple(command),
        expected_output=output / "TRANSLATIONAL_V6_INTENT_ONLY_REPORT.md",
    )


def build_translational_v7_auxiliary_direction_step(
    physionet_dir: str | Path,
    openbmi_dir: str | Path,
    liu_dir: str | Path,
    output_root: str | Path,
    *,
    hard_negative_manifest: str | Path | None = None,
    target_intent_sensitivity: float = 0.70,
    temporal_confirmations: int = 3,
    temporal_refractory_seconds: float = 2.0,
    bootstrap_iterations: int = 1000,
    rebuild: bool = False,
    neural_device: str = "auto",
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build graph/LDA-enhanced v7 with binary device-facing output."""
    root = Path(output_root).expanduser().resolve()
    output = root / "translational_v7_graph_lda"
    reusable_caches = (
        root
        / "translational_v7_auxiliary_direction"
        / "translational_windows_v3.npz",
        root / "translational_v6_intent_only" / "translational_windows_v3.npz",
        root / "translational_v5_patient_adaptive" / "translational_windows_v3.npz",
        root / "translational_v3" / "translational_windows_v3.npz",
    )
    cache_path = output / "translational_windows_v3.npz"
    if not rebuild:
        cache_path = next(
            (candidate for candidate in reusable_caches if candidate.is_file()),
            cache_path,
        )
    bakeoff = root / "eeg_decoder_model_bakeoff"
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "30_translational_motor_intent_v3.py"),
        "--physionet-dir",
        str(Path(physionet_dir).expanduser().resolve()),
        "--openbmi-dir",
        str(Path(openbmi_dir).expanduser().resolve()),
        "--liu-dir",
        str(Path(liu_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--cache-path",
        str(cache_path),
        "--neural-bakeoff-dir",
        str(bakeoff),
        "--neural-seeds",
        "42",
        "43",
        "44",
        "45",
        "46",
        "--neural-architectures",
        "mrcp_fbcnet",
        "shallow_convnet",
        "--neural-device",
        str(neural_device),
        "--target-intent-sensitivity",
        str(float(target_intent_sensitivity)),
        "--temporal-confirmations",
        str(int(temporal_confirmations)),
        "--temporal-refractory-seconds",
        str(float(temporal_refractory_seconds)),
        "--bootstrap-iterations",
        str(int(bootstrap_iterations)),
        "--auxiliary-direction",
    ]
    if hard_negative_manifest:
        command.extend(
            (
                "--hard-negative-manifest",
                str(Path(hard_negative_manifest).expanduser().resolve()),
            )
        )
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="translational_v7_auxiliary_direction",
        name="Translational v7 graph/LDA-enhanced binary output",
        command=tuple(command),
        expected_output=output / "TRANSLATIONAL_V7_AUXILIARY_DIRECTION_REPORT.md",
    )


def build_v7h_mind_step(
    mind_dir: str | Path,
    output_root: str | Path,
    *,
    target_specificity: float = 0.93,
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the participant-isolated MIND EEG-fNIRS V7-H benchmark."""
    root = Path(output_root).expanduser().resolve()
    output = root / "v7h_mind"
    python = str(python_executable or sys.executable)
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "51_v7h_mind_benchmark.py"),
        "--mind-dir",
        str(Path(mind_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--target-specificity",
        str(float(target_specificity)),
    ]
    if rebuild:
        command.append("--rebuild-cache")
    return PipelineStep(
        key="v7h_mind",
        name="V7-H MIND hybrid EEG-fNIRS benchmark",
        command=tuple(command),
        expected_output=output / "V7H_MIND_REPORT.md",
    )


def build_emg_supervised_transfer_steps(
    mrcp_eeg_emg_dir: str | Path,
    output_root: str | Path,
    *,
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> tuple[PipelineStep, ...]:
    """Build cached adapter and EEG-only transfer commands for the GUI."""
    python = str(python_executable or sys.executable)
    output = Path(output_root).expanduser().resolve() / "emg_supervised_transfer"
    source_cache = output / "mrcp_eeg_emg_source_features.npz"
    target_cache = output / "pure_eeg_target_features.npz"
    steps: list[PipelineStep] = []
    if rebuild or not source_cache.is_file() or not target_cache.is_file():
        steps.append(
            PipelineStep(
                key="prepare_mrcp_eeg_emg_transfer",
                name="Prepare synchronized MRCP EEG-EMG transfer caches",
                command=(
                    python,
                    str(PROJECT_ROOT / "scripts" / "33_prepare_mrcp_eeg_emg_transfer.py"),
                    "--data-dir",
                    str(Path(mrcp_eeg_emg_dir).expanduser().resolve()),
                    "--output-dir",
                    str(output),
                ),
                expected_output=output / "MRCP_EEG_EMG_ADAPTER_REPORT.md",
            )
        )
    steps.append(
        PipelineStep(
            key="emg_supervised_eeg_transfer",
            name="Compare EMG-supervised and target-only EEG intent models",
            command=(
                python,
                str(PROJECT_ROOT / "scripts" / "32_emg_supervised_eeg_transfer.py"),
                "--source-cache",
                str(source_cache),
                "--target-cache",
                str(target_cache),
                "--output-dir",
                str(output),
            ),
            expected_output=output / "EMG_SUPERVISED_EEG_TRANSFER_REPORT.md",
        )
    )
    return tuple(steps)


def build_way_eeg_gal_transfer_steps(
    way_eeg_gal_dir: str | Path,
    output_root: str | Path,
    *,
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> tuple[PipelineStep, ...]:
    """Build WAY privileged-source preparation and EEG-only transfer commands."""
    python = str(python_executable or sys.executable)
    output = Path(output_root).expanduser().resolve() / "way_eeg_gal_transfer"
    source_cache = output / "way_eeg_gal_source_features.npz"
    target_cache = (
        PROJECT_ROOT
        / "artifacts"
        / "emg_supervised_transfer"
        / "pure_eeg_target_features.npz"
    )
    steps: list[PipelineStep] = []
    if rebuild or not source_cache.is_file():
        steps.append(
            PipelineStep(
                key="prepare_way_eeg_gal_transfer",
                name="Prepare WAY-EEG-GAL privileged source features",
                command=(
                    python,
                    str(PROJECT_ROOT / "scripts" / "35_prepare_way_eeg_gal_transfer.py"),
                    "--data-dir",
                    str(Path(way_eeg_gal_dir).expanduser().resolve()),
                    "--target-cache",
                    str(target_cache),
                    "--output-dir",
                    str(output),
                ),
                expected_output=output / "WAY_EEG_GAL_ADAPTER_REPORT.md",
            )
        )
    steps.append(
        PipelineStep(
            key="way_eeg_gal_eeg_transfer",
            name="Compare WAY-assisted and target-only EEG intent models",
            command=(
                python,
                str(PROJECT_ROOT / "scripts" / "32_emg_supervised_eeg_transfer.py"),
                "--source-cache",
                str(source_cache),
                "--target-cache",
                str(target_cache),
                "--output-dir",
                str(output),
            ),
            expected_output=output / "EMG_SUPERVISED_EEG_TRANSFER_REPORT.md",
        )
    )
    return tuple(steps)


def build_multisource_privileged_ensemble_step(
    output_root: str | Path,
    *,
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the cached MRCP + WAY + target-only EEG ensemble command."""
    python = str(python_executable or sys.executable)
    output = Path(output_root).expanduser().resolve() / "multisource_privileged_ensemble"
    return PipelineStep(
        key="multisource_privileged_ensemble",
        name="Train validation-selected multi-source EEG ensemble",
        command=(
            python,
            str(PROJECT_ROOT / "scripts" / "36_multisource_privileged_ensemble.py"),
            "--mrcp-source-cache",
            str(
                PROJECT_ROOT
                / "artifacts"
                / "emg_supervised_transfer"
                / "mrcp_eeg_emg_source_features.npz"
            ),
            "--way-source-cache",
            str(
                PROJECT_ROOT
                / "artifacts"
                / "way_eeg_gal_transfer"
                / "way_eeg_gal_source_features.npz"
            ),
            "--target-cache",
            str(
                PROJECT_ROOT
                / "artifacts"
                / "emg_supervised_transfer"
                / "pure_eeg_target_features.npz"
            ),
            "--output-dir",
            str(output),
        ),
        expected_output=output / "MULTISOURCE_PRIVILEGED_ENSEMBLE_REPORT.md",
    )


def build_way_eeg_ssl_step(
    way_eeg_gal_dir: str | Path,
    output_root: str | Path,
    *,
    rebuild: bool = False,
    device: str = "auto",
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the EEG-only WAY self-supervised transfer command."""
    python = str(python_executable or sys.executable)
    output = Path(output_root).expanduser().resolve() / "way_eeg_only_ssl"
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "37_way_eeg_only_ssl_transfer.py"),
        "--way-dir",
        str(Path(way_eeg_gal_dir).expanduser().resolve()),
        "--way-cache",
        str(
            PROJECT_ROOT
            / "artifacts"
            / "way_eeg_only_ssl"
            / "way_unlabeled_eeg_windows.npz"
        ),
        "--target-raw-cache",
        str(
            PROJECT_ROOT.parent
            / "artifacts"
            / "gui_experiments"
            / "translational_v3"
            / "translational_windows_v3.npz"
        ),
        "--target-feature-cache",
        str(
            PROJECT_ROOT
            / "artifacts"
            / "emg_supervised_transfer"
            / "pure_eeg_target_features.npz"
        ),
        "--comparison-model",
        str(
            PROJECT_ROOT
            / "artifacts"
            / "multisource_privileged_ensemble"
            / "multisource_eeg_intent_ensemble.joblib"
        ),
        "--output-dir",
        str(output),
        "--device",
        str(device),
    ]
    if rebuild:
        command.append("--rebuild-way-cache")
    return PipelineStep(
        key="way_eeg_only_ssl",
        name="Pretrain on unlabeled WAY EEG and fine-tune intent decoder",
        command=tuple(command),
        expected_output=output / "WAY_EEG_ONLY_SSL_TRANSFER_REPORT.md",
    )


def build_way_eeg_ssl_seed_stability_step(
    way_eeg_gal_dir: str | Path,
    output_root: str | Path,
    *,
    device: str = "auto",
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the controlled five-seed EEG-only WAY stability command."""
    python = str(python_executable or sys.executable)
    output = Path(output_root).expanduser().resolve() / "way_eeg_only_ssl_seed_stability"
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "38_way_eeg_ssl_seed_stability.py"),
        "--way-dir",
        str(Path(way_eeg_gal_dir).expanduser().resolve()),
        "--output-dir",
        str(output),
        "--seeds",
        "42",
        "43",
        "44",
        "45",
        "46",
        "--device",
        str(device),
    ]
    if rebuild:
        command.append("--force")
    return PipelineStep(
        key="way_eeg_only_ssl_seed_stability",
        name="Confirm WAY EEG-only SSL across five neural seeds",
        command=tuple(command),
        expected_output=output / "WAY_EEG_SSL_SEED_STABILITY_REPORT.md",
    )


def build_decoder_bakeoff_step(
    output_root: str | Path,
    *,
    device: str = "auto",
    rebuild: bool = False,
    python_executable: str | Path | None = None,
) -> PipelineStep:
    """Build the controlled five-seed EEG decoder architecture bakeoff."""
    python = str(python_executable or sys.executable)
    output = Path(output_root).expanduser().resolve() / "eeg_decoder_model_bakeoff"
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "39_eeg_decoder_model_bakeoff.py"),
        "--output-dir",
        str(output),
        "--seeds",
        "42",
        "43",
        "44",
        "45",
        "46",
        "--device",
        str(device),
    ]
    if rebuild:
        command.append("--force")
    return PipelineStep(
        key="eeg_decoder_model_bakeoff",
        name="Compare EEGNet, ShallowConvNet, FBCNet, and MRCP + FBCNet",
        command=tuple(command),
        expected_output=output / "EEG_DECODER_MODEL_BAKEOFF_REPORT.md",
    )


def append_hard_negative_manifest(
    manifest_path: str | Path,
    recording_path: str | Path,
    *,
    subject_key: str,
    session: int,
    negative_kind: str = "continuous_no_control",
    start_seconds: float = 0.0,
    stop_seconds: float | None = None,
    step_seconds: float = 1.0,
) -> Path:
    """Register an explicit continuous no-control recording for later evaluation."""
    manifest = Path(manifest_path).expanduser().resolve()
    source = Path(recording_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "Hard-negative registration requires a saved recording file."
        )
    payload = (
        json.loads(manifest.read_text(encoding="utf-8"))
        if manifest.is_file()
        else {"schema_version": 1, "recordings": []}
    )
    entry: dict[str, object] = {
        "path": str(source),
        "dataset_id": "hard_negative",
        "subject_key": subject_key.strip(),
        "session": int(session),
        "recording": 1,
        "negative_kind": negative_kind,
        "start_seconds": float(start_seconds),
        "step_seconds": float(step_seconds),
    }
    if stop_seconds is not None:
        entry["stop_seconds"] = float(stop_seconds)
    existing = {
        (
            str(value.get("path")),
            str(value.get("subject_key")),
            int(value.get("session", 1)),
        )
        for value in payload["recordings"]
    }
    identity = (str(source), subject_key.strip(), int(session))
    if identity not in existing:
        payload["recordings"].append(entry)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    return manifest


def write_prospective_preparation(
    output_dir: str | Path,
    template: ProspectiveAcquisitionTemplate,
) -> tuple[Path, Path]:
    """Export the disabled acquisition contract and frozen controller package."""
    paths = write_research_contract(output_dir, acquisition=template)
    model_path = (
        PROJECT_ROOT
        / "artifacts"
        / "gui_experiments"
        / "translational_v4_neural_integrated"
        / "translational_v4_neural_models.joblib"
    )
    prepare_prospective_package(
        output_dir,
        model_path,
        acquisition=asdict(template),
    )
    return paths


def optional_adapter_status() -> dict[str, bool]:
    return {
        "lsl": importlib.util.find_spec("pylsl") is not None,
        "brainflow": importlib.util.find_spec("brainflow") is not None,
    }
