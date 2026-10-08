"""Quality-aware native-montage preprocessing for translational EEG studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
import warnings

import mne
import numpy as np

from .dataset import normalize_channel_name
from .feature_atlas import recording_quality


DEFAULT_MOTOR_CHANNELS = ("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4")


@dataclass(frozen=True)
class BranchPreprocessingProfile:
    sfreq: float = 100.0
    mrcp_band_hz: tuple[float, float] = (0.1, 4.0)
    smr_band_hz: tuple[float, float] = (7.0, 35.0)
    quality_band_hz: tuple[float, float] = (1.0, 45.0)
    filter_order: int = 4
    phase: str = "zero"
    reference: str = "average_after_bad_channel_repair"
    montage: str = "standard_1005"
    max_bad_channel_fraction: float = 0.25
    selected_motor_channels: tuple[str, ...] = DEFAULT_MOTOR_CHANNELS
    compute_csd: bool = True
    profile_version: int = 3

    @property
    def is_causal(self) -> bool:
        return self.phase != "zero"

    def fingerprint(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PreprocessedEEGBranches:
    """Parallel EEG branches produced only after full-montage QC and repair."""

    cleaned_full: mne.io.BaseRaw
    mrcp: mne.io.BaseRaw
    smr_car: mne.io.BaseRaw
    smr_csd: mne.io.BaseRaw | None
    quality: mne.io.BaseRaw
    auxiliary: mne.io.BaseRaw | None
    recording_qc: dict[str, object]
    selected_channels: tuple[str, ...]
    profile: BranchPreprocessingProfile


class RecordingQualityError(RuntimeError):
    """Raised when native-montage QC makes a recording unsafe to reference."""

    def __init__(self, message: str, recording_qc: dict[str, object]) -> None:
        super().__init__(message)
        self.recording_qc = recording_qc


def _standardize_montage(
    raw: mne.io.BaseRaw,
    montage_name: str,
) -> tuple[mne.io.BaseRaw, bool]:
    """Rename standard channels and attach positions without inventing locations."""
    adjusted = raw.copy()
    montage = mne.channels.make_standard_montage(montage_name)
    lookup = {
        normalize_channel_name(name): name
        for name in montage.ch_names
    }
    rename = {
        name: lookup[normalize_channel_name(name)]
        for name in adjusted.ch_names
        if normalize_channel_name(name) in lookup
    }
    if rename:
        adjusted.rename_channels(rename)
    recognized = sum(name in montage.ch_names for name in adjusted.ch_names)
    if recognized != len(adjusted.ch_names):
        return adjusted, False
    adjusted.set_montage(montage, on_missing="raise")
    return adjusted, True


def _pick_normalized(
    raw: mne.io.BaseRaw,
    requested: Sequence[str],
) -> mne.io.BaseRaw:
    lookup = {
        normalize_channel_name(name): name
        for name in raw.ch_names
    }
    normalized = [normalize_channel_name(name) for name in requested]
    missing = [name for name in normalized if name not in lookup]
    if missing:
        raise ValueError(f"Required motor channels are missing: {missing}")
    return raw.copy().pick([lookup[name] for name in normalized])


def _iir_filter(
    raw: mne.io.BaseRaw,
    band: tuple[float, float],
    profile: BranchPreprocessingProfile,
) -> mne.io.BaseRaw:
    nyquist = float(raw.info["sfreq"]) / 2.0
    low = max(float(band[0]), 0.001)
    high = min(float(band[1]), nyquist * 0.9)
    if low >= high:
        raise ValueError(
            f"Invalid branch band {band} for {raw.info['sfreq']:.1f} Hz sampling."
        )
    return raw.copy().filter(
        low,
        high,
        method="iir",
        iir_params={"order": profile.filter_order, "ftype": "butter"},
        phase=profile.phase,
        verbose=False,
    )


def _auxiliary_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw | None:
    picks = mne.pick_types(
        raw.info,
        eeg=False,
        eog=True,
        emg=True,
        misc=False,
        stim=False,
        exclude=[],
    )
    if not len(picks):
        return None
    auxiliary = raw.copy().pick(picks).load_data()
    return auxiliary


def preprocess_native_branches(
    raw: mne.io.BaseRaw,
    profile: BranchPreprocessingProfile = BranchPreprocessingProfile(),
) -> PreprocessedEEGBranches:
    """QC and repair the native EEG montage before creating physiology branches."""
    auxiliary = _auxiliary_channels(raw)
    eeg = raw.copy().pick("eeg").load_data()
    if len(eeg.ch_names) < 3:
        raise ValueError("At least three EEG channels are required.")
    eeg, montage_available = _standardize_montage(eeg, profile.montage)
    native_sfreq = float(eeg.info["sfreq"])
    qc = recording_quality(eeg.get_data(), native_sfreq, eeg.ch_names)
    qc.update(
        {
            "qc_before_reference": True,
            "native_channel_count": len(eeg.ch_names),
            "native_sampling_hz": native_sfreq,
            "montage_available": montage_available,
        }
    )
    if float(qc["bad_channel_fraction"]) > profile.max_bad_channel_fraction:
        raise RecordingQualityError(
            "Recording rejected before referencing: "
            f"{qc['bad_channel_count']}/{len(eeg.ch_names)} bad EEG channels.",
            qc,
        )

    cleaned = eeg.copy()
    cleaned.info["bads"] = list(qc["bad_channels"])
    interpolation = "not_needed"
    if cleaned.info["bads"]:
        if montage_available:
            candidate = cleaned.copy()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*divide by zero.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*overflow.*",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*invalid value.*",
                    category=RuntimeWarning,
                )
                candidate.interpolate_bads(reset_bads=True, verbose=False)
            if np.all(np.isfinite(candidate.get_data())):
                cleaned = candidate
                interpolation = "spherical_spline"
            else:
                bad_indices = [
                    cleaned.ch_names.index(name)
                    for name in cleaned.info["bads"]
                ]
                good_indices = [
                    index
                    for index in range(len(cleaned.ch_names))
                    if index not in bad_indices
                ]
                if not good_indices:
                    raise RuntimeError("No usable channels remain for bad-channel repair.")
                replacement = np.median(
                    cleaned.get_data(picks=good_indices),
                    axis=0,
                )
                for index in bad_indices:
                    cleaned._data[index] = replacement
                cleaned.info["bads"] = []
                interpolation = "median_of_good_channels_fallback"
        else:
            bad_indices = [
                cleaned.ch_names.index(name)
                for name in cleaned.info["bads"]
            ]
            good_indices = [
                index
                for index in range(len(cleaned.ch_names))
                if index not in bad_indices
            ]
            if not good_indices:
                raise RuntimeError("No usable channels remain for bad-channel repair.")
            replacement = np.median(
                cleaned.get_data(picks=good_indices),
                axis=0,
            )
            for index in bad_indices:
                cleaned._data[index] = replacement
            cleaned.info["bads"] = []
            interpolation = "median_of_good_channels_no_montage"
    if profile.reference != "average_after_bad_channel_repair":
        raise ValueError(f"Unsupported reference mode: {profile.reference}")
    cleaned.set_eeg_reference("average", projection=False, verbose=False)
    if abs(float(cleaned.info["sfreq"]) - profile.sfreq) > 1e-6:
        cleaned.resample(profile.sfreq, verbose=False)
    if auxiliary is not None and abs(float(auxiliary.info["sfreq"]) - profile.sfreq) > 1e-6:
        auxiliary.resample(profile.sfreq, verbose=False)
    qc["interpolation"] = interpolation
    qc["output_sampling_hz"] = float(cleaned.info["sfreq"])

    mrcp_full = _iir_filter(cleaned, profile.mrcp_band_hz, profile)
    smr_full = _iir_filter(cleaned, profile.smr_band_hz, profile)
    quality_full = _iir_filter(cleaned, profile.quality_band_hz, profile)
    mrcp = _pick_normalized(mrcp_full, profile.selected_motor_channels)
    smr_car = _pick_normalized(smr_full, profile.selected_motor_channels)
    quality = _pick_normalized(quality_full, profile.selected_motor_channels)

    smr_csd = None
    if profile.compute_csd and montage_available and len(cleaned.ch_names) >= 16:
        try:
            csd = mne.preprocessing.compute_current_source_density(
                cleaned.copy(),
                verbose=False,
            )
            smr_csd = _pick_normalized(
                _iir_filter(csd, profile.smr_band_hz, profile),
                profile.selected_motor_channels,
            )
            qc["csd_status"] = "available"
        except (RuntimeError, ValueError):
            qc["csd_status"] = "failed"
    else:
        qc["csd_status"] = "unavailable"

    return PreprocessedEEGBranches(
        cleaned_full=cleaned,
        mrcp=mrcp,
        smr_car=smr_car,
        smr_csd=smr_csd,
        quality=quality,
        auxiliary=auxiliary,
        recording_qc=qc,
        selected_channels=tuple(mrcp.ch_names),
        profile=profile,
    )


def extract_branch_window(
    branches: PreprocessedEEGBranches,
    onset_seconds: float,
    duration_seconds: float,
) -> dict[str, np.ndarray | None]:
    """Extract aligned windows from each branch and optional auxiliary channels."""
    sfreq = float(branches.mrcp.info["sfreq"])
    start = int(round(onset_seconds * sfreq))
    stop = start + int(round(duration_seconds * sfreq))
    if start < 0 or stop > branches.mrcp.n_times:
        raise ValueError("Requested branch window falls outside the recording.")

    def values(raw: mne.io.BaseRaw | None) -> np.ndarray | None:
        return None if raw is None else raw.get_data(start=start, stop=stop).copy()

    return {
        "mrcp": values(branches.mrcp),
        "smr_car": values(branches.smr_car),
        "smr_csd": values(branches.smr_csd),
        "quality": values(branches.quality),
        "auxiliary": values(branches.auxiliary),
    }


def window_quality_features(
    quality_window: np.ndarray,
    auxiliary_window: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return finite quality and auxiliary features used by an abstention gate."""
    signal = np.asarray(quality_window, dtype=np.float64)
    ptp_uv = np.ptp(signal, axis=-1) * 1e6
    rms_uv = np.sqrt(np.mean(signal**2, axis=-1)) * 1e6
    values = [
        float(np.max(ptp_uv)),
        float(np.median(ptp_uv)),
        float(np.max(rms_uv)),
        float(np.median(rms_uv)),
        float(np.max(np.abs(signal)) * 1e6),
        float(np.mean(np.abs(np.diff(signal, axis=-1))) * 1e6),
    ]
    names = [
        "eeg_max_peak_to_peak_uv",
        "eeg_median_peak_to_peak_uv",
        "eeg_max_rms_uv",
        "eeg_median_rms_uv",
        "eeg_max_absolute_uv",
        "eeg_mean_absolute_difference_uv",
    ]
    if auxiliary_window is None or not auxiliary_window.size:
        values.extend((0.0, 0.0, 0.0))
        names.extend(
            (
                "auxiliary_available",
                "auxiliary_max_peak_to_peak_uv",
                "auxiliary_max_rms_uv",
            )
        )
    else:
        auxiliary = np.asarray(auxiliary_window, dtype=np.float64)
        values.extend(
            (
                1.0,
                float(np.max(np.ptp(auxiliary, axis=-1)) * 1e6),
                float(np.max(np.sqrt(np.mean(auxiliary**2, axis=-1))) * 1e6),
            )
        )
        names.extend(
            (
                "auxiliary_available",
                "auxiliary_max_peak_to_peak_uv",
                "auxiliary_max_rms_uv",
            )
        )
    return np.asarray(values, dtype=np.float32), tuple(names)
