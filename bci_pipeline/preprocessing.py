"""Versioned offline and causal EEG preprocessing profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import mne
import numpy as np

from .dataset import normalize_channel_name


@dataclass(frozen=True)
class PreprocessingProfile:
    sfreq: float = 160.0
    l_freq: float = 4.0
    h_freq: float = 35.0
    filter_order: int = 4
    reference: str = "average"
    phase: str = "zero"
    profile_version: int = 1

    @property
    def is_causal(self) -> bool:
        return self.phase != "zero"

    def fingerprint(self) -> dict[str, object]:
        return asdict(self)


OFFLINE_PROFILE = PreprocessingProfile()
CAUSAL_PROFILE = PreprocessingProfile(phase="forward")


def preprocess_run(
    raw: mne.io.BaseRaw,
    profile: PreprocessingProfile,
    selected_channels: Sequence[str] | None = None,
) -> mne.io.BaseRaw:
    """Preprocess full-scalp EEG, reference it, and optionally select channels."""
    eeg = raw.copy().pick("eeg").load_data()
    eeg.filter(
        profile.l_freq,
        profile.h_freq,
        method="iir",
        iir_params={"order": profile.filter_order, "ftype": "butter"},
        phase=profile.phase,
        verbose=False,
    )
    if profile.reference == "average":
        eeg.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
    elif profile.reference != "none":
        raise ValueError(f"Unsupported reference profile: {profile.reference}")
    if abs(float(eeg.info["sfreq"]) - profile.sfreq) > 1e-6:
        eeg.resample(profile.sfreq, verbose=False)

    if selected_channels is None:
        return eeg

    channel_lookup = {normalize_channel_name(name): name for name in eeg.ch_names}
    normalized = [normalize_channel_name(name) for name in selected_channels]
    missing = [name for name in normalized if name not in channel_lookup]
    if missing:
        raise ValueError(f"Required channels are missing: {missing}")
    return eeg.pick([channel_lookup[name] for name in normalized])


def is_valid_epoch(*signals: np.ndarray, peak_to_peak_uv: float) -> bool:
    for signal in signals:
        if signal.ndim != 2 or signal.shape[1] == 0:
            return False
        if np.any(~np.isfinite(signal)):
            return False
        if np.any(np.ptp(signal, axis=1) * 1e6 > peak_to_peak_uv):
            return False
    return True
