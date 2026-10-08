"""Shared sensorimotor montage and filter-bank covariance primitives."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.covariance import oas


MOTOR_CHANNELS = (
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4",
)

TEMPORAL_CHANNELS = ("FC3", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP4")

FILTER_BANK: Tuple[Tuple[str, float, float], ...] = (
    ("mu_low", 7.0, 10.0),
    ("mu_high", 10.0, 13.0),
    ("beta_low", 13.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("mu_beta", 8.0, 30.0),
)


def regularized_covariance(signal: np.ndarray) -> np.ndarray:
    covariance, _ = oas(signal.T, assume_centered=False)
    covariance = 0.5 * (covariance + covariance.T)
    covariance /= max(float(np.trace(covariance)), 1e-12)
    return covariance.astype(np.float32)


def filter_bank_covariances(
    epoch: np.ndarray,
    sfreq: float,
    filter_bank: Tuple[Tuple[str, float, float], ...] = FILTER_BANK,
) -> np.ndarray:
    views = []
    for _, low_hz, high_hz in filter_bank:
        sos = butter(4, (low_hz, high_hz), btype="bandpass", fs=sfreq, output="sos")
        views.append(regularized_covariance(sosfiltfilt(sos, epoch, axis=-1)))
    return np.stack(views)

