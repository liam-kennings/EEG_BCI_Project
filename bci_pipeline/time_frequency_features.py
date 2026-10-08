"""Compact Morlet time-frequency-spatial features for motor-intent decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import fftconvolve


MORLET_FREQUENCIES = np.asarray(
    (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0, 28.0, 30.0),
    dtype=np.float64,
)
MORLET_BANDS = (
    ("mu", 8.0, 13.0),
    ("beta_low", 13.0, 21.0),
    ("beta_high", 21.0, 31.0),
)
MORLET_TIME_BINS = 4


@dataclass
class MorletEEGFeatures:
    values: np.ndarray
    names: list[str]
    fingerprint: str

    def validate(self, rows: int) -> None:
        if self.values.shape != (rows, len(self.names)):
            raise ValueError("Morlet feature matrix and names are not aligned.")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("Morlet EEG features contain non-finite values.")


def morlet_cycles(frequencies: np.ndarray = MORLET_FREQUENCIES) -> np.ndarray:
    """Increase cycle count with frequency while retaining sub-second support."""
    return np.interp(
        frequencies,
        (float(frequencies[0]), float(frequencies[-1])),
        (3.5, 8.0),
    )


def morlet_kernel(
    frequency: float,
    cycles: float,
    sfreq: float,
) -> np.ndarray:
    """Return a unit-energy complex Morlet wavelet."""
    sigma = cycles / (2.0 * np.pi * frequency)
    half_width = max(1.0 / sfreq, 3.5 * sigma)
    times = np.arange(
        -half_width,
        half_width + 0.5 / sfreq,
        1.0 / sfreq,
    )
    wavelet = np.exp(2j * np.pi * frequency * times) * np.exp(
        -(times**2) / (2.0 * sigma**2)
    )
    wavelet -= np.mean(wavelet)
    return wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))


def _reflect_convolution(
    signals: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    half = len(kernel) // 2
    padded = np.pad(signals, ((0, 0), (0, 0), (half, half)), mode="reflect")
    convolution = fftconvolve(
        padded,
        kernel[None, None, :],
        mode="same",
        axes=-1,
    )
    return convolution[..., half : half + signals.shape[-1]]


def morlet_frequency_power(
    epochs: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Return ``epochs x channels x frequencies x time`` Morlet power."""
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 3:
        raise ValueError("Morlet input must have shape epochs x channels x time.")
    power = np.zeros(
        (
            len(epochs),
            epochs.shape[1],
            len(MORLET_FREQUENCIES),
            epochs.shape[-1],
        ),
        dtype=np.float32,
    )
    for frequency_index, (frequency, cycles) in enumerate(
        zip(MORLET_FREQUENCIES, morlet_cycles())
    ):
        convolution = _reflect_convolution(
            epochs,
            morlet_kernel(float(frequency), float(cycles), sfreq),
        )
        power[:, :, frequency_index] = np.abs(convolution) ** 2
    return power


def morlet_band_power(
    epochs: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Return ``epochs x channels x bands x time`` variable-cycle power."""
    frequency_power = morlet_frequency_power(epochs, sfreq)
    blocks = []
    for _, low, high in MORLET_BANDS:
        selected = (MORLET_FREQUENCIES >= low) & (MORLET_FREQUENCIES < high)
        if not np.any(selected):
            raise RuntimeError("At least one Morlet band has no frequency samples.")
        blocks.append(np.mean(frequency_power[:, :, selected], axis=2))
    return np.stack(blocks, axis=2).astype(np.float32)


def _time_bin_log_power(power: np.ndarray) -> np.ndarray:
    edges = np.linspace(0, power.shape[-1], MORLET_TIME_BINS + 1, dtype=int)
    return np.stack(
        [
            np.log(
                np.clip(
                    np.mean(power[..., edges[index] : edges[index + 1]], axis=-1),
                    1e-30,
                    None,
                )
            )
            for index in range(MORLET_TIME_BINS)
        ],
        axis=-1,
    )


def _extract_batch(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    log_bins = _time_bin_log_power(morlet_band_power(epochs, sfreq))
    time_axis = np.arange(MORLET_TIME_BINS, dtype=np.float64)
    time_axis -= np.mean(time_axis)
    slope = np.sum(log_bins * time_axis[None, None, None, :], axis=-1)
    slope /= np.sum(time_axis**2)
    late_early = np.mean(log_bins[..., 2:], axis=-1) - np.mean(
        log_bins[..., :2],
        axis=-1,
    )
    variability = np.std(log_bins, axis=-1)

    blocks = [
        log_bins.reshape(len(epochs), -1),
        np.stack((slope, late_early, variability), axis=-1).reshape(
            len(epochs),
            -1,
        ),
    ]
    names = [
        f"{channel}_{band}_morlet_log_power_bin_{bin_index + 1}"
        for channel in channel_names
        for band, _, _ in MORLET_BANDS
        for bin_index in range(MORLET_TIME_BINS)
    ]
    names.extend(
        f"{channel}_{band}_morlet_{descriptor}"
        for channel in channel_names
        for band, _, _ in MORLET_BANDS
        for descriptor in ("trajectory_slope", "late_minus_early", "variability")
    )

    lookup = {name.upper(): index for index, name in enumerate(channel_names)}
    spatial_values = []
    spatial_names = []
    for left, right in (("FC3", "FC4"), ("C3", "C4"), ("CP3", "CP4")):
        for band_index, (band, _, _) in enumerate(MORLET_BANDS):
            difference = (
                log_bins[:, lookup[left], band_index]
                - log_bins[:, lookup[right], band_index]
            )
            for bin_index in range(MORLET_TIME_BINS):
                spatial_values.append(difference[:, bin_index])
                spatial_names.append(
                    f"{left}_minus_{right}_{band}_morlet_bin_{bin_index + 1}"
                )
            spatial_values.append(
                np.mean(difference[:, 2:], axis=-1)
                - np.mean(difference[:, :2], axis=-1)
            )
            spatial_names.append(
                f"{left}_minus_{right}_{band}_morlet_late_minus_early"
            )

    central = [
        lookup[channel.upper()] for channel in ("C3", "Cz", "C4")
    ]
    central_mean = np.mean(log_bins[:, central], axis=1)
    for band_index, (band, _, _) in enumerate(MORLET_BANDS):
        for bin_index in range(MORLET_TIME_BINS):
            spatial_values.append(central_mean[:, band_index, bin_index])
            spatial_names.append(
                f"central_mean_{band}_morlet_bin_{bin_index + 1}"
            )
        spatial_values.append(
            np.mean(central_mean[:, band_index, 2:], axis=-1)
            - np.mean(central_mean[:, band_index, :2], axis=-1)
        )
        spatial_names.append(
            f"central_mean_{band}_morlet_late_minus_early"
        )
    blocks.append(np.column_stack(spatial_values))
    names.extend(spatial_names)
    return np.concatenate(blocks, axis=1), names


def _fingerprint(
    dataset_fingerprint: str,
    channel_names: Sequence[str],
    sfreq: float,
) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "dataset_fingerprint": dataset_fingerprint,
            "channels": list(channel_names),
            "sampling_hz": float(sfreq),
            "frequencies_hz": MORLET_FREQUENCIES.tolist(),
            "cycles": morlet_cycles().tolist(),
            "bands": list(MORLET_BANDS),
            "time_bins": MORLET_TIME_BINS,
            "edge_handling": "reflect_padding",
            "power_transform": "natural_log",
        },
        sort_keys=True,
    )


def build_morlet_feature_cache(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
    dataset_fingerprint: str,
    path: Path,
    *,
    rebuild_cache: bool = False,
    batch_size: int = 256,
) -> MorletEEGFeatures:
    fingerprint = _fingerprint(dataset_fingerprint, channel_names, sfreq)
    if path.is_file() and not rebuild_cache:
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) == fingerprint:
                result = MorletEEGFeatures(
                    values=cached["values"],
                    names=[str(name) for name in cached["names"]],
                    fingerprint=fingerprint,
                )
                result.validate(len(epochs))
                print(f"[+] Loaded Morlet features from {path}")
                return result

    rows = []
    names: list[str] | None = None
    print("[+] Extracting variable-cycle Morlet spatial-temporal features...")
    for start in range(0, len(epochs), batch_size):
        stop = min(start + batch_size, len(epochs))
        values, batch_names = _extract_batch(
            epochs[start:stop],
            sfreq,
            channel_names,
        )
        if names is None:
            names = batch_names
        elif names != batch_names:
            raise RuntimeError("Morlet feature names changed between batches.")
        rows.append(values.astype(np.float32))
        if stop % (5 * batch_size) == 0 or stop == len(epochs):
            print(f"    extracted {stop:,}/{len(epochs):,} windows")
    if names is None:
        raise ValueError("No EEG epochs were supplied.")
    result = MorletEEGFeatures(
        values=np.concatenate(rows, axis=0),
        names=names,
        fingerprint=fingerprint,
    )
    result.validate(len(epochs))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=result.values,
        names=np.asarray(result.names),
        fingerprint=np.asarray(result.fingerprint),
    )
    print(f"[+] Saved Morlet features: {path}")
    return result
