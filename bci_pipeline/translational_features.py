"""Physiology-oriented MRCP, SMR, covariance, and quality features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

from .graph_features import fixed_graph_features
from .openbmi_controlled import (
    tangent_features,
    tangent_features_from_reference,
)
from .translational_dataset import TranslationalWindowDataset


SMR_FILTER_BANK = (
    ("mu_low", 7.0, 10.0),
    ("mu_high", 10.0, 14.0),
    ("beta_low", 14.0, 22.0),
    ("beta_high", 22.0, 30.0),
)
MRCP_TIME_EDGES_SECONDS = (0.0, 0.5, 1.0, 1.5, 2.0)
SMR_TIME_EDGES_SECONDS = (0.0, 0.75, 1.5, 2.0)


@dataclass
class TranslationalFeatures:
    branches: dict[str, np.ndarray]
    names: dict[str, tuple[str, ...]]
    covariances: np.ndarray
    individualized_mu_hz: np.ndarray
    metadata: dict[str, object]

    def validate(self, rows: int) -> None:
        for key, values in self.branches.items():
            if values.shape[0] != rows:
                raise ValueError(f"Feature branch {key} is not row-aligned.")
            if values.shape[1] != len(self.names[key]):
                raise ValueError(f"Feature branch {key} names do not align.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Feature branch {key} contains non-finite values.")
        if self.covariances.shape[0] != rows:
            raise ValueError("Covariances are not row-aligned.")
        if not np.all(np.isfinite(self.covariances)):
            raise ValueError("Covariances contain non-finite values.")


def _time_bin_means(
    signal: np.ndarray,
    sfreq: float,
    edges: tuple[float, ...],
) -> np.ndarray:
    values = []
    for start, stop in zip(edges[:-1], edges[1:]):
        first = int(round(start * sfreq))
        last = int(round(stop * sfreq))
        values.append(np.mean(signal[..., first:last], axis=-1))
    return np.stack(values, axis=-1)


def mrcp_features(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize slow trajectory, slope, extrema, and central laterality."""
    epochs_uv = np.asarray(epochs, dtype=np.float64) * 1e6
    binned = _time_bin_means(epochs_uv, sfreq, MRCP_TIME_EDGES_SECONDS)
    times = np.arange(epochs.shape[-1], dtype=np.float64) / sfreq
    centered_time = times - np.mean(times)
    slopes = np.sum(
        epochs_uv * centered_time[None, None, :],
        axis=-1,
    ) / np.sum(centered_time**2)
    area = np.trapezoid(epochs_uv, times, axis=-1)
    blocks = [
        binned.reshape(len(epochs), -1),
        slopes,
        np.min(epochs_uv, axis=-1),
        np.max(epochs_uv, axis=-1),
        area,
    ]
    names = [
        f"{channel}_mrcp_mean_{start:.2f}_{stop:.2f}s_uv"
        for channel in channel_names
        for start, stop in zip(
            MRCP_TIME_EDGES_SECONDS[:-1],
            MRCP_TIME_EDGES_SECONDS[1:],
        )
    ]
    for suffix in ("slope_uv_per_s", "minimum_uv", "maximum_uv", "area_uv_s"):
        names.extend(f"{channel}_mrcp_{suffix}" for channel in channel_names)
    lookup = {name.upper(): index for index, name in enumerate(channel_names)}
    if "C3" in lookup and "C4" in lookup:
        difference = binned[:, lookup["C3"]] - binned[:, lookup["C4"]]
        blocks.append(difference)
        names.extend(
            f"C3_minus_C4_mrcp_{start:.2f}_{stop:.2f}s_uv"
            for start, stop in zip(
                MRCP_TIME_EDGES_SECONDS[:-1],
                MRCP_TIME_EDGES_SECONDS[1:],
            )
        )
    return np.concatenate(blocks, axis=1).astype(np.float32), tuple(names)


def mrcp_phase_features(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize slow-potential phase without exposing raw phase discontinuities."""
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(channel_names):
        raise ValueError("MRCP epochs must have shape trials x channels x samples.")
    if values.shape[-1] < 8 or sfreq <= 0:
        raise ValueError("MRCP phase extraction requires a valid sampled window.")

    analytic = hilbert(values, axis=-1)
    unit_phase = analytic / np.clip(np.abs(analytic), 1e-15, None)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for start, stop in zip(
        MRCP_TIME_EDGES_SECONDS[:-1],
        MRCP_TIME_EDGES_SECONDS[1:],
    ):
        first = int(round(start * sfreq))
        last = min(int(round(stop * sfreq)), values.shape[-1])
        if last <= first:
            raise ValueError("MRCP phase time bins contain no samples.")
        mean_vector = np.mean(unit_phase[..., first:last], axis=-1)
        blocks.extend((mean_vector.real, mean_vector.imag, np.abs(mean_vector)))
        names.extend(
            f"{channel}_mrcp_phase_cos_{start:.2f}_{stop:.2f}s"
            for channel in channel_names
        )
        names.extend(
            f"{channel}_mrcp_phase_sin_{start:.2f}_{stop:.2f}s"
            for channel in channel_names
        )
        names.extend(
            f"{channel}_mrcp_phase_concentration_{start:.2f}_{stop:.2f}s"
            for channel in channel_names
        )

    times = np.arange(values.shape[-1], dtype=np.float64) / sfreq
    centered_time = times - np.mean(times)
    unwrapped = np.unwrap(np.angle(analytic), axis=-1)
    slope = np.sum(
        unwrapped * centered_time[None, None, :],
        axis=-1,
    ) / np.sum(centered_time**2)
    blocks.append(slope)
    names.extend(f"{channel}_mrcp_phase_slope_rad_per_s" for channel in channel_names)

    lookup = {name.upper(): index for index, name in enumerate(channel_names)}
    if "C3" in lookup and "C4" in lookup:
        phase_difference = (
            unit_phase[:, lookup["C3"]]
            * np.conjugate(unit_phase[:, lookup["C4"]])
        )
        for start, stop in zip(
            MRCP_TIME_EDGES_SECONDS[:-1],
            MRCP_TIME_EDGES_SECONDS[1:],
        ):
            first = int(round(start * sfreq))
            last = min(int(round(stop * sfreq)), values.shape[-1])
            mean_difference = np.mean(phase_difference[..., first:last], axis=-1)
            blocks.extend(
                (
                    mean_difference.real[:, None],
                    mean_difference.imag[:, None],
                    np.abs(mean_difference)[:, None],
                )
            )
            names.extend(
                (
                    f"C3_minus_C4_mrcp_phase_cos_{start:.2f}_{stop:.2f}s",
                    f"C3_minus_C4_mrcp_phase_sin_{start:.2f}_{stop:.2f}s",
                    f"C3_minus_C4_mrcp_phase_concentration_{start:.2f}_{stop:.2f}s",
                )
            )

    features = np.concatenate(
        [block.reshape(len(values), -1) for block in blocks],
        axis=1,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32), tuple(names)


def _band_power(
    epochs: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
) -> np.ndarray:
    frequencies, spectrum = welch(
        epochs,
        fs=sfreq,
        nperseg=epochs.shape[-1],
        axis=-1,
    )
    selected = (frequencies >= low) & (frequencies < high)
    return np.trapezoid(spectrum[..., selected], frequencies[selected], axis=-1)


def _individual_mu_centers(
    dataset: TranslationalWindowDataset,
    calibration_mask: np.ndarray,
) -> np.ndarray:
    """Estimate each subject's task-relative 8-14 Hz center from calibration only."""
    frequencies, spectrum = welch(
        dataset.smr_epochs,
        fs=dataset.sfreq,
        nperseg=dataset.smr_epochs.shape[-1],
        axis=-1,
    )
    mu = (frequencies >= 8.0) & (frequencies <= 14.0)
    centers = np.full(len(dataset.labels), 10.0, dtype=np.float32)
    for subject in np.unique(dataset.subject_keys):
        calibration = calibration_mask & (dataset.subject_keys == subject)
        rest = calibration & (dataset.labels == 0)
        intent = calibration & (dataset.labels != 0)
        if not np.any(rest) or not np.any(intent):
            continue
        rest_power = np.median(spectrum[rest], axis=(0, 1))
        intent_power = np.median(spectrum[intent], axis=(0, 1))
        erd = 10.0 * np.log10(
            np.clip(intent_power, 1e-30, None)
            / np.clip(rest_power, 1e-30, None)
        )
        center = float(frequencies[mu][np.argmin(erd[mu])])
        centers[dataset.subject_keys == subject] = center
    return centers


def smr_features(
    dataset: TranslationalWindowDataset,
    calibration_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Extract fixed and individualized spectral-temporal motor features."""
    epochs = np.asarray(dataset.smr_epochs, dtype=np.float64)
    fixed, fixed_names = fixed_smr_features(
        epochs,
        dataset.sfreq,
        dataset.channel_names,
    )
    values = [fixed]
    names: list[str] = list(fixed_names)
    centers = _individual_mu_centers(dataset, calibration_mask)
    individual = np.empty(
        (len(epochs), len(dataset.channel_names)),
        dtype=np.float64,
    )
    for center in np.unique(centers):
        selected = centers == center
        individual[selected] = 10.0 * np.log10(
            np.clip(
                _band_power(
                    epochs[selected],
                    dataset.sfreq,
                    max(7.0, float(center) - 2.0),
                    min(15.0, float(center) + 2.0),
                ),
                1e-30,
                None,
            )
        )
    values.append(individual)
    names.extend(
        f"{channel}_individual_mu_log_power"
        for channel in dataset.channel_names
    )
    lookup = {name.upper(): index for index, name in enumerate(dataset.channel_names)}
    if "C3" in lookup and "C4" in lookup:
        values.append(
            (individual[:, lookup["C3"]] - individual[:, lookup["C4"]])[:, None]
        )
        names.append("C3_minus_C4_individual_mu_log_power")
    return np.concatenate(values, axis=1).astype(np.float32), tuple(names), centers


def fixed_smr_features(
    epochs: np.ndarray,
    sfreq: float,
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract fixed filter-bank features transferable across datasets."""
    signal = np.asarray(epochs, dtype=np.float64)
    values = []
    names: list[str] = []
    for band_name, low, high in SMR_FILTER_BANK:
        sos = butter(
            4,
            (low, high),
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        filtered = sosfiltfilt(sos, signal, axis=-1)
        power = filtered**2
        binned = _time_bin_means(power, sfreq, SMR_TIME_EDGES_SECONDS)
        values.append(
            10.0 * np.log10(np.clip(binned, 1e-30, None)).reshape(
                len(signal),
                -1,
            )
        )
        names.extend(
            f"{channel}_{band_name}_{start:.2f}_{stop:.2f}s_log_power"
            for channel in channel_names
            for start, stop in zip(
                SMR_TIME_EDGES_SECONDS[:-1],
                SMR_TIME_EDGES_SECONDS[1:],
            )
        )
    return np.concatenate(values, axis=1).astype(np.float32), tuple(names)


def filter_bank_covariances(
    epochs: np.ndarray,
    sfreq: float,
    *,
    shrinkage: float = 0.05,
) -> np.ndarray:
    count, channels = epochs.shape[:2]
    covariances = np.empty(
        (count, len(SMR_FILTER_BANK), channels, channels),
        dtype=np.float32,
    )
    identity = np.eye(channels, dtype=np.float64) / channels
    for band_index, (_, low, high) in enumerate(SMR_FILTER_BANK):
        filtered = sosfiltfilt(
            butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos"),
            epochs,
            axis=-1,
        )
        centered = filtered - np.mean(filtered, axis=-1, keepdims=True)
        covariance = np.einsum(
            "nct,ndt->ncd",
            centered,
            centered,
            optimize=True,
        )
        covariance /= max(epochs.shape[-1] - 1, 1)
        traces = np.trace(covariance, axis1=1, axis2=2)
        covariance /= np.clip(traces[:, None, None], 1e-20, None)
        covariance = (1.0 - shrinkage) * covariance + shrinkage * identity
        covariances[:, band_index] = covariance.astype(np.float32)
    return covariances


def build_translational_features(
    dataset: TranslationalWindowDataset,
    training_mask: np.ndarray,
    calibration_mask: np.ndarray | None = None,
    *,
    include_graph: bool = False,
) -> TranslationalFeatures:
    """Build leakage-safe branches using only calibration/training references."""
    dataset.validate()
    training_mask = np.asarray(training_mask, dtype=bool)
    if training_mask.shape != dataset.labels.shape or not np.any(training_mask):
        raise ValueError("A non-empty row-aligned training mask is required.")
    reference_mask = (
        training_mask
        if calibration_mask is None
        else np.asarray(calibration_mask, dtype=bool)
    )
    if reference_mask.shape != dataset.labels.shape:
        raise ValueError("Calibration mask must align with dataset rows.")
    mrcp, mrcp_names = mrcp_features(
        dataset.mrcp_epochs,
        dataset.sfreq,
        dataset.channel_names,
    )
    smr, smr_names, individual_mu = smr_features(dataset, reference_mask)
    covariances = filter_bank_covariances(dataset.smr_epochs, dataset.sfreq)
    riemann, riemann_metadata = tangent_features(
        covariances,
        training_mask,
        dataset.subject_keys,
        dataset.sessions,
        dataset.recordings,
        alignment="none",
    )
    riemann_names = tuple(
        f"{band_name}_tangent_{index:03d}"
        for band_name, _, _ in SMR_FILTER_BANK
        for index in range(riemann.shape[1] // len(SMR_FILTER_BANK))
    )
    branches = {
        "mrcp": mrcp,
        "smr": smr,
        "riemann": riemann,
        "quality": dataset.quality_features.astype(np.float32, copy=False),
    }
    names = {
        "mrcp": mrcp_names,
        "smr": smr_names,
        "riemann": riemann_names,
        "quality": dataset.quality_feature_names,
    }
    graph_metadata = None
    if include_graph:
        graph, graph_names, graph_metadata = fixed_graph_features(
            dataset.mrcp_epochs,
            dataset.smr_epochs,
            dataset.sfreq,
            dataset.channel_names,
            SMR_FILTER_BANK,
            mrcp_time_edges=MRCP_TIME_EDGES_SECONDS,
            smr_time_edges=SMR_TIME_EDGES_SECONDS,
        )
        branches["gsp"] = graph
        names["gsp"] = graph_names
    features = TranslationalFeatures(
        branches=branches,
        names=names,
        covariances=covariances,
        individualized_mu_hz=individual_mu,
        metadata={
            "mrcp_band_hz": [0.1, 4.0],
            "smr_filter_bank": [list(value) for value in SMR_FILTER_BANK],
            "individual_mu_reference": "explicit calibration rows only",
            "riemann": riemann_metadata,
            "gsp": graph_metadata,
            "quality_branch_role": "availability gate only",
        },
    )
    features.validate(len(dataset.labels))
    return features


def build_external_translational_features(
    dataset: TranslationalWindowDataset,
    reference_dataset: TranslationalWindowDataset,
    reference_training_mask: np.ndarray,
    *,
    include_graph: bool = False,
) -> TranslationalFeatures:
    """Transform a locked cohort without estimating parameters from it."""
    dataset.validate()
    reference_dataset.validate()
    if dataset.channel_names != reference_dataset.channel_names:
        raise ValueError("External and frozen-reference channels do not match.")
    if abs(dataset.sfreq - reference_dataset.sfreq) > 1e-6:
        raise ValueError("External and frozen-reference sampling rates do not match.")
    training_mask = np.asarray(reference_training_mask, dtype=bool)
    if training_mask.shape != reference_dataset.labels.shape or not np.any(training_mask):
        raise ValueError("A non-empty frozen-reference training mask is required.")

    mrcp, mrcp_names = mrcp_features(
        dataset.mrcp_epochs,
        dataset.sfreq,
        dataset.channel_names,
    )
    no_external_calibration = np.zeros(len(dataset.labels), dtype=bool)
    smr, smr_names, individual_mu = smr_features(
        dataset,
        no_external_calibration,
    )
    covariances = filter_bank_covariances(dataset.smr_epochs, dataset.sfreq)
    reference_covariances = filter_bank_covariances(
        reference_dataset.smr_epochs[training_mask],
        reference_dataset.sfreq,
    )
    riemann, riemann_metadata = tangent_features_from_reference(
        covariances,
        reference_covariances,
    )
    riemann_names = tuple(
        f"{band_name}_tangent_{index:03d}"
        for band_name, _, _ in SMR_FILTER_BANK
        for index in range(riemann.shape[1] // len(SMR_FILTER_BANK))
    )
    branches = {
        "mrcp": mrcp,
        "smr": smr,
        "riemann": riemann,
        "quality": dataset.quality_features.astype(np.float32, copy=False),
    }
    names = {
        "mrcp": mrcp_names,
        "smr": smr_names,
        "riemann": riemann_names,
        "quality": dataset.quality_feature_names,
    }
    graph_metadata = None
    if include_graph:
        graph, graph_names, graph_metadata = fixed_graph_features(
            dataset.mrcp_epochs,
            dataset.smr_epochs,
            dataset.sfreq,
            dataset.channel_names,
            SMR_FILTER_BANK,
            mrcp_time_edges=MRCP_TIME_EDGES_SECONDS,
            smr_time_edges=SMR_TIME_EDGES_SECONDS,
        )
        branches["gsp"] = graph
        names["gsp"] = graph_names
    features = TranslationalFeatures(
        branches=branches,
        names=names,
        covariances=covariances,
        individualized_mu_hz=individual_mu,
        metadata={
            "mode": "locked_external",
            "individual_mu_hz": 10.0,
            "individual_mu_reference": "none_from_external_cohort",
            "riemann": riemann_metadata,
            "gsp": graph_metadata,
            "reference_training_rows": int(np.sum(training_mask)),
            "quality_branch_role": "frozen_availability_gate_only",
        },
    )
    features.validate(len(dataset.labels))
    return features
