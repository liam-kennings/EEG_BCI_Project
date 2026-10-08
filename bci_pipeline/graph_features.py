"""Fixed-geometry graph signal features for the shared motor montage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt


MOTOR_GRAPH_CHANNELS = ("FC3", "FC4", "C3", "CZ", "C4", "CP3", "CP4")

# Cartesian positions from MNE's standard_1005 montage, stored here so the
# deployment transform does not depend on a runtime montage lookup.
MOTOR_GRAPH_COORDINATES = np.asarray(
    (
        (-0.0601819, 0.0227162, 0.0555440),
        (0.0622931, 0.0237228, 0.0556300),
        (-0.0653581, -0.0116317, 0.0643580),
        (0.0004009, -0.0091670, 0.1002440),
        (0.0671179, -0.0109003, 0.0635800),
        (-0.0635562, -0.0470088, 0.0656240),
        (0.0666118, -0.0466372, 0.0655800),
    ),
    dtype=np.float64,
)

GRAPH_ENERGY_GROUPS = (
    ("low", (1, 2)),
    ("mid", (3, 4)),
    ("high", (5, 6)),
)


@dataclass(frozen=True)
class FixedElectrodeGraph:
    channel_names: tuple[str, ...]
    coordinates: np.ndarray
    adjacency: np.ndarray
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    input_order: np.ndarray
    neighbor_count: int
    distance_scale: float

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "construction": "fixed_standard_1005_geometry",
            "channel_names": list(self.channel_names),
            "coordinates_m": self.coordinates.tolist(),
            "adjacency": self.adjacency.tolist(),
            "laplacian_eigenvalues": self.eigenvalues.tolist(),
            "neighbor_count": int(self.neighbor_count),
            "distance_scale_m": float(self.distance_scale),
            "edge_parameters_learned_from_eeg": False,
        }


def fixed_electrode_graph(
    channel_names: Sequence[str],
    *,
    neighbor_count: int = 3,
) -> FixedElectrodeGraph:
    """Build a channel-order-invariant graph from fixed electrode positions."""
    normalized = tuple(str(name).upper() for name in channel_names)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Graph channels must be unique.")
    missing = [name for name in MOTOR_GRAPH_CHANNELS if name not in normalized]
    if missing:
        raise ValueError(f"Shared motor graph channels are missing: {missing}")
    if not 1 <= neighbor_count < len(MOTOR_GRAPH_CHANNELS):
        raise ValueError("neighbor_count must be between one and channels minus one.")

    input_order = np.asarray(
        [normalized.index(name) for name in MOTOR_GRAPH_CHANNELS],
        dtype=np.int64,
    )
    coordinates = MOTOR_GRAPH_COORDINATES.copy()
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    nonzero = distances[distances > 0.0]
    distance_scale = float(np.median(np.min(
        np.where(distances > 0.0, distances, np.inf),
        axis=1,
    )))
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        distance_scale = float(np.median(nonzero))

    dense = np.exp(-(distances**2) / (2.0 * distance_scale**2))
    np.fill_diagonal(dense, 0.0)
    keep = np.zeros_like(dense, dtype=bool)
    for index in range(len(MOTOR_GRAPH_CHANNELS)):
        nearest = np.argsort(distances[index])[1 : neighbor_count + 1]
        keep[index, nearest] = True
    keep |= keep.T
    adjacency = np.where(keep, dense, 0.0)
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = np.diag(np.sum(adjacency, axis=1))
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues[np.abs(eigenvalues) < 1e-12] = 0.0
    return FixedElectrodeGraph(
        channel_names=MOTOR_GRAPH_CHANNELS,
        coordinates=coordinates,
        adjacency=adjacency,
        laplacian=laplacian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        input_order=input_order,
        neighbor_count=neighbor_count,
        distance_scale=distance_scale,
    )


def _time_bin_means(
    values: np.ndarray,
    sfreq: float,
    edges: Sequence[float],
) -> np.ndarray:
    blocks = []
    for start, stop in zip(edges[:-1], edges[1:]):
        first = int(round(float(start) * sfreq))
        last = min(int(round(float(stop) * sfreq)), values.shape[-1])
        if last <= first:
            raise ValueError("Graph feature time bins contain no samples.")
        blocks.append(np.mean(values[..., first:last], axis=-1))
    return np.stack(blocks, axis=-1)


def _graph_summary(
    node_values: np.ndarray,
    graph: FixedElectrodeGraph,
) -> np.ndarray:
    """Return smoothness and normalized non-constant graph-band energies."""
    values = np.asarray(node_values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(graph.channel_names):
        raise ValueError("Graph values must have shape trials x channels x bins.")
    values = values - np.mean(values, axis=1, keepdims=True)
    denominator = np.sum(values**2, axis=1)
    smoothness = np.einsum(
        "ncb,cd,ndb->nb",
        values,
        graph.laplacian,
        values,
        optimize=True,
    ) / np.clip(denominator, 1e-20, None)
    coefficients = np.einsum(
        "ck,ncb->nkb",
        graph.eigenvectors,
        values,
        optimize=True,
    )
    energy = coefficients**2
    nonconstant = np.sum(energy[:, 1:], axis=1)
    fractions = []
    for _, modes in GRAPH_ENERGY_GROUPS:
        fractions.append(
            np.sum(energy[:, list(modes)], axis=1)
            / np.clip(nonconstant, 1e-20, None)
        )
    return np.stack((smoothness, *fractions), axis=-1)


def fixed_graph_features(
    mrcp_epochs: np.ndarray,
    smr_epochs: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
    smr_filter_bank: Sequence[tuple[str, float, float]],
    *,
    mrcp_time_edges: Sequence[float],
    smr_time_edges: Sequence[float],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
    """Extract fixed graph-spectral summaries from MRCP and SMR windows."""
    graph = fixed_electrode_graph(channel_names)
    mrcp = np.asarray(mrcp_epochs, dtype=np.float64)[:, graph.input_order]
    smr = np.asarray(smr_epochs, dtype=np.float64)[:, graph.input_order]
    if mrcp.shape[0] != smr.shape[0] or sfreq <= 0.0:
        raise ValueError("MRCP and SMR graph inputs must be row-aligned.")

    feature_blocks = []
    names: list[str] = []
    descriptors = (
        "smoothness",
        "low_energy_fraction",
        "mid_energy_fraction",
        "high_energy_fraction",
    )

    mrcp_bins = _time_bin_means(mrcp * 1e6, sfreq, mrcp_time_edges)
    feature_blocks.append(_graph_summary(mrcp_bins, graph).reshape(len(mrcp), -1))
    names.extend(
        f"mrcp_graph_{start:.2f}_{stop:.2f}s_{descriptor}"
        for start, stop in zip(mrcp_time_edges[:-1], mrcp_time_edges[1:])
        for descriptor in descriptors
    )

    for band_name, low, high in smr_filter_bank:
        filtered = sosfiltfilt(
            butter(4, (low, high), btype="bandpass", fs=sfreq, output="sos"),
            smr,
            axis=-1,
        )
        log_power_bins = 10.0 * np.log10(
            np.clip(
                _time_bin_means(filtered**2, sfreq, smr_time_edges),
                1e-30,
                None,
            )
        )
        feature_blocks.append(
            _graph_summary(log_power_bins, graph).reshape(len(smr), -1)
        )
        names.extend(
            f"{band_name}_graph_{start:.2f}_{stop:.2f}s_{descriptor}"
            for start, stop in zip(smr_time_edges[:-1], smr_time_edges[1:])
            for descriptor in descriptors
        )

    features = np.concatenate(feature_blocks, axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = graph.metadata()
    metadata.update(
        {
            "feature_count": int(features.shape[1]),
            "feature_families": [
                "normalized_graph_smoothness",
                "low_mid_high_graph_frequency_energy",
            ],
            "mrcp_time_edges_seconds": [float(value) for value in mrcp_time_edges],
            "smr_time_edges_seconds": [float(value) for value in smr_time_edges],
            "smr_filter_bank": [list(value) for value in smr_filter_bank],
        }
    )
    return features.astype(np.float32), tuple(names), metadata

