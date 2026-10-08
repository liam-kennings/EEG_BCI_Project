"""Load and ensemble the selected MRCP + FBCNet motor-intent decoders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .eeg_bakeoff_models import build_bakeoff_model
from .eeg_ssl import (
    BranchEpochDataset,
    BranchScales,
    predict_intent_probabilities,
)
from .torch_device import resolve_torch_device


DEFAULT_NEURAL_SEEDS = (42, 43, 44, 45, 46)


def _mean_pairwise_correlation(matrix: np.ndarray) -> float:
    values = []
    for first in range(len(matrix)):
        for second in range(first + 1, len(matrix)):
            left = matrix[first]
            right = matrix[second]
            if np.std(left) == 0.0 or np.std(right) == 0.0:
                values.append(1.0 if np.array_equal(left, right) else 0.0)
            else:
                values.append(float(np.corrcoef(left, right)[0, 1]))
    return float(np.mean(values)) if values else 1.0


def _load_payload(
    path: Path,
    expected_architecture: str = "mrcp_fbcnet",
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {
        "architecture",
        "model_state",
        "branch_scales",
        "channel_names",
        "neural_seed",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Neural payload {path} is missing: {sorted(missing)}")
    if payload["architecture"] != expected_architecture:
        raise ValueError(
            f"Expected {expected_architecture} payload, "
            f"got {payload['architecture']!r}."
        )
    if payload.get("peripheral_inputs_required", False):
        raise ValueError("The integrated decoder must remain EEG-only at inference.")
    return payload


def predict_bakeoff_ensemble(
    mrcp_epochs: np.ndarray,
    smr_epochs: np.ndarray,
    channel_names: Sequence[str],
    bakeoff_dir: str | Path,
    *,
    architecture: str,
    seeds: Sequence[int] = DEFAULT_NEURAL_SEEDS,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Return a bakeoff architecture's seed-mean intent probabilities."""
    if architecture not in {"shallow_convnet", "mrcp_fbcnet"}:
        raise ValueError(
            "Integrated architecture must be shallow_convnet or mrcp_fbcnet."
        )
    selected_seeds = tuple(int(seed) for seed in seeds)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Neural ensemble seeds must be non-empty and unique.")
    root = Path(bakeoff_dir).expanduser().resolve()
    target_channels = tuple(str(channel) for channel in channel_names)
    dataset = BranchEpochDataset(mrcp_epochs, smr_epochs)
    inference_device = resolve_torch_device(device)
    per_seed: dict[int, np.ndarray] = {}
    model_records = []
    for seed in selected_seeds:
        path = root / "seeds" / f"seed_{seed}" / f"{architecture}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_payload(path, architecture)
        payload_channels = tuple(str(value) for value in payload["channel_names"])
        if payload_channels != target_channels:
            raise ValueError(
                f"Neural model channels {payload_channels} do not match {target_channels}."
            )
        if int(payload["neural_seed"]) != seed:
            raise ValueError(f"Neural payload seed mismatch in {path}.")
        scales = BranchScales(
            mrcp=float(payload["branch_scales"]["mrcp"]),
            smr=float(payload["branch_scales"]["smr"]),
        )
        scales.validate()
        model = build_bakeoff_model(architecture)
        model.load_state_dict(payload["model_state"])
        probability = predict_intent_probabilities(
            model,
            dataset,
            scales,
            batch_size=batch_size,
            device=inference_device,
        ).astype(np.float64)
        if probability.shape != (len(dataset),) or not np.all(np.isfinite(probability)):
            raise ValueError(f"Neural model {path} emitted invalid probabilities.")
        per_seed[seed] = probability
        model_records.append(
            {
                "seed": seed,
                "path": str(path),
                "mrcp_scale": scales.mrcp,
                "smr_scale": scales.smr,
            }
        )
    matrix = np.vstack([per_seed[seed] for seed in selected_seeds])
    metadata = {
        "architecture": architecture,
        "ensemble_method": "arithmetic_mean_of_intent_probabilities",
        "seeds": list(selected_seeds),
        "models": model_records,
        "rows": int(matrix.shape[1]),
        "channels": list(target_channels),
        "peripheral_inputs_required": False,
        "device": str(inference_device),
        "mean_probability_seed_sd": float(np.mean(np.std(matrix, axis=0))),
        "mean_pairwise_seed_correlation": _mean_pairwise_correlation(matrix),
    }
    return np.mean(matrix, axis=0), per_seed, metadata


def predict_mrcp_fbcnet_ensemble(
    mrcp_epochs: np.ndarray,
    smr_epochs: np.ndarray,
    channel_names: Sequence[str],
    bakeoff_dir: str | Path,
    *,
    seeds: Sequence[int] = DEFAULT_NEURAL_SEEDS,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Backward-compatible MRCP + FBCNet ensemble entry point."""
    return predict_bakeoff_ensemble(
        mrcp_epochs,
        smr_epochs,
        channel_names,
        bakeoff_dir,
        architecture="mrcp_fbcnet",
        seeds=seeds,
        batch_size=batch_size,
        device=device,
    )


def predict_mrcp_fbcnet_files(
    mrcp_epochs: np.ndarray,
    smr_epochs: np.ndarray,
    channel_names: Sequence[str],
    model_files: Sequence[str | Path],
    *,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    """Run an ensemble from immutable model files copied into a lock package."""
    paths = tuple(Path(value).expanduser().resolve() for value in model_files)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Neural model files must be non-empty and unique.")
    target_channels = tuple(str(channel) for channel in channel_names)
    dataset = BranchEpochDataset(mrcp_epochs, smr_epochs)
    inference_device = resolve_torch_device(device)
    per_seed: dict[int, np.ndarray] = {}
    model_records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_payload(path, "mrcp_fbcnet")
        payload_channels = tuple(str(value) for value in payload["channel_names"])
        if payload_channels != target_channels:
            raise ValueError(
                f"Neural model channels {payload_channels} do not match {target_channels}."
            )
        seed = int(payload["neural_seed"])
        if seed in per_seed:
            raise ValueError(f"Duplicate neural seed {seed} in locked model files.")
        scales = BranchScales(
            mrcp=float(payload["branch_scales"]["mrcp"]),
            smr=float(payload["branch_scales"]["smr"]),
        )
        scales.validate()
        model = build_bakeoff_model("mrcp_fbcnet")
        model.load_state_dict(payload["model_state"])
        probability = predict_intent_probabilities(
            model,
            dataset,
            scales,
            batch_size=batch_size,
            device=inference_device,
        ).astype(np.float64)
        if probability.shape != (len(dataset),) or not np.all(np.isfinite(probability)):
            raise ValueError(f"Neural model {path} emitted invalid probabilities.")
        per_seed[seed] = probability
        model_records.append(
            {
                "seed": seed,
                "path": str(path),
                "mrcp_scale": scales.mrcp,
                "smr_scale": scales.smr,
            }
        )
    selected_seeds = tuple(sorted(per_seed))
    matrix = np.vstack([per_seed[seed] for seed in selected_seeds])
    metadata = {
        "architecture": "mrcp_fbcnet",
        "ensemble_method": "arithmetic_mean_of_intent_probabilities",
        "seeds": list(selected_seeds),
        "models": model_records,
        "rows": int(matrix.shape[1]),
        "channels": list(target_channels),
        "peripheral_inputs_required": False,
        "device": str(inference_device),
        "mean_probability_seed_sd": float(np.mean(np.std(matrix, axis=0))),
        "mean_pairwise_seed_correlation": _mean_pairwise_correlation(matrix),
    }
    return np.mean(matrix, axis=0), per_seed, metadata
