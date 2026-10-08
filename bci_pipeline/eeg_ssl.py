"""Compact masked-reconstruction EEG encoder and intent fine-tuning helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class BranchScales:
    mrcp: float
    smr: float

    def validate(self) -> None:
        if not np.isfinite(self.mrcp) or not np.isfinite(self.smr):
            raise ValueError("EEG branch scales must be finite.")
        if self.mrcp <= 0.0 or self.smr <= 0.0:
            raise ValueError("EEG branch scales must be positive.")


def robust_branch_scales(mrcp_epochs: np.ndarray, smr_epochs: np.ndarray) -> BranchScales:
    """Estimate fixed source scales while preserving between-window power changes."""
    mrcp = np.asarray(mrcp_epochs, dtype=np.float64)
    smr = np.asarray(smr_epochs, dtype=np.float64)
    if mrcp.shape != smr.shape or mrcp.ndim != 3 or not len(mrcp):
        raise ValueError("MRCP and SMR branches must be matching non-empty 3-D arrays.")
    mrcp_rms = np.sqrt(np.mean(mrcp**2, axis=(1, 2)))
    smr_rms = np.sqrt(np.mean(smr**2, axis=(1, 2)))
    scales = BranchScales(
        mrcp=float(np.median(mrcp_rms)),
        smr=float(np.median(smr_rms)),
    )
    scales.validate()
    return scales


def normalized_branch_tensor(
    mrcp: torch.Tensor,
    smr: torch.Tensor,
    scales: BranchScales,
    *,
    clip: float = 12.0,
) -> torch.Tensor:
    scales.validate()
    if mrcp.shape != smr.shape or mrcp.ndim != 3:
        raise ValueError("MRCP and SMR tensors must have matching batch shapes.")
    combined = torch.cat((mrcp / scales.mrcp, smr / scales.smr), dim=1)
    return torch.clamp(combined, -clip, clip)


class BranchEpochDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        mrcp_epochs: np.ndarray,
        smr_epochs: np.ndarray,
        indices: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        mrcp = np.asarray(mrcp_epochs, dtype=np.float32)
        smr = np.asarray(smr_epochs, dtype=np.float32)
        if mrcp.shape != smr.shape or mrcp.ndim != 3:
            raise ValueError("Branch epochs must be matching 3-D arrays.")
        selected = np.arange(len(mrcp)) if indices is None else np.asarray(indices, dtype=np.int64)
        self.mrcp = mrcp
        self.smr = smr
        self.indices = selected
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.sample_weights = (
            None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)
        )
        for values, name in (
            (self.labels, "labels"),
            (self.sample_weights, "sample weights"),
        ):
            if values is not None and len(values) != len(mrcp):
                raise ValueError(f"EEG {name} must align with all branch rows.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        values: list[torch.Tensor] = [
            torch.from_numpy(self.mrcp[index]),
            torch.from_numpy(self.smr[index]),
        ]
        if self.labels is not None:
            values.append(torch.tensor(self.labels[index], dtype=torch.long))
        if self.sample_weights is not None:
            values.append(torch.tensor(self.sample_weights[index], dtype=torch.float32))
        return tuple(values)


class EEGSSLBackbone(nn.Module):
    """Temporal encoder shared by masked reconstruction and intent decoding."""

    def __init__(self, input_channels: int = 14) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 24, 9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, 48, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(6, 48),
            nn.GELU(),
            nn.Conv1d(48, 72, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 72),
            nn.GELU(),
            nn.Conv1d(72, 96, 5, stride=1, padding=2, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.embedding_size = 192

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.encoder(signal)

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        encoded = self.forward(signal)
        return torch.cat((encoded.mean(dim=-1), encoded.std(dim=-1)), dim=1)


class MaskedEEGAutoencoder(nn.Module):
    def __init__(self, backbone: EEGSSLBackbone | None = None) -> None:
        super().__init__()
        self.backbone = backbone or EEGSSLBackbone()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(96, 72, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 72),
            nn.GELU(),
            nn.ConvTranspose1d(72, 48, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(6, 48),
            nn.GELU(),
            nn.ConvTranspose1d(48, 24, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            nn.Conv1d(24, self.backbone.input_channels, 7, padding=3),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.backbone(signal))


class EEGIntentNetwork(nn.Module):
    def __init__(self, backbone: EEGSSLBackbone | None = None) -> None:
        super().__init__()
        self.backbone = backbone or EEGSSLBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embedding_size, 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.embedding(signal))


def copied_backbone(backbone: EEGSSLBackbone) -> EEGSSLBackbone:
    return copy.deepcopy(backbone)


def corrupt_eeg_batch(
    signal: torch.Tensor,
    *,
    element_fraction: float = 0.15,
    channel_fraction: float = 0.10,
    noise_scale: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask independent samples and whole channels without using labels."""
    element_mask = torch.rand_like(signal) < element_fraction
    channel_mask = (
        torch.rand(
            (signal.shape[0], signal.shape[1], 1),
            device=signal.device,
        )
        < channel_fraction
    )
    mask = element_mask | channel_mask
    corrupted = signal.masked_fill(mask, 0.0)
    corrupted = corrupted + noise_scale * torch.randn_like(corrupted)
    return corrupted, mask


def train_masked_autoencoder(
    model: MaskedEEGAutoencoder,
    dataset: BranchEpochDataset,
    scales: BranchScales,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        rows = 0
        for mrcp, smr in loader:
            signal = normalized_branch_tensor(
                mrcp.to(device),
                smr.to(device),
                scales,
            )
            corrupted, mask = corrupt_eeg_batch(signal)
            reconstruction = model(corrupted)
            weight = 0.1 + mask.to(reconstruction.dtype)
            loss = torch.sum(weight * (reconstruction - signal) ** 2) / torch.sum(weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(signal)
            rows += len(signal)
        losses.append(total / rows)
        print(f"    SSL epoch {epoch + 1:02d}/{epochs:02d} | reconstruction loss={losses[-1]:.5f}")
    model.to("cpu")
    return losses


def train_intent_network(
    model: EEGIntentNetwork,
    dataset: BranchEpochDataset,
    scales: BranchScales,
    *,
    epochs: int,
    batch_size: int,
    encoder_learning_rate: float,
    head_learning_rate: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        (
            {"params": model.backbone.parameters(), "lr": encoder_learning_rate},
            {"params": model.classifier.parameters(), "lr": head_learning_rate},
        ),
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        mass = 0.0
        for mrcp, smr, labels, weights in loader:
            signal = normalized_branch_tensor(
                mrcp.to(device),
                smr.to(device),
                scales,
            )
            target = labels.to(device)
            sample_weight = weights.to(device)
            per_row = criterion(model(signal), target)
            loss = torch.sum(per_row * sample_weight) / torch.sum(sample_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(torch.sum(per_row.detach() * sample_weight))
            mass += float(torch.sum(sample_weight))
        losses.append(total / mass)
        print(f"    fine-tune epoch {epoch + 1:02d}/{epochs:02d} | weighted loss={losses[-1]:.5f}")
    model.to("cpu")
    return losses


@torch.no_grad()
def predict_intent_probabilities(
    model: EEGIntentNetwork,
    dataset: BranchEpochDataset,
    scales: BranchScales,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device)
    model.eval()
    probabilities: list[np.ndarray] = []
    for batch in loader:
        mrcp, smr = batch[:2]
        signal = normalized_branch_tensor(mrcp.to(device), smr.to(device), scales)
        probability = torch.softmax(model(signal), dim=1)[:, 1]
        probabilities.append(probability.cpu().numpy())
    model.to("cpu")
    return np.concatenate(probabilities)


def model_payload(
    model: EEGIntentNetwork,
    scales: BranchScales,
    thresholds: dict[str, float],
    channel_names: Sequence[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "architecture": "masked_reconstruction_conv1d_intent_network",
        "model_state": model.state_dict(),
        "branch_scales": {"mrcp": scales.mrcp, "smr": scales.smr},
        "thresholds": dict(thresholds),
        "channel_names": tuple(str(value) for value in channel_names),
        "input_branches": ("mrcp_0.1_4_hz", "smr_7_35_hz"),
        "peripheral_inputs_required": False,
    }


def load_intent_model_payload(
    path: str | Path,
) -> tuple[EEGIntentNetwork, BranchScales, dict[str, float], tuple[str, ...]]:
    """Restore a portable EEG-only decoder payload on CPU."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "model_state",
        "branch_scales",
        "thresholds",
        "channel_names",
        "peripheral_inputs_required",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"EEG SSL model payload is missing keys: {sorted(missing)}")
    if payload["peripheral_inputs_required"]:
        raise ValueError("Expected an EEG-only model payload.")
    model = EEGIntentNetwork()
    model.load_state_dict(payload["model_state"])
    model.eval()
    scales = BranchScales(
        mrcp=float(payload["branch_scales"]["mrcp"]),
        smr=float(payload["branch_scales"]["smr"]),
    )
    scales.validate()
    thresholds = {
        str(domain): float(value)
        for domain, value in payload["thresholds"].items()
    }
    channels = tuple(str(value) for value in payload["channel_names"])
    return model, scales, thresholds, channels
