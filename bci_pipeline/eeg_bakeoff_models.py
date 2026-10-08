"""Physiology-informed neural backbones for the controlled EEG decoder bakeoff."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.signal import firwin
from torch import nn

from .torch_layers import (
    TemporalAdaptiveAveragePool2d,
    temporal_adaptive_average_pool2d,
)


CHANNELS = 7
SMR_BANDS = (
    (7.0, 10.0),
    (10.0, 14.0),
    (14.0, 22.0),
    (22.0, 30.0),
)


def _broadband_input(signal: torch.Tensor) -> torch.Tensor:
    """Recombine separately normalized MRCP and SMR branches by channel."""
    if signal.ndim != 3 or signal.shape[1] != 2 * CHANNELS:
        raise ValueError("Bakeoff models require seven MRCP and seven SMR channels.")
    return signal[:, :CHANNELS] + signal[:, CHANNELS:]


class EEGNetBackbone(nn.Module):
    """Compact depthwise/separable EEGNet-style feature extractor."""

    embedding_size = 128

    def __init__(self) -> None:
        super().__init__()
        temporal_filters = 8
        spatial_filters = 16
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, 33), padding=(0, 16), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                (CHANNELS, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.35),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                (1, 15),
                padding=(0, 7),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, (1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.35),
            TemporalAdaptiveAveragePool2d(8),
        )

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        return self.features(_broadband_input(signal)[:, None]).flatten(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.embedding(signal)


class ShallowConvNetBackbone(nn.Module):
    """Shallow band-power ConvNet with square, pool, and log nonlinearities."""

    embedding_size = 192

    def __init__(self) -> None:
        super().__init__()
        filters = 24
        self.temporal = nn.Conv2d(1, filters, (1, 25), padding=(0, 12), bias=False)
        self.spatial = nn.Conv2d(filters, filters, (CHANNELS, 1), bias=False)
        self.normalization = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout(0.35)

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        values = _broadband_input(signal)[:, None]
        values = self.normalization(self.spatial(self.temporal(values)))
        values = torch.square(values)
        values = functional.avg_pool2d(values, (1, 25), stride=(1, 5))
        values = torch.log(torch.clamp(values, min=1e-6))
        values = self.dropout(values)
        return temporal_adaptive_average_pool2d(values, 8).flatten(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.embedding(signal)


class FBCNetBackbone(nn.Module):
    """Fixed filter-bank, learned spatial filters, and segmented log variance."""

    def __init__(self, sfreq: float = 100.0, bins: int = 4) -> None:
        super().__init__()
        kernel_size = 33
        filters = np.stack(
            [
                firwin(
                    kernel_size,
                    (low, high),
                    pass_zero=False,
                    fs=sfreq,
                )
                for low, high in SMR_BANDS
            ]
        ).astype(np.float32)
        self.register_buffer("filter_bank", torch.from_numpy(filters[:, None]))
        self.padding = kernel_size // 2
        self.bins = bins
        bands = len(SMR_BANDS)
        spatial_multiplier = 2
        self.spatial = nn.Conv2d(
            bands,
            bands * spatial_multiplier,
            (CHANNELS, 1),
            groups=bands,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(bands * spatial_multiplier)
        self.dropout = nn.Dropout(0.35)
        self.embedding_size = bands * spatial_multiplier * bins

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        smr = signal[:, CHANNELS:]
        batch, channels, samples = smr.shape
        filtered = functional.conv1d(
            smr.reshape(batch * channels, 1, samples),
            self.filter_bank,
            padding=self.padding,
        )
        filtered = filtered.reshape(batch, channels, len(SMR_BANDS), samples)
        filtered = filtered.permute(0, 2, 1, 3)
        projected = self.normalization(self.spatial(filtered)).squeeze(2)
        usable = (samples // self.bins) * self.bins
        segments = projected[..., :usable].reshape(
            batch,
            projected.shape[1],
            self.bins,
            usable // self.bins,
        )
        variance = torch.var(segments, dim=-1, unbiased=False)
        return self.dropout(torch.log(torch.clamp(variance, min=1e-6))).flatten(1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.embedding(signal)


class MRCPBackbone(nn.Module):
    """Spatial-temporal slow-potential encoder for motor-intent evidence."""

    embedding_size = 96

    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 12, (CHANNELS, 1), bias=False),
            nn.BatchNorm2d(12),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(12, 24, 15, padding=7, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv1d(24, 24, 9, padding=8, dilation=2, bias=False),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Dropout(0.25),
        )

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        values = self.spatial(signal[:, None, :CHANNELS, :]).squeeze(2)
        values = self.temporal(values)
        return torch.cat(
            (
                values.mean(dim=-1),
                values.std(dim=-1),
                values.amin(dim=-1),
                values.amax(dim=-1),
            ),
            dim=1,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.embedding(signal)


class MRCPFBCNetBackbone(nn.Module):
    """Recommended two-stream slow-potential and sensorimotor representation."""

    def __init__(self) -> None:
        super().__init__()
        self.mrcp = MRCPBackbone()
        self.fbcnet = FBCNetBackbone()
        self.embedding_size = self.mrcp.embedding_size + self.fbcnet.embedding_size

    def embedding(self, signal: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.mrcp.embedding(signal), self.fbcnet.embedding(signal)), dim=1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.embedding(signal)


class BakeoffIntentNetwork(nn.Module):
    """Common classifier head used by every candidate backbone."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(int(backbone.embedding_size), 64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.embedding(signal))


MODEL_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "eegnet": EEGNetBackbone,
    "shallow_convnet": ShallowConvNetBackbone,
    "fbcnet": FBCNetBackbone,
    "mrcp_fbcnet": MRCPFBCNetBackbone,
}


def build_bakeoff_model(name: str) -> BakeoffIntentNetwork:
    if name not in MODEL_FACTORIES:
        raise KeyError(f"Unknown EEG bakeoff model: {name}")
    return BakeoffIntentNetwork(MODEL_FACTORIES[name]())


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
