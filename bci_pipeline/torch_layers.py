"""Small neural layers implemented consistently across CPU, CUDA, and MPS."""

from __future__ import annotations

import torch
from torch import nn


def temporal_adaptive_average_pool2d(
    values: torch.Tensor,
    output_bins: int,
) -> torch.Tensor:
    """Match temporal adaptive average pooling without unsupported MPS kernels."""
    if values.ndim != 4 or values.shape[-2] != 1:
        raise ValueError("Temporal pooling expects [batch, channels, 1, time].")
    samples = int(values.shape[-1])
    if output_bins <= 0 or samples < output_bins:
        raise ValueError("Temporal output bins must be positive and no greater than time.")
    pooled = []
    for index in range(output_bins):
        start = (index * samples) // output_bins
        stop = ((index + 1) * samples + output_bins - 1) // output_bins
        pooled.append(values[..., start:stop].mean(dim=-1, keepdim=True))
    return torch.cat(pooled, dim=-1)


class TemporalAdaptiveAveragePool2d(nn.Module):
    """Parameter-free temporal adaptive average pooling with full MPS support."""

    def __init__(self, output_bins: int) -> None:
        super().__init__()
        self.output_bins = int(output_bins)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return temporal_adaptive_average_pool2d(values, self.output_bins)

    def extra_repr(self) -> str:
        return f"output_bins={self.output_bins}"
