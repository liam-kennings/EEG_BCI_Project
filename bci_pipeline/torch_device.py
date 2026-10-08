"""Shared PyTorch device selection and audit metadata."""

from __future__ import annotations

from typing import Any

import torch


TORCH_DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def mps_is_built() -> bool:
    """Return whether this PyTorch build includes the Apple MPS backend."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built())


def mps_is_available() -> bool:
    """Return whether Apple MPS can be used in the current process."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_torch_device(requested: str | torch.device | None = "auto") -> torch.device:
    """Resolve an explicit or automatic neural-compute backend."""
    value = "auto" if requested is None else str(requested).strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value not in TORCH_DEVICE_CHOICES[1:]:
        choices = ", ".join(TORCH_DEVICE_CHOICES)
        raise ValueError(f"Unknown PyTorch device {value!r}; choose one of: {choices}.")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if value == "mps" and not mps_is_available():
        reason = (
            "this PyTorch build includes MPS, but the current macOS hardware/runtime "
            "does not expose it"
            if mps_is_built()
            else "this PyTorch build does not include MPS"
        )
        raise RuntimeError(f"Apple MPS was requested, but {reason}.")
    return torch.device(value)


def torch_device_metadata(
    requested: str | torch.device | None,
    resolved: torch.device | None = None,
) -> dict[str, Any]:
    """Return serializable hardware metadata for experiment reports."""
    device = resolved or resolve_torch_device(requested)
    return {
        "requested": "auto" if requested is None else str(requested),
        "resolved": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "mps_built": mps_is_built(),
        "mps_available": mps_is_available(),
    }
