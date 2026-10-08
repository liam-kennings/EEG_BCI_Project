"""Shared infrastructure for patient-adaptive EEG motor-intent experiments."""

from __future__ import annotations

import os
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MNE and matplotlib otherwise attempt to write into the user's home directory.
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba"))
os.environ.setdefault("MPLBACKEND", "Agg")

# NumPy linked against Accelerate on this environment emits false-positive
# matmul RuntimeWarnings even when both inputs and outputs are finite. Scientific
# boundaries explicitly validate finiteness before model fitting.
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning,
)
