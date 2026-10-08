"""EEG-only unlabeled WAY-EEG-GAL windows for self-supervised pretraining."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .way_eeg_gal import (
    OUTPUT_SFREQ,
    TRANSFER_CHANNELS,
    WINDOW_SECONDS,
    _channel_names,
    _eeg_branches,
    resolve_way_eeg_gal_root,
)


DEFAULT_STRIDE_SECONDS = 2.0
DEFAULT_GUARD_SECONDS = 3.0
DEFAULT_MAX_WINDOWS_PER_SERIES = 160


@dataclass(frozen=True)
class WAYEEGSeries:
    participant: int
    series: str
    challenge_series: bool
    path: Path


@dataclass(frozen=True)
class WAYSSLWindowDataset:
    mrcp_epochs: np.ndarray
    smr_epochs: np.ndarray
    participants: np.ndarray
    series: np.ndarray
    challenge_series: np.ndarray
    window_starts_seconds: np.ndarray
    channel_names: tuple[str, ...]
    sfreq: float

    def validate(self) -> None:
        mrcp = np.asarray(self.mrcp_epochs)
        smr = np.asarray(self.smr_epochs)
        if mrcp.shape != smr.shape or mrcp.ndim != 3 or not len(mrcp):
            raise ValueError("WAY SSL branches must be matching non-empty 3-D arrays.")
        expected = int(round(WINDOW_SECONDS * self.sfreq))
        if mrcp.shape[1:] != (len(self.channel_names), expected):
            raise ValueError("WAY SSL epoch dimensions do not match the signal contract.")
        if not np.all(np.isfinite(mrcp)) or not np.all(np.isfinite(smr)):
            raise ValueError("WAY SSL epochs contain non-finite values.")
        count = len(mrcp)
        for values in (
            self.participants,
            self.series,
            self.challenge_series,
            self.window_starts_seconds,
        ):
            if len(values) != count:
                raise ValueError("WAY SSL metadata are not row-aligned.")


def discover_way_eeg_series(root: str | Path) -> list[WAYEEGSeries]:
    """Discover every continuous EEG series, including the EEG-only ST files."""
    dataset_root = resolve_way_eeg_gal_root(root)
    discovered: list[WAYEEGSeries] = []
    for participant_dir in sorted((dataset_root / "participants").glob("P[0-9][0-9]")):
        participant = int(participant_dir.name[1:])
        for path in sorted(participant_dir.glob(f"HS_P{participant}_S*.mat")):
            series = path.stem.rsplit("_", maxsplit=1)[-1]
            if not (series == "ST" or (series.startswith("S") and series[1:].isdigit())):
                continue
            discovered.append(
                WAYEEGSeries(
                    participant=participant,
                    series=series,
                    challenge_series=series == "ST",
                    path=path,
                )
            )
    return discovered


def _window_starts(
    samples: int,
    *,
    stride_seconds: float,
    guard_seconds: float,
    max_windows: int,
) -> np.ndarray:
    window_samples = int(round(WINDOW_SECONDS * OUTPUT_SFREQ))
    stride_samples = int(round(stride_seconds * OUTPUT_SFREQ))
    guard_samples = int(round(guard_seconds * OUTPUT_SFREQ))
    if stride_samples <= 0 or max_windows <= 0:
        raise ValueError("Stride and maximum windows must be positive.")
    starts = np.arange(
        guard_samples,
        samples - guard_samples - window_samples + 1,
        stride_samples,
        dtype=np.int64,
    )
    if len(starts) > max_windows:
        selected = np.linspace(0, len(starts) - 1, max_windows, dtype=np.int64)
        starts = starts[np.unique(selected)]
    return starts


def build_way_eeg_ssl_windows(
    root: str | Path,
    *,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    guard_seconds: float = DEFAULT_GUARD_SECONDS,
    max_windows_per_series: int = DEFAULT_MAX_WINDOWS_PER_SERIES,
) -> tuple[WAYSSLWindowDataset, dict[str, object]]:
    """Build fixed-grid EEG windows without reading peripheral or event fields."""
    dataset_root = resolve_way_eeg_gal_root(root)
    discovered = discover_way_eeg_series(dataset_root)
    if not discovered:
        raise RuntimeError("No continuous WAY EEG series were found.")
    mrcp_epochs: list[np.ndarray] = []
    smr_epochs: list[np.ndarray] = []
    participants: list[int] = []
    series_values: list[str] = []
    challenge_values: list[bool] = []
    starts_seconds: list[float] = []
    series_qc: list[dict[str, object]] = []
    rejected_amplitude = 0

    for entry in discovered:
        try:
            payload = loadmat(
                entry.path,
                squeeze_me=True,
                struct_as_record=False,
                variable_names=("hs",),
            )["hs"]
            eeg = np.asarray(payload.eeg.sig, dtype=np.float64)
            names = _channel_names(payload.eeg.names)
            sfreq = float(payload.eeg.samplingrate)
            mrcp, smr = _eeg_branches(eeg, names, sfreq)
        except (KeyError, OSError, ValueError) as error:
            series_qc.append(
                {
                    "participant": entry.participant,
                    "series": entry.series,
                    "challenge_series": entry.challenge_series,
                    "status": "rejected",
                    "reason": str(error),
                    "path": str(entry.path),
                }
            )
            continue
        starts = _window_starts(
            mrcp.shape[-1],
            stride_seconds=stride_seconds,
            guard_seconds=guard_seconds,
            max_windows=max_windows_per_series,
        )
        accepted = 0
        window_samples = int(round(WINDOW_SECONDS * OUTPUT_SFREQ))
        for start in starts:
            stop = int(start + window_samples)
            mrcp_epoch = mrcp[:, start:stop]
            smr_epoch = smr[:, start:stop]
            if (
                max(np.ptp(mrcp_epoch), 0.0) * 1e6 > 1000.0
                or max(np.ptp(smr_epoch), 0.0) * 1e6 > 500.0
            ):
                rejected_amplitude += 1
                continue
            mrcp_epochs.append(mrcp_epoch.astype(np.float32))
            smr_epochs.append(smr_epoch.astype(np.float32))
            participants.append(entry.participant)
            series_values.append(entry.series)
            challenge_values.append(entry.challenge_series)
            starts_seconds.append(float(start / OUTPUT_SFREQ))
            accepted += 1
        series_qc.append(
            {
                "participant": entry.participant,
                "series": entry.series,
                "challenge_series": entry.challenge_series,
                "status": "accepted" if accepted else "no_usable_windows",
                "candidate_windows": int(len(starts)),
                "accepted_windows": accepted,
                "path": str(entry.path),
            }
        )
    if not mrcp_epochs:
        raise RuntimeError("No unlabeled WAY EEG windows passed quality control.")
    dataset = WAYSSLWindowDataset(
        mrcp_epochs=np.stack(mrcp_epochs),
        smr_epochs=np.stack(smr_epochs),
        participants=np.asarray(participants, dtype=np.int16),
        series=np.asarray(series_values),
        challenge_series=np.asarray(challenge_values, dtype=bool),
        window_starts_seconds=np.asarray(starts_seconds, dtype=np.float32),
        channel_names=TRANSFER_CHANNELS,
        sfreq=OUTPUT_SFREQ,
    )
    dataset.validate()
    accepted_series = [entry for entry in series_qc if entry["status"] == "accepted"]
    qc = {
        "dataset_root": str(dataset_root),
        "eeg_only_contract": {
            "emg_read": False,
            "kinematics_read": False,
            "force_read": False,
            "event_labels_read": False,
        },
        "configuration": {
            "window_seconds": WINDOW_SECONDS,
            "stride_seconds": stride_seconds,
            "guard_seconds": guard_seconds,
            "max_windows_per_series": max_windows_per_series,
            "output_sampling_hz": OUTPUT_SFREQ,
            "channels": list(TRANSFER_CHANNELS),
        },
        "discovered_series": len(discovered),
        "accepted_series": len(accepted_series),
        "challenge_series": int(sum(entry.challenge_series for entry in discovered)),
        "participants": int(len(np.unique(dataset.participants))),
        "windows": len(dataset.mrcp_epochs),
        "challenge_windows": int(np.sum(dataset.challenge_series)),
        "rejected_amplitude_windows": rejected_amplitude,
        "series": series_qc,
    }
    return dataset, qc


def save_way_ssl_cache(
    path: str | Path,
    dataset: WAYSSLWindowDataset,
    qc: dict[str, object],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        mrcp_epochs=dataset.mrcp_epochs,
        smr_epochs=dataset.smr_epochs,
        participants=dataset.participants,
        series=dataset.series,
        challenge_series=dataset.challenge_series,
        window_starts_seconds=dataset.window_starts_seconds,
        channel_names=np.asarray(dataset.channel_names),
        sfreq=np.asarray(dataset.sfreq),
    )
    output.with_name(f"{output.stem}.qc.json").write_text(
        json.dumps(qc, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_way_ssl_cache(path: str | Path) -> WAYSSLWindowDataset:
    with np.load(path, allow_pickle=False) as cached:
        dataset = WAYSSLWindowDataset(
            mrcp_epochs=cached["mrcp_epochs"],
            smr_epochs=cached["smr_epochs"],
            participants=cached["participants"],
            series=cached["series"],
            challenge_series=cached["challenge_series"],
            window_starts_seconds=cached["window_starts_seconds"],
            channel_names=tuple(str(value) for value in cached["channel_names"].tolist()),
            sfreq=float(np.asarray(cached["sfreq"]).item()),
        )
    dataset.validate()
    return dataset
