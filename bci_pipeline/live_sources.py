"""Optional live EEG sources plus a dependency-free file replay source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mne
import numpy as np


@dataclass(frozen=True)
class LiveChunk:
    data: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class StreamDescriptor:
    name: str
    stream_type: str
    channels: int
    sampling_hz: float
    source_id: str


class FileReplaySource:
    """Replay an MNE recording in deterministic chunks without sleeping."""

    def __init__(
        self,
        raw: mne.io.BaseRaw,
        *,
        chunk_seconds: float = 0.25,
        loop: bool = False,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive.")
        self._raw = raw.copy().pick("eeg").load_data()
        self.sampling_hz = float(self._raw.info["sfreq"])
        self.channel_names = tuple(self._raw.ch_names)
        self.default_chunk_samples = max(1, int(round(chunk_seconds * self.sampling_hz)))
        self.loop = loop
        self._cursor = 0
        self._closed = False

    @property
    def finished(self) -> bool:
        return self._closed or (not self.loop and self._cursor >= self._raw.n_times)

    def pull_chunk(
        self,
        max_samples: int | None = None,
        timeout: float = 0.0,
    ) -> LiveChunk:
        del timeout
        if self._closed:
            return LiveChunk(
                np.empty((len(self.channel_names), 0)),
                np.empty(0),
            )
        count = max_samples or self.default_chunk_samples
        if count <= 0:
            raise ValueError("max_samples must be positive.")
        if self._cursor >= self._raw.n_times:
            if not self.loop:
                return LiveChunk(
                    np.empty((len(self.channel_names), 0)),
                    np.empty(0),
                )
            self._cursor = 0
        stop = min(self._cursor + count, self._raw.n_times)
        data = self._raw.get_data(start=self._cursor, stop=stop)
        timestamps = np.arange(self._cursor, stop, dtype=float) / self.sampling_hz
        self._cursor = stop
        return LiveChunk(data, timestamps)

    def close(self) -> None:
        self._closed = True


def discover_lsl_streams(timeout: float = 2.0) -> tuple[StreamDescriptor, ...]:
    """Discover LSL streams while keeping pylsl an optional dependency."""
    try:
        import pylsl
    except ImportError as error:
        raise RuntimeError(
            "LSL support requires the optional 'pylsl' package."
        ) from error
    streams = pylsl.resolve_streams(wait_time=float(timeout))
    return tuple(
        StreamDescriptor(
            name=stream.name(),
            stream_type=stream.type(),
            channels=int(stream.channel_count()),
            sampling_hz=float(stream.nominal_srate()),
            source_id=stream.source_id(),
        )
        for stream in streams
    )


def _lsl_channel_names(info: Any) -> tuple[str, ...]:
    count = int(info.channel_count())
    names: list[str] = []
    channel = info.desc().child("channels").child("channel")
    for index in range(count):
        label = channel.child_value("label") if channel else ""
        names.append(label or f"EEG{index + 1:03d}")
        channel = channel.next_sibling() if channel else channel
    return tuple(names)


class LSLSource:
    """Receive a selected Lab Streaming Layer EEG stream."""

    def __init__(self, inlet: Any, info: Any) -> None:
        self._inlet = inlet
        self._info = info
        self.sampling_hz = float(info.nominal_srate())
        self.channel_names = _lsl_channel_names(info)
        self._closed = False

    @classmethod
    def connect(
        cls,
        *,
        stream_name: str = "",
        stream_type: str = "EEG",
        timeout: float = 3.0,
    ) -> "LSLSource":
        try:
            import pylsl
        except ImportError as error:
            raise RuntimeError(
                "LSL support requires the optional 'pylsl' package."
            ) from error
        streams = pylsl.resolve_streams(wait_time=float(timeout))
        candidates = [
            stream
            for stream in streams
            if (not stream_type or stream.type().lower() == stream_type.lower())
            and (not stream_name or stream.name() == stream_name)
        ]
        if not candidates:
            requested = stream_name or stream_type or "any stream"
            raise RuntimeError(f"No matching LSL stream found for {requested}.")
        info = candidates[0]
        inlet = pylsl.StreamInlet(info, max_buflen=60)
        inlet.open_stream(timeout=float(timeout))
        return cls(inlet, info)

    def pull_chunk(
        self,
        max_samples: int | None = None,
        timeout: float = 0.0,
    ) -> LiveChunk:
        if self._closed:
            return LiveChunk(
                np.empty((len(self.channel_names), 0)),
                np.empty(0),
            )
        samples, timestamps = self._inlet.pull_chunk(
            timeout=float(timeout),
            max_samples=max_samples or max(1, int(self.sampling_hz)),
        )
        if not samples:
            return LiveChunk(
                np.empty((len(self.channel_names), 0)),
                np.empty(0),
            )
        return LiveChunk(
            np.asarray(samples, dtype=np.float64).T,
            np.asarray(timestamps, dtype=np.float64),
        )

    def close(self) -> None:
        if not self._closed:
            self._inlet.close_stream()
            self._closed = True


class BrainFlowSource:
    """Read board-agnostic EEG chunks through the optional BrainFlow SDK."""

    def __init__(
        self,
        board: Any,
        eeg_rows: Sequence[int],
        sampling_hz: float,
    ) -> None:
        self._board = board
        self._eeg_rows = np.asarray(eeg_rows, dtype=int)
        self.sampling_hz = float(sampling_hz)
        self.channel_names = tuple(
            f"EEG{index + 1:03d}" for index in range(len(self._eeg_rows))
        )
        self._closed = False
        self._sample_counter = 0

    @classmethod
    def connect(
        cls,
        board_id: int,
        *,
        serial_port: str = "",
        mac_address: str = "",
        ip_address: str = "",
        serial_number: str = "",
        timeout: int = 10,
    ) -> "BrainFlowSource":
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams
        except ImportError as error:
            raise RuntimeError(
                "BrainFlow support requires the optional 'brainflow' package."
            ) from error
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        params.ip_address = ip_address
        params.serial_number = serial_number
        params.timeout = int(timeout)
        board = BoardShim(int(board_id), params)
        try:
            board.prepare_session()
            board.start_stream()
            rows = BoardShim.get_eeg_channels(int(board_id))
            sfreq = BoardShim.get_sampling_rate(int(board_id))
        except Exception:
            if board.is_prepared():
                board.release_session()
            raise
        return cls(board, rows, sfreq)

    def pull_chunk(
        self,
        max_samples: int | None = None,
        timeout: float = 0.0,
    ) -> LiveChunk:
        del timeout
        if self._closed:
            return LiveChunk(
                np.empty((len(self.channel_names), 0)),
                np.empty(0),
            )
        count = max_samples or max(1, int(round(self.sampling_hz * 0.25)))
        board_data = self._board.get_current_board_data(int(count))
        if board_data.size == 0:
            return LiveChunk(
                np.empty((len(self.channel_names), 0)),
                np.empty(0),
            )
        data = np.asarray(board_data[self._eeg_rows], dtype=np.float64) * 1e-6
        timestamps = (
            np.arange(self._sample_counter, self._sample_counter + data.shape[1])
            / self.sampling_hz
        )
        self._sample_counter += data.shape[1]
        return LiveChunk(data, timestamps)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._board.stop_stream()
        finally:
            self._board.release_session()
            self._closed = True


def raw_from_live_chunks(
    chunks: Sequence[np.ndarray],
    sampling_hz: float,
    channel_names: Sequence[str],
) -> mne.io.RawArray:
    if not chunks:
        raise ValueError("No live samples were captured.")
    data = np.concatenate([np.asarray(chunk) for chunk in chunks], axis=1)
    if data.shape[0] != len(channel_names):
        raise ValueError("Live data channel count does not match channel names.")
    info = mne.create_info(list(channel_names), float(sampling_hz), ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)
