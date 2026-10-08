"""EEGMMI discovery, annotation semantics, and dataset validation helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from . import PROJECT_ROOT


LABEL_NAMES = {0: "rest", 1: "left_fist", 2: "right_fist"}
UNILATERAL_LABEL_NAMES = LABEL_NAMES
BILATERAL_LABEL_NAMES = {0: "rest", 1: "both_fists", 2: "both_feet"}

# Backward-compatible names for the unilateral hand experiments.
EXECUTION_RUNS = (3, 7, 11)
IMAGERY_RUNS = (4, 8, 12)
UNILATERAL_EXECUTION_RUNS = EXECUTION_RUNS
UNILATERAL_IMAGERY_RUNS = IMAGERY_RUNS
UNILATERAL_RUNS = EXECUTION_RUNS + IMAGERY_RUNS

BILATERAL_EXECUTION_RUNS = (5, 9, 13)
BILATERAL_IMAGERY_RUNS = (6, 10, 14)
BILATERAL_RUNS = BILATERAL_EXECUTION_RUNS + BILATERAL_IMAGERY_RUNS
ALL_TASK_RUNS = tuple(range(3, 15))


@dataclass(frozen=True)
class RecordingRecord:
    subject: int
    run: int
    path: str
    size_bytes: int
    sha256: Optional[str] = None


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated integers and inclusive ranges such as ``1-3,8``."""
    if not text:
        return None

    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def resolve_data_dir(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    fallbacks = (
        candidate,
        PROJECT_ROOT / candidate,
        PROJECT_ROOT / "files" / "eegmmidb" / "1.0.0",
    )
    for path in fallbacks:
        if path.exists() and any(path.glob("S[0-9][0-9][0-9]")):
            return path.resolve()
    raise FileNotFoundError("Could not find the EEGMMI folder containing S001 ... S109.")


def normalize_channel_name(name: str) -> str:
    return name.upper().replace(".", "").replace(" ", "")


def available_subjects(
    data_dir: Path,
    requested_subjects: Optional[Sequence[int]] = None,
    max_subjects: Optional[int] = None,
) -> List[int]:
    discovered = sorted(
        int(path.name[1:])
        for path in data_dir.glob("S[0-9][0-9][0-9]")
        if path.is_dir()
    )
    if requested_subjects:
        requested = set(requested_subjects)
        discovered = [subject for subject in discovered if subject in requested]
    if max_subjects is not None:
        discovered = discovered[:max_subjects]
    return discovered


def find_run_file(data_dir: Path, subject: int, run_id: int) -> Optional[Path]:
    subject_name = f"S{subject:03d}"
    file_name = f"{subject_name}R{run_id:02d}.edf"
    subject_dir = data_dir / subject_name
    candidates = (subject_dir / file_name, subject_dir / subject_name / file_name)
    for path in candidates:
        if path.exists():
            return path
    return next(subject_dir.rglob(file_name), None) if subject_dir.exists() else None


def map_task_annotation(description: str) -> Optional[int]:
    """Map an EEGMMIDB event code; its class name must be resolved from the run family."""
    return {"T0": 0, "T1": 1, "T2": 2}.get(description.strip().upper())


# Backward-compatible names used by the unilateral and imagery-only experiments.
map_unilateral_annotation = map_task_annotation
map_imagery_annotation = map_task_annotation


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_recording_manifest(
    data_dir: Path,
    subjects: Sequence[int],
    runs: Iterable[int] = IMAGERY_RUNS,
    include_checksums: bool = False,
) -> List[RecordingRecord]:
    records: List[RecordingRecord] = []
    missing: List[str] = []
    for subject in subjects:
        for run in runs:
            path = find_run_file(data_dir, subject, run)
            if path is None:
                missing.append(f"S{subject:03d}R{run:02d}")
                continue
            records.append(
                RecordingRecord(
                    subject=subject,
                    run=run,
                    path=str(path.resolve()),
                    size_bytes=path.stat().st_size,
                    sha256=file_sha256(path) if include_checksums else None,
                )
            )
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required recordings: {', '.join(missing[:8])}")
    return records


def save_manifest(records: Sequence[RecordingRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "recordings": [asdict(record) for record in records]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
