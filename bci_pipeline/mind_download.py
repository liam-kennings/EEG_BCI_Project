"""ScienceDB manifest handling for the MIND EEG-fNIRS dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse


MIND_DATASET_ID = "d0f67f38c5bc4416bdcce619f33e57da"
MIND_DOI = "10.57760/sciencedb.34326"
MIND_VERSION = "V3"
MIND_LICENSE = "CC BY-NC 4.0"
MIND_RELEASE_BYTES = 100_010_786_552
MIND_MANIFEST_URL = (
    "https://www.scidb.cn/api/sdb-filetree-service/getAllUrl"
    f"?dataSetId={MIND_DATASET_ID}&version={MIND_VERSION}&global=New_York"
)


@dataclass(frozen=True)
class MINDDownloadRecord:
    """One stable public file record returned by ScienceDB."""

    file_id: str
    relative_path: str
    download_url: str

    @property
    def subject(self) -> int | None:
        for part in PurePosixPath(self.relative_path).parts:
            if part.startswith("sub-") and part[4:].isdigit():
                return int(part[4:])
        return None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_sciencedb_manifest(text: str) -> tuple[MINDDownloadRecord, ...]:
    """Parse and validate ScienceDB's newline-delimited public URL manifest."""
    records: list[MINDDownloadRecord] = []
    seen_paths: set[str] = set()
    prefix = f"/{MIND_VERSION}/MIND_BIDS/"
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parsed = urlparse(line)
        query = parse_qs(parsed.query)
        if "fileId" not in query or "path" not in query:
            raise ValueError(f"Malformed MIND manifest line {line_number}.")
        file_id = str(query["fileId"][0])
        source_path = unquote(str(query["path"][0]))
        if not source_path.startswith(prefix):
            raise ValueError(
                f"Unexpected MIND release path on line {line_number}: {source_path}"
            )
        relative = source_path[len(prefix) :].lstrip("/")
        path = PurePosixPath(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe MIND path on line {line_number}: {relative}")
        if relative in seen_paths:
            raise ValueError(f"Duplicate MIND release path: {relative}")
        seen_paths.add(relative)
        records.append(
            MINDDownloadRecord(
                file_id=file_id,
                relative_path=relative,
                download_url=line,
            )
        )
    if not records:
        raise RuntimeError("ScienceDB returned an empty MIND manifest.")
    return tuple(sorted(records, key=lambda item: item.relative_path))


def select_mind_records(
    records: Iterable[MINDDownloadRecord],
    *,
    profile: str,
    subjects: set[int] | None = None,
) -> tuple[MINDDownloadRecord, ...]:
    """Select a documented download profile without losing root metadata."""
    normalized = profile.strip().lower()
    if normalized not in {"derivatives", "analysis", "full"}:
        raise ValueError("MIND profile must be derivatives, analysis, or full.")

    selected: list[MINDDownloadRecord] = []
    for record in records:
        path = PurePosixPath(record.relative_path)
        subject = record.subject
        is_root_metadata = len(path.parts) == 1
        is_derivative = record.relative_path.startswith("derivatives/preproc-v1/")
        is_source = record.relative_path.startswith("sourcedata/")
        if normalized == "derivatives":
            include = is_root_metadata or is_derivative
        elif normalized == "analysis":
            include = not is_source
        else:
            include = True
        if not include:
            continue
        if subjects is not None and subject is not None and subject not in subjects:
            continue
        selected.append(record)
    if not selected:
        raise RuntimeError("The requested MIND profile selected no files.")
    return tuple(selected)


def local_path(output_root: Path, record: MINDDownloadRecord) -> Path:
    """Return a safe local destination below MIND_BIDS."""
    root = (output_root / "MIND_BIDS").resolve()
    destination = (root / record.relative_path).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Unsafe MIND destination: {destination}")
    return destination
