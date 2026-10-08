#!/usr/bin/env python3
"""Download the official MIND EEG-fNIRS release from ScienceDB."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_pipeline.mind_download import (
    MIND_DATASET_ID,
    MIND_DOI,
    MIND_LICENSE,
    MIND_MANIFEST_URL,
    MIND_RELEASE_BYTES,
    MIND_VERSION,
    MINDDownloadRecord,
    local_path,
    parse_sciencedb_manifest,
    select_mind_records,
)


def _curl_text(url: str) -> str:
    result = subprocess.run(
        ("curl", "-fsSL", "--retry", "5", url),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_subjects(value: str | None) -> set[int] | None:
    if not value:
        return None
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid subject range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    if any(subject < 1 or subject > 30 for subject in selected):
        raise ValueError("MIND subjects must be between 1 and 30.")
    return selected


def _load_state(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(value) for value in payload.get("completed", [])}


def _write_state(path: Path, completed: set[str]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _download_record(output: Path, record: MINDDownloadRecord) -> str:
    destination = local_path(output, record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            "curl",
            "-fL",
            "--silent",
            "--show-error",
            "--retry",
            "8",
            "--retry-delay",
            "3",
            "--continue-at",
            "-",
            "--output",
            str(destination),
            record.download_url,
        ),
        check=True,
    )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"MIND download is empty: {record.relative_path}")
    return record.relative_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "MIND",
    )
    parser.add_argument(
        "--profile",
        choices=("derivatives", "analysis", "full"),
        default="derivatives",
        help=(
            "derivatives: processed EEG/HbO/HbR only; analysis: standardized "
            "BIDS plus derivatives; full: complete approximately 100 GB release"
        ),
    )
    parser.add_argument(
        "--subjects",
        help="Optional subjects such as 2,3,5-10. Root metadata is always retained.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm acceptance of the public CC BY-NC 4.0 release terms.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8.")
    try:
        subjects = _parse_subjects(args.subjects)
    except ValueError as error:
        parser.error(str(error))
    if not args.accept_license and not args.metadata_only and not args.dry_run:
        parser.error(
            "Downloading requires --accept-license for the CC BY-NC 4.0 release."
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_text = _curl_text(MIND_MANIFEST_URL)
    all_records = parse_sciencedb_manifest(manifest_text)
    records = select_mind_records(
        all_records,
        profile=args.profile,
        subjects=subjects,
    )
    manifest = {
        "schema_version": 1,
        "dataset": "MIND",
        "dataset_id": MIND_DATASET_ID,
        "doi": MIND_DOI,
        "version": MIND_VERSION,
        "license": MIND_LICENSE,
        "official_release_bytes": MIND_RELEASE_BYTES,
        "manifest_url": MIND_MANIFEST_URL,
        "profile": args.profile,
        "subjects": sorted(subjects) if subjects is not None else "all",
        "official_file_count": len(all_records),
        "selected_file_count": len(records),
        "files": [record.as_dict() for record in records],
    }
    manifest_path = output / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[+] MIND {MIND_VERSION}: selected {len(records):,}/{len(all_records):,} "
        f"files using profile={args.profile}."
    )
    print(
        f"[i] Complete release size: {MIND_RELEASE_BYTES / 1024**3:.1f} GiB; "
        "profile size is determined while files download."
    )
    print(f"[+] Saved manifest: {manifest_path}")
    if args.metadata_only or args.dry_run:
        return

    state_path = output / ".mind_download_state.json"
    completed = _load_state(state_path)
    pending = [
        record
        for record in records
        if record.relative_path not in completed
        or not local_path(output, record).is_file()
    ]
    print(
        f"[+] Reusing {len(records) - len(pending):,} completed files; "
        f"downloading {len(pending):,}."
    )
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download_record, output, record): record
            for record in pending
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            record = futures[future]
            try:
                relative_path = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Failed to download {record.relative_path}: {error}"
                ) from error
            with lock:
                completed.add(relative_path)
                _write_state(state_path, completed)
            if index == 1 or index % 25 == 0 or index == len(pending):
                print(f"[+] Downloaded {index:,}/{len(pending):,} pending files.")

    manifest["download_complete"] = all(
        record.relative_path in completed for record in records
    )
    manifest["downloaded_bytes"] = sum(
        local_path(output, record).stat().st_size
        for record in records
        if local_path(output, record).is_file()
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[COMPLETE] MIND profile saved under {output / 'MIND_BIDS'}")


if __name__ == "__main__":
    main()
