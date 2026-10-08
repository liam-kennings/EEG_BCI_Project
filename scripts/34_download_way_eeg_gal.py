#!/usr/bin/env python3
"""Download and optionally extract the official WAY-EEG-GAL release."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ID = 988376
COLLECTION_URL = (
    "https://springernature.figshare.com/collections/"
    "WAY_EEG_GAL_Multi_channel_EEG_Recordings_During_3_936_"
    "Grasp_and_Lift_Trials_with_Varying_Weight_and_Friction/988376/2"
)
API_ROOT = "https://api.figshare.com/v2"
PARTICIPANT_PATTERN = re.compile(r"Participant\s+(\d+)", re.IGNORECASE)


def _json(url: str) -> object:
    result = subprocess.run(
        ("curl", "-fsSL", "--retry", "5", url),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (output / member.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP member in {archive_path}: {member.filename}")
        archive.extractall(output)


def _participant_records(selected: set[int] | None) -> list[dict[str, object]]:
    articles = _json(
        f"{API_ROOT}/collections/{COLLECTION_ID}/articles?page_size=100"
    )
    if not isinstance(articles, list):
        raise RuntimeError("Figshare returned an unexpected collection response.")
    records = []
    for article in articles:
        match = PARTICIPANT_PATTERN.search(str(article.get("title", "")))
        if match is None:
            continue
        participant = int(match.group(1))
        if selected is not None and participant not in selected:
            continue
        details = _json(f"{API_ROOT}/articles/{int(article['id'])}")
        if not isinstance(details, dict) or len(details.get("files", [])) != 1:
            raise RuntimeError(f"Unexpected file metadata for participant {participant}.")
        file_info = details["files"][0]
        records.append(
            {
                "participant": participant,
                "article_id": int(article["id"]),
                "article_url": str(article["url_public_html"]),
                "file_id": int(file_info["id"]),
                "filename": str(file_info["name"]),
                "size_bytes": int(file_info["size"]),
                "md5": str(file_info["supplied_md5"]),
                "download_url": str(file_info["download_url"]),
            }
        )
    records.sort(key=lambda row: int(row["participant"]))
    if not records:
        raise RuntimeError("No WAY-EEG-GAL participant records were found.")
    return records


def _prepare_participant(
    record: dict[str, object],
    archives: Path,
    extracted: Path,
    extract: bool,
    delete_archives: bool,
) -> int:
    participant = int(record["participant"])
    archive_path = archives / str(record["filename"])
    expected_size = int(record["size_bytes"])
    expected_md5 = str(record["md5"])
    valid = (
        archive_path.is_file()
        and archive_path.stat().st_size == expected_size
        and _md5(archive_path) == expected_md5
    )
    if not valid:
        if archive_path.is_file() and archive_path.stat().st_size >= expected_size:
            archive_path.unlink()
        print(
            f"[+] Downloading P{participant} ({expected_size / 1024**3:.2f} GiB)...",
            flush=True,
        )
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
                str(archive_path),
                str(record["download_url"]),
            ),
            check=True,
        )
        if archive_path.stat().st_size != expected_size:
            raise RuntimeError(f"Size verification failed for {archive_path.name}.")
        if _md5(archive_path) != expected_md5:
            raise RuntimeError(f"MD5 verification failed for {archive_path.name}.")
        print(f"[+] Verified {archive_path.name}.", flush=True)
    else:
        print(f"[+] Reusing verified {archive_path.name}.", flush=True)

    if extract:
        participant_output = extracted / f"P{participant:02d}"
        marker = participant_output / ".way_eeg_gal_extracted"
        if not marker.is_file() or marker.read_text(encoding="ascii").strip() != expected_md5:
            print(f"[+] Extracting {archive_path.name}...", flush=True)
            _safe_extract(archive_path, participant_output)
            marker.write_text(expected_md5 + "\n", encoding="ascii")
        if delete_archives:
            archive_path.unlink()
    return participant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "WAY-EEG-GAL",
    )
    parser.add_argument(
        "--participants",
        help="Optional comma-separated participant numbers; default is all 12.",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--delete-archives", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.delete_archives and not args.extract:
        parser.error("--delete-archives requires --extract.")
    if args.workers < 1 or args.workers > 6:
        parser.error("--workers must be between 1 and 6.")
    selected = None
    if args.participants:
        try:
            selected = {int(value.strip()) for value in args.participants.split(",")}
        except ValueError as error:
            parser.error(f"Invalid --participants value: {error}")

    output = args.output_dir.expanduser().resolve()
    archives = output / "archives"
    extracted = output / "participants"
    output.mkdir(parents=True, exist_ok=True)
    records = _participant_records(selected)
    manifest = {
        "schema_version": 1,
        "dataset": "WAY-EEG-GAL",
        "collection_id": COLLECTION_ID,
        "collection_url": COLLECTION_URL,
        "article_url": "https://www.nature.com/articles/sdata201447",
        "participants": records,
        "total_download_bytes": sum(int(row["size_bytes"]) for row in records),
    }
    manifest_path = output / "download_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("download_complete", "archives_verified", "extracted"):
            if key in previous:
                manifest[key] = previous[key]
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    total_gib = manifest["total_download_bytes"] / 1024**3
    print(f"[+] Official Figshare manifest: {len(records)} participant files, {total_gib:.2f} GiB")
    if args.metadata_only:
        print(f"[+] Saved metadata: {output / 'download_manifest.json'}")
        return

    archives.mkdir(parents=True, exist_ok=True)
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _prepare_participant,
                record,
                archives,
                extracted,
                args.extract,
                args.delete_archives,
            )
            for record in records
        ]
        for future in concurrent.futures.as_completed(futures):
            participant = future.result()
            completed.append(participant)
            print(f"[+] Participant {participant} ready.", flush=True)

    manifest["download_complete"] = len(completed) == len(records)
    manifest["archives_verified"] = sorted(completed)
    manifest["extracted"] = bool(args.extract)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / ".way_eeg_gal_ready").write_text(
        "Official Figshare archives verified"
        + (" and extracted" if args.extract else "")
        + ".\n",
        encoding="ascii",
    )
    print(f"[+] WAY-EEG-GAL ready: {output}")


if __name__ == "__main__":
    main()
