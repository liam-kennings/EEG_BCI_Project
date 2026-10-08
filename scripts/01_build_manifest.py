#!/usr/bin/env python3
"""Inventory EEGMMI recordings and freeze leakage-safe subject partitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bci_pipeline.dataset import (  # noqa: E402
    UNILATERAL_RUNS,
    available_subjects,
    build_recording_manifest,
    parse_int_list,
    resolve_data_dir,
    save_manifest,
)
from bci_pipeline.splits import save_partitions, three_way_subject_split  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--subjects", default=None, help="Optional IDs, e.g. 1-30,42")
    parser.add_argument("--runs", default="4,8,12", help="Unilateral imagery runs")
    parser.add_argument("--target-fraction", type=float, default=0.2)
    parser.add_argument("--controller-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checksums", action="store_true", help="Hash every EDF (slower)")
    parser.add_argument("--output-dir", default="artifacts/dataset")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    requested = parse_int_list(args.subjects)
    runs = parse_int_list(args.runs)
    if not runs or not set(runs).issubset(UNILATERAL_RUNS):
        raise ValueError(f"This stage accepts only unilateral fist runs {UNILATERAL_RUNS}.")
    if not 0.0 < args.target_fraction < 1.0 or not 0.0 < args.controller_fraction < 1.0:
        raise ValueError("Split fractions must be between zero and one.")

    subjects = available_subjects(data_dir, requested)
    if len(subjects) < 5:
        raise ValueError("At least five subjects are required.")
    records = build_recording_manifest(data_dir, subjects, runs, args.checksums)
    partitions = three_way_subject_split(
        subjects,
        target_fraction=args.target_fraction,
        controller_fraction=args.controller_fraction,
        random_seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    save_manifest(records, output_dir / "manifest.json")
    save_partitions(partitions, output_dir / "splits.json")
    print(f"[+] Indexed {len(records)} recordings from {len(subjects)} subjects")
    print(
        f"[+] Split subjects: development={len(partitions.development)}, "
        f"controller-validation={len(partitions.controller_validation)}, "
        f"final-target={len(partitions.final_target)}"
    )
    print(f"[+] Saved manifest: {(output_dir / 'manifest.json').resolve()}")
    print(f"[+] Saved splits: {(output_dir / 'splits.json').resolve()}")


if __name__ == "__main__":
    main()
