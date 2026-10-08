"""Subject-isolated split creation and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SubjectPartitions:
    development: Tuple[int, ...]
    controller_validation: Tuple[int, ...]
    final_target: Tuple[int, ...]
    random_seed: int
    schema_version: int = 1

    def validate(self) -> None:
        groups = [set(self.development), set(self.controller_validation), set(self.final_target)]
        if any(not group for group in groups):
            raise ValueError("Every subject partition must be non-empty.")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Subject partitions overlap.")


def source_target_split(
    labels: np.ndarray,
    subjects: np.ndarray,
    test_fraction: float,
    random_seed: int,
    explicit_targets: Optional[Sequence[int]] = None,
) -> Tuple[list[int], list[int]]:
    all_subjects = sorted(np.unique(subjects).astype(int).tolist())
    if explicit_targets:
        targets = sorted(set(explicit_targets) & set(all_subjects))
        sources = sorted(set(all_subjects) - set(targets))
        if not targets or not sources:
            raise ValueError("Explicit target subjects must leave non-empty source and target groups.")
        return sources, targets

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_seed)
    source_idx, target_idx = next(splitter.split(np.zeros(len(labels)), labels, groups=subjects))
    return (
        sorted(np.unique(subjects[source_idx]).astype(int).tolist()),
        sorted(np.unique(subjects[target_idx]).astype(int).tolist()),
    )


def three_way_subject_split(
    subjects: Sequence[int],
    target_fraction: float = 0.2,
    controller_fraction: float = 0.2,
    random_seed: int = 42,
) -> SubjectPartitions:
    unique = np.asarray(sorted(set(int(subject) for subject in subjects)), dtype=np.int64)
    if len(unique) < 5:
        raise ValueError("At least five subjects are required for three non-empty partitions.")
    first = GroupShuffleSplit(n_splits=1, test_size=target_fraction, random_state=random_seed)
    source_idx, target_idx = next(first.split(unique, groups=unique))
    source_pool, targets = unique[source_idx], unique[target_idx]
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=controller_fraction,
        random_state=random_seed + 1,
    )
    development_idx, controller_idx = next(second.split(source_pool, groups=source_pool))
    partitions = SubjectPartitions(
        development=tuple(sorted(source_pool[development_idx].tolist())),
        controller_validation=tuple(sorted(source_pool[controller_idx].tolist())),
        final_target=tuple(sorted(targets.tolist())),
        random_seed=random_seed,
    )
    partitions.validate()
    return partitions


def save_partitions(partitions: SubjectPartitions, path: Path) -> None:
    partitions.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(partitions), indent=2) + "\n", encoding="ascii")


def load_partitions(path: Path) -> SubjectPartitions:
    payload = json.loads(path.read_text(encoding="ascii"))
    partitions = SubjectPartitions(
        development=tuple(payload["development"]),
        controller_validation=tuple(payload["controller_validation"]),
        final_target=tuple(payload["final_target"]),
        random_seed=int(payload["random_seed"]),
        schema_version=int(payload.get("schema_version", 1)),
    )
    partitions.validate()
    return partitions

