from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from bci_pipeline.cho2017 import (
    ChoEvent,
    calibration_event_indices,
    cho2017_snapshot,
    discover_cho2017_recordings,
    discover_cho2017_source_records,
    load_cho2017_events,
    resolve_cho2017_root,
)


class Cho2017AdapterTests(unittest.TestCase):
    def _minimal_root(self, base: Path) -> tuple[Path, Path]:
        root = base / "nm000245"
        eeg = root / "sub-1" / "ses-0" / "eeg"
        eeg.mkdir(parents=True)
        (root / "dataset_description.json").write_text("{}\n", encoding="ascii")
        (root / "participants.tsv").write_text("participant_id\nsub-1\n", encoding="ascii")
        sourcedata = root / "sourcedata"
        sourcedata.mkdir()
        (sourcedata / "s01.mat").touch()
        (sourcedata / "sourcedata_provenance.json").write_text(
            json.dumps(
                {
                    "files": [
                        {"file": "s01.mat", "bytes": 0, "sha256": "test"}
                    ]
                }
            )
            + "\n",
            encoding="ascii",
        )
        (eeg / "sub-1_ses-0_task-imagery_run-0_eeg.bdf").touch()
        events = eeg / "sub-1_ses-0_task-imagery_run-0_events.tsv"
        return root, events

    def test_resolve_and_discover_accept_parent_or_bids_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root, events = self._minimal_root(parent)
            events.write_text(
                "onset\tduration\ttrial_type\n"
                "2\t3\tleft_hand\n"
                "9\t3\tright_hand\n",
                encoding="ascii",
            )
            self.assertEqual(resolve_cho2017_root(parent), root.resolve())
            self.assertEqual(resolve_cho2017_root(root), root.resolve())
            recordings = discover_cho2017_recordings(parent, subjects=[1])
            self.assertEqual(len(recordings), 1)
            self.assertEqual(recordings[0].subject, 1)
            source = discover_cho2017_source_records(parent, subjects=[1])
            self.assertEqual(source[0].mat_path.name, "s01.mat")

    def test_event_parser_handles_bom_and_maps_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.tsv"
            path.write_text(
                "\ufeffonset\tduration\ttrial_type\n"
                "2\t3\tleft_hand\n"
                "9\t3\tright_hand\n",
                encoding="utf-8",
            )
            events = load_cho2017_events(path)
            self.assertEqual([event.label for event in events], [1, 2])
            self.assertEqual([event.onset for event in events], [2.0, 9.0])

    def test_calibration_reserves_ten_rows_per_source_class(self) -> None:
        events = tuple(
            ChoEvent(
                onset=float(index * 7),
                duration=3.0,
                label=1 if index < 20 else 2,
                trial_type="left_hand" if index < 20 else "right_hand",
            )
            for index in range(40)
        )
        intent, rest = calibration_event_indices(events)
        self.assertEqual(len(intent), 20)
        self.assertEqual(len(rest), 10)
        self.assertEqual(sum(events[index].label == 1 for index in intent), 10)
        self.assertEqual(sum(events[index].label == 2 for index in intent), 10)

    def test_snapshot_is_deterministic_and_inference_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, events = self._minimal_root(Path(directory))
            events.write_text(
                "onset\tduration\ttrial_type\n"
                "2\t3\tleft_hand\n"
                "9\t3\tright_hand\n",
                encoding="ascii",
            )
            first = cho2017_snapshot(root, subjects=[1])
            second = cho2017_snapshot(root, subjects=[1])
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertEqual(first["event_totals"], {"left_hand": 1, "right_hand": 1})


if __name__ == "__main__":
    unittest.main()
