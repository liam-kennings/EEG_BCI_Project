#!/usr/bin/env python3
"""Create a full-scalp ICA artifact diagnostic for one EEGMMI recording.

The script is deliberately diagnostic-only. It ranks components correlated with
frontal electrodes, but does not automatically remove them without EOG/EMG
reference channels or a reviewed component decision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MNE_HOME", str(PROJECT_ROOT / ".mne"))
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from bci_pipeline.dataset import find_run_file, resolve_data_dir  # noqa: E402


def normalized_name(name: str) -> str:
    return name.upper().replace(".", "").replace(" ", "")


def save_figures(figures: object, path: Path) -> None:
    if isinstance(figures, (list, tuple)):
        for index, figure in enumerate(figures, start=1):
            figure.savefig(path.with_stem(f"{path.stem}_{index}"), dpi=180, bbox_inches="tight")
            plt.close(figure)
    else:
        figures.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="files/eegmmidb/1.0.0")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--run", type=int, default=4)
    parser.add_argument("--components", type=int, default=30)
    parser.add_argument("--output-dir", default="artifacts/ica_diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    path = find_run_file(data_dir, args.subject, args.run)
    if path is None:
        raise FileNotFoundError(f"Could not find S{args.subject:03d}R{args.run:02d}.edf")
    output_dir = Path(args.output_dir) / f"S{args.subject:03d}_R{args.run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False).pick("eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    canonical_names = {normalized_name(name): name for name in montage.ch_names}
    raw.rename_channels(
        {name: canonical_names.get(normalized_name(name), normalized_name(name)) for name in raw.ch_names}
    )
    raw.set_montage(montage, on_missing="ignore")
    raw.filter(1.0, 35.0, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    component_count = min(args.components, len(raw.ch_names) - 1)
    ica = mne.preprocessing.ICA(
        n_components=component_count,
        method="fastica",
        random_state=args.seed,
        max_iter="auto",
    ).fit(raw, decim=3, verbose=False)
    sources = ica.get_sources(raw).get_data()
    frontal_names = [name for name in ("Fp1", "Fp2", "AF7", "AF8") if name in raw.ch_names]
    frontal = raw.get_data(picks=frontal_names).mean(axis=0)
    correlations = np.asarray(
        [np.corrcoef(source, frontal)[0, 1] for source in sources], dtype=np.float64
    )
    ranked = np.argsort(np.abs(correlations))[::-1]

    save_figures(
        ica.plot_components(inst=raw, picks=np.arange(min(component_count, 20)), show=False),
        output_dir / "ica_component_maps.png",
    )
    save_figures(
        ica.plot_sources(raw, picks=ranked[:6], show=False),
        output_dir / "frontal_correlated_component_sources.png",
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(min(component_count, 15)), correlations[ranked[:15]], color="#b65c4a")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(min(component_count, 15)), [f"IC {index:02d}" for index in ranked[:15]], rotation=45)
    ax.set_ylabel("Correlation with frontal EEG surrogate")
    ax.set_title("ICA components ranked for visual artifact review")
    fig.tight_layout()
    fig.savefig(output_dir / "frontal_component_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    report = {
        "recording": str(path),
        "preprocessing": "full scalp, 1-35 Hz, common average reference",
        "frontal_surrogate_channels": frontal_names,
        "automatic_exclusion": False,
        "component_ranking": [
            {"component": int(index), "frontal_correlation": float(correlations[index])}
            for index in ranked.tolist()
        ],
    }
    with (output_dir / "ica_diagnostic.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[+] Saved ICA diagnostic to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
