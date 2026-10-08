#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"

python3 -m py_compile \
  bci_pipeline/*.py \
  bci_pipeline/models/*.py \
  bci_pipeline/representations/*.py \
  scripts/22_bci_pipeline_gui.py \
  scripts/30_translational_motor_intent_v3.py \
  scripts/31_multimodal_readiness_audit.py \
  scripts/32_emg_supervised_eeg_transfer.py \
  scripts/33_prepare_mrcp_eeg_emg_transfer.py \
  scripts/34_download_way_eeg_gal.py \
  scripts/40_prepare_prospective_controller.py \
  scripts/41_mrcp_phase_ablation.py \
  scripts/50_download_mind.py \
  scripts/51_v7h_mind_benchmark.py

python3 scripts/22_bci_pipeline_gui.py --check
python3 scripts/32_emg_supervised_eeg_transfer.py --check
python3 scripts/40_prepare_prospective_controller.py --verify-only
python3 -m unittest tests.test_bci_pipeline
python3 -m unittest tests.test_v7h

test -d files/eegmmidb/1.0.0
test -d files/external/openbmi/nm000338
test -d "Liu acute-stroke dataset"
test -d "EEG and EMG Dataset for Analyzing Movement-Related/SUBJECTS"
test -f "WAY-EEG-GAL/.way_eeg_gal_ready"
test -f "WAY-EEG-GAL/download_manifest.json"
test -f artifacts/gui_experiments/translational_v3/translational_windows_v3.npz

print "EEG_BCI_Project verification complete."
