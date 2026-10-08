#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR"
source scripts/project_python.sh

export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
export NUMBA_CACHE_DIR="$PROJECT_DIR/.numba_cache"

print "Using Python: $PYTHON_BIN"

"$PYTHON_BIN" -m py_compile \
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

"$PYTHON_BIN" scripts/22_bci_pipeline_gui.py --check
"$PYTHON_BIN" scripts/32_emg_supervised_eeg_transfer.py --check
"$PYTHON_BIN" scripts/40_prepare_prospective_controller.py --verify-only
"$PYTHON_BIN" -m unittest tests.test_bci_pipeline
"$PYTHON_BIN" -m unittest tests.test_v7h
"$PYTHON_BIN" -m unittest tests.test_cho2017
"$PYTHON_BIN" -m unittest tests.test_quality_gate
"$PYTHON_BIN" -m unittest tests.test_participant_metrics

test -d files/eegmmidb/1.0.0
test -d files/external/openbmi/nm000338
test -d "Liu acute-stroke dataset"
test -d "EEG and EMG Dataset for Analyzing Movement-Related/SUBJECTS"
test -f "WAY-EEG-GAL/.way_eeg_gal_ready"
test -f "WAY-EEG-GAL/download_manifest.json"
test -f artifacts/gui_experiments/translational_v3/translational_windows_v3.npz

print "EEG_BCI_Project verification complete."
