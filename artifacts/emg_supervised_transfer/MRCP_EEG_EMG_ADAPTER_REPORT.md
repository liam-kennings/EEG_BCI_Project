# MRCP EEG-EMG Adapter Report

## Source Data

- Root: `/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project/EEG and EMG Dataset for Analyzing Movement-Related`.
- EEG files: **195**; EMG files: **200**.
- Paired recordings: **195** across **39** participants.
- Accepted movement events: **1308**.
- Balanced source rows after matched rest extraction: **2616**.
- EMG onset latency: median **0.70 s**, 5th-95th percentile **0.15-1.60 s** after cue.
- Rejected because EMG onset was not confirmed: **474** events.
- Rejected because synchronized windows were out of bounds: **147** events.

Subject 11 contains five EMG files but no EEG files and is excluded. Files with incomplete trigger sequences are retained only for individually paired cue events; missing cues are never imputed.

## Leakage Controls

- EEG positive windows are two seconds long and end at estimated EMG onset.
- EMG confirmation features are taken after onset and are used only in source training.
- Rest windows end before the movement cue.
- Source and target use the same seven channels, MRCP features, and fixed mu/beta features.
- The pure-EEG target cache contains no EMG features.

## Target Cache

- Rows: **36903** from **187** subjects.
- Shared EEG features: **144**.
- Datasets: `{'liu2024': {'rows': 3867, 'subjects': 49}, 'openbmi': {'rows': 23200, 'subjects': 29}, 'physionet': {'rows': 9836, 'subjects': 109}}`.
- Roles: `{'controller_validation': 6162, 'development': 23188, 'development_observed': 7553}`.

## Outputs

- Source: `/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project/artifacts/emg_supervised_transfer/mrcp_eeg_emg_source_features.npz`.
- Target: `/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project/artifacts/emg_supervised_transfer/pure_eeg_target_features.npz`.
- Figure: `00_mrcp_eeg_emg_source_qc.png`.
