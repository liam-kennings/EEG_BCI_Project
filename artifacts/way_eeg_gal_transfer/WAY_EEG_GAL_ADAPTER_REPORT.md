# WAY-EEG-GAL Privileged-Transfer Adapter

## Decoder Role

WAY-EEG-GAL supplies additional **intent-versus-rest** source training. It cannot train the left/right direction head because every recorded action uses the right hand.

## Source Cohort

- Root: `/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project/WAY-EEG-GAL`.
- Accepted series: **108/108** across **12** participants.
- Accepted grasp-and-lift events: **3512**.
- Balanced source rows: **7024** intent/rest examples.
- Cue-to-hand-start latency: median **0.41 s**, 5th-95th percentile **0.28-1.12 s**.
- Rejected event counts: `{'invalid_event_timing': 3, 'window_out_of_bounds': 0, 'eeg_amplitude': 13, 'series_load_or_montage': 0}`.
- The 12 EEG-only challenge series are excluded because they do not contain the peripheral evidence required for privileged supervision.

## Leakage Boundary

- Positive EEG windows are two seconds long and end 100 ms before kinematic hand start.
- Rest EEG windows end 100 ms before the visual cue.
- EMG verification windows begin 50 ms after hand start and are available only to the source-label verifier.
- The trained student and every target evaluation receive EEG features only.
- Some positive windows contain the visual cue. The result therefore measures cue-conditioned motor preparation and must not be presented as asynchronous intention decoding.

## Montage Harmonization

The target contract uses `['FC3', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CP4']`. WAY directly supplies C3, Cz, and C4; missing mid-lateral sites are reconstructed as `['FC3=mean(FC1, FC5)', 'FC4=mean(FC2, FC6)', 'CP3=mean(CP1, CP5)', 'CP4=mean(CP2, CP6)']` after full-montage common-average reference.

## Shared Target Contract

- Target rows: **36903** from **187** subjects.
- Shared EEG features: **144**.
- Target datasets: `{'liu2024': 3867, 'openbmi': 23200, 'physionet': 9836}`.
- Locked roles: `{'controller_validation': 6162, 'development': 23188, 'development_observed': 7553}`.

## Outputs

- Source cache: `artifacts/way_eeg_gal_transfer/way_eeg_gal_source_features.npz`.
- Existing target cache: `/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project/artifacts/emg_supervised_transfer/pure_eeg_target_features.npz`.
- Source audit figure: `00_way_eeg_gal_source_qc.png`.
