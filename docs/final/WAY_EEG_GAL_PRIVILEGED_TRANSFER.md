# WAY-EEG-GAL Privileged EEG Transfer

## Purpose

WAY-EEG-GAL is used as a synchronized movement-confirmation source, not as a deployment-time hybrid controller. Kinematics identify hand movement onset, EMG verifies that overt movement occurred, and only pre-movement EEG is passed into the deployable student. The dataset contains right-hand grasp-and-lift execution only, so it augments the intent-versus-rest gate and does not train the left/right direction head.

The source release and protocol are described in the [Scientific Data article](https://www.nature.com/articles/sdata201447) and the [official Figshare collection](https://doi.org/10.6084/m9.figshare.c.988376.v2).

## Training Flow

1. Load each participant's nine continuous paired EEG/EMG series and the `AllLifts` event table. The EEG-only challenge series are excluded from privileged training.
2. Apply full-montage common-average reference. Reconstruct FC3, FC4, CP3, and CP4 from adjacent 10-10 channels, then select FC3, FC4, C3, Cz, C4, CP3, and CP4.
3. Produce a 0.1-4 Hz MRCP branch and a 7-35 Hz sensorimotor-rhythm branch, resampled from 500 Hz to 100 Hz.
4. Extract a two-second positive EEG window ending 100 ms before kinematic hand start. Extract a matched rest window ending 100 ms before the visual cue.
5. Use five-channel, 20-450 Hz EMG from 50-800 ms after hand start as privileged evidence. EMG never enters the target cache or prediction API.
6. Convert both source and target EEG to the same 144 MRCP/mu/beta features. Apply robust source-to-target median/IQR alignment.
7. Cross-fit the EMG verifier by source participant, reject low-confidence source rows, and train an EEG-only logistic student with participant-balanced source weights and dataset/class-balanced target weights.
8. Select source weight and intent threshold on `controller_validation`; report performance once on `development_observed`.

## Local Source Audit

- Continuous paired series accepted: **108/108** across **12** participants.
- Usable movement events: **3,512/3,528**.
- Balanced source rows: **7,024**.
- Rejections: 3 invalid event timings and 13 high-amplitude EEG events.
- Median cue-to-hand-start latency: **0.414 s**; 5th-95th percentile: **0.276-1.116 s**.
- Cross-validated EMG verifier: **74.6% balanced accuracy**, **0.833 AUROC**.
- Rows retained at confidence 0.60: **4,835/7,024 (68.8%)**.
- Controller-selected source weight: **0.35**.

## Pure-EEG Target Result

| Development-observed model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Target-only EEG | 53.02% | 10.74% | 95.31% | 0.683 | 0.645 |
| WAY-assisted EEG student | 52.94% | 10.50% | 95.39% | 0.697 | 0.656 |

WAY improved ranking but did not improve pooled balanced accuracy at the conservative controller-selected operating point. Across 38 paired target participants, the equal-participant mean balanced-accuracy change was **+0.60 points** with a 95% participant-bootstrap interval of **+0.10 to +1.16 points**; 47.4% of participants improved.

| Target dataset | Target-only BA | WAY-assisted BA | Change |
|---|---:|---:|---:|
| Liu acute stroke | 56.07% | 56.95% | +0.88 points |
| PhysioNet | 52.91% | 53.72% | +0.82 points |
| OpenBMI | 52.54% | 51.94% | -0.60 points |

This is a domain-specific transfer signal, not evidence of universal improvement. The pooled thresholded result is essentially unchanged, and `development_observed` is not a still-locked external clinical cohort.

## Run It

Double-click `EEG_BCI_Project_RUN_WAY_EEG_GAL_TRANSFER.command`, or choose `WAY-EEG-GAL privileged EEG transfer` in the GUI's Dataset Experiments tab.

The run writes the standardized source cache, QC report, two presentation figures, metrics, and EEG-only model to `artifacts/way_eeg_gal_transfer/`.

## Interpretation Limits

- Positive epochs are cue-conditioned and can include visual/attention responses; they should not be described as asynchronous intent detections.
- WAY is overt right-hand movement from healthy participants, not weak attempted movement after stroke.
- Four shared channels are spatial estimates rather than measured WAY electrodes.
- EMG confirms movement after the EEG window; it does not prove that every accepted EEG pattern is cortical motor intent.
- Continuous no-control testing is still required before reporting false activations per minute or enabling a prosthetic command.
