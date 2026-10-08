# WAY EEG-Only Self-Supervised Transfer

## Purpose

This controlled experiment tests whether the EEG recordings in WAY-EEG-GAL are useful without EMG, kinematics, force, task events, or WAY movement labels. It avoids treating overt movement as motor imagery and instead uses WAY only to initialize a general EEG representation.

## Method

1. Load the EEG field from all 120 continuous WAY series, including the 12 EEG-only challenge series.
2. Apply common-average reference, harmonize FC3, FC4, C3, Cz, C4, CP3, and CP4, and produce the same 0.1-4 Hz MRCP and 7-35 Hz sensorimotor branches used by the target pipeline.
3. Extract unlabeled two-second windows on a fixed temporal grid. No cue or movement timing is read.
4. Pretrain a compact convolutional encoder to reconstruct randomly masked samples and complete dropped channels.
5. Fine-tune the encoder on PhysioNet, OpenBMI, and Liu development subjects for binary intent-versus-rest detection.
6. Train an architecture-matched random-initialized CNN using the same target rows, sample weights, epochs, classifier initialization, and batch order.
7. Select separate safety thresholds for each target dataset using controller-validation subjects and evaluate development-observed subjects once.

## Source Audit

- Continuous EEG series accepted: **120/120** across **12 participants**.
- Unlabeled two-second windows: **16,960**.
- EEG-only challenge windows: **1,534**.
- High-amplitude windows rejected: **73**.
- Peripheral signals read: **none**.
- WAY event labels read: **none**.

## Development-Observed Results

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Statistical target-only | 52.82% | 10.58% | 95.07% | 0.683 | 0.645 |
| Raw CNN, random initialization | 53.06% | 14.81% | 91.30% | 0.614 | 0.590 |
| **WAY EEG-only self-supervised** | **56.13%** | **18.38%** | **93.88%** | **0.683** | **0.664** |
| WAY peripheral-supervised | 52.92% | 10.63% | 95.20% | 0.697 | 0.656 |

WAY EEG-only initialization improved balanced accuracy **+3.07 points** over the matched random CNN and **+3.31 points** over the statistical target-only baseline. Relative to the statistical baseline, sensitivity improved **+7.80 points** while specificity decreased **1.19 points**.

Across **38 unseen participants**, the equal-participant SSL change versus the random CNN was **+3.82 balanced-accuracy points** with a 95% participant-bootstrap interval of **+2.25 to +5.37 points**. **78.9%** of participants improved.

## Dataset Results

| Dataset | Statistical BA | Random CNN BA | WAY SSL BA | Peripheral-supervised BA |
|---|---:|---:|---:|---:|
| Liu acute stroke | 54.85% | 49.98% | 53.78% | 55.25% |
| OpenBMI | 52.25% | 54.06% | **56.52%** | 51.81% |
| PhysioNet | 53.36% | 51.83% | **56.12%** | 54.64% |

The result supports EEG-only self-supervised initialization for the healthy imagery datasets. It does not yet outperform the established alternatives on Liu stroke, where clinical domain mismatch remains substantial.

## Interpretation Limits

- Five deterministic neural seeds now confirm that the improvement over the architecture-matched random CNN is stable to training randomness. This does not provide an independent patient replication.
- `development_observed` is not a still-locked external clinical cohort.
- Target trials remain cue-locked; the result is not asynchronous intent detection.
- The experiment evaluates binary intent versus rest and does not modify left/right direction decoding.
- Continuous no-control recordings remain necessary for false activations per minute.

## Run And Outputs

Run `EEG_BCI_Project_RUN_WAY_EEG_SSL.command`, or choose `WAY EEG-only self-supervised transfer` in the GUI. The expensive unlabeled cache is reused unless rebuilding is requested.

For the controlled five-seed confirmation, run `EEG_BCI_Project_RUN_WAY_EEG_SSL_SEEDS.command` or choose `WAY EEG-only five-seed stability` in the GUI.

Outputs are stored in `artifacts/way_eeg_only_ssl/`:

- `way_eeg_ssl_intent_decoder.pt`: EEG-only fine-tuned decoder.
- `way_eeg_ssl_pretrained_backbone.pt`: unlabeled WAY encoder.
- `way_eeg_only_ssl_metrics.json`: complete metrics and split contract.
- `WAY_EEG_ONLY_SSL_TRANSFER_REPORT.md`: generated concise report.
- Three presentation figures covering training, overall performance, domains, and participant effects.

## Five-Seed Stability Confirmation

Seeds 42-46 changed only neural initialization, masked corruption, dropout, and batch randomness. The WAY windows, target participants and roles, architecture, training schedule, weighting, and controller-derived thresholds remained fixed.

| Seed | Random CNN BA | WAY SSL BA | Paired change |
|---:|---:|---:|---:|
| 42 | 53.06% | 56.13% | +3.07 points |
| 43 | 53.43% | 56.68% | +3.25 points |
| 44 | 54.25% | 56.02% | +1.77 points |
| 45 | 53.92% | 56.05% | +2.13 points |
| 46 | 52.78% | 56.59% | +3.81 points |

The five-seed SSL mean was **56.29% balanced accuracy**, **19.59% sensitivity**, **93.00% specificity**, **0.683 AUROC**, and **0.662 AUPRC**. The paired improvement over the random CNN was **+2.81 balanced-accuracy points** (SD **0.84**; 95% seed-level t interval **+1.77 to +3.85**), and all five seeds were positive.

After averaging each of the 38 unseen participants across seeds, the equal-participant effect was **+2.60 points** with a participant-bootstrap 95% CI of **+1.48 to +3.79**; **78.9%** of participants improved. Mean domain changes were **+2.91 points** for OpenBMI and **+3.64 points** for PhysioNet, both positive in all five seeds. Liu stroke remained uncertain at **+0.16 points**, positive in three of five seeds, with a seed-level interval spanning zero.

Compared with the fixed statistical target-only baseline, SSL improved balanced accuracy by **+3.47 points** and sensitivity by **+9.01 points**, while specificity decreased by **2.07 points**. Therefore the repeated-seed result supports representation transfer for healthy imagery data, but does not resolve the stroke-domain gap or the need for continuous false-activation testing.

The full JSON, report, individual seed models, and four presentation figures are in `artifacts/way_eeg_only_ssl_seed_stability/`. The same output folder includes `ORIGINAL_EEG_ONLY_FIVE_SEED_REPORT.md`, which records the randomly initialized paired control whose gradients and labels use only PhysioNet, OpenBMI, and Liu. Its branch-normalization scales were shared with the unlabeled WAY condition, so it is not a strict target-only preprocessing baseline. That paired control reached **53.49% mean balanced accuracy**, **15.60% sensitivity**, and **91.37% specificity**. The later strict target-development-only Conv1D bakeoff baseline reached **60.63% balanced accuracy**, demonstrating that source-derived scaling materially confounded the earlier absolute comparison.
