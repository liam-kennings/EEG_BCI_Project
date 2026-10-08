# WAY EEG-Only Self-Supervised Transfer

## Contract

- All 120 continuous WAY EEG series are eligible, including 12 EEG-only challenge series.
- WAY EMG, kinematics, force, task events, and class labels are never read by this experiment.
- The encoder learns by reconstructing masked samples and channels in unlabeled two-second EEG windows.
- The pretrained and random-control networks use the same architecture, fine-tuning rows, sample weighting, epochs, and controller threshold procedure.
- Fine-tuning uses original-dataset development subjects; thresholds use controller subjects; development-observed subjects are evaluated once.

## Source Audit

- Unlabeled windows: **16960**.
- Continuous series: **120/120** across **12** participants.
- EEG-only challenge windows: **1534**.

## Development-Observed Result

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Statistical target-only | 52.82% | 10.58% | 95.07% | 0.683 | 0.645 |
| Raw CNN random init | 53.92% | 17.64% | 90.19% | 0.613 | 0.596 |
| WAY EEG-only SSL | 56.05% | 20.18% | 91.91% | 0.671 | 0.654 |
| WAY peripheral-supervised | 52.92% | 10.63% | 95.20% | 0.697 | 0.656 |

Against the architecture-matched random CNN, the equal-participant mean SSL change was **+2.26 points** (95% participant-bootstrap CI **+0.72 to +3.76**); **60.5%** of participants improved.

## Dataset Results

| Dataset | Random CNN BA | WAY SSL BA | Change |
|---|---:|---:|---:|
| liu2024 | 51.00% | 52.64% | +1.63 points |
| openbmi | 54.75% | 56.77% | +2.02 points |
| physionet | 53.06% | 55.66% | +2.60 points |

This experiment isolates the value of unlabeled WAY EEG. It does not claim that overt-movement EEG is motor imagery, and it does not use WAY cue timing as an intent label. Continuous no-control and prospective patient testing remain necessary.
