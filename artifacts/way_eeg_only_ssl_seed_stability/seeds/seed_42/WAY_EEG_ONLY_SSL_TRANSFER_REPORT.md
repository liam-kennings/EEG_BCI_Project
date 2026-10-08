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
| Raw CNN random init | 53.06% | 14.81% | 91.30% | 0.614 | 0.590 |
| WAY EEG-only SSL | 56.13% | 18.38% | 93.88% | 0.683 | 0.664 |
| WAY peripheral-supervised | 52.92% | 10.63% | 95.20% | 0.697 | 0.656 |

Against the architecture-matched random CNN, the equal-participant mean SSL change was **+3.82 points** (95% participant-bootstrap CI **+2.25 to +5.37**); **78.9%** of participants improved.

## Dataset Results

| Dataset | Random CNN BA | WAY SSL BA | Change |
|---|---:|---:|---:|
| liu2024 | 49.98% | 53.78% | +3.79 points |
| openbmi | 54.06% | 56.52% | +2.46 points |
| physionet | 51.83% | 56.12% | +4.28 points |

This experiment isolates the value of unlabeled WAY EEG. It does not claim that overt-movement EEG is motor imagery, and it does not use WAY cue timing as an intent label. Continuous no-control and prospective patient testing remain necessary.
