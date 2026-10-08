# Multi-Source Privileged EEG Ensemble

## Leakage Boundary

- MRCP EMG and WAY EMG/kinematics are used only to verify synchronized source examples.
- Every deployed component receives the same 144 EEG features; the fusion API additionally receives only the dataset identifier.
- Source weights are selected on controller-validation subjects.
- Dataset-specific stackers and thresholds use group-held controller predictions.
- Development-observed subjects are evaluated once and never used for model, fusion, or threshold selection.
- The separate left/right direction head is unchanged.

## Development-Observed Result

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Target-only EEG | 52.82% | 10.58% | 95.07% | 0.683 | 0.645 |
| MRCP-assisted EEG | 53.02% | 11.03% | 95.02% | 0.682 | 0.643 |
| WAY-assisted EEG | 52.92% | 10.63% | 95.20% | 0.697 | 0.656 |
| Validation-selected multi-source | 52.98% | 11.00% | 94.96% | 0.698 | 0.657 |

Across **38** paired unseen participants, the equal-participant mean balanced-accuracy change versus the domain-thresholded target-only baseline was **+0.92 points** (95% participant-bootstrap CI **+0.37 to +1.50**).

## Dataset Policies And Results

| Dataset | Controller-selected policy | Target-only BA | Multi-source BA | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| liu2024 | mrcp_student at 0.740 | 54.85% | 55.85% | 25.00% | 86.70% |
| openbmi | way_student at 0.705 | 52.25% | 51.81% | 5.83% | 97.79% |
| physionet | way_student at 0.665 | 53.36% | 54.64% | 17.94% | 91.34% |

## Interpretation

The ensemble is an exploratory cue-locked intent gate, not an asynchronous prosthetic controller. Improvement must be judged against the paired participant interval and domain results, not only pooled window accuracy. Continuous no-control recordings and a still-locked clinical cohort remain required before a device claim.
