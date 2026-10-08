# EMG-Supervised EEG Transfer Experiment

## Leakage Boundary

- EMG is used only in the synchronized source training cohort.
- Positive source EEG windows end at or before confirmed movement onset.
- The source adapter may use synchronized EMG and/or kinematics to establish that boundary.
- Target development, controller, and observed-test rows contain EEG features only.
- The deployed student prediction API accepts EEG only.

## Development-Observed Pure-EEG Result

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC |
|---|---:|---:|---:|---:|
| Target-only EEG | 53.0% | 10.7% | 95.3% | 0.683 |
| EMG-supervised EEG student | 52.9% | 10.5% | 95.4% | 0.697 |

Balanced-accuracy change: **-0.1 percentage points**.

Across **38** paired target participants, the equal-participant mean change was **+0.6 points** (95% participant-bootstrap CI **+0.1 to +1.2**); **47.4%** of participants improved.

The transfer source weight was selected as **0.35** using only the controller-validation cohort. Target development training used equal aggregate weight per dataset and intent class; synchronized source training used equal aggregate weight per participant.

## Dataset Results

| Dataset | Model | Balanced accuracy | Sensitivity | Specificity |
|---|---|---:|---:|---:|
| liu2024 | Target-only EEG | 56.1% | 28.0% | 84.1% |
| liu2024 | EMG-supervised EEG | 56.9% | 29.5% | 84.4% |
| openbmi | Target-only EEG | 52.5% | 7.5% | 97.6% |
| openbmi | EMG-supervised EEG | 51.9% | 6.4% | 97.5% |
| physionet | Target-only EEG | 52.9% | 11.6% | 94.2% |
| physionet | EMG-supervised EEG | 53.7% | 12.8% | 94.6% |

This result measures transfer to EEG-only intent detection. It does not measure an EEG-EMG hybrid controller and does not alter the separate left/right direction head.
It is exploratory because `development_observed` is not the still-locked external cohort.
