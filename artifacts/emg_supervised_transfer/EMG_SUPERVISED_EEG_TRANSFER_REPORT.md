# EMG-Supervised EEG Transfer Experiment

## Leakage Boundary

- EMG is used only in the synchronized source training cohort.
- Positive source EEG windows end at or before confirmed EMG onset.
- Target development, controller, and observed-test rows contain EEG features only.
- The deployed student prediction API accepts EEG only.

## Development-Observed Pure-EEG Result

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC |
|---|---:|---:|---:|---:|
| Target-only EEG | 53.0% | 10.7% | 95.3% | 0.683 |
| EMG-supervised EEG student | 53.3% | 11.5% | 95.1% | 0.682 |

Balanced-accuracy change: **+0.3 percentage points**.

Across **38** paired target participants, the equal-participant mean change was **+0.4 points** (95% participant-bootstrap CI **-0.1 to +0.9**); **39.5%** of participants improved.

The transfer source weight was selected as **0.35** using only the controller-validation cohort. Target development training used equal aggregate weight per dataset and intent class; synchronized source training used equal aggregate weight per participant.

## Dataset Results

| Dataset | Model | Balanced accuracy | Sensitivity | Specificity |
|---|---|---:|---:|---:|
| liu2024 | Target-only EEG | 56.1% | 28.0% | 84.1% |
| liu2024 | EMG-supervised EEG | 57.6% | 30.8% | 84.4% |
| openbmi | Target-only EEG | 52.5% | 7.5% | 97.6% |
| openbmi | EMG-supervised EEG | 52.8% | 7.8% | 97.7% |
| physionet | Target-only EEG | 52.9% | 11.6% | 94.2% |
| physionet | EMG-supervised EEG | 52.9% | 12.5% | 93.2% |

This result measures transfer to EEG-only intent detection. It does not measure an EEG-EMG hybrid controller and does not alter the separate left/right direction head.
It is exploratory because `development_observed` is not the still-locked external cohort.
