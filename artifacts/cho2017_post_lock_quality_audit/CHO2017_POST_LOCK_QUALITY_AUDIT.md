# Cho2017 Post-Lock Quality-Gate Audit

> Exploratory only. The locked result remains 0% sensitivity because the
> frozen OOD gate abstained on 100% of external windows.

## Calibration-Only Diagnostic

| Configuration | BA | Sensitivity | Specificity | False activation/rest window | Availability | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Locked Conservative | 50.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.925 |
| Recalibrated Conservative | 58.5% | 17.0% | 100.0% | 0.0% | 93.7% | 0.925 |
| Recalibrated Sensitivity Targeted | 72.0% | 45.2% | 98.8% | 1.2% | 93.7% | 0.925 |
| Bypass Sensitivity Targeted | 73.1% | 47.7% | 98.6% | 1.4% | 100.0% | 0.925 |

The patient-calibrated gate was fitted from each participant's 30 reserved
calibration windows only. No evaluation labels or evaluation probabilities
were used to fit it. Nevertheless, this method was proposed after the locked
failure was observed, so these values are not confirmatory.

## Recommended Protocol Change

Pre-register hardware/session-specific quality normalization as part of the
30-window calibration session, while preserving the population intent model,
probability calibration, and operating thresholds. Confirm that revision on a
new untouched dataset or later-session cohort.
