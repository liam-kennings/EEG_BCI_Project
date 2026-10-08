# Morlet ERD Feature Audit

The new pre-ML audit uses all available unilateral motor-imagery trials: 9,185 trials from 108 subjects. For every trial, it computes complex Morlet power from 4 to 30 Hz and reports post-cue power relative to that trial's one-second pre-cue baseline:

```text
ERD/ERS (dB) = 10 * log10(task power / baseline power)
```

The output is intentionally not a classifier. It provides an interpretable feature table and visual checks before any model is fitted.

## Aggregate Findings

The early 0.5–1.5 s window shows the expected stronger mu and low-beta suppression during imagery than rest. The lateralization feature is C3 minus C4, where a negative value means stronger C3 suppression.

| Condition | C3 mu ERD | C4 mu ERD | C3-C4 mu | C3-C4 low beta |
| --- | ---: | ---: | ---: | ---: |
| Rest | -1.589 dB | -1.564 dB | -0.025 dB | 0.129 dB |
| Left imagery | -3.270 dB | -3.429 dB | 0.159 dB | 0.222 dB |
| Right imagery | -3.481 dB | -2.955 dB | -0.525 dB | -0.516 dB |

This has the expected direction: left imagery is relatively more suppressed at C4, and right imagery is relatively more suppressed at C3. However, per-trial standard deviations are about 3–4 dB, so the direction signal is much smaller than the between-trial and between-subject variability. This supports feature normalization and patient calibration before attempting a more complex model.

## Outputs

- [Morlet ERD maps](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_feature_audit/morlet_erd_tfr_c3_c4.png)
- [ERD time courses](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_feature_audit/erd_timecourses_c3_c4.png)
- [Motor-cortex topomaps](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_feature_audit/erd_topomaps.png)
- [Lateralization distributions](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_feature_audit/lateralization_feature_distributions.png)
- [Trial quality summary](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_feature_audit/trial_quality_peak_to_peak.png)
- [Trial feature table](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_feature_audit/morlet_erd_trial_features.csv)

## Next Model Comparison

The next honest experiment is a locked-subject comparison using this exported table:

1. Baseline-normalized ERD and C3-C4 features with logistic regression or shrinkage LDA.
2. Filter-bank CSP with the same subject split.
3. A combined traditional feature model, using feature selection fitted only on development subjects.

The validation and final patient groups must remain unseen until threshold selection and final evaluation, respectively.
