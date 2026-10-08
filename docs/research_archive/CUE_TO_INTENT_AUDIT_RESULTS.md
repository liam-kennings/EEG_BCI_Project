# Cue-to-Intent Audit

This analysis separates early visual cue processing from later motor-imagery physiology. EEG was common-average referenced over the full scalp, then visual-region and motor-region features were extracted. The visual feature window was `80–300 ms` after cue onset; motor mu/beta ERD was computed from `0.5–3.5 s` after cue onset against the trial's pre-cue baseline.

The correlation analysis removes each subject's mean and each cue side's mean before measuring the association. This prevents a simple left/right difference from being mislabeled as a cue-to-intent relationship.

| Measurement | Result |
| --- | ---: |
| Mean patient residual correlation, early visual response versus later motor mu ERD | `r = 0.088` |
| Group test of patient correlations | `p = 0.0001` |
| Final unseen-subject left/right balanced accuracy, early visual features | `48.8%` |
| Final unseen-subject left/right balanced accuracy, late motor ERD features | `56.8%` |

The visual-to-motor association is statistically detectable because many patients contribute data, but its effect size is small. More importantly, early posterior visual features do not decode left versus right above chance on the untouched subjects. This audit does not support the claim that the current motor decoder is merely reading cue location or eye-related visual activity.

The correct BCI design is therefore a cue-gated sequence:

1. The known visual cue opens a short evaluation interval.
2. Only later central motor-channel features can supply evidence for motor intent.
3. The early visual response is retained for quality/timing research, not for the command classifier.

Saved outputs: [correlation figure](/Users/LiamKennings_1/Documents/BME%20473/artifacts/cue_intent_audit/cue_to_motor_correlation.png), [cue-confound audit](/Users/LiamKennings_1/Documents/BME%20473/artifacts/cue_intent_audit/visual_vs_motor_direction_decoding.png), [trial features](/Users/LiamKennings_1/Documents/BME%20473/artifacts/cue_intent_audit/cue_intent_trial_features.csv), and [full metrics](/Users/LiamKennings_1/Documents/BME%20473/artifacts/cue_intent_audit/cue_intent_correlation_metrics.json).
