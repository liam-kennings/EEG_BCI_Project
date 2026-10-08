# Execution vs. Motor-Imagery Physiology

## Purpose

This experiment establishes the physiological basis for the project before any
machine-learning result is interpreted.  The question is whether motor imagery
contains the same sensorimotor signature as actual unilateral hand movement,
and, if so, how much weaker that signal is.

The execution recordings are used here as a *physiological reference*, not as
extra imagery training data.  Pooling them into an imagery classifier would
change the task and could overstate performance on the intended rehabilitation
use case.

## Protocol

- Dataset: EEG Motor Movement/Imagery Dataset, 108 subjects and 18,381 usable
  unilateral-fist trials.
- Execution reference: runs `3`, `7`, and `11`.
- Motor-imagery condition: runs `4`, `8`, and `12`.
- Labels: left-fist and right-fist trials only; rest is excluded from this
  physiological comparison.
- Preprocessing: 4--35 Hz filtering and common-average referencing, followed
  by analysis of the central motor montage (`FC3`, `FC4`, `C3`, `C1`, `Cz`,
  `C2`, `C4`, `CP3`, `CP4`).
- Measurement window: 0.5--3.5 seconds after the visual cue.
- Features: baseline-normalized (one-second pre-cue) mu power (8--13 Hz) and
  beta power (13--30 Hz).  The primary measurement is contralateral ERD
  magnitude: `C4` for left fist and `C3` for right fist.
- Statistics: paired execution-versus-imagery test within each subject.

## Results

| Intended action | Band | Execution ERD (dB) | Imagery ERD (dB) | Execution minus imagery | Paired p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| Left fist | Mu | 4.22 | 3.27 | 0.95 | 7.13e-11 |
| Left fist | Beta | 4.34 | 3.27 | 1.06 | 5.82e-20 |
| Right fist | Mu | 3.95 | 3.29 | 0.65 | 8.92e-06 |
| Right fist | Beta | 4.14 | 3.36 | 0.78 | 6.31e-13 |

All four paired comparisons show a reliably stronger contralateral mu/beta
modulation during physical movement.  Crucially, imagery still has a clear
modulation in the same sensorimotor bands; it is smaller by approximately
`0.65--1.06 dB`.  This is the project's measurable "whisper" of motor intent.

## Interpretation

The figures show a useful, honest framing for the project: motor imagery is
not absent, but it is a lower-amplitude and more variable version of the motor
system response seen in execution.  That lower separation explains why a
generic subject-independent classifier can have both missed intents and false
activations, even after conventional filtering.

This result supports the present imagery-only direction:

- Preserve imagery-only held-out testing as the primary rehabilitation claim.
- Use execution data to validate expected mu/beta spatial-temporal physiology
  and to set reasonable visualization/feature hypotheses.
- Focus the imagery pipeline on baseline normalization, contralateral motor
  features, subject calibration, abstention/OOD handling, and temporal
  confirmation rather than claiming that execution labels solve imagery
  decoding.

## Outputs

- [Analysis script](/Users/LiamKennings_1/Documents/BME%20473/scripts/13_execution_vs_imagery_erd_audit.py)
- [ERD time-course comparison](/Users/LiamKennings_1/Documents/BME%20473/artifacts/execution_imagery_erd_audit/execution_vs_imagery_erd_timecourses.png)
- [Paired subject comparison](/Users/LiamKennings_1/Documents/BME%20473/artifacts/execution_imagery_erd_audit/execution_vs_imagery_paired_erd.png)
- [Subject-level feature table](/Users/LiamKennings_1/Documents/BME%20473/artifacts/execution_imagery_erd_audit/execution_imagery_subject_features.csv)
- [Machine-readable metrics](/Users/LiamKennings_1/Documents/BME%20473/artifacts/execution_imagery_erd_audit/execution_imagery_erd_metrics.json)
