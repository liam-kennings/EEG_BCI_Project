# HS-Stroke Locked External Evaluation

> This is the first evaluation of the frozen translational v4 decoder on the
> corrected HS-Stroke v6 chronic-stroke cohort. No HS-Stroke rows were used
> to fit preprocessing, features, models, calibration, thresholds, or timing.

## Cohort

- Patients: **14**
- Sessions: **171**
- Trials represented: **13,324**
- Evaluation windows: **26,483**
- Automatically excluded sessions: **106**
- Preparation window: 6.5-8.5 seconds
- Imagery window: 9.5-11.5 seconds
- Feedback: excluded

## Frozen Results

| Operating point | Sensitivity | Specificity | Intent BA | False activation/window | Missed intent | Direction BA* | Availability |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conservative | 0.0% | 100.0% | 50.0% | 0.0% | 100.0% | nan% | 0.0% |
| Sensitivity-targeted | 0.0% | 100.0% | 50.0% | 0.0% | 100.0% | nan% | 0.0% |

*Direction is scored against numeric MAT classes 1 and 2.* The public
release does not explicitly document which numeric value means left versus
right, so semantic direction names remain provisional. Intent metrics are
unaffected by that mapping.

## Interpretation Limits

- The independent sample size is 14 patients; sessions and trials are repeated observations.
- Preparation is structured cue-locked rest, not natural idle, so the false-activation fraction cannot be reported as commands per minute.
- Quality abstentions count as missed intent but as no issued command during preparation.
- These results must not be used to retune the frozen model and then re-labelled confirmatory.
- Patient-specific calibration is a separate secondary experiment using early sessions only, with later sessions re-locked.

## Figures

- `01_locked_external_summary.png`
- `02_patient_variability.png`
- `03_locked_confusion_matrices.png`
- `04_intent_probability_distribution.png`
