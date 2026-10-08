# Cho2017 Locked External V7 Evaluation

> V7 was frozen before EEG inference. Cho2017 contributed no rows to
> training, architecture selection, probability calibration, or threshold selection.

## Cohort

- Healthy participants: **32**
- Adapter-development participant excluded from inference: **S001**
- External EEG windows: **4,983**
- Adapted evaluation windows: **4,023**
- Source EEG: 64 native channels at 512 Hz; V7 uses seven shared motor channels at 100 Hz.
- Intent: 0.5-2.5 s after the visual imagery cue.
- Rest: independent continuous resting-state data, divided into 2-second windows with a 1-second step.

## Frozen Results

| Configuration | BA | Sensitivity | Specificity | False activation/rest window | AUROC | Abstained |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot conservative | 50.0% | 0.0% | 100.0% | 0.0% | 0.575 | 100.0% |
| 30-window adapted conservative | 50.0% | 0.0% | 100.0% | 0.0% | 0.925 | 100.0% |
| 30-window adapted sensitivity-targeted | 50.0% | 0.0% | 100.0% | 0.0% | 0.925 | 100.0% |

Primary participant-bootstrap 95% confidence intervals:
BA **50.0%-50.0%**, sensitivity **0.0%-0.0%**, specificity **100.0%-100.0%**.

## Interpretation Boundary

- The zero-shot result is a fully external population-model test.
- The adapted result uses a frozen 30-window stratified calibration subset from the same session; it is external few-shot adaptation, not prospective later-session validation.
- The negative class is genuine continuous rest, but commands/minute remain unmeasured because temporal confirmation is not applied here.
- Left/right labels are used only as internal calibration evidence; device-facing predictions remain REST or INTENT.
- Retuning V7 from these results would make Cho2017 development data for any later experiment.

## Outputs

- `01_external_summary.png`
- `02_adapted_probability_distribution.png`
- `03_participant_variability.png`
- `04_external_roc.png`
- `cho2017_v7_locked_predictions.csv`
- `cho2017_v7_locked_metrics.json`
