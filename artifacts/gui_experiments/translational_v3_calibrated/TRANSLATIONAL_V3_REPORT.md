# Calibrated Translational Motor-Intent Pipeline v3

## Status

- The previous final cohort is now labeled `development_observed`.
- No confirmatory accuracy is reported because `locked_external` is empty.
- Fusion and Platt-model fitting use controller subjects disjoint from the controller subjects used to select calibration shrinkage and operating points.
- Thresholds are dataset-specific and selected without using the development-observed cohort.

## Development-Observed Results

- Legacy late-fusion BA: **60.1%**.
- Calibrated conservative BA: **59.9%** (subject-bootstrap 95% CI 55.2-65.6%).
- Calibrated conservative sensitivity: **23.5%** (95% CI 13.8-35.5%).
- Calibrated conservative specificity: **96.3%**.
- Sensitivity-targeted population result: **61.4% sensitivity**, **79.9% specificity**.
- Patient-adapted conservative result: **67.9% BA**, **41.3% sensitivity**.
- Patient-adapted sensitivity-targeted result: **46.1% sensitivity**, **89.9% specificity**.

The patient-adapted conservative operating point is the current preferred cue-locked configuration because it improves intent recovery while retaining high specificity. The sensitivity-targeted population point is exploratory, not a device-command policy: its Liu specificity is only **38.7%**.

All values are exploratory. They are not a new locked final result, and cue-locked specificity is not a continuous false-activation rate.

## Per-Dataset Results

| Dataset | Population BA | Population sensitivity | Population specificity | Adapted BA | Adapted sensitivity |
|---|---:|---:|---:|---:|---:|
| physionet | 57.2% | 22.9% | 91.4% | 63.3% | 40.1% |
| openbmi | 60.9% | 24.4% | 97.5% | 69.4% | 42.3% |
| liu2024 | 54.9% | 16.0% | 93.7% | 61.8% | 34.0% |

### Sensitivity-Targeted Population Check

| Dataset | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|
| physionet | 59.0% | 65.1% | 62.1% |
| openbmi | 59.3% | 87.0% | 73.1% |
| liu2024 | 84.5% | 38.7% | 61.6% |

Subject-bootstrap intervals for the pooled and per-dataset endpoints are stored in `translational_v3_metrics.json`.

## Controller Design

- Intent fusion weights: `{'mrcp': 0.3333333333333333, 'riemann': 0.3333333333333333, 'smr': 0.3333333333333333}`.
- Direction fusion weights: `{'riemann': 0.5, 'fbcsp': 0.3, 'mrcp': 0.2}`.
- Probability calibration: per-dataset Platt scaling with controller-selected identity shrinkage.
- Intent calibration blends: `{'__global__': 0.0, 'liu2024': 0.0, 'openbmi': 0.0, 'physionet': 1.0}`.
- Patient calibration: early recording trials for every dataset; later recording trials remain evaluation-only.

## Dataset Quality Control

- Accepted recordings: **492**.
- Excluded recordings or subjects: **1**.
- Excluded subjects: `liu2024:S013`.

Full exclusion reasons are stored in `translational_v3_metrics.json` and beside the window cache in `translational_windows_v3.qc.json`.

## Implemented Changes

1. Stage-specific intent and direction fusion.
2. Per-dataset probability calibration and operating thresholds.
3. Conservative and sensitivity-targeted operating points.
4. Patient calibration for all three datasets.
5. Subject-level bootstrap confidence intervals.
6. Temporal confirmation and refractory logic prepared for continuous no-control recordings.
7. Legacy late-fusion results retained as a fixed reference.

## Continuous No-Control Boundary

Not evaluated: no hard-negative manifest was supplied. Cue-locked window specificity is not converted into a deployment false-activation rate.

Temporal confirmation status: `prepared_not_evaluated`.

A false-activations-per-minute claim is emitted only when a hard-negative manifest supplies explicitly continuous no-control recordings. Otherwise the script reports cue-locked window false activation and marks continuous safety as not evaluated.

## Figures

- `01_decoder_comparison.png`
- `02_safety_sensitivity_frontier.png`
- `03_patient_calibration_burden.png`
- `04_physiology_audit.png`
- `05_quality_gate_availability.png`
- `06_domain_probability_calibration.png`
- `07_operating_point_comparison.png`
