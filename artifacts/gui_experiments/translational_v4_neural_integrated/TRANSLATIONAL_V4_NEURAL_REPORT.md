# Neural-Integrated Translational Motor-Intent Pipeline v4

## Status

- The previous final cohort is now labeled `development_observed`.
- No confirmatory accuracy is reported because `locked_external` is empty.
- Fusion and Platt-model fitting use controller subjects disjoint from the controller subjects used to select calibration shrinkage and operating points.
- Thresholds are dataset-specific and selected without using the development-observed cohort.
- MRCP + FBCNet architecture selection previously observed the development-observed cohort; this integrated result is exploratory and cannot be relabeled as external validation.
- The five neural seeds are averaged before controller-only fusion, calibration, patient adaptation, quality rejection, and temporal confirmation.

## Development-Observed Results

- Legacy late-fusion BA: **60.1%**.
- Raw MRCP + FBCNet seed-ensemble result: **66.8% BA**, **37.9% sensitivity**, **95.6% specificity**.
- Calibrated conservative BA: **69.8%** (subject-bootstrap 95% CI 62.3-77.0%).
- Calibrated conservative sensitivity: **45.6%** (95% CI 29.2-61.7%).
- Calibrated conservative specificity: **94.0%**.
- Sensitivity-targeted population result: **62.2% sensitivity**, **85.3% specificity**.
- Patient-adapted conservative result: **70.6% BA**, **46.7% sensitivity**, **94.4% specificity**.
- Paired equal-participant change versus the physiology-only patient-adapted pipeline: **+0.16 points** (95% participant-bootstrap interval **-3.67 to +3.97**); **55.3%** of participants improved.
- Patient-adapted sensitivity-targeted result: **47.7% sensitivity**, **92.1% specificity**.

The patient-adapted conservative operating point is the preferred integrated candidate, but the equal-participant interval crosses zero; v3 remains the fixed reference until a new unseen cohort confirms the gain. The sensitivity-targeted population point is exploratory, not a device-command policy: its Liu specificity is only **42.9%**.

PhysioNet patient-adapted specificity is **87.5%**, so its observed false-activation rate exceeds the 10% controller target. This domain requires recalibration or stronger temporal confirmation before prospective use.

All values are exploratory. They are not a new locked final result, and cue-locked specificity is not a continuous false-activation rate.

## Per-Dataset Results

| Dataset | Population BA | Population sensitivity | Population specificity | Adapted BA | Adapted sensitivity | Adapted specificity |
|---|---:|---:|---:|---:|---:|---:|
| physionet | 57.0% | 24.2% | 89.9% | 61.6% | 35.8% | 87.5% |
| openbmi | 73.9% | 53.4% | 94.3% | 72.9% | 49.9% | 95.9% |
| liu2024 | 54.0% | 10.0% | 97.9% | 63.8% | 35.5% | 92.1% |

### Sensitivity-Targeted Population Check

| Dataset | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|
| physionet | 56.9% | 71.6% | 64.2% |
| openbmi | 61.3% | 92.3% | 76.8% |
| liu2024 | 79.5% | 42.9% | 61.2% |

Subject-bootstrap intervals for the pooled and per-dataset endpoints are stored in `translational_v4_neural_metrics.json`.

## Controller Design

- Intent fusion weights: `{'mrcp_fbcnet': 0.6, 'riemann': 0.25, 'smr': 0.15}`.
- Direction fusion weights: `{'riemann': 0.5, 'fbcsp': 0.3, 'mrcp': 0.2}`.
- Probability calibration: per-dataset Platt scaling with controller-selected identity shrinkage.
- Intent calibration blends: `{'__global__': 0.0, 'liu2024': 0.0, 'openbmi': 0.0, 'physionet': 1.0}`.
- Patient calibration: early recording trials for every dataset; later recording trials remain evaluation-only.
- Neural ensemble seeds: `[42, 43, 44, 45, 46]`.
- Neural inference remains EEG-only; FBCSP/Riemann physiology retains responsibility for conditional direction evidence.

## Dataset Quality Control

- Accepted recordings: **492**.
- Excluded recordings or subjects: **1**.
- Excluded subjects: `liu2024:S013`.

Full exclusion reasons are stored in `translational_v4_neural_metrics.json` and beside the window cache in `translational_windows_v3.qc.json`.

## Implemented Changes

1. Stage-specific intent and direction fusion.
2. Per-dataset probability calibration and operating thresholds.
3. Conservative and sensitivity-targeted operating points.
4. Patient calibration for all three datasets.
5. Subject-level bootstrap confidence intervals.
6. Temporal confirmation and refractory logic prepared for continuous no-control recordings.
7. Legacy late-fusion results retained as a fixed reference.
8. Five-seed MRCP + FBCNet intent proposal integrated through controller-only fusion and calibration.
9. Patient-adapted neural-integrated probabilities passed through the existing quality and temporal safety layers.

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
- `08_neural_integration_summary.png`
