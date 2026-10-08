# Patient-Adaptive Neural-Fusion Motor-Intent Pipeline v5

## Status

- The previous final cohort is now labeled `development_observed`.
- No confirmatory accuracy is reported because `locked_external` is empty.
- Fusion and Platt-model fitting use controller subjects disjoint from the controller subjects used to select calibration shrinkage and operating points.
- Thresholds are dataset-specific and selected without using the development-observed cohort.
- Neural architecture selection previously observed the development-observed cohort; this integrated result is exploratory and cannot be relabeled as external validation.
- Each neural architecture is averaged across five seeds before controller-only fusion, calibration, patient adaptation, quality rejection, and temporal confirmation.

## Development-Observed Results

- Legacy late-fusion BA: **60.1%**.
- Raw ShallowConvNet seed-ensemble result: **65.1% BA**, **35.3% sensitivity**, **95.0% specificity**.
- Raw MRCP + FBCNet seed-ensemble result: **66.9% BA**, **37.9% sensitivity**, **95.9% specificity**.
- Calibrated conservative BA: **70.6%** (subject-bootstrap 95% CI 63.1-77.8%).
- Calibrated conservative sensitivity: **47.6%** (95% CI 31.3-63.6%).
- Calibrated conservative specificity: **93.6%**.
- Sensitivity-targeted population result: **71.8% sensitivity**, **79.6% specificity**.
- Patient-adapted conservative result: **73.3% BA**, **52.9% sensitivity**, **93.8% specificity**.
- Paired equal-participant change versus the physiology-only patient-adapted pipeline: **+2.16 points** (95% participant-bootstrap interval **-0.73 to +4.77**); **68.4%** of participants improved.
- Patient-adapted sensitivity-targeted result: **60.2% sensitivity**, **89.0% specificity**.

The patient-adapted conservative operating point is the preferred integrated candidate, but it remains development-observed until a new unseen cohort confirms the gain. The sensitivity-targeted population point is exploratory, not a device-command policy: its Liu specificity is only **30.4%**.

PhysioNet patient-adapted specificity is **86.2%**, so its observed false-activation rate exceeds the 10% controller target. This domain requires recalibration or stronger temporal confirmation before prospective use.

All values are exploratory. They are not a new locked final result, and cue-locked specificity is not a continuous false-activation rate.

## Per-Dataset Results

| Dataset | Population BA | Population sensitivity | Population specificity | Adapted BA | Adapted sensitivity | Adapted specificity |
|---|---:|---:|---:|---:|---:|---:|
| physionet | 56.9% | 24.5% | 89.3% | 62.2% | 38.2% | 86.2% |
| openbmi | 74.7% | 55.4% | 94.0% | 76.4% | 57.3% | 95.4% |
| liu2024 | 56.2% | 15.5% | 96.9% | 64.3% | 37.0% | 91.6% |

### Sensitivity-Targeted Population Check

| Dataset | Sensitivity | Specificity | Balanced accuracy |
|---|---:|---:|---:|
| physionet | 65.7% | 62.7% | 64.2% |
| openbmi | 71.2% | 87.8% | 79.5% |
| liu2024 | 87.0% | 30.4% | 58.7% |

Subject-bootstrap intervals for the pooled and per-dataset endpoints are stored in `translational_v5_patient_adaptive_metrics.json`.

## Controller Design

- Intent fusion weights: `{'mrcp_fbcnet': 0.6, 'riemann': 0.25, 'smr': 0.15}`.
- Direction fusion weights: `{'riemann': 0.5, 'fbcsp': 0.3, 'mrcp': 0.2}`.
- Probability calibration: per-dataset Platt scaling with controller-selected identity shrinkage.
- Intent calibration blends: `{'__global__': 0.0, 'liu2024': 0.0, 'openbmi': 0.0, 'physionet': 1.0}`.
- Patient calibration: early recording trials provide labeled rest, left, and right examples; later recording trials remain evaluation-only.
- Patient-adapter inputs: `['population_stage', 'mrcp', 'smr', 'riemann']`.
- Controller-selected calibration configurations: `{'physionet': {'budget_per_class': 5, 'blend_weight': 0.5}, 'openbmi': {'budget_per_class': 10, 'blend_weight': 0.5}, 'liu2024': {'budget_per_class': 10, 'blend_weight': 0.5}}`.
- Neural ensemble seeds: `[42, 43, 44, 45, 46]`.
- Neural ensemble architectures: `['mrcp_fbcnet', 'shallow_convnet']`.
- Neural inference remains EEG-only; FBCSP/Riemann physiology retains responsibility for conditional direction evidence.
- Portable general-model bundle: `/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project/artifacts/gui_experiments/translational_v5_patient_adaptive/general_model_bundle` with 10 fingerprinted neural seed models and the rapid-calibration recipe.

## Dataset Quality Control

- Accepted recordings: **492**.
- Excluded recordings or subjects: **1**.
- Excluded subjects: `liu2024:S013`.

Full exclusion reasons are stored in `translational_v5_patient_adaptive_metrics.json` and beside the window cache in `translational_windows_v3.qc.json`.

## Implemented Changes

1. Stage-specific intent and direction fusion.
2. Per-dataset probability calibration and operating thresholds.
3. Conservative and sensitivity-targeted operating points.
4. Patient calibration for all three datasets.
5. Subject-level bootstrap confidence intervals.
6. Temporal confirmation and refractory logic prepared for continuous no-control recordings.
7. Legacy late-fusion results retained as a fixed reference.
8. Five-seed neural intent proposals integrated through controller-only fusion and calibration.
9. Population score, MRCP, SMR, and Riemann features used for rapid patient adaptation.
10. Patient-adapted probabilities passed through the existing quality and temporal safety layers.

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
