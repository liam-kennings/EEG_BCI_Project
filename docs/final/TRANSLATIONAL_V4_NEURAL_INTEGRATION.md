# Translational v4 Neural Integration

## Purpose

This experiment integrates the five-seed MRCP + FBCNet winner into the existing
quality-aware translational pipeline without changing the locked preprocessing,
subject roles, direction decoder, patient-calibration boundary, or temporal
safety rules.

The result remains `development_observed`. The same cohort contributed to the
earlier architecture bakeoff, so v4 is exploratory and cannot be presented as
new external validation.

## Integrated Pipeline

1. Load the same 36,903 native-montage MRCP and SMR windows used by v3.
2. Reject unusual windows with the development-fitted domain-aware quality/OOD
   gate.
3. Average intent probabilities from MRCP + FBCNet seeds 42-46.
4. Use controller-only subjects to select intent fusion. The selected weights
   were 60% MRCP + FBCNet, 25% Riemannian, and 15% SMR.
5. Preserve the separate direction fusion: 50% Riemannian, 30% FBCSP, and 20%
   MRCP.
6. Apply per-dataset Platt calibration and dataset-specific conservative
   thresholds.
7. Adapt with early recording trials from each patient while evaluating only
   later recording trials.
8. Pass patient-adapted probabilities through the quality gate and prepared
   two-window temporal-confirmation and two-second refractory policy.

The neural path is EEG-only at inference. EMG, kinematics, force, and peripheral
labels are not deployment inputs.

## Results

| Configuration | Balanced accuracy | Sensitivity | Specificity | AUROC |
|---|---:|---:|---:|---:|
| Physiology-only v3, patient adapted | 67.9% | 41.3% | 94.5% | 0.863 |
| Raw MRCP + FBCNet five-seed ensemble | 66.8% | 37.9% | 95.6% | 0.851 |
| Integrated population, calibrated | 69.8% | 45.6% | 94.0% | 0.856 |
| **Integrated v4, patient adapted** | **70.6%** | **46.7%** | **94.4%** | **0.882** |

The pooled v4 change relative to v3 is +2.7 balanced-accuracy points and +5.4
sensitivity points, with essentially unchanged overall specificity. Conditional
direction balanced accuracy is 71.4%.

Equal-participant analysis is much more conservative: the mean change is +0.16
points, the 95% participant-bootstrap interval is -3.67 to +3.97, and 55.3% of
participants improve. The pooled gain is therefore not evidence of universal
patient improvement.

| Dataset | v4 adapted BA | v4 sensitivity | v4 specificity | Equal-participant BA change vs v3 |
|---|---:|---:|---:|---:|
| PhysioNet | 61.6% | 35.8% | 87.5% | -1.63 points |
| OpenBMI | 72.9% | 49.9% | 95.9% | +3.50 points |
| Liu stroke | 63.8% | 35.5% | 92.1% | +2.08 points |

PhysioNet’s observed 12.5% false-activation rate exceeds the 10% controller
target. The integrated model should not be used prospectively in that domain
without recalibration or measured temporal confirmation.

## Safety Boundary

The current data contain no explicit continuous no-control recording. Temporal
confirmation is implemented and connected to the final patient-adapted neural
probability source, but false activations per minute and command latency remain
`not_evaluated`.

The next claim-bearing test must freeze the v4 model, collect a new unseen
cohort or prospective session, include continuous no-control periods, and
report false activations per minute, missed intents, and command latency.

## Reproduction

Run:

```bash
./EEG_BCI_Project_RUN_NEURAL_INTEGRATION.command
```

The generated report, complete JSON metrics, model manifest, and eight
presentation figures are stored in
`artifacts/gui_experiments/translational_v4_neural_integrated/`.
