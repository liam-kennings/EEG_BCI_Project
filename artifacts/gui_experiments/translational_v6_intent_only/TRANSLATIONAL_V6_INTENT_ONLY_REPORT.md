# Binary Rest-versus-Intent Patient-Adaptive Pipeline v6

## Status

- Left and right imagery labels are merged into one `intent` class.
- No direction decoder, direction calibration model, or direction metric is fitted.
- Recording 0 supplies patient calibration; all calibration rows are excluded from adapted evaluation.
- Later recordings from the same participant are evaluation-only.
- Fusion, calibration, adapter configuration, and thresholds use controller participants only.
- The result remains development-observed because `locked_external` is empty.

## Development-Observed Results

- General population conservative: **70.6% BA**, **47.6% sensitivity**, **93.6% specificity**.
- General population sensitivity-targeted: **75.7% BA**, **71.8% sensitivity**, **79.6% specificity**.
- Population model on the same later-recording rows and patient thresholds: **69.9% BA**.
- Binary patient-adapted conservative: **71.9% BA** (95% CI 66.7-76.8%), **49.1% sensitivity** (95% CI 38.6-60.0%), and **94.7% specificity**.
- Binary patient-adapted sensitivity-targeted: **74.5% BA**, **58.8% sensitivity**, **90.3% specificity**.
- Raw ShallowConvNet five-seed ensemble: **65.1% BA**.
- Raw MRCP + FBCNet five-seed ensemble: **66.9% BA**.
- Neural integration changed equal-participant adapted BA by **+1.63 points** versus physiology-only adaptation.

The matched later-recording comparison isolates adaptation benefit using the same evaluation rows and operating thresholds.

## V5 Directional Comparison

- Conservative v5 to v6: BA 73.3% to 71.9%, sensitivity 52.9% to 49.1%, and specificity 93.8% to 94.7%.
- Sensitivity-targeted v5 to v6: BA 74.6% to 74.5%, sensitivity 60.2% to 58.8%, and specificity 89.0% to 90.3%.

Both pipelines evaluate rest versus intent at the headline endpoint. V6 removes direction-derived adapter inputs and uses a uniform 20-window binary calibration contract.

## Per-Dataset Results

| Dataset | Calibration windows | Matched population BA | Adapted BA | Adapted sensitivity | Adapted specificity |
|---|---:|---:|---:|---:|---:|
| physionet | 20 | 61.8% | 65.6% | 45.9% | 85.3% |
| openbmi | 20 | 72.3% | 74.1% | 51.7% | 96.6% |
| liu2024 | 20 | 61.1% | 61.8% | 31.0% | 92.7% |

## Binary Controller

- Classes: `['rest', 'intent']`.
- Selected intent fusion: `{'mrcp_fbcnet': 0.6, 'riemann': 0.25, 'smr': 0.15}`.
- Intent calibration blends: `{'__global__': 0.0, 'liu2024': 0.0, 'openbmi': 0.0, 'physionet': 1.0}`.
- Patient-adapter inputs: `['population_intent', 'mrcp', 'smr', 'riemann']`.
- Calibration configurations: `{'physionet': {'budget_per_binary_class': 10, 'total_calibration_windows': 20, 'blend_weight': 0.5}, 'openbmi': {'budget_per_binary_class': 10, 'total_calibration_windows': 20, 'blend_weight': 0.5}, 'liu2024': {'budget_per_binary_class': 10, 'total_calibration_windows': 20, 'blend_weight': 0.5}}`.
- Adapted evaluation rows: **4,645**.
- Known visual or auditory cue supplies the requested movement; EEG only confirms rest versus intent.
- Three-window confirmation can gate commands, but it is not reflected in cue-locked metrics.

## Safety Boundary

Not evaluated: no hard-negative manifest was supplied. Cue-locked window specificity is not converted into a deployment false-activation rate.

Temporal confirmation: `{'status': 'prepared_not_evaluated', 'reason': 'no continuous no-control manifest', 'confirmations': 3, 'refractory_seconds': 2.0, 'probability_source': 'patient_adapted_neural_integrated_intent'}`.

Specificity is measured per cue-locked rest window. False activations per minute and command latency require continuous no-control recordings.

## Protocol Note

V6 makes the patient-adaptation boundary explicit and regression-tested: recording-0 calibration windows are never counted in adapted metrics. The v5 split masks also selected recording 1 for evaluation, so the v5 result remains a valid directional benchmark.

## Outputs

- `translational_v6_intent_only_metrics.json`
- `translational_v6_intent_only_models.joblib`
- `general_model_bundle/`
- `01_decoder_comparison.png`
- `02_safety_sensitivity_frontier.png`
- `03_patient_calibration_burden.png`
- `04_physiology_audit.png`
- `05_quality_gate_availability.png`
- `06_domain_probability_calibration.png`
- `07_operating_point_comparison.png`
- `08_neural_integration_summary.png`
- `09_v5_v6_intent_comparison.png`
