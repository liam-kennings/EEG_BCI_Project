# Auxiliary-Direction Binary Motor-Intent Pipeline v7

## Status

- The external output remains binary `rest` versus `intent`; left/right cue labels are used only by an internal auxiliary model.
- Auxiliary direction is collapsed to `P(left) + P(right)` and never exposed as a device command.
- Recording 0 supplies patient calibration; all calibration rows are excluded from adapted evaluation.
- Later recordings from the same participant are evaluation-only.
- Fusion, calibration, adapter configuration, and thresholds use controller participants only.
- V7 graph/LDA extension: fixed 10-10 electrode geometry supplies graph features; no EEG-derived edge is fitted.
- Weighted shrinkage LDA and graph models remain controller-selected candidates rather than mandatory fusion components.
- The result remains development-observed because `locked_external` is empty.

## Development-Observed Results

- General population conservative: **70.6% BA**, **47.6% sensitivity**, **93.6% specificity**.
- General population sensitivity-targeted: **75.7% BA**, **71.8% sensitivity**, **79.6% specificity**.
- Population model on the same later-recording rows and patient thresholds: **69.1% BA**.
- Binary patient-adapted conservative: **73.6% BA** (95% CI 67.5-78.7%), **53.5% sensitivity** (95% CI 41.4-65.8%), and **93.7% specificity**.
- Binary patient-adapted sensitivity-targeted: **74.1% BA**, **58.5% sensitivity**, **89.6% specificity**.
- Raw auxiliary-direction intent model: **59.6% BA**.
- Raw fixed-geometry graph model: **55.5% BA**.
- Raw weighted shrinkage-LDA model: **62.8% BA**.
- Raw ShallowConvNet five-seed ensemble: **65.1% BA**.
- Raw MRCP + FBCNet five-seed ensemble: **66.9% BA**.
- Neural integration changed equal-participant adapted BA by **+1.97 points** versus physiology-only adaptation.

The matched later-recording comparison isolates adaptation benefit using the same evaluation rows and operating thresholds.

## V5/V6 Reference Comparison

- Conservative BA: v5 73.3%, v6 71.9%, v7 73.6%.
- Conservative sensitivity/specificity: v5 52.9%/93.8%, v6 49.1%/94.7%, v7 53.5%/93.7%.
- Sensitivity-targeted BA: v5 74.6%, v6 74.5%, v7 74.1%.

V7 is retained only if controller-selected auxiliary weights and later-recording results justify the added calibration labels.

## Per-Dataset Results

| Dataset | Calibration windows | Matched population BA | Adapted BA | Adapted sensitivity | Adapted specificity |
|---|---:|---:|---:|---:|---:|
| physionet | 30 | 61.5% | 63.8% | 40.7% | 86.9% |
| openbmi | 30 | 71.1% | 76.3% | 57.3% | 95.3% |
| liu2024 | 30 | 64.0% | 65.0% | 40.5% | 89.5% |

## Binary Controller

- Classes: `['rest', 'intent']`.
- Auxiliary training labels: `['rest', 'left', 'right']`.
- Auxiliary direction output exposed: `False`.
- Selected intent fusion: `{'mrcp_fbcnet': 0.6, 'riemann': 0.25, 'smr': 0.15}`.
- Intent calibration blends: `{'__global__': 0.0, 'liu2024': 0.0, 'openbmi': 0.0, 'physionet': 1.0}`.
- Patient-adapter inputs: `{'physionet': ['population_intent', 'mrcp', 'smr', 'riemann'], 'openbmi': ['population_intent', 'mrcp', 'smr', 'riemann'], 'liu2024': ['population_intent', 'mrcp', 'smr', 'riemann']}`.
- V7 graph/LDA extension metadata: `{'enabled': True, 'scope': 'v7_auxiliary_direction_only', 'fixed_graph_feature_count': 64, 'graph_edge_source': 'fixed_standard_1005_geometry', 'graph_edges_learned_from_eeg': False, 'shrinkage_lda': 'weighted_OAS_shared_covariance', 'population_feature_branches': ['mrcp', 'smr', 'riemann', 'gsp'], 'controller_selected_intent_weights': {'mrcp_fbcnet': 0.6, 'riemann': 0.25, 'smr': 0.15}, 'patient_adapter_graph_candidate_available': True, 'patient_adapter_graph_selected_by_dataset': {'physionet': False, 'openbmi': False, 'liu2024': False}, 'diagnostic_only_methods': ['PCA', 'k-means']}`.
- Calibration configurations: `{'physionet': {'budget_per_source_class': 10, 'total_calibration_windows': 30, 'blend_weight': 0.5, 'adapter_branch_names': ['population_intent', 'mrcp', 'smr', 'riemann'], 'graph_branch_enabled': False, 'graph_selection_guard': {'status': 'not_applicable'}, 'auxiliary_weight': 0.25}, 'openbmi': {'budget_per_source_class': 10, 'total_calibration_windows': 30, 'blend_weight': 0.5, 'adapter_branch_names': ['population_intent', 'mrcp', 'smr', 'riemann'], 'graph_branch_enabled': False, 'graph_selection_guard': {'status': 'not_applicable'}, 'auxiliary_weight': 0.5}, 'liu2024': {'budget_per_source_class': 10, 'total_calibration_windows': 30, 'blend_weight': 0.5, 'adapter_branch_names': ['population_intent', 'mrcp', 'smr', 'riemann'], 'graph_branch_enabled': False, 'graph_selection_guard': {'status': 'rejected_to_base', 'minimum_gain': 0.01, 'controller_sensitivity_gain': 0.006249999999999978, 'controller_balanced_accuracy_gain': 0.0031249999999999334}, 'auxiliary_weight': 0.0}}`.
- Adapted evaluation rows: **4,645**.
- Known visual or auditory cue supplies the requested movement; EEG only confirms rest versus intent.
- Three-window confirmation can gate commands, but it is not reflected in cue-locked metrics.

## Safety Boundary

Not evaluated: no hard-negative manifest was supplied. Cue-locked window specificity is not converted into a deployment false-activation rate.

Temporal confirmation: `{'status': 'prepared_not_evaluated', 'reason': 'no continuous no-control manifest', 'confirmations': 3, 'refractory_seconds': 2.0, 'probability_source': 'patient_adapted_neural_integrated_intent'}`.

Specificity is measured per cue-locked rest window. False activations per minute and command latency require continuous no-control recordings.

## Protocol Note

V7 uses recording-0 cue labels only for calibration and excludes those rows from evaluation. Its binary endpoint is directly comparable with v5 and v6 on recording 1.

## Outputs

- `translational_v7_auxiliary_direction_metrics.json`
- `translational_v7_auxiliary_direction_models.joblib`
- `general_model_bundle/`
- `01_decoder_comparison.png`
- `02_safety_sensitivity_frontier.png`
- `03_patient_calibration_burden.png`
- `04_physiology_audit.png`
- `05_quality_gate_availability.png`
- `06_domain_probability_calibration.png`
- `07_operating_point_comparison.png`
- `08_neural_integration_summary.png`
- `09_v5_v6_v7_comparison.png`
- `10_v7_graph_signal_audit.png`
- `11_v7_graph_representation_diagnostics.png`
