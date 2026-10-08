# Multi-Source Privileged EEG Ensemble

## Question

Can synchronized EEG-EMG execution datasets improve EEG-only motor-imagery intent detection when peripheral signals are restricted to source supervision?

## Experiment

The experiment combines three EEG-only component scores:

1. Target-only imagery/attempted-movement student trained on PhysioNet, OpenBMI, and Liu development subjects.
2. MRCP-assisted student trained with EMG-confirmed pre-movement source examples.
3. WAY-assisted student trained with pre-kinematic-onset EEG and post-onset EMG verification.

Both source weights are selected on controller-validation subjects. For each target dataset, a logistic stacker is trained with group-held controller folds and compared against all three individual components. The controller chooses the method and threshold subject to at most 10% false activation per rest window on controller data. Development-observed subjects are evaluated only after this contract is fixed.

EMG and kinematics never enter target training, observed evaluation, the serialized inference method, or the deployment feature matrix. The left/right direction decoder is unchanged.

## Controller Decisions

- MRCP source weight: **0.35**.
- WAY source weight: **0.35**.
- Liu policy: **MRCP-assisted student**, threshold **0.740**.
- OpenBMI policy: **WAY-assisted student**, threshold **0.705**.
- PhysioNet policy: **WAY-assisted student**, threshold **0.665**.
- Cross-fitted stackers were evaluated but not selected in any domain.

The resulting model is therefore a validation-selected domain router, not a weighted average that is forced to use every source.

## Development-Observed Results

| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| Domain-thresholded target-only | 52.82% | 10.58% | 95.07% | 0.683 | 0.645 |
| MRCP-assisted | 53.02% | 11.03% | 95.02% | 0.682 | 0.643 |
| WAY-assisted | 52.92% | 10.63% | 95.20% | 0.697 | 0.656 |
| Validation-selected multi-source | **52.98%** | **11.00%** | **94.96%** | **0.698** | **0.657** |

The pooled improvement is small: **+0.16 balanced-accuracy points** and **+0.42 sensitivity points**, with specificity decreasing **0.11 points**. Across 38 paired unseen participants, the equal-participant mean balanced-accuracy change was **+0.92 points** with a 95% participant-bootstrap interval of **+0.37 to +1.50 points**; 52.6% of participants improved.

## Domain Results

| Dataset | Target-only BA | Selected multi-source BA | Change |
|---|---:|---:|---:|
| Liu acute stroke | 54.85% | 55.85% | +1.00 points |
| OpenBMI | 52.25% | 51.81% | -0.44 points |
| PhysioNet | 53.36% | 54.64% | +1.28 points |

OpenBMI demonstrates the remaining domain-selection risk: WAY looked best on its five controller subjects but did not transfer to the six observed subjects. A larger controller cohort or nested repeated subject resampling is needed before treating this router as stable.

## Run And Outputs

Run `EEG_BCI_Project_RUN_MULTISOURCE_ENSEMBLE.command`, or choose `Multi-source privileged EEG ensemble` in the GUI.

The standardized model, full JSON metrics, report, and two presentation figures are written to `artifacts/multisource_privileged_ensemble/`. The model API accepts EEG features and dataset identifiers only.

## Conclusion

Privileged execution supervision provides a small, measurable participant-level benefit, especially for Liu and PhysioNet, but it does not solve weak motor-imagery detection. This result supports retaining source-supervised students as optional priors while prioritizing patient calibration, continuous no-control validation, and prospective attempted-movement data.
