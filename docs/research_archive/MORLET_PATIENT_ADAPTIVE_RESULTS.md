# Patient-Adaptive Morlet Controller

This extension follows the known EEGMMI cue structure. The Morlet audit now covers `0.5–4.1 s` after each unilateral imagery cue and exports early (`0.5–1.5 s`), middle (`1.5–2.5 s`), terminal (`2.5–3.5 s`), and full-cue ERD features. The controller uses Runs 4/8 for each target patient's unlabeled feature alignment and small labeled calibration, then evaluates that patient only on Run 12.

The window, adaptation blend, rest-score quantile, and optional temporal two-of-three confirmation rule are selected on controller-validation subjects. Final target patients are never used to choose those settings.

| Controller operating point | Selected policy | Final balanced accuracy | Final false activation | Final missed intent |
| --- | --- | ---: | ---: | ---: |
| Responsive, controller FP <= 15% | Early ERD, 70% local adapter, 10% rest quantile | 63.2% | 25.9% | 47.7% |
| Conservative, controller FP <= 5% | Early/middle/terminal 2-of-3 vote, global model, 5% rest quantile | 54.3% | 8.8% | 82.5% |

The responsive policy improves sensitivity compared with the earlier safety gate, but fails to transfer its false-activation limit to final patients. The conservative temporal rule protects against false activation but suppresses too many genuine intents. This is an honest negative result: temporal voting is a safety layer, not a replacement for reliable patient/session calibration.

## Artifacts and Visualization

- [Full-cue Morlet ERD maps](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_fullcue_audit/morlet_erd_tfr_c3_c4.png)
- [ICA component review for S001 Run 4](/Users/LiamKennings_1/Documents/BME%20473/artifacts/ica_diagnostic/S001_R04/ica_component_maps.png)
- [Controller-selection frontier](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_patient_controller/morlet_patient_controller_frontier.png)
- [Responsive final operating point](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_patient_controller/morlet_patient_controller_final.png)
- [Conservative final operating point](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_patient_controller_safety05/morlet_patient_controller_final.png)

The ICA diagnostic ranks components by correlation with frontal EEG electrodes for review but does not automatically delete them. Automatic deletion would be unsafe here because EEGMMI does not provide dedicated EOG/EMG reference channels.
