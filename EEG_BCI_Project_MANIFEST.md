# EEG BCI Project Manifest

## Final Entry Points

| File | Role |
|---|---|
| `scripts/22_bci_pipeline_gui.py` | Desktop GUI for dataset inspection, experiments, study preparation, and future live acquisition |
| `scripts/30_translational_motor_intent_v3.py` | Final calibrated three-dataset motor-intent experiment |
| `scripts/32_emg_supervised_eeg_transfer.py` | Validation-selected EMG-supervised EEG transfer comparison |
| `scripts/33_prepare_mrcp_eeg_emg_transfer.py` | Downloaded MRCP EEG-EMG adapter and matched target-cache exporter |
| `scripts/35_prepare_way_eeg_gal_transfer.py` | Continuous WAY EEG/EMG/kinematic privileged-source adapter |
| `scripts/36_multisource_privileged_ensemble.py` | Dataset-specific validation-selected MRCP + WAY EEG ensemble |
| `scripts/37_way_eeg_only_ssl_transfer.py` | Unlabeled WAY EEG pretraining and original-dataset fine-tuning ablation |
| `scripts/38_way_eeg_ssl_seed_stability.py` | Controlled five-seed SSL repetition and stability aggregation |
| `scripts/39_eeg_decoder_model_bakeoff.py` | Matched five-seed EEGNet, ShallowConvNet, FBCNet, and MRCP-FBCNet comparison |
| `scripts/40_prepare_prospective_controller.py` | Freeze v4, prepare cue-constrained control, verify hashes, and replay probability streams |
| `scripts/41_mrcp_phase_ablation.py` | Development-only grouped ablation of explicit MRCP phase features |
| `scripts/42_hs_stroke_locked_external.py` | One-shot corrected-v6 HS-Stroke adapter and frozen-v4 external evaluation |
| `scripts/43_hs_stroke_post_lock_audit.py` | Post-lock quality-gate and probability-shift failure audit without retuning |
| `scripts/50_download_mind.py` | Official ScienceDB MIND V3 manifest reader and resumable profile downloader |
| `scripts/51_v7h_mind_benchmark.py` | Participant-isolated V7-H EEG-fNIRS benchmark, patient calibration, safety modes, and figures |
| `EEG_BCI_Project_RUN_GUI.command` | One-click macOS GUI launcher |
| `EEG_BCI_Project_RUN_FINAL_PIPELINE.command` | Reproduce the final cached experiment |
| `EEG_BCI_Project_VERIFY.command` | Syntax, GUI backend, data-link, and unit-test verification |
| `EEG_BCI_Project_RUN_MULTIMODAL_AUDIT.command` | Rebuild the EEG-EMG readiness audit and three figures |
| `EEG_BCI_Project_RUN_EMG_SUPERVISED_TRANSFER.command` | Inspect or run EMG-confirmed source to EEG-only target transfer |
| `EEG_BCI_Project_DOWNLOAD_WAY_EEG_GAL.command` | Resume, verify, and extract all official WAY-EEG-GAL participant archives |
| `EEG_BCI_Project_RUN_WAY_EEG_GAL_TRANSFER.command` | Prepare WAY source features and train/evaluate the EEG-only transfer student |
| `EEG_BCI_Project_RUN_MULTISOURCE_ENSEMBLE.command` | Reproduce the multi-source domain selector and presentation outputs |
| `EEG_BCI_Project_RUN_WAY_EEG_SSL.command` | Reproduce EEG-only WAY self-supervised pretraining and transfer |
| `EEG_BCI_Project_RUN_WAY_EEG_SSL_SEEDS.command` | Reproduce the five-seed neural stability confirmation |
| `EEG_BCI_Project_RUN_DECODER_BAKEOFF.command` | Reproduce the controlled EEG decoder architecture bakeoff |
| `EEG_BCI_Project_RUN_NEURAL_INTEGRATION.command` | Run five-seed MRCP-FBCNet fusion, calibration, patient adaptation, and temporal-safety preparation |
| `EEG_BCI_Project_PREPARE_PROSPECTIVE_STUDY.command` | Freeze v4 and prepare the prospective controller plus phase ablation |
| `EEG_BCI_Project_DOWNLOAD_MIND.sh` | Bash entry point for the analysis-ready MIND derivative download |
| `EEG_BCI_Project_DOWNLOAD_MIND.command` | One-click macOS MIND derivative downloader |
| `EEG_BCI_Project_RUN_V7H_MIND.sh` | Bash entry point for the V7-H MIND benchmark |
| `EEG_BCI_Project_RUN_V7H_MIND.command` | One-click macOS V7-H benchmark launcher |
| `EEG_BCI_Project_FINAL_REPORT.md` | Top-level shortcut to the calibrated report |
| `EEG_BCI_Project_RESULTS_OVERVIEW.png` | Top-level shortcut to the operating-point comparison |
| `EEG_BCI_Project_IMPLEMENTATION.md` | Top-level shortcut to the implementation guide |
| `EEG_BCI_Project_PRESENTATION_GUIDE.md` | Top-level shortcut to the complete project study and presentation guide |
| `EEG_BCI_Project_COMPARABLE_BCI_RESEARCH.md` | Top-level shortcut to the comparable-project methods and results review |
| `EEG_BCI_Project_PROSPECTIVE_CONTROLLER.md` | Top-level shortcut to the prospective controller protocol and current ablation |
| `EEG_BCI_Project_POSTER_PRESENTATION_GUIDE.md` | Poster-specific section walkthrough, technical glossary, safe claims, and likely Q&A |
| `EEG_BCI_Project_BME473_COURSE_GUIDE.md` | Top-level shortcut to the course-concept and V7/V7-H mapping guide |

## Final Code

`bci_pipeline/` contains the complete package used by the GUI and experiments.
The primary final modules are:

- `quality_preprocessing.py`: full-montage QC, repair, referencing, and branches
- `translational_dataset.py`: PhysioNet, OpenBMI, and Liu harmonization
- `translational_features.py`: MRCP, SMR, covariance, and quality features
- `graph_features.py`: fixed 10-10 motor-electrode graph, graph Fourier basis,
  smoothness, and graph-frequency energy features
- `translational_decoder.py`: domain-aware heads, stage fusion, calibration,
  availability gating, and patient adaptation
- `translational_metrics.py`: operating points, temporal safety metrics, and
  subject bootstrap intervals
- `privileged_transfer.py`: cross-fitted EMG confirmation and EEG-only transfer student
- `mrcp_eeg_emg.py`: synchronized MRCP EEG-EMG raw adapter and EMG-onset labeling
- `way_eeg_gal.py`: WAY continuous-MAT timing, montage harmonization, and privileged-source extraction
- `multisource_transfer.py`: group-held probability fusion and EEG-only domain-policy inference
- `way_eeg_ssl.py`: fixed-grid unlabeled EEG extraction from all WAY continuous series
- `eeg_ssl.py`: masked-reconstruction encoder, controlled fine-tuning, and portable EEG-only payload
- `seed_stability.py`: paired neural-seed summaries, intervals, participant aggregation, and criteria
- `eeg_bakeoff_models.py`: compact EEGNet, ShallowConvNet, FBCNet, and two-stream MRCP-FBCNet candidates
- `neural_integration.py`: validated five-seed MRCP-FBCNet loading and EEG-only probability ensembling
- `prospective_controller.py`: frozen-model hashing, cue-constrained temporal confirmation, contamination veto, replay metrics, and protocol preparation
- `hs_stroke.py`: corrected-v6 raw MAT discovery, full-session QC, and predeclared preparation/imagery windows
- `mind_download.py`: validated ScienceDB manifest parsing and derivative/analysis/full release profiles
- `mind_dataset.py`: paired MIND EEG-HbO-HbR adapter and interpretable dual-timescale features
- `v7h.py`: cross-fitted modality heads, quality-aware fusion, rapid patient calibration, and sequential confirmation
- `scripts/34_download_way_eeg_gal.py`: official Figshare manifest, resumable download, MD5 verification, and safe extraction
- `models/fbcsp.py`: hierarchical FBCSP baseline
- `gui_backend.py`: GUI inspection, experiment, and data-link backend

The complete `scripts/` directory is retained because the GUI exposes selected
historical audits and because those scripts document the path to the final
pipeline.

The first locked external result is retained in
`artifacts/hs_stroke_locked_external/`. It includes the immutable metrics,
execution receipt, predictions, QC audit, five figures, and a separate post-lock
failure analysis.

The V7-only graph/LDA ablation is available in:

```text
artifacts/gui_experiments/translational_v7_graph_lda
```

It contains the protected 73.6% V7 result, weighted shrinkage-LDA and graph
candidate metrics, controller rejection evidence, the general model bundle,
and eleven figures. The original `translational_v7_auxiliary_direction`
folder remains unchanged as the historical reference.

## Final Artifacts

`EEG_BCI_Project_Final_Results` points to:

```text
artifacts/gui_experiments/translational_v3_calibrated
```

It contains:

- `TRANSLATIONAL_V3_REPORT.md`
- `translational_v3_metrics.json`
- `translational_v3_models.joblib`
- `research_contract.json`
- `RESEARCH_CONTRACT.md`
- `subject_roles.json`
- Seven publication/presentation PNG figures

The original v3 feature cache is available through:

```text
artifacts/gui_experiments/translational_v3
```

The five-seed EEG-only controls and transfer comparison are available in:

```text
artifacts/way_eeg_only_ssl_seed_stability
```

This folder contains the WAY SSL stability report, the standalone original
PhysioNet/OpenBMI/Liu EEG-only report, five independent target-only decoders,
five SSL decoders, aggregate JSON, and four presentation figures.

The completed model bakeoff is available in:

```text
artifacts/eeg_decoder_model_bakeoff
```

It contains the strict target-development-only Conv1D control, five-seed neural
models and probabilities, deterministic references, aggregate and
participant-level metrics, a concise report, and five presentation figures.

The neural-integrated candidate is available in:

```text
artifacts/gui_experiments/translational_v4_neural_integrated
```

It contains the v4 report, complete metrics, deployment manifest, paired
participant analysis against v3, and eight presentation figures. The fixed v3
folder is retained because v4 has not yet been confirmed on a new external
cohort.

## Documentation

`docs/final/` contains the current project narrative and implementation
documents. `docs/research_archive/` contains the earlier benchmark reports,
feature audits, and design explorations.

The presentation-focused complete project guide is
`docs/final/EEG_BCI_PROJECT_PRESENTATION_GUIDE.md`. It explains the entire
pipeline, EEG physiology, feature representations, decoder logic, calibration,
patient adaptation, safety endpoints, acronyms, side experiments, and the
qualified interpretation of the current v4 result.

The course-concept mapping is
`docs/final/BME473_COURSE_CONCEPTS_AND_PROJECT_MAPPING.md`. It synthesizes the
six BME 473/ELE 573 lecture decks and identifies which signal-processing,
classification, optimization, validation, and graph-signal concepts are
implemented, diagnostic-only, or proposed as controlled future work.

The completed implementation and ablation decision are documented in
`docs/final/V7_GRAPH_LDA_COURSE_EXTENSION.md`.

The updated primary-study comparison is
`docs/final/COMPARABLE_BCI_PROJECTS_METHODS_RESULTS_REVIEW_2026.md`. It compares
the methods and results of cross-subject, stroke, asynchronous, and
brain-contingent rehabilitation BCIs, explains why their headline percentages
are not directly interchangeable, and translates the evidence into a frozen
external-test and cue-constrained controller plan.

The current multimodal research audit is
`docs/final/EEG_EMG_MULTIMODAL_PROJECT_AUDIT_2026.md`. Its reproducible output
is stored in `artifacts/multimodal_readiness/`.

The hybrid EEG-fNIRS extension is documented in
`docs/final/FNIRS_INTEGRATION_SYNOPSIS_V7.md`. It reviews the MIND and Yin
papers, comparable studies, modality-specific quality control, a dual-timescale
V7-H controller, public-data validation, prospective acquisition, GUI support,
and locked advancement criteria. Its first MIND implementation is documented
in `docs/final/V7H_MIND_IMPLEMENTATION.md`. The implementation does not alter
the current EEG-only V7 result, and no real MIND metric is claimed before the
downloaded-data benchmark is run.

The decoder architecture review and completed EEGNet, ShallowConvNet, FBCNet,
and MRCP-FBCNet bakeoff are documented in
`docs/final/DECODER_MODEL_RESEARCH_RECOMMENDATION.md`. MRCP + FBCNet was the
only candidate to pass every predeclared advancement check.

The implementation and qualified v4 interpretation are documented in
`docs/final/TRANSLATIONAL_V4_NEURAL_INTEGRATION.md`.

The synchronized MRCP EEG-EMG transfer experiment is documented in
`docs/final/EMG_SUPERVISED_EEG_TRANSFER_PROTOCOL.md`; its report, metrics,
EEG-only student model, and three figures are in
`artifacts/emg_supervised_transfer/`.

## External Data

The following are links, not duplicated files:

- `data_links/EEG_BCI_Project_PhysioNet_EEGMMIDB`
- `data_links/EEG_BCI_Project_OpenBMI`
- `data_links/EEG_BCI_Project_Liu_Stroke`

Keeping the datasets external avoids adding approximately 41 GB to this
consolidated folder.

WAY-EEG-GAL is intentionally stored locally under `WAY-EEG-GAL/` rather than
linked. Its 12 official archives and extracted MATLAB files occupy about 20 GB;
the verified inventory is recorded in `WAY-EEG-GAL/DOWNLOAD_AUDIT.md`.
