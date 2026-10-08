# Documentation Index

Status labels as of 2026-10-07, after the independent audit
(`../EEG_BCI_Project_AUDIT_2026-10-07.md`). Where documents disagree, the
**Current** documents take precedence. Documents written before the audit
report V7 as a window-pooled 73.6% balanced accuracy and do not include the
confound controls.

## Current

| Document | Purpose |
|---|---|
| [`final/PROJECT_STATUS_AND_RESULTS.md`](final/PROJECT_STATUS_AND_RESULTS.md) | **Start here.** Canonical methods, results, and limitations after the audit |
| [`final/CONFOUND_CONTROL_FINDINGS.md`](final/CONFOUND_CONTROL_FINDINGS.md) | Whether the rest-vs-intent contrast survives cue and trial-phase controls |
| [`final/PREREGISTRATION_DRAFT_EXTERNAL_TEST.md`](final/PREREGISTRATION_DRAFT_EXTERNAL_TEST.md) | Draft protocol for the next locked external test |
| [`final/TRANSLATIONAL_V3_IMPLEMENTATION.md`](final/TRANSLATIONAL_V3_IMPLEMENTATION.md) | Base preprocessing, feature, and decoder pipeline that V5-V7 extend |
| [`final/V7_GRAPH_LDA_COURSE_EXTENSION.md`](final/V7_GRAPH_LDA_COURSE_EXTENSION.md) | V7 graph and shrinkage-LDA extension and its rejection by the controller |
| [`final/BME473_COURSE_CONCEPTS_AND_PROJECT_MAPPING.md`](final/BME473_COURSE_CONCEPTS_AND_PROJECT_MAPPING.md) | Course concepts mapped to the pipeline |
| [`final/EEG_BCI_DATASET_CITATIONS.md`](final/EEG_BCI_DATASET_CITATIONS.md) | Dataset citations |
| [`final/COMPARABLE_BCI_PROJECTS_METHODS_RESULTS_REVIEW_2026.md`](final/COMPARABLE_BCI_PROJECTS_METHODS_RESULTS_REVIEW_2026.md) | Literature comparison. Study details remain useful; the V7 numbers in it predate the audit |

## Supporting experiments

All of these were scored on the same `development_observed` participants as
V7, so none of them is an independent confirmation.

| Document | Experiment |
|---|---|
| [`final/DECODER_MODEL_RESEARCH_RECOMMENDATION.md`](final/DECODER_MODEL_RESEARCH_RECOMMENDATION.md) | Five-seed neural decoder bakeoff (rankings used development_observed) |
| [`final/TRANSLATIONAL_V4_NEURAL_INTEGRATION.md`](final/TRANSLATIONAL_V4_NEURAL_INTEGRATION.md) | V4 neural integration |
| [`final/EMG_SUPERVISED_EEG_TRANSFER_PROTOCOL.md`](final/EMG_SUPERVISED_EEG_TRANSFER_PROTOCOL.md) | EMG-supervised transfer (CI crosses zero) |
| [`final/WAY_EEG_GAL_PRIVILEGED_TRANSFER.md`](final/WAY_EEG_GAL_PRIVILEGED_TRANSFER.md) | WAY-EEG-GAL privileged transfer |
| [`final/WAY_EEG_ONLY_SELF_SUPERVISED_TRANSFER.md`](final/WAY_EEG_ONLY_SELF_SUPERVISED_TRANSFER.md) | WAY self-supervised pretraining |
| [`final/MULTISOURCE_PRIVILEGED_EEG_ENSEMBLE.md`](final/MULTISOURCE_PRIVILEGED_EEG_ENSEMBLE.md) | Multi-source ensemble |
| [`final/PROSPECTIVE_CUE_CONSTRAINED_CONTROLLER.md`](final/PROSPECTIVE_CUE_CONSTRAINED_CONTROLLER.md) | Frozen v4 cue-constrained controller (not run prospectively) |
| [`final/FNIRS_INTEGRATION_SYNOPSIS_V7.md`](final/FNIRS_INTEGRATION_SYNOPSIS_V7.md), [`final/V7H_MIND_IMPLEMENTATION.md`](final/V7H_MIND_IMPLEMENTATION.md) | V7-H EEG-fNIRS design. **Never run on real data** |

## Poster and presentation (pre-audit)

These describe the August poster. Their headline numbers are window-pooled and
they say no locked external cohort existed, which was already untrue for
HS-Stroke. Update them before reuse.

- [`final/EEG_BCI_V7_POSTER_CONTENT.md`](final/EEG_BCI_V7_POSTER_CONTENT.md)
- [`final/EEG_BCI_POSTER_PRESENTATION_GUIDE.md`](final/EEG_BCI_POSTER_PRESENTATION_GUIDE.md)
- [`final/EEG_BCI_POSTER_REFERENCES_DROP_IN.md`](final/EEG_BCI_POSTER_REFERENCES_DROP_IN.md)
- [`final/EEG_BCI_PROJECT_PRESENTATION_GUIDE.md`](final/EEG_BCI_PROJECT_PRESENTATION_GUIDE.md)

## Historical design and research notes

Kept for provenance. They describe earlier versions and plans.

- [`final/BCI_FULL_WORKFLOW_RESEARCH.md`](final/BCI_FULL_WORKFLOW_RESEARCH.md)
- [`final/BCI_SYSTEM_ARCHITECTURE.md`](final/BCI_SYSTEM_ARCHITECTURE.md)
- [`final/DUAL_PIPELINE_PROJECT.md`](final/DUAL_PIPELINE_PROJECT.md)
- [`final/EEG_MOTOR_INTENT_RESEARCH_PROJECT.md`](final/EEG_MOTOR_INTENT_RESEARCH_PROJECT.md)
- [`final/EXTERNAL_DATASETS_AND_RESEARCH_CONSOLE.md`](final/EXTERNAL_DATASETS_AND_RESEARCH_CONSOLE.md)
- [`final/FULL_PROJECT_BCI_AUDIT_2026.md`](final/FULL_PROJECT_BCI_AUDIT_2026.md) (July self-audit)
- [`final/EEG_EMG_MULTIMODAL_PROJECT_AUDIT_2026.md`](final/EEG_EMG_MULTIMODAL_PROJECT_AUDIT_2026.md)
- [`research_archive/`](research_archive/): earlier experiment reports (v1-v2 era)
