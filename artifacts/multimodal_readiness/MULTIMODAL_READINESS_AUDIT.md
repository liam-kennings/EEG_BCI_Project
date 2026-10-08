# Multimodal Motor-Intent Readiness Audit

## Current Result

- Patient-adapted conservative balanced accuracy: **67.9%**.
- Intent sensitivity: **41.3%**.
- Cue-locked specificity: **94.5%**.
- Conditional direction balanced accuracy: **71.0%**.
- Status: development-observed; continuous false activations per minute are not evaluated.

## Readiness Summary

- Ready: **3** checks.
- Prepared: **2** checks.
- Gaps: **5** checks.

| Check | Status | Evidence | Required action |
|---|---|---|---|
| Subject-isolated development roles | READY | 187 subjects are assigned by subject role. | Preserve subject and session grouping in every new adapter. |
| Nested controller selection | READY | Fusion/calibration and operating-point selection use separate controller subroles. | Keep all fusion, calibration, and threshold choices outside observed test subjects. |
| Full-montage QC before reference | READY | 492 recordings accepted; 1 excluded. | Continue reporting exclusions and sensitivity to QC thresholds. |
| Existing synchronized EMG source | PREPARED | OpenBMI loader recognizes four EMG channels, but v3 stores only aggregate auxiliary quality. | Retain EMG waveforms and derive silence, onset, RMS, and waveform-length labels. |
| External multimodal teacher catalog | PREPARED | 7 EEG-EMG sources are cataloged with modality-specific roles. | Implement the MRCP EEG-EMG adapter first; do not pool execution labels with imagery. |
| Modality-specific auxiliary processing | GAP | EMG and EOG are currently merged into generic auxiliary peak-to-peak/RMS features. | Split EMG, EOG, and device channels before feature extraction and gating. |
| Robustly scaled quality gate | GAP | The current heterogeneous-distance threshold is 301,511.3, indicating scale domination. | Fit median/IQR scaling per feature and report per-domain rejection rates. |
| Causal deployable preprocessing | GAP | Final reported profile uses phase=zero filtering. | Re-run the locked candidate with causal filters and measure added latency/performance loss. |
| Continuous no-control safety | GAP | Not evaluated: no hard-negative manifest was supplied. Cue-locked window specificity is not converted into a deployment false-activation rate. | Add continuous idle, gaze, speech, jaw, posture, and device-motion hard negatives. |
| Locked external confirmation | GAP | The final cohort remains development-observed. | Freeze v4, then evaluate once on untouched subjects/sessions. |

## Interpretation

The next improvement should not be a single fused classifier trained on mixed imagery and execution labels. Build an EEG-only cortical-intent track, an EMG/kinematic teacher and validation track, and an optional hybrid control track. Require EEG-only, EMG-only, and fused ablations for every multimodal claim.

See `docs/final/EEG_EMG_MULTIMODAL_PROJECT_AUDIT_2026.md` for the cited research comparison and implementation sequence.

## Figures

- `01_multimodal_readiness_scorecard.png`
- `02_published_results_context.png`
- `03_multimodal_dataset_matrix.png`
