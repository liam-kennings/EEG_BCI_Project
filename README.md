# EEG BCI Project

Binary REST-vs-INTENT EEG detection for cue-guided rehabilitation. A known cue
specifies the requested movement; EEG confirms whether motor intent follows.

## Current status (2026-10-07)

**Start with [`docs/final/PROJECT_STATUS_AND_RESULTS.md`](docs/final/PROJECT_STATUS_AND_RESULTS.md).**
It is the canonical summary after the independent audit
([`EEG_BCI_Project_AUDIT_2026-10-07.md`](EEG_BCI_Project_AUDIT_2026-10-07.md)).
[`docs/README.md`](docs/README.md) labels every other document as current,
supporting, pre-audit poster material, or historical.

- **V7, participant level:** **66.0% balanced accuracy** (95% CI 62.3-69.6%)
  across 38 evaluation participants, with 21/38 above their individual chance
  threshold. Patient calibration adds +6.5 points per participant (95% CI
  +3.7 to +9.3).
- **V7, window-pooled:** 73.6% balanced accuracy, 53.5% sensitivity, 93.7%
  specificity. This is the August poster number. It is dominated by 6 OpenBMI
  participants who supply 77.5% of evaluation windows.
- **Evaluation set:** the 38 participants have been scored by every version
  since v3 and were used to rank neural architectures. They are a validation
  set, not a test set.
- **Confound controls**
  ([findings](docs/final/CONFOUND_CONTROL_FINDINGS.md)):
  - **PhysioNet:** about 0.70 AUROC of imagery-related signal survives
    pre-cue rest, later windows, and causal filters, and beats an
    amplitude-only model by 0.12-0.16.
  - **OpenBMI:** separability is matched by amplitude alone.
  - **Liu stroke:** physiology does no better than amplitude or EOG.
- **Locked external tests:** both failed. HS-Stroke (v4) and Cho2017 (V7)
  windows were 100% rejected, primarily by a quality-gate defect that is now
  fixed. Cho2017 zero-shot AUROC was 0.575. Its "adapted 0.925" is a
  recording-identity artifact, not evidence.
- **Safety:** specificity is per cue-locked window. Continuous false
  activations per minute have never been measured.

## Quick start

```bash
./EEG_BCI_Project_SETUP_ENV.command              # one-time: build the project .venv
./EEG_BCI_Project_VERIFY.command                 # syntax, GUI check, model hashes, 115 tests (~10 s)
./EEG_BCI_Project_RUN_FINAL_PIPELINE.command     # reproduce V7 with participant-level metrics (~2 min)
./EEG_BCI_Project_RUN_CONFOUND_CONTROLS.command  # confound-control benchmark (~15 min first run)
./EEG_BCI_Project_RUN_GUI.command                # research console
```

### Environment

- **Interpreter.** Every launcher sources `scripts/project_python.sh`, which
  picks one interpreter, in this order: `$EEG_BCI_PYTHON`, then `.venv/`, then
  the framework Python 3.12, then `python3`.
- **Setup.** `EEG_BCI_Project_SETUP_ENV.command` creates `.venv` with access to
  the framework Python 3.12 packages, which provide scikit-learn 1.6.1 for the
  saved `.joblib` models and PyTorch with Metal. It pins scipy 1.16.3, because
  scipy 1.15.2 does not load on macOS 27.
- **Neural compute.** Neural steps default to `--device auto` / `--neural-device
  auto` (CUDA, then Metal, then CPU). Completed seed models are reused unless
  `--force` is passed.

### Version control

The project is a git repository.

- **Branches.** `main` holds the pre-audit baseline snapshot, and
  `audit-fixes` holds the audit follow-up.
- **What is not tracked.** Source datasets (about 67 GB), cached `.npz` window
  arrays, third-party PDFs, and poster design binaries (`.pptx`, `.pxd`) are
  excluded by `.gitignore`.

## Main results and reports

| What | Where |
|---|---|
| Canonical status and results | `docs/final/PROJECT_STATUS_AND_RESULTS.md` (also `EEG_BCI_Project_FINAL_REPORT.md`) |
| V7 reproduction: report, participant-level metrics, per-window predictions | `artifacts/gui_experiments/translational_v7_graph_lda_reproduction/` (also `EEG_BCI_Project_Final_Results`) |
| Original V7 run (pooled metrics only) | `artifacts/gui_experiments/translational_v7_graph_lda/` |
| Confound controls | `artifacts/confound_controls/`, `docs/final/CONFOUND_CONTROL_FINDINGS.md` |
| HS-Stroke locked external (v4) | `artifacts/hs_stroke_locked_external/` |
| Cho2017 locked external (V7) | `artifacts/cho2017_locked_external_v7/`, `artifacts/cho2017_post_lock_quality_audit/` |
| Next locked test (draft protocol) | `docs/final/PREREGISTRATION_DRAFT_EXTERNAL_TEST.md` |
| Base pipeline implementation (v3; V7 extends it) | `docs/final/TRANSLATIONAL_V3_IMPLEMENTATION.md` (also `EEG_BCI_Project_IMPLEMENTATION.md`) |
| Course-concept mapping | `docs/final/BME473_COURSE_CONCEPTS_AND_PROJECT_MAPPING.md` |
| Poster and presentation material (pre-audit numbers) | `docs/final/EEG_BCI_V7_POSTER_CONTENT.md`, `artifacts/poster_v7/` |

### V7 binary contract

- **Output.** The device-facing output is strictly `rest` versus `intent`.
- **Calibration.** Recording 0 supplies 10 rest, 10 left, and 10 right
  calibration windows. Only later recordings are evaluated.
- **Direction.** An internal auxiliary model may use left/right cue labels.
  Its probabilities are collapsed to `P(left) + P(right)`, and direction is
  never exposed as a command.
- **Auxiliary weight.** Controller selection kept a weight of 0.25 for
  PhysioNet and 0.50 for OpenBMI, and selected 0.00 for Liu stroke.
- **Graph/LDA extension.** It received zero fusion weight and was rejected
  for every patient adapter.
- **Model bundle.** The portable bundle is in
  `artifacts/gui_experiments/translational_v7_graph_lda/general_model_bundle`.

### Quality gate

The fix is in `QualityAvailabilityGate` (`bci_pipeline/translational_decoder.py`).

- **The defect.** Scoring version 1 floored the zero spread of the binary
  indicators `auxiliary_available` and `csd_available` at 1e-6. Any dataset
  without auxiliary channels or CSD was therefore rejected outright.
- **The fix.** Version 2, the default for new fits, excludes indicator and
  constant features from the distance and reports them via
  `compatibility_report()`. It also offers a label-free within-recording mode.
- **Frozen models are unchanged.** Pickled gates keep version 1 scoring.
  `scripts/30_translational_motor_intent_v3.py --legacy-quality-gate`
  reproduces historical runs exactly.

### Locked external tests

**HS-Stroke (14 chronic-stroke patients, frozen v4).**

- The gate rejected every window.
- With the fixed gate, 95.1% of windows are available.
- The descriptive intent AUROC before gating was 0.524. The contrast there was
  cued preparation vs imagery, so both classes had stimuli.
- HS-Stroke has now been inspected and cannot serve as a locked test again.
- The one-shot evaluator `scripts/42_hs_stroke_locked_external.py` refuses to
  overwrite its result.

**Cho2017 (32 retained healthy participants, V7).**

- The gate rejected every window, for two reasons:
  - the indicator defect, and
  - Cho2017 amplitude features about 20× the training data (median RMS 105.7
    vs 5.2), which is unresolved and may be a unit issue.
- Zero-shot AUROC was **0.575**.
- The reported "adapted AUROC 0.925" and the post-hoc 72.0% BA are not evidence.
  Rest windows come from a separately recorded and separately preprocessed
  resting file, so a feature that is constant within each recording separates
  the classes at about 0.87 AUROC on its own.

### Open the GUI

Double-click:

```text
EEG_BCI_Project_RUN_GUI.command
```

Or run:

```bash
cd EEG_BCI_Project
./EEG_BCI_Project_RUN_GUI.command
```

Choose `Translational v7 graph/LDA-enhanced intent (recommended)` in the
Dataset Experiments tab. V6 and the fixed legacy references remain available
in the same menu.

## Supporting experiments

Every experiment below was scored on the same validation participants as V7.
Treat each as exploratory.

### Run the EEG decoder model bakeoff

```bash
./EEG_BCI_Project_RUN_DECODER_BAKEOFF.command
```

The completed matched five-seed comparison evaluates EEGNet, ShallowConvNet,
FBCNet, a two-stream MRCP + FBCNet model, corrected target-only Conv1D, WAY
SSL, Riemannian, and calibrated-fusion references. **MRCP + FBCNet was the only
candidate to pass every predeclared advancement check**, reaching **62.66%
balanced accuracy**, **31.29% sensitivity**, **94.03% specificity**, and
**0.804 AUROC**. It improved equal-participant balanced accuracy by **+3.10
points** over target-only Conv1D (95% participant-bootstrap interval **+1.50 to
+4.73**) and helped **76.3%** of participants.

The workflow is also available in the GUI as `EEG decoder model bakeoff`.
Research rationale and the completed interpretation are documented in
`docs/final/DECODER_MODEL_RESEARCH_RECOMMENDATION.md`; the generated report and
five presentation figures are in `artifacts/eeg_decoder_model_bakeoff/`.
This is a base intent-gate architecture audit; it does not replace the
patient-adapted V7 result. The ranking used the V7 evaluation participants, and
the top two architectures (MRCP + FBCNet, ShallowConvNet) were then built into
V5-V7, so V7's evaluation set also served for architecture selection.

### Run the neural-integrated pipeline

```bash
./EEG_BCI_Project_RUN_NEURAL_INTEGRATION.command
```

This averages the five selected MRCP + FBCNet seeds, applies controller-only
stage fusion and probability calibration, performs early-recording patient
adaptation, retains the quality/OOD gate, and prepares two-window temporal
confirmation. Results are written to
`artifacts/gui_experiments/translational_v4_neural_integrated/`.

### Prepare the frozen prospective controller

```bash
./EEG_BCI_Project_PREPARE_PROSPECTIVE_STUDY.command
```

This freezes and hashes the v4 model plus all five MRCP-FBCNet seeds, exports a
cue-constrained controller, continuous no-control and locked-external protocol,
replay template, synthetic state-machine figure, and a development-only MRCP
phase ablation. Device output remains disabled and no external result is
claimed. Outputs are written to `artifacts/prospective_cue_controller/` and
`artifacts/mrcp_phase_ablation/`.

### Run the EEG-EMG research audit

```bash
./EEG_BCI_Project_RUN_MULTIMODAL_AUDIT.command
```

The cited audit is in
`docs/final/EEG_EMG_MULTIMODAL_PROJECT_AUDIT_2026.md`. Reproducible checks and
three presentation figures are written to `artifacts/multimodal_readiness/`.

### Run EMG-supervised EEG transfer

```bash
./EEG_BCI_Project_RUN_EMG_SUPERVISED_TRANSFER.command
```

With no arguments, this detects the downloaded MRCP EEG-EMG dataset, prepares
the synchronized-source and pure-EEG target caches when needed, and runs the
validation-selected transfer comparison. The method and leakage controls are
documented in `docs/final/EMG_SUPERVISED_EEG_TRANSFER_PROTOCOL.md`; generated
metrics, reports, models, and figures are in
`artifacts/emg_supervised_transfer/`.

The same experiment is available in the GUI under Dataset Experiments by
selecting `EMG-supervised EEG transfer`; the MRCP EEG-EMG root is prefilled
with the downloaded project folder.

Current exploratory development-observed result:

- Target-only EEG: **53.0%** balanced accuracy, **10.7%** sensitivity, **95.3%** specificity.
- EMG-supervised EEG: **53.3%** balanced accuracy, **11.5%** sensitivity, **95.1%** specificity.
- Stroke subset: **56.1% to 57.6%** balanced accuracy and **28.0% to 30.8%** sensitivity.
- Equal-participant mean change: **+0.4 points**, 95% bootstrap CI **-0.1 to +0.9** points.

The interval crosses zero, so this supports feasibility but not a claim that
EMG supervision improves general EEG intent detection.

### Download WAY-EEG-GAL

The official 12-participant WAY-EEG-GAL release can be downloaded and
extracted with:

```bash
./EEG_BCI_Project_DOWNLOAD_WAY_EEG_GAL.command
```

The launcher reads the current Figshare collection API, resumes partial
archives, verifies the published file size and MD5 for every participant, and
extracts into `WAY-EEG-GAL/participants/`. The official archive set is 9.63 GiB
compressed; the extracted local MATLAB release occupies approximately 9.8 GB,
or about 20 GB when the verified ZIP archives are retained.

### Train the EEG decoder with WAY-EEG-GAL

```bash
./EEG_BCI_Project_RUN_WAY_EEG_GAL_TRANSFER.command
```

This extracts pre-movement EEG using kinematic hand-start timing, uses
post-onset EMG only to verify source examples, and trains an EEG-only intent
student against the existing PhysioNet/OpenBMI/Liu target contract. It is also
available in the GUI as `WAY-EEG-GAL privileged EEG transfer`.

The full local result, leakage controls, and interpretation limits are in
`docs/final/WAY_EEG_GAL_PRIVILEGED_TRANSFER.md`. Outputs are written to
`artifacts/way_eeg_gal_transfer/`. The current run improved target AUROC from
**0.683 to 0.697** but left conservative pooled balanced accuracy essentially
flat (**53.02% to 52.94%**); Liu stroke and PhysioNet improved while OpenBMI
declined.

### Run the multi-source privileged ensemble

```bash
./EEG_BCI_Project_RUN_MULTISOURCE_ENSEMBLE.command
```

This compares the target-only, MRCP-assisted, and WAY-assisted EEG students,
then uses group-held controller subjects to select a separate method and
threshold for Liu, OpenBMI, and PhysioNet. The same workflow is available in
the GUI as `Multi-source privileged EEG ensemble`.

The development-observed result was **52.98% balanced accuracy**, **11.00%
sensitivity**, **94.96% specificity**, and **0.698 AUROC**. The equal-participant
change was **+0.92 points** with a 95% bootstrap interval of **+0.37 to +1.50
points**. Full interpretation is in
`docs/final/MULTISOURCE_PRIVILEGED_EEG_ENSEMBLE.md`.

### Run WAY EEG-only self-supervised transfer

```bash
./EEG_BCI_Project_RUN_WAY_EEG_SSL.command
```

This pretrains a masked-reconstruction encoder on all 120 continuous WAY EEG
series without reading EMG, kinematics, force, events, or WAY class labels. It
then fine-tunes on the original PhysioNet/OpenBMI/Liu development subjects and
compares against an architecture-matched random CNN.

Across five neural seeds, SSL reached **56.29% mean balanced accuracy**,
**19.59% sensitivity**, **93.00% specificity**, and **0.683 AUROC**. Every seed
improved over its matched random CNN; the paired gain was **+2.81 points**
(95% seed-level t interval **+1.77 to +3.85**). Run the confirmation with
`./EEG_BCI_Project_RUN_WAY_EEG_SSL_SEEDS.command` and see
`docs/final/WAY_EEG_ONLY_SELF_SUPERVISED_TRANSFER.md`.

The same controlled run also produces a target-label-only paired neural
control. Without WAY pretraining, that CNN reached **53.49% balanced accuracy**,
**15.60% sensitivity**, and **91.37% specificity**. Its normalization scales
were shared with the unlabeled WAY condition, so it is not the strict
target-only preprocessing baseline. See
`artifacts/way_eeg_only_ssl_seed_stability/ORIGINAL_EEG_ONLY_FIVE_SEED_REPORT.md`.

### Download MIND and run V7-H (not yet run on real data)

No `MIND/` download or `artifacts/v7h_mind/` output exists yet. MIND is a
healthy 4-direction right-arm imagery dataset, and the slow fNIRS response
(peak 5-8 s) cannot speed a sub-second trigger, so this extension is
deprioritized.

V7-H is the paired EEG-fNIRS extension of V7. The default Bash launcher reads
the official ScienceDB V3 manifest and downloads only the processed EEG, HbO,
and HbR derivatives needed for the first benchmark:

```bash
cd "/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project"
bash EEG_BCI_Project_DOWNLOAD_MIND.sh
```

The download is resumable and writes its manifest and progress state under
`MIND/`. To validate the workflow with a smaller subset first:

```bash
bash EEG_BCI_Project_DOWNLOAD_MIND.sh --subjects 3-7
```

The current V3 derivative manifest yields 27 usable complete EEG/HbO/HbR
participant pairs: the adapter reports and skips incomplete `sub-02` and the
documented corrupt fNIRS derivative for `sub-13`.

The complete MIND release is approximately **93.1 GiB**. It is not required
for V7-H, but it can be selected explicitly with:

```bash
bash EEG_BCI_Project_DOWNLOAD_MIND.sh --profile full
```

After the derivative download completes, run the participant-isolated hybrid
benchmark with:

```bash
bash EEG_BCI_Project_RUN_V7H_MIND.sh
```

V7-H compares EEG-only, fNIRS-only, population late fusion, rapid
patient-adapted fusion, and sequential EEG-proposal/fNIRS-confirmation modes.
It reserves recording 0 for calibration and evaluates later recordings. The
same workflow is available in the GUI as `V7-H hybrid EEG-fNIRS (MIND)`.
Implementation details and output definitions are in
`docs/final/V7H_MIND_IMPLEMENTATION.md`. No MIND performance result should be
reported until the real-data command completes; `--synthetic` is only a
software smoke test.

### Reference runs

- `EEG_BCI_Project_RUN_V3_REFERENCE.command` reruns the fixed physiology-only
  v3 reference into `artifacts/gui_experiments/translational_v3_calibrated`.
  On the window-pooled metric it scored 67.9% BA, 41.3% sensitivity, and 94.5%
  specificity.

## Folder layout

| Location | Purpose |
|---|---|
| `bci_pipeline/` | Signal processing, feature extraction, decoders, calibration, metrics, and confound controls |
| `scripts/` | Experiment history and entry points (`30_` = V7 pipeline, `54_` = confound controls, `project_python.sh` = interpreter selection) |
| `tests/` | Automated tests (115) |
| `docs/` | Documentation; see `docs/README.md` for status labels |
| `artifacts/gui_experiments/` | V3-V7 runs, including the V7 reproduction with participant-level metrics |
| `artifacts/confound_controls/` | Cue and trial-phase control benchmark |
| `artifacts/hs_stroke_locked_external/`, `artifacts/cho2017_*` | Locked external tests and post-lock audits |
| `artifacts/eeg_decoder_model_bakeoff/`, `artifacts/way_*`, `artifacts/emg_supervised_transfer/`, `artifacts/multisource_privileged_ensemble/`, `artifacts/multimodal_readiness/` | Supporting experiments |
| `artifacts/prospective_cue_controller/`, `artifacts/mrcp_phase_ablation/` | Frozen v4 controller preparation and phase ablation |
| `artifacts/poster_v7/` | Poster figures and their generator outputs |
| `data_links/`, `files/`, `Liu acute-stroke dataset` | Dataset links (machine-specific; not tracked) |
| `WAY-EEG-GAL/` | Local WAY-EEG-GAL release (about 20 GB; only the audit and manifest are tracked) |
| `30747806/` | HS-Stroke release (about 13 GB; only the project's intake and contract JSON are tracked) |
| `files/external/cho2017/` | Cho2017 release (about 33 GB; not tracked) |

## Moving This Folder

The code, documentation, models, metrics, figures, and local 20 GB
WAY-EEG-GAL release can be moved together.
The dataset links will need to be recreated on another computer or after the
source data are moved:

```bash
./EEG_BCI_Project_CONFIGURE_DATA.command \
  "/path/to/eegmmidb/1.0.0" \
  "/path/to/openbmi/nm000338" \
  "/path/to/liu-dataset" \
  "/optional/path/to/translational_v3-cache-folder"
```

See `data_links/README.md` for details.

On another machine, run `./EEG_BCI_Project_SETUP_ENV.command` to create the
`.venv` (it installs the full `requirements.txt` when the framework Python 3.12
stack is absent; add `--with-optional` for PyTorch). Tracked symlinks inside the
project are relative. The dataset links in `data_links/` are machine-specific
and not tracked.
