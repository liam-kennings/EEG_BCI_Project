# EEG BCI Project

## Neural Compute Device

Neural experiments default to `--device auto`, which selects CUDA first, then
Apple Metal (`mps`), and finally CPU. The WAY EEG-only experiments and decoder
bakeoff accept `--device auto|cpu|mps|cuda`; translational v4 inference uses the
equivalent `--neural-device` option. Classical MNE, SciPy, NumPy, covariance,
and scikit-learn processing remains CPU-based.

Completed seed results are reused by default for reproducibility. Pass `--force`
to the five-seed stability or decoder-bakeoff command when a deliberate retrain
on the newly selected device is required. The HS Stroke locked external script
remains explicitly CPU-bound so its completed locked result is not altered.
On Apple Silicon, the GUI exposes a `Neural compute` selector and defaults to
`mps` when Metal is available. Enable `Rebuild cached features` to retrain and
replace completed seed models on the selected neural device.

Use `EEG_BCI_Project_RUN_GUI.command` to launch the console. On this project
machine, that launcher prefers the Python 3.12 installation matching the pinned
scikit-learn 1.6.1 artifacts and the working PyTorch Metal backend. Set
`EEG_BCI_PYTHON=/path/to/python` only when intentionally using another complete
environment.

## Current Patient-Adaptive Model

The recommended development workflow is
`Translational v7 graph/LDA-enhanced intent`. Its device-facing output is
strictly binary `rest` versus `intent`. During training and rapid calibration,
an internal auxiliary model may use the known left/right cue labels, but its
probabilities are collapsed to `P(left) + P(right)` and direction is never
exposed as a command. Recording 0 supplies 10 rest, 10 left, and 10 right
calibration windows; only later recordings are evaluated.

The development-observed patient-adapted conservative result is **73.6%
balanced accuracy**, **53.5% sensitivity**, and **93.7% specificity**. The
sensitivity-targeted point reaches **74.1% balanced accuracy**, **58.5%
sensitivity**, and **89.6% specificity**. Controller selection retained an
auxiliary weight of 0.25 for PhysioNet and 0.50 for OpenBMI, but correctly
selected 0.00 for Liu stroke data. This is a modest conservative improvement
over v5 (**73.3% BA**) and v6 (**71.9% BA**), not external confirmation. V6
remains available when its simpler 20-window binary calibration contract is
preferred. Continuous false commands per minute remain unevaluated. Cho2017
now supplies an external continuous-rest window test, but temporal confirmation
and commands per minute are still not measured.

The V7-only course-concept extension adds a fixed seven-electrode graph,
64 graph-spectral features, weighted OAS shrinkage LDA, and development-fitted
PCA/K-means diagnostics. Controller validation assigned zero operational
fusion weight to the graph and LDA candidates and rejected graph augmentation
for all patient adapters. The protected V7 result therefore remains 73.6% BA.
This negative ablation is retained because added feature complexity did not
improve later-recording generalization. Shrinkage LDA remains a useful
interpretable reference at 62.8% raw BA and 0.803 AUROC.

The portable model package is under
`artifacts/gui_experiments/translational_v7_graph_lda/general_model_bundle`.
It contains the population controller, fingerprinted neural seed models,
preprocessing and channel contracts, operating thresholds, and the selected
rapid-calibration configuration. The API class
`RapidAuxiliaryIntentPatientAdapter` can be fitted from labeled rest/left/right
calibration windows and serialized with joblib for later patient-specific
binary inference. A known rehabilitation cue supplies the requested movement
while EEG confirms whether intent is present.

This folder is the consolidated home for the final weak motor-intent EEG
project. It contains the runnable pipeline, GUI, tests, documentation, trained
model bundle, metrics, and presentation figures.

The large source datasets are not duplicated. The `files/`,
`Liu acute-stroke dataset`, and `data_links/` entries point to the existing
local datasets. The requested WAY-EEG-GAL release is the exception: its
verified archives and extracted MATLAB files are stored directly in
`WAY-EEG-GAL/` and occupy approximately 20 GB.

## Start Here

### Learn the complete project for a presentation

- Course-to-project technical study guide:
  `EEG_BCI_Project_BME473_COURSE_GUIDE.md`
- Connects the supplied BME 473/ELE 573 lectures on baseline normalization,
  Hilbert analysis, CSP, LDA, Bayesian classification, clustering, neural
  networks, optimization, and graph signal processing to V7/V7-H.
- Poster-specific section walkthrough, talk track, glossary, and likely Q&A:
  `EEG_BCI_Project_POSTER_PRESENTATION_GUIDE.md`
- Top-level shortcut: `EEG_BCI_Project_PRESENTATION_GUIDE.md`
- Presentation and study guide:
  `docs/final/EEG_BCI_PROJECT_PRESENTATION_GUIDE.md`
- Includes the full pipeline, physiology, preprocessing, feature and decoder
  explanations, acronyms, metrics, side experiments, likely questions, and
  presentation-safe interpretation of the current results.
- Comparable-project research review:
  `docs/final/COMPARABLE_BCI_PROJECTS_METHODS_RESULTS_REVIEW_2026.md`
- Top-level research shortcut:
  `EEG_BCI_Project_COMPARABLE_BCI_RESEARCH.md`
- Compares decoder, transfer, asynchronous-control, and rehabilitation studies
  by cohort, method, validation contract, and result, then defines the next
  recommended experiment for this project.
- Prospective frozen-controller guide:
  `docs/final/PROSPECTIVE_CUE_CONSTRAINED_CONTROLLER.md`
- Top-level prospective shortcut:
  `EEG_BCI_Project_PROSPECTIVE_CONTROLLER.md`
- Proposed EEG-fNIRS extension of V7:
  `docs/final/FNIRS_INTEGRATION_SYNOPSIS_V7.md`
- Reviews the supplied MIND and Yin studies, comparable hybrid BCI evidence,
  a dual-timescale V7-H architecture, public-data benchmark, fNIRS quality
  control, prospective acquisition protocol, GUI changes, and advancement
  criteria.

### View the final result

- Poster clinical-motivation panel and recommended abstract replacement:
  `artifacts/poster_v7/why_project/00_why_this_project.png`
- Poster graphical abstract:
  `artifacts/poster_v7/01_v7_poster_graphical_abstract.png`
- Poster dataset and V7 pipeline:
  `artifacts/poster_v7/02_v7_poster_dataset_pipeline.png`
- Poster V7 results dashboard:
  `artifacts/poster_v7/03_v7_poster_results_dashboard.png`
- Poster figure guide and vector exports:
  `artifacts/poster_v7/POSTER_FIGURES.md`
- Standalone poster visuals for Sections 3-7:
  `artifacts/poster_v7/sections/SECTION_VISUALS.md`
- Poster-ready section text and updated abstract:
  `EEG_BCI_Project_POSTER_CONTENT.md`
- Current v7 report:
  `artifacts/gui_experiments/translational_v7_graph_lda/TRANSLATIONAL_V7_AUXILIARY_DIRECTION_REPORT.md`
- V7 fixed-geometry graph signal audit:
  `artifacts/gui_experiments/translational_v7_graph_lda/10_v7_graph_signal_audit.png`
- V7 PCA, shrinkage-LDA, and K-means diagnostics:
  `artifacts/gui_experiments/translational_v7_graph_lda/11_v7_graph_representation_diagnostics.png`
- V5/v6/v7 presentation comparison:
  `artifacts/gui_experiments/translational_v7_graph_lda/09_v5_v6_v7_comparison.png`
- Immediate report shortcut: `EEG_BCI_Project_FINAL_REPORT.md`
- Immediate figure shortcut: `EEG_BCI_Project_RESULTS_OVERVIEW.png`
- Immediate implementation shortcut: `EEG_BCI_Project_IMPLEMENTATION.md`
- Report:
  `EEG_BCI_Project_Final_Results/TRANSLATIONAL_V3_REPORT.md`
- Main comparison figure:
  `EEG_BCI_Project_Final_Results/07_operating_point_comparison.png`
- Full implementation guide:
  `docs/final/TRANSLATIONAL_V3_IMPLEMENTATION.md`
- Neural-integration report:
  `artifacts/gui_experiments/translational_v4_neural_integrated/TRANSLATIONAL_V4_NEURAL_REPORT.md`
- Neural-integration overview figure:
  `artifacts/gui_experiments/translational_v4_neural_integrated/08_neural_integration_summary.png`
- Project manifest:
  `EEG_BCI_Project_MANIFEST.md`

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

### Re-run the final pipeline

```bash
cd EEG_BCI_Project
./EEG_BCI_Project_RUN_FINAL_PIPELINE.command
```

This reuses the existing 36,903-window cache and writes results to:

```text
artifacts/gui_experiments/translational_v3_calibrated
```

Pass additional pipeline arguments after the launcher if needed:

```bash
./EEG_BCI_Project_RUN_FINAL_PIPELINE.command --bootstrap-iterations 200
```

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

### Inspect the HS-Stroke locked external result

The corrected HS-Stroke v6 cohort was evaluated once with the frozen v4 model:

```bash
open artifacts/hs_stroke_locked_external/HS_STROKE_LOCKED_EXTERNAL_REPORT.md
open artifacts/hs_stroke_locked_external/POST_LOCK_DIAGNOSTIC_AUDIT.md
```

The one-shot evaluator is `scripts/42_hs_stroke_locked_external.py`. It refuses
to overwrite the completed result. `scripts/43_hs_stroke_post_lock_audit.py`
recreates only the explanatory post-lock audit and does not refit or rescore the
locked model.

### Inspect the Cho2017 locked external result

Cho2017 provides an additional healthy motor-imagery domain that was absent
from V7 training. Subject 1 was used only to validate the source-data adapter
and was excluded before inference; subjects 2-52 formed the frozen external
cohort. Native 64-channel, 512 Hz EEG was mapped through the unchanged V7
seven-channel preprocessing contract. Intent came from cue-period imagery,
while negatives came from each participant's independent continuous resting
recording.

```bash
open artifacts/cho2017_locked_external_v7/CHO2017_V7_LOCKED_EXTERNAL_REPORT.md
open artifacts/cho2017_post_lock_quality_audit/CHO2017_POST_LOCK_QUALITY_AUDIT.md
```

The locked run retained 32 participants and 4,983 QC-passing windows. The
frozen population OOD gate rejected 100% of external windows, producing 0%
sensitivity despite adapted intent scores reaching 0.925 AUROC. This is a
real external failure of quality calibration, not evidence that the intent
representation contains no information.

A separate post-lock analysis fitted quality normalization from the same 30
reserved patient-calibration windows without evaluation labels. At the
unchanged sensitivity-targeted threshold it restored 93.7% availability and
reached **72.0% balanced accuracy, 45.2% sensitivity, and 98.8% specificity**.
Because this correction was designed after observing the locked failure, it is
exploratory and must be pre-registered before testing another untouched cohort.

The reproducible scripts are:

```text
scripts/52_cho2017_locked_external_v7.py
scripts/53_cho2017_post_lock_quality_audit.py
```

### Verify the project

```bash
./EEG_BCI_Project_VERIFY.command
```

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

### Download MIND and run V7-H

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
This is a base intent-gate architecture audit; it does not replace the separate
patient-adapted calibrated result reported below.

## Folder Layout

| Location | Purpose |
|---|---|
| `bci_pipeline/` | Signal processing, feature extraction, decoders, calibration, and metrics |
| `scripts/` | Complete experiment history and final GUI/pipeline entry points |
| `tests/` | Automated pipeline tests |
| `docs/final/` | Current architecture, audit, workflow, and implementation documents |
| `artifacts/multimodal_readiness/` | EEG-EMG readiness checks and presentation figures |
| `artifacts/way_eeg_gal_transfer/` | WAY source QC, transfer metrics, EEG-only student, and figures |
| `artifacts/multisource_privileged_ensemble/` | Domain-selected multi-source model, metrics, report, and figures |
| `artifacts/way_eeg_only_ssl/` | Unlabeled WAY cache, pretrained encoder, EEG-only decoder, metrics, and figures |
| `artifacts/eeg_decoder_model_bakeoff/` | Five-seed architecture comparison, participant analysis, models, metrics, and presentation figures |
| `artifacts/prospective_cue_controller/` | Frozen model hashes, cue controller, locked protocol, replay template, and synthetic functional figure |
| `artifacts/mrcp_phase_ablation/` | Development-only explicit MRCP phase audit and presentation figure |
| `artifacts/hs_stroke_locked_external/` | First frozen-v4 chronic-stroke external result, QC records, predictions, five figures, and post-lock audit |
| `MIND/` | Official MIND V3 files, downloader manifest, and resumable state |
| `artifacts/v7h_mind/` | V7-H features, model, locked internal metrics, predictions, report, and figures |
| `docs/research_archive/` | Earlier experiment reports and research notes |
| `artifacts/gui_experiments/translational_v3_calibrated/` | Final model, metrics, report, and seven figures |
| `artifacts/gui_experiments/translational_v4_neural_integrated/` | Neural-integrated candidate, paired v3 comparison, metrics, model manifest, and eight figures |
| `data_links/` | Clearly named links to the three source datasets |
| `WAY-EEG-GAL/` | Verified 12-participant EEG/EMG/kinematic/force release and download audit |

## Current Results

The fixed physiology-only v3 reference remains:

- Balanced accuracy: **67.9%**
- Intent sensitivity: **41.3%**
- Intent specificity: **94.5%**
- Conditional direction balanced accuracy: **71.0%**

The patient-adapted neural-integrated v4 candidate reached:

- Balanced accuracy: **70.6%**
- Intent sensitivity: **46.7%**
- Intent specificity: **94.4%**
- Conditional direction balanced accuracy: **71.4%**

The pooled change is **+2.7 BA points** and **+5.4 sensitivity points** versus
v3, but the equal-participant BA change is only **+0.16 points** (95%
participant-bootstrap interval **-3.67 to +3.97**). V4 is therefore the
preferred integration candidate, not a confirmed replacement for v3.

The first locked external test has now been completed on corrected HS-Stroke v6:

- Patients: **14** across **278** downloaded sessions
- Frozen recording-QC acceptance: **172/278 (61.9%)** sessions
- Retained paired-window opportunities: **26,483/46,764 (56.6%)**
- Frozen availability gate: **0.0% available**
- Locked sensitivity/specificity: **0.0% / 100.0%**
- Descriptive intent AUROC before command gating: **52.4%**
- Descriptive numeric direction balanced accuracy: **51.1%**

This external result does not confirm the development-observed improvement. The
quality gate rejected every HS window because missing auxiliary channels and
unavailable CSD were encoded as extreme deviations. Removing the gate alone is
unsafe: at the frozen conservative threshold, ungated sensitivity was 99.9% but
specificity was only 0.05%. HS-Stroke is therefore now an observed adaptation
cohort; a different untouched dataset or prospective cohort is required to
confirm any fixes. Continuous natural-idle recordings also remain necessary
before reporting false commands per minute or enabling device commands.

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
