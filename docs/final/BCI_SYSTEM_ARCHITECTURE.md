# Research-Backed Patient-Adaptive Motor-Intent BCI

## 1. Objective

The system should answer a stricter question than ordinary EEG classification:

> Can a model trained on many people detect weak motor intent in a completely
> unseen patient, improve safely from a small patient calibration set, and avoid
> issuing commands during rest or artifact-contaminated EEG?

This requires two connected but separately evaluated systems:

1. An **offline learning pipeline** that builds and adapts the decoder.
2. An **online controller** that converts uncertain window predictions into a
   conservative prosthetic command.

The complete flow is:

```text
Raw EDF + annotations
        |
        v
Dataset validation and immutable manifest
        |
        v
Continuous EEG preprocessing + quality-control report
        |
        v
Trial epochs + asynchronous sliding windows
        |
        +----------------+------------------+------------------+
        |                |                  |                  |
        v                v                  v                  v
 Filter-bank       FBCSP/log-power     Raw EEGNet       Morlet ERD maps
 covariance         features            windows          (experimental)
        |                |                  |                  |
        +----------------+------------------+------------------+
                                 |
                                 v
                Global hierarchical intent decoder
                    rest/task -> left/right
                                 |
                                 v
                  Small unseen-patient calibration
                                 |
                                 v
                Patient alignment, calibration, profile
                                 |
                                 v
          Untouched patient session and continuous-stream test
                                 |
                                 v
        uncertainty/OOD gate -> evidence accumulator -> command
```

Wavelets are one candidate representation, not the pipeline itself. Large BCI
benchmarks find that Riemannian covariance methods are consistently strong and
that deep models become more competitive as the number of independent training
trials grows. The first deep baseline should therefore be compact EEGNet, not a
large ResNet or transformer.

## 2. Dataset Roles and Leakage Boundary

The PhysioNet EEGMMI unilateral imagery runs are 4, 8, and 12. The event labels
must be interpreted by run; T1/T2 do not have the same meaning in bilateral
hand/foot runs. Baseline runs 1/2 are not substitutes for the within-task T0
state and should not be mixed into a three-class model without a separate study.

Assign every subject to exactly one of these partitions before preprocessing
parameters or model choices are fitted:

| Partition | Purpose | May influence model choices? |
|---|---|---:|
| Development source subjects | Fit global models and feature transforms | Yes |
| Controller-validation subjects | Select thresholds, dwell, OOD, and calibration policy | Yes |
| Final target subjects | Simulate new patients | No |

For each final target subject:

- Runs 4/8 are patient calibration only.
- Run 12 is locked final evaluation only.
- No statistic estimated from Run 12 may change filtering, alignment, feature
  scaling, source selection, thresholds, or controller settings.
- All windows from one cue remain in one partition.

The current project has source and target isolation, but it should permanently
reserve controller-validation subjects too. Reusing target results to redesign
the controller gradually turns the target cohort into another validation set.

## 3. Stage A: Ingest and Validate Raw EEG

Create one row per recording in a versioned manifest. Record the source path,
file checksum, subject, run, task semantics, sampling rate, units, channel names,
duration, annotation counts, and software/configuration version.

Validation must fail loudly for:

- Missing or duplicated runs.
- Unexpected sampling rates or physical units.
- Missing motor channels or non-unique normalized channel names.
- Event codes that are invalid for the selected task.
- Epochs extending outside a recording.
- Implausible flat, saturated, or non-finite signals.

Recommended outputs:

```text
artifacts/dataset/manifest.csv
artifacts/dataset/validation_report.json
artifacts/dataset/splits.json
```

The subject split should be saved once and reused by every experiment.

## 4. Stage B: Preprocess Continuous EEG

Preprocessing should operate on all usable scalp channels before selecting the
sensorimotor montage. A noisy channel can contaminate ordinary common-average
referencing, so channel quality should be estimated before the final reference.

Recommended offline sequence:

1. Load continuous data in volts and attach the standard montage.
2. Detect flat, saturated, high-variance, low-correlation, and line-noise-heavy
   channels; save every reason and threshold in the QC report.
3. Remove 50/60 Hz interference only when it is inside the retained bandwidth or
   a measured spectrum shows that it is needed. A 4-35 Hz low-pass already
   excludes direct 60 Hz power, although harmonics and filter behavior still need
   inspection.
4. Apply a broad offline zero-phase bandpass, initially 4-35 Hz for imagery.
5. Interpolate marked channels when positions are available, then compute a
   robust/common-average reference across the cleaned full montage.
6. Optionally compare ASR and ICA against the simpler pipeline as ablations.
7. Select a broad sensorimotor region only after cleaning and referencing.
8. Produce pre/post spectra, channel quality, rejection counts, and retained
   trial counts by subject and class.

ASR and ICA should not be assumed to improve decoding automatically. ASR is
useful for transient high-amplitude bursts when it has suitable clean calibration
data. ICA needs enough continuous multi-channel data and a defensible component
labeling rule; fitting ICA independently to short two-minute runs is unstable.
Both methods can remove neural signal as well as artifact, so they should be kept
only if subject-held-out performance or artifact-stress tests improve.

The online implementation must use causal filters and account for filter delay.
It cannot silently reuse the offline zero-phase result. Train/evaluate a second
"causal deployment" preprocessing profile before making latency claims.

Recommended outputs:

```text
artifacts/preprocessed/<subject>/<run>.fif
artifacts/preprocessed/<subject>/<run>_qc.json
artifacts/preprocessed/cohort_qc.csv
```

## 5. Stage C: Epoch Once, Create Two Views

Create trial identity before generating overlapping windows.

### Trial view

- Baseline: approximately -1.0 to 0 seconds before the annotation.
- Motor-imagery response: initially 0.5 to 3.5 seconds after onset.
- Store rest/task label, direction label, subject, run, cue, and QC flags.
- Keep alternate onset/duration choices as source-validation hyperparameters.

### Continuous-controller view

- Use 0.75-1.5 second causal windows with 0.125-0.25 second steps.
- Preserve their parent trial/recording and absolute timestamps.
- Include transitions and long idle intervals, not only balanced cue epochs.

EEGMMI is cue-driven and does not contain the quantity or diversity of idle data
needed to estimate real prosthetic false activations per hour. It is suitable for
model development, but continuous healthy and clinical recordings are required
for asynchronous safety validation.

## 6. Stage D: Interchangeable Representations

Every representation must consume the same epoch table and use transforms fitted
only on development training subjects.

### Required baseline 1: filter-bank Riemannian covariance

- Bands: narrow mu and beta bands plus a broad 8-30 Hz view.
- OAS/shrinkage covariance for short windows.
- Euclidean alignment per source subject and calibration-only alignment for a
  target patient.
- Tangent-space logistic regression or minimum-distance-to-mean classifier.
- Include task covariance and pre-cue-relative covariance.

This is the strongest current project branch and a well-supported cross-subject
baseline.

### Required baseline 2: FBCSP

- Learn spatial filters independently inside each training fold.
- Rank/select log-variance features without seeing held-out subjects.
- Use shrinkage LDA or logistic regression.
- Let calibration choose a small set of patient-relevant frequency bands rather
  than fitting a high-dimensional patient classifier.

FBCSP directly models lateralized sensorimotor rhythms and provides an important
check that the covariance pipeline is not obscuring left/right information.

### Required deep baseline: raw-window EEGNet

- Input minimally processed channels by time, with training-only normalization.
- Train rest/task and left/right heads, or two small networks, rather than only a
  three-class softmax.
- Use channel dropout, time masking, small amplitude scaling, and small temporal
  shifts. Avoid arbitrary spatial permutations or transformations that destroy
  lateralization.
- Calibrate probabilities on controller-validation subjects.

Only advance to EEG Conformer or ATCNet if EEGNet improves the strict held-out
patient benchmark. Larger architectures add capacity but do not fix a weak
representation, label ambiguity, or domain shift.

### Experimental branch: Morlet time-frequency maps

- Retain baseline-normalized time-frequency structure and sensorimotor topology.
- Do not reduce every channel/band to only mean, variance, skewness, and kurtosis;
  those moments discard the timing and spatial pattern of ERD/ERS.
- Compare scalograms or compact ERD maps against log-bandpower features.
- Add the branch to an ensemble only when errors are demonstrably complementary.

### Quality/OOD features

Maintain a separate vector for amplitude, peak-to-peak, kurtosis, spectral slope,
line-noise ratio, high-frequency muscle ratio, channel dropout, and distance from
the calibration distribution. These features should gate or abstain; they should
not become shortcuts for predicting motor class.

## 7. Stage E: Train the Global Decoder

Use a hierarchical decision because the two errors have different meanings:

```text
Stage 1: Is usable motor intent present?
Stage 2: If yes, which direction?
```

Recommended training procedure:

1. Fit all preprocessing-derived transforms, alignment references, feature
   selection, scaling, and calibration inside subject-grouped folds.
2. Optimize intent discrimination and conditional direction discrimination
   separately.
3. Report source cross-validation by subject, not pooled windows.
4. Compare Riemannian, FBCSP, and EEGNet under identical folds.
5. Try late fusion only after measuring disagreement and paired per-subject gains.
6. Freeze a global candidate before looking at final target subjects.

Execution and imagery runs should be separate tasks initially. Execution EEG can
contain class-correlated EMG, so mixing it into imagery training can inflate
accuracy while teaching the model a feature unavailable to a severely impaired
patient. Execution may later be used for auxiliary or self-supervised pretraining
if the imagery-only test improves.

## 8. Stage F: Adapt to an Unseen Patient

The patient profile should be small, auditable, and reversible:

- Estimate Euclidean/reference alignment from calibration data only.
- Estimate patient-specific mu/beta weighting.
- Select similar source subjects only using calibration distributions.
- Fit temperature/probability calibration and an intent bias/adapter.
- Keep the global feature extractor and usually the direction model frozen when
  calibration data are scarce.
- Estimate patient-specific artifact and OOD limits from calibration rest trials.

Save the global model and patient profile separately. A patient update should not
overwrite the research model.

```text
models/global/<version>/
models/patients/<patient>/<version>/profile.json
models/patients/<patient>/<version>/adapter.joblib
```

## 9. Stage G: Online Safety Controller

A classifier probability is evidence, not a prosthetic command. The controller
should combine:

1. **Signal-quality veto:** reject saturated, disconnected, or severe artifact
   windows before neural inference.
2. **Intent prescreener:** a rest/task model prevents direction confidence from
   activating the device during rest.
3. **Uncertainty/OOD abstention:** reject low-confidence or distribution-shifted
   windows. The current Mahalanobis result shows that distance alone is
   insufficient; compare calibrated entropy, energy, ensemble disagreement, and
   calibration-relative density.
4. **Sequential evidence:** accumulate log likelihood ratios or use an SPRT-like
   rule across consecutive windows.
5. **Direction consistency:** require the intended direction to remain stable.
6. **Dwell and refractory periods:** require sustained evidence, then prevent
   repeated commands until the system returns to rest.
7. **Shared-control veto:** the prosthetic controller should still reject unsafe
   commands based on device state and sensors.

Thresholds should be chosen from a cost function rather than accuracy alone:

```text
cost = C_false * false activations
     + C_miss  * missed intents
     + C_delay * command latency
     + C_abstain * no-decision time
```

The costs must be selected with the clinical/device team. In rehabilitation,
false feedback can be harmful, but a controller that almost never activates is
also unusable. Sequential evidence is the main way to gain specificity without
raising one static threshold until nearly all intent is missed.

## 10. Stage H: Evaluation Ladder

Do not summarize the system with one pooled accuracy number. Advance through:

| Level | Question |
|---|---|
| Within-subject | Is there decodable signal at all? |
| Cross-session | Does it survive same-person drift? |
| Zero-shot cross-subject | Does the global model transfer? |
| Few-shot cross-subject | Does calibration improve it? |
| Pseudo-asynchronous | Does the controller manage continuous cue-run windows? |
| True asynchronous | How many false commands occur during long idle use? |
| Clinical external | Does it transfer to stroke/rehabilitation patients? |

Required metrics:

- Three-class balanced accuracy, macro F1, and confusion matrix.
- Rest false-activation rate and false commands per minute/hour.
- Intent sensitivity and missed-intent rate.
- Conditional left/right balanced accuracy after intent is detected.
- No-decision/abstention rate.
- Median and upper-quantile command latency.
- Per-patient metrics, confidence intervals, and worst-patient behavior.
- Calibration burden and performance versus trials/minutes of calibration.

Use paired subject-level bootstrap confidence intervals and comparisons. A new
method should not be accepted because pooled window accuracy rises by a small
amount while several patients or the false-activation rate worsen.

## 11. Recommended Project Modules

```text
bci_pipeline/
  config.py
  ingest.py                 # EDF discovery, run semantics, manifest
  quality.py                # channel/recording QC and artifact flags
  preprocess.py             # offline and causal profiles
  epochs.py                 # trial and sliding-window datasets
  splits.py                 # immutable subject partitions
  representations/
    riemann.py
    fbcsp.py
    eegnet.py
    morlet.py
  models/
    hierarchical.py
    calibration.py
    uncertainty.py
  controller/
    evidence.py
    state_machine.py
  evaluation/
    protocols.py
    metrics.py
    reports.py
scripts/
  01_build_manifest.py
  02_preprocess.py
  03_build_epochs.py
  04_train_global.py
  05_calibrate_patient.py
  06_evaluate_locked.py
  07_replay_stream.py
```

Each stage should write a configuration, code version, input fingerprint, output
fingerprint, and QC summary. This makes an experiment reproducible and prevents
stale feature caches from being paired with a different preprocessing profile.

## 12. Implementation Order

1. Refactor ingestion, preprocessing, epoching, split management, and metrics out
   of the current experiment scripts. Preserve the current Riemannian result as a
   regression test.
2. Add robust channel QC and compare plain CAR, robust CAR, ASR, and ICA as
   preprocessing ablations. Do not combine all cleaners by default.
3. Add leakage-safe FBCSP and EEGNet baselines under one evaluation harness.
4. Lock a global model, then compare zero-shot and few-shot patient adaptation.
5. Replace the weak one-second covariance gate with the best calibrated raw-window
   intent model; retest sequential evidence and uncertainty/OOD rejection.
6. Collect or obtain long continuous idle and clinical stroke EEG before making
   claims about real-world false activations or rehabilitation transfer.

This order attacks the current bottleneck directly: the temporal controller
cannot recover weak intent if its per-window model is poor, and a larger model
cannot rescue mislabeled, leaking, or artifact-driven data.

## Current Implementation

The shared foundation is implemented under `bci_pipeline/`. Build the recording
manifest and immutable three-way subject split with:

```bash
python3 scripts/01_build_manifest.py
```

The current cohort contains 327 unilateral-imagery recordings from 109 subjects.
With seed 42, `artifacts/dataset/splits.json` assigns 69 development subjects, 18
controller-validation subjects, and 22 final-target subjects. Existing patient
and temporal experiments now share dataset semantics, preprocessing profiles,
filter-bank definitions, subject splitting, and metrics through this package.

The first common zero-shot comparison is implemented in
`scripts/04_benchmark_models.py`. It trains hierarchical FBCSP and EEGNet models
on the same raw trial cache and reports both pooled and per-patient safety
metrics. See `ZERO_SHOT_MODEL_BENCHMARK.md` for the locked full-cohort result.

Options for adding physical execution Runs 3/7/11 without contaminating the
imagery claim are evaluated in `MOTOR_EXECUTION_IMAGERY_RESEARCH_OPTIONS.md`.
The recommended sequence is a cross-task diagnostic matrix, followed by
execution pretraining and imagery fine-tuning, before attempting domain
adaptation or MRCP fusion.

The completed full-cohort result is in `CROSS_TASK_EXECUTION_RESULTS.md`. It
shows negative transfer from naive execution/imagery pooling and no overall gain
from execution pretraining. Execution-derived features should therefore remain
an optional direction auxiliary, not the primary intent gate.

## Research Basis

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [MOABB large-scale BCI benchmark](https://arxiv.org/abs/2404.15319)
- [Riemannian approaches to BCI](https://pubmed.ncbi.nlm.nih.gov/27845666/)
- [Euclidean alignment for cross-subject transfer](https://arxiv.org/abs/1808.05464)
- [Filter Bank Common Spatial Pattern](https://pmc.ncbi.nlm.nih.gov/articles/PMC3314883/)
- [EEGNet](https://pubmed.ncbi.nlm.nih.gov/29932424/)
- [EEG Conformer](https://pubmed.ncbi.nlm.nih.gov/37015413/)
- [ATCNet](https://doi.org/10.1109/TII.2022.3197419)
- [Morlet/STFT time-frequency motor-imagery decoding](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.655599/pdf)
- [PREP robust EEG preprocessing](https://doi.org/10.3389/fninf.2015.00016)
- [Artifact Subspace Reconstruction technical description](https://raw.githubusercontent.com/wiki/sccn/clean_rawdata/files/AsrDescription.pdf)
- [EEG-BIDS](https://pmc.ncbi.nlm.nih.gov/articles/PMC6592877/)
- [Sequential probability ratio testing for motor-imagery BCI](https://pubmed.ncbi.nlm.nih.gov/29348781/)
- [Asynchronous dwell and refractory control](https://doi.org/10.1109/TNSRE.2004.827220)
- [Stroke BCI false-positive rejection pilot](https://pubmed.ncbi.nlm.nih.gov/36172602/)
- [Stroke BCI-FES feasibility study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6010454/)
- [Effect of false-positive feedback in rehabilitation BCI](https://pubmed.ncbi.nlm.nih.gov/40102969/)
