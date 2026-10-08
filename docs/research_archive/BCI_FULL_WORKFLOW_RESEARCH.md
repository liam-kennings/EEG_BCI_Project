# Full-Workflow Research: Extracting More Motor-Intent Information from EEG

## Executive Finding

The current project is not a minimal bandpower classifier. It already includes
five-band FBCSP, baseline-normalized Morlet ERD summaries, individualized mu,
Euclidean-aligned Riemannian covariance, EEGNet, hard-negative rest examples,
patient calibration, OOD experiments, and chronological evidence accumulation.

The main opportunity is therefore not to append many generic statistics. It is
to preserve physiological structure that is currently discarded:

1. Slow cue-locked preparation below 4 Hz is removed by the current filter.
2. Morlet output is compressed into broad time summaries instead of retaining
   the evolving time-frequency-spatial response.
3. Covariance is calculated independently in each one-second window, without
   explicitly describing how the spatial state changes across the trial.
4. Signal quality is measured after 4-35 Hz filtering and primarily by
   peak-to-peak amplitude, so bad channels, 60 Hz contamination, and much of the
   available high-frequency muscle contamination are not characterized.
5. CAR is applied before a robust bad-channel stage. A severely noisy channel
   can therefore influence the common average used by every retained channel.
6. The individualized mu estimate is a resting-power peak. A calibration-derived
   task-versus-rest ERD peak may be more specific to the intended physiology.

The recommended next system is a multibranch feature pipeline with auditable
late fusion:

```text
Raw 64-channel EEG
    |
    +--> raw recording/channel quality --> bad-channel repair + quality/OOD gate
    |
    +--> 0.1-4 Hz slow-potential branch --> cue-locked preparation evidence
    |
    +--> 4-40 Hz Morlet/ERD branch ------> time-frequency-spatial evidence
    |
    +--> filter-bank covariance branch --> spatial-state evidence
    |
    +--> optional CSD/source branch -----> localized motor-cortex evidence
    |
    +--> calibrated late fusion ---------> rest/intent proposal
                                             |
                                             +--> left/right direction
                                             +--> sequential safety controller
```

## What the Current Pipeline Extracts

| Component | Current implementation | Main limitation |
| --- | --- | --- |
| Early preprocessing | 4-35 Hz Butterworth, full-scalp CAR | No robust bad-channel stage; removes slow potentials |
| Quality gate | Filtered peak-to-peak amplitude | Does not describe raw line, muscle, flatline, or channel noise |
| Spectral power | Mu, low-beta, high-beta log power | One-second averages lose within-window evolution |
| ERD | Calibration-rest or pre-cue-relative dB power | Coarse early/middle/late summaries |
| Spatial features | FBCSP and shrinkage covariance | No phase-specific covariance trajectory or CSD baseline |
| Subject adaptation | Euclidean alignment and local logistic adapter | Small gain; threshold transfer remains unstable |
| Deep baseline | Hierarchical EEGNet | High sensitivity but poor rest specificity |
| Controller | Leaky evidence, hysteresis, hold, cooldown | Cannot meet sensitivity and false-activation targets together |

The strongest locked-subject trial result remains `65.0%` intent balanced
accuracy after Euclidean alignment and 10 labels per class. In the continuous
test, the safety policy reached `1.06` false activations/min but only `8.8%`
sensitivity; the sensitivity policy reached `70.9%` sensitivity but produced
`7.94` false activations/min. Those results indicate overlap in the extracted
rest and intent representations, not simply an underpowered classifier.

## Recommended Full Workflow

### Stage 1: Preserve Raw Data and Build Recording-Level QC

Quality measurements should be extracted before the task bandpass filter and
before selecting the motor montage. For each recording and channel, calculate:

- Robust amplitude deviation and peak-to-peak amplitude.
- Flatline duration and near-zero variance.
- Correlation and predictability from neighboring channels.
- High-frequency noise ratio, using the available range below the 80 Hz
  Nyquist limit.
- 60 Hz line-noise ratio.
- Spectral slope and broadband discontinuity counts.
- Channel dropout and clipping/saturation indicators.

Use these values to identify bad channels, interpolate them from the montage,
and then calculate a robust average reference. Save every decision to a QC
table. The PREP study specifically showed that early line-noise handling,
bad-channel detection, and robust referencing make large EEG collections more
uniform and prevent bad channels from degrading downstream analyses
([Bigdely-Shamlo et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4471356/)).

Recommended comparison:

```text
A. Current CAR with no bad-channel repair
B. Robust channel QC + interpolation + CAR
C. Robust channel QC + interpolation + CSD/Laplacian
```

The classifier must not receive quality values as motor-class shortcuts.
Quality features should veto, down-weight, or mark a window as uncertain.

### Stage 2: Keep Offline and Causal Signal Views Separate

Use one standardized cleaned recording to derive two reproducible profiles:

- **Offline physiology profile:** zero-phase filters for descriptive plots and
  cue-locked physiological audits.
- **Causal controller profile:** forward filters with measured group delay and
  no future samples.

Each reported representation should identify which profile generated it.
Performance gains that exist only under zero-phase filtering must not be
reported as real-time controller gains.

### Stage 3: Add a Slow-Potential Preparation Branch

Create a separate approximately `0.1-4 Hz` branch using central and
frontocentral channels. Candidate features include:

- Mean voltage in predefined 250-500 ms bins.
- Negative slope and slope-change points.
- Minimum amplitude and time-to-minimum.
- Area under the negative deflection.
- Correlation with a development-only preparation template.
- CSD topography over FC, C, and CP regions.

MRCP studies use temporal morphology, matched filters, and low-frequency
spectral information to detect attempted or imagined movement. A direct
comparison in healthy participants and stroke patients found that both temporal
and spectral preparation features can carry useful intention information
([Jochumsen et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4487719/)).

There is an important validity constraint for EEGMMIDB: its imagery is externally
cued, and there is no separately recorded physical onset. A low-frequency
classifier may learn a visual ERP, expectation, or cue timing. Until cue-only
catch trials and a randomized delayed-go protocol are available, this branch
must be called **cue-locked slow-potential evidence**, not proven self-paced
MRCP intent. Validate it by comparing central versus occipital topographies and
by repeating the existing early-visual confound audit.

### Stage 4: Retain the Morlet Time-Frequency-Spatial Trajectory

The existing Morlet audit is physiologically useful, but the model consumes
early, late, minimum, and mean summaries. Build a compact tensor that preserves:

- 250 ms time bins from approximately 0.5-3.5 s.
- A modest 4-40 Hz variable-cycle Morlet grid.
- Nine sensorimotor channels or CSD-derived motor regions.
- Pre-cue-relative power in dB.
- Contralateral-minus-ipsilateral maps for direction.

Do not export every time-frequency pixel to an unconstrained MLP. Compare three
controlled representations:

1. Hand-engineered trajectory features: onset, slope, integral, minimum,
   time-to-minimum, duration below baseline, and terminal beta rebound.
2. Time-frequency-spatial CSP or supervised low-rank projection fitted only on
   development subjects.
3. A small separable CNN operating on the compact tensor.

Adaptive time-frequency segmentation has been developed specifically because
motor-imagery patterns occur in subject-specific time-frequency regions
([Ince et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17010962/)). Joint methods
that learn time, frequency, and spatial patterns together provide a direct
precedent for preserving these dimensions before classification
([Higashi and Tanaka, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3835885/)).

### Stage 5: Make Frequency Selection Task-Relative

The current individual-mu estimate selects the largest resting spectral peak
between 7 and 13 Hz over C3/Cz/C4. Its lack of improvement is not evidence that
all personalization is ineffective; a high resting peak is not necessarily the
frequency with the strongest task ERD.

For each target patient, use calibration trials only to estimate a shrunk
task-versus-rest effect-size curve. Select either:

- One mu sub-band and one beta sub-band with the strongest stable ERD effect, or
- A small set of filter-bank weights regularized toward population weights.

Frequency selection must use nested development/controller folds and a small
candidate grid. FBCSP was designed around subject-specific spectral selection
and remains an essential baseline
([Ang et al., 2012](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2012.00039/full)).

### Stage 6: Compare CAR with CSD/Laplacian Spatial Processing

CAR suppresses common activity but does not explicitly localize radial cortical
sources. Add a spherical-spline CSD/Laplacian branch calculated from the full
64-channel montage before reducing to motor channels.

Use identical features and folds for CAR and CSD. A primary MI-BCI comparison
reported improvements of `3.02%` across binary tasks and `5.59%` for a
four-class task with CSD relative to common referencing, and found that using
more channels during the Laplacian preprocessing stage was beneficial
([Rathee et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28715332/)). These values
are evidence for an ablation, not an expected gain on this project.

### Stage 7: Describe Spatial-State Changes, Not Only Static Covariance

The current Riemannian branch is well supported: covariance matrices are useful
spatial EEG descriptors on a non-Euclidean manifold
([Barachant et al., 2012](https://pubmed.ncbi.nlm.nih.gov/22010143/)). The next
extension should preserve dynamics without creating thousands of features.

For each trial, calculate aligned covariance in several non-overlapping phases,
then describe:

- Tangent-space coordinates in each phase.
- Riemannian distance from patient calibration rest.
- Distance traveled between consecutive phases.
- Distance to left/right development prototypes.
- Stability or reversal of the predicted direction across phases.

This creates a short covariance trajectory. It is preferable to treating four
highly overlapping windows as independent training samples.

### Stage 8: Treat Source Space and Connectivity as Secondary Ablations

Source-space ROI-CSP can use anatomy to reduce volume conduction and has been
demonstrated on multiclass MI data
([Edelman et al., 2018](https://pubmed.ncbi.nlm.nih.gov/30154834/)). Test a
small set of bilateral M1/S1, premotor, and SMA regions only after CSD is
working. Generic source reconstruction without an individual head model should
be labeled approximate.

Connectivity should not be the first response to weak classification. PLV,
coherence, and correlation are high-dimensional, sensitive to common sources,
and unstable in short windows. If tested, use CSD first and restrict the branch
to a few prespecified motor-region pairs with wPLI or imaginary coherence.
Accept it only if it adds subject-level gains beyond power and covariance.

### Stage 9: Use ICA and ASR as Measured Cleaning Ablations

ICA and ASR should not be assumed to improve results automatically. Fit them on
continuous calibration/development data, never independently on evaluation
epochs. Compare:

- No source removal after robust channel QC.
- Automated ICA with EOG/EMG-informed component labeling.
- ASR with a documented clean-rest reference and a small cutoff grid.

Report retained data, ERD attenuation, spatial-map changes, and final safety
metrics. Automatic online artifact-removal studies show that ICA-based methods
can improve BCI classification, but component identification and reference
sensors are central to the procedure
([Schloegl et al., 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC2234090/)).
ASR validation also shows that its behavior depends on the selected cutoff and
motor task, so it must remain an ablation rather than an undocumented default
([Chang et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7560919/)).

### Stage 10: Calibrated Late Fusion and Hierarchical Decisions

Train each physiological branch separately and preserve an interpretable output:

```text
quality gate
    -> rest versus usable intent
        -> left versus right
            -> temporal evidence and command state machine
```

Fuse calibrated branch probabilities rather than concatenating all raw
features. Select weights on controller subjects using the clinical cost:

```text
cost = C_false * false commands
     + C_miss  * missed intents
     + C_delay * latency
     + C_abstain * unavailable time
```

EEGNet should remain the compact raw-signal baseline, not the default answer to
every new feature. Its design intentionally learns temporal filters followed by
spatial filters and has been evaluated on both MRCP and sensorimotor-rhythm BCI
tasks ([Lawhern et al., 2018](https://arxiv.org/abs/1611.08024)). A larger
transformer should be considered only after a compact time-frequency model and
EEGNet are compared under the same subject/session splits.

## Required Pre-ML Visual and Statistical Audit

Before fitting a classifier, the new extraction pipeline should save:

1. **Recording QC dashboard:** bad channels, raw PSD, line-noise ratio,
   high-frequency ratio, flatlines, and retained data.
2. **Slow-potential figure:** central and occipital cue-locked averages with
   confidence intervals for rest, left, and right.
3. **Morlet trajectory figure:** class-average time-frequency ERD maps at C3,
   Cz, and C4.
4. **Time-frequency-space effect atlas:** training-only effect size at each
   channel, time bin, and frequency.
5. **CSD topographies:** rest-relative mu/beta maps over time.
6. **Covariance trajectory plot:** distance from rest and between phases.
7. **Patient heterogeneity panel:** each subject's strongest band, time window,
   ERD magnitude, and signal-quality burden.
8. **Feature reliability panel:** calibration Run 4 versus Run 8 agreement.
9. **Confound panel:** motor versus occipital evidence, quality versus class,
   and execution high-frequency contamination.

These outputs answer whether the signal is present, where it appears, when it
appears, and whether it repeats before asking a model to classify it.

## Prioritized Experiments

| Priority | Experiment | Why it comes next | Main risk |
| ---: | --- | --- | --- |
| 1 | Raw QC + robust bad-channel repair + CAR/CSD ablation | Corrects a weakness before every representation | Interpolation can hide widespread poor data |
| 2 | Compact Morlet trajectory and phase-specific covariance | Preserves information already computed but currently averaged away | High dimensionality if not constrained |
| 3 | Cue-locked slow-potential branch | Adds genuinely missing sub-4 Hz physiology | Visual/cue confounding in EEGMMIDB |
| 4 | Calibrated late fusion of ERD, covariance, and slow potential | Tests complementarity at the decision level | Correlated branches may add no value |
| 5 | Calibration-derived band/time selection | Targets patient-specific ERD rather than resting peak power | Overfitting small calibration sets |
| 6 | ICA/ASR and source-space ablations | Potential robustness and localization improvements | Signal removal or model complexity |
| 7 | Restricted connectivity branch | Possible network-level clinical information | Poor short-window reliability |

The first implementation should be a feature-atlas experiment, not a new final
controller. It should produce the physiological/QC figures and cached branch
features using the existing fixed splits. Only branches that pass the audit
should enter a locked model comparison.

## Acceptance Rules

A new branch should advance only if it satisfies all applicable rules:

- It improves controller-subject performance before final subjects are opened.
- On final subjects, its paired subject-level confidence interval supports a
  real gain or a useful safety/sensitivity tradeoff.
- It improves sensitivity at a fixed false-activation budget, not only pooled
  window accuracy.
- It does not obtain its gain from cue location, occipital activity, raw
  amplitude, EMG contamination, rejected trials, or subject identity.
- It improves a meaningful fraction of subjects rather than only the pooled
  majority class.
- Its causal version retains the gain with latency included.
- Its calibration burden and unavailable/abstention time are reported.

Large reproducibility benchmarks have found that Riemannian methods remain
strong across EEG datasets and that deep networks need substantial data to be
competitive. They also show why results from one dataset or one evaluation
regime should not be treated as a general ranking
([Chevallier et al., MOABB](https://pubmed.ncbi.nlm.nih.gov/30177583/)). The
project's subject-disjoint and later-run evaluation should therefore remain
fixed throughout these experiments.

## Implemented Research Unit

The recommended two-stage unit is now implemented:

```text
scripts/18_build_motor_imagery_feature_atlas.py
    raw QC and robust channel handling
    CAR versus CSD physiology
    slow-potential, Morlet trajectory, and covariance-trajectory caches
    pre-ML figures and reliability/confound tables

scripts/19_multibranch_motor_imagery_benchmark.py
    branch-specific intent and direction models
    calibrated late fusion
    patient calibration
    locked trial-level evidence/OOD ablations

scripts/20_build_extended_motor_imagery_features.py
    calibration-derived spatio-spectral decomposition
    explicit hemispheric asymmetry and compact ERD morphology
    restricted sensorimotor phase coupling
    participant spectral profiles and development-only repeatability

scripts/21_controlled_extended_feature_ablation.py
    fixed slow-plus-covariance reference
    validation-only morphology/SSD selection
    paired final-subject comparison and bootstrap uncertainty
```

The feature atlas was built and reviewed before the second script fit models.
It contains 9,835 trials from all 109 subjects and audits all 327 unilateral
imagery recordings. On the 22 locked final subjects, controller-selected fusion
of the slow-potential and covariance branches reached 68.8% intent balanced
accuracy, 23.9% false activation, and 38.5% missed intent. The safety point
reached 6.7% false activation but missed 70.0% of intent.

Run 4 versus Run 8 repeatability was 0.26 for the slow-potential contrast, 0.72
for Morlet mu, 0.72 for CSD mu, and 0.78 for covariance distance. The low
slow-potential repeatability limits its interpretation as a patient-specific
biomarker even though it helped the classifier. The more repeatable covariance
and ERD branches should provide confirmation in the next causal experiment.

Patient calibration and evidence/OOD rejection did not improve the locked final
cohort. Those negative results are retained in
[`artifacts/multibranch_benchmark/MULTIBRANCH_BENCHMARK_RESULTS.md`](artifacts/multibranch_benchmark/MULTIBRANCH_BENCHMARK_RESULTS.md).
The full physiology audit is in
[`artifacts/feature_atlas/FEATURE_ATLAS_RESULTS.md`](artifacts/feature_atlas/FEATURE_ATLAS_RESULTS.md).

The extended pre-ML audit adds 1,020 aligned trial features. On 69 development
subjects, median Run 4/8 effect-pattern cosine similarity was 0.50 for ERD
morphology, 0.39 for SSD, 0.05 for explicit asymmetry, and 0.02 for restricted
phase. Morphology and SSD entered an isolated controlled ablation; asymmetry
remains restricted to future prespecified direction contrasts, and phase
remains exploratory.

The participant profile also shows that 76.8% of development subjects have a
resting mu peak at least 3 dB above local background and 82.6% have Run 4/8
resting spectral correlation of at least 0.80. Individual bands are plausible
for most participants, but a fixed/broad-band fallback is still required. The
cache, profile table, diagnostics, and presentation figures are documented in
[`artifacts/extended_features/EXTENDED_FEATURE_RESULTS.md`](artifacts/extended_features/EXTENDED_FEATURE_RESULTS.md).

The controlled ablation selected morphology on controller validation, but it
did not establish a reliable final improvement. Balanced intent BA changed from
68.8% to 69.5% (paired subject-bootstrap change interval -3.2 to +4.5 points);
false activation increased from 23.9% to 26.4% while missed intent decreased
from 38.5% to 34.5%. At the safety point, both models had 6.7% false activation
and intent BA changed from 61.7% to 61.5%. SSD reduced controller-validation
performance and did not advance. Full results are in
[`artifacts/controlled_ablation/CONTROLLED_ABLATION_RESULTS.md`](artifacts/controlled_ablation/CONTROLLED_ABLATION_RESULTS.md).

The slow-plus-covariance reference therefore remains the default decoder.
Morphology may be retained as an exploratory sensitivity-oriented proposal
signal, but should not replace the safety policy. The next isolated unit should
carry the reference into a causal continuous-window comparison with the
existing script 17 controller under the same false activations per minute,
latency, and missed-intent definitions.
