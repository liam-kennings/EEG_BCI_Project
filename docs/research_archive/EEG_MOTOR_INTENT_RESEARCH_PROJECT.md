# Weak EEG Motor-Intent Detection for Rehabilitation and Prosthetic Control

> **Current full-project audit:** See
> [`FULL_PROJECT_BCI_AUDIT_2026.md`](FULL_PROJECT_BCI_AUDIT_2026.md) for the
> July 2026 evidence review, redesigned pipeline, dataset strategy, validation
> contract, and prioritized implementation plan.

> **Audit implementation:** See
> [`TRANSLATIONAL_V3_IMPLEMENTATION.md`](TRANSLATIONAL_V3_IMPLEMENTATION.md) for
> the native-montage MRCP/SMR/quality pipeline, domain-aware decoders, safety
> endpoints, GUI preparation, reproducible commands, and verification results.

## Project Purpose

The primary goal of this project is to improve detection of weak motor intent
in EEG for future rehabilitation and prosthetic-control applications. The
central engineering problem is not initially to decode a large set of movement
commands. It is to determine whether credible cortical motor intent is present
while limiting unsafe false activations during rest.

The research follows a staged translational sequence:

1. Establish the signal-processing and modeling process with healthy
   participants, using physical movement as a stronger physiological reference
   and motor imagery as the intended noninvasive control signal.
2. Validate the resulting intent detector on unseen healthy participants and
   across independently acquired healthy datasets.
3. Test the locked healthy-trained detector on unseen stroke patients, then
   measure how much patient-specific calibration is required to recover useful
   intent detection.
4. Quantify differences in physiology, feature distributions, calibration,
   false activations, and missed intents between healthy and stroke cohorts.
5. Use those retrospective findings to design later synchronized, cue-gated
   in-person experiments.

The current retrospective datasets cannot by themselves prove that every
healthy-versus-stroke difference is caused by stroke. Population is confounded
with recording hardware, referencing, cue timing, and experimental protocol.
Accordingly, this phase reports cross-population and cross-dataset transfer
differences; a prospective healthy-control and stroke study using the same
equipment and protocol will be required for a causal physiological comparison.

The expanded benchmark now uses both motor-task families in the PhysioNet EEG
Motor Movement/Imagery Dataset (EEGMMIDB):

- Unilateral tasks: `rest`, `left_fist`, and `right_fist`.
- Bilateral tasks: `rest`, `both_fists`, and `both_feet`.

Movement and imagery are evaluated with matched preprocessing, features,
models, subject partitions, and operating-threshold selection. Execution data
are not pooled into a task-specific imagery decoder, preserving an honest
imagery result. A separate pooled binary experiment tests whether all task runs
can improve generalized `rest` versus `motor_intent` detection. This matched
benchmark is the physiological and methodological foundation of the project,
not its final clinical endpoint.

The primary shared target across PhysioNet, OpenBMI, and the Liu acute-stroke
dataset is binary hand `motor_intent` versus `rest`. Left-versus-right intent is
a conditional secondary output after intent has been detected. Hand-versus-foot
decoding remains exploratory because all three current datasets do not share
compatible foot-imagery labels.

## Research Questions

1. Which EEG representations most consistently separate weak motor imagery or
   attempted-movement intent from rest in unseen healthy participants?
2. Does a healthy-trained intent detector generalize across independently
   acquired healthy datasets, subjects, sessions, and recording systems?
3. How much performance is lost when the locked healthy-trained detector is
   transferred without labels to unseen stroke patients?
4. Which physiological and statistical differences distinguish healthy and
   stroke cohorts after acquisition and protocol effects are explicitly
   modeled?
5. How much labeled calibration from a new stroke patient is needed to improve
   sensitivity without producing an unacceptable false-activation rate?
6. Once intent is detected, how reliably can the system conditionally identify
   left-versus-right hand intent or the intended limb family?
7. What cue-gating, OOD rejection, temporal confirmation, and abstention policy
   best balances false activations, missed intents, and command latency?

## Translational Study Design

The three datasets must have different roles rather than being pooled before
the first clinical-transfer measurement:

| Phase | Data role | Primary question |
| --- | --- | --- |
| Healthy physiology reference | PhysioNet movement and imagery | How much weaker and more variable is imagery than execution under a matched analysis? |
| Healthy model development | PhysioNet and OpenBMI development subjects | Which intent features and alignment methods generalize across healthy subjects and recording systems? |
| Zero-shot stroke transfer | Completely untouched Liu stroke patients | What performance remains when a frozen healthy model is applied without stroke labels? |
| Stroke-domain adaptation | Separate Liu development patients | How much does labeled or label-free stroke adaptation improve transfer? |
| Locked clinical test | Liu patients excluded from all development and threshold selection | Does the final adapted process generalize to unseen stroke patients? |
| Prospective feasibility | Future healthy controls and stroke participants recorded with one synchronized protocol | Are the retrospective differences reproducible when acquisition and cue timing are controlled? |

The zero-shot stroke result must be recorded before fitting any component on
Liu labels. A later mixed healthy-plus-stroke model may use only designated
stroke-development patients. Dataset-balanced training, dataset-specific
normalization, and leave-one-dataset-out evaluation are required so that the
decoder does not succeed by identifying recording-system differences.

The primary endpoint is binary `motor_intent` versus `rest`. Required metrics
are intent sensitivity, specificity, balanced accuracy, AUROC, AUPRC, missed
intents, false activations per rest minute, abstention rate, and calibration
burden. Left/right accuracy is reported only for windows in which intent is
present, and hand/foot performance is reported separately when compatible
labels exist.

Healthy-versus-stroke analysis should compare patient-level distributions of
mu/beta ERD, MRCP timing and amplitude, spatial covariance, affected- versus
unaffected-hemisphere laterality, uncertainty, OOD rate, and calibration gain.
These analyses characterize transfer-relevant differences; they are not used
as a healthy/stroke command classifier.

## Dataset and Correct Event Semantics

EEGMMIDB contains 64-channel EEG sampled at 160 Hz. Runs `1` and `2` are
baseline recordings and are not part of this cue-locked task benchmark. Runs
`3` through `14` are included as follows:

| Task family | Modality | Runs | `T0` | `T1` | `T2` |
| --- | --- | --- | --- | --- | --- |
| Unilateral | Movement | `3/7/11` | Rest | Left fist | Right fist |
| Unilateral | Imagery | `4/8/12` | Rest | Imagined left fist | Imagined right fist |
| Bilateral | Movement | `5/9/13` | Rest | Both fists | Both feet |
| Bilateral | Imagery | `6/10/14` | Rest | Imagined both fists | Imagined both feet |

The event codes are context-dependent. In particular, bilateral `T1/T2`
events must not be labeled left/right fist. Internally, IDs `0/1/2` remain
stable for model fitting, while class names are resolved from the run family.

The full cache contains 36,650 accepted trials from 108 represented subjects.
`S109` contributed no accepted trials to the cache under the current extraction
and quality-control rules, so it is excluded from development. The fixed
controller-validation and final-test cohorts are unaffected.

## Expanded Matched Pipeline

Four task-specific three-class decoders are trained:

1. Unilateral movement.
2. Unilateral imagery.
3. Bilateral movement.
4. Bilateral imagery.

Every decoder uses the same:

- Cue-locked epoch definition.
- 4--35 Hz offline filtering.
- Common-average reference (CAR).
- 250-microvolt peak-to-peak artifact rejection.
- Per-trial pre-cue baseline centering.
- Nine central sensorimotor channels.
- Five-view mu/beta filter-bank CSP representation.
- Hierarchical, class-balanced logistic-regression decoder.
- Subject-isolated development, controller-validation, and final-test groups.

The only intended differences are modality and task family.

![Expanded movement-versus-imagery pipeline](artifacts/dual_task_benchmark/dual_pipeline_overview.png)

![Subject-isolated evaluation protocol](artifacts/dual_task_benchmark/dual_task_subject_isolation_protocol.png)

## Preprocessing

### Full-scalp filtering and referencing

Each EDF recording is loaded using EEG channels only. A fourth-order
Butterworth band-pass filter retains 4--35 Hz activity. CAR is applied over the
full EEG montage before selecting motor channels, reducing spatially shared
noise while retaining localized sensorimotor activity.

### Motor-focused channels

The model receives:

`FC3`, `FC4`, `C3`, `C1`, `Cz`, `C2`, `C4`, `CP3`, and `CP4`.

This montage covers bilateral hand areas around C3/C4 and includes Cz, which is
particularly relevant when adding bilateral foot imagery. A future
task-family audit should test whether the bilateral pipeline benefits from a
wider midline channel set.

### Epochs and quality control

For every task annotation:

- A one-second pre-cue interval is retained as baseline.
- The analyzed interval begins 0.5 seconds after cue onset.
- The post-cue epoch lasts 3.0 seconds, producing 480 samples.
- Each channel is centered using its own pre-cue mean.
- A trial is rejected if the baseline or post-cue segment exceeds 250
  microvolts peak-to-peak or contains a nonfinite value.

The 0.5-second delay reduces direct visual-evoked cue activity and emphasizes
the subsequent sensorimotor response.

The figure below shows the same C3/Cz/C4 samples before and after the benchmark
preprocessing. Raw traces receive only a per-channel display offset; processed
traces use the actual filter, CAR, and pre-cue centering pipeline. The dashed
line marks cue onset and the shaded region is the 0.5--3.5-second interval sent
to feature extraction. This is an illustrative development-subject example,
not a model-performance metric.

![Representative raw and processed EEG](artifacts/dual_task_benchmark/raw_vs_processed_eeg.png)

The cue-centered view above deliberately sets cue onset to zero. The full-trial
view below instead preserves the original `T0`-to-task timeline, placing the
first `T1/T2` event near 4.1 seconds. It displays the processed broadband signal
alongside isolated mu (7--13 Hz) and beta (13--30 Hz) waveforms for the same
movement and imagery trials. These band waveforms illustrate the oscillations;
the quantitative FBCSP features are computed from spatially projected,
log-normalized band variance rather than from visual inspection.

![Full EEG trial with mu and beta bands](artifacts/dual_task_benchmark/full_trial_mu_beta_eeg.png)

## Features and Decoders

### Filter-bank common spatial patterns

Each model fits four CSP spatial filters in each frequency view using
development subjects only:

| View | Frequency range |
| --- | ---: |
| Low mu | 7--10 Hz |
| High mu | 10--13 Hz |
| Low beta | 13--20 Hz |
| High beta | 20--30 Hz |
| Combined mu/beta | 8--30 Hz |

Log-normalized CSP projection variances become the features. This is a
traditional and interpretable motor-BCI representation of spatial mu/beta
power modulation.

### Hierarchical three-class model

Each task-specific decoder has two stages:

1. **Intent gate:** `rest` versus any motor intent in that run family.
2. **Task-specific head:** left versus right for unilateral runs, or both fists
   versus both feet for bilateral runs.

An intent probability below the selected threshold produces `rest`. Otherwise,
the task-specific head produces the command class. This separates unsafe rest
activation from incorrect effector or direction selection.

### Pooled generalized intent gate

A fifth FBCSP model is trained across all 12 task runs. It is evaluated only as
a binary `rest` versus `any motor intent` gate. Its second-stage class output is
ignored because `T1/T2` do not share one body-part meaning across task families.

The pooled experiment asks whether broader motor examples improve intent
detection. It does not claim a five-class decoder and does not mix incompatible
labels.

## Subject-Isolated Protocol

| Partition | Subjects | Data used | Role |
| --- | ---: | --- | --- |
| Development | 68 represented | All three repeats for each condition | Fit CSP, scalers, and classifiers |
| Controller validation | 18 | Runs `11/12/13/14` | Select condition-specific thresholds |
| Final unseen test | 22 | Runs `11/12/13/14` | Report final results only |

No final-test participant contributes to spatial filters, feature scaling,
classifier fitting, or threshold selection.

Two operating points are retained:

- **Balanced:** controller-selected threshold balancing rest rejection and
  intent sensitivity. This is the primary scientific comparison.
- **Safety:** controller-selected threshold targeting no more than 10% false
  activation. This exposes the corresponding increase in missed intent.

## Final Three-Class Results

All values below use the balanced controller-selected operating point and 22
completely unseen final-test subjects.

| Condition | Trials | Accuracy | Balanced accuracy | Macro F1 | Intent BA | Task-class BA | False activation | Missed intent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unilateral movement | 613 | 61.3% | 55.4% | 55.7% | 72.7% | 65.2% | 17.7% | 37.0% |
| Unilateral imagery | 612 | 42.3% | 44.3% | 42.2% | 57.1% | 61.8% | 64.4% | 21.5% |
| Bilateral movement | 617 | 58.2% | 57.9% | 56.8% | 72.0% | 69.0% | 40.7% | 15.3% |
| Bilateral imagery | 628 | 43.8% | 43.1% | 41.9% | 58.6% | 60.0% | 53.8% | 28.9% |

![Four-condition headline results](artifacts/dual_task_benchmark/dual_task_abstract_key_findings.png)

### Movement-to-imagery gaps

| Task family | Balanced-accuracy gap | Intent-BA gap | Task-class-BA gap |
| --- | ---: | ---: | ---: |
| Unilateral | 11.2 points | 15.6 points | 3.4 points |
| Bilateral | 14.8 points | 13.3 points | 9.0 points |

Both task families support the same broad conclusion: execution is more
separable than imagery under the matched pipeline. In both cases, the imagery
intent BA is only about 57--59%, and false activation during rest exceeds 50%.
Task-specific discrimination remains above chance, but the system first has to
decide reliably whether an intent exists.

### Class-wise F1

| Condition | Rest | Class 1 | Class 2 |
| --- | ---: | ---: | ---: |
| Unilateral movement | 74.0% | Left fist: 50.0% | Right fist: 43.2% |
| Unilateral imagery | 44.9% | Left fist: 39.5% | Right fist: 42.1% |
| Bilateral movement | 67.4% | Both fists: 53.9% | Both feet: 49.0% |
| Bilateral imagery | 52.1% | Both fists: 43.9% | Both feet: 29.7% |

The bilateral imagery result adds a second challenge: both-feet imagery has the
lowest task-class F1. This may reflect weaker imagery, cross-subject topographic
variability, or a sensor montage/feature configuration that is still too
hand-focused.

![Four task-specific confusion matrices](artifacts/dual_task_benchmark/dual_task_confusion_matrices.png)

## Pooled Rest-vs.-Intent Result

The generalized intent gate evaluates 2,470 final trials pooled across runs
`11/12/13/14`:

| Metric | Balanced operating point |
| --- | ---: |
| Intent balanced accuracy | 63.0% |
| ROC AUC | 67.6% |
| Rest specificity | 56.2% |
| Intent sensitivity | 69.8% |
| False-activation rate | 43.8% |
| Missed-intent rate | 30.2% |

Pooling increases the diversity and number of motor-intent examples, but it
does not solve rest rejection. At the safety threshold, false activation falls
to 9.3%, while missed intent rises to 80.6%. The pooled gate is therefore a
useful baseline, not a deployable controller.

![Pooled generalized intent gate](artifacts/dual_task_benchmark/pooled_intent_gate_summary.png)

## Subject Variability and Safety Tradeoff

Median final-subject balanced accuracy was 50.6% versus 39.3% for unilateral
movement and imagery, and 59.2% versus 44.9% for bilateral movement and imagery.
The participant-level spread confirms that a generalized model will likely
need a patient-specific calibration stage.

![Per-subject task-family comparisons](artifacts/dual_task_benchmark/dual_task_per_subject_comparison.png)

The safety curves show that no single trial-level threshold simultaneously
provides low false activation and low missed intent. This motivates confidence
abstention, temporal evidence accumulation, and explicit rest/OOD modeling.

![Intent safety tradeoff](artifacts/dual_task_benchmark/dual_task_safety_tradeoff.png)

## What This Benchmark Establishes

- The movement-to-imagery decoding gap is present in both EEGMMIDB task
  families, not only left/right hand trials.
- The context-dependent `T1/T2` semantics must be preserved; the expanded
  pipeline does not create an invalid five-class label mapping.
- Imagery remains difficult primarily because weak intent overlaps rest,
  producing high false activation at balanced thresholds.
- Bilateral foot imagery may require broader midline spatial coverage or
  task-specific feature tuning.
- More pooled training data improves the breadth of the intent experiment but
  does not remove the clinical safety tradeoff.

## Limitations

- Participants are healthy volunteers in a cue-based laboratory study, not
  stroke patients controlling a rehabilitation device.
- The test is cue-locked. A deployable device requires continuous-window
  evaluation, false activations per minute, latency, and temporal confirmation.
- Offline zero-phase filtering is appropriate for this benchmark but not a
  complete real-time implementation.
- The same nine channels are used for fair comparison; they may not be optimal
  for both-feet imagery around the midline sensorimotor cortex.
- Amplitude-based quality control is not a substitute for clinically validated
  artifact handling.
- Task-family and modality are associated with different runs. The four
  matched comparisons are valid within the documented protocol, but a pooled
  multiclass model could learn run context and is therefore intentionally not
  reported.

## Recommended Next Steps

1. Keep all four task-specific results fixed as reference benchmarks.
2. Audit bilateral channel coverage, especially additional midline electrodes,
   without changing the unilateral reference pipeline.
3. Develop imagery-specific rest features and calibrated abstention using only
   development and controller-validation subjects.
4. Evaluate a continuous-window confirmed-command policy with false activations
   per minute, latency, and missed-intent rate.
5. Add small patient-specific calibration from early imagery runs while
   preserving later trials for unseen evaluation.
6. Compare every proposed improvement with the same final subjects and figures.

## Existing Intent-Gate Follow-up

The unilateral imagery follow-up in
[`scripts/15_rest_aware_sequential_imagery_controller.py`](scripts/15_rest_aware_sequential_imagery_controller.py)
adds per-subject calibration, quality context, OOD rejection, and sequential
evidence. Its stricter three-window policy reduced estimated false activations
from 4.18 to 2.68 per minute but increased missed intent from 50.4% to 64.4%.
This remains a useful tradeoff result rather than a clinical controller.

## Advanced Imagery Follow-up

The recommended imagery improvements are implemented together in
[`scripts/16_advanced_imagery_research_suite.py`](scripts/16_advanced_imagery_research_suite.py).
The suite preserves the fixed subject-disjoint benchmark and adds hard-negative
rest training, artifact-threshold auditing, individualized mu features,
Riemannian tangent-space features, Euclidean alignment, patient calibration,
and a 9- versus 15-channel bilateral montage comparison.

On the locked final subjects, the strongest full-test result was the aligned
Riemannian model with 10 labeled calibration trials per rest/left/right class:

| Metric | Final result |
|---|---:|
| Intent balanced accuracy | 65.0% |
| Intent sensitivity | 73.7% |
| False-activation rate | 43.7% |
| Missed-intent rate | 26.3% |

At its conservative safety point, the same calibrated model reached 61.4%
balanced accuracy, 23.4% false activation, and 53.8% missed intent. A strict
100 microvolt artifact screen reached 66.5% balanced accuracy, but retained
only 69.4% of final trials, so it is reported as a quality-control ablation
rather than a directly comparable full-test gain.

Important negative results are also retained. Individualized mu features
reached 62.3% versus 63.4% with a fixed 8-13 Hz band. The expanded montage
changed aligned unilateral imagery from 61.1% to 61.3%, while bilateral intent
fell from 57.9% to 55.0%. Euclidean alignment itself was useful on the
15-channel covariance model, improving balanced intent accuracy from 54.1% to
61.3% before patient calibration.

The full generated report is
[`artifacts/advanced_imagery/ADVANCED_IMAGERY_RESULTS.md`](artifacts/advanced_imagery/ADVANCED_IMAGERY_RESULTS.md).
Its presentation figures include the final model comparison, safety-response
plane, feature ablations, artifact retention, patient calibration, cue-confound
validation, and temporal-confirmation tradeoff.

## Physiology-First Multibranch Follow-up

The full-workflow research recommendations are now implemented in two isolated
stages:

1. [`scripts/18_build_motor_imagery_feature_atlas.py`](scripts/18_build_motor_imagery_feature_atlas.py)
   performs raw recording QC before referencing, interpolates robustly detected
   bad channels, applies CAR and surface-Laplacian CSD, and extracts separate
   slow-potential, Morlet, CSD, covariance, and quality branches.
2. [`scripts/19_multibranch_motor_imagery_benchmark.py`](scripts/19_multibranch_motor_imagery_benchmark.py)
   tunes branch models and late fusion on the controller-validation subjects,
   then evaluates the locked policy on Run 12 from the 22 final subjects.

The feature atlas contains 9,835 trials from all 109 subjects:

| Atlas item | Value |
| --- | ---: |
| Rest trials | 4,918 |
| Left-imagery trials | 2,479 |
| Right-imagery trials | 2,438 |
| Raw recordings audited | 327 |
| Slow-potential features | 55 |
| Morlet trajectory features | 1,080 |
| CSD trajectory features | 900 |
| Covariance trajectory | 5 bands x 4 phases x 15 channels |

Only one recording reached the documented QC warning level. The mean number of
interpolated channels was 3.32 per recording and the maximum was 14 of 64. No
trial was removed because of signal amplitude; signal quality remains explicit
metadata so quality-aware rejection can be evaluated rather than hidden in the
training sample.

Runs 4 and 8 provide each subject's calibration-rest reference and
task-relative mu estimate. This reference is then applied to Run 12. The
pipeline therefore does not normalize a `T0` rest trial against the movement
period immediately before it, and it does not use held-out Run 12 labels to
construct the subject reference.

Calibration repeatability was measured by comparing each development subject's
Run 4 and Run 8 effect:

| Physiology summary | Run 4 vs. Run 8 correlation |
| --- | ---: |
| Slow-potential contrast | 0.26 |
| Morlet mu contrast | 0.72 |
| CSD mu contrast | 0.72 |
| Covariance intent-rest distance | 0.78 |

The slow branch is useful to the classifier but its simple subject-level effect
is not a stable patient biomarker. Covariance and mu ERD are more repeatable
calibration targets. The prospective controller should therefore use the slow
response as complementary evidence and require confirmation from the more
repeatable branches.

![Run 4 versus Run 8 feature repeatability](artifacts/feature_atlas/08_calibration_feature_repeatability.png)

### Extended pre-ML feature audit

[`scripts/20_build_extended_motor_imagery_features.py`](scripts/20_build_extended_motor_imagery_features.py)
adds a separate extension cache without changing the reference atlas or
reopening the final cohort. It derives:

| Feature family | Trial features | Calibration-derived information |
| --- | ---: | --- |
| Spatio-spectral decomposition (SSD) | 108 | Three components in individual mu, low beta, and high beta |
| Explicit asymmetry | 528 | C3/C4, hemisphere, lateral/midline, Morlet, and CSD contrasts |
| ERD morphology | 264 | Depth, timing, area, persistence, and recovery |
| Restricted phase | 120 | PLV and signed lag for five sensorimotor channel pairs |

Eleven participant-level spectral and calibration values are stored separately
as audit metadata. They are not repeated into trial rows, which would let a
classifier use subject identity as a shortcut.

All label-based diagnostics use only the 69 development subjects and
calibration Runs 4/8. Run 12 is transformed with filters and references learned
from Runs 4/8 but does not fit those transformations.

| Feature family | Median Run 4/8 pattern cosine | Effect-magnitude correlation |
| --- | ---: | ---: |
| SSD | 0.39 | 0.71 |
| Asymmetry | 0.05 | 0.79 |
| ERD morphology | 0.50 | 0.62 |
| Restricted phase | 0.02 | 0.25 |

Morphology has the strongest within-subject effect-pattern stability and SSD is
the second strongest. Asymmetry strength repeats across subjects, but its
detailed direction/time pattern does not; only a compact, prespecified
direction subset should advance. The restricted phase branch is retained as a
negative/exploratory result and should not enter the primary fusion yet.

The median development-subject resting mu-peak SNR is 4.46 dB, with 76.8% of
subjects reaching at least 3 dB. Median Run 4/8 resting spectral correlation is
0.90, and 82.6% of subjects reach at least 0.80. These results support
participant-specific band calibration while also identifying a substantial
minority for whom a fixed or broad mu band remains necessary.

![Extended feature repeatability](artifacts/extended_features/06_extended_feature_repeatability.png)

The full audit, cache, participant profile table, and six presentation figures
are in [`artifacts/extended_features/`](artifacts/extended_features/).

### Controlled morphology/SSD ablation

The follow-up is implemented in
[`scripts/21_controlled_extended_feature_ablation.py`](scripts/21_controlled_extended_feature_ablation.py).
The development cohort fits each branch; the controller-validation cohort
selects among four prespecified equal-weight fusions; and the final cohort is
used only for a paired comparison of the reference with the
validation-selected winner. Patient adaptation, OOD rejection, temporal
confirmation, and fusion-weight search are intentionally excluded.

Controller validation selected the morphology extension:

| Validation configuration | Intent BA | False activation | Missed intent | Direction BA |
| --- | ---: | ---: | ---: | ---: |
| Slow + covariance reference | 78.3% | 21.0% | 22.5% | 76.0% |
| Reference + morphology | 78.8% | 23.2% | 19.1% | 74.9% |
| Reference + SSD | 77.5% | 21.0% | 24.0% | 73.4% |
| Reference + morphology + SSD | 76.4% | 27.0% | 20.2% | 71.9% |

On the 22 final subjects, morphology changed balanced intent BA from 68.8% to
69.5%, false activation from 23.9% to 26.4%, missed intent from 38.5% to 34.5%,
and direction BA from 76.1% to 77.9%. The paired subject-bootstrap interval for
intent-BA change was -3.2 to +4.5 percentage points, so the apparent +0.8-point
gain is not reliable. At the safety threshold, intent BA changed from 61.7% to
61.5% with the same 6.7% false activation.

Intent BA improved for 8 subjects, worsened for 8, and was unchanged for 6.
Morphology reduced missed intent for 11 subjects but increased false activation
for 11. It should therefore be retained only as an exploratory,
sensitivity-oriented proposal signal. The slow-plus-covariance reference
remains the default locked decoder, and SSD does not advance.

![Locked reference versus morphology comparison](artifacts/controlled_ablation/02_locked_final_reference_vs_winner.png)

The complete report and paired metrics are in
[`artifacts/controlled_ablation/CONTROLLED_ABLATION_RESULTS.md`](artifacts/controlled_ablation/CONTROLLED_ABLATION_RESULTS.md).

### Locked multibranch results

The controller-validation cohort selected equal late fusion of the cue-locked
slow-potential and phase-specific covariance branches. Final Run 12 results
from 22 unseen subjects were:

| Operating point | Intent BA | Sensitivity | False activation | Missed intent | Direction BA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced fusion | 68.8% | 61.5% | 23.9% | 38.5% | 76.1% |
| Safety threshold | 61.7% | 30.0% | 6.7% | 70.0% | 76.1% |
| Evidence + OOD rejection | 61.2% | 28.8% | 6.4% | 71.2% | 76.1% |
| Five-trial/class calibration | 68.3% | 61.8% | 25.2% | 38.2% | 74.5% |

The balanced fusion improves the previous strongest trial-level result from
65.0% to 68.8% intent balanced accuracy and reduces false activation from
43.7% to 23.9%. It also increases missed intent from 26.3% to 38.5%, so this is
a safer tradeoff rather than a universal gain.

Subject-clustered bootstrapping gives a 63.8--73.6% interval for balanced intent
accuracy and a 17.6--31.5% interval for false activation. The final-subject
median intent balanced accuracy is 71.7%, but two subjects are at or below
chance. This between-subject spread remains a central clinical-translation
problem even though pooled performance improved.

The ablations are informative negative findings. Patient calibration selected
on the controller cohort did not improve the final cohort. Evidence/OOD
rejection was effectively tied with threshold-only safety: it reduced false
activation by 0.3 percentage points while increasing missed intent by 1.2
points. These methods should not be described as improvements on this dataset.

![Physiology branches and late fusion](artifacts/multibranch_benchmark/01_branch_and_fusion_comparison.png)

![Locked safety-response operating points](artifacts/multibranch_benchmark/02_safety_responsiveness_operating_points.png)

![Final policy confusion matrices](artifacts/multibranch_benchmark/03_final_policy_confusion_matrices.png)

![Final-subject performance variability](artifacts/multibranch_benchmark/05_final_subject_variability.png)

The pre-ML physiology figures, QC tables, and generated interpretation report
are in [`artifacts/feature_atlas/`](artifacts/feature_atlas/). The complete
locked benchmark report is
[`artifacts/multibranch_benchmark/MULTIBRANCH_BENCHMARK_RESULTS.md`](artifacts/multibranch_benchmark/MULTIBRANCH_BENCHMARK_RESULTS.md).

## Continuous Fusion Follow-up

The device-oriented follow-up is implemented in
[`scripts/17_continuous_calibrated_fusion_controller.py`](scripts/17_continuous_calibrated_fusion_controller.py).
It processes complete imagery runs with causal filtering, one-second windows
updated every 250 ms, a 150 microvolt artifact-abstention gate, fixed-band ERD,
Euclidean-aligned covariance features, 10 calibration segments per class, and
a leaky evidence accumulator with hysteresis and cooldown.

The full cache contains 158,770 windows from 108 represented subjects. Subject
109 was excluded from development because no calibration-rest windows passed
the predefined quality gate; every locked controller and final subject remained
represented. The three controller-selected operating modes produced:

| Final operating mode | False activations/min | Sensitivity | Missed intent | Median latency |
|---|---:|---:|---:|---:|
| Safety | 1.06 | 8.8% | 91.2% | 1.95 s |
| Balanced | 5.53 | 48.8% | 51.2% | 1.90 s |
| Sensitivity target | 7.94 | 70.9% | 29.1% | 1.72 s |

The predefined joint target of at most 1 false activation per minute, at least
60% sensitivity, and at most 2.5 seconds latency was not met. Artifact
abstention was directionally useful but not sufficient: removing it from the
safety policy changed 1.06 to 1.24 false activations per minute and 8.8% to
9.4% sensitivity. This result supports a separate user-arming or confirmation
mechanism for rehabilitation control rather than further threshold tightening.

The complete report and figures are in
[`artifacts/continuous_fusion/CONTINUOUS_FUSION_RESULTS.md`](artifacts/continuous_fusion/CONTINUOUS_FUSION_RESULTS.md).

## Future Work: Confirmed-Intent Rehabilitation Control

The continuous results indicate that EEG should provide a **proposed motor
intent**, not directly energize the rehabilitation device. The safety policy
approaches the false-activation target but misses 91.2% of intents, while the
sensitivity-target policy detects 70.9% of intents but produces 7.94 false
activations per rest minute. Further threshold tightening cannot resolve this
overlap by itself.

Future development should therefore use a confirmed-intent architecture:

1. Use the patient-calibrated ERD/Riemannian fusion model to generate a
   time-limited left/right intent proposal. Compare the balanced and
   sensitivity-target modes as proposal generators rather than actuator
   controllers.
2. Require an independent confirmation signal before movement. Candidate
   confirmation sources include residual EMG when available, a deliberate
   physical switch, therapist approval during early rehabilitation, or a
   device-side interaction signal that is independent of the EEG features.
3. Implement explicit `disarmed`, `armed`, `intent proposed`, `confirmed`, and
   `timeout` controller states. EEG evidence alone may enter the proposal state
   but must never directly enter the movement state.
4. Evaluate the combined system on long asynchronous recordings containing
   natural no-control periods, transitions, distraction, jaw/neck movement,
   eye movement, and electrode disturbances. Cue-isolated accuracy should
   remain secondary to command-event safety.
5. Preserve patient-specific calibration and update only from confirmed,
   high-quality examples. Later sessions must remain held out to measure
   session-to-session drift and recalibration requirements.
6. Validate the final controller with the intended stroke-rehabilitation
   population. EEGMMIDB establishes the algorithmic challenge but cannot
   establish clinical safety or performance.

The next-stage joint acceptance criteria should remain at most **1 false
activation per rest minute**, at least **60% intent sensitivity**, and at most
**2.5 seconds median detection latency**. Direction error among confirmed
commands, confirmation rejection rate, artifact-abstention rate, calibration
time, and performance across later sessions should also be reported. A command
should be considered successful only when the EEG proposal and independent
confirmation agree within a predefined time window.

## Reproducibility

Run the expanded benchmark from the project root:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
  scripts/14_dual_execution_imagery_benchmark.py
```

The first run builds `artifacts/cross_task/all_task_epochs.npz`. Later runs
reuse that cache unless `--rebuild-cache` is supplied.

Outputs in `artifacts/dual_task_benchmark/` include:

- `dual_task_metrics.json`: full protocol, four-condition metrics, and pooled
  intent metrics.
- `dual_task_final_predictions.csv`: task-specific final probabilities and
  predictions with correct class names.
- `pooled_intent_final_predictions.csv`: generalized binary intent outputs.
- `dual_task_fbcsp_models.joblib`: four task-specific decoders and pooled gate.
- `DUAL_TASK_BENCHMARK_RESULTS.md`: concise generated result report.
- `dual_pipeline_overview.png`: expanded four-condition pipeline.
- `raw_vs_processed_eeg.png`: cue-aligned raw and processed C3/Cz/C4 traces.
- `full_trial_mu_beta_eeg.png`: full T0-to-task timing with broadband, mu, and
  beta traces.
- `dual_task_abstract_key_findings.png`: headline performance and safety figure.
- `dual_task_confusion_matrices.png`: four normalized confusion matrices.
- `dual_task_per_subject_comparison.png`: subject-level family comparisons.
- `dual_task_safety_tradeoff.png`: task-specific and pooled intent curves.
- `pooled_intent_gate_summary.png`: generalized intent-gate result.

The implementation is in
[`scripts/14_dual_execution_imagery_benchmark.py`](scripts/14_dual_execution_imagery_benchmark.py).

Run the advanced imagery suite with the same interpreter:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/16_advanced_imagery_research_suite.py
```

The first run builds reusable caches in `artifacts/advanced_imagery/`. Later
runs reuse those caches unless `--rebuild-feature-cache` or
`--rebuild-bilateral-cache` is supplied.

Run the continuous fusion experiment with:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/17_continuous_calibrated_fusion_controller.py
```

The first run creates `artifacts/continuous_fusion/continuous_features.npz`.
Later runs reuse the feature cache unless `--rebuild-cache` is supplied.

Build or refresh the physiology feature atlas with:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/18_build_motor_imagery_feature_atlas.py
```

The first run reads all 327 unilateral imagery EDF recordings and creates
`artifacts/feature_atlas/motor_imagery_feature_atlas.npz`. Later runs reuse that
cache unless `--rebuild-cache` is supplied.

Run the locked multibranch benchmark with:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/19_multibranch_motor_imagery_benchmark.py
```

The benchmark reuses the atlas and the aligned covariance cache. Supply
`--rebuild-covariance-cache` only when the covariance representation changes.

Build or refresh the extended pre-ML feature audit with:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/20_build_extended_motor_imagery_features.py
```

The script reuses `artifacts/extended_features/extended_motor_imagery_features.npz`
unless `--rebuild-cache` is supplied. It does not train a classifier or evaluate
the final cohort. The next model stage should add morphology, SSD, and then the
compact asymmetry subset one at a time using only development and
controller-validation decisions.

Run the controlled morphology/SSD ablation with:

```bash
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/21_controlled_extended_feature_ablation.py
```

This experiment is complete. It preserves the slow-plus-covariance reference
as the default because morphology produced an uncertain sensitivity/false-
activation tradeoff and SSD did not improve controller validation.

## Future Research and Clinical Translation Roadmap

The detailed signal-extraction audit and proposed multibranch workflow are in
[`BCI_FULL_WORKFLOW_RESEARCH.md`](BCI_FULL_WORKFLOW_RESEARCH.md). That report
prioritizes raw-channel QC, robust referencing, CSD, slow cue-locked potentials,
compact Morlet trajectories, phase-specific covariance, and calibrated late
fusion before any larger classifier is introduced.

### Proposed Research Direction

The next stage should focus on protocol design, patient calibration, and
controller safety rather than only increasing model capacity. A suitable
long-term research theme is:

> **Safety-aware, patient-adaptive detection of weak cortical motor intent for
> stroke rehabilitation.**

The intended system would combine a pretrained population model, a short
patient-specific calibration, cue-constrained EEG detection, sequential
confirmation, and an independent confirmation signal when one is available.
The initial clinical objective is scheduled rehabilitation, where a patient is
asked to attempt or imagine a movement and the BCI determines whether credible
motor intent follows. Fully self-paced prosthetic control is a separate and
more difficult endpoint that requires long natural-idle recordings.

### Cue-Constrained Calibration

A visual object or command can improve calibration by identifying when motor
intent is expected. It must arm the detector rather than directly trigger the
device. The cue identity, screen position, and event timestamp must not be
provided as predictive inputs to the EEG intent decoder; otherwise, the model
could learn visual processing or experimental timing instead of motor intent.

The existing cue-to-intent audit supports this separation. On untouched
subjects, early visual-region features produced `48.8%` left/right balanced
accuracy, while later motor ERD features produced `56.8%`. The visual cue is
therefore useful for opening an evaluation interval, but command evidence
should come from later central sensorimotor activity.

A practical clinical calibration session could use:

| Stage | Approximate duration | Purpose |
| --- | ---: | --- |
| Signal-quality baseline | 5 min | Rest distribution, impedance, and usable channels |
| Artifact characterization | 3 min | Blinks, jaw/neck activity, motion, and saturation limits |
| Open-loop labeled calibration | 10 min | Initial patient alignment and intent features |
| Model configuration | 1-2 min | Bands, channels, calibration, and controller thresholds |
| Closed-loop assisted calibration | 10 min | Co-adaptation with contingent feedback |
| Locked evaluation | 10 min | Unadapted test on later trials |

Each trial should separate visual processing from motor preparation:

```text
Fixation/rest
    -> neutral alert cue
    -> randomized delay (approximately 0.8-1.5 s)
    -> left/right instruction
    -> randomized delay
    -> identical "attempt now" cue
    -> EEG evidence interval (approximately 0.5-3.0 s)
    -> feedback only after sufficient motor-intent evidence
```

The prospective protocol should:

- Use the same final go cue for every direction and randomize delay durations.
- Include cue-only catch trials in which the instruction appears but the patient
  must remain at rest.
- Include observation-only, passive-movement, imagery, and active-attempt
  conditions as separately labeled controls where fatigue permits.
- Synchronize cue presentation and EEG with hardware event markers or a
  photodiode rather than relying only on software timestamps.
- Record EOG, residual EMG, force or bend sensors, and arm IMUs during
  calibration. Initially use them for artifact rejection and ground truth, not
  as EEG decoder features.
- Freeze later blocks and later days before model development so calibration
  performance is not confused with longitudinal generalization.

A cue-constrained asynchronous BCI in people with stroke reported `84.88%`
true-positive detection and `0.52` false positives per minute, compared with
`0.99` false positives per minute for its continuously active asynchronous
condition. These values are not directly comparable with this project's
EEGMMIDB results, but they provide evidence that cue-based arming can reduce
unsafe opportunities without making the cue itself the command
([Niazi et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/)).

### Dual-Physiology Intent Decoder

The next decoder should preserve complementary physiological signals in
separate branches:

| Branch | Candidate range | Primary information |
| --- | ---: | --- |
| MRCP | `0.1-3 Hz` | Movement preparation and attempted-movement onset |
| ERD | Patient-specific mu/beta | Sustained imagery or attempted movement |
| Quality/OOD | Broadband, EOG, amplitude, impedance | Artifact rejection and abstention |
| Optional EMG | Approximately `20-450 Hz` | Residual peripheral confirmation |

The current causal `4-35 Hz` pipeline excludes most movement-related cortical
potential (MRCP) information. The MRCP branch should therefore have its own
causal filter, baseline correction, and spatial features before late fusion
with ERD/Riemannian evidence. In a closed-loop stroke exoskeleton study, MRCP
intent was detected an average of `367 ms` before movement onset, supporting
its inclusion as a complementary preparation signal
([Bhagat et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4815250/)).

Fusion should occur at the calibrated evidence level rather than by simply
concatenating every feature. The controller should separately expose MRCP, ERD,
quality, and optional EMG contributions so that false commands can be audited.

### Patient Profile and Longitudinal Adaptation

The global representation should initially remain frozen. A small, reversible
patient profile should estimate:

- Rest covariance and session-specific alignment.
- Individual mu and beta peak frequencies.
- Relevant contralateral and ipsilesional motor channels.
- MRCP polarity, timing, and spatial pattern.
- Intent probability calibration and operating threshold.
- Artifact, uncertainty, and OOD limits.
- Sequential evidence, hysteresis, and cooldown parameters.

Short co-adaptive calibration is plausible: a study with severely impaired end
users automatically calibrated a three-channel motor-imagery BCI in less than
five minutes while explicitly modeling a non-control state
([Faller et al., 2014](https://pubmed.ncbi.nlm.nih.gov/25014055/)). Calibration
must also be evaluated longitudinally. In an 11-patient, 12-session stroke
dataset, combining earlier patient sessions eventually outperformed a model
trained only on the calibration immediately preceding the test session
([Arvaneh et al., 2013](https://pubmed.ncbi.nlm.nih.gov/23366835/)).

Updates should use only confirmed, high-quality examples. Every update must be
versioned, reversible, and evaluated against a locked later session to detect
negative adaptation.

### Proposed PhD Research Aims

| Phase | Research question | Primary deliverable |
| --- | --- | --- |
| 1. Retrospective benchmark | Which physiological representation transfers across unseen subjects? | Locked MRCP, ERD, Riemannian, FBCSP, and EEGNet comparison |
| 2. Calibration study | How little patient data is needed for useful adaptation? | Performance curves at 2, 5, 10, and 15 min of calibration |
| 3. Longitudinal study | Does the profile remain reliable across days and weeks? | Drift analysis and safe update policy |
| 4. Cue-gated feasibility | Can intended attempts be detected safely during scheduled therapy? | Online controller reporting FP/min, sensitivity, and latency |
| 5. Hybrid confirmation | When does residual EMG improve an EEG proposal? | EEG-only, EMG-only, and calibrated-fusion comparison |
| 6. Clinical pilot | Is repeated use safe, usable, and physiologically meaningful? | Prospective stroke feasibility study |
| 7. Controlled trial | Does contingent EEG feedback improve recovery? | Clinically powered trial, likely beyond the initial PhD |

The most coherent thesis would connect the first six phases around one claim:
a population EEG model can be safely individualized with limited patient data,
and cue-constrained sequential detection can improve the sensitivity/false-
activation tradeoff across later sessions. A definitive rehabilitation-efficacy
trial should follow only after the engineering and feasibility endpoints are
met.

### Required Outcomes and Advancement Criteria

Evaluation should include three outcome layers:

- **Engineering:** false activations per minute/hour, intent sensitivity,
  missed-intent rate, command latency, conditional direction error,
  abstention, artifact rejection, and calibration burden.
- **Physiology:** MRCP amplitude and timing, mu/beta ERD, laterality, EEG-EMG
  timing, and longitudinal feature stability.
- **Clinical and human factors:** FMA-UE, ARAT, Box and Block, Motor Activity
  Log, fatigue, workload, usability, and adverse events selected with the
  clinical team.

Contingent cortical feedback may have value beyond controller accuracy. In a
randomized EEG-triggered FES study, the BCI group improved by `6.6 +/- 5.6`
FMA-UE points, retained the improvement at 6-12 months, and showed related
motor-network connectivity changes
([Biasiucci et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29925890/)). This is
motivation for a clinical study, not evidence that the current pipeline will
produce the same outcome.

The immediate advancement target remains no more than `1` false activation per
rest minute, at least `60%` intent sensitivity, and no more than `2.5 s` median
latency on completely locked subjects and sessions. The next concrete
experiment should add the MRCP branch and cue-gated state machine, then compare
them with the current ERD/Riemannian controller under identical final-subject
splits. Prospective stroke collection should begin only with clinical
collaboration, ethics approval, synchronized validation sensors, predefined
stopping rules, and a device-side safety layer independent of the EEG decoder.

## External Dataset Expansion and Research Console

The next retrospective phase should expand beyond EEGMMIDB without changing
the locked EEGMMIDB final-subject benchmark. The first healthy addition should
be the 51-subject, three-session WBCIC Yang 2025 two-class dataset. Cho2017 and
OpenBMI can then add 106 more healthy participants. This creates a
multi-device, multi-session development pool while preserving EEGMMIDB as a
fixed comparison.

Clinical transfer should be measured separately. The 50-patient Liu acute
stroke dataset and 30-patient UET175 post-stroke dataset are the first
priorities. The new 14-patient HS-Stroke longitudinal dataset is potentially
valuable, but its corrected release must be audited before use: the repository
now reports 23,382 trials after removing within-session duplicates, rather
than the article-in-press total of 57,902.

External recordings must pass a dataset-specific adapter that verifies units,
channel names, references, cue timing, imagery timing, rest semantics, and
subject/session IDs. Evaluation must report zero-shot transfer,
patient-calibration curves, leave-one-dataset-out results, and continuous
false activations rather than only pooled trial accuracy.

### Interim OpenBMI Evidence

The first controlled OpenBMI experiment is now complete on the 29 locally
available subjects. It uses the original study's 20 central channels,
1.0-3.5 second motor-imagery window, paired 5.0-7.5 second rest, and separate
development, controller-validation, and untouched final subjects.
Calibration-only log-Euclidean alignment produced 82.7% intent balanced
accuracy with 12.4% false activation and 22.2% missed intent on six untouched
subjects. Twenty labeled calibration trials per class improved these values to
87.7%, 11.5%, and 13.1%, respectively.

The direction head remains the limiting stage: left/right balanced accuracy
was 51.6% globally and 53.7% after calibration. By contrast, same-subject
CSP/LDA averaged 68.0%, indicating subject-specific direction information that
does not transfer reliably. The early occipital cue control reached 57.9% but
fell to chance in the later imagery window, making cue leakage visible rather
than silently crediting it as motor decoding. The full reproducible result is
in `artifacts/openbmi_controlled/OPENBMI_CONTROLLED_REPORT.md`.

### Harmonized Multi-Dataset Result

The follow-up comparison now holds channels, filtering, sampling, imagery
window, filter-bank covariance features, log-Euclidean alignment, decoder,
subject isolation, calibration budgets, and safety selection constant across
PhysioNet EEGMMIDB and OpenBMI. Equal total dataset weighting prevents the
larger OpenBMI trial count from dominating pooled training.

The matched experiment shows two different strengths. Dataset-specific
calibration produced 65.8% intent balanced accuracy and 58.4% direction
balanced accuracy on PhysioNet, compared with 79.8% intent and 52.4% direction
on OpenBMI. Pooled calibration reached 59.4% intent and 59.9% direction on
PhysioNet, and 83.3% intent and 53.2% direction on OpenBMI. Pooling therefore
does not provide a universal gain: it improves OpenBMI intent and PhysioNet
direction, but harms PhysioNet intent relative to dataset-specific
calibration.

Label-free aligned transfer is the clearest warning. A PhysioNet-trained model
reached 61.8% intent balanced accuracy on OpenBMI, while an OpenBMI-trained
model reached only 52.5% on PhysioNet and missed 89.5% of PhysioNet intent
under its source-selected safety policy. Alignment used unlabeled target
calibration recordings, but decoder and threshold fitting used no target
labels. This supports a multi-domain architecture: share motor representations
where possible, but use dataset/device/patient specific intent calibration and
safety thresholds. The full result is in
`artifacts/harmonized_physionet_openbmi/HARMONIZED_BENCHMARK_REPORT.md`.

### Healthy-to-Stroke Hand-Intent Result

The first complete healthy-to-stroke experiment is implemented in
[`scripts/27_healthy_to_stroke_hand_intent.py`](scripts/27_healthy_to_stroke_hand_intent.py).
It excludes all PhysioNet bilateral fist/foot runs and makes binary hand
`motor_intent` versus `rest` the primary endpoint. Conditional left/right hand
decoding remains a secondary measurement.

The three-dataset intersection contains seven central channels: `FC3`, `FC4`,
`C3`, `Cz`, `C4`, `CP3`, and `CP4`. All datasets are harmonized to full-EEG
CAR, 4-38 Hz filtering, 100 Hz, two-second windows, five-band covariance
features, and tangent-space logistic regression. PhysioNet contributes only
runs `4/8/12`; OpenBMI contributes left/right hand imagery and paired rest;
Liu contributes left/right hand imagery and stage-3 rest. Instruction periods
are never labeled as rest.

The Liu extraction retained 3,798 windows from all 50 patients. It rejected
101 complete intent/rest trial pairs under the shared 250-microvolt
peak-to-peak criterion. The fixed split contains 31 stroke-development, 8
controller-validation, and 10 final patients. The final patients do not
contribute labels to model or threshold selection; their early trials are used
only in the explicitly labeled patient-calibration condition, with later
trials retained for evaluation.

On the ten locked Liu patients:

| Method | Intent BA | Sensitivity | Specificity | False activation | Missed intent |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen healthy, no target alignment | 56.5% | 48.4% | 64.5% | 35.5% | 51.6% |
| Frozen healthy, label-free patient alignment | 56.2% | 34.9% | 77.4% | 22.6% | 65.1% |
| Healthy plus stroke population adaptation | 62.1% | 40.9% | 83.3% | 16.7% | 59.1% |
| Population model plus patient calibration | 61.8% | 32.8% | 90.9% | 9.1% | 67.2% |

The subject-bootstrap 95% interval for intent balanced accuracy was
54.6-70.4% for population adaptation and 53.5-70.5% after patient
calibration. Patient calibration selected five labeled windows per class with
a 25% local-model blend. It should be interpreted as a safety operating point:
it reduced false activation below 10%, but it did not improve sensitivity.
The population-adapted model is preferable when missed intent is more costly;
the calibrated model is preferable when false activation is the primary risk.

The corresponding pooled healthy model reached 72.6% intent balanced accuracy,
60.1% sensitivity, and 85.2% specificity across the combined final healthy
cohort. Performance remained dataset dependent: PhysioNet intent balanced
accuracy was 54.3%, compared with 77.6% on OpenBMI. This difference, together
with the exploratory mu/beta feature shifts, confirms that acquisition and
protocol remain major confounds in the healthy-versus-stroke comparison.

Run the complete cached experiment with:

```bash
python scripts/27_healthy_to_stroke_hand_intent.py
```

The full report, machine-readable metrics, final predictions, patient-level
CSV data, models, and six poster-ready figures are in
`artifacts/healthy_to_stroke_hand_intent/`. These results do not yet satisfy
the combined advancement target of at least 60% sensitivity and no more than
10% false activation. The next retrospective work should therefore improve
the intent evidence itself, especially with a separate MRCP branch, before
tightening the safety threshold further.

### Compact Classical-Feature Ablation

The inertial-sensor-style statistical/PSD workflow was translated into an EEG-
appropriate controlled ablation in
[`scripts/28_classical_feature_model_ablation.py`](scripts/28_classical_feature_model_ablation.py).
The extractor generates 310 compact features grouped into statistical, Welch
spectral, temporal-morphology, and spatial-asymmetry families. It preserves
mu/beta information rather than copying the source project's 0.5-10 Hz filter,
and all normalization uses only each subject's early calibration recording.

Feature family and model type were selected only on the eight Liu controller
patients. The statistical family won with 56 features: log standard deviation,
skewness, excess kurtosis, log line length, zero-crossing rate, Hjorth mobility,
Hjorth complexity, and log peak-to-peak amplitude for each of seven channels.
Regularized logistic regression outperformed RBF-SVM, gradient boosting,
random forest, and a shallow decision tree on the controller criterion.

On the same ten final Liu patients:

| Pipeline | Intent BA | Sensitivity | Specificity | False activation | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Covariance population reference | 63.2% | 43.0% | 83.3% | 16.7% | 68.5% | 70.0% |
| Statistical population model | 68.5% | 50.5% | 86.6% | 13.4% | 75.9% | 77.3% |
| Covariance plus calibration | 65.9% | 41.4% | 90.3% | 9.7% | 72.2% | 73.5% |
| Statistical model plus calibration | 68.5% | 49.5% | 87.6% | 12.4% | 77.9% | 79.4% |

The population intent-BA change was +5.4 percentage points, but its
subject-bootstrap 95% interval was -5.3 to +14.4. The calibrated change was
+2.8 points (95% CI -6.0 to +12.2), improving six patients, tying one, and
worsening three. The statistical features therefore provide promising
additional intent evidence, but the uncertainty still includes no effect and
the calibrated false-activation rate exceeds the 10% target.

The compact statistical branch should be retained as a candidate proposal
signal for late fusion with the covariance safety gate. It should not replace
the covariance controller until nested subject-level validation or a new
locked cohort confirms the gain. This final cohort had already been observed
in the preceding benchmark, so this is an exploratory paired ablation rather
than a new untouched confirmation. The full report and five figures are in
`artifacts/classical_feature_model_ablation/`.

### Welch PSD and Morlet Spatial-Temporal Fusion

The follow-up experiment in
[`scripts/29_psd_morlet_fusion_ablation.py`](scripts/29_psd_morlet_fusion_ablation.py)
tests whether preserving more frequency, time, and scalp information improves
the three-dataset hand-intent pipeline. Welch PSD is treated as feature
extraction rather than preprocessing: it contributes absolute and relative
band powers, spectral entropy, spectral centroid, and mu/beta peak summaries.
The new variable-cycle Morlet branch extracts 207 features from 8-30 Hz,
including four temporal bins, trajectory slopes, late-minus-early changes,
left-right FC/C/CP contrasts, and central-channel means. All features are
normalized from each subject/session's recording-zero calibration windows
only.

Controller-only selection favored the 263-feature statistical-plus-Morlet
branch. On the eight controller stroke patients it reached 70.5% intent
balanced accuracy, 51.0% sensitivity, 90.1% specificity, and 9.9% false
activation. Morlet alone reached 69.5% balanced accuracy, while Welch PSD alone
reached 57.0%. Direct PSD-plus-Morlet fusion reached 65.9%, showing that more
features do not automatically provide more transferable information.

On the same ten previously observed final Liu patients:

| Pipeline | Intent BA | Sensitivity | Specificity | False activation |
| --- | ---: | ---: | ---: | ---: |
| Covariance population reference | 63.2% | 43.0% | 83.3% | 16.7% |
| Statistical population reference | 68.5% | 50.5% | 86.6% | 13.4% |
| Statistical plus Morlet population | 68.5% | 55.9% | 81.2% | 18.8% |
| Covariance plus calibration | 65.9% | 41.4% | 90.3% | 9.7% |
| Statistical plus calibration | 68.5% | 49.5% | 87.6% | 12.4% |
| Statistical plus Morlet and calibration | 68.3% | 52.2% | 84.4% | 15.6% |

Morlet therefore recovered some additional intent sensitivity, but the gain
was accompanied by more false activation and no balanced-accuracy
improvement. Relative to the statistical branch, calibrated balanced accuracy
changed by -0.2 percentage points with a paired subject-bootstrap 95%
confidence interval of -6.6 to +6.8 points. The Morlet branch should be kept
for physiological audit, patient phenotyping, and future temporally confirmed
evidence fusion. It should not replace the covariance safety gate or compact
statistical proposal branch in the current command controller.

The output directory `artifacts/psd_morlet_fusion_ablation/` contains exact
JSON/CSV metrics, trained models, a Welch PSD comparison, Morlet
time-frequency contrasts, late-window scalp maps, and an animated GIF of
healthy-versus-stroke mu and beta scalp dynamics. These visuals report
intent-minus-rest power in dB. They are not strict pre-cue ERD or source
localization because the harmonized benchmark uses two-second task/rest
windows and only seven shared electrodes.

The local research console is launched with:

```bash
python scripts/22_bci_pipeline_gui.py
```

It supports standard EEG and array uploads, raw-versus-processed signal audit,
mu/beta visualization, the fixed EEGMMIDB feature and ablation workflow, file
replay, and optional LSL/BrainFlow capture. Live data are intentionally limited
to capture and signal audit until synchronized cue markers, channel identity,
causal preprocessing, and patient calibration have been validated.

The full dataset catalog, harmonization contract, training sequence, metrics,
and console instructions are in
[EXTERNAL_DATASETS_AND_RESEARCH_CONSOLE.md](EXTERNAL_DATASETS_AND_RESEARCH_CONSOLE.md).
