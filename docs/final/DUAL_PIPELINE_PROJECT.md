# Expanded EEG Benchmark: Physical Movement vs. Motor Imagery

> **Historical reference benchmark:** This document preserves the original
> matched movement-versus-imagery study. The current cross-dataset binary
> decoder, patient-calibration protocol, and V7 results are documented in
> [`EEG_MOTOR_INTENT_RESEARCH_PROJECT.md`](EEG_MOTOR_INTENT_RESEARCH_PROJECT.md).

## Project Purpose

This project investigates a central challenge in rehabilitation brain-computer
interfaces (BCIs): detecting weak motor intent from EEG without issuing unsafe
commands while the user is at rest. Physical movement provides a stronger
physiological reference signal; motor imagery is the intended noninvasive
control signal.

The expanded benchmark now uses both motor-task families in the PhysioNet EEG
Motor Movement/Imagery Dataset (EEGMMIDB):

- Unilateral tasks: `rest`, `left_fist`, and `right_fist`.
- Bilateral tasks: `rest`, `both_fists`, and `both_feet`.

Movement and imagery are evaluated with matched preprocessing, features,
models, subject partitions, and operating-threshold selection. Execution data
are not pooled into a task-specific imagery decoder, preserving an honest
imagery result. A separate pooled binary experiment tests whether all task runs
can improve generalized `rest` versus `motor_intent` detection.

## Research Questions

1. How much harder is imagery than execution when the analysis is held fixed?
2. Does the movement-to-imagery gap appear in both unilateral hand and
   bilateral hand/foot tasks?
3. Is imagery limited mainly by task-specific discrimination or by detecting
   weak intent against rest?
4. Can all correctly mapped task runs support a more general binary intent
   gate?
5. What false-activation versus missed-intent tradeoff would a rehabilitation
   controller inherit from these decoders?

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
