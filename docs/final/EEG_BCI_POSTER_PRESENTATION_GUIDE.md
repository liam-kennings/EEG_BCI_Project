# EEG BCI Poster Presentation Guide

## Purpose

This guide follows the reading order of the poster **Improving Weak
Motor-Intent Detection in EEG-Based Rehabilitation BCIs**. It is designed for
review immediately before presenting. The poster reports the EEG-only V7
development-observed experiment; it does not report prospective clinical,
continuous-control, or real MIND EEG-fNIRS results.

## The Story in One Sentence

This project trains a general EEG model across healthy and stroke datasets,
adapts it with a short patient calibration, and uses a conservative binary
REST-versus-INTENT gate to prioritize low false activation for future
rehabilitation control.

## Recommended Presentation Flow

Use the poster from upper left to upper right, then lower left to lower right:

1. **Current Methods:** explain the clinical and engineering problem.
2. **Data Processing and Decoder Pipeline:** explain how the project responds
   to that problem.
3. **Results:** explain the safety-performance tradeoff and domain differences.
4. **Next Steps:** explain what is required before device or clinical use.

For a short poster interaction, spend roughly one minute on the problem, two
minutes on the pipeline, two minutes on results, and one minute on limitations
and next steps.

## 30-Second Opening

> Noninvasive EEG could help confirm that a rehabilitation patient is trying or
> imagining a movement even when visible movement or reliable muscle output is
> limited. The difficulty is that motor-intent EEG is weak, noisy, and highly
> variable between people and recording systems. My project developed a model
> across three EEG datasets, then adapted it to each patient using a brief
> calibration. The conservative V7 result reached 73.6% balanced accuracy and
> 93.7% rest specificity, but it still missed 46.5% of intent windows. That
> makes this a promising intent-confirmation framework, not yet a clinically
> validated device controller.

---

## Section 1: Current Methods

### Main message

Common motor-imagery research pipelines can report useful offline accuracy but
still fail the safety and generalization requirements of rehabilitation.

### Common research approach

The top row shows a typical sequence:

1. A controlled visual or auditory cue tells the participant which task to do.
2. A clean, cue-locked EEG window is extracted.
3. A classifier predicts REST, LEFT, or RIGHT.
4. The classifier is forced to choose a class for every window.

The method is useful for algorithm development, but it is easier than
continuous real-world operation because the model already knows approximately
when the intent should occur.

### Why this breaks down in rehabilitation

**Weak signal:** Motor imagery and attempted movement can produce faint EEG
changes that are obscured by eye activity, muscle activity, electrode noise,
and environmental interference.

**Domain shift:** EEG distributions change across people, sessions, devices,
datasets, and stroke status. A model that performs well on one group may not
transfer directly to another.

**Safety gap:** A forced multiclass decision can convert uncertain data into a
device command. Overall accuracy can also hide poor intent detection or poor
rest rejection.

### This project's response

**Generalize:** learn common structure across multiple datasets rather than one
small patient-specific recording.

**Personalize:** adapt the population model with a brief early recording from
the new patient.

**Protect:** issue only a binary REST or INTENT decision, reject weak evidence,
and prepare temporal confirmation before future device action.

### Transition to the pipeline

> The rest of the poster shows how I turned those three goals into the V7
> processing and decoder pipeline.

---

## Section 2: Data Processing and Decoder Pipeline

### 01 - Datasets

The population model uses three EEG domains:

| Dataset | Population | Poster contribution |
|---|---:|---:|
| PhysioNet EEGMMIDB | 109 healthy participants | 9,836 windows |
| OpenBMI | 29 healthy participants | 23,200 windows |
| Liu acute stroke | 49 stroke participants | 3,867 windows |
| **Total represented** | **187 participants** | **36,903 windows** |

Each dataset has different hardware, montage, timing, and rest definitions.
Dataset identity is preserved so calibration and controller choices can account
for those differences instead of pretending every recording is identical.

**What to say:**

> The purpose of combining datasets was not simply to increase sample count. It
> was to expose the model to multiple recording domains, including healthy and
> acute-stroke EEG, while preserving which domain each window came from.

### 02 - Native QC and harmonization

The recordings are cleaned in their native montage before being reduced to the
shared motor-region channels.

1. Detect and repair bad EEG channels.
2. Apply common-average referencing, or CAR.
3. Resample all recordings to 100 Hz.
4. Select seven shared motor-region channels: FC3, FC4, C3, Cz, C4, CP3, CP4.
5. Retain dataset identity for later calibration.

**Why repair before CAR:** a severely corrupted channel can contaminate the
average reference if it is included before repair.

**Why seven channels:** these channels cover bilateral frontal-central,
central, and centro-parietal sensorimotor regions while remaining available
across all three datasets.

### 03 - Windowing

The poster simplifies windowing as a three-second cue interval containing REST,
LEFT, or RIGHT labels. The implemented harmonized V7 model input is a
**two-second analysis window beginning 0.5 seconds after event onset**. Treat
the poster's 0-3 second graphic as the task/cue timeline, not the exact feature
window specification.

- Recording 0 is reserved for patient calibration.
- Later recordings are held out for evaluation.
- Calibration and evaluation windows therefore come from different recording
  periods.

**Important limitation:** cue-locked windowing is not the same as continuous,
asynchronous control. The cue provides timing information that would not be
available during unrestricted everyday use.

### 04 - EEG representations

The system represents the same EEG window in complementary ways.

**MRCP, 0.1-4 Hz:** captures slow cortical voltage changes associated with
movement preparation and intention.

**SMR, 7-35 Hz:** captures sensorimotor rhythms, especially changes in mu and
beta activity during motor imagery.

**Riemannian covariance:** captures coordinated spatial activity across
sensorimotor channels in four frequency bands. The method respects the curved
geometry of covariance matrices rather than treating them as ordinary flat
vectors.

**Quality and OOD evidence:** quality features remain available so unusual or
artifact-like windows can be rejected rather than forced into a command.

**Why multiple representations:** slow preparation potentials, oscillatory
desynchronization, and spatial covariance contain different information. No
single representation was consistently strongest across every dataset.

### 05 - General model

The population model uses controller-selected late fusion:

| Branch | Fusion weight |
|---|---:|
| MRCP + FBCNet | 60% |
| Riemannian covariance | 25% |
| SMR statistics | 15% |

**Late fusion** means each branch first produces its own evidence or
probability, and those outputs are combined afterward. This is different from
concatenating all raw features before classification.

**FBCNet** is a filter-bank convolutional neural network. It learns spatial and
spectral patterns from multiple frequency-filtered EEG views.

After fusion, each dataset receives probability calibration and an operating
threshold selected without using the final later-recording evaluation rows.

### 06 - Patient calibration

Each patient contributes 30 early calibration windows:

- 10 REST
- 10 LEFT
- 10 RIGHT

LEFT and RIGHT are auxiliary internal labels. The device-facing probability is:

```text
P(INTENT) = P(LEFT) + P(RIGHT)
```

The direction labels help characterize the patient's EEG response, but the
external output remains only REST or INTENT. In the intended rehabilitation
workflow, the cue defines which movement is requested and EEG confirms whether
credible intent followed.

### 07 - Intent gate

The conservative threshold produces:

- **REST:** no command.
- **INTENT:** confirm that motor-intent evidence followed the known cue.

The prepared temporal controller requires three qualifying windows before a
future command and then applies a two-second refractory period. This
confirmation rule was not evaluated in the poster result, so the poster reports
window-level specificity rather than false commands per minute.

### One-sentence pipeline summary

> V7 quality-checks and harmonizes EEG, extracts complementary slow, rhythmic,
> and spatial evidence, fuses calibrated population models, adapts the result
> using 30 early patient windows, and applies a conservative binary intent gate
> to later recordings.

---

## Section 3: Results

### Evaluation contract

The headline result uses 4,645 later-recording evaluation windows. It is
development-observed: model and controller design decisions were made during
the project using the available datasets. It is not a newly locked prospective
or external clinical test.

### Headline metrics

| Metric | Result | Plain-language meaning |
|---|---:|---|
| Balanced accuracy | **73.6%** | Average of intent sensitivity and rest specificity |
| Intent sensitivity | **53.5%** | Detected 53.5% of true intent windows |
| Rest specificity | **93.7%** | Correctly rejected 93.7% of analyzed rest windows |
| Intent AUROC | **0.896** | Strong threshold-independent ranking of intent above rest |
| 95% bootstrap CI for BA | **67.5-78.7%** | Participant-level uncertainty around balanced accuracy |

The balanced-accuracy calculation is:

```text
(53.5% sensitivity + 93.7% specificity) / 2 = 73.6%
```

### Safety result and its limitation

Rest specificity of 93.7% corresponds to a 6.3% false-positive fraction per
analyzed rest window:

```text
False-positive fraction = 1 - specificity = 6.3%
```

Do not call this 6.3% false commands. Overlapping windows, temporal
confirmation, refractory periods, and continuous recording duration determine
the number of actual command events. False activations per minute have not yet
been measured.

### Sensitivity tradeoff

The conservative point detects 53.5% of intent and therefore misses 46.5%.
The sensitivity-targeted option detects 58.5% but lowers specificity to 89.6%.

**Interpretation:** reducing missed intent currently requires accepting more
false-positive windows. The conservative point was chosen because unintended
device activation is the primary safety concern for this project.

### Patient adaptation

The matched later-recording balanced accuracy increased from 69.1% with the
population model to 73.6% after early patient calibration, a gain of 4.5
percentage points.

This supports rapid personalization, but it does not prove the same gain will
occur for every future patient. The 95% interval and domain-level results show
substantial variability.

### Model progression graph

| Version | Main design | BA | Sensitivity | Specificity |
|---|---|---:|---:|---:|
| V5 | Directional calibration | 73.3% | 52.9% | 93.8% |
| V6 | Strict binary calibration | 71.9% | 49.1% | 94.7% |
| V7 | Auxiliary direction, binary output | 73.6% | 53.5% | 93.7% |

V7 recovered some sensitivity relative to V6 while preserving high rest
specificity. Its balanced-accuracy improvement over V5 is only 0.3 points, so
the result should be described as the preferred conservative configuration,
not a dramatic breakthrough.

### Domain-dependence graph

| Dataset | BA | Sensitivity | Specificity | Auxiliary direction weight |
|---|---:|---:|---:|---:|
| PhysioNet | 63.8% | 40.7% | 86.9% | 0.25 |
| OpenBMI | 76.3% | 57.3% | 95.3% | 0.50 |
| Liu stroke | 65.0% | 40.5% | 89.5% | 0.00 |

The model performs differently across recording domains. OpenBMI is strongest,
while PhysioNet and Liu stroke show lower sensitivity. The controller assigned
zero auxiliary-direction weight to Liu, meaning left/right internal evidence
did not improve that stroke domain under the validation rule.

### Published comparison

The grey comparison panel provides context, not a leaderboard:

- Ang et al., 2011: 74% mean accuracy for affected-hand imagery versus rest in
  46 stroke patients with offline patient-specific calibration.
- Nagarajan et al., 2024: 71.15% mean accuracy for stroke imagery versus rest
  after healthy pretraining and domain adaptation in 71 patients.
- Lopez-Larraz et al., 2025: 69.7% mean accuracy for paralyzed-hand attempt
  versus rest using EEG and EMG in 35 analyzed patients.

These studies use different cohorts, endpoints, metrics, and validation
contracts. The poster's 73.6% is balanced accuracy, while the comparison values
are mean accuracy. They cannot be treated as direct head-to-head evidence that
V7 is better.

### Safest results statement

> On 4,645 later-recording cue-locked windows, patient-adapted V7 achieved 73.6%
> balanced accuracy, 53.5% intent sensitivity, 93.7% rest specificity, and
> 0.896 AUROC. The result supports rapid calibration and conservative rest
> rejection, but almost half of intent windows remain missed and continuous
> false activations per minute have not been measured.

---

## Section 4: Next Steps

### 1 - Define the experimental protocol

Standardize when intent should occur using randomized visual or auditory cues,
rest periods, and continuous no-control periods. Repeat sessions in both healthy
participants and stroke patients.

**Why:** public datasets use different protocols, and cue-locked windows do not
measure spontaneous false commands.

### 2 - Build a prospective patient dataset

Use an early session only for calibration, then lock later sessions for testing.
Preserve raw EEG, cue timing, artifacts, functional scores, fatigue, session
metadata, and recovery outcomes.

**Why:** this creates a true prospective evaluation rather than repeatedly
reusing development datasets.

### 3 - Expand complementary sensing

Keep EEG as the intended deployment signal while using other sensors to verify
or characterize intent during data collection:

- EMG: confirms attempted muscle activation.
- fNIRS: adds slower cortical hemodynamic evidence.
- IMU and force sensors: confirm movement and device response.

The V7-H EEG-fNIRS software branch has now been implemented for the MIND public
dataset, but no real MIND V7-H result appears on this poster. It remains a
follow-up experiment.

### 4 - Validate closed-loop safety

Run the actual confirmation, abstention, and device-control state machine.
Measure:

- False activations per minute.
- Missed-intent rate.
- Command latency.
- Calibration time.
- Usability and fatigue.

### Translational goal

> Rapidly personalize a general EEG model, then confirm motor intent safely
> before assistive action.

---

## Acronyms and Technical Terms

### Core BCI terms

| Term | Meaning |
|---|---|
| BCI | Brain-computer interface: a system that converts brain-signal evidence into feedback or control |
| EEG | Electroencephalography: noninvasive measurement of scalp electrical potentials |
| Motor imagery | Mental rehearsal of movement without intended overt movement |
| Attempted movement | Intention and effort to move when physical output may be weak or absent |
| Motor intent | The project-level binary concept that purposeful motor preparation or imagery is present |
| Decoder | The algorithm that converts processed EEG features into probabilities or decisions |
| Cue-locked | An analysis window positioned relative to a known experimental cue |
| Asynchronous | Continuous operation without knowing when intent will begin |
| Closed loop | A system in which decoded intent changes feedback or device behavior, which then affects the user |

### Signal-processing terms

| Term | Meaning |
|---|---|
| QC | Quality control: checks for unusable recordings, channels, or windows |
| CAR | Common-average reference: subtracts the average EEG voltage across available channels from each channel |
| Hz | Hertz: cycles or samples per second, depending on context |
| MRCP | Movement-related cortical potential: a slow EEG voltage shift linked to movement preparation and execution |
| SMR | Sensorimotor rhythm: oscillatory activity over sensorimotor cortex, especially mu and beta rhythms |
| Mu rhythm | Approximately 7-14 Hz sensorimotor activity that often decreases during movement or imagery |
| Beta rhythm | Approximately 14-30 Hz activity associated with motor preparation, maintenance, and rebound |
| ERD | Event-related desynchronization: a task-related reduction in rhythmic band power |
| ERS | Event-related synchronization: a task-related increase or rebound in rhythmic band power |
| Covariance | A measure of how signals from different EEG channels vary together |
| Riemannian geometry | A mathematical framework that preserves the non-Euclidean structure of covariance matrices |
| Resampling | Converting recordings to a common sampling rate, here 100 Hz |

### Machine-learning and controller terms

| Term | Meaning |
|---|---|
| ML | Machine learning |
| CNN | Convolutional neural network |
| FBCNet | Filter-bank convolutional network: a neural model that learns spatial patterns from frequency-filtered EEG |
| Late fusion | Combining probabilities or decisions from separately trained branches |
| General model | Population model trained across development participants and datasets |
| Patient adaptation | Updating or calibrating the general model using a small early recording from the new individual |
| Probability calibration | Mapping model scores to probabilities that better reflect observed event frequency |
| Platt scaling | Logistic-regression-based probability calibration |
| Operating threshold | Probability cutoff above which the controller calls INTENT |
| Auxiliary label | Internal information used to improve learning but not exposed as the final device output |
| OOD | Out of distribution: sufficiently different from the model's training data to be considered unreliable |
| Abstention | Refusing to classify or command when evidence or quality is inadequate |
| Temporal confirmation | Requiring qualifying evidence across multiple consecutive windows |
| Refractory period | A lockout interval after a command that prevents immediate repeated activation |

### Evaluation terms

| Term | Meaning |
|---|---|
| TP | True positive: an intent window correctly detected as intent |
| TN | True negative: a rest window correctly rejected as rest |
| FP | False positive: a rest window incorrectly labeled intent |
| FN | False negative: an intent window missed as rest |
| Sensitivity | TP / (TP + FN): fraction of intent detected; also called recall |
| Specificity | TN / (TN + FP): fraction of rest correctly rejected |
| FPR | False-positive rate: 1 - specificity |
| Missed-intent rate | 1 - sensitivity |
| BA | Balanced accuracy: average of sensitivity and specificity |
| ROC | Receiver operating characteristic curve: sensitivity versus false-positive rate across thresholds |
| AUROC | Area under the ROC curve: threshold-independent ranking performance; 0.5 is chance and 1.0 is perfect |
| CI | Confidence interval: an uncertainty interval from repeated statistical resampling or a model assumption |
| Participant bootstrap | Resampling participants, rather than individual windows, to preserve within-person dependence |
| Development-observed | Performance viewed during model development rather than a one-time locked confirmatory test |
| Prospective | Data and analysis collected forward under a prespecified protocol |
| Locked test | A held-out cohort or session evaluated once after models, thresholds, and analysis are frozen |

### Sensors and rehabilitation terms

| Term | Meaning |
|---|---|
| EMG | Electromyography: electrical measurement of muscle activation |
| fNIRS | Functional near-infrared spectroscopy: optical estimation of cortical oxygenation changes |
| IMU | Inertial measurement unit: measures acceleration and rotation |
| FES | Functional electrical stimulation: electrically activates muscles to assist functional movement |
| Latency | Time between the intended event or cue and a confirmed system response |
| No-control period | Continuous period in which the user should issue no command; used to measure false activation |

### Dataset names

| Name | Meaning in this project |
|---|---|
| EEGMMIDB | PhysioNet EEG Motor Movement/Imagery Database |
| OpenBMI | Public EEG dataset and toolbox containing motor-imagery experiments |
| Liu acute stroke | Acute-stroke motor-imagery EEG dataset used as the stroke development domain |
| HS-Stroke | Chronic-stroke motor-imagery dataset used in a separate external transfer audit |
| WAY-EEG-GAL | Synchronized EEG, EMG, kinematic, and force dataset used in supporting transfer experiments |
| MIND | Simultaneous EEG-fNIRS unilateral-limb motor-imagery dataset used for the V7-H follow-up |

---

## Likely Questions and Short Answers

### Why not classify left and right directly?

Direction is a harder and less reliable problem. In rehabilitation, the visual
or auditory cue already specifies the requested movement. The safety-critical
EEG question is whether credible intent occurred. V7 may use left/right labels
internally but exposes only REST or INTENT.

### Is 73.6% accuracy good?

It is 73.6% **balanced accuracy**, not ordinary accuracy. It combines 53.5%
intent sensitivity and 93.7% rest specificity. The result is encouraging for a
conservative cue-locked intent gate, but it is not sufficient alone for
clinical device control.

### Why is sensitivity only 53.5%?

The chosen operating point deliberately prioritizes rest rejection. Weak
imagery, patient and session variability, domain mismatch, and conservative
thresholding all reduce sensitivity. Lowering the threshold recovers some
intent but increases false-positive windows.

### Does 93.7% specificity mean the device is safe?

No. It means 93.7% of analyzed cue-locked rest windows were rejected. Actual
device safety requires continuous no-control testing, temporal confirmation,
false activations per minute, latency, and closed-loop evaluation.

### Why use balanced accuracy?

Rest and intent may have different numbers of windows. Ordinary accuracy could
look high by favoring the larger or easier class. Balanced accuracy gives equal
weight to intent sensitivity and rest specificity.

### What does AUROC 0.896 add?

It shows that the probability ranking separates intent from rest well across
many possible thresholds. It does not choose the final threshold and does not
replace sensitivity or specificity at the deployed operating point.

### Why combine three model branches?

MRCP, sensorimotor rhythms, and spatial covariance capture complementary slow,
oscillatory, and spatial evidence. Validation-selected late fusion was more
stable than relying on one representation across all datasets.

### Why did direction help healthy datasets but not Liu stroke?

The controller selected auxiliary weights using validation evidence. Direction
received weight in PhysioNet and OpenBMI but zero in Liu, suggesting the
left/right internal structure did not transfer reliably to that acute-stroke
domain. This is evidence for dataset-specific calibration rather than one fixed
fusion rule.

### Is patient calibration leaking test information?

Calibration uses only the designated early recording. Evaluation uses later
recordings. The calibration rows are not included in the reported later-session
metrics.

### Is the model window two seconds or three seconds?

The poster graphic summarizes the cue timeline as 0-3 seconds. The harmonized
V7 implementation extracts a two-second analysis window beginning 0.5 seconds
after event onset. The cue timeline and exact model input window are related but
not identical.

### Are the published comparisons directly comparable?

No. They use different patient groups, task definitions, metrics, and
validation designs. They show that the result is in a plausible literature
range, not that V7 outperforms those systems.

### Why EEG instead of EMG?

EMG is often more direct when usable muscle activation remains. EEG is valuable
when muscle output is absent, unreliable, or occurs too late for the intended
neural-feedback question. A future system can use EMG as confirmation while
retaining EEG for cortical intent.

### Why add fNIRS?

fNIRS provides slower hemodynamic evidence that is physiologically different
from EEG. It may help confirm sustained cortical engagement or support
calibration, but its delay makes it less suitable for rapid commands. The V7-H
follow-up tests this as a quality-aware confirmation branch.

### What is the most important next experiment?

A prespecified prospective protocol with early calibration, later locked
sessions, continuous no-control periods, and direct measurement of false
activations per minute, missed intents, and latency.

---

## Claims to Make and Claims to Avoid

### Supported claims

- V7 is a reproducible, patient-adaptive binary EEG intent benchmark.
- The conservative result favors rest rejection over maximum intent recovery.
- Patient calibration improved matched later-recording balanced accuracy from
  69.1% to 73.6%.
- Performance remains strongly dataset dependent.
- V7 remains development-observed and cue-locked.

### Avoid these claims

- The system has 93.7% overall accuracy.
- The system produces only 6.3 false commands per 100 opportunities.
- The system is ready to control a prosthesis or FES device.
- The poster proves superiority over the cited published systems.
- V7 has been prospectively validated on an untouched clinical cohort.
- The temporal confirmation rule has already reduced false activations per
  minute.

---

## Final 20-Second Close

> The main contribution is not a claim that EEG motor intent is solved. It is a
> structured path from a multi-dataset population model to rapid patient
> calibration and conservative intent confirmation. The current result rejects
> rest well but still misses too much intent. The next phase is therefore a
> prospective, continuous, closed-loop study that measures safety and latency
> directly and tests whether complementary signals can improve confirmation.

## Five Numbers to Memorize

1. **187 participants** represented across three development datasets.
2. **36,903 cue-locked windows** used in the V7 dataset.
3. **73.6% balanced accuracy** after patient adaptation.
4. **53.5% sensitivity and 93.7% specificity** at the conservative point.
5. **0.896 AUROC** on 4,645 later-recording evaluation windows.
