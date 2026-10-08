# Comparable EEG Motor-Intent BCI Projects

## Methods, Results, and Implications for the Current Project

**Research update:** August 3, 2026

## Executive Summary

This review compares the current weak motor-intent EEG project with primary
research on motor imagery (MI), attempted movement, stroke rehabilitation,
cross-subject decoding, asynchronous control, and brain-contingent feedback.

The central finding is that published BCI percentages are not directly
interchangeable. A result can be made substantially easier by using:

- Direction-only classification without rest.
- Random trial splits from the same patient and session.
- Long, cue-locked windows with known task timing.
- Patient-specific calibration before every session.
- Prescreening participants for BCI performance.
- Cross-trial rather than cross-patient validation.
- A clinical outcome rather than a decoder-performance endpoint.

The current v4 result addresses a comparatively difficult question: intent
versus no intent across three datasets, including stroke EEG, with subject-held
roles, early-recording patient adaptation, dataset calibration, quality/OOD
rejection, and a separate conditional direction head. Its 70.6% pooled balanced
accuracy, 46.7% sensitivity, 94.4% specificity, and 71.4% conditional direction
balanced accuracy are credible development results, but not yet a continuous or
externally confirmed device result.

The most important lessons from comparable projects are:

1. **Cue-constrained asynchronous control is the strongest immediate clinical
   design.** A stroke study reduced false activations from 0.99 to 0.52 per
   minute while retaining an 84.88% true-positive rate by allowing an MRCP
   detector to operate only during a cue-defined attempt interval.
2. **False-positive rejection requires non-motor evidence.** A separate online
   stroke pilot rejected visual, auditory, and other contamination using a
   two-phase classifier and non-region-of-interest channels.
3. **MRCP and SMR should remain separate.** Transfer studies show value in MRCP
   amplitude and phase, while FBCNet and Riemannian methods are strong for
   spectro-spatial MI information.
4. **A modest decoder can still support meaningful therapy.** A BCI-FES trial
   used compact PSD features, a Gaussian classifier, accumulated evidence, and
   no-decision trials; its single-sample detection was approximately 64%, yet
   the contingent BCI group improved 6.6 FMA-UE points versus 2.1 with sham.
5. **Patient calibration remains normal in successful systems.** Strong online
   studies typically calibrate each person or session rather than relying on a
   universal zero-shot decoder.
6. **Cross-trial accuracy is not evidence of cross-patient generalization.** A
   new chronic-stroke dataset reports up to 82.65% cross-trial accuracy, but its
   14 patients and 278 longitudinal sessions should be reevaluated with entire
   patients and future sessions held out.
7. **The current project architecture is well aligned with the literature.**
   Full-montage QC, MRCP plus SMR branches, Riemannian covariance, FBCNet,
   patient adaptation, calibrated late fusion, abstention, and temporal
   confirmation all have strong methodological support.

The best next experiment is not another unrestricted model bakeoff. It is a
frozen, cue-constrained asynchronous study with patient calibration, explicit
contamination veto channels, MRCP phase ablation, continuous no-control data,
and brain-contingent feedback.

---

## 1. Scope and Review Method

### Research questions

This review addresses five questions:

1. What signals and preprocessing do comparable EEG motor-intent systems use?
2. Which feature and decoder families perform consistently?
3. How do successful projects handle calibration and subject variability?
4. How do online systems reduce false activation without missing all intent?
5. Which published results are genuinely comparable with the current project?

### Evidence included

The review prioritizes primary studies that include at least one of the
following:

- Healthy or stroke motor imagery.
- Attempted movement or movement-related cortical potential detection.
- Subject-independent or cross-session evaluation.
- Online or asynchronous operation.
- Brain-controlled FES, robot, orthosis, or exoskeleton feedback.
- Explicit false-positive, calibration, or no-decision handling.

Large reproducibility benchmarks and data descriptors are included when they
directly test algorithms or define relevant datasets.

### Important limitation

This is a focused technical review, not a formal systematic review or
meta-analysis. The studies use different tasks, cohorts, windows, labels,
splits, and metrics. Results are therefore compared by evaluation contract, not
ranked as if they came from one shared test set.

---

## 2. A Comparability Scale for BCI Results

### Tier 1: Within-trial or random cross-validation

Trials from the same participant and recording context can appear in both
training and test folds. This is useful for proving that a dataset contains
class information, but it says little about a new patient.

### Tier 2: Within-patient cross-session

The model is calibrated on one session and tested on another session from the
same person. This tests day-to-day drift but still relies on patient-specific
data.

### Tier 3: Cross-subject, same dataset

Entire participants are held out, but hardware, protocol, cue timing, and class
definitions remain consistent. This is a stronger generalized-decoder test.

### Tier 4: Cross-dataset or cross-clinical-domain

The model must transfer across people and acquisition protocols. This is harder
because disease status is confounded with hardware, timing, montage, and task
instructions unless the study controls them prospectively.

### Tier 5: Patient-adapted external evaluation

A population model adapts using a predefined early calibration block and is
evaluated on later data from unseen patients. This is closest to the intended
clinical workflow of the current project.

### Tier 6: Continuous online control

The system must operate without knowing the exact intent onset, reject prolonged
idle and artifacts, accumulate evidence, and report false activations per unit
time and latency.

### Where the current project stands

The current v4 pipeline is between Tiers 4 and 5, but its evaluation cohort is
`development_observed`, not a new locked external cohort. Temporal confirmation
is implemented, but Tier 6 has not been evaluated because continuous no-control
recordings are unavailable.

---

## 3. Comparable Decoder and Transfer Studies

### 3.1 Large stroke MI-versus-rest clinical decoding study

**Study:** Ang et al., 2011, 54 stroke patients and 16 healthy participants.

**Task:** Affected-hand motor imagery or finger tapping versus background rest.

**Design:** Forty-six stroke patients performed MI. Eleven returned for a
second calibration and a third independent online session.

**Results:** Mean offline MI detection accuracy was 74% in stroke and 78% in
healthy participants. Finger tapping reached 87%. In the 11 returning patients,
the first and second offline sessions reached 72% and 76%, and the third online
session reached 82%. Six stroke patients performed at chance.

**Interpretation:** This is one of the closest historical comparisons to an
intent-versus-rest gate. Its strong online result includes patient-specific
calibration and a small returning subgroup, so it should not be treated as a
zero-shot population benchmark. The chance-level subgroup supports reporting
patient distributions and abstention rather than only a mean.

**Source:** [Ang et al., large clinical stroke BCI study](https://pubmed.ncbi.nlm.nih.gov/22208123/)

### 3.2 Subject-independent MRCP template detection

**Study:** Jochumsen et al., 2013, 20 healthy participants and five stroke
patients.

**Task:** Executed or imagined ballistic ankle dorsiflexion in healthy
participants and attempted movement in stroke.

**Method:** Optimized spatial filtering followed by a population-average MRCP
template. It compared a subject-independent average template with individual
templates.

**Results:** The subject-independent template detected executed movement with a
69% true-positive rate in healthy participants and 58% in stroke patients. MI
detection in healthy participants reached 65%. Individual templates did not
produce a large consistent advantage.

**Interpretation:** Population MRCP information can transfer, but performance
is modest and the primary reported endpoint is true-positive rate rather than a
complete asynchronous false-activation analysis. This supports keeping a
population MRCP proposal branch while adding calibration and a separate safety
gate.

**Source:** [Subject-independent MRCP detection](https://pubmed.ncbi.nlm.nih.gov/23283643/)

### 3.3 MRCP amplitude and phase transfer

**Study:** Sburlea et al., 2016, 10 healthy participants and nine chronic-stroke
participants across three sessions.

**Task:** Self-initiated gait intention.

**Method:** MRCP phase, amplitude, and power features under session-specific,
intersession, and intersubject transfer.

**Results:** Phase features were strongest with session-specific calibration,
detecting 66.5% of healthy and 63.3% of stroke trials. Combined phase and
amplitude retained the best transfer performance, reaching 62.5% in healthy and
59% in stroke participants.

**Interpretation:** Explicit MRCP phase may contribute information not fully
captured by mean amplitude, slope, and extrema. This is a justified ablation for
the current MRCP branch, especially for transfer and timing.

**Source:** [MRCP phase transfer study](https://arxiv.org/abs/1605.04533)

### 3.4 Passive movement does not substitute for MI training

**Study:** Edelman et al., 2018, intersubject left-versus-right motor-imagery
decoding with EEG and MEG.

**Method:** CSP and multitask logistic-regression approaches trained on either
other participants' MI or passive movement.

**Results:** Training on other participants' MI reached 67.7% mean EEG accuracy.
Training on passive movement did not exceed the study's chance threshold,
reaching 58.7%.

**Interpretation:** This directly supports the current project's negative WAY
and movement-transfer findings. Physical or passive movement may identify
physiology, but it should not replace imagery examples in the final decoder.

**Source:** [Across-subject MI decoding from EEG and MEG](https://pubmed.ncbi.nlm.nih.gov/29973645/)

### 3.5 FBCNet on healthy and chronic-stroke MI

**Study:** Mane et al., FBCNet.

**Method:** Multiple fixed frequency views, learned spatial filtering, a
variance aggregation layer, and a compact neural classifier.

**Results:** FBCNet reached 76.20% four-class holdout accuracy on BCI Competition
IV-2a. Under the reported cross-validation setting, it reached 74.70% on OpenBMI
and 79.16% and 81.11% on two chronic-stroke datasets, outperforming the tested
baselines by up to approximately eight points.

**Interpretation:** These stroke percentages are mainly within-dataset,
participant-specific cross-validation results rather than unseen-patient
intent-versus-rest transfer. They strongly support the architecture's
spectro-spatial bias, but not a direct expectation that the current cross-domain
gate should reach 80%.

**Source:** [FBCNet primary paper](https://arxiv.org/abs/2104.01233)

### 3.6 Acute-stroke Riemannian decoding

**Study:** Liu et al., 2024, 50 acute-stroke patients.

**Task:** Left-hand versus right-hand MI, with 20 trials per patient.

**Preprocessing:** High-pass filtering, rereferencing, 8-30 Hz spectral
filtering, and a 0-4 second post-stimulus window.

**Methods:** CSP-LDA, FBCSP-SVM, Riemannian tangent-space and geodesic methods,
and an optimized time-window/filter-bank geodesic minimum-distance method.

**Results:** Mean accuracies were 55.57% for CSP-LDA, 57.57% for FBCSP-SVM,
61.20% for the initial Riemannian fusion, and 72.21% for the optimized method.
Training and testing were split within each patient using 60/40 trial division
and repeated cross-validation.

**Interpretation:** The 72.21% result confirms that individualized direction
information exists in acute-stroke EEG. It is not a subject-held-out
intent-versus-rest result. The current project's 71.4% conditional direction BA
is broadly plausible, but it is measured under a different and harder transfer
contract.

**Source:** [Liu acute-stroke dataset and validation](https://www.nature.com/articles/s41597-023-02787-8)

### 3.7 Stroke unilateral multi-task transfer

**Study:** Ma et al., 2024, 22 stroke patients.

**Task:** Four motor-imagery tasks involving the same unilateral upper limb.

**Method:** A spatial self-attention CNN with partial-prior transfer learning.
The model used target-domain data to preserve patient specificity while
retaining a source-domain representation.

**Results:** Four-class accuracy reached 55.4% under the paper's transfer mode.
The study reported approximately 0.5 seconds of inference after a four-second MI
task and 16 seconds for target-model training on upgraded hardware.

**Interpretation:** Same-limb multi-task decoding is much harder than coarse
left/right lateralization. The result supports the current project's decision
to solve intent and direction before attempting detailed hand-action classes.

**Source:** [SACNN partial-prior transfer study](https://www.nature.com/articles/s41598-024-79202-8)

### 3.8 New chronic-stroke longitudinal dataset

**Study:** Lu et al., 2026, 14 chronic-stroke patients, 278 sessions, and 57,902
left/right MI trials.

**Result:** State-of-the-art deep models reached up to 82.65% accuracy under a
cross-trial evaluation setting. MI features were also associated with Fugl-Meyer
scores.

**Interpretation:** The dataset is exceptionally valuable for longitudinal
adaptation, session drift, and calibration-burden research. The result should
not be compared directly with subject-held-out performance because trials from
the same small patient cohort support training and validation. For this project,
the best use is to reserve entire patients and later sessions before any model
selection.

**Source:** [HS chronic-stroke longitudinal MI dataset](https://www.nature.com/articles/s41597-026-07742-x)

### 3.9 Recent subject-independent PhysioNet decoding

**Study:** A 2026 multi-branch CNN-LSTM evaluation on PhysioNet EEGMMIDB and an
independent wearable-EEG dataset.

**Method:** Four temporal convolution branches with different kernel sizes,
LSTM sequence modeling, group normalization, subwindow supervision, and late
fusion.

**Results:** The study reports 82.63% subject-independent binary left/right MI
accuracy with four-second trials, 74.9% leave-one-subject-out three-class
accuracy, 64.60% zero-shot three-class transfer to the independent setup, and
81.38% after fine-tuning on 20% of the target session.

**Interpretation:** This is a relevant upper benchmark for direction-oriented
PhysioNet decoding. Its four-second, full-trial sequence model and same-dataset
training are easier than the current 2-second, seven-channel, three-dataset,
high-specificity intent gate. The zero-shot cross-setup drop and large recovery
after 20% fine-tuning strongly support patient adaptation.

**Source:** [Multi-branch CNN-LSTM cross-subject study](https://pubmed.ncbi.nlm.nih.gov/42280831/)

---

## 4. Comparable Online and False-Activation Studies

### 4.1 Two-phase asynchronous MI rejection

**Study:** Song et al., 2022, eight healthy participants and nine patients with
stroke.

**Goal:** Reduce false-positive feedback in an online asynchronous MI-BCI while
using only four EEG channels.

**Method:** The authors identified visual-evoked, auditory-evoked, and other EEG
contaminations that caused false positives. A two-phase classifier first
detected MI-related ERD and then rejected contamination using channels outside
the motor region.

**Stroke result:** 71.76% selectivity and a 13.70% false-positive rate. The
non-motor rejection stage rejected 76.04% of candidate false positives. The
system also included recalibration for day-to-day variation.

**Interpretation:** A motor classifier cannot reliably reject every artifact
using only motor evidence. The current full-montage preprocessing can support a
parallel contamination-veto branch using frontal, temporal, and posterior
channels without feeding those channels into the motor classifier.

**Source:** [Song et al., asynchronous false-positive rejection](https://pubmed.ncbi.nlm.nih.gov/36172602/)

### 4.2 Cue-constrained asynchronous MRCP detection

**Study:** Niazi et al., 2022, nine analyzed chronic-stroke patients in a
repeated-measures intervention.

**Task:** Attempted paretic ankle dorsiflexion with afferent electrical
stimulation.

**Method:** Fifty attempted movements were collected for training. The standard
asynchronous MRCP detector operated continuously, while the cue-constrained
version operated only inside a period when the patient had been instructed to
attempt movement.

**Results:** Cue-constrained operation reached an 84.88% true-positive rate and
0.52 false positives per minute. Fully asynchronous operation reached 82.68%
and 0.99 false positives per minute. The cue reduced false positives
significantly without a significant sensitivity loss.

**Interpretation:** This is the clearest published workaround for the project's
false-activation versus missed-intent tradeoff. A cue or therapist command can
serve as an arming signal, while EEG still determines whether and when the
patient produced credible motor intent. This does not solve spontaneous device
control, but it is highly appropriate for structured rehabilitation.

**Source:** [Niazi et al., cue-constrained asynchronous BCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/)

### 4.3 BCI-FES with evidence accumulation and no decision

**Study:** Biasiucci et al., 2018, 27 chronic-stroke patients assigned to
BCI-FES or sham FES.

**Acquisition and calibration:** A 16-electrode sensorimotor montage, 512 Hz EEG,
and one individualized calibration session for hand-extension attempt versus
rest.

**Signal processing:** Laplacian spatial derivations, Welch PSD from 4-40 Hz in
2 Hz bins, 500 ms Hanning windows with 75% overlap, and probability updates every
62.5 ms.

**Model:** Up to 10 manually selected discriminant and physiologically plausible
spatio-spectral features, followed by a Gaussian classifier. Probabilities were
integrated over time. The trial ended when a class reached the confidence
threshold or returned no decision after a 7-second timeout.

**Decoder behavior:** Single-sample detection was approximately 63.8% in the BCI
group. The study's purpose was temporal contingency between cortical attempt
and feedback rather than maximizing standalone classifier accuracy.

**Clinical result:** The BCI-FES group improved 6.6 +/- 5.6 FMA-UE points versus
2.1 +/- 3.0 with sham. Eight BCI participants versus two sham participants
improved by at least five points, and the BCI improvement was retained at
follow-up.

**Interpretation:** This study strongly supports calibrated probabilities,
evidence accumulation, abstention/no decision, and contingent feedback. It also
shows that therapeutic value may not require a high single-window accuracy if
feedback timing is sufficiently brain contingent.

**Source:** [Biasiucci et al., brain-actuated FES](https://www.nature.com/articles/s41467-018-04673-z)

---

## 5. Comparable Rehabilitation Trials

Decoder accuracy and clinical benefit are different endpoints. A clinical trial
may test whether contingent feedback improves impairment even if it does not
publish a complete sensitivity/specificity analysis.

| Study | Cohort and intervention | Main result | Relevance to this project |
|---|---|---|---|
| Ramos-Murguialday et al., 2013 | Severe chronic stroke; BMI plus physiotherapy versus sham feedback | BMI training produced superior motor improvement and changes associated with brain/EMG measures | Supports close intent-feedback contingency and dose-matched sham design |
| Ang et al., 2014 | 26 chronic-stroke participants; BCI-Manus versus intensive robot therapy | Both groups improved; no significant intergroup FMA difference, although BCI used fewer robot repetitions | Shows feasibility does not guarantee superiority over active therapy |
| Pichiorri et al., 2015 | 28 subacute-stroke participants; BCI-supported MI versus MI without feedback | BCI group had a higher probability of clinically relevant FMA improvement and stronger ipsilesional alpha/beta changes | Supports contingent feedback and physiology endpoints |
| Multicenter hand-exoskeleton trial, 2017 | 74 severe-stroke participants; BCI-contingent versus noncontingent exoskeleton | Only BCI group showed significant grasp and pinch gains; responder proportions favored BCI | Supports multicenter feasibility and active-control design |
| Biasiucci et al., 2018 | 27 chronic-stroke participants; BCI-FES versus sham FES | 6.6 versus 2.1 FMA-UE point improvement, retained at follow-up | Strong model for a future brain-contingent intervention |
| Sebastian-Romagosa et al., 2024 | 35 severe subacute-stroke participants; MI-BCI-FES versus conventional treatment | No significant group difference; corticospinal-tract integrity predicted meaningful improvement | Clinical phenotype may matter more than algorithm choice for some patients |
| He et al., 2025 | 48 ischemic-stroke participants; 8-electrode EEG, VR, and robot feedback versus matched non-real-time control | Upper-extremity FMA improved by 4 versus 2 points, p=0.046; EEG, EMG, and fNIRS changes supported multimodal effects | Supports low-channel deployment and multimodal outcome assessment |
| Closed-loop glove pilot, 2026 | 40 subacute-stroke participants; contingent MI-BCI glove versus dose-matched noncontingent glove | Adjusted week-4 FMA-UE difference 13.40 points; no serious adverse events | Promising but still a single-center pilot requiring replication |

Primary sources:

- [Ramos-Murguialday et al., controlled chronic-stroke BMI study](https://pubmed.ncbi.nlm.nih.gov/23494615/)
- [Ang et al., BCI-Manus randomized trial](https://pubmed.ncbi.nlm.nih.gov/24756025/)
- [Pichiorri et al., BCI-supported MI trial](https://pubmed.ncbi.nlm.nih.gov/25712802/)
- [Multicenter BCI hand-exoskeleton trial](https://pubmed.ncbi.nlm.nih.gov/28775677/)
- [Biasiucci et al., BCI-FES trial](https://www.nature.com/articles/s41467-018-04673-z)
- [Severe-stroke MI-BCI-FES trial](https://pubmed.ncbi.nlm.nih.gov/38245782/)
- [2025 multimodal BCI randomized trial](https://pubmed.ncbi.nlm.nih.gov/40859358/)
- [2026 closed-loop soft-glove pilot](https://pubmed.ncbi.nlm.nih.gov/42422218/)

### Clinical interpretation

The trial literature is encouraging but heterogeneous. Some controlled studies
show clinically meaningful gains, while others find no advantage over intensive
robotic or conventional therapy. Successful studies generally emphasize the
contingency between detected cortical effort and immediate sensory feedback,
not left/right command accuracy alone.

For a rehabilitation PhD project, the decoder should be treated as one component
of an intervention. FMA-UE, ARAT, calibration burden, engagement, clinical
phenotype, and neurophysiological change should be co-primary or secondary
outcomes alongside false activations and sensitivity.

---

## 6. Large Benchmark Evidence

The 2024 MOABB reproducibility analysis reimplemented 30 pipelines across 36
public datasets, including 14 MI datasets.

The main relevant findings were:

- Riemannian covariance pipelines were consistently strong.
- Tangent-space approaches generally outperformed direct manifold classifiers.
- Riemannian pipelines worked especially well with limited electrode counts.
- Deep learning usually needed more than approximately 150 trials per class to
  become consistently competitive.
- Right-hand-versus-feet MI was easier than left-versus-right MI across methods.
- Performance varied substantially across participants and datasets.

These findings support retaining the current Riemannian branch and using a
compact, physiology-informed neural model rather than replacing everything with
a larger network. They also explain why adding foot trials could inflate a
benchmark without improving the intended left/right hand rehabilitation task.

**Source:** [Large-scale MOABB reproducibility benchmark](https://arxiv.org/abs/2404.15319)

---

## 7. Like-for-Like Comparison with the Current Project

| Project or study | Main task | Test separation | Calibration | Primary result | Direct comparability |
|---|---|---|---|---|---|
| Current v4 | Intent/rest across PhysioNet, OpenBMI, and Liu; conditional left/right | Subject-held roles plus later-recording evaluation; cohort is development observed | Early recording plus dataset calibration | 70.6% BA, 46.7% sensitivity, 94.4% specificity; direction BA 71.4% | Reference |
| Ang 2011 | Affected-hand MI/rest | Patient-specific sessions; 11-person online subset | Yes | 74% stroke offline; 82% returning online subset | Medium |
| Jochumsen 2013 | Movement/MI/attempt onset | Population MRCP template | No individual template required | 58% stroke TPR | Medium for MRCP transfer, low for specificity |
| Sburlea 2016 | Self-initiated gait intent | Intersession and intersubject | Compared | 59% stroke transfer detection | Medium for MRCP transfer |
| FBCNet | Primarily MI class discrimination | Mostly subject-specific CV or session holdout | Training data per person | 79.16-81.11% on two stroke datasets | Low for cross-patient intent gate; high for architecture |
| Liu validation | Left/right MI | Within-patient trial split | Implicit within-patient training | 72.21% accuracy | Medium for direction, low for intent/rest transfer |
| SACNN-PPTL | Four same-limb MI tasks | Transfer with target-domain prior | Yes | 55.4% four-class accuracy | Medium for patient adaptation |
| HS stroke 2026 | Left/right MI | Cross-trial over 14 longitudinal patients | Same cohort | Up to 82.65% accuracy | Low until patient/session held out |
| CNN-LSTM 2026 | PhysioNet MI classes | Subject-independent and cross-setup tests | Optional 20% target fine-tuning | 74.9% LOSO three-class; 64.6% zero-shot cross-setup | Medium-high for cross-subject direction |
| Song 2022 | Asynchronous MI/rest with contamination rejection | Online healthy and stroke | Day-to-day recalibration | 13.70% FP rate, 71.76% selectivity | High for false-positive design |
| Niazi 2022 | Cue-constrained attempted-movement onset | Online within patient | 50 attempts | 84.88% TPR, 0.52 FP/min | High for future clinical controller |
| Biasiucci 2018 | Attempt/rest controlling FES | Individual calibration and repeated therapy | Yes | Approximately 64% sample detection; 6.6-point FMA gain | High for full intervention design |

### What the table means

The current 70.6% balanced accuracy is not obviously below the field once the
test contract is matched. Many 75-85% results use direction-only classes,
within-patient trials, longer windows, or patient-specific training. The most
relevant gap is not another five points of cue-locked BA. It is the absence of a
locked external cohort and continuous false-activation measurement.

---

## 8. Methods Shared by Successful Projects

### 8.1 Physiological frequency separation

Most comparable systems explicitly model slow MRCP or mu/beta SMR rather than
using one broad undifferentiated signal. The current 0.1-4 Hz MRCP and 7-35 Hz
SMR branches follow this pattern.

### 8.2 Spatial filtering

Common approaches include Laplacian derivations, CSP/FBCSP, covariance tangent
space, and learned spatial convolutions. The current use of CAR after repair,
optional CSD, FBCSP, Riemannian covariance, and FBCNet is well supported.

### 8.3 Compact feature sets or inductive bias

Clinical systems often use a small number of physiologically reviewed features.
Successful neural systems encode filter-bank and variance structure. Neither
approach supports unrestricted feature concatenation.

### 8.4 Individual calibration

The strongest online systems generally include a calibration run. Population
training reduces burden, but does not remove the need to estimate individual
frequency, threshold, spatial pattern, or probability mapping.

### 8.5 Temporal evidence accumulation

Single windows are rarely trusted as final commands. Systems integrate
probability, require confirmation, use a timeout, or restrict operation to an
armed state.

### 8.6 No-decision behavior

Clinical BCI systems can time out or reject uncertain trials. This is directly
aligned with the current OOD/quality abstention design.

### 8.7 Contingent feedback

The therapeutic mechanism depends on pairing cortical attempt with immediate
afferent feedback. Accuracy is a means to preserve that contingency, not the
only outcome.

---

## 9. Priority Recommendations for This Project

### Priority 1: Freeze v4 and create a truly locked external test

Use the new HS chronic-stroke dataset or a prospectively recorded cohort without
reopening model selection. Before inspecting outcomes:

1. Freeze preprocessing, seven motor channels, window timing, model weights,
   patient-calibration budget, thresholds, and failure rules.
2. Reserve complete patients, not random trials.
3. Reserve later sessions for longitudinal evaluation.
4. Record the zero-shot result before any target adaptation.
5. Apply the predefined early-session adaptation once.
6. Report all patients, including nonresponders and QC exclusions.

### Priority 2: Implement cue-constrained asynchronous operation

For structured rehabilitation, use a state machine:

```text
IDLE
  -> therapist/visual/audio cue
ARMED WINDOW
  -> quality valid + calibrated EEG evidence
CONFIRMED INTENT
  -> left/right proposal or single affected-limb command
FEEDBACK
  -> FES, robot, glove, VR, or visual feedback
REFRACTORY
  -> return to IDLE
```

The cue should arm the detector, not automatically trigger movement. EEG still
determines whether credible intent occurred. This preserves brain contingency
while reducing the amount of time in which false activation is possible.

### Priority 3: Add a non-motor contamination-veto branch

Retain the seven-channel motor decoder, but calculate veto evidence from the
cleaned full montage:

- Frontal channels for blink and eye-movement activity.
- Temporal-edge channels for jaw and facial EMG.
- Occipital channels for visual-evoked responses.
- Auxiliary EOG/EMG when available.
- Broadband discontinuity and electrode-shift features.

This branch should only abstain or delay. It should not predict left or right.
Evaluate it with explicit blink, saccade, jaw clench, speech, head movement,
visual distractor, and auditory distractor blocks.

### Priority 4: Run an explicit MRCP phase ablation

Add Hilbert-phase and phase-consistency features to the existing 0.1-4 Hz MRCP
branch. Compare:

1. Existing MRCP temporal statistics.
2. MRCP phase only.
3. Amplitude plus phase.
4. Current MRCP + FBCNet neural representation.
5. Neural representation plus explicit phase proposal.

Advance phase only if participant-level sensitivity improves at the same
specificity and the effect survives across PhysioNet, OpenBMI, and Liu.

### Priority 5: Make temporal confirmation a measured controller

Collect at least 10-20 minutes of continuous no-control EEG per session. Include
natural and challenged idle. Measure:

- False confirmed commands per minute.
- Sensitivity per instructed attempt.
- Missed attempts.
- Median and 95th-percentile latency.
- Abstained fraction.
- Commands during artifact challenges.
- Performance before and after short recalibration.

Compare single-window, two-confirmation, evidence-accumulator, and cue-constrained
policies under one fixed probability stream.

### Priority 6: Use a two-stage calibration protocol

Stage A should estimate recording quality, individual mu frequency, baseline
probability, and gross spatial pattern from approximately 20-50 instructed
attempts.

Stage B should adapt online using only high-confidence, cue-consistent events.
It should update calibration and thresholds conservatively, not continuously
retrain the entire neural model.

### Priority 7: Connect the decoder to a therapeutic endpoint

The first closed-loop study can use one affected-limb command rather than
requiring left/right control from every patient. For hemiparetic rehabilitation,
the clinical question may be whether affected-limb cortical effort is present,
not which hand was selected.

Use contingent visual feedback first, then FES or a glove after safety validation.
Include a yoked or randomized noncontingent-feedback control so the effect of
brain contingency can be separated from exercise dose and device exposure.

### Priority 8: Stratify clinical participants

Collect and analyze:

- FMA-UE and ARAT.
- Affected hemisphere and lesion location.
- Time since stroke.
- Corticospinal-tract integrity when clinically available.
- Residual EMG and active range of motion.
- Motor-imagery ability and strategy.
- Fatigue, attention, medication, and session time.

The 2024 severe-stroke trial found corticospinal-tract integrity more predictive
of meaningful improvement than treatment assignment. A decoder should not be
expected to erase clinical heterogeneity.

---

## 10. Approaches Not Recommended as the Next Main Experiment

### Do not optimize on the current development-observed cohort again

Another improvement measured on the same people would increase selection bias
and further weaken the confirmatory meaning of the percentage.

### Do not replace the complete pipeline with a large transformer

Large architecture capacity does not solve negative-class mismatch, cue
confounding, or missing continuous evaluation. MOABB supports retaining
Riemannian baselines, especially with limited trials.

### Do not add foot trials merely to increase accuracy

Right-hand-versus-feet classification is systematically easier than
left-versus-right. It is a different rehabilitation question and can inflate the
headline result.

### Do not use movement or passive movement as the primary MI teacher

Movement data can support representation learning and physiology audits, but
cross-subject evidence shows that passive movement does not reliably train an MI
decoder. The project's WAY and EMG-supervision findings are consistent with
this.

### Do not use cross-trial stroke accuracy as the external target

The 72-83% stroke results are valuable within-patient benchmarks. They should
not be used as a direct standard for unseen-patient intent-versus-rest transfer.

### Do not hide nonresponders

Some stroke participants remain near chance even in large clinical studies.
Report calibration failures, excluded recordings, and patient distributions.

---

## 11. Proposed Next Experiment

### Research objective

Determine whether the frozen v4 model plus cue-constrained temporal control can
detect affected-limb motor attempt with low false activation after a short
patient calibration.

### Participants

Begin with a prospective healthy pilot, followed by a small stroke feasibility
cohort. Use the same hardware and protocol in both groups so disease is not
confounded with acquisition.

### Session structure

1. Five minutes of natural eyes-open no-control EEG.
2. Artifact challenge block.
3. Twenty to fifty cue-constrained calibration attempts.
4. A short recalibration pause.
5. Cue-constrained online feedback trials.
6. Ten minutes of mixed cued and uncued continuous operation.
7. Later-session or later-day repeat without redesigning the model.

### Decoder comparison

Use one frozen probability source and compare policies:

1. V4 single-window threshold.
2. V4 two-window confirmation.
3. V4 evidence accumulation with no-decision timeout.
4. V4 cue-constrained evidence accumulation.
5. V4 cue-constrained accumulation plus contamination veto.

### Primary endpoints

- Sensitivity at no more than one false confirmed command per minute.
- False commands during artifact challenges.
- Median confirmation latency no more than 2.5 seconds.
- Missed-intent rate.
- Available versus abstained time.

### Secondary endpoints

- Conditional direction accuracy when direction is clinically relevant.
- Calibration burden.
- Day-to-day stability.
- MRCP and ERD physiological validity.
- User effort, fatigue, and comfort.
- FMA-UE and ARAT in the later clinical phase.

### Advancement rule

Advance to device-triggered FES or robotic movement only if the policy meets the
false-activation target on continuous no-control recordings and does not obtain
its improvement by rejecting most intent attempts.

---

## 12. Final Assessment

The current project is not fundamentally behind comparable BCI research. Its
architecture now resembles the more defensible systems in the field: separate
MRCP and SMR representations, Riemannian and neural evidence, calibration,
patient adaptation, quality rejection, abstention, and temporal confirmation.

Its headline accuracy is also not obviously low after accounting for protocol
difficulty. Many higher published numbers solve direction-only, within-patient,
or cross-trial tasks. The current project solves a more safety-oriented
intent-versus-rest problem across heterogeneous cohorts.

The real shortcoming is experimental, not primarily algorithmic:

- No newly locked external cohort.
- No continuous no-control recording.
- No measured false activations per minute or latency.
- No prospective causal preprocessing validation.
- No brain-contingent feedback intervention.

The literature points to a practical way forward: freeze the decoder, constrain
the first clinical controller with a cue-defined armed state, use full-montage
contamination vetoes, accumulate calibrated evidence, permit no decision, adapt
from a short patient block, and evaluate the entire system in continuous time.

---

## 13. Primary Source Index

### Decoder and transfer studies

1. [Ang et al., stroke MI BCI ability](https://pubmed.ncbi.nlm.nih.gov/22208123/)
2. [Jochumsen et al., subject-independent MRCP detection](https://pubmed.ncbi.nlm.nih.gov/23283643/)
3. [Sburlea et al., MRCP phase and transfer](https://arxiv.org/abs/1605.04533)
4. [Edelman et al., across-subject MI and passive-movement training](https://pubmed.ncbi.nlm.nih.gov/29973645/)
5. [Mane et al., FBCNet](https://arxiv.org/abs/2104.01233)
6. [Liu et al., acute-stroke MI dataset](https://www.nature.com/articles/s41597-023-02787-8)
7. [Ma et al., partial-prior transfer in stroke](https://www.nature.com/articles/s41598-024-79202-8)
8. [Lu et al., chronic-stroke longitudinal dataset](https://www.nature.com/articles/s41597-026-07742-x)
9. [2026 multi-branch CNN-LSTM subject-independent study](https://pubmed.ncbi.nlm.nih.gov/42280831/)
10. [Lawhern et al., EEGNet](https://pubmed.ncbi.nlm.nih.gov/29932424/)
11. [Large MOABB reproducibility benchmark](https://arxiv.org/abs/2404.15319)

### Online and asynchronous studies

12. [Song et al., online false-positive rejection](https://pubmed.ncbi.nlm.nih.gov/36172602/)
13. [Niazi et al., cue-constrained asynchronous BCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/)
14. [Biasiucci et al., brain-actuated FES](https://www.nature.com/articles/s41467-018-04673-z)

### Rehabilitation trials

15. [Ramos-Murguialday et al., chronic-stroke BMI trial](https://pubmed.ncbi.nlm.nih.gov/23494615/)
16. [Ang et al., BCI-Manus randomized trial](https://pubmed.ncbi.nlm.nih.gov/24756025/)
17. [Pichiorri et al., BCI-supported MI practice](https://pubmed.ncbi.nlm.nih.gov/25712802/)
18. [Multicenter BCI-exoskeleton trial](https://pubmed.ncbi.nlm.nih.gov/28775677/)
19. [Severe-stroke MI-BCI-FES pilot](https://pubmed.ncbi.nlm.nih.gov/38245782/)
20. [2025 multimodal randomized BCI trial](https://pubmed.ncbi.nlm.nih.gov/40859358/)
21. [2026 closed-loop MI-BCI glove pilot](https://pubmed.ncbi.nlm.nih.gov/42422218/)
