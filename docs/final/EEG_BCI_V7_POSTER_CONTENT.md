# Patient-Adaptive EEG Detection of Weak Motor Intent

## Poster Title

**Patient-Adaptive Multi-Dataset EEG Detection of Weak Motor Intent for
Rehabilitation**

Shorter alternative:

**Detecting Weak Motor Intent from EEG with Rapid Patient Calibration**

### Author Line

**Liam Kennings, [Coauthors or Faculty Mentor]**  
Department of Biomedical Engineering, [Institution]

### Take-Home Banner

> A multi-dataset EEG model combined complementary slow-potential and
> sensorimotor-rhythm evidence with a short patient calibration. The V7 binary
> intent gate reached 73.6% balanced accuracy and 93.7% cue-locked rest
> specificity, but continuous false activations and locked external
> generalization remain untested.

### Recommended Abstract Replacement

For the physical poster, replace the abstract text block with the
[Why This Project visual](../../artifacts/poster_v7/why_project/00_why_this_project.png).
Editable [SVG](../../artifacts/poster_v7/why_project/00_why_this_project.svg)
and print-ready
[PDF](../../artifacts/poster_v7/why_project/00_why_this_project.pdf) versions
are available. The written abstracts below should be retained for conference
submission forms, reports, or a QR-linked project page.

---

## Updated Poster Abstract

Motor-imagery electroencephalography (EEG) offers a noninvasive pathway for
controlling rehabilitation and assistive devices when reliable muscle activity
is unavailable. Real-world translation remains limited, however, by weak and
variable motor-intent patterns, differences among users and recording systems,
calibration burden, and false activation during rest. This project therefore
focused on the clinically relevant first decision of whether credible hand
motor intent is present, rather than exposing an unreliable left-versus-right
device command.

A quality-aware, patient-adaptive binary decoder was developed using 36,903
cue-locked EEG windows from 187 represented participants across PhysioNet
EEGMMIDB, OpenBMI, and a Liu acute-stroke motor-imagery dataset. Recordings
underwent native-montage quality control, bad-channel repair, common-average
referencing, resampling to 100 Hz, and reduction to seven shared sensorimotor
channels. Complementary branches represented movement-related cortical
potentials (MRCP, 0.1-4 Hz), sensorimotor rhythms (SMR, 7-35 Hz), and
Riemannian spatial covariance. A five-seed MRCP-FBCNet neural ensemble was
combined with classical physiological evidence using controller-selected late
fusion, dataset-specific probability calibration, and conservative operating
thresholds.

For each evaluation participant, the first recording supplied 30 labeled
calibration windows: 10 rest, 10 left imagery, and 10 right imagery. Later
recordings were reserved for evaluation. Direction labels provided optional
internal calibration evidence only; left and right probabilities were summed
to produce a strictly binary REST-versus-INTENT output. The patient-adapted V7
conservative operating point achieved 73.6% balanced accuracy
(participant-bootstrap 95% CI: 67.5-78.7%), 53.5% intent sensitivity, 93.7%
rest specificity, and an AUROC of 0.896. Balanced accuracy remained
domain-dependent at 63.8% for PhysioNet, 76.3% for OpenBMI, and 65.0% for the
stroke cohort. Direction-assisted calibration was retained for the healthy
datasets but assigned zero weight for stroke data.

These development-observed results support a rapidly calibrated EEG intent
gate for cue-guided rehabilitation, where the external cue specifies the
requested movement and EEG confirms whether intent follows. However,
specificity was measured per cue-locked rest window, not as continuous false
activations per minute, and no locked external cohort was available. Continuous
no-control testing, causal online processing, and prospective patient
evaluation are therefore required before clinical or autonomous device-control
claims.

### Short Abstract Option

Weak and variable motor-imagery EEG limits reliable control of rehabilitation
and assistive devices, particularly when false activation during rest is a
safety concern. We developed a patient-adaptive decoder that detects binary
hand motor intent versus rest using 36,903 cue-locked windows from 187
represented participants across PhysioNet EEGMMIDB, OpenBMI, and an
acute-stroke dataset. Native-montage quality control was followed by
sensorimotor representations of movement-related cortical potentials,
mu/beta rhythms, and Riemannian covariance. These features were combined with
a five-seed neural ensemble, dataset-specific probability calibration, and a
30-window patient calibration. On later recordings, V7 achieved 73.6%
balanced accuracy, 53.5% intent sensitivity, 93.7% rest specificity, and 0.896
AUROC. Direction labels were used only as optional internal calibration
evidence; the device-facing output remained REST or INTENT. Results support a
rapidly calibrated intent gate for cue-guided rehabilitation, but remain
development-observed and cue-locked. Continuous no-control and locked external
testing are required before clinical use.

---

## 1. Background and Clinical Motivation

### Poster Copy

Rehabilitation brain-computer interfaces (BCIs) aim to connect cortical motor
activity with feedback, stimulation, or robotic assistance. This is especially
important when stroke or paralysis limits measurable muscle output. Motor
imagery and attempted movement can modulate slow cortical potentials and
mu/beta rhythms, but these changes are weak, noisy, and highly variable.

The immediate engineering challenge is therefore not decoding many commands.
It is deciding whether credible motor intent is present while avoiding unsafe
activation during rest.

### Clinical Framing

```text
Known visual or auditory cue: Which movement is requested?
EEG decoder: Did credible motor intent follow?
Output: REST or INTENT
```

### Highlight Box

**Why EEG instead of EMG alone?** EEG may remain available when peripheral
muscle activation is absent, inconsistent, or below a usable threshold. EMG
remains valuable for ground truth, artifact assessment, and optional future
confirmation.

---

## 2. Objective and Research Question

### Objective

Develop a general EEG motor-intent model that can be rapidly individualized
using a short calibration recording while retaining high rest specificity.

### Primary Question

> Can complementary EEG physiology and rapid patient calibration improve
> binary rest-versus-intent detection across healthy and stroke datasets
> without exposing an unreliable left/right command?

### Hypothesis

A population model combining MRCP, SMR, covariance, and compact neural
representations will provide a transferable starting point, while early
patient data will improve later-recording intent detection. Direction labels
may help calibration in some domains but should be rejected when they do not
transfer.

---

## 3. Datasets

| Dataset | Cohort | Participants | Windows | Negative-state definition |
|---|---|---:|---:|---|
| PhysioNet EEGMMIDB | Healthy | 109 | 9,836 | Protocol rest inside imagery runs |
| OpenBMI | Healthy | 29 locally complete | 23,200 | Paired post-task rest |
| Liu acute stroke | Stroke | 49 accepted | 3,867 | Protocol break/rest |
| **Combined** | **Healthy + stroke** | **187** | **36,903** | **Dataset identity retained** |

### Dataset Interpretation

- The datasets increase participant and acquisition diversity.
- Their rest states are not equivalent to unrestricted real-world idle.
- One Liu participant was excluded before referencing because 12 of 29 EEG
  channels failed recording-level quality control.
- Dataset identity remained explicit during model fitting, calibration, and
  threshold selection.

### Figure Placement

Use the dedicated
[Section 3 dataset visual](../../artifacts/poster_v7/sections/03_datasets_and_cohorts.png)
for a standalone poster panel. Vector
[SVG](../../artifacts/poster_v7/sections/03_datasets_and_cohorts.svg) and
[PDF](../../artifacts/poster_v7/sections/03_datasets_and_cohorts.pdf) exports
are available. The left panel of
[Figure 1](../../artifacts/poster_v7/01_v7_poster_graphical_abstract.png), or
the input-domain portion of
[Figure 2](../../artifacts/poster_v7/02_v7_poster_dataset_pipeline.png), can be
used when less detail is needed.

---

## 4. Signal Processing and Feature Pipeline

### Preprocessing

1. Load each recording in its native EEG montage.
2. Detect flat, nonfinite, high-amplitude, noisy, or poorly correlated
   channels.
3. Repair acceptable bad channels before referencing.
4. Apply common-average reference (CAR).
5. Resample to 100 Hz.
6. Select the shared motor-region channels: `FC3`, `FC4`, `C3`, `Cz`, `C4`,
   `CP3`, and `CP4`.

### Parallel Physiological Views

| Branch | Frequency range | Information represented |
|---|---:|---|
| MRCP | 0.1-4 Hz | Slow preparation, onset trajectory, slope, and area |
| SMR | 7-35 Hz | Mu/beta desynchronization and individualized mu activity |
| Riemannian | Four SMR bands | Spatial covariance across motor-region channels |
| Quality/OOD | 1-45 Hz | Availability, artifact, and distribution-shift evidence |

### Key Design Choice

The branches were fused after each produced calibrated evidence. This preserved
physiological interpretability and prevented a single high-dimensional feature
set from dominating through dataset-specific shortcuts.

### Standalone Poster Visual

Use the dedicated
[Section 4 signal-processing visual](../../artifacts/poster_v7/sections/04_signal_processing_and_features.png).
Vector [SVG](../../artifacts/poster_v7/sections/04_signal_processing_and_features.svg)
and [PDF](../../artifacts/poster_v7/sections/04_signal_processing_and_features.pdf)
exports are available for print layout.

---

## 5. V7 Decoder and Patient Calibration

### Population Model

Controller participants selected the following conservative population fusion:

| Intent evidence | Fusion weight |
|---|---:|
| Five-seed MRCP + FBCNet ensemble | 60% |
| Riemannian covariance | 25% |
| SMR statistical features | 15% |

Probabilities were calibrated separately by dataset using Platt scaling and
controller-selected shrinkage. Dataset-specific operating thresholds were then
selected for conservative and sensitivity-targeted policies.

### Rapid Patient Adaptation

Recording 0 supplied:

```text
10 REST + 10 LEFT + 10 RIGHT windows = 30 calibration windows
```

The auxiliary direction model was converted to binary evidence:

```text
P(intent) = P(left) + P(right) = 1 - P(rest)
```

Later recordings were evaluation-only. No calibration row contributed to the
reported patient-adapted metrics.

### Selected Auxiliary Contribution

| Dataset | Auxiliary direction weight |
|---|---:|
| PhysioNet | 0.25 |
| OpenBMI | 0.50 |
| Liu stroke | 0.00 |

The zero Liu weight is a key result. Direction structure helped healthy-domain
calibration but was not reliable enough to improve stroke calibration.

### Output Contract

The deployed decoder exposes only:

```text
REST    or    INTENT
```

The known rehabilitation cue supplies the requested side. EEG does not issue a
left/right command.

### Standalone Poster Visual

Use the dedicated
[Section 5 decoder and calibration visual](../../artifacts/poster_v7/sections/05_v7_decoder_and_patient_calibration.png).
Vector [SVG](../../artifacts/poster_v7/sections/05_v7_decoder_and_patient_calibration.svg)
and [PDF](../../artifacts/poster_v7/sections/05_v7_decoder_and_patient_calibration.pdf)
exports are available for print layout.

---

## 6. Evaluation Protocol

### Subject Roles

- **Development:** fit representations and model parameters.
- **Controller validation:** select fusion, calibration burden, auxiliary
  weight, probability calibration, and operating thresholds.
- **Development observed:** report exploratory V7 performance.
- **Locked external:** currently empty.

### Primary Metrics

- Balanced accuracy: equal average of sensitivity and specificity.
- Sensitivity: percentage of motor-intent windows detected.
- Specificity: percentage of rest windows correctly rejected.
- AUROC: threshold-independent ranking quality.
- Patient-bootstrap confidence intervals: uncertainty estimated by resampling
  participants rather than treating windows as independent.

### Operating Points

- **Conservative:** prioritizes rest rejection and lower false activation.
- **Sensitivity targeted:** detects more intent but accepts more rest errors.

### Standalone Poster Visual

Use the dedicated
[Section 6 evaluation visual](../../artifacts/poster_v7/sections/06_evaluation_protocol_and_leakage_control.png).
Vector [SVG](../../artifacts/poster_v7/sections/06_evaluation_protocol_and_leakage_control.svg)
and [PDF](../../artifacts/poster_v7/sections/06_evaluation_protocol_and_leakage_control.pdf)
exports are available for print layout.

---

## 7. Results

### Headline V7 Result

| Metric | Patient-adapted conservative result |
|---|---:|
| Balanced accuracy | **73.6%** |
| Participant-bootstrap 95% CI | **67.5-78.7%** |
| Intent sensitivity | **53.5%** |
| Rest specificity | **93.7%** |
| False activation per cue-locked rest window | **6.3%** |
| Missed-intent rate | **46.5%** |
| AUROC | **0.896** |
| Available windows | **99.5%** |

Patient adaptation increased balanced accuracy from 69.1% for the matched
population model to 73.6% on the same later-recording rows and patient
thresholds.

### Version Comparison

| Version | Calibration design | BA | Sensitivity | Specificity |
|---|---|---:|---:|---:|
| V5 | Directional adaptation inputs | 73.3% | 52.9% | 93.8% |
| V6 | Strict binary, 20 total windows | 71.9% | 49.1% | 94.7% |
| **V7** | Optional direction evidence, 30 total windows | **73.6%** | **53.5%** | **93.7%** |

V7 produced the strongest conservative result, but its advantage over V5 was
only 0.3 percentage points. The result supports the V7 design without proving
that it is universally superior.

### Domain-Specific Results

| Dataset | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|
| PhysioNet | 63.8% | 40.7% | 86.9% |
| OpenBMI | 76.3% | 57.3% | 95.3% |
| Liu stroke | 65.0% | 40.5% | 89.5% |

Performance remained strongly domain dependent. OpenBMI contributed most of
the pooled strength, while PhysioNet and Liu retained lower intent sensitivity.

### Sensitivity-Specificity Tradeoff

| Patient-adapted policy | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|
| Conservative | 73.6% | 53.5% | 93.7% |
| Sensitivity targeted | 74.1% | 58.5% | 89.6% |

The targeted policy recovered more intent but approximately doubled the
cue-locked false-activation fraction from 6.3% to 10.4%.

### Figure Placement

Use the dedicated
[Section 7 results visual](../../artifacts/poster_v7/sections/07_v7_results_and_domain_variation.png)
as the detailed Results panel. Vector
[SVG](../../artifacts/poster_v7/sections/07_v7_results_and_domain_variation.svg)
and [PDF](../../artifacts/poster_v7/sections/07_v7_results_and_domain_variation.pdf)
exports are available. Use
[Figure 3](../../artifacts/poster_v7/03_v7_poster_results_dashboard.png) when a
more compact dashboard is preferred.

---

## 8. Interpretation

### Results Versus Current Studies

Use the poster-ready
[study-comparison panel](../../artifacts/poster_v7/study_comparison/08_v7_results_vs_current_studies.png)
to place V7 in the published literature. Editable
[SVG](../../artifacts/poster_v7/study_comparison/08_v7_results_vs_current_studies.svg)
and print-ready
[PDF](../../artifacts/poster_v7/study_comparison/08_v7_results_vs_current_studies.pdf)
versions are available. The accompanying
[source and caveat sheet](../../artifacts/poster_v7/study_comparison/STUDY_COMPARISON_SOURCES.md)
contains the primary-study links and the validation differences that must be
stated during a presentation.

For a narrow poster column, use the vertically stacked
[closest-studies panel](../../artifacts/poster_v7/study_comparison/08b_v7_closest_studies_vertical.png),
with editable
[SVG](../../artifacts/poster_v7/study_comparison/08b_v7_closest_studies_vertical.svg)
and print-ready
[PDF](../../artifacts/poster_v7/study_comparison/08b_v7_closest_studies_vertical.pdf)
exports.

**Recommended poster statement:** V7 achieved a cue-locked intent-detection
result in the same broad range as representative stroke motor-intent studies
while evaluating a larger, multi-dataset population with rapid patient
adaptation and explicit rest specificity. It should not be described as
outperforming the literature because the studies use different endpoints,
metrics, calibration strategies, and validation splits.

The closest modern transfer comparison reported 71.15% mean MI-versus-rest
accuracy across 71 stroke patients after healthy-to-stroke domain adaptation.
V7 reached 73.6% balanced accuracy across 187 represented participants, but
remains development observed. A 2026 chronic-stroke dataset reported up to
82.65% accuracy for left-versus-right imagery; that higher percentage is not a
rest-rejection result and used cross-trial evaluation in 14 patients. An online
cue-gated stroke BCI reported 84.88% true-positive detection and 0.52 false
positives per minute, providing the more relevant target for V7's future
continuous safety evaluation.

### What Worked

- Complementary slow and oscillatory physiology outperformed relying on one
  representation alone.
- Five neural seeds reduced dependence on a single random initialization.
- Dataset-specific calibration preserved known acquisition differences.
- Rapid patient adaptation improved the matched later-recording result.
- Optional direction evidence could be retained or rejected by domain.

### Central Finding

> The strongest conservative result came from combining a general population
> model with controlled patient adaptation, not from replacing the complete
> pipeline with a single larger neural network.

### Clinical Meaning

V7 is best interpreted as an EEG **intent proposal** for a scheduled,
cue-guided rehabilitation task. It is not yet an autonomous prosthetic command.
A device-side state machine and independent safety layer should remain between
the EEG proposal and physical movement.

---

## 9. Limitations

- The reported cohort is development observed because it influenced earlier
  architecture development.
- No newly locked external cohort is available.
- Rest examples are protocol specific and do not represent unrestricted daily
  idle behavior.
- Specificity is measured per cue-locked window, not as false activations per
  minute.
- Temporal confirmation is prepared but not evaluated on continuous data.
- Offline zero-phase filtering must be replaced with causal filtering for live
  use.
- The stroke result is affected by differences in hardware, montage, protocol,
  lesion, impairment, and negative-state construction.
- A 30-window calibration is short but still more burdensome than V6's
  20-window binary calibration.

### Required Poster Caveat

> Development-observed, cue-locked result. Continuous no-control safety,
> command latency, and locked external generalization have not been evaluated.

---

## 10. Conclusions

1. A multi-dataset population model can provide a useful starting point for
   binary EEG motor-intent detection.
2. Rapid patient adaptation improved later-recording balanced accuracy while
   retaining high cue-locked rest specificity.
3. Direction-assisted calibration was domain dependent and was rejected for
   the stroke cohort.
4. V7 is the strongest conservative development candidate, but its gain over
   V5 is modest and remains within participant-level uncertainty.
5. The next advance must come from prospective signal validation and safety
   testing, not continued tuning on the same observed participants.

### Final Takeaway

> V7 moves the project from generic EEG classification toward a clinically
> framed, patient-adaptive intent gate, while clearly exposing the remaining
> sensitivity, domain-transfer, and continuous-control challenges.

---

## 11. Future Work

### Immediate Next Study

1. Freeze the V7 preprocessing, model weights, fusion, calibration, and
   thresholds.
2. Implement causal MRCP and SMR processing with measured delay.
3. Collect synchronized visual or auditory cues, EEG, EOG, bilateral EMG,
   force, and movement sensors.
4. Include cue-only catch trials, natural idle periods, distraction, eye
   movement, jaw/neck contraction, and electrode disturbances.
5. Evaluate the prepared three-window confirmation and two-second refractory
   policy.
6. Report false activations per minute, sensitivity, missed intent, command
   latency, abstention, and recalibration burden.
7. Preserve later sessions and a new external cohort as locked confirmation.

### Long-Term Research Direction

Develop a longitudinal patient profile that adapts rest covariance, individual
mu/beta ranges, MRCP timing, artifact limits, probability calibration, and
operating thresholds using only confirmed, high-quality examples.

---

## 12. Poster Figure Plan

### Motivation Panel: Why This Project?

[Open motivation panel](../../artifacts/poster_v7/why_project/00_why_this_project.png)

**Caption:** EEG may preserve cortical evidence of attempted or imagined
movement when usable muscle output is limited, but motor-intent EEG is weak,
patient specific, and difficult to separate safely from rest. The project
therefore combines rapid calibration and complementary EEG evidence in a
binary rest-versus-intent gate after a known rehabilitation cue.

**Recommended placement:** Upper content area in place of a conventional
abstract block.

### Figure 1: Main Graphical Abstract

[Open Figure 1](../../artifacts/poster_v7/01_v7_poster_graphical_abstract.png)

**Caption:** Three EEG datasets were harmonized through native-montage quality
control and represented with parallel MRCP and SMR branches. Controller-selected
late fusion and a 30-window patient calibration produced a binary rest-versus-
intent output. V7 reached 73.6% balanced accuracy and 93.7% cue-locked rest
specificity on later recordings. Results remain development observed.

**Recommended placement:** Full-width centerpiece directly below the motivation
panel.

### Figure 2: Dataset-to-Decoder Pipeline

[Open Figure 2](../../artifacts/poster_v7/02_v7_poster_dataset_pipeline.png)

**Caption:** Dataset identity remains explicit throughout quality control,
physiology extraction, fusion, per-domain probability calibration, and rapid
patient adaptation. Left/right labels are collapsed to binary intent evidence;
the decoder never exposes direction as a device command.

**Recommended placement:** Methods column or center panel.

### Complete V7 Pipeline Alternative

[Open the complete V7 pipeline](../../artifacts/poster_v7/complete_pipeline/09_v7_complete_pipeline.png)

Use this figure when the poster or presentation needs the entire workflow in a
single panel. It separates population development from rapid patient
adaptation, then shows the binary safety gate, later-recording evaluation,
headline results, and validation boundary. Editable
[SVG](../../artifacts/poster_v7/complete_pipeline/09_v7_complete_pipeline.svg)
and print-ready
[PDF](../../artifacts/poster_v7/complete_pipeline/09_v7_complete_pipeline.pdf)
versions are available.

**Caption:** V7 first learns a general EEG intent model from three datasets
using native-montage quality control and parallel MRCP, SMR, and Riemannian
representations. Recording 0 then supplies 30 labeled calibration windows for
patient adaptation; only later recordings are evaluated. The output remains
binary `REST` or `INTENT`, while the rehabilitation cue supplies the requested
movement. The conservative cue-locked result was 73.6% balanced accuracy,
53.5% sensitivity, 93.7% specificity, and 0.896 AUROC. The result remains
development observed because a locked external cohort is not yet available.

### Figure 3: Results Dashboard

[Open Figure 3](../../artifacts/poster_v7/03_v7_poster_results_dashboard.png)

**Caption:** V7 produced the strongest conservative result among the three
patient-adaptation versions, although its advantage over V5 was small.
Performance varied substantially by dataset, and the auxiliary direction
weight was set to zero for Liu stroke data.

**Recommended placement:** Results column.

### Figure 4: Results Versus Current Studies

[Open Figure 4](../../artifacts/poster_v7/study_comparison/08_v7_results_vs_current_studies.png)

**Caption:** V7 reached 73.6% balanced accuracy with 93.7% cue-locked rest
specificity after rapid patient adaptation. Representative motor-intent
studies report results in a similar broad range, but direct ranking is not
valid because endpoints, metrics, cohorts, calibration, and validation splits
differ. Direction-only and online-safety studies are shown separately. V7's
next required proof is a locked external or prospective continuous evaluation
reporting false activations per minute and command latency.

**Recommended placement:** Results or Discussion column beside Figure 3.

Vector PDF and SVG versions are available in
[`artifacts/poster_v7/`](../../artifacts/poster_v7/).

---

## 13. Suggested Landscape Poster Layout

### Header: 12-15% of Poster Height

- Title, authors, institution, and logos.
- One-sentence take-home banner.

### Upper Content: 20-25%

- Use the `Why This Project?` panel across the upper content area.
- Do not repeat the full abstract on the physical poster.
- Keep only a short objective or hypothesis nearby if space allows.

### Centerpiece: 30-35%

- Place Figure 1 across most or all of the poster width.
- Avoid repeating every number from the figure in nearby text.

### Lower Content: 25-30%

- Left: Methods summary and Figure 2 if space allows.
- Center/right: Results summary and Figure 3.
- Use the Results text only for interpretation not visible in the chart.

### Footer: 8-10%

- Conclusions, limitations, future work, references, acknowledgments, QR code,
  and contact information.
- Keep the development-observed and continuous-validation caveat visible.

---

## 14. Thirty-Second Poster Explanation

> This project asks whether EEG can confirm weak motor intent after a known
> rehabilitation cue. We combined 36,903 windows from 187 healthy and stroke
> participants, cleaned each recording in its native montage, and extracted
> complementary slow MRCP and mu/beta SMR information. V7 fuses neural and
> Riemannian evidence, then adapts using 30 early patient windows while scoring
> only later recordings. The conservative result was 73.6% balanced accuracy,
> 53.5% sensitivity, and 93.7% specificity. Direction information helped the
> healthy datasets but was rejected for stroke data. The key limitation is that
> this remains cue locked and development observed, so the next step is frozen
> prospective testing with continuous no-control periods.

---

## 15. Acronyms for a Small Poster Key

| Acronym | Meaning |
|---|---|
| EEG | Electroencephalography |
| BCI | Brain-computer interface |
| MRCP | Movement-related cortical potential |
| SMR | Sensorimotor rhythm |
| CAR | Common-average reference |
| FBCNet | Filter-bank convolutional network |
| OOD | Out of distribution |
| BA | Balanced accuracy |
| AUROC | Area under the receiver operating characteristic curve |

---

## 16. Selected References

### Core V7 Datasets

1. Schalk G. EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet;
   2009. [doi:10.13026/C28G6P](https://doi.org/10.13026/C28G6P).
2. Lee MH, Kwon OY, Kim YJ, et al. EEG dataset and OpenBMI toolbox for three
   BCI paradigms. *GigaScience*. 2019;8(5):giz002.
   [doi:10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002).
3. Liu H, Wei P, Wang H, et al. An EEG motor imagery dataset for brain computer
   interface in acute stroke patients. *Scientific Data*. 2024;11:131.
   [doi:10.1038/s41597-023-02787-8](https://doi.org/10.1038/s41597-023-02787-8).

### Methods

4. Pfurtscheller G, Lopes da Silva FH. Event-related EEG/MEG synchronization
   and desynchronization. *Clinical Neurophysiology*. 1999.
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/10576479/).
5. Barachant A et al. Multiclass brain-computer interface classification by
   Riemannian geometry. *IEEE Transactions on Biomedical Engineering*. 2012.
   [PubMed](https://pubmed.ncbi.nlm.nih.gov/22010143/).
6. Mane R et al. FBCNet: A multi-view convolutional neural network for
   brain-computer interface. 2021.
   [Preprint](https://arxiv.org/abs/2104.01233).
7. Bhattacharyya S et al. Cue-constrained versus asynchronous motor-imagery BCI
   after stroke. *Frontiers in Neuroscience*. 2022.
   [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/).

The complete citation list for every dataset analyzed in the project,
including MRCP EEG-EMG, WAY-EEG-GAL, and HS-Stroke, is provided in
[EEG_BCI_DATASET_CITATIONS.md](EEG_BCI_DATASET_CITATIONS.md). Those auxiliary
datasets should be added to the poster references only if their experiments
are shown.

---

## 17. Claims Checklist

### Supported

- V7 is a reproducible, multi-dataset, patient-adaptive binary intent pipeline.
- The development-observed conservative result is 73.6% BA, 53.5%
  sensitivity, and 93.7% specificity.
- Patient adaptation improved the matched later-recording comparison.
- Performance and useful auxiliary information differed by dataset.

### Not Yet Supported

- Safe autonomous prosthetic or rehabilitation-device control.
- A known continuous false-activation rate per minute.
- Superiority for the average future patient.
- Generalization to all stroke lesions, severities, devices, or sessions.
- Spontaneous asynchronous intent detection.
- Clinical efficacy or improved rehabilitation outcome.
