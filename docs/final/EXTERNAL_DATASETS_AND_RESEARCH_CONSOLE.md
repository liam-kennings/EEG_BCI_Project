# External EEG Datasets and Research Console

## Purpose

The current EEGMMIDB experiments appear to be approaching the information
available from a single healthy-subject protocol. More recordings can improve
representation learning and robustness, but heterogeneous datasets should not
be concatenated as if they were acquired by one device and one task.

The recommended expansion has two linked goals:

1. Increase healthy-subject diversity and day-to-day variation during model
   development.
2. Reserve clinically relevant stroke recordings for transfer learning,
   calibration experiments, and genuinely external validation.

The accompanying desktop research console supports recording upload, signal
audit, the current EEGMMIDB experiment, and optional live acquisition. It does
not issue an autonomous rehabilitation-device command.

## Dataset Shortlist

| Tier | Dataset | Cohort | Scale | Main value | Important limitation |
|---|---|---:|---:|---|---|
| 0 | EEGMMIDB | 109 healthy | 64 EEG, 160 Hz | Fixed reference, rest/task ontology, execution-imagery comparison | Cue-locked, single protocol, T1/T2 depend on run |
| 1 | WBCIC Yang 2025 two-class cohort | 51 healthy, 3 days | 59 raw EEG, 1000 Hz | Large multi-day left/right hand set; strongest first healthy addition | Different cue timing, reference, and hardware |
| 1 | Liu 2024 acute stroke | 50 patients | 29 EEG, 500 Hz, 40 trials/patient | Direct clinical transfer and external testing | Small per-patient calibration set; known artifact-heavy recordings |
| 1 | UET175 | 30 post-stroke patients | 22 EEG, 128 Hz, 1-3 sessions | Commercial headset and clinical acquisition domain | Four limb classes; archive password must be requested |
| 1 | HS-Stroke 2026 | 14 chronic-stroke patients | Repeated sessions | Longitudinal clinical adaptation and recovery analysis | New article-in-press dataset; metadata and corrected release require QA |
| 2 | Cho 2017 | 52 healthy | 64 EEG, 512 Hz | Large left/right cross-subject pretraining pool | One session and no independent rest class |
| 2 | OpenBMI | 54 healthy, 2 days | 62 EEG, 1000 Hz | Cross-session healthy pretraining | Complete three-paradigm archive is about 209 GB |
| 3 | BCI Competition IV 2a | 9 healthy | 22 EEG, 250 Hz | Standard adapter and four-class benchmark | Too small to be the main generalization solution |
| 3 | BCI Competition IV Dataset 1 | 7 healthy | 64 EEG, 1000 Hz | Continuous idle-period false-activation stress test | Very small cohort |

The machine-readable version of this table is
`bci_pipeline/dataset_registry.py` and is shown in the GUI's Dataset Library.

### Why These Are the Best Candidates

The WBCIC dataset contains 51 participants in its two-class left/right
hand-grasping cohort, each recorded on three days. Its raw data are BDF, its
processed data are MAT, and it includes 200 balanced trials per session. This
makes it the best first addition for learning session-invariant motor imagery
features. The paper reports 85.32% average EEGNet accuracy, but that is a
within-dataset technical-validation result and is not directly comparable to
this project's unseen-subject, rest-aware controller metrics.
[Yang et al., 2025](https://doi.org/10.1038/s41597-025-04826-y)

The Liu dataset contributes 2,000 left/right hand-grip imagery trials from 50
acute stroke patients. It provides raw MAT, processed EDF, clinical metadata,
and a BIDS mirror. Its published TWFB+DGFMDM result was 72.21% for binary
direction decoding. It does not by itself solve rest-versus-intent detection,
but it is substantially more relevant than another healthy-only benchmark.
[Liu et al., 2024](https://doi.org/10.1038/s41597-023-02787-8),
[NEMAR access](https://ww2.nemar.org/dataset/nm000158)

UET175 contains 30 post-stroke patients recorded with a 22-channel Emotiv EPOC
Flex montage at 128 Hz. It includes left/right arm and left/right leg imagery,
explicit relax intervals, event timestamps, and raw CSV files. This dataset is
particularly valuable for testing whether a model trained on research-grade
EEG transfers to a lower-cost clinical device.
[Ma Thi et al., 2025](https://doi.org/10.3389/fnins.2025.1580931),
[dataset repository](https://github.com/nmk-k66-uet/UET175)

The HS-Stroke article reports 14 chronic-stroke patients and extensive
longitudinal sessions. However, its repository now states that duplicate
within-session trials were removed and that the updated release contains
23,382 trials rather than the article-in-press total of 57,902. This correction
is a reason to perform a complete BIDS and duplicate audit before using the
dataset.
[Lu et al., 2026](https://doi.org/10.1038/s41597-026-07742-x),
[correction and code](https://github.com/KINLAWWW/HS-Stroke)

Cho2017 and OpenBMI add 106 healthy participants across compatible left/right
hand paradigms. Cho2017 provides 64 EEG channels from 52 participants, while
OpenBMI provides 62 channels and two sessions from 54 participants.
[Cho2017 data](https://doi.org/10.5524/100295),
[OpenBMI data](https://doi.org/10.5524/100542)

The BCI Competition datasets remain useful controls. Dataset 2a provides a
widely reproduced four-class GDF benchmark, while Dataset 1 contains continuous
evaluation data with idle periods that are useful for false-activation testing.
They should not outweigh the larger cohorts during training.
[BNCI data catalog](https://bnci-horizon-2020.eu/database/data-sets),
[BCI Competition IV](https://www.bbci.de/competition/iv/)

## Expected Data Expansion

The first healthy expansion wave adds 157 participant datasets:

- WBCIC Yang 2025: 51
- Cho2017: 52
- OpenBMI: 54

Together with EEGMMIDB, this yields 266 healthy participant records across
multiple devices and protocols. The three prioritized stroke resources add 94
clinical participant records:

- Liu acute stroke: 50
- UET175 post-stroke: 30
- HS-Stroke chronic stroke: 14

These counts describe dataset participants, not guaranteed independent clinical
populations, and they do not imply a proportional accuracy increase.

## Harmonization Contract

Every external loader should produce the same intermediate BIDS-like trial
table before any model sees the data:

| Field | Required meaning |
|---|---|
| `dataset` | Immutable source-domain identifier |
| `subject` | Dataset-qualified subject identifier |
| `session` | Recording day/session; never merge silently |
| `run` | Acquisition block |
| `cue_onset_s` | Recorded visual/auditory cue onset |
| `intent_start_s` | Protocol-defined imagery interval start |
| `intent_end_s` | Protocol-defined imagery interval end |
| `native_label` | Original event code |
| `canonical_label` | `rest`, `left_hand`, or `right_hand` |
| `channels` | Normalized 10-10 names and original reference |
| `sampling_hz` | Native sampling frequency |
| `line_frequency_hz` | 50 or 60 Hz acquisition environment |
| `quality_flags` | Missing, flat, noisy, interpolated, or rejected channels |

### Liu Acute-Stroke Import

The local Liu dataset now has a validated EDF adapter. Prepare its clinical
metadata, subject-specific trigger timing, trial manifest, and overview figure
with:

```bash
python scripts/23_prepare_liu_stroke_dataset.py \
  --data-dir "Liu acute-stroke dataset"
```

Derived files are written under `artifacts/liu_stroke/`; the downloaded source
files are never modified. The adapter marks HEOL/HEOR as EOG, excludes the flat
CPz online-reference signal from EEG, and identifies the unnamed EDF channel as
the trigger stream. Instruction, imagery, and break timing remain distinct.
The importer reconciles isolated missing or extra pulses against the fixed
8-second protocol and flags every fallback in the trial and recording manifests.

### OpenBMI Import

The OpenBMI NEMAR BIDS mirror has a validated adapter for string-based event
mapping, EMG exclusion, session-aware run IDs, paired imagery/rest windows, and
the common model-ready `RawEpochDataset` cache:

```bash
python scripts/24_prepare_openbmi_dataset.py \
  --data-dir files/external/openbmi/nm000338
```

Add `--build-epoch-cache` to read the BDF signals, apply CAR and 4-35 Hz
preprocessing, select the common nine-channel sensorimotor montage, resample to
160 Hz, and save `artifacts/openbmi/openbmi_raw_epochs.npz`. The local NEMAR
zip currently contains complete signals for 29 of 54 subjects; final OpenBMI
claims must wait for the remaining BDF files.

### Controlled OpenBMI Benchmark

The validated adapter now supports a stricter OpenBMI experiment using the
paper's 20-channel central montage, the 1.0-3.5 second imagery window, paired
5.0-7.5 second rest, subject-isolated development/controller/final partitions,
calibration-only alignment, and an occipital visual-cue control:

```bash
python scripts/25_openbmi_controlled_benchmark.py \
  --data-dir files/external/openbmi/nm000338 \
  --output-dir artifacts/openbmi_controlled
```

The available 29-subject cohort produced 23,006 valid windows. The
controller-selected log-Euclidean aligned tangent model reached 82.7% intent
balanced accuracy on six untouched subjects, with 12.4% false activation and
22.2% missed intent under the locked safety policy. Twenty calibration trials
per class improved intent balanced accuracy to 87.7%, with 11.5% false
activation and 13.1% missed intent.

This did not solve command direction. Cross-subject left/right balanced
accuracy remained 51.6% before calibration and 53.7% after calibration, while
the easier same-subject CSP/LDA reproduction averaged 68.0%. Early 0-0.5 second
occipital direction decoding reached 57.9% but returned to 50.6% in the
1.0-3.5 second window. These controls support the conclusion that population
intent gating is promising, but subject-specific direction decoding and
continuous safety validation remain required.

The locked metrics, predictions, model artifacts, report, and seven
presentation figures are under `artifacts/openbmi_controlled/`.

### Harmonized PhysioNet-OpenBMI Comparison

PhysioNet EEGMMIDB and OpenBMI now also run through one matched pipeline:
nine common central channels, 4-38 Hz CAR, 100 Hz, 1.0-3.5 second windows, the
same five-band covariances, calibration-only log-Euclidean alignment,
hierarchical decoding, subject isolation, and controller-selected safety
policies. Pooled training gives the datasets equal total weight.

```bash
python scripts/26_harmonized_physionet_openbmi_benchmark.py
```

On untouched subjects, the within-dataset global intent balanced accuracies
were 54.8% for PhysioNet and 79.3% for OpenBMI. Dataset-specific calibration
raised these to 65.8% and 79.8%. Equal-weight pooled calibration produced 59.4%
on PhysioNet and 83.3% on OpenBMI: adding OpenBMI helped the OpenBMI intent
gate but did not improve PhysioNet over its own calibration.

Direction showed the opposite pattern. The pooled calibrated direction scores
were 59.9% for PhysioNet and 53.2% for OpenBMI. Label-free aligned transfer
retained some direction information but failed at intent gating, particularly
OpenBMI-to-PhysioNet, which missed 89.5% of PhysioNet intent. The dominant
cross-dataset mismatch is therefore likely task/rest protocol and cohort
distribution, not simply channel or preprocessing differences.

The complete report, locked metrics, predictions, model artifact, calibration
curves, transfer matrices, and presentation figures are under
`artifacts/harmonized_physionet_openbmi/`.

### Signal Processing

1. Preserve raw files and source metadata before conversion.
2. Detect bad channels before CAR so one failed electrode is not spread across
   the montage.
3. Apply the source-appropriate 50 or 60 Hz notch before resampling.
4. Use a common offline bandpass, with separate branches for slow potentials
   and 4-35 Hz sensorimotor activity.
5. Re-reference consistently after bad-channel handling.
6. Resample the current handcrafted-feature pipeline to 160 Hz. A future raw
   EEGNet experiment may use 250 Hz.
7. Retain a 15-channel central motor montage when it exists:
   `FC3, FC1, FCz, FC2, FC4, C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4`.
8. Do not interpolate a missing channel until its effect is tested in an
   ablation against a strict channel-intersection model.
9. Fit robust scaling, alignment references, CSP, covariance centering, and
   feature selection only on source/development subjects.

### Label and Timing Rules

- Do not use the same epoch offset for every dataset merely because the labels
  are both called left hand.
- Preserve cue onset separately from protocol-defined imagery onset.
- Treat long breaks as continuous rest only when their marker timing is known.
- Do not manufacture a rest class from an unmarked edge of a trial.
- Map UET175 leg trials and WBCIC foot trials to a separate limb task; do not
  relabel them as left/right hand.
- Keep execution trials out of the imagery classifier unless they are used in a
  declared auxiliary or pretraining objective.

## Leakage-safe Training Plan

### Phase A: Healthy Representation Development

1. Freeze the existing EEGMMIDB development, controller-validation, and final
   subject IDs.
2. Add WBCIC Yang 2025 first because it provides three sessions and balanced
   left/right imagery.
3. Retain the controlled OpenBMI result as an external reference and add
   Cho2017 after its MAT adapter passes channel, unit, and event tests.
4. Balance batches by dataset and subject so the largest archive does not
   define the entire decision boundary.
5. Compare a strict common-montage model with a geometry-aware model that can
   tolerate missing channels.

### Phase B: Clinical Transfer

1. Hold out complete Liu and UET175 patients.
2. Evaluate the healthy model zero-shot before using any patient labels.
3. Adapt with fixed budgets of 1, 3, 5, and 10 trials per class.
4. Preserve later sessions or runs as unseen evaluation data.
5. Tune thresholds only on a clinical validation cohort, never on the final
   patient cohort.

### Phase C: External and Longitudinal Validation

1. Integrate HS-Stroke only after validating the corrected release and BIDS
   metadata.
2. Run leave-one-dataset-out evaluation.
3. Report performance separately for healthy, acute stroke, post-stroke, and
   chronic-stroke domains.
4. Use BCI Competition IV Dataset 1 for continuous idle-period testing.

## Required Metrics

Every cross-dataset experiment should report:

- Intent balanced accuracy and macro F1
- Direction balanced accuracy conditional on detected intent
- False-activation rate and false activations per minute
- Missed-intent rate
- Command precision and accepted-command coverage
- Detection latency
- Calibration error or reliability curve
- Per-subject and per-dataset distributions
- Zero-shot result and each patient-calibration budget
- Bootstrap confidence intervals for changes from the fixed reference

A result should not be called improved when pooled accuracy rises but a
held-out dataset, patient subgroup, or false-activation metric deteriorates.

## Research Console

Launch the desktop application from the project directory:

```bash
python scripts/22_bci_pipeline_gui.py
```

Check backend readiness without opening a window:

```bash
python scripts/22_bci_pipeline_gui.py --check
```

Export the dataset catalog:

```bash
python scripts/22_bci_pipeline_gui.py \
  --export-catalog artifacts/dataset/external_motor_imagery_catalog.json
```

### Recording Workspace

Supported standard formats are EDF/EDF+, BDF, GDF, FIF, BrainVision VHDR,
EEGLAB SET, and CNT. CSV, TSV, NPY, and NPZ arrays are also supported when the
sampling frequency, channel names, and voltage unit are supplied.

The signal audit produces:

- Recording metadata and annotation inventory
- Gross bad-channel and amplitude checks
- Raw central-channel EEG traces
- CAR and 4-35 Hz processed traces
- Processed power spectral density
- Mu and beta relative power
- A PNG figure and JSON audit record

This verifies that data can enter the pipeline. It is intentionally not
reported as a motor-intent prediction.

### Dataset Experiments

Select the EEGMMIDB `1.0.0` folder and run:

1. Base physiology feature atlas
2. Extended feature audit
3. Controlled morphology/SSD ablation

The GUI streams command output and writes all results below the selected
experiment output folder. It uses the current fixed subject split.

### Live Device

File replay works with the existing environment. The other adapters are
optional:

```bash
python -m pip install pylsl
python -m pip install brainflow
```

LSL is the preferred generic transport because it supports synchronized EEG
and event-marker streams. BrainFlow provides a board-agnostic API for devices
including OpenBCI hardware. The console retains up to ten minutes and can run
the same post-capture signal audit.

[Lab Streaming Layer](https://labstreaminglayer.org/) provides synchronized
stream discovery, subscription, and recording. [BrainFlow](https://brainflow.readthedocs.io/en/stable/)
provides a common data-acquisition interface across supported biosensor boards.
OpenBCI can stream data and markers through its GUI networking widget.
[OpenBCI LSL documentation](https://docs.openbci.com/Software/CompatibleThirdPartySoftware/LSL/)

### Safety Boundary

Live capture is not yet live control. Before decoder output can drive a device,
the system still needs:

- Verified device channel names and units
- Synchronized cue or command markers
- Patient-specific rest and imagery calibration
- A causal preprocessing profile
- Prospective false-activation and emergency-stop tests
- Human-supervised command confirmation
- Clinical and regulatory review
