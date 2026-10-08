# EEG-EMG Motor-Intent Project Audit 2026

## Scope and conclusion

This audit asks whether synchronized EEG, EMG, force, or kinematic data can improve weak motor-intent detection without changing the scientific claim from cortical intent to muscle-activity recognition.

The central recommendation is a **three-track system**, not one model trained on every available signal:

1. **EEG-only cortical-intent track:** the primary scientific and deployment fallback model. It must work when no usable muscle activity is present.
2. **EMG/kinematic teacher and validation track:** establishes physiological movement onset, identifies covert movement during imagery, detects compensatory activity, and supervises premovement EEG representation learning.
3. **Optional hybrid control track:** combines EEG and residual EMG when a patient has usable peripheral activity. It must always be reported beside EEG-only and EMG-only ablations.

Synchronized movement datasets should therefore improve **label quality, timing, artifact control, representation learning, and optional hybrid control**. Their overt-movement labels should not be pooled directly with motor-imagery labels.

Audit date: **July 31, 2026**.

## Current project baseline

The consolidated v3 experiment contains 36,903 cue-locked windows from 187 participants: 109 PhysioNet participants, 29 locally complete OpenBMI participants, and 49 accepted Liu acute-stroke participants. It uses full-montage QC and repair before average reference, parallel MRCP and SMR branches, CSD where available, FBCSP, Riemannian covariance, domain probability calibration, patient calibration, and subject-level role isolation.

The preferred development-observed operating point is:

| Metric | Current result |
|---|---:|
| Patient-adapted intent balanced accuracy | **67.9%** |
| Intent sensitivity | **41.3%** |
| Cue-locked specificity | **94.5%** |
| Missed-intent rate | **58.7%** |
| Conditional left/right balanced accuracy | **71.0%** |
| Continuous false activations/minute | **Not evaluated** |

This is a useful research baseline, but it is not yet a device-readiness result. The test cohort is development-observed, the final preprocessing is zero-phase rather than causal, and the negatives are protocol-defined windows rather than long continuous no-control recordings.

## Research findings

### Synchronized EEG-EMG can solve the timing problem

The most directly relevant new resource is the [EEG and EMG MRCP hand-gesture dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12969298/), with raw EEG, EMG, and visual-interface triggers from 40 healthy participants over five sessions. It records 32-channel EEG at 128 Hz and forearm EMG at 440 Hz during right-fist closure. Its validation pipeline uses CAR, an anti-Laplacian spatial filter, and 0.1-1 Hz filtering to reveal MRCPs; the raw files are downloadable from [Mendeley Data](https://data.mendeley.com/datasets/y23s2xg6x4/1).

This dataset is a strong fit for the project because EMG onset can define time zero. The EEG model can then train only on windows ending before EMG onset, which is much closer to detecting a cortical preparation signal than the current fixed post-cue window. Its limitations are equally important: it contains right-hand execution only, healthy young adults, and 50 movements per participant.

### EEG and EMG are complementary in severe stroke, but they answer different questions

A 35-patient chronic severe-stroke study recorded 16-channel EEG, EOG, and bilateral EMG while people with no visible finger extension attempted hand opening/closing. The [published analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC12557861/) used one-second windows, small-Laplacian EEG, alpha/beta spectral power, and EMG waveform length. Ipsilesional EEG plus involved-muscle EMG produced 69.7% mean accuracy over all patients and 73.8% among successful users; 79% of patients exceeded chance with the fusion. EEG and EMG performance were weakly correlated, supporting complementary information rather than simple duplication.

That result supports an optional hybrid controller for patients with residual muscle activation. It does not establish that EEG-only motor intent can reach the fused score. Its validation is within-patient block cross-validation, not unseen-patient or cross-dataset evaluation, so its 69.7% cannot be directly ranked against this project's 67.9% balanced accuracy.

### Multimodal fusion can create a peripheral shortcut

The original [WAY-EEG-GAL dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/) contains 3,936 grasp-and-lift trials from 12 participants with 32 EEG channels, five arm/hand EMG channels, kinematics, grip/load force, and 16 event times. It is unusually valuable for studying exact movement phases and force coordination.

However, a 2026 LOSO analysis of WAY-EEG-GAL reported that [EEG-only weight and surface decoding was near chance while EMG-only and fused models were substantially higher](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1874302/full). In several comparisons, adding EEG did not improve EMG and sometimes reduced performance. This is the clearest warning for the proposed project: a higher fused score may only prove that EMG detects executed movement. Every multimodal experiment therefore needs EEG-only, EMG-only, fused, and preferably shuffled-modality controls.

### Robot and kinematic datasets can improve robustness and labeling

The [NeBULA reaching dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092770/) records 127-channel EEG and 11-channel EMG from 40 healthy participants performing free, low-assistance, and high-assistance reaches with an upper-limb exoskeleton. It includes GO/reach markers and robot telemetry. This is a strong source for testing movement artifact, cable/device noise, assistance-domain shift, and compensatory muscle patterns.

The 2026 [EEG-EMG WristMotion dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC13287717/) contains 45 participants with 320 wrist-pointing trials each, synchronized EEG, four bipolar EMG channels, robotic encoders, event markers, and trajectory metrics. It is well suited to direction, latency, electromechanical-delay, and continuous-control research, although its eight-channel EEG montage and overt wrist task differ from the present hand-imagery target.

The 2026 [high-density prehension dataset](https://doi.org/10.1038/s41597-026-07242-y) adds 64-channel EEG, 13 upper-limb muscles, three grip types, and a sensorized grasping box in 14 healthy participants. Its main value is physiological: muscle synergies, corticomuscular coherence, contact timing, and separation of dynamic and holding phases. Its small execution-only cohort makes it a feature-development resource rather than a final generalization benchmark.

### Cue gating is a practical workaround for false activation

In a chronic-stroke comparison, a [cue-based asynchronous BCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/) enabled the detector only during intended attempt intervals. It reported 84.88% true-positive rate and 0.52 false activations/minute, compared with 82.68% and 0.99 false activations/minute for a continuously enabled asynchronous system.

This supports the user's proposed clinical workflow: present a visual or auditory command, open a short intent-detection interval, accumulate evidence, then close the gate. It does not remove the need for continuous testing because speech, gaze, jaw tension, posture changes, and device motion can still occur inside the enabled interval.

### Published accuracy values are not a common leaderboard

| Source | Reported result | What was classified | Validation | Comparison with this project |
|---|---:|---|---|---|
| This project v3 | 67.9% BA | Intent vs rest; direction conditional on accepted intent | Patient-adapted, development-observed, three datasets | Strong conservative cue-locked baseline; 41.3% sensitivity and no continuous FPM |
| [OpenBMI original paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6501944/) | 71.1% accuracy | Left vs right imagery | Subject-dependent; all 54 participants | Similar numeric range to the project's 71.0% conditional direction BA, but not the same subjects, endpoint, or split |
| OpenBMI original, selected performers | 75.5% accuracy | Left vs right imagery | Excludes 11 near-chance users | Not an appropriate target for an all-patient system |
| [Severe-stroke EEG+EMG](https://pmc.ncbi.nlm.nih.gov/articles/PMC12557861/) | 69.7% accuracy | Attempt vs rest | Within-patient block validation; fused EEG+EMG | Supports hybrid control; does not prove unseen-patient EEG-only generalization |
| Severe-stroke responders | 73.8% accuracy | Attempt vs rest | Above-chance users only | Responder-only value should not be compared with an all-user result |
| [HS-Stroke](https://www.nature.com/articles/s41597-026-07742-x) | Up to 82.65% accuracy | Left vs right imagery | Cross-trial evaluation over repeated sessions | Valuable longitudinal data, but cross-trial splitting is easier than held-out patient/session confirmation |
| [Cue-gated asynchronous stroke BCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/) | 84.88% TPR; 0.52 FPM | Event detection during enabled intervals | Nine-person intervention comparison | Sets a useful safety/latency reporting pattern, not an offline balanced-accuracy target |

The current project's accuracy is credible for its harder intent-gating objective, but it cannot yet be called better or worse than these studies. A valid comparison requires the same endpoint, patient adaptation budget, subject/session split, causal signal path, and false-activation definition.

## Project audit

### Critical findings

1. **No locked external result.** All reported performance remains development-observed. The code labels this correctly, which is a strength, but no claim-bearing estimate exists.
2. **No continuous no-control evaluation.** Cue-locked specificity cannot be converted into false activations per minute. Device safety and usability therefore remain unknown.
3. **Cue onset is being treated as a proxy for physiological intent onset.** PhysioNet and most imagery records provide task cues, not EMG-confirmed or kinematically confirmed movement onset. A post-cue window can contain visual evoked responses, attention, protocol timing, and movement artifacts.
4. **Final preprocessing is noncausal.** The reported profile uses zero-phase filters. A live system must use causal filters and report the resulting delay and performance change.

### High-priority findings

5. **EMG and EOG identity is lost.** `quality_preprocessing.py` combines all EOG and EMG channels into `auxiliary`, then stores only availability, maximum peak-to-peak, and maximum RMS. The current model cannot distinguish eye artifact from muscle activation, determine EMG onset, or audit silent imagery.
6. **OpenBMI already contains an unused validation opportunity.** The local loader recognizes four EMG channels, but the v3 cache does not retain their waveforms. Before downloading another large dataset, OpenBMI imagery should be audited for covert muscle activity and labeled `emg_silent`, `possible_contraction`, or `unavailable`.
7. **The quality gate is sensitive to feature scale.** Its distance threshold is approximately 301,511 because heterogeneous amplitude and availability features are not represented on a comparable robust scale. A median/IQR transform and per-domain rejection report would make the gate interpretable.
8. **Negative semantics vary by dataset.** PhysioNet uses protocol rest, OpenBMI uses post-task rest, and Liu uses protocol breaks. Domain calibration helps, but these are not equivalent examples of real-world idle operation.
9. **The common seven-channel montage may be too restrictive for stroke.** It improves cross-dataset compatibility, but stroke reorganization can be bilateral or shifted. A full-montage teacher with a compact deployment student, or hemisphere-aware channel subsets based on lesion side, should be tested.
10. **Patient adaptation is promising but not operationally characterized.** Selected budgets reach 5-10 trials per class. The project should report minutes of setup, failed calibration rate, day-to-day retention, and recalibration triggers, not only accuracy after calibration.

### Strengths to preserve

11. **Full-montage QC precedes average reference.** Bad-channel repair before CAR prevents one corrupted electrode from contaminating every channel.
12. **Physiological branches are explicit.** MRCP, SMR, CSD, FBCSP, Riemannian, PSD/Morlet, and compact statistical features have been evaluated rather than assuming one representation is sufficient.
13. **Controller selection is now nested.** Fusion/probability-calibration subjects are separated from operating-point-selection subjects.
14. **Subject-level roles and bootstrap intervals are implemented.** These should remain mandatory for all adapters.
15. **The software refuses to claim continuous safety without continuous data.** This reporting boundary is scientifically appropriate.

## Recommended v4 pipeline

### Stage 1: preserve synchronized modalities and ontology

Each recording should retain EEG, EOG, EMG, device channels, event triggers, native sampling rates, channel locations, lesion side, session/day, and acquisition hardware. Do not collapse auxiliary signals before synchronization and epoch definition.

Use these state labels:

| State | Meaning | Primary evidence |
|---|---|---|
| `no_control` | Natural idle behavior outside a command | Continuous protocol and device state |
| `cue_perception` | Visual/auditory cue processing before inferred intent | Cue timestamp; posterior/frontal ERP audit |
| `premovement_intent` | Cortical preparation before peripheral onset | EEG window ending before EMG/kinematic onset |
| `imagery_emg_silent` | Imagery with no detectable task or compensatory EMG | Cue plus EMG silence test |
| `attempt_with_residual_emg` | Attempted movement with residual task-muscle activation | Cue plus EMG onset |
| `overt_movement` | Confirmed physical movement | EMG, kinematics, force, or contact sensor |
| `compensation_or_artifact` | Jaw, neck, contralateral, ocular, device, or posture contamination | Modality-specific quality and compensation channels |

### Stage 2: modality-specific preprocessing and features

EEG should keep a causal MRCP branch around 0.1-4 Hz and an SMR filter-bank branch around 7-35 Hz. Compare CAR, small Laplacian/CSD, and lesion-aware bilateral montages. Preserve MRCP amplitude, slope, negative-area, readiness-potential timing, mu/beta ERD/ERS, log-PSD, Morlet summaries, FBCSP, and Riemannian covariance.

EMG should be processed separately with acquisition-appropriate high-pass/band-pass and notch settings. Extract RMS, mean absolute value, waveform length, zero crossings with a noise threshold, slope-sign changes, envelope, onset confidence, and task-versus-compensatory muscle groups. Store raw and normalized values because stroke EMG can be weak and highly patient-specific.

EOG should support blink/saccade regression and artifact labels. ICA or ASR can be included as an offline sensitivity analysis, but neither should silently alter the online signal path. The final candidate must be rerun with the exact causal preprocessing used for deployment.

### Stage 3: teacher-student learning without peripheral leakage

1. Detect EMG or kinematic onset in execution datasets.
2. Extract EEG windows that end before peripheral onset, such as `[-1.5, 0]` seconds relative to onset.
3. Train an execution EEG teacher to recognize premovement preparation, with subject/session grouping.
4. Transfer the cortical representation to imagery using domain-adversarial alignment, contrastive learning, or conservative feature-level initialization.
5. Fine-tune an EEG-only imagery/attempt model on the current healthy and stroke data.
6. Evaluate the EEG-only student without EMG input.
7. Separately train an optional late-fusion controller with reliability gating and modality dropout.

The optional hybrid controller should abstain or fall back to EEG-only when EMG is unavailable, saturated, inconsistent with the requested limb, or dominated by compensation.

### Stage 4: event controller and clinical calibration

Use a cue-gated asynchronous state machine:

`idle -> cue received -> artifact check -> accumulating evidence -> confirmed intent -> direction -> refractory period -> idle`

The detector should require repeated compatible windows, not one high probability. Calibration should include supervised rest, left/right imagery or attempted movement, jaw/eye/speech/posture hard negatives, and known device motion. Store calibration trials chronologically and evaluate only on later trials or a later session.

### Stage 5: evaluation contract

Every final report should include:

- EEG-only, EMG-only, and fused ablations.
- Population-only and patient-adapted results.
- Subject- and session-disjoint validation.
- A final untouched external cohort or session.
- Event sensitivity, false activations/minute, missed-intent rate, latency, refractory failures, abstention, and calibration burden.
- Performance by dataset, stroke/healthy cohort, lesion side, subject, and session.
- Causal versus zero-phase performance and latency.
- EMG-silent imagery results separate from possible covert movement.
- Confidence intervals with the participant, not the window, as the resampling unit.

## Dataset integration order

| Priority | Dataset | Immediate use | Do not use it for |
|---:|---|---|---|
| 1 | [MRCP EEG-EMG hand gestures](https://data.mendeley.com/datasets/y23s2xg6x4/1) | EMG-onset labeling, premovement EEG, five-session drift | Left/right imagery or stroke claims |
| 2 | Existing local OpenBMI EMG | Imagery muscle-leakage audit and EMG-silent labels | Hybrid intent benefit without an ablation |
| 3 | [WAY-EEG-GAL](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/) | Event/force/kinematic teacher and leakage controls | Direct pooling with imagery labels |
| 4 | [WristMotion](https://pmc.ncbi.nlm.nih.gov/articles/PMC13287717/) | Direction, continuous trajectory, latency, and device synchronization | Direct hand-imagery transfer without validation |
| 5 | [NeBULA](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092770/) | Exoskeleton assistance and artifact-domain robustness | Trial-rich classifier training by itself |
| 6 | [HS-Stroke](https://www.nature.com/articles/s41597-026-07742-x) | Longitudinal clinical imagery adaptation and session drift | Claiming 82.65% under a held-out-patient protocol |
| 7 | Severe-stroke attempted-movement cohort | Clinical protocol template or author-request validation | Assuming public patient-level data are available |

The first new adapter should be the MRCP EEG-EMG dataset because it is small enough to integrate quickly and addresses the most important current ambiguity: when cortical preparation occurs relative to actual movement.

**Implementation update (2026-07-31):** this adapter is now complete. The local copy yielded 1,308 EMG-confirmed movement events from 39 paired participants, standardized to 144 MRCP/fixed-SMR EEG features shared exactly with the three-dataset target cache. The first development-observed EEG-only transfer result improved pooled balanced accuracy by 0.3 points and equal-participant balanced accuracy by 0.4 points (95% bootstrap CI -0.1 to +0.9), so the result establishes feasibility but does not yet establish benefit.

## Decision

Do not replace the current v3 model immediately. Freeze it as the comparison baseline and develop v4 beside it. The first v4 milestone should demonstrate that an EEG-only model trained with EMG-defined premovement labels improves sensitivity at matched specificity on untouched imagery/attempt subjects. Only after that should an EEG-EMG fused controller be evaluated.

The executable contract for this milestone is documented in `EMG_SUPERVISED_EEG_TRANSFER_PROTOCOL.md`. It includes target-only, cue-labeled execution, EMG-confirmed, and shuffled-EMG controls so that any gain can be attributed to physiological confirmation rather than merely adding execution trials.

Success should be defined as one of these outcomes:

1. Higher EEG-only event sensitivity at the same false-activation rate.
2. Lower false activations/minute at the same event sensitivity.
3. Lower calibration burden while preserving both.
4. A hybrid controller that improves performance over both unimodal models without losing the EEG-only fallback.

If none occurs, the synchronized data are still valuable because they can show whether the present ceiling is caused by label timing, non-neural contamination, absent physiological intent, or cross-domain variability rather than insufficient model complexity.
