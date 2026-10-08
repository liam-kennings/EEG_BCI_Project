# Project Status and Results

Canonical summary as of 2026-10-07, after the independent audit
(`../../EEG_BCI_Project_AUDIT_2026-10-07.md`) and its follow-up fixes. Where
older documents disagree with this one, this one is current.

## One-paragraph summary

The project builds a binary REST-vs-INTENT EEG gate for cue-guided
rehabilitation. A known cue specifies the movement; EEG confirms whether
intent follows. V7 is trained across PhysioNet EEGMMIDB, OpenBMI, and the Liu
acute-stroke dataset (187 participants, 36,903 cue-locked windows) and is
evaluated on 38 held-out participants.

- **Each participant counted once:** **66.0% balanced accuracy** (95% CI
  62.3–69.6%), and 21/38 participants beat their individual chance threshold.
- **Patient calibration:** adds +6.5 points per participant.
- **Confound controls:** transparent models show genuine imagery-related signal
  in PhysioNet, about 0.70 AUROC after removing cue and trial-phase confounds.
  In OpenBMI, separability is matched by an amplitude-only model. In the stroke
  cohort, physiology does no better than amplitude or eye-movement signals.
- **External tests:** both locked runs (HS-Stroke and Cho2017) failed,
  primarily because of a quality-gate defect that has now been fixed.
- **Status:** none of these results is yet a confirmatory, clinically
  meaningful demonstration of motor-intent detection.

## System

Native-montage QC, bad-channel repair, and common-average reference come
first. The signal is then resampled to 100 Hz and reduced to seven channels
(FC3, FC4, C3, Cz, C4, CP3, CP4). Four evidence branches feed the decoder:

- MRCP (0.1–4 Hz)
- SMR (7–35 Hz)
- Riemannian covariance
- a five-seed MRCP-FBCNet and ShallowConvNet neural ensemble

These are combined by controller-selected late fusion (MRCP-FBCNet 0.60,
Riemannian 0.25, SMR 0.15), followed by per-domain thresholds. A 30-window
patient calibration uses recording 0 (10 rest, 10 left, 10 right). Left and
right are summed into P(intent), so the output is only REST or INTENT. A
quality gate abstains on unusual windows.

Base implementation: `TRANSLATIONAL_V3_IMPLEMENTATION.md`. V7 extensions:
`V7_GRAPH_LDA_COURSE_EXTENSION.md`.

## Evaluation design

| Role | PhysioNet | OpenBMI | Liu stroke | Used for |
|---|---:|---:|---:|---|
| development | 70 | 18 | 31 | Fitting features, models, neural seeds |
| controller_validation | 17 | 5 | 8 | Fusion weights, calibration, thresholds |
| development_observed | 22 | 6 | 10 | Reported results (later recordings only) |
| locked_external | 0 | 0 | 0 | none |

- **The 38 `development_observed` participants are a validation set, not a
  test set.** They scored every version from v3 to V7 and were used to rank
  neural architectures.
- **Evaluation rows are uneven.** There are 4,645 rows: PhysioNet 654 (about 30
  per person), OpenBMI 3,600 (600 per person, 77.5% of all rows), and Liu 391
  (about 39 per person).
- **About half the "later recordings" are from the same session.** PhysioNet
  run 12 vs runs 4/8, and Liu trials 20–39 vs 0–19, are within-session. Only
  OpenBMI session 2 is a different day. In total, 48% of rows are
  within-session.
- **Windows are 2 s long and cue-locked.**
  - Intent starts 0.5 s after the cue (Liu: 1.0 s after motor-imagery onset).
  - Rest starts immediately after a task ends: PhysioNet T0 + 0.5 s, OpenBMI
    cue + 5.5 s, Liu break onset.
  - Filters are zero-phase over whole recordings, which is not usable online.

## Results

### V7, participant level (primary)

From `artifacts/gui_experiments/translational_v7_graph_lda_reproduction`. This
re-run reproduces the original pooled metrics exactly.

| | Value |
|---|---|
| Mean balanced accuracy (38 participants) | **66.0%** (95% CI 62.3–69.6%; dataset-stratified 62.6–69.3%) |
| Median / range | 66.7% / 40.0–87.3% |
| Mean sensitivity / specificity / AUROC | 43.2% / 88.9% / 0.800 |
| Above individual chance threshold | **21/38** |
| Dataset-equal BA | 68.3% |
| Patient adaptation vs population model (own thresholds) | **+6.5 points** (95% CI +3.7 to +9.3); 63% of participants improved |

| Dataset | Participants | Mean BA (95% CI) | Mean AUROC | Above chance |
|---|---:|---:|---:|---:|
| PhysioNet | 22 | 63.7% (59.8–67.8%) | 0.785 | 10/22 |
| OpenBMI | 6 | 76.3% (69.0–82.6%) | 0.936 | 6/6 |
| Liu stroke | 10 | 65.0% (56.7–73.1%) | 0.752 | 5/10 |

### V7, window-pooled (secondary)

73.6% BA, 53.5% sensitivity, 93.7% specificity, AUROC 0.896. Pooling lets
OpenBMI's 6 participants dominate. This is the number on the August poster.

The v5, v6, and V7 conservative results (73.3%, 71.9%, 73.6% pooled) are
statistically indistinguishable.

### Confound controls

Full details in `CONFOUND_CONTROL_FINDINGS.md`.

| Dataset | Combined physiology, V7 windows | Strictest control (pre-cue rest, late window, causal) | Physiology minus amplitude-only | Rest-vs-rest control (no imagery) |
|---|---:|---:|---:|---:|
| PhysioNet | 0.748 | 0.703 | **+0.12 to +0.16** (CIs exclude 0) | 0.59 |
| OpenBMI | 0.844 | 0.799 | −0.07 to +0.03 | **0.72** |
| Liu stroke | 0.667 | 0.663 | −0.07 to +0.05 (CIs include 0) | 0.56 |

Values are participant-mean AUROC from transparent population models on V7's
evaluation rows.

### Locked external tests

| Test | Model | What happened | Corrected reading |
|---|---|---|---|
| HS-Stroke (14 chronic stroke, Aug 3) | v4 population | Quality gate rejected 100% of windows; descriptive AUROC 0.524 | The gate rejection was the indicator-feature bug. With the fixed gate, 95.1% of windows are available. The contrast (cued preparation vs imagery) had cues in both classes and was near chance. HS-Stroke has now been inspected and cannot serve as a locked test again. |
| Cho2017 (32 retained healthy, Aug 17) | V7 | Gate rejected 100%; zero-shot AUROC 0.575; "adapted" AUROC 0.925; post-hoc 72.0% BA | The gate failure had two causes: the indicator bug, and Cho2017 amplitude features about 20× the training data (median RMS 105.7 vs 5.2), which is unresolved and may be a unit issue. The 0.925 is not evidence: rest came from a separately recorded and separately preprocessed resting file, and a feature that is constant within each recording separates the classes at about 0.87 AUROC on its own. Report **zero-shot 0.575**. |

### Supporting experiments

These were all scored on the same validation participants and are exploratory:

- EMG-supervised transfer: +0.4 points, CI crosses zero.
- WAY-EEG-GAL transfer: AUROC 0.683 to 0.697.
- WAY self-supervised pretraining: +2.8 points over a matched CNN.
- Multi-source ensemble: +0.9 points.
- Decoder bakeoff: MRCP-FBCNet chosen.
- V7 graph/LDA extension: rejected by the controller.
- V7-H EEG-fNIRS: implemented but **never run on real data**.

## Safety and clinical status

- **Specificity is per cue-locked window.** At one decision every 1–2 s, a 6.3%
  per-window false-activation rate corresponds to roughly 1.9–3.8 false
  activations per minute before temporal confirmation. Published asynchronous
  stroke systems report 0.5–2.7 per minute. Continuous no-control data have
  never been evaluated, so the project's own primary endpoint (≤ 1 false
  activation per minute) is unmeasured.
- **Stroke sensitivity is 40.5%.** More than half of genuine attempts would
  receive no feedback. In BCI-FES trials, contingency accuracy predicted motor
  recovery (Biasiucci 2018: 86% in the therapeutic arm vs 54.5% under sham).
- **The cue already supplies timing.** The gate only adds value over "fire
  after every cue" if it rejects cued trials with no attempt, and none of the
  current datasets contain such trials.

## What is supported

- The pipeline has clean participant-level role separation. Thresholds and
  calibration never touch evaluation rows.
- **Healthy imagery (PhysioNet):** about 0.70 AUROC of imagery-related EEG
  information survives cue and trial-phase controls and exceeds an amplitude
  baseline.
- Patient calibration improves participant-level BA (+6.5 points), within the
  limits of a reused validation set.

## What is not supported

- Detection of motor intent in stroke beyond amplitude and ocular signals.
- Any locked external generalization.
- Continuous false-activation rates, latency, or safety for device control.
- That V7 is better than v5 or v6.

## Reproducing

```bash
./EEG_BCI_Project_SETUP_ENV.command             # one-time: project .venv
./EEG_BCI_Project_VERIFY.command                # 115 tests + hash checks, ~10 s
./EEG_BCI_Project_RUN_FINAL_PIPELINE.command    # V7 reproduction, ~2 min
./EEG_BCI_Project_RUN_CONFOUND_CONTROLS.command # confound benchmark, ~15 min first run
```

The project is under git. `main` holds the pre-audit baseline, and
`audit-fixes` holds the changes described here.

## Next steps

1. Rebuild V7 on controlled windows: pre-cue negatives, causal filters, and an
   intent offset chosen on controller participants. Retrain the neural seeds,
   and report V7 next to amplitude-only and EOG/EMG-only baselines.
2. Resolve the Cho2017 amplitude-scale question.
3. Run the locked test in `PREREGISTRATION_DRAFT_EXTERNAL_TEST.md` on an
   untouched cohort, preferably with attempted movement, cue-only trials, and
   continuous idle.
