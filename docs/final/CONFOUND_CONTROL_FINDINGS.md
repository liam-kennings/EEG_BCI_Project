# Confound-Control Findings

Date: 2026-10-07. Generated tables and figures:
`artifacts/confound_controls/CONFOUND_CONTROL_REPORT.md`,
`01_confound_conditions.png`, and `02_intent_offset_sweep.png`. Reproduce
with `./EEG_BCI_Project_RUN_CONFOUND_CONTROLS.command` (about 15 min the
first time, while it builds the feature cache).

## Question

In every V7 dataset the intent window is taken while a visual cue is on screen
and the rest window is taken right after a task or stimulus ends. The question
is whether V7's separability reflects motor imagery, or the difference between
"cue just appeared" and "task just ended".

## What was done

Windows were re-extracted from the native recordings, using the same
native-montage preprocessing and the same 7 channels, under five definitions:

1. **V7 windows.** Cue-period intent vs post-task rest.
2. **V7 windows with causal filters.**
3. **Pre-cue rest.** Rest is the 2 s before the cue:
   - PhysioNet: the end of the inter-trial rest.
   - OpenBMI: inside the fixation cross.
   - Liu: the instruction stage. Liu trials are contiguous, so no cue-free
     rest exists; this window still shows the hand cue.
4. **Pre-cue rest with late intent.** The intent window starts 1 s later.
5. **The strictest version:** item 4 with causal filters.

There is also a negative control in which **neither class contains imagery**:
post-task rest vs pre-cue rest.

Every model was trained on the development and controller participants. Each
was scored on V7's 38 evaluation participants, on their later recordings only,
using participant-mean AUROC. Four model families were compared:

- **Transparent physiology:**
  - baseline-normalized mu/beta bandpower,
  - re-centered Riemannian covariance,
  - MRCP bins,
  - all three combined.
- **Amplitude-only:** 6 peak-to-peak and RMS statistics.
- **EOG/EMG-only.**
- **Within-subject versions.** These are fitted on each participant's
  recording 0.

These are not the V7 model. They are transparent stand-ins that isolate what
the windows themselves contain.

## Results

### PhysioNet (healthy, 22 evaluated)

There is real imagery-related signal, and it survives every control.

- Combined physiology reaches AUROC 0.748 on V7 windows. It is 0.703 under the
  strictest condition (95% CI 0.64–0.76).
- It beats amplitude-only by +0.12 to +0.16 AUROC under every intent condition
  (paired CIs exclude zero).
- It does not beat amplitude in the rest-vs-rest control (+0.01, CI −0.06 to
  +0.09). That is the pattern expected if the extra information is about
  imagery rather than trial phase.
- About 0.05–0.08 AUROC of the original separability is cue-locked. AUROC falls
  from 0.73 to 0.65 as the intent window moves from 0 to 2 s after the cue.
  The MRCP features are the most cue-sensitive: with zero-phase filters they
  drop from 0.687 to 0.594 when the window moves 1 s later.
- With within-subject calibration, combined physiology reaches 0.72–0.82 AUROC
  across the intent conditions, against 0.63–0.69 for amplitude-only.

### OpenBMI (healthy, 6 evaluated)

The apparent strength is mostly amplitude and trial phase.

- On V7 windows, **amplitude alone (0.910) beats combined physiology (0.844)**.
  The paired difference is −0.07 (CI −0.09 to −0.04).
- The two no-imagery periods are highly separable: post-task rest vs fixation
  scores 0.72–0.73 AUROC. The original rest window therefore carries a strong
  trial-phase signal unrelated to intent.
- With pre-cue rest, separability is still 0.78–0.80, but amplitude alone
  matches it in every condition (differences −0.01 to +0.03). Some of this may
  be genuine event-related desynchronization (ERD), which lowers central EEG
  amplitude. The data cannot attribute it to anything more specific than an
  overall power change.
- EOG/EMG alone reaches 0.64 on V7 windows and 0.56 with pre-cue rest.

### Liu acute stroke (10 evaluated)

There is no evidence of intent-specific EEG beyond amplitude and eye signals.

- Combined physiology sits at 0.64–0.67 AUROC in every condition and never
  beats amplitude-only. Paired differences run from −0.07 to +0.05, and every
  CI includes zero.
- EOG alone reaches 0.60–0.65, about the same as the EEG models.
- MRCP features are at or below chance (0.45–0.54).
- At threshold 0.5, only 1–3 of 10 patients exceed their individual chance
  threshold.
- Liu has no cue-free rest, so its strictest contrast (instruction vs imagery)
  is cue-vs-cue. It is still the most clinically relevant contrast available:
  "the cue is showing — is the patient imagining?"

### Filtering

- Causal filters did not reduce separability in any dataset. That matters for
  deployment, because V7's zero-phase filtering cannot run online.
- Causal filters do add group delay, though, which can move early cue-evoked
  activity later into the window. That likely explains why MRCP scores rise
  under the causal late-intent condition in PhysioNet (0.594 to 0.671). A
  causal filter alone is not a confound control.

## What this changes in the project's claims

- **V7's pooled 73.6% overstates the evidence.** It is dominated by OpenBMI
  windows, and OpenBMI's contrast is explained by amplitude and trial phase as
  well as by physiology.
- **The strongest motor-imagery evidence is PhysioNet.** About 0.70 AUROC
  survives pre-cue rest, late windows, and causal filters, and clearly exceeds
  amplitude. That is a modest but real, defensible healthy-imagery result.
- **The stroke result is not yet evidence of motor-intent detection.** Simple
  amplitude and EOG models do as well as physiology. Stroke patients in rehab
  *attempt* movement, which is decoded more reliably than imagery, so
  attempt-based stroke data are the natural next test.
- **This audit's earlier description of PhysioNet was too strong.** The audit
  said much of PhysioNet's signal "looks like" a cue transient. The controls
  show a cue-locked component of about 0.05–0.08 AUROC, with most of the signal
  remaining.

## Limitations

- These are transparent models, not V7. V7 retrained on controlled windows
  could score differently. Its neural seeds would need retraining, and the
  window offset would have to be chosen on controller participants, not on
  the 38.
- The 38 evaluation participants are a reused validation set, and OpenBMI has
  only 6 of them.
- The within-subject models use all recording-0 windows, not V7's 30-window
  budget.
- PhysioNet "pre-cue" rest is 2–4 s after the previous task ended, so a small
  post-task residue remains.

## Recommended next steps

1. Rebuild the V7 window cache with pre-cue negatives and causal filters.
   Choose the intent offset on controller participants and retrain the neural
   seeds. Then report V7 against the amplitude baseline on the same rows.
2. Make the amplitude-only and EOG/EMG-only baselines a required row in every
   future results table.
3. For stroke, prioritize attempted-movement data and datasets with cue-only
   (no-go) trials. See `PREREGISTRATION_DRAFT_EXTERNAL_TEST.md`.
