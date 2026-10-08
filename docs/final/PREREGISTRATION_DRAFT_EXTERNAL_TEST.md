# Pre-Registration Draft: Next Locked External Test

Status: **DRAFT, not frozen** (created 2026-10-07 during the audit follow-up).
Nothing in this document has been run. Freeze it by completing Section 6 and
committing it with a git tag *before* any external data are loaded.

## 1. Why a new test is needed

- The 38 `development_observed` participants have scored every pipeline
  version since v3 and were used to rank neural architectures. They are a
  validation set, not a test set.
- HS-Stroke (v4) and Cho2017 (V7) are spent. They were inspected after their
  locked runs, and both failures traced to a quality-gate defect that has now
  been fixed (`QualityAvailabilityGate` scoring version 2).
- The original rest-vs-intent contrast is confounded by trial phase (cue on
  screen vs task just ended). See `CONFOUND_CONTROL_FINDINGS.md`. A new test
  must use negatives that do not carry that difference.

## 2. Candidate cohorts (choose one before freezing)

| Cohort | Why | Status |
|---|---|---|
| Remaining OpenBMI participants (54 total; 29 used) | Same hardware, so isolates participant generalization. Cheapest. | Not downloaded; check that the other 25 are complete |
| Medvedeva 2026 stroke EEG+fNIRS (figshare 28904942) | Contains cues **without** movement, the gate's real failure case | Not downloaded; verify link and licence |
| CBCIC WCCI 2020 stroke challenge (10 patients) | Attempted grasp, not imagery; 3-s pre-cue baseline | Not downloaded; verify link |
| Affected-hand stroke MI, Sci Data 2026 (figshare 30509165) | Raw continuous sessions with long labelled rest, so FP/min is computable | Not downloaded; verify link |

The candidate links came from a literature search on 2026-10-07 and have not
been opened in this project. Downloading needs the dataset owner's terms to be
accepted.

## 3. What gets frozen

All of the following are fixed using only existing development, controller,
and validation participants. No external row may influence any of them.

1. **Preprocessing.** `BranchPreprocessingProfile(phase="forward")`: causal
   filters, same bands and channels.
2. **Windows.** Negative = the 2 s before the cue (or the dataset's cue-free
   idle). Intent = the 2-s window starting at the offset chosen on the
   validation set from the offset sweep in `artifacts/confound_controls`.
3. **Decoder.** One model, selected on controller participants. The
   validation participants only confirm it is not worse than the amplitude
   baseline.
4. **Calibration protocol.** Number and type of calibration windows per
   participant, taken from the participant's first recording.
5. **Thresholds.** Per-domain conservative threshold from controller
   participants. A new domain uses the pooled fallback threshold, with no
   tuning on external data.
6. **Quality gate.** Version 2, within-recording mode. Report the
   compatibility record, which shows which indicators differ.

## 4. Endpoints (declared now)

### Primary

Participant-mean balanced accuracy at the frozen threshold, with a 95%
participant-bootstrap CI.

### Secondary

- Participant-mean within-participant AUROC.
- Number of participants above their binomial chance threshold.
- Participant-mean sensitivity and specificity.

### Mandatory baselines, run on identical rows

- Amplitude-only model.
- EOG/EMG-only model, where those channels exist.
- Fire-on-every-cue rule: sensitivity 100%, specificity 0% on cue-only trials.

### If the cohort supports it

- Continuous data: true-positive rate at ≤ 1 false activation per minute, and
  median detection latency.
- Cue-only (no-go) trials: false-activation fraction on cue-only trials.

## 5. Success criteria (declared now)

The pipeline is considered to generalize only if **all** of these hold:

1. The lower 95% CI bound of participant-mean AUROC is above 0.60.
2. Participant-mean AUROC exceeds the amplitude-only baseline by at least 0.05.
3. The EOG/EMG-only baseline is below 0.60 participant-mean AUROC, so ocular
   and muscle artifact are not driving the result.
4. At least half of participants exceed their own chance threshold.

Any other outcome is reported as a failure, with the same prominence.

## 6. Freeze record (complete before loading external data)

- [ ] Cohort chosen:
- [ ] Decoder and window offset chosen (controller evidence attached):
- [ ] Model artifact SHA-256 values listed:
- [ ] Git commit and tag (`locked-external-<cohort>-v1`):
- [ ] Execution receipt path:

## 7. Execution rules

- Run once. The evaluator refuses to overwrite its output, as
  `scripts/42_hs_stroke_locked_external.py` already does.
- Record the git commit hash in the receipt. Do not edit evaluation code
  after the run. Any later analysis is labelled post-lock exploratory.
- Deviations from this document are listed in the report, not silently
  absorbed.
