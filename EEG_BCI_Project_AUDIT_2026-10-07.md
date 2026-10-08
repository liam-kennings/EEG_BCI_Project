# EEG BCI Project — Independent Audit (2026-10-07)

Scope: code, evaluation methodology, results, external tests, reproducibility,
poster/claims, and a literature + clinical-application review. Read-only audit;
no project file other than this one was created or changed.

File:line references point to the pre-audit baseline commit (`git show df2279e:<path>`); later fixes moved some lines.

Evidence labels: **[verified]** = re-computed or read directly during the audit;
**[reported]** = from a delegated code/literature pass, spot-checked but not
fully re-derived. Literature items marked *(verify link)* were found by search
and should be checked before citing.

---

## 1. Bottom line

The project has unusually good *engineering hygiene* for a course project —
subject-level role splits, a separate controller-validation set, thresholds
never chosen on evaluation rows, calibration rows excluded, one-shot locked
external tests that refuse to overwrite themselves, and honest "development
observed" caveats. Those checks came back clean.

The core *scientific* claim — that V7 detects weak **motor intent** at 73.6% BA —
is not yet supported, for four reasons:

1. **The negative class differs from the positive class in ways unrelated to
   intent.** In every dataset, intent windows are taken while a visual cue is on
   screen and rest windows are taken immediately after a task/stimulus ends. A
   6-feature amplitude-only logistic regression (no spectral, spatial, MRCP, or
   neural features, no patient calibration) reproduces the **pooled 73.6%**
   on the same evaluation rows, and beats V7 on OpenBMI (78.7% vs 76.3%).
2. **The headline is a window-pooled number dominated by 6 people.** 77.5% of
   the 4,645 evaluation windows are from 6 OpenBMI participants. Weighted per
   participant, V7 BA is **66.0% (95% CI 62.5–69.6)** over 38 evaluated people —
   not 73.6% across 187.
3. **The 38 evaluation participants have been reused for 7 pipeline versions
   and for the neural-architecture bakeoff**, so they function as a validation
   set, not a test set.
4. **Every evaluation where the cue/trial-phase structure was absent or
   balanced came out near chance** (Cho2017 zero-shot AUROC 0.575; HS-Stroke
   preparation-vs-imagery AUROC 0.524), and neither result appears on the
   poster, although both predate the final poster file.

**Defensible claim today:** an exploratory, cue-locked discrimination of
"cue period" vs "post-task period" windows at ~66% participant-level BA, with
a well-engineered pipeline and protocol ready for a properly controlled test.

> **Update after the follow-up experiments (same day; see §0).**
> - The confound controls refine this claim. PhysioNet retains about 0.70
>   AUROC of imagery-related signal under pre-cue rest, later windows, and
>   causal filters, and beats amplitude-only by 0.12–0.16.
> - OpenBMI's separability is matched by amplitude alone, and Liu stroke shows
>   no physiology advantage over amplitude or EOG.
> - See `docs/final/CONFOUND_CONTROL_FINDINGS.md`.

---

## 0. Follow-up status (2026-10-07)

All changes are on git branch `audit-fixes`. `main` is the pre-audit baseline.

| # | Recommendation | Status |
|---|---|---|
| 1 | Trivial baselines in every table | **Done** for the confound benchmark: amplitude-only, EOG/EMG-only, and a rest-vs-rest control. Not yet added to the V7 pipeline report. |
| 2 | Pre-cue negatives | **Done** (`scripts/54_confound_control_benchmark.py`). Liu has no cue-free rest, so its instruction stage is used instead. |
| 3 | Later intent windows / offset curve | **Done**: offsets 0–2 s against both rest definitions. |
| 4 | Baseline-normalized ERD | **Done**: bandpower normalized to each participant's recording-0 pre-cue windows, plus Riemannian re-centering. |
| 5 | Causal filtering | **Done** in the confound benchmark. V7 itself has not been re-run causally. |
| 6 | Participant-level headline | **Done**: script 30 reports it. V7 reproduction gives 66.0% (62.3–69.6), 21/38 above chance, plus per-window predictions. |
| 7 | Treat the 38 as a validation set | **Done** in the reports, `docs/README.md`, and `PROJECT_STATUS_AND_RESULTS.md` |
| 8 | Report external tests honestly | **Done** in the README and status doc. Cho2017 is reported as zero-shot 0.575. |
| 9 | Safety wording | **Done** in the README and status doc. Poster not edited (on hold). |
| 10 | Citation and window label | Citation **fixed** in the docs. Poster window label still on hold, as is `scripts/47` (poster mockup generator). |
| 11 | Gate fix | **Done** (scoring version 2). HS-Stroke availability went from 0% to 95.1%. Cho2017 stays at 0% because of a separate 20× amplitude-scale shift. |
| 12 | Per-recording re-centering | **Done**: optional within-recording gate mode, and Riemannian re-centering in the controls |
| 13 | Gate regression test | **Done**: `tests/test_quality_gate.py` (9 tests) |
| 14 | Freeze one pipeline | **Not done.** The controls show V7 should first be rebuilt on controlled windows. |
| 15 | Pre-register | **Drafted**: `docs/final/PREREGISTRATION_DRAFT_EXTERNAL_TEST.md` |
| 16 | Untouched cohort | **Not done.** It needs a dataset download and licence acceptance by you. |
| 17 | git | **Done**: repository plus a baseline commit. Data, caches, and poster binaries are ignored. |
| 18 | Environment | **Done**: `.venv` with scipy 1.16.3, all launchers on one interpreter, pins added. VERIFY passes 115 tests. |
| 19 | Canonical docs and shortcuts | **Done**: status doc, docs index, root shortcuts repointed to V7 and the controls, duplicate archive docs removed. Older docs are labelled rather than rewritten. |

### Corrections to this audit

- **PhysioNet (C1).** This audit said a large share of PhysioNet's signal
  looks like a cue transient. The controls show a cue-locked component of
  about 0.05–0.08 AUROC, with most of the signal surviving.
- **Cho2017 confound (C3).** The recording-identity feature separates the
  classes at about 0.87 within-subject |AUROC| (re-verified on all 32
  participants), not 0.958 as first reported. The conclusion is unchanged.

## 2. What is genuinely strong

- **Split discipline [reported, spot-checked].** Roles are assigned by a seeded
  subject shuffle before fitting (`scripts/30_translational_motor_intent_v3.py:355-453`).
  Scalers, Riemannian reference, neural training, and the quality gate are fit on
  development subjects. Fusion, Platt calibration, blends, thresholds, and
  adapter settings are fit on controller subjects (`scripts/30…py:637-719, 1500-1600`).
  Calibration rows are excluded from scoring (`bci_pipeline/translational_decoder.py:1560`).
  No evaluation-row threshold tuning was found.
- **Locked external protocol.** One-shot evaluators that refuse to overwrite
  (`scripts/42_hs_stroke_locked_external.py`, `scripts/52_cho2017_locked_external_v7.py`)
  and a README that reports the failures. Most student projects never attempt
  this.
- **Clinically sensible output contract.** The cue supplies the side and EEG
  only gates rest/intent. This matches how BCI-FES trials (Biasiucci 2018) and
  the FDA-authorized IpsiHand device are structured.
- **Negative results kept:** graph/LDA rejected by the controller, EMG transfer
  CI crossing zero, explicit "not yet supported" claims list.
- **Poster design** is polished and visually coherent.

---

## 3. Critical findings

### C1. Trial-phase / cue confound in the rest-vs-intent contrast [verified]

Window timing (all 2-s windows, one per event; `bci_pipeline/translational_dataset.py`):

| Dataset | Intent window | Rest window |
|---|---|---|
| PhysioNet | 0.5–2.5 s after T1/T2 target **appears** | 0.5–2.5 s after T0, i.e. after the target **disappears** and the imagery ends |
| OpenBMI | 0.5–2.5 s after arrow appears | 5.5–7.5 s after arrow onset = 1.5–3.5 s after imagery **ends**, same trial |
| Liu stroke | 1–3 s into MI stage (grip video playing) | 0–2 s into the break image, ending as the next instruction appears |

So "rest" is always the period immediately after a task ends: post-imagery beta
rebound/termination activity, blinks, swallowing, posture shifts. "Intent" is
always "a visual stimulus is on screen." The literature shows both are
decodable without motor intent. A cue alone evokes central beta ERD on no-go
trials (Pfurtscheller 2013, doi:10.3389/fnhum.2013.00379). CNV/anticipation
potentials in 0.1–1 Hz are single-trial decodable (Garipelli 2013). MI
termination has its own beta signature (Orset 2021).

**Amplitude-only probe [verified].** Logistic regression trained on
development + controller participants and tested on the exact V7 evaluation
rows (development_observed, later recordings). Script in the Appendix.

| Feature set | PhysioNet BA / AUROC | OpenBMI BA / AUROC | Liu BA / AUROC | Pooled BA | Participant-equal BA |
|---|---|---|---|---|---|
| **V7 (full pipeline + patient adaptation)** | 63.8% / 0.774 | 76.3% / 0.926 | 65.0% / 0.737 | **73.6%** | **66.0%** |
| 6 EEG amplitude stats (peak-to-peak, RMS, max abs, mean abs diff) | 52.9% / 0.552 | **78.7% / 0.861** | 61.0% / 0.666 | **73.6%** | 59.2% |
| Classic mu/beta lateralization only (C3−C4 etc.) | 50.3% | 52.8% | 52.5% | 52.5% | — |
| MRCP first 0.5 s only | 61.8% / 0.672 | 57.4% | 51.0% | 57.4% | — |
| MRCP last 1.0 s only | 56.6% / 0.609 | 51.9% | 48.7% | 52.3% | — |

Observations:

- OpenBMI rest windows have **53% larger median peak-to-peak amplitude** than
  intent windows (32.0 vs 20.9 µV).
- The project's own `04_physiology_audit.png` shows OpenBMI intent/rest power
  lower by 3–4.5 dB **across 7–35 Hz**. That is a broadband offset, not a
  band-limited mu/beta ERD (there *is* also a genuine dip at ~11 Hz on top).
- **PhysioNet [reported]:**
  - The Cz "MRCP" difference is a −9.9 µV transient at 0.5–0.75 s after the
    event that has decayed by 2.25–2.5 s. That is event-locked, not a slow
    readiness potential.
  - At FC3, intent vs post-target T0 is −8.0 µV (t=−9.1). Intent vs the first
    T0 of each run (no preceding target) is +0.5 µV (t=0.1).
  - About 90% of the mu contrast comes from rest windows being post-task. This
    rests on only 3 first-T0 windows per subject, so treat it as strong
    suggestive evidence.
- Classical lateralized ERD — the textbook MI signature — is at chance in
  every dataset with these 7 channels and windows.

Fair reading: V7 *does* add value over amplitude on PhysioNet and Liu
(+7 points participant-equal), so it is not *only* amplitude. But on the
dataset that drives the headline, a trivial artifact-sensitive baseline is as
good or better, and the contrast cannot distinguish motor intent from
trial-phase effects. **This is the single most important issue to fix.**

*Follow-up:* the controls in `docs/final/CONFOUND_CONTROL_FINDINGS.md` show
PhysioNet keeps about 0.70 AUROC and beats amplitude under every control.
OpenBMI and Liu do not beat amplitude.

### C2. Headline metric construction [verified]

- `score_intent_operating_point` (`bci_pipeline/translational_metrics.py:188-231`)
  pools all 4,645 rows.
  - Composition: PhysioNet 654 rows / 22 people; OpenBMI **3,600 rows / 6 people**;
    Liu 391 rows / 10 people.
  - Per participant that is about 30 PhysioNet windows and 39 Liu windows,
    versus 600 OpenBMI windows.
- Equal-participant BA [reported, from the metrics JSON per-participant records]:
  - Mean **66.0%** (bootstrap 62.5–69.6), median 66.7%, range 40.0–87.3%.
  - 12/38 participants are below 60%.
- Dataset-equal BA is 68.4%. Without OpenBMI the pooled BA is ≈64%.
- The binomial chance threshold (α=0.05) with ~30–40 windows per person is
  ≈65–67%. **Roughly half the participants are not individually distinguishable
  from chance.**
- AUROC 0.896 is pooled; the dataset-equal mean is 0.812.
- The poster's "187 participants / 36,903 windows" describes the training pool.
  Readers will assume it describes the evaluation.

### C3. External results omitted from the poster; both failures came from a known gate bug [verified]

- **Timeline:**

  | Event | Date |
  |---|---|
  | HS-Stroke locked failure | Aug 3 |
  | Poster content doc ("no locked external cohort was available") | Aug 8 |
  | Poster PDF | Aug 13 |
  | `Poster_V2.pptx` | Aug 16 |
  | Cho2017 V7 locked failure | Aug 17 23:43 |
  | `Poster_V1.pptx` (latest edit) | Aug 18 16:31 |

  HS-Stroke appears only in the acknowledgments line; Cho2017 not at all.
- **The bug.** `QualityAvailabilityGate._statistics` floors the MAD scale at
  1e-6 (`translational_decoder.py:1268`). The binary features
  `auxiliary_available` and `csd_available` have zero MAD, so any dataset
  without auxiliary channels or CSD scores ~1e6 standard deviations.
  - The global threshold is exactly 1e6/√11 = 301,511.3. Cho2017 windows scored
    0.0003–0.001 above it.
  - The README calls this "a real external failure of quality calibration". It
    is the same defect, known since Aug 3, firing again.
  - The only gate test (`tests/test_bci_pipeline.py:1591`) checks finiteness.
- **Cho2017 "adapted AUROC 0.925" is confounded [reported].**
  - Intent comes from the imagery recordings; rest comes from a separately
    preprocessed resting recording.
  - Calibration is drawn from the same arrays.
  - A per-recording constant (`recording_bad_channel_fraction`) alone separates
    the classes at within-subject |AUROC| ≈0.87 (corrected from 0.958 on
    re-verification).
  - So 0.925 measures "which recording", and the bolded post-hoc 72.0% BA in
    the README should not be presented as evidence. The honest number is
    **zero-shot AUROC 0.575**.
- **HS-Stroke 0.524 [reported].** This tested v4, not V7. It contrasted a
  preparation phase (with a lateralized sensory cue) against imagery, so *both*
  classes had stimuli. That near-chance result is consistent with C1: when the
  trial-phase difference is removed, little intent signal remains.

---

## 4. Major findings

- **M1. Evaluation-set reuse [reported, spot-checked].**
  - `subject_roles.json` is byte-identical across v3→V7_graph_lda.
  - BA history on the same 38 people: v3 67.9 → v4 70.6 → v5 73.3 →
    v6 71.9 → V7 73.6.
  - The bakeoff ranked architectures on development_observed
    (`scripts/39_eeg_decoder_model_bakeoff.py:280-327`; its report literally
    says "highest development-observed mean BA"). The top two
    (MRCP-FBCNet, ShallowConvNet) were then hard-wired into V5–V7
    (`bci_pipeline/gui_backend.py:1106-1108`).
  - The v5/v6/V7 differences (≤1.7 points) are well inside the participant CI.
    The "Model Progression" chart shows noise, not progress.
- **M2. "Later recordings" are mostly same-session [reported].**
  - PhysioNet: run 12 vs runs 4/8, minutes apart.
  - Liu: trials 20–39 vs 0–19 of one EDF.
  - Overall, 48% of evaluation rows are within-session. Only the 2,400 OpenBMI
    session-2 rows are cross-day.
  - The adaptation result is mostly within-session adaptation.
- **M3. Non-causal filtering [reported].** Zero-phase filtering runs on whole
  continuous recordings before epoching (`bci_pipeline/quality_preprocessing.py:105-124`).
  - A `--causal` flag exists but was never used for reported results.
  - Zero-phase filtering smears post-cue transients backward and next-cue
    onsets (Liu) into windows.
  - It is not train/test leakage, but it cannot run online and likely inflates
    the MRCP branch.
- **M4. Safety framing outruns the evidence [verified].**
  - The poster labels 93.7% as "SAFETY RESULT" and "LOW FALSE ACTIVATION."
  - Per-window false-positive rate is 6.3%. At one decision per 2-s window that
    is ≈1.9 false activations/min; at 1-s stride, ≈3.8/min.
  - Published asynchronous stroke systems report 0.5–2.7 FP/min. For example,
    Niazi 2022 cue-gated: 84.9% TPR at 0.52 FP/min.
  - The project's own research contract names *event sensitivity at ≤1 FP/min*
    as the primary endpoint, and it has never been measured.
  - Three-window confirmation cannot be credited by assuming independent
    windows. Errors are strongly correlated.
- **M5. Environment no longer runs [reported].**
  - On this macOS (Darwin 27), scipy 1.15.2's Fortran extensions fail to load in
    the pinned Python 3.12. As a result `RUN_GUI.command` and `--check` crash.
  - `VERIFY.command` uses the anaconda `python3` (numpy 1.26), where
    `np.trapezoid` doesn't exist, so it aborts.
  - With an import workaround, **99/99 tests pass in ~21 s**. *(Fixed: project `.venv`; VERIFY now runs 115 tests.)*
- **M6. Tests are mostly smoke tests [reported].**
  - About 45% of asserts check shape or finiteness on synthetic data.
  - Untested: the window builders (where timing/label bugs would live), exact
    metric values, bootstrap correctness, the gate with constant features, and
    any regression test reproducing 73.6%.
- **M7. No version control [verified].** The locked-test provenance relies on
  file timestamps. `scripts/52_…` was modified after its locked run, and the
  contract hashes no code. `git` would make the locked-test story auditable.

---

## 5. Minor findings

- **Window length mismatch.** The poster shows "3 s cue-locked window, 0–3 s".
  Cached windows are 2 s, at 0.5–2.5 s post-event (`translational_windows_v3.npz`:
  200 samples at 100 Hz).
- **Platt calibration.** "Per-dataset Platt calibration" is effectively
  PhysioNet-only; the selected blends are 0.0 for OpenBMI and Liu [reported].
- **Adaptation gain.** The "+4.5 BA" gain (69.1→73.6) scores the population
  model at the *adapter's* thresholds. At its own thresholds the population
  model is 70.6%, so the gain is ≈+3.0 points, with no participant-level CI
  reported.
- **Extra labeled calibration data.** The individualized mu band uses *all*
  labeled recording-0 windows (≈200 OpenBMI, 60 PhysioNet, 40 Liu), not just
  the advertised 30 (`translational_features.py:224-237`) [reported].
- **Bad PhysioNet subjects.** S088, S092, and S100 have known 128 Hz / timing
  defects; S089 has wrong labels. S100 is in the evaluation set. These are
  usually excluded.
- **Wrong citation.** "Bhattacharyya S et al., Frontiers in Neuroscience 2022"
  (`docs/final/EEG_BCI_V7_POSTER_CONTENT.md:703`,
  `docs/final/FULL_PROJECT_BCI_AUDIT_2026.md:193,963`) is wrong. PMC9082379 is
  **Niazi IK et al., *Ann Clin Transl Neurol* 2022;9(5):722–733**: 9 chronic
  stroke patients, attempted **ankle dorsiflexion**, not hand MI [verified].
  It is not on the final poster, but the study-comparison panel's caveat sheet
  should be checked.
- **Model mismatch.** Cho2017 evaluated the Aug-7 `translational_v7_auxiliary_direction`
  model, not the `graph_lda` bundle the README recommends.
- **Stale "final" pointers.** `EEG_BCI_Project_FINAL_REPORT.md`,
  `EEG_BCI_Project_Final_Results`, `IMPLEMENTATION`, `RESULTS_OVERVIEW`, and
  `RUN_FINAL_PIPELINE.command` all point to **v3**. The README "Current Results"
  still headlines v3/v4.
- **V7-H MIND never run.** The README documents `MIND/` and `artifacts/v7h_mind/`
  outputs that don't exist. MIND is also a healthy 4-direction right-arm MI
  dataset, so it is not a good match for this question.
- **Hygiene:**
  - Absolute-path symlinks; the required v3 cache lives outside the project.
  - Script number collisions (41/42/43).
  - 5 docs duplicated byte-for-byte between `docs/final` and `docs/research_archive`.
  - About 65,000 words in `docs/final` alone.
  - "Datasets are not duplicated" is false: Cho2017 (~33 GB) and HS-Stroke
    (~13 GB) are inside the project.
  - `Poster_V1.pptx` is newer than `Poster_V2.pptx`.

---

## 6. Research context

### 6.1 How the numbers compare

| Study | Population | Task / validation | Result |
|---|---|---|---|
| **This project (V7)** | 22 healthy + 6 healthy + 10 acute stroke | Cue-locked MI vs post-task rest; 30-window calibration | 66.0% participant-equal BA; stroke 65.0% BA, 40.5% sens. |
| Ang 2011 (doi:10.1177/155005941104200411) | 46 stroke | MI vs rest, subject-specific offline | 0.74 |
| Nagarajan 2024, JNE (doi:10.1088/1741-2552/ad152f) | 71 chronic stroke | MI vs rest; healthy pretraining + adaptation on ~78 trials/patient | 71.15% (53.0% before adaptation; LOSO 55.9%) |
| López-Larraz 2025, JNER (doi:10.1186/s12984-025-01687-9) | 34 severe chronic stroke | Attempted hand opening vs rest | EEG-only 61.9% (50% of patients above chance); EEG+EMG 69.7% |
| Han et al. (arXiv:2112.09243) | OpenBMI, 54 healthy | L/R/rest, LOSO | 84–85% BA — OpenBMI rest is easy to detect |
| Niazi 2022 (PMC9082379) | 9 chronic stroke | Attempted ankle dorsiflexion, online, cue-gated | 84.9% TPR, 0.52 FP/min |
| Jochumsen 2015 (JNE) | 5 stroke | Attempted grasp, pseudo-online | ~73% TPR, 2.7 FP/min |
| Song 2022 (Front Neurorobot) | 9 stroke | Asynchronous MI, online | 34.8% TPR, 1.03 FP/min |

The stroke result (65%) sits in the normal range for few-shot cross-subject
stroke MI-vs-rest (53–72%). It is neither an outlier nor a weakness. The
OpenBMI number is inflated by easy rest windows, as other OpenBMI work also
shows.

Poster citation checks:

- **Nagarajan 71.15%:** accurate, but that study used ~78 adaptation trials
  per patient vs your 30.
- **HS-Stroke DOI:** the poster's `s41597-026-07742-x` is correct. The local
  file `s41597-026-07807-x.pdf` is the **MIND** fNIRS-EEG paper, not HS-Stroke.

### 6.2 Clinical application

- **Effect sizes are modest.** Recent meta-analyses put BCI rehab at
  **+2.5 to +3.9 FMA-UE** vs control (e.g., Li 2025 JNER, 21 RCTs, MD 3.69).
  That is below the chronic-stroke MCID of ~5.25. BCI-FES subgroups reach
  ~5–5.8. The benefit belongs to the closed-loop *system* and is small, so a
  weak gate can erase it.
- **Contingency is the mechanism.** In Biasiucci 2018 (Nat Commun,
  doi:10.1038/s41467-018-04673-z):
  - Contingency accuracy was 86% in the BCI-FES arm vs 54.5% in sham.
  - Contingency accuracy correlated with FMA gain (r=0.48). False negatives
    correlated negatively.
  - The amount of FES did not predict gain.
  - **Implication:** at 40.5% stroke sensitivity, more than half of genuine
    attempts get no reward. That is closer to sham contingency than to the
    therapeutic arm.
- **The right baseline for a cue-gated device is "fire at cue + fixed delay."**
  When the patient is asked to attempt on every cue, the gate only adds value
  if it correctly *withholds* feedback on cued-but-not-attempted trials. The
  current datasets contain **no such trials**, so the gate's actual job is
  untested. Mrachacz-Kersting 2018 found cue-locked fixed timing produced
  plasticity similar to online detection in healthy participants.
- **Attempted movement beats imagery in stroke.** Chen 2021 found better
  decoding for attempts. Meta-analyses favor attempt-based BCIs: Bai 2020 SMD
  0.69 vs 0.16, Mansour 2022 g=1.21 vs 0.55.
  - Patients in rehab attempt. A model trained mostly on healthy imagery
    shifts task, population, and lesion all at once.
  - IpsiHand (FDA De Novo DEN200046, 2021) deliberately reads the
    *contralesional* hemisphere.
- **Regulatory.** IpsiHand's special controls require evidence that "the EEG
  signal can be translated into intended motion" plus a hazard analysis of
  unintended motion. A continuous false-activation evaluation is therefore
  required, not optional.
- **Practical constraints.**
  - Setup should take under ~15 min.
  - BCI inefficiency is 15–30% classically. OpenBMI's own paper reports 53.7%
    MI-illiterate, and only 50% of severe stroke patients were above chance
    with EEG alone.
  - Plan for a large non-responder fraction and report it.
- **fNIRS (V7-H).** The hemodynamic response peaks 5–8 s after onset, so fNIRS
  cannot speed a sub-second trigger. Its hybrid gains are typically +1–5%. At
  most it fits as a slow confirmation or assessment channel. Deprioritize it.

### 6.3 Better external-validation candidates *(verify links)*

| Dataset | Why it fits |
|---|---|
| **Remaining OpenBMI subjects** (54 total, only 29 used) | Cheapest truly untouched holdout from a familiar domain, if the other 25 are obtainable |
| Medvedeva 2026 stroke EEG+fNIRS (figshare 28904942) | Includes **cues without movement** (non-target flashes): exactly the gate's real failure case |
| CBCIC WCCI 2020 (10 stroke, **attempted** grasp, 3 s pre-cue) | Attempted movement; a pre-cue baseline in every trial |
| "Sixth-finger"/affected-hand stroke Sci Data 2026 (figshare 30509165) | Raw continuous sessions with long labeled rest, so FP/min is computable |

---

## 7. Recommendations (in priority order)

### A. Fix the contrast (days, existing data)

1. **Add mandatory trivial baselines** to every results table:
   - amplitude-only;
   - EOG-only (Liu) and EMG-only (OpenBMI), which should be at chance;
   - an "event-time" baseline.
2. **Re-define negatives** using within-trial pre-cue periods:
   - OpenBMI has a 3 s fixation before each arrow;
   - for PhysioNet, use the *late* part of T0, just before the next target.
3. **Move intent windows later** (e.g., 1.0–3.0 s post-cue) and plot AUROC vs
   window offset. A cue-evoked confound decays with offset; sustained
   ERD/attempt activity should not.
4. **Compute ERD relative to a within-trial pre-cue baseline.** This is
   Concept 1 in your own BME 473 mapping doc, and the standard
   Pfurtscheller-style analysis.
5. **Re-run with causal filtering** applied per window, or with a causal
   streaming filter.

### B. Fix the reporting (hours)

6. **Headline participant-equal BA** with participant-bootstrap CI.
   - State N=38 evaluated (22/6/10).
   - Show the per-participant distribution.
   - Report how many participants exceed their binomial chance threshold.
7. **Relabel the 38 as a validation set.** Stop ranking versions on them.
8. **Report both external tests on the poster/README.** List the gate bug as
   the cause, and give Cho2017 as zero-shot AUROC 0.575, not 0.925/72.0%.
9. **Replace safety wording.** Say "cue-locked rest-window specificity", drop
   "LOW FALSE ACTIVATION", and give the FP/min conversion range.
10. **Fix the Niazi/Bhattacharyya citation and the 3 s window label.**

### C. Fix the gate (hours)

11. **Exclude binary/availability flags** from the robust z-score. Handle
    them as categorical compatibility checks, and floor the MAD scale at a
    physically meaningful value per feature.
12. **Re-center per recording** (Euclidean or Riemannian alignment) before
    gating.
13. **Add a regression test** in which a constant feature differs at inference.

### D. Then test once, cleanly (weeks)

14. **Freeze one pipeline** (V7 or V5; they are statistically
    indistinguishable).
15. **Pre-register thresholds and endpoints:**
    - participant-equal BA;
    - TPR at ≤1 FP/min on continuous data;
    - contingency accuracy vs a fire-at-cue baseline.
16. **Run on an untouched cohort** from §6.3, ideally one with cue-only/no-go
    trials and continuous idle.

### E. Engineering

17. **`git init` now.** Commit code plus small artifacts, not data. Tag the
    frozen model.
18. **Rebuild the environment** for current macOS.
    - Make `VERIFY.command` use the same interpreter.
    - Pin `pyriemann` and `pytest`.
    - Pin `brainflow` and `pylsl` in `requirements-optional.txt`.
19. **Point every "FINAL" shortcut at one canonical V7 report.** Archive side
    experiments (WAY SSL, EMG transfer, multisource, V7-H). Collapse the docs to
    a single methods + results document plus an archive.

### Poster edits (if it will be presented again)

- **Title:** "…for Rehabilitation Device Control" overclaims. Use
  "…Cue-Locked Motor-Intent Detection…" or similar.
- **Results headline:** participant-equal BA, "N=38 evaluated", per-dataset
  distribution.
- **Add a small "External checks" box:** HS-Stroke and Cho2017, gate bug, and
  zero-shot AUROC.
- **Replace "Model Progression"** with "V5/V6/V7 equivalent within CI", or drop
  it.
- **Fix the window diagram** to show 2 s at 0.5–2.5 s, and show what the rest
  window is.
- **Add the amplitude baseline row.** Showing your own confound control
  strengthens the poster.

---

## Appendix A — Amplitude-only probe (reproducible)

Runs in ~1 min with numpy / scipy / scikit-learn on the cached windows.

```python
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = "/Users/LiamKennings_1/Documents/BME 473"
z = np.load(f"{ROOT}/artifacts/gui_experiments/translational_v3/translational_windows_v3.npz", allow_pickle=True)
roles = json.load(open(f"{ROOT}/EEG_BCI_Project/artifacts/gui_experiments/translational_v7_graph_lda/subject_roles.json"))
role_of = {s: r for d in roles for r, v in roles[d].items() for s in v}

y = (z["labels"] > 0).astype(int)
ds, subj, rec = z["dataset_ids"], z["subject_keys"], z["recordings"]
role = np.array([role_of.get(s, "none") for s in subj])
X = z["quality_features"][:, :6].astype(float)  # EEG peak-to-peak / RMS / abs / diff only

probs, labels = [], []
for d in ["physionet", "openbmi", "liu2024"]:
    tr = (ds == d) & np.isin(role, ["development", "controller_validation"])
    te = (ds == d) & (role == "development_observed") & (rec > 0)   # the V7 evaluation rows
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    p = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(d, f"BA {balanced_accuracy_score(y[te], p > 0.5):.3f}  AUROC {roc_auc_score(y[te], p):.3f}")
    probs.append(p); labels.append(y[te])
p, t = np.concatenate(probs), np.concatenate(labels)
print("pooled", f"BA {balanced_accuracy_score(t, p > 0.5):.3f}  AUROC {roc_auc_score(t, p):.3f}")
```

Output on 2026-10-07:

| Dataset | BA | AUROC |
|---|---|---|
| PhysioNet | 0.529 | 0.552 |
| OpenBMI | 0.787 | 0.861 |
| Liu | 0.610 | 0.666 |
| Pooled | **0.736** | 0.823 |

## Appendix B — Key file references

| Topic | Location |
|---|---|
| Window timing | `bci_pipeline/translational_dataset.py` (PhysioNet ~:48, :517-525; OpenBMI ~:612-629; Liu ~:711-732) |
| Pooled metric | `bci_pipeline/translational_metrics.py:188-231`; bootstrap :310-377 |
| Gate bug | `bci_pipeline/translational_decoder.py:1262-1271` |
| Zero-phase filtering | `bci_pipeline/quality_preprocessing.py:105-124, 246-248` |
| Bakeoff ranking on evaluation subjects | `scripts/39_eeg_decoder_model_bakeoff.py:280-327, 890-905` |
| Neural architectures hard-wired into V5–V7 | `bci_pipeline/gui_backend.py:1100-1112` |
| Cho2017 per-array preprocessing | `bci_pipeline/cho2017.py:360-421` |
