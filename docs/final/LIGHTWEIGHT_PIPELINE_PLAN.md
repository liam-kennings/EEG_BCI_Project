# Lightweight EEG Motor-Intent Pipeline: Plan for MATLAB and Python

Status: plan only (2026-10-22). No code has been written for it.

Scope: a binary REST vs INTENT gate for cue-guided rehabilitation. One shared
specification, two independent implementations (MATLAB and Python). Each runs
on recorded data and in real time, and produces the same outputs.

Sources: the lessons in `EEG_BCI_Project_AUDIT_2026-10-07.md` and
`CONFOUND_CONTROL_FINDINGS.md`, and the band-power pipeline of Lee et al.
(2022, *Front. Neurosci.* 16:1009878).

---

## 1. Design principles

1. **One code path for offline and online.** Every processing step is a
   *streaming block*: it takes a chunk of samples plus its saved state, and
   returns output plus the updated state. Offline analysis feeds the recorded
   file through the same blocks in chunks ("replay"), then cuts epochs from the
   resulting **feature stream**. This guarantees that offline results are
   achievable online, which V7's zero-phase filtering was not.
2. **Light by construction.**
   - Causal IIR filters.
   - Mu/beta band power on a 4 Hz feature stream.
   - A closed-form linear classifier (shrinkage LDA).
   - Optional extras: small-channel Riemannian covariance and MRCP. No deep
     learning is required.
   - Target: under 5 ms of compute per 0.25 s update for 32 channels on a
     laptop. This is an estimate; measure it in milestone M5.
3. **Configuration is data.** All parameters live in one versioned
   `pipeline_config.json`, readable by both languages. Every output records the
   config hash and the git commit.
4. **Built-in safeguards** from the audit:
   - Negatives are pre-cue rest, cue-only (no-go) trials, or continuous idle.
     Never the window right after a task ends.
   - Results are reported per participant. Cross-validation groups by run or
     session, not by trial.
   - The time point or window is fixed in advance, never picked as the best
     one afterwards (the Lee et al. weakness).
   - Amplitude-only, EOG/EMG-only, and rest-vs-rest baselines are run on the
     same rows as every result.
   - The quality gate scores only continuous features. Missing-modality flags
     are reported, never used as rejection reasons.
   - A locked test cohort is chosen before any analysis.

---

## 2. Architecture

```
            ┌────────────── shared pipeline_config.json ──────────────┐
            │                                                         │
 OFFLINE:  file reader ──► chunker ─┐                                 │
 (EDF/BDF/XDF/.mat)                 │                                 │
                                    ▼                                 ▼
 ONLINE:   LSL EEG inlet ──► [S1 channel map] ─► [S2 health] ─► [S3 causal filter bank]
           LSL marker inlet ──┐                               │
                              │                     ┌─────────┴──────────┐
                              │                     ▼                    ▼
                              │            [S4 re-reference]     [S6 artifact features]
                              │                     │                    │
                              │                     ▼                    │
                              │            [S5 feature stream: log band power,
                              │                drift compensation, optional cov/MRCP]
                              │                     │                    │
                              ▼                     ▼                    ▼
                         event table ───► [S7 decoder P(intent)] ◄─ [S6 gate: available?]
                              │                     │
                              │                     ▼
                              │            [S8 decision layer: cue-gated / async,
                              │                evidence accumulation, refractory]
                              │                     │
                              ▼                     ▼
 OFFLINE ANALYSIS:   epoch the feature stream    ONLINE OUTPUT: LSL outlet
   → statistics, topomaps, reports               (P(intent), decision, gate flag)
```

The state of every block is a small struct (MATLAB) or dataclass (Python). It
can be saved and reloaded, so a session can resume after a dropout.

---

## 3. Shared specification

### 3.1 Inputs

| Mode | Source | Notes |
|---|---|---|
| Offline | EDF/BDF (PhysioNet, Liu, OpenBMI), XDF (LSL recordings), lab `.mat` | Each reader returns `data [ch × samples]`, `fs`, `channel table` (name, type, x/y/z), `event table` |
| Online | LSL stream `type='EEG'` plus a `type='Markers'` stream | Uses LSL timestamps. Chunks of about 50 ms |

**Event table** (one row per event, identical in both languages): `onset_s`,
`type` (cue_left, cue_right, cue_nogo, rest_start, idle_start, …), `trial`,
`run`, `session`.

**Window definitions** (one table per dataset, not code):

| Field | Meaning |
|---|---|
| `class` | rest or intent |
| `anchor` | which event the window is timed from (e.g. cue_left) |
| `start_s`, `end_s` | window bounds relative to the anchor |
| `role` | train, calibration, test, or locked |

Default intent window: +0.5 to +3.0 s after the cue. Default negative: −2.0 to
0 s before the cue, plus cue_nogo windows and idle segments where they exist.

### 3.2 Default parameters

| Block | Parameter | Default | Rationale |
|---|---|---|---|
| Resample | `fs_target` | 200 Hz (100 Hz acceptable) | Lee et al.; enough for 8–45 Hz content |
| Line noise | notch | 50 or 60 Hz, 4th-order Butterworth band-stop (±2 Hz), causal | Lee et al. cascade |
| Health | bad-channel rule | z(log power 8–25 Hz) > 6 at calibration; flat-line; saturation | Lee et al.; simple and auditable |
| Health | bad-run rule | more than 10% of channels bad | Lee et al. |
| Reference | `reference` | common average over good channels, or a small Laplacian for C3/Cz/C4 | Laplacian is cheap and sharpens sensorimotor signals |
| Bands | `bands` | mu 8–12 Hz, beta 13–30 Hz (optionally split 13–20 / 20–30) | Motor ERD/ERS |
| Filters | `order` | 4th-order Butterworth band-pass, second-order sections, **causal** | Real-time capable |
| Band power | block / smoothing | square → 0.25 s block mean → **trailing** 0.75 s moving average → log10 | Lee et al., made causal |
| Drift | `drift_window_s` | 25 s trailing mean subtracted (toggle); alternative: subtract pre-cue reference | Lee "power shift compensation"; tests the amplitude confound |
| Feature rate | | 4 Hz | One decision every 0.25 s |
| Optional | covariance | 2 s window, 0.25 s hop, ≤ 16 channels, tangent space at the calibration mean | Riemannian branch |
| Optional | MRCP | causal 0.1–3 Hz, 1 s of amplitude at 4 Hz | **Only if it survives the confound controls** |
| Artifacts | EOG / EMG | EOG peak-to-peak; 20–45 Hz power on EMG or temporal channels; EEG peak-to-peak | Gate and baselines |
| Decoder | `model` | shrinkage LDA (Ledoit-Wolf / OAS), then logistic calibration | Closed form, fast, works with few trials |
| Calibration | `n_cal_per_class` | 20 (run a burden curve over 5–40) | Short patient calibration |
| Decision | `mode` | cue-gated (evaluate 0–4 s after the cue) or asynchronous | Clinical framing |
| Decision | accumulation | 2-of-3 consecutive updates above the threshold, 2 s refractory | Cuts single-update false positives |

### 3.3 Outputs (identical schema in both languages)

| File | Content |
|---|---|
| `run_manifest.json` | config hash, git commit, software versions, input file hashes, excluded channels and runs |
| `features.h5` | 4 Hz feature stream per recording (band power, artifact features, timestamps) |
| `epochs.h5` | epoched features `[trial × channel × band × time]` plus trial metadata |
| `predictions.csv` | one row per decision update: participant, run, time, P(intent), threshold, gate, decision, true label if known |
| `participant_metrics.csv` | one row per participant per model or baseline |
| `group_metrics.json` | group summaries with confidence intervals and p-values |
| `figures/` | PNG plus SVG (Section 5) |
| `REPORT.md` / `REPORT.html` | auto-generated run report linking the above |

Both languages read and write `.h5` (MATLAB `h5read`/`h5write`, Python
`h5py`), `.csv`, and `.json`, so outputs can be compared across languages.

---

## 4. Processing stages

Each stage lists its algorithm, whether it is causal, its MATLAB and Python
building blocks, and its cost.

**S0 Input and chunking**
- Offline readers emit fixed-length chunks (default 50 ms) with timestamps.
- Online inlets pull chunks from LSL.
- MATLAB: `edfread`; EEGLAB `pop_biosig` for BDF; `load` for `.mat`;
  `liblsl-Matlab` (**add-on, not installed**).
- Python: `mne.io.read_raw_edf` / `read_raw_bdf` (input only); `pyxdf`;
  `pylsl`.

**S1 Channel map**
- Pick channels and attach positions from a standard 10-05 montage.
- Store neighbour lists (for the Laplacian) and topomap coordinates.
- Runs once per session.

**S2 Channel health**
- **At calibration** (offline, on the calibration run): the Lee et al. rule.
  Re-reference to the average, band-pass 8–25 Hz, take log mean power,
  z-score it, and mark z > 6 as bad. Also flag flat lines (variance below a
  floor) and saturation.
- **Online**: a rolling 2 s variance and flatline check per channel, flagged
  only. A channel failing for more than 5 s is dropped from the reference with
  a logged event.
- Never interpolate silently. If interpolation is enabled, precompute the
  spherical-spline matrix at calibration, which makes it one matrix multiply
  online.

**S3 Causal filter bank**
- Anti-alias low-pass, then integer decimation.
- Notch, then one band-pass per band.
- Second-order sections with state carried between chunks.
- MATLAB: `butter(...,'s')` or `zp2sos`, then `filter(b,a,x,zi)` per section
  (MATLAB's `sosfilt` has no state argument).
- Python: `scipy.signal.butter(output='sos')`, `sosfilt(sos, x, zi=zi)`,
  `sosfilt_zi`.
- Cost: O(channels × samples × sections). Negligible.

**S4 Re-reference**
- Common average over good channels, or Laplacian: channel minus the mean of
  its 4 neighbours.
- No state; a matrix multiply.

**S5 Feature stream** (4 Hz)
1. Square the band-limited signal.
2. Average within each 0.25 s block.
3. Take a trailing 3-block moving average.
4. Take log10.
5. Optionally subtract the trailing 25 s mean (circular buffer of 100 values per
   channel and band).

Optional branches:
- **Covariance**: 2 s ring buffer; tangent-space projection at the reference
  matrix estimated during calibration.
- **MRCP**: low-pass stream, decimated.

Stored state: ring buffers only.

**S6 Artifact features and gate**
- Per update: EEG peak-to-peak and RMS, EOG peak-to-peak, high-band (EMG)
  power.
- Gate: robust z against the **participant's own calibration data**, using
  continuous features only. The gate is unavailable if any z exceeds the
  threshold (default 5).
- Artifact features are **always logged**, because they are the inputs of the
  amplitude-only and EOG/EMG-only baselines.

**S7 Decoder**
- Training samples: one vector per trial per class, the mean feature vector
  over the class window. That avoids pseudo-replicating correlated 4 Hz
  samples.
- Model: standardize, then shrinkage LDA, then logistic calibration on
  calibration data.
- Applied to the running feature vector at every update.
- MATLAB: `fitcdiscr(...,'DiscrimType','linear','Gamma',γ)` with γ chosen by
  inner cross-validation, or a closed-form OAS solution (about 20 lines);
  `fitclinear` / `fitcsvm` as comparators.
- Python: `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`;
  `LogisticRegression`; `LinearSVC` as comparator.
- Optional population prior: pool the training participants, re-center each new
  participant on their own calibration mean, then fine-tune with
  `n_cal_per_class` trials.

**S8 Decision layer**
- **Cue-gated mode**: evaluate only from 0 to 4 s after each cue. Output INTENT
  once 2 of 3 consecutive updates exceed the threshold; otherwise REST at
  window end.
- **Asynchronous mode**: the same rule runs continuously, with a 2 s
  refractory period.
- In both modes, gate-unavailable updates count as REST and are logged as
  abstentions.
- Device output is **disabled by default**. Enabling it needs an explicit flag
  plus a watchdog that forces REST if the stream stalls for more than 0.5 s.

### 4.1 Latency budget (estimate; measure in M5)

| Component | Delay |
|---|---|
| LSL chunk (50 ms) | ≤ 50 ms |
| Band-pass group delay (4th-order, mu band) | about 50–80 ms |
| 0.25 s block plus trailing 0.75 s average | about 0.5 s mean |
| 2-of-3 accumulation at 4 Hz | 0.25–0.5 s |
| **Total after ERD onset** | **about 0.8–1.2 s** |

Shorter smoothing reduces delay but adds false positives. The trade-off is
reported as a curve, not tuned on test data.

---

## 5. Offline analysis and outputs

Everything below runs on `epochs.h5` and `predictions.csv`, i.e. on the
streamed features.

### 5.1 Data-quality report

- Grid of channels by runs showing kept and rejected (as in Lee et al. Fig. 4).
  Per-run artifact rate; line-noise ratio; gate abstention rate.
- Trials per class after rejection, per participant.
- Distribution of amplitude, EOG, and EMG features by class. If any of them
  separates the classes on its own, this panel is the first warning.

### 5.2 Topomaps

The montage needs at least 16–32 channels for useful topomaps. The decoder can
still use a sensorimotor subset; seven channels give only a sparse map.

| Figure | Content |
|---|---|
| `topo_erds_<band>_<class>.png` | ERD/S in dB, referenced to pre-cue (−1.0 to 0 s), in 0.5 s steps from −0.5 s to the end of the task. Rows: classes (intent, rest/no-go). Columns: time |
| `topo_contrast_<band>.png` | Intent minus rest, as t-values, with channels significant after cluster correction marked |
| `topo_patterns.png` | Decoder **activation patterns**, not weights (Haufe et al. 2014: pattern = feature covariance × weights). This shows which channels carry the signal |
| `topo_group.png` | Grand average across participants (left-handers mirrored) plus per-participant small multiples |

- MATLAB: EEGLAB `topoplot`. Python: `mne.viz.plot_topomap`.
- Use one diverging colour map centred at 0 dB, with the same limits across
  panels in a figure.

### 5.3 Time and time-frequency

- `erds_timecourse.png`: ERD/S over time at C3/Cz/C4 (or the strongest
  pattern channel), with participant-bootstrap 95% CIs.
- `tfr_<channel>.png`:
  - spectrogram, 1 s Hann window, 0.05 s hop, 4–40 Hz;
  - log, then ERD/S against the pre-cue reference;
  - mask showing significant clusters.
  - MATLAB `spectrogram`; Python `scipy.signal.spectrogram`.
- `decoding_timecourse.png`: AUROC per 0.25 s step, with cluster-permutation
  significance over time. Shown for interpretation, **not** to pick the
  reported time point.

### 5.4 Statistics

| Question | Method | MATLAB | Python |
|---|---|---|---|
| Does ERD/S differ from baseline (per channel and time)? | One-sample cluster permutation (sign-flip, 1,000 permutations) | Max-statistic / cluster permutation (about 40 lines), or FieldTrip if the lab uses it | `mne.stats.permutation_cluster_1samp_test` |
| Intent vs rest ERD/S contrast | Paired cluster permutation across participants | same | same |
| Does the decoder beat chance per participant? | Label permutation **within runs** (1,000 shuffles) plus the binomial chance level | custom loop; `binocdf` | custom loop; `scipy.stats.binomtest` |
| Primary endpoint | Participant-mean BA and AUROC; participant-bootstrap 95% CI; number above chance | `bootci`, `perfcurve` | `numpy` bootstrap; `sklearn.metrics.roc_auc_score` |
| Model A vs B, or model vs amplitude baseline | Paired per-participant difference: Wilcoxon signed-rank plus bootstrap CI | `signrank` | `scipy.stats.wilcoxon` |
| Several datasets or sessions | Mixed-effects model: AUROC ~ model + (1 \| participant) | `fitlme` | `statsmodels` `mixedlm` |
| Multiple tests | Benjamini-Hochberg false-discovery rate | `mafdr` or a manual version | `statsmodels.stats.multitest` |

Cross-validation:
- **Group by run** (leave-one-run-out) within a participant.
- **Group by participant** for population models.
- **Nested** whenever a hyperparameter (shrinkage, threshold) is chosen.
- Seeds are fixed and recorded.

### 5.5 Pseudo-online and real-time metrics

Produced by replaying files through S0–S8:

| Metric | Definition |
|---|---|
| True-positive rate | fraction of intent trials with INTENT inside the decision window |
| False activations per minute | INTENT decisions during idle/no-go segments ÷ minutes |
| Cue-only false activation | fraction of no-go cues answered with INTENT |
| Latency | time from cue (or EMG onset if recorded) to INTENT; median and IQR |
| Abstention rate | fraction of updates with the gate unavailable |
| Contingency accuracy | fraction of trials where feedback matched the true state (Biasiucci 2018) |
| Compute per update | median and 99th percentile in ms |

The required comparator is a "fire on every cue" rule: 100% true positives,
100% cue-only false activations.

### 5.6 Calibration and burden

- Reliability diagram and expected calibration error of P(intent).
- Calibration-burden curve: participant-mean AUROC against calibration trials
  per class (5, 10, 20, 40).

### 5.7 Run report

Every run auto-generates `REPORT.md` / `.html`:
- the manifest;
- the quality panel;
- the headline table (participant-level metrics with all baselines in adjacent
  rows);
- figures;
- the deviations-from-plan section.

MATLAB can also export a Live Script to HTML; Python can render a Jupyter
notebook or plain Markdown.

---

## 6. Implementation layout

### 6.1 MATLAB

```
eeg_intent_matlab/            git + MATLAB Project
  buildfile.m                 tasks: test, replay, analyze, report, online
  config/pipeline_config.json
  +ei/+io/        readEDF, readBDF, readXDF, lslInlet, lslOutlet, eventTable
  +ei/+stream/    Chunker, ChannelHealth, CausalFilterBank, Rereference,
                  BandPowerStream, ArtifactFeatures, Gate, DecisionLayer   (classdef, state as properties)
  +ei/+model/     trainLDA, calibrate, recenter, haufePatterns
  +ei/+analysis/  epochFeatures, erds, tfr, clusterPerm, bootstrapCI, chanceLevel, pseudoOnline
  +ei/+viz/       topoERDS, topoPatterns, timecourse, tfrPlot, qualityGrid
  +ei/+report/    writeManifest, writeReport
  apps/           OnlineMonitor.mlapp (optional live view: P(intent), gate, channel health)
  tests/          matlab.unittest
```

Toolboxes already installed: Signal Processing, Statistics and Machine
Learning, Parallel Computing, EEGLAB (input and `topoplot`). Add:
`liblsl-Matlab`. Optional: FieldTrip.

### 6.2 Python

```
eeg_intent_py/                git + pyproject.toml
  config/pipeline_config.json
  eeg_intent/io/        readers.py (mne/pyxdf for input only), lsl.py, events.py
  eeg_intent/stream/    chunker.py, health.py, filters.py, reref.py, bandpower.py,
                        artifacts.py, gate.py, decision.py      (dataclass state)
  eeg_intent/model/     lda.py, calibrate.py, recenter.py, patterns.py
  eeg_intent/analysis/  epochs.py, erds.py, tfr.py, stats.py, pseudo_online.py
  eeg_intent/viz/       topomaps.py, timecourse.py, tfr.py, quality.py
  eeg_intent/report/    manifest.py, report.py
  cli.py                eeg-intent replay | analyze | report | online
  tests/                pytest
```

Dependencies (about eight; pin versions):

- `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `h5py`, `pylsl`
- `mne`: input and topomaps only
- optional: `pyxdf`, `pyriemann`, `statsmodels`

### 6.3 Keeping the two implementations equivalent

- Same config file, event and window tables, and output schema.
- A **golden test set**: one short recording plus synthetic signals, with
  expected outputs checked in.
- A cross-language test compares `features.h5` and `predictions.csv` within a
  tolerance:
  - band power: 1e-6 relative;
  - P(intent): 1e-4;
  - decisions: identical.

---

## 7. Testing and validation

| Test | What it proves |
|---|---|
| Sine at 10 Hz plus noise: band power peaks in mu only | Filter bank and band power are correct |
| Chunked vs single-shot filtering give identical output | Filter state is carried correctly |
| **Causality test**: changing samples after time *t* never changes outputs at or before *t* | No look-ahead anywhere |
| **Replay equivalence**: file replayed through LSL equals offline replay | Online and offline paths are the same |
| Labels shuffled within runs: AUROC about 0.5 | No leakage through windows, normalization, or CV |
| Participant/run grouping: no group in both train and test | Split hygiene |
| Gate with a missing EOG channel: still available | The 2026 indicator bug cannot recur |
| Timing: 99th-percentile update time under budget for 32 and 64 channels | Real-time capable |
| MATLAB vs Python golden comparison | The two implementations agree |

---

## 8. Build plan (each milestone must pass before the next)

| Milestone | Deliverable | Acceptance |
|---|---|---|
| M0 Spec (1 wk) | `pipeline_config.json`, event and window tables, output schema, locked-cohort choice | Reviewed and committed before any analysis |
| M1 Streaming core (2 wk, per language) | S0–S5 plus replay; `features.h5` | Causality, chunking, and sine tests pass |
| M2 Offline analysis (2 wk) | Epochs, ERD/S, TFR, topomaps, quality report | Figures look right on PhysioNet; ERD over C3/C4 visible |
| M3 Decoder and statistics (2 wk) | S6–S7, grouped CV, baselines, permutation and bootstrap, burden curve | Shuffle test ≈ 0.5; baselines in every table |
| M4 Pseudo-online (1 wk) | S8 and pseudo-online metrics | FP/min, latency, and contingency reported on replayed data |
| M5 Real time (1–2 wk) | LSL in and out, live monitor, watchdog | Replay equivalence; update time within budget |
| M6 Equivalence and lock (1 wk) | Cross-language golden test; frozen config and model hashes | Both implementations agree; locked test run once |

The two implementations can be built in parallel after M0, or one after the
other. If one language is built first, it becomes the reference the other is
tested against.

---

## 9. What this deliberately leaves out

- Deep learning, multi-dataset fusion stacks, and left/right decoding. Add
  these only if the simple pipeline is beaten on validation data.
- Zero-phase filtering, centred smoothing, or whole-session normalization
  anywhere on the decision path.
- Reporting the best time point or best participant as a headline.
