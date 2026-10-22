# eeg_intent_matlab

MATLAB implementation of the lightweight motor-intent pipeline described in
[`../docs/final/LIGHTWEIGHT_PIPELINE_PLAN.md`](../docs/final/LIGHTWEIGHT_PIPELINE_PLAN.md).

Every stage is causal and keeps its own state. Recorded files are replayed
through the same code that a live stream would use, so offline results are
achievable online.

## Status

| Milestone | State |
|---|---|
| M0 shared spec | `config/pipeline_config.json` |
| M1 streaming core | **Done**: causal filter bank, re-referencing, block band power with drift compensation, artifact summaries, offline replay |
| M2 offline analysis | **Started**: channel quality, epoching, ERD/S, topomaps, time courses |
| M3 decoder and statistics | Not started |
| M4 pseudo-online metrics | Not started |
| M5 real time (LSL) | Not started; needs `liblsl-Matlab` |
| M6 MATLAB/Python equivalence | Not started |

## Requirements

- MATLAB R2024a or later. Built and tested on R2026b.
- Signal Processing Toolbox (`butter`, `zp2sos`, `edfread`).
- Statistics and Machine Learning Toolbox (`prctile`, `zscore`).
- EEGLAB, for electrode locations and `topoplot` only.

## Quick start

From this folder in MATLAB:

```matlab
buildtool                 % Code Analyzer + 17 unit tests
buildtool demo            % PhysioNet motor-imagery demo (20 participants, ~40 s)
```

Or use the pieces directly:

```matlab
config = ei.config.loadConfig();
recording = ei.io.readEdfRecording(ei.io.findPhysionetRun(dataRoot, 1, 4));
quality = ei.qc.findBadChannels(recording.Data, recording.Fs, config);
stream = ei.pipeline.replayRecording(recording, config, IsGood=~quality.IsBad);
events = ei.io.physionetEvents(recording.Annotations, 4);
epochs = ei.analysis.epochFeatures(stream, events, ["cue_left", "cue_right"]);
erds = ei.analysis.computeErds(epochs, [-1, 0]);
```

The online path is the same object, fed chunk by chunk:

```matlab
pipeline = ei.pipeline.StreamingPipeline(config, fs, labels, IsGood=isGood);
rows = pipeline.step(chunk);   % chunk is [samples x channels] in microvolts
```

## Layout

| Package | Contents |
|---|---|
| `+ei/+config` | `loadConfig`: reads the shared JSON and records its SHA-256 |
| `+ei/+io` | EDF reader, label normalization, PhysioNet events, electrode locations (EEGLAB), HDF5 feature export |
| `+ei/+dsp` | `SosFilter` (stateful causal second-order sections), `designButterworth` |
| `+ei/+qc` | Calibration-time bad-channel and bad-run rules |
| `+ei/+stream` | `rereferenceMatrix` (average / Laplacian), `BandPowerStream` (0.25 s blocks, trailing 0.75 s smoothing, log10, 25 s drift compensation) |
| `+ei/+pipeline` | `StreamingPipeline` (the online object) and `replayRecording` (offline) |
| `+ei/+analysis` | `epochFeatures`, `computeErds` |
| `+ei/+viz` | Topomap grid, ERD/S time courses, channel-quality grid, diverging colormap |
| `demos/` | `runPhysionetDemo` |
| `tests/` | `matlab.unittest` suites |

## Design decisions and deviations from the plan

- **Native sampling rate.** The plan proposed resampling to 200 Hz, but
  PhysioNet is 160 Hz, which would mean upsampling, and streaming fractional
  resampling adds complexity. Filters run at the recording's own rate.
  Feature blocks are defined in samples (block *k* ends at
  `floor(k*0.25*fs)`), so any rate works, including 62.5-sample blocks at
  250 Hz.
- **Robust z for bad channels.** With a plain z-score (Lee et al. 2022), one
  outlier among *n* channels cannot score above (n−1)/√n. That is 5.4 for 31
  channels, so "z > 6" never fires on a 32-channel cap. The default uses
  median/MAD. `channel_health.robust_z = false` restores the original rule,
  and a unit test documents the ceiling.
- **Feature time stamps.** Features are stamped at the end of their causal
  window. A feature at +1.0 s summarizes about +0.25 to +1.0 s, so ERD
  appears roughly 0.4–0.5 s later than with centred smoothing.
- **EEGLAB options are character arrays.** `readlocs` and `topoplot` parse
  options with `strcmp`, so string options break them.

## Checks so far

| Check | Result |
|---|---|
| Unit tests | 17 pass. They cover chunked vs single-shot filtering, causality (future samples never change past features), replay equivalence, fractional block timing, hand-calculated smoothing/log, drift removal, bad-channel exclusion from the reference, the bad-channel rules, labels, events, and epoching/ERD |
| Code Analyzer | 0 issues |
| PhysioNet demo, 20 participants | Contralateral mu ERD over C3 for right-hand and C4 for left-hand imagery (about −1 to −1.5 dB) |
| Speed | 64 channels: 0.12 ms median and 1.4 ms worst-case 99th percentile per 50 ms chunk |
