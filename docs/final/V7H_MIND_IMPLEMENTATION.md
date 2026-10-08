# V7-H MIND Implementation

## Purpose

V7-H tests whether synchronized fNIRS can improve safe binary motor-intent
detection beyond EEG alone. It preserves V7's deployment endpoint:
`REST` versus `INTENT`. Direction remains metadata and is not exposed as a
device command.

The first implementation uses MIND, a healthy-participant simultaneous
64-channel EEG and 51-channel fNIRS motor-imagery dataset. This is an internal
public-data benchmark, not stroke validation or a prospective clinical result.

## Download

The recommended download contains the preprocessed EEG and HbO/HbR derivatives
needed by the benchmark:

```bash
cd "/Users/LiamKennings_1/Documents/BME 473/EEG_BCI_Project"
bash EEG_BCI_Project_DOWNLOAD_MIND.sh
```

Useful options:

```bash
# Small adapter test
bash EEG_BCI_Project_DOWNLOAD_MIND.sh --subjects 3-7

# More simultaneous transfers
bash EEG_BCI_Project_DOWNLOAD_MIND.sh --workers 6

# Inspect the official manifest without downloading files
bash EEG_BCI_Project_DOWNLOAD_MIND.sh --metadata-only

# Complete raw, source, standardized, and derivative release (~93.1 GiB)
bash EEG_BCI_Project_DOWNLOAD_MIND.sh --profile full
```

The downloader resumes partial files with `curl`, records every selected URL in
`MIND/download_manifest.json`, and tracks completed files in
`MIND/.mind_download_state.json`. Running the same command again resumes rather
than restarting. Use is subject to the dataset's CC BY-NC 4.0 license.

## Run

```bash
bash EEG_BCI_Project_RUN_V7H_MIND.sh
```

For a quick end-to-end software check without real data:

```bash
bash EEG_BCI_Project_RUN_V7H_MIND.sh \
  --synthetic \
  --rebuild-cache \
  --output-dir /tmp/v7h_mind_smoke
```

The GUI exposes the same experiment as `V7-H hybrid EEG-fNIRS (MIND)`.

## Data Contract

- The current V3 derivative manifest provides 27 usable complete EEG/HbO/HbR
  pairs. `sub-02` is incomplete and `sub-13` is excluded because its released
  fNIRS derivative is documented as corrupted.
- Motor-imagery EEG uses the cue-locked 0-3 second interval and a -2 to 0 second
  baseline.
- fNIRS uses a slower 0-6 second response interval and a -5 to 0 second
  baseline.
- Independent REST windows come from the released rest recording rather than
  immediate post-imagery samples with residual hemodynamic activity.
- Participants are split into development, controller-validation, and locked
  internal-test roles.
- Test-patient recording 0 contributes at most 10 REST and 20 INTENT calibration
  windows. Only later recordings are scored.

## Representations

The EEG branch extracts MRCP preparation statistics, theta/mu/beta power and
baseline ratios, and four-band sensorimotor covariance. Seven channels shared
with V7 are used: FC3, FC4, C3, Cz, C4, CP3, and CP4.

The fNIRS branch extracts HbO and HbR mean change, slope, area, variability, and
late-minus-early response over 51 channels. A derivative-level response-quality
score remains available to the rejection policy.

## Decoder Modes

1. `EEG only`: calibrated electrical evidence and the V7-H safety threshold.
2. `fNIRS only`: calibrated hemodynamic evidence.
3. `Population fusion`: cross-fitted late fusion learned only from development
   participants.
4. `Patient-adapted V7-H`: population fusion plus recording-0 probability
   calibration; poor fNIRS quality falls back to EEG.
5. `Sequential confirmation`: an EEG proposal must be accompanied by usable
   fNIRS evidence; poor fNIRS quality abstains.

Operating thresholds are selected on controller-validation participants at the
requested rest-specificity target. They are not optimized on the locked test
participants.

## Outputs

`artifacts/v7h_mind/` contains:

- `V7H_MIND_REPORT.md`
- `v7h_mind_metrics.json`
- `v7h_mind_model.joblib`
- `v7h_mind_locked_internal_predictions.csv`
- `mind_v7h_features.npz`
- `01_v7h_model_comparison.png`
- `02_v7h_roc.png`
- `03_v7h_safety_sensitivity_frontier.png`
- `04_v7h_architecture.png`

## Interpretation Boundary

A hybrid improvement is meaningful only if it increases intent sensitivity at
the same rest-specificity target, or reduces false activation at matched
sensitivity, without an unacceptable delay or abstention burden. Cue-locked
window specificity is not false activations per minute. That endpoint still
requires continuous no-control recordings and execution of the full temporal
controller.

MIND contains healthy participants, so a positive result would justify further
hybrid work but would not establish generalization to stroke. The next planned
replication is a separate Shin EEG-NIRS adapter followed by frozen cross-dataset
testing and, eventually, prospective healthy and stroke cohorts.
