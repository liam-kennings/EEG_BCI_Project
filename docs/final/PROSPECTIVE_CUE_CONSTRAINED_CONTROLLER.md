# Prospective Cue-Constrained Motor-Intent Stage

## Purpose

This stage converts the best current retrospective decoder into a protected
prospective experiment. It does not claim that the system is clinically ready.
The current v4 result remains development-observed because no untouched
external or prospectively collected cohort has been evaluated.

## Frozen Starting Point

The selected v4 bundle combines the five-seed MRCP-FBCNet intent output with
Riemannian and SMR branches, calibrated domain thresholds, patient adaptation,
and the existing OOD/quality gate. The preparation command copies the bundle
and all five neural seed files into `artifacts/prospective_cue_controller/`,
records their SHA-256 hashes, and verifies those hashes before use.

No model weights, preprocessing choices, fusion weights, calibration models,
or thresholds may change after the locked evaluation data are opened. A hash
mismatch invalidates the frozen evaluation.

## Command Logic

```text
visual/audio cue
    -> opens a 4 s command opportunity only
    -> frozen v4 EEG intent and direction probabilities
    -> frozen OOD availability gate
    -> full-montage contamination veto
    -> two consecutive, direction-consistent EEG confirmations
    -> command or explicit no-decision
    -> 2 s refractory period
```

The cue does not provide the decoded direction and cannot issue a command.
High EEG evidence outside an armed opportunity is logged as a latent candidate
but suppressed. This separates cue timing assistance from EEG motor-intent
evidence.

## Full-Montage Veto

The prospective contamination veto uses channels outside the seven-channel
motor decoder. It measures global peak-to-peak amplitude, derivative RMS,
frontal low-frequency dominance, peripheral high-frequency dominance, and the
fraction of channel-amplitude outliers. Limits are fitted only on clean
calibration/rest windows with robust statistics.

This veto complements rather than replaces the frozen v4 OOD gate. EOG and EMG
remain validation channels and are never decoder inputs.

## Required Prospective Blocks

1. A short labeled calibration block with left, right, rest, and no-go trials.
2. A later locked cue-locked evaluation block.
3. Continuous natural-idle/no-control recording with periodic sham arming
   opportunities.
4. A documented artifact challenge block containing blinks, jaw activity,
   swallowing, and posture changes.

The study should synchronize full-montage EEG, vertical and horizontal EOG,
bilateral task-muscle EMG, force or grip, limb IMU, and a hardware trigger or
photodiode. Device output remains disabled until separate safety validation and
a supervised hardware stop are available.

## Required Endpoints

- Issued false commands per minute on continuous no-control data.
- Latent unarmed EEG candidates per minute.
- Correct motor-command sensitivity and motor no-decision rate.
- Wrong-direction command rate.
- Rest/no-go specificity.
- Median and 95th-percentile command latency.
- OOD abstention and contamination-veto rates.

These endpoints must be reported together. Low false activation is not useful
if it is obtained only by abstaining from nearly every true intent.

## MRCP Phase Ablation

An explicit phase candidate was evaluated without opening the
`development_observed` or future `locked_external` roles. The feature contains
Hilbert unit-phase cosine, sine, phase concentration, phase slope, and C3-C4
phase difference in predeclared 0.5-second MRCP bins.

On controller-validation subjects:

| Development candidate | Balanced accuracy | Sensitivity | Specificity | AUROC | Conditional direction BA |
|---|---:|---:|---:|---:|---:|
| MRCP amplitude | 63.5% | 39.7% | 87.4% | 74.1% | 64.3% |
| MRCP amplitude + phase | 64.5% | 41.1% | 87.9% | 75.6% | 61.3% |

The candidate improved intent discrimination but missed the predeclared
one-point balanced-accuracy gate by 0.04 point and reduced direction balance.
It therefore remains a future candidate and is not integrated into frozen v4.
The advancement rule was not changed after observing the result.

## Run It

```bash
cd EEG_BCI_Project
./EEG_BCI_Project_PREPARE_PROSPECTIVE_STUDY.command
```

To verify the frozen files later:

```bash
python3 scripts/40_prepare_prospective_controller.py --verify-only
```

To replay causal decoder probabilities:

```bash
python3 scripts/40_prepare_prospective_controller.py \
  --replay-probabilities /path/to/probability_stream.csv
```

The GUI's **Prospective Study** export button now creates the same frozen
controller package in addition to the acquisition and research contracts.

## Interpretation

The implementation closes several methodological gaps: model freezing,
cue-versus-EEG role separation, explicit no-decisions, contamination vetoing,
continuous-no-control planning, and clinically meaningful safety metrics. It
does not close the central evidence gap. The next claim-bearing result must
come from a truly untouched cohort or a prospective later session acquired
under the locked protocol.
