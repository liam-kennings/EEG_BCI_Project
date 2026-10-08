# Calibrated Translational Motor-Intent Pipeline v3

## Purpose

This implementation converts the July 2026 full-project audit into a runnable
retrospective workflow. It improves signal validity, domain handling, split
governance, and safety reporting without claiming that public cue-locked data
are equivalent to a prospective clinical experiment.

The primary command is:

```bash
python3 scripts/30_translational_motor_intent_v3.py
```

The desktop entry point is:

```bash
python3 scripts/22_bci_pipeline_gui.py
```

## Audit Recommendations Implemented

### 1. Reset the confirmatory split

The previous stroke final cohort is now labeled:

```text
development_observed
```

The new split vocabulary is:

```text
development
controller_validation
development_observed
locked_external
```

`locked_external` remains empty. The v3 report therefore does not emit a
claim-bearing final accuracy. A new external subject/session cohort is still
required for confirmation.

The machine-readable decision record is written to:

```text
research_contract.json
RESEARCH_CONTRACT.md
subject_roles.json
```

### 2. Full-montage QC before channel reduction

Each native recording is processed in this order:

1. Preserve EEG, EOG, and EMG channel roles.
2. Load the complete native EEG montage.
3. Check finite data, flat channels, RMS outliers, high-frequency contamination,
   line-noise ratio, and channel correlation.
4. Reject recordings with more than 25% bad EEG channels.
5. Repair bad channels before referencing.
6. Use spherical-spline interpolation when valid positions are available.
7. Fall back to the median of good channels if spline interpolation is
   numerically unstable or positions are unavailable.
8. Apply common-average reference after repair.
9. Resample to 100 Hz.
10. Select the common motor channels only after native-montage processing.

The common decoder channels remain:

```text
FC3 FC4 C3 Cz C4 CP3 CP4
```

This preserves cross-dataset compatibility without using the seven-channel
intersection to perform the original cleaning.

### 3. Parallel MRCP, SMR, and quality branches

The new preprocessing output contains:

| Branch | Band | Role |
|---|---:|---|
| MRCP | 0.1-4 Hz | Slow cortical trajectory, slope, extrema, and area |
| SMR-CAR | 7-35 Hz | Mu/beta spectral and covariance representation |
| SMR-CSD | 7-35 Hz | Optional full-montage surface-Laplacian representation |
| Quality | 1-45 Hz | Availability and artifact/OOD gate |
| Auxiliary | Native EOG/EMG | Validation and abstention evidence only |

The MRCP branch is cue-aligned in the current public datasets. It must therefore
be interpreted as a slow cortical response candidate, not proof of a
pre-movement readiness potential.

SMR features include:

- Fixed mu-low, mu-high, beta-low, and beta-high temporal power
- Participant-specific 8-14 Hz center estimated from calibration rows only
- C3/C4 individualized-mu contrast
- Four-band regularized covariance
- Riemannian tangent-space representation

### 4. Decoder anchors and model ladder

The controlled comparison contains:

- FBCSP with hierarchical intent and direction heads
- Riemannian filter-bank covariance
- MRCP-only domain-aware decoder
- SMR-only domain-aware decoder
- The original equal-weight late fusion as a fixed reference
- Controller-selected stage-specific fusion for intent and direction

Every late-fusion branch uses regularized logistic probabilities. The model
fits:

- One shared population head
- One dataset-specific head when that dataset contains the required classes
- An equal dataset-weighted training objective

The shared and dataset-specific probabilities are blended at prediction time.
The calibrated controller then:

1. Selects intent-branch and direction-branch weights on a controller subset.
2. Fits per-dataset Platt transforms on that subset.
3. Uses a disjoint controller subset to retain or shrink each transform toward
   the identity mapping according to Brier score.
4. Selects a conservative threshold and a separate 60%-sensitivity target for
   each dataset.

EEGNet remains available elsewhere in the project but is intentionally not the
primary v3 model. Deep or self-supervised models should be added only after this
data and evaluation contract is stable.

### 5. Continuous no-control support

The current public negative classes remain explicitly labeled:

| Dataset | Negative subtype |
|---|---|
| PhysioNet | `protocol_rest` |
| OpenBMI | `post_task_rest` |
| Liu | `protocol_break` |

These rows are not called unrestricted idle.

The GUI can register saved continuous no-control recordings in:

```text
hard_negative_manifest.json
```

The v3 builder extracts overlapping two-second windows and labels them:

```text
continuous_no_control
```

When a manifest is present, threshold selection can use confirmed command
events and a false-activations-per-minute budget. Without one, the report says:

```text
not_evaluated_no_continuous_negative_recording
```

It does not multiply a trial false-positive fraction into a deployment claim.

### 6. Domain and patient adaptation

Dataset identity remains explicit throughout the pipeline. PhysioNet, OpenBMI,
and Liu do not become anonymous pooled rows.

The training process uses:

- Equal aggregate weight per dataset
- Equal aggregate weight for intent and no-intent
- Shared population heads
- Dataset-specific heads
- Calibration-only individualized mu estimates
- Early patient recordings for local heads
- Later patient recordings for evaluation
- Per-dataset probability calibration and operating thresholds
- Subject-level bootstrap confidence intervals overall and by dataset

Patient-calibration candidates use 2, 5, or 10 trials per class and local/global
blend weights of 0.25, 0.50, or 0.75. Controller patients choose one
configuration per dataset before it is applied to development-observed
patients. This calibration is now evaluated for PhysioNet, OpenBMI, and Liu,
not only the stroke cohort.

### 7. Safety-centered endpoints

Every model reports:

- Intent balanced accuracy
- Intent sensitivity
- Intent specificity
- False activation per window
- Missed-intent rate
- Available and abstained fractions
- AUROC and AUPRC
- Brier score
- Expected calibration error
- Conditional direction accuracy
- Subject-bootstrap 95% confidence intervals

The controller selects the operating threshold. If explicit continuous
no-control data exist, the primary target is:

```text
<=1 false activation/minute
>=60% intent sensitivity
<=2.5 seconds median latency
```

Latency requires a continuous event stream and is not inferred from independent
trial windows.

### Recording exclusion and missing-data handling

Native-montage QC is intentionally completed before referencing. A recording
with more than 25% bad EEG channels is not repaired aggressively or allowed to
stop the entire experiment. Instead, the pipeline:

1. Excludes the complete subject so calibration and evaluation rows remain
   paired.
2. Continues with the remaining subjects.
3. Stores the bad-channel list, QC values, source path, and reason in
   `translational_windows_v3.qc.json`.
4. Includes the exclusion summary in `translational_v3_metrics.json` and
   `TRANSLATIONAL_V3_REPORT.md`.
5. Stops before training only when a core dataset has fewer than five complete
   usable subjects for the disjoint split roles.

Missing PhysioNet runs, incomplete OpenBMI subject recordings, and missing Liu
recordings or participant metadata follow the same subject-level exclusion
policy.

## Future Acquisition Preparation

The GUI contains a **Study Preparation** tab that exports a disabled prospective
protocol template with:

- Warning duration
- Randomized instruction delay
- Randomized go delay
- Imagery/attempt duration
- Rest duration
- No-go fraction
- Required EOG, bilateral EMG, force, IMU, and hardware timing channels

The **Live Device** tab remains acquisition/audit-only. It does not issue
decoder or prosthetic commands.

This prepares the software contract for a future study without implying that
the prospective validation has happened.

## Running the Pipeline

### GUI

```bash
python3 scripts/22_bci_pipeline_gui.py
```

Choose:

```text
Dataset Experiments
Workflow: Translational v3 calibrated (recommended)
```

Select the PhysioNet, OpenBMI, and Liu roots. A no-control manifest is optional.
The GUI also exposes the target intent sensitivity, number of temporal
confirmations, and refractory period. The temporal settings are evaluated only
when explicit continuous no-control recordings are registered.

The recommended workflow writes to:

```text
artifacts/gui_experiments/translational_v3_calibrated
```

If the original `translational_v3/translational_windows_v3.npz` cache exists,
the GUI reuses it without modifying the original report or figures. Selecting
**Rebuild cache** creates a new cache in the calibrated output folder.

### Build-only smoke test

This verifies all native adapters without fitting models:

```bash
python3 scripts/30_translational_motor_intent_v3.py \
  --physionet-dir files/eegmmidb/1.0.0 \
  --openbmi-dir files/external/openbmi/nm000338 \
  --liu-dir "Liu acute-stroke dataset" \
  --max-subjects-per-dataset 1 \
  --disable-csd \
  --build-only
```

### Small end-to-end development test

At least five subjects per core dataset are required for non-empty development,
controller, and development-observed roles:

```bash
python3 scripts/30_translational_motor_intent_v3.py \
  --max-subjects-per-dataset 5 \
  --disable-csd
```

### Full retrospective experiment

```bash
python3 scripts/30_translational_motor_intent_v3.py \
  --physionet-dir files/eegmmidb/1.0.0 \
  --openbmi-dir files/external/openbmi/nm000338 \
  --liu-dir "Liu acute-stroke dataset" \
  --rebuild-cache
```

This uses every locally complete subject and enables CSD. It can take
substantially longer than the smoke run because OpenBMI is recorded at 1000 Hz.

### Causal preprocessing comparison

```bash
python3 scripts/30_translational_motor_intent_v3.py \
  --causal \
  --rebuild-cache
```

Use the causal result for deployment-oriented claims. The zero-phase profile is
for retrospective physiological characterization.

### Continuous no-control comparison

```bash
python3 scripts/30_translational_motor_intent_v3.py \
  --hard-negative-manifest artifacts/translational_motor_intent_v3/hard_negative_manifest.json
```

No-control subjects remain isolated by role. Small optional no-control sets can
select a controller threshold but are not called a locked final safety test.

## Generated Outputs

```text
TRANSLATIONAL_V3_REPORT.md
translational_v3_metrics.json
translational_v3_models.joblib
translational_windows_v3.npz
translational_windows_v3.qc.json
research_contract.json
RESEARCH_CONTRACT.md
subject_roles.json
01_decoder_comparison.png
02_safety_sensitivity_frontier.png
03_patient_calibration_burden.png
04_physiology_audit.png
05_quality_gate_availability.png
06_domain_probability_calibration.png
07_operating_point_comparison.png
```

## Current Full-Cohort Result

The calibrated validation reused the 36,903-window native-montage cache:

- Subjects: **187** across PhysioNet, OpenBMI, and Liu
- Accepted recordings: **492**
- QC exclusions: **1** (`liu2024:S013`)
- Legacy late-fusion BA: **60.1%**
- Calibrated conservative: **59.9% BA**, **23.5% sensitivity**, **96.3% specificity**
- Population sensitivity-targeted: **70.7% BA**, **61.4% sensitivity**, **79.9% specificity**
- Patient-adapted conservative: **67.9% BA**, **41.3% sensitivity**, **94.5% specificity**
- Patient-adapted sensitivity-targeted: **68.0% BA**, **46.1% sensitivity**, **89.9% specificity**
- Subject-level bootstrap resamples: **1,000**
- Automated tests: **48 passed**
- GUI backend check: **passed**

The patient-adapted conservative operating point is the preferred current
cue-locked result. It recovers more intent than the population conservative
gate while preserving high specificity. The population sensitivity-targeted
point is an exploratory tradeoff, not a safe command policy: held-out Liu
specificity falls to **38.7%**.

The report, model bundle, metrics, and all seven figures generated
successfully.

## Important Interpretation

The MRCP branch contains strong cue-locked information. This may represent a
mixture of slow motor-related, sensory, anticipatory, and protocol-timing
activity. It should not be described as a pure motor-readiness potential until
randomized cue timing, catch trials, and auxiliary movement validation are
available.

The next acquisition action is to collect continuous no-control and no-go data
so temporal confirmation can be assessed in false activations per minute. After
that policy is frozen, a new external subject/session cohort can populate
`locked_external` for the first claim-bearing evaluation.
