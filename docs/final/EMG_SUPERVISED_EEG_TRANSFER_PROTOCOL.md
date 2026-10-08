# EMG-Supervised EEG Transfer Protocol

## Research question

Can synchronized EMG make execution labels precise enough to teach an EEG-only model a reusable cortical motor-intent representation, improving intent-versus-rest detection on held-out pure-EEG imagery or attempted-movement data?

This is a **learning using privileged information** experiment. EMG is available only in the source training cohort. It is unavailable to the target test cohort and to the deployed decoder.

## Model boundary

The experiment modifies the motor-intent gate only. It does not use right-fist execution to supervise a left-versus-right imagery direction head. Direction remains a separate target-EEG problem because the highest-priority MRCP EEG-EMG source contains right-hand execution only.

The implemented prediction API is:

```python
probability = model.predict_intent_probability(eeg_features)
```

There is no EMG inference argument.

## Source labeling

1. Detect task-muscle EMG onset independently for each movement trial.
2. Define positive EEG windows that end at or before EMG onset.
3. Define rest windows from pre-cue or verified no-movement periods.
4. Extract the same MRCP, SMR, PSD/Morlet, and spatial features used in the target EEG cache.
5. Train an EMG movement verifier using subject-grouped cross-validation.
6. Weight or reject source examples according to out-of-fold EMG confirmation confidence.
7. Robustly align accepted source EEG features to target-development EEG without inspecting target test data.

The code rejects a source cache if any positive EEG window ends after EMG onset. This is deliberately conservative because post-onset EEG is vulnerable to movement and muscle contamination.

## Required comparison

| Condition | Source information | Target inference | Purpose |
|---|---|---|---|
| Target-only baseline | None | EEG only | Fixed reference |
| Cue-labeled execution transfer | EEG plus protocol labels | EEG only | Tests whether more execution EEG alone helps |
| EMG-confirmed transfer | EEG plus out-of-fold EMG confirmation | EEG only | Tests the proposed physiological-label benefit |
| Shuffled-EMG control | EEG plus subject-safe shuffled confirmation | EEG only | Tests whether confirmation carries meaningful timing information |
| EMG-only diagnostic | EMG | Not applicable to pure-EEG target | Measures source confirmation quality, not deployment accuracy |

The claim is supported only if EMG-confirmed transfer improves over both the target-only and cue-labeled transfer controls on the same untouched target subjects.

## Splits

Source participants must be grouped by participant and session for EMG verification. Target participants retain the existing `development`, `controller_validation`, `development_observed`, and future `locked_external` roles.

Target development EEG is used to fit the target-only baseline and the transfer student. Controller EEG selects each model's threshold at the same false-activation budget. Development-observed EEG is evaluated once per experiment configuration. A later locked external cohort is required for a claim-bearing result.

## Endpoints

Primary comparison:

- Change in EEG-only intent sensitivity at matched controller false activation.

Required secondary results:

- Intent balanced accuracy.
- Specificity and missed-intent rate.
- AUROC and AUPRC.
- Per-dataset and per-participant results.
- EMG verifier cross-validated balanced accuracy and AUROC.
- Fraction of source windows accepted by EMG confirmation.
- Transfer gain with confidence intervals resampled by target participant.
- Continuous event sensitivity, latency, and false activations/minute when no-control recordings become available.

## Standardized caches

The source NPZ must contain:

```text
eeg_features
emg_features
labels
subject_keys
premovement_end_seconds
eeg_feature_names
emg_feature_names
dataset_id
```

The target NPZ must contain:

```text
eeg_features
labels
subject_keys
dataset_ids
roles
eeg_feature_names
```

`eeg_feature_names` must match exactly. This prevents a source and target representation mismatch from being silently accepted.

## Running the experiment

Run the complete adapter and transfer experiment with:

```bash
./EEG_BCI_Project_RUN_EMG_SUPERVISED_TRANSFER.command
```

After the source and target feature caches exist:

```bash
./EEG_BCI_Project_RUN_EMG_SUPERVISED_TRANSFER.command \
  --source-cache /path/to/mrcp_eeg_emg_source_features.npz \
  --target-cache /path/to/pure_eeg_target_features.npz
```

The launcher reuses prepared caches or creates them from the downloaded dataset when absent. The output includes metrics, the fitted EEG student, a Markdown report, a source-quality figure, an overall comparison figure, and a dataset/participant transfer figure.

## Implementation status

- Privileged-transfer estimator: implemented.
- Cross-fitted EMG confirmation: implemented.
- Premovement leakage guard: implemented.
- Robust source-to-target alignment: implemented using target development only.
- Matched target-only baseline: implemented.
- Pure-EEG observed-test reporting: implemented.
- MRCP EEG-EMG raw-data adapter: implemented for the downloaded 40-subject folder; 39 subjects have paired EEG and EMG.
- Current three-dataset target feature exporter: implemented with 144 exactly matched MRCP and fixed-SMR features.
- Subject-balanced source weighting and dataset/class-balanced target weighting: implemented.
- Transfer-weight selection: implemented using controller validation only.
- Dataset-specific and paired-participant reporting: implemented.
- Exploratory real transfer result: +0.3 pooled balanced-accuracy points; equal-participant mean +0.4 points (95% bootstrap CI -0.1 to +0.9).
- Cue-only and shuffled-EMG controls: still required before attributing any gain specifically to EMG supervision.
- Locked external confirmation: unavailable because the locked cohort remains empty.
