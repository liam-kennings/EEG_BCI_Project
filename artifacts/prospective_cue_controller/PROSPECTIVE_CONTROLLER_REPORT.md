# Prospective Cue-Constrained EEG Controller

## Status

The v4 decoder and all five MRCP-FBCNet seed weights are copied and SHA-256 fingerprinted. The current result remains `development_observed`; no locked external cohort has been evaluated.

The controller is research-only and device output is disabled. The included timeline and metrics are a synthetic functional test, not participant evidence.

## Decision Path

`cue arms a 4 s opportunity -> frozen EEG decoder -> frozen OOD availability -> full-montage contamination veto -> two direction-consistent EEG confirmations -> command or explicit no-decision -> refractory period`

The cue never supplies the command direction. Unarmed EEG detections are logged as latent candidates but cannot issue a command.

## Required Confirmation

1. Fit only subject normalization and contamination limits on the labeled calibration block.
2. Keep the later cue-locked block and continuous no-control block untouched.
3. Verify the frozen hashes immediately before inference.
4. Report false commands/min, sensitivity, no-decisions, wrong direction, latency, OOD abstention, and contamination veto rate together.
5. Do not claim clinical utility until a prospective safety protocol and supervised hardware stop have been validated.

## Prepared Files

- `frozen_model/frozen_model_manifest.json`: immutable model provenance and hashes.
- `prospective_controller_config.json`: predeclared command policy.
- `locked_external_protocol.json`: split, acquisition, and endpoint contract.
- `hard_negative_manifest.template.json`: continuous no-control registration template.
- `probability_replay_template.csv`: decoder-output replay interface.
- `functional_demo_metrics.json`: synthetic state-machine verification.
- `01_cue_constrained_controller_demo.png`: presentation-ready functional diagram.
