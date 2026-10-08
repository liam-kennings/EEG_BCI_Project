# Translational Motor-Intent Research Contract

## Evaluation Status

- Current stroke test status: `development_observed`.
- Locked external test available: `False`.
- Primary endpoint: event sensitivity at a fixed continuous false-activation budget.
- Operating point: <=1 false activation/minute, >=60% sensitivity, <=2.5 s median latency.
- Confirmation rule: A claim-bearing result requires a never-inspected subject/session or external dataset after every preprocessing, feature, model, fusion, and threshold choice.

## Event Semantics

| Dataset | Intent | Negative state | Subtype | Continuous idle | Match |
|---|---|---|---|---:|---|
| physionet | T1/T2 cue-locked unilateral fist imagery | T0 intervals inside repetitive imagery runs | `protocol_rest` | no | low |
| openbmi | left/right hand imagery after a visual instruction | fixed post-imagery interval from the same trial | `post_task_rest` | no | low |
| liu2024 | left/right hand-grip imagery during protocol stage 2 | protocol break/rest during stage 3 | `protocol_break` | no | moderate |
| hard_negative | none | continuous no-control EEG containing ordinary idle and documented artifacts | `continuous_no_control` | yes | high |

## Prospective Acquisition

- Mode: `preparation_only`.
- Enabled: `False`.
- Safety: Configuration only. No clinical command or autonomous device output is enabled.

This file is a decision and provenance record. It does not make the retrospective datasets equivalent to a prospective clinical recording.
