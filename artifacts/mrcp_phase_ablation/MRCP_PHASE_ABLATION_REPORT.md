# MRCP Phase Feature Ablation

## Scope

This is a development-only feature audit. It does not retrain or replace the frozen v4 model, and it does not inspect the `development_observed` or future `locked_external` cohorts.

## Controller-Validation Result

- MRCP amplitude balanced accuracy: **63.5%**.
- MRCP amplitude + phase balanced accuracy: **64.5%**.
- AUROC change: **+1.5 points**.
- Specificity change: **+0.5 points**.
- Decision: the phase candidate **did not pass the development advancement rule**.

A passing result only licenses future locked testing; it is not evidence of external or clinical benefit.
