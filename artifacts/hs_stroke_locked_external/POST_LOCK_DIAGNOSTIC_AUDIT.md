# HS-Stroke Post-Lock Diagnostic Audit

> This document explains the immutable locked result. It is post hoc, is not a
> second confirmatory analysis, and must not be used to replace the saved locked
> metrics.

## What Happened

- The corrected release supplied 278 sessions from 14 patients.
- Frozen recording QC accepted **172/278 (61.9%)** sessions.
- The hard window rules retained **26,483/46,764 (56.6%)** theoretical paired windows.
- The frozen availability gate accepted **0.0%** of retained windows.
- Consequently, both locked operating points issued zero commands: 0% sensitivity, 100% structured-rest specificity, and 50% binary balanced accuracy.

## Root Cause

The gate treated `auxiliary_available` and `csd_available` as continuous quality
features. Both had training median 1 and a clipped MAD scale of `1e-6`. HS-Stroke
has no auxiliary EOG/EMG channels and only 11 EEG channels, so both indicators are
0. Each becomes a **1,000,000-MAD** deviation, forcing every quality score above
the frozen threshold (301,511.3). This is a quality-gate design flaw exposed
by the new montage, not evidence that every HS window is physiologically unusable.

## Why Removing the Gate Is Not Enough

- Descriptive intent AUROC: **52.4%**.
- Numeric class-1/class-2 direction BA: **51.1%**.
- Ungated conservative sensitivity/specificity: **99.9% / 0.05%**.
- Ungated sensitivity-targeted sensitivity/specificity: **100.0% / 0.01%**.

The external probabilities are shifted high for both preparation and imagery.
Bypassing the gate would therefore replace total abstention with near-continuous
false activation, while discrimination remains near chance.

## Research Consequence

HS-Stroke is now an observed development/adaptation cohort. We can use it to fix
modality-aware quality gating, domain calibration, lesion-aware normalization, and
early-session patient adaptation, but those changes require a different untouched
dataset or a prospective cohort for confirmation.
