# WAY-EEG-GAL Download Audit

## Provenance

- Dataset: WAY-EEG-GAL grasp-and-lift EEG-EMG dataset.
- Paper: <https://www.nature.com/articles/sdata201447>.
- Official collection: <https://doi.org/10.6084/m9.figshare.c.988376.v2>.
- Download metadata: `download_manifest.json`.

## Verified Local Inventory

- Participants: **12 of 12**.
- Official participant ZIP archives: **12**, totaling **9.63 GiB**.
- Every archive matched Figshare's published byte count and MD5 before extraction.
- Windowed series files (`WS_*.mat`): **108**.
- Holistic series and EEG test files (`HS_*.mat`): **120**.
- Event/behavior summaries (`P*_AllLifts.mat`): **12**.
- Extraction markers: **12**.
- Local storage: approximately **20 GB** including archives and extracted files.

## Sample Structure Check

`participants/P01/WS_P1_S1.mat` loaded successfully with:

- 34 grasp-and-lift trials.
- 32 EEG channels sampled at 500 Hz.
- Five EMG channels sampled at 4,000 Hz.
- 45 kinematic, force, and derived channels sampled at 500 Hz.
- Explicit LED-on and LED-off times and object weight/surface metadata.

## Project Use

This is an overt, right-hand grasp-and-lift dataset rather than motor imagery.
Its primary value is privileged physiological supervision: EMG, hand velocity,
contact, load force, and lift-off can define motor phases more precisely than a
visual cue alone. A future adapter should train EEG-only premovement features
using these peripheral measurements as source-only labels, then evaluate the
student on the existing pure-EEG imagery and stroke cohorts.

The dataset must not be pooled as if its labels were left/right imagery, and a
higher EEG-EMG fusion result must not be described as improved cortical intent
detection without EEG-only and shuffled/peripheral-label controls.
