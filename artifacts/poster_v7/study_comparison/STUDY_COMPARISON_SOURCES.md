# V7 Results Versus Current Studies

This file documents the sources and comparison limits for
`08_v7_results_vs_current_studies` and the portrait
`08b_v7_closest_studies_vertical` variant. Sources were checked on August 7,
2026.

## Current Project

The V7 patient-adapted conservative result is **73.6% balanced accuracy**
(participant-bootstrap 95% CI: **67.5-78.7%**), **53.5% intent sensitivity**,
**93.7% cue-locked rest specificity**, and **0.896 AUROC**. It represents
36,903 windows from 187 participants across three datasets. Thirty early
calibration windows were used per evaluation participant; later recordings
were used for evaluation.

This result is development observed. There is no locked external cohort, and
cue-locked specificity has not been converted into continuous false
activations per minute.

## Published Comparisons

| Ref. | Study | Reported result | Why it is relevant | Important difference from V7 |
|---|---|---|---|---|
| 1 | [Ang et al., 2011](https://pubmed.ncbi.nlm.nih.gov/22208123/) | 74% mean offline accuracy for stroke MI versus rest; 82% online in 11 returning patients | One of the closest historical motor-imagery-versus-rest studies | Patient-specific design; the independent online result came from a small returning subset |
| 2 | [Nagarajan et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38091617/) | 71.15% +/- 12.46% mean MI-versus-rest accuracy across 71 stroke patients after healthy-to-stroke domain adaptation | Closest modern transfer-learning comparison | Reports mean accuracy rather than V7's balanced accuracy and uses a different transfer/calibration contract |
| 3 | [Lopez-Larraz et al., 2025](https://doi.org/10.1186/s12984-025-01687-9) | 69.7% mean accuracy for ipsilesional EEG plus involved-muscle EMG over the full cohort; 73.8% among successful users | Directly evaluates attempted movement versus rest in severe paralysis | Multimodal EEG+EMG and within-patient block validation; not an EEG-only cross-dataset result |
| 4 | [Lu et al., HS-Stroke, 2026](https://www.nature.com/articles/s41597-026-07742-x) | Up to 82.65% left-versus-right MI accuracy | Current longitudinal chronic-stroke benchmark | Direction-only cross-trial evaluation in 14 patients; does not test rest rejection or unseen-patient transfer |
| 5 | [Niazi et al., 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9082379/) | 84.88% true-positive rate and 0.52 false positives/minute for cue-gated operation | Best direct model for V7's future online safety evaluation | Different endpoint and metric; online within-patient attempted ankle movement in nine participants |

## Defensible Poster Interpretation

> V7 achieved a cue-locked intent-detection result in the same broad range as
> representative stroke motor-intent studies while evaluating a larger,
> multi-dataset population with rapid patient adaptation and explicit rest
> specificity. It should not be described as outperforming the literature,
> because the published studies use different class definitions, metrics,
> calibration strategies, and validation splits. The next required comparison
> is a locked prospective or external evaluation reporting continuous false
> activations per minute and command latency.

## Regeneration

```bash
python scripts/44_create_v7_study_comparison_visual.py
```

The script exports a 300-DPI PNG and vector SVG/PDF versions.

The portrait `08b_v7_closest_studies_vertical` export preserves the original
comparison-card proportions and content while stacking the three closest
intent-versus-rest studies vertically. It uses the same references and
comparison limits as the full panel.
