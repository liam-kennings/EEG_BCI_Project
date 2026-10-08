# WAY EEG-Only SSL Five-Seed Stability

## Fixed Contract

Neural seeds **42, 43, 44, 45, 46** change only initialization, masking, dropout, and batch randomness. Source windows, target rows, participant roles, architecture, epochs, sample weights, controller thresholding, and development-observed evaluation are fixed.

## Seed Results

| Seed | Random CNN BA | WAY SSL BA | BA change | Sensitivity change | Specificity change |
|---:|---:|---:|---:|---:|---:|
| 42 | 53.06% | 56.13% | +3.07 points | +3.57 points | +2.57 points |
| 43 | 53.43% | 56.68% | +3.25 points | +5.42 points | +1.09 points |
| 44 | 54.25% | 56.02% | +1.77 points | +4.42 points | -0.87 points |
| 45 | 53.92% | 56.05% | +2.13 points | +2.54 points | +1.72 points |
| 46 | 52.78% | 56.59% | +3.81 points | +3.99 points | +3.63 points |

## Aggregate Result

The paired SSL effect was **+2.81 balanced-accuracy points** across seeds (SD **0.84**, 95% t interval **+1.77 to +3.85**).
After averaging each participant across all five seeds, the equal-participant effect was **+2.60 points** (participant-bootstrap 95% CI **+1.48 to +3.79**); **78.9%** improved.
The five-seed SSL mean was **56.29% BA**, **19.59% sensitivity**, and **93.00% specificity**. Relative to the fixed statistical target-only baseline, this is **+3.47 BA points**, **+9.01 sensitivity points**, and **-2.07 specificity points**.

| Dataset | Mean BA change | Positive seeds |
|---|---:|---:|
| Liu stroke | +0.16 points | 60% |
| OpenBMI | +2.91 points | 100% |
| PhysioNet | +3.64 points | 100% |

## Predeclared Check

Overall criteria met: **yes**.

- Positive BA effect in at least four of five seeds: **yes**.
- Positive mean BA effect with seed-level 95% t interval above zero: **yes**.
- Mean specificity erosion no worse than one point: **yes**.
- Positive mean effect in both OpenBMI and PhysioNet: **yes**.

## Interpretation Boundary

This confirms or rejects robustness to neural-training randomness only. The same development-observed participants are reused for every seed, so this is not a new external patient validation and does not replace continuous no-control testing.
