# Target-Label-Only Paired CNN Five-Seed Result

## Contract

- Gradient training and labels use only PhysioNet EEGMMIDB, OpenBMI, and Liu EEG development participants.
- The decoder starts from random weights; WAY labels, EMG, kinematics, force, and peripheral supervision are not used.
- For the original paired SSL ablation, branch-normalization scales are shared with the unlabeled WAY source condition. This is a matched transfer control, not a strict target-only preprocessing baseline.
- Controller-validation participants select dataset-specific thresholds; development-observed participants provide the reported result.
- Seeds 42-46 change neural initialization, dropout, and batch randomness while all data and training settings remain fixed.

## Overall Result

Across five seeds, the paired random-initialization control reached **53.49% balanced accuracy** (SD **0.60**, 95% seed-level t interval **52.74 to 54.23**), **15.60% sensitivity**, **91.37% specificity**, **0.610 AUROC**, and **0.593 AUPRC**.

| Seed | Balanced accuracy | Sensitivity | Specificity | AUROC |
|---:|---:|---:|---:|---:|
| 42 | 53.06% | 14.81% | 91.30% | 0.614 |
| 43 | 53.43% | 14.47% | 92.39% | 0.612 |
| 44 | 54.25% | 16.19% | 92.31% | 0.619 |
| 45 | 53.92% | 17.64% | 90.19% | 0.613 |
| 46 | 52.78% | 14.92% | 90.64% | 0.591 |

## Dataset Stability

| Dataset | Mean BA | SD | Mean sensitivity | Mean specificity | Mean AUROC |
|---|---:|---:|---:|---:|---:|
| Liu stroke | 51.29% | 1.41 | 10.45% | 92.12% | 0.527 |
| OpenBMI | 54.38% | 0.78 | 17.73% | 91.02% | 0.630 |
| PhysioNet | 52.20% | 0.56 | 12.50% | 91.91% | 0.593 |

## Interpretation

The paired random-initialization control is reproducible but remains close to chance at the selected low-false-activation operating point. WAY EEG-only pretraining raises mean balanced accuracy from **53.49%** to **56.29%** relative to this normalization-matched control. A later strict target-development-only normalization audit reached **60.63%** with the same Conv1D family, showing that source-derived scaling was a material confound. The Liu result remains unstable, reinforcing the need for stroke-specific calibration and domain adaptation.
