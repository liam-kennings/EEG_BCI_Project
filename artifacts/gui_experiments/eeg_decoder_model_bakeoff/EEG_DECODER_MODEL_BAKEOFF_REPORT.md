# EEG Decoder Model Bakeoff

## Contract

All neural candidates use seeds 42-46, identical target development rows, class/dataset weights, epochs, batch order, and controller-derived dataset thresholds. Target-only models estimate normalization from development subjects only. The WAY SSL model is an explicitly labeled transfer condition. Development-observed subjects are reused for exploratory comparison and are not a new locked clinical cohort.

## Overall Results

| Model | BA mean (SD) | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | Train time/seed | CPU ms/256 windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ShallowConvNet | 62.49% (1.27) | 30.32% | 94.67% | 0.779 | 0.762 | 0.200 | 17162 | 8.2 s | 0.38 |
| MRCP + FBCNet | 62.43% (0.85) | 31.06% | 93.80% | 0.805 | 0.793 | 0.182 | 18166 | 13.3 s | 0.99 |
| EEGNet | 58.97% (0.50) | 24.43% | 93.52% | 0.765 | 0.732 | 0.201 | 9338 | 9.2 s | 0.53 |
| Conv1D target-only | 58.55% (4.66) | 22.09% | 95.01% | 0.748 | 0.729 | 0.237 | 75890 | 9.2 s | 0.93 |
| MRCP/SMR/Riemann fusion | 58.51% (0.00) | 22.22% | 94.80% | 0.780 | 0.767 | 0.206 | deterministic reference | not retrained | not benchmarked |
| FBCNet | 57.08% (1.03) | 20.38% | 93.77% | 0.684 | 0.677 | 0.226 | 2314 | 5.2 s | 0.21 |
| Conv1D + WAY SSL | 56.29% (0.32) | 19.59% | 93.00% | 0.683 | 0.662 | 0.235 | 75890 | not retrained | 0.62 |
| Riemannian tangent | 55.86% (0.00) | 19.36% | 92.36% | 0.693 | 0.673 | 0.222 | deterministic reference | not retrained | not benchmarked |
| Statistical target-only | 52.82% (0.00) | 10.58% | 95.07% | 0.683 | 0.645 | 0.227 | deterministic reference | not retrained | not benchmarked |

## Neural Advancement Check

| Candidate | BA change vs target Conv1D | 95% seed interval | Specificity change | All checks met |
|---|---:|---:|---:|---:|
| Conv1D + WAY SSL | -2.26 points | -8.24 to +3.73 | -2.01 points | no |
| EEGNet | +0.42 points | -5.17 to +6.02 | -1.50 points | no |
| ShallowConvNet | +3.94 points | -1.30 to +9.19 | -0.34 points | no |
| FBCNet | -1.47 points | -7.11 to +4.16 | -1.24 points | no |
| MRCP + FBCNet | +3.88 points | -1.31 to +9.07 | -1.21 points | no |

## Dataset Results

| Model | Liu stroke BA | OpenBMI BA | PhysioNet BA |
|---|---:|---:|---:|
| ShallowConvNet | 50.49% | 67.59% | 54.91% |
| MRCP + FBCNet | 53.99% | 64.78% | 60.09% |
| EEGNet | 52.05% | 59.65% | 60.12% |
| Conv1D target-only | 51.86% | 60.93% | 55.42% |
| MRCP/SMR/Riemann fusion | 57.68% | 58.29% | 59.38% |
| FBCNet | 62.07% | 58.87% | 50.64% |
| Conv1D + WAY SSL | 51.45% | 57.29% | 55.84% |
| Riemannian tangent | 50.86% | 57.33% | 54.33% |
| Statistical target-only | 54.85% | 52.25% | 53.36% |

## Equal-Participant Analysis

| Model | Participant-mean BA | Change vs target Conv1D | 95% participant-bootstrap interval | Participants improved |
|---|---:|---:|---:|---:|
| ShallowConvNet | 55.72% | +0.39 points | -1.37 to +2.40 | 34.2% |
| MRCP + FBCNet | 59.21% | +3.88 points | +2.37 to +5.49 | 86.8% |
| EEGNet | 57.91% | +2.57 points | +0.84 to +4.45 | 57.9% |
| Conv1D target-only | 55.33% | +0.00 points | +0.00 to +0.00 | 0.0% |
| MRCP/SMR/Riemann fusion | 58.74% | +3.41 points | +1.19 to +5.81 | 68.4% |
| FBCNet | 54.95% | -0.39 points | -3.18 to +2.89 | 31.6% |
| Conv1D + WAY SSL | 54.90% | -0.43 points | -1.77 to +0.95 | 39.5% |
| Riemannian tangent | 53.88% | -1.46 points | -2.72 to -0.13 | 28.9% |
| Statistical target-only | 53.56% | -1.77 points | -3.35 to -0.18 | 31.6% |

## Interpretation

The highest development-observed mean BA was **ShallowConvNet** at **62.49%**. No neural candidate met every predeclared advancement check.

MRCP + FBCNet also improved the equal-participant mean over the corrected target-only Conv1D in 76.3% of participants. ShallowConvNet's pooled improvement did not satisfy the PhysioNet or Liu checks, so its nearly identical pooled score should not be interpreted as equally robust cross-domain evidence.

This bakeoff evaluates cue-locked intent-versus-rest windows. False activation is reported per rest window, not per minute. Model selection for a prospective study must be frozen before collecting a new external or in-person cohort.
