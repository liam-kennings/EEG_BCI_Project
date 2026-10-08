# EEG Decoder Model Bakeoff

## Contract

All neural candidates use seeds 42-46, identical target development rows, class/dataset weights, epochs, batch order, and controller-derived dataset thresholds. Target-only models estimate normalization from development subjects only. The WAY SSL model is an explicitly labeled transfer condition. Development-observed subjects are reused for exploratory comparison and are not a new locked clinical cohort.

## Overall Results

| Model | BA mean (SD) | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | Train time/seed | CPU ms/256 windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MRCP + FBCNet | 62.66% (1.01) | 31.29% | 94.03% | 0.804 | 0.794 | 0.183 | 18166 | 164.2 s | 102.44 |
| ShallowConvNet | 62.63% (0.80) | 30.45% | 94.81% | 0.782 | 0.767 | 0.198 | 17162 | 39.7 s | 15.78 |
| Conv1D target-only | 60.63% (0.72) | 27.49% | 93.78% | 0.746 | 0.728 | 0.233 | 75890 | 78.4 s | 46.30 |
| EEGNet | 58.92% (0.53) | 24.28% | 93.55% | 0.763 | 0.731 | 0.201 | 9338 | 144.1 s | 53.82 |
| MRCP/SMR/Riemann fusion | 58.51% (0.00) | 22.22% | 94.80% | 0.780 | 0.767 | 0.206 | deterministic reference | not retrained | not benchmarked |
| FBCNet | 56.94% (1.05) | 20.34% | 93.54% | 0.683 | 0.676 | 0.226 | 2314 | 25.3 s | 8.47 |
| Conv1D + WAY SSL | 56.29% (0.32) | 19.59% | 93.00% | 0.683 | 0.662 | 0.235 | 75890 | not retrained | 46.61 |
| Riemannian tangent | 55.86% (0.00) | 19.36% | 92.36% | 0.693 | 0.673 | 0.222 | deterministic reference | not retrained | not benchmarked |
| Statistical target-only | 52.82% (0.00) | 10.58% | 95.07% | 0.683 | 0.645 | 0.227 | deterministic reference | not retrained | not benchmarked |

## Neural Advancement Check

| Candidate | BA change vs target Conv1D | 95% seed interval | Specificity change | All checks met |
|---|---:|---:|---:|---:|
| Conv1D + WAY SSL | -4.34 points | -5.46 to -3.21 | -0.78 points | no |
| EEGNet | -1.72 points | -3.00 to -0.43 | -0.22 points | no |
| ShallowConvNet | +2.00 points | +1.09 to +2.90 | +1.03 points | no |
| FBCNet | -3.69 points | -5.11 to -2.27 | -0.23 points | no |
| MRCP + FBCNet | +2.03 points | +1.08 to +2.97 | +0.25 points | yes |

## Dataset Results

| Model | Liu stroke BA | OpenBMI BA | PhysioNet BA |
|---|---:|---:|---:|
| MRCP + FBCNet | 53.93% | 64.95% | 60.58% |
| ShallowConvNet | 51.20% | 67.81% | 54.60% |
| Conv1D target-only | 51.30% | 63.75% | 56.78% |
| EEGNet | 52.02% | 59.78% | 59.60% |
| MRCP/SMR/Riemann fusion | 57.68% | 58.29% | 59.38% |
| FBCNet | 61.22% | 58.82% | 50.60% |
| Conv1D + WAY SSL | 51.45% | 57.29% | 55.84% |
| Riemannian tangent | 50.86% | 57.33% | 54.33% |
| Statistical target-only | 54.85% | 52.25% | 53.36% |

## Equal-Participant Analysis

| Model | Participant-mean BA | Change vs target Conv1D | 95% participant-bootstrap interval | Participants improved |
|---|---:|---:|---:|---:|
| MRCP + FBCNet | 59.52% | +3.10 points | +1.50 to +4.73 | 76.3% |
| ShallowConvNet | 55.77% | -0.66 points | -2.40 to +1.26 | 28.9% |
| Conv1D target-only | 56.42% | +0.00 points | +0.00 to +0.00 | 0.0% |
| EEGNet | 57.62% | +1.19 points | -0.41 to +2.91 | 50.0% |
| MRCP/SMR/Riemann fusion | 58.74% | +2.32 points | +0.03 to +4.77 | 60.5% |
| FBCNet | 54.69% | -1.73 points | -4.73 to +1.80 | 28.9% |
| Conv1D + WAY SSL | 54.90% | -1.52 points | -3.15 to +0.07 | 36.8% |
| Riemannian tangent | 53.88% | -2.55 points | -4.03 to -1.02 | 23.7% |
| Statistical target-only | 53.56% | -2.86 points | -4.82 to -0.89 | 28.9% |

## Interpretation

The highest development-observed mean BA was **MRCP + FBCNet** at **62.66%**. Candidates meeting every predeclared neural advancement check: **MRCP + FBCNet**.

MRCP + FBCNet also improved the equal-participant mean over the corrected target-only Conv1D in 76.3% of participants. ShallowConvNet's pooled improvement did not satisfy the PhysioNet or Liu checks, so its nearly identical pooled score should not be interpreted as equally robust cross-domain evidence.

This bakeoff evaluates cue-locked intent-versus-rest windows. False activation is reported per rest window, not per minute. Model selection for a prospective study must be frozen before collecting a new external or in-person cohort.
