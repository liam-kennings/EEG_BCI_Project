# Temporal Evidence and OOD-Rejection Experiment

## Purpose

`temporal_evidence_ood_bci.py` tests whether command confirmation over multiple
EEG windows can reduce false prosthetic activations without relying only on a
strict single-trial threshold.

The experiment is isolated from `patient_adaptive_bci.py` and preserves the same
transfer protocol:

- Entire target subjects remain unseen during source-model training.
- Runs 4/8 provide 12 patient calibration trials per class.
- Run 12 is the untouched final evaluation session.
- Controller parameters are selected using separate source validation subjects
  that undergo the same calibration/evaluation procedure.

Each three-second cue trial contains nine overlapping one-second windows with a
0.25-second step. The window representation uses five mu/beta covariance bands,
Euclidean subject alignment, pre-cue-relative covariance, and log-covariance
features.

## Controller

For every accepted window, the controller accumulates clipped intent log-odds
and left/right log-odds. A command is emitted only when both evidence thresholds
are reached. Class-conditional Ledoit-Wolf Mahalanobis distance rejects windows
that are far from every learned rest/left/right source distribution.

The default source-only search selected:

```text
decay                 = 1.0
intent evidence       = 4.0
direction evidence    = 1.0
OOD coverage quantile = 0.95
```

## Full-Cohort Results

The cache contains 9,185 trials and nine windows per trial. The final evaluation
contains the same 22 held-out patients used by the patient-adaptive benchmark.

| Metric | Evidence only | Evidence + OOD |
|---|---:|---:|
| Balanced accuracy | 40.90% | 40.72% |
| False-activation rate | 14.83% | 14.14% |
| Missed-intent rate | 71.07% | 71.70% |
| Median command latency | 2.25 s | 2.25 s |
| OOD window rejection | 0.00% | 7.40% |

The OOD gate removes a small number of false commands, but it also removes true
intent windows and does not improve the overall tradeoff. Forced 95% and 97.5%
OOD experiments reached the same conclusion.

A less conservative 20% source-validation target reduced missed intent to
53.77%, but held-out false activation rose to 25.86%. At a comparable false
activation rate, the current full-trial safety gate remains better.

## Interpretation

This is a useful negative result. EEGMMI contains clean, closed-set, cue-driven
rest/left/right states. Most false activations therefore look like valid source
data rather than feature-space outliers, so Mahalanobis OOD rejection cannot
identify them reliably. The one-second covariance classifier is also less
sensitive to weak intent than the three-second model.

The next justified experiment is an EEGNet raw-window gate with calibrated deep
ensembles or energy/density OOD scoring. Better per-window representations are
needed before temporal evidence can recover weak intent. Final asynchronous
safety validation will still require long continuous idle recordings, which
EEGMMI does not provide.

## Run It

```bash
python3 temporal_evidence_ood_bci.py
```

Force an OOD operating point:

```bash
python3 temporal_evidence_ood_bci.py --force-ood-quantile 0.95
```

The report, model, plot, and window cache are stored under
`artifacts/temporal_ood/`.

## Research Basis

- [Sequential probability ratio testing for motor-imagery BCI](https://pubmed.ncbi.nlm.nih.gov/29348781/)
- [Continuous asynchronous MI with dwell and refractory periods](https://doi.org/10.1109/TNSRE.2004.827220)
- [Sliding-window prescreening for asynchronous MI](https://arxiv.org/abs/2412.09006)
- [Temporal OOD detection for asynchronous MI](https://arxiv.org/abs/2605.01014)
