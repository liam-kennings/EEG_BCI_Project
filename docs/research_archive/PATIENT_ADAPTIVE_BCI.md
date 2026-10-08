# Patient-Adaptive Motor-Intent Pipeline

## Why the experiment changed

The earlier approximately 60% result held out a run but not a person. The same
subjects contributed training and verification windows, and multiple overlapping
windows came from each cue. That is useful for within-subject debugging, but it
does not support the claim that a model will generalize to a new rehabilitation
patient.

`patient_adaptive_bci.py` tests the intended translational claim directly:

1. Train a global decoder on source subjects only.
2. Keep entire target subjects unseen during global training and preprocessing fit.
3. For each target subject, use imagery Runs 4 and 8 only for labeled calibration.
4. Evaluate only on that subject's untouched imagery Run 12.
5. Report performance at 3, 7, 10, and 12 calibration trials per class.

One cue produces one sample. Overlapping windows from the same cue are not split
across training and testing.

## Signal and model design

The new representation keeps information that the previous per-channel wavelet
moments discarded:

- Full-scalp common-average referencing is applied before selecting 15
  sensorimotor channels.
- Five mu/beta filter-bank views accommodate patient-specific peak frequencies.
- OAS-regularized spatial covariance preserves relationships among C3, C4, Cz,
  and adjacent motor channels.
- Euclidean alignment normalizes each source subject and uses only Run 4
  calibration data to align a target patient.
- Riemannian tangent-space features represent spatial covariance without treating
  covariance matrices as ordinary flat vectors.
- Calibration-derived band weights emphasize the patient's most discriminative
  mu/beta ranges when matching source subjects.
- The 30 source subjects with covariance patterns closest to the patient train a
  patient-matched source decoder, reducing negative transfer from dissimilar users.
- A second covariance branch is relative to the one-second pre-cue baseline and
  captures ERD/ERS changes that help separate rest from weak motor intent.
- A hierarchical decoder first detects rest versus intent, then selects left
  versus right. This matches the safety logic of a prosthetic controller better
  than a single three-class argmax.
- The patient adapter is deliberately small. It uses at most twice the number of
  calibration trials per class in the source-learned PCA space, preventing a
  40-dimensional model from being fit to only a handful of trials.
- Patient calibration adapts only the intent gate. The stronger all-source
  left/right decoder is frozen to prevent few-shot direction overfitting.

This follows the central transfer-learning conclusion in the research document:
alignment must happen before classification, and a general model should be
adapted with a small amount of target-subject data. It uses a compact covariance
model instead of immediately moving to a transformer because the number of
independent trials per patient is small.

## Full-cohort result

The completed run extracted 9,148 clean trials from 108 usable subjects. Subject
109 was excluded by the fixed artifact criterion because its selected motor
channels repeatedly exceeded 250 microvolts peak-to-peak.

The seeded split used 86 source subjects and 22 entirely unseen target subjects.
At 12 labeled trials per class from Runs 4/8, evaluated on untouched Run 12:

| Metric | Global model | Similar-source | Patient-adapted |
|---|---:|---:|---:|
| Three-class accuracy | 52.30% | 52.80% | 53.12% |
| Three-class balanced accuracy | 46.06% | 46.49% | 51.17% |
| Intent balanced accuracy | 62.57% | 62.92% | 64.75% |
| Left/right direction balanced accuracy | 67.87% | 64.07% | 67.87% |
| Missed-intent rate | 49.69% | 49.69% | 30.50% |
| False-activation rate | 25.17% | 24.48% | 40.00% |

The calibration primarily improves detection of the patient's weak intent while
preserving the global direction decoder. The gain comes with a clinically
important tradeoff: more true intent is detected, but false activations increase.
A deployable controller therefore needs confidence-based abstention and temporal
confirmation before actuating hardware.

## Safety operating points

The pipeline now reports a separate safety-gated decoder. Its intent threshold
is selected from source-subject out-of-fold predictions and can only be made more
conservative by the new patient's calibration rest trials. Run 12 is never used
to choose the threshold.

With the default 10% calibration target at 12 trials per class, the untouched
Run 12 result is:

| Metric | Accuracy-optimized | Safety-gated |
|---|---:|---:|
| Ordinary accuracy | 53.12% | 54.28% |
| False-activation rate | 40.00% | 17.24% |
| Missed-intent rate | 30.50% | 57.86% |
| Direction balanced accuracy | 67.87% | 67.87% |

A stricter `--max-false-activation 0.05` calibration target lowers held-out false
activation to 10.34%, but misses 72.64% of intent. The calibration target is not
an exact guarantee on a later session because EEG distributions drift between
Runs 8 and 12. These results show why thresholding should be combined with
multi-window temporal confirmation rather than made arbitrarily strict.

These numbers are not directly comparable to the old 60% because this protocol
uses unseen people, unseen sessions, independent trials, and includes rest.

## Run it

The final experiment is one command; the covariance cache makes later reruns much
faster:

```bash
python3 patient_adaptive_bci.py
```

For the stricter safety operating point:

```bash
python3 patient_adaptive_bci.py --max-false-activation 0.05
```

For a quick pipeline check:

```bash
python3 patient_adaptive_bci.py \
  --max-subjects 20 \
  --calibration-budgets 3,7 \
  --cache-path artifacts/patient_adaptive/smoke20_cache.npz \
  --output-dir artifacts/patient_adaptive/smoke20
```

To evaluate specific held-out patients:

```bash
python3 patient_adaptive_bci.py --test-subjects 96,107
```

The final model, metrics, calibration curve, and per-patient plot are written to
`artifacts/patient_adaptive/`. The false-activation tradeoff is saved as
`false_activation_curve.png`.

## Research basis

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [He and Wu: Euclidean Alignment for BCI Transfer Learning](https://arxiv.org/abs/1808.05464)
- [Wu et al.: A Complete Transfer-Learning Pipeline for Motor Imagery BCI](https://arxiv.org/abs/2007.03746)
- [Zanini et al.: Riemannian Transfer Learning for BCIs](https://pubmed.ncbi.nlm.nih.gov/28841546/)
- [Pan et al.: Riemannian Cross-Session Motor-Imagery Decoding](https://pubmed.ncbi.nlm.nih.gov/37931299/)

## Next clinical step

The next model change should not be a larger network. First repeat this protocol
on a stroke or impaired-patient dataset, because EEGMMI contains healthy,
cue-driven participants and cannot validate clinical transfer by itself. Then add
confidence-based abstention and temporal evidence accumulation so a prosthetic
command requires several consistent intent estimates rather than one trial-level
decision.
