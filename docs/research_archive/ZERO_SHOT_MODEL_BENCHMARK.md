# Locked Zero-Shot FBCSP and EEGNet Benchmark

## Protocol

`scripts/04_benchmark_models.py` uses the immutable split in
`artifacts/dataset/splits.json`:

- 69 development subjects and Runs 4/8/12 train model parameters.
- 18 controller-validation subjects and Run 12 select early stopping and intent
  thresholds.
- 22 final-target subjects and Run 12 provide the zero-shot result.
- No subject occurs in more than one partition.
- Every model receives the same 9 channels, 0.5-3.5 second epochs, 4-35 Hz
  preprocessing, artifact rule, and one-sample-per-cue labels.

The raw cache contains 9,185 retained trials. Development has 5,815 trials,
controller validation has 485 Run 12 trials, and final evaluation has 612 Run 12
trials.

## Full-Cohort Results

| Model/operating point | Balanced accuracy | Intent BA | Direction BA | False activation | Missed intent |
|---|---:|---:|---:|---:|---:|
| FBCSP balanced | 44.26% | 57.07% | 61.83% | 64.41% | 21.45% |
| FBCSP safety | 36.00% | 52.75% | 61.83% | 7.12% | 87.38% |
| EEGNet balanced | 41.56% | 54.36% | 65.21% | 51.53% | 39.75% |
| EEGNet safety | 36.56% | 52.36% | 65.21% | 9.15% | 86.12% |

The validation-selected safety threshold controls pooled final-subject false
activation reasonably well. It does not produce a usable detector: the median
final patient has zero false activation but 100% missed intent. This is not fixed
by choosing a different global architecture or a stricter threshold.

Direction discrimination is more transferable than rest-versus-intent
discrimination. EEGNet reaches 65.21% conditional direction balanced accuracy,
but its global intent gate remains weak. The next experiment should therefore
adapt the intent gate from each patient's Runs 4/8 while keeping Run 12 locked and
initially freezing the global EEGNet direction head.

## Run

The first full run builds the 187 MB raw cache. Later runs reuse it:

```bash
python3 scripts/04_benchmark_models.py
```

For a small end-to-end check:

```bash
python3 scripts/04_benchmark_models.py \
  --max-subjects 15 \
  --cache-path artifacts/benchmark/smoke15_raw_epochs.npz \
  --output-dir artifacts/benchmark/smoke15 \
  --eegnet-epochs 3
```

Models and the per-patient report are saved under `artifacts/benchmark/`.

