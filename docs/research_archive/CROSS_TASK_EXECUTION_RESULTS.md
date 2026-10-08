# Motor Execution to Motor Imagery: Full-Cohort Results

## Question

Does adding healthy physical fist movement help a model detect the weaker motor
imagery signal in completely unseen subjects?

The experiment used 18,381 retained trials from 109 subjects and the immutable
69/18/22 development, controller-validation, and final-target split. Physical
execution Runs 3/7/11 and imagery Runs 4/8/12 were cached together, but final
imagery performance was evaluated only on Run 12. Execution Run 11 was never
used to calibrate the final imagery test.

## Cross-Task Results

### FBCSP

| Train -> test | Balanced accuracy | Intent BA | Direction BA | Safety FP | Safety missed intent |
|---|---:|---:|---:|---:|---:|
| Execution -> execution | 55.43% | 72.66% | 65.20% | 9.2% | 63.9% |
| Execution -> imagery | 40.93% | 55.87% | 64.01% | 9.5% | 83.3% |
| Imagery -> imagery | **44.26%** | **57.07%** | 61.83% | **7.1%** | 87.4% |
| Combined -> imagery | 39.86% | 55.38% | 63.69% | 9.8% | 83.3% |

Execution contains a substantially clearer rest-versus-movement signal in its
own domain, but that advantage does not transfer to imagery. Direction transfers
somewhat better than intent.

### EEGNet

All source models used the same deterministic architecture and 20-epoch limit.
The fine-tuning stage used 10 additional imagery epochs at one-fifth the learning
rate.

| Train -> test | Balanced accuracy | Intent BA | Direction BA | Safety FP | Safety missed intent |
|---|---:|---:|---:|---:|---:|
| Execution -> execution | 50.89% | 69.42% | 62.71% | 10.5% | 57.7% |
| Execution -> imagery | 40.24% | 55.81% | 63.39% | 11.2% | 84.9% |
| Imagery -> imagery | **46.15%** | **56.81%** | 64.64% | 11.5% | 82.6% |
| Combined -> imagery | 37.59% | 53.59% | 63.68% | 8.5% | 85.2% |
| Execution pretrain + imagery fine-tune | 40.79% | 55.21% | **66.56%** | 18.3% | 79.2% |

Physical execution did not improve overall imagery decoding. Naive pooling was
the worst deep protocol. Pretraining improved direction but shifted the intent
operating point toward more false activation.

## Patient-Level Result

Compared with the imagery-only model:

| Model | Transfer method | Patients improved | Median BA change | Mean BA change |
|---|---|---:|---:|---:|
| FBCSP | Execution -> imagery | 9/22 | -4.76 points | -3.97 points |
| FBCSP | Combined -> imagery | 7/22 | -5.06 points | -4.32 points |
| EEGNet | Execution -> imagery | 8/22 | -2.68 points | -5.13 points |
| EEGNet | Combined -> imagery | 4/22 | -8.63 points | -7.42 points |
| EEGNet | Pretrain + fine-tune | 9/22 | -3.57 points | -6.14 points |

For pooled EEGNet, the paired mean change had a 95% bootstrap interval of
-12.67 to -2.08 percentage points and the paired Wilcoxon test gave p=0.0129.
This is evidence of negative transfer, not merely a neutral result. The
pretrained model helped a minority of patients and improved direction, but did
not produce a robust overall gain.

## Band and Time Audit

FBCSP was trained on execution and evaluated on imagery separately by interval
and frequency band.

| Interval | Mu 7-13 | Low beta 13-20 | High beta 20-30 | Broad 7-30 |
|---|---:|---:|---:|---:|
| Full 0.5-3.5 s | 36.08% | 37.85% | 37.88% | 40.02% |
| Early 0.5-1.5 s | 39.87% | 41.70% | **43.06%** | 39.21% |
| Late 1.5-3.5 s | 36.15% | 36.31% | 37.76% | 34.89% |

The best direct transfer occurred in early high beta. That may reflect genuine
contralateral execution/planning structure, but EEGMMI lacks EMG and exact
movement onset, so an execution artifact contribution cannot be excluded. Late
execution features transferred poorly, arguing against using movement and
sensory-feedback periods as a generic imagery template.

## Direction-Only Hybrid

Because execution helped direction more than intent, a final no-retraining test
kept the imagery-trained intent gate and thresholds while substituting another
model's direction probabilities:

| Intent gate | Direction decoder | Three-class BA | Direction BA | Safety FP | Safety miss |
|---|---|---:|---:|---:|---:|
| Imagery | Imagery | 46.15% | 64.64% | 11.53% | 82.65% |
| Imagery | Execution | 44.05% | 63.39% | 11.53% | 82.65% |
| Imagery | Execution-pretrained/fine-tuned | **46.16%** | **66.56%** | 11.53% | 82.65% |

Execution pretraining adds about 1.9 points of conditional direction accuracy,
but the three-class result changes by only 0.01 points because intent detection
remains the limiting stage.

## Interpretation

The original idea was physiologically reasonable, and the experiment confirms a
shared signal. However, healthy physical movement is not simply a higher-SNR
version of imagery:

- It is easier to detect in-domain.
- Its left/right structure transfers modestly.
- Its strong rest/movement separation does not transfer.
- Pooling the domains damages imagery performance.
- Pretraining does not improve the safety tradeoff.

The project should therefore **not** mix execution trials into the main imagery
intent model. If execution is retained, the defensible use is a low-weight
auxiliary direction task with domain-specific heads or class-conditional
alignment. Even then, improvement must be demonstrated patient by patient.

The more clinically relevant next direction is attempted-movement data from
stroke patients, potentially with a separate MRCP preparation branch and measured
EMG/kinematic onset. Healthy execution remains useful as a diagnostic and
representation pretraining source, not as ground truth for deployable intent.

## Visual Results

- `artifacts/cross_task/fbcsp_cross_task_heatmap.png`
- `artifacts/cross_task/eegnet_cross_task_heatmap.png`
- `artifacts/cross_task/safety_tradeoff.png`
- `artifacts/cross_task/per_patient_cross_task.png`
- `artifacts/cross_task/fbcsp_band_time_audit.png`
- `artifacts/cross_task/hybrid_direction_heads.png`

The complete pooled, safety, per-patient, bootstrap, and confusion-matrix results
are in `artifacts/cross_task/cross_task_metrics.json`.

## Commands

Reproduce the cross-task experiment:

```bash
python3 scripts/05_cross_task_execution_imagery.py \
  --eegnet-epochs 20 \
  --fine-tune-epochs 10 \
  --eegnet-patience 4 \
  --batch-size 256
```

Regenerate all figures without retraining:

```bash
python3 scripts/05_cross_task_execution_imagery.py --plots-only
```

Re-evaluate the saved direction-head combinations:

```bash
python3 scripts/06_evaluate_hybrid_heads.py
```

