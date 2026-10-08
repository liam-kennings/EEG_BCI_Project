# Using Physical Motor Execution to Improve Motor-Intent Decoding

> **Experiment completed:** The full 109-subject results are documented in
> `CROSS_TASK_EXECUTION_RESULTS.md`. Execution improved in-domain detection and
> modestly improved direction representations, but direct transfer, pooling, and
> pretraining did not improve overall unseen-subject imagery performance.

## Research Question

Can the stronger EEG response during physical fist movement teach a model a
representation that improves detection of weaker motor imagery or attempted
movement in a completely unseen patient?

The answer from current evidence is **potentially yes, but not by treating
execution and imagery as identical samples**. They share motor preparation and
sensorimotor rhythm modulation, but execution adds muscle activity, actual limb
motion, and proprioceptive/somatosensory feedback. A model can exploit those
execution-only signals and appear better without learning transferable intent.

The best formulation is therefore cross-task transfer:

```text
Motor execution = strong but contaminated source domain
Motor imagery   = weak healthy-subject target domain
Motor attempt   = eventual rehabilitation-patient target domain
```

## What the Evidence Says

### Execution and imagery share useful physiology

Both tasks commonly produce alpha/mu and beta event-related desynchronization
over sensorimotor regions, although their timing and spatial emphasis differ.
In a 28-person experiment, an execution-trained model obtained 65.93% on imagery,
statistically similar to the study's imagery-trained result. Combining execution
with half of the imagery trials increased imagery accuracy to 69.21%, with the
largest benefit among lower-performing users. This is direct evidence for the
basic idea, although it was not the strict unseen-patient/rest-inclusive protocol
used in this project.

A second study using EEGMMI reported that execution-trained deep models tested
on imagery performed comparably to imagery-trained models. Its attribution
analysis found shared patterns but also task-specific emphasis: execution focused
more on contralateral central activity from 0.5-1.0 seconds, while imagery also
used ipsilateral frontocentral activity.

A 2026 domain-adaptation study also used EEGMMI Runs for execution-to-imagery
transfer. It found direct execution-trained EEGNet performance comparable to an
imagery-trained baseline and larger gains when substantial labeled imagery from
the same target subject was available. Its best result used 80% of each target
subject's imagery samples for training, classified only left versus right, and
did not include rest. Its reported 81.75% therefore does **not** establish
zero-shot patient transfer or safe asynchronous intent detection, but it supports
class-conditional domain alignment as an experiment.

### Execution is not simply a louder version of imagery

Execution includes movement and sensory reafference after motor output begins.
Movement-related cortical potentials also contain post-movement components tied
to kinesthetic feedback. Imagery lacks that peripheral loop and can involve a
broader or differently timed planning network.

Muscle contamination is a serious shortcut. In a neuromuscular-blockade study,
scalp power above 20 Hz differed by 10-200 times between paralyzed and
unparalyzed conditions, leading the investigators to conclude that much scalp
activity above 20 Hz was EMG-derived. That does not mean beta above 20 Hz should
always be discarded, but execution-trained high-beta features require a specific
transfer and artifact audit.

EEGMMI has cue annotations, not measured EMG or exact movement-onset timestamps.
We cannot cleanly divide pre-movement intent from actual movement and feedback in
these files. Execution can still be useful, but we must demonstrate that gains
survive evaluation on imagery-only data.

### Motor attempt may be the real clinical target

For a patient with paralysis, the desired action is often an active attempt to
move rather than an abstract imagery exercise. In 14 patients with hemiplegia,
motor-attempt decoding averaged 79% versus 66.5% for imagery, while 8-30 Hz ERD
and lateralization did not differ significantly between tasks. A separate study
detected attempted finger extension during motor planning in people with complete
hand paralysis and showed that conservative thresholds plus consecutive
confirmations could eliminate false triggers in some sessions while retaining
useful true triggers.

Healthy EEGMMI execution is not the same as impaired-patient motor attempt.
Execution data can help develop a representation, but only attempted-movement
patient recordings can validate the clinical claim.

## EEGMMI Run Design

For unilateral fists:

| Domain | Runs | Labels |
|---|---|---|
| Physical execution | 3, 7, 11 | T0 rest, T1 left, T2 right |
| Motor imagery | 4, 8, 12 | T0 rest, T1 left, T2 right |

The immutable subject split remains unchanged:

- 69 development subjects fit model parameters.
- 18 controller-validation subjects choose architecture, transfer strength,
  thresholds, and calibration policy.
- 22 final-target subjects remain untouched until the method is frozen.

Every result must be evaluated on **imagery Run 12**. Execution accuracy is a
diagnostic result, not the main outcome.

## Option 0: Cross-Task Diagnostic Matrix

**Recommendation: implement first.**

Before designing a complex transfer network, run four experiments with the same
channels, epochs, model, and subject partitions:

| Training data | Test data | Purpose |
|---|---|---|
| Execution | Execution | Confirm that strong motor structure is learnable |
| Imagery | Imagery | Existing baseline |
| Execution | Imagery | Measure direct transfer and domain gap |
| Execution + imagery | Imagery | Test whether simple additional data helps |

Run the matrix separately for:

- Rest versus intent.
- Left versus right conditional on intent.
- FBCSP and EEGNet.
- Mu only (7-13 Hz), low beta (13-20 Hz), high beta (20-30 Hz), and full
  7-30 Hz.
- Early 0.5-1.5 s and late 1.5-3.5 s intervals.

This identifies where execution helps. For example, improvement only in late
high-beta execution but not imagery would suggest EMG or reafference rather than
transferable intent.

**Cost:** Low.  
**Scientific value:** Very high.  
**Risk:** Low, because it makes no deployment claim.

## Option 1: Execution Pretraining, Imagery Fine-Tuning

**Recommendation: first actual transfer model.**

1. Pretrain EEGNet on development-subject execution Runs 3/7/11.
2. Initialize a new imagery model from the execution backbone.
3. Reset or separately initialize the intent and direction heads.
4. Fine-tune on development-subject imagery Runs 4/8/12.
5. Select freezing duration, learning rates, and thresholds using only controller
   subjects.
6. Evaluate final subjects on imagery Run 12.

Variants to compare:

- Freeze the temporal/spatial backbone and train only imagery heads.
- Freeze for several epochs, then unfreeze with a 5-10 times smaller backbone
  learning rate.
- Keep domain-specific batch-normalization statistics for execution and imagery.
- Pretrain direction only on execution; do not let execution determine the final
  rest/intent threshold.

This is easy to interpret. If it fails, more complex adaptation is unlikely to be
justified until the domain diagnostics explain why.

**Cost:** Low to moderate.  
**Evidence:** Strongest direct support among practical options.  
**Main risk:** The backbone may preserve execution-specific shortcuts.

## Option 2: Shared Backbone with Domain-Specific Heads

Train execution and imagery simultaneously rather than pooling their labels:

```text
                 shared EEG backbone
                 /                 \
       execution adapter       imagery adapter
       intent/direction        intent/direction
```

Use balanced batches containing both domains. The shared backbone learns common
mu/beta spatial-temporal structure, while small domain-specific adapters or batch
normalization layers retain legitimate differences. The deployment output comes
only from the imagery head.

Loss:

```text
L = L_imagery_intent + L_imagery_direction
  + lambda_me * (L_execution_intent + L_execution_direction)
```

Tune `lambda_me` on controller subjects. Start small so the stronger execution
signal does not dominate the target task.

**Cost:** Moderate.  
**Benefit:** More protection against negative transfer than naive pooling.  
**Main risk:** Shared features can still encode domain-specific artifact.

## Option 3: Class-Conditional Domain Alignment

Add explicit alignment between execution and imagery representations:

- Align execution-left with imagery-left.
- Align execution-right with imagery-right.
- Align execution-rest with imagery-rest separately.
- Preserve class separation while minimizing task-domain differences.

Candidate losses are class-conditional CORAL/MMD or supervised contrastive loss.
The 2026 EEGMMI study found that global and class-level alignment both mattered,
and warned that indiscriminate alignment can distort class structure and cause
negative transfer.

For this project, supervised contrastive alignment is the cleaner initial form:

- Positive pairs: same subject and motor class across execution and imagery.
- Hard negatives: opposite direction from the same subject.
- Domain nuisance: execution versus imagery should not determine distance.
- Patient and final-test trials are never used to form development pairs.

EEGMMI trials are same-subject/class examples, not synchronized trial pairs, so
the loss should use class/domain groups rather than claim one-to-one pairing.

**Cost:** Moderate to high.  
**Potential:** High if direct transfer shows a clear domain gap.  
**Main risk:** Alignment can erase useful imagery-specific structure.

## Option 4: Execution Teacher, Imagery Student

Train a high-capacity teacher on execution, then transfer only selected
relationships to an imagery student:

- Distill inter-trial similarity or spatial attention, not just teacher logits.
- Distill mu/low-beta representations more strongly than high-beta features.
- Keep the student's supervised imagery loss dominant.
- Reject teacher features whose attribution lies outside sensorimotor channels or
  whose performance disappears below 20 Hz.

This is attractive when the execution teacher is strong but its output
distribution does not calibrate on imagery. It is less directly supported for
execution-to-imagery than pretraining or domain adaptation, so it should come
later.

**Cost:** High.  
**Potential:** Moderate.  
**Main risk:** Distilling confident EMG shortcuts into the student.

## Option 5: Self-Supervised Pretraining on Both Tasks

Ignore labels initially and learn from all development-subject continuous EEG:

- Mask and reconstruct temporal segments.
- Contrast two valid augmentations of the same window.
- Predict temporal order or nearby-window relationships.
- Then fine-tune an imagery-only hierarchical decoder.

Because self-supervision does not force execution and imagery to share a class
boundary, it can exploit the additional signal volume with less label-induced
shortcut pressure. However, it may still learn movement artifact unless
augmentations and channel/band audits are carefully controlled.

**Cost:** High.  
**Potential:** Long-term, especially when adding more EEG datasets.  
**Recommendation:** Not before Options 0-3.

## Option 6: Separate MRCP and Sensorimotor-Rhythm Branches

The current 4-35 Hz preprocessing removes movement-related cortical potentials,
which are slow signals associated with movement preparation. A future detector
could fuse:

```text
0.1-3 Hz MRCP branch      -> preparatory intent evidence
7-30 Hz SMR branch       -> mu/beta ERD and direction evidence
artifact/EMG quality     -> veto, never motor class evidence
```

Execution data are useful for developing this idea, but EEGMMI lacks exact
movement onset and EMG timing. Its visual cue also produces cue-locked activity
that can masquerade as preparation. Therefore:

- Treat EEGMMI MRCP results as exploratory.
- Use pre-cue and early post-cue windows, not only the existing 0.5-3.5 s epoch.
- Test whether the branch distinguishes left/right rather than merely cue/rest.
- Require a future patient dataset with EMG, bend sensors, or kinematics to mark
  actual and attempted movement onset.

This option is closely aligned with detecting the "whisper" before movement and
may ultimately be more clinically meaningful than larger imagery classifiers.

**Cost:** Moderate now, high for proper validation.  
**Potential:** High for early intent detection.  
**Main limitation:** EEGMMI timing cannot validate true pre-movement detection.

## Option 7: Make Motor Attempt the Deployment Paradigm

Instead of asking impaired patients to reproduce healthy motor imagery, instruct
them to actively attempt the desired movement. For patients with residual EMG,
EMG can be recorded as a quality/ground-truth channel during calibration while
EEG remains the neural command source. For complete paralysis, unaffected-limb
timing or device sensors can help define intended onset during calibration.

This changes the project claim from:

> Detect imagined left/right fist movement.

to:

> Detect attempted left/right movement early enough to trigger contingent
> rehabilitation assistance.

That claim is more consistent with the intended prosthetic use and clinical
attempted-movement literature. It still requires external stroke data because
healthy execution cannot simulate lesion-dependent cortical reorganization.

**Cost:** Requires new/clinical data.  
**Clinical relevance:** Highest.  
**Recommendation:** Use as the long-term target while EEGMMI develops methods.

## Required Shortcut and Safety Audits

Any execution-assisted model must pass all of these checks:

1. **Imagery-only final evaluation:** improvement must appear on unseen imagery
   trials, never only on execution.
2. **Band audit:** report 7-13, 13-20, and 20-30 Hz transfer separately.
3. **Time audit:** compare early and late windows to identify feedback/artifact
   dependence.
4. **Channel audit:** compare central sensorimotor channels with peripheral
   frontal/temporal channels. Strong peripheral-only execution accuracy is a
   warning, not a success.
5. **Domain probe:** train a probe to distinguish execution from imagery from the
   learned representation. High domain accuracy indicates incomplete alignment.
6. **Attribution/topography:** verify physiologically plausible contralateral
   central emphasis and inspect subject variability.
7. **Rest safety:** execution must not raise imagery false activation or worsen
   the safety-threshold missed-intent tradeoff.
8. **Per-patient effect:** report how many final patients improve, not only pooled
   accuracy.
9. **Negative-transfer stop rule:** reject a transfer method if median patient
   intent balanced accuracy or safety utility worsens, even if pooled direction
   accuracy rises.

## Recommended Experimental Sequence

### Phase 1: Establish whether transferable signal exists

Implement Option 0 for FBCSP and EEGNet using Runs 3/7/11 and 4/8/12. This should
produce a cross-task matrix for intent, direction, frequency band, and time
window. It is the lowest-cost way to determine whether execution helps our weak
intent gate or only the already-stronger direction decoder.

### Phase 2: Simple transfer before complex transfer

Implement Option 1. Compare execution-pretrained EEGNet against the existing
deterministic imagery-only EEGNet on the same locked protocol. Add Option 2 only
if pretraining helps some subjects but shows domain-specific failure.

### Phase 3: Alignment and patient calibration

If the domain probe remains strong, implement class-conditional contrastive or
CORAL/MMD alignment. For final-patient simulation:

- Calibration: imagery Runs 4/8.
- Optional able-bodied execution calibration: Runs 3/7, reported separately.
- Locked evaluation: imagery Run 12.
- Do not use execution Run 11 in the primary result because it immediately
  precedes Run 12 and represents an easier same-session calibration condition.
- A secondary "same-day physical calibration" experiment may use Run 11, but it
  must be labeled separately.

### Phase 4: Clinically aligned intent

Move from healthy imagery to attempted-movement stroke data. Add the MRCP branch,
measured movement/EMG timing, long rest periods, temporal confirmation, and
false-command-per-hour evaluation.

## Ranked Recommendation

| Rank | Option | Why |
|---:|---|---|
| 1 | Cross-task diagnostic matrix | Tells us whether and where the idea works |
| 2 | Execution pretrain -> imagery fine-tune | Strong evidence and low complexity |
| 3 | Shared backbone, domain-specific heads | Uses both tasks without pretending they are identical |
| 4 | Class-conditional alignment/contrastive learning | Best response to a measured domain gap |
| 5 | MRCP + SMR dual branch | Closest to early motor-intent physiology, but timing-limited in EEGMMI |
| 6 | Self-supervised joint pretraining | Valuable after the supervised baselines are understood |
| 7 | Teacher-student distillation | Most likely to transfer execution shortcuts if used too early |

The recommended immediate experiment is not naive pooling. It is the diagnostic
matrix followed by execution pretraining and imagery fine-tuning, with final
imagery Run 12 and all false-activation metrics unchanged.

## Primary Research Sources

- [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- [Gwon and Ahn: motor task-to-task transfer learning](https://www.sciencedirect.com/science/article/pii/S1053811924004038)
- [Pérez-Velasco et al.: execution-to-imagery deep transfer](https://doi.org/10.1016/j.bspc.2025.107834)
- [Wang et al.: domain-class adaptation on EEGMMI](https://doi.org/10.3389/fnins.2026.1851006)
- [Chen et al.: motor attempt versus imagery in hemiplegia](https://doi.org/10.3389/fnbot.2021.706630)
- [Muralidharan et al.: attempted movement in complete hand paralysis](https://doi.org/10.3389/fnins.2011.00039)
- [Blokland et al.: attempted movement during neuromuscular block](https://pubmed.ncbi.nlm.nih.gov/26248679/)
- [Whitham et al.: EMG contamination above 20 Hz](https://pubmed.ncbi.nlm.nih.gov/17574912/)
- [Combined MRCP and ERD analysis of movement preparation/execution](https://pmc.ncbi.nlm.nih.gov/articles/PMC6182054/)
- [Supervised contrastive domain generalization for MI/ME](https://pubmed.ncbi.nlm.nih.gov/39046861/)
