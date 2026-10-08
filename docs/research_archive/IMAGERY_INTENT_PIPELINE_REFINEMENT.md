# Motor-Imagery Intent Controller

## Question

The clinical task is not simply left-versus-right classification. A rehabilitation device must decide when to remain idle, when a patient has a credible motor-imagery intent, and only then which action to decode. A false activation can deliver an unintended movement; a missed intent makes the device unusable. The appropriate objective is therefore a measured operating point on false activations versus intent sensitivity, not raw three-class accuracy alone.

## Evidence Used

An idle/non-control state should be explicitly modeled rather than treated as an afterthought. A classic asynchronous motor-imagery BCI used one classifier for idle versus motor imagery and a second classifier for direction, which is the basis for the hierarchy here ([Pfurtscheller et al., 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC1994518/)).

This matters particularly for neurorehabilitation: an asynchronous clinical study describes false positives as potentially harmful and used a two-phase classifier with recalibration across sessions ([Lopez-Larraz et al., 2022](https://pubmed.ncbi.nlm.nih.gov/36172602/)). A reject option is also a real tradeoff, not a free accuracy gain. In an FBCSP study, rejection reduced false-discovery rate from 34.6% to 6.9%, but reduced decoded trials from 93.2% to 34.2%; a less strict F0.5 operating point retained more trials at a higher false-discovery rate ([Liu et al., 2017](https://pubmed.ncbi.nlm.nih.gov/29060520/)).

Short calibration and re-calibration are appropriate for a patient-adaptive controller. An online BCI with severely impaired users re-estimated the classifier after small batches of artifact-free trials and used a short refractory interval after a command ([Friedrich et al., 2014](https://pubmed.ncbi.nlm.nih.gov/25014055/)). For a future continuous controller, sequential evidence accumulation is preferable to one irreversible window decision because it exposes the speed/error tradeoff directly ([Hsieh et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29348781/)).

Artifact control remains important. A study of simulated muscle artifacts found that broad channel sets degraded CSP more than a central-channel set, supporting this project’s motor-cortex channel focus rather than assuming all scalp channels help ([Castermans et al., 2015](https://pubmed.ncbi.nlm.nih.gov/26736664/)).

## Refined Pipeline

1. Keep only unilateral fist imagery runs: Runs 4 and 8 calibrate a target patient; Run 12 is held out for that patient. Execution runs are not used in this intent decoder because the previous cross-task experiment found no general intent-transfer benefit.

2. Preprocess full scalp EEG before choosing the nine motor channels: 4–35 Hz bandpass, common-average reference, resampling to 160 Hz, pre-cue baseline extraction, and peak-to-peak / non-finite rejection. A 50/60 Hz notch is unnecessary in this specific profile because 35 Hz is the upper passband; add it only if the passband is widened.

3. Build filter-bank task and baseline-relative covariance features, then apply per-subject Euclidean alignment and Riemannian tangent-space projection. The baseline-relative view is intended to preserve person-specific ERD/ERS change rather than absolute amplitude.

4. Train two separate models on development subjects: a rest-versus-imagery gate and a left-versus-right decoder. For a new patient, use 12 labeled trials per imagery class to adapt the gate and decoder, plus 18 labeled rest trials to estimate the safety threshold more reliably.

5. Issue a command only when both stages agree: the individualized intent probability must exceed the patient threshold and the direction posterior must exceed the configured confidence margin. Otherwise output `rest` explicitly. The controller-validation cohort chooses conservative, balanced, and responsive policies before the final target cohort is evaluated.

6. For deployment, add a separate raw-window temporal controller after the trial-level policy is satisfactory: consecutive-evidence confirmation, a refractory interval after a command, and ongoing rest-window recalibration. The cue-locked EEGMMI recordings are useful for trial-level validation but cannot measure real false activations per minute in self-paced idle periods.

## What To Report

The experiment reports mean per-patient false activation rate, missed-intent rate, intent balanced accuracy, direction balanced accuracy, command rate, and patient-specific thresholds. It saves a controller-validation tradeoff frontier, final operating-point bars, and a per-patient safety-versus-sensitivity scatter plot. A policy that misses the final-cohort false-activation budget must be treated as a failed generalization result, even if its conventional accuracy improves.
