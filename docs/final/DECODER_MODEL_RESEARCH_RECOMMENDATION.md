# Decoder Model Research Recommendation

## Question

Would an EEG-specific neural network or a different machine-learning model be a better decoder than the current randomly initialized temporal CNN?

`Random CNN` means that the network starts from random weights and then learns normally. It is not a random-guess classifier. The original paired WAY ablation reached 53.49% mean balanced accuracy without source pretraining, but it shared WAY-derived normalization scales. The strict target-development-only Conv1D control in the completed bakeoff reached 60.63%.

## Recommendation

The selected neural architecture uses a physiology-informed two-stream network:

1. An MRCP temporal branch operating on the existing 0.1-4 Hz signal for intent-versus-rest detection.
2. An FBCNet-style branch operating on explicit sensorimotor filter banks, spatial depthwise convolutions, and log-variance aggregation for mu/beta ERD and direction information.
3. Separate intent and direction heads. Intent fusion should initially favor MRCP; direction fusion should favor the filter-bank branch.
4. Dataset-specific probability calibration, abstention, and controller-selected thresholds should remain outside the neural network.

This is a better match to the project than replacing the compact CNN with a large transformer. FBCNet was designed for limited, noisy motor-imagery data and was evaluated on OpenBMI and chronic-stroke datasets. The current project already shows that MRCP is the strongest single intent branch, while FBCSP and Riemannian representations are more useful for spatial/directional evidence.

## Completed Bakeoff Result

The recommendation was implemented under one matched five-seed contract. The
two-stream **MRCP + FBCNet** model was the only neural candidate to pass every
predeclared advancement check.

| Model | BA mean (SD) | Sensitivity | Specificity | AUROC |
|---|---:|---:|---:|---:|
| **MRCP + FBCNet** | **62.66% (1.01)** | **31.29%** | **94.03%** | **0.804** |
| ShallowConvNet | 62.63% (0.80) | 30.45% | 94.81% | 0.782 |
| Strict target-only Conv1D | 60.63% (0.72) | 27.49% | 93.78% | 0.746 |
| EEGNet | 58.92% (0.53) | 24.28% | 93.55% | 0.763 |
| FBCNet | 56.94% (1.05) | 20.34% | 93.54% | 0.683 |

MRCP + FBCNet improved pooled balanced accuracy by **+2.03 points** over
target-only Conv1D (95% seed-level t interval **+1.08 to +2.97**), improved the
equal-participant mean by **+3.10 points** (95% participant-bootstrap interval
**+1.50 to +4.73**), and helped **76.3%** of the 38 observed participants.
ShallowConvNet was nearly tied in pooled accuracy and had slightly higher
specificity, but its gain was concentrated in OpenBMI and failed the PhysioNet
and Liu advancement checks.

This is an exploratory development-observed result, not locked clinical
validation. The full report and five presentation figures are in
`artifacts/eeg_decoder_model_bakeoff/`.

## Local Evidence

| Existing model | BA | Sensitivity | Specificity | AUROC | Interpretation |
|---|---:|---:|---:|---:|---|
| WAY-scale-matched random CNN, five-seed mean | 53.49% | 15.60% | 91.37% | 0.610 | Paired SSL control; not strict target-only preprocessing |
| WAY-pretrained temporal CNN, five-seed mean | 56.29% | 19.59% | 93.00% | 0.683 | Pretraining helps, but sensitivity remains low |
| Strict target-only Conv1D, completed bakeoff | 60.63% | 27.49% | 93.78% | 0.746 | Corrected development-only normalization baseline |
| MRCP + FBCNet, completed bakeoff | 62.66% | 31.29% | 94.03% | 0.804 | Only candidate passing every advancement check |
| MRCP branch, translational v3 | 60.28% | 25.44% | 95.13% | 0.778 | Strongest individual intent evidence |
| Riemannian branch, translational v3 | 57.90% | 21.27% | 94.52% | 0.721 | Stable covariance representation |
| FBCSP branch, translational v3 | 52.89% | 11.56% | 94.22% | 0.676 | Weak intent gate, useful conditional direction evidence |
| MRCP/SMR/Riemann late fusion | 60.11% | 22.99% | 97.24% | 0.804 | Best population safety-oriented system |

The older translational-v3 rows use a different established artifact and remain contextual rather than head-to-head evidence. The completed bakeoff rows use one identical split, normalization, training, and threshold contract.

## Research Findings

### 1. Riemannian covariance remains the strongest non-neural anchor

The 2024 MOABB reproducibility study reimplemented 30 pipelines across 36 EEG datasets, including 14 motor-imagery datasets. Riemannian covariance approaches were generally superior, and deep neural methods required substantially more data to become competitive. This argues against removing the existing Riemannian branch.

**Project use:** retain shrinkage covariance, tangent-space logistic regression, and subject alignment as the non-neural reference and likely ensemble member.

### 2. FBCNet is the best next motor-imagery-specific neural candidate

FBCNet applies multiple frequency views, learns spatial filters, and aggregates temporal information with a variance layer. This closely parallels filter-bank CSP while allowing supervised spatial features. Its evaluation included BCI Competition IV-2a, OpenBMI, and two chronic-stroke datasets, making it unusually relevant to this project.

**Project use:** adapt FBCNet to the existing 7-35 Hz SMR branch and add a separate MRCP stream rather than asking FBCNet alone to detect slow cortical intent.

### 3. EEGNet is the mandatory compact neural baseline

EEGNet uses depthwise spatial and separable temporal convolutions and was explicitly tested in both within-subject and cross-subject settings across MRCP and SMR paradigms. It is compact enough for the available data and future real-time deployment. The project already has an implementation, but it has not been compared against the five-seed WAY experiment under the exact current target contract.

**Project use:** run the existing EEGNet as an architecture-matched five-seed baseline before introducing more complex attention models.

### 4. ShallowConvNet is more appropriate than DeepConvNet initially

The ShallowConvNet design approximates log-band-power/FBCSP processing with temporal convolution, spatial convolution, squaring, pooling, and logarithmic activation. The original work found cropped training important for competitive EEG decoding. DeepConvNet has more capacity and a higher overfitting risk under cross-subject clinical shift.

**Project use:** include ShallowConvNet in the first bakeoff; defer DeepConvNet unless it wins controller-only validation without losing specificity.

### 5. Contrastive pretraining is a stronger follow-up than reconstruction alone

A 2024 cross-subject motor-imagery study combined convolution and attention with self-supervised contrastive learning and reported improvements on three public MI datasets. The present masked-reconstruction result is already positive across five seeds, but reconstruction can emphasize waveform detail that is unrelated to intent.

**Project use:** after selecting the best backbone, compare masked reconstruction against physiology-preserving contrastive learning. Suitable augmentations include channel dropout, time masking, small amplitude scaling, low noise, and limited frequency-band dropout. Channel permutation and time reversal should not be used because they can destroy laterality and temporal physiology.

### 6. Transformers and attention networks are second-stage candidates

EEG Conformer combines convolutional local features with self-attention for global dependencies. ATCNet and related temporal-attention models are promising for motor imagery, but their added capacity raises data, tuning, and calibration requirements. Published headline accuracies frequently use within-subject left/right classification and cannot be compared directly with this project's cross-subject intent-versus-rest safety task.

**Project use:** evaluate a compact Conformer only after EEGNet, ShallowConvNet, and FBCNet. It should advance only if it improves participant-level effects and safety metrics, not merely pooled accuracy.

### 7. Transfer and calibration may matter more than architecture

Euclidean and Riemannian alignment studies show that reducing subject-domain mismatch can improve transfer before classification. A healthy-to-stroke study reported 71.15% mean MI detection after domain adaptation across 71 stroke patients, but its protocol and endpoint differ from this project. The important lesson is that a stronger backbone does not remove the need for stroke-specific adaptation.

**Project use:** apply alignment using development or patient-calibration data only. Keep controller and development-observed participants isolated from gradient training.

### 8. Conventional SVM and tree ensembles are not priority candidates

The project's controlled compact-feature ablation already found regularized logistic regression superior to RBF-SVM, gradient boosting, random forest, and a decision tree on controller patients. These models remain useful checks, but changing only the classifier is less promising than improving physiological representation and subject alignment.

## Controlled Bakeoff Design (Completed)

### Tier 1: completed

1. Current random temporal CNN.
2. Current WAY-pretrained temporal CNN.
3. EEGNet from random initialization.
4. ShallowConvNet from random initialization.
5. FBCNet on the SMR filter bank.
6. Two-stream MRCP plus FBCNet model.
7. Riemannian tangent-space logistic regression as the deterministic anchor.
8. Existing MRCP/SMR/Riemann late fusion as the system reference.

### Tier 2: only after Tier 1

1. Contrastive WAY EEG pretraining on the winning neural backbone.
2. Compact EEG Conformer or ATCNet.
3. Alignment-aware or domain-adversarial fine-tuning.
4. Five-seed probability ensemble only if the gain survives single-seed deployment analysis.

## Fixed Evaluation Contract

- Use the same PhysioNet, OpenBMI, and Liu windows and subject roles.
- Fit gradients only on `development` subjects.
- Use `controller_validation` subjects for early stopping, model choice, calibration, and dataset-specific thresholds.
- Evaluate `development_observed` once after the candidate set and operating rule are frozen.
- Run every neural model with the same five seeds: 42-46.
- Report mean and SD across seeds plus a paired seed interval against the current CNN.
- Report equal-participant effects and each dataset separately.
- Report balanced accuracy, sensitivity, specificity, AUROC, AUPRC, calibration error, and abstention.
- Do not claim false activations per minute until continuous no-control recordings are available.
- Match parameter budgets where practical and report parameter count, training time, and inference latency.

## Advancement Rules

A model should advance only if all of the following hold:

1. Mean balanced accuracy exceeds the matched random CNN in at least four of five seeds.
2. The paired seed-level interval for balanced-accuracy change excludes zero.
3. Mean specificity is not reduced by more than one point versus the matched neural control.
4. OpenBMI and PhysioNet mean effects are positive.
5. Liu does not show a clinically concerning sensitivity collapse.
6. Participant-level improvement is not driven by only a few high-performing subjects.

## Selected Forward Architecture

The completed result supports a multi-branch rather than monolithic classifier. The forward architecture is:

> MRCP proposal signal + FBCNet/SMR evidence + Riemannian covariance safety evidence + calibrated late fusion + patient adaptation.

This keeps the interpretable physiological branches that already outperform the generic CNN while testing whether EEG-specific neural spatial and frequency learning adds transferable information.

## Primary Sources

1. [Chevallier et al., largest MOABB reproducibility benchmark, 2024](https://arxiv.org/abs/2404.15319)
2. [Lawhern et al., EEGNet, 2018](https://pubmed.ncbi.nlm.nih.gov/29932424/)
3. [Mane et al., FBCNet](https://arxiv.org/abs/2104.01233)
4. [Schirrmeister et al., ShallowConvNet and DeepConvNet, 2017](https://pubmed.ncbi.nlm.nih.gov/28782865/)
5. [Song et al., EEG Conformer, 2023](https://pubmed.ncbi.nlm.nih.gov/37015413/)
6. [Li et al., self-supervised contrastive cross-subject MI, 2024](https://pubmed.ncbi.nlm.nih.gov/38565100/)
7. [He and Wu, Euclidean alignment for BCI transfer](https://arxiv.org/abs/1808.05464)
8. [Zanini et al., Riemannian transfer framework](https://pubmed.ncbi.nlm.nih.gov/28841546/)
9. [Nagarajan et al., healthy-to-stroke deep transfer, 2024](https://pubmed.ncbi.nlm.nih.gov/38091617/)
10. [Xie et al., cross-dataset MI pretraining, 2023](https://pubmed.ncbi.nlm.nih.gov/37774694/)
