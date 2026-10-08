# BME 473 Course Concepts and EEG BCI Project Mapping

## Purpose

This guide connects the six supplied BME 473/ELE 573 lecture decks to the
current weak motor-intent EEG project. It distinguishes among concepts already
implemented, controlled side experiments, diagnostic tools, and possible
future methods.

The development-observed V7 result remains unchanged. A lecture concept is not
evidence of improved performance until it is evaluated with the same
participant-isolated development, validation, calibration, and later-recording
test contract.

The strongest low-risk additions from the lectures have now been tested in V7:
a fixed-geometry graph-signal branch, weighted OAS shrinkage LDA, and
development-fitted PCA and K-means diagnostics. Controller validation rejected
the graph branch from the final population fusion and every patient adapter,
so the protected V7 result remains 73.6% balanced accuracy. Shrinkage LDA
provided a useful interpretable reference at 62.8% raw BA and 0.803 AUROC.

## The Complete Mental Model

The course material forms one connected workflow:

```text
raw EEG
  -> quality control, referencing, filtering, and baseline definition
  -> time, frequency, spatial, covariance, or graph representations
  -> feature scaling and optional dimensionality reduction
  -> classifier or neural decoder trained by empirical risk minimization
  -> validation-selected probability calibration and operating threshold
  -> evaluation on unseen people, sessions, or locked external data
  -> safe REST / INTENT decision with abstention or temporal confirmation
```

The main lesson is that the mathematical objective used during training is not
the clinical objective. Minimizing training loss is useful only when it
produces reliable decisions on unseen patients while controlling false
activation during rest.

## 1. Time-Frequency Power and Baseline Normalization

### Core idea

EEG power decreases approximately according to a 1/f relationship. Raw low
frequency power is therefore much larger than raw high frequency power, making
direct visual or statistical comparisons misleading. A baseline transforms
task-related power into a relative change from a physiologically meaningful
reference period.

Common normalizations include:

```text
dB change       = 10 log10(P_task / P_baseline)
percent change  = 100 (P_task - P_baseline) / P_baseline
z score         = (P_task - mean(P_baseline)) / std(P_baseline)
```

### Project relevance

Motor imagery is commonly associated with event-related desynchronization
(ERD), meaning a reduction of mu or beta power relative to baseline. The
project already contains cue-locked baselines, Morlet time-frequency power,
Welch power spectral density, and ERD-style relative features.

Relevant implementation areas include:

- `bci_pipeline/epochs.py`: pre-cue baseline extraction and centering;
- `bci_pipeline/time_frequency_features.py`: complex Morlet power in mu and
  beta bands;
- `bci_pipeline/classical_eeg_features.py`: Welch PSD and spectral statistics;
- `scripts/08_audit_morlet_erd_features.py`: baseline-normalized Morlet/ERD
  audit; and
- `scripts/29_psd_morlet_fusion_ablation.py`: controlled PSD plus Morlet
  comparison.

### Interpretation rules

- The baseline must end before the analyzed response to avoid temporal
  leakage, especially after smoothing or long wavelet convolution.
- The same baseline rule must be applied to development and held-out data.
- Color limits can make a weak effect appear large or hide a real effect.
  Topographic and spectrogram figures must report their scale.
- More trials improve the signal-to-noise ratio of averaged ERD, but trial
  averaging cannot be used to claim single-window controller performance.

## 2. Hilbert Transform and Analytic Signals

After bandpass filtering, the Hilbert transform creates an analytic signal:

```text
z(t) = x(t) + j H{x(t)} = A(t) exp(j phi(t))
```

This provides instantaneous amplitude, power, phase, and a band envelope. The
Hilbert transform shifts frequency components by 90 degrees. It does not create
frequency specificity by itself, so the signal must be bandpass filtered before
interpreting a mu- or beta-band envelope.

Hilbert envelope and phase descriptors are included in
`bci_pipeline/classical_eeg_features.py`. Explicit MRCP phase features were
also evaluated in `scripts/41_mrcp_phase_ablation.py`.

Its appropriate role is a controlled complementary feature family, not an
assumed replacement for Morlet power. Morlet convolution provides explicit
time-frequency resolution across several frequencies, while a
bandpass-Hilbert branch provides a compact band envelope and phase estimate.

## 3. Common Spatial Pattern and FBCSP

Common Spatial Pattern (CSP) is a supervised spatial filter for two classes. It
finds projections with high variance for one class and low variance for the
other. One form of its objective is:

```text
J(w) = (w^T C1 w) / (w^T (C1 + C2) w)
```

Solving the generalized eigenvalue problem yields spatial filters. Filters at
both ends of the eigenvalue spectrum emphasize opposite classes.
Log-normalized projected variances become classifier features.

Filter-bank CSP (FBCSP) repeats the process across frequency bands. This is
important because each patient can express sensorimotor rhythms at slightly
different frequencies.

The project implementation in `bci_pipeline/models/fbcsp.py` follows the class
algorithm:

1. Bandpass each training epoch in a filter bank.
2. Calculate trace-normalized class covariance matrices.
3. Solve a regularized generalized eigenvalue problem.
4. Retain low- and high-eigenvalue spatial filters.
5. Calculate log-normalized projected variance.
6. Fit a scaled logistic classifier.

The hierarchical version first classifies REST versus INTENT and then fits an
optional left-versus-right direction head on movement trials.

### Leakage rule

CSP is supervised. Its covariance matrices and filters must be fitted only on
training participants inside each split. Fitting CSP once on the complete
dataset before cross-validation would expose test labels and inflate results.
The current `fit` and `transform` separation supports the correct pattern.

## 4. LDA, Bayesian Classification, and Calibration

### Linear Discriminant Analysis

Fisher Linear Discriminant Analysis (LDA) finds a projection that increases
between-class separation while reducing within-class spread:

```text
J(w) = [w^T (mu1 - mu2)]^2 / (w^T Sw w)
w* proportional to Sw^-1 (mu1 - mu2)
```

LDA differs from principal component analysis (PCA):

- PCA preserves directions of greatest overall variance without labels.
- LDA preserves directions that discriminate known classes.

LDA remains a strong BCI baseline because it is fast, interpretable, and often
stable when the feature count is controlled. Shrinkage is important when the
covariance estimate is poorly conditioned or the feature count is large
relative to the number of independent participants.

### Bayesian classification

Bayes' rule combines a likelihood and a prior:

```text
P(class | X) = P(X | class) P(class) / P(X)
```

Maximum a posteriori classification selects the class with the largest
posterior. Naive Bayes assumes conditional feature independence. EEG features
are often correlated, so this assumption is rarely physiologically exact, but
Naive Bayes can still serve as a compact diagnostic baseline.

### Project relevance

V7 uses discriminative models and then performs Platt-style probability
calibration in `bci_pipeline/translational_decoder.py`. Calibration asks
whether a predicted probability matches the observed event frequency. V7
evaluates this with Brier score and expected calibration error in
`bci_pipeline/translational_metrics.py`.

The threshold is selected after calibration because a rehabilitation
controller needs an operating point, not only a class ranking. The conservative
threshold prioritizes rest specificity; a sensitivity-targeted threshold
accepts more false activation to recover more intent.

## 5. K-Means, PCA, t-SNE, and UMAP

K-means is an unsupervised clustering method. Given `K` centroids, it alternates
between assigning each point to the nearest centroid and replacing each
centroid with the mean of its assigned points. It minimizes within-cluster
squared Euclidean distance:

```text
min sum_k sum_{x_i in cluster k} ||x_i - mu_k||^2
```

The objective decreases at each alternating step, but the solution can depend
on initialization and can become trapped in a local optimum. Euclidean
K-means also favors approximately spherical clusters.

- PCA is linear and emphasizes global variance.
- t-SNE emphasizes preservation of local neighborhoods for visualization.
- UMAP is another nonlinear neighborhood embedding method.

These methods are best used for exploratory diagnostics in this project:

- visualize whether datasets form separate domains;
- identify gross artifact or quality clusters;
- inspect whether calibration windows shift toward later recordings;
- inspect representation drift between healthy and stroke cohorts; and
- summarize unsupervised quality groups.

They should not be used to claim class separability without a held-out decoder
test. A visually separated t-SNE plot is not an accuracy result, and its axes
have no direct physiological meaning. Embeddings and cluster centroids must be
fitted on development data if they affect preprocessing, rejection, or the
decoder.

## 6. Neural Networks, ERM, and Optimization

A multilayer perceptron or convolutional network applies weighted linear
transformations followed by nonlinear activation functions. Hidden nonlinear
layers can represent relations that a single linear boundary cannot.
Backpropagation computes gradients of the loss with respect to each parameter.

Supervised training commonly minimizes empirical risk:

```text
L(theta) = (1/N) sum_i loss(y_i, f_theta(x_i))
```

A regularized objective is:

```text
f(theta) = L(theta) + lambda r(theta)
```

Gradient descent updates parameters according to:

```text
theta_(k+1) = theta_k - h_k gradient f(theta_k)
```

The step size `h_k` matters. A large value can diverge or oscillate; a small
value can train very slowly. Stochastic and mini-batch gradient descent use
subsets of samples to estimate the full gradient.

For a convex objective, a stationary point is a global optimum. Neural network
training is generally nonconvex, so different random initializations can
produce different results. This is why the project evaluates five neural seeds
rather than relying on one favorable run.

Training loss can continue to decrease while held-out performance worsens. The
course validation material supports the project's use of:

- subject-isolated train, controller-validation, and later-recording roles;
- early stopping for neural models;
- validation-selected fusion weights and thresholds;
- multiple random seeds;
- participant-level bootstrap confidence intervals; and
- a separate locked external cohort when available.

Early stopping can act as implicit regularization by stopping before the model
overfits the development set.

## 7. Graph Signal Processing

Conventional signal processing assumes a regular structure such as equally
spaced time samples or an image grid. EEG electrodes instead form an irregular
spatial network. Graph signal processing (GSP) represents electrodes as
vertices, spatial relationships as weighted edges, and voltage or band power
at each electrode as a graph signal.

For weighted adjacency matrix `W` and degree matrix `D`, the graph Laplacian is:

```text
L = D - W
```

Its quadratic form measures graph roughness:

```text
x^T L x = (1/2) sum_ij W_ij (x_i - x_j)^2
```

If `L = U Lambda U^T`, the graph Fourier transform is:

```text
x_hat = U^T x
x = U x_hat
```

Small-eigenvalue eigenvectors vary smoothly across connected electrodes and
are interpreted as low graph frequencies. Large-eigenvalue components
represent rapid spatial changes. Spectral graph wavelets add localization in
graph position and graph scale. Graph neural networks learn feature
aggregation and transformation over neighboring nodes.

### Why it could help EEG

The current Riemannian covariance branch represents inter-electrode spatial
structure but does not explicitly encode electrode geometry. GSP could add:

- graph-smoothness or graph-total-variation features;
- low-, mid-, and high-graph-frequency energy in mu/beta power maps;
- graph spectral filtering before classification;
- spatially localized graph wavelet coefficients; or
- a compact graph convolutional encoder.

### Recommended first GSP experiment

Start with an interpretable feature ablation, not a new deep GNN:

1. Use the seven shared motor electrodes as a fixed graph.
2. Define edge weights from standard 10-10 sensor distances, not class labels.
3. Keep the graph identical across datasets after channel harmonization.
4. For MRCP, mu, and beta representations, calculate graph smoothness and
   graph spectral energy in predeclared eigenvalue ranges.
5. Add these features to the existing SMR statistical branch.
6. Select inclusion using controller-validation subjects only.
7. Evaluate once on the unchanged later-recording role and report paired
   participant effects, sensitivity, specificity, AUROC, and calibration.

This test is lower risk than a full-montage GNN and remains interpretable. A
full-montage graph model can follow if the seven-node ablation is beneficial.

### GSP leakage and robustness rules

- Do not estimate connectivity from held-out participants.
- Do not construct graph edges using class labels from evaluation data.
- If covariance or coherence defines the graph, fit it on development subjects
  only and freeze it before calibration and evaluation.
- Verify invariance to channel ordering.
- Recompute the graph contract explicitly when a dataset lacks electrodes.
- Compare against a parameter-matched non-graph control.

## 8. Direct Mapping to the Project

| Course concept | Project use | Status |
|---|---|---|
| Signal filtering | Separate MRCP, SMR, and quality branches | Implemented |
| Baseline normalization | Cue-locked ERD/Morlet physiology audits | Implemented in audits, not a primary V7 branch |
| Hilbert transform | Envelope and phase feature experiments | Controlled side experiment |
| CSP | FBCSP spatial filters and log variance | Implemented comparison baseline, not the V7 final head |
| Covariance and eigendecomposition | CSP and Riemannian tangent features | Implemented |
| LDA | Weighted OAS shrinkage baseline | Implemented in V7; controller rejected from final fusion |
| Bayesian probability | Calibrated probability interpretation | Implemented through calibration metrics |
| K-means | Artifact/domain exploration | Implemented as V7 diagnostic only |
| PCA/t-SNE/UMAP | Representation and domain-shift visualization | PCA implemented as V7 diagnostic only |
| Neural networks | FBCNet and MRCP-FBCNet seed ensembles | Implemented |
| ERM and regularization | Loss-based fitting and early stopping | Implemented |
| Nonconvex optimization | Five-seed stability requirement | Implemented |
| Validation | Participant isolation and controller-only selection | Implemented |
| GSP | Fixed seven-electrode graph spectral features | Implemented V7 ablation; rejected operationally |

## 9. Recommended Priorities

1. Protect the validation contract. No feature, graph, embedding, scaler, CSP
   filter, or threshold should learn from held-out participants.
2. Standardize baseline-normalized physiology. Use the same pre-cue baseline
   definition for PSD, Hilbert, Morlet, and graph features.
3. Retain the completed graph/LDA ablation as a negative result. Do not add its
   features to the command path unless a future cohort clears the predeclared
   one-point controller gain guard.
4. Keep shrinkage LDA as the fast interpretable comparison and preserve
   MRCP-FBCNet/Riemannian/SMR fusion as the validated population model.
5. Continue using clustering only for quality and domain analysis. The current
   diagnostic does not show clean unsupervised REST/INTENT clusters; stronger
   alignment with dataset identity would indicate remaining harmonization
   limits.

## 10. Presentation-Safe Explanation

> EEG is first cleaned and separated into slow movement-preparation and faster
> sensorimotor-rhythm branches. Time-frequency, spatial, and covariance
> features expose weak motor-related changes before machine learning. Models
> are trained by minimizing loss, but selected using unseen development
> participants because low training loss does not guarantee patient
> generalization. A short calibration adjusts probabilities for one patient,
> and a conservative threshold converts those probabilities into a binary REST
> or INTENT decision while prioritizing rejection of false commands.

## 11. Glossary

| Term | Meaning in this project |
|---|---|
| Analytic signal | Complex signal produced from a real signal and its Hilbert transform |
| AUROC | Area under the receiver operating characteristic curve; threshold-independent ranking of intent above rest |
| Baseline | Pre-cue reference interval used to express task-related change |
| BCI | Brain-computer interface |
| CAR | Common average reference |
| CSP | Common Spatial Pattern, a supervised spatial variance filter |
| Decoder | Model and decision logic mapping EEG representations to probabilities or commands |
| ERD | Event-related desynchronization, usually a task-related power reduction |
| ERM | Empirical risk minimization |
| FBCSP | Filter-bank Common Spatial Pattern |
| FBCNet | Neural architecture combining temporal filter banks and spatial variance structure |
| GCN | Graph convolutional network |
| GFT | Graph Fourier transform |
| GNN | Graph neural network |
| GSP | Graph signal processing |
| Hilbert envelope | Time-varying amplitude of a band-limited analytic signal |
| LDA | Linear Discriminant Analysis |
| MAP | Maximum a posteriori decision rule |
| MRCP | Movement-related cortical potential, a slow cortical change preceding or accompanying movement intent |
| OOD | Out of distribution |
| PCA | Principal component analysis |
| PSD | Power spectral density |
| Riemannian features | Geometry-aware features from positive-definite EEG covariance matrices |
| SGWT | Spectral graph wavelet transform |
| SMR | Sensorimotor rhythm, primarily mu and beta activity over motor cortex |
| t-SNE | Nonlinear neighborhood-preserving visualization method |
| UMAP | Nonlinear manifold visualization and embedding method |

## Source Lecture Decks

- `GSP_Slides.pdf`: graph Laplacians, graph spectra, GFT, graph convolution,
  spectral clustering, graph wavelets, spectral filters, and GNNs.
- `k-Means.pdf`: unsupervised clustering, Euclidean distance, alternating
  optimization, initialization, and local optima.
- `Lecture 8.pdf`: baseline-normalized time-frequency power, SNR, Hilbert
  transforms, analytic signals, and CSP.
- `Lecture-9.pdf`: LDA, Bayesian and Naive Bayes classifiers, multilayer neural
  networks, backpropagation, and validation.
- `Machine_Learning_BME_473 2.pdf`: machine-learning task types, loss, ERM,
  generalization, neural networks, PCA, t-SNE, and UMAP context.
- `optimization.pdf`: first-order optimization, convexity, regularization,
  gradient methods, step size, initialization, and validation during training.
