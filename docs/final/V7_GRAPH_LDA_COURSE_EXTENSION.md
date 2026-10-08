# V7 Graph, LDA, and Representation-Diagnostic Extension

## Purpose

This V7-only experiment translated the strongest low-risk additions from the
BME 473 lecture decks into the existing participant-isolated motor-intent
benchmark. It did not modify V3 through V6, subject roles, the calibration
recording, the later-recording evaluation set, neural seeds, or safety
thresholds.

## Added Methods

### Fixed-geometry graph signal processing

The shared motor montage `FC3 FC4 C3 Cz C4 CP3 CP4` is represented as a fixed
graph using standard 10-10 electrode coordinates. A symmetric three-neighbor
RBF adjacency matrix and its graph Laplacian are frozen before EEG is loaded.
No EEG value, participant, or class label defines an edge.

Each MRCP time bin and each SMR filter-bank time bin yields:

- normalized graph smoothness;
- low graph-frequency energy fraction;
- middle graph-frequency energy fraction; and
- high graph-frequency energy fraction.

The resulting branch contains 64 features. Channel-order invariance and graph
connectivity are covered by unit tests.

### Weighted shrinkage LDA

The LDA comparison concatenates MRCP, SMR, Riemannian, and graph features. It
uses median imputation, sample-weighted scaling, dataset-balanced sample
weights, class means, and an Oracle Approximating Shrinkage covariance. A
population head is blended with dataset-specific heads.

### Diagnostic visualization

PCA scaling and components are fitted on development participants only and
later recordings are projected without refitting. K-means is also fitted on
development participants only. It is used to inspect domain and class
structure, never to issue a prediction.

## Controller Guard

The original patient adapter remains an eligible candidate. A graph-augmented
adapter must improve both controller sensitivity and controller balanced
accuracy by at least one percentage point. Otherwise V7 falls back to the
simpler adapter. This prevents a small controller fluctuation from adding a
weak high-dimensional branch.

## Results

The final protected V7 result was unchanged:

| Endpoint | Result |
|---|---:|
| Balanced accuracy | 73.6% |
| Intent sensitivity | 53.5% |
| Rest specificity | 93.7% |
| AUROC | 0.896 |
| Evaluation windows | 4,645 |

Candidate results on the development-observed role were:

| Candidate | Balanced accuracy | AUROC |
|---|---:|---:|
| Fixed graph branch | 55.5% | 0.688 |
| Graph late fusion | 61.4% | 0.815 |
| Weighted shrinkage LDA | 62.8% | 0.803 |
| MRCP + FBCNet seed ensemble | 66.9% | retained in final fusion |

The controller did not assign graph or LDA evidence to the final population
fusion. PhysioNet and OpenBMI directly selected the base patient adapter. The
Liu controller showed only +0.63 sensitivity points and +0.31 balanced-accuracy
points for graph augmentation, below the one-point guard, so the base adapter
was retained. This avoided the later-recording sensitivity loss seen in the
unrestricted ablation.

## Interpretation

This is a useful negative result. Fixed graph features reveal domain-dependent
spatial structure and shrinkage LDA supplies an interpretable baseline, but
neither improved the validated command path. The PCA overlap and K-means
composition further show that the graph representation does not form clean,
universal REST and INTENT clusters across datasets.

The operational recommendation is therefore to keep the original V7 fusion
and patient adapter while retaining graph/LDA outputs for physiology,
harmonization, and future-cohort audits. A full graph neural network is not
justified by this seven-electrode result.

## Outputs

- `artifacts/gui_experiments/translational_v7_graph_lda/TRANSLATIONAL_V7_AUXILIARY_DIRECTION_REPORT.md`
- `artifacts/gui_experiments/translational_v7_graph_lda/10_v7_graph_signal_audit.png`
- `artifacts/gui_experiments/translational_v7_graph_lda/11_v7_graph_representation_diagnostics.png`
- `artifacts/gui_experiments/translational_v7_graph_lda/translational_v7_auxiliary_direction_metrics.json`
