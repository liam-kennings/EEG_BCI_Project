# Traditional Morlet ERD Intent Models

This experiment trains only on the baseline-normalized Morlet ERD and C3-C4 lateralization features produced by the feature audit. It excludes metadata and amplitude-quality columns, so the classifiers cannot depend on subject identifiers, run identifiers, or residual artifact magnitude.

The locked protocol uses 68 available development subjects for fitting, controller-validation subjects on Run 12 for threshold selection, and 22 untouched final patients on Run 12 for evaluation. The safety threshold was chosen to target a 10% controller false-activation rate.

| Model | Final balanced accuracy | Final false activation | Final missed intent | Final ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Shrinkage LDA | 62.4% | 19.7% | 55.5% | 0.676 |
| Logistic regression | 63.0% | 20.7% | 53.3% | 0.675 |
| RBF-SVM | 60.9% | 21.7% | 56.5% | 0.666 |

Logistic regression is the strongest overall traditional baseline, but it is not deployment-ready: the controller cohort threshold did not hold its 10% false-activation target on the final patients. The good result here is scientific rather than clinical: the visualized mu/beta ERD features do contain generalizable motor-intent information, and a nonlinear SVM did not improve on the simpler interpretable models.

Saved outputs:

- [Model comparison](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_intent_models/morlet_intent_model_comparison.png)
- [Logistic feature weights](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_intent_models/logistic_erd_feature_weights.png)
- [Full metrics](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_intent_models/morlet_intent_model_metrics.json)
- [Saved pipelines](/Users/LiamKennings_1/Documents/BME%20473/artifacts/morlet_intent_models/morlet_intent_models.joblib)

The next justified improvement is patient-specific calibration of the Morlet feature distribution, then temporal evidence confirmation on continuous raw windows. It should not be another larger global classifier, because the final results show that threshold calibration, not model capacity, is the limiting failure mode.
