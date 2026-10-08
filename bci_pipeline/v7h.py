"""Quality-aware V7-H EEG-fNIRS intent fusion and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .mind_dataset import MINDHybridFeatureDataset


V7H_MODES = (
    "eeg_only",
    "fnirs_only",
    "late_fusion_general",
    "late_fusion_patient_adapted",
    "sequential_confirmation",
)


def _probability_model(feature_count: int, seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=min(128, feature_count)),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        ),
    )


def _calibration_model(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        ),
    )


def _positive_probability(model: object, features: np.ndarray) -> np.ndarray:
    classes = model.classes_.tolist()
    return model.predict_proba(features)[:, classes.index(1)]


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _fusion_features(
    eeg_probability: np.ndarray,
    fnirs_probability: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    eeg = np.asarray(eeg_probability, dtype=np.float64)
    fnirs = np.asarray(fnirs_probability, dtype=np.float64)
    return np.column_stack(
        (
            _logit(eeg),
            _logit(fnirs),
            np.asarray(quality, dtype=np.float64),
            eeg * fnirs,
            np.abs(eeg - fnirs),
        )
    )


@dataclass
class V7HProbabilities:
    eeg: np.ndarray
    fnirs: np.ndarray
    fusion_general: np.ndarray
    fusion_adapted: np.ndarray
    fnirs_available: np.ndarray


@dataclass
class PatientProbabilityCalibrator:
    eeg_model: object | None
    fnirs_model: object | None
    fusion_model: object | None

    @staticmethod
    def _transform(model: object | None, probability: np.ndarray) -> np.ndarray:
        if model is None:
            return probability
        return _positive_probability(model, _logit(probability)[:, None])

    def transform(
        self,
        eeg: np.ndarray,
        fnirs: np.ndarray,
        fusion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self._transform(self.eeg_model, eeg),
            self._transform(self.fnirs_model, fnirs),
            self._transform(self.fusion_model, fusion),
        )


class V7HIntentModel:
    """Population EEG/fNIRS heads with cross-fitted late fusion."""

    def __init__(
        self,
        *,
        seed: int = 42,
        patient_blend_weight: float = 0.30,
        quality_rejection_quantile: float = 0.05,
    ):
        if not 0.0 <= patient_blend_weight <= 1.0:
            raise ValueError("patient_blend_weight must be between zero and one.")
        if not 0.0 <= quality_rejection_quantile < 0.5:
            raise ValueError("quality_rejection_quantile must be in [0, 0.5).")
        self.seed = int(seed)
        self.patient_blend_weight = float(patient_blend_weight)
        self.quality_rejection_quantile = float(quality_rejection_quantile)

    def fit(self, dataset: MINDHybridFeatureDataset) -> "V7HIntentModel":
        labels = np.asarray(dataset.labels, dtype=np.int64)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("V7-H population training requires REST and INTENT rows.")
        unique_subjects = np.unique(dataset.subjects)
        if len(unique_subjects) < 3:
            raise ValueError("V7-H cross-fitted fusion requires at least 3 subjects.")
        folds = min(5, len(unique_subjects))
        splitter = GroupKFold(n_splits=folds)
        oof_eeg = np.zeros(len(labels), dtype=np.float64)
        oof_fnirs = np.zeros(len(labels), dtype=np.float64)
        for fold, (train, validation) in enumerate(
            splitter.split(dataset.eeg_features, labels, dataset.subjects)
        ):
            eeg_model = _probability_model(dataset.eeg_features.shape[1], self.seed + fold)
            fnirs_model = _probability_model(
                dataset.fnirs_features.shape[1],
                self.seed + 100 + fold,
            )
            eeg_model.fit(dataset.eeg_features[train], labels[train])
            fnirs_model.fit(dataset.fnirs_features[train], labels[train])
            oof_eeg[validation] = _positive_probability(
                eeg_model,
                dataset.eeg_features[validation],
            )
            oof_fnirs[validation] = _positive_probability(
                fnirs_model,
                dataset.fnirs_features[validation],
            )
        self.fusion_model_ = _calibration_model(self.seed + 200).fit(
            _fusion_features(oof_eeg, oof_fnirs, dataset.fnirs_quality),
            labels,
        )
        self.eeg_model_ = _probability_model(
            dataset.eeg_features.shape[1], self.seed
        ).fit(dataset.eeg_features, labels)
        self.fnirs_model_ = _probability_model(
            dataset.fnirs_features.shape[1], self.seed + 100
        ).fit(dataset.fnirs_features, labels)
        self.quality_threshold_ = float(
            np.quantile(dataset.fnirs_quality, self.quality_rejection_quantile)
        )
        self.patient_calibrators_: dict[str, PatientProbabilityCalibrator] = {}
        self.training_subjects_ = tuple(sorted(unique_subjects.astype(str).tolist()))
        return self

    def _general_probabilities(
        self,
        dataset: MINDHybridFeatureDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eeg = _positive_probability(self.eeg_model_, dataset.eeg_features)
        fnirs = _positive_probability(self.fnirs_model_, dataset.fnirs_features)
        fusion = _positive_probability(
            self.fusion_model_,
            _fusion_features(eeg, fnirs, dataset.fnirs_quality),
        )
        return eeg, fnirs, fusion

    @staticmethod
    def _fit_patient_head(
        probability: np.ndarray,
        labels: np.ndarray,
        seed: int,
    ) -> object | None:
        if len(probability) < 6 or len(np.unique(labels)) != 2:
            return None
        return _calibration_model(seed).fit(_logit(probability)[:, None], labels)

    def fit_patient_calibrators(
        self,
        calibration: MINDHybridFeatureDataset,
    ) -> "V7HIntentModel":
        eeg, fnirs, fusion = self._general_probabilities(calibration)
        for offset, subject in enumerate(np.unique(calibration.subjects)):
            selected = calibration.subjects == subject
            labels = calibration.labels[selected]
            self.patient_calibrators_[str(subject)] = PatientProbabilityCalibrator(
                eeg_model=self._fit_patient_head(
                    eeg[selected], labels, self.seed + 300 + offset * 3
                ),
                fnirs_model=self._fit_patient_head(
                    fnirs[selected], labels, self.seed + 301 + offset * 3
                ),
                fusion_model=self._fit_patient_head(
                    fusion[selected], labels, self.seed + 302 + offset * 3
                ),
            )
        return self

    def predict(self, dataset: MINDHybridFeatureDataset) -> V7HProbabilities:
        eeg, fnirs, fusion = self._general_probabilities(dataset)
        adapted_eeg = eeg.copy()
        adapted_fnirs = fnirs.copy()
        adapted_fusion = fusion.copy()
        for subject in np.unique(dataset.subjects):
            calibrator = self.patient_calibrators_.get(str(subject))
            if calibrator is None:
                continue
            selected = dataset.subjects == subject
            local_eeg, local_fnirs, local_fusion = calibrator.transform(
                eeg[selected],
                fnirs[selected],
                fusion[selected],
            )
            weight = self.patient_blend_weight
            adapted_eeg[selected] = (1.0 - weight) * eeg[selected] + weight * local_eeg
            adapted_fnirs[selected] = (
                (1.0 - weight) * fnirs[selected] + weight * local_fnirs
            )
            adapted_fusion[selected] = (
                (1.0 - weight) * fusion[selected] + weight * local_fusion
            )
        available = dataset.fnirs_quality >= self.quality_threshold_
        fallback_fusion = np.where(available, adapted_fusion, adapted_eeg)
        return V7HProbabilities(
            eeg=adapted_eeg,
            fnirs=adapted_fnirs,
            fusion_general=fusion,
            fusion_adapted=fallback_fusion,
            fnirs_available=available,
        )


def split_subjects(
    subjects: np.ndarray,
    *,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    unique = np.unique(np.asarray(subjects).astype(str))
    if len(unique) < 9:
        raise ValueError("V7-H requires at least 9 subjects for development/validation/test.")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    test_count = max(2, int(round(0.20 * len(shuffled))))
    validation_count = max(2, int(round(0.20 * len(shuffled))))
    if test_count + validation_count > len(shuffled) - 3:
        validation_count = max(2, len(shuffled) - test_count - 3)
    return {
        "development": tuple(sorted(shuffled[: -validation_count - test_count])),
        "controller_validation": tuple(
            sorted(shuffled[-validation_count - test_count : -test_count])
        ),
        "locked_internal_test": tuple(sorted(shuffled[-test_count:])),
    }


def calibration_and_evaluation_masks(
    dataset: MINDHybridFeatureDataset,
    subjects: tuple[str, ...],
    *,
    rest_budget: int = 10,
    intent_budget: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve recording 0 calibration and later recordings for evaluation."""
    calibration = np.zeros(len(dataset.labels), dtype=bool)
    subject_set = set(subjects)
    for subject in subjects:
        local = (dataset.subjects == subject) & (dataset.sessions == 0)
        rest = np.flatnonzero(local & (dataset.labels == 0))[:rest_budget]
        intent = np.flatnonzero(local & (dataset.labels == 1))[:intent_budget]
        calibration[np.concatenate((rest, intent))] = True
    evaluation = np.asarray(
        [subject in subject_set for subject in dataset.subjects],
        dtype=bool,
    ) & (dataset.sessions > 0)
    return calibration, evaluation


def _rates(labels: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    rest = labels == 0
    intent = labels == 1
    specificity = float(np.mean(~predicted[rest])) if np.any(rest) else 0.0
    sensitivity = float(np.mean(predicted[intent])) if np.any(intent) else 0.0
    return sensitivity, specificity


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_specificity: float,
    available: np.ndarray | None = None,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    usable = np.ones(len(labels), dtype=bool) if available is None else np.asarray(available, dtype=bool)
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predicted = usable & (probabilities >= threshold)
        sensitivity, specificity = _rates(labels, predicted)
        if specificity + 1e-12 >= target_specificity:
            candidates.append((sensitivity, specificity, float(threshold)))
    if candidates:
        return max(candidates, key=lambda row: (row[0], row[1], -row[2]))[2]
    return 0.95


def select_sequential_thresholds(
    labels: np.ndarray,
    eeg_probability: np.ndarray,
    fnirs_probability: np.ndarray,
    available: np.ndarray,
    *,
    target_specificity: float,
) -> tuple[float, float]:
    candidates = []
    for eeg_threshold in np.linspace(0.15, 0.85, 29):
        for fnirs_threshold in np.linspace(0.15, 0.85, 29):
            predicted = (
                np.asarray(available, dtype=bool)
                & (eeg_probability >= eeg_threshold)
                & (fnirs_probability >= fnirs_threshold)
            )
            sensitivity, specificity = _rates(labels, predicted)
            if specificity + 1e-12 >= target_specificity:
                candidates.append(
                    (sensitivity, specificity, float(eeg_threshold), float(fnirs_threshold))
                )
    if not candidates:
        return 0.85, 0.85
    best = max(candidates, key=lambda row: (row[0], row[1], -row[2] - row[3]))
    return best[2], best[3]


def score_binary_mode(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted: np.ndarray,
    *,
    available: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=bool)
    usable = np.ones(len(labels), dtype=bool) if available is None else np.asarray(available, dtype=bool)
    sensitivity, specificity = _rates(labels, predicted)
    return {
        "rows": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_activation_rate_per_window": 1.0 - specificity,
        "missed_intent_rate": 1.0 - sensitivity,
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "abstained_fraction": float(np.mean(~usable)),
        "continuous_false_activations_per_minute": "not_evaluated",
    }


def _bootstrap_balanced_accuracy(
    labels: np.ndarray,
    predicted: np.ndarray,
    subjects: np.ndarray,
    *,
    seed: int,
    iterations: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    estimates = []
    for _ in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indexes = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
        if len(np.unique(labels[indexes])) == 2:
            estimates.append(balanced_accuracy_score(labels[indexes], predicted[indexes]))
    if not estimates:
        return float("nan"), float("nan")
    return tuple(np.percentile(estimates, (2.5, 97.5)).astype(float))


def evaluate_v7h(
    dataset: MINDHybridFeatureDataset,
    *,
    seed: int = 42,
    target_specificity: float = 0.93,
    calibration_rest: int = 10,
    calibration_intent: int = 20,
) -> tuple[V7HIntentModel, dict[str, object], dict[str, np.ndarray]]:
    """Run population development, patient calibration, and locked internal test."""
    roles = split_subjects(dataset.subjects, seed=seed)
    development_mask = np.isin(dataset.subjects, roles["development"])
    development = dataset.subset(development_mask)
    model = V7HIntentModel(seed=seed).fit(development)

    validation_calibration, validation_evaluation = calibration_and_evaluation_masks(
        dataset,
        roles["controller_validation"],
        rest_budget=calibration_rest,
        intent_budget=calibration_intent,
    )
    model.fit_patient_calibrators(dataset.subset(validation_calibration))
    validation = dataset.subset(validation_evaluation)
    validation_probabilities = model.predict(validation)
    thresholds = {
        "eeg_only": select_threshold(
            validation.labels,
            validation_probabilities.eeg,
            target_specificity=target_specificity,
        ),
        "fnirs_only": select_threshold(
            validation.labels,
            validation_probabilities.fnirs,
            target_specificity=target_specificity,
            available=validation_probabilities.fnirs_available,
        ),
        "late_fusion_general": select_threshold(
            validation.labels,
            validation_probabilities.fusion_general,
            target_specificity=target_specificity,
        ),
        "late_fusion_patient_adapted": select_threshold(
            validation.labels,
            validation_probabilities.fusion_adapted,
            target_specificity=target_specificity,
        ),
    }
    thresholds["sequential_confirmation"] = select_sequential_thresholds(
        validation.labels,
        validation_probabilities.eeg,
        validation_probabilities.fnirs,
        validation_probabilities.fnirs_available,
        target_specificity=target_specificity,
    )

    test_calibration, test_evaluation = calibration_and_evaluation_masks(
        dataset,
        roles["locked_internal_test"],
        rest_budget=calibration_rest,
        intent_budget=calibration_intent,
    )
    model.fit_patient_calibrators(dataset.subset(test_calibration))
    test = dataset.subset(test_evaluation)
    probabilities = model.predict(test)
    mode_probabilities: Mapping[str, np.ndarray] = {
        "eeg_only": probabilities.eeg,
        "fnirs_only": probabilities.fnirs,
        "late_fusion_general": probabilities.fusion_general,
        "late_fusion_patient_adapted": probabilities.fusion_adapted,
    }
    metrics: dict[str, dict[str, float | int | str]] = {}
    predictions: dict[str, np.ndarray] = {}
    for mode, values in mode_probabilities.items():
        available = (
            probabilities.fnirs_available if mode == "fnirs_only" else np.ones(len(test.labels), dtype=bool)
        )
        predicted = available & (values >= float(thresholds[mode]))
        predictions[mode] = predicted
        metrics[mode] = score_binary_mode(
            test.labels,
            values,
            predicted,
            available=available,
        )
    eeg_threshold, fnirs_threshold = thresholds["sequential_confirmation"]
    sequential = (
        probabilities.fnirs_available
        & (probabilities.eeg >= eeg_threshold)
        & (probabilities.fnirs >= fnirs_threshold)
    )
    predictions["sequential_confirmation"] = sequential
    sequential_score = np.sqrt(probabilities.eeg * probabilities.fnirs)
    metrics["sequential_confirmation"] = score_binary_mode(
        test.labels,
        sequential_score,
        sequential,
        available=probabilities.fnirs_available,
    )
    for mode in V7H_MODES:
        lower, upper = _bootstrap_balanced_accuracy(
            test.labels,
            predictions[mode],
            test.subjects,
            seed=seed + 500 + V7H_MODES.index(mode),
        )
        metrics[mode]["balanced_accuracy_ci95"] = [lower, upper]

    patient_metrics = []
    final_prediction = predictions["late_fusion_patient_adapted"]
    for subject in np.unique(test.subjects):
        selected = test.subjects == subject
        patient_metrics.append(
            {
                "subject": str(subject),
                **score_binary_mode(
                    test.labels[selected],
                    probabilities.fusion_adapted[selected],
                    final_prediction[selected],
                ),
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "experiment": "v7h_mind_hybrid_intent",
        "endpoint": "binary_rest_vs_intent",
        "development_status": "locked_internal_public_dataset_test",
        "external_clinical_status": "not_evaluated",
        "seed": seed,
        "target_specificity": target_specificity,
        "roles": roles,
        "calibration_budget": {
            "rest_windows": calibration_rest,
            "intent_windows": calibration_intent,
        },
        "quality_threshold": model.quality_threshold_,
        "thresholds": thresholds,
        "metrics": metrics,
        "patient_metrics": patient_metrics,
        "dataset": {
            "rows": int(len(dataset.labels)),
            "subjects": int(len(np.unique(dataset.subjects))),
            "eeg_features": int(dataset.eeg_features.shape[1]),
            "fnirs_features": int(dataset.fnirs_features.shape[1]),
            "rest_rows": int(np.sum(dataset.labels == 0)),
            "intent_rows": int(np.sum(dataset.labels == 1)),
            "test_rows": int(len(test.labels)),
        },
        "caveats": [
            "MIND is a healthy young-adult cohort and is not external stroke validation.",
            "The derivative profile lacks raw optical scalp-coupling indices and short-separation regression.",
            "False activations per minute require a future continuous no-control recording.",
            "fNIRS confirmation latency is represented by a six-second feature window.",
        ],
    }
    arrays = {
        "labels": test.labels,
        "subjects": test.subjects,
        "sessions": test.sessions,
        "quality": test.fnirs_quality,
        "eeg_probability": probabilities.eeg,
        "fnirs_probability": probabilities.fnirs,
        "fusion_general_probability": probabilities.fusion_general,
        "fusion_adapted_probability": probabilities.fusion_adapted,
        "fnirs_available": probabilities.fnirs_available,
        "sequential_score": sequential_score,
        **{f"prediction_{mode}": values for mode, values in predictions.items()},
    }
    return model, report, arrays
