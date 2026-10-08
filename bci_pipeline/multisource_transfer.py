"""Domain-specific fusion of EEG-only students trained with privileged sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .privileged_transfer import EMGConfirmedEEGTransfer


COMPONENT_NAMES = ("target_only", "mrcp_student", "way_student")
GLOBAL_DOMAIN = "__global__"


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a non-empty finite two-dimensional array.")
    return matrix


def _binary(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Fusion labels must be a one-dimensional binary array.")
    return labels


def logit_component_features(probabilities: np.ndarray) -> np.ndarray:
    """Convert component probabilities to bounded log-odds for calibration."""
    values = _finite_matrix(probabilities, "component probabilities")
    if values.shape[1] != len(COMPONENT_NAMES):
        raise ValueError(
            f"Expected {len(COMPONENT_NAMES)} component probabilities per row."
        )
    clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def _stacker(seed: int) -> Pipeline:
    return Pipeline(
        (
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        )
    )


def fit_cross_fitted_stacker(
    component_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[Pipeline, np.ndarray]:
    """Fit a final stacker and return group-held probabilities for selection."""
    features = logit_component_features(component_probabilities)
    target = _binary(labels)
    group_values = np.asarray(groups).astype(str)
    if len(group_values) != len(target):
        raise ValueError("Fusion groups must be row-aligned.")
    unique_groups = np.unique(group_values)
    held_out = np.empty(len(target), dtype=np.float64)
    if len(unique_groups) >= 2:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        folds = splitter.split(features, target, group_values)
    else:
        class_count = min(int(np.sum(target == 0)), int(np.sum(target == 1)))
        if class_count < 2:
            raise ValueError("Fusion needs at least two examples per class.")
        splitter = StratifiedKFold(
            n_splits=min(5, class_count),
            shuffle=True,
            random_state=seed,
        )
        folds = splitter.split(features, target)
    for fold, (train, test) in enumerate(folds):
        if len(np.unique(target[train])) != 2:
            raise ValueError("Every fusion training fold must contain both classes.")
        model = _stacker(seed + fold).fit(features[train], target[train])
        held_out[test] = model.predict_proba(features[test])[:, 1]
    final_model = _stacker(seed + 100).fit(features, target)
    return final_model, held_out


@dataclass(frozen=True)
class DomainFusionRule:
    method: str
    threshold: float
    controller_balanced_accuracy: float
    controller_sensitivity: float
    controller_specificity: float

    def validate(self) -> None:
        if self.method not in (*COMPONENT_NAMES, "stacked"):
            raise ValueError(f"Unknown fusion method: {self.method}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Fusion thresholds must be probabilities.")


class MultiSourceEEGEnsemble:
    """EEG-only intent ensemble with dataset-specific calibrated policies."""

    def __init__(
        self,
        *,
        mrcp_model: EMGConfirmedEEGTransfer,
        way_model: EMGConfirmedEEGTransfer,
        stackers: Mapping[str, Pipeline],
        rules: Mapping[str, DomainFusionRule],
        feature_names: Sequence[str],
    ) -> None:
        self.mrcp_model = mrcp_model
        self.way_model = way_model
        self.stackers = dict(stackers)
        self.rules = dict(rules)
        self.feature_names = tuple(str(value) for value in feature_names)
        if GLOBAL_DOMAIN not in self.rules:
            raise ValueError("A global fallback fusion rule is required.")
        for domain, rule in self.rules.items():
            rule.validate()
            if rule.method == "stacked" and domain not in self.stackers:
                raise ValueError(f"Stacked rule for {domain} has no fitted stacker.")
        if self.mrcp_model.feature_names_ != self.feature_names:
            raise ValueError("MRCP student features do not match the ensemble contract.")
        if self.way_model.feature_names_ != self.feature_names:
            raise ValueError("WAY student features do not match the ensemble contract.")

    def component_probabilities(self, eeg_features: np.ndarray) -> np.ndarray:
        """Return target-only, MRCP-assisted, and WAY-assisted EEG scores."""
        features = _finite_matrix(eeg_features, "EEG features")
        if features.shape[1] != len(self.feature_names):
            raise ValueError("EEG feature count does not match the ensemble contract.")
        target = self.mrcp_model.predict_target_only_probability(features)
        mrcp = self.mrcp_model.predict_intent_probability(features)
        way = self.way_model.predict_intent_probability(features)
        return np.column_stack((target, mrcp, way))

    def _domain_rule(self, domain: str) -> tuple[str, DomainFusionRule]:
        key = domain if domain in self.rules else GLOBAL_DOMAIN
        return key, self.rules[key]

    def predict_intent_probability(
        self,
        eeg_features: np.ndarray,
        dataset_ids: np.ndarray,
    ) -> np.ndarray:
        """Predict intent from EEG and a target-domain identifier only."""
        components = self.component_probabilities(eeg_features)
        domains = np.asarray(dataset_ids).astype(str)
        if domains.ndim != 1 or len(domains) != len(components):
            raise ValueError("Dataset identifiers must be row-aligned with EEG features.")
        result = np.empty(len(domains), dtype=np.float64)
        for domain in np.unique(domains):
            selected = domains == domain
            key, rule = self._domain_rule(str(domain))
            if rule.method in COMPONENT_NAMES:
                result[selected] = components[selected, COMPONENT_NAMES.index(rule.method)]
            else:
                result[selected] = self.stackers[key].predict_proba(
                    logit_component_features(components[selected])
                )[:, 1]
        return result

    def threshold_vector(self, dataset_ids: np.ndarray) -> np.ndarray:
        domains = np.asarray(dataset_ids).astype(str)
        if domains.ndim != 1:
            raise ValueError("Dataset identifiers must be one-dimensional.")
        return np.asarray(
            [self._domain_rule(domain)[1].threshold for domain in domains],
            dtype=np.float64,
        )

    def predict_intent(
        self,
        eeg_features: np.ndarray,
        dataset_ids: np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_intent_probability(eeg_features, dataset_ids)
        return (probabilities >= self.threshold_vector(dataset_ids)).astype(np.int64)
