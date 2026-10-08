"""Domain-aware hierarchical branch decoders and patient calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.covariance import oas
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .openbmi_controlled import StageProbabilities


def dataset_balanced_weights(
    labels: np.ndarray,
    domains: np.ndarray,
) -> np.ndarray:
    """Give each dataset and binary intent class equal aggregate training mass."""
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    weights = np.ones(len(labels), dtype=np.float64)
    unique_domains = np.unique(domains)
    for domain in unique_domains:
        domain_mask = domains == domain
        weights[domain_mask] *= len(labels) / (
            len(unique_domains) * int(np.sum(domain_mask))
        )
    binary = labels != 0
    for value in (False, True):
        selected = binary == value
        mass = float(np.sum(weights[selected]))
        if mass:
            weights[selected] *= float(np.sum(weights)) / (2.0 * mass)
    return weights


def _model(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        ),
    )


def _multiclass_model(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        ),
    )


class WeightedShrinkageLDA:
    """Binary LDA with weighted scaling and OAS covariance shrinkage."""

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedShrinkageLDA":
        values = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or len(values) != len(labels):
            raise ValueError("LDA features and labels must be row-aligned.")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("WeightedShrinkageLDA requires binary labels.")
        weights = (
            np.ones(len(labels), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        if weights.shape != labels.shape or np.any(weights < 0.0):
            raise ValueError("LDA sample weights must be non-negative and row-aligned.")
        if float(np.sum(weights)) <= 0.0:
            raise ValueError("LDA sample weights must have positive mass.")

        self.imputer_ = SimpleImputer(strategy="median", add_indicator=True)
        imputed = self.imputer_.fit_transform(values)
        self.scaler_ = StandardScaler().fit(imputed, sample_weight=weights)
        transformed = self.scaler_.transform(imputed)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.means_ = np.stack(
            [
                np.average(
                    transformed[labels == value],
                    axis=0,
                    weights=weights[labels == value],
                )
                for value in self.classes_
            ]
        )
        residuals = transformed - self.means_[labels]
        normalized_weights = weights * len(weights) / np.sum(weights)
        weighted_residuals = residuals * np.sqrt(normalized_weights[:, None])
        covariance, self.shrinkage_ = oas(
            weighted_residuals,
            assume_centered=True,
        )
        jitter = 1e-8 * max(float(np.trace(covariance)), 1.0) / len(covariance)
        covariance = covariance + jitter * np.eye(len(covariance))
        difference = self.means_[1] - self.means_[0]
        self.coef_ = np.linalg.solve(covariance, difference)
        class_mass = np.asarray(
            [np.sum(weights[labels == value]) for value in self.classes_],
            dtype=np.float64,
        )
        priors = class_mass / np.sum(class_mass)
        self.intercept_ = float(
            -0.5 * (self.means_[1] + self.means_[0]) @ self.coef_
            + np.log(np.clip(priors[1], 1e-12, None))
            - np.log(np.clip(priors[0], 1e-12, None))
        )
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        imputed = self.imputer_.transform(np.asarray(features, dtype=np.float64))
        transformed = self.scaler_.transform(imputed)
        return transformed @ self.coef_ + self.intercept_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = np.clip(self.decision_function(features), -40.0, 40.0)
        positive = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - positive, positive))


class DomainAwareShrinkageLDAIntentDecoder:
    """Blend population and dataset-specific weighted shrinkage-LDA heads."""

    def __init__(self, domain_weight: float = 0.5):
        if not 0.0 <= domain_weight <= 1.0:
            raise ValueError("domain_weight must be between zero and one.")
        self.domain_weight = domain_weight

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DomainAwareShrinkageLDAIntentDecoder":
        values = np.asarray(features)
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        binary = (labels != 0).astype(np.int64)
        if set(np.unique(binary)) != {0, 1}:
            raise ValueError("Shrinkage LDA requires rest and motor-intent labels.")
        weights = (
            dataset_balanced_weights(labels, domains)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        self.global_ = WeightedShrinkageLDA().fit(values, binary, weights)
        self.domains_: dict[str, WeightedShrinkageLDA] = {}
        for domain in np.unique(domains):
            selected = domains == domain
            if set(np.unique(binary[selected])) == {0, 1}:
                self.domains_[domain] = WeightedShrinkageLDA().fit(
                    values[selected],
                    binary[selected],
                    weights[selected],
                )
        return self

    def predict_intent_probabilities(
        self,
        features: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(features)
        domains = np.asarray(domains).astype(str)
        intent = self.global_.predict_proba(values)[:, 1]
        for domain in np.unique(domains):
            selected = domains == domain
            if domain in self.domains_:
                local = self.domains_[domain].predict_proba(values[selected])[:, 1]
                intent[selected] = (
                    (1.0 - self.domain_weight) * intent[selected]
                    + self.domain_weight * local
                )
        return intent


class DomainAwareHierarchicalDecoder:
    """Blend a shared population model with dataset-specific model heads."""

    def __init__(self, seed: int = 42, domain_weight: float = 0.5):
        if not 0.0 <= domain_weight <= 1.0:
            raise ValueError("domain_weight must be between zero and one.")
        self.seed = seed
        self.domain_weight = domain_weight

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DomainAwareHierarchicalDecoder":
        features = np.asarray(features)
        labels = np.asarray(labels)
        domains = np.asarray(domains)
        movement = labels != 0
        if set(np.unique(labels)) != {0, 1, 2}:
            raise ValueError("Training requires rest, left, and right labels.")
        weights = (
            dataset_balanced_weights(labels, domains)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        self.intent_global_ = _model(self.seed).fit(
            features,
            movement.astype(np.int64),
            logisticregression__sample_weight=weights,
        )
        self.direction_global_ = _model(self.seed + 1).fit(
            features[movement],
            labels[movement],
            logisticregression__sample_weight=weights[movement],
        )
        self.intent_domains_: dict[str, object] = {}
        self.direction_domains_: dict[str, object] = {}
        for offset, domain in enumerate(np.unique(domains), start=1):
            local = domains == domain
            if len(np.unique(movement[local])) == 2:
                self.intent_domains_[str(domain)] = _model(
                    self.seed + 10 * offset
                ).fit(
                    features[local],
                    movement[local].astype(np.int64),
                    logisticregression__sample_weight=weights[local],
                )
            local_movement = local & movement
            if set(np.unique(labels[local_movement])) == {1, 2}:
                self.direction_domains_[str(domain)] = _model(
                    self.seed + 10 * offset + 1
                ).fit(
                    features[local_movement],
                    labels[local_movement],
                    logisticregression__sample_weight=weights[local_movement],
                )
        return self

    @staticmethod
    def _positive(model: object, features: np.ndarray, label: int) -> np.ndarray:
        classes = model.classes_.tolist()
        return model.predict_proba(features)[:, classes.index(label)]

    @staticmethod
    def _direction(model: object, features: np.ndarray) -> np.ndarray:
        classes = model.classes_.tolist()
        values = model.predict_proba(features)
        return np.column_stack(
            (
                values[:, classes.index(1)],
                values[:, classes.index(2)],
            )
        )

    def predict_stage_probabilities(
        self,
        features: np.ndarray,
        domains: np.ndarray,
    ) -> StageProbabilities:
        features = np.asarray(features)
        domains = np.asarray(domains)
        intent = self._positive(self.intent_global_, features, 1)
        direction = self._direction(self.direction_global_, features)
        for domain in np.unique(domains):
            selected = domains == domain
            key = str(domain)
            if key in self.intent_domains_:
                local = self._positive(
                    self.intent_domains_[key],
                    features[selected],
                    1,
                )
                intent[selected] = (
                    (1.0 - self.domain_weight) * intent[selected]
                    + self.domain_weight * local
                )
            if key in self.direction_domains_:
                local_direction = self._direction(
                    self.direction_domains_[key],
                    features[selected],
                )
                direction[selected] = (
                    (1.0 - self.domain_weight) * direction[selected]
                    + self.domain_weight * local_direction
                )
        return StageProbabilities(intent=intent, direction=direction)


class DomainAwareIntentDecoder:
    """Blend shared and dataset-specific binary rest-versus-intent heads."""

    def __init__(self, seed: int = 42, domain_weight: float = 0.5):
        if not 0.0 <= domain_weight <= 1.0:
            raise ValueError("domain_weight must be between zero and one.")
        self.seed = seed
        self.domain_weight = domain_weight

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DomainAwareIntentDecoder":
        features = np.asarray(features)
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        binary = (labels != 0).astype(np.int64)
        if set(np.unique(binary)) != {0, 1}:
            raise ValueError("Intent training requires rest and motor-intent labels.")
        weights = (
            dataset_balanced_weights(labels, domains)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        self.intent_global_ = _model(self.seed).fit(
            features,
            binary,
            logisticregression__sample_weight=weights,
        )
        self.intent_domains_: dict[str, object] = {}
        for offset, domain in enumerate(np.unique(domains), start=1):
            local = domains == domain
            if set(np.unique(binary[local])) == {0, 1}:
                self.intent_domains_[domain] = _model(
                    self.seed + 10 * offset
                ).fit(
                    features[local],
                    binary[local],
                    logisticregression__sample_weight=weights[local],
                )
        return self

    @staticmethod
    def _positive(model: object, features: np.ndarray) -> np.ndarray:
        classes = model.classes_.tolist()
        return model.predict_proba(features)[:, classes.index(1)]

    def predict_intent_probabilities(
        self,
        features: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        features = np.asarray(features)
        domains = np.asarray(domains).astype(str)
        intent = self._positive(self.intent_global_, features)
        for domain in np.unique(domains):
            selected = domains == domain
            if domain in self.intent_domains_:
                local = self._positive(
                    self.intent_domains_[domain],
                    features[selected],
                )
                intent[selected] = (
                    (1.0 - self.domain_weight) * intent[selected]
                    + self.domain_weight * local
                )
        return intent


class DomainAwareAuxiliaryIntentDecoder:
    """Collapse a jointly trained rest/left/right model into binary intent."""

    def __init__(self, seed: int = 42, domain_weight: float = 0.5):
        if not 0.0 <= domain_weight <= 1.0:
            raise ValueError("domain_weight must be between zero and one.")
        self.seed = seed
        self.domain_weight = domain_weight

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "DomainAwareAuxiliaryIntentDecoder":
        features = np.asarray(features)
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        if set(np.unique(labels)) != {0, 1, 2}:
            raise ValueError(
                "Auxiliary intent training requires rest, left, and right labels."
            )
        weights = (
            dataset_balanced_weights(labels, domains)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        self.global_ = _multiclass_model(self.seed).fit(
            features,
            labels,
            logisticregression__sample_weight=weights,
        )
        self.domains_: dict[str, object] = {}
        for offset, domain in enumerate(np.unique(domains), start=1):
            local = domains == domain
            if set(np.unique(labels[local])) == {0, 1, 2}:
                self.domains_[domain] = _multiclass_model(
                    self.seed + 10 * offset
                ).fit(
                    features[local],
                    labels[local],
                    logisticregression__sample_weight=weights[local],
                )
        return self

    @staticmethod
    def _intent(model: object, features: np.ndarray) -> np.ndarray:
        classes = model.classes_.tolist()
        probabilities = model.predict_proba(features)
        return 1.0 - probabilities[:, classes.index(0)]

    def predict_intent_probabilities(
        self,
        features: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        features = np.asarray(features)
        domains = np.asarray(domains).astype(str)
        intent = self._intent(self.global_, features)
        for domain in np.unique(domains):
            selected = domains == domain
            if domain in self.domains_:
                local = self._intent(self.domains_[domain], features[selected])
                intent[selected] = (
                    (1.0 - self.domain_weight) * intent[selected]
                    + self.domain_weight * local
                )
        return intent


@dataclass
class BranchEnsemble:
    """Equal-weight late fusion of calibrated logistic branch probabilities."""

    branch_names: tuple[str, ...]
    models: dict[str, DomainAwareHierarchicalDecoder]
    seed: int
    domain_weight: float

    @classmethod
    def fit(
        cls,
        branches: Mapping[str, np.ndarray],
        labels: np.ndarray,
        domains: np.ndarray,
        *,
        branch_names: Sequence[str] = ("mrcp", "smr", "riemann"),
        seed: int = 42,
        domain_weight: float = 0.5,
    ) -> "BranchEnsemble":
        selected = tuple(branch_names)
        models = {}
        weights = dataset_balanced_weights(labels, domains)
        for offset, name in enumerate(selected):
            if name not in branches:
                raise KeyError(f"Missing decoder feature branch: {name}")
            models[name] = DomainAwareHierarchicalDecoder(
                seed=seed + 100 * offset,
                domain_weight=domain_weight,
            ).fit(branches[name], labels, domains, weights)
        return cls(selected, models, seed, domain_weight)

    def predict_stage_probabilities(
        self,
        branches: Mapping[str, np.ndarray],
        domains: np.ndarray,
    ) -> StageProbabilities:
        values = [
            self.models[name].predict_stage_probabilities(branches[name], domains)
            for name in self.branch_names
        ]
        return StageProbabilities(
            intent=np.mean([value.intent for value in values], axis=0),
            direction=np.mean([value.direction for value in values], axis=0),
        )


@dataclass
class IntentBranchEnsemble:
    """Equal-weight ensemble containing no left-versus-right model."""

    branch_names: tuple[str, ...]
    models: dict[str, DomainAwareIntentDecoder]
    seed: int
    domain_weight: float

    @classmethod
    def fit(
        cls,
        branches: Mapping[str, np.ndarray],
        labels: np.ndarray,
        domains: np.ndarray,
        *,
        branch_names: Sequence[str] = ("mrcp", "smr", "riemann"),
        seed: int = 42,
        domain_weight: float = 0.5,
    ) -> "IntentBranchEnsemble":
        selected = tuple(branch_names)
        models = {}
        weights = dataset_balanced_weights(labels, domains)
        for offset, name in enumerate(selected):
            if name not in branches:
                raise KeyError(f"Missing decoder feature branch: {name}")
            models[name] = DomainAwareIntentDecoder(
                seed=seed + 100 * offset,
                domain_weight=domain_weight,
            ).fit(branches[name], labels, domains, weights)
        return cls(selected, models, seed, domain_weight)

    def predict_intent_probabilities(
        self,
        branches: Mapping[str, np.ndarray],
        domains: np.ndarray,
    ) -> np.ndarray:
        values = [
            self.models[name].predict_intent_probabilities(
                branches[name],
                domains,
            )
            for name in self.branch_names
        ]
        return np.mean(values, axis=0)


@dataclass
class AuxiliaryIntentBranchEnsemble:
    """Ensemble of joint rest/left/right models exposed as binary intent."""

    branch_names: tuple[str, ...]
    models: dict[str, DomainAwareAuxiliaryIntentDecoder]
    seed: int
    domain_weight: float

    @classmethod
    def fit(
        cls,
        branches: Mapping[str, np.ndarray],
        labels: np.ndarray,
        domains: np.ndarray,
        *,
        branch_names: Sequence[str] = ("mrcp", "smr", "riemann"),
        seed: int = 42,
        domain_weight: float = 0.5,
    ) -> "AuxiliaryIntentBranchEnsemble":
        selected = tuple(branch_names)
        models = {}
        weights = dataset_balanced_weights(labels, domains)
        for offset, name in enumerate(selected):
            if name not in branches:
                raise KeyError(f"Missing decoder feature branch: {name}")
            models[name] = DomainAwareAuxiliaryIntentDecoder(
                seed=seed + 100 * offset,
                domain_weight=domain_weight,
            ).fit(branches[name], labels, domains, weights)
        return cls(selected, models, seed, domain_weight)

    def predict_intent_probabilities(
        self,
        branches: Mapping[str, np.ndarray],
        domains: np.ndarray,
    ) -> np.ndarray:
        values = [
            self.models[name].predict_intent_probabilities(
                branches[name],
                domains,
            )
            for name in self.branch_names
        ]
        return np.mean(values, axis=0)


def _weighted_stage_probabilities(
    probabilities: Mapping[str, StageProbabilities],
    weights: Mapping[str, float],
    stage: str,
) -> np.ndarray:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Fusion weights must have positive mass.")
    values = [
        float(weight) * getattr(probabilities[name], stage)
        for name, weight in weights.items()
    ]
    return np.sum(values, axis=0) / total


def _intent_fusion_candidates(
    probabilities: Mapping[str, StageProbabilities],
) -> tuple[dict[str, float], ...]:
    candidates = (
        {"mrcp": 1.0},
        {"mrcp": 0.75, "riemann": 0.25},
        {"mrcp": 0.60, "riemann": 0.25, "smr": 0.15},
        {"mrcp": 0.50, "riemann": 0.50},
        {"mrcp": 1 / 3, "riemann": 1 / 3, "smr": 1 / 3},
    )
    if "gsp" in probabilities:
        candidates += (
            {"gsp": 1.0},
            {"mrcp": 0.60, "riemann": 0.20, "gsp": 0.20},
            {"mrcp": 0.50, "smr": 0.25, "gsp": 0.25},
            {"graph_late_fusion": 1.0},
        )
    if "shrinkage_lda" in probabilities:
        candidates += (
            {"shrinkage_lda": 1.0},
            {"mrcp": 0.70, "shrinkage_lda": 0.30},
            {"riemann": 0.70, "shrinkage_lda": 0.30},
        )
    if "mrcp_fbcnet" in probabilities:
        candidates += (
            {"mrcp_fbcnet": 1.0},
            {"mrcp_fbcnet": 0.75, "mrcp": 0.25},
            {"mrcp_fbcnet": 0.70, "riemann": 0.30},
            {"mrcp_fbcnet": 0.60, "mrcp": 0.20, "riemann": 0.20},
            {"mrcp_fbcnet": 0.60, "riemann": 0.25, "smr": 0.15},
        )
        if "gsp" in probabilities:
            candidates += (
                {"mrcp_fbcnet": 0.65, "riemann": 0.20, "gsp": 0.15},
                {"mrcp_fbcnet": 0.65, "graph_late_fusion": 0.35},
            )
        if "shrinkage_lda" in probabilities:
            candidates += (
                {"mrcp_fbcnet": 0.70, "shrinkage_lda": 0.30},
                {
                    "mrcp_fbcnet": 0.60,
                    "shrinkage_lda": 0.20,
                    "gsp": 0.20,
                },
            )
    if "shallow_convnet" in probabilities:
        candidates += (
            {"shallow_convnet": 1.0},
            {"shallow_convnet": 0.75, "mrcp": 0.25},
            {"shallow_convnet": 0.70, "riemann": 0.30},
            {"shallow_convnet": 0.60, "mrcp": 0.20, "riemann": 0.20},
        )
    if {"mrcp_fbcnet", "shallow_convnet"}.issubset(probabilities):
        candidates += (
            {"mrcp_fbcnet": 0.50, "shallow_convnet": 0.50},
            {
                "mrcp_fbcnet": 0.50,
                "shallow_convnet": 0.30,
                "riemann": 0.20,
            },
            {
                "mrcp_fbcnet": 0.40,
                "shallow_convnet": 0.40,
                "mrcp": 0.20,
            },
        )
    if "auxiliary_direction_intent" in probabilities:
        candidates += (
            {"auxiliary_direction_intent": 1.0},
            {"mrcp": 0.75, "auxiliary_direction_intent": 0.25},
            {"riemann": 0.75, "auxiliary_direction_intent": 0.25},
        )
        if "mrcp_fbcnet" in probabilities:
            candidates += (
                {
                    "mrcp_fbcnet": 0.60,
                    "auxiliary_direction_intent": 0.25,
                    "riemann": 0.15,
                },
                {
                    "mrcp_fbcnet": 0.60,
                    "auxiliary_direction_intent": 0.20,
                    "smr": 0.20,
                },
            )
    if "graph_auxiliary_direction_intent" in probabilities:
        candidates += (
            {"graph_auxiliary_direction_intent": 1.0},
            {"mrcp": 0.70, "graph_auxiliary_direction_intent": 0.30},
            {"riemann": 0.70, "graph_auxiliary_direction_intent": 0.30},
        )
        if "mrcp_fbcnet" in probabilities:
            candidates += (
                {
                    "mrcp_fbcnet": 0.60,
                    "graph_auxiliary_direction_intent": 0.25,
                    "gsp": 0.15,
                },
            )
    return tuple(
        candidate
        for candidate in candidates
        if set(candidate).issubset(probabilities)
    )


def _mean_domain_score(
    labels: np.ndarray,
    scores: np.ndarray,
    domains: np.ndarray,
    *,
    direction: bool,
) -> float:
    results = []
    for domain in np.unique(domains):
        selected = domains == domain
        local_labels = labels[selected]
        local_scores = scores[selected]
        if direction:
            selected_intent = local_labels != 0
            local_labels = local_labels[selected_intent]
            local_scores = local_scores[selected_intent]
            if set(np.unique(local_labels)) != {1, 2}:
                continue
            predictions = np.argmax(local_scores, axis=1) + 1
            results.append(
                balanced_accuracy_score(local_labels, predictions)
            )
        else:
            binary = local_labels != 0
            if len(np.unique(binary)) != 2:
                continue
            results.append(roc_auc_score(binary, local_scores))
    return float(np.mean(results)) if results else 0.0


@dataclass
class StageSpecificFusion:
    """Use separately selected branch weights for intent and direction."""

    intent_weights: dict[str, float]
    direction_weights: dict[str, float]
    selection_scores: dict[str, float]

    @classmethod
    def select(
        cls,
        probabilities: Mapping[str, StageProbabilities],
        labels: np.ndarray,
        domains: np.ndarray,
        calibration_mask: np.ndarray,
    ) -> "StageSpecificFusion":
        intent_candidates = (
            {"mrcp": 1.0},
            {"mrcp": 0.75, "riemann": 0.25},
            {"mrcp": 0.60, "riemann": 0.25, "smr": 0.15},
            {"mrcp": 0.50, "riemann": 0.50},
            {"mrcp": 1 / 3, "riemann": 1 / 3, "smr": 1 / 3},
        )
        if "mrcp_fbcnet" in probabilities:
            intent_candidates += (
                {"mrcp_fbcnet": 1.0},
                {"mrcp_fbcnet": 0.75, "mrcp": 0.25},
                {"mrcp_fbcnet": 0.70, "riemann": 0.30},
                {"mrcp_fbcnet": 0.60, "mrcp": 0.20, "riemann": 0.20},
                {"mrcp_fbcnet": 0.60, "riemann": 0.25, "smr": 0.15},
            )
        if "shallow_convnet" in probabilities:
            intent_candidates += (
                {"shallow_convnet": 1.0},
                {"shallow_convnet": 0.75, "mrcp": 0.25},
                {"shallow_convnet": 0.70, "riemann": 0.30},
                {"shallow_convnet": 0.60, "mrcp": 0.20, "riemann": 0.20},
            )
        if {"mrcp_fbcnet", "shallow_convnet"}.issubset(probabilities):
            intent_candidates += (
                {"mrcp_fbcnet": 0.50, "shallow_convnet": 0.50},
                {
                    "mrcp_fbcnet": 0.50,
                    "shallow_convnet": 0.30,
                    "riemann": 0.20,
                },
                {
                    "mrcp_fbcnet": 0.40,
                    "shallow_convnet": 0.40,
                    "mrcp": 0.20,
                },
            )
        direction_candidates = (
            {"riemann": 1.0},
            {"fbcsp": 1.0},
            {"riemann": 0.50, "fbcsp": 0.50},
            {"riemann": 0.50, "fbcsp": 0.30, "mrcp": 0.20},
            {"riemann": 0.40, "fbcsp": 0.40, "smr": 0.20},
        )
        selected_probabilities = {
            name: StageProbabilities(
                value.intent[calibration_mask],
                value.direction[calibration_mask],
            )
            for name, value in probabilities.items()
        }
        selected_labels = np.asarray(labels)[calibration_mask]
        selected_domains = np.asarray(domains)[calibration_mask]
        intent_ranked = [
            (
                _mean_domain_score(
                    selected_labels,
                    _weighted_stage_probabilities(
                        selected_probabilities,
                        candidate,
                        "intent",
                    ),
                    selected_domains,
                    direction=False,
                ),
                candidate,
            )
            for candidate in intent_candidates
        ]
        direction_ranked = [
            (
                _mean_domain_score(
                    selected_labels,
                    _weighted_stage_probabilities(
                        selected_probabilities,
                        candidate,
                        "direction",
                    ),
                    selected_domains,
                    direction=True,
                ),
                candidate,
            )
            for candidate in direction_candidates
        ]
        intent_score, intent_weights = max(
            intent_ranked,
            key=lambda item: item[0],
        )
        direction_score, direction_weights = max(
            direction_ranked,
            key=lambda item: item[0],
        )
        return cls(
            intent_weights=dict(intent_weights),
            direction_weights=dict(direction_weights),
            selection_scores={
                "mean_domain_intent_auroc": float(intent_score),
                "mean_domain_direction_balanced_accuracy": float(
                    direction_score
                ),
            },
        )

    def predict(
        self,
        probabilities: Mapping[str, StageProbabilities],
    ) -> StageProbabilities:
        return StageProbabilities(
            intent=_weighted_stage_probabilities(
                probabilities,
                self.intent_weights,
                "intent",
            ),
            direction=_weighted_stage_probabilities(
                probabilities,
                self.direction_weights,
                "direction",
            ),
        )


@dataclass
class IntentSpecificFusion:
    """Select and apply fusion using only binary motor-intent evidence."""

    intent_weights: dict[str, float]
    selection_score: float

    @classmethod
    def select(
        cls,
        probabilities: Mapping[str, StageProbabilities],
        labels: np.ndarray,
        domains: np.ndarray,
        calibration_mask: np.ndarray,
    ) -> "IntentSpecificFusion":
        selected_labels = np.asarray(labels)[calibration_mask]
        selected_domains = np.asarray(domains)[calibration_mask]
        ranked = []
        for candidate in _intent_fusion_candidates(probabilities):
            scores = np.sum(
                [
                    float(weight)
                    * probabilities[name].intent[calibration_mask]
                    for name, weight in candidate.items()
                ],
                axis=0,
            ) / float(sum(candidate.values()))
            ranked.append(
                (
                    _mean_domain_score(
                        selected_labels,
                        scores,
                        selected_domains,
                        direction=False,
                    ),
                    candidate,
                )
            )
        score, weights = max(ranked, key=lambda item: item[0])
        return cls(dict(weights), float(score))

    def predict(
        self,
        probabilities: Mapping[str, StageProbabilities],
    ) -> np.ndarray:
        return _weighted_stage_probabilities(
            probabilities,
            self.intent_weights,
            "intent",
        )


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _calibration_model(
    scores: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    ).fit(_logit(scores), labels)


@dataclass
class DomainProbabilityCalibrator:
    """Platt-calibrate intent and direction scores within each dataset."""

    seed: int = 42

    def fit(
        self,
        probabilities: StageProbabilities,
        labels: np.ndarray,
        domains: np.ndarray,
    ) -> "DomainProbabilityCalibrator":
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        binary = (labels != 0).astype(np.int64)
        self.intent_global_ = _calibration_model(
            probabilities.intent,
            binary,
            self.seed,
        )
        movement = labels != 0
        self.direction_global_ = _calibration_model(
            probabilities.direction[movement, 1],
            (labels[movement] == 2).astype(np.int64),
            self.seed + 1,
        )
        self.intent_domains_: dict[str, LogisticRegression] = {}
        self.direction_domains_: dict[str, LogisticRegression] = {}
        self.intent_blends_: dict[str, float] = {"__global__": 1.0}
        self.direction_blends_: dict[str, float] = {"__global__": 1.0}
        for offset, domain in enumerate(np.unique(domains), start=1):
            local = domains == domain
            if len(np.unique(binary[local])) == 2:
                self.intent_domains_[domain] = _calibration_model(
                    probabilities.intent[local],
                    binary[local],
                    self.seed + 10 * offset,
                )
            local_movement = local & movement
            if set(np.unique(labels[local_movement])) == {1, 2}:
                self.direction_domains_[domain] = _calibration_model(
                    probabilities.direction[local_movement, 1],
                    (labels[local_movement] == 2).astype(np.int64),
                    self.seed + 10 * offset + 1,
                )
        return self

    def select_blends(
        self,
        probabilities: StageProbabilities,
        labels: np.ndarray,
        domains: np.ndarray,
        validation_mask: np.ndarray,
    ) -> "DomainProbabilityCalibrator":
        """Shrink calibration toward identity when it does not transfer."""
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        validation_mask = np.asarray(validation_mask, dtype=bool)
        candidates = np.linspace(0.0, 1.0, 5)
        transformed = self._predict_transformed(probabilities, domains)
        binary = (labels != 0).astype(np.int64)
        movement = labels != 0
        for key, selected in (
            ("__global__", validation_mask),
            *(
                (
                    domain,
                    validation_mask & (domains == domain),
                )
                for domain in np.unique(domains[validation_mask])
            ),
        ):
            self.intent_blends_[str(key)] = min(
                candidates,
                key=lambda blend: brier_score_loss(
                    binary[selected],
                    (1.0 - blend) * probabilities.intent[selected]
                    + blend * transformed.intent[selected],
                ),
            )
            selected_direction = selected & movement
            self.direction_blends_[str(key)] = min(
                candidates,
                key=lambda blend: brier_score_loss(
                    (labels[selected_direction] == 2).astype(np.int64),
                    (1.0 - blend)
                    * probabilities.direction[selected_direction, 1]
                    + blend
                    * transformed.direction[selected_direction, 1],
                ),
            )
        return self

    @staticmethod
    def _calibrated(
        model: LogisticRegression,
        scores: np.ndarray,
    ) -> np.ndarray:
        return model.predict_proba(_logit(scores))[:, 1]

    def _predict_transformed(
        self,
        probabilities: StageProbabilities,
        domains: np.ndarray,
    ) -> StageProbabilities:
        domains = np.asarray(domains).astype(str)
        intent = self._calibrated(
            self.intent_global_,
            probabilities.intent,
        )
        right = self._calibrated(
            self.direction_global_,
            probabilities.direction[:, 1],
        )
        for domain in np.unique(domains):
            selected = domains == domain
            if domain in self.intent_domains_:
                intent[selected] = self._calibrated(
                    self.intent_domains_[domain],
                    probabilities.intent[selected],
                )
            if domain in self.direction_domains_:
                right[selected] = self._calibrated(
                    self.direction_domains_[domain],
                    probabilities.direction[selected, 1],
                )
        return StageProbabilities(
            intent=intent,
            direction=np.column_stack((1.0 - right, right)),
        )

    def predict(
        self,
        probabilities: StageProbabilities,
        domains: np.ndarray,
    ) -> StageProbabilities:
        domains = np.asarray(domains).astype(str)
        transformed = self._predict_transformed(probabilities, domains)
        intent = probabilities.intent.copy()
        right = probabilities.direction[:, 1].copy()
        for domain in np.unique(domains):
            selected = domains == domain
            intent_blend = self.intent_blends_.get(
                domain,
                self.intent_blends_.get("__global__", 1.0),
            )
            direction_blend = self.direction_blends_.get(
                domain,
                self.direction_blends_.get("__global__", 1.0),
            )
            intent[selected] = (
                (1.0 - intent_blend) * probabilities.intent[selected]
                + intent_blend * transformed.intent[selected]
            )
            right[selected] = (
                (1.0 - direction_blend)
                * probabilities.direction[selected, 1]
                + direction_blend * transformed.direction[selected, 1]
            )
        return StageProbabilities(
            intent=intent,
            direction=np.column_stack((1.0 - right, right)),
        )


@dataclass
class DomainIntentProbabilityCalibrator:
    """Platt-calibrate binary intent without fitting a direction model."""

    seed: int = 42

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
    ) -> "DomainIntentProbabilityCalibrator":
        probabilities = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        binary = (labels != 0).astype(np.int64)
        self.intent_global_ = _calibration_model(
            probabilities,
            binary,
            self.seed,
        )
        self.intent_domains_: dict[str, LogisticRegression] = {}
        self.intent_blends_: dict[str, float] = {"__global__": 1.0}
        for offset, domain in enumerate(np.unique(domains), start=1):
            local = domains == domain
            if set(np.unique(binary[local])) == {0, 1}:
                self.intent_domains_[domain] = _calibration_model(
                    probabilities[local],
                    binary[local],
                    self.seed + 10 * offset,
                )
        return self

    @staticmethod
    def _calibrated(
        model: LogisticRegression,
        scores: np.ndarray,
    ) -> np.ndarray:
        return model.predict_proba(_logit(scores))[:, 1]

    def _predict_transformed(
        self,
        probabilities: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        domains = np.asarray(domains).astype(str)
        transformed = self._calibrated(self.intent_global_, probabilities)
        for domain in np.unique(domains):
            selected = domains == domain
            if domain in self.intent_domains_:
                transformed[selected] = self._calibrated(
                    self.intent_domains_[domain],
                    probabilities[selected],
                )
        return transformed

    def select_blends(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        validation_mask: np.ndarray,
    ) -> "DomainIntentProbabilityCalibrator":
        labels = np.asarray(labels)
        domains = np.asarray(domains).astype(str)
        validation_mask = np.asarray(validation_mask, dtype=bool)
        transformed = self._predict_transformed(probabilities, domains)
        binary = (labels != 0).astype(np.int64)
        candidates = np.linspace(0.0, 1.0, 5)
        for key, selected in (
            ("__global__", validation_mask),
            *(
                (domain, validation_mask & (domains == domain))
                for domain in np.unique(domains[validation_mask])
            ),
        ):
            self.intent_blends_[str(key)] = min(
                candidates,
                key=lambda blend: brier_score_loss(
                    binary[selected],
                    (1.0 - blend) * probabilities[selected]
                    + blend * transformed[selected],
                ),
            )
        return self

    def predict(
        self,
        probabilities: np.ndarray,
        domains: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        domains = np.asarray(domains).astype(str)
        transformed = self._predict_transformed(probabilities, domains)
        intent = probabilities.copy()
        for domain in np.unique(domains):
            selected = domains == domain
            blend = self.intent_blends_.get(
                domain,
                self.intent_blends_.get("__global__", 1.0),
            )
            intent[selected] = (
                (1.0 - blend) * probabilities[selected]
                + blend * transformed[selected]
            )
        return intent


class QualityAvailabilityGate:
    """Domain-aware robust quality gate that abstains on unusual windows.

    Scoring version 2 (the default for new fits) measures robust distance only
    over continuous features with nonzero spread in the reference data.
    Indicator features whose reference values are all 0 or 1 (for example
    ``auxiliary_available`` and ``csd_available``) and features that are
    constant in the reference are excluded from the distance. Version 1 floored
    their zero MAD at 1e-6, so any change in them scored as a roughly
    1e6-sigma deviation and the gate rejected every window of the HS-Stroke and
    Cho2017 external cohorts. Missing modalities are a compatibility question,
    reported by :meth:`compatibility_report`, not an artifact.

    When ``recording_groups`` are passed to :meth:`fit`, the gate instead
    scores each window against its own recording's median and spread. This
    label-free mode flags windows that are unusual within a recording and
    does not reject a new dataset only because its amplitude scale differs.

    Gates pickled before version 2 have no ``scoring_version_`` attribute and
    keep version 1 scoring, so frozen artifacts reproduce exactly. Pass
    ``legacy=True`` to fit a new version 1 gate.
    """

    def __init__(self, rejection_quantile: float = 0.99, legacy: bool = False):
        if not 0.5 < rejection_quantile < 1.0:
            raise ValueError("rejection_quantile must be between 0.5 and 1.0.")
        self.rejection_quantile = rejection_quantile
        self.legacy = bool(legacy)

    @property
    def scoring_version(self) -> int:
        return int(getattr(self, "scoring_version_", 1))

    @property
    def within_recording(self) -> bool:
        return bool(getattr(self, "within_recording_", False))

    def fit(
        self,
        quality_features: np.ndarray,
        domains: np.ndarray | None = None,
        recording_groups: np.ndarray | None = None,
    ) -> "QualityAvailabilityGate":
        values = np.asarray(quality_features, dtype=np.float64)
        domain_values = self._domains(values, domains)
        self.scoring_version_ = 1 if getattr(self, "legacy", False) else 2
        if recording_groups is not None and self.scoring_version_ == 1:
            raise ValueError("Within-recording scoring requires scoring version 2.")
        self.within_recording_ = recording_groups is not None
        if self.within_recording_:
            groups = self._groups(values, recording_groups)
            distances = self._within_recording_distances(values, groups)
            self.statistics_ = {}
            self.threshold_ = float(np.quantile(distances, self.rejection_quantile))
            self.domain_thresholds_ = {"__global__": self.threshold_}
            for domain in np.unique(domain_values):
                selected = domain_values == domain
                if np.sum(selected) >= 10:
                    self.domain_thresholds_[str(domain)] = float(
                        np.quantile(distances[selected], self.rejection_quantile)
                    )
            return self
        self.statistics_: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        global_statistics = self._statistics(values)
        self.statistics_["__global__"] = global_statistics
        for domain in np.unique(domain_values):
            selected = domain_values == domain
            if np.sum(selected) >= 10:
                self.statistics_[str(domain)] = self._statistics(values[selected])
        self.threshold_ = global_statistics[2]
        return self

    @staticmethod
    def _domains(values: np.ndarray, domains: np.ndarray | None) -> np.ndarray:
        domain_values = (
            np.full(len(values), "__global__", dtype="<U16")
            if domains is None
            else np.asarray(domains).astype(str)
        )
        if len(domain_values) != len(values):
            raise ValueError("Quality domains must align with feature rows.")
        return domain_values

    @staticmethod
    def _groups(values: np.ndarray, recording_groups: np.ndarray | None) -> np.ndarray:
        if recording_groups is None:
            raise ValueError("This gate was fitted within recordings; pass recording_groups.")
        groups = np.asarray(recording_groups).astype(str)
        if len(groups) != len(values):
            raise ValueError("Recording groups must align with feature rows.")
        return groups

    def _robust_center_scale(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        if self.scoring_version == 1:
            return median, np.clip(1.4826 * mad, 1e-6, None)
        scale = 1.4826 * mad
        # A feature that is mostly constant (e.g. bad-channel fraction) has
        # zero MAD; fall back to its standard deviation. Indicators and truly
        # constant features keep a scale of 0 and are excluded by the caller.
        degenerate = scale <= 1e-12
        scale[degenerate] = np.std(values[:, degenerate], axis=0)
        indicator = np.all((values == 0.0) | (values == 1.0), axis=0)
        scale[indicator] = 0.0
        return median, scale

    @staticmethod
    def _distances(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
        usable = scale > 1e-12
        if not np.any(usable):
            return np.zeros(len(values), dtype=np.float64)
        z = (values[:, usable] - median[usable]) / scale[usable]
        return np.sqrt(np.mean(z**2, axis=1))

    def _statistics(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        median, scale = self._robust_center_scale(values)
        distances = self._distances(values, median, scale)
        threshold = float(np.quantile(distances, self.rejection_quantile))
        return median, scale, threshold

    def _within_recording_distances(
        self,
        values: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        distances = np.empty(len(values), dtype=np.float64)
        for group in np.unique(groups):
            selected = groups == group
            median, scale = self._robust_center_scale(values[selected])
            distances[selected] = self._distances(values[selected], median, scale)
        return distances

    def score(
        self,
        quality_features: np.ndarray,
        domains: np.ndarray | None = None,
        recording_groups: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(quality_features, dtype=np.float64)
        domain_values = self._domains(values, domains)
        if self.within_recording:
            return self._within_recording_distances(
                values,
                self._groups(values, recording_groups),
            )
        scores = np.empty(len(values), dtype=np.float64)
        for domain in np.unique(domain_values):
            selected = domain_values == domain
            median, scale, _ = self.statistics_.get(
                str(domain),
                self.statistics_["__global__"],
            )
            if self.scoring_version == 1:
                scores[selected] = np.sqrt(
                    np.mean(((values[selected] - median) / scale) ** 2, axis=1)
                )
            else:
                scores[selected] = self._distances(values[selected], median, scale)
        return scores

    def available(
        self,
        quality_features: np.ndarray,
        domains: np.ndarray | None = None,
        recording_groups: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(quality_features, dtype=np.float64)
        domain_values = self._domains(values, domains)
        scores = self.score(values, domain_values, recording_groups)
        available = np.empty(len(values), dtype=bool)
        for domain in np.unique(domain_values):
            selected = domain_values == domain
            if self.within_recording:
                threshold = self.domain_thresholds_.get(
                    str(domain),
                    self.domain_thresholds_["__global__"],
                )
            else:
                _, _, threshold = self.statistics_.get(
                    str(domain),
                    self.statistics_["__global__"],
                )
            available[selected] = scores[selected] <= threshold
        return available

    def compatibility_report(
        self,
        quality_features: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Report excluded indicator/constant features that differ from the reference.

        Only meaningful for version 2 gates fitted across recordings. A
        mismatch (for example no auxiliary channels in a new dataset) should
        be documented, not silently turned into abstention.
        """
        if self.scoring_version == 1 or self.within_recording:
            return {"status": "not_applicable"}
        values = np.asarray(quality_features, dtype=np.float64)
        median, scale, _ = self.statistics_["__global__"]
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{index}" for index in range(values.shape[1])]
        )
        excluded = np.flatnonzero(scale <= 1e-12)
        return {
            "status": "evaluated",
            "excluded_features": {
                names[index]: {
                    "reference_value": float(median[index]),
                    "mismatch_fraction": float(np.mean(values[:, index] != median[index])),
                }
                for index in excluded
            },
        }


def population_stage_features(
    probabilities: StageProbabilities,
) -> np.ndarray:
    """Create stable logit features from the general population controller."""
    intent = np.clip(probabilities.intent, 1e-5, 1.0 - 1e-5)
    right = np.clip(probabilities.direction[:, 1], 1e-5, 1.0 - 1e-5)
    return np.column_stack(
        (
            np.log(intent / (1.0 - intent)),
            np.log(right / (1.0 - right)),
        )
    )


def population_intent_features(probabilities: np.ndarray) -> np.ndarray:
    """Create a single stable logit feature from population intent evidence."""
    intent = np.clip(np.asarray(probabilities), 1e-5, 1.0 - 1e-5)
    return np.log(intent / (1.0 - intent)).reshape(-1, 1)


@dataclass
class RapidPatientAdapter:
    """Serializable low-parameter adapter fitted from one calibration session."""

    model: BranchEnsemble
    blend_weight: float
    branch_names: tuple[str, ...]
    calibration_rows: int

    @classmethod
    def fit(
        cls,
        branches: Mapping[str, np.ndarray],
        labels: np.ndarray,
        domain: str,
        *,
        branch_names: Sequence[str],
        blend_weight: float,
        seed: int,
    ) -> "RapidPatientAdapter":
        selected_labels = np.asarray(labels)
        if set(np.unique(selected_labels)) != {0, 1, 2}:
            raise ValueError("Patient calibration requires rest, left, and right labels.")
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("Patient adapter blend_weight must be between zero and one.")
        names = tuple(branch_names)
        model = BranchEnsemble.fit(
            {name: np.asarray(branches[name]) for name in names},
            selected_labels,
            np.full(len(selected_labels), str(domain)),
            branch_names=names,
            seed=seed,
            domain_weight=1.0,
        )
        return cls(model, float(blend_weight), names, len(selected_labels))

    def predict(
        self,
        branches: Mapping[str, np.ndarray],
        domain: str,
        population: StageProbabilities,
    ) -> StageProbabilities:
        rows = len(population.intent)
        local = self.model.predict_stage_probabilities(
            {name: np.asarray(branches[name]) for name in self.branch_names},
            np.full(rows, str(domain)),
        )
        return StageProbabilities(
            intent=(1.0 - self.blend_weight) * population.intent
            + self.blend_weight * local.intent,
            direction=(1.0 - self.blend_weight) * population.direction
            + self.blend_weight * local.direction,
        )


@dataclass
class RapidIntentPatientAdapter:
    """Serializable binary adapter fitted from labeled rest and intent windows."""

    model: IntentBranchEnsemble
    blend_weight: float
    branch_names: tuple[str, ...]
    calibration_rows: int

    @classmethod
    def fit(
        cls,
        branches: Mapping[str, np.ndarray],
        labels: np.ndarray,
        domain: str,
        *,
        branch_names: Sequence[str],
        blend_weight: float,
        seed: int,
    ) -> "RapidIntentPatientAdapter":
        labels = np.asarray(labels)
        binary = (labels != 0).astype(np.int64)
        if set(np.unique(binary)) != {0, 1}:
            raise ValueError("Patient calibration requires rest and intent labels.")
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("Patient adapter blend_weight must be between zero and one.")
        names = tuple(branch_names)
        model = IntentBranchEnsemble.fit(
            {name: np.asarray(branches[name]) for name in names},
            binary,
            np.full(len(binary), str(domain)),
            branch_names=names,
            seed=seed,
            domain_weight=1.0,
        )
        return cls(model, float(blend_weight), names, len(binary))

    def predict(
        self,
        branches: Mapping[str, np.ndarray],
        domain: str,
        population: np.ndarray,
    ) -> np.ndarray:
        population = np.asarray(population, dtype=np.float64)
        local = self.model.predict_intent_probabilities(
            {name: np.asarray(branches[name]) for name in self.branch_names},
            np.full(len(population), str(domain)),
        )
        return (
            (1.0 - self.blend_weight) * population
            + self.blend_weight * local
        )


@dataclass
class RapidAuxiliaryIntentPatientAdapter:
    """Binary adapter regularized by internal rest/left/right discrimination."""

    primary_model: IntentBranchEnsemble
    auxiliary_model: AuxiliaryIntentBranchEnsemble
    blend_weight: float
    auxiliary_weight: float
    branch_names: tuple[str, ...]
    calibration_rows: int

    @classmethod
    def fit(
        cls,
        branches: Mapping[str, np.ndarray],
        labels: np.ndarray,
        domain: str,
        *,
        branch_names: Sequence[str],
        blend_weight: float,
        auxiliary_weight: float,
        seed: int,
    ) -> "RapidAuxiliaryIntentPatientAdapter":
        labels = np.asarray(labels)
        if set(np.unique(labels)) != {0, 1, 2}:
            raise ValueError(
                "Auxiliary patient calibration requires rest, left, and right cues."
            )
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("Patient adapter blend_weight must be between zero and one.")
        if not 0.0 <= auxiliary_weight <= 1.0:
            raise ValueError("auxiliary_weight must be between zero and one.")
        names = tuple(branch_names)
        selected = {name: np.asarray(branches[name]) for name in names}
        domains = np.full(len(labels), str(domain))
        primary = IntentBranchEnsemble.fit(
            selected,
            labels,
            domains,
            branch_names=names,
            seed=seed,
            domain_weight=1.0,
        )
        auxiliary = AuxiliaryIntentBranchEnsemble.fit(
            selected,
            labels,
            domains,
            branch_names=names,
            seed=seed + 500,
            domain_weight=1.0,
        )
        return cls(
            primary,
            auxiliary,
            float(blend_weight),
            float(auxiliary_weight),
            names,
            len(labels),
        )

    def predict(
        self,
        branches: Mapping[str, np.ndarray],
        domain: str,
        population: np.ndarray,
    ) -> np.ndarray:
        population = np.asarray(population, dtype=np.float64)
        selected = {
            name: np.asarray(branches[name]) for name in self.branch_names
        }
        domains = np.full(len(population), str(domain))
        primary = self.primary_model.predict_intent_probabilities(
            selected,
            domains,
        )
        auxiliary = self.auxiliary_model.predict_intent_probabilities(
            selected,
            domains,
        )
        local = (
            (1.0 - self.auxiliary_weight) * primary
            + self.auxiliary_weight * auxiliary
        )
        return (
            (1.0 - self.blend_weight) * population
            + self.blend_weight * local
        )


def patient_adapted_auxiliary_intent_probabilities(
    branches: Mapping[str, np.ndarray],
    labels: np.ndarray,
    domains: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    subject_mask: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    branch_names: Sequence[str],
    budget_per_source_class: int,
    blend_weight: float,
    auxiliary_weight: float,
    seed: int,
    calibration_recording: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt binary intent using cue direction only as an auxiliary signal."""
    labels = np.asarray(labels)
    domains = np.asarray(domains).astype(str)
    subjects = np.asarray(subjects).astype(str)
    recordings = np.asarray(recordings)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    result = np.asarray(global_probabilities, dtype=np.float64).copy()
    evaluation_mask = subject_mask & (recordings != calibration_recording)
    adapted_mask = np.zeros(len(labels), dtype=bool)
    for subject in np.unique(subjects[subject_mask]):
        evaluate = evaluation_mask & (subjects == subject)
        calibration = (
            (subjects == subject) & (recordings == calibration_recording)
        )
        selected: list[int] = []
        for label in (0, 1, 2):
            candidates = np.flatnonzero(calibration & (labels == label))
            selected.extend(candidates[:budget_per_source_class])
        selected_indices = np.asarray(selected, dtype=np.int64)
        if not len(selected_indices) or set(labels[selected_indices]) != {0, 1, 2}:
            continue
        if not np.any(evaluate):
            continue
        adapter = RapidAuxiliaryIntentPatientAdapter.fit(
            {name: values[selected_indices] for name, values in branches.items()},
            labels[selected_indices],
            domains[selected_indices][0],
            branch_names=branch_names,
            blend_weight=blend_weight,
            auxiliary_weight=auxiliary_weight,
            seed=seed + sum(ord(value) for value in subject),
        )
        result[evaluate] = adapter.predict(
            {name: values[evaluate] for name, values in branches.items()},
            domains[evaluate][0],
            result[evaluate],
        )
        adapted_mask[evaluate] = True
    return result, adapted_mask


def patient_adapted_intent_probabilities(
    branches: Mapping[str, np.ndarray],
    labels: np.ndarray,
    domains: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    subject_mask: np.ndarray,
    global_probabilities: np.ndarray,
    *,
    branch_names: Sequence[str],
    budget_per_class: int,
    blend_weight: float,
    seed: int,
    calibration_recording: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt binary intent and return a mask excluding calibration recordings."""
    labels = np.asarray(labels)
    domains = np.asarray(domains).astype(str)
    subjects = np.asarray(subjects).astype(str)
    recordings = np.asarray(recordings)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    result = np.asarray(global_probabilities, dtype=np.float64).copy()
    evaluation_mask = subject_mask & (recordings != calibration_recording)
    adapted_mask = np.zeros(len(labels), dtype=bool)
    binary = labels != 0
    for subject in np.unique(subjects[subject_mask]):
        evaluate = evaluation_mask & (subjects == subject)
        calibration = (
            (subjects == subject) & (recordings == calibration_recording)
        )
        selected: list[int] = []
        for value in (False, True):
            candidates = np.flatnonzero(calibration & (binary == value))
            selected.extend(candidates[:budget_per_class])
        selected_indices = np.asarray(selected, dtype=np.int64)
        if not len(selected_indices) or set(binary[selected_indices]) != {False, True}:
            continue
        if not np.any(evaluate):
            continue
        adapter = RapidIntentPatientAdapter.fit(
            {name: values[selected_indices] for name, values in branches.items()},
            labels[selected_indices],
            domains[selected_indices][0],
            branch_names=branch_names,
            blend_weight=blend_weight,
            seed=seed + sum(ord(value) for value in subject),
        )
        result[evaluate] = adapter.predict(
            {name: values[evaluate] for name, values in branches.items()},
            domains[evaluate][0],
            result[evaluate],
        )
        adapted_mask[evaluate] = True
    return result, adapted_mask


def patient_adapted_probabilities(
    branches: Mapping[str, np.ndarray],
    labels: np.ndarray,
    domains: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    evaluation_mask: np.ndarray,
    global_probabilities: StageProbabilities,
    *,
    branch_names: Sequence[str],
    budget_per_class: int,
    blend_weight: float,
    seed: int,
) -> StageProbabilities:
    """Blend per-patient calibration heads into population predictions."""
    result = StageProbabilities(
        intent=global_probabilities.intent.copy(),
        direction=global_probabilities.direction.copy(),
    )
    for subject in np.unique(subjects[evaluation_mask]):
        evaluate = evaluation_mask & (subjects == subject)
        calibration = (subjects == subject) & (recordings == 0)
        selected: list[int] = []
        for label in (0, 1, 2):
            candidates = np.flatnonzero(calibration & (labels == label))
            selected.extend(candidates[:budget_per_class])
        selected_indices = np.asarray(selected, dtype=np.int64)
        if set(np.unique(labels[selected_indices])) != {0, 1, 2}:
            continue
        adapter = RapidPatientAdapter.fit(
            {
                name: values[selected_indices]
                for name, values in branches.items()
            },
            labels[selected_indices],
            str(domains[selected_indices][0]),
            branch_names=branch_names,
            blend_weight=blend_weight,
            seed=seed + sum(ord(value) for value in str(subject)),
        )
        local_probabilities = adapter.predict(
            {
                name: values[evaluate]
                for name, values in branches.items()
            },
            str(domains[evaluate][0]),
            StageProbabilities(
                result.intent[evaluate],
                result.direction[evaluate],
            ),
        )
        result.intent[evaluate] = local_probabilities.intent
        result.direction[evaluate] = local_probabilities.direction
    return result
