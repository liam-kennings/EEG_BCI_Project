"""Leakage-safe filter-bank CSP hierarchical decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..representations.riemann import FILTER_BANK


class BinaryFBCSP:
    def __init__(
        self,
        sfreq: float,
        components_per_band: int = 4,
        filter_bank: Sequence[Tuple[str, float, float]] = FILTER_BANK,
    ):
        if components_per_band < 2 or components_per_band % 2:
            raise ValueError("components_per_band must be a positive even number.")
        self.sfreq = sfreq
        self.components_per_band = components_per_band
        self.filter_bank = tuple(filter_bank)
        self.filters_: List[np.ndarray] = []

    def _bandpass(self, epochs: np.ndarray, low: float, high: float) -> np.ndarray:
        sos = butter(4, (low, high), btype="bandpass", fs=self.sfreq, output="sos")
        return sosfiltfilt(sos, epochs, axis=-1)

    @staticmethod
    def _covariances(epochs: np.ndarray) -> np.ndarray:
        centered = epochs - np.mean(epochs, axis=-1, keepdims=True)
        covariances = centered @ np.swapaxes(centered, -1, -2)
        traces = np.trace(covariances, axis1=1, axis2=2)
        return covariances / np.clip(traces[:, None, None], 1e-12, None)

    def fit(self, epochs: np.ndarray, labels: np.ndarray) -> "BinaryFBCSP":
        if not np.all(np.isfinite(epochs)):
            raise ValueError("FBCSP received non-finite training epochs.")
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError("BinaryFBCSP requires exactly two classes.")
        self.filters_ = []
        half = self.components_per_band // 2
        for _, low, high in self.filter_bank:
            filtered = self._bandpass(epochs, low, high)
            covariances = self._covariances(filtered)
            first = np.mean(covariances[labels == classes[0]], axis=0)
            second = np.mean(covariances[labels == classes[1]], axis=0)
            composite = first + second
            regularizer = 1e-6 * np.trace(composite) / composite.shape[0]
            _, vectors = eigh(first, composite + regularizer * np.eye(composite.shape[0]))
            self.filters_.append(np.column_stack((vectors[:, :half], vectors[:, -half:])))
        return self

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        if len(self.filters_) != len(self.filter_bank):
            raise RuntimeError("BinaryFBCSP must be fitted before transformation.")
        blocks = []
        for filters, (_, low, high) in zip(self.filters_, self.filter_bank):
            filtered = self._bandpass(epochs, low, high)
            projected = np.einsum("cf,nct->nft", filters, filtered, optimize=True)
            variances = np.var(projected, axis=-1)
            normalized = variances / np.clip(np.sum(variances, axis=1, keepdims=True), 1e-12, None)
            blocks.append(np.log(np.clip(normalized, 1e-12, None)))
        features = np.concatenate(blocks, axis=1)
        if not np.all(np.isfinite(features)):
            raise ValueError("FBCSP produced non-finite features.")
        return features

    def fit_transform(self, epochs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(epochs, labels).transform(epochs)


@dataclass
class HierarchicalFBCSP:
    sfreq: float
    components_per_band: int = 4
    random_seed: int = 42
    filter_bank: Sequence[Tuple[str, float, float]] = FILTER_BANK

    def fit(self, epochs: np.ndarray, labels: np.ndarray) -> "HierarchicalFBCSP":
        intent_labels = (labels != 0).astype(np.int64)
        movement = labels != 0
        self.intent_features_ = BinaryFBCSP(
            self.sfreq,
            self.components_per_band,
            self.filter_bank,
        )
        intent_x = self.intent_features_.fit_transform(epochs, intent_labels)
        self.intent_model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=self.random_seed,
            ),
        ).fit(intent_x, intent_labels)
        self.direction_features_ = BinaryFBCSP(
            self.sfreq,
            self.components_per_band,
            self.filter_bank,
        )
        direction_x = self.direction_features_.fit_transform(epochs[movement], labels[movement])
        self.direction_model_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=self.random_seed,
            ),
        ).fit(direction_x, labels[movement])
        return self

    def predict_stage_probabilities(self, epochs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        intent = self.intent_model_.predict_proba(self.intent_features_.transform(epochs))
        intent_classes = self.intent_model_.classes_.tolist()
        direction = self.direction_model_.predict_proba(self.direction_features_.transform(epochs))
        direction_classes = self.direction_model_.classes_.tolist()
        return (
            intent[:, intent_classes.index(1)],
            np.column_stack(
                (direction[:, direction_classes.index(1)], direction[:, direction_classes.index(2)])
            ),
        )
