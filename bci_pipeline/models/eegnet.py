"""Compact hierarchical EEGNet for raw sensorimotor epochs."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..torch_device import resolve_torch_device
from ..torch_layers import TemporalAdaptiveAveragePool2d


class EEGNetBackbone(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.5):
        super().__init__()
        temporal_filters, depth_multiplier = 8, 2
        spatial_filters = temporal_filters * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                (channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                (1, 15),
                padding=(0, 7),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, (1, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            TemporalAdaptiveAveragePool2d(8),
        )
        self.output_features = spatial_filters * 8

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.features(signal[:, None]).flatten(1)


class HierarchicalEEGNetModule(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.5):
        super().__init__()
        self.backbone = EEGNetBackbone(channels, dropout)
        self.intent_head = nn.Linear(self.backbone.output_features, 2)
        self.direction_head = nn.Linear(self.backbone.output_features, 2)

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(signal)
        return self.intent_head(features), self.direction_head(features)


@dataclass
class EEGNetTrainingConfig:
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    dropout: float = 0.5
    random_seed: int = 42
    device: str = "auto"


class HierarchicalEEGNet:
    def __init__(self, channels: int, config: EEGNetTrainingConfig):
        self.channels = channels
        self.config = config
        self._set_seed()
        torch.use_deterministic_algorithms(True)
        self.device = self._resolve_device(config.device)
        self.model = HierarchicalEEGNetModule(channels, config.dropout).to(self.device)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.history_: Dict[str, list[float]] = {"train_loss": [], "validation_loss": []}

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        return resolve_torch_device(requested)

    def _set_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

    def _normalize(self, epochs: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("EEGNet normalization has not been fitted.")
        return ((epochs - self.mean_) / self.std_).astype(np.float32)

    @staticmethod
    def _class_weights(labels: np.ndarray, classes: np.ndarray) -> torch.Tensor:
        counts = np.asarray([np.sum(labels == label) for label in classes], dtype=np.float64)
        weights = len(labels) / (len(classes) * np.clip(counts, 1.0, None))
        return torch.as_tensor(weights, dtype=torch.float32)

    def fit(
        self,
        train_epochs: np.ndarray,
        train_labels: np.ndarray,
        validation_epochs: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "HierarchicalEEGNet":
        self._set_seed()
        self.mean_ = np.mean(train_epochs, axis=(0, 2), keepdims=True)
        self.std_ = np.std(train_epochs, axis=(0, 2), keepdims=True)
        self.std_ = np.clip(self.std_, 1e-7, None)
        train_x = torch.from_numpy(self._normalize(train_epochs))
        train_y = torch.from_numpy(train_labels.astype(np.int64))
        validation_x = torch.from_numpy(self._normalize(validation_epochs))
        validation_y = torch.from_numpy(validation_labels.astype(np.int64))
        generator = torch.Generator().manual_seed(self.config.random_seed)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            TensorDataset(validation_x, validation_y),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        intent_labels = (train_labels != 0).astype(np.int64)
        intent_loss = nn.CrossEntropyLoss(
            weight=self._class_weights(intent_labels, np.asarray([0, 1])).to(self.device)
        )
        movement_labels = train_labels[train_labels != 0] - 1
        direction_loss = nn.CrossEntropyLoss(
            weight=self._class_weights(movement_labels, np.asarray([0, 1])).to(self.device)
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            training_total = 0.0
            training_count = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                intent_logits, direction_logits = self.model(batch_x)
                movement = batch_y != 0
                loss = intent_loss(intent_logits, (batch_y != 0).long())
                if torch.any(movement):
                    loss = loss + direction_loss(direction_logits[movement], batch_y[movement] - 1)
                loss.backward()
                optimizer.step()
                training_total += float(loss.detach()) * len(batch_y)
                training_count += len(batch_y)

            validation_total = 0.0
            validation_count = 0
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y in validation_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    intent_logits, direction_logits = self.model(batch_x)
                    movement = batch_y != 0
                    loss = intent_loss(intent_logits, (batch_y != 0).long())
                    if torch.any(movement):
                        loss = loss + direction_loss(direction_logits[movement], batch_y[movement] - 1)
                    validation_total += float(loss) * len(batch_y)
                    validation_count += len(batch_y)
            train_value = training_total / max(training_count, 1)
            validation_value = validation_total / max(validation_count, 1)
            self.history_["train_loss"].append(train_value)
            self.history_["validation_loss"].append(validation_value)
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"    EEGNet epoch {epoch:03d} | train={train_value:.4f} | "
                    f"validation={validation_value:.4f}"
                )
            if validation_value < best_loss - 1e-4:
                best_loss = validation_value
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    print(f"    EEGNet early stopping at epoch {epoch}")
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict_stage_probabilities(
        self,
        epochs: np.ndarray,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            torch.from_numpy(self._normalize(epochs)),
            batch_size=batch_size,
            shuffle=False,
        )
        intent_rows, direction_rows = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                intent_logits, direction_logits = self.model(batch.to(self.device))
                intent_rows.append(torch.softmax(intent_logits, dim=1)[:, 1].cpu().numpy())
                direction_rows.append(torch.softmax(direction_logits, dim=1).cpu().numpy())
        return np.concatenate(intent_rows), np.vstack(direction_rows)

    def payload(self) -> Dict[str, object]:
        return {
            "channels": self.channels,
            "config": self.config.__dict__,
            "mean": self.mean_,
            "std": self.std_,
            "state_dict": self.model.state_dict(),
            "history": self.history_,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "HierarchicalEEGNet":
        config = EEGNetTrainingConfig(**payload["config"])
        instance = cls(int(payload["channels"]), config)
        instance.model.load_state_dict(payload["state_dict"])
        instance.mean_ = np.asarray(payload["mean"])
        instance.std_ = np.asarray(payload["std"])
        instance.history_ = payload.get("history", {"train_loss": [], "validation_loss": []})
        return instance


def select_balanced_intent_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    true_intent = labels != 0
    candidates = np.linspace(0.2, 0.8, 121)
    scores = np.asarray(
        [balanced_accuracy_score(true_intent, probabilities >= threshold) for threshold in candidates]
    )
    tied = candidates[np.isclose(scores, np.max(scores))]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def select_safety_intent_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    max_false_activation_rate: float,
) -> float:
    rest = probabilities[labels == 0]
    if rest.size == 0:
        raise ValueError("Safety threshold selection requires rest examples.")
    threshold = np.quantile(rest, 1.0 - max_false_activation_rate, method="higher")
    return float(np.nextafter(threshold, np.inf))
