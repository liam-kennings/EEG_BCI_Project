#!/usr/bin/env python3
"""Train traditional ML motor-intent models on baseline-normalized Morlet ERD features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from bci_pipeline.splits import load_partitions  # noqa: E402


def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load only physiological ERD features, never metadata or QC measurements."""
    with path.open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Feature table has no header.")
        feature_names = [name for name in reader.fieldnames if name.endswith("_erd_db")]
        rows = list(reader)
    if not feature_names:
        raise ValueError("No ERD feature columns were found.")
    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64
    )
    labels = np.asarray([int(row["label"]) != 0 for row in rows], dtype=np.int64)
    subjects = np.asarray([int(row["subject"]) for row in rows], dtype=np.int64)
    runs = np.asarray([int(row["run"]) for row in rows], dtype=np.int64)
    if not np.all(np.isfinite(features)):
        raise ValueError("Morlet feature table contains non-finite values.")
    return features, labels, subjects, runs, feature_names


def make_models(feature_count: int, seed: int) -> Dict[str, Pipeline]:
    selected_count = min(40, feature_count)

    def preprocessing() -> List[Tuple[str, object]]:
        return [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selection", SelectKBest(f_classif, k=selected_count)),
        ]

    return {
        "shrinkage_lda": Pipeline(
            preprocessing()
            + [("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]
        ),
        "logistic_regression": Pipeline(
            preprocessing()
            + [
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                )
            ]
        ),
        "rbf_svm": Pipeline(
            preprocessing()
            + [
                (
                    "model",
                    SVC(
                        C=1.0,
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                )
            ]
        ),
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray, max_false_activation: float) -> Tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(labels, probabilities >= threshold) for threshold in candidates]
    best = np.max(scores)
    balanced = candidates[np.isclose(scores, best)]
    balanced_threshold = float(balanced[np.argmin(np.abs(balanced - 0.5))])
    rest_scores = probabilities[labels == 0]
    safety_threshold = float(
        np.nextafter(
            np.quantile(rest_scores, 1.0 - max_false_activation, method="higher"), np.inf
        )
    )
    return balanced_threshold, safety_threshold


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, object]:
    predictions = probabilities >= threshold
    rest = labels == 0
    intent = labels == 1
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "false_activation_rate": float(np.mean(predictions[rest])) if np.any(rest) else 0.0,
        "missed_intent_rate": float(np.mean(~predictions[intent])) if np.any(intent) else 0.0,
        "intent_sensitivity": float(np.mean(predictions[intent])) if np.any(intent) else 0.0,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def top_logistic_features(model: Pipeline, feature_names: List[str], count: int = 15) -> List[Dict[str, float]]:
    selector = model.named_steps["selection"]
    selected_names = np.asarray(feature_names)[selector.get_support()]
    coefficients = model.named_steps["model"].coef_[0]
    indices = np.argsort(np.abs(coefficients))[::-1][:count]
    return [
        {"feature": str(selected_names[index]), "coefficient": float(coefficients[index])}
        for index in indices
    ]


def plot_results(output_dir: Path, results: Dict[str, object]) -> None:
    names = list(results)
    colors = ["#396a93", "#4f9c70", "#b97831"]
    measures = (
        ("balanced_accuracy", "Balanced accuracy"),
        ("false_activation_rate", "False activation"),
        ("missed_intent_rate", "Missed intent"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for axis, (metric_name, title) in zip(axes, measures):
        values = [100 * results[name]["safety"][metric_name] for name in names]
        bars = axis.bar(names, values, color=colors)
        axis.set_title(f"Safety threshold: {title}")
        axis.set_ylim(0, 100)
        axis.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    axes[0].set_ylabel("Final unseen-patient rate (%)")
    fig.suptitle("Traditional Morlet-ERD motor-intent models", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "morlet_intent_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    logistic = results["logistic_regression"]["top_features"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    feature_names = [item["feature"].replace("_", " ") for item in logistic][::-1]
    coefficients = [item["coefficient"] for item in logistic][::-1]
    colors = ["#b65c4a" if value < 0 else "#3b7ea1" for value in coefficients]
    ax.barh(feature_names, coefficients, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title("Most influential selected Morlet ERD features")
    fig.tight_layout()
    fig.savefig(output_dir / "logistic_erd_feature_weights.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path", default="artifacts/morlet_feature_audit/morlet_erd_trial_features.csv"
    )
    parser.add_argument("--splits", default="artifacts/dataset/splits.json")
    parser.add_argument("--output-dir", default="artifacts/morlet_intent_models")
    parser.add_argument("--max-false-activation", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.max_false_activation < 1:
        raise ValueError("--max-false-activation must be between zero and one.")

    features, labels, subjects, runs, feature_names = load_features(Path(args.features_path))
    partitions = load_partitions(Path(args.splits))
    development = np.isin(subjects, partitions.development)
    controller = np.isin(subjects, partitions.controller_validation) & (runs == 12)
    final_target = np.isin(subjects, partitions.final_target) & (runs == 12)
    if not np.any(development) or not np.any(controller) or not np.any(final_target):
        raise RuntimeError("One of the locked subject partitions is empty in the feature table.")

    print("=== Traditional Morlet ERD Motor-Intent Models ===")
    print(
        f"[+] ERD features: {features.shape[1]} | development trials: {np.sum(development)} | "
        f"controller Run 12: {np.sum(controller)} | final Run 12: {np.sum(final_target)}"
    )
    result: Dict[str, object] = {
        "protocol": {
            "target": "rest versus unilateral motor imagery intent",
            "features": "baseline-normalized Morlet ERD and C3-C4 lateralization only",
            "development_subjects": list(partitions.development),
            "controller_validation_subjects": list(partitions.controller_validation),
            "final_target_subjects": list(partitions.final_target),
            "threshold_selection": "controller Run 12 only",
            "final_evaluation": "final-target Run 12 only",
            "max_controller_false_activation": args.max_false_activation,
        },
        "dataset": {
            "feature_count": int(features.shape[1]),
            "development_trials": int(np.sum(development)),
            "controller_trials": int(np.sum(controller)),
            "final_trials": int(np.sum(final_target)),
        },
        "models": {},
    }
    fitted: Dict[str, Pipeline] = {}
    for name, model in make_models(features.shape[1], args.seed).items():
        print(f"[+] Training {name}...")
        model.fit(features[development], labels[development])
        controller_probabilities = model.predict_proba(features[controller])[:, 1]
        balanced_threshold, safety_threshold = select_threshold(
            labels[controller], controller_probabilities, args.max_false_activation
        )
        final_probabilities = model.predict_proba(features[final_target])[:, 1]
        model_result: Dict[str, object] = {
            "balanced": metrics(labels[final_target], final_probabilities, balanced_threshold),
            "safety": metrics(labels[final_target], final_probabilities, safety_threshold),
            "balanced_threshold": balanced_threshold,
            "safety_threshold": safety_threshold,
        }
        if name == "logistic_regression":
            model_result["top_features"] = top_logistic_features(model, feature_names)
        result["models"][name] = model_result
        fitted[name] = model
        safety = model_result["safety"]
        print(
            f"    final safety BA={safety['balanced_accuracy'] * 100:.1f}% | "
            f"FP={safety['false_activation_rate'] * 100:.1f}% | "
            f"miss={safety['missed_intent_rate'] * 100:.1f}% | AUC={safety['roc_auc']:.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "morlet_intent_model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    joblib.dump(
        {"feature_names": feature_names, "models": fitted, "protocol": result["protocol"]},
        output_dir / "morlet_intent_models.joblib",
    )
    plot_results(output_dir, result["models"])
    print(f"[+] Saved models, report, and figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
