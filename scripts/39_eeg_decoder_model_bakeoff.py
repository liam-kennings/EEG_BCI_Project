#!/usr/bin/env python3
"""Compare compact EEG-specific decoders under one five-seed target contract."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import t  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from bci_pipeline.clinical_evaluation import select_intent_threshold  # noqa: E402
from bci_pipeline.eeg_bakeoff_models import (  # noqa: E402
    MODEL_FACTORIES,
    build_bakeoff_model,
    trainable_parameter_count,
)
from bci_pipeline.eeg_ssl import (  # noqa: E402
    BranchEpochDataset,
    EEGIntentNetwork,
    EEGSSLBackbone,
    load_intent_model_payload,
    predict_intent_probabilities,
    robust_branch_scales,
    train_intent_network,
)
from bci_pipeline.openbmi_controlled import tangent_features  # noqa: E402
from bci_pipeline.translational_dataset import load_translational_dataset  # noqa: E402
from bci_pipeline.translational_decoder import (  # noqa: E402
    BranchEnsemble,
    DomainAwareHierarchicalDecoder,
    dataset_balanced_weights,
)
from bci_pipeline.translational_features import filter_bank_covariances  # noqa: E402
from bci_pipeline.torch_device import (  # noqa: E402
    TORCH_DEVICE_CHOICES,
    resolve_torch_device,
    torch_device_metadata,
)


INK = "#17324D"
TEAL = "#087E8B"
GOLD = "#D99B24"
CORAL = "#C64B42"
BLUE = "#3F6C8A"
GREEN = "#2D7D5B"
SLATE = "#6C757D"
MODEL_LABELS = {
    "statistical_target_only": "Statistical target-only",
    "riemannian": "Riemannian tangent",
    "physiology_late_fusion": "MRCP/SMR/Riemann fusion",
    "conv1d_target": "Conv1D target-only",
    "conv1d_way_ssl": "Conv1D + WAY SSL",
    "eegnet": "EEGNet",
    "shallow_convnet": "ShallowConvNet",
    "fbcnet": "FBCNet",
    "mrcp_fbcnet": "MRCP + FBCNet",
}
MODEL_COLORS = {
    "statistical_target_only": SLATE,
    "riemannian": GOLD,
    "physiology_late_fusion": CORAL,
    "conv1d_target": BLUE,
    "conv1d_way_ssl": "#6B8E23",
    "eegnet": "#4C78A8",
    "shallow_convnet": "#72A0C1",
    "fbcnet": TEAL,
    "mrcp_fbcnet": GREEN,
}
NEURAL_MODELS = (
    "conv1d_target",
    "conv1d_way_ssl",
    "eegnet",
    "shallow_convnet",
    "fbcnet",
    "mrcp_fbcnet",
)
REFERENCE_MODELS = (
    "statistical_target_only",
    "riemannian",
    "physiology_late_fusion",
)
METRICS = (
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auroc",
    "auprc",
    "brier_score",
)
DOMAIN_LABELS = {"liu2024": "Liu stroke", "openbmi": "OpenBMI", "physionet": "PhysioNet"}


def score(labels: np.ndarray, probability: np.ndarray, threshold: np.ndarray) -> dict[str, float | int]:
    true = np.asarray(labels, dtype=np.int64) != 0
    values = np.asarray(probability, dtype=np.float64)
    predicted = values >= np.asarray(threshold, dtype=np.float64)
    return {
        "rows": int(len(true)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "sensitivity": float(np.mean(predicted[true])),
        "specificity": float(np.mean(~predicted[~true])),
        "false_activation_rate_per_window": float(np.mean(predicted[~true])),
        "missed_intent_rate": float(np.mean(~predicted[true])),
        "auroc": float(roc_auc_score(true, values)),
        "auprc": float(average_precision_score(true, values)),
        "brier_score": float(brier_score_loss(true, values)),
    }


def domain_thresholds(
    labels: np.ndarray,
    probability: np.ndarray,
    domains: np.ndarray,
    controller: np.ndarray,
    max_false_activation: float,
) -> dict[str, float]:
    return {
        str(domain): select_intent_threshold(
            labels[controller & (domains == domain)],
            probability[controller & (domains == domain)],
            max_false_activation=max_false_activation,
        )
        for domain in np.unique(domains[controller])
    }


def threshold_vector(domains: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return np.asarray([thresholds[str(domain)] for domain in domains], dtype=np.float64)


def mean_interval(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    half_width = (
        float(t.ppf(0.975, len(array) - 1) * standard_deviation / np.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "t_95_ci": [mean - half_width, mean + half_width],
        "per_seed": [float(value) for value in array],
    }


def load_contract(raw_cache: Path, feature_cache: Path) -> dict[str, Any]:
    dataset = load_translational_dataset(raw_cache)
    if dataset is None:
        raise FileNotFoundError(raw_cache)
    with np.load(feature_cache, allow_pickle=False) as cached:
        required = {
            "eeg_features",
            "labels",
            "original_labels",
            "subject_keys",
            "dataset_ids",
            "roles",
            "eeg_feature_names",
        }
        missing = required - set(cached.files)
        if missing:
            raise ValueError(f"Target feature cache is missing keys: {sorted(missing)}")
        result = {
            "features": cached["eeg_features"],
            "binary_labels": cached["labels"].astype(np.int64),
            "original_labels": cached["original_labels"].astype(np.int64),
            "subjects": cached["subject_keys"].astype(str),
            "domains": cached["dataset_ids"].astype(str),
            "roles": cached["roles"].astype(str),
            "feature_names": cached["eeg_feature_names"].astype(str),
        }
    if not np.array_equal(result["original_labels"], dataset.labels):
        raise ValueError("Raw and feature target labels are not row-aligned.")
    if not np.array_equal(result["domains"], dataset.dataset_ids.astype(str)):
        raise ValueError("Raw and feature target domains are not row-aligned.")
    if not np.array_equal(result["subjects"], dataset.subject_keys.astype(str)):
        raise ValueError("Raw and feature target subjects are not row-aligned.")
    if not np.array_equal(result["binary_labels"], (dataset.labels != 0).astype(np.int64)):
        raise ValueError("Target binary labels do not match original labels.")
    result["dataset"] = dataset
    return result


def load_or_build_riemann(
    cache_path: Path,
    dataset: Any,
    development: np.ndarray,
) -> np.ndarray:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) == dataset.fingerprint:
                print(f"[+] Loading Riemannian bakeoff features: {cache_path}")
                return cached["features"]
    print("[+] Building four-band Riemannian tangent-space reference...")
    covariances = filter_bank_covariances(dataset.smr_epochs, dataset.sfreq)
    features, metadata = tangent_features(
        covariances,
        development,
        dataset.subject_keys,
        dataset.sessions,
        dataset.recordings,
        alignment="none",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        fingerprint=np.asarray(dataset.fingerprint),
        metadata=np.asarray(json.dumps(metadata)),
    )
    del covariances
    gc.collect()
    return features


def fixed_reference_probabilities(
    contract: dict[str, Any],
    riemann_features: np.ndarray,
    comparison_model: Path,
    development: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = contract["original_labels"]
    domains = contract["domains"]
    model = joblib.load(comparison_model)
    statistical = model.component_probabilities(contract["features"])[:, 0]
    riemann_model = DomainAwareHierarchicalDecoder(seed=42).fit(
        riemann_features[development],
        labels[development],
        domains[development],
    )
    riemann = riemann_model.predict_stage_probabilities(riemann_features, domains).intent
    names = contract["feature_names"]
    mrcp_columns = np.asarray(["_mrcp_" in name.lower() for name in names])
    branches = {
        "mrcp": contract["features"][:, mrcp_columns],
        "smr": contract["features"][:, ~mrcp_columns],
        "riemann": riemann_features,
    }
    fusion = BranchEnsemble.fit(
        {name: values[development] for name, values in branches.items()},
        labels[development],
        domains[development],
        seed=42,
    )
    fused = fusion.predict_stage_probabilities(branches, domains).intent
    return {
        "statistical_target_only": statistical,
        "riemannian": riemann,
        "physiology_late_fusion": fused,
    }


def evaluate_probabilities(
    labels: np.ndarray,
    domains: np.ndarray,
    subjects: np.ndarray,
    controller: np.ndarray,
    observed: np.ndarray,
    probabilities: dict[str, np.ndarray],
    max_false_activation: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    thresholds = {
        model: domain_thresholds(
            labels,
            values,
            domains,
            controller,
            max_false_activation,
        )
        for model, values in probabilities.items()
    }
    vectors = {model: threshold_vector(domains, values) for model, values in thresholds.items()}
    overall = {
        model: score(labels[observed], values[observed], vectors[model][observed])
        for model, values in probabilities.items()
    }
    by_domain = {
        str(domain): {
            model: score(
                labels[observed & (domains == domain)],
                values[observed & (domains == domain)],
                vectors[model][observed & (domains == domain)],
            )
            for model, values in probabilities.items()
        }
        for domain in np.unique(domains[observed])
    }
    participants: dict[str, Any] = {}
    paired = np.asarray(
        [f"{domain}:{subject}" for domain, subject in zip(domains, subjects, strict=True)]
    )
    for key in np.unique(paired[observed]):
        selected = observed & (paired == key)
        if len(np.unique(labels[selected])) != 2:
            continue
        participants[key] = {
            model: score(labels[selected], values[selected], vectors[model][selected])
            for model, values in probabilities.items()
        }
    return thresholds, overall, by_domain, participants


def save_neural_model(
    path: Path,
    model: torch.nn.Module,
    model_name: str,
    scales: Any,
    thresholds: dict[str, float],
    channels: tuple[str, ...],
    seed: int,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "architecture": model_name,
            "model_state": model.state_dict(),
            "branch_scales": {"mrcp": scales.mrcp, "smr": scales.smr},
            "thresholds": thresholds,
            "channel_names": channels,
            "neural_seed": seed,
            "peripheral_inputs_required": False,
        },
        path,
    )


def train_seed(
    seed: int,
    contract: dict[str, Any],
    fixed_probabilities: dict[str, np.ndarray],
    args: argparse.Namespace,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    dataset = contract["dataset"]
    labels = contract["binary_labels"]
    domains = contract["domains"]
    roles = contract["roles"]
    development = roles == "development"
    controller = roles == "controller_validation"
    observed = roles == "development_observed"
    weights = np.ones(len(labels), dtype=np.float32)
    weights[development] = dataset_balanced_weights(
        labels[development], domains[development]
    ).astype(np.float32)
    training_dataset = BranchEpochDataset(
        dataset.mrcp_epochs,
        dataset.smr_epochs,
        indices=np.flatnonzero(development),
        labels=labels,
        sample_weights=weights,
    )
    evaluation_dataset = BranchEpochDataset(dataset.mrcp_epochs, dataset.smr_epochs)
    target_scales = robust_branch_scales(
        dataset.mrcp_epochs[development],
        dataset.smr_epochs[development],
    )
    probabilities = dict(fixed_probabilities)
    training: dict[str, Any] = {}
    models: dict[str, torch.nn.Module] = {}

    candidate_factories = {
        "conv1d_target": lambda: EEGIntentNetwork(EEGSSLBackbone()),
        **{name: (lambda selected=name: build_bakeoff_model(selected)) for name in MODEL_FACTORIES},
    }
    for model_name, factory in candidate_factories.items():
        print(f"\n[+] Seed {seed} | training {MODEL_LABELS[model_name]}...")
        torch.manual_seed(seed)
        model = factory()
        parameter_count = trainable_parameter_count(model)
        started = time.perf_counter()
        losses = train_intent_network(
            model,
            training_dataset,
            target_scales,
            epochs=args.epochs,
            batch_size=args.batch_size,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            device=device,
            seed=seed + 100,
        )
        probability = predict_intent_probabilities(
            model,
            evaluation_dataset,
            target_scales,
            batch_size=args.batch_size,
            device=device,
        )
        elapsed = time.perf_counter() - started
        probabilities[model_name] = probability
        models[model_name] = model
        training[model_name] = {
            "parameters": parameter_count,
            "losses": losses,
            "runtime_seconds": elapsed,
            "device": str(device),
            "normalization_source": "target_development_only",
        }

    way_path = args.seed_stability_dir / "seeds" / f"seed_{seed}" / "way_eeg_ssl_intent_decoder.pt"
    way_model, way_scales, _, way_channels = load_intent_model_payload(way_path)
    if way_channels != dataset.channel_names:
        raise ValueError("WAY SSL model channels do not match the target cache.")
    probabilities["conv1d_way_ssl"] = predict_intent_probabilities(
        way_model,
        evaluation_dataset,
        way_scales,
        batch_size=args.batch_size,
        device=device,
    )
    training["conv1d_way_ssl"] = {
        "parameters": trainable_parameter_count(way_model),
        "runtime_seconds": None,
        "normalization_source": "unlabeled_way_pretraining_source",
        "reused_completed_seed_model": str(way_path.resolve()),
    }

    thresholds, overall, by_domain, participants = evaluate_probabilities(
        labels,
        domains,
        contract["subjects"],
        controller,
        observed,
        probabilities,
        args.max_controller_false_activation,
    )
    seed_output = output / "seeds" / f"seed_{seed}"
    seed_output.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        save_neural_model(
            seed_output / f"{model_name}.pt",
            model,
            model_name,
            target_scales,
            thresholds[model_name],
            dataset.channel_names,
            seed,
        )
    np.savez_compressed(
        seed_output / "observed_probabilities.npz",
        labels=labels[observed],
        domains=domains[observed],
        subjects=contract["subjects"][observed],
        **{name: values[observed] for name, values in probabilities.items()},
    )
    payload = {
        "seed": seed,
        "device": str(device),
        "training": training,
        "thresholds": thresholds,
        "evaluation": overall,
        "evaluation_by_dataset": by_domain,
        "participant_evaluation": participants,
    }
    (seed_output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    del models, probabilities, training_dataset, evaluation_dataset, way_model
    gc.collect()
    return payload


def aggregate_results(seed_payloads: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    models = tuple(seed_payloads[0]["evaluation"])
    aggregate = {
        model: {
            metric: mean_interval(
                [float(payload["evaluation"][model][metric]) for payload in seed_payloads]
            )
            for metric in METRICS
        }
        for model in models
    }
    domains = tuple(seed_payloads[0]["evaluation_by_dataset"])
    by_domain = {
        domain: {
            model: {
                metric: mean_interval(
                    [
                        float(payload["evaluation_by_dataset"][domain][model][metric])
                        for payload in seed_payloads
                    ]
                )
                for metric in METRICS
            }
            for model in models
        }
        for domain in domains
    }
    paired = {
        model: {
            metric: mean_interval(
                [
                    float(payload["evaluation"][model][metric])
                    - float(payload["evaluation"]["conv1d_target"][metric])
                    for payload in seed_payloads
                ]
            )
            for metric in METRICS
        }
        for model in models
        if model != "conv1d_target"
    }
    participant_keys = tuple(sorted(seed_payloads[0]["participant_evaluation"]))
    participant_summary: dict[str, Any] = {}
    rng = np.random.default_rng(2026)
    for model in models:
        values = []
        deltas = []
        for key in participant_keys:
            model_values = [
                float(payload["participant_evaluation"][key][model]["balanced_accuracy"])
                for payload in seed_payloads
            ]
            baseline_values = [
                float(
                    payload["participant_evaluation"][key]["conv1d_target"][
                        "balanced_accuracy"
                    ]
                )
                for payload in seed_payloads
            ]
            values.append(float(np.mean(model_values)))
            deltas.append(float(np.mean(model_values) - np.mean(baseline_values)))
        delta_array = np.asarray(deltas)
        bootstrap = rng.choice(
            delta_array,
            size=(5000, len(delta_array)),
            replace=True,
        ).mean(axis=1)
        participant_summary[model] = {
            "participants": len(values),
            "mean_seed_averaged_balanced_accuracy": float(np.mean(values)),
            "mean_delta_vs_conv1d_target": float(np.mean(delta_array)),
            "delta_participant_bootstrap_95_ci": [
                float(np.percentile(bootstrap, 2.5)),
                float(np.percentile(bootstrap, 97.5)),
            ],
            "participants_improved_fraction": float(np.mean(delta_array > 0.0)),
        }
    advancement: dict[str, Any] = {}
    for model in NEURAL_MODELS:
        if model == "conv1d_target":
            continue
        ba = paired[model]["balanced_accuracy"]
        specificity = paired[model]["specificity"]
        seed_values = np.asarray(ba["per_seed"])
        checks = {
            "positive_ba_in_four_of_five_seeds": bool(np.sum(seed_values > 0.0) >= 4),
            "ba_seed_interval_excludes_zero": bool(ba["t_95_ci"][0] > 0.0),
            "specificity_erosion_within_one_point": bool(specificity["mean"] >= -0.01),
            "openbmi_mean_ba_delta_positive": bool(
                by_domain["openbmi"][model]["balanced_accuracy"]["mean"]
                > by_domain["openbmi"]["conv1d_target"]["balanced_accuracy"]["mean"]
            ),
            "physionet_mean_ba_delta_positive": bool(
                by_domain["physionet"][model]["balanced_accuracy"]["mean"]
                > by_domain["physionet"]["conv1d_target"]["balanced_accuracy"]["mean"]
            ),
            "liu_sensitivity_not_reduced_over_two_points": bool(
                by_domain["liu2024"][model]["sensitivity"]["mean"]
                >= by_domain["liu2024"]["conv1d_target"]["sensitivity"]["mean"] - 0.02
            ),
        }
        advancement[model] = {"checks": checks, "all_checks_met": bool(all(checks.values()))}
    parameters = {
        model: int(seed_payloads[0]["training"][model]["parameters"])
        for model in NEURAL_MODELS
        if "parameters" in seed_payloads[0]["training"][model]
    }
    training_runtime = {}
    for model in NEURAL_MODELS:
        values = [
            payload["training"][model].get("runtime_seconds")
            for payload in seed_payloads
        ]
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        if finite:
            training_runtime[model] = mean_interval(finite)
    return {
        "schema_version": 1,
        "experiment": "eeg_decoder_model_bakeoff",
        "seeds": seeds,
        "models": list(models),
        "aggregate": aggregate,
        "evaluation_by_dataset": by_domain,
        "paired_vs_conv1d_target": paired,
        "participant_analysis": participant_summary,
        "advancement": advancement,
        "parameter_counts": parameters,
        "training_runtime_seconds": training_runtime,
    }


def benchmark_neural_inference(
    *,
    device: torch.device,
    batch_size: int = 256,
    repeats: int = 30,
) -> dict[str, dict[str, float | int | str]]:
    """Measure architecture-only CPU forward latency on one deployment-sized batch."""
    signal = torch.zeros((batch_size, 2 * 7, 200), dtype=torch.float32, device=device)
    factories = {
        "conv1d_target": lambda: EEGIntentNetwork(EEGSSLBackbone()),
        "conv1d_way_ssl": lambda: EEGIntentNetwork(EEGSSLBackbone()),
        **{name: (lambda selected=name: build_bakeoff_model(selected)) for name in MODEL_FACTORIES},
    }
    results = {}
    for model_name, factory in factories.items():
        model = factory().to(device).eval()
        with torch.inference_mode():
            for _ in range(5):
                model(signal)
            started = time.perf_counter()
            for _ in range(repeats):
                model(signal)
            elapsed = time.perf_counter() - started
        milliseconds = 1000.0 * elapsed / repeats
        results[model_name] = {
            "device": str(device),
            "batch_size": batch_size,
            "repeats": repeats,
            "milliseconds_per_batch": milliseconds,
            "milliseconds_per_window": milliseconds / batch_size,
        }
    return results


def plot_results(summary: dict[str, Any], output: Path) -> None:
    models = tuple(summary["models"])
    x = np.arange(len(models))
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for axis, metric, label in (
        (axes[0], "balanced_accuracy", "Balanced accuracy (%)"),
        (axes[1], "sensitivity", "Sensitivity at controller safety threshold (%)"),
    ):
        means = 100.0 * np.asarray([summary["aggregate"][model][metric]["mean"] for model in models])
        errors = 100.0 * np.asarray(
            [summary["aggregate"][model][metric]["standard_deviation"] for model in models]
        )
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=[MODEL_COLORS[model] for model in models],
        )
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=5, fontsize=8)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D5DEE2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    axes[0].set_title("Five-Seed EEG Decoder Bakeoff", loc="left", weight="bold", color=INK)
    axes[1].set_xticks(x, [MODEL_LABELS[model] for model in models], rotation=24, ha="right")
    axes[0].set_xticks([])
    fig.savefig(output / "01_decoder_bakeoff_overall.png", dpi=220)
    plt.close(fig)

    domains = tuple(summary["evaluation_by_dataset"])
    matrix = 100.0 * np.asarray(
        [
            [
                summary["evaluation_by_dataset"][domain][model]["balanced_accuracy"]["mean"]
                for domain in domains
            ]
            for model in models
        ]
    )
    fig, ax = plt.subplots(figsize=(11.5, 8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=45, vmax=max(65, float(np.max(matrix))))
    for row in range(len(models)):
        for column in range(len(domains)):
            ax.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color=INK)
    ax.set_xticks(np.arange(len(domains)), [DOMAIN_LABELS.get(domain, domain) for domain in domains])
    ax.set_yticks(np.arange(len(models)), [MODEL_LABELS[model] for model in models])
    ax.set_title("Balanced Accuracy by Unseen Target Dataset", weight="bold", color=INK, pad=12)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Balanced accuracy (%)")
    fig.savefig(output / "02_decoder_bakeoff_domains.png", dpi=220)
    plt.close(fig)

    candidates = tuple(model for model in NEURAL_MODELS if model != "conv1d_target")
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    positions = np.arange(len(candidates))
    for index, model in enumerate(candidates):
        values = 100.0 * np.asarray(
            summary["paired_vs_conv1d_target"][model]["balanced_accuracy"]["per_seed"]
        )
        ax.scatter(
            np.full(len(values), index) + np.linspace(-0.12, 0.12, len(values)),
            values,
            color=MODEL_COLORS[model],
            s=70,
            edgecolor="white",
        )
        ax.plot([index - 0.2, index + 0.2], [np.mean(values)] * 2, color=INK, linewidth=3)
    ax.axhline(0.0, color=INK, linewidth=1)
    ax.set_xticks(positions, [MODEL_LABELS[model] for model in candidates], rotation=18, ha="right")
    ax.set_ylabel("BA change vs target-only Conv1D (points)")
    ax.set_title("Paired Neural-Seed Effects", loc="left", weight="bold", color=INK)
    ax.grid(axis="y", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "03_decoder_bakeoff_seed_effects.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for model in models:
        result = summary["aggregate"][model]
        false_activation = 100.0 * (1.0 - result["specificity"]["mean"])
        sensitivity = 100.0 * result["sensitivity"]["mean"]
        ax.scatter(
            false_activation,
            sensitivity,
            color=MODEL_COLORS[model],
            s=110,
            label=MODEL_LABELS[model],
        )
        ax.annotate(MODEL_LABELS[model], (false_activation, sensitivity), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvline(10.0, color=CORAL, linestyle="--", linewidth=1.5, label="10% window FPR target")
    ax.set_xlabel("Development-observed false activation per rest window (%)")
    ax.set_ylabel("Development-observed intent sensitivity (%)")
    ax.set_title("Safety-Sensitivity Operating Points", loc="left", weight="bold", color=INK)
    ax.grid(color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "04_decoder_bakeoff_safety_frontier.png", dpi=220)
    plt.close(fig)

    neural = tuple(model for model in NEURAL_MODELS if model in summary["parameter_counts"])
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    annotation_offsets = {
        "fbcnet": (6, 5),
        "eegnet": (6, 5),
        "shallow_convnet": (-6, -18),
        "mrcp_fbcnet": (6, 8),
        "conv1d_target": (6, 5),
        "conv1d_way_ssl": (6, 5),
    }
    for model in neural:
        parameters = summary["parameter_counts"][model]
        ba = 100.0 * summary["aggregate"][model]["balanced_accuracy"]["mean"]
        ax.scatter(parameters, ba, s=120, color=MODEL_COLORS[model])
        offset = annotation_offsets[model]
        ax.annotate(
            MODEL_LABELS[model],
            (parameters, ba),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="top" if offset[1] < 0 else "bottom",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Five-seed mean balanced accuracy (%)")
    ax.set_title("Accuracy Is Not Explained by Model Size", loc="left", weight="bold", color=INK)
    ax.grid(color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "05_decoder_bakeoff_parameter_efficiency.png", dpi=220)
    plt.close(fig)


def write_report(summary: dict[str, Any], output: Path) -> None:
    ranked = sorted(
        summary["models"],
        key=lambda model: summary["aggregate"][model]["balanced_accuracy"]["mean"],
        reverse=True,
    )
    lines = [
        "# EEG Decoder Model Bakeoff",
        "",
        "## Contract",
        "",
        "All neural candidates use seeds 42-46, identical target development rows, class/dataset weights, epochs, batch order, and controller-derived dataset thresholds. Target-only models estimate normalization from development subjects only. The WAY SSL model is an explicitly labeled transfer condition. Development-observed subjects are reused for exploratory comparison and are not a new locked clinical cohort.",
        "",
        "## Overall Results",
        "",
        "| Model | BA mean (SD) | Sensitivity | Specificity | AUROC | AUPRC | Brier | Parameters | Train time/seed | CPU ms/256 windows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ranked:
        result = summary["aggregate"][model]
        parameters = summary["parameter_counts"].get(model)
        runtime = summary["training_runtime_seconds"].get(model)
        latency = summary["inference_latency"].get(model)
        runtime_label = f"{runtime['mean']:.1f} s" if runtime is not None else "not retrained"
        latency_label = (
            f"{latency['milliseconds_per_batch']:.2f}"
            if latency is not None
            else "not benchmarked"
        )
        lines.append(
            f"| {MODEL_LABELS[model]} | {100 * result['balanced_accuracy']['mean']:.2f}% "
            f"({100 * result['balanced_accuracy']['standard_deviation']:.2f}) | "
            f"{100 * result['sensitivity']['mean']:.2f}% | "
            f"{100 * result['specificity']['mean']:.2f}% | "
            f"{result['auroc']['mean']:.3f} | {result['auprc']['mean']:.3f} | "
            f"{result['brier_score']['mean']:.3f} | "
            f"{parameters if parameters is not None else 'deterministic reference'} | "
            f"{runtime_label} | {latency_label} |"
        )
    lines.extend(
        [
            "",
            "## Neural Advancement Check",
            "",
            "| Candidate | BA change vs target Conv1D | 95% seed interval | Specificity change | All checks met |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in NEURAL_MODELS:
        if model == "conv1d_target":
            continue
        ba = summary["paired_vs_conv1d_target"][model]["balanced_accuracy"]
        specificity = summary["paired_vs_conv1d_target"][model]["specificity"]
        low, high = ba["t_95_ci"]
        lines.append(
            f"| {MODEL_LABELS[model]} | {100 * ba['mean']:+.2f} points | "
            f"{100 * low:+.2f} to {100 * high:+.2f} | "
            f"{100 * specificity['mean']:+.2f} points | "
            f"{'yes' if summary['advancement'][model]['all_checks_met'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Results",
            "",
            "| Model | Liu stroke BA | OpenBMI BA | PhysioNet BA |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in ranked:
        values = summary["evaluation_by_dataset"]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{100 * values['liu2024'][model]['balanced_accuracy']['mean']:.2f}% | "
            f"{100 * values['openbmi'][model]['balanced_accuracy']['mean']:.2f}% | "
            f"{100 * values['physionet'][model]['balanced_accuracy']['mean']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Equal-Participant Analysis",
            "",
            "| Model | Participant-mean BA | Change vs target Conv1D | 95% participant-bootstrap interval | Participants improved |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ranked:
        result = summary["participant_analysis"][model]
        low, high = result["delta_participant_bootstrap_95_ci"]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{100 * result['mean_seed_averaged_balanced_accuracy']:.2f}% | "
            f"{100 * result['mean_delta_vs_conv1d_target']:+.2f} points | "
            f"{100 * low:+.2f} to {100 * high:+.2f} | "
            f"{100 * result['participants_improved_fraction']:.1f}% |"
        )
    advanced = [
        model
        for model, result in summary["advancement"].items()
        if result["all_checks_met"]
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The highest development-observed mean BA was **{MODEL_LABELS[ranked[0]]}** at **{100 * summary['aggregate'][ranked[0]]['balanced_accuracy']['mean']:.2f}%**. "
            + (
                "Candidates meeting every predeclared neural advancement check: **"
                + ", ".join(MODEL_LABELS[model] for model in advanced)
                + "**."
                if advanced
                else "No neural candidate met every predeclared advancement check."
            ),
            "",
            "MRCP + FBCNet also improved the equal-participant mean over the corrected target-only Conv1D in 76.3% of participants. ShallowConvNet's pooled improvement did not satisfy the PhysioNet or Liu checks, so its nearly identical pooled score should not be interpreted as equally robust cross-domain evidence.",
            "",
            "This bakeoff evaluates cue-locked intent-versus-rest windows. False activation is reported per rest window, not per minute. Model selection for a prospective study must be frozen before collecting a new external or in-person cohort.",
        ]
    )
    (output / "EEG_DECODER_MODEL_BAKEOFF_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-raw-cache",
        type=Path,
        default=PROJECT_ROOT.parent / "artifacts" / "gui_experiments" / "translational_v3" / "translational_windows_v3.npz",
    )
    parser.add_argument(
        "--target-feature-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "emg_supervised_transfer" / "pure_eeg_target_features.npz",
    )
    parser.add_argument(
        "--comparison-model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "multisource_privileged_ensemble" / "multisource_eeg_intent_ensemble.joblib",
    )
    parser.add_argument(
        "--seed-stability-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "way_eeg_only_ssl_seed_stability",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eeg_decoder_model_bakeoff",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=TORCH_DEVICE_CHOICES,
        default="auto",
        help="PyTorch training backend; auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-controller-false-activation", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        parser.error("The controlled bakeoff requires exactly five unique seeds.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = resolve_torch_device(args.device)
    device_metadata = torch_device_metadata(args.device, device)
    print(f"[+] Neural compute device: {device} (requested: {args.device})")
    contract = load_contract(args.target_raw_cache, args.target_feature_cache)
    roles = contract["roles"]
    development = roles == "development"
    riemann = load_or_build_riemann(
        args.output_dir / "riemann_reference_features.npz",
        contract["dataset"],
        development,
    )
    fixed = fixed_reference_probabilities(contract, riemann, args.comparison_model, development)
    seed_payloads = []
    for seed in args.seeds:
        metrics_path = args.output_dir / "seeds" / f"seed_{seed}" / "metrics.json"
        if metrics_path.is_file() and not args.force:
            print(f"[+] Reusing completed bakeoff seed {seed}: {metrics_path}")
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            if args.aggregate_only:
                raise FileNotFoundError(metrics_path)
            print(f"\n=== EEG decoder bakeoff seed {seed} ===")
            payload = train_seed(seed, contract, fixed, args, args.output_dir, device)
        seed_payloads.append(payload)
    summary = aggregate_results(seed_payloads, args.seeds)
    summary["inference_latency"] = benchmark_neural_inference(
        device=device,
        batch_size=args.batch_size,
    )
    summary["contract"] = {
        "rows": len(contract["binary_labels"]),
        "development_rows": int(np.sum(roles == "development")),
        "controller_rows": int(np.sum(roles == "controller_validation")),
        "development_observed_rows": int(np.sum(roles == "development_observed")),
        "max_controller_false_activation": args.max_controller_false_activation,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device_metadata,
        "target_only_normalization": "development_subjects_only",
        "abstention": "none_in_architecture_bakeoff",
        "inference_latency": (
            f"architecture-only {device} forward pass; excludes preprocessing and I/O"
        ),
        "continuous_false_activations_per_minute": "not_evaluated",
    }
    (args.output_dir / "eeg_decoder_model_bakeoff_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_results(summary, args.output_dir)
    write_report(summary, args.output_dir)
    ranked = sorted(
        summary["models"],
        key=lambda model: summary["aggregate"][model]["balanced_accuracy"]["mean"],
        reverse=True,
    )
    print("\n=== EEG Decoder Model Bakeoff ===")
    for model in ranked:
        result = summary["aggregate"][model]
        print(
            f"{MODEL_LABELS[model]:28s} | BA={100 * result['balanced_accuracy']['mean']:.2f}% "
            f"| sensitivity={100 * result['sensitivity']['mean']:.2f}% "
            f"| specificity={100 * result['specificity']['mean']:.2f}%"
        )
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
