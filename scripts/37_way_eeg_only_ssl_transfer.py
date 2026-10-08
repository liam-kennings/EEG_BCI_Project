#!/usr/bin/env python3
"""Pretrain on unlabeled WAY EEG, then fine-tune on motor-imagery targets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from bci_pipeline.clinical_evaluation import select_intent_threshold  # noqa: E402
from bci_pipeline.eeg_ssl import (  # noqa: E402
    BranchEpochDataset,
    EEGIntentNetwork,
    EEGSSLBackbone,
    MaskedEEGAutoencoder,
    copied_backbone,
    model_payload,
    predict_intent_probabilities,
    robust_branch_scales,
    train_intent_network,
    train_masked_autoencoder,
)
from bci_pipeline.translational_decoder import dataset_balanced_weights  # noqa: E402
from bci_pipeline.torch_device import (  # noqa: E402
    TORCH_DEVICE_CHOICES,
    resolve_torch_device,
    torch_device_metadata,
)
from bci_pipeline.way_eeg_ssl import (  # noqa: E402
    build_way_eeg_ssl_windows,
    load_way_ssl_cache,
    save_way_ssl_cache,
)


INK = "#17324D"
TEAL = "#087E8B"
GOLD = "#D99B24"
CORAL = "#C64B42"
BLUE = "#3F6C8A"
MODEL_LABELS = {
    "classical_target_only": "Statistical target-only",
    "random_cnn": "Raw CNN random init",
    "way_ssl": "WAY EEG-only SSL",
    "way_privileged": "WAY peripheral-supervised",
}
MODEL_COLORS = {
    "classical_target_only": GOLD,
    "random_cnn": BLUE,
    "way_ssl": TEAL,
    "way_privileged": CORAL,
}


def load_target_contract(
    raw_path: Path,
    feature_path: Path,
) -> dict[str, np.ndarray | tuple[str, ...] | float]:
    with np.load(raw_path, allow_pickle=False) as raw:
        required_raw = {
            "mrcp_epochs",
            "smr_epochs",
            "labels",
            "dataset_ids",
            "subject_keys",
            "channel_names",
            "sfreq",
        }
        missing = required_raw - set(raw.files)
        if missing:
            raise ValueError(f"Raw target cache is missing keys: {sorted(missing)}")
        result: dict[str, np.ndarray | tuple[str, ...] | float] = {
            "mrcp_epochs": raw["mrcp_epochs"],
            "smr_epochs": raw["smr_epochs"],
            "labels": (raw["labels"] != 0).astype(np.int64),
            "dataset_ids": raw["dataset_ids"].astype(str),
            "subject_keys": raw["subject_keys"].astype(str),
            "channel_names": tuple(str(value) for value in raw["channel_names"].tolist()),
            "sfreq": float(np.asarray(raw["sfreq"]).item()),
        }
    with np.load(feature_path, allow_pickle=False) as features:
        required_features = {"eeg_features", "labels", "dataset_ids", "subject_keys", "roles"}
        missing = required_features - set(features.files)
        if missing:
            raise ValueError(f"Feature target cache is missing keys: {sorted(missing)}")
        if not np.array_equal(np.asarray(result["labels"]), features["labels"].astype(np.int64)):
            raise ValueError("Raw and feature target labels are not row-aligned.")
        if not np.array_equal(np.asarray(result["dataset_ids"]), features["dataset_ids"].astype(str)):
            raise ValueError("Raw and feature target datasets are not row-aligned.")
        if not np.array_equal(np.asarray(result["subject_keys"]), features["subject_keys"].astype(str)):
            raise ValueError("Raw and feature target subjects are not row-aligned.")
        result["eeg_features"] = features["eeg_features"]
        result["roles"] = features["roles"].astype(str)
    if np.asarray(result["mrcp_epochs"]).shape != np.asarray(result["smr_epochs"]).shape:
        raise ValueError("Target MRCP and SMR branches do not match.")
    return result


def score(
    labels: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float | int]:
    true = np.asarray(labels, dtype=np.int64) != 0
    probability = np.asarray(probability, dtype=np.float64)
    predicted = probability >= np.asarray(thresholds, dtype=np.float64)
    return {
        "rows": int(len(true)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "sensitivity": float(np.mean(predicted[true])),
        "specificity": float(np.mean(~predicted[~true])),
        "missed_intent_rate": float(np.mean(~predicted[true])),
        "false_activation_rate_per_window": float(np.mean(predicted[~true])),
        "auroc": float(roc_auc_score(true, probability)),
        "auprc": float(average_precision_score(true, probability)),
        "brier_score": float(brier_score_loss(true, probability)),
    }


def domain_thresholds(
    labels: np.ndarray,
    probability: np.ndarray,
    dataset_ids: np.ndarray,
    mask: np.ndarray,
    max_false_activation: float,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for domain in np.unique(dataset_ids[mask]):
        selected = mask & (dataset_ids == domain)
        thresholds[str(domain)] = select_intent_threshold(
            labels[selected],
            probability[selected],
            max_false_activation=max_false_activation,
        )
    return thresholds


def threshold_vector(dataset_ids: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return np.asarray([thresholds[str(domain)] for domain in dataset_ids], dtype=np.float64)


def participant_analysis(
    labels: np.ndarray,
    random_probability: np.ndarray,
    ssl_probability: np.ndarray,
    random_threshold: np.ndarray,
    ssl_threshold: np.ndarray,
    subject_keys: np.ndarray,
    dataset_ids: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    paired = np.asarray(
        [f"{dataset}:{subject}" for dataset, subject in zip(dataset_ids, subject_keys, strict=True)]
    )
    for key in np.unique(paired):
        selected = paired == key
        if len(np.unique(labels[selected])) != 2:
            continue
        random_result = score(
            labels[selected],
            random_probability[selected],
            random_threshold[selected],
        )
        ssl_result = score(
            labels[selected],
            ssl_probability[selected],
            ssl_threshold[selected],
        )
        dataset, subject = key.split(":", maxsplit=1)
        records.append(
            {
                "dataset_id": dataset,
                "subject_key": subject,
                "rows": int(np.sum(selected)),
                "random_cnn_balanced_accuracy": random_result["balanced_accuracy"],
                "way_ssl_balanced_accuracy": ssl_result["balanced_accuracy"],
                "balanced_accuracy_delta": ssl_result["balanced_accuracy"]
                - random_result["balanced_accuracy"],
                "random_cnn_sensitivity": random_result["sensitivity"],
                "way_ssl_sensitivity": ssl_result["sensitivity"],
            }
        )
    deltas = np.asarray([record["balanced_accuracy_delta"] for record in records])
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True)
    bootstrap = np.mean(samples, axis=1)
    return {
        "participants": len(records),
        "mean_balanced_accuracy_delta": float(np.mean(deltas)),
        "median_balanced_accuracy_delta": float(np.median(deltas)),
        "bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "participants_improved_fraction": float(np.mean(deltas > 0.0)),
        "bootstrap_iterations": iterations,
        "records": records,
    }


def _plot_losses(
    ssl_losses: list[float],
    random_losses: list[float],
    pretrained_losses: list[float],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    axes[0].plot(np.arange(1, len(ssl_losses) + 1), ssl_losses, marker="o", color=TEAL, linewidth=2.5)
    axes[0].set_title("Unlabeled WAY masked reconstruction", loc="left", weight="bold", color=INK)
    axes[0].set_xlabel("Pretraining epoch")
    axes[0].set_ylabel("Reconstruction loss")
    axes[1].plot(
        np.arange(1, len(random_losses) + 1),
        random_losses,
        marker="o",
        color=BLUE,
        linewidth=2.5,
        label="Random initialization",
    )
    axes[1].plot(
        np.arange(1, len(pretrained_losses) + 1),
        pretrained_losses,
        marker="o",
        color=TEAL,
        linewidth=2.5,
        label="WAY EEG-only pretrained",
    )
    axes[1].set_title("Original-dataset intent fine-tuning", loc="left", weight="bold", color=INK)
    axes[1].set_xlabel("Fine-tuning epoch")
    axes[1].set_ylabel("Weighted training loss")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(color="#D5DEE2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle("WAY EEG-Only Self-Supervised Training Audit", fontsize=17, weight="bold", color=INK)
    fig.savefig(output / "01_way_eeg_ssl_training_curves.png", dpi=220)
    plt.close(fig)


def _plot_results(
    evaluation: dict[str, Any],
    evaluation_by_dataset: dict[str, Any],
    participant: dict[str, Any],
    output: Path,
) -> None:
    models = tuple(MODEL_LABELS)
    metrics = ("balanced_accuracy", "sensitivity", "specificity", "auroc")
    metric_labels = ("Balanced accuracy", "Sensitivity", "Specificity", "AUROC")
    positions = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.5, 6.5), constrained_layout=True)
    for index, model in enumerate(models):
        values = 100.0 * np.asarray([evaluation[model][metric] for metric in metrics])
        offset = (index - 1.5) * width
        ax.bar(positions + offset, values, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
        for position, value in zip(positions + offset, values, strict=True):
            ax.text(position, value + 1.0, f"{value:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(positions, metric_labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Development-observed result (%)")
    ax.set_title("Does Unlabeled WAY EEG Improve Motor-Imagery Intent Detection?", loc="left", weight="bold", color=INK)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color="#D5DEE2", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "02_way_eeg_ssl_overall_comparison.png", dpi=220)
    plt.close(fig)

    domains = tuple(evaluation_by_dataset)
    x = np.arange(len(domains))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), constrained_layout=True)
    for index, model in enumerate(models):
        values = 100.0 * np.asarray(
            [evaluation_by_dataset[domain][model]["balanced_accuracy"] for domain in domains]
        )
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, values, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    labels = [domain.upper() if domain == "openbmi" else domain.title() for domain in domains]
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(40, 70)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[0].set_title("EEG-only transfer by unseen target domain", loc="left", weight="bold", color=INK)
    axes[0].legend(frameon=False, fontsize=8, ncols=2)

    records = participant["records"]
    colors = {"liu2024": CORAL, "openbmi": TEAL, "physionet": GOLD}
    for index, domain in enumerate(domains):
        deltas = 100.0 * np.asarray(
            [record["balanced_accuracy_delta"] for record in records if record["dataset_id"] == domain]
        )
        jitter = np.linspace(-0.13, 0.13, len(deltas)) if len(deltas) > 1 else np.zeros(len(deltas))
        axes[1].scatter(
            np.full(len(deltas), index) + jitter,
            deltas,
            color=colors.get(domain, BLUE),
            edgecolor="white",
            linewidth=0.5,
            s=44,
            alpha=0.85,
        )
        if len(deltas):
            axes[1].plot([index - 0.18, index + 0.18], [np.mean(deltas)] * 2, color=INK, linewidth=3)
    axes[1].axhline(0.0, color=INK, linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("SSL BA change vs random CNN (points)")
    axes[1].set_title("Paired participant pretraining effect", loc="left", weight="bold", color=INK)
    for axis in axes:
        axis.grid(axis="y", color="#D5DEE2", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
    low, high = participant["bootstrap_95_ci"]
    fig.suptitle(
        f"WAY EEG-Only SSL | Mean participant change {100 * participant['mean_balanced_accuracy_delta']:+.1f} points "
        f"(95% CI {100 * low:+.1f} to {100 * high:+.1f})",
        fontsize=16,
        weight="bold",
        color=INK,
    )
    fig.savefig(output / "03_way_eeg_ssl_domain_and_participants.png", dpi=220)
    plt.close(fig)


def _write_report(payload: dict[str, Any], output: Path) -> None:
    evaluation = payload["evaluation"]
    participant = payload["participant_analysis"]
    low, high = participant["bootstrap_95_ci"]
    lines = [
        "# WAY EEG-Only Self-Supervised Transfer",
        "",
        "## Contract",
        "",
        "- All 120 continuous WAY EEG series are eligible, including 12 EEG-only challenge series.",
        "- WAY EMG, kinematics, force, task events, and class labels are never read by this experiment.",
        "- The encoder learns by reconstructing masked samples and channels in unlabeled two-second EEG windows.",
        "- The pretrained and random-control networks use the same architecture, fine-tuning rows, sample weighting, epochs, and controller threshold procedure.",
        "- Fine-tuning uses original-dataset development subjects; thresholds use controller subjects; development-observed subjects are evaluated once.",
        "",
        "## Source Audit",
        "",
        f"- Unlabeled windows: **{payload['source_audit']['windows']}**.",
        f"- Continuous series: **{payload['source_audit']['accepted_series']}/{payload['source_audit']['discovered_series']}** across **{payload['source_audit']['participants']}** participants.",
        f"- EEG-only challenge windows: **{payload['source_audit']['challenge_windows']}**.",
        "",
        "## Development-Observed Result",
        "",
        "| Model | Balanced accuracy | Sensitivity | Specificity | AUROC | AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_LABELS:
        result = evaluation[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {100 * result['balanced_accuracy']:.2f}% | "
            f"{100 * result['sensitivity']:.2f}% | {100 * result['specificity']:.2f}% | "
            f"{result['auroc']:.3f} | {result['auprc']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Against the architecture-matched random CNN, the equal-participant mean SSL change was **{100 * participant['mean_balanced_accuracy_delta']:+.2f} points** (95% participant-bootstrap CI **{100 * low:+.2f} to {100 * high:+.2f}**); **{100 * participant['participants_improved_fraction']:.1f}%** of participants improved.",
            "",
            "## Dataset Results",
            "",
            "| Dataset | Random CNN BA | WAY SSL BA | Change |",
            "|---|---:|---:|---:|",
        ]
    )
    for domain, comparison in payload["evaluation_by_dataset"].items():
        random_result = comparison["random_cnn"]
        ssl_result = comparison["way_ssl"]
        delta = ssl_result["balanced_accuracy"] - random_result["balanced_accuracy"]
        lines.append(
            f"| {domain} | {100 * random_result['balanced_accuracy']:.2f}% | "
            f"{100 * ssl_result['balanced_accuracy']:.2f}% | {100 * delta:+.2f} points |"
        )
    lines.extend(
        [
            "",
            "This experiment isolates the value of unlabeled WAY EEG. It does not claim that overt-movement EEG is motor imagery, and it does not use WAY cue timing as an intent label. Continuous no-control and prospective patient testing remain necessary.",
        ]
    )
    (output / "WAY_EEG_ONLY_SSL_TRANSFER_REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--way-dir", type=Path, default=PROJECT_ROOT / "WAY-EEG-GAL")
    parser.add_argument(
        "--way-cache",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "way_eeg_only_ssl" / "way_unlabeled_eeg_windows.npz",
    )
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "way_eeg_only_ssl",
    )
    parser.add_argument("--rebuild-way-cache", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows-per-series", type=int, default=160)
    parser.add_argument("--max-controller-false-activation", type=float, default=0.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=TORCH_DEVICE_CHOICES,
        default="auto",
        help="PyTorch training backend; auto prefers CUDA, then Apple MPS, then CPU.",
    )
    args = parser.parse_args()
    if args.pretrain_epochs <= 0 or args.fine_tune_epochs <= 0 or args.batch_size <= 0:
        parser.error("Epochs and batch size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    device = resolve_torch_device(args.device)
    device_metadata = torch_device_metadata(args.device, device)
    print(f"[+] Neural compute device: {device} (requested: {args.device})")

    qc_path = args.way_cache.with_name(f"{args.way_cache.stem}.qc.json")
    if args.rebuild_way_cache or not args.way_cache.is_file():
        print("[+] Building unlabeled fixed-grid WAY EEG windows from all continuous series...")
        source, source_qc = build_way_eeg_ssl_windows(
            args.way_dir,
            max_windows_per_series=args.max_windows_per_series,
        )
        save_way_ssl_cache(args.way_cache, source, source_qc)
    else:
        print(f"[+] Loading unlabeled WAY EEG cache: {args.way_cache}")
        source = load_way_ssl_cache(args.way_cache)
        source_qc = json.loads(qc_path.read_text(encoding="utf-8"))
    scales = robust_branch_scales(source.mrcp_epochs, source.smr_epochs)
    source_dataset = BranchEpochDataset(source.mrcp_epochs, source.smr_epochs)
    torch.manual_seed(args.seed)
    autoencoder = MaskedEEGAutoencoder()
    print(
        f"[+] Self-supervised pretraining on {len(source_dataset)} unlabeled EEG windows "
        f"from {source_qc['accepted_series']} series..."
    )
    ssl_losses = train_masked_autoencoder(
        autoencoder,
        source_dataset,
        scales,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        device=device,
        seed=args.seed,
    )
    pretrained_backbone = copied_backbone(autoencoder.backbone)
    torch.save(
        {
            "schema_version": 1,
            "backbone_state": pretrained_backbone.state_dict(),
            "branch_scales": {"mrcp": scales.mrcp, "smr": scales.smr},
            "source_qc": source_qc,
            "peripheral_inputs_used": False,
        },
        args.output_dir / "way_eeg_ssl_pretrained_backbone.pt",
    )
    del source_dataset, source, autoencoder
    gc.collect()

    print("[+] Loading row-aligned original-dataset MRCP/SMR windows...")
    target = load_target_contract(args.target_raw_cache, args.target_feature_cache)
    mrcp = np.asarray(target["mrcp_epochs"], dtype=np.float32)
    smr = np.asarray(target["smr_epochs"], dtype=np.float32)
    labels = np.asarray(target["labels"], dtype=np.int64)
    datasets = np.asarray(target["dataset_ids"]).astype(str)
    subjects = np.asarray(target["subject_keys"]).astype(str)
    roles = np.asarray(target["roles"]).astype(str)
    development = roles == "development"
    controller = roles == "controller_validation"
    observed = roles == "development_observed"
    full_weights = np.ones(len(labels), dtype=np.float32)
    full_weights[development] = dataset_balanced_weights(
        labels[development],
        datasets[development],
    ).astype(np.float32)
    training_dataset = BranchEpochDataset(
        mrcp,
        smr,
        indices=np.flatnonzero(development),
        labels=labels,
        sample_weights=full_weights,
    )
    evaluation_dataset = BranchEpochDataset(mrcp, smr)

    torch.manual_seed(args.seed + 1)
    random_model = EEGIntentNetwork(EEGSSLBackbone())
    torch.manual_seed(args.seed + 2)
    pretrained_model = EEGIntentNetwork(pretrained_backbone)
    random_model.classifier.load_state_dict(pretrained_model.classifier.state_dict())
    print("[+] Fine-tuning architecture-matched random control on original datasets...")
    random_losses = train_intent_network(
        random_model,
        training_dataset,
        scales,
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        encoder_learning_rate=3e-4,
        head_learning_rate=1e-3,
        device=device,
        seed=args.seed + 100,
    )
    print("[+] Fine-tuning WAY EEG-only pretrained encoder on original datasets...")
    pretrained_losses = train_intent_network(
        pretrained_model,
        training_dataset,
        scales,
        epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        encoder_learning_rate=3e-4,
        head_learning_rate=1e-3,
        device=device,
        seed=args.seed + 100,
    )
    print("[+] Scoring controller and development-observed subjects...")
    random_probability = predict_intent_probabilities(
        random_model,
        evaluation_dataset,
        scales,
        batch_size=args.batch_size,
        device=device,
    )
    ssl_probability = predict_intent_probabilities(
        pretrained_model,
        evaluation_dataset,
        scales,
        batch_size=args.batch_size,
        device=device,
    )
    comparison = joblib.load(args.comparison_model)
    components = comparison.component_probabilities(np.asarray(target["eeg_features"], dtype=np.float64))
    model_probabilities = {
        "classical_target_only": components[:, 0],
        "random_cnn": random_probability,
        "way_ssl": ssl_probability,
        "way_privileged": components[:, 2],
    }
    model_thresholds = {
        model: domain_thresholds(
            labels,
            probability,
            datasets,
            controller,
            args.max_controller_false_activation,
        )
        for model, probability in model_probabilities.items()
    }
    threshold_vectors = {
        model: threshold_vector(datasets, thresholds)
        for model, thresholds in model_thresholds.items()
    }
    evaluation = {
        model: score(
            labels[observed],
            probability[observed],
            threshold_vectors[model][observed],
        )
        for model, probability in model_probabilities.items()
    }
    evaluation_by_dataset: dict[str, Any] = {}
    for domain in np.unique(datasets[observed]):
        selected = observed & (datasets == domain)
        evaluation_by_dataset[str(domain)] = {
            model: score(
                labels[selected],
                probability[selected],
                threshold_vectors[model][selected],
            )
            for model, probability in model_probabilities.items()
        }
    participant = participant_analysis(
        labels[observed],
        random_probability[observed],
        ssl_probability[observed],
        threshold_vectors["random_cnn"][observed],
        threshold_vectors["way_ssl"][observed],
        subjects[observed],
        datasets[observed],
        seed=args.seed,
        iterations=args.bootstrap_iterations,
    )
    payload = {
        "schema_version": 1,
        "experiment": "way_eeg_only_masked_reconstruction_transfer",
        "neural_seed": args.seed,
        "inputs": {
            "way_cache": str(args.way_cache.resolve()),
            "target_raw_cache": str(args.target_raw_cache.resolve()),
            "target_feature_cache": str(args.target_feature_cache.resolve()),
            "comparison_model": str(args.comparison_model.resolve()),
        },
        "source_audit": source_qc,
        "source_branch_scales": {"mrcp": scales.mrcp, "smr": scales.smr},
        "training": {
            "pretrain_epochs": args.pretrain_epochs,
            "fine_tune_epochs": args.fine_tune_epochs,
            "batch_size": args.batch_size,
            "neural_seed": args.seed,
            "max_controller_false_activation": args.max_controller_false_activation,
            "device": str(device),
            "device_metadata": device_metadata,
            "development_rows": int(np.sum(development)),
            "controller_rows": int(np.sum(controller)),
            "development_observed_rows": int(np.sum(observed)),
            "ssl_reconstruction_losses": ssl_losses,
            "random_control_losses": random_losses,
            "pretrained_fine_tune_losses": pretrained_losses,
        },
        "leakage_boundary": {
            "way_labels_used": False,
            "way_events_used": False,
            "way_emg_used": False,
            "way_kinematics_used": False,
            "way_force_used": False,
            "controller_used_for_gradient_training": False,
            "observed_used_for_selection": False,
        },
        "domain_thresholds": model_thresholds,
        "evaluation": evaluation,
        "evaluation_by_dataset": evaluation_by_dataset,
        "participant_analysis": participant,
    }
    (args.output_dir / "way_eeg_only_ssl_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        model_payload(
            pretrained_model,
            scales,
            model_thresholds["way_ssl"],
            target["channel_names"],
        ),
        args.output_dir / "way_eeg_ssl_intent_decoder.pt",
    )
    torch.save(
        model_payload(
            random_model,
            scales,
            model_thresholds["random_cnn"],
            target["channel_names"],
        ),
        args.output_dir / "random_init_intent_decoder_control.pt",
    )
    _plot_losses(ssl_losses, random_losses, pretrained_losses, args.output_dir)
    _plot_results(evaluation, evaluation_by_dataset, participant, args.output_dir)
    _write_report(payload, args.output_dir)
    ssl_result = evaluation["way_ssl"]
    random_result = evaluation["random_cnn"]
    print(
        "[+] Development-observed WAY EEG-only SSL: "
        f"BA={100 * ssl_result['balanced_accuracy']:.2f}% | "
        f"sensitivity={100 * ssl_result['sensitivity']:.2f}% | "
        f"specificity={100 * ssl_result['specificity']:.2f}% | "
        f"AUROC={ssl_result['auroc']:.3f}"
    )
    print(
        f"[+] Architecture-matched BA change: "
        f"{100 * (ssl_result['balanced_accuracy'] - random_result['balanced_accuracy']):+.2f} points"
    )
    print(f"[+] Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
