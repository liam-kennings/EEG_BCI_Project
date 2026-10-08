"""Aggregate controlled neural-seed repetitions for WAY EEG-only SSL."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import t


SUMMARY_METRICS = (
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "auroc",
    "auprc",
)


def _mean_interval(values: Sequence[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Seed statistics require a non-empty finite vector.")
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if len(array) > 1:
        half_width = float(t.ppf(0.975, len(array) - 1) * standard_deviation / np.sqrt(len(array)))
    else:
        half_width = 0.0
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "t_95_ci": [mean - half_width, mean + half_width],
    }


def _assert_fixed_contract(payloads: Sequence[dict[str, Any]]) -> None:
    first = payloads[0]
    expected = {
        "source_windows": first["source_audit"]["windows"],
        "source_series": first["source_audit"]["accepted_series"],
        "development_rows": first["training"]["development_rows"],
        "controller_rows": first["training"]["controller_rows"],
        "observed_rows": first["training"]["development_observed_rows"],
        "pretrain_epochs": first["training"]["pretrain_epochs"],
        "fine_tune_epochs": first["training"]["fine_tune_epochs"],
        "batch_size": first["training"]["batch_size"],
    }
    for payload in payloads[1:]:
        observed = {
            "source_windows": payload["source_audit"]["windows"],
            "source_series": payload["source_audit"]["accepted_series"],
            "development_rows": payload["training"]["development_rows"],
            "controller_rows": payload["training"]["controller_rows"],
            "observed_rows": payload["training"]["development_observed_rows"],
            "pretrain_epochs": payload["training"]["pretrain_epochs"],
            "fine_tune_epochs": payload["training"]["fine_tune_epochs"],
            "batch_size": payload["training"]["batch_size"],
        }
        if observed != expected:
            raise ValueError("Seed runs do not share an identical data and training contract.")
        if payload.get("inputs") != first.get("inputs"):
            raise ValueError("Seed runs do not use the same input artifacts.")
        if payload["source_branch_scales"] != first["source_branch_scales"]:
            raise ValueError("Seed runs do not use the same source normalization.")
    for payload in payloads:
        if any(payload["leakage_boundary"].values()):
            raise ValueError("A seed run violates the declared leakage boundary.")


def summarize_way_ssl_seeds(
    payloads: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    *,
    participant_bootstrap_iterations: int = 5000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Summarize paired random-control and pretrained results across seeds."""
    if len(payloads) != len(seeds) or not payloads:
        raise ValueError("Provide one non-empty payload for every seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Neural seeds must be unique.")
    _assert_fixed_contract(payloads)

    per_seed: list[dict[str, Any]] = []
    for seed, payload in zip(seeds, payloads, strict=True):
        random_result = payload["evaluation"]["random_cnn"]
        ssl_result = payload["evaluation"]["way_ssl"]
        per_seed.append(
            {
                "seed": int(seed),
                "random_cnn": {metric: float(random_result[metric]) for metric in SUMMARY_METRICS},
                "way_ssl": {metric: float(ssl_result[metric]) for metric in SUMMARY_METRICS},
                "ssl_minus_random": {
                    metric: float(ssl_result[metric] - random_result[metric])
                    for metric in SUMMARY_METRICS
                },
                "participant_mean_balanced_accuracy_delta": float(
                    payload["participant_analysis"]["mean_balanced_accuracy_delta"]
                ),
            }
        )

    aggregate = {
        model: {
            metric: _mean_interval(
                [float(payload["evaluation"][model][metric]) for payload in payloads]
            )
            for metric in SUMMARY_METRICS
        }
        for model in ("random_cnn", "way_ssl")
    }
    fixed_references = {
        model: {
            metric: float(payloads[0]["evaluation"][model][metric])
            for metric in SUMMARY_METRICS
        }
        for model in ("classical_target_only", "way_privileged")
    }
    for payload in payloads[1:]:
        for model, reference in fixed_references.items():
            observed = np.asarray(
                [payload["evaluation"][model][metric] for metric in SUMMARY_METRICS]
            )
            expected_reference = np.asarray([reference[metric] for metric in SUMMARY_METRICS])
            if not np.allclose(observed, expected_reference, rtol=0.0, atol=1e-12):
                raise ValueError("Non-neural reference metrics changed across seed runs.")
    paired_delta = {
        metric: _mean_interval(
            [record["ssl_minus_random"][metric] for record in per_seed]
        )
        for metric in SUMMARY_METRICS
    }

    domains = tuple(sorted(payloads[0]["evaluation_by_dataset"]))
    domain_summary: dict[str, Any] = {}
    domain_model_summary: dict[str, Any] = {}
    for domain in domains:
        values = [
            float(
                payload["evaluation_by_dataset"][domain]["way_ssl"]["balanced_accuracy"]
                - payload["evaluation_by_dataset"][domain]["random_cnn"]["balanced_accuracy"]
            )
            for payload in payloads
        ]
        domain_summary[domain] = {
            "balanced_accuracy_delta": _mean_interval(values),
            "per_seed": values,
            "positive_seed_fraction": float(np.mean(np.asarray(values) > 0.0)),
        }
        domain_model_summary[domain] = {}
        for model in ("random_cnn", "way_ssl"):
            domain_model_summary[domain][model] = {}
            for metric in SUMMARY_METRICS:
                model_values = [
                    float(payload["evaluation_by_dataset"][domain][model][metric])
                    for payload in payloads
                ]
                metric_summary = _mean_interval(model_values)
                metric_summary["per_seed"] = model_values
                domain_model_summary[domain][model][metric] = metric_summary

    participant_keys = {
        (record["dataset_id"], record["subject_key"])
        for record in payloads[0]["participant_analysis"]["records"]
    }
    participant_deltas: list[float] = []
    for key in sorted(participant_keys):
        values = []
        for payload in payloads:
            lookup = {
                (record["dataset_id"], record["subject_key"]): record
                for record in payload["participant_analysis"]["records"]
            }
            if key not in lookup:
                raise ValueError("Participant records are not aligned across seed runs.")
            values.append(float(lookup[key]["balanced_accuracy_delta"]))
        participant_deltas.append(float(np.mean(values)))
    participant_array = np.asarray(participant_deltas, dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = rng.choice(
        participant_array,
        size=(participant_bootstrap_iterations, len(participant_array)),
        replace=True,
    ).mean(axis=1)
    participant_summary = {
        "participants": len(participant_deltas),
        "mean_seed_averaged_balanced_accuracy_delta": float(np.mean(participant_array)),
        "median_seed_averaged_balanced_accuracy_delta": float(np.median(participant_array)),
        "participant_bootstrap_95_ci": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "participants_improved_fraction": float(np.mean(participant_array > 0.0)),
        "bootstrap_iterations": participant_bootstrap_iterations,
    }

    ba_delta = np.asarray(
        [record["ssl_minus_random"]["balanced_accuracy"] for record in per_seed]
    )
    specificity_delta = paired_delta["specificity"]["mean"]
    primary_stability = {
        "at_least_four_of_five_positive": bool(np.sum(ba_delta > 0.0) >= 4),
        "positive_mean_balanced_accuracy_delta": bool(np.mean(ba_delta) > 0.0),
        "seed_t_interval_excludes_zero": bool(
            paired_delta["balanced_accuracy"]["t_95_ci"][0] > 0.0
        ),
    }
    safety = {
        "mean_specificity_erosion_within_one_point": bool(specificity_delta >= -0.01),
    }
    cross_domain = {
        "openbmi_mean_delta_positive": bool(
            domain_summary.get("openbmi", {})
            .get("balanced_accuracy_delta", {})
            .get("mean", -np.inf)
            > 0.0
        ),
        "physionet_mean_delta_positive": bool(
            domain_summary.get("physionet", {})
            .get("balanced_accuracy_delta", {})
            .get("mean", -np.inf)
            > 0.0
        ),
    }
    criteria = {
        "primary_stability": primary_stability,
        "safety": safety,
        "cross_domain_replication": cross_domain,
        "all_predeclared_criteria_met": bool(
            all(primary_stability.values())
            and all(safety.values())
            and all(cross_domain.values())
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "way_eeg_only_ssl_neural_seed_stability",
        "seeds": [int(seed) for seed in seeds],
        "seed_count": len(seeds),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "fixed_references": fixed_references,
        "paired_ssl_minus_random": paired_delta,
        "domain_balanced_accuracy_delta": domain_summary,
        "domain_model_summary": domain_model_summary,
        "participant_analysis": participant_summary,
        "predeclared_criteria": criteria,
        "interpretation": {
            "tests_training_randomness": True,
            "independent_patient_cohort": False,
            "development_observed_reused_across_seeds": True,
        },
    }
