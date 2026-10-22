"""Validation-only representation analysis and decision for Stage 2C."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.representation_health import (
    assess_layer_health,
    compute_layer_health,
)
from evaluation.representations import extract_representations
from models import ConvAutoencoder
from schemas import load_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum
from utils.results import write_json


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("h1", "h2", "z")
ENCODER_NAMES = {"h1": "enc1", "h2": "enc2", "z": "enc3"}
DIVERSITY_CHECKS = (
    "finite",
    "active_unit_ratio",
    "nonzero_variance_ratio",
    "effective_rank",
    "normalized_effective_rank",
)
WTA_CHECKS = (
    "winner_density_matches_topk",
    "winner_coverage_ratio",
    "winner_entropy",
    "max_winner_share",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validation_loader(protocol: dict[str, Any]) -> tuple[DataLoader, np.ndarray]:
    spec = protocol["representation"]
    subset = np.load(_resolve(spec["subset_manifest"]), allow_pickle=False)
    sample_ids = subset["sample_ids"].astype(np.int64)
    labels = subset["labels"].astype(np.int64)
    if sample_ids.size != 2000 or not all(
        int((labels == class_id).sum()) == 200 for class_id in range(10)
    ):
        raise RuntimeError("Representation subset is not fixed class-balanced 2,000")
    training = datasets.MNIST(
        root=str(_resolve(spec["data_root"])),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        IndexedDataset(training, sample_ids),
        batch_size=int(spec["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return loader, labels


def _stable_rank(activations: torch.Tensor, epsilon: float) -> float:
    values = (
        activations.permute(0, 2, 3, 1)
        .reshape(-1, activations.shape[1])
        .to(torch.float64)
    )
    values -= values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / values.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    largest = float(eigenvalues.max().item())
    return 0.0 if largest <= epsilon else float(eigenvalues.sum().item() / largest)


def _mean_abs_channel_correlation(
    activations: torch.Tensor, epsilon: float
) -> float:
    values = (
        activations.permute(0, 2, 3, 1)
        .reshape(-1, activations.shape[1])
        .to(torch.float64)
    )
    values -= values.mean(dim=0, keepdim=True)
    std = values.square().mean(dim=0).sqrt()
    valid = std > epsilon
    if int(valid.sum()) < 2:
        return 0.0
    standardized = values[:, valid] / std[valid]
    correlation = standardized.T @ standardized / standardized.shape[0]
    mask = ~torch.eye(correlation.shape[0], dtype=torch.bool)
    return float(correlation[mask].abs().mean().item())


def _mean_abs_filter_cosine(weight: torch.Tensor, epsilon: float) -> float:
    filters = weight.detach().cpu().flatten(start_dim=1).to(torch.float64)
    norms = filters.norm(dim=1, keepdim=True)
    valid = norms.squeeze(1) > epsilon
    filters = filters[valid]
    if filters.shape[0] < 2:
        return 0.0
    normalized = filters / norms[valid]
    similarities = normalized @ normalized.T
    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool)
    return float(similarities[mask].abs().mean().item())


def _separation_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    scaler = StandardScaler()
    values = scaler.fit_transform(features)
    global_mean = values.mean(axis=0)
    between = 0.0
    within = 0.0
    for class_id in range(10):
        class_values = values[labels == class_id]
        centroid = class_values.mean(axis=0)
        between += class_values.shape[0] * float(
            np.square(centroid - global_mean).sum()
        )
        within += float(np.square(class_values - centroid).sum())
    return between / max(within, 1e-12)


def _cross_validated_scores(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
    neighbors: int,
) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    linear_scores: list[float] = []
    knn_scores: list[float] = []
    for train_index, validation_index in splitter.split(features, labels):
        scaler = StandardScaler()
        train = scaler.fit_transform(features[train_index])
        validation = scaler.transform(features[validation_index])
        linear = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
            shuffle=True,
        )
        linear.fit(train, labels[train_index])
        linear_scores.append(
            accuracy_score(labels[validation_index], linear.predict(validation))
        )
        knn = KNeighborsClassifier(n_neighbors=neighbors)
        knn.fit(train, labels[train_index])
        knn_scores.append(
            accuracy_score(labels[validation_index], knn.predict(validation))
        )
    return float(np.mean(linear_scores)), float(np.mean(knn_scores))


def _final_probe_metrics(run_dir: Path) -> dict[str, float]:
    rows = []
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                raise RuntimeError(f"Test metric found in Stage 2C: {run_dir}")
            if row["stage"] == "linear_probe_final" and row["split"] == "validation":
                rows.append(row)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one final validation probe row: {run_dir}")
    row = rows[0]
    return {
        "validation_accuracy": float(row["accuracy"]),
        "validation_macro_f1": float(row["macro_f1"]),
        "validation_classification_ce": float(row["classification_ce"]),
    }


def _plots(
    output_dir: Path,
    representation_rows: list[dict[str, Any]],
    performance_rows: list[dict[str, Any]],
) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = [row["method_id"] for row in performance_rows]
    display = {row["method_id"]: row["display_name"] for row in performance_rows}
    x = np.arange(len(LAYERS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, method in enumerate(methods):
        rows = [row for row in representation_rows if row["method_id"] == method]
        ax.bar(
            x + (index - 1.5) * width,
            [row["effective_rank"] for row in rows],
            width,
            label=display[method],
        )
    ax.set_xticks(x, LAYERS)
    ax.set_ylabel("Participation-ratio effective rank")
    ax.legend()
    fig.tight_layout()
    path_rank = figures / "layerwise_effective_rank.png"
    fig.savefig(path_rank, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, method in enumerate(methods):
        rows = [row for row in representation_rows if row["method_id"] == method]
        ax.bar(
            x + (index - 1.5) * width,
            [row["filter_pairwise_abs_cosine"] for row in rows],
            width,
            label=display[method],
        )
    ax.set_xticks(x, LAYERS)
    ax.set_ylabel("Mean |filter cosine|")
    ax.legend()
    fig.tight_layout()
    path_filter = figures / "layerwise_filter_similarity.png"
    fig.savefig(path_filter, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        [display[row["method_id"]] for row in performance_rows],
        [row["validation_accuracy"] for row in performance_rows],
    )
    axes[0].axhline(0.8863, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Frozen-probe validation accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(
        [display[row["method_id"]] for row in performance_rows],
        [row["validation_reconstruction_mse"] for row in performance_rows],
    )
    axes[1].set_ylabel("Validation reconstruction MSE")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path_performance = figures / "validation_performance.png"
    fig.savefig(path_performance, dpi=180)
    plt.close(fig)

    z_rows = [row for row in representation_rows if row["layer"] == "z"]
    accuracy = {row["method_id"]: row["validation_accuracy"] for row in performance_rows}
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in z_rows:
        ax.scatter(
            row["effective_rank"],
            accuracy[row["method_id"]],
            s=70,
            label=display[row["method_id"]],
        )
    ax.set_xlabel("z effective rank")
    ax.set_ylabel("Validation accuracy")
    ax.legend()
    fig.tight_layout()
    path_diversity = figures / "diversity_vs_performance.png"
    fig.savefig(path_diversity, dpi=180)
    plt.close(fig)

    matrix = np.asarray(
        [
            [
                next(
                    row["applicable_health_pass_fraction"]
                    for row in representation_rows
                    if row["method_id"] == method and row["layer"] == layer
                )
                for layer in LAYERS
            ]
            for method in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(LAYERS)), LAYERS)
    ax.set_yticks(
        range(len(methods)), [display[method] for method in methods]
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax, label="Applicable health checks passed")
    fig.tight_layout()
    path_matrix = figures / "method_layer_health_matrix.png"
    fig.savefig(path_matrix, dpi=180)
    plt.close(fig)
    return [
        str(path_rank),
        str(path_filter),
        str(path_performance),
        str(path_diversity),
        str(path_matrix),
    ]


def run_analysis(config_path: str | Path) -> Path:
    config_path = _resolve(config_path)
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if protocol["version"] != "hybrid-depth-ablation-v1":
        raise ValueError("Unsupported hybrid-depth protocol")
    output_dir = _resolve(protocol["output_dir"])
    pairing_gate = _json(output_dir / "pairing_gate.json")
    if pairing_gate["decision"] != "PASS":
        raise RuntimeError("Pairing gate must pass before representation analysis")
    health_source = yaml.safe_load(
        _resolve(protocol["health_threshold_source"]).read_text(encoding="utf-8")
    )
    thresholds = health_source["thresholds"]
    loader, expected_labels = _validation_loader(protocol)
    representation_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    device = torch.device("cpu")
    spec = protocol["representation"]

    for method in protocol["methods"]:
        method_id = method["id"]
        method_config = load_config(_resolve(method["config"]))
        run_dir = output_dir / "runs" / f"{method_id}_seed42"
        model = ConvAutoencoder(
            method_config["model"]["latent_dim"],
            seed=int(protocol["seed"]),
        )
        model.load_state_dict(
            torch.load(
                run_dir / "model_best.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        checksum_before = state_dict_checksum(model)
        extracted = extract_representations(
            model, loader, device=device, layers=LAYERS
        )
        checksum_after = state_dict_checksum(model)
        if checksum_before != checksum_after:
            raise RuntimeError("Representation extraction mutated the model")
        labels = extracted["label"].numpy()
        if not np.array_equal(labels, expected_labels):
            raise RuntimeError("Representation labels differ from fixed manifest")
        integrity[method_id] = {
            "checkpoint_sha256": file_sha256(run_dir / "model_best.pt"),
            "state_dict_sha256_before": checksum_before,
            "state_dict_sha256_after": checksum_after,
            "unchanged": checksum_before == checksum_after,
        }

        for layer in LAYERS:
            activations = extracted[layer]
            health, _, _ = compute_layer_health(
                activations,
                winner_fraction=float(spec["winner_fraction"]),
                activation_epsilon=float(spec["activation_epsilon"]),
                variance_epsilon=float(spec["variance_epsilon"]),
            )
            assessment = assess_layer_health(health, thresholds)
            encoder_layer = ENCODER_NAMES[layer]
            layer_rule = method_config["hybrid"]["encoder_layer_rules"][
                encoder_layer
            ]
            applicable = list(DIVERSITY_CHECKS)
            if layer_rule == "hebbian":
                applicable.extend(WTA_CHECKS)
            applicable_pass_count = sum(
                bool(assessment["checks"][check]) for check in applicable
            )
            features = activations.flatten(start_dim=1).numpy().astype(
                np.float32, copy=False
            )
            linear_accuracy, knn_accuracy = _cross_validated_scores(
                features,
                labels,
                folds=int(spec["cv_folds"]),
                seed=int(spec["cv_seed"]),
                neighbors=int(spec["knn_neighbors"]),
            )
            weight = getattr(model.encoder, encoder_layer).weight
            row = {
                "method_id": method_id,
                "display_name": method_config["hybrid"]["display_name"],
                "layer": layer,
                "encoder_layer": encoder_layer,
                "layer_rule": layer_rule,
                "effective_rank": float(health["effective_rank"]),
                "normalized_effective_rank": float(
                    health["normalized_effective_rank"]
                ),
                "stable_rank": _stable_rank(
                    activations, float(spec["variance_epsilon"])
                ),
                "activation_variance": float(health["mean_unit_variance"]),
                "pairwise_activation_abs_correlation": (
                    _mean_abs_channel_correlation(
                        activations, float(spec["variance_epsilon"])
                    )
                ),
                "filter_pairwise_abs_cosine": _mean_abs_filter_cosine(
                    weight, float(spec["variance_epsilon"])
                ),
                "dead_unit_ratio": float(health["dead_unit_ratio"]),
                "winner_coverage_ratio": (
                    float(health["winner_coverage_ratio"])
                    if layer_rule == "hebbian"
                    else None
                ),
                "winner_entropy": (
                    float(health["winner_entropy"])
                    if layer_rule == "hebbian"
                    else None
                ),
                "max_winner_share": (
                    float(health["max_winner_share"])
                    if layer_rule == "hebbian"
                    else None
                ),
                "layerwise_linear_probe_cv_accuracy": linear_accuracy,
                "knn_cv_accuracy": knn_accuracy,
                "between_within_separation_ratio": _separation_ratio(
                    features, labels
                ),
                "applicable_health_checks": ";".join(applicable),
                "applicable_health_pass_count": applicable_pass_count,
                "applicable_health_check_count": len(applicable),
                "applicable_health_pass_fraction": (
                    applicable_pass_count / len(applicable)
                ),
                "applicable_layer_health_pass": applicable_pass_count
                == len(applicable),
                "all_values_finite": bool(
                    torch.isfinite(activations).all().item()
                ),
            }
            representation_rows.append(row)

        probe = _final_probe_metrics(run_dir)
        training = _json(run_dir / "hybrid_training_summary.json")
        method_rows = [
            row for row in representation_rows if row["method_id"] == method_id
        ]
        performance_rows.append(
            {
                "method_id": method_id,
                "display_name": method_config["hybrid"]["display_name"],
                **probe,
                "validation_reconstruction_mse": training[
                    "best_validation_reconstruction_mse"
                ],
                "samples_seen": training["samples_seen"],
                "steps_completed": training["steps_completed"],
                "training_wall_time_sec": training["wall_time_sec"],
                "performance_gate_pass": probe["validation_accuracy"]
                >= float(protocol["performance_floor"]),
                "representation_health_gate_pass": all(
                    row["applicable_layer_health_pass"] for row in method_rows
                ),
                "test_samples_accessed": 0,
            }
        )

    full_hebbian = {
        row["layer"]: row
        for row in representation_rows
        if row["method_id"] == "full_hebbian"
    }
    performance = {row["method_id"]: row for row in performance_rows}

    candidate_records: dict[str, Any] = {}
    for candidate in ("hybrid_hhb", "hybrid_hbb"):
        candidate_layers = {
            row["layer"]: row
            for row in representation_rows
            if row["method_id"] == candidate
        }
        key_improvements = []
        for layer in LAYERS:
            baseline = full_hebbian[layer]
            current = candidate_layers[layer]
            for field, threshold_key in (
                ("effective_rank", "min_effective_rank"),
                (
                    "normalized_effective_rank",
                    "min_normalized_effective_rank",
                ),
            ):
                if (
                    baseline[field] < thresholds[threshold_key]
                    and current[field] >= thresholds[threshold_key]
                ):
                    key_improvements.append(f"{layer}:{field}:fail_to_pass")
        numerical_ok = all(
            row["all_values_finite"] for row in candidate_layers.values()
        )
        eligible = (
            performance[candidate]["performance_gate_pass"]
            and bool(key_improvements)
            and numerical_ok
            and pairing_gate["decision"] == "PASS"
        )
        candidate_records[candidate] = {
            "performance_gate_pass": performance[candidate][
                "performance_gate_pass"
            ],
            "representation_health_gate_pass": performance[candidate][
                "representation_health_gate_pass"
            ],
            "key_representation_improvements": key_improvements,
            "numerical_integrity_pass": numerical_ok,
            "pairing_gate_pass": pairing_gate["decision"] == "PASS",
            "eligible_for_confirmation": eligible,
        }

    eligible = [
        method
        for method in ("hybrid_hhb", "hybrid_hbb")
        if candidate_records[method]["eligible_for_confirmation"]
    ]
    selected_candidate = None
    if "hybrid_hhb" in eligible:
        selected_candidate = "hybrid_hhb"
    elif "hybrid_hbb" in eligible:
        selected_candidate = "hybrid_hbb"

    if selected_candidate is not None and not performance[selected_candidate][
        "representation_health_gate_pass"
    ]:
        outcome = "D"
        conclusion = (
            "Hybrid BP suffix training improves validation performance and at "
            "least one frozen diversity check, while the existing health gate "
            "still fails at earlier Hebbian layers."
        )
    elif selected_candidate == "hybrid_hhb":
        outcome = "A"
        conclusion = (
            "The primary bottleneck is consistent with Hebbian training of Enc3."
        )
    elif selected_candidate == "hybrid_hbb":
        outcome = "B"
        conclusion = (
            "The failure accumulates across Enc2 and Enc3, while Hebbian Enc1 "
            "remains usable."
        )
    else:
        outcome = "C"
        conclusion = (
            "The bottleneck begins at Enc1 or arises from the broader "
            "Hebbian/BP objective mismatch."
        )

    figures = _plots(output_dir, representation_rows, performance_rows)
    _write_csv(output_dir / "representation_metrics.csv", representation_rows)
    _write_csv(output_dir / "performance_metrics.csv", performance_rows)
    decision = {
        "schema_version": "hybrid-depth-decision-v1",
        "completed_at_utc": utc_now(),
        "stage_status": "PASS",
        "selected_outcome": outcome,
        "reliable_conclusion": conclusion,
        "stage3_candidate": selected_candidate,
        "stage3_readiness": (
            "NEEDS CONFIRMATION SEEDS"
            if selected_candidate is not None
            else "NOT ALLOWED"
        ),
        "candidate_records": candidate_records,
        "performance_floor": float(protocol["performance_floor"]),
        "health_thresholds": thresholds,
        "pairing_gate": pairing_gate["decision"],
        "test_samples_accessed": 0,
        "formal_seeds_started": False,
        "confirmation_seeds_started": False,
        "diagnostic_seed": 42,
        "figures": figures,
        "integrity": integrity,
        "source_commit": git_provenance(str(ROOT))["git_commit"],
        "single_next_task": (
            f"Run preregistered confirmation seeds for {selected_candidate}."
            if selected_candidate is not None
            else "Preregister one alternative Hebbian learning rule."
        ),
    }
    write_json(output_dir, "decision.json", decision, overwrite=True)
    write_json(
        output_dir,
        "analysis_manifest.json",
        {
            "schema_version": "hybrid-depth-analysis-manifest-v1",
            "completed_at_utc": utc_now(),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "subset_manifest": protocol["representation"]["subset_manifest"],
            "subset_manifest_sha256": file_sha256(
                _resolve(protocol["representation"]["subset_manifest"])
            ),
            "sample_count": 2000,
            "class_counts": {str(class_id): 200 for class_id in range(10)},
            "cv_protocol": {
                "folds": int(spec["cv_folds"]),
                "seed": int(spec["cv_seed"]),
                "linear_model": "SGDClassifier(log_loss, alpha=1e-4)",
                "knn_neighbors": int(spec["knn_neighbors"]),
                "standardization": "fit within each validation CV training fold",
            },
            "test_samples_accessed": 0,
            **git_provenance(str(ROOT)),
        },
        overwrite=True,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/hybrid_depth_ablation_v1.yaml",
    )
    args = parser.parse_args()
    print(run_analysis(args.config).resolve())


if __name__ == "__main__":
    main()
