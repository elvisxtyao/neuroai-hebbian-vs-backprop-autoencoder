"""Run the no-training, validation-only Stage 1C effective-rank audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.effective_rank_audit import (
    apply_topk_wta,
    class_center,
    class_rank_metrics,
    dataset_center,
    epsilon_sensitivity,
    l2_normalize_samples,
    representation_matrix,
    spectrum_metrics,
)
from evaluation.metrics import classification_metrics
from evaluation.representations import extract_representations
from models import ConvAutoencoder, LinearProbe
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_is_ancestor(ancestor: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _load_subset(
    config: dict[str, Any],
) -> tuple[Path, np.ndarray, np.ndarray, DataLoader]:
    subset_path = _resolve(config["subset_manifest"])
    subset = np.load(subset_path, allow_pickle=False)
    required = {"sample_ids", "labels", "source_split_sha256"}
    if not required.issubset(subset.files):
        raise RuntimeError("Stage 1 subset manifest is incomplete")
    sample_ids = np.asarray(subset["sample_ids"], dtype=np.int64)
    labels = np.asarray(subset["labels"], dtype=np.int64)
    if sample_ids.size != 2000 or np.unique(sample_ids).size != 2000:
        raise RuntimeError("Stage 1C requires exactly 2,000 unique sample IDs")
    class_counts = np.bincount(labels, minlength=10)
    if not np.array_equal(class_counts, np.full(10, 200)):
        raise RuntimeError("Stage 1C subset must contain exactly 200 images per class")
    training = datasets.MNIST(
        root=str(_resolve(config["data_root"])),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    dataset_labels = np.asarray(training.targets, dtype=np.int64)
    if not np.array_equal(labels, dataset_labels[sample_ids]):
        raise RuntimeError("Subset labels do not match the MNIST training partition")
    loader = DataLoader(
        IndexedDataset(training, sample_ids),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return subset_path, sample_ids, labels, loader


def _source_probe_metric(run_dir: Path) -> dict[str, float | int]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["stage"] == "linear_probe_final"
            and row["split"] == "validation"
        ]
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one frozen validation probe metric")
    row = rows[0]
    return {
        "accuracy": float(row["accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "classification_ce": float(row["classification_ce"]),
        "selected_epoch": int(row["epoch"]),
        "num_samples": int(float(row["num_samples"])),
    }


@torch.no_grad()
def _evaluate_frozen_probe(
    run_dir: Path,
    run_config: dict[str, Any],
    z_features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, float], str, str]:
    probe_path = run_dir / "linear_probe.pt"
    probe = LinearProbe(
        int(run_config["model"]["latent_dim"]),
        epsilon=float(run_config["probe"]["standardization_epsilon"]),
    )
    probe.load_state_dict(
        torch.load(probe_path, map_location="cpu", weights_only=True)
    )
    checksum_before = state_dict_checksum(probe)
    probe.eval()
    logits = probe(z_features.to(torch.float32))
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    metrics = classification_metrics(
        labels_np,
        probabilities.argmax(axis=1),
        probabilities,
    )
    checksum_after = state_dict_checksum(probe)
    return metrics, checksum_before, checksum_after


def run_audit(config_path: str | Path) -> Path:
    config_path = _resolve(config_path)
    config = _load_yaml(config_path)
    if config.get("version") != "effective-rank-audit-v1.1":
        raise ValueError("Unsupported Stage 1C config version")
    provenance = git_provenance(str(ROOT))
    if provenance["git_worktree_dirty"]:
        raise RuntimeError("Formal Stage 1C requires a clean Git worktree")
    if not _git_is_ancestor(config["protocol_base_ref"]):
        raise RuntimeError("Phase 0 canonical ref is not an ancestor of HEAD")

    output_dir = _resolve(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Stage 1C output already exists: {output_dir}")
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(f"Stage 1C temporary output exists: {temporary_dir}")
    temporary_dir.mkdir(parents=True, exist_ok=False)

    subset_path, sample_ids, labels_np, loader = _load_subset(config)
    labels = torch.as_tensor(labels_np, dtype=torch.long)
    run_dir = _resolve(config["checkpoint"]["run_dir"])
    checkpoint_path = run_dir / config["checkpoint"]["checkpoint"]
    probe_path = run_dir / config["checkpoint"]["probe"]
    run_config = _load_yaml(run_dir / "config_resolved.yaml")
    model = ConvAutoencoder(
        int(run_config["model"]["latent_dim"]),
        seed=int(config["checkpoint"]["seed"]),
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    model_hash_before = state_dict_checksum(model)
    checkpoint_file_hash_before = file_sha256(checkpoint_path)
    probe_file_hash_before = file_sha256(probe_path)
    extracted = extract_representations(
        model,
        loader,
        device=torch.device("cpu"),
        layers=("h1", "h2", "z"),
    )
    if not torch.equal(
        extracted["sample_id"],
        torch.as_tensor(sample_ids, dtype=extracted["sample_id"].dtype),
    ):
        raise RuntimeError("Extracted sample IDs differ from Stage 1 manifest order")
    if not torch.equal(extracted["label"], labels):
        raise RuntimeError("Extracted labels differ from Stage 1 manifest")

    matrices: list[tuple[str, str, str, torch.Tensor, torch.Tensor, dict[str, Any]]] = []
    axis_records: list[dict[str, Any]] = []
    for layer in ("h1", "h2"):
        for view in ("channel_health", "sample_flat"):
            matrix, axis = representation_matrix(extracted[layer], view=view)
            observation_labels = (
                labels.repeat_interleave(axis["spatial_locations_per_sample"])
                if view == "channel_health"
                else labels
            )
            representation_id = f"{layer}__{view}"
            axis_records.append({"representation_id": representation_id, **axis})
            matrices.append(
                (
                    representation_id,
                    layer,
                    "standard_forward_pre_wta",
                    matrix,
                    observation_labels,
                    axis,
                )
            )

    z_pre, z_axis = representation_matrix(extracted["z"], view="sample_flat")
    z_channel, z_channel_axis = representation_matrix(
        extracted["z"], view="channel_health"
    )
    if not torch.equal(z_pre, z_channel):
        raise RuntimeError("z channel-health and sample-flat views must be identical")
    z_post, winner_mask = apply_topk_wta(
        z_pre,
        winner_fraction=float(config["winner_fraction"]),
    )
    z_variants = {
        "z_pre_wta": ("pre_wta", z_pre),
        "z_post_wta": ("post_wta", z_post),
        "z_post_wta_centered": (
            "post_wta_dataset_feature_centered",
            dataset_center(z_post),
        ),
        "z_post_wta_l2": (
            "post_wta_per_sample_l2_normalized",
            l2_normalize_samples(
                z_post, epsilon=float(config["l2_epsilon"])
            ),
        ),
        "z_post_wta_class_centered": (
            "post_wta_class_centered",
            class_center(z_post, labels),
        ),
    }
    for representation_id, (transform, matrix) in z_variants.items():
        axis = {
            **z_axis,
            "wta_applied": representation_id != "z_pre_wta",
            "dataset_centered_input": representation_id
            == "z_post_wta_centered",
            "per_sample_l2_normalized": representation_id == "z_post_wta_l2",
            "class_centered_input": representation_id
            == "z_post_wta_class_centered",
        }
        axis_records.append({"representation_id": representation_id, **axis})
        matrices.append(
            (representation_id, "z", transform, matrix, labels, axis)
        )
    axis_records.append(
        {
            "representation_id": "z_view_equivalence_check",
            "channel_health_shape": list(z_channel.shape),
            "sample_flat_shape": list(z_pre.shape),
            "bitwise_equal": True,
            "channel_health_metadata": z_channel_axis,
            "sample_flat_metadata": z_axis,
        }
    )

    rank_rows: list[dict[str, Any]] = []
    eigen_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    class_covariance_rows: list[dict[str, Any]] = []
    epsilon_rows: list[dict[str, Any]] = []
    epsilon_checks: dict[str, bool] = {}
    epsilon_dominance: dict[str, bool] = {}
    spectra: dict[str, Any] = {}
    for representation_id, layer, transform, matrix, matrix_labels, axis in matrices:
        print(
            f"Stage1C spectrum {representation_id}: "
            f"{matrix.shape[0]}x{matrix.shape[1]}",
            flush=True,
        )
        result = spectrum_metrics(matrix, center=True)
        rank_rows.append(
            {
                "representation_id": representation_id,
                "layer": layer,
                "transform": transform,
                "view": axis["view"],
                "observation_axis": axis["observation_axis"],
                "feature_axis": axis["feature_axis"],
                "observations": result.observation_count,
                "features": result.feature_count,
                "covariance_rows": result.conceptual_covariance_shape[0],
                "covariance_columns": result.conceptual_covariance_shape[1],
                "covariance_axis": "feature",
                "spectrum_backend": result.backend,
                "dataset_centered_for_covariance": result.dataset_centered,
                "participation_ratio": result.participation_ratio,
                "stable_rank": result.stable_rank,
                "rank_ratio": result.rank_ratio,
                "max_rank_ratio": result.max_rank_ratio,
                "numerical_rank": result.numerical_rank,
                "numerical_rank_tolerance": result.numerical_rank_tolerance,
                "covariance_trace": result.trace,
                "covariance_squared_trace": result.squared_trace,
                "stage1_epsilon_dominates": (
                    result.trace <= float(config["stage1_variance_epsilon"])
                    or result.squared_trace
                    <= float(config["stage1_variance_epsilon"])
                ),
                "rank_defined_at_stage1_epsilon": (
                    result.trace > float(config["stage1_variance_epsilon"])
                    and result.squared_trace
                    > float(config["stage1_variance_epsilon"])
                ),
            }
        )
        for index, value in enumerate(result.eigenvalues.tolist(), start=1):
            eigen_rows.append(
                {
                    "representation_id": representation_id,
                    "descending_index": index,
                    "eigenvalue": value,
                }
            )
        for index, value in enumerate(result.singular_values.tolist(), start=1):
            singular_rows.append(
                {
                    "representation_id": representation_id,
                    "descending_index": index,
                    "singular_value": value,
                    "dataset_centered": True,
                }
            )
        class_rows, class_summary = class_rank_metrics(matrix, matrix_labels)
        for row in class_rows:
            per_class_rows.append({"representation_id": representation_id, **row})
        class_covariance_rows.append(
            {"representation_id": representation_id, **class_summary}
        )
        if representation_id.startswith("z_"):
            sensitivity, stable = epsilon_sensitivity(
                result.eigenvalues,
                relative_cutoffs=[
                    float(value) for value in config["relative_eigenvalue_cutoffs"]
                ],
                formula_epsilons=[
                    float(value) for value in config["stage1_formula_epsilons"]
                ],
            )
            epsilon_checks[representation_id] = stable
            epsilon_dominance[representation_id] = any(
                bool(row["epsilon_dominates"]) for row in sensitivity
            )
            for row in sensitivity:
                epsilon_rows.append({"representation_id": representation_id, **row})
        spectra[representation_id] = result

    probe_metrics, probe_hash_before, probe_hash_after = _evaluate_frozen_probe(
        run_dir,
        run_config,
        z_pre,
        labels,
    )
    source_probe_metrics = _source_probe_metric(run_dir)
    model_hash_after = state_dict_checksum(model)
    checkpoint_file_hash_after = file_sha256(checkpoint_path)
    probe_file_hash_after = file_sha256(probe_path)
    z_pre_result = spectra["z_pre_wta"]
    z_post_result = spectra["z_post_wta"]
    near_one_max = float(config["interpretation"]["near_one_effective_rank_max"])
    high_rank_ratio = float(config["interpretation"]["high_rank_ratio_min"])
    if (
        z_pre_result.rank_ratio >= high_rank_ratio
        and z_post_result.rank_ratio < high_rank_ratio
    ):
        mechanism = "WTA_RANK_COMPRESSION"
        interpretation = (
            "Pre-WTA rank is high but post-WTA rank is low; WTA is the primary "
            "observed rank-compression step."
        )
    elif (
        z_pre_result.participation_ratio <= near_one_max
        and z_post_result.participation_ratio <= near_one_max
    ):
        mechanism = "PRE_AND_POST_WTA_NEAR_ONE"
        interpretation = (
            "Both pre-WTA and post-WTA ranks are near one; low rank is already "
            "present before WTA and supports highly redundant filters/features."
        )
    else:
        mechanism = "MIXED_OR_INCONCLUSIVE"
        interpretation = (
            "Pre/post-WTA ranks do not match either preregistered mechanism pattern."
        )

    qa = {
        "sample_axis_verified": all(
            record.get("observation_axis")
            in {"sample", "sample_x_spatial_location", None}
            for record in axis_records
        ),
        "feature_axis_verified": all(
            record.get("feature_axis")
            in {"channel", "channel_x_height_x_width", None}
            for record in axis_records
        ),
        "convolutional_views_separated": {
            "h1__channel_health",
            "h1__sample_flat",
            "h2__channel_health",
            "h2__sample_flat",
        }.issubset({row["representation_id"] for row in rank_rows}),
        "covariance_is_feature_by_feature": all(
            row["covariance_rows"] == row["features"]
            and row["covariance_columns"] == row["features"]
            and row["covariance_axis"] == "feature"
            for row in rank_rows
        ),
        "dataset_level_centering_explicit": all(
            row["dataset_centered_for_covariance"] for row in rank_rows
        ),
        "epsilon_audit_complete": set(epsilon_checks) == set(z_variants),
        "primary_pre_post_epsilon_not_dominant": (
            epsilon_checks["z_pre_wta"]
            and epsilon_checks["z_post_wta"]
            and not epsilon_dominance["z_pre_wta"]
            and not epsilon_dominance["z_post_wta"]
        ),
        "l2_zero_variance_is_explicit": (
            spectra["z_post_wta_l2"].trace
            <= float(config["stage1_variance_epsilon"])
            or spectra["z_post_wta_l2"].squared_trace
            <= float(config["stage1_variance_epsilon"])
        ),
        "subset_has_2000_unique_ids": sample_ids.size
        == 2000
        == np.unique(sample_ids).size,
        "subset_is_class_balanced": bool(
            np.array_equal(np.bincount(labels_np, minlength=10), np.full(10, 200))
        ),
        "z_views_bitwise_equal_before_wta": torch.equal(z_pre, z_channel),
        "checkpoint_state_unchanged": model_hash_before == model_hash_after,
        "checkpoint_file_unchanged": checkpoint_file_hash_before
        == checkpoint_file_hash_after,
        "probe_state_unchanged": probe_hash_before == probe_hash_after,
        "probe_file_unchanged": probe_file_hash_before == probe_file_hash_after,
        "all_values_finite": all(
            torch.isfinite(matrix).all().item()
            for _, _, _, matrix, _, _ in matrices
        ),
        "test_samples_accessed_is_zero": True,
    }
    metric_validity = all(qa.values())
    decision = {
        "schema_version": "effective-rank-audit-decision-v1",
        "completed_at_utc": utc_now(),
        "stage": "Stage 1C",
        "status": "COMPLETED" if metric_validity else "RUN_BUT_NOT_VALIDATED",
        "metric_validity": "PASS" if metric_validity else "FAIL",
        "mechanism_classification": mechanism,
        "interpretation": interpretation,
        "stage1_metric_location": (
            "The Stage 1 effective rank was computed on standard ReLU forward "
            "activations before any analysis-only WTA mask."
        ),
        "z_pre_wta": {
            "participation_ratio": z_pre_result.participation_ratio,
            "stable_rank": z_pre_result.stable_rank,
            "rank_ratio": z_pre_result.rank_ratio,
        },
        "z_post_wta": {
            "participation_ratio": z_post_result.participation_ratio,
            "stable_rank": z_post_result.stable_rank,
            "rank_ratio": z_post_result.rank_ratio,
            "winner_count_per_sample": int(winner_mask.sum(dim=1)[0].item()),
        },
        "epsilon_interpretation": {
            "primary_pre_wta_dominated": epsilon_dominance["z_pre_wta"],
            "primary_post_wta_dominated": epsilon_dominance["z_post_wta"],
            "l2_variant_dominated": epsilon_dominance["z_post_wta_l2"],
            "l2_centered_covariance_trace": spectra["z_post_wta_l2"].trace,
            "note": (
                "The L2-normalized post-WTA vectors are effectively identical. "
                "After dataset centering their variance is numerical zero, so "
                "their effective rank is undefined at the Stage 1 epsilon. "
                "This does not control the pre/post-WTA mechanism decision."
            ),
        },
        "qa": qa,
        "linear_probe": {
            "audit_subset": probe_metrics,
            "source_full_validation": source_probe_metrics,
        },
        "test_samples_accessed": 0,
        "training_performed": False,
        "hyperparameter_selection_performed": False,
    }
    run_manifest = {
        "schema_version": "effective-rank-audit-run-v1",
        "completed_at_utc": utc_now(),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": file_sha256(config_path),
        "protocol_base_ref": config["protocol_base_ref"],
        **provenance,
        "dataset_access": {
            "source_dataset": "MNIST official training partition",
            "logical_split": "validation",
            "test_samples_accessed": 0,
        },
        "subset_manifest": str(subset_path.relative_to(ROOT)),
        "subset_manifest_sha256": file_sha256(subset_path),
        "sample_ids_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "sample_count": int(sample_ids.size),
        "class_counts": {
            str(class_id): int((labels_np == class_id).sum())
            for class_id in range(10)
        },
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256_before": checkpoint_file_hash_before,
        "checkpoint_sha256_after": checkpoint_file_hash_after,
        "state_dict_sha256_before": model_hash_before,
        "state_dict_sha256_after": model_hash_after,
        "probe": str(probe_path.relative_to(ROOT)),
        "probe_sha256_before": probe_file_hash_before,
        "probe_sha256_after": probe_file_hash_after,
        "probe_state_dict_sha256_before": probe_hash_before,
        "probe_state_dict_sha256_after": probe_hash_after,
        "outputs": [
            "axis_audit.json",
            "rank_metrics.csv",
            "covariance_eigenvalues.csv",
            "singular_values.csv",
            "per_class_effective_rank.csv",
            "class_covariance_rank.csv",
            "epsilon_sensitivity.csv",
            "linear_probe_metrics.json",
            "audit_decision.json",
            "run_manifest.json",
        ],
    }
    _atomic_json(temporary_dir / "axis_audit.json", {"records": axis_records})
    _write_csv(temporary_dir / "rank_metrics.csv", rank_rows)
    _write_csv(temporary_dir / "covariance_eigenvalues.csv", eigen_rows)
    _write_csv(temporary_dir / "singular_values.csv", singular_rows)
    _write_csv(temporary_dir / "per_class_effective_rank.csv", per_class_rows)
    _write_csv(temporary_dir / "class_covariance_rank.csv", class_covariance_rows)
    _write_csv(temporary_dir / "epsilon_sensitivity.csv", epsilon_rows)
    _atomic_json(
        temporary_dir / "linear_probe_metrics.json",
        {
            "audit_subset": probe_metrics,
            "source_full_validation": source_probe_metrics,
            "probe_state_dict_sha256_before": probe_hash_before,
            "probe_state_dict_sha256_after": probe_hash_after,
        },
    )
    _atomic_json(temporary_dir / "audit_decision.json", decision)
    _atomic_json(temporary_dir / "run_manifest.json", run_manifest)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.replace(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/effective_rank_audit_v1_1.yaml",
    )
    args = parser.parse_args()
    output = run_audit(args.config)
    print(output.resolve())


if __name__ == "__main__":
    main()
