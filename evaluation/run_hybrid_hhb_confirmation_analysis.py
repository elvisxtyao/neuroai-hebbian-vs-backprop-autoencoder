"""Analyze Stage 2D HHB confirmation runs and apply the frozen gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.representation_health import compute_layer_health
from evaluation.representations import extract_representations
from models import ConvAutoencoder
from schemas import load_config
from training.run_hybrid_hhb_confirmation import (
    EXPECTED_METHODS,
    EXPECTED_SEEDS,
    validate_confirmation_protocol,
)
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance, state_dict_checksum
from utils.results import write_json


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("h1", "h2", "z")


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
        raise RuntimeError("Stage 2D subset must contain 200 validation images/class")
    training = datasets.MNIST(
        root=str(_resolve(spec["data_root"])),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    return (
        DataLoader(
            IndexedDataset(training, sample_ids),
            batch_size=int(spec["batch_size"]),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        ),
        labels,
    )


def _final_probe_metrics(run_dir: Path) -> dict[str, float]:
    with (run_dir / "metrics.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["stage"] == "linear_probe_final"
            and row["split"] == "validation"
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one validation probe row in {run_dir}")
    return {
        "validation_accuracy": float(rows[0]["accuracy"]),
        "validation_macro_f1": float(rows[0]["macro_f1"]),
        "validation_classification_ce": float(rows[0]["classification_ce"]),
    }


def evaluate_seed_confirmation(
    *,
    seed: int,
    performance: dict[str, dict[str, float]],
    ranks: dict[str, dict[str, float]],
    thresholds: dict[str, float],
    pairing_pass: bool,
    numerical_integrity_pass: bool,
    zero_test_access: bool,
) -> dict[str, Any]:
    """Apply the preregistered per-seed confirmation gate."""

    epsilon = float(thresholds["epsilon"])
    hhb = performance["hybrid_hhb"]
    bp = performance["full_bp"]
    hhb_ranks = ranks["hybrid_hhb"]
    hebbian_ranks = ranks["full_hebbian"]
    standardized_ratio = (
        hhb["standardized_decoder_validation_mse"]
        / bp["standardized_decoder_validation_mse"]
    )
    z_to_hebbian = hhb_ranks["z"] / (hebbian_ranks["z"] + epsilon)
    z_to_h2 = hhb_ranks["z"] / (hhb_ranks["h2"] + epsilon)
    checks = {
        "validation_accuracy_floor": hhb["validation_accuracy"]
        >= float(thresholds["validation_accuracy_floor"]),
        "standardized_decoder_near_bp": standardized_ratio
        <= float(thresholds["standardized_decoder_mse_ratio_to_bp_max"]),
        "z_effective_rank_absolute": hhb_ranks["z"]
        >= float(thresholds["z_effective_rank_min"]),
        "z_effective_rank_vs_full_hebbian": z_to_hebbian
        >= float(thresholds["z_effective_rank_ratio_to_full_hebbian_min"]),
        "z_compensates_low_rank_h2": z_to_h2
        >= float(thresholds["z_to_h2_effective_rank_ratio_min"]),
        "pairing_integrity": pairing_pass,
        "numerical_integrity": numerical_integrity_pass,
        "zero_test_access": zero_test_access,
    }
    return {
        "seed": seed,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {
            "validation_accuracy": hhb["validation_accuracy"],
            "system_reconstruction_mse": hhb[
                "system_validation_reconstruction_mse"
            ],
            "standardized_decoder_reconstruction_mse": hhb[
                "standardized_decoder_validation_mse"
            ],
            "paired_bp_standardized_decoder_reconstruction_mse": bp[
                "standardized_decoder_validation_mse"
            ],
            "standardized_decoder_mse_ratio_to_bp": standardized_ratio,
            "h2_effective_rank": hhb_ranks["h2"],
            "z_effective_rank": hhb_ranks["z"],
            "full_hebbian_z_effective_rank": hebbian_ranks["z"],
            "z_effective_rank_ratio_to_full_hebbian": z_to_hebbian,
            "z_to_h2_effective_rank_ratio": z_to_h2,
        },
    }


def run_analysis(protocol_path: str | Path) -> Path:
    protocol_path = _resolve(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_confirmation_protocol(protocol)
    output_dir = _resolve(protocol["output_dir"])
    pairing_gate = _json(output_dir / "pairing_gate.json")
    if pairing_gate["decision"] != "PASS":
        raise RuntimeError("Pairing gate must pass before Stage 2D analysis")
    loader, expected_labels = _validation_loader(protocol)
    representation_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    spec = protocol["representation"]

    for seed in EXPECTED_SEEDS:
        for method in EXPECTED_METHODS:
            config = load_config(_resolve(protocol["configs"][str(seed)][method]))
            run_dir = output_dir / "runs" / f"seed_{seed}" / method
            model = ConvAutoencoder(config["model"]["latent_dim"], seed=seed)
            model.load_state_dict(
                torch.load(
                    run_dir / "model_best.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            checksum_before = state_dict_checksum(model)
            extracted = extract_representations(
                model, loader, device=torch.device("cpu"), layers=LAYERS
            )
            checksum_after = state_dict_checksum(model)
            labels = extracted["label"].numpy()
            if not np.array_equal(labels, expected_labels):
                raise RuntimeError("Stage 2D representation subset order changed")
            if checksum_before != checksum_after:
                raise RuntimeError("Stage 2D representation extraction mutated model")
            integrity[f"seed_{seed}/{method}"] = {
                "checkpoint_sha256": file_sha256(run_dir / "model_best.pt"),
                "state_dict_sha256_before": checksum_before,
                "state_dict_sha256_after": checksum_after,
                "unchanged": True,
            }

            for layer in LAYERS:
                health, _, _ = compute_layer_health(
                    extracted[layer],
                    winner_fraction=float(spec["winner_fraction"]),
                    activation_epsilon=float(spec["activation_epsilon"]),
                    variance_epsilon=float(spec["variance_epsilon"]),
                )
                representation_rows.append(
                    {
                        "seed": seed,
                        "method_id": method,
                        "layer": layer,
                        "layer_rule": config["hybrid"]["encoder_layer_rules"][
                            {"h1": "enc1", "h2": "enc2", "z": "enc3"}[layer]
                        ],
                        "effective_rank": float(health["effective_rank"]),
                        "normalized_effective_rank": float(
                            health["normalized_effective_rank"]
                        ),
                        "active_unit_ratio": float(health["active_unit_ratio"]),
                        "nonzero_variance_ratio": float(
                            health["nonzero_variance_ratio"]
                        ),
                        "winner_coverage_ratio": float(
                            health["winner_coverage_ratio"]
                        ),
                        "winner_entropy": float(health["winner_entropy"]),
                        "max_winner_share": float(health["max_winner_share"]),
                        "all_values_finite": bool(
                            torch.isfinite(extracted[layer]).all().item()
                        ),
                    }
                )

            probe = _final_probe_metrics(run_dir)
            system = _json(run_dir / "hybrid_training_summary.json")
            standard = _json(
                run_dir
                / "standardized_decoder"
                / "standardized_decoder_summary.json"
            )
            performance_rows.append(
                {
                    "seed": seed,
                    "method_id": method,
                    **probe,
                    "system_validation_reconstruction_mse": float(
                        system["best_validation_reconstruction_mse"]
                    ),
                    "standardized_decoder_validation_mse": float(
                        standard["best_validation_reconstruction_mse"]
                    ),
                    "system_samples_seen": int(system["samples_seen"]),
                    "standardized_decoder_samples_seen": int(
                        standard["samples_seen"]
                    ),
                    "system_wall_time_sec": float(system["wall_time_sec"]),
                    "standardized_decoder_wall_time_sec": float(
                        standard["wall_time_sec"]
                    ),
                    "test_samples_accessed": 0,
                }
            )

    seed_decisions: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        performance = {
            row["method_id"]: row
            for row in performance_rows
            if row["seed"] == seed
        }
        ranks = {
            method: {
                row["layer"]: row["effective_rank"]
                for row in representation_rows
                if row["seed"] == seed and row["method_id"] == method
            }
            for method in EXPECTED_METHODS
        }
        numerical_ok = all(
            row["all_values_finite"]
            and math.isfinite(float(row["effective_rank"]))
            for row in representation_rows
            if row["seed"] == seed
        ) and all(
            all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key
                in {
                    "validation_accuracy",
                    "validation_macro_f1",
                    "validation_classification_ce",
                    "system_validation_reconstruction_mse",
                    "standardized_decoder_validation_mse",
                }
            )
            for row in performance_rows
            if row["seed"] == seed
        )
        seed_decisions[str(seed)] = evaluate_seed_confirmation(
            seed=seed,
            performance=performance,
            ranks=ranks,
            thresholds=protocol["thresholds"],
            pairing_pass=pairing_gate["per_seed"][str(seed)]["decision"]
            == "PASS",
            numerical_integrity_pass=numerical_ok,
            zero_test_access=True,
        )

    confirmed = all(
        seed_decisions[str(seed)]["decision"] == "PASS"
        for seed in EXPECTED_SEEDS
    )
    decision = {
        "schema_version": "hybrid-hhb-confirmation-decision-v1",
        "completed_at_utc": utc_now(),
        "stage": "Stage 2D",
        "decision": "PASS" if confirmed else "FAIL",
        "candidate": "hybrid_hhb",
        "confirmation_seeds": list(EXPECTED_SEEDS),
        "per_seed": seed_decisions,
        "pairing_gate": pairing_gate["decision"],
        "stage3_readiness": (
            "CONFIRMED FOR FORMAL STAGE 3"
            if confirmed
            else "BLOCKED — CONFIRMATION FAILED"
        ),
        "no_third_confirmation_seed": True,
        "test_samples_accessed": 0,
        "thresholds": protocol["thresholds"],
        "integrity": integrity,
        "source_commit": git_provenance(str(ROOT))["git_commit"],
        "reliable_conclusion": (
            "Across both preregistered confirmation seeds, BP Enc3 consistently "
            "converted a low-rank Hebbian h2 representation into a higher-rank "
            "z while meeting the frozen validation and standardized-decoder gates."
            if confirmed
            else "Hybrid-HHB did not reproduce every preregistered gate across "
            "both confirmation seeds."
        ),
    }
    _write_csv(output_dir / "performance_metrics.csv", performance_rows)
    _write_csv(output_dir / "representation_metrics.csv", representation_rows)
    write_json(output_dir, "confirmation_decision.json", decision, overwrite=True)
    write_json(
        output_dir,
        "analysis_manifest.json",
        {
            "schema_version": "hybrid-hhb-confirmation-analysis-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "subset_manifest": protocol["representation"]["subset_manifest"],
            "subset_manifest_sha256": file_sha256(
                _resolve(protocol["representation"]["subset_manifest"])
            ),
            "sample_count": 2000,
            "class_counts": {str(class_id): 200 for class_id in range(10)},
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
        default="configs/experiments/hybrid_hhb_confirmation_v1.yaml",
    )
    args = parser.parse_args()
    print(run_analysis(args.config).resolve())


if __name__ == "__main__":
    main()
