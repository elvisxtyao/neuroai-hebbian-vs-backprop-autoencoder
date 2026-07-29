"""Formal paired deterministic-noise evaluation for Stage 3 / Q3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from data.mnist import build_mnist_dataloaders
from evaluation.analyze_stage3_q1 import bootstrap_mean_ci
from evaluation.deterministic_noise import apply_deterministic_noise
from evaluation.metrics import classification_metrics
from models import ConvAutoencoder, LinearProbe, autoencoder_from_config
from schemas import load_config
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import state_dict_checksum


ROOT = Path(__file__).resolve().parents[1]
METRIC_FIELDS = (
    "accuracy",
    "macro_f1",
    "classification_ce",
    "system_reconstruction_mse",
    "standardized_reconstruction_mse",
    "representation_cosine",
    "prediction_js_divergence",
)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def load_components(run_dir: Path, config: dict):
    seed = int(config["training"]["seed"])
    latent_dim = int(config["model"]["latent_dim"])
    system = autoencoder_from_config(config, seed=seed)
    system.load_state_dict(
        torch.load(run_dir / "model_best.pt", map_location="cpu", weights_only=True)
    )
    standardized = autoencoder_from_config(config, seed=seed)
    standardized.load_state_dict(system.state_dict())
    standardized.decoder.load_state_dict(
        torch.load(
            run_dir / "standardized_decoder" / "decoder_best.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    probe = LinearProbe(
        latent_dim,
        epsilon=float(config["probe"]["standardization_epsilon"]),
    )
    probe.load_state_dict(
        torch.load(run_dir / "linear_probe.pt", map_location="cpu", weights_only=True)
    )
    for component in (system, standardized, probe):
        component.eval()
        for parameter in component.parameters():
            parameter.requires_grad_(False)
    return system, standardized, probe


def _cosine_per_sample(clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    clean_values = clean.flatten(start_dim=1)
    noisy_values = noisy.flatten(start_dim=1)
    clean_norm = clean_values.norm(dim=1)
    noisy_norm = noisy_values.norm(dim=1)
    cosine = F.cosine_similarity(clean_values, noisy_values, dim=1, eps=1e-12)
    both_zero = (clean_norm <= 1e-12) & (noisy_norm <= 1e-12)
    one_zero = (clean_norm <= 1e-12) ^ (noisy_norm <= 1e-12)
    cosine[both_zero] = 1.0
    cosine[one_zero] = 0.0
    return cosine


def _js_divergence(clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-12
    midpoint = (clean + noisy) / 2.0
    clean_kl = (
        clean * (clean.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())
    ).sum(dim=1)
    noisy_kl = (
        noisy * (noisy.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())
    ).sum(dim=1)
    return (clean_kl + noisy_kl) / 2.0


@torch.no_grad()
def evaluate_condition(
    system,
    standardized,
    probe,
    loader,
    *,
    noise_type: str,
    severity: float,
    noise_seed: int,
    salt_probability: float,
    clean_reference: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], str]:
    criterion = nn.MSELoss(reduction="sum")
    labels_all = []
    ids_all = []
    probabilities_all = []
    representations_all = []
    system_error = 0.0
    standardized_error = 0.0
    pixel_count = 0
    first_noise_hash = ""

    for batch_index, (clean_images, labels, sample_ids) in enumerate(loader):
        noisy_images = apply_deterministic_noise(
            clean_images,
            sample_ids,
            noise_type=noise_type,
            severity=severity,
            noise_seed=noise_seed,
            salt_probability=salt_probability,
        )
        if batch_index == 0:
            first_noise_hash = tensor_sha256(noisy_images)
        z = system.encode(noisy_images)
        probabilities = torch.softmax(probe(z), dim=1)
        system_reconstruction = system.decode(z)
        standardized_reconstruction = standardized.decode(z)
        system_error += float(criterion(system_reconstruction, clean_images).item())
        standardized_error += float(
            criterion(standardized_reconstruction, clean_images).item()
        )
        pixel_count += clean_images.numel()
        labels_all.append(labels)
        ids_all.append(torch.as_tensor(sample_ids))
        probabilities_all.append(probabilities.cpu())
        representations_all.append(z.cpu())

    labels_tensor = torch.cat(labels_all).long()
    ids_tensor = torch.cat(ids_all).long()
    probabilities_tensor = torch.cat(probabilities_all)
    representations_tensor = torch.cat(representations_all)
    if ids_tensor.unique().numel() != ids_tensor.numel():
        raise RuntimeError("noise evaluation sample IDs are not unique")
    order = torch.argsort(ids_tensor)
    ids_tensor = ids_tensor[order]
    labels_tensor = labels_tensor[order]
    probabilities_tensor = probabilities_tensor[order]
    representations_tensor = representations_tensor[order]
    metrics = classification_metrics(
        labels_tensor.numpy(),
        probabilities_tensor.argmax(dim=1).numpy(),
        probabilities_tensor.numpy(),
    )
    reference = {
        "sample_ids": ids_tensor,
        "probabilities": probabilities_tensor,
        "representations": representations_tensor,
    }
    if clean_reference is None:
        representation_cosine = 1.0
        prediction_js = 0.0
    else:
        if not torch.equal(ids_tensor, clean_reference["sample_ids"]):
            raise RuntimeError("clean/noisy sample IDs differ")
        representation_cosine = float(
            _cosine_per_sample(
                clean_reference["representations"], representations_tensor
            )
            .mean()
            .item()
        )
        prediction_js = float(
            _js_divergence(
                clean_reference["probabilities"], probabilities_tensor
            )
            .mean()
            .item()
        )
    result = {
        **metrics,
        "system_reconstruction_mse": system_error / pixel_count,
        "standardized_reconstruction_mse": standardized_error / pixel_count,
        "representation_cosine": representation_cosine,
        "prediction_js_divergence": prediction_js,
        "num_samples": int(labels_tensor.numel()),
        "num_pixels": int(pixel_count),
        "sample_ids_sha256": tensor_sha256(ids_tensor),
    }
    return result, reference, first_noise_hash


def degradation_rows(records: list[dict]) -> list[dict]:
    clean = {
        (row["seed"], row["method"]): row
        for row in records
        if row["noise_type"] == "clean" and row["severity"] == 0.0
    }
    output = []
    for row in records:
        baseline = clean[(row["seed"], row["method"])]
        output.append(
            {
                **row,
                "accuracy_absolute_degradation": baseline["accuracy"]
                - row["accuracy"],
                "accuracy_relative_degradation": (
                    baseline["accuracy"] - row["accuracy"]
                )
                / max(baseline["accuracy"], 1e-12),
                "macro_f1_absolute_degradation": baseline["macro_f1"]
                - row["macro_f1"],
                "system_reconstruction_mse_increase": row[
                    "system_reconstruction_mse"
                ]
                - baseline["system_reconstruction_mse"],
                "standardized_reconstruction_mse_increase": row[
                    "standardized_reconstruction_mse"
                ]
                - baseline["standardized_reconstruction_mse"],
            }
        )
    return output


def summarize(records: list[dict]) -> tuple[list[dict], list[dict]]:
    fields = METRIC_FIELDS + (
        "accuracy_absolute_degradation",
        "accuracy_relative_degradation",
        "macro_f1_absolute_degradation",
        "system_reconstruction_mse_increase",
        "standardized_reconstruction_mse_increase",
    )
    summary = []
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["noise_type"], row["severity"])].append(row)
    for (method, noise_type, severity), rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "method": method,
            "noise_type": noise_type,
            "severity": severity,
        }
        for field in fields:
            values = np.asarray([row[field] for row in rows], dtype=np.float64)
            ci = bootstrap_mean_ci(values)
            result[f"{field}_mean"] = float(values.mean())
            result[f"{field}_sd"] = float(values.std(ddof=1))
            result[f"{field}_ci_low"] = ci[0]
            result[f"{field}_ci_high"] = ci[1]
        summary.append(result)

    contrasts = []
    pairs = (
        ("HHB_minus_HHH", "HHB", "HHH"),
        ("HBB_minus_HHB", "HBB", "HHB"),
        ("BBB_minus_HHB", "BBB", "HHB"),
    )
    indexed = {
        (row["seed"], row["method"], row["noise_type"], row["severity"]): row
        for row in records
    }
    conditions = sorted(
        {(row["noise_type"], row["severity"]) for row in records}
    )
    for contrast, left, right in pairs:
        for noise_type, severity in conditions:
            for field in fields:
                differences = np.asarray(
                    [
                        indexed[(seed, left, noise_type, severity)][field]
                        - indexed[(seed, right, noise_type, severity)][field]
                        for seed in range(5)
                    ]
                )
                ci = bootstrap_mean_ci(differences)
                contrasts.append(
                    {
                        "contrast": contrast,
                        "left": left,
                        "right": right,
                        "noise_type": noise_type,
                        "severity": severity,
                        "metric": field,
                        "mean_paired_difference": float(differences.mean()),
                        "sd_paired_difference": float(differences.std(ddof=1)),
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                    }
                )
    return summary, contrasts


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output_dir: Path, summary: list[dict]) -> None:
    figures = output_dir / "figures"
    figures.mkdir()
    methods = ("BBB", "HHH", "HHB", "HBB")
    noise_types = ("gaussian", "salt_pepper", "pixel_masking")
    colors = {"BBB": "#2463A3", "HHH": "#D95F02", "HHB": "#1B9E77", "HBB": "#66A61E"}
    indexed = {
        (row["method"], row["noise_type"], float(row["severity"])): row
        for row in summary
    }

    for metric, ylabel, filename in (
        ("accuracy", "Accuracy", "accuracy_severity"),
        (
            "representation_cosine",
            "Clean–noisy z cosine",
            "representation_stability",
        ),
        (
            "prediction_js_divergence",
            "Prediction JS divergence",
            "prediction_js",
        ),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, noise_type in zip(axes, noise_types):
            for method in methods:
                severities = [0.0, 0.1, 0.2, 0.3, 0.4]
                rows = [
                    indexed[
                        (
                            method,
                            "clean" if severity == 0 else noise_type,
                            severity,
                        )
                    ]
                    for severity in severities
                ]
                ax.errorbar(
                    severities,
                    [row[f"{metric}_mean"] for row in rows],
                    yerr=[row[f"{metric}_sd"] for row in rows],
                    marker="o",
                    capsize=3,
                    color=colors[method],
                    label=method,
                )
            ax.set_title(noise_type.replace("_", " "))
            ax.set_xlabel("Severity")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(figures / f"{filename}.png", dpi=160)
        fig.savefig(figures / f"{filename}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for column, noise_type in enumerate(noise_types):
        for method in methods:
            severities = [0.0, 0.1, 0.2, 0.3, 0.4]
            rows = [
                indexed[
                    (
                        method,
                        "clean" if severity == 0 else noise_type,
                        severity,
                    )
                ]
                for severity in severities
            ]
            axes[0, column].plot(
                severities,
                [row["system_reconstruction_mse_increase_mean"] for row in rows],
                marker="o",
                color=colors[method],
                label=method,
            )
            axes[1, column].plot(
                severities,
                [
                    row["standardized_reconstruction_mse_increase_mean"]
                    for row in rows
                ],
                marker="o",
                color=colors[method],
                label=method,
            )
        axes[0, column].set_title(noise_type.replace("_", " "))
        axes[1, column].set_xlabel("Severity")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("System MSE increase")
    axes[1, 0].set_ylabel("Standardized MSE increase")
    axes[0, -1].legend()
    fig.tight_layout()
    fig.savefig(figures / "reconstruction_degradation.png", dpi=160)
    fig.savefig(figures / "reconstruction_degradation.pdf")
    plt.close(fig)


def run(config_path: Path) -> Path:
    protocol_path = resolve(config_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["version"] != "stage3-q3-noise-v1":
        raise ValueError("unsupported Q3 protocol")
    source_root = resolve(protocol["source_results_root"])
    gate = read_json(source_root / "freeze_gate.json")
    test_summary = read_json(source_root / "test_evaluation" / "summary.json")
    if gate["decision"] != "PASS" or not test_summary["records_complete"]:
        raise RuntimeError("Stage 3 core is not formally frozen/evaluated")
    output_dir = resolve(protocol["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"immutable Q3 output exists: {output_dir}")
    output_dir.mkdir(parents=True)

    first_config = load_config(
        source_root / "runs" / "seed_0" / "full_bp" / "config_resolved.yaml"
    )
    loaders = build_mnist_dataloaders(
        first_config, seed=0, include_test=True, download=False
    )
    loader = loaders["test"]
    noise_spec = protocol["noise"]
    records = []
    integrity = {}
    condition_fingerprints: dict[tuple[str, float], str] = {}

    for seed in protocol["seeds"]:
        for method_id, method in protocol["methods"].items():
            run_dir = source_root / "runs" / f"seed_{seed}" / method_id
            config = load_config(run_dir / "config_resolved.yaml")
            system, standardized, probe = load_components(run_dir, config)
            before = {
                "system": state_dict_checksum(system),
                "standardized": state_dict_checksum(standardized),
                "probe": state_dict_checksum(probe),
            }
            clean_metrics, clean_reference, clean_hash = evaluate_condition(
                system,
                standardized,
                probe,
                loader,
                noise_type="gaussian",
                severity=0.0,
                noise_seed=int(noise_spec["seed"]),
                salt_probability=float(noise_spec["salt_probability"]),
                clean_reference=None,
            )
            records.append(
                {
                    "seed": int(seed),
                    "method_id": method_id,
                    "method": method,
                    "noise_type": "clean",
                    "severity": 0.0,
                    **clean_metrics,
                }
            )
            condition_fingerprints.setdefault(("clean", 0.0), clean_hash)
            if condition_fingerprints[("clean", 0.0)] != clean_hash:
                raise RuntimeError("clean first-batch tensor differs across methods")

            for noise_type in noise_spec["types"]:
                for severity in noise_spec["severities"][1:]:
                    metrics, _, noise_hash = evaluate_condition(
                        system,
                        standardized,
                        probe,
                        loader,
                        noise_type=noise_type,
                        severity=float(severity),
                        noise_seed=int(noise_spec["seed"]),
                        salt_probability=float(noise_spec["salt_probability"]),
                        clean_reference=clean_reference,
                    )
                    key = (noise_type, float(severity))
                    condition_fingerprints.setdefault(key, noise_hash)
                    if condition_fingerprints[key] != noise_hash:
                        raise RuntimeError(
                            f"noisy tensor differs across methods: {key}"
                        )
                    records.append(
                        {
                            "seed": int(seed),
                            "method_id": method_id,
                            "method": method,
                            "noise_type": noise_type,
                            "severity": float(severity),
                            **metrics,
                        }
                    )
            after = {
                "system": state_dict_checksum(system),
                "standardized": state_dict_checksum(standardized),
                "probe": state_dict_checksum(probe),
            }
            integrity[f"seed_{seed}/{method}"] = {
                "before": before,
                "after": after,
                "unchanged": before == after,
                "system_checkpoint_sha256": file_sha256(
                    run_dir / "model_best.pt"
                ),
                "standardized_decoder_sha256": file_sha256(
                    run_dir / "standardized_decoder" / "decoder_best.pt"
                ),
                "probe_sha256": file_sha256(run_dir / "linear_probe.pt"),
            }

    degraded = degradation_rows(records)
    summary, contrasts = summarize(degraded)
    write_csv(output_dir / "per_seed_condition_metrics.csv", degraded)
    write_csv(output_dir / "condition_summary.csv", summary)
    write_csv(output_dir / "paired_degradation_contrasts.csv", contrasts)
    plot_results(output_dir, summary)
    write_json(
        output_dir / "noise_fingerprints.json",
        {
            f"{key[0]}:{key[1]:.1f}": value
            for key, value in sorted(condition_fingerprints.items())
        },
    )
    write_json(
        output_dir / "integrity.json",
        {
            "schema_version": "stage3-q3-integrity-v1",
            "completed_at_utc": utc_now(),
            "records": integrity,
            "all_components_unchanged": all(
                value["unchanged"] for value in integrity.values()
            ),
            "checkpoint_count": len(integrity),
            "expected_checkpoint_count": 20,
            "condition_count_per_checkpoint": 13,
            "metric_row_count": len(records),
            "expected_metric_row_count": 260,
            "all_metrics_finite": all(
                np.isfinite(float(row[field]))
                for row in records
                for field in METRIC_FIELDS
            ),
            "same_noise_tensor_across_methods": True,
            "test_samples_per_condition": 10000,
            "test_samples_total": 10000 * len(records),
            "test_used_for_selection": False,
            "training_performed": False,
        },
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "stage3-q3-run-manifest-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "source_freeze_gate": "PASS",
            "source_test_records_complete": True,
            "methods": protocol["methods"],
            "seeds": protocol["seeds"],
            "noise": noise_spec,
            "records": len(records),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/stage3_q3_noise_v1.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args.config).resolve())


if __name__ == "__main__":
    main()
