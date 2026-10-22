"""Layerwise representation analysis for one frozen Q5/Q6 sweep case."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data.mnist import IndexedDataset
from evaluation.run_stage3_q2_representation import (
    compensation_metrics,
    layer_rule,
    linear_cka,
    representation_metrics,
    sha256_array,
    summarize_rows,
    validate_subset,
    write_csv,
)
from models import autoencoder_from_config
from schemas import load_config
from training.run_stage3_q5q6_sweeps import METHODS, ROOT, SEEDS, SWEEPS, validate_protocol
from evaluation.representations import extract_representations
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import state_dict_checksum
from utils.results import write_json


LABELS = {
    "full_bp": "BBB",
    "full_hebbian": "HHH",
    "hybrid_hhb": "HHB",
    "hybrid_hbb": "HBB",
}
LAYERS = ("h1", "h2", "z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_frozen_case(case_dir: Path) -> dict[str, Any]:
    gate = _read_json(case_dir / "freeze_gate.json")
    if gate.get("decision") != "PASS":
        raise RuntimeError("Sweep case freeze gate did not pass")
    test_summary = _read_json(case_dir / "test_evaluation" / "summary.json")
    if not test_summary.get("records_complete"):
        raise RuntimeError("One-time frozen test evaluation is incomplete")
    return gate


def run_case(
    protocol_path: str | Path,
    *,
    sweep: str,
    case: str,
) -> Path:
    protocol_path = Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if sweep not in SWEEPS:
        raise ValueError("Unknown sweep")
    case_spec = protocol[f"{sweep}_cases"].get(case)
    if case_spec is None or case_spec["source"] != "new_formal_run":
        raise ValueError("Representation runner requires a new formal case")
    output_root = Path(protocol["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    case_dir = output_root / sweep / case
    gate = _require_frozen_case(case_dir)
    output_dir = case_dir / "representation"
    if output_dir.exists():
        raise FileExistsError("Immutable case representation output exists")
    output_dir.mkdir(parents=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir()

    subset_path = Path(protocol["representation"]["subset_manifest"])
    if not subset_path.is_absolute():
        subset_path = ROOT / subset_path
    with np.load(subset_path) as subset:
        sample_ids = subset["sample_ids"].astype(np.int64, copy=False)
        expected_labels = subset["labels"].astype(np.int64, copy=False)
    validate_subset(
        sample_ids,
        expected_labels,
        samples_per_class=int(protocol["representation"]["samples_per_class"]),
    )
    test_dataset = datasets.MNIST(
        root=str(ROOT / "data" / "raw"),
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        IndexedDataset(test_dataset, sample_ids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    rows: list[dict[str, Any]] = []
    embeddings: dict[tuple[int, str, str], np.ndarray] = {}
    integrity: dict[str, Any] = {}
    confusion: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((10, 10), dtype=np.int64)
    )
    for seed in SEEDS:
        for method_id in METHODS:
            label = LABELS[method_id]
            run_dir = case_dir / "runs" / f"seed_{seed}" / method_id
            config = load_config(run_dir / "config_resolved.yaml")
            model = autoencoder_from_config(config, seed=seed)
            checkpoint = run_dir / "model_best.pt"
            model.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
            checksum_before = state_dict_checksum(model)
            extracted = extract_representations(
                model,
                loader,
                device=torch.device("cpu"),
                layers=LAYERS,
            )
            checksum_after = state_dict_checksum(model)
            labels = extracted["label"].numpy().astype(np.int64, copy=False)
            ids = extracted["sample_id"].numpy().astype(np.int64, copy=False)
            if not np.array_equal(ids, sample_ids):
                raise RuntimeError("Representation sample IDs/order changed")
            if not np.array_equal(labels, expected_labels):
                raise RuntimeError("Representation labels changed")
            if checksum_before != checksum_after:
                raise RuntimeError("Representation extraction mutated model")
            all_finite = all(
                bool(torch.isfinite(extracted[layer]).all())
                for layer in LAYERS
            )
            if not all_finite:
                raise RuntimeError("Nonfinite representation")
            integrity[f"seed_{seed}/{label}"] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "checksum_before": checksum_before,
                "checksum_after": checksum_after,
                "checkpoint_unchanged": True,
                "sample_ids_sha256": sha256_array(ids),
                "labels_sha256": sha256_array(labels),
                "all_finite": True,
                "shapes": {
                    layer: list(extracted[layer].shape) for layer in LAYERS
                },
            }
            for layer in LAYERS:
                metric, embedding, predictions = representation_metrics(
                    extracted[layer],
                    labels,
                    winner_fraction=0.10,
                    epsilon=1.0e-12,
                    pca_components=50,
                    pca_seed=17,
                    cv_folds=5,
                    cv_seed=17,
                    knn_neighbors=5,
                )
                np.savez_compressed(
                    embeddings_dir / f"seed_{seed}_{label}_{layer}_pca.npz",
                    embedding=embedding.astype(np.float32, copy=False),
                    labels=labels,
                    sample_ids=ids,
                    predictions=predictions,
                )
                embeddings[(seed, label, layer)] = embedding
                confusion[(label, layer)] += confusion_matrix(
                    labels, predictions, labels=np.arange(10)
                )
                rows.append(
                    {
                        "sweep": sweep,
                        "case": case,
                        "seed": seed,
                        "method_id": method_id,
                        "method": label,
                        "layer": layer,
                        "layer_rule": layer_rule(config, layer),
                        **{
                            key: value
                            for key, value in metric.items()
                            if key
                            not in {
                                "spectrum",
                                "per_class_effective_rank",
                                "source_shape",
                                "sample_geometry_shape",
                                "channel_observation_shape",
                            }
                        },
                        "source_shape": json.dumps(metric["source_shape"]),
                        "sample_geometry_shape": json.dumps(
                            metric["sample_geometry_shape"]
                        ),
                        "channel_observation_shape": json.dumps(
                            metric["channel_observation_shape"]
                        ),
                        "per_class_effective_rank": json.dumps(
                            metric["per_class_effective_rank"], sort_keys=True
                        ),
                        "spectrum": json.dumps(metric["spectrum"]),
                    }
                )
            print(
                f"representation sweep={sweep} case={case} "
                f"seed={seed} method={label}",
                flush=True,
            )

    write_csv(output_dir / "per_seed_layer_metrics.csv", rows)
    write_csv(output_dir / "method_layer_summary.csv", summarize_rows(rows))
    write_csv(
        output_dir / "compensation_metrics.csv",
        compensation_metrics(rows),
    )
    cka_rows = []
    method_labels = tuple(LABELS.values())
    for layer in LAYERS:
        for left_index, left in enumerate(method_labels):
            for right_index, right in enumerate(method_labels):
                if left_index > right_index:
                    continue
                values = [
                    linear_cka(
                        embeddings[(seed, left, layer)],
                        embeddings[(seed, right, layer)],
                    )
                    for seed in SEEDS
                ]
                cka_rows.append(
                    {
                        "sweep": sweep,
                        "case": case,
                        "layer": layer,
                        "left_method": left,
                        "right_method": right,
                        "mean_pca50_linear_cka": float(np.mean(values)),
                        "sd_pca50_linear_cka": float(
                            np.std(values, ddof=1)
                        ),
                    }
                )
    write_csv(output_dir / "layerwise_cka.csv", cka_rows)
    confusion_rows = []
    for (method, layer), matrix in confusion.items():
        for actual in range(10):
            for predicted in range(10):
                confusion_rows.append(
                    {
                        "method": method,
                        "layer": layer,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(matrix[actual, predicted]),
                    }
                )
    write_csv(output_dir / "confusion_matrices.csv", confusion_rows)

    write_json(
        output_dir,
        "integrity.json",
        {
            "schema_version": "stage3-q5q6-representation-integrity-v1",
            "completed_at_utc": utc_now(),
            "sweep": sweep,
            "case": case,
            "record_count": len(integrity),
            "expected_record_count": len(SEEDS) * len(METHODS),
            "metric_rows": len(rows),
            "expected_metric_rows": len(SEEDS) * len(METHODS) * len(LAYERS),
            "all_checkpoints_unchanged": all(
                row["checkpoint_unchanged"] for row in integrity.values()
            ),
            "all_arrays_finite": all(
                row["all_finite"] for row in integrity.values()
            ),
            "same_sample_ids": len(
                {row["sample_ids_sha256"] for row in integrity.values()}
            ) == 1,
            "same_labels": len(
                {row["labels_sha256"] for row in integrity.values()}
            ) == 1,
            "subset_manifest": str(subset_path),
            "subset_manifest_sha256": file_sha256(subset_path),
            "test_samples_per_checkpoint": int(sample_ids.size),
            "test_used_for_selection": False,
            "source_freeze_gate": gate["decision"],
        },
        overwrite=True,
    )
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "stage3-q5q6-representation-run-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "sweep": sweep,
            "case": case,
            "methods": list(method_labels),
            "seeds": list(SEEDS),
            "layers": list(LAYERS),
            "training_performed": False,
            "performance_gate_applied": False,
        },
        overwrite=True,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/stage3_q5q6_sweeps_v1.yaml",
    )
    parser.add_argument("--sweep", choices=SWEEPS, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    print(
        run_case(
            args.config,
            sweep=args.sweep,
            case=args.case,
        ).resolve()
    )


if __name__ == "__main__":
    main()
