"""One-time test evaluation for Stage 3 RBB/RRB matched controls."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from data.mnist import build_mnist_dataloaders
from evaluation.run_stage3_test_evaluation import (
    _bootstrap_ci,
    _load_frozen_components,
    _read_json,
    evaluate_frozen_components,
)
from schemas import load_config
from utils.results import write_json


SEEDS = (0, 1, 2, 3, 4)
METHODS = ("random_hbb", "random_rrb")


def _summary(records):
    metrics = (
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
    )
    result = {}
    for method in METHODS:
        result[method] = {}
        rows = [row for row in records if row["method_id"] == method]
        for metric in metrics:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            result[method][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "bootstrap_95_ci": _bootstrap_ci(values),
                "values": values.tolist(),
            }
    return result


def run(results_root: Path, *, device: torch.device | None = None) -> Path:
    gate = _read_json(results_root / "freeze_gate.json")
    if gate.get("decision") != "PASS" or int(gate.get("test_samples_accessed", -1)) != 0:
        raise RuntimeError("Matched-control freeze gate is not eligible for test")
    if not all(bool(value) for value in gate.get("global_checks", {}).values()):
        raise RuntimeError("Matched-control freeze gate has a failed check")
    output_dir = results_root / "test_evaluation"
    if output_dir.exists():
        raise FileExistsError("Matched-control test evaluation already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    try:
        for seed in SEEDS:
            for method in METHODS:
                run_dir = results_root / "runs" / f"seed_{seed}" / method
                config = load_config(run_dir / "config_resolved.yaml")
                system, standardized, probe = _load_frozen_components(
                    run_dir, config, device
                )
                loader = build_mnist_dataloaders(
                    config, seed=seed, download=False, include_test=True
                )["test"]
                metrics = evaluate_frozen_components(
                    system, standardized, probe, loader, device=device
                )
                record = {
                    "schema_version": "stage3-matched-one-time-test-run-v1",
                    "seed": seed,
                    "method_id": method,
                    "split": "test",
                    "test_access_ordinal": 1,
                    **metrics,
                }
                write_json(output_dir, f"seed_{seed}_{method}.json", record)
                records.append(record)
                print(
                    f"seed={seed} method={method} accuracy={metrics['accuracy']:.4f} "
                    f"standardized_mse={metrics['standardized_reconstruction_mse']:.6f}",
                    flush=True,
                )
    except Exception:
        write_json(
            output_dir,
            "FAILED.json",
            {
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_records": len(records),
            },
        )
        raise
    payload = {
        "schema_version": "stage3-matched-one-time-test-summary-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_complete": len(records) == 10,
        "each_checkpoint_accessed_once": all(
            row["test_access_ordinal"] == 1 for row in records
        ),
        "test_samples_per_checkpoint": records[0]["num_samples"],
        "by_method": _summary(records),
    }
    write_json(output_dir, "summary.json", payload)
    fields = (
        "seed",
        "method_id",
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
        "num_samples",
    )
    with (output_dir / "per_run_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in records)
    return output_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/formal/phase0_v1_1/stage3_matched_controls",
    )
    args = parser.parse_args()
    print(run(Path(args.results_root)).resolve())


if __name__ == "__main__":
    main()
