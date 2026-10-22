"""One-time frozen test evaluation for a Q5/Q6 sweep case."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from data.mnist import build_mnist_dataloaders
from evaluation.run_stage3_test_evaluation import (
    _load_frozen_components,
    evaluate_frozen_components,
)
from schemas import load_config
from training.run_stage3_q5q6_sweeps import (
    METHODS,
    ROOT,
    SEEDS,
    SWEEPS,
    validate_protocol,
)
from utils.checkpointing import file_sha256
from utils.results import write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2026) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values, size=(10_000, values.size), replace=True
    ).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _require_gate(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "freeze_gate.json"
    if not path.exists():
        raise RuntimeError("Case freeze gate is missing; refusing test access")
    gate = _read_json(path)
    if gate.get("decision") != "PASS":
        raise RuntimeError("Case freeze gate did not pass")
    if not all(bool(value) for value in gate.get("global_checks", {}).values()):
        raise RuntimeError("Case freeze gate contains a failed check")
    if int(gate.get("test_samples_accessed", -1)) != 0:
        raise RuntimeError("Pre-test gate must report zero test access")
    return gate


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
    )
    by_method: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method_id"] == method]
        by_method[method] = {}
        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in method_rows], dtype=np.float64
            )
            by_method[method][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "bootstrap_95_ci": _bootstrap_ci(values),
                "values": values.tolist(),
            }
    lookup = {(row["seed"], row["method_id"]): row for row in rows}
    contrasts = (
        ("HHB_minus_HHH", "hybrid_hhb", "full_hebbian"),
        ("HBB_minus_HHB", "hybrid_hbb", "hybrid_hhb"),
        ("BBB_minus_HHB", "full_bp", "hybrid_hhb"),
    )
    paired: dict[str, Any] = {}
    for name, left, right in contrasts:
        paired[name] = {}
        for metric in metrics:
            differences = np.asarray(
                [
                    lookup[(seed, left)][metric]
                    - lookup[(seed, right)][metric]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            paired[name][metric] = {
                "mean_difference": float(differences.mean()),
                "sd_difference": float(differences.std(ddof=1)),
                "paired_bootstrap_95_ci": _bootstrap_ci(differences),
                "paired_differences": differences.tolist(),
            }
    return {"by_method": by_method, "paired_contrasts": paired}


def run_case(
    protocol_path: str | Path,
    *,
    sweep: str,
    case: str,
    device: torch.device | None = None,
) -> Path:
    protocol_path = Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    if sweep not in SWEEPS:
        raise ValueError("Unknown sweep")
    case_spec = protocol[f"{sweep}_cases"].get(case)
    if case_spec is None:
        raise ValueError("Unknown sweep case")
    if case_spec["source"] != "new_formal_run":
        raise ValueError("Reused core case must not access test a second time")
    output_root = Path(protocol["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    case_dir = output_root / sweep / case
    gate = _require_gate(case_dir)
    evaluation_dir = case_dir / "test_evaluation"
    if (evaluation_dir / "summary.json").exists():
        raise FileExistsError("One-time case test evaluation already completed")
    if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
        raise RuntimeError(
            "Incomplete prior test access exists; manual audit required"
        )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    rows: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            for method in METHODS:
                run_dir = (
                    case_dir / "runs" / f"seed_{seed}" / method
                )
                config = load_config(run_dir / "config_resolved.yaml")
                system, standardized, probe = _load_frozen_components(
                    run_dir, config, device
                )
                loader = build_mnist_dataloaders(
                    config,
                    seed=seed,
                    download=False,
                    include_test=True,
                )["test"]
                metrics = evaluate_frozen_components(
                    system,
                    standardized,
                    probe,
                    loader,
                    device=device,
                )
                row = {
                    "schema_version": "stage3-q5q6-test-run-v1",
                    "sweep": sweep,
                    "case": case,
                    "seed": seed,
                    "method_id": method,
                    "split": "test",
                    "test_access_ordinal": 1,
                    "source_checkpoint_sha256": file_sha256(
                        run_dir / "model_best.pt"
                    ),
                    **metrics,
                }
                write_json(
                    evaluation_dir,
                    f"seed_{seed}_{method}.json",
                    row,
                )
                rows.append(row)
                print(
                    f"sweep={sweep} case={case} seed={seed} method={method} "
                    f"accuracy={metrics['accuracy']:.4f}",
                    flush=True,
                )
    except Exception:
        write_json(
            evaluation_dir,
            "FAILED.json",
            {
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completed_records": len(rows),
                "manual_audit_required_before_retry": True,
            },
        )
        raise

    summary = {
        "schema_version": "stage3-q5q6-test-summary-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep": sweep,
        "case": case,
        "case_spec": case_spec,
        "freeze_gate_decision": gate["decision"],
        "source_git_commit": gate["source_git_commit"],
        "records_complete": len(rows) == len(SEEDS) * len(METHODS),
        "total_checkpoint_test_evaluations": len(rows),
        "each_checkpoint_accessed_once": all(
            row["test_access_ordinal"] == 1 for row in rows
        ),
        **_summarize(rows),
    }
    write_json(evaluation_dir, "summary.json", summary)
    fields = [
        "sweep",
        "case",
        "seed",
        "method_id",
        "accuracy",
        "macro_f1",
        "classification_ce",
        "system_reconstruction_mse",
        "standardized_reconstruction_mse",
        "num_samples",
    ]
    with (evaluation_dir / "per_run_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields} for row in rows
        )
    return evaluation_dir / "summary.json"


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
