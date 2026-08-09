"""Deterministic noise evaluation for one frozen Q5/Q6 sweep case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from data.mnist import build_mnist_dataloaders
from evaluation.run_stage3_q3_noise import (
    METRIC_FIELDS,
    degradation_rows,
    evaluate_condition,
    load_components,
    plot_results,
    summarize,
    write_csv,
)
from schemas import load_config
from training.run_stage3_q5q6_sweeps import METHODS, ROOT, SEEDS, SWEEPS, validate_protocol
from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import state_dict_checksum
from utils.results import write_json


LABELS = {
    "full_bp": "BBB",
    "full_hebbian": "HHH",
    "hybrid_hhb": "HHB",
    "hybrid_hbb": "HBB",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise ValueError("Noise runner requires a new formal case")
    output_root = Path(protocol["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    case_dir = output_root / sweep / case
    gate = _read_json(case_dir / "freeze_gate.json")
    test_summary = _read_json(case_dir / "test_evaluation" / "summary.json")
    if gate.get("decision") != "PASS" or not test_summary.get(
        "records_complete"
    ):
        raise RuntimeError("Sweep case is not formally frozen/evaluated")
    output_dir = case_dir / "noise"
    if output_dir.exists():
        raise FileExistsError("Immutable case noise output exists")
    output_dir.mkdir(parents=True)

    first_config = load_config(
        case_dir / "runs" / "seed_0" / "full_bp" / "config_resolved.yaml"
    )
    loader = build_mnist_dataloaders(
        first_config,
        seed=0,
        include_test=True,
        download=False,
    )["test"]
    noise_spec = protocol["noise"]
    severities = {
        "gaussian": [float(value) for value in noise_spec["gaussian"]],
        "salt_pepper": [
            float(value) for value in noise_spec["salt_pepper"]
        ],
        "pixel_masking": [
            float(value) for value in noise_spec["pixel_masking"]
        ],
    }
    if any(values != [0.0, 0.1, 0.2, 0.3, 0.4] for values in severities.values()):
        raise RuntimeError("Frozen noise severities changed")
    noise_seed = int(noise_spec["noise_seed"])
    records = []
    integrity = {}
    condition_fingerprints: dict[tuple[str, float], str] = {}
    for seed in SEEDS:
        for method_id in METHODS:
            method = LABELS[method_id]
            run_dir = case_dir / "runs" / f"seed_{seed}" / method_id
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
                noise_seed=noise_seed,
                salt_probability=0.5,
                clean_reference=None,
            )
            records.append(
                {
                    "sweep": sweep,
                    "case": case,
                    "seed": seed,
                    "method_id": method_id,
                    "method": method,
                    "noise_type": "clean",
                    "severity": 0.0,
                    **clean_metrics,
                }
            )
            condition_fingerprints.setdefault(("clean", 0.0), clean_hash)
            if condition_fingerprints[("clean", 0.0)] != clean_hash:
                raise RuntimeError("Clean tensor differs across checkpoints")
            for noise_type, values in severities.items():
                for severity in values[1:]:
                    metrics, _, noise_hash = evaluate_condition(
                        system,
                        standardized,
                        probe,
                        loader,
                        noise_type=noise_type,
                        severity=severity,
                        noise_seed=noise_seed,
                        salt_probability=0.5,
                        clean_reference=clean_reference,
                    )
                    key = (noise_type, severity)
                    condition_fingerprints.setdefault(key, noise_hash)
                    if condition_fingerprints[key] != noise_hash:
                        raise RuntimeError(
                            f"Noise tensor differs across checkpoints: {key}"
                        )
                    records.append(
                        {
                            "sweep": sweep,
                            "case": case,
                            "seed": seed,
                            "method_id": method_id,
                            "method": method,
                            "noise_type": noise_type,
                            "severity": severity,
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
            print(
                f"noise sweep={sweep} case={case} seed={seed} method={method}",
                flush=True,
            )

    degraded = degradation_rows(records)
    summary, contrasts = summarize(degraded)
    write_csv(output_dir / "per_seed_condition_metrics.csv", degraded)
    write_csv(output_dir / "condition_summary.csv", summary)
    write_csv(output_dir / "paired_degradation_contrasts.csv", contrasts)
    plot_results(output_dir, summary)
    write_json(
        output_dir,
        "noise_fingerprints.json",
        {
            f"{key[0]}:{key[1]:.1f}": value
            for key, value in sorted(condition_fingerprints.items())
        },
    )
    expected_rows = len(SEEDS) * len(METHODS) * 13
    write_json(
        output_dir,
        "integrity.json",
        {
            "schema_version": "stage3-q5q6-noise-integrity-v1",
            "completed_at_utc": utc_now(),
            "sweep": sweep,
            "case": case,
            "records": integrity,
            "all_components_unchanged": all(
                value["unchanged"] for value in integrity.values()
            ),
            "checkpoint_count": len(integrity),
            "expected_checkpoint_count": len(SEEDS) * len(METHODS),
            "condition_count_per_checkpoint": 13,
            "metric_row_count": len(records),
            "expected_metric_row_count": expected_rows,
            "all_metrics_finite": all(
                np.isfinite(float(row[field]))
                for row in records
                for field in METRIC_FIELDS
            ),
            "same_noise_tensor_across_methods": True,
            "test_samples_per_condition": 10000,
            "test_used_for_selection": False,
            "training_performed": False,
            "source_freeze_gate": gate["decision"],
        },
        overwrite=True,
    )
    write_json(
        output_dir,
        "run_manifest.json",
        {
            "schema_version": "stage3-q5q6-noise-run-v1",
            "completed_at_utc": utc_now(),
            "protocol": str(protocol_path),
            "protocol_sha256": file_sha256(protocol_path),
            "sweep": sweep,
            "case": case,
            "methods": LABELS,
            "seeds": list(SEEDS),
            "noise": noise_spec,
            "records": len(records),
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
