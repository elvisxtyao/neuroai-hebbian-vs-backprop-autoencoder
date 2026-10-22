"""Compare the frozen Oja/WTA baseline with one output-centered candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from utils.checkpointing import file_sha256, utc_now
from utils.reproducibility import git_provenance


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ("h1", "h2", "z")
HEALTH_FIELDS = (
    "effective_rank",
    "winner_coverage_ratio",
    "max_winner_share",
)
Q4_FIELDS = (
    "batch_alignment_mean",
    "batch_norm_ratio_mean",
    "alpha_star",
    "scale_matched_relative_bias",
    "candidate_snr_linear",
    "bp_snr_linear",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _validation_accuracy(run_dir: Path) -> float:
    result = None
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                raise RuntimeError(f"Validation mechanism run contains test row: {run_dir}")
            if (
                row["stage"] == "linear_probe_final"
                and row["split"] == "validation"
            ):
                result = float(row["accuracy"])
    if result is None:
        raise RuntimeError(f"Missing final validation accuracy: {run_dir}")
    return result


def _q4_rows(root: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with (root / "aggregate_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if row["layer"] != "enc3":
                continue
            result[row["variant"]] = {
                field: float(row[field]) for field in Q4_FIELDS
            }
    if set(result) != {"hebbian_raw", "hebbian_effective"}:
        raise RuntimeError(f"Incomplete Enc3 Q4 results: {root}")
    gate = _load_json(root / "gate_decision.json")
    if (
        gate["decision"] != "PASS"
        or gate["analysis_optimizer_steps"] != 0
        or gate["test_samples_accessed"] != 0
    ):
        raise RuntimeError(f"Q4 integrity gate did not pass: {root}")
    return result


def compare(config_path: str | Path) -> Path:
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["version"] != "output-filter-centering-comparison-v1":
        raise ValueError("Unsupported comparison config")

    baseline_health_path = _resolve(config["baseline"]["health"])
    candidate_health_path = _resolve(config["candidate"]["health"])
    baseline_health = _load_json(baseline_health_path)
    candidate_health = _load_json(candidate_health_path)
    selection = _load_json(_resolve(config["candidate"]["selection"]))
    if (
        selection["test_samples_accessed"] != 0
        or candidate_health["test_samples_accessed"] != 0
    ):
        raise RuntimeError("Candidate accessed test data")
    trials = [
        trial
        for trial in selection["trials"]
        if trial["trial_id"] == config["candidate_id"]
    ]
    if len(trials) != 1:
        raise RuntimeError("Frozen candidate is missing or duplicated")
    trial = trials[0]

    baseline_accuracy = _validation_accuracy(
        _resolve(config["baseline"]["validation_run"])
    )
    candidate_run = Path(trial["run_dir"])
    candidate_accuracy = _validation_accuracy(candidate_run)
    performance_floor = float(config["performance_floor"])

    representation_rows: list[dict[str, Any]] = []
    representation_exactly_unchanged = True
    for layer in LAYERS:
        baseline_layer = baseline_health["layers"][layer]
        candidate_layer = candidate_health["layers"][layer]
        row: dict[str, Any] = {
            "layer": layer,
            "baseline_gate_pass": baseline_layer["gate_pass"],
            "candidate_gate_pass": candidate_layer["gate_pass"],
        }
        for field in HEALTH_FIELDS:
            baseline_value = float(baseline_layer["metrics"][field])
            candidate_value = float(candidate_layer["metrics"][field])
            row[f"baseline_{field}"] = baseline_value
            row[f"candidate_{field}"] = candidate_value
            row[f"delta_{field}"] = candidate_value - baseline_value
            representation_exactly_unchanged &= candidate_value == baseline_value
        representation_rows.append(row)

    baseline_q4 = _q4_rows(_resolve(config["baseline"]["q4"]))
    candidate_q4 = _q4_rows(_resolve(config["candidate"]["q4"]))
    q4_rows: list[dict[str, Any]] = []
    direction_statistics_unchanged = True
    for variant in ("hebbian_raw", "hebbian_effective"):
        row = {"variant": variant}
        for field in Q4_FIELDS:
            baseline_value = baseline_q4[variant][field]
            candidate_value = candidate_q4[variant][field]
            row[f"baseline_{field}"] = baseline_value
            row[f"candidate_{field}"] = candidate_value
            row[f"delta_{field}"] = candidate_value - baseline_value
            if field != "batch_norm_ratio_mean":
                direction_statistics_unchanged &= candidate_value == baseline_value
        q4_rows.append(row)

    health_pass = bool(candidate_health["health_pass"])
    accuracy_pass = candidate_accuracy >= performance_floor
    eligible = health_pass and accuracy_pass
    if eligible:
        conclusion = "COMMON_MODE_REMOVAL_IMPROVES_REPRESENTATION_HEALTH"
    elif representation_exactly_unchanged and direction_statistics_unchanged:
        conclusion = "ONLY_CHANGES_UPDATE_SCALE"
    else:
        conclusion = "DOES_NOT_RESOLVE_FAILURE"

    output_dir = _resolve(config["output_dir"])
    _write_csv(output_dir / "representation_comparison.csv", representation_rows)
    _write_csv(output_dir / "q4_enc3_comparison.csv", q4_rows)
    payload = {
        "schema_version": "output-filter-centering-comparison-v1",
        "completed_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "notebook_sha256": config["notebook_sha256"],
        "candidate_id": config["candidate_id"],
        "candidate_run": str(candidate_run),
        "baseline_validation_accuracy": baseline_accuracy,
        "candidate_validation_accuracy": candidate_accuracy,
        "performance_floor": performance_floor,
        "accuracy_pass": accuracy_pass,
        "health_pass": health_pass,
        "eligible_to_replace_baseline": eligible,
        "conclusion": conclusion,
        "test_samples_accessed": 0,
        "baseline_health_sha256": file_sha256(baseline_health_path),
        "candidate_health_sha256": file_sha256(candidate_health_path),
        "baseline_q4_gate_sha256": file_sha256(
            _resolve(config["baseline"]["q4"]) / "gate_decision.json"
        ),
        "candidate_q4_gate_sha256": file_sha256(
            _resolve(config["candidate"]["q4"]) / "gate_decision.json"
        ),
        **git_provenance(str(ROOT)),
    }
    _write_json(output_dir / "comparison_summary.json", payload)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/output_filter_centering_comparison_v1.yaml",
    )
    args = parser.parse_args()
    print(compare(args.config).resolve())


if __name__ == "__main__":
    main()
