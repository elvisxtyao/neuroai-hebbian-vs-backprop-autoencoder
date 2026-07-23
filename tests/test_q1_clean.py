from pathlib import Path

import numpy as np

from schemas import load_config
from training.run_q1_clean import Q1Runner, _bootstrap_ci, _curve_summary


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "experiments" / "q1_clean_v1.yaml"


def test_q1_manifest_uses_five_paired_seeds_and_shared_model(tmp_path):
    runner = Q1Runner(MANIFEST, tmp_path / "q1")

    assert runner.manifest["paired_seeds"] == [0, 1, 2, 3, 4]
    assert runner.manifest["test_policy"] == "evaluate_once_after_config_freeze"
    assert runner.bp_base["model"] == runner.hebbian_base["model"]


def test_curve_summary_reports_aligned_aulc_and_threshold():
    rows = [
        {
            "epoch": str(epoch),
            "accuracy": str(accuracy),
            "samples_seen": str(epoch * 100),
            "wall_time_sec": str(epoch * 2),
        }
        for epoch, accuracy in enumerate((0.80, 0.86, 0.90), start=1)
    ]

    summary = _curve_summary(rows, threshold=0.85)

    assert np.isclose(summary["epoch_aulc"], (0.80 + 0.86 + 0.90) / 3)
    assert summary["samples_to_threshold"] == 200
    assert summary["wall_time_to_threshold_sec"] == 4
    assert summary["probe_samples_seen"] == 300


def test_paired_bootstrap_is_reproducible_and_bounded():
    differences = np.asarray([-0.0182, -0.0093], dtype=np.float64)

    first = _bootstrap_ci(differences, seed=2026, resamples=1_000)
    second = _bootstrap_ci(differences, seed=2026, resamples=1_000)

    assert first == second
    assert differences.min() <= first[0] <= first[1] <= differences.max()


def test_stop_after_seed_pauses_before_later_seeds(tmp_path, monkeypatch):
    runner = Q1Runner(MANIFEST, tmp_path / "q1")
    calls: list[tuple[str, int] | tuple[str, tuple[int, ...]]] = []

    monkeypatch.setattr(
        runner,
        "_representation_run",
        lambda rule, seed: calls.append((rule, seed)),
    )
    monkeypatch.setattr(
        runner,
        "_random_probe_run",
        lambda seed: calls.append(("random", seed)),
    )
    monkeypatch.setattr(runner, "_validate_pairing", lambda seed: None)
    monkeypatch.setattr(
        runner,
        "_analyze",
        lambda seeds: calls.append(("analyze", tuple(seeds))),
    )

    runner.run(stop_after_seed=1)

    assert ("bp", 2) not in calls
    assert ("hebbian", 2) not in calls
    assert ("analyze", (0, 1)) in calls
    assert runner.state["status"] == "paused"
    assert runner.state["paused_after_seed"] == 1
