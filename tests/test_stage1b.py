from dataclasses import replace
from pathlib import Path

from training.run_stage1b import (
    Stage1BResult,
    Stage1BRunner,
    select_eligible,
)


ROOT = Path(__file__).resolve().parents[1]


def _result(
    trial_id: str,
    *,
    accuracy: float,
    ce: float,
    health_pass: bool,
    accuracy_pass: bool,
) -> Stage1BResult:
    return Stage1BResult(
        trial_id=trial_id,
        config_sha256=trial_id,
        run_dir=trial_id,
        competition_mode="channel_rms",
        competition_power=1.0,
        winner_fraction=0.1,
        validation_accuracy=accuracy,
        validation_macro_f1=accuracy,
        validation_ce=ce,
        health_pass=health_pass,
        accuracy_pass=accuracy_pass,
        eligible=health_pass and accuracy_pass,
        h1_effective_rank=3.0,
        h2_effective_rank=4.0,
        z_effective_rank=5.0,
        h1_winner_coverage=1.0,
        h2_winner_coverage=1.0,
        z_winner_coverage=1.0,
        status="completed",
        error=None,
    )


def test_stage1b_manifest_freezes_four_unique_equal_budget_candidates(tmp_path):
    runner = Stage1BRunner.__new__(Stage1BRunner)
    import yaml

    manifest_path = ROOT / "configs" / "tuning" / "stage1b_homeostasis_v1.yaml"
    runner.manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    runner.base = __import__("schemas").load_config(
        ROOT / "configs" / "selected" / "hebbian_validation_selected.yaml",
        validate=False,
    )
    configs = [
        Stage1BRunner.candidate_config(runner, candidate)
        for candidate in runner.manifest["candidates"]
    ]

    assert len(configs) == 4
    assert {config["training"]["seed"] for config in configs} == {42}
    assert {config["model"]["latent_dim"] for config in configs} == {64}
    assert {config["hebbian"]["lr"] for config in configs} == {0.0005}
    assert {
        config["training"]["hebbian_epochs_per_layer"] for config in configs
    } == {10}


def test_stage1b_selection_requires_health_and_accuracy_then_maximizes_accuracy():
    health_only = _result(
        "health_only",
        accuracy=0.88,
        ce=0.2,
        health_pass=True,
        accuracy_pass=False,
    )
    accuracy_only = _result(
        "accuracy_only",
        accuracy=0.95,
        ce=0.1,
        health_pass=False,
        accuracy_pass=True,
    )
    eligible_lower = _result(
        "eligible_lower",
        accuracy=0.90,
        ce=0.3,
        health_pass=True,
        accuracy_pass=True,
    )
    eligible_best = replace(
        eligible_lower,
        trial_id="eligible_best",
        validation_accuracy=0.91,
        validation_ce=0.4,
    )

    selected = select_eligible(
        [health_only, accuracy_only, eligible_lower, eligible_best]
    )

    assert selected is not None
    assert selected.trial_id == "eligible_best"
