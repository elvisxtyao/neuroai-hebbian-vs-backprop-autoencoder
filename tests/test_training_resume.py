import csv
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from schemas import load_config
from training.train_representation import train_config
from utils.reproducibility import state_dict_checksum
from utils.results import METRIC_FIELDS, read_metadata, read_run_status


ROOT = Path(__file__).resolve().parents[1]


def _config(rule: str) -> dict:
    config = load_config(ROOT / "configs" / f"{rule}_main.yaml")
    config = deepcopy(config)
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["model"]["latent_dim"] = 4
    config["training"]["bp_epochs"] = 2
    config["training"]["hebbian_epochs_per_layer"] = 1
    config["training"]["decoder_epochs"] = 1
    return config


def _loaders(seed: int = 0):
    generator = torch.Generator().manual_seed(999)
    images = torch.rand(8, 1, 28, 28, generator=generator)
    labels = torch.arange(8) % 10
    sample_ids = torch.arange(8)
    dataset = TensorDataset(images, labels, sample_ids)
    return {
        "train": DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        "validation": DataLoader(dataset, batch_size=4, shuffle=False),
        "test": DataLoader(dataset, batch_size=4, shuffle=False),
    }


def _model_checksum(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state_dict_checksum(state)


def _metric_rows(run_dir: Path):
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scientific_metrics(rows):
    ignored = {"run_id", "wall_time_sec", "git_commit"}
    return [{key: value for key, value in row.items() if key not in ignored} for row in rows]


def test_bp_epoch_resume_matches_uninterrupted_run_and_records_status(tmp_path):
    config = _config("bp")
    full = train_config(config, loaders=_loaders(), run_root=tmp_path / "full")
    interrupted = train_config(
        config,
        loaders=_loaders(),
        run_root=tmp_path / "resumed",
        stop_after_global_epoch=1,
    )
    assert read_run_status(interrupted)["status"] == "paused"

    resumed = train_config(config, loaders=_loaders(), resume_run_dir=interrupted)

    assert resumed == interrupted
    status = read_run_status(resumed)
    assert status["status"] == "completed"
    assert status["resume_count"] == 1
    assert status["global_epoch"] == 2
    assert _model_checksum(full / "model_last.pt") == _model_checksum(
        resumed / "model_last.pt"
    )
    assert _scientific_metrics(_metric_rows(full)) == _scientific_metrics(
        _metric_rows(resumed)
    )
    assert len(_metric_rows(resumed)) == 4
    assert (resumed / "checkpoints" / "bp_representation_epoch_001.pt").exists()
    assert (resumed / "checkpoints" / "bp_representation_epoch_002.pt").exists()
    assert (resumed / "checkpoints" / "initial_state.pt").exists()
    assert (resumed / "resume_checkpoint.pt").exists()
    metadata = read_metadata(resumed)
    assert metadata["config_sha256"]
    assert metadata["run_id"] == resumed.name
    assert set(_metric_rows(resumed)[0]) == set(METRIC_FIELDS)


def test_hebbian_layerwise_and_decoder_resume_matches_uninterrupted(tmp_path):
    config = _config("hebbian")
    full = train_config(config, loaders=_loaders(), run_root=tmp_path / "full")
    interrupted = train_config(
        config,
        loaders=_loaders(),
        run_root=tmp_path / "resumed",
        stop_after_global_epoch=2,
    )

    resumed = train_config(config, loaders=_loaders(), resume_run_dir=interrupted)

    assert read_run_status(resumed)["status"] == "completed"
    assert _model_checksum(full / "model_last.pt") == _model_checksum(
        resumed / "model_last.pt"
    )
    assert _scientific_metrics(_metric_rows(full)) == _scientific_metrics(
        _metric_rows(resumed)
    )
    assert len(_metric_rows(resumed)) == 5
    assert (resumed / "checkpoints" / "hebbian_enc1_epoch_001.pt").exists()
    assert (resumed / "checkpoints" / "hebbian_enc2_epoch_001.pt").exists()
    assert (resumed / "checkpoints" / "hebbian_enc3_epoch_001.pt").exists()
    assert (resumed / "checkpoints" / "decoder_epoch_001.pt").exists()


def test_decoder_optimizer_and_rng_resume_match_uninterrupted(tmp_path):
    config = _config("hebbian")
    config["training"]["decoder_epochs"] = 2
    full = train_config(config, loaders=_loaders(), run_root=tmp_path / "full")
    interrupted = train_config(
        config,
        loaders=_loaders(),
        run_root=tmp_path / "resumed",
        stop_after_global_epoch=4,
    )
    assert read_run_status(interrupted)["stage"] == "decoder"

    resumed = train_config(config, loaders=_loaders(), resume_run_dir=interrupted)

    assert read_run_status(resumed)["status"] == "completed"
    assert _model_checksum(full / "model_last.pt") == _model_checksum(
        resumed / "model_last.pt"
    )
    assert _scientific_metrics(_metric_rows(full)) == _scientific_metrics(
        _metric_rows(resumed)
    )


def test_resume_rejects_modified_config(tmp_path):
    config = _config("bp")
    interrupted = train_config(
        config,
        loaders=_loaders(),
        run_root=tmp_path,
        stop_after_global_epoch=1,
    )
    modified = deepcopy(config)
    modified["backprop"]["lr"] *= 2

    with pytest.raises(RuntimeError, match="config"):
        train_config(modified, loaders=_loaders(), resume_run_dir=interrupted)
