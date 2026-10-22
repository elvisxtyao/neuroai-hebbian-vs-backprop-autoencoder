import csv
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from schemas import load_config
from training.train_random_encoder_decoder import train_random_decoder_config
from utils.reproducibility import state_dict_checksum
from utils.results import read_run_status


ROOT = Path(__file__).resolve().parents[1]


def _config():
    config = deepcopy(load_config(ROOT / "configs" / "bp_main.yaml"))
    config["data"]["batch_size"] = 4
    config["training"]["batch_size"] = 4
    config["training"]["decoder_epochs"] = 2
    config["model"]["latent_dim"] = 4
    return config


def _loaders():
    generator = torch.Generator().manual_seed(31)
    images = torch.rand(8, 1, 28, 28, generator=generator)
    labels = torch.arange(8)
    sample_ids = torch.arange(8)
    dataset = TensorDataset(images, labels, sample_ids)
    return {
        "train": DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        ),
        "validation": DataLoader(dataset, batch_size=4),
        "test": DataLoader(dataset, batch_size=4),
    }


def _checksum(path: Path):
    return state_dict_checksum(torch.load(path, map_location="cpu", weights_only=True))


def _scientific_rows(run_dir: Path):
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ignored = {"run_id", "wall_time_sec", "git_commit"}
    return [{key: value for key, value in row.items() if key not in ignored} for row in rows]


def test_random_encoder_is_frozen_and_decoder_only_run_resumes_exactly(tmp_path):
    config = _config()
    full = train_random_decoder_config(
        config,
        loaders=_loaders(),
        run_root=tmp_path / "full",
        samples_per_class=0,
    )
    interrupted = train_random_decoder_config(
        config,
        loaders=_loaders(),
        run_root=tmp_path / "resume",
        samples_per_class=0,
        stop_after_epoch=1,
    )
    assert read_run_status(interrupted)["status"] == "paused"

    resumed = train_random_decoder_config(
        config,
        loaders=_loaders(),
        resume_run_dir=interrupted,
        samples_per_class=0,
    )

    assert read_run_status(resumed)["status"] == "completed"
    assert _checksum(full / "model_last.pt") == _checksum(resumed / "model_last.pt")
    assert _scientific_rows(full) == _scientific_rows(resumed)
    with (resumed / "random_encoder_decoder_summary.json").open(
        encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    assert summary["encoder_unchanged"]
    assert summary["checksums"]["initial_encoder"] == summary["checksums"]["trained_encoder"]
    assert summary["checksums"]["initial_decoder"] != summary["checksums"]["trained_decoder"]
