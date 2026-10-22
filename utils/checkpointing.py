"""Atomic training checkpoints with reproducible RNG restoration."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = "training-checkpoint-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def loader_generator_state(loader) -> torch.Tensor | None:
    generator = getattr(loader, "generator", None)
    return None if generator is None else generator.get_state()


def restore_loader_generator(loader, state: torch.Tensor | None) -> None:
    if state is None:
        return
    generator = getattr(loader, "generator", None)
    if generator is None:
        raise RuntimeError("Checkpoint contains a DataLoader generator state")
    generator.set_state(state)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_epoch_checkpoint(
    run_dir: str | Path,
    *,
    archive_name: str,
    config: dict[str, Any],
    rule: str,
    stage: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    progress: dict[str, Any],
    train_loader,
) -> Path:
    run_dir = Path(run_dir)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "saved_at_utc": utc_now(),
        "config_fingerprint": config_fingerprint(config),
        "rule": rule,
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "progress": progress,
        "rng_state": capture_rng_state(),
        "train_loader_generator_state": loader_generator_state(train_loader),
    }
    archive_path = run_dir / "checkpoints" / archive_name
    if archive_path.exists():
        # A process may die after committing the immutable archive but before
        # advancing resume_checkpoint.pt. Replaying that deterministic epoch
        # must recover the transaction without overwriting its archive.
        existing = torch.load(archive_path, map_location="cpu", weights_only=False)
        identity_fields = (
            "stage",
            "active_layer",
            "completed_epoch",
            "global_epoch",
            "samples_seen",
            "steps_completed",
        )
        existing_progress = existing.get("progress", {})
        same_identity = (
            existing.get("config_fingerprint") == payload["config_fingerprint"]
            and existing.get("rule") == rule
            and existing.get("stage") == stage
            and all(existing_progress.get(key) == progress.get(key) for key in identity_fields)
        )
        existing_state = existing.get("model_state_dict", {})
        new_state = payload["model_state_dict"]
        same_model = existing_state.keys() == new_state.keys() and all(
            torch.equal(existing_state[key].detach().cpu(), new_state[key].detach().cpu())
            for key in existing_state
        )
        if not same_identity or not same_model:
            raise FileExistsError(f"Refusing to overwrite checkpoint: {archive_path}")
        payload = existing
    else:
        atomic_torch_save(payload, archive_path)
    atomic_torch_save(payload, run_dir / "resume_checkpoint.pt")
    return archive_path


def load_resume_checkpoint(
    run_dir: str | Path, config: dict[str, Any], *, rule: str
) -> dict[str, Any]:
    path = Path(run_dir) / "resume_checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing resume checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Unsupported training checkpoint schema")
    if payload.get("config_fingerprint") != config_fingerprint(config):
        raise RuntimeError("Resume config does not match the original resolved config")
    if payload.get("rule") != rule:
        raise RuntimeError("Resume checkpoint learning rule does not match config")
    return payload
