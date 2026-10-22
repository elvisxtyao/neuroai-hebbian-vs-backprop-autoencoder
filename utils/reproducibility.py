"""Deterministic seed and model-state helpers."""

from __future__ import annotations

import hashlib
import random
import subprocess
from collections.abc import Mapping

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def state_dict_checksum(state: Mapping[str, torch.Tensor] | torch.nn.Module) -> str:
    """Return a stable SHA-256 checksum for a module or state dict."""

    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def git_provenance(workdir: str | None = None) -> dict[str, str | bool | None]:
    """Return the current commit and dirty flag without requiring Git."""

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=workdir,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain") if commit is not None else None
    branch = run("branch", "--show-current") if commit is not None else None
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_worktree_dirty": None if status is None else bool(status),
    }
