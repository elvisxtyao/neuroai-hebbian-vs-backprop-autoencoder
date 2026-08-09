"""Sample-ID keyed deterministic image corruption for formal robustness tests."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
C1 = np.uint64(0x9E3779B97F4A7C15)
C2 = np.uint64(0xBF58476D1CE4E5B9)
C3 = np.uint64(0x94D049BB133111EB)


def _namespace_key(
    *, noise_seed: int, noise_type: str, severity: float, stream: int
) -> np.uint64:
    payload = (
        f"stage3-noise-v1|{noise_seed}|{noise_type}|{severity:.8f}|{stream}"
    ).encode("utf-8")
    return np.uint64(
        int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    z = (values + C1) & MASK64
    z = ((z ^ (z >> np.uint64(30))) * C2) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * C3) & MASK64
    return z ^ (z >> np.uint64(31))


def _uniform(
    sample_ids: np.ndarray,
    pixel_count: int,
    *,
    noise_seed: int,
    noise_type: str,
    severity: float,
    stream: int,
) -> np.ndarray:
    ids = np.asarray(sample_ids, dtype=np.uint64).reshape(-1, 1)
    pixels = np.arange(pixel_count, dtype=np.uint64).reshape(1, -1)
    key = _namespace_key(
        noise_seed=noise_seed,
        noise_type=noise_type,
        severity=severity,
        stream=stream,
    )
    counters = key ^ ((ids + np.uint64(1)) * C1) ^ ((pixels + 1) * C2)
    values = _splitmix64(counters)
    return ((values >> np.uint64(11)).astype(np.float64) + 0.5) / float(
        1 << 53
    )


def apply_deterministic_noise(
    images: torch.Tensor,
    sample_ids: torch.Tensor | np.ndarray,
    *,
    noise_type: str,
    severity: float,
    noise_seed: int = 2026,
    salt_probability: float = 0.5,
) -> torch.Tensor:
    """Corrupt CPU NCHW images without using global random state."""

    if images.ndim != 4:
        raise ValueError("images must have NCHW shape")
    if images.device.type != "cpu":
        raise ValueError("deterministic noise is generated on CPU")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0,1]")
    if noise_type not in {"gaussian", "salt_pepper", "pixel_masking"}:
        raise ValueError(f"unsupported noise type: {noise_type}")
    if not 0.0 <= salt_probability <= 1.0:
        raise ValueError("salt_probability must be in [0,1]")
    ids = np.asarray(torch.as_tensor(sample_ids).cpu(), dtype=np.int64)
    if ids.ndim != 1 or ids.size != images.shape[0]:
        raise ValueError("sample_ids must contain one ID per image")
    if severity == 0.0:
        return images.clone()

    original = images.detach().cpu().numpy().astype(np.float32, copy=False)
    flat = original.reshape(original.shape[0], -1)
    uniform = _uniform(
        ids,
        flat.shape[1],
        noise_seed=noise_seed,
        noise_type=noise_type,
        severity=severity,
        stream=0,
    )
    if noise_type == "gaussian":
        second = _uniform(
            ids,
            flat.shape[1],
            noise_seed=noise_seed,
            noise_type=noise_type,
            severity=severity,
            stream=1,
        )
        gaussian = np.sqrt(-2.0 * np.log(np.maximum(uniform, 1e-15))) * np.cos(
            2.0 * np.pi * second
        )
        corrupted = flat + float(severity) * gaussian.astype(np.float32)
    elif noise_type == "salt_pepper":
        salt_draw = _uniform(
            ids,
            flat.shape[1],
            noise_seed=noise_seed,
            noise_type=noise_type,
            severity=severity,
            stream=1,
        )
        corrupted = flat.copy()
        selected = uniform < severity
        corrupted[selected] = (salt_draw[selected] < salt_probability).astype(
            np.float32
        )
    else:
        corrupted = flat.copy()
        corrupted[uniform < severity] = 0.0
    corrupted = np.clip(corrupted, 0.0, 1.0).reshape(original.shape)
    return torch.from_numpy(corrupted.astype(np.float32, copy=False))
