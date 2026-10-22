# Phase 0 v1.1 environment snapshot

Captured: 2026-07-23 (Asia/Shanghai)

Execution device: CPU only

## Runtime

| Field | Frozen snapshot |
|---|---|
| Operating system | Windows 10 build 26200 (`Windows-10-10.0.26200-SP0`) |
| Architecture | AMD64 |
| Processor | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| Python | 3.11.6 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| CUDA available | false |
| CUDA runtime | none |
| cuDNN | none |
| Torch CPU threads | 12 |

Exact direct dependency versions are recorded in
`environment/phase0_v1_1_requirements.txt`. The source requirements remain
bounded in `requirements.txt`.

## Determinism contract

Every training/evaluation entry point must call `set_global_seed(seed)` before
model construction and DataLoader iteration. It performs:

- Python `random.seed(seed)`;
- NumPy `numpy.random.seed(seed)`;
- `torch.manual_seed(seed)` and CUDA seeds when CUDA exists;
- `torch.use_deterministic_algorithms(True, warn_only=True)`;
- `torch.backends.cudnn.deterministic = True`;
- `torch.backends.cudnn.benchmark = False`.

Verified after `set_global_seed(0)` in this environment:

| Flag | Value |
|---|---|
| `torch.are_deterministic_algorithms_enabled()` | true |
| `torch.backends.cudnn.deterministic` | true |
| `torch.backends.cudnn.benchmark` | false |

Resume checkpoints additionally preserve Python, NumPy, Torch CPU/CUDA RNG and
the shuffled train DataLoader generator state.

## Data identity

| Field | Value |
|---|---|
| Manifest | `data/splits/mnist_split_v1.npz` |
| Split seed | 0 |
| Sizes | train 50,000 / validation 10,000 / test 10,000 |
| SHA-256 | `e7e92e0252a4ffd8b80651b9fe630f4914b563d2b6b802c0c397a8cf1c31ee54` |

No MNIST sample was read while creating this Stage 0 snapshot. The hash was
computed directly from the existing split-manifest bytes.

## Verification command

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The captured full-suite output is
`verification/phase0_v1_1/pytest_full.log`. Tests use synthetic fixtures or
read-only metadata where possible; Stage 0 launches no formal experiment and
does not evaluate the real MNIST test set.
