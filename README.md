# NeuroAI BP–Hebbian Shared Skeleton

Shared Phase 0 framework for comparing backpropagation and explicit local Hebbian learning in a 3-layer convolutional autoencoder on MNIST.

The frozen settings are in [PHASE0_STANDARD_V1.md](PHASE0_STANDARD_V1.md). The research plan is in [HEBBIAN_PROJECT_PLAN.md](HEBBIAN_PROJECT_PLAN.md).

## Current seed-0 baseline

Both runs use the same deterministic MNIST split, autoencoder architecture,
initial model state and frozen linear-probe protocol.

| Test metric | Backpropagation | Hebbian encoder |
|---|---:|---:|
| Reconstruction MSE | 0.003289 | 0.019896 |
| Linear-probe accuracy | 91.81% | 89.00% |
| Linear-probe macro-F1 | 91.67% | 88.82% |

The Hebbian encoder learns useful linearly separable features, but the current
seed-0 run also shows winner concentration at depth: the final active-neuron
ratios are 100%, 46.88% and 20.31% for `enc1`, `enc2` and `enc3`. These are
single-seed development results rather than final multi-seed conclusions.

The Hebbian rule trains the encoder only. Its reconstruction decoder and
classification probe are trained with backpropagation after the encoder is
frozen. See [HEBBIAN_IMPLEMENTATION_REPORT.md](HEBBIAN_IMPLEMENTATION_REPORT.md)
for formulas, diagnostics, checksums, limitations and the complete result
record.

## Setup

Python 3.11 is supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Validate the shared skeleton

```powershell
python -m pytest
python -m scripts.smoke_test --config configs/bp_main.yaml
```

The smoke test uses synthetic inputs and does not download MNIST.

## Generate the fixed MNIST split

```powershell
python -m data.mnist --config configs/common_mnist.yaml --create-split
```

## Train and evaluate the BP baseline

```powershell
python -m training.train_representation --config configs/bp_main.yaml
python -m evaluation.evaluate_reconstruction --config configs/bp_main.yaml --run-dir results/<run-id>
python -m training.train_linear_probe --config configs/bp_main.yaml --run-dir results/<run-id>
```

## Train and evaluate the Hebbian encoder

The encoder uses explicit local competitive Oja updates, top-k WTA,
per-filter L2 normalization and greedy `enc1 -> enc2 -> enc3` training. The
trained encoder is frozen before the shared decoder and linear probe are
trained with backpropagation.

```powershell
python -m training.train_representation --config configs/hebbian_main.yaml
python -m evaluation.evaluate_reconstruction --config configs/hebbian_main.yaml --run-dir results/<run-id>
python -m training.train_linear_probe --config configs/hebbian_main.yaml --run-dir results/<run-id>
python -m evaluation.plot_run_metrics --hebbian-run results/<hebbian-run-id> --bp-run results/<bp-run-id>
```

The completed seed-0 implementation and result record is in
[HEBBIAN_IMPLEMENTATION_REPORT.md](HEBBIAN_IMPLEMENTATION_REPORT.md).

## Project layout

```text
configs/          shared and rule-specific YAML
data/             deterministic split and MNIST loaders
evaluation/       representation extraction and clean metrics
learning_rules/   BP trainer and explicit Hebbian WTA/Oja trainer
models/           shared encoder, decoder and linear probe
results/          generated runs (not source code)
schemas/          config loading and validation
scripts/          synthetic smoke test
tests/            shape, freeze, split and reproducibility checks
training/         common training entry points
utils/            seed, hash and result helpers
```
