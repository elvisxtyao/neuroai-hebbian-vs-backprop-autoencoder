# NeuroAI BP–Hebbian Shared Skeleton

Shared Phase 0 framework for comparing backpropagation and explicit local Hebbian learning in a 3-layer convolutional autoencoder on MNIST.

The frozen settings are in [PHASE0_STANDARD_V1.md](PHASE0_STANDARD_V1.md). The research plan is in [HEBBIAN_PROJECT_PLAN.md](HEBBIAN_PROJECT_PLAN.md).

## Documentation map

- [PROJECT_STATUS.md](PROJECT_STATUS.md): current completion state, blockers, and next actions; this is the live status source.
- [PHASE0_STANDARD_V1.md](PHASE0_STANDARD_V1.md): frozen BP/Hebbian comparison contract.
- [HEBBIAN_PROJECT_PLAN.md](HEBBIAN_PROJECT_PLAN.md): research questions, formulas, WBS, and acceptance criteria.
- [docs/tutorial_migration.md](docs/tutorial_migration.md): source provenance and notebook-to-module migration boundary.
- [docs/phase0_team_confirmation.md](docs/phase0_team_confirmation.md): pending BP teammate compliance evidence.
- [docs/validation_tuning.md](docs/validation_tuning.md): formal seed-42 validation-only search and frozen configs.
- [docs/q1_clean_performance.md](docs/q1_clean_performance.md): paused
  two-seed Q1 run, preliminary results, and recovery instructions.

Run directories, checkpoints, generated figures, and run-specific reports are
local-only artifacts excluded by `.gitignore`. Reproducible protocols and
current conclusions remain in tracked source, plan, and status files.

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
frozen. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for current conclusions,
limitations, and remaining gates.

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

## Pause and resume a run

Training commits one immutable checkpoint per completed epoch. BP joint
training, each Hebbian encoder layer, and the frozen decoder can all resume
without repeating completed epochs. The following option is mainly useful for
preemption tests or scheduled jobs:

```powershell
python -m training.train_representation --config configs/hebbian_main.yaml --stop-after-global-epoch 5
python -m training.train_representation --resume-run-dir results/<run-id>
```

Each run contains `metadata.json`, `run_status.json`, `metrics.csv`,
`resume_checkpoint.pt`, immutable files under `checkpoints/`, and the resolved
configuration. Resume rejects a changed configuration. See
`utils/checkpointing.py`, `utils/results.py`, and `tests/test_training_resume.py`
for the exact transaction and test coverage.

## Random-encoder decoder-only reconstruction control

This control keeps the paired encoder at its initial random weights and trains
only the shared BP decoder. It tests how much reconstruction quality can be
created by the decoder without encoder learning.

```powershell
python -m training.train_random_encoder_decoder `
  --config configs/bp_main.yaml `
  --bp-run results/<bp-run-id> `
  --hebbian-run results/<hebbian-run-id>
```

The seed-0 result and interpretation are summarized in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

## Validation-only tuning

The formal tuning manifest uses seed 42, balanced eight-unique-trial budgets,
and never evaluates a test metric. The runner is safe to restart.

```powershell
python -m training.run_validation_tuning `
  --manifest configs/tuning/validation_tuning_v1.yaml `
  --output-dir results/tuning/validation_tuning_v1
```

Selected L=64 main-comparison configs are in `configs/selected/`. See
[docs/validation_tuning.md](docs/validation_tuning.md).

## Q1 clean-performance run

The resumable Q1 runner executes paired BP, Hebbian, and random-encoder
controls, then writes raw rows, paired differences, bootstrap summaries, and
learning-curve figures. It can stop safely after a complete seed:

```powershell
python -m training.run_q1_clean `
  --manifest configs/experiments/q1_clean_v1.yaml `
  --output-dir results/q1_clean_v1 `
  --stop-after-seed 1
```

The current run is paused after seeds 0–1. Its preliminary mean test accuracy
is 91.595% for BP, 90.220% for Hebbian, and 82.765% for the random encoder.
See [docs/q1_clean_performance.md](docs/q1_clean_performance.md) before citing
these values; five paired seeds are still required for the final Q1 claim.
Resume the full matrix by omitting `--stop-after-seed`.

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
