# Comparing Backpropagation, Hebbian Learning, and Minimal Hybrid Credit Assignment

Shared three-layer convolutional-autoencoder framework for comparing Full BP,
Full Hebbian, Hybrid-HHB/Hybrid-HBB, and matched random-prefix controls on
MNIST. The three-layer encoder is held fixed while the number of
Hebbian-trained layers changes from 0 to 3.

The parent settings are in [PHASE0_STANDARD_V1.md](PHASE0_STANDARD_V1.md).
Formal experiments use the versioned
[Phase 0 v1.1 addendum](PHASE0_STANDARD_V1_1_ADDENDUM.md), which freezes the
BP learning rate at `0.003` and adds reproducibility gates. The research plan
is in [HEBBIAN_PROJECT_PLAN.md](HEBBIAN_PROJECT_PLAN.md).

## Documentation map

- [PROJECT_STATUS.md](PROJECT_STATUS.md): current completion state, blockers, and next actions; this is the live status source.
- [PHASE0_STANDARD_V1.md](PHASE0_STANDARD_V1.md): frozen BP/Hebbian comparison contract.
- [PHASE0_STANDARD_V1_1_ADDENDUM.md](PHASE0_STANDARD_V1_1_ADDENDUM.md):
  formal experiment override, source snapshot, test policy, and artifact naming.
- [environment/phase0_v1_1_environment.md](environment/phase0_v1_1_environment.md):
  exact CPU runtime, deterministic flags, split hash, dependencies, and test evidence.
- [HEBBIAN_PROJECT_PLAN.md](HEBBIAN_PROJECT_PLAN.md): expanded Full
  BP/Full Hebbian/Hybrid research questions, formal model matrix,
  reconstruction controls, WBS, and acceptance criteria.
- [docs/tutorial_migration.md](docs/tutorial_migration.md): source provenance and notebook-to-module migration boundary.
- [docs/phase0_team_confirmation.md](docs/phase0_team_confirmation.md): pending BP teammate compliance evidence.
- [docs/validation_tuning.md](docs/validation_tuning.md): formal seed-42 validation-only search and frozen configs.
- [docs/q1_clean_performance.md](docs/q1_clean_performance.md): paused
  two-seed Q1 run, preliminary results, and recovery instructions.
- [docs/representation_health_gate.md](docs/representation_health_gate.md):
  completed validation-only Stage 1 gate, corrected collapse definition, and
  evidence for the required Stage 1B repair.
- [docs/stage1b_hebbian_repair.md](docs/stage1b_hebbian_repair.md):
  frozen validation-only repair outcome; all eight preregistered candidates
  failed and no replacement Hebbian config was selected.
- [docs/stage1c_effective_rank_audit.md](docs/stage1c_effective_rank_audit.md):
  completed no-training audit of rank axes, centering, spectra, epsilon
  sensitivity, pre/post-WTA mechanism and frozen-probe interpretation.
- [docs/q4_update_mechanism_seed42.md](docs/q4_update_mechanism_seed42.md):
  completed seed-42 frozen-snapshot Q4 tooling gate, update definitions,
  integrity evidence, results, and single-failure-case limitations.
- [docs/output_filter_centering_mechanism.md](docs/output_filter_centering_mechanism.md):
  notebook audit and the finite validation-only output-filter update-centering
  experiment; the sole candidate failed both frozen gates.
- [docs/hebbian_failure_case_protocol_addendum.md](docs/hebbian_failure_case_protocol_addendum.md):
  Branch-D restrictions after common-mode removal failed to repair the model.
- [docs/hybrid_depth_ablation_protocol.md](docs/hybrid_depth_ablation_protocol.md):
  preregistered validation-only Hybrid-HHB/Hybrid-HBB depth-ablation contract.
- [docs/hybrid_depth_ablation_results.md](docs/hybrid_depth_ablation_results.md):
  completed seed-42 depth-ablation metrics, Outcome D and candidate decision.
- [docs/hybrid_hhb_confirmation_protocol.md](docs/hybrid_hhb_confirmation_protocol.md):
  immutable Stage 2D seeds 43/44 confirmation gates, paired references, and
  system-versus-standardized reconstruction contract.
- [docs/hybrid_hhb_confirmation_results.md](docs/hybrid_hhb_confirmation_results.md):
  completed two-seed validation result. HHB passed the accuracy and
  representation gates in both seeds, but seed 43 failed the standardized
  reconstruction gate; this historical decision is preserved by Stage 3 v1.
- [docs/stage3_formal_protocol_v1.md](docs/stage3_formal_protocol_v1.md):
  approved formal five-seed matrix. HHB enters as a rank-repair condition with
  unresolved reconstruction stability; standardized reconstruction is a
  measured outcome rather than an entry gate.
- [docs/stage3_formal_core_results.md](docs/stage3_formal_core_results.md):
  completed five-seed core matrix, technical freeze gate, one-time test table,
  paired contrasts, learning budgets and remaining matched-control limits.

Run directories, checkpoints, generated figures, and run-specific reports are
local-only artifacts excluded by `.gitignore`. Reproducible protocols and
current conclusions remain in tracked source, plan, and status files.

Formal runs must use `configs/formal/`, live below
`results/formal/phase0_v1_1/`, and start from the immutable Git ref
`phase0-v1.1-formal`. Existing `results/` runs remain preliminary.

## Historical seed-0 baseline

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

The Full Hebbian rule trains the encoder only. Its reconstruction decoder and
classification probe are trained with backpropagation after the encoder is
frozen. Hybrid-HHB and Hybrid-HBB additionally train an encoder BP suffix.
Future reconstruction comparisons therefore report both the decoder produced
by the actual system and a standardized decoder retrained from paired
initialization after freezing the whole encoder. See
[PROJECT_STATUS.md](PROJECT_STATUS.md) for current conclusions, limitations,
and remaining gates.

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

The immutable Stage 0 full-suite record is
`verification/phase0_v1_1/pytest_full.log`.

## Stage 3 formal core matrix

Stage 3 first runs validation-only representation training, the frozen linear
probe and the paired standardized decoder. Running a selected seed is the
recommended recoverable unit:

```powershell
python -m training.run_stage3_formal_core `
  --config configs/experiments/stage3_formal_core_v1.yaml `
  --seed 0
```

Omit `--seed` only when intentionally running the complete five-seed matrix.
The pre-test freeze gate is written only after all 25 method/seed combinations
are complete. This command does not construct the official test loader.

The Stage 3 implementation test record is
`verification/phase0_v1_1/stage3_formal_core_implementation_pytest.log`.
The completed formal results are summarized in
`docs/stage3_formal_core_results.md`. The one-time test evaluator is guarded by
the saved technical freeze gate and refuses to run again when its output
directory already exists:

```powershell
python -m evaluation.run_stage3_test_evaluation `
  --results-root results/formal/phase0_v1_1/stage3_core
```

## Stage 1 representation-health gate

The gate uses a fixed, class-balanced 2,000-image validation subset and compares
per-location WTA density with dataset-wide winner coverage, entropy, variance,
and effective rank. It performs no training and never accesses MNIST test data.

```powershell
python -m evaluation.run_representation_health `
  --config configs/experiments/representation_health_v1.yaml
```

The selected Hebbian seed-42 checkpoint failed the gate: its `z` representation
uses the same seven winners on all 2,000 validation images and has effective
rank `1.0186`. Stage 1B then evaluated eight preregistered validation-only
repair candidates; none passed both the unchanged health gate and the
validation-accuracy floor. Stage 1B is frozen with no selected replacement.
Stage 1C then verified the rank implementation and found `z` participation
rank `1.0186` before WTA and `1.0000` after analysis-only WTA. The same-subset
frozen probe accuracy is `90.4%`, so low raw-covariance rank is evidence of
strong anisotropy/redundancy, not by itself absence of class information.

Reproduce the final validation-only, no-training Stage 1C audit:

```powershell
python -m evaluation.run_effective_rank_audit `
  --config configs/experiments/effective_rank_audit_v1_1.yaml
```

## Stage 2 / Q4 update-mechanism tooling

The seed-42 tooling gate compares raw reconstruction negative gradients with
raw and effective Hebbian deltas at the three greedy layer-boundary snapshots.
It uses 50 fixed training batches, performs zero analysis optimizer steps, and
accesses no test samples. The source checkpoint is a Stage 1 health-gate
failure case, so this validates the tool and supplies preliminary mechanism
evidence rather than a formal multi-seed Q4 answer.

```powershell
python -m evaluation.run_q4_tooling `
  --config configs/experiments/q4_tooling_seed42_v1.yaml
```

The immutable test record is
`verification/phase0_v1_1/q4_tooling_pytest.log` (`62 passed in 16.70s`).

The notebook-inspired output-filter update-centering candidate was then tested
once at seed 42 without test-set access. It reduced validation accuracy from
`0.9063` to `0.1944`, did not improve effective rank or winner coverage, and
made the `enc3` update anti-aligned with the BP reference. It is rejected and
does not replace the original Oja + WTA baseline. Reproduce the bounded run and
comparison with:

```powershell
python -m training.run_stage1b `
  --config configs/tuning/output_filter_centering_v1.yaml
python -m evaluation.run_q4_tooling `
  --config configs/experiments/q4_output_filter_centering_seed42_v1.yaml
python -m evaluation.compare_output_filter_centering `
  --config configs/experiments/output_filter_centering_comparison_v1.yaml
```

The corresponding full-suite record is
`verification/phase0_v1_1/output_filter_centering_pytest.log`
(`70 passed in 27.06s`).

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
