# Step 3 — Formal validation-only tuning

Date: 2026-07-23  
Status: complete  
Tuning seed: 42  
Selection metric: frozen linear-probe validation accuracy

## Protocol and leakage controls

The search used the fixed MNIST training/validation split. Test samples were
not extracted by the probe and no test metric was computed or written. Model
selection maximized validation accuracy; validation CE and lexical trial ID
were preregistered tie-breakers.

- Encoder/AE training budget: 10 epochs per BP trial or 10 epochs per Hebbian
  layer.
- Probe budget: 30 epochs, standardized frozen features, shared SGD protocol.
- Hebbian tuning skipped decoder training because decoder state cannot affect
  a frozen-encoder probe. Final configs restore the standard 10-epoch decoder.
- Normalization/stabilization was a preregistered singleton: Oja plus
  per-filter L2 normalization. No unsupported alternative was inferred.
- Eight unique physical trials were allowed for each learning rule.
- All 896 metric rows across the 16 physical trials were audited; test rows:
  **0**.

The Hebbian sequential budget consisted of four LR runs, two new WTA runs
(`0.20` reused the selected LR run), and two new dimension runs (`L=64` reused
the selected WTA run). `L=32` is deferred to the complete Q5 matrix so the
search remains within eight unique runs. There are 10 logical Hebbian rows but
only 8 physical runs. BP used an explicit 4×2 LR/weight-decay grid.

## Hebbian results

### Learning-rate stage — fixed L=64, winner fraction 0.20

| LR | Validation accuracy | Macro-F1 | CE |
|---:|---:|---:|---:|
| 1e-4 | 89.40% | 89.24% | 0.3697 |
| **5e-4** | **89.65%** | **89.50%** | 0.3505 |
| 1e-3 | 89.32% | 89.18% | 0.3549 |
| 5e-3 | 89.41% | 89.29% | 0.3449 |

Selected LR: `5e-4`.

### Winner-fraction stage — fixed L=64, LR 5e-4

| Winner fraction | Validation accuracy | Macro-F1 | CE |
|---:|---:|---:|---:|
| 0.05 | 89.29% | 89.17% | 0.3518 |
| **0.10** | **90.63%** | **90.50%** | **0.3113** |
| 0.20 | 89.65% | 89.50% | 0.3505 |

Selected main-comparison winner fraction: `0.10`.

### Coarse dimensionality screen — fixed LR 5e-4, winner fraction 0.10

| Latent dimension | Validation accuracy | Macro-F1 | CE |
|---:|---:|---:|---:|
| 16 | 76.71% | 76.36% | 0.7482 |
| **64 (main architecture)** | **90.63%** | **90.50%** | 0.3113 |
| 128 | 93.10% | 93.01% | 0.2299 |

`L=128` is the best Hebbian-only dimension screen, improving 2.47 percentage
points over L=64. It is **not** substituted into the Q1 main configuration:
all BP trials used the frozen L=64 architecture, so BP-L64 versus
Hebbian-L128 would confound learning rule and capacity. The L=128 result is
retained as Q5 evidence and requires a paired BP-L128 run there.

## BP results — shared L=64 architecture

| LR | Weight decay | Validation accuracy | Macro-F1 | CE |
|---:|---:|---:|---:|---:|
| 3e-4 | 0 | 91.89% | 91.75% | 0.2885 |
| 3e-4 | 1e-4 | 91.56% | 91.43% | 0.2992 |
| 1e-3 | 0 | 91.56% | 91.44% | 0.2965 |
| 1e-3 | 1e-4 | 91.67% | 91.54% | 0.2994 |
| **3e-3** | **0** | **92.26%** | **92.16%** | **0.2621** |
| 3e-3 | 1e-4 | 90.96% | 90.82% | 0.3134 |
| 1e-2 | 0 | 91.17% | 91.09% | 0.2875 |
| 1e-2 | 1e-4 | 91.06% | 90.92% | 0.3145 |

Selected BP configuration: `lr=3e-3`, `weight_decay=0`.

## Frozen main-comparison configs

The Q1 configs keep the same L=64 model and differ only in learning-rule and
selected optimizer/rule parameters.

| Rule | Selected parameters | Validation accuracy | Selected config SHA-256 |
|---|---|---:|---|
| Hebbian | `lr=5e-4`, `winner_fraction=0.10`, Oja + filter L2 | 90.63% | `d08edbba2bf16e4a76b076bd732a0fa70b258c6a87f2e3a79d02c64562eba377` |
| BP | `lr=3e-3`, `weight_decay=0` | 92.26% | `400aa9a79179c045af29ef180287a407d3c0f9ae1a10dc5d65f413e3a5586a99` |

The immutable seed-42 selected files are:

- `configs/selected/hebbian_validation_selected.yaml`
- `configs/selected/bp_validation_selected.yaml`

The operational `hebbian_main.yaml` and `bp_main.yaml` now contain the
selected hyperparameters while inheriting seed 0 for the first paired main
run. Their resolved hashes are
`8ee94009cff7840522b9fc9a11bf8f663daf0d137fb9c70cc7b0e946a7f9c3d7` and
`2562c2cff7242040e228146c0b3dd007dce1c84d7fb377e5bb19fc1d14e313b3`.

## Mechanism warning

Accuracy selection does not resolve Hebbian competition collapse. In the
selected L=64 trial, final enc3 active-neuron ratio was `0.109375`, winner
entropy `0.4679`, and `collapse_detected=True`; activation variance was nonzero
(`146.81`). The formal validation criterion therefore selects a useful
classifier, but the mechanism gate still fails at depth. This must remain
visible in Q2 and must not be described as if tuning solved collapse.

At seed 42 and shared L=64, BP exceeds Hebbian validation accuracy by 1.63
percentage points. This is a tuning-seed observation, not a Q1 result; Q1
requires fresh paired seeds 0–4 and confidence intervals.

## Recovery and completeness

One one-hour execution window ended during a BP trial. A runner-level orphan
detection defect briefly created a duplicate run. The defect was fixed; the
newer run resumed from its epoch checkpoint (`resume_count=1`), and the older
duplicate was marked `Superseded duplicate` and excluded. No scientific trial
was dropped or counted twice. Final logical results: 18 completed, 0 failed;
physical budget: 8 Hebbian + 8 BP.

## Reproduction and local evidence

```powershell
python -m training.run_validation_tuning `
  --manifest configs/tuning/validation_tuning_v1.yaml `
  --output-dir results/tuning/validation_tuning_v1
```

The manifest and implementation are tracked:

- `configs/tuning/validation_tuning_v1.yaml`
- `training/run_validation_tuning.py`
- `training/train_linear_probe.py`
- `evaluation/plot_tuning_results.py`
- `tests/test_validation_tuning.py`

Run-specific trial tables, decisions, checkpoints, and the figure are retained
locally under `results/tuning/validation_tuning_v1/` according to repository
policy. No test accuracy, test macro-F1, test CE, or test reconstruction metric
was used or produced in this step.

Final verification:

```text
29 passed in 60.5s
physical trials=16
test metric rows=0
incomplete physical trials=0
shared selected model config=True
selected config hashes match decision=True
```
