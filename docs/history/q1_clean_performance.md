# Step 4 — Q1 clean performance

Date: 2026-07-23  
Status: paused after paired seeds 0–1; preliminary, not confirmatory  
Planned confirmatory sample: five paired seeds `[0, 1, 2, 3, 4]`

## Scope and stopping point

This step compares the frozen L=64 BP and Hebbian configurations selected in
Step 3. Each model uses the same MNIST split, forward architecture,
initialization protocol, decoder class, frozen standardized linear probe, and
test evaluation code. A random-encoder frozen-probe control is paired to each
seed.

The user requested a pause after seed 1. Seeds 0 and 1 are complete for BP,
Hebbian, and random-encoder control. During the polling interval before the
runner was stopped, BP seed 2 completed and Hebbian seed 2 reached enc1 epoch
7. These seed-2 artifacts are preserved for exact resume but are excluded from
all tables, plots, bootstrap intervals, and conclusions below because the
paired seed is incomplete.

The summary therefore has `analysis_status=preliminary`, `n=2` paired seeds,
and must not be presented as the planned five-seed confirmatory result.

## Frozen protocol

- Dataset: fixed stratified MNIST 50,000 train / 10,000 validation / 10,000
  official test split.
- Architecture: shared `conv3_ae_v1`, flatten latent dimension 64.
- BP: end-to-end autoencoder, Adam `lr=3e-3`, weight decay 0, 10 epochs.
- Hebbian encoder: greedy enc1 → enc2 → enc3, 10 epochs per layer, explicit
  local WTA/Oja update, `lr=5e-4`, winner fraction 0.10, per-filter L2
  normalization.
- Hebbian reconstruction: encoder frozen, shared BP decoder trained for 10
  epochs.
- Probe: encoder frozen, standardized latent features, shared linear
  classifier trained for 30 epochs and selected by validation accuracy.
- Test policy: one evaluation after configuration and validation checkpoint
  selection; test metrics never choose a checkpoint or hyperparameter.
- Pairing: BP, Hebbian, and random control have an identical initial-state
  hash within each completed seed.
- Statistics: raw seed values, mean ± sample SD, paired Hebbian-minus-BP
  differences, and a seed-level paired bootstrap with 10,000 resamples.

## Per-seed results

| Seed | Rule | Test accuracy | Macro-F1 | Classification CE | Reconstruction MSE |
|---:|---|---:|---:|---:|---:|
| 0 | BP | 91.60% | 91.46% | 0.2845 | 0.003143 |
| 0 | Hebbian | 89.78% | 89.63% | 0.3538 | 0.018789 |
| 0 | Random encoder | 82.21% | 81.98% | 0.5865 | n/a |
| 1 | BP | 91.59% | 91.48% | 0.2877 | 0.003122 |
| 1 | Hebbian | 90.66% | 90.54% | 0.3124 | 0.017566 |
| 1 | Random encoder | 83.32% | 83.10% | 0.5302 | n/a |

## Preliminary aggregate results

| Metric | BP mean ± SD | Hebbian mean ± SD | Random mean ± SD | Hebbian − BP |
|---|---:|---:|---:|---:|
| Test accuracy | 91.595 ± 0.007% | 90.220 ± 0.622% | 82.765 ± 0.785% | −1.375 pp |
| Macro-F1 | 91.468 ± 0.012% | 90.088 ± 0.642% | 82.539 ± 0.795% | −1.380 pp |
| Classification CE | 0.2861 ± 0.0023 | 0.3331 ± 0.0293 | 0.5584 ± 0.0398 | +0.0470 |
| Reconstruction MSE | 0.003132 ± 0.000015 | 0.018178 ± 0.000865 | n/a | +0.015045 |
| Probe epoch-AULC | 0.9085 ± 0.0004 | 0.8930 ± 0.0079 | 0.8180 ± 0.0106 | — |

The seed-level bootstrap intervals for Hebbian minus BP are:

| Metric | Mean paired difference | Preliminary bootstrap 95% CI |
|---|---:|---:|
| Test accuracy | −1.375 percentage points | [−1.820, −0.930] pp |
| Macro-F1 | −1.380 percentage points | [−1.825, −0.935] pp |
| Classification CE | +0.0470 | [+0.0247, +0.0693] |
| Reconstruction MSE | +0.015045 | [+0.014445, +0.015646] |

With only two paired seeds, these bootstrap intervals have very low support
and are effectively bounded by the two observed differences. They are included
to validate the analysis pipeline, not to support a final uncertainty claim.

## Learning rate and computational exposure

Both learned representations exceeded the 85% probe threshold after the first
probe epoch, corresponding to 50,000 probe samples. The random encoder did not
reach 85%. This threshold is too low to distinguish BP and Hebbian learning
speed once the encoders are frozen, so epoch/samples/wall-time AULC is more
informative for this run.

| Quantity | BP mean ± SD | Hebbian mean ± SD |
|---|---:|---:|
| Representation-training samples | 500,000 | 1,500,000 |
| Representation-training wall time | 255.2 ± 14.8 s | 287.6 ± 50.1 s |
| Full model-training samples | 500,000 | 2,000,000 |
| Full model-training wall time | 255.2 ± 14.8 s | 468.8 ± 49.0 s |
| Probe wall time | 34.2 ± 5.6 s | 36.1 ± 7.1 s |

The exposure difference is structural: BP updates all three encoder layers
simultaneously for 10 dataset passes, whereas greedy Hebbian learning gives
each layer 10 passes, totaling 30 encoder passes. The Hebbian decoder adds
another 10 passes. Wall-clock values are machine-specific CPU measurements;
sample exposure is the more portable comparison.

## Interpretation for Q1

The current evidence supports three preliminary observations:

1. The Hebbian encoder learns useful class information. Its mean accuracy is
   7.455 percentage points above the paired random encoder, so its performance
   cannot be attributed only to the shared architecture or probe.
2. BP remains better on clean classification in both completed seeds. The
   mean gap is 1.375 percentage points, with lower BP variance in this
   two-seed slice.
3. BP is much better for reconstruction. Mean Hebbian reconstruction MSE is
   5.80 times the BP MSE. Qualitatively, BP reconstructions retain sharp digit
   strokes, while Hebbian reconstructions show more blur, broken strokes, and
   shape distortion.

These observations answer the direction of Q1 only provisionally. Seeds 2–4
must be completed before making a confirmatory claim about the size and
uncertainty of the performance gap.

## Recovery, outputs, and quality checks

The runner now supports a safe seed boundary:

```powershell
python -m training.run_q1_clean `
  --manifest configs/experiments/q1_clean_v1.yaml `
  --output-dir results/q1_clean_v1 `
  --stop-after-seed 1
```

To resume seed 2 and finish seeds 3–4, run the same command without
`--stop-after-seed`. Completed stages and test rows are detected and skipped;
Hebbian seed 2 resumes from its saved enc1 epoch-7 checkpoint.

Generated local outputs are retained under `results/q1_clean_v1/`:

- `q1_state.json`: paused state and run registry;
- `q1_run_table.csv`: six included rule/seed rows;
- `paired_differences.csv`: two included paired differences;
- `q1_summary.json`: preliminary aggregate statistics;
- `q1_clean_accuracy.png`: paired accuracy and random-control comparison;
- `q1_learning_curves.png`: epoch-, sample-, and wall-time-aligned probe curves;
- per-run checkpoints, metrics, metadata, and reconstruction grids.

Quality checks completed for seeds 0–1:

- identical paired initial-state hash for BP, Hebbian, and random control;
- exactly one test probe row per included run;
- exactly one test reconstruction row per BP/Hebbian run;
- zero random-control reconstruction rows by design;
- all six expected run rows included and no seed-2 row included;
- paused state records `paused_after_seed=1`;
- reconstruction images were visually inspected against the numerical MSE.

Run-specific outputs remain local and ignored by Git. The protocol,
implementation, tests, and this result summary are tracked.
