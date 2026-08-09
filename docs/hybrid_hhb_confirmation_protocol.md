# Stage 2D — Hybrid-HHB Validation-Only Confirmation Protocol

Date frozen: 2026-07-27

Status: **PREREGISTERED — execution pending**

Parent evidence: Stage 2C seed-42 diagnostic, Outcome D.

## 1. Purpose

Stage 2D tests whether the seed-42 observation that a BP-trained Enc3 can
compensate for a low-rank Hebbian Enc1/Enc2 prefix reproduces without tuning.
It is a validation-only confirmation gate, not a formal multi-seed result.

## 2. Frozen scope

- Confirmation seeds: `43`, `44`.
- Methods per seed: `BBB` (Full BP), `HHH` (Full Hebbian), and `HHB`
  (Hybrid-HHB).
- Architecture: `conv3_ae_v1`, latent dimension `64`.
- Hebbian rule: Oja + WTA + per-filter L2 normalization.
- Hebbian learning rate: `0.0005`; winner fraction: `0.10`.
- BP learning rate: `0.003`.
- Frozen linear probe: the Phase 0 protocol.
- Analysis subset: the same class-balanced 2,000-image validation manifest
  used by Stage 1–2C.
- Test access: forbidden.

Seed 43 cannot alter seed 44. No third confirmation seed may be appended after
a failure. No hyperparameter, threshold, data, model, probe, or selection rule
may change during this stage.

## 3. Reconstruction fairness

Both reconstruction protocols must be reported.

### System reconstruction

Use the decoder produced by the actual method. For HHB, BP Enc3 and the decoder
are jointly optimized. This measures final system performance and is not, by
itself, evidence about encoder information.

### Standardized-decoder reconstruction

After each encoder is complete:

1. freeze the complete encoder;
2. initialize a new decoder from the method/seed-paired model initialization;
3. train only that decoder with Adam `lr=0.003`, betas `(0.9, 0.999)`, no
   weight decay, pixel-mean MSE, identical train batches/order, and 10 epochs;
4. select the checkpoint with minimum validation reconstruction MSE;
5. verify the encoder checksum is unchanged.

The standardized-decoder result is the primary reconstruction evidence for how
much recoverable information is contained in each encoder representation.

## 4. Per-seed confirmation gates

Each seed passes only when every condition is true:

1. HHB validation linear-probe accuracy is at least `0.8863`.
2. HHB standardized-decoder validation MSE is no more than `1.25 ×` the paired
   BBB standardized-decoder MSE.
3. `ER_z(HHB) >= 2.0`.
4. `ER_z(HHB) >= 2 × ER_z(HHH)`.
5. `ER_z(HHB) / (ER_h2(HHB) + 1e-12) >= 2.0`.
6. Per-seed pairing, frozen-layer checksum, standardized-decoder checksum,
   resume/artifact integrity, finite-value checks, and zero test access pass.

Stage 2D passes only when seeds 43 and 44 both pass. A failure stops progression
to the expanded formal Stage 3/Phase 4 matrix.

## 5. Required artifacts

- immutable protocol and resolved configs;
- clean source commit and protocol hash;
- six complete method/seed run directories;
- system and standardized-decoder checkpoints and validation metrics;
- frozen linear-probe summaries;
- h1/h2/z effective-rank rows on the fixed validation subset;
- per-seed and global pairing/integrity gates;
- machine-readable confirmation decision;
- immutable full-test log;
- final results report and live status update.
