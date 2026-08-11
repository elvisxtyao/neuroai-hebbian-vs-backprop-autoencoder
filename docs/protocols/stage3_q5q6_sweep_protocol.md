# Stage 3 Q5/Q6 Dimension and Architecture Sweep Protocol

Date frozen: 2026-07-29

Protocol config: `configs/experiments/stage3_q5q6_sweeps_v1.yaml`

## Purpose

This protocol completes the post-core experiments needed for:

- Q5: how bottleneck dimensionality and encoder-width allocation affect
  classification, dual reconstruction, robustness, representation metrics,
  and method sensitivity;
- Q6: how width asymmetry changes h1/h2/z geometry and the layer at which
  HHB/HBB compensate for low-rank Hebbian prefixes.

No hyperparameter is selected from these runs. All cases, seeds, methods,
metrics, noise severities, and stopping rules are frozen before execution.

## Frozen matrices

Every new case uses `BBB/HHH/HHB/HBB × seeds 0–4`, BP learning rate `0.003`,
Hebbian learning rate `0.0005`, winner fraction `0.10`, 10 epochs per local
layer, 10 BP epochs, the Phase 0 split, paired initializations, the same
frozen probe, and the same standardized decoder protocol.

Latent dimension:

- `L16`: new formal runs;
- `L32`: new formal runs;
- `L64`: reuse the accepted Stage 3 core;
- `L128`: new formal runs.

Encoder-width allocation at `L=64`:

- `early_heavy = [64,28,64]`: new formal runs;
- `balanced = [16,32,64]`: reuse the accepted Stage 3 core;
- `late_heavy = [4,33,64]`: new formal runs.

The measured encoder parameter counts are `104,512`, `105,104`, and `104,712`.
Their range divided by their mean is below 1%. Decoder and total parameter
counts are reported separately because decoder mirroring does not match total
capacity exactly.

The matrix therefore adds 60 dimension system/probe/standardized-decoder runs
and 40 architecture runs. Reused L64/balanced checkpoints are not retrained
and their test set is not accessed a second time.

## Training and freeze gate

Each new case must satisfy:

- 20/20 system runs complete and finite;
- 20/20 frozen linear probes complete;
- 20/20 paired standardized decoders complete;
- all four methods share initial model, system decoder, probe, split, and
  standardized-decoder initialization within each seed;
- HHH/HHB/HBB Enc1 checkpoints match and HHH/HHB Enc2 checkpoints match;
- frozen layers and standardized encoders remain unchanged;
- exact-resume tests pass;
- all artifacts are present;
- one clean Git source commit across the case;
- zero pre-freeze test access.

There is no performance, rank, or reconstruction gate. An unexpected outcome
is retained as a formal result.

## Post-freeze evaluation

Only after a case gate passes:

1. perform one test evaluation per frozen checkpoint for classification and
   system/standardized reconstruction;
2. extract h1/h2/z metrics on the same class-balanced 2,000-image subset used
   by Q2, including effective/stable rank, winner statistics, cross-validated
   probes/k-NN, class geometry, compensation ratios, confusion, PCA, and CKA;
3. evaluate the same sample-ID-keyed Gaussian, salt-and-pepper, and masking
   curves used by Q3;
4. aggregate relative-to-balanced changes, sensitivity scores, paired
   contrasts, and method × architecture interactions.

The official test set is an outcome dataset only. It is never used to choose
dimensions, widths, checkpoints, or hyperparameters.

## Recoverability

Training is resumable at epoch boundaries and validates the resolved-config
fingerprint before resuming. Each case has an independent freeze gate and
manifest. The one-time test runner refuses a reused core case and refuses a
second completed evaluation.

## Expected cost

The 100 new system/probe/standardized-decoder records are expected to require
approximately 15–20 CPU hours on the current host. Execution proceeds in
case/seed boundaries so completed cases remain independently auditable.
