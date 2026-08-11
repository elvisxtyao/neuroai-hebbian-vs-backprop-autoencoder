# Stage 2 / Q4 — Seed-42 update-mechanism tooling validation

Date: 2026-07-23

Status: **COMPLETED — TOOLING GATE PASS**

Scope: **single-seed failure-case mechanism snapshot**

This stage implements and validates the Q4 frozen-snapshot analysis pipeline.
It uses the exact seed-42 Hebbian run authorized by Stage 1C. That checkpoint
failed the representation-health gate, so this result validates the analysis
tool and provides failure-case mechanism evidence; it is not a
representation-health PASS, a selected formal Hebbian configuration, or a
multi-seed answer to Q4.

## 1. Frozen inputs and source

| Item | Value |
|---|---|
| Tooling source commit | `6a1b10a68e1e0a4d42c552458540216ac1708d69` |
| Config | `configs/experiments/q4_tooling_seed42_v1.yaml` |
| Config SHA-256 | `df0ddb314bf2506feaf5c5155842126de1eefcdcd7ace239842a94b240432c36` |
| Source run | `results/tuning/validation_tuning_v1/runs/20260723T071035Z_hebbian_seed42` |
| Analysis split | MNIST training split only |
| Batch manifest | `data/splits/mnist_q4_update_batches_seed31415_v1.npz` |
| Manifest SHA-256 | `43b3fcaad66c5f6c9949cd5561acad0903504e6b7189758cbafc312a5f34fb0c` |
| Batch-ID SHA-256 | `f94590bed6094399f0cc80756273d0e1d6fcdec14a7bb57e65dcef0a42d988cd` |
| Sampling | 50 fixed batches × 128 images; 6,400 unique IDs |
| Test samples accessed | 0 |

The snapshot/layer pairs are `enc1_end/enc1`, `enc2_end/enc2`, and
`enc3_end/enc3`. Their source-file SHA-256 values are:

| Snapshot | File SHA-256 | State-dict SHA-256 |
|---|---|---|
| `enc1_end` | `f2e8e3568be8552324343644cfcf161690991701d2254fe561464c23aeb9766c` | `9f57b06ba505979bc88e7aa30c4cd58fec5ad4ce3ddabe468aade1853e0303d7` |
| `enc2_end` | `46ac507c0996485841005167814567c7c9b77766cdd7df93036d3eecb51e5657` | `9b68c0dcbacfddbf366d612d36a27dacc7083b7ad1e2ee9b14f94de86a411743` |
| `enc3_end` | `a430e984916beb11f5a11456f9785355ee56630709f7eabeda133d7c47314b2c` | `118545a8766e116327def6b27ed46ee91b3c8eb51b8d472fda9e037969ed35c3` |

The integrity gate also confirmed that earlier layers were frozen after their
greedy training stage and that later layers were still at their required
initial states at each boundary.

## 2. Reference and candidate definitions

- **Raw BP direction:** the reconstruction loss negative gradient for the
  active encoder layer, computed with `torch.autograd.grad` on the frozen
  snapshot and its trained reference decoder. It excludes Adam state,
  momentum, weight decay, and optimizer steps.
- **Raw Hebbian delta:** the explicit WTA/Oja local candidate returned before
  learning-rate scaling and filter normalization.
- **Effective Hebbian delta:** the exact parameter displacement that would
  result from applying the configured Hebbian learning rate followed by
  per-filter L2 normalization, evaluated on a copy without mutating the
  analysis model.
- **Alignment:** cosine similarity between a Hebbian candidate and the raw BP
  direction on the same batch and state.
- **Norm ratio:** candidate norm divided by BP-direction norm.
- **Scale match:** `alpha*` is fitted from the two 50-batch mean updates; the
  primary bias is the relative residual after scaling the Hebbian mean by
  `alpha*`.
- **SNR:** squared norm of the across-batch mean divided by mean squared
  deviation around that mean. BP and each Hebbian variant are calculated
  separately on the same frozen state.

No target clamping is used. Candidate collection creates no optimizer and
performs no `optimizer.step()` or local-update application.

## 3. Paired reference decoders

Each snapshot received a separately trained BP reference decoder. All three
decoders started from the same initialization hash
`71edcbbd558ae0e615f9590a3a3862cc143dc176e54f47c2a19f42ba211aa751`.
The encoder was frozen throughout. Adam used the frozen BP learning rate
`0.003`; selection used validation MSE only, and no test data was read.

| Snapshot | Best epoch | Validation MSE | Encoder unchanged |
|---|---:|---:|---|
| `enc1_end` | 10 | 0.0130146 | Yes |
| `enc2_end` | 10 | 0.0115040 | Yes |
| `enc3_end` | 10 | 0.0176709 | Yes |

Each decoder training run took 3,910 optimizer steps. These are decoder
training steps, not update-analysis steps; update analysis performed exactly
zero optimizer steps. The snapshot-boundary resume path was used once after
the process was interrupted after the first two decoders. The same source
commit, config, output directory, completed decoder artifacts, and hashes were
retained, and the final gate found no duplication or state change.

## 4. Seed-42 results

The table reports means across the 50 fixed batches. SNR is linear, not dB.

| Layer | Hebbian variant | Alignment mean ± SD | Norm ratio | `alpha*` | Scale-matched bias | Candidate SNR | BP SNR |
|---|---|---:|---:|---:|---:|---:|---:|
| enc1 | raw | 0.34176 ± 0.06912 | 101.5570 | 0.003473 | 0.90020 | 434.1682 | 1.5433 |
| enc1 | effective | 0.34164 ± 0.06910 | 0.05078 | 6.94408 | 0.90027 | 427.3153 | 1.5433 |
| enc2 | raw | 0.00806 ± 0.04601 | 23.1930 | 0.019537 | 0.99504 | 0.02029 | 0.7453 |
| enc2 | effective | 0.00807 ± 0.04601 | 0.01160 | 38.83507 | 0.99504 | 0.02029 | 0.7453 |
| enc3 | raw | 0.00075 ± 0.00095 | 1786.6386 | 0.000000715 | 0.9999995 | 0.58457 | 1.5051 |
| enc3 | effective | 0.00061 ± 0.00094 | 0.89257 | 0.001035 | 0.9999997 | 0.58540 | 1.5051 |

Mean-update cosine gives the same depth trend: `0.43547`, `0.09943`, and
`0.00100` for raw Hebbian at enc1–enc3; effective values are `0.43532`,
`0.09945`, and `0.00073`.

## 5. Mechanism interpretation

For this failed-health seed-42 model, Hebbian/BP direction alignment decreases
sharply with depth: moderately positive at enc1, near zero at enc2, and
effectively orthogonal at enc3. The very large raw norm ratios mostly reflect
the different native scales of the two rules. Applying the configured
learning rate and normalization changes magnitude substantially, but it does
not improve direction alignment.

Scale-matched bias rises from about `0.90` at enc1 to `0.995` at enc2 and
almost `1.0` at enc3. Thus a single optimal rescaling cannot reconcile the
deep Hebbian mean update with the reconstruction BP direction.

The enc1 Hebbian update is highly stable across batches, but its high SNR does
not imply agreement with BP. Enc2 instead has very low candidate SNR, while
enc3 retains moderate SNR with almost no BP alignment. SNR and alignment
therefore measure distinct properties.

These observations are descriptive evidence for one collapsed/failure-case
checkpoint. They do not establish that update misalignment caused the
representation pathology, and they cannot be generalized across seeds or
compared with a healthy Hebbian configuration until such a configuration is
approved.

## 6. Acceptance and reproducibility checks

- All three snapshot/layer pairs are complete.
- All 50 batches use identical IDs for BP, raw Hebbian, and effective Hebbian.
- The combined CSV contains 150 batch rows and six aggregate rows.
- Stored raw tensors have shapes `50×16×1×3×3`,
  `50×32×16×3×3`, and `50×64×32×7×7`.
- All recorded tensors and metrics are finite.
- BP direction excludes optimizer state; analysis optimizer steps equal zero.
- Target clamping is false.
- Reference-decoder encoder hashes are identical before and after training.
- Analysis-state checksums and all three source snapshot file hashes are
  identical before and after sampling.
- Test samples accessed equal zero.
- Synthetic and full-suite tests: **62 passed in 16.70s**.
- Immutable test-log SHA-256:
  `5710f8bebc3e2ec168934591a1184745032ef393608172622c81ea02286882a9`.
- Independent artifact audit reloaded every NPZ, compared every batch-ID array
  with the fixed manifest, verified all indexed SHA-256 values, and passed.

## 7. Decision and remaining Q4 work

**Stage 2 / Q4 seed-42 tooling gate = PASS.**

This completes implementation and seed-42 validation of frozen-snapshot
alignment, norm ratio, `alpha*`, scale-matched bias, and across-batch SNR.
Formal Q4 remains partial: multi-seed execution, uncertainty across seeds,
comparison with a representation-health-passing Hebbian configuration, and
the preregistered exploratory correlation with performance/representation
metrics have not been completed.

## 8. Outputs

Local artifact root:

```text
results/formal/phase0_v1_1/stage2_q4_tooling/seed42_v1/
```

| Artifact | SHA-256 |
|---|---|
| `gate_decision.json` | `c4556301c47c13bb14a94a9ae5b73667fee806625fdc2faac6f71e652fafdc0d` |
| `run_manifest.json` | `d67892546fa50325985a406428e06948f3ef9914ce703d2819c593d32142c688` |
| `batch_update_metrics.csv` | `695c7eac4cc517c8e3c2e53229511d1db53461b0d6211479a710427838e190ce` |
| `aggregate_metrics.csv` | `9bd8a887810c468108b35a1ade79596e6c9498a4b9941acf5a07338cc973a2e5` |
| `update_tensor_index.json` | `957c73ee87fdd7eeb8f8fd3fbaabee83a2a3a434f1175940c2942032c50deca9` |
| `snapshot_integrity_gate.json` | `c5f861a5be6f124e56c88f013e9a1aff69350dbe40e36ae554ce6ba20c36b42d` |
| `q4_seed42_panels.png` | `23e51613d82ab29e076da677984e4ecf6d238283f40bced9443996beb7858c46` |
