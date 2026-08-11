# Stage 1C — Effective-rank metric audit

Date: 2026-07-23

Status: **COMPLETED — METRIC VALIDITY PASS**

Mechanism classification: **PRE_AND_POST_WTA_NEAR_ONE**

Stage 1C audited the mathematical definition and implementation of the Stage 1
effective-rank metric. It reused the exact seed-42 Hebbian checkpoint and
class-balanced 2,000-image validation subset. It performed no training,
optimizer step, local update, probe fitting, hyperparameter selection, or test
access.

## 1. Frozen inputs

| Item | Value |
|---|---|
| Final audit source commit | `b1108ce3645d41db891f2c818fcf6c8e8bbce0c5` |
| Config | `configs/experiments/effective_rank_audit_v1_1.yaml` |
| Config SHA-256 | `c77b2a7d9da8f2ec85e29d8274baf5cd160ef4fd6903f9087c257d9708a42f22` |
| Checkpoint | validation-selected Hebbian seed 42 |
| Checkpoint SHA-256 | `96aaf6f5b036dfccc31f7f51c04d4f3450a048589b0ee132a306ff2bc24c34c4` |
| Frozen probe SHA-256 | `75553eb19e398830f65d49977816f13058c3a5dc2dd424d5381a9b657113be5e` |
| Subset manifest | `data/splits/mnist_validation_health_v1.npz` |
| Subset SHA-256 | `5198fcd030eb1b37f9c9b29767cb2b84870c674f66e805f57b77b0bcd9e2fe1f` |
| Subset | 2,000 unique validation images; exactly 200 per class |
| Test samples accessed | 0 |

The final run started from a clean worktree. Model, checkpoint-file, probe-state
and probe-file hashes were identical before and after the audit.

## 2. What Stage 1 actually measured

The encoder's normal forward path is:

```text
Conv2d -> ReLU -> next layer
```

WTA is used only inside the explicit Hebbian local-update rule. It is not
applied to the encoder output used by the decoder or linear probe. Therefore:

- Stage 1 effective rank was computed on standard post-ReLU, **pre-WTA**
  activations;
- Stage 1 winner statistics applied an analysis-only top-k operation to the
  same activations;
- Stage 1C explicitly constructs `z_post_wta` to isolate what WTA would do,
  but this masked vector is not the normal model representation.

This distinction rules out the interpretation that the Stage 1 rank of
`1.0186` was itself calculated after WTA.

## 3. Axis and flattening audit

For convolutional `h1` and `h2`, Stage 1 used the channel-health view:

```text
rows = sample × spatial location
columns = channels
```

Stage 1C also computed a separate sample-representation view:

```text
rows = samples
columns = channel × height × width
```

| Layer/view | Matrix shape | Participation rank | Stable rank | PR / feature dimension |
|---|---:|---:|---:|---:|
| h1 channel-health | 392,000 × 16 | 1.1456 | 1.0712 | 0.07160 |
| h1 sample-flat | 2,000 × 3,136 | 20.3794 | 7.8574 | 0.00650 |
| h2 channel-health | 98,000 × 32 | 1.0384 | 1.0191 | 0.03245 |
| h2 sample-flat | 2,000 × 1,568 | 8.8313 | 4.2676 | 0.00563 |
| z pre-WTA | 2,000 × 64 | 1.0186 | 1.0092 | 0.01592 |

The two convolutional views answer different questions. Channel-health rank
measures diversity among filters across all locations. Sample-flat rank also
contains spatial-position variation and is appropriate for sample geometry.
The higher h1/h2 sample-flat ranks therefore do not invalidate the Stage 1
channel calculation, but the two ranks must not share a name or threshold.

For `z`, height and width are both one, so channel-health and sample-flat
matrices are bitwise identical.

## 4. Pre/post-WTA and transformation results

All covariance calculations used float64, explicit dataset-level feature
centering, and an observations-by-features matrix. The conceptual covariance
shape is always feature × feature. When features exceeded observations, the
implementation used the mathematically equivalent dual Gram matrix only to
obtain the nonzero feature-covariance spectrum.

| z representation | Participation rank | Stable rank | Rank ratio | Numerical rank | Interpretation |
|---|---:|---:|---:|---:|---|
| pre-WTA | 1.018568 | 1.009246 | 0.015915 | 57 | Strongly anisotropic before WTA |
| post-WTA | 1.000000 | 1.000000 | 0.015625 | 1 | WTA makes the existing low-rank structure exactly rank-1 |
| post-WTA, dataset centered input | 1.000000 | 1.000000 | 0.015625 | 1 | Same covariance, as expected |
| post-WTA, per-sample L2 | undefined at Stage 1 epsilon | undefined | — | — | All normalized samples are effectively identical |
| post-WTA, class centered | 1.000000 | 1.000000 | 0.015625 | 1 | Class centering does not restore dimensions |

For the L2-normalized variant, the centered covariance trace is only
`1.43e-31`. A raw floating-point calculation returns a rank near one from
roundoff, but the Stage 1 epsilon correctly marks the variance as numerical
zero. This derived variant is therefore recorded as undefined rather than used
in the mechanism decision.

Per-class pre-WTA participation ranks range from `1.0144` to `1.0226`.
The pooled within-class and between-class covariance participation ranks are
`1.0171` and `1.0209`. Post-WTA, every class and both covariance components are
rank one.

## 5. Epsilon audit

The primary pre-WTA and post-WTA conclusions are stable over relative
eigenvalue cutoffs from `0` through `1e-6`. Their covariance totals are far
above every Stage 1 denominator epsilon, so epsilon does not produce the main
near-one result.

The only epsilon-dominated case is the L2-normalized post-WTA representation,
because normalization makes the already collinear samples effectively
identical before dataset centering. This is a real zero-variance boundary
case, not evidence that the primary rank calculation is unstable.

The first local `audit_v1` attempt incorrectly treated this expected
zero-variance derived case as a failure of the entire audit. It is retained as
an unvalidated attempt. The acceptance logic was corrected and tested before
the final clean-source `audit_v1_1` run; no data, model or hyperparameter
choice changed.

## 6. Frozen linear probe

| Scope | Accuracy | Macro-F1 | CE |
|---|---:|---:|---:|
| Same 2,000-image audit subset | 0.9040 | 0.90395 | 0.31646 |
| Original full validation set | 0.9063 | 0.90505 | 0.31126 |

The probe was loaded from its frozen checkpoint and evaluated without fitting.
Its standardizer rescales each feature, so low-variance directions can still
carry discriminative information. Consequently, participation rank near one
does not mean that the representation is mathematically one-dimensional or
unclassifiable: pre-WTA numerical rank is 57 and the standardized linear probe
remains strong.

Effective rank is therefore valid as a measure of raw covariance anisotropy
and channel redundancy. It should remain a mechanism/health diagnostic used
with winner concentration and standardized probe performance; it must not be
interpreted alone as proof that all class information has disappeared.

## 7. Acceptance checks

- Sample and feature axes are explicit and verified.
- Convolutional channel-health and sample-flat views are separate.
- Covariance is defined over feature dimensions.
- Dataset-level feature centering is explicit.
- Float64 spectra and epsilon/cutoff sensitivity are saved.
- All 2,000 sample IDs are unique and each class count is 200.
- Covariance eigenvalue and singular-value spectra are saved.
- Participation, stable, normalized, per-class, within-class and between-class
  ranks are saved.
- Frozen subset and full-validation probe metrics are saved.
- All representations are finite.
- Checkpoint and probe hashes are unchanged.
- Training performed: false.
- Hyperparameter selection performed: false.
- Test samples accessed: 0.
- Full test suite: `54 passed in 25.45s`.

## 8. Decision and consequence

**Stage 1C = COMPLETED; metric validity = PASS.**

Both pre-WTA and post-WTA ranks are near one. The result therefore supports
the preregistered interpretation that highly redundant filters/features are
already present before WTA; WTA further compresses them but is not the initial
cause. This is support, not direct proof of filter-weight duplication; a
weight-similarity analysis would be needed for that stronger claim.

The Stage 1/1B low-rank finding is methodologically valid, but its meaning is
narrowed: it demonstrates extreme raw-covariance anisotropy, not absence of
linearly decodable class information.

Stage 1B remains closed with no selected repair. Stage 2 / Q4 tooling may use
this exact seed-42 checkpoint as a frozen **failure-case mechanism snapshot**.
It is not thereby approved as a replacement formal Q1 configuration.

## 9. Outputs

Final local artifact root:

```text
results/formal/phase0_v1_1/stage1c_effective_rank_audit/audit_v1_1/
```

| Artifact | SHA-256 |
|---|---|
| `audit_decision.json` | `046b893219de5bdb8e41f9e2127f5832506ed22b475acede582234db36a4abd0` |
| `rank_metrics.csv` | `f25bf63d08bce9e2d98db22879979b4c14ea1b7f8f247306ec3d5d0136080db2` |
| `covariance_eigenvalues.csv` | `2c5644a77babe27aa1cc06cae0d568275de1aeca29a88ff8fba25b0f9bdfc184` |
| `singular_values.csv` | `7ae7201e900ac6664dbf3cc3411883ae5c4cf1c3a9b276432b3a3cc07a9a6d77` |
| `per_class_effective_rank.csv` | `aa2ffbb2531c6837e9863326993166149c70bea3bd00b29832b1a31e8325aba2` |
| `class_covariance_rank.csv` | `6eb8676585ecd364f7b7897c2244152c67d597f0e0cd9f81ff7e8fa7a6e6d791` |
| `epsilon_sensitivity.csv` | `74a6a09890a09e3a7f53feae52d85158fed59d6a407767ab2d334f9360210505` |
| `linear_probe_metrics.json` | `86641046747db73f90bd1905621465a697b86fe92835cc87b90562686168d524` |
| `axis_audit.json` | `1206c0397cf3e2e2ff115cdbc4e7b189a9405e80221f5496709f1f16498b52b1` |
| `run_manifest.json` | `16ce7df46c3518f3b6dfd05b110c37b7d4b467d2bf3e2042d0a6c5baeccf1d3d` |
