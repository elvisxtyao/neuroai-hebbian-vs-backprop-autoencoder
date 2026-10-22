# Stage 1B — Hebbian repair/reselection

Date: 2026-07-23

Status: **COMPLETED — NO CANDIDATE PASSED**

Stage 1B tested a small, preregistered validation-only repair matrix after the
selected Hebbian encoder failed the Stage 1 representation-health gate. The
stage is now frozen. No v3/v4 candidates will be added, no candidate is
selected, and no formal Hebbian config is superseded by these trials.

## 1. Frozen acceptance rule

Each candidate had to satisfy both conditions:

1. pass the unchanged Stage 1 health gate at `h1`, `h2`, and `z`; and
2. reach validation linear-probe accuracy at or above `0.8863`.

The noninferiority floor is the prior selected validation accuracy `0.9063`
minus the preregistered `0.02` tolerance. Eligible candidates would have been
ranked by validation accuracy, then classification CE, then lexical trial ID.
The fixed analysis subset is
`data/splits/mnist_validation_health_v1.npz`: 2,000 validation images, exactly
200 per class, with SHA-256
`5198fcd030eb1b37f9c9b29767cb2b84870c674f66e805f57b77b0bcd9e2fe1f`.

All tuning, probe selection, and health analysis used train/validation data
only. Official MNIST test samples accessed: **0**.

## 2. Candidate matrices

### v1 — stateless competition normalization

Manifest: `configs/tuning/stage1b_homeostasis_v1.yaml`

Source commit: `86b454d4da57f2f488d679ccc4db111b3fcaab39`

| Candidate | Competition | WTA | Validation accuracy | h1 rank | h2 rank | z rank | z winner coverage | Health | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `rms_power_0p5_wta_0p10` | channel RMS, power 0.5 | 0.10 | 0.9025 | 1.1733 | 1.1139 | 1.0271 | 0.1094 | FAIL | No |
| `rms_power_1p0_wta_0p10` | channel RMS, power 1.0 | 0.10 | 0.8248 | 1.2296 | 1.1932 | 1.1751 | 0.9844 | FAIL | No |
| `standardized_wta_0p10` | channel standardized | 0.10 | 0.8272 | 1.2291 | 1.1924 | 1.1661 | 0.9844 | FAIL | No |
| `standardized_wta_0p20` | channel standardized | 0.20 | 0.7945 | 1.1510 | 1.1409 | 1.1397 | 0.9844 | FAIL | No |

The weak RMS normalization retained acceptable probe accuracy but preserved
the fixed-winner, rank-near-one `z`. Stronger normalization distributed
winners across channels, but covariance rank stayed near one and validation
accuracy fell below the floor. Winner coverage alone therefore did not repair
the representation.

### v2 — centered local inputs

Manifest: `configs/tuning/stage1b_centered_v2.yaml`

Source commit: `1c64dac11147a7ae5e8a3caa071538ca53413d1b`

The four v2 candidates were preregistered and launched before the Stage 1B
freeze instruction. They were allowed to finish; no later candidate was
created or started.

| Candidate | Competition | Hebbian LR | Validation accuracy | h1 rank | h2 rank | z rank | z winner coverage | Health | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `centered_rms_lr_0p0005` | channel RMS | 0.0005 | 0.8148 | 1.2420 | 1.4169 | 4.3638 | 0.7656 | FAIL | No |
| `centered_rms_lr_0p001` | channel RMS | 0.0010 | 0.2031 | 1.2004 | 1.3259 | 1.0279 | 0.2812 | FAIL | No |
| `centered_standardized_lr_0p0005` | channel standardized | 0.0005 | 0.8394 | 1.2415 | 1.4066 | 4.3348 | 0.7344 | FAIL | No |
| `centered_standardized_lr_0p001` | channel standardized | 0.0010 | 0.1810 | 1.1997 | 1.3208 | 1.1024 | 0.3125 | FAIL | No |

Centering the local input patches at learning rate `0.0005` increased `z`
participation-ratio rank to about 4.3 and broadened winner coverage. It still
failed the normalized-rank requirement: the best normalized `z` rank was
`4.3638 / 64 = 0.0682`, below `0.10`. Both corresponding validation
accuracies were also below `0.8863`. At learning rate `0.001`, representation
and probe performance collapsed sharply.

## 3. Integrity and reproducibility evidence

- Both manifests were committed before their corresponding runs.
- Both runners recorded a clean worktree and the exact source commit.
- All eight trials completed 30 greedy encoder epochs
  (`10 × enc1`, `10 × enc2`, `10 × enc3`).
- Every trial has a resolved config, run status, checkpoints, validation probe
  metrics, per-layer health JSON, and trial-table row.
- Every health JSON records identical `state_dict_sha256_before` and
  `state_dict_sha256_after`; health extraction did not mutate a checkpoint.
- Both selection records contain `decision="FAIL"`, `selected=null`, and
  `test_samples_accessed=0`.
- v1 full suite: `45 passed in 19.28s`.
- v2 full suite: `47 passed in 27.77s`.

Immutable tracked test logs:

```text
verification/phase0_v1_1/stage1b_pytest.log
verification/phase0_v1_1/stage1b_v2_pytest.log
```

Local result roots:

```text
results/tuning/stage1b_homeostasis_v1/
results/tuning/stage1b_centered_v2/
```

Key artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| v1 `trial_table.csv` | `17067364eac0665f8d6430015d7ce13efbf13fc4d24e1aa78e34099faf0a576f` |
| v1 `selection_decision.json` | `1e9b174ac20f65e0e78f7a70abd23dabf2507937fe2f112fe0b61ceafd22bec8` |
| v2 `trial_table.csv` | `25f12287471e305cf3ad08350101671ce2bcb2268f36fbbc5a396e98cec59d7b` |
| v2 `selection_decision.json` | `59d9d8539d1d9632a1376ebc3e089ffbca023851d40500e9becbcf1d7f9ba5a5` |

## 4. Decision and limits

**Stage 1B outcome: COMPLETED — NO CANDIDATE PASSED.**

No repaired configuration satisfies both representation health and validation
performance. The repository therefore does not freeze a replacement formal
Hebbian config, and these seed-42 validation trials must not be presented as
formal multi-seed or test results.

The trials support two limited mechanism observations:

- distributing WTA winners across channels is not sufficient to produce a
  high-rank representation; and
- centered local inputs can raise the measured bottleneck rank, but the
  improvement did not meet the frozen rank or validation-performance gates.

Before Q4, Stage 1C will audit whether the effective-rank computation and
representation axes are scientifically appropriate. Stage 1C performs no
training and no hyperparameter selection. It does not reopen Stage 1B.
