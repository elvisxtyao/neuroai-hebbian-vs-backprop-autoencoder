# Output-filter update-centering mechanism audit

Date: 2026-07-25

Status: **COMPLETED — CANDIDATE FAILED; BASELINE NOT REPLACED**

Scope: validation-only, seed 42, one preregistered candidate, zero test access

## 1. Question and decision

This finite mechanism experiment tested whether the update-centering operation
used by the supplied Neuromatch Microlearning notebook can repair the
representation-health failure of the frozen convolutional Oja + WTA baseline.
It did not.

The candidate failed both frozen gates:

- validation accuracy `0.1944 < 0.8863`;
- the unchanged representation-health gate failed at `h1`, `h2`, and `z`.

The result classification is therefore:

> **Common-mode output-filter update removal does not resolve the failure.**

It also does more than rescale the update: at `enc3`, its mean raw/effective
alignment with the BP reference becomes negative. The original Oja + WTA
configuration remains the named baseline, but it is still a known
representation-health failure case and is not approved as a formal replacement
configuration.

## 2. Notebook audit

Source notebook: local historical teaching notebook (not distributed in this
repository). Its content identity is preserved by the checksum below.

SHA-256:
`7d4437af298a81b128106ef0236aa00d4d9a8108e6199799cfd90026b603eca3`

The relevant `HebbianFunction` is a linear custom-autograd teaching
implementation. Its active default computes a batch-mean correlation update:

```python
grad_weight = output_for_update.t().mm(input) / len(input)
grad_weight = grad_weight - grad_weight.mean(axis=0)
```

For a linear weight matrix shaped `[out_features, in_features]`, `axis=0`
removes the update component shared across output neurons for every input
feature. The notebook's Oja-subtraction variant is present only as commented
code and is not its default execution path.

This differs from the repository's earlier centered-input-patch candidate.
Input-patch centering changes the presynaptic signal before the Hebbian
correlation is formed. Output-filter update centering instead subtracts a
common direction after the full Hebb–Oja candidate has been computed.

The notebook was not copied as the experiment implementation. Target clamping
remained disabled, the convolutional WTA and Oja terms were retained, bias
updates remained disabled, and per-filter L2 normalization remained active.
Only the output-filter common-mode removal operation was transferred.

## 3. Implementation and ordering

For convolutional weights shaped
`[out_channels, in_channels, kernel_height, kernel_width]`, the correct
output-filter axis is dimension `0`. The candidate is:

```python
raw_update_centered = (
    raw_update - raw_update.mean(dim=0, keepdim=True)
)
```

The implementation:

1. computes the complete raw Hebb–Oja candidate;
2. verifies that dimension `0` equals `out_channels`;
3. optionally centers across dimension `0`;
4. returns the candidate without changing weights;
5. applies learning-rate scaling only in `apply_local_update()`;
6. preserves the existing post-update per-filter L2 normalization.

The public `compute_local_update()` and `apply_local_update()` responsibilities
therefore remain separated.

## 4. Frozen experimental protocol

| Item | Value |
|---|---|
| Seed | `42` |
| Data used for selection/evaluation | train + validation only |
| Test samples accessed | `0` |
| Candidate count | exactly `1` |
| Candidate ID | `oja_wta_output_filter_centered` |
| Competition | raw WTA |
| Centered input patches | `false` |
| Output-filter update centering | `output_filters` |
| Hebbian learning rate | `0.0005` |
| Winner fraction | `0.10` |
| Greedy training budget | 10 epochs per layer |
| Frozen accuracy floor | `0.8863` |
| Health thresholds | unchanged |
| Candidate source commit | `9b53308b76e2b9ccc9a29b1ca2217bb12e272213` |

Training completed for 30 greedy layer epochs, 11,730 update steps, and
1,500,000 samples seen. Encoder training took 441.68 seconds; the frozen
linear probe took 37.02 seconds and selected validation epoch 9.

## 5. Performance and representation health

| Metric | Original Oja + WTA | Output-filter centered | Decision |
|---|---:|---:|---|
| Validation accuracy | 0.9063 | 0.1944 | fail |
| Accuracy floor | 0.8863 | 0.8863 | unchanged |
| `h1` effective rank | 1.1456 | 1.1352 | worse |
| `h1` winner coverage | 0.7500 | 0.7500 | unchanged |
| `h1` max winner share | 0.4046 | 0.4103 | worse |
| `h2` effective rank | 1.0384 | 1.0136 | worse |
| `h2` winner coverage | 0.3125 | 0.1875 | worse |
| `h2` max winner share | 0.2499 | 0.2500 | unchanged |
| `z` effective rank | 1.0186 | 1.0000 | worse |
| `z` winner coverage | 0.1094 | 0.1094 | unchanged |
| `z` max winner share | 0.1429 | 0.1429 | unchanged |

All three candidate layers failed the unchanged health gate. At `z`, only
7/64 channels win anywhere in the fixed validation subset and the
participation-ratio effective rank is numerically 1. The common-mode removal
therefore did not improve representation health.

## 6. Q4 frozen-snapshot comparison

The accepted baseline Stage 2 Q4 result was verified before the candidate was
run. Both analyses used three layer-end snapshots and 50 fixed batches per
snapshot. Candidate computation caused zero optimizer steps, sample IDs were
fixed, and model checksums were unchanged before and after analysis.

The deepest-layer results are:

| `enc3` metric | Baseline raw | Centered raw | Baseline effective | Centered effective |
|---|---:|---:|---:|---:|
| Mean batch alignment | 0.000749 | -0.107798 | 0.000610 | -0.103273 |
| Mean norm ratio | 1,786.64 | 97,621.94 | 0.8926 | 48.7903 |
| `alpha*` | 7.15e-7 | -4.36e-6 | 0.001035 | -0.008401 |
| Scale-matched bias | 1.000000 | 0.968463 | 1.000000 | 0.970819 |
| Hebbian SNR | 0.5846 | 0.3954 | 0.5854 | 0.3957 |
| BP-reference SNR | 1.5051 | 5.1871 | 1.5051 | 5.1871 |

The candidate's numerically smaller scale-matched bias is not evidence of
better same-direction agreement. Its `alpha*` is negative and its
mean-update alignment is `-0.2492` raw and `-0.2398` effective, so the optimal
least-squares rescaling includes a sign reversal. The candidate also greatly
increases the update norm ratio and lowers Hebbian SNR. It changes both
direction and scale; it is not a scale-only transformation.

The BP-reference SNR differs between runs because each candidate produces a
different trained encoder snapshot and paired reference-decoder trajectory.
It is not a repeated measurement on the baseline checkpoint.

## 7. Integrity and tests

The implementation tests include:

- identical filter updates are removed to zero;
- filter-specific residual directions are preserved;
- the mean centered update across output filters is zero;
- a zero update remains finite and produces no NaN;
- `compute_local_update()` does not alter weights;
- interrupted and uninterrupted training remain exactly deterministic;
- the runner accepts exactly one preregistered candidate;
- Q4 resolves and records the intended candidate snapshot.

The immutable full-suite log reports `70 passed in 27.06s`:

`verification/phase0_v1_1/output_filter_centering_pytest.log`

Log SHA-256:
`c7dd4894e7b92546f7c77cc4230133aefe4a8b906efe46950625638e0014c8dd`

The candidate run contains resolved configuration, provenance, epoch metrics,
best/last/resume checkpoints, three layer snapshots, probe artifacts, health
metrics, 150 fixed-batch Q4 rows, six Q4 aggregate rows, and 6,400 unique fixed
sample IDs. All stored update tensors are finite. No test rows were found.

## 8. Evidence locations

- Candidate run:
  `results/tuning/output_filter_centering_v1/runs/20260725T062527Z_hebbian_seed42`
- Selection decision:
  `results/tuning/output_filter_centering_v1/selection_decision.json`
- Candidate health:
  `results/tuning/output_filter_centering_v1/health/oja_wta_output_filter_centered.json`
- Candidate Q4:
  `results/tuning/output_filter_centering_v1/q4_seed42`
- Comparison:
  `results/tuning/output_filter_centering_v1/comparison/comparison_summary.json`
- Baseline Q4:
  `results/formal/phase0_v1_1/stage2_q4_tooling/seed42_v1`

The final comparison record has conclusion
`DOES_NOT_RESOLVE_FAILURE`, `eligible_to_replace_baseline=false`, and
`test_samples_accessed=0`.

## 9. Scope limit

This is a validation-only, single-seed, one-candidate mechanism experiment.
It does not reopen Stage 1B, alter any gate threshold, justify another
candidate, complete formal multi-seed Q4, or support a general comparison
between Hebbian learning and backpropagation. The negative candidate result is
retained as mechanism evidence only.
