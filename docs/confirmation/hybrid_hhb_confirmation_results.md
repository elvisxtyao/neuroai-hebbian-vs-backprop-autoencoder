# Stage 2D — Hybrid-HHB Confirmation Results

## Outcome

**COMPLETED — CONFIRMATION FAILED**

Stage 2D executed the immutable validation-only protocol for seeds 43 and 44.
Both seeds completed paired Full BP (`BBB`), Full Hebbian (`HHH`) and
Hybrid-HHB (`HHB`) runs, including system reconstruction, a separately trained
standardized decoder and a frozen linear probe. The preregistered confirmation
decision is `FAIL` because seed 43 exceeded the standardized-decoder
reconstruction limit. No third confirmation seed was added.

The source snapshot was:

```text
b57082d764565bc3a6291428cc2592a79ab5c696
```

The snapshot was clean and detached during every run. The protocol SHA-256 was
`d32895e6b23bdd72676d72289df209d48337fff8f7929d095407edba783d6854`.

## Scope and protocol

- Dataset: frozen MNIST train/validation split.
- Confirmation seeds: exactly `43` and `44`.
- Methods per seed: `BBB`, `HHH` and `HHB`.
- BP learning rate: `0.003`.
- HHB allocation: Enc1/Enc2 use the frozen WTA/Oja/L2 rule; Enc3 and the
  system decoder use BP.
- Selection and all reported metrics: validation only.
- Representation subset: the fixed class-balanced 2,000-image validation
  manifest, with 200 images per class.
- Test samples accessed: `0`.
- Standardized decoder: the completed encoder is frozen; a fresh decoder is
  trained from the same paired initialization with the same optimizer, data
  order, epochs and validation selection within each seed.

The immutable contract is in
[`hybrid_hhb_confirmation_protocol.md`](hybrid_hhb_confirmation_protocol.md).

## Performance results

| Seed | Method | Validation accuracy | Macro-F1 | System MSE | Standardized-decoder MSE |
|---:|---|---:|---:|---:|---:|
| 43 | BBB | 0.9140 | 0.9128 | 0.003043 | 0.003116 |
| 43 | HHH | 0.8626 | 0.8607 | 0.020963 | 0.020990 |
| 43 | HHB | 0.9163 | 0.9149 | 0.004830 | 0.004888 |
| 44 | BBB | 0.9239 | 0.9228 | 0.007536 | 0.007574 |
| 44 | HHH | 0.8956 | 0.8938 | 0.018711 | 0.019219 |
| 44 | HHB | 0.9118 | 0.9101 | 0.006965 | 0.006901 |

HHB exceeded the frozen accuracy floor of `0.8863` in both seeds. Relative to
HHH, HHB improved validation accuracy by 5.37 percentage points in seed 43 and
1.62 percentage points in seed 44.

System and standardized-decoder MSE are close within each HHB run. This shows
that the reconstruction result is not merely an artifact of retaining the
system decoder, but the paired BBB comparison remains seed-dependent.

## Representation results

| Seed | Method | ER(h1) | ER(h2) | ER(z) | z winner coverage |
|---:|---|---:|---:|---:|---:|
| 43 | BBB | 1.7543 | 3.6753 | 22.0484 | 0.9063 |
| 43 | HHH | 1.2837 | 1.0500 | 1.0158 | 0.1094 |
| 43 | HHB | 1.2837 | 1.0500 | 10.3835 | 0.7031 |
| 44 | BBB | 1.9100 | 6.2202 | 11.2176 | 0.3594 |
| 44 | HHH | 1.3349 | 1.0471 | 1.0174 | 0.1094 |
| 44 | HHB | 1.3349 | 1.0471 | 9.0809 | 0.4844 |

The Hebbian h1/h2 values are exactly paired between HHH and HHB within each
seed. Replacing only Enc3 with BP increased bottleneck effective rank from
approximately 1.02 to 10.38 and 9.08. Thus a BP-trained Enc3 consistently
compensated for the low-rank Hebbian h2 representation under the frozen health
definition.

## Preregistered gate decision

| Seed | Accuracy ≥ 0.8863 | Standardized MSE / BBB ≤ 1.25 | ER(z) ≥ 2 | ER(z) / ER(z HHH) ≥ 2 | ER(z) / ER(h2) ≥ 2 | Integrity | Seed decision |
|---:|---|---|---|---|---|---|---|
| 43 | PASS (0.9163) | **FAIL (1.5688)** | PASS (10.3835) | PASS (10.2218) | PASS (9.8890) | PASS | **FAIL** |
| 44 | PASS (0.9118) | PASS (0.9111) | PASS (9.0809) | PASS (8.9259) | PASS (8.6725) | PASS | PASS |

The protocol is conjunctive: both seeds must pass every gate. The final
decision is therefore:

```text
Stage 2D: CONFIRMATION FAILED
Stage 3 readiness: BLOCKED — CONFIRMATION FAILED
```

The thresholds were not changed after observing results, and a third seed was
not run.

## Integrity and acceptance evidence

- Pairing gate: `PASS` for both seeds.
- Six method runs and six standardized decoders: complete.
- Same split, model initialization, probe initialization, system-decoder
  initialization and standardized-decoder initialization within each seed:
  verified.
- Frozen-layer checksums: unchanged.
- Encoder checksums before and after standardized-decoder training: unchanged.
- Analysis checksums before and after representation extraction: unchanged for
  all six checkpoints.
- Source commit: identical and clean for all runs.
- Numerical integrity: `PASS`.
- Test samples accessed: `0`.
- Full test suite: `84/84` passed on the immutable source snapshot.
- Training and analysis stderr logs: empty.

Local reproducibility artifacts are retained under
`results/hybrid_hhb_confirmation/`, including:

- `run_manifest.json`
- `pairing_gate.json`
- `performance_metrics.csv`
- `representation_metrics.csv`
- `confirmation_decision.json`
- `analysis_manifest.json`
- `acceptance_pytest.log`
- seed-specific checkpoints, metrics, configs and resume checkpoints

These generated artifacts remain local-only under the repository artifact
policy; the protocol, implementation, tests and this result summary are
tracked.

## Supported and unsupported conclusions

The two confirmation seeds support the limited conclusion that minimal BP
credit assignment in Enc3 consistently repairs the severe HHH bottleneck-rank
failure and yields validation classification above the preregistered floor.

They do **not** support the stronger claim that HHB produces reconstruction
consistently close to BBB: the standardized-decoder fairness gate failed in
seed 43. They also do not authorize test evaluation, a five-seed formal matrix,
or a claim that one or two Hebbian layers outperform matched random prefixes.

Stage 3 remains blocked until a new, explicitly versioned research decision is
approved. Such a decision must preserve this failed result and cannot reinterpret
seed 44 alone as confirmation.
