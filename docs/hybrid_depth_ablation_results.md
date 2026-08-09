# Stage 2C — Hybrid Hebbian–BP Depth Ablation Results

Date: 2026-07-25

Status: **PASS — diagnostic experiment complete**

Selected outcome: **D**

Stage 3 candidate: **Hybrid-HHB, confirmation required**

## 1. Scope and integrity

All four methods were rerun from clean implementation commit
`d6a105f1c8951199a318662b7fb34ebb1439026f`. They used diagnostic seed 42,
the same model/split/latent dimension, the same full-model and decoder
initialization, and the same frozen linear-probe initialization and training
protocol.

The pairing gate passed every check:

- four methods complete on the same clean commit;
- identical full-model, decoder, split and probe-classifier initialization
  hashes;
- frozen encoder layers excluded from optimizer groups and unchanged;
- complete resume/checkpoint artifacts;
- no test rows and `test_samples_accessed=0`.

The immutable implementation suite reports `78 passed in 31.68s`.

## 2. Final validation performance

| Method | Accuracy | Macro-F1 | CE | Reconstruction MSE | Performance gate |
|---|---:|---:|---:|---:|---|
| Full BP | 0.9226 | 0.9216 | 0.2621 | 0.006196 | pass |
| Full Hebbian | 0.9063 | 0.9050 | 0.3113 | 0.017981 | pass |
| Hybrid-HHB | 0.9097 | 0.9082 | 0.3050 | 0.006492 | pass |
| Hybrid-HBB | 0.9161 | 0.9147 | 0.2831 | 0.005613 | pass |

Both controls exceed the frozen non-inferiority floor `0.8863`. Relative to
Full Hebbian, Hybrid-HHB improves accuracy by 0.34 percentage points and
reduces reconstruction MSE by 63.9%. Hybrid-HBB improves accuracy by 0.98
percentage points and reduces reconstruction MSE by 68.8%.

These are seed-42 diagnostic differences, not multi-seed estimates.

## 3. Layerwise representation results

### Effective rank

| Method | h1 | h2 | z |
|---|---:|---:|---:|
| Full BP | 1.5426 | 5.9564 | 11.8533 |
| Full Hebbian | 1.1456 | 1.0384 | 1.0186 |
| Hybrid-HHB | 1.1456 | 1.0384 | 10.0850 |
| Hybrid-HBB | 1.1456 | 5.0657 | 12.9754 |

Hybrid-HHB leaves the Hebbian h1/h2 geometry exactly unchanged and changes the
z effective rank from `1.0186` to `10.0850`. Hybrid-HBB additionally changes
h2 from `1.0384` to `5.0657`, while z reaches `12.9754`.

The normalized effective-rank changes tell the same story:

- Full Hebbian h2/z: `0.0325 / 0.0159`;
- Hybrid-HHB h2/z: `0.0325 / 0.1576`;
- Hybrid-HBB h2/z: `0.1583 / 0.2027`.

Thus BP at Enc3 repairs bottleneck diversity despite collapsed Hebbian inputs,
while BP at both Enc2 and Enc3 provides an additional h2 and z diversity gain.

### Other quantitative geometry

| Method/layer | Stable rank | Mean abs activation correlation | Mean abs filter cosine | CV linear probe | CV 5-NN |
|---|---:|---:|---:|---:|---:|
| Full Hebbian h2 | 1.0191 | 0.2788 | 0.0824 | 0.8790 | 0.8195 |
| Hybrid-HHB h2 | 1.0191 | 0.2788 | 0.0824 | 0.8790 | 0.8195 |
| Hybrid-HHB z | 3.9532 | 0.1967 | 0.0569 | 0.8625 | 0.8965 |
| Hybrid-HBB h2 | 2.5640 | 0.2586 | 0.1271 | 0.9030 | 0.8695 |
| Hybrid-HBB z | 4.9897 | 0.1280 | 0.0599 | 0.8660 | 0.9140 |
| Full BP z | 4.8005 | 0.1229 | 0.0598 | 0.8775 | 0.9220 |

The low Full-Hebbian z effective rank coexists with low filter cosine
similarity (`0.0297`). The failure therefore cannot be explained simply as
identical convolution filters; activation geometry and the training objective
remain important.

Winner metrics were only computed for Hebbian-trained layers. The unchanged
Hebbian h1/h2 portions of both hybrids retain the baseline winner statistics:

- h1 coverage `0.75`, entropy `0.5432`, max share `0.4046`;
- h2 coverage `0.3125`, entropy `0.4225`, max share `0.2499`.

## 4. Health-gate result

Neither hybrid passes the complete applicable health gate:

- Hybrid-HHB: h1 and h2 fail; BP-trained z passes all applicable diversity
  checks.
- Hybrid-HBB: Hebbian h1 fails; BP-trained h2 and z pass.

Full BP itself fails the strict complete gate at h1 and z while reaching
`0.9226` validation accuracy. This supports Outcome D: the present gate is not
a sufficient predictor of MNIST frozen-probe classification performance. No
threshold was changed.

The gate still identifies the severe Full-Hebbian rank/winner-concentration
failure and may remain useful for reconstruction, robustness, or broad
representation quality. Those predictive roles are not established by this
single diagnostic experiment.

## 5. Outcome and localization

**Outcome D**

> The current health gate is not a sufficient predictor of MNIST
> classification performance.

The architecture-localization evidence is more nuanced than “Enc3 alone”:

- replacing only Hebbian Enc3 with BP restores z rank and reconstruction,
  showing Enc3 local training is a major bottleneck;
- replacing Enc2 and Enc3 provides a further h2/z diversity and classification
  gain;
- Hebbian h1 remains below the frozen health thresholds in both controls.

The failure therefore accumulates across Enc2/Enc3, while Hebbian Enc1 still
supports useful classification but does not satisfy the current general
health definition. Representation diversity and performance are related but
not interchangeable: large rank improvements produce only modest accuracy
changes on MNIST, while reconstruction improves substantially.

## 6. Candidate decision

Both Hybrid-HHB and Hybrid-HBB satisfy the preregistered minimum requirements
for a confirmation candidate. The protocol prefers the smaller BP intervention
when both are eligible, so the selected candidate is:

`Hybrid-HHB`

This is a hybrid diagnostic candidate, not a pure Hebbian baseline. It requires
separately approved confirmation seeds before Stage 3 and cannot yet support a
cross-seed claim.

## 7. Evidence

- `results/hybrid_depth_ablation/decision.json`
- `results/hybrid_depth_ablation/pairing_gate.json`
- `results/hybrid_depth_ablation/run_manifest.json`
- `results/hybrid_depth_ablation/performance_metrics.csv`
- `results/hybrid_depth_ablation/representation_metrics.csv`
- `results/hybrid_depth_ablation/analysis_manifest.json`
- `results/hybrid_depth_ablation/runs/*`
- `results/hybrid_depth_ablation/figures/*`
- `verification/phase0_v1_1/hybrid_depth_implementation_pytest.log`

## 8. Single next task

Preregister and run the two validation-only confirmation seeds for the frozen
Hybrid-HHB config. This requires further user approval; it was not started in
this task.
