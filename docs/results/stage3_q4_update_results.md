# Stage 3 Q4 Formal Weight-Update Mechanism Results

Date completed: 2026-07-29

Analysis source: `d2eddabfbfe6f1630a624811206f024d8ba3df43`

Protocol: `configs/experiments/stage3_q4_updates_v1.yaml`

## Scope and integrity

The formal analysis reuses the frozen Stage 3 checkpoints for seeds `0–4`.
It compares:

- HHH local updates at `enc1_end`, `enc2_end`, and `enc3_end` with a
  reconstruction BP reference computed at the same frozen weights;
- the shared Hebbian prefixes of HHB (`enc1/enc2`) and HBB (`enc1`);
- raw BP gradients at the final BBB (`enc1/enc2/enc3`), HHB (`enc3`), and
  HBB (`enc2/enc3`) checkpoints.

Each layer/snapshot comparison uses the same 50 fixed training batches of 128
images. A paired decoder is trained for 10 epochs using train/validation data
only before the reference gradient is measured. The analysis itself never
performs an optimizer step.

The formal integrity gate reports:

- five complete paired seeds;
- three validated layer-boundary snapshots per seed;
- 50/50 identical batch IDs per comparison, with batch manifest SHA-256
  `f94590bed6094399f0cc80756273d0e1d6fcdec14a7bb57e65dcef0a42d988cd`;
- 90 finite seed/method/layer/rule summary rows;
- all source checkpoint hashes unchanged before and after analysis;
- zero analysis optimizer steps;
- zero test samples accessed;
- empty stderr and successful resumable completion.

Because HHH, HHB, and HBB have exactly paired Hebbian prefixes, their
identical prefix rows are deliberately mapped from one shared computation.
They are not independent observations.

## Five-seed update results

Values are mean ± sample SD across the five paired seeds. `Effective` is the
actual post-normalization Hebbian parameter delta; `raw` is the unscaled Oja
candidate. The BP reference is a raw reconstruction negative gradient.

| Condition/layer | Rule | Alignment | Norm ratio | alpha-star | Scale-matched bias | Update SNR |
|---|---|---:|---:|---:|---:|---:|
| HHH/HHB/HBB Enc1 | Hebbian effective | 0.3680 ± 0.1265 | 0.04547 ± 0.01065 | 8.283 ± 2.552 | 0.9214 ± 0.0597 | 453.09 ± 37.85 |
| HHH/HHB Enc1 | Hebbian effective | 0.03563 ± 0.04110 | 0.002242 ± 0.000573 | 14.56 ± 17.15 | 0.99869 ± 0.00212 | 0.01806 ± 0.00442 |
| HHH Enc3 | Hebbian effective | 0.000046 ± 0.000276 | 0.5687 ± 0.1823 | 0.000146 ± 0.000524 | 0.99999997 ± 0.00000003 | 1.293 ± 0.600 |
| BBB Enc1 | BP raw | 1.0 | 1.0 | 1.0 | 0.0 | 17.48 ± 17.47 |
| BBB Enc2 | BP raw | 1.0 | 1.0 | 1.0 | 0.0 | 6.922 ± 5.631 |
| BBB Enc3 | BP raw | 1.0 | 1.0 | 1.0 | 0.0 | 5.736 ± 0.492 |
| HBB Enc2 | BP raw | 1.0 | 1.0 | 1.0 | 0.0 | 6.368 ± 5.787 |
| HBB Enc3 | BP raw | 1.0 | 1.0 | 1.0 | 0.0 | 5.147 ± 2.327 |
| HHB Enc3 | BP raw | 1.0 | 1.0 | 1.0 | 0.0 | 8.523 ± 1.016 |

Raw Oja candidates have much larger norms than the BP reference: mean norm
ratios are `90.93`, `4.484`, and `1138.33` for Enc1, Enc2, and Enc3. The
per-filter normalization changes the scale substantially, but it does not
repair the direction: raw/effective alignment remains nearly the same at
Enc1/Enc2 and remains approximately zero at Enc3.

## Layerwise mechanism interpretation

1. **Enc1 contains a reproducible task-compatible component.** Its local
   direction has positive alignment with the reconstruction BP direction in
   all five seeds, although the magnitude varies substantially by seed and
   batch. This is consistent with Q1's small positive HBB−RBB and HHB−RRB
   classification contrasts.
2. **Enc2 is effectively misaligned and noisy.** Mean alignment is only
   `0.0356`, scale-matched bias is `0.9987`, and effective-update SNR is
   `0.0181` (`−17.54 dB`). High alpha-star uncertainty cannot turn this into a
   stable BP-like direction.
3. **Enc3 is essentially orthogonal to the reconstruction objective.** Its
   mean alignment is `4.6×10⁻⁵` and scale-matched bias is effectively one.
   Its relatively large effective norm ratio (`0.569`) therefore reflects
   movement in a different direction, not useful agreement.
4. **Minimal BP suffixes replace the failing deep local updates with
   coherent task gradients.** HHB Enc3 BP-gradient SNR is `8.52`, while HBB
   Enc2/Enc3 SNRs are `6.37/5.15`. This provides a mechanism consistent with
   the Q2 rank repair: HHB repairs at z, whereas HBB starts repairing at h2.
5. **Weight normalization is a scale control, not the principal repair.**
   Raw and effective directions show the same depth-dependent loss of
   alignment. The main difference between pure Hebbian and Hybrid conditions
   is the presence of deep credit-assigned BP directions.

These are associations under the frozen MNIST protocol. They do not establish
that alignment or SNR alone causally determines accuracy, rank, or robustness.

## Exploratory cross-metric analysis

Across the 15 shared-prefix method/seed rows, mean effective alignment has
descriptive Spearman correlations of:

- `rho=0.425` with test accuracy;
- `rho=0.721` with z effective rank;
- `rho=−0.654` with Gaussian-0.4 accuracy degradation.

Mean effective SNR has corresponding correlations of `0.511`, `0.889`, and
`−0.746`. These values are exploratory only. The 15 rows contain duplicated
HHH/HHB/HBB shared-prefix measurements and therefore violate ordinary
independence assumptions; the p-values must not be treated as confirmatory.

## Q4 conclusion

Q4 is formally answerable for the frozen three-layer MNIST architecture.
Hebbian updates become progressively less aligned with the reconstruction BP
direction with depth. Enc1 retains a modest common direction, Enc2 is weak and
low-SNR, and Enc3 is effectively orthogonal. Hybrid credit assignment does not
change the paired Hebbian prefix; it replaces the poorly aligned deep local
updates with task-coherent BP gradients, matching the layer at which Q2
observes representation-rank repair.

This conclusion does not yet establish whether the same update pattern holds
at other bottleneck dimensions or encoder-width allocations. That question is
part of Q5/Q6.

## Artifacts

Local formal root:
`results/formal/phase0_v1_1/stage3_q4_updates/`

- `per_seed_layer_update_metrics.csv`;
- `method_layer_update_summary.csv`;
- `cross_metric_join.csv`;
- `exploratory_correlations.csv`;
- `integrity.json`, `run_manifest.json`;
- per-seed reference-decoder histories, batch metrics, raw/effective update
  tensors, and BP-gradient tensors;
- mechanism and BP-gradient-SNR figures in `figures/`.

Implementation validation:
`verification/phase0_v1_1/stage3_q4_implementation_junit.xml` (`120/120`
tests).
