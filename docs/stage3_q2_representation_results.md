# Stage 3 Q2 Formal Layerwise Representation Results

Date completed: 2026-07-29

Analysis source: `e488a056434f007745cfc0cc7efd96f0be6015ae`

Protocol: `configs/experiments/stage3_q2_representation_v1.yaml`

## Scope and integrity

The formal analysis covers `BBB/HHH/HHB/HBB/RBB/RRB × seeds 0–4 ×
h1/h2/z`: 30 frozen checkpoints and 90 method/seed/layer records. Full Random
is excluded because Q2 is about trained Full/Hybrid representations and the
matched random-prefix controls.

All conditions use the identical seed-17 subset of 2,000 official MNIST test
images, with 200 images per class and an identical sample-ID order. This subset
was accessed only after the Stage 3 technical freeze and one-time clean
performance evaluation were complete. It was not used for training, checkpoint
selection, hyperparameter selection, or exclusion of a method.

The integrity gate reports:

- `30/30` representation records;
- `90/90` method/seed/layer metric rows;
- checkpoint state checksums unchanged before/after every extraction;
- all arrays finite;
- identical sample-ID and label hashes across all conditions;
- 30 compressed `input/h1/h2/z` archives and metadata records;
- 90 PCA embeddings, 18 fixed seed-0 UMAP embeddings, and 14 figures.

## Metric views

Two axis-explicit views are kept separate:

1. **Channel-health view:** `(sample × spatial location) × channel`. Exact
   feature-covariance spectra, participation-ratio effective rank, stable rank,
   active units, sparsity, and winner metrics use this view.
2. **Sample-geometry view:** `sample × flattened(channel × spatial)`. Unified
   five-fold linear probes and k-NN use the raw flattened representation.
   Fixed PCA-50 is used for class covariance, silhouette, CKA, PCA plots, and
   as the input to the deterministic exact-neighbor UMAP visualization.

The UMAP implementation uses exact kNN, smooth-kNN fuzzy-set union, spectral
initialization, and fixed-seed cross-entropy optimization. Its frozen parameters
are `n_neighbors=15`, `min_dist=0.1`, `spread=1`, `epochs=200`,
`negative_samples=5`, and `seed=17`. PCA remains the primary deterministic
visualization; UMAP is auxiliary.

## Five-seed layerwise results

Values are mean ± sample SD.

| Method | Layer | Effective rank | Linear probe | k-NN | Separability |
|---|---|---:|---:|---:|---:|
| BBB | h1 | 1.882 ± 0.262 | 0.9004 ± 0.0050 | 0.8428 | 0.1100 |
| BBB | h2 | 4.310 ± 0.423 | **0.9190 ± 0.0051** | 0.8748 | 0.1210 |
| BBB | z | **20.269 ± 2.754** | 0.8605 ± 0.0040 | **0.9242** | 0.1097 |
| HHH | h1 | 1.361 ± 0.078 | 0.8919 ± 0.0048 | 0.8070 | 0.1088 |
| HHH | h2 | 1.045 ± 0.014 | 0.9001 ± 0.0036 | 0.8524 | 0.1106 |
| HHH | z | 1.016 ± 0.002 | 0.8460 ± 0.0143 | 0.8242 | 0.1048 |
| HHB | h1 | 1.361 ± 0.078 | 0.8919 ± 0.0048 | 0.8070 | 0.1088 |
| HHB | h2 | 1.045 ± 0.014 | 0.9001 ± 0.0036 | 0.8524 | 0.1106 |
| HHB | z | 11.541 ± 0.771 | **0.8778 ± 0.0057** | 0.9183 | 0.1095 |
| HBB | h1 | 1.361 ± 0.078 | 0.8919 ± 0.0048 | 0.8070 | 0.1088 |
| HBB | h2 | 4.171 ± 0.618 | 0.9138 ± 0.0027 | **0.8763** | 0.1194 |
| HBB | z | 19.450 ± 2.506 | 0.8634 ± 0.0105 | 0.9233 | 0.1092 |
| RBB | h1 | 1.893 ± 0.503 | 0.8908 ± 0.0046 | 0.7998 | 0.1089 |
| RBB | h2 | 4.180 ± 0.400 | 0.9161 ± 0.0062 | 0.8705 | 0.1195 |
| RBB | z | 19.513 ± 2.699 | 0.8649 ± 0.0079 | 0.9201 | 0.1096 |
| RRB | h1 | 1.893 ± 0.503 | 0.8908 ± 0.0046 | 0.7998 | 0.1089 |
| RRB | h2 | 3.424 ± 0.549 | 0.8951 ± 0.0042 | 0.8329 | 0.1110 |
| RRB | z | 17.901 ± 3.209 | 0.8672 ± 0.0093 | 0.9099 | 0.1086 |

Separability is the between-class/within-class scatter ratio in the fixed
PCA-50 geometry. Negative silhouette values for all methods show that the ten
classes are not globally separated as compact Euclidean clusters; this does
not contradict their linear or local-neighbor decodability.

## Paired mechanism contrasts

Bootstrap intervals use the five seed-level paired differences, 10,000
resamples, and seed 2026.

| Contrast and layer | Metric | Mean difference | Paired 95% CI | Cohen's dz |
|---|---|---:|---:|---:|
| HHB − HHH, z | effective rank | +10.525 | [+9.845, +11.075] | +13.63 |
| HHB − HHH, z | linear probe | +0.0318 | [+0.0224, +0.0428] | +2.41 |
| HHB − HHH, z | k-NN | +0.0941 | [+0.0821, +0.1061] | +6.09 |
| HBB − HHB, h2 | effective rank | +3.126 | [+2.659, +3.586] | +5.00 |
| HBB − HHB, h2 | linear probe | +0.0137 | [+0.0096, +0.0186] | +2.42 |
| HBB − HHB, z | effective rank | +7.908 | [+6.441, +9.376] | +4.17 |
| HBB − HHB, z | linear probe | −0.0144 | [−0.0250, −0.0070] | −1.25 |
| HBB − RBB, h1 | effective rank | −0.532 | [−0.919, −0.280] | −1.23 |
| HBB − RBB, h1 | k-NN | +0.0072 | [+0.0019, +0.0133] | +1.00 |
| HHB − RRB, h2 | effective rank | −2.379 | [−2.807, −1.950] | −4.27 |
| HHB − RRB, h2 | linear probe | +0.0050 | [+0.0018, +0.0082] | +1.21 |
| HHB − RRB, h2 | k-NN | +0.0195 | [+0.0123, +0.0270] | +2.03 |

## Compensation and CKA

The mean channel-rank ratios `ER(z)/ER(h2)` are:

| Method | ER(z)/ER(h2) | ER(z) − ER(h2) | z − h2 linear probe |
|---|---:|---:|---:|
| BBB | 4.741 | +15.959 | −0.0585 |
| HHH | 0.972 | −0.029 | −0.0541 |
| HHB | **11.048** | +10.496 | −0.0223 |
| HBB | 4.767 | +15.279 | −0.0504 |
| RBB | 4.686 | +15.333 | −0.0512 |
| RRB | 5.224 | +14.477 | −0.0279 |

HHB therefore performs the strongest explicit low-rank-to-higher-rank
transformation at Enc3, although its z rank remains below BBB/HBB.

Mean paired-seed PCA-50 linear CKA further localizes the intervention:

- HHH–HHB CKA is `1.000` at h1 and h2, then `0.672` at z;
- HHB–HBB is `1.000` at h1, `0.865` at h2, and `0.916` at z;
- BBB–HBB is `0.993` at h1, `0.994` at h2, and `0.970` at z;
- HBB–RBB is `0.987`, `0.995`, and `0.971` across h1/h2/z.

These values agree with the frozen/shared-prefix design and show that the main
HHB geometry change occurs exactly at BP Enc3.

## Q2 conclusions

1. **HHH exhibits stable, cumulative channel-rank compression.** Its h2 and z
   ranks remain near one across all five seeds. The seed-42 failure was not an
   isolated anomaly.
2. **Minimal BP Enc3 credit assignment repairs the bottleneck geometry.** HHB
   leaves the low-rank Hebbian h1/h2 exactly unchanged, but raises z effective
   rank by 10.53, z linear-probe accuracy by 3.18 points, and z k-NN accuracy by
   9.41 points relative to HHH.
3. **BP Enc2 adds an earlier repair.** HBB raises h2 rank and decoding, and its
   z rank becomes close to BBB. However, the higher rank does not automatically
   yield higher linear decoding: HBB z linear-probe accuracy is lower than HHB
   on this 2,000-image cross-validation analysis.
4. **Rank is diversity, not class information.** Hebbian h1/h2 have lower rank
   than matched random prefixes, yet show small positive k-NN/linear-decoding
   differences. This is consistent with Q1's small matched-prefix
   classification benefit and prevents interpreting rank alone as task value.
5. **The Hybrid mechanism is localized, not a full Hebbian recovery.** HHB is a
   BP transformation of a compressed Hebbian prefix. It should be described as
   minimal Hybrid rank repair, not as evidence that pure Hebbian depth is
   healthy.

## Artifacts

Local formal root:
`results/formal/phase0_v1_1/stage3_q2_representation/`

- `integrity.json`, `run_manifest.json`;
- `per_seed_layer_metrics.csv`, `method_layer_summary.csv`;
- `compensation_metrics.csv`, `layerwise_cka.csv`;
- `representations/`: 30 compressed raw representation archives plus metadata;
- `embeddings/`: 90 PCA and 18 seed-0 UMAP artifacts;
- `figures/`: rank, probes, separability, PCA, UMAP, CKA, and confusion plots.

Implementation validation:
`verification/phase0_v1_1/stage3_q2_implementation_junit.xml` (`109/109`
tests).

