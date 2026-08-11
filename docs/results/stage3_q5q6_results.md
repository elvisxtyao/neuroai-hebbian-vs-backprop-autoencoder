# Stage 3 Q5/Q6 Formal Dimension and Architecture Results

Date completed: 2026-08-09

Protocol: `configs/experiments/stage3_q5q6_sweeps_v1.yaml`

Training source snapshot: `a924932685fd634d0bd054c6171b66859c1c74a2`

Final analysis source snapshot: `4d09209` (after the two aggregation fixes)

## Outcome

**Stage 3 Q5/Q6 is complete.** The preregistered matrix contains
`BBB/HHH/HHB/HBB × seeds 0–4` for latent dimensions
`L=[16,32,64,128]` and encoder-width allocations early-heavy, balanced, and
late-heavy. L64/balanced reuses the accepted core checkpoints; the other five
cases add 100 complete system/probe/standardized-decoder records. Every new
case passed its technical freeze gate before its single frozen test
evaluation.

The main finding is not that one architecture is uniformly best. Added latent
capacity benefits BP and Hybrid representations, while Full Hebbian remains
near rank one even at L128. Early-heavy is stable; late-heavy exposes large
seed sensitivity, especially for HHB. A BP suffix repairs rank after a
Hebbian prefix in all width allocations on average, but it cannot guarantee
recovery after an unstable narrow early layer.

## Protocol and integrity

- Methods: BBB, HHH, HHB, HBB.
- Seeds: 0–4, paired within every case.
- BP learning rate: 0.003.
- Hebbian learning rate: 0.0005; winner fraction: 0.10.
- Selection: validation only; no performance, rank, or reconstruction gate.
- Reconstruction: both system and paired standardized-decoder MSE.
- Representation subset: fixed class-balanced 2,000-image test subset.
- Noise: identical sample-ID-keyed Gaussian, salt-and-pepper, and masking
  realizations at the frozen severities.
- Architecture encoder parameters: 104,512 / 105,104 / 104,712, with
  range/mean below 1%. Total autoencoder parameters differ because the decoder
  is not exactly parameter-matched; architecture reconstruction is therefore
  secondary and both decoder protocols are reported.

For each of L16, L32, L128, early-heavy, and late-heavy:

- freeze gate: PASS;
- 20/20 system runs, probes, and standardized decoders represented in the
  case gate;
- 20/20 one-time frozen test records complete;
- representation: 20 checkpoints and 60 layer rows, finite and unchanged;
- noise: 20 checkpoints and 260 condition rows, finite and unchanged;
- pre-freeze test access: zero.

The aggregate integrity gate records 140 performance rows, 420 representation
rows, 420 severity-0.4 noise rows, and 270 architecture update rows. All are
finite, and test results were not used for selection. Early-heavy and
late-heavy Q4 analyses each used five seeds, 50 fixed batches of 128 samples,
90 formal update rows, zero optimizer steps, zero test samples, and unchanged
source-checkpoint hashes.

The final full suite passed **138/138 tests** with zero failures or errors.
The saved log contains joblib's physical-core detection fallback and a related
Windows subprocess-decoding thread warning; neither changed test outcomes or
the already saved numerical results.

## Q5: dimensionality and architecture performance

### Latent dimension

Test results are mean ± sample SD across five paired seeds.

| L | Method | Accuracy | System MSE | Standardized MSE | z effective rank |
|---:|:---:|---:|---:|---:|---:|
| 16 | BBB | 0.8804 ± 0.0878 | 0.01833 | 0.01827 | 7.08 ± 3.29 |
| 16 | HHH | 0.6801 ± 0.0559 | 0.03752 | 0.03741 | 1.01 ± 0.00 |
| 16 | HHB | 0.7337 ± 0.3475 | 0.02974 | 0.03174 | 5.52 ± 3.24 |
| 16 | HBB | 0.8667 ± 0.1296 | 0.01897 | 0.01886 | 7.21 ± 3.32 |
| 32 | BBB | 0.9242 ± 0.0046 | 0.00857 | 0.00852 | 11.64 ± 2.93 |
| 32 | HHH | 0.8102 ± 0.0285 | 0.02779 | 0.02755 | 1.01 ± 0.00 |
| 32 | HHB | 0.9084 ± 0.0075 | 0.01015 | 0.00978 | 7.36 ± 0.76 |
| 32 | HBB | 0.9182 ± 0.0072 | 0.00965 | 0.00953 | 10.79 ± 3.27 |
| 64 | BBB | 0.9158 ± 0.0024 | 0.00320 | 0.00323 | 20.27 ± 2.75 |
| 64 | HHH | 0.8941 ± 0.0082 | 0.01850 | 0.01869 | 1.02 ± 0.00 |
| 64 | HHB | 0.9199 ± 0.0028 | 0.00465 | 0.00457 | 11.54 ± 0.77 |
| 64 | HBB | 0.9164 ± 0.0019 | 0.00327 | 0.00331 | 19.45 ± 2.51 |
| 128 | BBB | 0.9246 ± 0.0043 | 0.00217 | 0.00209 | 28.36 ± 0.92 |
| 128 | HHH | 0.9290 ± 0.0040 | 0.01447 | 0.01411 | 1.02 ± 0.00 |
| 128 | HHB | 0.9342 ± 0.0023 | 0.00367 | 0.00344 | 15.61 ± 0.30 |
| 128 | HBB | 0.9263 ± 0.0040 | 0.00227 | 0.00219 | 26.24 ± 1.34 |

Interpretation:

1. L16 is an unstable regime, not a reliable compression optimum. HHB seed 1
   reached only 0.1135 accuracy, while the other HHB seeds were 0.8485–0.9053.
   BBB and HBB also showed seed-specific failures.
2. From L32 upward, all BP-suffix methods are stable. HHB−HHH accuracy is
   +0.0982 at L32, +0.0258 at L64, and +0.0052 at L128. As capacity grows,
   HHH classification catches up, but its z covariance rank does not: it stays
   approximately one at every dimension.
3. Added latent dimensions therefore provide usable directions mainly when a
   BP-trained bottleneck is present. Dimensionality alone does not repair the
   Full-Hebbian representation geometry.
4. Both reconstruction measures improve with L. HHH remains substantially
   worse than BBB/HBB at L128, so high classification accuracy does not imply
   that the collapsed/anisotropic latent preserves equivalent reconstructive
   information.

The preregistered method × dimension interaction is not significant for
accuracy (`p=0.2629`) or either reconstruction metric (`p>0.88`) with five
seeds and the high-variance L16 outcomes. Representation interactions are much
clearer: z effective rank (`p=1.39e-16`) and h2 linear-probe accuracy
(`p=1.07e-13`) vary by method and dimension. The performance patterns should
therefore be reported with seed dispersion rather than as a universal ordering.

### Encoder-width asymmetry at L64

| Width allocation | Method | Accuracy | System MSE | Standardized MSE | z effective rank |
|:---|:---:|---:|---:|---:|---:|
| early-heavy [64,28,64] | BBB | 0.9188 ± 0.0013 | 0.00309 | 0.00499 | 16.86 ± 1.61 |
| early-heavy [64,28,64] | HHH | 0.8973 ± 0.0064 | 0.02727 | 0.02982 | 1.02 ± 0.00 |
| early-heavy [64,28,64] | HHB | 0.9208 ± 0.0009 | 0.00486 | 0.00565 | 9.96 ± 1.06 |
| early-heavy [64,28,64] | HBB | 0.9188 ± 0.0028 | 0.00347 | 0.00484 | 15.22 ± 2.46 |
| balanced [16,32,64] | BBB | 0.9158 ± 0.0024 | 0.00320 | 0.00323 | 20.27 ± 2.75 |
| balanced [16,32,64] | HHH | 0.8941 ± 0.0082 | 0.01850 | 0.01869 | 1.02 ± 0.00 |
| balanced [16,32,64] | HHB | 0.9199 ± 0.0028 | 0.00465 | 0.00457 | 11.54 ± 0.77 |
| balanced [16,32,64] | HBB | 0.9164 ± 0.0019 | 0.00327 | 0.00331 | 19.45 ± 2.51 |
| late-heavy [4,33,64] | BBB | 0.8606 ± 0.1208 | 0.01423 | 0.01437 | 15.53 ± 10.14 |
| late-heavy [4,33,64] | HHH | 0.8376 ± 0.0564 | 0.02430 | 0.02443 | 1.04 ± 0.02 |
| late-heavy [4,33,64] | HHB | 0.7950 ± 0.2555 | 0.01983 | 0.01969 | 11.16 ± 6.92 |
| late-heavy [4,33,64] | HBB | 0.8970 ± 0.0361 | 0.01202 | 0.01219 | 14.84 ± 8.50 |

Early-heavy and balanced are stable, with HHB retaining its rank-repair and
classification effect. Late-heavy is the stress case. Seed 4 produced BBB,
HHB, and HBB accuracies of 0.6446, 0.3380, and 0.8330, respectively, while
HHH reached 0.8898. This seed was retained exactly as required by the frozen
technical-only gate. HBB has the smallest architecture accuracy sensitivity
among the four methods (0.024); HHB has the largest (0.137).

The method × architecture accuracy interaction is not significant
(`p=0.8169`) because late-heavy has very large within-method seed variance.
Architecture is therefore demonstrated to create instability, but the current
five-seed test does not establish a population-level ordering of sensitivity.

### Noise robustness

At severity 0.4, dimension changes interact with method for Gaussian noise
(`p=0.00048`) and salt-and-pepper noise (`p=0.0392`), but not masking
(`p=0.1358`). More dimensions do not monotonically improve corruption
robustness: for example, HHB Gaussian accuracy falls from 0.492 at L64 to
0.198 at L128 despite better clean accuracy. Representation capacity and noise
stability are distinct outcomes.

Across architectures, the salt-and-pepper interaction is significant
(`p=0.0326`); Gaussian and masking interactions are not. Early-heavy HHB has
the best salt-and-pepper result among its four methods (0.376), whereas
late-heavy HHB falls to 0.224 and has large seed variability. HBB is the most
consistent Hybrid condition across the clean and noisy width comparisons.

### Learning cost

The fixed protocol exposes a deliberate sample-budget gradient. System
training sees 0.5M/1.0M/1.5M/2.0M samples for BBB/HBB/HHB/HHH; the paired
standardized decoder adds 0.5M samples to every method. At balanced L64, mean
system-plus-standardized wall times are 9.4, 11.4, 13.9, and 15.6 minutes,
respectively. Early-heavy is computationally slower (14.5–18.5 minutes total)
than late-heavy (5.6–10.1 minutes) even with matched encoder parameter counts,
because early channels operate on larger spatial maps. These times exclude
the probe and post-freeze analyses.

## Reconstruction fairness control

System and standardized-decoder reconstruction lead to the same broad
scientific conclusion: HHH loses more recoverable image information, while BP
suffixes repair much of the gap. Their differences are nevertheless material.
For early-heavy BBB, system MSE is 0.00309 but standardized MSE is 0.00499;
for early-heavy HHB they are 0.00486 and 0.00565. Thus system reconstruction
alone would partially mix encoder information with decoder/suffix adaptation.
All representation-level reconstruction claims use the standardized result;
system MSE is retained as the end-to-end system outcome.

## Q6: architecture asymmetry and hidden representations

The layerwise results localize both the persistent failure and the Hybrid
repair:

- HHH z effective rank remains 1.02, 1.02, and 1.04 for early, balanced, and
  late allocations. Moving channels does not repair the final local Hebbian
  layer.
- HHB raises z rank to 9.96, 11.54, and 11.16. Its mean z/h2 rank ratio is
  9.82, 11.05, and 5.48. BP Enc3 therefore repairs rank in every allocation,
  but the relative repair is weaker and much more variable in late-heavy.
- HBB raises z rank to 15.22, 19.45, and 14.84 and is more accurate/stable in
  late-heavy. Moving BP credit assignment one layer earlier is valuable when
  the shallow path is narrow.
- The shared Hebbian h2 competition regime changes strongly with width. HHB
  and HHH h2 winner entropy/coverage are 0.330/0.107 in early-heavy,
  0.423/0.344 in balanced, and 0.734/0.945 in late-heavy
  (`method × architecture p=1.99e-10` for entropy). Winner diversity alone is
  not sufficient for semantic quality: late-heavy has broader winning but is
  less stable.
- Cross-architecture CKA supports the same distinction. Early-heavy versus
  balanced CKA is at least 0.958 through z for all methods. Late-heavy z CKA
  falls to 0.828±0.220 (BBB), 0.789±0.215 (HHB), 0.863±0.095 (HBB), and
  0.865±0.258 (HHH), showing substantial seed-dependent geometry changes.

The frozen Q4 extension provides mechanism-consistent evidence. Effective
Hebbian/BP cosine alignment for Enc1/Enc2 is 0.2797/0.0206 in early-heavy and
0.7961/0.3443 in late-heavy; Enc3 remains approximately zero in both. Enc2
SNR rises from 0.0224 to 174.86 in late-heavy, yet HHH still ends at rank one.
Consequently, neither local-update alignment nor winner diversity alone
predicts the final representation. The failure is concentrated at the deep
all-local boundary, while BP suffix depth determines how much upstream loss
can be compensated.

This is an association from frozen snapshots, not a causal intervention on
alignment or SNR. The defensible answer to Q6 is that asymmetry changes the
competition regime, geometry, and stability of the Hebbian prefix, while the
location of BP credit assignment controls whether those changes are repaired
at h2 or only at z.

## Recovery and anomaly record

No completed formal run was overwritten or repeated. The resumable
orchestrator skipped accepted artifacts and quarantined partial outputs under
`results/.../stage3_q5q6_sweeps/_recovery/`. Three analysis-only issues were
fixed and regression-tested:

1. native sklearn PCA warnings were incorrectly treated as fatal by
   PowerShell;
2. asymmetric Q4 snapshots were initially loaded into the balanced model;
3. aggregate update layer names and heterogeneous CSV fields were not mapped
   correctly.

The first complete aggregate was also preserved before adding the planned
training-cost fields. These recovery directories are historical evidence, not
formal inputs. The accepted aggregate is the current `analysis/` directory.

## Answers and limits

**Q5:** Latent dimension strongly changes usable representation capacity and
stability. It does not cure HHH rank collapse. Width allocation can create
large seed instability even at matched encoder parameter count; HBB is the
most stable Hybrid response in the late-heavy stress case. Architecture and
dimension effects are metric-specific, especially under noise.

**Q6:** HHB is a reproducible rank-repair mechanism on average, but not a
complete or uniformly stable repair. HBB moves credit assignment earlier and
better protects against late-heavy failures. The deep Full-Hebbian layer is
the persistent rank bottleneck across all tested widths.

These conclusions are restricted to MNIST, the frozen three-layer encoder,
the selected Oja/WTA rule, five paired seeds, and clean-trained models.
Matched RBB/RRB controls exist only at the preregistered key L64/balanced
reference. The sweeps compare learning systems across dimensions and widths,
but do not support new claims that Hebbian prefixes add value at every
non-reference configuration. CIFAR-10, non-stationary learning, noisy
training, adversarial robustness, and alternate Hebbian rules remain Phase 9
extensions rather than missing Q5/Q6 acceptance items.

## Evidence paths

- Aggregate integrity and tables:
  `results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/`
- Case freeze/test/representation/noise artifacts:
  `results/formal/phase0_v1_1/stage3_q5q6_sweeps/{dimension,architecture}/`
- Final immutable test records:
  `verification/phase0_v1_1/stage3_final_junit.xml` and
  `verification/phase0_v1_1/stage3_final_pytest.log`
- Recovery records:
  `results/recovery/` and
  `results/formal/phase0_v1_1/stage3_q5q6_sweeps/_recovery/`
