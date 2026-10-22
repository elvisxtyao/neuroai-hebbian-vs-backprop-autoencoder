# Results Summary

## Research Question

Where does local Hebbian learning remain useful in a hierarchical representation-learning system, and at what depth does global credit assignment become necessary?

The answer is specific to the frozen study: in a three-layer MNIST convolutional autoencoder using a competitive Oja/WTA Hebbian rule, a shallow learned Hebbian prefix adds a small amount of classification value, but stacking the rule through the encoder compresses the deepest representation to nearly one effective dimension. BP restores representation dimensionality where it is introduced, with earlier intervention providing stronger reconstruction and corruption-robustness outcomes.

## Experimental Matrix

| Method | Enc1 | Enc2 | Enc3 | Role |
|---|---|---|---|---|
| BBB | BP | BP | BP | Full-BP reference |
| HBB | Hebbian | BP | BP | Hybrid; BP begins at Enc2 |
| HHB | Hebbian | Hebbian | BP | Hybrid; BP begins at Enc3 |
| HHH | Hebbian | Hebbian | Hebbian | Full competitive Hebbian encoder |
| RBB | Random | BP | BP | HBB prefix-matched control |
| RRB | Random | Random | BP | HHB prefix-matched control |
| Random | Random | Random | Random | Overall random-encoder lower bound |

The common task is MNIST with a frozen 50,000/10,000/10,000 train/validation/test split. Formal inference uses five paired seeds `[0, 1, 2, 3, 4]`, with seed as the statistical unit. Frozen confidence intervals use 10,000 nonparametric bootstrap resamples, bootstrap seed 2026, and 95% coverage. HHB and HBB are hybrid models, not pure Hebbian models.

## Q1 Performance

### Headline outcomes

| Method | Test accuracy, mean ± SD | System reconstruction MSE, mean | Standardized decoder MSE, mean |
|---|---:|---:|---:|
| BBB | 0.91584 ± 0.00238 | 0.003201 | 0.003234 |
| HBB | 0.91638 ± 0.00186 | 0.003270 | 0.003312 |
| HHB | 0.91986 ± 0.00283 | 0.004647 | 0.004569 |
| HHH | 0.89410 ± 0.00817 | 0.018500 | 0.018689 |
| Random | 0.80804 ± 0.02729 | 0.019446 | 0.019446 |
| RBB | 0.91528 | — | 0.003302 |
| RRB | 0.91640 | — | 0.004559 |

### Primary paired contrasts

| Contrast | Accuracy difference (95% CI) | Standardized-MSE difference (95% CI) | Interpretation |
|---|---:|---:|---|
| HHB − HHH | +0.02576 (0.01664, 0.03328) | −0.014120 (−0.015455, −0.012754) | BP at Enc3 repairs classification and much of reconstruction |
| HBB − HHB | −0.00348 (−0.00540, −0.00104) | −0.001256 (−0.001510, −0.000969) | Earlier BP improves reconstruction but not mean accuracy |
| BBB − HHB | −0.00402 (−0.00816, −0.00106) | −0.001334 (−0.001504, −0.001132) | HHB's reconstruction repair remains incomplete |

### Protocol-mandated secondary contrasts

HBB exceeds RBB in accuracy by 0.00110 (95% CI 0.00036–0.00180), and HHB exceeds RRB by 0.00346 (95% CI 0.00204–0.00554). Their standardized-MSE differences are both approximately +0.000010 with intervals crossing zero. Learned Hebbian prefixes therefore add small, reproducible classification value, without evidence of incremental standardized-reconstruction value.

Observed system training exposure is 500,000 samples for BBB, 1,000,000 for HBB, 1,500,000 for HHB, and 2,000,000 for HHH. Mean wall-clock times are approximately 313, 436, 579, and 695 seconds, respectively. These are training-cost summaries, not sample-efficiency claims.

## Q2 Representation

### Layerwise mean effective rank / frozen linear probe accuracy

| Method | h1 | h2 | z |
|---|---:|---:|---:|
| BBB | 1.882 / 0.9004 | 4.310 / 0.9190 | 20.269 / 0.8605 |
| HBB | 1.361 / 0.8919 | 4.171 / 0.9138 | 19.450 / 0.8634 |
| HHB | 1.361 / 0.8919 | 1.045 / 0.9001 | 11.541 / 0.8778 |
| HHH | 1.361 / 0.8919 | 1.045 / 0.9001 | 1.016 / 0.8460 |

HHB raises z effective rank over HHH by 10.525 (95% CI 9.845–11.075) and z linear-probe accuracy by 0.0318 (95% CI 0.0224–0.0428). HBB raises h2 rank over HHB by 3.126 and z rank by 7.908. The representation change is localized: HHH–HHB CKA is 1.0 at h1 and h2, then 0.672 at z.

Effective rank is a geometry/diversity measure, not semantic information. HHH remains decodable despite rank collapse, and HHB has higher z linear-probe accuracy than HBB despite lower z rank.

## Q3 Robustness

### Mean test accuracy at corruption severity 0.4

| Method | Gaussian | Salt-and-pepper | Masking |
|---|---:|---:|---:|
| BBB | 0.59344 | 0.35620 | 0.84526 |
| HBB | 0.67516 | 0.40156 | 0.84752 |
| HHB | 0.49168 | 0.35302 | 0.82708 |
| HHH | 0.37902 | 0.28472 | 0.80230 |

Relative to HHB, HBB has less clean-to-severity-0.4 degradation under Gaussian noise by 0.1870 (95% CI 0.1180–0.2487) and masking by 0.0239 (95% CI 0.0090–0.0373). The salt-and-pepper difference is 0.0520 with an interval crossing zero. Whole-curve AUC contrasts are supplementary/exploratory rather than primary. HHH's near-one noisy z cosine is a collapsed-direction artifact, not semantic robustness.

## Q4 Update Mechanism

### HHH effective Hebbian update versus matched BP reconstruction direction

| Layer | Cosine alignment, mean ± SD | Scale-matched bias | Norm ratio | Linear SNR |
|---|---:|---:|---:|---:|
| Enc1 | 0.3680 ± 0.1265 | 0.9214 | 0.04547 | 453.09 |
| Enc2 | 0.03563 ± 0.04110 | 0.998687 | 0.002242 | 0.01806 |
| Enc3 | 0.0000459 ± 0.0002756 | 0.99999997 | 0.5687 | 1.293 |

Alignment deteriorates sharply with depth and is approximately zero at Enc3. The same depths show the strongest HHH compression in Q2. This is mechanistic association, not strict causal proof; the matched BP reconstruction direction is a defined comparator rather than a universally optimal update. Shared HBB/HHB prefix rows duplicate accepted local updates and are not independent observations. Cross-metric correlation analyses remain exploratory.

## Q5 Dimension and Architecture

### Latent-dimension sweep

| Dimension | BBB accuracy / z rank | HBB accuracy / z rank | HHB accuracy / z rank | HHH accuracy / z rank |
|---:|---:|---:|---:|---:|
| 16 | 0.8804 / 7.08 | 0.8667 / 7.21 | 0.7337 / 5.52 | 0.6801 / 1.01 |
| 32 | 0.9242 / 11.64 | 0.9182 / 10.79 | 0.9084 / 7.36 | 0.8102 / 1.01 |
| 64 | 0.9158 / 20.27 | 0.9164 / 19.45 | 0.9199 / 11.54 | 0.8941 / 1.02 |
| 128 | 0.9246 / 28.36 | 0.9263 / 26.24 | 0.9342 / 15.61 | 0.9290 / 1.02 |

HHH classification improves with nominal dimension while z effective rank stays near one. The method×dimension interaction is non-significant for accuracy (*p*=0.2629) but significant for z effective rank (*p*=1.39×10⁻¹⁶). Dimension therefore changes performance without removing the full-Hebbian compression pattern.

### Architecture sweep at latent dimension 64

| Architecture | BBB accuracy / z rank | HBB accuracy / z rank | HHB accuracy / z rank | HHH accuracy / z rank |
|---|---:|---:|---:|---:|
| Early-heavy | 0.9188 / 16.86 | 0.9188 / 15.22 | 0.9208 / 9.96 | 0.8973 / 1.02 |
| Balanced | 0.9158 / 20.27 | 0.9164 / 19.45 | 0.9199 / 11.54 | 0.8941 / 1.02 |
| Late-heavy | 0.8606 / 15.53 | 0.8970 / 14.84 | 0.7950 / 11.16 | 0.8376 / 1.04 |

The retained late-heavy seed 4 accuracies are 0.6446 (BBB), 0.8330 (HBB), 0.3380 (HHB), and 0.8898 (HHH). HBB is descriptively least sensitive across architectures (0.024), and HHB most sensitive (0.137), but the method×architecture accuracy interaction is non-significant (*p*=0.8169). The stability statement is descriptive, not a global significance claim.

## Q6 Layerwise Compensation

The mean effective-rank compensation ratio, ER(z)/ER(h2), is 4.741 for BBB, 4.767 for HBB, 11.048 for HHB, and 0.972 for HHH. BP at Enc3 can therefore expand a low-rank h2 into a substantially broader z representation, while BP beginning at Enc2 repairs h2 before the final expansion. Across the dimension and architecture sweeps, HHB generally repairs z rank without fully matching the stability or reconstruction profile of HBB. Compensation restores geometry where BP is introduced; it does not imply recovery of all information or a complete reconstruction repair.

## Main Takeaways

1. A shallow learned competitive Hebbian prefix contributes a small but reproducible classification benefit over matched random prefixes, while the fully local stack retains class information but reconstructs poorly.
2. Deep stacking compresses HHH h2 and z toward one effective dimension; BP at Enc3 repairs z, and BP from Enc2 repairs both h2 and z, showing a depth-dependent credit-assignment boundary in this tested system.
3. Earlier BP intervention improves reconstruction, corruption robustness, and descriptive architecture stability, while rank, class information, and winner diversity remain distinct measurements rather than interchangeable definitions of representation quality.
