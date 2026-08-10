# Final figure captions

## Figure 1 - Performance and reconstruction tradeoff

MNIST held-out test classification and reconstruction outcomes for BBB, HBB, HHB, HHH, and Random. Open markers are the five formal seeds (0-4; statistical unit: seed); filled markers are method means; whiskers are the frozen 95% nonparametric bootstrap confidence intervals (10,000 resamples, bootstrap seed 2026). Reconstruction errors are shown on logarithmic axes for the common standardized BP decoder (B) and each method's system decoder (C). HHB has the highest mean accuracy (0.9199, 95% CI 0.9177-0.9222), whereas HHH has lower mean accuracy and substantially higher reconstruction error. Small accuracy differences should be read together with their seed-level dispersion and intervals.

## Figure 2 - Prefix value and training cost

(A) Paired seed-level MNIST held-out test accuracy effects of learned Hebbian prefixes relative to depth-matched random-prefix controls. HBB-RBB has a mean difference of 0.0011 (95% CI 0.00036-0.00180); HHB-RRB has a mean difference of 0.00346 (95% CI 0.00204-0.00554). Whiskers are the frozen paired-seed bootstrap intervals (five paired seeds, statistical unit: paired seed, 10,000 resamples, bootstrap seed 2026). (B) Validation reconstruction MSE at every observed samples-seen checkpoint for BBB, HBB, HHB, and HHH. Thin lines are individual seeds; heavy lines are means; bands are frozen 95% intervals. All 200 plotted points are direct observations from the complete 350-row audit table; no trajectory is interpolated. (C) System training samples and wall-clock cost for the same methods. Open markers are seeds and filled markers are means. Wall-clock is descriptive and hardware-specific; training cost is not labeled sample efficiency.

## Figure 3 - Layerwise representation structure

Effective rank (A) and frozen linear-probe cross-validation accuracy (B) across h1, h2, and z for BBB, HBB, HHB, and HHH. Faint lines and points are the five formal seeds (statistical unit: seed); heavy lines are method means; whiskers are +/- one sample standard deviation as stored in the frozen Q2 summary. At z, mean effective rank is 20.27 for BBB, 19.45 for HBB, 11.54 for HHB, and 1.016 for HHH. The rank collapse of fully local HHH is therefore visible even though its frozen-probe accuracy remains above chance. Effective rank is a diversity/geometry metric, not a direct measure of semantic information. The linear probe is an evaluation instrument on frozen representations, not end-to-end model selection.

## Figure 4 - Depth-dependent local-update mechanism

Canonical effective Hebbian updates along the HHH encoder path, compared at each layer with the same-layer BP direction induced by the matched reconstruction objective and snapshot. Panels show cosine alignment (A), scale-matched bias (B), norm ratio to matched BP (C), and linear update SNR (D). Open markers are the five formal seeds (statistical unit: seed); filled markers are means; whiskers are +/- one sample standard deviation from the frozen Q4 summary. Alignment falls from 0.368 at Enc1 to 0.0356 at Enc2 and approximately zero at Enc3, while scale-matched bias approaches one. The matched BP reconstruction direction is a defined comparator, not a universally optimal or ground-truth learning direction. Shared HBB/HHB prefix rows are deliberately omitted because they duplicate the same accepted local updates rather than providing independent observations.

## Figure 5 - Corruption robustness

MNIST held-out test accuracy across Gaussian noise (A), salt-and-pepper noise (B), and pixel masking (C) for the four core methods. Faint lines are individual formal seeds (statistical unit: seed); heavy lines are means; bands are frozen 95% bootstrap confidence intervals (10,000 resamples, bootstrap seed 2026). The clean condition is reused only as the directly observed severity-zero anchor for each corruption family. HBB is strongest at severity 0.4 under Gaussian noise (mean 0.675) and salt-and-pepper noise (0.402), whereas masking produces smaller separations. These are protocol-mandated severity-wise summaries; exploratory aggregate AUC contrasts are not plotted or promoted to primary evidence here. Representation cosine stability is not interpreted as semantic robustness.

## Figure 6 - Dimension and architecture variability

MNIST held-out test classification accuracy (A, B) and z-layer effective rank (C, D) across latent dimensions 16/32/64/128 and early-heavy/balanced/late-heavy architecture allocations. Faint trajectories show the five formal seeds (statistical unit: seed); heavy lines and whiskers show means and frozen 95% bootstrap confidence intervals. Filled seed markers identify seed 4, including the retained late-heavy outlier runs. HHH remains near rank one across dimensions and architectures. Late-heavy performance and z-rank variability are shown without replacing unusual runs. The panels are descriptive sensitivity analyses and do not establish a formal dimension-by-method or architecture-by-method interaction.

## Method design schematic (not a formal hero figure)

Learning-rule allocation for BBB, HBB, HHB, HHH, RBB, and RRB. Each method uses the shared BP decoder and frozen-probe evaluation path; only the encoder layer rules differ. The schematic is a design aid, contains no measured values, and is not counted among the six formal figures.
