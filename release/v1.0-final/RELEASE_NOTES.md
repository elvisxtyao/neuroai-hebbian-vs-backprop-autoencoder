# v1.0-final

## Study Scope

This release freezes a five-seed study of where local competitive Hebbian learning remains useful in a three-layer convolutional autoencoder and where global BP credit assignment becomes necessary. The formal comparison covers BBB, HBB, HHB, and HHH, with RBB/RRB matched random-prefix controls and a fully random encoder lower bound. Claims are limited to MNIST, the frozen architecture family, and the competitive Oja/WTA protocol.

## Main Findings

- Full-Hebbian HHH retains substantial classification information (mean test accuracy 0.8941) but compresses z to mean effective rank 1.016 and has high standardized reconstruction MSE (0.018689).
- Hybrid HHB introduces BP at Enc3, raising z effective rank to 11.541 and mean test accuracy to 0.9199. This is a rank and classification repair, not a complete reconstruction repair.
- Hybrid HBB introduces BP at Enc2, raising z effective rank to 19.450, close to BBB's 20.269, while improving reconstruction relative to HHB and showing formal degradation advantages under Gaussian noise and masking.
- Learned Hebbian prefixes add small paired classification gains over matched random prefixes: +0.00110 for HBB–RBB and +0.00346 for HHB–RRB. Neither matched contrast establishes a standardized-reconstruction advantage.
- Hebbian alignment with the matched BP reconstruction direction decreases with depth: 0.368 at Enc1, 0.0356 at Enc2, and approximately zero at Enc3. This is mechanistic association rather than strict causal proof.
- Latent dimension and channel allocation change performance and variability without removing HHH's near-rank-one z pattern. The method×architecture accuracy interaction remains non-significant.

## Included Evidence

The compact evidence bundle contains accepted CSV/JSON tables for Q1–Q6, the final audit supplement, protocol/statistical audit records, the Stage 2D governance decision, an artifact registry, a manifest with logical source paths, and SHA-256 checksums. It excludes checkpoints, representation archives, update tensors, raw logs, recovery outputs, partial runs, and exploratory results.

Six publication figures and a non-scientific method-design schematic are stored in `figures/final/`. Figure provenance includes frozen source hashes, plotted-value exports, and a builder hash. The release notes are presentation metadata and intentionally remain outside the frozen evidence checksum set.

## Reproducibility

The compact bundle can be verified without loading MNIST, checkpoints, or model code:

```bash
python scripts/verify_final_release.py
```

The final figures can be regenerated from compact frozen tables only:

```bash
python scripts/plot_final_figures.py --output figures/rebuilt-v1.0-final
```

Detailed clean-export instructions are provided in `REPRODUCIBILITY.md`, and the executed artifact-only walkthrough is available as `project_demo.ipynb`. No scientific retraining or model evaluation is required to verify the compact evidence, rebuild the figures, or execute the demo.

## Limitations

- MNIST only; one competitive Oja/WTA-style Hebbian rule; one three-layer ConvAE family.
- Five paired formal seeds and limited latent-dimension/channel-allocation sweeps.
- Corruption evaluation is not adversarial robustness and does not study noisy training.
- Effective rank is not semantic information, and winner diversity alone is not representation quality.
- Q4 update alignment is an objective-specific association, not direct causal evidence or a universal endorsement of the BP direction.
- HHB's larger architecture variability is descriptive; the overall method×architecture accuracy interaction is non-significant.

## Governance

Stage 2D ended as **CONFIRMATION FAILED** because seed 43 failed the preregistered standardized-decoder reconstruction criterion (HHB/BBB MSE ratio 1.5688 versus the required maximum 1.25). Stage 2D nevertheless supported classification stability and z-rank repair. Its threshold and outcome were not retrospectively changed.

Stage 3 was a later, independently documented post-confirmation re-scoping. HHB was retained as a rank-repair and minimal-credit-assignment condition, while standardized reconstruction became an outcome rather than an eligibility assumption. Stage 2D seeds 43 and 44 are confirmation evidence and are excluded from the final formal seed set `[0, 1, 2, 3, 4]`.
