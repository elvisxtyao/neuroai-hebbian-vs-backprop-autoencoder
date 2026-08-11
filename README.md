# Where Does Hebbian Learning Help?

This repository studies a focused credit-assignment question: **how far can local Hebbian learning be stacked into a hierarchical encoder before global backpropagation becomes necessary?** In a three-layer convolutional autoencoder on MNIST, it compares full BP, a competitive Oja/WTA Hebbian rule, and two hybrid allocations that introduce BP at different encoder depths. The formal release uses five paired seeds and separates classification, reconstruction, representation geometry, robustness, and update mechanism.

## Key Finding

Under this tested architecture and rule, shallow Hebbian learning remains useful, but a fully local encoder compresses its deepest representation toward one effective dimension. Mean z effective rank is **1.016 for HHH**, rises to **11.541 when BP is introduced at Enc3 (HHB)**, and reaches **19.450 when BP begins at Enc2 (HBB)**, close to **20.269 for full BP (BBB)**. HHH still achieves 0.8941 mean test accuracy, while HHB reaches 0.9199, showing that effective dimensionality and decodable class information are not the same quantity.

## Experimental Design

![Learning-rule allocation for the core methods and matched controls.](figures/final/method_design.png)

- **BBB:** BP in Enc1, Enc2, and Enc3.
- **HBB:** competitive Hebbian Enc1; BP Enc2 and Enc3.
- **HHB:** competitive Hebbian Enc1 and Enc2; BP Enc3.
- **HHH:** competitive Hebbian learning throughout the encoder.
- **RBB/RRB:** frozen random-prefix controls matched to HBB/HHB depth.
- **Random:** fully random encoder, used only as an overall lower bound.

All methods use a BP decoder and a frozen linear probe evaluation path. HHB and HBB are hybrid models, not pure Hebbian models. Formal inference uses five paired seeds `[0, 1, 2, 3, 4]`; seed is the statistical unit, and uncertainty is reported with frozen 95% bootstrap confidence intervals.

## Main Results

### Deep local stacking compresses representation geometry

![Layerwise effective rank and frozen linear probe performance.](figures/final/fig3_layerwise_representation.png)

HHH approaches rank one at h2 and z, even though class information remains linearly accessible. BP at Enc3 expands HHB's z representation; BP from Enc2 repairs both h2 and z. This is the study's central result: representation dimensionality and semantic decodability are related but distinct.

### Classification alone hides reconstruction deficits

![Classification and reconstruction outcomes.](figures/final/fig1_performance.png)

HHB improves mean accuracy over HHH by 0.02576 (95% CI 0.01664–0.03328) and greatly reduces standardized decoder error. The repair is incomplete: BBB standardized MSE is 0.003234 versus 0.004569 for HHB and 0.018689 for HHH. Figure 1A uses a zoomed 0.76–0.93 accuracy axis; seed-level points and intervals should guide interpretation.

### Local-update alignment deteriorates with depth

![Hebbian updates compared with the matched BP reconstruction direction.](figures/final/fig4_update_mechanism.png)

Mean Hebbian/BP cosine alignment decreases from 0.368 at Enc1 to 0.0356 at Enc2 and approximately zero at Enc3. This depth trend is associated with the rank pattern, but it is not strict causal proof, and the matched BP reconstruction direction is not assumed to be universally optimal.

## Why This Matters

The project treats encoder depth as a controlled learning-rule ablation rather than asking only whether BP beats Hebbian learning. Matched random-prefix controls show that learned Hebbian prefixes add small, reproducible classification value. Layerwise rank, frozen decoding, corruption tests, and update geometry then reveal why that value does not justify stacking the same local objective without deeper credit assignment.

The result is intentionally scoped to MNIST, this three-layer convolutional autoencoder, and this competitive Oja/WTA protocol. It does not claim that Hebbian learning generally collapses.

## Reproducibility

The repository separates three reproducibility tiers:

1. **Compact artifact verification:** `release/v1.0-final/` contains accepted CSV/JSON evidence, a registry, relative provenance, and SHA-256 checksums. It can be verified without datasets or checkpoints using `python scripts/verify_final_release.py`.
2. **Figure regeneration:** `scripts/plot_final_figures.py --output figures/rebuilt-v1.0-final` rebuilds the final figures only from the compact frozen tables into a new directory. It does not train or evaluate a model.
3. **Full research archive:** protocols, configurations, training/evaluation code, and frozen formal outputs retain the original research lineage. Large checkpoints and raw arrays are intentionally excluded from the compact release bundle.

See the [reproducibility guide](REPRODUCIBILITY.md) for exact commands and the executed [artifact-only project demo](project_demo.ipynb) for a 3–5 minute walkthrough.

## Repository Structure

```text
ae/, learning_rules/, models/   Model and learning-rule implementation
training/, evaluation/          Training and evaluation infrastructure
configs/                        Frozen experiment configurations
results/                        Ignored local full archive; not distributed in the compact GitHub release
release/v1.0-final/             Compact accepted evidence and integrity metadata
figures/final/                  Publication figures and plotted-value provenance
docs/final/                     Figure audit and full captions
docs/                           Protocols, amendments, and frozen result reports
scripts/                        Release verification and artifact-only plotting
tests/                          Unit, release-bundle, and figure validation tests
```

## Documentation

- [Final research report](FINAL_REPORT.md)
- [Compact numerical results](RESULTS_SUMMARY.md)
- [Reproducibility guide](REPRODUCIBILITY.md) and [artifact-only project demo](project_demo.ipynb)
- [Figure captions](docs/final/FIGURE_CAPTIONS.md) and [figure audit](docs/final/FIGURE_AUDIT.md)
- [Compact evidence manifest](release/v1.0-final/manifest.json) and [artifact registry](release/v1.0-final/artifact_registry.csv)
- [Stage 3 formal protocol](docs/stage3_formal_protocol_v1.md) and [final statistical audit](docs/final_statistical_protocol_audit.md)
- [Project status](PROJECT_STATUS.md) and [historical research plan](HEBBIAN_PROJECT_PLAN.md)

## Scope and Limitations

The study uses MNIST, one local competitive rule, one three-layer ConvAE family, five formal seeds, and bounded latent/channel sweeps. Corruption tests are neither adversarial robustness nor noisy training. Update alignment is mechanistic association, not causal identification. The architecture interaction for accuracy is non-significant, and reported HHB/HBB architecture stability differences are descriptive.

## Project Context

This work originated as a NeuroAI research project and developed into a frozen study of layerwise credit assignment, representation compression, and hybrid local/global learning. The release preserves negative results and governance history, including the failed Stage 2D confirmation and the later independently documented Stage 3 re-scoping.

## License

Original project code is released under the [MIT License](LICENSE). The
output-filter update-centering adaptation from the Neuromatch Academy NeuroAI
Course remains subject to its BSD 3-Clause notice; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Neuromatch Academy does not
endorse this project.
