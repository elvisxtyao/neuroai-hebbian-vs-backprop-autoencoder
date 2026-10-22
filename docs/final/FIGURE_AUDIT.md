# Final figure audit

## Scope and release rule

This audit was performed before promoting any figure into the final release. The numerical source of every final hero figure is restricted to the checksum-verified compact bundle at `release/v1.0-final/`. Existing figures under `results/` and `figures/presentation_stage1b/` were inspected as historical visual references only. They were not used as numerical inputs, modified, or deleted.

The governance boundary remains unchanged: Stage 2D ended in `CONFIRMATION FAILED`; the figures below describe the later, separately approved Stage 3 post-confirmation re-scoping and must not be read as reversing that outcome. Test access was zero before the corresponding freeze/model-selection gates; formal test evaluation occurred after those gates; Phase 1 and Phase 2 add zero test access.

The pre-Phase-2 inventory contained 89 formal PNG/PDF artifacts outside `_recovery/` and 10 tracked presentation PNG/PDF artifacts. Eight figure artifacts under `_recovery/` remain excluded and preserved. The apparent odd formal count is expected: Stage 2 Q4 tooling contains one PNG without a PDF counterpart; the other formal figures are PNG/PDF pairs.

## Formal figure inventory and disposition

| Historical location | Files | Research question / underlying source | Evidence role | Current/reuse audit | Final disposition |
|---|---:|---|---|---|---|
| `results/formal/phase0_v1_1/stage2_q4_tooling/seed42_v1/q4_seed42_panels.png` | 1 | Does Q4 instrumentation behave correctly? Seed-42 tooling output. | Formal governance/tooling | Current for its tooling role, but not final Stage 3 evidence and lacks five-seed uncertainty. | Preserve in place; do not promote. |
| `results/formal/phase0_v1_1/stage3_final_audit_supplement/q1_samples_seen_curve.{png,pdf}` | 2 | How does reconstruction change with observed samples? Frozen 350-row Q1 trajectory supplement. | Formal supplement | Values remain current and complete; historical view omits prefix attribution and cost context. | Reworked as Figure 2; no interpolation. |
| `results/formal/phase0_v1_1/stage3_q1_complete/{hebbian_depth_dose,learning_curves,matched_prefix_controls}.{png,pdf}` | 6 | What are the depth tradeoff and learned-prefix effects? Frozen Q1 aggregates and paired controls. | Formal Q1 | Values remain current, but seed observations, reconstruction tradeoff, and matched-control context are split across overlapping figures. | Reworked as Figures 1 and 2. |
| `results/formal/phase0_v1_1/stage3_q2_representation/figures/{effective_rank,layerwise_cka,linear_probe,pca_seed0_grid,separability,umap_seed0_grid,z_confusion_matrices}.{png,pdf}` | 14 | Where does representation structure degrade or recover? Frozen Q2 layer aggregates plus seed-specific embedding views. | Formal Q2 | Aggregate values remain current; summary plots omit visible seeds. PCA/UMAP/confusion grids are seed-specific and require archives intentionally absent from the compact release. Several panels overlap on the same layerwise story. | Rank and probe evidence reworked as Figure 3; embedding grids remain supporting only. |
| `results/formal/phase0_v1_1/stage3_q3_noise/figures/{accuracy_severity,prediction_js,reconstruction_degradation,representation_stability}.{png,pdf}` | 8 | How do methods respond to corruption severity? Frozen Q3 condition summaries. | Formal Q3 | Values remain current; central curves do not foreground individual seeds and multiple metric families overlap narratively. | Accuracy-severity evidence reworked as Figure 5; other metrics remain supporting. |
| `results/formal/phase0_v1_1/stage3_q4_updates/figures/{bp_gradient_snr,hebbian_update_mechanisms}.{png,pdf}` | 4 | How does the local update compare with matched BP reconstruction direction? Frozen Q4 layer-update summaries. | Formal Q4 | Values remain current, but repeated shared HBB/HHB/HHH prefix rows can look like independent evidence; uncertainty is not sufficiently foregrounded. | Reworked as Figure 4 using the canonical HHH `hebbian_effective` path only. |
| `results/formal/phase0_v1_1/stage3_q5q6_sweeps/analysis/figures/{classification_interactions,standardized_reconstruction_interactions,z_effective_rank_interactions}.{png,pdf}` | 6 | How do dimension and architecture configurations vary descriptively? Frozen Q5/Q6 analysis tables. | Formal Q5/Q6 analysis | Values remain current; interaction-style summaries hide seed trajectories and the retained late-heavy seed 4, and can over-emphasize non-significant interaction framing. | Reworked as Figure 6 with descriptive wording. |
| `results/formal/phase0_v1_1/stage3_q5q6_sweeps/architecture/{early_heavy,late_heavy}/noise/figures/{accuracy_severity,prediction_js,reconstruction_degradation,representation_stability}.{png,pdf}` | 16 | How robust is each architecture case? Accepted Q6 case summaries. | Formal Q6 case evidence | Values remain current, but the case-by-case files are fragmented and overlap the main Q3 robustness visual. | Preserve as supporting evidence; no direct promotion. |
| `results/formal/phase0_v1_1/stage3_q5q6_sweeps/architecture/{early_heavy,late_heavy}/update_mechanisms/figures/{bp_gradient_snr,hebbian_update_mechanisms}.{png,pdf}` | 8 | How do update metrics vary by architecture case? Accepted Q6 case summaries. | Formal Q6 case evidence | Values remain current; panels overlap Figure 4 and are too detailed for the six-figure release narrative. | Preserve as supporting evidence; no direct promotion. |
| `results/formal/phase0_v1_1/stage3_q5q6_sweeps/dimension/{L16,L32,L128}/noise/figures/{accuracy_severity,prediction_js,reconstruction_degradation,representation_stability}.{png,pdf}` | 24 | How robust is each latent-dimension case? Accepted Q5 case summaries. | Formal Q5 case evidence | Values remain current, but files are fragmented and overlap the primary robustness story. | Preserve as supporting evidence; no direct promotion. |

## Presentation figure inventory and disposition

All five pairs use the historical `figures/presentation_stage1b/plotted_values.csv` source and are exploratory/presentation evidence rather than Stage 3 final evidence.

| Figure pair | Research question | Current/reuse audit |
|---|---|---|
| `preliminary_classification_accuracy.{png,pdf}` | Preliminary classification comparison | Superseded by the accepted Stage 3 Q1 matrix; lacks final controls and frozen uncertainty. Visual reference only. |
| `preliminary_reconstruction_mse.{png,pdf}` | Preliminary reconstruction comparison | Superseded by accepted system and standardized-decoder results. Visual reference only. |
| `representation_health_heatmap.{png,pdf}` | Early representation-health screening | Stage 1B exploratory overview; thresholds and scope do not represent final Q2 evidence. Do not reuse numerically. |
| `stage1b_tradeoff_effective_rank.{png,pdf}` | Early accuracy/rank tradeoff | Overlaps the final layerwise story but uses preliminary Stage 1B values and lacks final five-seed uncertainty. Visual reference only. |
| `stage1b_tradeoff_winner_coverage.{png,pdf}` | Early accuracy/coverage tradeoff | Preliminary endpoint not selected for the six-figure final narrative. Visual reference only. |

These ten artifacts remain tracked and unchanged. None is promoted, copied, or cited as final numerical evidence.

## Excluded and preserved figures

- The eight PNG/PDF artifacts under `results/formal/phase0_v1_1/stage3_q5q6_sweeps/_recovery/` remain excluded from the release and untouched.
- Timestamped seed-0 runs, `q1_clean_v1`, tuning, reconstruction sanity, and hybrid-depth diagnostic figures remain exploratory or diagnostic. They are not inputs to the final figures.
- No historical figure file was overwritten. The final figures live only in `figures/final/`.

## Final six-figure narrative

| Final figure | Question answered | Frozen compact inputs | Statistical display |
|---|---|---|---|
| Figure 1 | What performance/reconstruction tradeoff does each learning-rule allocation achieve? | Q1 per-seed and method summary tables | Five seed observations, mean, frozen 95% bootstrap CI |
| Figure 2 | Do learned prefixes add value over matched random controls, and what training cost accompanies them? | Q1 per-seed, paired contrasts, and complete samples-seen supplement | Paired seed effects with frozen 95% CI; observed trajectories only; per-seed cost points |
| Figure 3 | How does representation rank and probe utility change with encoder depth? | Q2 per-seed layer metrics and method-layer summary | Five seed trajectories, mean +/- sample SD |
| Figure 4 | How does the effective local Hebbian update compare with the matched BP reconstruction direction by depth? | Q4 per-seed update metrics and method-layer summary | Canonical HHH path, five seed points, mean +/- sample SD |
| Figure 5 | How robust are the four core methods across corruption families and severity? | Q3 per-seed condition metrics and condition summary | Five seed trajectories, mean, frozen 95% bootstrap CI |
| Figure 6 | How do dimension and architecture allocation change performance and latent rank? | Q5/Q6 performance and z-layer representation tables | Five seed trajectories, mean, frozen 95% bootstrap CI; seed 4 retained |

The separate `method_design` schematic contains no numerical evidence and is not counted among the six formal hero figures.
