# Stage 3 Q1 Complete Five-Seed Results

Date completed: 2026-07-29

Core training source: `3c422aa4a0bd6f8e65ce0b1f45a8b001239c965d`

Matched-control source: `23b61502449fecec9cf776eb35588eca09874108`

Formal aggregation source: `bb245897d6d987f378a223c030ca904cdc491cc8`

## Scope and integrity

This report completes Q1 for the seven-condition matrix:
`BBB/HHH/HHB/HBB/Full Random/RBB/RRB × seeds 0–4`.

The 25 headline system runs and 10 matched-control system runs all passed their
technical freeze gates before test access. Each frozen system decoder,
standardized decoder, and probe was evaluated on the official 10,000-image
MNIST test set once. The aggregation step loaded only immutable CSV/JSON
artifacts; it loaded no dataset and no checkpoint and added zero test accesses.

The matched controls have the following layer rules:

- `RBB`: frozen random Enc1, BP Enc2, BP Enc3;
- `RRB`: frozen random Enc1, frozen random Enc2, BP Enc3.

They are paired to `HBB` and `HHB`, respectively, by seed, initial model state,
system decoder, standardized decoder, probe initialization, data split, and
batch order.

## Complete clean-performance table

Values are five-seed mean; accuracy also reports sample SD.

| Method | Accuracy | SD | Macro-F1 | CE | System MSE | Standardized MSE | Recon AULC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBB | 0.915840 | 0.002379 | 0.914492 | 0.282652 | 0.003201 | 0.003234 | 0.004565 |
| HHH | 0.894100 | 0.008167 | 0.892730 | 0.355258 | 0.018500 | 0.018689 | 0.021681 |
| HHB | **0.919860** | 0.002825 | **0.918618** | **0.277340** | 0.004647 | 0.004569 | 0.006393 |
| HBB | 0.916380 | 0.001859 | 0.915151 | 0.283203 | 0.003270 | 0.003312 | 0.004656 |
| Full Random | 0.808040 | 0.027290 | 0.804958 | 0.607885 | 0.019446 | 0.019446 | 0.021437 |
| RBB | 0.915280 | 0.001624 | 0.913990 | 0.285304 | 0.003309 | 0.003302 | 0.004724 |
| RRB | 0.916400 | **0.001208** | 0.915141 | 0.285940 | 0.004617 | 0.004559 | 0.006240 |

Reconstruction AULC is the mean validation reconstruction MSE over the ten
system decoder/joint-training epochs; lower is better.

## Preregistered paired contrasts

Differences are `left − right`. For CE and MSE, a negative difference favors
the left method.

| Contrast | Accuracy difference | Paired bootstrap 95% CI | Cohen's dz | Standardized-MSE difference | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| HHB − HHH | +0.02576 | [+0.01664, +0.03328] | +2.455 | −0.014120 | [−0.015455, −0.012754] |
| HBB − HHB | −0.00348 | [−0.00540, −0.00104] | −1.213 | −0.001256 | [−0.001510, −0.000969] |
| BBB − HHB | −0.00402 | [−0.00816, −0.00106] | −0.863 | −0.001334 | [−0.001504, −0.001132] |
| HBB − RBB | +0.00110 | [+0.00036, +0.00180] | +1.173 | +0.000010 | [−0.000143, +0.000175] |
| HHB − RRB | +0.00346 | [+0.00204, +0.00554] | +1.477 | +0.000010 | [−0.000281, +0.000205] |

The confidence intervals bootstrap the five seed-level paired differences
(`10,000` resamples, seed `2026`). With only five seeds, effect sizes and
confidence intervals are descriptive; no isolated p-value is used as the main
claim.

## Learning budgets

| Method | System samples | System wall time (s) | Standardized decoder samples | Probe samples |
|---|---:|---:|---:|---:|
| BBB | 500,000 | 313.0 | 500,000 | 1,500,000 |
| HHH | 2,000,000 | 695.3 | 500,000 | 1,500,000 |
| HHB | 1,500,000 | 579.3 | 500,000 | 1,500,000 |
| HBB | 1,000,000 | 435.9 | 500,000 | 1,500,000 |
| Full Random | 500,000 | 223.2 | 500,000 | 1,500,000 |
| RBB | 500,000 | 208.7 | 500,000 | 1,500,000 |
| RRB | 500,000 | 189.9 | 500,000 | 1,500,000 |

The system budget includes the local Hebbian passes and actual BP
decoder/suffix passes for that method. Per-layer exposure is matched, but total
system samples are necessarily larger when more layers are trained greedily.

No absolute samples-to-threshold value was preregistered before these formal
runs. It is therefore recorded as `NA_NOT_PREREGISTERED`; selecting a threshold
after inspecting the curves would be post-hoc.

## Q1 conclusions

1. Full Hebbian (`HHH`) is worse than `BBB` on classification and much worse on
   both reconstruction outcomes. It also requires four times the system sample
   budget because its three local layers are trained sequentially before its
   decoder.
2. The minimal BP Enc3 suffix (`HHB`) repairs the main classification deficit:
   it exceeds `HHH` by 2.576 percentage points and has the highest mean test
   accuracy in this matrix. It does not fully repair recoverable information:
   its standardized MSE remains worse than `BBB`.
3. `HBB` moves reconstruction close to `BBB`, but its classification is
   slightly below `HHB`.
4. Against matched random prefixes, one Hebbian layer (`HBB − RBB`) adds
   0.110 percentage points and two Hebbian layers (`HHB − RRB`) add 0.346
   percentage points. Both accuracy intervals are above zero, so the shallow
   Hebbian features have a small, reproducible classification value beyond
   merely providing a frozen random prefix.
5. The matched standardized-reconstruction contrasts are effectively zero and
   both intervals cross zero. These runs do not show that the Hebbian prefixes
   contain more decoder-recoverable pixel information than random prefixes.
6. `HHB` remains a **rank-repair hybrid condition with unresolved
   reconstruction stability**, not a complete repair and not a biologically
   plausible end-to-end learner.

## Artifacts

Local immutable results:
`results/formal/phase0_v1_1/stage3_q1_complete/`

- `per_seed_complete.csv`: all 35 seed/method rows;
- `method_summary.csv`: mean, SD and bootstrap CI for all outcomes and budgets;
- `paired_contrasts.csv`: all five contrasts, metrics, CIs and Cohen's dz;
- `summary.json`: machine-readable complete result;
- `learning_curves.{png,pdf}`;
- `hebbian_depth_dose.{png,pdf}`;
- `matched_prefix_controls.{png,pdf}`;
- `provenance.json`: confirms no dataset/checkpoint load and zero new test
  access.

Implementation validation:
`verification/phase0_v1_1/stage3_q1_analysis_junit.xml` (`101/101` tests).

