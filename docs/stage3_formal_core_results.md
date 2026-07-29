# Stage 3 Formal Core Results

Date completed: 2026-07-29

Training source commit: `3c422aa4a0bd6f8e65ce0b1f45a8b001239c965d`

Evaluator commit: `4595ac340e785df489ff295fdf42adca9d23403c`

Protocol: `docs/stage3_formal_protocol_v1.md`

## Scope and decision

The frozen `BBB/HHH/HHB/HBB/Full Random × seeds 0–4` matrix is complete.
All 25 system runs, 25 standardized-decoder runs, and 25 frozen probes were
run with paired initialization and the same split/probe/decoder protocol.

The pre-test technical freeze gate returned `PASS`. Every global and per-seed
check passed: artifact completeness, finite metrics, identical source commit,
paired initial state, paired system and standardized decoders, paired probe
initialization, unchanged frozen layers and encoders, matched Hebbian prefixes,
and zero pre-gate test access. No performance threshold was used to remove a
condition.

After the gate passed, the frozen test evaluator accessed each checkpoint once.
It did not train, select, or modify any model or probe. All 25 evaluations used
10,000 unique official MNIST test images, and all before/after checksums matched.

## Five-seed test results

Values are mean across five paired seeds. Accuracy dispersion is sample SD.

| Method | Accuracy | SD | Macro-F1 | CE | System MSE | Standardized-decoder MSE |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.91584 | 0.00238 | 0.91449 | 0.28265 | 0.003201 | 0.003234 |
| HHH | 0.89410 | 0.00817 | 0.89273 | 0.35526 | 0.018500 | 0.018689 |
| HHB | **0.91986** | 0.00283 | **0.91862** | **0.27734** | 0.004647 | 0.004569 |
| HBB | 0.91638 | **0.00186** | 0.91515 | 0.28320 | 0.003270 | 0.003312 |
| Full Random | 0.80804 | 0.02729 | 0.80496 | 0.60789 | 0.019446 | 0.019446 |

Bootstrap 95% CIs for test accuracy were:

- BBB: `[0.91376, 0.91760]`
- HHH: `[0.88830, 0.90132]`
- HHB: `[0.91766, 0.92224]`
- HBB: `[0.91498, 0.91778]`
- Full Random: `[0.78438, 0.82782]`

## Preregistered paired contrasts available in the core matrix

Differences use `left − right`. Reconstruction is an error metric, so a
negative MSE difference favors the left condition.

| Contrast | Test accuracy difference | Paired bootstrap 95% CI | Standardized MSE difference | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| HHB − HHH | +0.02576 | `[+0.01664, +0.03328]` | −0.014120 | `[−0.015455, −0.012754]` |
| HBB − HHB | −0.00348 | `[−0.00540, −0.00104]` | −0.001256 | `[−0.001510, −0.000969]` |
| BBB − HHB | −0.00402 | `[−0.00816, −0.00106]` | −0.001334 | `[−0.001504, −0.001132]` |

The core matrix therefore supports the following limited conclusions:

1. Replacing the third HHH layer with BP (`HHB`) consistently repairs the
   classification deficit and greatly improves reconstruction relative to HHH.
2. HHB test classification is slightly higher than BBB in these five seeds,
   but its standardized reconstruction remains worse. HHB is therefore a
   **rank/classification repair with unresolved information-recovery gap**, not
   a complete repair.
3. Moving from two Hebbian layers (`HHB`) to one (`HBB`) slightly reduces
   classification but improves standardized reconstruction to near-BBB levels.
4. Full Random is not a matched control for the value of a one- or two-layer
   Hebbian prefix. Claims that Hebbian prefixes outperform random prefixes
   remain prohibited until `HBB − RBB` and `HHB − RRB` are completed.

## Validation and learning-budget summary

| Method | Validation accuracy | System val MSE | Standardized val MSE | Reconstruction AULC | System samples seen | System wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| BBB | 0.91216 | 0.003320 | 0.003363 | 0.004565 | 500,000 | 315.6 |
| HHH | 0.88894 | 0.018863 | 0.019015 | 0.021681 | 2,000,000 | 698.1 |
| HHB | 0.91572 | 0.004803 | 0.004742 | 0.006393 | 1,500,000 | 582.2 |
| HBB | 0.91242 | 0.003393 | 0.003442 | 0.004656 | 1,000,000 | 438.7 |
| Full Random | 0.80362 | 0.019754 | 0.019754 | 0.021437 | 500,000 | 226.3 |

The system sample counts include the local Hebbian layer passes and the BP
joint/decoder passes actually used by each condition. The standardized decoder
adds 500,000 samples per run and the probe adds 1,500,000 samples per run.
Mean standardized-decoder wall time was 245–261 seconds; mean probe wall time
was 42–44 seconds. AULC is the mean validation reconstruction MSE over the ten
system decoder/joint-training epochs; lower is better.

## Integrity evidence

- Freeze gate:
  `results/formal/phase0_v1_1/stage3_core/freeze_gate.json`
- Run manifest:
  `results/formal/phase0_v1_1/stage3_core/run_manifest.json`
- One-time test summary:
  `results/formal/phase0_v1_1/stage3_core/test_evaluation/summary.json`
- Per-run test table:
  `results/formal/phase0_v1_1/stage3_core/test_evaluation/per_run_metrics.csv`
- Training logs:
  `results/formal/phase0_v1_1/stage3_core/seed{0..4}.stdout.log`
- All five training stderr logs are empty.
- Evaluator full-suite log:
  `verification/phase0_v1_1/stage3_one_time_test_evaluator_pytest.log`
  (`93/93` tests passed).

Generated checkpoints and results remain local and are not committed to Git.

## What remains outside this completed core

- `RBB/RRB × seeds 0–4` matched-prefix controls for the net value of Hebbian
  shallow layers;
- Q2 formal multi-seed layerwise representation extraction and geometry;
- Q3 deterministic noise robustness;
- Q4 formal Hybrid rule-boundary update analysis and cross-metric correlation;
- Q5/Q6 dimension and encoder-asymmetry sweeps.

Those tasks must reuse the frozen Stage 3 checkpoints and must not reinterpret
the present result as proof that HHB completely repairs reconstruction.
