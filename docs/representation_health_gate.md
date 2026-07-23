# Stage 1 — Representation health gate

Date: 2026-07-23

Status: **completed; gate decision = FAIL**

This stage evaluates whether the validation-selected Hebbian encoder is healthy
enough to enter Q4 tooling validation and formal multi-seed experiments. It
uses validation data only and performs no training or optimizer update.

## 1. Frozen inputs

| Item | Value |
|---|---|
| Protocol base | `phase0-v1.1-formal` |
| Analysis source commit | `dc0210652e65d6b5682ebf89dc7f6412f16f9392` |
| Primary checkpoint | selected Hebbian seed 42, `lr=0.0005`, `winner_fraction=0.10` |
| Seed-42 reference | selected BP seed 42, Adam `lr=0.003` |
| Repeat checks | preliminary Q1 BP/Hebbian seeds 0 and 1 |
| Logical data split | validation indices from the official MNIST training partition |
| Subset | 2,000 images; exactly 200 per class; subset seed 17 |
| Subset manifest | `data/splits/mnist_validation_health_v1.npz` |
| Subset SHA-256 | `5198fcd030eb1b37f9c9b29767cb2b84870c674f66e805f57b77b0bcd9e2fe1f` |
| Source split SHA-256 | `e7e92e0252a4ffd8b80651b9fe630f4914b563d2b6b802c0c397a8cf1c31ee54` |
| Test samples accessed | 0 |

The threshold configuration, checkpoint paths and roles were committed before
the formal gate was run. The primary decision uses only the selected Hebbian
seed-42 checkpoint. Q1 seeds 0–1 are repeat checks, not additional formal
seeds.

## 2. Corrected collapse definition

The former `active_neuron_ratio` name conflated two different quantities:

1. **Per-location WTA density:** with `winner_fraction=0.10`,
   `k=ceil(0.10*C)` units are selected at each spatial location. At `z`,
   `k/C=7/64=0.109375`. This sparsity is expected.
2. **Winner coverage across the dataset:** the fraction of channels that win
   at least once over all validation sample/spatial locations. With 2,000
   samples, healthy competition should distribute wins across substantially
   more than seven channels; under uniform independent selection its expected
   coverage is effectively 1.

Stage 1 therefore uses three separate classifications:

- expected WTA sparsity is consistent when observed positive-winner density
  matches `k/C` within the frozen tolerance;
- pathological winner concentration is present when coverage, normalized
  winner entropy or maximum winner share fails its threshold;
- representation degeneracy is present when active-unit coverage,
  nonzero-variance coverage or effective rank fails;
- **pathological collapse** requires winner concentration and representation
  degeneracy together.

The frozen thresholds are:

| Check | Threshold |
|---|---:|
| Active-unit ratio | ≥ 0.50 |
| Nonzero-variance unit ratio | ≥ 0.50 |
| Participation-ratio effective rank | ≥ 2.0 |
| Effective rank / channels | ≥ 0.10 |
| Winner coverage | ≥ 0.50 |
| Normalized winner entropy | ≥ 0.50 |
| Maximum winner share | ≤ 0.25 |
| Winner-density tolerance | max(0.01, 10% of expected density) |

Effective rank is computed from the centered channel covariance over all
sample/spatial observations:

```text
r_eff = (sum(lambda))^2 / sum(lambda^2)
```

## 3. Primary seed-42 result

| Layer | Expected/observed winner density | Winner coverage | Winner entropy | Max share | Effective rank | Normalized rank | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| h1 | 0.1250 / 0.0401 | 0.7500 | 0.5432 | 0.4046 | 1.1456 | 0.0716 | FAIL; confirmed collapse |
| h2 | 0.1250 / 0.0683 | 0.3125 | 0.4225 | 0.2499 | 1.0384 | 0.0325 | FAIL; confirmed collapse |
| z | 0.109375 / 0.109375 | 0.109375 | 0.4679 | 0.1429 | 1.0186 | 0.0159 | FAIL; confirmed collapse |

At `z`, activation active-unit ratio is still 0.9844 and nonzero-variance ratio
is 0.9844. The failure is therefore not simply “most ReLU units are exactly
zero.” Instead, variation is almost one-dimensional and top-k competition is
monopolized.

For every one of the 2,000 validation images, the same seven seed-42 Hebbian
units (`2, 22, 25, 27, 39, 45, 55`) occupy all seven `z` winner positions.
Each wins exactly 2,000 times; the other 57 units never win. Thus
`0.109375` matching the WTA fraction is evidence of a fixed-winner monopoly,
not evidence that the old detector was merely observing expected sparsity.

## 4. Reference and repeat evidence at z

| Checkpoint | Winner coverage | Winner entropy | Effective rank | Normalized rank |
|---|---:|---:|---:|---:|
| Hebbian selected seed 42 | 0.1094 | 0.4679 | 1.0186 | 0.0159 |
| BP selected seed 42 | 0.4531 | 0.7758 | 11.8533 | 0.1852 |
| Hebbian preliminary seed 0 | 0.1094 | 0.4679 | 1.0159 | 0.0159 |
| BP preliminary seed 0 | 0.8438 | 0.8983 | 20.9398 | 0.3272 |
| Hebbian preliminary seed 1 | 0.1094 | 0.4679 | 1.0167 | 0.0159 |
| BP preliminary seed 1 | 0.8438 | 0.9041 | 21.6949 | 0.3390 |

The exact identities of the seven winning Hebbian units vary by seed, but each
Hebbian checkpoint uses one fixed seven-unit set for all 2,000 images. This
repeats the mechanism across the two available preliminary Q1 seeds.

BP is a descriptive reference rather than a model required to satisfy a WTA
gate. Some absolute early-layer thresholds are strict enough to flag BP h1,
so early-layer cross-rule comparisons should be interpreted quantitatively.
The Hebbian `z` conclusion is nevertheless robust: its fixed-winner pattern and
rank near one are qualitatively different from all three BP checkpoints.

## 5. Integrity and acceptance checks

- All 2,000 sample IDs are unique and class counts are exactly balanced.
- Extracted sample IDs and labels exactly match the frozen manifest and order.
- All six checkpoint hashes are identical before and after extraction.
- The run manifest records a clean worktree and source commit `dc02106...`.
- `phase0-v1.1-formal` is an ancestor of the analysis commit.
- The runner constructs only the MNIST training partition and indexes its
  validation subset; official test data are not constructed or read.
- Unit tests distinguish correct WTA sparsity from fixed-winner collapse and
  verify bitwise deterministic metrics.
- Full test suite: `41 passed in 16.79s`.

Raw local artifacts:

```text
results/formal/phase0_v1_1/stage1_representation_health/gate_v1/
  gate_decision.json
  health_metrics.csv
  run_manifest.json
  winner_frequencies.csv
```

Artifact SHA-256:

| Artifact | SHA-256 |
|---|---|
| `gate_decision.json` | `c06d4fe97857299666a14446d854779198ed32f3a343c8d8d43746b1f096860d` |
| `health_metrics.csv` | `cf28fa04de20fb0f30c0fdf7fae575ab768e5370dd3d383aece226027b9822ee` |
| `run_manifest.json` | `00798fee072fe6669b3680ee840491f5776f374e9ac548bfd8cb17b945f183ec` |
| `winner_frequencies.csv` | `152b5ba2bc49dfd612b37903e1f9659d9163e0e7224df9e7aabbe2afe42931b4` |

## 6. Decision and consequence

**Stage 1 = completed, FAIL.**

The validation-selected Hebbian configuration is not approved for Stage 2 or
formal Q1 generation. Stage 1B must repair or reselect the Hebbian
configuration using train/validation only, then rerun this unchanged gate.
No Q4 seed-42 run, additional Q1 seed, dimension sweep or architecture sweep
is authorized by this result.

This stage supports a preliminary mechanism statement for Q2: the current
Hebbian encoder develops increasingly concentrated, near-rank-one channel
representations with depth. It does not yet answer the formal multi-seed Q2
class-geometry question.
