# NeuroAI BP–Hebbian Project Status

Last updated: 2026-07-23

This file is the single source of truth for current execution status. It tracks
what is complete, what is only exploratory, what is blocked, and what should be
done next. Research definitions remain in `HEBBIAN_PROJECT_PLAN.md`; parent
comparison rules remain in `PHASE0_STANDARD_V1.md`, with formal-run overrides
in `PHASE0_STANDARD_V1_1_ADDENDUM.md`.

## 1. Status vocabulary

| Status | Meaning |
|---|---|
| Complete | Implemented, verified, and supported by saved evidence |
| Partial | Useful work exists, but at least one acceptance gate is still open |
| Blocked | Cannot be completed without external information or coordination |
| Planned | Defined but not yet implemented or run |
| Exploratory | Development evidence only; not a confirmatory result |
| Archived | Historical snapshot; not a live status source |

## 2. Document roles

| Document | Role | Maintenance policy |
|---|---|---|
| `PROJECT_STATUS.md` | Live operational status and next actions | Update whenever a task changes state |
| `PHASE0_STANDARD_V1.md` | Frozen normative comparison contract | Do not add progress notes; version any material protocol change |
| `PHASE0_STANDARD_V1_1_ADDENDUM.md` | Frozen formal-run override and reproducibility gate | Takes precedence over v1 where explicitly stated |
| `HEBBIAN_PROJECT_PLAN.md` | Research design, formulas, WBS IDs, and acceptance criteria | Update design decisions; use this file for requirements, not live status |
| `docs/tutorial_migration.md` | Source provenance and migration boundary | Update when a source notebook/tutorial is added or replaced |
| `docs/phase0_team_confirmation.md` | BP-team compliance evidence/template | Replace pending fields with dated teammate evidence |
| `docs/validation_tuning.md` | Formal validation-only tuning table, decisions, hashes, and limitations | Preserve seed-42 results; do not add test metrics |
| `docs/q1_clean_performance.md` | Preliminary paired clean-performance results and exact recovery point | Keep the n=2 limitation until seeds 2–4 complete |
| `README.md` | Repository entry point and reproducible commands | Keep concise; link to the documents above |

Run-specific reports are retained locally only. Root `*_REPORT.md`,
`docs/*_report.md`, `results/`, generated figures, and checkpoints are excluded
from Git. Public status claims instead point to tracked source code and tests,
so the repository does not contain broken links to private run records.

## 3. Current project summary

| Phase | Status | Evidence | Open gate |
|---|---|---|---|
| Stage 0: formal governance and reproducible snapshot | Complete | Phase 0 v1.1 addendum, formal configs/schema, environment and split identity, deferred test access, immutable full pytest log (37 passed), canonical Git ref | None; external tutorial/team records remain archival follow-ups |
| Phase 1: explicit Hebbian rule | Partial | Conv2d WTA/Oja, L2 normalization, `lr=0`, 500-step stability, reproducible trajectory, legacy collapse detector | Representation-health definition and BP-reference compatibility |
| Phase 2: seed-0 end-to-end engineering loop | Partial | Encoder, frozen decoder/probe, exact resume, reconstruction, random-encoder decoder-only control, paired BP–Hebbian diagnostics | Stage 1 health gate; historical collapse interpretation is unresolved |
| Phase 3: validation-only tuning | Complete | Seed 42; 8 Hebbian + 8 BP unique trials; zero test rows; selected config hashes | Hebbian enc3 collapse remains a downstream mechanism warning |
| Phase 4 / Q1: clean performance | Partial | Paired seeds 0–1; BP, Hebbian, and random controls; test metrics, reconstruction, AULC, timing, preliminary paired CI | Resume partial Hebbian seed 2 and finish paired seeds 2–4 |
| Phase 5 / Q2: representations | Planned | Extraction helper exists | Fixed 2,000-sample manifest and quantitative layerwise analysis |
| Phase 6 / Q3: robustness | Planned | Noise severities specified in config | Deterministic noise generator and paired evaluation |
| Phase 7 / Q4: update mechanisms | Planned | Hebbian candidate update exists | Frozen BP reference, alignment, bias, variance, and SNR |
| Phase 8 / Q5–Q6: dimension/asymmetry | Planned | Dimensions and architecture IDs specified in plan | Configurable channel architecture, sweeps, and representation analysis |
| Phase 9: extensions | Planned | Scope listed | Core MNIST matrix must finish first |

## 4. Phase 0 evidence

### Complete

- “3-layer” means three learnable encoder Conv2d layers; shapes are frozen.
- MNIST is split deterministically into 50,000 train, 10,000 validation, and
  10,000 official test samples.
- BP and Hebbian use the same `ConvAutoencoder`, initialization, data loaders,
  latent dimension, decoder class, linear probe, and evaluation metrics.
- The encoder representation-training interface does not accept labels.
- Main experiments disable target clamping.
- Notebook cells are not accepted as official experiment entry points.
- The official entry points are validated YAML plus Python module CLIs.
- Decoder/probe language is explicit: only the encoder is Hebbian-trained;
  decoder and probe use backpropagation after encoder freezing.
- Phase 0 v1.1 freezes BP Adam `lr=0.003`, formal seeds `[0,1,2,3,4]`,
  tuning seed `42`, formal/preliminary artifact isolation, initialization
  pairing, and validation-before-test access.
- Formal configs are validated by the same schema and carry the protocol block
  into resolved configs and run metadata.
- The canonical formal source ref is `phase0-v1.1-formal`; environment and
  full-suite test evidence are tracked under `environment/` and `verification/`.

### Pending external evidence

- The original Hebbian tutorial/notebook containing the custom-autograd,
  BP-reference, cosine, or SNR teaching prototype has not been provided. Its
  URL/path/version/hash therefore cannot yet be verified.
- Written confirmation from the BP teammate that their implementation is
  `phase0-v1 compliant` has not been attached.

These two items are tracked in `docs/tutorial_migration.md` and
`docs/phase0_team_confirmation.md`. They must remain pending until real evidence
is supplied; they must not be self-certified by the Hebbian implementation.
They do not authorize protocol drift and are not blockers for the local
canonical source freeze.

## 5. Phase 1 implementation status

### Complete

- Explicit `compute_local_update()` and `apply_local_update()` under
  `torch.no_grad()`.
- Per-sample, per-spatial-location channel top-k WTA.
- Local Hebbian correlation plus Oja stabilization.
- Per-output-filter L2 normalization after each applied update.
- Greedy layer isolation and frozen earlier-layer checks.
- Candidate computation does not mutate weights.
- Tiny deterministic Oja update matches a hand calculation.
- Encoder parameters are excluded from gradient-based representation updates.
- Labels do not enter the main representation-learning path.
- `learning_rate=0` is an exact no-op, including no normalization side effect.
- A 500-update stress test remains finite with unit filter norms.
- A fixed-seed multi-step update trajectory is bitwise reproducible.
- Active-filter and maximum-winner-share thresholds produce a legacy
  `collapse_detected` field in every Hebbian epoch record. Its scientific
  interpretation is not accepted until Stage 1 validates it against expected
  top-k sparsity and additional representation-health metrics.

### Open engineering debt

- Add a matched-state interface test that produces Hebbian and BP-reference
  candidates without changing the snapshot.
- `HebbianLinear` remains optional for formula pedagogy; it is not required by
  the convolutional main experiment unless the team chooses to restore it as a
  deliverable.

## 6. Phase 2 seed-0 status

The single-seed engineering loop is complete and reproducible:

- greedy `enc1 -> enc2 -> enc3` training;
- layer-end encoder checkpoints;
- frozen BP-trained decoder;
- frozen standardized linear probe;
- reconstruction grid and diagnostic plots;
- paired BP–Hebbian training diagnostics restricted to common metrics, with
  the different reconstruction protocols explicitly labelled;
- encoder checksum protection;
- test accuracy, macro-F1, CE, and reconstruction MSE.

Step 2 engineering hardening is also complete for all future runs:

- immutable epoch archives plus an atomic `resume_checkpoint.pt`;
- exact restoration of model, optimizer, Python/NumPy/Torch RNG, CUDA RNG when
  present, and the shuffled train-loader generator;
- resumable BP representation training, greedy Hebbian layers, and frozen
  decoder training;
- atomic metric upsert keyed by stage/split/layer/epoch, so replay cannot add
  duplicate rows;
- `running`, `paused`, `completed`, and `failed` run states with progress,
  sample, step, elapsed-time, checkpoint, error, and resume counters;
- config fingerprint validation and Git/runtime/split provenance.

The existing seed-0 run directories are historical records and are not
silently rewritten with unavailable timing data. The expanded schema applies
to newly launched runs. Exact interrupted-versus-uninterrupted equivalence is
verified on synthetic BP, Hebbian-layer, and decoder runs.

### Step 2.2 random-encoder reconstruction control

The paired seed-0 encoder was left exactly at initialization and only its BP
decoder was trained for 10 epochs. The encoder checksum remained unchanged.
Test MSE fell from `0.230406` for the fully untrained autoencoder to `0.018890`
after decoder-only training, a 91.80% reduction. This is slightly lower than
the current Hebbian run's `0.019896`, but remains 5.74 times the BP
autoencoder's `0.003289`.

Therefore the decoder **can strongly compensate for a random encoder for
reconstruction**, and at seed 0 it fully matches the current Hebbian
reconstruction result. It does not match BP. Reconstruction quality alone
cannot demonstrate that the Hebbian encoder learned a better latent
representation. The random-encoder linear-probe control is still required for
classification and representation claims.

The run remains exploratory because the representation-health gate has not
been run. The historical threshold produced the following diagnostic:

| Layer | Final active-neuron ratio | Final winner entropy | Gate interpretation |
|---|---:|---:|---|
| enc1 | 1.0000 | 0.7466 | Legacy detector did not flag active-filter collapse |
| enc2 | 0.4688 | 0.5711 | Participation falls; needs Stage 1 context |
| enc3 | 0.2031 | 0.6167 | Unresolved; active ratio alone is not a valid collapse verdict |

Another current checkpoint reports `active_neuron_ratio=0.109375`, close to
the configured `winner_fraction=0.10`; this is exactly the condition under
which expected WTA sparsity may be confused with pathological winner
concentration. Stage 1 must reconcile checkpoints on one fixed validation
subset. Effective rank was not saved in the historical seed-0 run. A paired
random-encoder **linear-probe** control is now included in the formal Q1
pipeline; the two completed seeds average 82.765% test accuracy. The older
89.00% Hebbian result remains development evidence and is not mixed with the
new frozen-configuration Q1 runs.

## 7. Research-question status

| Question | Current answer | Required before claiming an answer |
|---|---|---|
| Q1: classification performance | Preliminary n=2: BP 91.595%, Hebbian 90.220%, random encoder 82.765%; paired Hebbian−BP gap −1.375 pp | Resume and complete paired seeds 2–4 before confirmatory claim |
| Q2: latent representations | Only sparsity/active ratio/winner entropy are available | h1/h2/z fixed-subset geometry and quantitative metrics |
| Q3: robustness | No result | Deterministic paired clean-to-noisy evaluation |
| Q4: weight updates | Only Hebbian candidate norms are logged | Matched-state BP reference, alignment, norm ratio, bias, variance, SNR |
| Q5: dimension/asymmetry performance | No result | Parameterized models, frozen matrices, sensitivity/interaction analysis |
| Q6: asymmetry and representations | No result | Q2 pipeline repeated across the Q5 architecture matrix |

## 8. Frozen decisions

- Existing seed-0 test results are development evidence and must not be used to
  select future hyperparameters.
- Formal tuning uses only `tuning_seed=42` and validation metrics.
- Phase 0 v1.1 overrides the parent BP default: Adam learning rate is `0.003`;
  BP learning-rate tuning is not repeated.
- Test representations and metrics are evaluated only after every
  validation-only choice is frozen and the best validation checkpoint restored.
- BP and Hebbian must continue to share forward architecture and evaluation
  code within each experiment variant.
- The main model must be described as a Hebbian-trained encoder with a
  BP-trained decoder and linear probe, not a fully Hebbian autoencoder.
- Any material change to data, shapes, loss, training budget, probe, or noise
  realization requires a documented standard-version change and paired reruns.
- Existing dirty/historical runs remain preliminary and cannot be copied into
  the formal artifact tree.
- No further Q1 seeds, Q4 run, dimension sweep, or architecture sweep may start
  before the Stage 1 health gate and Stage 2 Q4 tooling gate pass.

## 9. Immediate next actions

1. **Stage 1 — representation health gate:** create one deterministic,
   class-balanced 2,000-image validation manifest and evaluate expected versus
   observed sparsity, winner frequency/concentration, dead-unit ratio, entropy,
   activation variance, and effective rank on seed-42/current Q1 checkpoints.
   Output an explicit PASS/FAIL without reading test.
2. **Stage 1B — conditional Hebbian repair:** run only if Stage 1 FAILS. Use
   train/validation evidence and preregistered candidates; freeze a new config
   and hash only after it passes the same health gate.
3. **Stage 2 / Q4 tooling gate:** on the health-approved seed-42 snapshots and
   50 fixed batches, implement raw BP direction, raw/effective Hebbian delta,
   cosine, norm ratio, alpha-star, scale-matched bias, and per-rule SNR. Require
   synthetic tests, fixed sample IDs, no optimizer step, unchanged hashes, and
   complete seed-42 output.
4. **Stage 3 / Q1 formal runs:** only after Stages 1–2 pass, generate paired
   BP/Hebbian/random seeds 0–4 from the canonical ref and formal configs.
5. Continue in dependency order: Q4 five-seed analysis; Q1 final statistics;
   Q2 representations; Q3 deterministic noise; Q5 dimension; Q5/Q6 asymmetry;
   final paper and reproducibility package.

Tutorial provenance and teammate confirmation remain useful coordination items,
but they do not replace or reorder the scientific gates above.

## 10. Status change log

| Date | Change | Evidence |
|---|---|---|
| 2026-07-20 | Phase 0 shared skeleton and BP baseline completed | `models/`, `training/`, and shared tests |
| 2026-07-20 | Explicit Hebbian encoder seed-0 loop completed | `learning_rules/hebbian.py`, Hebbian tests |
| 2026-07-21 | Live-status governance introduced; historical and normative documents separated | This file and `docs/` records |
| 2026-07-22 | Step 2 run records, transactional epoch checkpoints, exact resume, collapse diagnostics, and recovery tests completed | `utils/checkpointing.py`, `utils/results.py`, `tests/test_training_resume.py` |
| 2026-07-22 | Step 2.2 random frozen encoder + trained decoder baseline completed and recovered from epoch 7 | `training/train_random_encoder_decoder.py`, `tests/test_random_encoder_decoder.py` |
| 2026-07-22 | Common-metric BP–Hebbian training diagnostic plot implemented | `evaluation/plot_run_metrics.py` |
| 2026-07-22 | Run-specific reports and generated outputs changed to local-only Git policy | `.gitignore`, this status file |
| 2026-07-23 | Step 3 validation-only tuning completed and shared-L64 configs frozen | `docs/validation_tuning.md`, tuning runner, manifests, selected configs |
| 2026-07-23 | Step 4 paused after paired seeds 0–1; preliminary Q1 table, random controls, AULC/timing plots, paired CI, and safe seed-boundary resume added | `docs/q1_clean_performance.md`, `training/run_q1_clean.py`, `tests/test_q1_clean.py` |
| 2026-07-23 | Stage 0 formal governance frozen: Phase 0 v1.1, BP `lr=0.003`, formal configs/schema, environment/split identity, deferred test access, full tests, and canonical source ref | `PHASE0_STANDARD_V1_1_ADDENDUM.md`, `configs/formal/`, `environment/`, `verification/phase0_v1_1/pytest_full.log` |
