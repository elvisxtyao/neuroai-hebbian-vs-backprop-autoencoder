# NeuroAI BP–Hebbian–Hybrid Project Status

Last updated: 2026-08-10

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
| `docs/tutorial_migration.md` | Source provenance, audited notebook semantics, and migration boundary | Update when a source notebook/tutorial is added or replaced |
| `docs/phase0_team_confirmation.md` | BP-team compliance evidence/template | Replace pending fields with dated teammate evidence |
| `docs/validation_tuning.md` | Formal validation-only tuning table, decisions, hashes, and limitations | Preserve seed-42 results; do not add test metrics |
| `docs/q1_clean_performance.md` | Preliminary paired clean-performance results and exact recovery point | Keep the n=2 limitation until seeds 2–4 complete |
| `docs/representation_health_gate.md` | Formal Stage 1 validation-only health result and corrected collapse definition | Preserve the FAIL decision |
| `docs/stage1b_hebbian_repair.md` | Frozen Stage 1B validation-only repair matrix, integrity evidence, and no-selection decision | Preserve `COMPLETED — NO CANDIDATE PASSED`; do not append v3/v4 candidates |
| `docs/stage1c_effective_rank_audit.md` | No-training metric audit, pre/post-WTA mechanism result, spectra, integrity evidence, and interpretation limits | Preserve the PASS decision and failure-case-snapshot scope |
| `docs/output_filter_centering_mechanism.md` | Notebook audit and bounded seed-42 output-filter-centering mechanism experiment | Preserve the failed-gate decision; do not treat it as a new Stage 1B round |
| `docs/hebbian_failure_case_protocol_addendum.md` | Branch-D scope and restrictions after the failed repair | Do not start another repair candidate or Stage 3 from this decision |
| `docs/hybrid_depth_ablation_protocol.md` | Preregistered two-control Hybrid-HHB/Hybrid-HBB depth ablation | Do not add a third control or tune from seed-42 results |
| `docs/hybrid_depth_ablation_results.md` | Completed seed-42 depth-ablation evidence and Outcome-D decision | Keep diagnostic-only language until confirmation seeds pass |
| `docs/hybrid_hhb_confirmation_protocol.md` | Immutable Stage 2D seeds 43/44 confirmation and dual-reconstruction contract | Do not tune, change thresholds, access test, or append a third seed |
| `docs/hybrid_hhb_confirmation_results.md` | Completed Stage 2D paired results, gate table, integrity evidence and failed confirmation decision | Preserve the seed-43 reconstruction failure and zero-test boundary |
| `docs/stage3_formal_protocol_v1.md` | Approved Stage 3 five-seed protocol and revised HHB research role | Keep HHB wording limited to rank repair; standardized reconstruction is an outcome, not an entry gate |
| `docs/stage3_formal_core_results.md` | Completed five-seed core matrix, technical gate, one-time test results, budgets and limitations | Preserve the 25-condition matrix and prohibit Hebbian-prefix value claims until RBB/RRB |
| `release/v1.0-final/manifest.json` | Phase 1 compact-evidence freeze, relative-path provenance, row-count contracts and release inventory | Derived from accepted aggregate artifacts only; never substitute recovery, exploratory or raw checkpoint outputs |
| `README.md` | Repository entry point and reproducible commands | Keep concise; link to the documents above |

Run-specific reports are retained locally only. Root `*_REPORT.md`,
`docs/*_report.md`, `results/`, generated figures, and checkpoints are excluded
from Git. Public status claims instead point to tracked source code and tests,
so the repository does not contain broken links to private run records.

## 3. Current project summary

| Phase | Status | Evidence | Open gate |
|---|---|---|---|
| Stage 0: formal governance and reproducible snapshot | Complete | Phase 0 v1.1 addendum, formal configs/schema, environment and split identity, deferred test access, immutable full pytest log (37 passed), canonical Git ref | None; external tutorial/team records remain archival follow-ups |
| Stage 1: representation health gate | Complete — FAIL, interpretation audited | Fixed 2,000-image validation manifest; density/coverage/entropy/variance/effective-rank metrics; clean seed-42 run; 41 tests | Stage 1C confirmed the calculation and narrowed it to raw-covariance anisotropy; Stage 1B still has no passing repair |
| Stage 1B: Hebbian repair/reselection | **Complete — no candidate passed** | Eight preregistered validation-only candidates across stateless competition normalization and centered local inputs; 45/47-test immutable logs; zero test access | No replacement Hebbian config selected; Stage 1B is frozen |
| Stage 1C: effective-rank metric audit | **Complete — metric validity PASS** | Same 2,000-image validation subset; explicit axes/centering; covariance and singular spectra; PR/stable/class ranks; epsilon audit; frozen probe; 54 tests; zero test access | Low rank exists pre-WTA; use exact seed-42 checkpoint only as Q4 failure-case snapshot |
| Phase 1: explicit Hebbian rule | **Complete — convolutional experiment scope** | Conv2d WTA/Oja, L2 normalization, `lr=0`, stability/reproducibility, non-mutating raw/effective candidates, and matched-state BP-reference tests | Optional `HebbianLinear` pedagogy remains out of main-experiment scope; formal configuration selection is tracked separately by Stage 1/1B |
| Phase 2: seed-0 end-to-end engineering loop | Partial | Encoder, frozen decoder/probe, exact resume, reconstruction, random-encoder decoder-only control, paired diagnostics | Current Hebbian selected config failed Stage 1 |
| Phase 3: validation-only tuning | Complete, selection not approved | Seed 42; initial 8 Hebbian + 8 BP trials, followed by eight frozen Stage 1B repair candidates; zero test rows | No Stage 1B candidate passed; no repaired Hebbian config is approved |
| Historical Phase 4 / Q1 two-rule run | **Exploratory — paused** | Paired seeds 0–1; BP, Hebbian, and random controls; test metrics, reconstruction, AULC, timing, preliminary paired CI | Superseded by the expanded Hybrid matrix; do not resume seeds 2–4 as formal evidence |
| Phase 5 / Q2: representations | **Complete — formal five-seed layerwise analysis** | BBB/HHH/HHB/HBB/RBB/RRB × seeds 0–4 × h1/h2/z; fixed class-balanced 2,000-image test subset; 30 raw representation archives, 90 metric rows, probes/k-NN/rank/class geometry/CKA/PCA/UMAP/confusion; all checksums and integrity gates PASS; 109 tests | None; architecture-dependent representation changes are now completed in Q6 |
| Phase 6 / Q3: robustness | **Complete — formal five-seed paired noise evaluation plus curve audit** | BBB/HHH/HHB/HBB × seeds 0–4; deterministic sample-ID keyed Gaussian/salt-pepper/masking; 260 condition rows; full severity accuracy/degradation AUC with paired bootstrap CIs; dual reconstruction, z cosine and prediction JS; all component/noise/checksum gates PASS | None for frozen clean-trained MNIST models; AUC is a post-hoc supplement, while adversarial, non-stationary and cross-dataset robustness remain extensions |
| Phase 7 / Q4: update mechanisms | **Complete — formal five-seed frozen-snapshot analysis** | HHH/HHB/HBB shared Hebbian boundaries and BBB/HHB/HBB BP-side gradients across seeds 0–4; three validated layer snapshots; 50 fixed batches; raw/effective updates; alignment, norm ratio, alpha-star, scale-matched bias, SNR and exploratory Q1–Q3 correlations; all hashes/integrity gates PASS; 120 tests; zero analysis optimizer steps and zero test access | None; early/late width-dependent Q4 extensions are now complete; correlations remain descriptive because shared-prefix rows are non-independent |
| Stage 2B: output-filter update-centering audit | **Complete — candidate failed** | Supplied notebook audited; exactly one validation-only seed-42 candidate; 70 tests; full health/Q4 comparison; zero test access | No replacement selected; Stage 1B remains frozen |
| Stage 2B decision overlay | **Complete — Branch D** | Unified snapshot/layer table, machine-readable branch decision, source/checksum manifest, immutable tests, zero new training and zero test access | Superseded only for the separately authorized two-control depth diagnostic |
| Stage 2C: Hybrid Hebbian–BP depth ablation | **Complete — PASS, Outcome D** | Four newly paired clean-commit runs; pairing/freeze PASS; 2,000-image layerwise metrics; five figures; 78 tests; zero test access | Historical confirmation requirement was tested in Stage 2D and later superseded only by Stage 3 protocol v1 |
| Stage 2D: Hybrid-HHB confirmation | **Complete — CONFIRMATION FAILED** | Six paired BBB/HHH/HHB validation runs; system and standardized decoders; both accuracy/rank gates passed; seed 43 standardized MSE ratio `1.5688 > 1.25`; pairing/checksums PASS; 84 tests; zero test access | Do not add a third seed or alter thresholds; preserve HHB as a mechanistic rank/classification result, not a confirmed formal candidate |
| Stage 3 / Phase 4 formal core matrix | **Complete — technical gate PASS and one-time test complete** | BBB/HHH/HHB/HBB/Full Random × seeds 0–4; 25 system runs, 25 standardized decoders and probes; all pairing/checksum gates PASS; 25 frozen test evaluations; 93 tests | None; RBB/RRB attribution and Q2–Q6 are now complete |
| Stage 3 / Q1 complete matrix and matched-prefix attribution | **Complete** | Core five methods plus RBB/RRB × seeds 0–4; 35 system/probe/standardized-decoder records; both freeze gates PASS; one-time test complete; five paired contrasts, effect sizes, bootstrap CIs, budgets and figures; 101 tests | None; Q1 supports small classification value for Hebbian prefixes at L64/balanced but no standardized-reconstruction advantage over matched random prefixes |
| Phase 8 / Q5–Q6: dimension/asymmetry | **Complete — formal five-seed sweeps and mechanism analysis** | BBB/HHH/HHB/HBB × seeds 0–4 for `L=[16,32,64,128]` and early/balanced/late widths; 100 new system/probe/standardized-decoder records plus reused L64/balanced core; five new freeze gates PASS; 100 one-time test records; 300 layer rows; 1,300 noise rows; early/late Q4 snapshots; interactions, sensitivity, CKA, figures, training cost and 138-test final suite | None for frozen MNIST Q5/Q6; matched-prefix value attribution remains restricted to the preregistered L64/balanced RBB/RRB controls; Phase 9 extensions are optional |
| Final statistical/protocol audit | **Complete — PASS** | Stage 2D post-confirmation re-scoping confirmed; paired seeds/CI/contrast roles/test timeline audited; late-heavy seed 4 retained; 135 standardized-decoder configs matched; Q1 samples-seen curve and Q3 curve AUC derived artifact-only; 142/142 tests passed; zero test-access increment | None; preserve primary/secondary/exploratory wording in the paper |
| v1.0-final release Phase 1 | **Complete — compact evidence frozen; awaiting review** | Release manifest, artifact registry, SHA-256 inventory, 43 accepted source CSV/JSON copies plus a 135-config hash/index table; artifact-only verifier; no dataset/checkpoint load, training, model evaluation or test-access increment | Phase 2 figures must not start until the Phase 1 review is accepted; GitHub authentication and licensing remain deferred to Phase 5 |
| Phase 9: extensions | Optional / not started | Scope listed | Requires a new frozen protocol; the core MNIST matrix is complete |

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

- The supplied Hebbian notebook is now locally verified by filename, size,
  access date, and SHA-256. Its upstream URL/version and license or sharing
  terms remain unknown.
- Written confirmation from the BP teammate that their implementation is
  `phase0-v1 compliant` has not been attached.

These remaining fields are tracked in `docs/tutorial_migration.md` and
`docs/phase0_team_confirmation.md`. They must not be inferred or self-certified
by the Hebbian implementation. They do not authorize protocol drift and are
not blockers for the local canonical source freeze.

## 5. Phase 1 implementation status

### Complete

- Explicit `compute_local_update()` and `apply_local_update()` under
  `torch.no_grad()`.
- Per-sample, per-spatial-location channel top-k WTA.
- Local Hebbian correlation plus Oja stabilization.
- Per-output-filter L2 normalization after each applied update.
- Greedy layer isolation and frozen earlier-layer checks.
- Candidate computation does not mutate weights.
- Matched-state tests produce Hebbian and raw BP-reference candidates without
  changing the snapshot or populating parameter gradients.
- Tiny deterministic Oja update matches a hand calculation.
- Encoder parameters are excluded from gradient-based representation updates.
- Labels do not enter the main representation-learning path.
- `learning_rate=0` is an exact no-op, including no normalization side effect.
- A 500-update stress test remains finite with unit filter norms.
- A fixed-seed multi-step update trajectory is bitwise reproducible.
- Active-filter and maximum-winner-share thresholds produce a legacy
  `collapse_detected` field in every Hebbian epoch record. Stage 1 showed that
  its `active_neuron_ratio` is dataset-wide winner coverage, not per-location
  WTA density; future reporting must use the explicit names.

### Non-blocking optional extension

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

The run remains exploratory. The historical threshold produced:

| Layer | Final active-neuron ratio | Final winner entropy | Gate interpretation |
|---|---:|---:|---|
| enc1 | 1.0000 | 0.7466 | Legacy detector did not flag active-filter collapse |
| enc2 | 0.4688 | 0.5711 | Participation falls; needs Stage 1 context |
| enc3 | 0.2031 | 0.6167 | Unresolved; active ratio alone is not a valid collapse verdict |

Stage 1 resolved the ambiguity around `active_neuron_ratio=0.109375`.
At `z`, `7/64=0.109375` is the expected number of winners per sample, but the
same seven Hebbian units win for every one of the 2,000 validation images.
Winner coverage is therefore only 0.109375 when its healthy expectation over
the dataset is near 1; effective rank is 1.0186. Q1 Hebbian seeds 0 and 1
repeat the fixed-seven and rank-near-one pattern, whereas their BP effective
ranks are 20.94 and 21.69. This is confirmed pathological winner
concentration plus representation degeneracy, not normal WTA sparsity.

A paired
random-encoder **linear-probe** control is now included in the formal Q1
pipeline; the two completed seeds average 82.765% test accuracy. The older
89.00% Hebbian result remains development evidence and is not mixed with the
new frozen-configuration Q1 runs.

## 7. Research-question status

| Question | Current answer | Required before claiming an answer |
|---|---|---|
| Q1: classification and reconstruction performance | **Formally answerable.** Seven-condition five-seed matrix complete. Test accuracy: BBB 0.91584, HHH 0.89410, HHB 0.91986, HBB 0.91638, Random 0.80804, RBB 0.91528, RRB 0.91640. HBB−RBB is +0.00110 [0.00036, 0.00180]; HHB−RRB is +0.00346 [0.00204, 0.00554]. Hebbian prefixes add a small classification benefit, but matched standardized-MSE contrasts are approximately zero with CIs crossing zero | Complete for the frozen MNIST reference; Q3/Q5/Q6 now document its noise, dimension, and width dependence |
| Q2: latent representations | **Formally answerable.** Five-seed layerwise analysis confirms HHH h2/z rank near one; HHB leaves h1/h2 unchanged but raises z rank by +10.525 [9.845, 11.075], z linear probe by +0.0318, and z k-NN by +0.0941 vs HHH. HBB repairs from h2 and reaches near-BBB z rank. Matched random controls show rank is not equivalent to class information | Complete for L64/balanced, with architecture-dependent compensation now answered by Q6 |
| Q3: robustness | **Formally answerable.** At severity 0.4 HBB has the highest accuracy for all three corruptions. Whole-curve AUC confirms HBB is strongest for Gaussian and masking; HHB and HBB have essentially equal salt-and-pepper degradation AUC. HHB remains more Gaussian/masking-sensitive than HBB/BBB, and HHH's near-one z cosine is a collapse artifact | Q3 complete for frozen clean-trained MNIST; curve AUC is explicitly supplementary, with no claim about adversarial, non-stationary or cross-dataset robustness |
| Q4: weight updates | **Formally answerable.** Five-seed frozen-snapshot analysis shows effective Hebbian–BP alignment falls from Enc1 `0.3680±0.1265` to Enc2 `0.0356±0.0411` and Enc3 `0.000046±0.000276`; scale-matched bias approaches one and Enc2 SNR collapses. HHB/HBB replace the failing deep local directions with BP gradients, matching Q2's z/h2 rank-repair boundaries | Complete for L64/balanced; early/late width extensions are included in Q6, and snapshot correlations remain non-causal |
| Q5: dimension/asymmetry performance | **Formally answerable.** Added dimension helps BP/Hybrid usable rank and reconstruction, but HHH z rank stays approximately one from L16 to L128. L16 and late-heavy are seed-unstable; HBB is the most stable Hybrid in late-heavy. Noise effects are metric- and corruption-specific. Performance interactions are not significant with five seeds/high variance, while dimension × method representation interactions are strong | Complete for frozen MNIST and the preregistered grid; do not generalize prefix-value attribution beyond the L64/balanced matched controls or claim a universal performance ordering |
| Q6: asymmetry and representations | **Formally answerable.** HHB raises z rank after a collapsed Hebbian h2 in every width allocation, but late-heavy weakens and destabilizes compensation; HBB repairs one layer earlier and is more stable. Width strongly changes h2 winner entropy/coverage and cross-architecture CKA, while HHH Enc3 remains near-zero BP alignment and rank one | Complete for the frozen three-layer widths; snapshot associations do not establish that alignment/SNR alone causally determines representation quality |

## 8. Frozen decisions

- Existing seed-0 test results are development evidence and must not be used to
  select future hyperparameters.
- Formal tuning uses only `tuning_seed=42` and validation metrics.
- Phase 0 v1.1 overrides the parent BP default: Adam learning rate is `0.003`;
  BP learning-rate tuning is not repeated.
- Test representations and metrics are evaluated only after every
  validation-only choice is frozen and the best validation checkpoint restored.
- BBB/HHH/HHB/HBB/RBB/RRB must share forward architecture, parameter shapes
  and evaluation code within each experiment variant.
- The fixed three-layer encoder defines total depth; BBB/HBB/HHB/HHH define
  0/1/2/3 Hebbian layers. HBB/HHB are Hybrid interventions, not Full Hebbian.
- Full Random is only an overall lower bound. Claims that one or two Hebbian
  layers add value require the matched `HBB−RBB` and `HHB−RRB` contrasts.
- Reconstruction must be reported as both **system reconstruction** and
  **standardized-decoder reconstruction**. The latter freezes the completed
  encoder and retrains a new decoder from paired initialization with identical
  optimizer, data, epochs and validation selection; encoder-information claims
  cannot rely on Hybrid system reconstruction alone.
- Any material change to data, shapes, loss, training budget, probe, or noise
  realization requires a documented standard-version change and paired reruns.
- Existing dirty/historical runs remain preliminary and cannot be copied into
  the formal artifact tree.
- The approved Stage 3 v1 decision supersedes only the old requirement that
  Stage 2D must pass every reconstruction gate before formal work can start.
  Stage 2D remains `CONFIRMATION FAILED`. HHB enters Stage 3 only as
  **a confirmed rank-repair candidate with unresolved reconstruction
  stability**, never as a confirmed complete repair.
- Standardized-decoder reconstruction is now a formal outcome and cannot be
  used to remove HHB, a seed, or any other preregistered condition.
- Stage 1 decision is FAIL. The selected Hebbian `lr=0.0005,
  winner_fraction=0.10` config is not approved for formal runs.
- Stage 1B is frozen as **COMPLETED — NO CANDIDATE PASSED**. Its eight
  preregistered validation-only candidates accessed zero test samples. No
  v3/v4 candidates may be added and no replacement formal config was selected.
- Stage 1C is a metric audit only. It may read the existing checkpoint and
  fixed validation subset, but must not train, update, retune, construct test
  features, or reopen Stage 1B.
- Stage 1C is complete with metric validity PASS. Its effective-rank result
  measures severe raw-covariance anisotropy/channel redundancy. It must not be
  used alone as evidence that linearly decodable class information is absent.
- Q4 use of the seed-42 snapshot is mechanism analysis of a failed health case;
  it does not approve that checkpoint/config for confirmatory Q1 runs.
- Stage 2 / Q4 seed-42 tooling gate is complete with PASS. This PASS covers
  implementation, fixed-batch sampling, metric correctness, stored updates,
  and state integrity only; it does not reverse Stage 1/1B or complete formal
  multi-seed Q4.
- The notebook-inspired output-filter update-centering audit is complete with
  `DOES_NOT_RESOLVE_FAILURE`. Its sole seed-42 candidate failed the frozen
  validation-accuracy and representation-health gates, so it cannot replace
  the original Oja + WTA baseline. No follow-up candidate or threshold change
  is authorized by this result.
- The frozen follow-up decision is
  `BRANCH D — FREEZE AS FAILURE-CASE BASELINE`. Record
  `COMMON-MODE UPDATE REMOVAL: NOT SUFFICIENT`; no new seed-42 repair run or
  Stage 3 execution is authorized by that repair branch alone.
- Stage 2C remains seed-42 diagnostic evidence. Stage 2D uses exactly
  validation-only seeds 43 and 44; it cannot tune, change thresholds, read
  test, or append a third confirmation seed after a failure.
- Stage 2D is complete with `CONFIRMATION FAILED`. Seed 43 failed only the
  standardized-decoder reconstruction ratio (`1.5688 > 1.25`); seed 44 passed
  all gates. Both seeds passed accuracy, z-rank, h2→z compensation, pairing,
  checksum, numerical-integrity and zero-test-access checks. The formal matrix
  remains blocked.
- The preceding “formal matrix remains blocked” statement is retained as the
  historical Stage 2D decision. On 2026-07-28, the user explicitly approved
  Stage 3 protocol v1, which authorizes the formal matrix under the narrower
  rank-repair wording and technical-only pre-test gate.

## 9. Immediate next actions

The frozen MNIST Q1–Q6 experiment program is complete. Release Phase 1 has now
frozen a compact, tracked evidence bundle from accepted aggregate tables only.
Stop for review before Phase 2: do not generate the six final figures until the
artifact registry, checksums, governance history and exclusions are accepted.
Phase 9 experiments (CIFAR-10, non-stationary data, noisy training, alternate
Hebbian rules, or a local decoder) remain optional extensions requiring a new
protocol; they must not alter or be mixed into the completed v1 results.

Tutorial provenance and teammate confirmation remain useful archival items,
but they are not open scientific gates for Q1–Q6.

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
| 2026-07-23 | Stage 1 completed with FAIL: corrected density-versus-coverage definition; selected Hebbian `z` has fixed 7/64 winners and effective rank 1.0186; no test access | `docs/representation_health_gate.md`, `evaluation/representation_health.py`, local formal gate artifacts |
| 2026-07-23 | Stage 1B frozen as **COMPLETED — NO CANDIDATE PASSED** after all eight preregistered v1/v2 validation candidates completed; no v3/v4 candidates and no test access | `docs/stage1b_hebbian_repair.md`, two tuning manifests, selection records, and immutable 45/47-test logs |
| 2026-07-23 | Stage 1C effective-rank metric audit inserted before Q4; no training or retuning authorized | `HEBBIAN_PROJECT_PLAN.md`, this status file |
| 2026-07-23 | Stage 1C completed with metric validity PASS: pre-WTA PR 1.0186, post-WTA PR 1.0000, same-subset frozen probe accuracy 0.9040; no training, tuning or test access | `docs/stage1c_effective_rank_audit.md`, audit implementation/config/tests, local `audit_v1_1` artifacts |
| 2026-07-23 | Stage 2 / Q4 seed-42 tooling gate passed on the authorized failure-case snapshots: 50 fixed batches, raw BP and raw/effective Hebbian updates, full metrics/tensors, 62 tests, zero analysis optimizer steps, unchanged hashes, and zero test access | `docs/q4_update_mechanism_seed42.md`, `evaluation/update_analysis.py`, `evaluation/run_q4_tooling.py`, local `seed42_v1` artifacts |
| 2026-07-25 | Audited the supplied notebook and completed the single seed-42 output-filter update-centering experiment; the candidate failed both frozen gates and is not eligible to replace the baseline | `docs/output_filter_centering_mechanism.md`, output-centering configs/tests, immutable 70-test log, and local comparison artifacts |
| 2026-07-25 | Applied the frozen follow-up decision tree and selected Branch D; stopped current Oja repair and froze the original configuration as a health-gate failure-case baseline | `docs/hebbian_failure_case_protocol_addendum.md`, `results/hebbian_followup_decision/` |
| 2026-07-25 | Completed Stage 2C Hybrid-HHB/Hybrid-HBB depth ablation with clean pairing, zero test access and Outcome D; froze Hybrid-HHB as confirmation candidate | `docs/hybrid_depth_ablation_protocol.md`, `docs/hybrid_depth_ablation_results.md`, local `results/hybrid_depth_ablation/` |
| 2026-07-27 | Expanded the research design to Full BP, Full Hebbian and Minimal Hybrid Credit Assignment in a fixed three-layer encoder; added HBB/HHB Hebbian-depth comparisons and RBB/RRB matched random-prefix controls | `HEBBIAN_PROJECT_PLAN.md`, this status file |
| 2026-07-27 | Added a reconstruction fairness control: report actual system reconstruction and paired standardized-decoder reconstruction separately | `HEBBIAN_PROJECT_PLAN.md`, `README.md`, this status file |
| 2026-07-27 | Completed Stage 2D on validation-only seeds 43/44. Both seeds passed accuracy and representation gates, but seed 43 failed the paired standardized-decoder MSE ratio; final decision `CONFIRMATION FAILED`, zero test access | `docs/hybrid_hhb_confirmation_results.md`, local `results/hybrid_hhb_confirmation/` artifacts |
| 2026-07-28 | Approved Stage 3 protocol v1: HHB enters as a confirmed rank-repair candidate with unresolved reconstruction stability; standardized reconstruction becomes a formal outcome rather than an entry gate | `docs/stage3_formal_protocol_v1.md`, `configs/experiments/stage3_formal_core_v1.yaml` |
| 2026-07-29 | Completed Stage 3 formal core: all 25 validation-only conditions and standardized decoders/probes, technical freeze gate PASS, then 25 one-time frozen test evaluations; HHB repaired classification but not the standardized reconstruction gap | `docs/stage3_formal_core_results.md`, local `results/formal/phase0_v1_1/stage3_core/` artifacts |
| 2026-08-09 | Completed formal Q5/Q6 dimension and matched-encoder-parameter width sweeps: five new case gates PASS, all one-time test/representation/noise/update outputs complete, aggregate integrity PASS, training costs recorded, and 138/138 final tests passed | `docs/stage3_q5q6_results.md`, `configs/experiments/stage3_q5q6_sweeps_v1.yaml`, local `results/formal/phase0_v1_1/stage3_q5q6_sweeps/` artifacts |
| 2026-08-10 | Final artifact-only audit PASS: confirmed Stage 2D→Stage 3 post-confirmation amendment, added Q1 samples-seen curves and Q3 whole-severity AUC/paired CIs, retained late-heavy seed 4, verified all 135 standardized-decoder configs, and passed 142/142 tests; no training, checkpoint/data load or new test access | `docs/final_statistical_protocol_audit.md`, updated Q1/Q3 reports, immutable final-audit test logs, local `stage3_final_audit_supplement/` artifacts |
| 2026-08-10 | v1.0-final Phase 1 compact-evidence freeze completed on `codex/final-release`: accepted aggregate CSV/JSON records copied under a relative-path manifest with SHA-256 checksums, 135 standardized-decoder config hashes indexed, evidence classes/exclusions recorded, and raw/ignored artifacts preserved | `release/v1.0-final/`, `scripts/build_final_release.py`, `scripts/verify_final_release.py`, `tests/test_final_release_bundle.py` |
