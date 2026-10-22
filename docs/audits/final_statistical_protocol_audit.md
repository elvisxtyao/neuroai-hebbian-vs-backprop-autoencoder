# Final Statistical and Protocol Audit

Date completed: 2026-08-10

Audit analysis source: `069c1870a9bad625bd89c8041545fc94ce258f8e`

Decision: **PASS**

This was an artifact-only audit. It loaded no dataset or checkpoint, performed
no training or model evaluation, created no experimental condition, and added
zero test accesses.

## 1. Stage 2D to formal Stage 3 history

The required amendment already exists in
`docs/stage3_formal_protocol_v1.md`, committed on 2026-07-28 after the completed
Stage 2D experiment. The historical sequence is:

1. Stage 2D completed on validation-only seeds 43/44 with
   **`CONFIRMATION FAILED`**. Seed 43 failed the preregistered standardized
   reconstruction ratio; seed 44 passed. Thresholds were not changed and no
   third seed was added.
2. The later Stage 3 protocol is a **post-confirmation re-scoping / protocol
   amendment**, not a successful preregistration result.
3. The amendment changed standardized reconstruction from an entry hard gate
   into a formal outcome and admitted HHB only as “a confirmed rank-repair
   candidate with unresolved reconstruction stability.”
4. Stage 2D seeds 43/44 were excluded from formal seeds 0–4.

The original confirmation report and its blocked decision remain unchanged.
The final paper should cite both the failed Stage 2D result and the subsequent
Stage 3 amendment; it must not imply that HHB passed the original conjunctive
gate.

## 2. Pairing and confidence intervals

- Formal paired seeds are exactly `[0,1,2,3,4]`.
- Pairing unit is the seed, not batches, epochs, images, or duplicated shared
  prefixes.
- Reported mean CIs use a nonparametric bootstrap of five seed-level values.
- Paired contrast CIs bootstrap the five seed-level paired differences.
- Bootstrap settings are 10,000 resamples, seed 2026, 95% confidence.
- With only five seeds, intervals and Cohen's dz are descriptive and should not
  be presented as high-powered population estimates.
- Q5/Q6 interaction F tests use a seed block. Unadjusted p-values outside the
  frozen contrasts remain exploratory; no family-wise multiplicity claim is
  made.

## 3. Primary and exploratory contrast roles

The Stage 3 protocol defines three primary contrasts:

- HHB − HHH: whether BP Enc3 repairs Full Hebbian;
- HBB − HHB: incremental value of moving BP credit assignment to Enc2;
- BBB − HHB: residual gap from HHB to Full BP.

Two additional contrasts are protocol-mandated **secondary attribution
controls**, not headline primary contrasts:

- HBB − RBB;
- HHB − RRB.

Full Random comparisons are descriptive lower-bound checks. Q4 cross-outcome
correlations are exploratory. The newly requested Q3 whole-curve AUC is also a
post-hoc supplementary summary; it does not replace the frozen severity-level
outcomes.

The implementation constant groups all five confirmatory/mandatory contrasts
together for table generation. This naming convenience does not change their
protocol roles above.

## 4. Test-usage timeline

The timeline is consistent with validation-only selection:

- 2026-07-27: Stage 2D decision completed with zero test samples.
- 2026-07-28: post-confirmation Stage 3 protocol amendment committed.
- 2026-07-29 07:40 UTC: core validation-only freeze gate PASS with zero test
  access.
- 2026-07-29 07:47 UTC: one-time core test evaluation completed.
- Later Q1 aggregation loaded immutable tables only and incremented test access
  by zero.
- Q2/Q3 used the already frozen models as outcome analyses and did not select
  models or hyperparameters.
- Every new Q5/Q6 case records freeze PASS and zero pre-freeze test access
  before its completed test table. L64/balanced reused the core and was not
  evaluated a second time.
- This final supplement reads existing CSV/JSON artifacts only; its test-access
  increment is zero.

No evidence of validation/test leakage or test-driven condition removal was
found.

## 5. Late-heavy seed 4

Late-heavy seed 4 passes every technical check:

- four methods complete and finite;
- same clean source commit, split, model, decoder and probe pairing;
- Hebbian prefix pairing valid;
- frozen layers and standardized encoders unchanged;
- zero pre-freeze test access;
- completed one-time test rows.

Its test accuracies are BBB 0.6446, HHH 0.8898, HHB 0.3380 and HBB 0.8330.
These unusual results are genuine formal outcomes under the technical-only
gate. The seed must not be excluded, retried, or relabelled. All reported
architecture means, SDs, CIs and interaction tests retain it.

## 6. Standardized-decoder fairness

The audit checked all **135** formal core, matched-control, dimension and width
resolved configs. Every config uses the same standardized-decoder contract:

- frozen completed encoder;
- paired decoder initialization;
- Adam with learning rate 0.003, betas `[0.9,0.999]`, zero weight decay;
- 10 epochs, batch size 128, frozen MNIST split;
- pixel-mean MSE;
- validation selection by minimum reconstruction MSE;
- encoder checksum unchanged.

All relevant freeze gates confirm paired initialization and unchanged
standardized encoders. System and standardized reconstruction remain separate
outcomes. Encoder-information claims rely on the standardized decoder, while
system reconstruction describes end-to-end performance.

## 7. Supplemental artifact integrity

The accepted local supplement is
`results/formal/phase0_v1_1/stage3_final_audit_supplement/` and contains:

- 350 Q1 samples-seen learning-curve rows;
- 60 Q3 seed-level curve-AUC rows;
- 12 Q3 method/noise summaries;
- 63 paired curve-AUC contrasts;
- source hashes and the machine-readable protocol audit.

All derived values are finite. The integrity record explicitly reports
`datasets_loaded=false`, `checkpoints_loaded=false`,
`training_performed=false`, `model_evaluation_performed=false`, and
`test_access_increment=0`.

## 8. Final verification

The complete repository test suite passed after generating this supplement:

- 142 tests collected and passed;
- 0 failures, 0 errors and 0 skipped tests;
- JUnit record: `verification/phase0_v1_1/stage3_final_audit_junit.xml`;
- console record: `verification/phase0_v1_1/stage3_final_audit_pytest.log`.

Two non-failing environment warnings were retained verbatim in the console
record: joblib fell back from physical-core detection to logical cores, and a
Windows subprocess reader encountered a console-encoding exception. Neither
warning changed a test result or any scientific artifact.
