# Stage 3 Formal Multi-Seed Protocol v1

## 1. Decision and precedence

This protocol is the explicitly approved governance decision that follows the
completed Stage 2D confirmation experiment.

Stage 2D remains historically recorded as `CONFIRMATION FAILED`: seed 43
failed the preregistered standardized-decoder reconstruction ratio while seed
44 passed. This result must not be deleted, relabelled or recomputed with a
post-hoc threshold.

For Stage 3, Hybrid-HHB enters the formal matrix as:

> a confirmed rank-repair candidate with unresolved reconstruction stability

It must not be described as a confirmed complete repair.

This protocol changes the role of standardized-decoder reconstruction from a
pre-Stage-3 hard gate to a formal research outcome. It does not change the
Phase 0 v1.1 data, architecture, learning-rate, pairing, validation-selection
or test-access rules.

## 2. Core conditions

The fixed three-layer encoder uses these rule allocations:

| Method | Enc1 | Enc2 | Enc3 | Role |
|---|---|---|---|---|
| Full BP (`BBB`) | BP | BP | BP | global-credit reference |
| Full Hebbian (`HHH`) | Hebbian | Hebbian | Hebbian | local-rule failure/reference condition |
| Hybrid-HHB (`HHB`) | Hebbian | Hebbian | BP | rank-repair condition |
| Hybrid-HBB (`HBB`) | Hebbian | BP | BP | deeper-credit hybrid condition |
| Full Random | frozen random | frozen random | frozen random | overall encoder lower bound |

The core formal seeds are exactly `[0,1,2,3,4]`.

## 3. Frozen training contract

- Dataset and split: frozen MNIST 50k train / 10k validation / 10k test.
- Architecture: `conv3_ae_v1`, balanced encoder, latent dimension 64.
- Batch size: 128.
- BP: Adam, learning rate `0.003`, 10 epochs.
- Hebbian: explicit Oja + WTA + per-filter L2 normalization, learning rate
  `0.0005`, winner fraction `0.10`, 10 epochs per applicable layer.
- Target clamping: disabled.
- Probe: frozen encoder, standardized train features, single linear layer,
  validation-selected checkpoint.
- System decoder: the decoder produced by the actual method-specific training
  procedure.
- Standardized decoder: freeze the completed encoder, initialize a new decoder
  from the paired seed-specific decoder initialization, and train it with the
  same Adam `0.003`, data order, 10 epochs and validation selection.

Within each seed, all methods must share initial model state, decoder state,
probe initialization protocol, standardized-decoder initialization, split and
data-order rule.

## 4. Test-access boundary

Core representation training, validation selection, probing, standardized
decoder training, pairing checks and artifact checks are validation-only.
They must construct no test loader and must record
`test_samples_accessed=0`.

Only after all 25 core runs and all 25 standardized decoders pass the technical
freeze gate may a separate evaluator load the official test split. Every
frozen checkpoint is evaluated once; test results cannot select, remove,
rename or retrain a condition.

## 5. Technical freeze gate

Stage 3 has no performance eligibility gate. Accuracy, effective rank and both
reconstruction measures are outcomes.

The pre-test gate requires:

- 25/25 core system runs complete;
- 25/25 frozen probes complete;
- 25/25 standardized decoders complete;
- exact seed/method matrix and resolved configs;
- clean, identical source commit for every run;
- paired model, decoder, probe and standardized-decoder initialization;
- HHH/HHB identical Hebbian Enc1/Enc2 prefixes within seed;
- HBB identical Hebbian Enc1 prefix within seed;
- all frozen layers and standardized encoders unchanged;
- complete checkpoints, metrics, metadata and resume state;
- finite stored metrics;
- `test_samples_accessed=0`.

A technical failure may be resumed or rerun with the same frozen config.
Observed model performance may not trigger replacement, tuning or seed
addition.

## 6. Formal outcomes

### Q1: clean performance

Report validation-selection history and one-time test accuracy, macro-F1,
classification CE, system reconstruction MSE, standardized-decoder MSE, AULC,
samples seen and wall-clock. Report per-seed rows, mean ± SD, paired
differences, effect sizes and bootstrap 95% confidence intervals.

Primary contrasts are `HHB-HHH`, `HBB-HHB` and `BBB-HHB`. Full Random is an
overall lower bound.

### Q2: representations

On a frozen class-balanced subset, extract h1/h2/z and report linear probe,
k-NN, class geometry, effective/stable rank, covariance spectra, winner
coverage/entropy, PCA, fixed-parameter UMAP and h2-to-z compensation.

### Q3: robustness

Evaluate clean-trained frozen models with identical sample-level Gaussian,
salt-and-pepper and masking noise realizations at the preregistered
severities. Report classification, both reconstruction protocols and
representation degradation.

### Q4: update mechanisms

Use stored frozen snapshots and the same 50 training mini-batches. Report raw
BP directions, raw/effective Hebbian candidates, cosine, norm ratio,
`alpha*`, scale-matched bias and cross-batch SNR. No analysis optimizer step is
allowed.

### Q5/Q6: dimension and asymmetry

After the core matrix is frozen, run BBB/HHH/HHB/HBB for latent dimensions
`[16,32,64,128]` and for the three frozen width allocations at L64:
early-heavy `[64,28,64]`, balanced `[16,32,64]`, and late-heavy
`[4,33,64]`. Reuse the balanced-L64 core runs rather than duplicating them.

## 7. Matched random-prefix controls

Full Random cannot establish whether one or two Hebbian prefix layers add
value. Claims about Hebbian-prefix value require the secondary matched
controls `RBB` and `RRB`, using the same five paired seeds. These controls are
not headline learning paradigms, but the contrasts `HBB-RBB` and `HHB-RRB`
are mandatory before making a net-value claim.

## 8. Stop and reporting rules

- Do not reuse the historical seeds 0–1 exploratory results.
- Do not include Stage 2D seeds 43/44 in the formal five-seed estimates.
- Do not tune from test results.
- Do not drop a seed because it is unfavorable.
- Do not turn standardized reconstruction back into an entry gate.
- Do not call HHB a complete repair.
- Preserve failures, resumes, logs and exact source provenance.
