# Stage 2C — Hybrid Hebbian–BP Depth Ablation Protocol

Date preregistered: 2026-07-25

Scope: validation-only, diagnostic seed 42, no test access, no confirmation or
formal seeds.

## 1. Purpose

This finite depth ablation asks whether the current representation failure is
primarily introduced by local Hebbian training at Enc3 or has already
accumulated through Enc2. It evaluates exactly two controls and two references.
No third candidate or post-result tuning is permitted.

The earlier Branch-D freeze prohibited automatically adding another Oja repair.
This ablation is a separately and explicitly authorized architectural
diagnostic. Neither hybrid may be described as a pure Hebbian model.

## 2. Preregistered methods

| Method | Enc1 | Enc2 | Enc3 | Decoder | Role |
|---|---|---|---|---|---|
| Full BP | BP | BP | BP | BP | reference |
| Full Hebbian | Hebbian | Hebbian | Hebbian | BP | health-failure reference |
| Hybrid-HHB | Hebbian | Hebbian | BP | BP | Control A |
| Hybrid-HBB | Hebbian | BP | BP | BP | Control B |

All methods use the same `ConvAutoencoder`, latent dimension 64, saved MNIST
split, seed 42 initialization, decoder initialization, validation-only probe,
and checkpoint-selection policy. Because the historical references do not
share the new clean implementation commit and the old Full Hebbian decoder
used BP learning rate `0.001`, both references are rerun.

Frozen values:

- Hebbian learning rate `0.0005`;
- winner fraction `0.10`;
- raw top-k WTA + Oja + per-filter L2 normalization;
- BP Adam learning rate `0.003`, betas `[0.9, 0.999]`, weight decay `0`;
- 10 epochs per trained Hebbian layer;
- 10 epochs for the joint BP suffix/decoder stage;
- frozen standardized linear probe: SGD, 30 epochs, learning rate `0.1`;
- validation performance floor `0.8863`.

## 3. Training and integrity

Hebbian layers are trained greedily in depth order. During the BP stage, only
layers assigned `bp` and the decoder are included in optimizer parameter
groups. Hebbian and explicitly frozen layers have `requires_grad=false`; their
parameter and buffer checksums must remain unchanged.

Each run saves:

- resolved config and source/config/split hashes;
- initial full-model, encoder, decoder and per-layer hashes;
- trainable/frozen parameter manifest;
- immutable epoch checkpoints and atomic resume checkpoint;
- best/last checkpoints, samples seen, steps and wall-clock;
- validation reconstruction MSE;
- frozen-probe validation accuracy, macro-F1 and CE;
- probe initialization hash and encoder checksum;
- `test_samples_accessed=0`.

All four runs must originate from one clean implementation commit. Full-model,
decoder and probe-classifier initialization hashes and the split hash must be
identical.

## 4. Representation analysis

The frozen `mnist_validation_health_v1` subset contains exactly 2,000 samples,
200 per class. At h1, h2 and z the analysis reports:

- participation-ratio and normalized effective rank;
- stable rank;
- mean channel activation variance;
- mean absolute pairwise channel-activation correlation;
- mean absolute filter cosine similarity;
- dead-unit ratio;
- five-fold validation-subset CV linear-probe and 5-NN accuracy;
- between/within-class separation ratio.

Winner coverage, normalized winner entropy and maximum winner share are only
reported for layers trained with the Hebbian rule. Layerwise CV standardization
is fitted inside each training fold. The final frozen probe continues to use
the full training split and validation-only checkpoint selection.

The existing Stage-1 numerical thresholds are unchanged. For BP-trained
layers, WTA-specific checks are not applicable; diversity checks retain their
frozen thresholds. A “clear key improvement” is preregistered as a frozen
effective-rank or normalized-effective-rank check changing from fail in Full
Hebbian to pass in the hybrid. No post-result effect-size threshold is added.

## 5. Frozen decision rule

A hybrid is eligible for confirmation only if:

1. validation accuracy is at least `0.8863`;
2. at least one key rank check changes from fail to pass relative to Full
   Hebbian;
3. all activations and saved metrics remain finite;
4. frozen-layer and pairing gates pass;
5. test access is zero and artifacts are complete.

Selection prefers the smaller BP intervention, Hybrid-HHB, when both controls
are eligible.

- **Outcome D has priority** when the selected hybrid is performance-eligible
  but its complete applicable health gate still fails.
- Otherwise Outcome A is selected when Hybrid-HHB is eligible.
- Otherwise Outcome B is selected when only Hybrid-HBB is eligible.
- Outcome C is selected when neither hybrid is eligible.

Outcome D does not change any threshold. An eligible hybrid remains only a
candidate for separately approved confirmation seeds.

## 6. Acceptance

- [ ] clean implementation commit recorded before training;
- [ ] complete test suite and immutable log pass;
- [ ] four validation-only runs complete on the same commit;
- [ ] pairing and frozen-layer gate pass;
- [ ] fixed-subset quantitative analysis and five core figures complete;
- [ ] machine-readable `decision.json` saved;
- [ ] no test, confirmation seed, formal seed or third candidate accessed.
