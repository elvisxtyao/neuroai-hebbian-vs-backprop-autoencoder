# Tutorial and Notebook Migration Record

Last updated: 2026-07-25

## 1. Purpose

This document records the provenance of teaching/reference material and defines
which ideas may influence the project. It does not make any notebook a runtime
dependency or an official experiment source.

Personal absolute paths are intentionally not committed. Source files remain
outside the repository unless the team has permission and a clear reason to
version them.

## 2. Source inventory

| Source ID | Artifact | Role | SHA-256 | Size | Access date | Status |
|---|---|---|---|---:|---|---|
| `SRC-BP-NB-001` | `NeuroAI_Project.ipynb` | Teammate/reference Phase 0 BP notebook | `AE4E7F61DDC1F59D62AAE341461BDA8144195FFDEB07DDA10BE5FA4D74E7C308` | 23,574 bytes | 2026-07-21 | Verified local file; upstream URL/version not recorded |
| `SRC-PLAN-001` | Initial Hebbian project-planning brief (`pasted-text.txt`) | Planning source, not executable tutorial | `2CFCE328088A0F5D852165669967C00A27980B668F3E853C44DC9AAC4A3FF593` | 21,168 bytes | 2026-07-21 | Verified attachment |
| `SRC-HEBB-TUTORIAL-001` | `Microlearning.ipynb` | Neuromatch teaching source for custom-autograd Hebbian updates and update-comparison concepts | `7D4437AF298A81B128106EF0236AA00D4D9A8108E6199799CFD90026B603ECA3` | 7,426,114 bytes | 2026-07-25 | Verified user-supplied local file; upstream URL/version and sharing terms not recorded |

### Remaining provenance information for `SRC-HEBB-TUTORIAL-001`

- author/title where known;
- upstream URL, commit, or release where available;
- license or sharing restriction if known.

The local artifact and the audited cells can now be cited by hash. The project
must still avoid claims about its upstream release or reuse license until those
details are supplied.

## 3. Verified BP notebook contents

`SRC-BP-NB-001` contains 12 cells covering:

1. frozen Phase 0 settings and reproducibility;
2. deterministic MNIST split;
3. class-balance checks;
4. shared three-layer convolutional autoencoder;
5. BP autoencoder training and validation checkpointing;
6. learning curves;
7. reconstruction evaluation;
8. frozen feature extraction and train-only standardization;
9. frozen linear-probe training;
10. final validation/test classification metrics;
11. compact reproducibility output.

It does not contain the original Hebbian custom-autograd/SNR teaching prototype.
It is therefore recorded as a BP reference notebook, not as the missing
Hebbian tutorial.

## 4. Verified Hebbian tutorial contents

`SRC-HEBB-TUTORIAL-001` contains 105 cells. The relevant
`HebbianFunction` is a linear custom-autograd teaching implementation. Its
active default forms a batch-mean correlation update and then subtracts
`grad_weight.mean(axis=0)`. For a linear weight matrix shaped
`[out_features, in_features]`, this removes the common update direction across
output neurons. The Oja-subtraction implementation is present as commented
alternative code and is not the notebook's active default.

The notebook operation is not equivalent to the repository's centered-input
patch option: the former centers an already formed update across output units,
whereas the latter changes the presynaptic patches before correlation. The
bounded convolutional migration and its negative validation result are
recorded in `docs/output_filter_centering_mechanism.md`.

## 5. Migration map

| Source concept | Project decision | Formal destination | Current status |
|---|---|---|---|
| One forward model compared under different learning rules | Retain the concept; learning rule must not select another architecture | `models/conv_autoencoder.py`, `learning_rules/__init__.py` | Implemented |
| Shared BP autoencoder baseline | Rewrite as a reusable trainer and validated CLI | `learning_rules/backprop.py`, `training/train_representation.py` | Implemented |
| Frozen linear classifier | Retain with train-only feature standardization and checksum protection | `models/linear_probe.py`, `training/train_linear_probe.py` | Implemented |
| Per-layer update inspection | Retain as structured diagnostics, not notebook print statements | `evaluation/update_analysis.py`, `evaluation/run_q4_tooling.py` | Implemented and seed-42 validated |
| Explicit Hebbian local rule | Implement outside autograd with separate compute/apply operations | `learning_rules/hebbian.py` | Implemented |
| Output-neuron/filter common-mode removal | Migrate only as an optional post-Oja candidate over Conv2d `dim=0`, before learning-rate scaling | `learning_rules/hebbian.py` | Implemented, tested, and rejected by frozen gates |
| BP reference at a Hebbian state | Recompute raw reconstruction negative gradients on frozen snapshots | `evaluation/update_analysis.py`, `evaluation/run_q4_tooling.py` | Implemented and seed-42 validated |
| Cosine comparison | Name it update alignment, not bias | `evaluation/update_analysis.py` | Implemented and seed-42 validated |
| Bias analysis | Use scale-matched relative bias in addition to alignment/norm ratio | `evaluation/update_analysis.py` | Implemented and seed-42 validated |
| SNR | Recompute on multiple mini-batch candidates from one frozen state | `evaluation/update_analysis.py` | Implemented and seed-42 validated |
| Notebook data split | Do not inherit; use the saved stratified Phase 0 manifest | `data/splits/mnist_split_v1.npz` | Implemented |
| Notebook output as official result | Prohibited; notebooks may demonstrate only | Validated YAML + module CLI + saved run directory | Enforced by policy |

## 6. Prohibited migrations

The following must not enter Q1–Q6 main experiments:

- custom `autograd.Function.backward()` that silently replaces gradients with
  Hebbian updates;
- target clamping or labels in encoder representation training;
- notebook-cell order as hidden training state;
- test-set metrics used for hyperparameter or checkpoint selection;
- transient arrays or screenshots as the only result record;
- calling cosine similarity “bias” without a separate bias definition;
- SNR estimated while weights change between sampled mini-batches.

## 7. Acceptance checklist

- [x] Reference BP notebook filename, hash, size, and access date recorded.
- [x] Initial project-planning brief hash and role recorded.
- [x] Concept-to-module migration table created.
- [x] custom autograd and target clamping marked as prohibited.
- [x] Official experiment-entry policy recorded.
- [x] Supplied Hebbian tutorial filename, local hash, size, and access date recorded.
- [x] Active update-centering axis and distinction from the commented Oja branch audited.
- [ ] Original source license/sharing status checked where available.

The remaining license/sharing item cannot be inferred from the local file.
