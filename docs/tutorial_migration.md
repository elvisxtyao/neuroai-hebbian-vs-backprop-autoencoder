# Tutorial and Notebook Migration Record

Last updated: 2026-07-21

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
| `SRC-HEBB-TUTORIAL-001` | Original Hebbian teaching notebook/tutorial | Intended source for custom-autograd, BP-reference, cosine, and SNR concepts | Pending | Pending | Pending | Blocked: source has not been provided |

### Information required for `SRC-HEBB-TUTORIAL-001`

- File, URL, or repository path;
- author/title where known;
- commit, release, or retrieval date;
- local SHA-256 if a file is supplied;
- license or sharing restriction if known.

Until this row is complete, project reports may describe tutorial-derived ideas
only in general terms and must not claim a verified implementation lineage.

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

## 4. Migration map

| Source concept | Project decision | Formal destination | Current status |
|---|---|---|---|
| One forward model compared under different learning rules | Retain the concept; learning rule must not select another architecture | `models/conv_autoencoder.py`, `learning_rules/__init__.py` | Implemented |
| Shared BP autoencoder baseline | Rewrite as a reusable trainer and validated CLI | `learning_rules/backprop.py`, `training/train_representation.py` | Implemented |
| Frozen linear classifier | Retain with train-only feature standardization and checksum protection | `models/linear_probe.py`, `training/train_linear_probe.py` | Implemented |
| Per-layer update inspection | Retain as structured diagnostics, not notebook print statements | `learning_rules/hebbian.py`, `metrics.csv` | Partially implemented |
| Explicit Hebbian local rule | Implement outside autograd with separate compute/apply operations | `learning_rules/hebbian.py` | Implemented |
| BP reference at a Hebbian state | Recompute raw reconstruction negative gradients on frozen snapshots | Future `analysis/update_analysis.py` or equivalent | Planned |
| Cosine comparison | Name it update alignment, not bias | Future update-analysis metrics | Planned |
| Bias analysis | Use scale-matched relative bias in addition to alignment/norm ratio | Future update-analysis metrics | Planned |
| SNR | Recompute on multiple mini-batch candidates from one frozen state | Future update-analysis metrics | Planned |
| Notebook data split | Do not inherit; use the saved stratified Phase 0 manifest | `data/splits/mnist_split_v1.npz` | Implemented |
| Notebook output as official result | Prohibited; notebooks may demonstrate only | Validated YAML + module CLI + saved run directory | Enforced by policy |

## 5. Prohibited migrations

The following must not enter Q1–Q6 main experiments:

- custom `autograd.Function.backward()` that silently replaces gradients with
  Hebbian updates;
- target clamping or labels in encoder representation training;
- notebook-cell order as hidden training state;
- test-set metrics used for hyperparameter or checkpoint selection;
- transient arrays or screenshots as the only result record;
- calling cosine similarity “bias” without a separate bias definition;
- SNR estimated while weights change between sampled mini-batches.

## 6. Acceptance checklist

- [x] Reference BP notebook filename, hash, size, and access date recorded.
- [x] Initial project-planning brief hash and role recorded.
- [x] Concept-to-module migration table created.
- [x] custom autograd and target clamping marked as prohibited.
- [x] Official experiment-entry policy recorded.
- [ ] Original Hebbian tutorial source/version/hash recorded.
- [ ] Original source license/sharing status checked where available.

The final two items remain blocked until the user or teammate supplies the
actual source.
