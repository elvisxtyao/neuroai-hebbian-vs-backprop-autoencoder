# Reproducibility Guide

This guide separates public artifact reproduction from maintenance of the full historical archive and from expensive retraining. The recommended path is Tier 1. It uses only committed compact evidence and does not load MNIST, checkpoints, raw representation arrays, update tensors, `results/`, or `_recovery/`.

The frozen research environment used Python 3.11.6. Python 3.11 is recommended for all commands below. Run commands from the repository root.

## Tier 1 — Artifact-only reproduction

### What this tier verifies

Tier 1 checks:

- the release-manifest schema and required evidence tables;
- SHA-256 checksums for every compact bundle file;
- manifest/bundle provenance records and byte counts;
- formal seeds `[0, 1, 2, 3, 4]`, with seed as the statistical unit;
- frozen row and required-column contracts;
- Stage 2D/Stage 3 governance fields and the no-new-test-access release record;
- the final figure source manifest, source-table hashes, plotted-value tables, and committed figure hashes.

It then supports regenerating all final figures and executing the project walkthrough. No training or model evaluation occurs.

### Minimal environment

Artifact verification itself uses the Python standard library. Figure regeneration and the notebook require the lightweight packages in `requirements-release.txt`; PyTorch, torchvision, scikit-learn, MNIST, and the training archive are not Tier 1 dependencies.

```bash
python -m venv .venv
python -m pip install -r requirements-release.txt
```

Installing missing packages may require a package source. Once the environment exists, every verification, figure, notebook, and test command below runs without network access.

### 1. Verify the compact release

```bash
python scripts/verify_final_release.py
```

Expected result: JSON with `"decision": "PASS"`, formal seeds `[0, 1, 2, 3, 4]`, 46 checked bundle files, 17 plotted-value tables, and `"test_access_increment": 0`. This single command verifies `manifest.json`, `checksums.sha256`, row/schema contracts, provenance records, and final figure sources.

The verifier is deliberately artifact-only. It imports neither training nor evaluation pipelines and has no code path to a dataset or checkpoint. `RELEASE_NOTES.md` is presentation metadata and intentionally sits outside the frozen scientific checksum set.

### 2. Inspect plotted values

The exact CSV tables supplied to the publication plots are committed under `figures/final/plotted_values/`. List them with:

```bash
python -c "from pathlib import Path; print(*sorted(p.as_posix() for p in Path('figures/final/plotted_values').glob('*.csv')), sep='\n')"
```

These files are derived only from compact accepted tables. Their hashes and source relationships are recorded in `figures/final/source_manifest.json`.

### 3. Regenerate all final figures

Choose a new, nonexistent output directory; the builder refuses to overwrite an existing directory or write inside the frozen release bundle.

```bash
python scripts/plot_final_figures.py --output build/v1.0-final-figures
```

Expected result: six PNG/PDF hero-figure pairs, the method-design schematic, 17 plotted-value CSVs, and a new source manifest. The builder verifies the release before reading it. It performs no interpolation, dataset access, checkpoint loading, training, or model evaluation.

Binary PNG/PDF identity can vary with Matplotlib/font metadata across platforms. Reproduction is therefore checked through source hashes, plotted-value equality, panel/data contracts, and valid vector/raster outputs rather than requiring regenerated image bytes to match across environments.

### 4. Run the quick demo

The committed [project_demo.ipynb](project_demo.ipynb) is already executed so GitHub can display its outputs. To execute a fresh copy without modifying the committed notebook:

```bash
python scripts/execute_demo_notebook.py --input project_demo.ipynb --output build/project_demo.executed.ipynb --working-directory .
```

The notebook reads Q1, Q2, Q3, and Q4 compact tables through the shared final-figure loader. It writes only the requested output notebook.

### 5. Run release-safe tests

```bash
python -m pytest -p no:cacheprovider tests/test_final_release_bundle.py tests/test_final_figures.py tests/test_release_narrative.py tests/test_phase4_reproducibility.py
```

These tests require neither `results/` nor MNIST. They verify the public release, plotted values, notebook structure/execution state, documentation links, and dependency boundary.

### 6. Run the clean-export audit

```bash
python scripts/qa_clean_export.py
```

This creates an isolated temporary tree from tracked and pending non-ignored files, confirms that no `results/` tree is present, and repeats release verification, figure generation, notebook execution, and release-safe tests. It also scans public text/notebook sources for private absolute paths and common secret/token patterns. The temporary tree is removed after the audit.

### Expected runtime

On a typical current CPU with the environment already installed:

| Step | Approximate runtime |
|---|---:|
| Compact release verification | under 5 seconds |
| Figure regeneration | about 10–30 seconds |
| Notebook execution | about 10–30 seconds |
| Release-safe tests | about 5–20 seconds |
| Complete clean-export audit | about 1–2 minutes |

Runtime depends on filesystem and plotting backend. None of these steps uses a GPU.

## Tier 2 — Existing full local archive verification

This optional maintainer tier applies only when the ignored historical archive is already present. It is not required for GitHub users and is not part of the compact release.

### Expected local layout

```text
results/
├── formal/phase0_v1_1/
│   ├── stage3_core/
│   ├── stage3_matched_controls/
│   ├── stage3_q1_complete/
│   ├── stage3_q2_representation/
│   ├── stage3_q3_noise/
│   ├── stage3_q4_updates/
│   ├── stage3_q5q6_sweeps/
│   └── stage3_final_audit_supplement/
├── hybrid_hhb_confirmation/
├── hybrid_depth_ablation/
└── ... historical diagnostic/exploratory material
```

At the Phase 1 freeze, the local `results/` inventory contained 11,200 files and 11,083,515,220 bytes; the formal subtree contained 9,085 files and 9,453,570,385 bytes. Those totals are provenance metadata, not files distributed in the compact GitHub release.

### Read-only maintainer verification

```bash
python scripts/verify_local_archive.py
```

The maintainer verifier first runs Tier 1, then checks accepted non-recovery artifact-root inventories, all 43 recorded aggregate source hashes, all 135 standardized decoder resolved-config fingerprints, and frozen governance-document fingerprints. Excluded and `_recovery/` roots are not opened or used as evidence. The script does not load checkpoint contents, datasets, representation archives, or update tensors.

This tier verifies that accepted aggregate outputs and configurations match the compact bundle. It does not claim a bitwise audit of every large binary checkpoint.

## Tier 3 — Expensive historical full reproduction

> **Expensive historical full reproduction — documentation only. Do not run this tier as part of artifact verification.**

Full training requires the training dependencies in `requirements.txt`, the frozen environment described in `environment/phase0_v1_1_environment.md`, and a local MNIST copy compatible with the committed split manifests. The captured host used Windows, Python 3.11.6, CPU-only PyTorch, and deterministic-algorithm settings. Exact captured direct versions are in `environment/phase0_v1_1_requirements.txt`.

### Data and seed policy

- Dataset: MNIST, with the frozen 50,000/10,000/10,000 train/validation/test split.
- Split identity: `data/splits/mnist_split_v1.npz`, SHA-256 recorded in the environment snapshot.
- Formal seeds: exactly `[0, 1, 2, 3, 4]`; do not add or replace seeds.
- Seed is the paired statistical unit.
- Validation-only training and selection must complete before the technical freeze gate.
- The official test split is loaded only by the post-freeze evaluators and is never used to select, remove, retry, or tune a run.

### Major historical entry points

The principal commands, in protocol order, were:

```bash
# Validation-only core and matched controls; each command writes its freeze gate.
python -m training.run_stage3_formal_core --config configs/experiments/stage3_formal_core_v1.yaml
python -m training.run_stage3_matched_controls --config configs/experiments/stage3_matched_controls_v1.yaml

# Only after the corresponding technical gate is PASS: one-time frozen test evaluation.
python -m evaluation.run_stage3_test_evaluation --results-root results/formal/phase0_v1_1/stage3_core
python -m evaluation.run_stage3_matched_test_evaluation --results-root results/formal/phase0_v1_1/stage3_matched_controls

# Aggregate Q1, then post-freeze representation/noise and fixed-snapshot update analyses.
python -m evaluation.analyze_stage3_q1 --core-root results/formal/phase0_v1_1/stage3_core --controls-root results/formal/phase0_v1_1/stage3_matched_controls --output-dir results/formal/phase0_v1_1/stage3_q1_complete
python -m evaluation.run_stage3_q2_representation --config configs/experiments/stage3_q2_representation_v1.yaml
python -m evaluation.run_stage3_q3_noise --config configs/experiments/stage3_q3_noise_v1.yaml
python -m evaluation.run_stage3_q4_updates --config configs/experiments/stage3_q4_updates_v1.yaml

# Q5/Q6 training sweeps. Balanced/L64 reuses the accepted core.
python -m training.run_stage3_q5q6_sweeps --config configs/experiments/stage3_q5q6_sweeps_v1.yaml --sweep dimension
python -m training.run_stage3_q5q6_sweeps --config configs/experiments/stage3_q5q6_sweeps_v1.yaml --sweep architecture

# Each non-reused case then passes its own gate before its one-time test,
# representation, noise, and accepted update analyses; aggregate only afterward.
python -m evaluation.analyze_stage3_q5q6 --config configs/experiments/stage3_q5q6_sweeps_v1.yaml
python -m evaluation.analyze_stage3_final_audit --output-dir results/formal/phase0_v1_1/stage3_final_audit_supplement
```

The per-case Q5/Q6 post-freeze commands are defined in `evaluation/run_stage3_q5q6_test.py`, `evaluation/run_stage3_q5q6_representation.py`, and `evaluation/run_stage3_q5q6_noise.py`. Historical execution must follow `docs/stage3_formal_protocol_v1.md` and `docs/stage3_q5q6_sweep_protocol.md`; the abbreviated command list above does not override their gates.

### Standardized decoder protocol

For each completed encoder, freeze the encoder, initialize a new decoder from the paired seed-controlled initialization, train the decoder with Adam at learning rate 0.003 for ten epochs, and select the minimum validation MSE. The 135 accepted resolved configurations have one common contract and are fingerprinted in the compact release. Standardized reconstruction is an outcome, not a Stage 3 eligibility gate.

### Compute, storage, and reproducibility boundary

- The Q5/Q6 protocol estimated approximately 15–20 CPU hours on the captured host.
- The complete local results archive at freeze was approximately 11.08 GB decimal, including approximately 9.45 GB of formal artifacts.
- Actual runtime depends on hardware, filesystem, and library versions.
- The project records deterministic seeds, environment versions, split identities, configs, checkpoints, and freeze gates. It does **not** promise bitwise-identical training across different hardware or library stacks.

For scientific interpretation and governance chronology, see [FINAL_REPORT.md](FINAL_REPORT.md), [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md), and the [final statistical audit](docs/final_statistical_protocol_audit.md).
