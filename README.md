# NeuroAI: Hebbian Learning vs Backpropagation in a 3-Layer Autoencoder

Private collaboration repository for a controlled comparison of backpropagation
and biologically plausible Hebbian learning in a three-layer convolutional
autoencoder on MNIST.

This initial commit contains the frozen `phase0-v1` data pipeline, shared model,
BP autoencoder trainer, and frozen linear-probe trainer. Hebbian-specific code
and experiment results are intentionally not part of this initial handoff.

## Included scope

- Deterministic MNIST 50k/10k/10k train-validation-test protocol
- Shared three-layer convolutional encoder and decoder
- Backpropagation autoencoder training with MSE and Adam
- Frozen single-layer linear probe
- Reproducible initialization, configuration validation, and run recording

The binding experiment settings are documented in `PHASE0_STANDARD_V1.md`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the BP autoencoder

```powershell
python -m training.train_representation --config configs/bp_main.yaml
```

The command creates a run directory under `results/`. Use that directory for
the frozen linear probe:

```powershell
python -m training.train_linear_probe `
  --config configs/bp_main.yaml `
  --run-dir results/<bp-run-id>
```

Raw MNIST files, checkpoints, results, virtual environments, and caches are
excluded from version control.
