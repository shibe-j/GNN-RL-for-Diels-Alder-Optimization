# RunPod quickstart

This folder contains a standalone training script and setup helpers for RunPod GPU instances.

## Setup

```bash
cd deepchem
./runpod_setup.sh
```

Optional environment variables:
- `VENV_DIR` to pick a custom venv path
- `TORCH_INDEX_URL` to force a specific PyTorch wheel index

### Conda setup (if your packages require conda)

```bash
cd deepchem
./runpod_setup_conda.sh
```

Optional environment variables:
- `ENV_NAME` to set the conda env name
- `ENV_FILE` to point at a different `environment.yml`

## Train

```bash
cd deepchem
./runpod_train.sh
```

Outputs are saved under `deepchem/outputs/run-YYYYmmdd-HHMMSS/`.

### Train with conda

```bash
cd deepchem
./runpod_train_conda.sh
```

## Common overrides

```bash
./runpod_train.sh \
  --epochs 200 \
  --batch-size 64 \
  --val-batch-size 128 \
  --lr 5e-4 \
  --output-dir outputs/my-run
```
