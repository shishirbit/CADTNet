# Reproducibility protocol

The repository separates result verification from full retraining so reviewers
can choose the appropriate cost.

## 1. Verify the released results

```bash
python -m pip install "numpy>=1.26,<3"
python scripts/verify_results.py
```

The script independently recomputes every seed-42 MAE, RMSE, WAPE, and R2
value from the released NPZ prediction arrays, recomputes the three-seed means
and sample standard deviations from the per-seed CSV, and checks the manuscript
claim that CA-DTNet has the lowest mean MAE and WAPE in 24 of 27
domain--horizon--target configurations.

## 2. Inspect or reuse the trained models

Install the full environment with `python -m pip install -r requirements.txt`.
The checkpoints contain the model state, seed, validation loss, training time,
training protocol, and complete experiment configuration. Use
`src.models.load_released_checkpoint` to load them. The published seeds are 42,
52, and 62.

## 3. Rebuild the input data

Run `python scripts/download_datasets_colab.py --project-root <path>`. Raw
pNEUMA, CICV5G, and SEE-V2X data are not redistributed because of their size
and upstream terms. The code records the official sources and prepares the
expected folder structure. `src.pneuma`, `src.communication`, and `src.replay`
implement trajectory parsing, communication-trace standardization, and
contiguous trace replay respectively.

## 4. Retrain from prepared chronological windows

The framework-neutral training entry point accepts a compressed NumPy archive
with `adjacency` and `train_`/`val_` arrays named `traffic`, `communication`,
`target`, and `clean_last`. Their shapes are `[W,60,N,8]`, `[W,60,N,27]`,
`[W,3,N,3]`, and `[W,N,8]`, respectively. Run, for example:

```bash
python scripts/train.py --data prepared_windows.npz --model ca_dtnet \
  --seed 42 --output checkpoints/ca_dtnet_seed_42_retrained.pt
```

Use the same command with `dcrnn`, `graph_wavenet`, or `graph_gru` and repeat
for seeds 52 and 62. The training entry point uses the checkpoint-recorded
optimizer, gradient clipping, epoch limits, early stopping, and loss weights.

## Protocol recorded in the checkpoints

- chronological split: 70% train, 15% validation, 15% test;
- input history: 60 one-second steps;
- horizons: 5, 10, and 30 seconds;
- targets: mean speed, vehicle count, and flow proxy;
- seeds: 42, 52, and 62;
- batch size: 16;
- hidden width: 64;
- optimizer settings: learning rate 0.001, weight decay 0.00001;
- early stopping patience: 6;
- maximum epochs: 35 for point baselines and 45 for CA-DTNet;
- quantiles: 0.1, 0.5, and 0.9;
- CA-DTNet reconstruction-loss weight: 0.3.

Training-only scalers and chronological splitting must be used to avoid future
information leakage. Hardware-dependent timing values should be compared only
on the same device and software stack.
