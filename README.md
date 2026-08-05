# CA-DTNet

Official reproducibility package for **CA-DTNet: A Trace-Driven
Communication-Aware Digital Twin for Robust High-Resolution Traffic
Forecasting under Heterogeneous V2X Networks**.

## Fast verification

```bash
python -m pip install "numpy>=1.26,<3"
python scripts/verify_results.py
```

This recomputes the released point-forecast metrics from prediction arrays,
rebuilds the three-seed aggregates, and verifies the headline 24/27 result.

## Repository contents

- `src/`: pNEUMA processing, communication standardization, trace replay,
  CA-DTNet, DCRNN, Graph WaveNet, Graph GRU, losses, and metrics;
- `scripts/`: official-dataset acquisition and independent result verification;
- `checkpoints/`: released weights for four models and seeds 42, 52, and 62;
- `artifacts/results/`: full three-seed metrics, statistical tests, and timing;
- `artifacts/predictions/seed42/`: raw arrays for independent metric recomputation;
- `artifacts/robustness/`: Phase 4 and Phase 5 robustness tables;
- `manuscript/`: LaTeX source, tables, data summaries, and vector figures.
- `SHA256SUMS`: integrity hashes for every released checkpoint and result artifact.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full protocol. Raw public
datasets are downloaded from their official sources and are intentionally not
duplicated in this repository.

## Citation

Please cite the accompanying manuscript. Bibliographic metadata will be added
after publication.
