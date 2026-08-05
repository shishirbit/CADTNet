"""Train a released model from deterministic prepared window arrays.

Expected NPZ keys are documented by ``--help`` and in REPRODUCIBILITY.md. The
interface keeps data preparation auditable and training independent of Colab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import CADTNet, DCRNN, GraphGRU, GraphWaveNet  # noqa: E402
from src.training import (  # noqa: E402
    EarlyStopping,
    WindowDataset,
    ca_dtnet_loss,
    point_forecast_loss,
    seed_everything,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train from an NPZ containing adjacency plus train_/val_ arrays: "
            "traffic [W,60,N,8], communication [W,60,N,27], target [W,3,N,3], "
            "and clean_last [W,N,8]."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=("ca_dtnet", "dcrnn", "graph_gru", "graph_wavenet"),
                        default="ca_dtnet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_model(name: str, adjacency: torch.Tensor) -> torch.nn.Module:
    constructors = {
        "ca_dtnet": CADTNet,
        "dcrnn": DCRNN,
        "graph_gru": GraphGRU,
        "graph_wavenet": GraphWaveNet,
    }
    return constructors[name](adjacency)


def dataset(payload: np.lib.npyio.NpzFile, prefix: str) -> WindowDataset:
    clean_key = f"{prefix}_clean_last"
    return WindowDataset(
        payload[f"{prefix}_traffic"],
        payload[f"{prefix}_communication"],
        payload[f"{prefix}_target"],
        payload[clean_key] if clean_key in payload.files else None,
    )


def batch_loss(model_name: str, model: torch.nn.Module, batch: dict[str, torch.Tensor],
               device: torch.device) -> torch.Tensor:
    traffic = batch["traffic"].to(device)
    target = batch["target"].to(device)
    if model_name == "ca_dtnet":
        if "clean_last" not in batch:
            raise ValueError("CA-DTNet training requires clean_last arrays")
        outputs = model(traffic, batch["communication"].to(device))
        return ca_dtnet_loss(outputs, target, batch["clean_last"].to(device))
    return point_forecast_loss(model(traffic), target)


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    payload = np.load(args.data)
    adjacency = torch.as_tensor(payload["adjacency"], dtype=torch.float32)
    device = torch.device(args.device)
    model = make_model(args.model, adjacency).to(device)
    train_loader = DataLoader(dataset(payload, "train"), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset(payload, "val"), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = args.epochs or (45 if args.model == "ca_dtnet" else 35)
    stopper = EarlyStopping(patience=6)
    best_state = None
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(args.model, model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in val_loader:
                validation_losses.append(float(batch_loss(args.model, model, batch, device).cpu()))
        validation = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                        "validation_loss": validation})
        if validation <= stopper.best:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(history[-1]))
        if stopper.update(validation):
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "metadata": {
                "model_name": args.model,
                "seed": args.seed,
                "best_validation_loss": stopper.best,
                "training_seconds": time.perf_counter() - started,
                "training_protocol": "combined CICV5G + SEE-V2X impaired training",
                "history": history,
            },
        },
        args.output,
    )


if __name__ == "__main__":
    main()
