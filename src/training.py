"""Deterministic training utilities used by CA-DTNet experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class WindowDataset(Dataset):
    """Chronological windows already standardized with training-only scalers."""

    def __init__(self, traffic: np.ndarray, communication: np.ndarray, targets: np.ndarray,
                 clean_last: np.ndarray | None = None):
        if len(traffic) != len(communication) or len(traffic) != len(targets):
            raise ValueError("All arrays must have the same number of windows")
        self.traffic = torch.as_tensor(traffic, dtype=torch.float32)
        self.communication = torch.as_tensor(communication, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.clean_last = None if clean_last is None else torch.as_tensor(clean_last, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.traffic)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {"traffic": self.traffic[index], "communication": self.communication[index],
                "target": self.targets[index]}
        if self.clean_last is not None:
            item["clean_last"] = self.clean_last[index]
        return item


def quantile_loss(prediction: Tensor, target: Tensor, quantiles=(0.1, 0.5, 0.9)) -> Tensor:
    error = target.unsqueeze(-1) - prediction
    q = prediction.new_tensor(quantiles)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def ca_dtnet_loss(outputs: dict[str, Tensor], target: Tensor, clean_last: Tensor,
                   reconstruction_weight: float = 0.3) -> Tensor:
    forecast = quantile_loss(outputs["quantiles"], target)
    reconstruction = torch.mean(torch.abs(outputs["reconstruction"] - clean_last))
    return forecast + reconstruction_weight * reconstruction


@dataclass
class EarlyStopping:
    patience: int = 6
    best: float = float("inf")
    stale_epochs: int = 0

    def update(self, value: float) -> bool:
        if value < self.best:
            self.best, self.stale_epochs = value, 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def point_forecast_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean(torch.abs(prediction - target))
