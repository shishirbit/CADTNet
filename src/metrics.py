"""Forecast and digital-twin evaluation metrics."""

from __future__ import annotations

import numpy as np


def point_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    error = prediction - target
    denominator = np.sum(np.abs(target))
    total_variation = np.sum((target - np.mean(target)) ** 2)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "WAPE": float(np.sum(np.abs(error)) / denominator) if denominator else float("nan"),
        "R2": float(1.0 - np.sum(error ** 2) / total_variation) if total_variation else float("nan"),
    }


def interval_metrics(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    target, lower, upper = map(lambda x: np.asarray(x, dtype=float), (target, lower, upper))
    return {
        "coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }
