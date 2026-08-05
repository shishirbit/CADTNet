"""Recompute the released metrics and manuscript headline counts.

This script deliberately needs only NumPy and the Python standard library.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
PREDICTIONS = ROOT / "artifacts" / "predictions" / "seed42"
HORIZONS = (5, 10, 30)
TARGETS = ("speed_mean_kmh", "vehicle_count", "flow_proxy")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target, prediction = target.reshape(-1), prediction.reshape(-1)
    error = prediction - target
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "WAPE": float(np.sum(np.abs(error)) / np.sum(np.abs(target))),
        "R2": float(1.0 - np.sum(error ** 2) / np.sum((target - np.mean(target)) ** 2)),
    }


def verify_seed42(tolerance: float = 1e-9) -> int:
    released = read_rows(RESULTS / "phase6_seed_metrics.csv")
    lookup = {
        (r["model"], r["domain"], int(r["horizon_s"]), r["target"], int(r["seed"])): r
        for r in released
    }
    checked = 0
    for archive in sorted(PREDICTIONS.glob("*.npz")):
        parts = archive.stem.split("_")
        seed_index = parts.index("seed") if "seed" in parts else None
        # Filenames follow <model>_seed_42_<domain>; model names may contain underscores.
        marker = "_seed_42_"
        model, domain = archive.stem.split(marker, 1)
        payload = np.load(archive)
        prediction, target = payload["prediction"], payload["target"]
        for h_index, horizon in enumerate(HORIZONS):
            for t_index, target_name in enumerate(TARGETS):
                computed = metrics(target[:, h_index, :, t_index], prediction[:, h_index, :, t_index])
                row = lookup[(model, domain, horizon, target_name, 42)]
                for name, value in computed.items():
                    if not np.isclose(value, float(row[name]), rtol=tolerance, atol=tolerance):
                        raise AssertionError(
                            f"{archive.name} {horizon}s {target_name} {name}: "
                            f"computed={value}, released={row[name]}"
                        )
                checked += 1
    return checked


def verify_aggregate(tolerance: float = 1e-9) -> int:
    seed_rows = read_rows(RESULTS / "phase6_seed_metrics.csv")
    aggregate_rows = read_rows(RESULTS / "phase6_aggregate_metrics.csv")
    grouped: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["model"], row["domain"], int(row["horizon_s"]), row["target"])].append(row)
    checked = 0
    for row in aggregate_rows:
        key = (row["model"], row["domain"], int(row["horizon_s"]), row["target"])
        members = grouped[key]
        if int(row["seed_count"]) != len(members):
            raise AssertionError(f"Seed count mismatch for {key}")
        for metric_name in ("MAE", "RMSE", "WAPE", "R2"):
            values = np.array([float(member[metric_name]) for member in members])
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            if not np.isclose(mean, float(row[f"{metric_name}_mean"]), rtol=tolerance, atol=tolerance):
                raise AssertionError(f"Aggregate mean mismatch for {key} {metric_name}")
            if not np.isclose(std, float(row[f"{metric_name}_std"]), rtol=tolerance, atol=tolerance):
                raise AssertionError(f"Aggregate std mismatch for {key} {metric_name}")
            checked += 1
    return checked


def headline_wins() -> tuple[int, int]:
    rows = read_rows(RESULTS / "phase6_aggregate_metrics.csv")
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], int(row["horizon_s"]), row["target"])].append(row)
    mae_wins = wape_wins = 0
    for members in grouped.values():
        if min(members, key=lambda r: float(r["MAE_mean"]))["model"] == "ca_dtnet":
            mae_wins += 1
        if min(members, key=lambda r: float(r["WAPE_mean"]))["model"] == "ca_dtnet":
            wape_wins += 1
    return mae_wins, wape_wins


def main() -> None:
    seed_checks = verify_seed42()
    aggregate_checks = verify_aggregate()
    mae_wins, wape_wins = headline_wins()
    if (mae_wins, wape_wins) != (24, 24):
        raise AssertionError(f"Expected 24/27 CA-DTNet wins; got MAE={mae_wins}, WAPE={wape_wins}")
    print(f"PASS: {seed_checks} seed-42 metric rows recomputed from NPZ predictions")
    print(f"PASS: {aggregate_checks} three-seed aggregate values recomputed")
    print(f"PASS: CA-DTNet has the lowest mean MAE and WAPE in {mae_wins}/27 configurations")


if __name__ == "__main__":
    main()
