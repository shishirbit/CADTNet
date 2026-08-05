from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import re
import numpy as np
import pandas as pd


def normalize_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_name(c) for c in out.columns]
    return out


def first_existing(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> Optional[str]:
    cols = set(columns)
    for candidate in candidates:
        c = normalize_name(candidate)
        if c in cols:
            return c
    return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_table(df: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output.suffix}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".txt", ".dat"}:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(path, sep=r"\s+", engine="python")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def stable_sort(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    return df.sort_values(available, kind="mergesort").reset_index(drop=True)


def chronological_split(
    values: pd.Series,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.10,
) -> pd.Series:
    unique = np.sort(pd.unique(values.dropna()))
    if len(unique) < 3:
        raise ValueError("At least three unique timestamps are required.")
    train_end = unique[max(0, int(len(unique) * train_fraction) - 1)]
    val_end_idx = max(
        int(len(unique) * (train_fraction + validation_fraction)) - 1,
        int(len(unique) * train_fraction),
    )
    val_end = unique[min(val_end_idx, len(unique) - 1)]
    split = pd.Series("test", index=values.index, dtype="object")
    split.loc[values <= val_end] = "validation"
    split.loc[values <= train_end] = "train"
    return split
