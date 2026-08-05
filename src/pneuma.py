from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from .common import save_table


META_COLUMNS = [
    "track_id",
    "vehicle_type",
    "traveled_distance_m",
    "trajectory_mean_speed_kmh",
]

REPEATED_COLUMNS = [
    "latitude",
    "longitude",
    "speed_kmh",
    "longitudinal_acceleration_mps2",
    "lateral_acceleration_mps2",
    "time_s",
]


def detect_delimiter(input_path: str | Path) -> str:
    """Detect the pNEUMA delimiter from the first non-empty physical line."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for line in stream:
            if line.strip():
                counts = {
                    ";": line.count(";"),
                    ",": line.count(","),
                    "\\t": line.count("\\t"),
                }
                delimiter = max(counts, key=counts.get)
                if counts[delimiter] == 0:
                    raise ValueError(
                        f"Unable to detect a delimiter in {input_path}. "
                        f"First line preview: {line[:200]!r}"
                    )
                return delimiter
    raise ValueError(f"The pNEUMA file is empty: {input_path}")


def _clean_tokens(tokens: list[str]) -> list[str]:
    cleaned = [str(value).strip().lstrip("\\ufeff") for value in tokens]
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


def _looks_like_header(tokens: list[str]) -> bool:
    if not tokens:
        return True
    joined = " ".join(tokens[:10]).strip().lower()
    header_terms = (
        "track_id", "track id", "vehicle type", "traveled", "travelled",
        "avg_speed", "average speed", "latitude", "longitude",
    )
    return any(term in joined for term in header_terms)


def _parse_vehicle_row(tokens: list[str], source_file: str) -> pd.DataFrame:
    tokens = _clean_tokens(tokens)
    if len(tokens) < 10 or _looks_like_header(tokens):
        return pd.DataFrame()

    metadata = tokens[:4]
    observations = tokens[4:]
    usable = len(observations) - (len(observations) % 6)
    if usable < 6:
        return pd.DataFrame()

    values = np.asarray(observations[:usable], dtype=object).reshape(-1, 6)
    frame = pd.DataFrame(values, columns=REPEATED_COLUMNS)

    for column in REPEATED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["track_id"] = str(metadata[0]).strip()
    frame["vehicle_type"] = str(metadata[1]).strip()
    frame["traveled_distance_m"] = pd.to_numeric(metadata[2], errors="coerce")
    frame["trajectory_mean_speed_kmh"] = pd.to_numeric(metadata[3], errors="coerce")
    frame["source_file"] = source_file

    valid = (
        frame["latitude"].between(-90, 90)
        & frame["longitude"].between(-180, 180)
        & frame["time_s"].notna()
    )
    return frame.loc[
        valid,
        META_COLUMNS + REPEATED_COLUMNS + ["source_file"],
    ].reset_index(drop=True)


def iter_pneuma_wide(
    input_path: str | Path,
    max_vehicles: int | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Read variable-width pNEUMA rows directly with csv.reader.

    The official raw files are semicolon-delimited and each vehicle row contains
    four metadata values followed by a variable number of six-value samples.
    """
    input_path = Path(input_path)
    delimiter = detect_delimiter(input_path)
    print(f"Detected delimiter {delimiter!r} for {input_path.name}")

    parsed_vehicles = 0
    skipped_rows = 0

    with input_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        for physical_row_number, tokens in enumerate(reader, start=1):
            tokens = _clean_tokens(tokens)
            if not tokens or _looks_like_header(tokens):
                continue

            parsed = _parse_vehicle_row(tokens, input_path.name)
            if parsed.empty:
                skipped_rows += 1
                if skipped_rows <= 5:
                    print(
                        f"Skipping row {physical_row_number}: "
                        f"{len(tokens)} fields, preview={tokens[:10]}"
                    )
                continue

            yield parsed
            parsed_vehicles += 1

            if parsed_vehicles % 25 == 0:
                print(
                    f"Parsed {parsed_vehicles} vehicles; "
                    f"latest row contributed {len(parsed):,} samples"
                )

            if max_vehicles is not None and parsed_vehicles >= max_vehicles:
                break

    if parsed_vehicles == 0:
        raise ValueError(
            f"No valid pNEUMA trajectories were parsed from {input_path}. "
            f"Detected delimiter={delimiter!r}; skipped_rows={skipped_rows}."
        )


def convert_pneuma_wide(
    input_path: str | Path,
    output_path: str | Path,
    chunksize_vehicles: int = 25,
    max_vehicles: int | None = None,
) -> pd.DataFrame:
    """Convert a pNEUMA variable-width CSV into a long-format Parquet/CSV table."""
    del chunksize_vehicles  # Kept for backward compatibility with the notebook.

    frames = list(iter_pneuma_wide(input_path, max_vehicles=max_vehicles))
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(
        ["source_file", "track_id", "time_s"],
        kind="mergesort",
    ).reset_index(drop=True)
    save_table(result, output_path)
    return result


def latlon_to_local_meters(
    latitude: pd.Series,
    longitude: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    lat0 = np.deg2rad(float(latitude.median()))
    lon0 = float(longitude.median())
    latitude0 = float(latitude.median())
    earth_radius_m = 6_371_000.0
    x = (
        np.deg2rad(longitude.astype(float) - lon0)
        * earth_radius_m
        * np.cos(lat0)
    )
    y = np.deg2rad(latitude.astype(float) - latitude0) * earth_radius_m
    return pd.Series(x, index=latitude.index), pd.Series(y, index=latitude.index)


def build_grid_traffic_states(
    trajectories: pd.DataFrame,
    interval_seconds: int = 1,
    cell_size_m: float = 40.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "track_id",
        "latitude",
        "longitude",
        "speed_kmh",
        "longitudinal_acceleration_mps2",
        "lateral_acceleration_mps2",
        "time_s",
    }
    missing = required.difference(trajectories.columns)
    if missing:
        raise ValueError(f"Missing required pNEUMA columns: {sorted(missing)}")

    df = trajectories.copy()
    df["x_m"], df["y_m"] = latlon_to_local_meters(df["latitude"], df["longitude"])
    df["grid_x"] = np.floor(df["x_m"] / cell_size_m).astype("int64")
    df["grid_y"] = np.floor(df["y_m"] / cell_size_m).astype("int64")
    df["node_id"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
    df["time_bin_s"] = (
        np.floor(df["time_s"] / interval_seconds) * interval_seconds
    ).astype("int64")

    grouped = df.groupby(["source_file", "time_bin_s", "node_id"], observed=True)
    states = grouped.agg(
        vehicle_count=("track_id", "nunique"),
        speed_mean_kmh=("speed_kmh", "mean"),
        speed_std_kmh=("speed_kmh", "std"),
        longitudinal_acc_mean_mps2=("longitudinal_acceleration_mps2", "mean"),
        lateral_acc_mean_mps2=("lateral_acceleration_mps2", "mean"),
        x_center_m=("x_m", "mean"),
        y_center_m=("y_m", "mean"),
    ).reset_index()

    states["speed_std_kmh"] = states["speed_std_kmh"].fillna(0.0)
    states["cell_area_m2"] = float(cell_size_m * cell_size_m)
    states["density_proxy_veh_per_km2"] = (
        states["vehicle_count"] / states["cell_area_m2"] * 1_000_000.0
    )
    states["flow_proxy_veh_kmh_per_km2"] = (
        states["density_proxy_veh_per_km2"] * states["speed_mean_kmh"].fillna(0.0)
    )
    states["occupancy_proxy"] = np.minimum(
        1.0,
        states["vehicle_count"] * 10.0 / states["cell_area_m2"],
    )
    states["timestamp_s"] = states["time_bin_s"]

    node_xy = states.groupby("node_id", observed=True).agg(
        grid_x=("node_id", lambda x: int(str(x.iloc[0]).split("_")[0])),
        grid_y=("node_id", lambda x: int(str(x.iloc[0]).split("_")[1])),
        x_center_m=("x_center_m", "mean"),
        y_center_m=("y_center_m", "mean"),
    ).reset_index()

    lookup = {
        (int(row.grid_x), int(row.grid_y)): str(row.node_id)
        for row in node_xy.itertuples()
    }
    edges = []
    for row in node_xy.itertuples():
        x, y = int(row.grid_x), int(row.grid_y)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = lookup.get((x + dx, y + dy))
            if neighbor is not None:
                edges.append((str(row.node_id), neighbor))

    graph = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    states = states.sort_values(
        ["source_file", "timestamp_s", "node_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return states, graph
