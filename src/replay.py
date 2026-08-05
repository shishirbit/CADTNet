from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplayConfig:
    seed: int = 42
    max_latency_seconds: float = 5.0
    initial_age_seconds: float = 0.0


def _safe_metric(row: pd.Series, column: str, default: float = np.nan) -> float:
    value = row.get(column, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _loss_probability(row: pd.Series) -> float:
    for column in (
        "packet_loss_10ms",
        "packet_loss_100ms",
        "packet_loss_pct",
    ):
        value = _safe_metric(row, column)
        if np.isfinite(value):
            if value > 1:
                value = value / 100.0
            return float(np.clip(value, 0.0, 1.0))
    return 0.0


def apply_contiguous_trace_replay(
    traffic_states: pd.DataFrame,
    communication: pd.DataFrame,
    config: ReplayConfig = ReplayConfig(),
) -> pd.DataFrame:
    required_traffic = {"timestamp_s", "node_id"}
    missing = required_traffic.difference(traffic_states.columns)
    if missing:
        raise ValueError(f"Traffic data missing: {sorted(missing)}")

    if "trace_id" not in communication.columns:
        raise ValueError("Communication table must include trace_id.")

    traffic = traffic_states.sort_values(
        ["node_id", "timestamp_s"], kind="mergesort"
    ).reset_index(drop=True)
    traces = {
        trace_id: frame.sort_values("sample_index", kind="mergesort")
        .reset_index(drop=True)
        for trace_id, frame in communication.groupby("trace_id", observed=True)
        if len(frame) > 0
    }
    if not traces:
        raise ValueError("No non-empty communication traces are available.")

    rng = np.random.default_rng(config.seed)
    trace_ids = sorted(traces)
    state_columns = [
        c for c in traffic.columns
        if c not in {
            "source_file", "time_bin_s", "timestamp_s", "node_id",
            "x_center_m", "y_center_m", "cell_area_m2"
        }
        and pd.api.types.is_numeric_dtype(traffic[c])
    ]

    outputs = []
    for node_index, (node_id, node_df) in enumerate(
        traffic.groupby("node_id", observed=True, sort=True)
    ):
        node_df = node_df.sort_values("timestamp_s", kind="mergesort").copy()
        trace_id = trace_ids[node_index % len(trace_ids)]
        trace = traces[trace_id]

        if len(trace) >= len(node_df):
            max_start = len(trace) - len(node_df)
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
            selected = trace.iloc[start:start + len(node_df)].reset_index(drop=True)
        else:
            repeats = int(np.ceil(len(node_df) / len(trace)))
            selected = pd.concat([trace] * repeats, ignore_index=True).iloc[
                :len(node_df)
            ]

        age = float(config.initial_age_seconds)
        latest_received = {}
        records = []

        for local_idx, (_, traffic_row) in enumerate(node_df.iterrows()):
            comm_row = selected.iloc[local_idx]
            loss_prob = _loss_probability(comm_row)
            received = bool(rng.random() >= loss_prob)

            latency_ms = _safe_metric(comm_row, "latency_ms", 0.0)
            latency_s = min(
                max(latency_ms / 1000.0, 0.0),
                config.max_latency_seconds,
            )

            dt = 0.0
            if local_idx > 0:
                dt = max(
                    0.0,
                    float(node_df.iloc[local_idx]["timestamp_s"])
                    - float(node_df.iloc[local_idx - 1]["timestamp_s"]),
                )

            if received:
                age = latency_s
                for column in state_columns:
                    latest_received[column] = traffic_row[column]
            else:
                age += dt

            record = traffic_row.to_dict()
            for column in state_columns:
                record[f"true_{column}"] = traffic_row[column]
                record[f"received_{column}"] = (
                    latest_received.get(column, np.nan)
                )

            record["packet_received"] = int(received)
            record["age_of_information_s"] = age
            record["trace_id"] = trace_id
            record["communication_domain"] = comm_row.get(
                "communication_domain", "unknown"
            )

            for column in (
                "latency_ms",
                "packet_loss_pct",
                "jitter_ms",
                "channel_busy_pct",
                "throughput_10ms_bps",
                "throughput_100ms_bps",
                "packet_loss_10ms",
                "packet_loss_100ms",
                "rsrp_dbm",
                "sinr_db",
                "vehicle_speed",
            ):
                record[column] = comm_row.get(column, np.nan)

            records.append(record)

        outputs.append(pd.DataFrame.from_records(records))

    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(
        ["timestamp_s", "node_id"], kind="mergesort"
    ).reset_index(drop=True)
