from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .common import (
    first_existing,
    normalized_columns,
    numeric,
    read_table,
    save_table,
)


def _all_data_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    allowed = {".csv", ".txt", ".dat"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in allowed)


def standardize_cicv5g(input_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in _all_data_files(input_dir):
        try:
            df = normalized_columns(read_table(path))
        except Exception:
            continue

        pub_col = first_existing(
            df.columns,
            ["pub_time", "publish_time", "tx_timestamp", "send_time"],
        )
        sub_col = first_existing(
            df.columns,
            ["sub_time", "subscribe_time", "rx_timestamp", "receive_time"],
        )
        delay_col = first_existing(
            df.columns,
            ["delay", "delay_ms", "delay_ms_", "rtt", "rtt_ms", "round_trip_delay"],
        )
        rsrp_col = first_existing(df.columns, ["rsrp", "reference_signal_received_power"])
        sinr_col = first_existing(df.columns, ["sinr", "snr", "signal_to_noise_ratio"])
        cell_col = first_existing(df.columns, ["cell_id", "cellid", "pci"])
        velocity_col = first_existing(
            df.columns, ["velocity", "speed", "vehicle_speed", "speed_kmh"]
        )
        x_col = first_existing(df.columns, ["x", "utmx", "utm_x", "pos_x", "position_x", "longitude"])
        y_col = first_existing(df.columns, ["y", "utmy", "utm_y", "pos_y", "position_y", "latitude"])

        out = pd.DataFrame(index=df.index)
        out["pub_time_ms"] = numeric(df[pub_col]) if pub_col else np.nan
        out["sub_time_ms"] = numeric(df[sub_col]) if sub_col else np.nan

        if delay_col:
            out["latency_ms"] = numeric(df[delay_col])
        elif pub_col and sub_col:
            out["latency_ms"] = out["sub_time_ms"] - out["pub_time_ms"]
        else:
            continue

        out["rsrp_dbm"] = numeric(df[rsrp_col]) if rsrp_col else np.nan
        out["sinr_db"] = numeric(df[sinr_col]) if sinr_col else np.nan
        out["cell_id"] = df[cell_col].astype(str) if cell_col else None
        out["vehicle_speed"] = numeric(df[velocity_col]) if velocity_col else np.nan
        out["position_x"] = numeric(df[x_col]) if x_col else np.nan
        out["position_y"] = numeric(df[y_col]) if y_col else np.nan
        out["source_file"] = str(path.relative_to(Path(input_dir)))
        out["trace_id"] = out["source_file"]
        out["communication_domain"] = "cicv5g"
        out["sample_index"] = np.arange(len(out), dtype=np.int64)

        out = out.loc[out["latency_ms"].notna() & (out["latency_ms"] >= 0)]
        if not out.empty:
            frames.append(out)

    if not frames:
        raise ValueError(
            "No CICV5G files could be standardized. Inspect the raw column names."
        )

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["trace_id", "sample_index"], kind="mergesort"
    ).reset_index(drop=True)


def standardize_see_v2x(input_dir: str | Path) -> pd.DataFrame:
    root = Path(input_dir)
    paths = sorted(root.rglob("rx_*_tx_*.csv"))
    frames = []

    for path in paths:
        df = normalized_columns(pd.read_csv(path))

        mapping = {
            "rx_timestamp_us": [
                "rx_timestamp_us", "rx_timestamp", "receive_timestamp_us"
            ],
            "latency_ms": ["latency_ms", "latency"],
            "packet_loss_pct": ["per_ue_loss_pct", "packet_loss_pct"],
            "jitter_ms": ["ipg_ms", "ipg", "jitter_ms"],
            "tx_equipment_id": ["tx_equipment_id"],
            "tx_seq_num": ["tx_seq_num", "seq_num"],
            "tx_priority": ["tx_priority", "priority"],
            "tx_timestamp_us": ["tx_timestamp_us", "tx_timestamp"],
            "channel_busy_pct": [
                "channel_busy_percentage", "channel_busy_pct"
            ],
            "packet_size_bytes": ["packet_size", "packet_size_bytes"],
            "throughput_10ms_bps": [
                "avg_throughput_10ms", "avg_throughput10ms"
            ],
            "throughput_100ms_bps": [
                "avg_throughput_100ms", "avg_throughput100ms"
            ],
            "packet_loss_10ms": [
                "avg_packet_loss_10ms", "avg_packet_loss10ms"
            ],
            "packet_loss_100ms": [
                "avg_packet_loss_100ms", "avg_packet_loss100ms"
            ],
        }

        out = pd.DataFrame(index=df.index)
        for target, aliases in mapping.items():
            source = first_existing(df.columns, aliases)
            out[target] = numeric(df[source]) if source else np.nan

        if out["latency_ms"].isna().all():
            continue

        out["source_file"] = str(path.relative_to(root))
        out["trace_id"] = out["source_file"]
        out["communication_domain"] = "see_v2x"
        out["sample_index"] = np.arange(len(out), dtype=np.int64)
        out["packet_received"] = 1

        frames.append(out)

    if not frames:
        raise ValueError(
            "No SEE-V2X receiver files matching rx_*_tx_*.csv were found."
        )

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["trace_id", "sample_index"], kind="mergesort"
    ).reset_index(drop=True)


def write_standardized(df: pd.DataFrame, output: str | Path) -> None:
    save_table(df, output)
