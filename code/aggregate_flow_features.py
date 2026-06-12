from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import label_to_binary
from utils.data_io import discover_dataset_files, ensure_dirs, processed_dir, project_root, read_table, results_tables_dir, safe_relpath


def aggregate_one_window(df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    working = df[["timestamp", "tcp_len", "label"]].copy()
    working["timestamp"] = pd.to_numeric(working["timestamp"], errors="coerce")
    working["tcp_len"] = pd.to_numeric(working["tcp_len"], errors="coerce")
    working = working.dropna(subset=["timestamp", "tcp_len", "label"])
    working["binary_label"] = working["label"].map(label_to_binary)
    working = working.dropna(subset=["binary_label"])
    working["binary_label"] = working["binary_label"].astype(int)
    working["multiclass_label"] = working["label"].astype(str).str.strip()
    working = working.sort_values("timestamp")
    t0 = working["timestamp"].min()
    working["window_id"] = np.floor((working["timestamp"] - t0) / window_seconds).astype(int)

    rows = []
    for window_id, group in working.groupby("window_id", sort=True):
        labels = group["multiclass_label"].tolist()
        label_counts = Counter(labels)
        majority_label, majority_count = label_counts.most_common(1)[0]
        binary = 1 if (group["binary_label"] == 1).any() else 0
        timestamps = group["timestamp"].to_numpy(dtype=float)
        iats = np.diff(timestamps) if len(timestamps) > 1 else np.array([], dtype=float)
        duration = float(timestamps.max() - timestamps.min()) if len(timestamps) > 1 else 0.0
        tcp_len = group["tcp_len"].to_numpy(dtype=float)
        byte_mean = float(np.mean(tcp_len)) if len(tcp_len) else 0.0
        byte_std = float(np.std(tcp_len, ddof=0)) if len(tcp_len) else 0.0
        rows.append(
            {
                "binary_label": binary,
                "multiclass_label": majority_label,
                "window_seconds": window_seconds,
                "packet_count": int(len(group)),
                "byte_sum": float(np.sum(tcp_len)),
                "byte_mean": byte_mean,
                "byte_std": byte_std,
                "byte_min": float(np.min(tcp_len)),
                "byte_max": float(np.max(tcp_len)),
                "duration": duration,
                "packets_per_second": float(len(group) / duration) if duration > 0 else float(len(group) / window_seconds),
                "bytes_per_second": float(np.sum(tcp_len) / duration) if duration > 0 else float(np.sum(tcp_len) / window_seconds),
                "iat_mean": float(np.mean(iats)) if len(iats) else 0.0,
                "iat_std": float(np.std(iats, ddof=0)) if len(iats) else 0.0,
                "iat_min": float(np.min(iats)) if len(iats) else 0.0,
                "iat_max": float(np.max(iats)) if len(iats) else 0.0,
                "burstiness": float(byte_std / byte_mean) if byte_mean > 0 else 0.0,
                "mixed_label_ratio": float(1.0 - majority_count / len(group)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate MQTTEEB-D packet rows into encrypted-flow time-window metadata.")
    parser.add_argument("--dataset", default="MQTTEEB-D")
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    root = project_root()
    ensure_dirs(root)
    if args.dataset != "MQTTEEB-D":
        raise ValueError("Only MQTTEEB-D flow aggregation is currently implemented.")

    frames = []
    sources = []
    for path in discover_dataset_files("MQTTEEB-D", root, extract_archives=True):
        if "dataset_loop" not in path.name:
            continue
        df = read_table(path, usecols=["timestamp", "tcp_len", "label"])
        frames.append(df)
        sources.append(safe_relpath(path, root))
    if not frames:
        raise FileNotFoundError("No MQTTEEB-D raw loop CSV files found.")

    raw = pd.concat(frames, ignore_index=True)
    profiles = []
    for window in args.windows:
        agg = aggregate_one_window(raw, window)
        out_path = processed_dir(root) / f"MQTTEEB-D_flow_windows_{window}s.parquet"
        agg.to_parquet(out_path, index=False)
        profiles.append(
            {
                "dataset": "MQTTEEB-D",
                "window_seconds": window,
                "rows": int(len(agg)),
                "columns": int(len(agg.columns)),
                "features": json.dumps([col for col in agg.columns if col not in {"binary_label", "multiclass_label"}], ensure_ascii=False),
                "mixed_windows": int((agg["mixed_label_ratio"] > 0).sum()) if not agg.empty else 0,
                "mean_mixed_label_ratio": float(agg["mixed_label_ratio"].mean()) if not agg.empty else 0.0,
                "output_file": safe_relpath(out_path, root),
                "source_files": json.dumps(sources, ensure_ascii=False),
            }
        )
    profile_path = results_tables_dir(root) / "MQTTEEB-D_flow_aggregation_profile.csv"
    pd.DataFrame(profiles).to_csv(profile_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {profile_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
