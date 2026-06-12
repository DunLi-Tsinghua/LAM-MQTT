from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils.data_io import ensure_dirs, processed_dir, project_root, results_tables_dir, safe_relpath


GROUP_CANDIDATES = (
    "source_file",
    "capture",
    "capture_name",
    "pcap",
    "pcap_name",
    "scenario",
    "scenario_name",
    "session",
    "session_id",
    "source_capture",
)
TIME_CANDIDATES = (
    "timestamp",
    "time_epoch",
    "frame_time",
    "datetime",
    "date_time",
    "record_order",
    "row_index",
    "index",
)


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def valid_label_frame(df: pd.DataFrame, label_col: str = "binary_label") -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"missing {label_col}")
    y = pd.to_numeric(df[label_col], errors="coerce") if label_col == "binary_label" else df[label_col].astype("string")
    valid = y.notna()
    return df.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def random_stratified_split(y: pd.Series, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(idx, y, test_size=0.30, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=y_temp)
    return train_idx, val_idx, test_idx


def find_group_field(df: pd.DataFrame) -> str | None:
    low_map = {str(col).lower(): col for col in df.columns}
    for candidate in GROUP_CANDIDATES:
        if candidate in low_map and df[low_map[candidate]].nunique(dropna=True) > 1:
            return str(low_map[candidate])
    return None


def group_split(df: pd.DataFrame, y: pd.Series, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    group_field = find_group_field(df)
    if group_field is None:
        return None
    groups = df[group_field].astype("string").fillna("<NA>").to_numpy()
    idx = np.arange(len(df))
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(splitter.split(idx, y, groups))
    temp_groups = groups[temp_idx]
    temp_y = y.iloc[temp_idx] if hasattr(y, "iloc") else y[temp_idx]
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_rel, test_rel = next(splitter2.split(temp_idx, temp_y, temp_groups))
    return train_idx, temp_idx[val_rel], temp_idx[test_rel], group_field


def find_time_field(df: pd.DataFrame) -> str | None:
    low_map = {str(col).lower(): col for col in df.columns}
    for candidate in TIME_CANDIDATES:
        if candidate in low_map:
            return str(low_map[candidate])
    return None


def time_block_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    time_field = find_time_field(df)
    if time_field:
        ordered = df.sort_values(time_field, kind="stable").index.to_numpy()
        group_field = time_field
    else:
        ordered = np.arange(len(df))
        group_field = "original_row_order"
    n = len(ordered)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return ordered[:n_train], ordered[n_train : n_train + n_val], ordered[n_train + n_val :], group_field


def available_splits(df: pd.DataFrame, y: pd.Series, seed: int = 42) -> dict[str, dict]:
    splits: dict[str, dict] = {}
    train_idx, val_idx, test_idx = random_stratified_split(y, seed)
    splits["random_stratified"] = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "group_field": "",
        "notes": "70/15/15 stratified split.",
    }
    grouped = group_split(df, y, seed)
    if grouped is not None:
        g_train, g_val, g_test, field = grouped
        splits["group_split"] = {
            "train_idx": g_train,
            "val_idx": g_val,
            "test_idx": g_test,
            "group_field": field,
            "notes": "Group split; same group is not shared across train/validation/test.",
        }
    t_train, t_val, t_test, time_field = time_block_split(df)
    splits["time_block_split"] = {
        "train_idx": t_train,
        "val_idx": t_val,
        "test_idx": t_test,
        "group_field": time_field,
        "notes": "First 70%, next 15%, last 15%; ordering is used only for splitting, not as a feature.",
    }
    return splits


def source_files(root: Path) -> list[tuple[str, Path]]:
    processed = processed_dir(root)
    out = []
    for dataset, path in [
        ("MQTTEEB-D", processed / "MQTTEEB-D__cleaned.csv"),
        ("MQTT-IoT-IDS2020", processed / "MQTT-IoT-IDS2020__cleaned.csv"),
    ]:
        if path.exists():
            out.append((dataset, path))
    for path in sorted(processed.glob("MQTTEEB-D_flow_windows_*s.parquet")):
        window = path.stem.replace("MQTTEEB-D_flow_windows_", "")
        out.append((f"MQTTEEB-D_flow_{window}", path))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Create split strategy availability summary.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    rows = []
    for dataset, path in source_files(root):
        try:
            df, y = valid_label_frame(read_any(path), "binary_label")
            splits = available_splits(df, y.astype(int), args.seed)
            for split_type, split in splits.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "split_type": split_type,
                        "group_field": split["group_field"],
                        "train_size": len(split["train_idx"]),
                        "val_size": len(split["val_idx"]),
                        "test_size": len(split["test_idx"]),
                        "notes": split["notes"] + f" Source: {safe_relpath(path, root)}",
                    }
                )
            if "group_split" not in splits:
                rows.append(
                    {
                        "dataset": dataset,
                        "split_type": "group_split_not_available",
                        "group_field": "",
                        "train_size": 0,
                        "val_size": 0,
                        "test_size": 0,
                        "notes": "No explicit source_file/capture/pcap/scenario/session group field is present in the processed modeling table; no fabricated group was used.",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "dataset": dataset,
                    "split_type": "split_summary_failed",
                    "group_field": "",
                    "train_size": 0,
                    "val_size": 0,
                    "test_size": 0,
                    "notes": repr(exc),
                }
            )
    out_path = results_tables_dir(root) / "split_strategy_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
