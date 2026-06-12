from __future__ import annotations

import json
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from inspect_datasets import detect_label_column, suspicious_fields
from utils.data_io import DATASETS, discover_dataset_files, ensure_dirs, processed_dir, project_root, read_table, safe_relpath, write_json


BENIGN_TOKENS = ("benign", "normal", "legitimate", "legit", "nonattack", "non_attack", "clean", "false", "no", "0")
LEAKAGE_TOKENS = ("label", "target", "attack", "scenario", "category", "class", "malicious", "benign", "normal", "is_attack")
FORBIDDEN_MAIN_FEATURE_TOKENS = (
    "payload",
    "topic",
    "username",
    "user_name",
    "password",
    "passwd",
    "clientid",
    "client_id",
    "client id",
    "secret",
    "token",
    "auth",
    "credential",
    "ip",
    "addr",
    "mac",
    "host",
    "source_file",
    "file_name",
    "filename",
)
RAW_CONTENT_EXACT = {"mqtt_msg", "mqtt_message", "msg", "message", "raw_message", "content"}
TEMPORAL_LEAKAGE_EXACT = {"timestamp", "time_epoch", "frame_time", "datetime", "date_time"}


def normalize_column_name(col: str) -> str:
    clean = str(col).strip()
    clean = re.sub(r"\s+", "_", clean)
    clean = clean.replace(".", "_").replace("-", "_")
    clean = re.sub(r"[^0-9A-Za-z_]+", "", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "unnamed"


def unique_columns(columns: list[str]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    out = []
    for col in columns:
        base = normalize_column_name(col)
        counts[base] += 1
        out.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return out


def label_to_binary(value: object) -> int:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "<na>", "nan", "none", "null", "unknown"}:
        return np.nan
    if text in BENIGN_TOKENS:
        return 0
    if text in {"false", "f"}:
        return 0
    if text in {"true", "t"}:
        return 1
    try:
        numeric = float(text)
        return 0 if numeric == 0 else 1
    except Exception:
        pass
    if any(token in text for token in BENIGN_TOKENS):
        return 0
    return 1


def infer_label_from_filename(path: Path) -> str:
    low = path.stem.lower()
    if any(token in low for token in ("normal", "benign", "legitimate", "legit")):
        return "benign"
    return path.stem


def drop_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    duplicates: list[str] = []
    seen: dict[int, str] = {}
    keep_cols: list[str] = []
    for col in df.columns:
        series = df[col]
        try:
            digest = int(pd.util.hash_pandas_object(series, index=False).sum())
        except Exception:
            digest = hash(tuple(series.astype("string").fillna("<NA>").head(1000).tolist()))
        if digest in seen and series.equals(df[seen[digest]]):
            duplicates.append(col)
            continue
        seen[digest] = col
        keep_cols.append(col)
    return df[keep_cols], duplicates


def exact_forbidden_columns(columns: list[str]) -> set[str]:
    forbidden = set()
    for col in columns:
        low = col.lower().replace(".", "_").replace("-", "_").replace(" ", "_")
        if low in RAW_CONTENT_EXACT or low in TEMPORAL_LEAKAGE_EXACT:
            forbidden.add(col)
    return forbidden


def add_common_flow_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized flow metadata names for cross-dataset evaluation."""
    out = df.copy()

    def num(col: str):
        if col not in out.columns:
            return None
        return pd.to_numeric(out[col], errors="coerce")

    fwd_pkts, bwd_pkts = num("fwd_num_pkts"), num("bwd_num_pkts")
    fwd_bytes, bwd_bytes = num("fwd_num_bytes"), num("bwd_num_bytes")
    if fwd_pkts is not None and bwd_pkts is not None:
        out["packet_count"] = fwd_pkts.fillna(0) + bwd_pkts.fillna(0)
    elif "num_pkts" in out.columns:
        out["packet_count"] = pd.to_numeric(out["num_pkts"], errors="coerce")

    if fwd_bytes is not None and bwd_bytes is not None:
        out["byte_sum"] = fwd_bytes.fillna(0) + bwd_bytes.fillna(0)
    elif "num_bytes" in out.columns:
        out["byte_sum"] = pd.to_numeric(out["num_bytes"], errors="coerce")

    pairs = {
        "byte_mean": ("fwd_mean_pkt_len", "bwd_mean_pkt_len", "mean_pkt_len"),
        "byte_std": ("fwd_std_pkt_len", "bwd_std_pkt_len", "std_pkt_len"),
        "iat_mean": ("fwd_mean_iat", "bwd_mean_iat", "mean_iat"),
        "iat_std": ("fwd_std_iat", "bwd_std_iat", "std_iat"),
    }
    for target, (fwd, bwd, uni) in pairs.items():
        fwd_s, bwd_s = num(fwd), num(bwd)
        if fwd_s is not None and bwd_s is not None:
            out[target] = pd.concat([fwd_s, bwd_s], axis=1).mean(axis=1)
        elif uni in out.columns:
            out[target] = pd.to_numeric(out[uni], errors="coerce")

    min_pairs = {
        "byte_min": ("fwd_min_pkt_len", "bwd_min_pkt_len", "min_pkt_len"),
        "iat_min": ("fwd_min_iat", "bwd_min_iat", "min_iat"),
    }
    for target, (fwd, bwd, uni) in min_pairs.items():
        fwd_s, bwd_s = num(fwd), num(bwd)
        if fwd_s is not None and bwd_s is not None:
            out[target] = pd.concat([fwd_s, bwd_s], axis=1).min(axis=1)
        elif uni in out.columns:
            out[target] = pd.to_numeric(out[uni], errors="coerce")

    max_pairs = {
        "byte_max": ("fwd_max_pkt_len", "bwd_max_pkt_len", "max_pkt_len"),
        "iat_max": ("fwd_max_iat", "bwd_max_iat", "max_iat"),
    }
    for target, (fwd, bwd, uni) in max_pairs.items():
        fwd_s, bwd_s = num(fwd), num(bwd)
        if fwd_s is not None and bwd_s is not None:
            out[target] = pd.concat([fwd_s, bwd_s], axis=1).max(axis=1)
        elif uni in out.columns:
            out[target] = pd.to_numeric(out[uni], errors="coerce")

    if "duration" not in out.columns and "iat_max" in out.columns:
        out["duration"] = pd.to_numeric(out["iat_max"], errors="coerce").fillna(0)
    if "packet_count" in out.columns and "duration" in out.columns:
        duration = pd.to_numeric(out["duration"], errors="coerce").replace(0, np.nan)
        out["packets_per_second"] = pd.to_numeric(out["packet_count"], errors="coerce") / duration
    if "byte_sum" in out.columns and "duration" in out.columns:
        duration = pd.to_numeric(out["duration"], errors="coerce").replace(0, np.nan)
        out["bytes_per_second"] = pd.to_numeric(out["byte_sum"], errors="coerce") / duration
    if "byte_std" in out.columns and "byte_mean" in out.columns:
        out["burstiness"] = pd.to_numeric(out["byte_std"], errors="coerce") / pd.to_numeric(out["byte_mean"], errors="coerce").replace(0, np.nan)

    for col in ["packets_per_second", "bytes_per_second", "burstiness"]:
        if col in out.columns:
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def load_dataset(dataset: str, root: Path) -> tuple[pd.DataFrame | None, list[dict]]:
    frames = []
    source_info = []
    files = discover_dataset_files(dataset, root, extract_archives=True)
    if dataset == "MQTT-IoT-IDS2020":
        biflow_files = [path for path in files if "biflow" in str(path).lower()]
        uniflow_files = [path for path in files if "uniflow" in str(path).lower()]
        if biflow_files:
            files = biflow_files
            selection_note = "selected_biflow_features"
        elif uniflow_files:
            files = uniflow_files
            selection_note = "selected_uniflow_features_fallback"
        else:
            selection_note = "no_biflow_or_uniflow_files_found"
    else:
        selection_note = "all_discovered_files"
    for path in files:
        try:
            df = read_table(path)
            original_columns = list(df.columns)
            df.columns = unique_columns(original_columns)
            sample = df.head(5000)
            label_col = detect_label_column(list(df.columns), sample)
            if label_col:
                original_label = df[label_col]
            else:
                original_label = pd.Series([infer_label_from_filename(path)] * len(df), index=df.index)
            binary = original_label.map(label_to_binary)
            missing_label_rows = int(binary.isna().sum())
            if missing_label_rows:
                df = df.loc[binary.notna()].copy()
                original_label = original_label.loc[binary.notna()]
                binary = binary.loc[binary.notna()]
            multiclass = original_label.astype("string").fillna("unknown").str.strip()
            binary = binary.astype("int8")
            if label_col and multiclass.nunique(dropna=True) <= 2 and binary.nunique(dropna=True) <= 2:
                inferred = infer_label_from_filename(path)
                multiclass = np.where(binary == 0, "benign", "attack" if inferred == "benign" else inferred)
            df.insert(0, "multiclass_label", multiclass)
            df.insert(0, "binary_label", binary)
            df.insert(0, "dataset_name", dataset)
            df.insert(1, "source_file", safe_relpath(path, root))
            frames.append(df)
            source_info.append(
                {
                    "file_path": safe_relpath(path, root),
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "detected_label_column": label_col,
                    "inferred_label_from_filename": None if label_col else infer_label_from_filename(path),
                    "dropped_missing_label_rows": missing_label_rows,
                    "selection_note": selection_note,
                }
            )
        except Exception as exc:
            source_info.append({"file_path": safe_relpath(path, root), "status": f"failed: {exc!r}"})
    if not frames:
        return None, source_info
    return pd.concat(frames, ignore_index=True, sort=False), source_info


def preprocess_dataset(dataset: str, root: Path) -> dict:
    out_dir = processed_dir(root)
    df, source_info = load_dataset(dataset, root)
    if df is None or df.empty:
        return {"dataset": dataset, "status": "no_data", "sources": source_info}

    raw_rows, raw_cols = df.shape
    labels = df[["binary_label", "multiclass_label"]].copy()

    empty_cols = [col for col in df.columns if df[col].isna().all()]
    df = df.drop(columns=empty_cols)

    constant_cols = [
        col
        for col in df.columns
        if col not in {"binary_label", "multiclass_label"} and df[col].nunique(dropna=False) <= 1
    ]
    df = df.drop(columns=constant_cols)

    df, duplicate_cols = drop_duplicate_columns(df)
    if "binary_label" not in df.columns:
        df.insert(0, "binary_label", labels["binary_label"])
    if "multiclass_label" not in df.columns:
        df.insert(1, "multiclass_label", labels["multiclass_label"])

    leakage_cols = set(suspicious_fields(list(df.columns), LEAKAGE_TOKENS))
    forbidden_cols = set(suspicious_fields(list(df.columns), FORBIDDEN_MAIN_FEATURE_TOKENS))
    forbidden_cols |= exact_forbidden_columns(list(df.columns))
    keep_label_cols = {"binary_label", "multiclass_label"}
    drop_cols = sorted((leakage_cols | forbidden_cols | {"dataset_name", "source_file"}) - keep_label_cols)
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    if dataset == "MQTT-IoT-IDS2020":
        df = add_common_flow_metadata(df)

    for col in list(df.columns):
        if col in keep_label_cols:
            continue
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            nunique = df[col].nunique(dropna=True)
            if nunique > max(1000, len(df) * 0.2):
                drop_cols.append(col)
                df = df.drop(columns=[col])

    cleaned_path = out_dir / f"{dataset}__cleaned.csv"
    metadata_path = out_dir / f"{dataset}__preprocess_metadata.json"
    df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    metadata = {
        "dataset": dataset,
        "status": "ok",
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_cols),
        "cleaned_rows": int(df.shape[0]),
        "cleaned_columns": int(df.shape[1]),
        "feature_columns": [col for col in df.columns if col not in keep_label_cols],
        "dropped_empty_columns": empty_cols,
        "dropped_constant_columns": constant_cols,
        "dropped_duplicate_columns": duplicate_cols,
        "dropped_leakage_or_forbidden_columns": sorted(set(drop_cols)),
        "binary_label_mapping": {"0": "benign/normal/legitimate", "1": "attack/anomalous"},
        "multiclass_labels": sorted(pd.Series(df["multiclass_label"]).astype(str).unique().tolist()),
        "sources": source_info,
        "cleaned_path": safe_relpath(cleaned_path, root),
    }
    write_json(metadata_path, metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess MQTT datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    reports = [preprocess_dataset(dataset, root) for dataset in DATASETS]
    write_json(processed_dir(root) / "preprocess_summary.json", reports)
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
