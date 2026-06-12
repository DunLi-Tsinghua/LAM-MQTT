from __future__ import annotations

import json
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from utils.data_io import DATASETS, discover_dataset_files, ensure_dirs, project_root, read_table, read_table_chunks, results_tables_dir, safe_relpath


LABEL_EXACT = {
    "label",
    "labels",
    "target",
    "class",
    "classification",
    "is_attack",
    "attack",
    "attack_label",
    "attack_type",
    "attack_cat",
    "category",
}
LABEL_TOKENS = ("label", "target", "class", "attack", "category", "malicious", "anomaly")
LEAKAGE_TOKENS = ("label", "target", "attack", "scenario", "category", "class", "malicious", "benign", "normal", "is_attack")
SENSITIVE_TOKENS = (
    "payload",
    "topic",
    "mqtt_msg",
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
    "hostname",
)


def detect_label_column(columns: list[str], sample: pd.DataFrame | None = None) -> str | None:
    lowered = {col.lower().strip(): col for col in columns}
    for key in LABEL_EXACT:
        if key in lowered:
            return lowered[key]
    scored: list[tuple[int, str]] = []
    for col in columns:
        low = col.lower().strip()
        score = 0
        if any(token in low for token in LABEL_TOKENS):
            score += 3
        if sample is not None and col in sample.columns:
            nunique = sample[col].nunique(dropna=True)
            if 1 < nunique <= 50:
                score += 1
            if pd.api.types.is_numeric_dtype(sample[col]) and nunique <= 5:
                score += 1
        if score > 0:
            scored.append((score, col))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]


def suspicious_fields(columns: list[str], tokens: tuple[str, ...]) -> list[str]:
    found = []
    for col in columns:
        normalized = col.lower().replace(".", "_").replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in tokens):
            found.append(col)
    return found


def inspect_file(dataset: str, path: Path, root: Path) -> tuple[dict, list[dict]]:
    rel = safe_relpath(path, root)
    try:
        sample = read_table(path, nrows=5000)
        columns = list(sample.columns)
        label_col = detect_label_column(columns, sample)
        row_count = 0
        missing_count = 0
        total_cells = 0
        label_counter: Counter = Counter()
        for chunk in read_table_chunks(path):
            row_count += len(chunk)
            missing_count += int(chunk.isna().sum().sum())
            total_cells += int(chunk.shape[0] * chunk.shape[1])
            if label_col and label_col in chunk.columns:
                label_counter.update(chunk[label_col].astype("string").fillna("<NA>").tolist())
        missing_ratio = missing_count / total_cells if total_cells else None
        profile = {
            "dataset": dataset,
            "file_path": rel,
            "size_bytes": path.stat().st_size,
            "rows": row_count,
            "columns": len(columns),
            "fields": json.dumps(columns, ensure_ascii=False),
            "label_column": label_col,
            "label_distribution": json.dumps(dict(label_counter), ensure_ascii=False),
            "missing_value_ratio": missing_ratio,
            "suspicious_leakage_fields": json.dumps(suspicious_fields(columns, LEAKAGE_TOKENS), ensure_ascii=False),
            "suspicious_sensitive_fields": json.dumps(suspicious_fields(columns, SENSITIVE_TOKENS), ensure_ascii=False),
            "read_status": "ok",
        }
        dist_rows = []
        total_labels = sum(label_counter.values())
        for value, count in label_counter.most_common():
            dist_rows.append(
                {
                    "dataset": dataset,
                    "file_path": rel,
                    "label_column": label_col,
                    "label_value": value,
                    "count": count,
                    "percent": count / total_labels if total_labels else None,
                }
            )
        return profile, dist_rows
    except Exception as exc:
        return (
            {
                "dataset": dataset,
                "file_path": rel,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "rows": None,
                "columns": None,
                "fields": "[]",
                "label_column": None,
                "label_distribution": "{}",
                "missing_value_ratio": None,
                "suspicious_leakage_fields": "[]",
                "suspicious_sensitive_fields": "[]",
                "read_status": f"failed: {exc!r}",
            },
            [],
        )


def inspect_processed_file(dataset: str, path: Path, root: Path) -> tuple[dict, list[dict]]:
    rel = safe_relpath(path, root)
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        columns = list(df.columns)
        label_col = "multiclass_label" if "multiclass_label" in df.columns else detect_label_column(columns, df.head(5000))
        label_counter: Counter = Counter()
        if label_col and label_col in df.columns:
            label_counter.update(df[label_col].astype("string").fillna("<NA>").tolist())
        missing_ratio = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])) if df.shape[0] and df.shape[1] else None
        profile = {
            "dataset": dataset,
            "file_path": rel,
            "size_bytes": path.stat().st_size,
            "rows": int(len(df)),
            "columns": int(len(columns)),
            "fields": json.dumps(columns, ensure_ascii=False),
            "label_column": label_col,
            "label_distribution": json.dumps(dict(label_counter), ensure_ascii=False),
            "missing_value_ratio": missing_ratio,
            "suspicious_leakage_fields": json.dumps(suspicious_fields(columns, LEAKAGE_TOKENS), ensure_ascii=False),
            "suspicious_sensitive_fields": json.dumps(suspicious_fields(columns, SENSITIVE_TOKENS), ensure_ascii=False),
            "read_status": "ok",
        }
        dist_rows = []
        total_labels = sum(label_counter.values())
        for value, count in label_counter.most_common():
            dist_rows.append(
                {
                    "dataset": dataset,
                    "file_path": rel,
                    "label_column": label_col,
                    "label_value": value,
                    "count": count,
                    "percent": count / total_labels if total_labels else None,
                }
            )
        return profile, dist_rows
    except Exception as exc:
        return (
            {
                "dataset": dataset,
                "file_path": rel,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "rows": None,
                "columns": None,
                "fields": "[]",
                "label_column": None,
                "label_distribution": "{}",
                "missing_value_ratio": None,
                "suspicious_leakage_fields": "[]",
                "suspicious_sensitive_fields": "[]",
                "read_status": f"failed: {exc!r}",
            },
            [],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect raw and processed datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    profiles: list[dict] = []
    distributions: list[dict] = []
    for dataset in DATASETS:
        files = discover_dataset_files(dataset, root, extract_archives=True)
        if not files:
            profiles.append(
                {
                    "dataset": dataset,
                    "file_path": "",
                    "size_bytes": None,
                    "rows": 0,
                    "columns": 0,
                    "fields": "[]",
                    "label_column": None,
                    "label_distribution": "{}",
                    "missing_value_ratio": None,
                    "suspicious_leakage_fields": "[]",
                    "suspicious_sensitive_fields": "[]",
                    "read_status": "no_tabular_files_found",
                }
            )
            continue
        for path in files:
            profile, dist = inspect_file(dataset, path, root)
            profiles.append(profile)
            distributions.extend(dist)

    processed_candidates = [
        ("IoT-23", root / "data" / "processed" / "IoT-23_flows.parquet"),
        ("Gotham2025", root / "data" / "processed" / "Gotham2025_flows.parquet"),
        ("CICIoT2023", root / "data" / "processed" / "CICIoT2023_sampled.parquet"),
    ]
    for dataset, path in processed_candidates:
        if path.exists():
            profile, dist = inspect_processed_file(dataset, path, root)
            profiles.append(profile)
            distributions.extend(dist)

    out_dir = results_tables_dir(root)
    profile_path = out_dir / "dataset_profile.csv"
    dist_path = out_dir / "label_distribution.csv"
    pd.DataFrame(profiles).to_csv(profile_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(distributions).to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {profile_path}")
    print(f"Wrote {dist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
