from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from build_feature_views import (
    COMMON_CROSS_FEATURES,
    hard_leakage_reasons,
    is_hard_leakage,
    is_legitimate_metadata,
    is_protocol_metadata,
)
from utils.data_io import ensure_dirs, processed_dir, project_root, results_tables_dir, safe_relpath


AUDIT_DATASET_PREFIXES = ("MQTTEEB-D", "MQTT-IoT-IDS2020", "IoT-23")
RUNNABLE_STATUSES = {"ok", "weak_view"}


def norm(col: str) -> str:
    return str(col).lower().replace(".", "_").replace("-", "_").replace(" ", "_")


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def parse_features(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return [str(col) for col in json.loads(value)]
    except Exception:
        return []


def dataset_in_scope(dataset: str) -> bool:
    return any(str(dataset).startswith(prefix) for prefix in AUDIT_DATASET_PREFIXES)


def source_tables(root: Path) -> list[tuple[str, Path]]:
    processed = processed_dir(root)
    out = []
    for dataset, path in [
        ("MQTTEEB-D", processed / "MQTTEEB-D__cleaned.csv"),
        ("MQTT-IoT-IDS2020", processed / "MQTT-IoT-IDS2020__cleaned.csv"),
        ("IoT-23", processed / "IoT-23_flows.parquet"),
    ]:
        if path.exists():
            out.append((dataset, path))
    for path in sorted(processed.glob("MQTTEEB-D_flow_windows_*s.parquet")):
        window = path.stem.replace("MQTTEEB-D_flow_windows_", "")
        out.append((f"MQTTEEB-D_flow_{window}", path))
    return out


def balanced_sample(df: pd.DataFrame, label_col: str, seed: int, max_per_class: int) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby(label_col, dropna=False):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    if not parts:
        return df
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def encode_single_feature(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.90:
        values = numeric.replace([np.inf, -np.inf], np.nan)
        fill = values.median()
        if pd.isna(fill):
            fill = 0.0
        return values.fillna(fill).to_numpy(dtype=float).reshape(-1, 1)
    codes, _ = pd.factorize(series.astype("string").fillna("<NA>"), sort=True)
    return codes.astype(float).reshape(-1, 1)


def single_feature_macro_f1(df: pd.DataFrame, feature: str, seed: int, max_per_class: int) -> float | None:
    if "binary_label" not in df.columns or feature not in df.columns:
        return None
    work = df[[feature, "binary_label"]].copy()
    work["binary_label"] = pd.to_numeric(work["binary_label"], errors="coerce")
    work = work.dropna(subset=["binary_label"])
    if work["binary_label"].nunique() < 2 or work["binary_label"].value_counts().min() < 3:
        return None
    work = balanced_sample(work, "binary_label", seed, max_per_class)
    y = work["binary_label"].astype(int).to_numpy()
    X = encode_single_feature(work[feature])
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(10, int(len(y_train) * 0.002)), class_weight="balanced", random_state=seed)
        clf.fit(X_train, y_train)
        return float(f1_score(y_test, clf.predict(X_test), average="macro", zero_division=0))
    except Exception:
        return None


def classify_feature(feature: str, score: float | None) -> tuple[str, str]:
    if is_hard_leakage(feature):
        return "hard_leakage_removed", "remove_from_all_model_views"
    if is_legitimate_metadata(feature):
        if score is not None and score > 0.90:
            return "legitimate_metadata_retained", "retain_in_audited_metadata_main; conservative_only_removed"
        return "legitimate_metadata_retained", "retain_in_audited_metadata_main"
    if is_protocol_metadata(feature):
        return "suspicious_protocol_metadata_retained", "retain_with_protocol_metadata_note"
    if score is not None and score > 0.90:
        return "conservative_only_removed", "retain_in_main_remove_in_conservative"
    return "retained_metadata", "retain"


def audit_feature_views(root: Path, seed: int, max_per_class: int) -> pd.DataFrame:
    summary_path = results_tables_dir(root) / "feature_views_audited_metadata_main.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run build_feature_views.py first: {summary_path}")
    summary = pd.read_csv(summary_path)
    rows: list[dict] = []
    df_cache: dict[str, pd.DataFrame] = {}
    score_cache: dict[tuple[str, str, str], float | None] = {}
    scoped = summary[summary["dataset"].astype(str).map(dataset_in_scope) & summary["status"].isin(RUNNABLE_STATUSES)]
    for _, view_row in scoped.iterrows():
        dataset = str(view_row["dataset"])
        view = str(view_row["view"])
        view_role = str(view_row.get("view_role", ""))
        rel = str(view_row.get("source_file", ""))
        path = root / rel
        if not path.exists():
            continue
        if rel not in df_cache:
            df_cache[rel] = read_any(path)
        df = df_cache[rel]
        for feature in parse_features(view_row.get("included_features", "[]")):
            cache_key = (dataset, rel, feature)
            if cache_key not in score_cache:
                score_cache[cache_key] = single_feature_macro_f1(df, feature, seed, max_per_class)
            score = score_cache[cache_key]
            classification, action = classify_feature(feature, score)
            if score is not None and score > 0.90 and classification == "legitimate_metadata_retained":
                reason = "single-feature Macro F1 > 0.90; legitimate encrypted-flow metadata, not leakage"
            elif score is not None and score > 0.80:
                reason = "single-feature Macro F1 > 0.80; discriminative metadata"
            else:
                reason = "no leakage indication under revised taxonomy"
            rows.append(
                {
                    "dataset": dataset,
                    "view": view,
                    "view_role": view_role,
                    "feature": feature,
                    "feature_class": classification,
                    "single_feature_macro_f1": score,
                    "action": action,
                    "notes": reason,
                }
            )
    return pd.DataFrame(rows)


def hard_rules_from_sources(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, path in source_tables(root):
        try:
            df = read_any(path)
        except Exception:
            continue
        for col in df.columns:
            reasons = hard_leakage_reasons(col)
            if not reasons:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "source_file": safe_relpath(path, root),
                    "feature": col,
                    "feature_class": "hard_leakage_removed",
                    "exclusion_reason": "; ".join(reasons),
                    "action": "remove_from_all_model_views",
                    "notes": "Hard leakage or sensitive field under revised Stage 2B taxonomy.",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Revised Stage 2B leakage audit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=30_000)
    args = parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    report = audit_feature_views(root, args.seed, args.max_per_class)
    hard_rules = hard_rules_from_sources(root)

    report_path = results_tables_dir(root) / "leakage_audit_report_revised.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    high = report[
        report["single_feature_macro_f1"].notna()
        & (report["single_feature_macro_f1"] > 0.80)
        & report["feature_class"].isin(["legitimate_metadata_retained", "suspicious_protocol_metadata_retained", "conservative_only_removed"])
    ].copy()
    high_path = results_tables_dir(root) / "highly_discriminative_metadata_report.csv"
    high.to_csv(high_path, index=False, encoding="utf-8-sig")

    hard_path = results_tables_dir(root) / "hard_leakage_exclusion_rules.csv"
    hard_rules.to_csv(hard_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {report_path}")
    print(f"Wrote {high_path}")
    print(f"Wrote {hard_path}")
    if not high.empty:
        print(high[["dataset", "view_role", "feature", "feature_class", "single_feature_macro_f1", "action"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
