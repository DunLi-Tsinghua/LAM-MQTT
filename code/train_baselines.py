from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.data_io import ensure_dirs, project_root, results_metrics_dir, results_tables_dir
from utils.metrics import binary_classification_metrics


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
}
RUNNABLE_STATUSES = {"ok", "weak_view"}


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def parse_feature_list(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return [str(col) for col in parsed]
    except Exception:
        return []


def sample_for_training(df: pd.DataFrame, sample_size: int, max_per_class: int, seed: int) -> pd.DataFrame:
    if "binary_label" not in df.columns:
        return df
    parts = []
    for _, group in df.groupby("binary_label", dropna=False):
        cap = min(len(group), max_per_class)
        if cap < len(group):
            parts.append(group.sample(n=cap, random_state=seed))
        else:
            parts.append(group)
    sampled = pd.concat(parts, ignore_index=True)
    if sample_size and len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(X: pd.DataFrame, y: pd.Series, seed: int):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def positive_scores(model: Pipeline, X_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = list(model.named_steps["model"].classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        if proba.shape[1] == 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if np.ndim(scores) == 1:
            return scores
    return None


def encoded_feature_count(pipeline: Pipeline) -> int | None:
    try:
        return int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))
    except Exception:
        return None


def fit_and_score(X_train, y_train, X_test, y_test, model_name: str, estimator, feature_count: int, seed: int, extra: dict) -> dict:
    if hasattr(estimator, "random_state"):
        estimator = clone(estimator)
        estimator.set_params(random_state=seed)
    else:
        estimator = clone(estimator)
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(X_train, scale_numeric=model_name == "Logistic Regression")),
            ("model", estimator),
        ]
    )
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    pred_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    infer_time = time.perf_counter() - pred_start
    metrics = binary_classification_metrics(y_test, y_pred, positive_scores(pipeline, X_test))
    return {
        **extra,
        "model": model_name,
        "status": "ok",
        **metrics,
        "training_time_seconds": train_time,
        "inference_time_per_sample_seconds": infer_time / len(X_test) if len(X_test) else None,
        "number_of_features": feature_count,
        "encoded_number_of_features": encoded_feature_count(pipeline),
        "train_sample_size": len(X_train),
        "test_sample_size": len(X_test),
        "random_seed": seed,
    }


def train_within(row: pd.Series, root: Path, sample_size: int, max_per_class: int, seed: int) -> list[dict]:
    dataset = row["dataset"]
    view = row["view"]
    status = row.get("status")
    source_file = row.get("source_file", "")
    features = parse_feature_list(row.get("included_features", "[]"))
    if status not in RUNNABLE_STATUSES or not features or not isinstance(source_file, str) or not source_file:
        return [{"dataset": dataset, "view": view, "model": name, "status": f"skipped_{status}", "number_of_features": len(features)} for name in MODELS]
    path = root / source_file
    if not path.exists():
        return [{"dataset": dataset, "view": view, "model": name, "status": "skipped_missing_view", "number_of_features": len(features)} for name in MODELS]
    df = sample_for_training(read_any(path), sample_size, max_per_class, seed)
    y = pd.to_numeric(df["binary_label"], errors="coerce")
    valid = y.notna()
    X = df.loc[valid, features]
    y = y.loc[valid].astype(int)
    if len(features) == 0 or y.nunique() < 2 or y.value_counts().min() < 3:
        return [{"dataset": dataset, "view": view, "model": name, "status": "skipped_insufficient_features_or_classes", "number_of_features": len(features)} for name in MODELS]
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    except Exception as exc:
        return [{"dataset": dataset, "view": view, "model": name, "status": f"skipped_split_failed: {exc!r}", "number_of_features": len(features)} for name in MODELS]
    rows = []
    for model_name, estimator in MODELS.items():
        rows.append(
            fit_and_score(
                X_train,
                y_train,
                X_test,
                y_test,
                model_name,
                estimator,
                len(features),
                seed,
                {
                    "dataset": dataset,
                    "view": view,
                    "validation_sample_size": len(X_val),
                    "feature_columns": json.dumps(features, ensure_ascii=False),
                },
            )
        )
    return rows


def external_validation(summary: pd.DataFrame, root: Path, sample_size: int, max_per_class: int, seed: int) -> list[dict]:
    rows = []
    primary = summary[
        summary["dataset"].astype(str).str.startswith("MQTTEEB-D_flow_")
        & (summary["view"] == "strict_encrypted_flow_metadata")
        & summary["status"].isin(RUNNABLE_STATUSES)
    ]
    external = summary[
        ~summary["dataset"].astype(str).str.startswith("MQTTEEB-D")
        & (summary["view"] == "strict_encrypted_flow_metadata")
        & summary["status"].isin(RUNNABLE_STATUSES)
    ]
    if primary.empty or external.empty:
        return [
            {
                "train_dataset": "MQTTEEB-D_flow",
                "test_dataset": "",
                "view": "strict_encrypted_flow_metadata",
                "model": name,
                "status": "skipped_no_primary_or_external_view",
            }
            for name in MODELS
        ]
    for _, train_row in primary.iterrows():
        train_features = set(parse_feature_list(train_row.get("included_features", "[]")))
        train_path = root / train_row["source_file"]
        if not train_path.exists():
            continue
        train_df = sample_for_training(read_any(train_path), sample_size, max_per_class, seed)
        for _, test_row in external.iterrows():
            test_features = set(parse_feature_list(test_row.get("included_features", "[]")))
            common = sorted(train_features & test_features)
            test_path = root / test_row["source_file"]
            if len(common) < 1 or not test_path.exists():
                for model_name in MODELS:
                    rows.append(
                        {
                            "train_dataset": train_row["dataset"],
                            "test_dataset": test_row["dataset"],
                            "view": "strict_encrypted_flow_metadata",
                            "model": model_name,
                            "status": "skipped_no_common_features",
                            "number_of_features": len(common),
                        }
                    )
                continue
            test_df = sample_for_training(read_any(test_path), sample_size, max_per_class, seed)
            train_y = pd.to_numeric(train_df["binary_label"], errors="coerce")
            test_y = pd.to_numeric(test_df["binary_label"], errors="coerce")
            train_valid = train_y.notna()
            test_valid = test_y.notna()
            X_train = train_df.loc[train_valid, common]
            y_train = train_y.loc[train_valid].astype(int)
            X_test = test_df.loc[test_valid, common]
            y_test = test_y.loc[test_valid].astype(int)
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                for model_name in MODELS:
                    rows.append(
                        {
                            "train_dataset": train_row["dataset"],
                            "test_dataset": test_row["dataset"],
                            "view": "strict_encrypted_flow_metadata",
                            "model": model_name,
                            "status": "skipped_insufficient_classes",
                            "number_of_features": len(common),
                        }
                    )
                continue
            for model_name, estimator in MODELS.items():
                rows.append(
                    fit_and_score(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        model_name,
                        estimator,
                        len(common),
                        seed,
                        {
                            "train_dataset": train_row["dataset"],
                            "test_dataset": test_row["dataset"],
                            "view": "strict_encrypted_flow_metadata",
                            "feature_columns": json.dumps(common, ensure_ascii=False),
                        },
                    )
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 1.5 binary baselines.")
    parser.add_argument("--task", default="binary", choices=["binary"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--max-per-class", type=int, default=10_000)
    args = parser.parse_args()

    root = project_root()
    ensure_dirs(root)
    summary_path = results_tables_dir(root) / "feature_views_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run build_feature_views.py first: {summary_path}")
    summary = pd.read_csv(summary_path)
    within_rows: list[dict] = []
    for _, row in summary.iterrows():
        within_rows.extend(train_within(row, root, args.sample_size, args.max_per_class, args.seed))
    within_path = results_metrics_dir(root) / "within_dataset_baseline_results.csv"
    pd.DataFrame(within_rows).to_csv(within_path, index=False, encoding="utf-8-sig")

    external_rows = external_validation(summary, root, args.sample_size, args.max_per_class, args.seed)
    external_path = results_metrics_dir(root) / "external_dataset_baseline_results.csv"
    pd.DataFrame(external_rows).to_csv(external_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {within_path}")
    print(f"Wrote {external_path}")
    print(pd.DataFrame(within_rows)[["dataset", "view", "model", "status", "accuracy", "macro_f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
