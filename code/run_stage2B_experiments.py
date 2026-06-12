from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from build_feature_views import COMMON_CROSS_FEATURES
from split_strategies import available_splits
from utils.data_io import ensure_dirs, project_root


SEEDS = [42, 7, 13, 21, 100]
SUPERVISED_MODELS = ["Logistic Regression", "Random Forest", "Extra Trees", "HistGradientBoostingClassifier"]
BINARY_MODELS = SUPERVISED_MODELS + ["Isolation Forest"]
OPTIONAL_SKIPPED_MODELS = ["MLP"]
RUNNABLE_STATUSES = {"ok", "weak_view"}
VIEW_ROLES = ["full", "broker", "strict"]
MAIN_DATASETS = ["MQTTEEB-D", "MQTT-IoT-IDS2020"]
FLOW_DATASETS = ["MQTTEEB-D_flow_1s", "MQTTEEB-D_flow_5s", "MQTTEEB-D_flow_10s"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


@dataclass
class View:
    dataset: str
    policy: str
    view_role: str
    view: str
    status: str
    path: Path
    features: list[str]
    notes: str = ""


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_pub(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_features(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return [str(x) for x in json.loads(value)]
    except Exception:
        return []


def load_views(root: Path, policy: str) -> dict[tuple[str, str], View]:
    table = root / "results" / "tables" / f"feature_views_{policy}.csv"
    df = pd.read_csv(table)
    views: dict[tuple[str, str], View] = {}
    for _, row in df.iterrows():
        source = str(row.get("source_file", ""))
        path = root / source if source else Path("")
        view = View(
            dataset=str(row["dataset"]),
            policy=str(row["policy"]),
            view_role=str(row["view_role"]),
            view=str(row["view"]),
            status=str(row["status"]),
            path=path,
            features=parse_features(row.get("included_features", "[]")),
            notes=str(row.get("notes", "")),
        )
        views[(view.dataset, view.view_role)] = view
    return views


def make_preprocessor(X: pd.DataFrame, model_name: str) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scaler", StandardScaler()))
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]), categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def model_factory(name: str, seed: int, task: str):
    if name == "Logistic Regression":
        kwargs = {"max_iter": 1000, "random_state": seed, "class_weight": "balanced"}
        if task == "multiclass":
            kwargs["multi_class"] = "auto"
        return LogisticRegression(**kwargs)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1, class_weight="balanced_subsample")
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=80, random_state=seed, n_jobs=-1, class_weight="balanced")
    if name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed, early_stopping=True)
    if name == "Isolation Forest":
        return IsolationForest(n_estimators=80, contamination="auto", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def prepare_task_frame(df: pd.DataFrame, features: list[str], task: str, min_class_count: int = 3):
    label_col = "binary_label" if task == "binary" else "multiclass_label"
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing[:5]}")
    work = df[features + [label_col]].copy()
    if task == "binary":
        work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
        work = work.dropna(subset=[label_col])
        y = work[label_col].astype(int).reset_index(drop=True)
        label_names = ["benign", "attack"]
    else:
        work[label_col] = work[label_col].astype("string").fillna("unknown")
        counts = work[label_col].value_counts()
        keep = counts[counts >= min_class_count].index
        work = work[work[label_col].isin(keep)].copy()
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(work[label_col].astype(str)), index=work.index).reset_index(drop=True)
        label_names = encoder.classes_.tolist()
    X = work[features].reset_index(drop=True)
    if y.nunique() < 2 or y.value_counts().min() < 3:
        raise ValueError("not enough classes after label preparation")
    return X, y, label_names


def score_metrics(y_true, y_pred, y_score=None, task: str = "binary") -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    if task == "binary" and y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
        try:
            out["pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            pass
    return out


def fit_predict(model_name: str, X_train, y_train, X_test, seed: int, task: str):
    start = time.perf_counter()
    if model_name == "Isolation Forest":
        if task != "binary":
            raise ValueError("Isolation Forest supports binary anomaly evaluation only")
        preprocessor = make_preprocessor(X_train, model_name)
        benign_mask = y_train == 0
        if benign_mask.sum() < 10:
            raise ValueError("not enough benign training samples for Isolation Forest")
        X_train_enc = preprocessor.fit_transform(X_train.loc[benign_mask])
        model = model_factory(model_name, seed, task)
        model.fit(X_train_enc)
        train_time = time.perf_counter() - start
        pred_start = time.perf_counter()
        X_test_enc = preprocessor.transform(X_test)
        raw = model.predict(X_test_enc)
        y_pred = np.where(raw == -1, 1, 0)
        y_score = -model.decision_function(X_test_enc)
        infer_time = time.perf_counter() - pred_start
        return y_pred, y_score, train_time, infer_time

    estimator = model_factory(model_name, seed, task)
    pipeline = Pipeline([("preprocess", make_preprocessor(X_train, model_name)), ("model", estimator)])
    fit_kwargs = {}
    if model_name == "HistGradientBoostingClassifier":
        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, **fit_kwargs)
    train_time = time.perf_counter() - start
    pred_start = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    infer_time = time.perf_counter() - pred_start
    y_score = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)
        if task == "binary":
            classes = list(pipeline.named_steps["model"].classes_)
            y_score = proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
        else:
            y_score = proba
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X_test)
    return y_pred, y_score, train_time, infer_time


def split_arrays(X: pd.DataFrame, y: pd.Series, split: dict):
    return (
        X.iloc[split["train_idx"]],
        X.iloc[split["val_idx"]],
        X.iloc[split["test_idx"]],
        y.iloc[split["train_idx"]].to_numpy(),
        y.iloc[split["val_idx"]].to_numpy(),
        y.iloc[split["test_idx"]].to_numpy(),
    )


def split_is_valid(y_train, y_test) -> tuple[bool, str]:
    if len(np.unique(y_train)) < 2:
        return False, "invalid_split_train_single_class"
    if len(np.unique(y_test)) < 2:
        return False, "invalid_split_test_single_class"
    return True, "ok"


def run_one_split(dataset: str, policy: str, view_role: str, view_name: str, task: str, model_name: str, split_type: str, split: dict, X, y, features, seed: int) -> dict:
    X_train, X_val, X_test, y_train, y_val, y_test = split_arrays(X, y, split)
    valid, status = split_is_valid(y_train, y_test)
    base = {
        "dataset": dataset,
        "policy": policy,
        "view_role": view_role,
        "view": view_name,
        "task": task,
        "model": model_name,
        "split_type": split_type,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": len(features),
        "seed": seed,
    }
    if not valid:
        return {**base, "status": status}
    if task == "multiclass" and model_name == "Isolation Forest":
        return {**base, "status": "skipped_model_not_applicable_to_multiclass"}
    try:
        y_pred, y_score, train_time, infer_time = fit_predict(model_name, X_train, y_train, X_test, seed, task)
        return {
            **base,
            "status": "ok",
            **score_metrics(y_test, y_pred, y_score if task == "binary" else None, task),
            "training_time_seconds": train_time,
            "inference_time_per_sample_seconds": infer_time / len(X_test) if len(X_test) else np.nan,
        }
    except Exception as exc:
        return {**base, "status": f"failed: {exc!r}"}


def run_within_policy(root: Path, policy: str, seed: int) -> pd.DataFrame:
    views = load_views(root, policy)
    rows: list[dict] = []
    for dataset in MAIN_DATASETS:
        for view_role in VIEW_ROLES:
            view = views.get((dataset, view_role))
            if not view or view.status not in RUNNABLE_STATUSES or not view.path.exists() or not view.features:
                continue
            df = read_any(view.path)
            for task in ["binary", "multiclass"]:
                try:
                    X, y, _ = prepare_task_frame(df, view.features, task)
                    split_source = X.copy()
                    split_source["binary_label"] = y if task == "binary" else pd.Series(1, index=X.index)
                    splits = available_splits(split_source, y, seed)
                except Exception as exc:
                    rows.append({"dataset": dataset, "policy": policy, "view_role": view_role, "view": view.view, "task": task, "model": "", "split_type": "", "status": f"skipped: {exc!r}", "n_features": len(view.features), "seed": seed})
                    continue
                models = BINARY_MODELS if task == "binary" else SUPERVISED_MODELS
                for split_type, split in splits.items():
                    for model_name in models:
                        rows.append(run_one_split(dataset, policy, view_role, view.view, task, model_name, split_type, split, X, y, view.features, seed))
                for skipped_model in OPTIONAL_SKIPPED_MODELS:
                    rows.append(
                        {
                            "dataset": dataset,
                            "policy": policy,
                            "view_role": view_role,
                            "view": view.view,
                            "task": task,
                            "model": skipped_model,
                            "split_type": "all",
                            "status": "skipped_optional_too_slow_or_unstable",
                            "n_features": len(view.features),
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def train_source_split(X: pd.DataFrame, y: pd.Series, seed: int):
    idx = np.arange(len(y))
    train_idx, temp_idx, y_train, y_temp = train_test_split(idx, y, test_size=0.30, random_state=seed, stratify=y)
    val_idx, _ = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=y_temp)
    return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx].to_numpy(), y.iloc[val_idx].to_numpy()


def run_cross_pair(train_view: View, test_view: View, common: list[str], seed: int, models: list[str] | None = None) -> list[dict]:
    rows: list[dict] = []
    models = models or BINARY_MODELS
    train_df = read_any(train_view.path)
    test_df = read_any(test_view.path)
    try:
        X_train_all, y_train_all, _ = prepare_task_frame(train_df, common, "binary")
        X_test, y_test, _ = prepare_task_frame(test_df, common, "binary")
        X_train, X_val, y_train, y_val = train_source_split(X_train_all, y_train_all, seed)
    except Exception as exc:
        for model_name in models:
            rows.append({"train_dataset": train_view.dataset, "test_dataset": test_view.dataset, "policy": train_view.policy, "view_role": "strict", "task": "binary", "model": model_name, "status": f"skipped: {exc!r}", "n_features": len(common), "seed": seed})
        return rows
    for model_name in models:
        base = {
            "train_dataset": train_view.dataset,
            "test_dataset": test_view.dataset,
            "policy": train_view.policy,
            "view_role": "strict",
            "task": "binary",
            "model": model_name,
            "status": "ok",
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "n_features": len(common),
            "seed": seed,
        }
        try:
            y_pred, y_score, train_time, infer_time = fit_predict(model_name, X_train, y_train, X_test, seed, "binary")
            rows.append(
                {
                    **base,
                    **score_metrics(y_test, y_pred, y_score, "binary"),
                    "training_time_seconds": train_time,
                    "inference_time_per_sample_seconds": infer_time / len(X_test) if len(X_test) else np.nan,
                    "features": json.dumps(common, ensure_ascii=False),
                }
            )
        except Exception as exc:
            rows.append({**base, "status": f"failed: {exc!r}", "features": json.dumps(common, ensure_ascii=False)})
    return rows


def run_mqtt_cross_dataset(root: Path, seed: int) -> pd.DataFrame:
    views = load_views(root, "audited_metadata_main")
    rows: list[dict] = []
    common_rows: list[dict] = []
    for flow_dataset in FLOW_DATASETS:
        for train_dataset, test_dataset in [(flow_dataset, "MQTT-IoT-IDS2020"), ("MQTT-IoT-IDS2020", flow_dataset)]:
            train_view = views[(train_dataset, "strict")]
            test_view = views[(test_dataset, "strict")]
            common = [feature for feature in COMMON_CROSS_FEATURES if feature in train_view.features and feature in test_view.features]
            common_rows.append(
                {
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "view": "common_cross_dataset_metadata",
                    "n_common_features": len(common),
                    "common_features": json.dumps(common, ensure_ascii=False),
                }
            )
            rows.extend(run_cross_pair(train_view, test_view, common, seed))
    write_csv(pd.DataFrame(common_rows), root / "results" / "tables" / "mqtt_cross_dataset_common_features_main.csv")
    return pd.DataFrame(rows)


def run_iot23_external(root: Path, seed: int) -> pd.DataFrame:
    views = load_views(root, "audited_metadata_main")
    iot = views.get(("IoT-23", "strict"))
    rows: list[dict] = []
    if not iot:
        return pd.DataFrame([{"validation_type": "external IoT robustness validation", "status": "skipped_no_iot23_view"}])
    for source_dataset in ["MQTTEEB-D_flow_5s", "MQTT-IoT-IDS2020"]:
        source = views[(source_dataset, "strict")]
        common = [feature for feature in COMMON_CROSS_FEATURES if feature in source.features and feature in iot.features]
        pair_rows = run_cross_pair(source, iot, common, seed)
        for row in pair_rows:
            row["validation_type"] = "external IoT robustness validation"
            row["features"] = json.dumps(common, ensure_ascii=False)
        rows.extend(pair_rows)
    return pd.DataFrame(rows)


def feature_family(features: list[str], family: str) -> list[str]:
    low = {f: f.lower() for f in features}
    if family == "packet_count_only":
        return [f for f, l in low.items() if "packet_count" in l or "num_pkts" in l]
    if family == "byte_statistics_only":
        return [f for f, l in low.items() if ("byte" in l or "pkt_len" in l or "num_bytes" in l) and "per_second" not in l]
    if family == "iat_statistics_only":
        return [f for f, l in low.items() if "iat" in l]
    if family == "rate_features_only":
        return [f for f, l in low.items() if "per_second" in l or "rate" in l or "burstiness" in l]
    if family == "tcp_flags_only":
        return [f for f, l in low.items() if "flag" in l]
    if family == "all_metadata":
        return features[:]
    if family == "all_without_byte_stats":
        return [f for f, l in low.items() if not (("byte" in l or "pkt_len" in l or "num_bytes" in l) and "per_second" not in l)]
    if family == "all_without_iat":
        return [f for f, l in low.items() if "iat" not in l]
    if family == "all_without_rate":
        return [f for f, l in low.items() if not ("per_second" in l or "rate" in l or "burstiness" in l)]
    raise ValueError(family)


def run_feature_family_ablation(root: Path, seed: int) -> pd.DataFrame:
    views = load_views(root, "audited_metadata_main")
    targets = [("MQTTEEB-D_flow_5s", "strict"), ("MQTT-IoT-IDS2020", "strict")]
    families = [
        "packet_count_only",
        "byte_statistics_only",
        "iat_statistics_only",
        "rate_features_only",
        "tcp_flags_only",
        "all_metadata",
        "all_without_byte_stats",
        "all_without_iat",
        "all_without_rate",
    ]
    rows: list[dict] = []
    for dataset, role in targets:
        view = views[(dataset, role)]
        df = read_any(view.path)
        X_all, y, _ = prepare_task_frame(df, view.features, "binary")
        splits = available_splits(X_all.assign(binary_label=y), y, seed)
        split = splits["random_stratified"]
        for family in families:
            feats = feature_family(view.features, family)
            if not feats:
                rows.append({"dataset": dataset, "view_role": role, "feature_family": family, "model": "", "status": "skipped_no_features", "n_features": 0, "seed": seed})
                continue
            X = X_all[feats]
            for model_name in BINARY_MODELS:
                rows.append(run_one_split(dataset, "audited_metadata_main", role, view.view, "binary", model_name, "random_stratified", split, X, y, feats, seed) | {"feature_family": family})
    return pd.DataFrame(rows)


def evaluate_binary_random_view(view: View, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    if view.status not in RUNNABLE_STATUSES or not view.path.exists() or not view.features:
        return pd.DataFrame(
            [
                {
                    "dataset": view.dataset,
                    "policy": view.policy,
                    "view_role": view.view_role,
                    "view": view.view,
                    "task": "binary",
                    "model": "",
                    "split_type": "random_stratified",
                    "status": "skipped_missing_view_or_features",
                    "n_features": len(view.features),
                    "seed": seed,
                }
            ]
        )
    df = read_any(view.path)
    X, y, _ = prepare_task_frame(df, view.features, "binary")
    split = available_splits(X.assign(binary_label=y), y, seed)["random_stratified"]
    for model_name in BINARY_MODELS:
        rows.append(run_one_split(view.dataset, view.policy, view.view_role, view.view, "binary", model_name, "random_stratified", split, X, y, view.features, seed))
    return pd.DataFrame(rows)


def run_crypto_observability_ablation(root: Path, within_main: pd.DataFrame, within_cons: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    ok_main = within_main[(within_main["status"] == "ok") & (within_main["task"] == "binary") & (within_main["split_type"] == "random_stratified")].copy()
    ok_cons = within_cons[(within_cons["status"] == "ok") & (within_cons["task"] == "binary") & (within_cons["split_type"] == "random_stratified")].copy()
    main_views = load_views(root, "audited_metadata_main")
    cons_views = load_views(root, "conservative_anti_leakage")
    evaluated_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    def frame_for(dataset: str, role: str, policy: str, existing: pd.DataFrame) -> pd.DataFrame:
        sub = existing[(existing["dataset"] == dataset) & (existing["view_role"] == role)]
        if not sub.empty:
            return sub
        key = (dataset, role, policy)
        if key not in evaluated_cache:
            view = (main_views if policy == "audited_metadata_main" else cons_views)[(dataset, role)]
            evaluated_cache[key] = evaluate_binary_random_view(view, seed)
        return evaluated_cache[key][evaluated_cache[key]["status"] == "ok"]

    specs = [
        ("MQTTEEB-D", "full", "audited_metadata_main", "full-feature audited", ok_main),
        ("MQTTEEB-D", "broker", "audited_metadata_main", "broker-side audited", ok_main),
        ("MQTTEEB-D_flow_1s", "strict", "audited_metadata_main", "strict encrypted-flow 1s", ok_main),
        ("MQTTEEB-D_flow_5s", "strict", "audited_metadata_main", "strict encrypted-flow 5s", ok_main),
        ("MQTTEEB-D_flow_10s", "strict", "audited_metadata_main", "strict encrypted-flow 10s", ok_main),
        ("MQTTEEB-D_flow_5s", "strict", "conservative_anti_leakage", "conservative strict 5s", ok_cons),
        ("MQTT-IoT-IDS2020", "full", "audited_metadata_main", "full-feature audited", ok_main),
        ("MQTT-IoT-IDS2020", "broker", "audited_metadata_main", "broker-side audited", ok_main),
        ("MQTT-IoT-IDS2020", "strict", "audited_metadata_main", "strict encrypted-flow audited", ok_main),
        ("MQTT-IoT-IDS2020", "strict", "conservative_anti_leakage", "conservative strict", ok_cons),
    ]
    for dataset, role, policy, label, frame in specs:
        sub = frame_for(dataset, role, policy, frame)
        if sub.empty:
            continue
        best = sub.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
        best["comparison_view"] = label
        rows.append(best)
    return pd.DataFrame(rows)


def fit_feature_importance(root: Path, seed: int) -> pd.DataFrame:
    views = load_views(root, "audited_metadata_main")
    rows = []
    for dataset in ["MQTTEEB-D_flow_5s", "MQTT-IoT-IDS2020"]:
        view = views[(dataset, "strict")]
        df = read_any(view.path)
        X, y, _ = prepare_task_frame(df, view.features, "binary")
        split = available_splits(X.assign(binary_label=y), y, seed)["random_stratified"]
        X_train, _, X_test, y_train, _, y_test = split_arrays(X, y, split)
        pipeline = Pipeline([("preprocess", make_preprocessor(X_train, "Extra Trees")), ("model", model_factory("Extra Trees", seed, "binary"))])
        pipeline.fit(X_train, y_train)
        try:
            names = pipeline.named_steps["preprocess"].get_feature_names_out()
        except Exception:
            names = np.array(view.features)
        importances = pipeline.named_steps["model"].feature_importances_
        for name, importance in sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:20]:
            rows.append({"dataset": dataset, "view_role": "strict", "model": "Extra Trees", "feature": str(name), "importance": float(importance)})
    return pd.DataFrame(rows)


def run_multiseed_stability(root: Path, within_main: pd.DataFrame, cross_main: pd.DataFrame) -> pd.DataFrame:
    views = load_views(root, "audited_metadata_main")
    rows: list[dict] = []
    within_specs = [
        ("MQTTEEB-D_flow_5s", "strict", "MQTTEEB-D strict 5s binary"),
        ("MQTT-IoT-IDS2020", "strict", "MQTT-IoT-IDS2020 strict binary"),
    ]
    for dataset, role, label in within_specs:
        prior = within_main[
            (within_main["dataset"] == dataset)
            & (within_main["view_role"] == role)
            & (within_main["task"] == "binary")
            & (within_main["split_type"] == "random_stratified")
            & (within_main["status"] == "ok")
        ]
        model_name = prior.sort_values("macro_f1", ascending=False).iloc[0]["model"] if not prior.empty else "HistGradientBoostingClassifier"
        view = views[(dataset, role)]
        df = read_any(view.path)
        X, y, _ = prepare_task_frame(df, view.features, "binary")
        for seed in SEEDS:
            split = available_splits(X.assign(binary_label=y), y, seed)["random_stratified"]
            row = run_one_split(dataset, "audited_metadata_main", role, view.view, "binary", model_name, "random_stratified", split, X, y, view.features, seed)
            row["stability_target"] = label
            rows.append(row)

    cross_specs = [
        ("MQTTEEB-D_flow_5s", "MQTT-IoT-IDS2020", "MQTTEEB-D strict 5s -> MQTT-IoT-IDS2020 strict"),
        ("MQTT-IoT-IDS2020", "MQTTEEB-D_flow_5s", "MQTT-IoT-IDS2020 strict -> MQTTEEB-D strict 5s"),
    ]
    for train_dataset, test_dataset, label in cross_specs:
        prior = cross_main[(cross_main["train_dataset"] == train_dataset) & (cross_main["test_dataset"] == test_dataset) & (cross_main["status"] == "ok")]
        model_name = prior.sort_values("macro_f1", ascending=False).iloc[0]["model"] if not prior.empty else "Extra Trees"
        train_view = views[(train_dataset, "strict")]
        test_view = views[(test_dataset, "strict")]
        common = [feature for feature in COMMON_CROSS_FEATURES if feature in train_view.features and feature in test_view.features]
        for seed in SEEDS:
            for row in run_cross_pair(train_view, test_view, common, seed, models=[model_name]):
                row["stability_target"] = label
                rows.append(row)

    runs = pd.DataFrame(rows)
    ok = runs[runs["status"] == "ok"].copy()
    summary = (
        ok.groupby(["stability_target", "model"], as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            n_runs=("macro_f1", "count"),
        )
    )
    write_csv(runs, root / "results" / "metrics" / "multiseed_stability_revised_runs.csv")
    return summary


def best_by(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    return ok.sort_values("macro_f1", ascending=False).groupby(keys, as_index=False).first()


def plot_bar(df: pd.DataFrame, x: str, y: str, hue: str | None, title: str, ylabel: str, stem: Path) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    fig_w = max(6.5, min(11, 0.42 * len(plot_df[x].astype(str).unique()) + 3.5))
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    if hue and hue in plot_df.columns:
        categories = plot_df[hue].astype(str).unique().tolist()
        xcats = plot_df[x].astype(str).unique().tolist()
        width = 0.78 / max(1, len(categories))
        positions = np.arange(len(xcats))
        for i, cat in enumerate(categories):
            sub = plot_df[plot_df[hue].astype(str) == cat]
            vals = [float(sub[sub[x].astype(str) == xc][y].iloc[0]) if not sub[sub[x].astype(str) == xc].empty else np.nan for xc in xcats]
            ax.bar(positions + (i - (len(categories) - 1) / 2) * width, vals, width=width, label=cat)
        ax.set_xticks(positions)
        ax.set_xticklabels(xcats, rotation=35, ha="right")
        ax.legend(loc="best")
    else:
        ax.bar(plot_df[x].astype(str), plot_df[y].astype(float), color="#4C78A8")
        ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, min(1.05, max(0.1, float(plot_df[y].max()) * 1.15)))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_pub(fig, stem)


def conceptual_figures(root: Path) -> None:
    fig_dir = root / "results" / "figures"
    rows = []
    concepts = [
        ("Fig1_problem_motivation", "Problem motivation", ["Encrypted MQTT traffic", "Payload/topic unavailable", "Metadata-only IDS"]),
        ("Fig2_framework", "Experimental framework", ["Raw open datasets", "Leakage audit", "Feature policies", "Robust evaluation"]),
        ("Fig3_crypto_observability_views", "Crypto-observability views", ["Full audited metadata", "Broker-side metadata", "Strict encrypted-flow metadata"]),
    ]
    for stem, title, labels in concepts:
        fig, ax = plt.subplots(figsize=(6.2, 2.2))
        ax.axis("off")
        xs = np.linspace(0.15, 0.85, len(labels))
        for i, (x, label) in enumerate(zip(xs, labels)):
            ax.text(x, 0.55, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#EEF3F7", ec="#4C78A8", lw=0.8))
            rows.append({"figure": stem, "step": i + 1, "label": label})
            if i < len(labels) - 1:
                ax.annotate("", xy=(xs[i + 1] - 0.09, 0.55), xytext=(x + 0.09, 0.55), arrowprops=dict(arrowstyle="->", lw=0.9, color="#4C78A8"))
        ax.set_title(title)
        fig.tight_layout()
        save_pub(fig, fig_dir / stem)
    write_csv(pd.DataFrame(rows), fig_dir / "conceptual_figures_source.csv")


def generate_figures(root: Path, within_main: pd.DataFrame, ablation: pd.DataFrame, cross: pd.DataFrame, family: pd.DataFrame, importance: pd.DataFrame) -> None:
    fig_dir = root / "results" / "figures"
    conceptual_figures(root)

    within_best = best_by(within_main[(within_main["task"] == "binary") & (within_main["split_type"] == "random_stratified")], ["dataset", "view_role"])
    if not within_best.empty:
        within_best["target"] = within_best["dataset"] + " / " + within_best["view_role"]
        write_csv(within_best, fig_dir / "Fig4_within_dataset_main_macro_f1.csv")
        plot_bar(within_best, "target", "macro_f1", "model", "Within-dataset main Macro F1", "Macro F1", fig_dir / "Fig4_within_dataset_main_macro_f1")

    if not ablation.empty:
        abl_best = best_by(ablation, ["dataset", "comparison_view"])
        write_csv(abl_best, fig_dir / "Fig5_crypto_observability_ablation_revised.csv")
        abl_best["target"] = abl_best["dataset"] + " / " + abl_best["comparison_view"]
        plot_bar(abl_best, "target", "macro_f1", "model", "Crypto-observability ablation", "Macro F1", fig_dir / "Fig5_crypto_observability_ablation_revised")
        # Compatibility filename requested in Experiment 2.
        plot_bar(abl_best, "target", "macro_f1", "model", "Crypto-observability ablation", "Macro F1", fig_dir / "Fig_crypto_observability_ablation_revised")

    cross_best = best_by(cross, ["train_dataset", "test_dataset"])
    if not cross_best.empty:
        cross_best["direction"] = cross_best["train_dataset"] + " -> " + cross_best["test_dataset"]
        write_csv(cross_best, fig_dir / "Fig6_mqtt_to_mqtt_cross_dataset_main.csv")
        plot_bar(cross_best, "direction", "macro_f1", "model", "MQTT-to-MQTT cross-dataset main results", "Macro F1", fig_dir / "Fig6_mqtt_to_mqtt_cross_dataset_main")
        plot_bar(cross_best, "direction", "macro_f1", "model", "MQTT-to-MQTT cross-dataset main results", "Macro F1", fig_dir / "Fig_mqtt_to_mqtt_cross_dataset_main")

    family_best = best_by(family, ["dataset", "feature_family"])
    if not family_best.empty:
        family_best["target"] = family_best["dataset"] + " / " + family_best["feature_family"]
        write_csv(family_best, fig_dir / "Fig7_feature_family_ablation.csv")
        plot_bar(family_best, "target", "macro_f1", "model", "Feature-family ablation", "Macro F1", fig_dir / "Fig7_feature_family_ablation")
        plot_bar(family_best, "target", "macro_f1", "model", "Feature-family ablation", "Macro F1", fig_dir / "Fig_feature_family_ablation")

    if not importance.empty:
        imp = importance.sort_values("importance", ascending=False).groupby("dataset").head(12)
        write_csv(imp, fig_dir / "Fig8_feature_importance.csv")
        fig, axes = plt.subplots(1, len(imp["dataset"].unique()), figsize=(8, 3.6), sharex=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        for ax, (dataset, sub) in zip(axes, imp.groupby("dataset")):
            sub = sub.sort_values("importance")
            ax.barh(sub["feature"], sub["importance"], color="#4C78A8")
            ax.set_title(dataset)
            ax.set_xlabel("Importance")
        fig.tight_layout()
        save_pub(fig, fig_dir / "Fig8_feature_importance")


def write_summary(root: Path, within_main: pd.DataFrame, within_cons: pd.DataFrame, ablation: pd.DataFrame, cross: pd.DataFrame, iot23: pd.DataFrame, family: pd.DataFrame, stability: pd.DataFrame) -> None:
    tables = root / "results" / "tables"
    hard = pd.read_csv(tables / "hard_leakage_exclusion_rules.csv") if (tables / "hard_leakage_exclusion_rules.csv").exists() else pd.DataFrame()
    high = pd.read_csv(tables / "highly_discriminative_metadata_report.csv") if (tables / "highly_discriminative_metadata_report.csv").exists() else pd.DataFrame()
    main_views = pd.read_csv(tables / "feature_views_audited_metadata_main.csv")
    cons_views = pd.read_csv(tables / "feature_views_conservative_anti_leakage.csv")

    main_best = best_by(within_main, ["dataset", "task", "view_role", "split_type"])
    cons_best = best_by(within_cons, ["dataset", "task", "view_role", "split_type"])
    ablation_best = best_by(ablation, ["dataset", "comparison_view"])
    cross_best = best_by(cross, ["train_dataset", "test_dataset"])
    iot_best = best_by(iot23, ["train_dataset", "test_dataset"])
    family_best = best_by(family, ["dataset", "feature_family"])

    hard_features = ", ".join(sorted(hard["feature"].dropna().astype(str).unique().tolist())) if not hard.empty else "none observed in processed modeling tables"
    legit = high[high["feature_class"] == "legitimate_metadata_retained"] if not high.empty else pd.DataFrame()
    legit_features = ", ".join(sorted(legit["feature"].dropna().astype(str).unique().tolist())) if not legit.empty else "none above audit threshold"
    conservative_removed = set()
    for value in cons_views.get("excluded_conservative_only_features", pd.Series(dtype=str)):
        conservative_removed.update(parse_features(value))
    cons_removed = ", ".join(sorted(conservative_removed)) if conservative_removed else "none"

    lines = [
        "# Stage 2B Results Summary",
        "",
        "This is a results-only experiment summary. It is not manuscript prose and contains no Abstract, Introduction, or Conclusion.",
        "",
        "## 1. Hard Leakage Features Removed",
        "",
        hard_features,
        "",
        "## 2. Legitimate Metadata Features Retained",
        "",
        legit_features,
        "",
        "## 3. Conservative Anti-Leakage Removals",
        "",
        cons_removed,
        "",
        "## 4. Main vs Conservative Results",
        "",
        cons_best[["dataset", "task", "view_role", "split_type", "model", "macro_f1", "weighted_f1", "n_features"]].to_markdown(index=False) if not cons_best.empty else "No conservative successful results.",
        "",
        "## 5. MQTTEEB-D Within-Dataset Main Results",
        "",
        main_best[main_best["dataset"] == "MQTTEEB-D"][["dataset", "task", "view_role", "split_type", "model", "macro_f1", "weighted_f1", "n_features"]].to_markdown(index=False) if not main_best.empty else "No MQTTEEB-D result.",
        "",
        "## 6. MQTT-IoT-IDS2020 Within-Dataset Main Results",
        "",
        main_best[main_best["dataset"] == "MQTT-IoT-IDS2020"][["dataset", "task", "view_role", "split_type", "model", "macro_f1", "weighted_f1", "n_features"]].to_markdown(index=False) if not main_best.empty else "No MQTT-IoT result.",
        "",
        "## 7. Crypto-Observability Ablation",
        "",
        ablation_best[["dataset", "comparison_view", "model", "macro_f1", "weighted_f1", "n_features"]].to_markdown(index=False) if not ablation_best.empty else "No ablation result.",
        "",
        "## 8. MQTT-to-MQTT Cross-Dataset Main Results",
        "",
        cross_best[["train_dataset", "test_dataset", "model", "macro_f1", "weighted_f1", "roc_auc", "pr_auc", "n_features"]].to_markdown(index=False) if not cross_best.empty else "No MQTT cross-dataset result.",
        "",
        "## 9. IoT-23 External Robustness",
        "",
        iot_best[["train_dataset", "test_dataset", "model", "macro_f1", "weighted_f1", "roc_auc", "pr_auc", "n_features", "validation_type"]].to_markdown(index=False) if not iot_best.empty else "No IoT-23 result.",
        "",
        "## 10. Feature-Family Ablation",
        "",
        family_best[["dataset", "feature_family", "model", "macro_f1", "weighted_f1", "n_features"]].to_markdown(index=False) if not family_best.empty else "No feature-family ablation result.",
        "",
        "## 11. Multi-Seed Stability",
        "",
        stability.to_markdown(index=False) if not stability.empty else "No multi-seed stability result.",
        "",
        "## 12. Results That Can Be Written",
        "",
        "- Hard leakage and sensitive identifiers were excluded from all model views.",
        "- Duration, packet/byte statistics, IAT statistics, rates, and burstiness are legitimate encrypted-flow metadata and were retained in audited main views.",
        "- Conservative anti-leakage is a sensitivity analysis, not the main result.",
        "- IoT-23 is external IoT robustness validation, not MQTT cross-dataset generalization.",
        "",
        "## 13. Results That Cannot Be Written",
        "",
        "- Do not call legitimate flow statistics label leakage merely because a single feature is highly discriminative.",
        "- Do not report invalid MQTT-IoT time-block splits as formal model performance.",
        "- Do not claim payload/topic/user/password/client-ID based detection.",
        "- Do not claim broad MQTT-to-MQTT generalization without noting the bidirectional cross-dataset results.",
    ]
    (root / "results" / "results_summary_stage2B.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 2B revised leakage-aware paper-level experiments.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = project_root()
    ensure_dirs(root)

    log_lines = [f"Stage 2B started with seed {args.seed}."]
    within_main = run_within_policy(root, "audited_metadata_main", args.seed)
    within_cons = run_within_policy(root, "conservative_anti_leakage", args.seed)
    write_csv(within_main, root / "results" / "metrics" / "within_dataset_main_results.csv")
    write_csv(within_cons, root / "results" / "metrics" / "within_dataset_conservative_results.csv")
    split_robust = pd.concat([within_main, within_cons], ignore_index=True, sort=False)
    write_csv(split_robust, root / "results" / "metrics" / "split_robustness_revised_results.csv")
    log_lines.append("Within-dataset main and conservative results completed. MLP rows are recorded as optional skipped.")

    cross = run_mqtt_cross_dataset(root, args.seed)
    write_csv(cross, root / "results" / "metrics" / "mqtt_to_mqtt_cross_dataset_main_results.csv")
    log_lines.append("MQTT-to-MQTT bidirectional main cross-dataset evaluation completed.")

    iot23 = run_iot23_external(root, args.seed)
    write_csv(iot23, root / "results" / "metrics" / "iot23_external_robustness_results.csv")
    log_lines.append("IoT-23 external IoT robustness validation completed.")

    ablation = run_crypto_observability_ablation(root, within_main, within_cons, args.seed)
    write_csv(ablation, root / "results" / "metrics" / "crypto_observability_ablation_revised.csv")

    family = run_feature_family_ablation(root, args.seed)
    write_csv(family, root / "results" / "metrics" / "feature_family_ablation_results.csv")

    importance = fit_feature_importance(root, args.seed)
    write_csv(importance, root / "results" / "tables" / "feature_importance_revised_top20.csv")

    stability = run_multiseed_stability(root, within_main, cross)
    write_csv(stability, root / "results" / "metrics" / "multiseed_stability_revised_results.csv")

    generate_figures(root, within_main, ablation, cross, family, importance)
    write_summary(root, within_main, within_cons, ablation, cross, iot23, family, stability)

    log_lines.extend(
        [
            "Crypto-observability ablation, feature-family ablation, feature importance, and multi-seed stability completed.",
            "No manuscript body, Abstract, Introduction, or Conclusion was generated.",
        ]
    )
    (root / "results" / "experiment_log_stage2B.md").write_text("# Stage 2B Experiment Log\n\n" + "\n".join(f"- {line}" for line in log_lines), encoding="utf-8")
    with (root / "reproducibility_report.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## Stage 2B Commands\n\n"
            "```powershell\n"
            "python code/build_feature_views.py --seed 42\n"
            "python code/audit_leakage.py --seed 42\n"
            "python code/build_feature_views.py --seed 42\n"
            "python code/run_stage2B_experiments.py --seed 42\n"
            "```\n"
        )
    print("Stage 2B experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
