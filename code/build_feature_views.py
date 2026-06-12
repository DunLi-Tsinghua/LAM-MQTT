from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.data_io import ensure_dirs, processed_dir, project_root, results_tables_dir, safe_relpath


LABEL_COLUMNS = {"binary_label", "binary_label_name", "multiclass_label", "label", "labels", "target"}
HARD_LEAKAGE_EXACT = {
    "label",
    "labels",
    "target",
    "attack",
    "attack_type",
    "attack_cat",
    "class",
    "category",
    "scenario",
    "scenario_name",
    "source_file",
    "file_name",
    "filename",
    "capture_name",
    "pcap_name",
    "client_id",
    "clientid",
    "device_id",
    "deviceid",
    "username",
    "user_name",
    "password",
    "passwd",
    "topic",
    "payload",
    "mqtt_msg",
    "mqtt_message",
    "raw_message",
    "message",
    "content",
    "timestamp",
    "time_epoch",
    "frame_time",
    "datetime",
    "date_time",
    "row_index",
    "record_index",
    "record_order",
    "index",
}
HARD_LEAKAGE_SUBSTRINGS = (
    "source_file",
    "file_name",
    "filename",
    "capture_name",
    "pcap_name",
    "scenario_name",
    "clientid",
    "client_id",
    "deviceid",
    "device_id",
    "username",
    "password",
    "passwd",
    "payload",
    "mqtt_msg",
    "timestamp",
    "time_epoch",
    "frame_time",
    "datetime",
    "date_time",
    "row_index",
    "record_index",
    "record_order",
    "window_id",
)
RAW_IDENTITY_SUBSTRINGS = (
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "ip_address",
    "attacker_ip",
    "victim_ip",
    "id_orig_h",
    "id_resp_h",
    "orig_h",
    "resp_h",
    "mac",
    "hostname",
    "uid",
)
ANALYSIS_PARAMETER_EXACT = {"window_seconds"}
MQTT_METADATA_TOKENS = (
    "mqtt",
    "tcp",
    "flag",
    "qos",
    "len",
    "length",
    "session",
    "connect",
    "conack",
    "kalive",
    "keep",
    "packet",
    "pkt",
    "byte",
    "duration",
    "iat",
    "rate",
    "burst",
    "flow",
    "proto",
    "prt",
    "port",
)
STRICT_TOKENS = (
    "duration",
    "packet",
    "pkt",
    "pkts",
    "byte",
    "bytes",
    "len",
    "length",
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "iat",
    "rate",
    "burst",
    "flag",
    "count",
    "proto",
    "protocol",
    "service",
    "conn_state",
    "missed",
    "history",
    "ttl",
    "variance",
    "prt",
    "port",
)
STRICT_EXCLUDE_TOKENS = ("mqtt_", "topic", "payload", "user", "pass", "client", "timestamp", "source_file")
COMMON_CROSS_FEATURES = [
    "packet_count",
    "byte_sum",
    "byte_mean",
    "byte_std",
    "byte_min",
    "byte_max",
    "duration",
    "iat_mean",
    "iat_std",
    "iat_min",
    "iat_max",
    "bytes_per_second",
    "packets_per_second",
    "burstiness",
]


def norm(col: str) -> str:
    return str(col).lower().replace(".", "_").replace("-", "_").replace(" ", "_")


def parts(col: str) -> set[str]:
    return {p for p in re.split(r"[^0-9a-zA-Z]+", norm(col)) if p}


def has_any(col: str, tokens: tuple[str, ...]) -> bool:
    low = norm(col)
    return any(token.replace(".", "_").replace("-", "_").replace(" ", "_") in low for token in tokens)


def parse_json_list(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        return [str(x) for x in json.loads(value)]
    except Exception:
        return []


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def num(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for col in candidates:
        series = num(df, col)
        if series is not None:
            return series
    return None


def add_normalized_flow_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "packet_count" not in out.columns:
        fwd_pkts, bwd_pkts = first_existing(out, ["fwd_num_pkts", "orig_pkts"]), first_existing(out, ["bwd_num_pkts", "resp_pkts"])
        if fwd_pkts is not None and bwd_pkts is not None:
            out["packet_count"] = fwd_pkts.fillna(0) + bwd_pkts.fillna(0)
        elif "num_pkts" in out.columns:
            out["packet_count"] = pd.to_numeric(out["num_pkts"], errors="coerce")

    if "byte_sum" not in out.columns:
        fwd_bytes = first_existing(out, ["fwd_num_bytes", "orig_bytes", "orig_ip_bytes"])
        bwd_bytes = first_existing(out, ["bwd_num_bytes", "resp_bytes", "resp_ip_bytes"])
        if fwd_bytes is not None and bwd_bytes is not None:
            out["byte_sum"] = fwd_bytes.fillna(0) + bwd_bytes.fillna(0)
        elif "num_bytes" in out.columns:
            out["byte_sum"] = pd.to_numeric(out["num_bytes"], errors="coerce")

    pair_means = {
        "byte_mean": ("fwd_mean_pkt_len", "bwd_mean_pkt_len", "mean_pkt_len"),
        "byte_std": ("fwd_std_pkt_len", "bwd_std_pkt_len", "std_pkt_len"),
        "iat_mean": ("fwd_mean_iat", "bwd_mean_iat", "mean_iat"),
        "iat_std": ("fwd_std_iat", "bwd_std_iat", "std_iat"),
    }
    for target, (fwd, bwd, uni) in pair_means.items():
        if target in out.columns:
            continue
        fwd_s, bwd_s = num(out, fwd), num(out, bwd)
        if fwd_s is not None and bwd_s is not None:
            out[target] = pd.concat([fwd_s, bwd_s], axis=1).mean(axis=1)
        elif uni in out.columns:
            out[target] = pd.to_numeric(out[uni], errors="coerce")

    pair_min = {
        "byte_min": ("fwd_min_pkt_len", "bwd_min_pkt_len", "min_pkt_len"),
        "iat_min": ("fwd_min_iat", "bwd_min_iat", "min_iat"),
    }
    for target, (fwd, bwd, uni) in pair_min.items():
        if target in out.columns:
            continue
        fwd_s, bwd_s = num(out, fwd), num(out, bwd)
        if fwd_s is not None and bwd_s is not None:
            out[target] = pd.concat([fwd_s, bwd_s], axis=1).min(axis=1)
        elif uni in out.columns:
            out[target] = pd.to_numeric(out[uni], errors="coerce")

    pair_max = {
        "byte_max": ("fwd_max_pkt_len", "bwd_max_pkt_len", "max_pkt_len"),
        "iat_max": ("fwd_max_iat", "bwd_max_iat", "max_iat"),
    }
    for target, (fwd, bwd, uni) in pair_max.items():
        if target in out.columns:
            continue
        fwd_s, bwd_s = num(out, fwd), num(out, bwd)
        if fwd_s is not None and bwd_s is not None:
            out[target] = pd.concat([fwd_s, bwd_s], axis=1).max(axis=1)
        elif uni in out.columns:
            out[target] = pd.to_numeric(out[uni], errors="coerce")

    if "byte_mean" not in out.columns and {"byte_sum", "packet_count"}.issubset(out.columns):
        denom = pd.to_numeric(out["packet_count"], errors="coerce").replace(0, np.nan)
        out["byte_mean"] = pd.to_numeric(out["byte_sum"], errors="coerce") / denom
    if "duration" not in out.columns and "iat_max" in out.columns:
        out["duration"] = pd.to_numeric(out["iat_max"], errors="coerce")
    if "packets_per_second" not in out.columns and {"packet_count", "duration"}.issubset(out.columns):
        denom = pd.to_numeric(out["duration"], errors="coerce").replace(0, np.nan)
        out["packets_per_second"] = pd.to_numeric(out["packet_count"], errors="coerce") / denom
    if "bytes_per_second" not in out.columns and {"byte_sum", "duration"}.issubset(out.columns):
        denom = pd.to_numeric(out["duration"], errors="coerce").replace(0, np.nan)
        out["bytes_per_second"] = pd.to_numeric(out["byte_sum"], errors="coerce") / denom
    if "burstiness" not in out.columns and {"byte_std", "byte_mean"}.issubset(out.columns):
        denom = pd.to_numeric(out["byte_mean"], errors="coerce").replace(0, np.nan)
        out["burstiness"] = pd.to_numeric(out["byte_std"], errors="coerce") / denom
    for col in COMMON_CROSS_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out


def discover_processed_sources(root: Path) -> list[dict]:
    processed = processed_dir(root)
    sources = []
    candidates = [
        ("MQTTEEB-D", "packet", processed / "MQTTEEB-D__cleaned.csv", True),
        ("MQTT-IoT-IDS2020", "flow", processed / "MQTT-IoT-IDS2020__cleaned.csv", True),
        ("IoT-23", "flow", processed / "IoT-23_flows.parquet", False),
    ]
    for dataset, source_kind, path, is_mqtt in candidates:
        if path.exists():
            sources.append({"dataset": dataset, "source_kind": source_kind, "path": path, "is_mqtt": is_mqtt})
    for path in sorted(processed.glob("MQTTEEB-D_flow_windows_*s.parquet")):
        window = path.stem.replace("MQTTEEB-D_flow_windows_", "")
        sources.append({"dataset": f"MQTTEEB-D_flow_{window}", "source_kind": "flow_window", "path": path, "is_mqtt": True})
    return sources


def hard_leakage_reasons(col: str) -> list[str]:
    low = norm(col)
    ps = parts(col)
    reasons: list[str] = []
    is_ip_byte_counter = low in {"orig_ip_bytes", "resp_ip_bytes"} or low.endswith("_ip_bytes")
    if low in {norm(x) for x in LABEL_COLUMNS} or low in HARD_LEAKAGE_EXACT:
        reasons.append("hard_leakage:label_or_explicit_forbidden_field")
    if any(token in low for token in HARD_LEAKAGE_SUBSTRINGS):
        reasons.append("hard_leakage:sensitive_content_identity_timestamp_or_order")
    if "attack" in ps or "scenario" in ps or "category" in ps or "class" in ps:
        reasons.append("hard_leakage:attack_scenario_or_class_semantics")
    if "topic" in ps:
        reasons.append("hard_leakage:raw_topic_string")
    if (("ip" in ps and not is_ip_byte_counter) or any(token in low for token in RAW_IDENTITY_SUBSTRINGS)):
        reasons.append("hard_leakage:raw_ip_or_endpoint_identity")
    if ("client" in ps and "id" in ps) or ("device" in ps and "id" in ps):
        reasons.append("hard_leakage:client_or_device_identity")
    return sorted(set(reasons))


def is_hard_leakage(col: str) -> bool:
    return bool(hard_leakage_reasons(col))


def is_analysis_parameter(col: str) -> bool:
    return norm(col) in ANALYSIS_PARAMETER_EXACT


def is_legitimate_metadata(col: str) -> bool:
    low = norm(col)
    tokens = (
        "duration",
        "packet_count",
        "num_pkts",
        "pkt",
        "pkts",
        "byte",
        "bytes",
        "pkt_len",
        "packet_len",
        "iat",
        "bytes_per_second",
        "packets_per_second",
        "burstiness",
        "flow_rate",
        "flag",
        "tcp_flag",
        "psh_flags",
        "rst_flags",
        "urg_flags",
    )
    return any(token in low for token in tokens)


def is_protocol_metadata(col: str) -> bool:
    low = norm(col)
    return any(token in low for token in ("prt_", "port", "proto", "protocol", "service", "conn_state", "history"))


def main_feature_columns(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    hard = [col for col in columns if is_hard_leakage(col)]
    analysis = [col for col in columns if is_analysis_parameter(col)]
    excluded = set(hard) | set(analysis) | LABEL_COLUMNS
    features = [col for col in columns if col not in excluded]
    return features, sorted(set(hard)), sorted(set(analysis))


def load_conservative_exclusions(root: Path) -> dict[str, set[str]]:
    path = results_tables_dir(root) / "highly_discriminative_metadata_report.csv"
    exclusions: dict[str, set[str]] = {}
    if not path.exists():
        return exclusions
    try:
        df = pd.read_csv(path)
    except Exception:
        return exclusions
    if "feature" not in df.columns or "single_feature_macro_f1" not in df.columns:
        return exclusions
    for _, row in df.iterrows():
        try:
            score = float(row.get("single_feature_macro_f1"))
        except Exception:
            continue
        if score <= 0.90:
            continue
        feature = str(row.get("feature", "")).strip()
        if not feature:
            continue
        dataset = str(row.get("dataset", "*")).strip() or "*"
        exclusions.setdefault(dataset, set()).add(feature)
    return exclusions


def select_view_features(source: dict, features: list[str]) -> dict[str, list[str]]:
    if source["is_mqtt"]:
        broker = features[:] if source["source_kind"] == "flow_window" else [col for col in features if has_any(col, MQTT_METADATA_TOKENS)]
    else:
        broker = []
    strict = [col for col in features if has_any(col, STRICT_TOKENS) and not has_any(col, STRICT_EXCLUDE_TOKENS)]
    return {
        "full": features,
        "broker": broker,
        "strict": strict,
    }


def make_view(
    df: pd.DataFrame,
    root: Path,
    dataset: str,
    policy: str,
    view_role: str,
    features: list[str],
    hard_excluded: list[str],
    analysis_excluded: list[str],
    extra_excluded: list[str],
    status_base: str,
    note: str,
) -> dict:
    label_cols = ["binary_label"] + (["multiclass_label"] if "multiclass_label" in df.columns else [])
    view_name = f"{view_role}_{policy}"
    out_path = processed_dir(root) / f"{dataset}__view__{view_name}.parquet"
    if status_base != "not_applicable":
        write_any(df[label_cols + features].copy(), out_path)
        source_file = safe_relpath(out_path, root)
    else:
        source_file = ""
    status = status_base if status_base != "ok" else ("weak_view" if len(features) < 5 else "ok")
    return {
        "dataset": dataset,
        "policy": policy,
        "view_role": view_role,
        "view": view_name,
        "status": status,
        "source_file": source_file,
        "n_features": len(features),
        "included_features": json.dumps(features, ensure_ascii=False),
        "excluded_hard_leakage_features": json.dumps(hard_excluded, ensure_ascii=False),
        "excluded_analysis_parameter_features": json.dumps(analysis_excluded, ensure_ascii=False),
        "excluded_conservative_only_features": json.dumps(extra_excluded, ensure_ascii=False),
        "notes": note,
    }


def build_for_source(source: dict, root: Path, conservative_exclusions: dict[str, set[str]]) -> tuple[list[dict], list[dict]]:
    df = add_normalized_flow_metadata(read_any(source["path"]))
    if "binary_label" not in df.columns:
        return [], []
    base_features, hard_excluded, analysis_excluded = main_feature_columns(list(df.columns))
    main_views = select_view_features(source, base_features)
    conservative_extra = conservative_exclusions.get(source["dataset"], set()) | conservative_exclusions.get("*", set())
    conservative_features = [col for col in base_features if col not in conservative_extra]
    conservative_views = select_view_features(source, conservative_features)

    rows: list[dict] = []
    policy_rows: list[dict] = []
    for view_role, features in main_views.items():
        status_base = "not_applicable" if view_role == "broker" and not source["is_mqtt"] else "ok"
        row = make_view(
            df,
            root,
            source["dataset"],
            "audited_metadata_main",
            view_role,
            features,
            hard_excluded,
            analysis_excluded,
            [],
            status_base,
            "Main policy: hard leakage and sensitive content removed; legitimate encrypted-flow metadata retained.",
        )
        rows.append(row)
        policy_rows.append(row)
    for view_role, features in conservative_views.items():
        status_base = "not_applicable" if view_role == "broker" and not source["is_mqtt"] else "ok"
        view_specific_extra = sorted(set(main_views.get(view_role, [])) - set(features))
        row = make_view(
            df,
            root,
            source["dataset"],
            "conservative_anti_leakage",
            view_role,
            features,
            hard_excluded,
            analysis_excluded,
            view_specific_extra,
            status_base,
            "Sensitivity policy: hard leakage removed plus single-feature Macro F1 > 0.90 metadata removed.",
        )
        rows.append(row)
        policy_rows.append(row)
    return rows, policy_rows


def write_policy_tables(root: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for policy, filename in [
        ("audited_metadata_main", "feature_views_audited_metadata_main.csv"),
        ("conservative_anti_leakage", "feature_views_conservative_anti_leakage.csv"),
    ]:
        out = df[df["policy"] == policy].copy() if not df.empty else pd.DataFrame()
        out.to_csv(results_tables_dir(root) / filename, index=False, encoding="utf-8-sig")

    common_rows = []
    for dataset in ["MQTT-IoT-IDS2020", "MQTTEEB-D_flow_1s", "MQTTEEB-D_flow_5s", "MQTTEEB-D_flow_10s"]:
        sub = df[(df["dataset"] == dataset) & (df["policy"] == "audited_metadata_main") & (df["view_role"] == "strict")]
        if sub.empty:
            continue
        features = parse_json_list(sub.iloc[0]["included_features"])
        common = [feature for feature in COMMON_CROSS_FEATURES if feature in features]
        common_rows.append(
            {
                "dataset": dataset,
                "view": sub.iloc[0]["view"],
                "n_common_metadata_features_available": len(common),
                "common_metadata_features": json.dumps(common, ensure_ascii=False),
                "notes": "Normalized metadata retained for MQTT-to-MQTT main cross-dataset evaluation.",
            }
        )
    pd.DataFrame(common_rows).to_csv(results_tables_dir(root) / "mqtt_cross_dataset_common_features_main.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 2B leakage-audited feature policies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    conservative_exclusions = load_conservative_exclusions(root)
    rows: list[dict] = []
    for source in discover_processed_sources(root):
        source_rows, _ = build_for_source(source, root, conservative_exclusions)
        rows.extend(source_rows)
    if not rows:
        rows.append(
            {
                "dataset": "",
                "policy": "",
                "view_role": "",
                "view": "",
                "status": "no_processed_sources",
                "source_file": "",
                "n_features": 0,
                "included_features": "[]",
                "excluded_hard_leakage_features": "[]",
                "excluded_analysis_parameter_features": "[]",
                "excluded_conservative_only_features": "[]",
                "notes": "Run preprocessing first.",
            }
        )
    out_path = results_tables_dir(root) / "feature_views_summary.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    write_policy_tables(root, rows)
    print(f"Wrote {out_path}")
    print(out_df[["dataset", "policy", "view_role", "status", "n_features"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
