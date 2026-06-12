from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from utils.data_io import ensure_dirs, processed_dir, project_root, raw_dir, results_tables_dir, safe_relpath


IOT23_BASE_URL = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
NUMERIC_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]
HISTORY_CHARS = ["D", "d", "S", "h", "H", "A", "F", "R", "r", "^", "I", "Q"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_public(url: str, timeout: int = 60) -> requests.Response:
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, verify=False)
        response.raise_for_status()
        return response


def parse_size_to_bytes(text: str) -> int | None:
    text = text.strip()
    match = re.match(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]?)", text, re.I)
    if not match:
        return None
    value = float(match.group("num"))
    unit = match.group("unit").upper()
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}.get(unit, 1)
    return int(value * multiplier)


def crawl_iot23_logs() -> list[dict]:
    root_html = get_public(IOT23_BASE_URL).text
    scenario_links = [
        href
        for href in re.findall(r'href="([^"]+)"', root_html)
        if href.endswith("/") and not href.startswith("?") and not href.startswith("/publicDatasets")
    ]
    candidates: list[dict] = []
    for scenario in scenario_links:
        scenario_url = urljoin(IOT23_BASE_URL, scenario)
        html = get_public(scenario_url).text
        for row in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S | re.I):
            hrefs = re.findall(r'href="([^"]+)"', row)
            for href in hrefs:
                lowered = href.lower()
                if "labeled" not in lowered:
                    continue
                if "zeek-conn" not in lowered and "conn.log" not in lowered:
                    continue
                row_text = " ".join(re.sub(r"<[^>]+>", " ", row).split())
                size_match = re.search(r"(\d+(?:\.\d+)?\s*[KMGTP])\s*(?:&nbsp;)?$", row_text)
                size_bytes = parse_size_to_bytes(size_match.group(1)) if size_match else None
                candidates.append(
                    {
                        "scenario": scenario.rstrip("/"),
                        "filename": href,
                        "url": urljoin(scenario_url, href),
                        "size_bytes": size_bytes,
                        "listing_text": row_text,
                    }
                )
    return candidates


def download_file(url: str, target: Path, expected_size: int | None = None) -> tuple[str, str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return "exists", "file already present", target.stat().st_size

    temp = target.with_suffix(target.suffix + ".part")
    resume_from = temp.stat().st_size if temp.exists() else 0
    headers = dict(REQUEST_HEADERS)
    mode = "wb"
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=120)
        if response.status_code == 416 and temp.exists():
            temp.replace(target)
            return "exists", "partial file already complete", target.stat().st_size
        response.raise_for_status()
    except requests.exceptions.SSLError:
        response = requests.get(url, headers=headers, stream=True, timeout=120, verify=False)
        response.raise_for_status()

    ctype = (response.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        return "skipped_no_direct_url", "URL returned HTML, not a data file", 0

    with open(temp, mode + ("" if "b" in mode else "b")) as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    temp.replace(target)
    actual = target.stat().st_size
    if expected_size and actual < expected_size * 0.98:
        return "partially_downloaded", f"expected about {expected_size} bytes, got {actual}", actual
    return "success", "downloaded", actual


def download_iot23_logs(root: Path, max_file_mb: float = 200.0) -> dict:
    out_dir = raw_dir("IoT-23", root)
    candidates = crawl_iot23_logs()
    downloaded: list[str] = []
    skipped_large = []
    failed = []
    max_bytes = int(max_file_mb * 1024 * 1024)
    for item in candidates:
        size_bytes = item.get("size_bytes")
        target = out_dir / item["scenario"] / item["filename"]
        if size_bytes and size_bytes > max_bytes:
            skipped_large.append(item)
            continue
        try:
            status, reason, bytes_written = download_file(item["url"], target, size_bytes)
            if status in {"success", "exists"}:
                downloaded.append(safe_relpath(target, root))
            else:
                failed.append({**item, "status": status, "reason": reason, "bytes": bytes_written})
        except Exception as exc:
            failed.append({**item, "status": "failed_network", "reason": repr(exc)})

    if downloaded and skipped_large:
        status = "partially_downloaded"
        reason = f"downloaded {len(downloaded)} Zeek logs; skipped {len(skipped_large)} logs larger than {max_file_mb:g} MB"
    elif downloaded:
        status = "success"
        reason = f"downloaded {len(downloaded)} Zeek labelled logs"
    elif failed:
        status = "failed_network"
        reason = f"no logs downloaded; {len(failed)} attempts failed"
    else:
        status = "skipped_no_direct_url"
        reason = "no zeek-conn labelled logs found in public directory"

    return {
        "dataset": "IoT-23",
        "source_name": "Stratosphere/CTU public directory",
        "attempted_url": IOT23_BASE_URL,
        "status": status,
        "reason": reason,
        "downloaded_files": downloaded,
        "total_size_mb": sum((root / p).stat().st_size for p in downloaded) / (1024 * 1024) if downloaded else 0.0,
        "timestamp": utc_now(),
        "candidate_count": len(candidates),
        "skipped_large_count": len(skipped_large),
        "failed_count": len(failed),
    }


def clean_value(value: str):
    if value in {"-", "(empty)", ""}:
        return np.nan
    return value


def to_float(value) -> float:
    value = clean_value(str(value))
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def history_features(history: str) -> dict[str, int]:
    history = "" if pd.isna(history) else str(history)
    features = {"history_len": len(history)}
    for char in HISTORY_CHARS:
        safe = char.replace("^", "caret")
        features[f"history_count_{safe}"] = history.count(char)
    return features


def parse_zeek_file(path: Path, root: Path, buckets: dict[str, list[dict]], max_per_class: int, remaining_budget: int | None) -> tuple[int, Counter]:
    fields: list[str] | None = None
    seen = 0
    labels: Counter = Counter()
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#fields"):
                fields = re.split(r"\s+", line.replace("#fields", "", 1).strip())
                continue
            if line.startswith("#"):
                continue
            if not fields:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) != len(fields):
                continue
            row = dict(zip(fields, parts))
            label = str(row.get("label", "")).strip().lower()
            if not label:
                continue
            binary = 0 if "benign" in label else 1
            detailed = str(row.get("detailed-label", "")).strip()
            multiclass = "benign" if binary == 0 else (detailed if detailed and detailed != "-" else label)
            labels[multiclass] += 1
            if len(buckets[multiclass]) >= max_per_class:
                continue
            if remaining_budget is not None and sum(len(v) for v in buckets.values()) >= remaining_budget:
                continue

            feature_row = {
                "binary_label": binary,
                "multiclass_label": multiclass,
                "source_file": safe_relpath(path, root),
                "duration": to_float(row.get("duration")),
                "orig_bytes": to_float(row.get("orig_bytes")),
                "resp_bytes": to_float(row.get("resp_bytes")),
                "orig_pkts": to_float(row.get("orig_pkts")),
                "resp_pkts": to_float(row.get("resp_pkts")),
                "orig_ip_bytes": to_float(row.get("orig_ip_bytes")),
                "resp_ip_bytes": to_float(row.get("resp_ip_bytes")),
                "missed_bytes": to_float(row.get("missed_bytes")),
                "proto": clean_value(row.get("proto", "")),
                "service": clean_value(row.get("service", "")),
                "conn_state": clean_value(row.get("conn_state", "")),
            }
            feature_row.update(history_features(row.get("history", "")))
            byte_total = np.nansum([feature_row["orig_bytes"], feature_row["resp_bytes"]])
            pkt_total = np.nansum([feature_row["orig_pkts"], feature_row["resp_pkts"]])
            duration = feature_row["duration"]
            feature_row["total_bytes"] = float(byte_total)
            feature_row["total_pkts"] = float(pkt_total)
            feature_row["bytes_per_second"] = float(byte_total / duration) if duration and duration > 0 else 0.0
            feature_row["packets_per_second"] = float(pkt_total / duration) if duration and duration > 0 else 0.0
            buckets[multiclass].append(feature_row)
            seen += 1
    return seen, labels


def process_iot23(root: Path, sample_size: int = 100_000, max_per_class: int = 10_000, seed: int = 42) -> dict:
    del seed
    files = sorted(raw_dir("IoT-23", root).rglob("*zeek*conn*labeled"))
    files = [path for path in files if path.is_file()]
    if not files:
        return {"dataset": "IoT-23", "status": "skipped_no_direct_url", "reason": "no downloaded Zeek labelled logs found"}

    buckets: dict[str, list[dict]] = defaultdict(list)
    raw_label_counts: Counter = Counter()
    parsed_rows = 0
    for path in files:
        seen, labels = parse_zeek_file(path, root, buckets, max_per_class=max_per_class, remaining_budget=sample_size)
        parsed_rows += seen
        raw_label_counts.update(labels)

    rows = [row for values in buckets.values() for row in values]
    if not rows:
        return {"dataset": "IoT-23", "status": "failed_network", "reason": "downloaded logs contained no parseable labelled rows"}

    df = pd.DataFrame(rows)
    out_path = processed_dir(root) / "IoT-23_flows.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    profile = pd.DataFrame(
        [
            {
                "dataset": "IoT-23",
                "file_path": safe_relpath(out_path, root),
                "rows": len(df),
                "columns": len(df.columns),
                "label_column": "binary_label;multiclass_label",
                "label_distribution": dict(df["multiclass_label"].value_counts()),
                "missing_value_ratio": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])),
                "read_status": "ok",
                "raw_parseable_rows_in_downloaded_logs": parsed_rows,
            }
        ]
    )
    profile_path = results_tables_dir(root) / "IoT-23_dataset_profile.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    dist = df["multiclass_label"].value_counts().reset_index()
    dist.columns = ["label_value", "count"]
    dist.insert(0, "dataset", "IoT-23")
    dist["percent"] = dist["count"] / dist["count"].sum()
    dist_path = results_tables_dir(root) / "IoT-23_label_distribution.csv"
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")

    return {
        "dataset": "IoT-23",
        "status": "success",
        "reason": f"processed {len(df)} sampled labelled flows from {len(files)} downloaded Zeek logs",
        "downloaded_files": [safe_relpath(path, root) for path in files],
        "processed_file": safe_relpath(out_path, root),
        "total_size_mb": out_path.stat().st_size / (1024 * 1024),
        "timestamp": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and process public IoT-23 Zeek labelled flow logs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--max-per-class", type=int, default=10_000)
    parser.add_argument("--max-file-mb", type=float, default=200.0)
    args = parser.parse_args()
    root = project_root()
    ensure_dirs(root)
    download_status = download_iot23_logs(root, max_file_mb=args.max_file_mb)
    process_status = process_iot23(root, sample_size=args.sample_size, max_per_class=args.max_per_class, seed=args.seed)
    print(download_status)
    print(process_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
