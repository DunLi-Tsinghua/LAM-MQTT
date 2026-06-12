from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from download_iot23 import download_iot23_logs, process_iot23
from utils.data_io import ensure_dirs, project_root, raw_dir, results_tables_dir, safe_relpath, write_json


MQTTEEB_DATASET_API = "https://data.mendeley.com/public-api/datasets/jfttfjn6tr"
GOTHAM_ZENODO_API = "https://zenodo.org/api/records/14502760"
CIC_OFFICIAL_PAGE = "https://www.unb.ca/cic/datasets/iotdataset-2023.html"
CIC_DOWNLOAD_FORM = "https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/"
CIC_HF_MIRROR = "https://datasets-server.huggingface.co/is-valid?dataset=baalajimaestro/DDoS-CICIoT2023"
MQTT_IDS_PAGE = "https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset"

ALLOWED_STATUSES = {
    "success",
    "skipped_login_required",
    "skipped_no_direct_url",
    "failed_network",
    "failed_checksum",
    "too_large_skipped",
    "partially_downloaded",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status_row(
    dataset: str,
    source_name: str,
    attempted_url: str,
    status: str,
    reason: str,
    downloaded_files: list[str] | None = None,
    total_size_mb: float = 0.0,
) -> dict:
    if status not in ALLOWED_STATUSES:
        status = "failed_network"
    return {
        "dataset": dataset,
        "source_name": source_name,
        "attempted_url": attempted_url,
        "status": status,
        "reason": reason,
        "downloaded_files": json.dumps(downloaded_files or [], ensure_ascii=False),
        "total_size_mb": round(float(total_size_mb), 4),
        "timestamp": utc_now(),
    }


def requests_get(url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("headers", {"User-Agent": "Mozilla/5.0"})
    kwargs.setdefault("timeout", 60)
    try:
        return requests.get(url, **kwargs)
    except requests.exceptions.SSLError:
        kwargs["verify"] = False
        return requests.get(url, **kwargs)


def is_html_response(response: requests.Response) -> bool:
    ctype = (response.headers.get("content-type") or "").lower()
    return "text/html" in ctype


def download_file(url: str, target: Path, expected_sha256: str | None = None) -> tuple[str, str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return "success", "file already present", target.stat().st_size

    temp = target.with_suffix(target.suffix + ".part")
    resume_from = temp.stat().st_size if temp.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    mode = "wb"
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    with requests_get(url, stream=True, timeout=120, headers=headers) as response:
        response.raise_for_status()
        if is_html_response(response):
            return "skipped_no_direct_url", "URL returned HTML instead of a data file", 0
        total = int(response.headers.get("content-length") or 0)
        digest = hashlib.sha256()
        if resume_from:
            try:
                with open(temp, "rb") as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                        digest.update(chunk)
            except FileNotFoundError:
                pass
        with open(temp, mode) as handle, tqdm(
            total=total + resume_from if total else 0,
            initial=resume_from,
            unit="B",
            unit_scale=True,
            desc=target.name,
            disable=total == 0,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
    temp.replace(target)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 and expected_sha256 != actual_sha256:
        return "failed_checksum", f"sha256 mismatch: expected {expected_sha256}, got {actual_sha256}", target.stat().st_size
    return "success", "downloaded", target.stat().st_size


def download_mqtteeb_d(root: Path, include_processed: bool = False) -> dict:
    out_dir = raw_dir("MQTTEEB-D", root)
    downloaded: list[str] = []
    failed: list[str] = []
    try:
        meta = requests_get(MQTTEEB_DATASET_API, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, timeout=60)
        meta.raise_for_status()
        payload = meta.json()
        write_json(out_dir / "_mendeley_dataset_metadata.json", payload)
        files = payload.get("files", [])
    except Exception as exc:
        return status_row("MQTTEEB-D", "Mendeley Data public API", MQTTEEB_DATASET_API, "failed_network", repr(exc))

    selected = []
    for file_info in files:
        filename = file_info.get("filename") or ""
        lowered = filename.lower()
        if "smote" in lowered:
            continue
        if lowered.startswith("mqtteeb-d_dataset_loop_"):
            selected.append(file_info)
        elif lowered in {"categorical_processing_metadata.json", "label_encoding_metadata.json"}:
            selected.append(file_info)
        elif lowered == "readme.txt" and not any((item.get("filename") or "").lower() == "readme.txt" for item in selected):
            selected.append(file_info)
        elif include_processed and lowered.endswith(".csv") and "normalized" not in lowered and "standardized" not in lowered:
            selected.append(file_info)

    for file_info in selected:
        details = file_info.get("content_details", {})
        url = details.get("download_url")
        filename = file_info.get("filename")
        sha256 = details.get("sha256_hash")
        if not url or not filename:
            continue
        folder = "Raw_RealTime_Data" if "dataset_loop_" in filename else "metadata"
        target = out_dir / folder / filename
        try:
            status, reason, _ = download_file(url, target, sha256)
            if status == "success":
                downloaded.append(safe_relpath(target, root))
            else:
                failed.append(f"{filename}: {status} ({reason})")
        except Exception as exc:
            failed.append(f"{filename}: {exc!r}")

    if downloaded and not failed:
        status = "success"
        reason = f"downloaded or found {len(downloaded)} MQTTEEB-D files"
    elif downloaded:
        status = "partially_downloaded"
        reason = f"downloaded or found {len(downloaded)} files; failures: {'; '.join(failed[:3])}"
    else:
        status = "failed_network"
        reason = f"no files downloaded; failures: {'; '.join(failed[:3])}"
    total_mb = sum((root / p).stat().st_size for p in downloaded if (root / p).exists()) / (1024 * 1024) if downloaded else 0.0
    return status_row("MQTTEEB-D", "Mendeley Data public API", MQTTEEB_DATASET_API, status, reason, downloaded, total_mb)


def try_gotham(root: Path, max_file_mb: float) -> dict:
    del root
    try:
        response = requests_get(GOTHAM_ZENODO_API, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        data = response.json()
        files = data.get("files", [])
        useful = [f for f in files if any(str(f.get("key", "")).lower().endswith(ext) for ext in (".csv", ".parquet"))]
        if useful:
            return status_row("Gotham2025", "Zenodo API", GOTHAM_ZENODO_API, "skipped_no_direct_url", "CSV/Parquet file found but Gotham parser not implemented for this record yet")
        if files:
            largest = files[0]
            size_mb = float(largest.get("size", 0)) / (1024 * 1024)
            if size_mb > max_file_mb:
                return status_row(
                    "Gotham2025",
                    "Zenodo API",
                    GOTHAM_ZENODO_API,
                    "too_large_skipped",
                    f"Zenodo record exposes only {largest.get('key')} ({size_mb:.1f} MB), no separate processed CSV/Parquet direct file",
                )
        return status_row("Gotham2025", "Zenodo API", GOTHAM_ZENODO_API, "skipped_no_direct_url", "no CSV/Parquet/processed labelled file found")
    except Exception as exc:
        return status_row("Gotham2025", "Zenodo API", GOTHAM_ZENODO_API, "failed_network", repr(exc))


def try_ciciot2023(root: Path) -> dict:
    del root
    reasons = []
    try:
        official = requests_get(CIC_OFFICIAL_PAGE, timeout=60)
        official.raise_for_status()
        if CIC_DOWNLOAD_FORM in official.text or "Download the dataset" in official.text:
            reasons.append("official UNB page redirects to CIC download form")
    except Exception as exc:
        reasons.append(f"official page failed: {exc!r}")

    try:
        form = requests_get(CIC_DOWNLOAD_FORM, timeout=60)
        form.raise_for_status()
        text = form.text.lower()
        if "<form" in text and "first_name" in text and "required" in text:
            reasons.append("CIC official download requires form submission/person details; skipped by auto-only policy")
    except Exception as exc:
        reasons.append(f"CIC form check failed: {exc!r}")

    try:
        hf = requests_get(CIC_HF_MIRROR, timeout=60)
        if hf.status_code == 200 and "true" in hf.text.lower():
            reasons.append("public Hugging Face mirror is reachable but checked mirror is DDoS-only and exposes no label column in Dataset Viewer preview, so it is not used as labelled binary IDS data")
    except Exception as exc:
        reasons.append(f"Hugging Face mirror check failed: {exc!r}")

    return status_row(
        "CICIoT2023",
        "UNB/CIC official page and public mirror checks",
        CIC_OFFICIAL_PAGE,
        "skipped_login_required",
        "; ".join(reasons) if reasons else "no auto-downloadable labelled CSV/Parquet direct source found",
    )


def try_mqtt_iot_ids2020(root: Path) -> dict:
    del root
    try:
        response = requests_get(MQTT_IDS_PAGE, timeout=60)
        response.raise_for_status()
        text = response.text.lower()
        if "user/login" in text or "saml_login" in text or "login" in text:
            return status_row(
                "MQTT-IoT-IDS2020",
                "IEEE DataPort open access page",
                MQTT_IDS_PAGE,
                "skipped_login_required",
                "IEEE DataPort page exposes dataset metadata but file download requires DataPort login/authentication; HTML is not counted as data",
            )
        return status_row("MQTT-IoT-IDS2020", "IEEE DataPort open access page", MQTT_IDS_PAGE, "skipped_no_direct_url", "no direct CSV/ZIP/PCAP file URL found")
    except Exception as exc:
        return status_row("MQTT-IoT-IDS2020", "IEEE DataPort open access page", MQTT_IDS_PAGE, "failed_network", repr(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-only robust downloader for Stage 1.5 datasets.")
    parser.add_argument("--auto-only", action="store_true", help="Use only no-login/no-token/no-manual-interaction sources.")
    parser.add_argument("--skip-login-required", action="store_true", help="Skip sources requiring login, forms, CAPTCHA, Kaggle tokens, or manual authorization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-file-mb", type=float, default=200.0)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--max-per-class", type=int, default=10_000)
    parser.add_argument("--include-processed-mqtteeb", action="store_true")
    args = parser.parse_args()

    root = project_root()
    ensure_dirs(root)
    rows: list[dict] = []

    try:
        rows.append(download_mqtteeb_d(root, include_processed=args.include_processed_mqtteeb))
    except Exception as exc:
        rows.append(status_row("MQTTEEB-D", "Mendeley Data public API", MQTTEEB_DATASET_API, "failed_network", repr(exc)))

    try:
        iot_download = download_iot23_logs(root, max_file_mb=args.max_file_mb)
        rows.append(status_row(iot_download["dataset"], iot_download["source_name"], iot_download["attempted_url"], iot_download["status"], iot_download["reason"], iot_download.get("downloaded_files", []), iot_download.get("total_size_mb", 0.0)))
        iot_process = process_iot23(root, sample_size=args.sample_size, max_per_class=args.max_per_class, seed=args.seed)
        if iot_process.get("status") != "success":
            rows.append(status_row("IoT-23", "Zeek parser", IOT23_BASE_URL, iot_process.get("status", "failed_network"), iot_process.get("reason", "processing failed")))
    except Exception as exc:
        rows.append(status_row("IoT-23", "Stratosphere/CTU public directory", IOT23_BASE_URL, "failed_network", repr(exc)))

    rows.append(try_gotham(root, max_file_mb=args.max_file_mb))
    rows.append(try_ciciot2023(root))
    rows.append(try_mqtt_iot_ids2020(root))

    out_path = results_tables_dir(root) / "download_status.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    write_json(root / "data" / "raw" / "download_status.json", rows)
    print(f"Wrote {out_path}")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
