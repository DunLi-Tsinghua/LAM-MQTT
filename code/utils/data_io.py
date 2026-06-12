from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASETS = ("MQTTEEB-D", "MQTT-IoT-IDS2020")
TABULAR_SUFFIXES = (".csv", ".csv.gz", ".tsv", ".txt")
EXCLUDED_NAME_TOKENS = ("smote",)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_dir(dataset: str, root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "raw" / dataset


def interim_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "interim"


def processed_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "processed"


def results_tables_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "results" / "tables"


def results_metrics_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "results" / "metrics"


def ensure_dirs(root: Path | None = None) -> None:
    root = root or project_root()
    for dataset in DATASETS:
        raw_dir(dataset, root).mkdir(parents=True, exist_ok=True)
    for rel in [
        "code/utils",
        "data/interim",
        "data/processed",
        "results/metrics",
        "results/tables",
        "results/figures",
        "paper",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def is_excluded_path(path: Path) -> bool:
    lowered = str(path).lower()
    return any(token in lowered for token in EXCLUDED_NAME_TOKENS)


def is_tabular_path(path: Path) -> bool:
    lowered = path.name.lower()
    if is_excluded_path(path):
        return False
    return lowered.endswith(TABULAR_SUFFIXES)


def sniff_delimiter(path: Path) -> str | None:
    if path.suffix.lower() == ".tsv":
        return "\t"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            sample = handle.read(8192)
        if not sample.strip():
            return None
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        if path.suffix.lower() == ".csv":
            return ","
        return None


def read_table(path: Path, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    sep = sniff_delimiter(path)
    if sep is None:
        raise ValueError(f"Could not infer delimiter for {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=sep,
                nrows=nrows,
                usecols=usecols,
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read {path}: {last_error}")


def read_table_chunks(path: Path, chunksize: int = 100_000, usecols: list[str] | None = None) -> Iterable[pd.DataFrame]:
    sep = sniff_delimiter(path)
    if sep is None:
        raise ValueError(f"Could not infer delimiter for {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=sep,
                chunksize=chunksize,
                usecols=usecols,
                encoding=encoding,
                low_memory=False,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to stream {path}: {last_error}")


def extract_zip_csvs(zip_path: Path, dataset: str, root: Path | None = None) -> list[Path]:
    root = root or project_root()
    out_dir = interim_dir(root) / "extracted" / dataset / zip_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename.replace("\\", "/")
            if member.is_dir() or is_excluded_path(Path(name)):
                continue
            if not name.lower().endswith((".csv", ".tsv", ".txt")):
                continue
            target = out_dir / Path(name).name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if is_tabular_path(target):
                extracted.append(target)
    return extracted


def discover_dataset_files(dataset: str, root: Path | None = None, extract_archives: bool = True) -> list[Path]:
    root = root or project_root()
    base = raw_dir(dataset, root)
    if not base.exists():
        return []
    files: list[Path] = []
    for path in sorted(base.rglob("*")):
        if path.is_dir() or is_excluded_path(path):
            continue
        if is_tabular_path(path):
            if path.name.lower().startswith("readme"):
                continue
            files.append(path)
        elif extract_archives and path.suffix.lower() == ".zip":
            files.extend(extract_zip_csvs(path, dataset, root))
    return sorted(dict.fromkeys(files))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relpath(path: Path, root: Path | None = None) -> str:
    root = root or project_root()
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)
