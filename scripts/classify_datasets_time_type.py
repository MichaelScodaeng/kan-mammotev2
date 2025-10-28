#!/usr/bin/env python3
"""
Inspect all dataset folders under `processed_data/` and try to classify each
as a continuous-time dynamic graph (CTDG) or discrete-time dynamic graph (DTDG)
using heuristics on timestamp columns and value distributions.

Output: CSV summary `sh_scripts/kmm_aftertune/dataset_time_type_summary.csv`

Usage:
  python scripts/classify_datasets_time_type.py --data-dir processed_data --out summary.csv

Heuristics (brief):
 - Look for timestamp-like column names: 'time','ts','timestamp','t','time_idx','snapshot', 'snap'
 - If timestamp values are floats or many unique values relative to rows -> continuous-time
 - If timestamp values are small-range integers (e.g., 0..T-1 with T small (<500)) -> discrete-time
 - If no timestamp found, mark as 'unknown'

This script is conservative and prints summary stats to help manual review.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


TIMESTAMP_KEYWORDS = [
    "time",
    "timestamp",
    "ts",
    "t",
    "time_sec",
    "time_ms",
    "time_us",
    "time_idx",
    "snapshot",
    "snap",
    "step",
    "bin",
]


def find_csv_file(folder: Path) -> Optional[Path]:
    # prefer ml_*.csv then *.csv
    ml_csvs = list(folder.glob("ml_*.csv"))
    if ml_csvs:
        return ml_csvs[0]
    csvs = [p for p in folder.glob("*.csv") if p.is_file()]
    return csvs[0] if csvs else None


def find_npy_file(folder: Path) -> Optional[Path]:
    ml_npy = list(folder.glob("ml_*.npy"))
    if ml_npy:
        return ml_npy[0]
    npys = [p for p in folder.glob("*.npy") if p.is_file()]
    return npys[0] if npys else None


def detect_timestamp_column(df_columns):
    # return column name or None
    for col in df_columns:
        lc = str(col).lower()
        for kw in TIMESTAMP_KEYWORDS:
            if kw in lc:
                return col
    # fallback: if a column named like 2nd/3rd that appears numeric could be timestamp
    return None


def analyze_series(arr: np.ndarray) -> Dict[str, Any]:
    # arr is 1D numeric
    stats = {}
    arr = arr.astype(float)
    n = arr.size
    stats["n_rows"] = int(n)
    if n == 0:
        stats.update({"n_unique": 0, "min": None, "max": None, "is_integer": None})
        return stats
    uniq = np.unique(arr)
    stats["n_unique"] = int(uniq.size)
    stats["min"] = float(arr.min())
    stats["max"] = float(arr.max())
    stats["range"] = float(stats["max"] - stats["min"]) if stats["max"] is not None else None
    # integer check: nearly integers
    is_int = np.all(np.abs(arr - np.round(arr)) < 1e-8)
    stats["is_integer"] = bool(is_int)
    # large-values heuristic (possible unix timestamps)
    stats["max_is_large"] = bool(stats["max"] is not None and stats["max"] > 1e9)
    stats["unique_ratio"] = float(stats["n_unique"]) / max(1, n)
    return stats


def classify_from_stats(stats: Dict[str, Any]) -> str:
    """Return one of: 'continuous', 'discrete', 'unknown'"""
    if stats["n_rows"] == 0 or stats["n_unique"] == 0:
        return "unknown"
    # If timestamps are large unix epoch (and many unique) -> continuous
    if stats.get("max_is_large") and stats.get("n_unique", 0) > min(1000, stats["n_rows"]):
        return "continuous"
    # If values are floats and many unique -> continuous
    if not stats.get("is_integer") and stats.get("n_unique", 0) > min(500, max(50, int(0.6 * stats["n_rows"]))):
        return "continuous"
    # If integer timestamps and small range -> discrete
    if stats.get("is_integer"):
        rng = stats.get("range", 0)
        nuniq = stats.get("n_unique", 0)
        # small number of time steps
        if nuniq <= 500 and rng <= 2000:
            return "discrete"
        # if unique ratio is very high (close to 1) even if int -> likely continuous event times recorded as ints
        if stats.get("unique_ratio", 0) > 0.7 and nuniq > 1000:
            return "continuous"
    # if many unique values relative to rows -> continuous
    if stats.get("unique_ratio", 0) > 0.6 and stats.get("n_unique", 0) > 200:
        return "continuous"
    # ambiguous
    return "unknown"


def inspect_csv(path: Path, sample_rows: int = 20000) -> Dict[str, Any]:
    info = {"source": str(path), "method": "csv", "detected_timestamp": None, "stats": None, "classification": "unknown"}
    if pd is None:
        info["error"] = "pandas not installed"
        return info
    try:
        # try reading with pandas but only a sample (for large files)
        # If file has header, keep it
        df = pd.read_csv(path, nrows=sample_rows)
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t", nrows=sample_rows)
        except Exception as e:
            info["error"] = f"could not read csv: {e}"
            return info
    col = detect_timestamp_column(df.columns)
    if col is None:
        # try any numeric column heuristics: choose numeric column with many unique values
        nums = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if nums:
            # pick the column with highest unique ratio
            best = None
            best_ratio = -1.0
            for c in nums:
                uniq = df[c].nunique(dropna=True)
                ratio = uniq / max(1, len(df))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = c
            col = best
    if col is None:
        return info
    info["detected_timestamp"] = str(col)
    # keep as Series so pd.to_numeric returns a Series and dropna() is available
    series = df[col].dropna()
    # coerce to numeric (returns Series), drop NaNs, then convert to numpy
    series = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    stats = analyze_series(series)
    info["stats"] = stats
    info["classification"] = classify_from_stats(stats)
    return info


def inspect_npy(path: Path, sample_rows: int = 20000) -> Dict[str, Any]:
    info = {"source": str(path), "method": "npy", "detected_timestamp": None, "stats": None, "classification": "unknown"}
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        info["error"] = f"could not load npy: {e}"
        return info
    # If structured array with named fields
    if hasattr(data, "dtype") and data.dtype.names:
        cols = data.dtype.names
        # pick timestamp-like
        col = None
        for c in cols:
            if any(kw in c.lower() for kw in TIMESTAMP_KEYWORDS):
                col = c
                break
        if col is None:
            # fallback: try last numeric field
            for c in reversed(cols):
                arr = data[c]
                if np.issubdtype(arr.dtype, np.number):
                    col = c
                    break
        if col is None:
            return info
        arr = data[col]
        stats = analyze_series(np.array(arr).astype(float))
        info["detected_timestamp"] = str(col)
        info["stats"] = stats
        info["classification"] = classify_from_stats(stats)
        return info

    # else if plain 2D array, try to guess timestamp column: many event files are (src, dst, t, ...)
    if data.ndim == 2 and data.shape[0] > 0 and data.shape[1] >= 3:
        # assume third column (index 2) is timestamp
        col_idx = 2
        arr = data[:sample_rows, col_idx]
        try:
            arr = np.asarray(arr, dtype=float)
        except Exception:
            return info
        stats = analyze_series(arr)
        info["detected_timestamp"] = f"col_{col_idx}"
        info["stats"] = stats
        info["classification"] = classify_from_stats(stats)
        return info

    # other cases: mark unknown
    return info


def classify_folder(folder: Path) -> Dict[str, Any]:
    out = {"dataset": folder.name, "path": str(folder), "result": "unknown", "details": None}
    csvf = find_csv_file(folder)
    if csvf:
        info = inspect_csv(csvf)
        out["result"] = info.get("classification", "unknown")
        out["details"] = info
        return out
    npyf = find_npy_file(folder)
    if npyf:
        info = inspect_npy(npyf)
        out["result"] = info.get("classification", "unknown")
        out["details"] = info
        return out
    # nothing found
    out["details"] = {"error": "no csv or npy found"}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="processed_data", help="folder containing dataset subfolders")
    p.add_argument("--out", default="sh_scripts/kmm_aftertune/dataset_time_type_summary.csv", help="CSV output path")
    p.add_argument("--write-json", default=None, help="optional JSON detailed report path")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"data dir not found: {data_dir}")
        return
    results = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        print(f"Inspecting: {child.name}")
        r = classify_folder(child)
        results.append(r)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "path", "result", "timestamp_col", "n_rows", "n_unique", "min", "max", "is_integer", "notes"])
        for r in results:
            details = r.get("details") or {}
            det_stats = details.get("stats") if isinstance(details, dict) else None
            timestamp_col = details.get("detected_timestamp") if isinstance(details, dict) else None
            notes = details.get("error") if isinstance(details, dict) and details.get("error") else details.get("method")
            n_rows = det_stats.get("n_rows") if det_stats else None
            n_unique = det_stats.get("n_unique") if det_stats else None
            mn = det_stats.get("min") if det_stats else None
            mx = det_stats.get("max") if det_stats else None
            is_int = det_stats.get("is_integer") if det_stats else None
            writer.writerow([r["dataset"], r["path"], r["result"], timestamp_col, n_rows, n_unique, mn, mx, is_int, notes])

    print(f"Wrote summary CSV -> {out_csv}")
    if args.write_json:
        with open(args.write_json, "w") as jfh:
            json.dump(results, jfh, indent=2)
        print(f"Wrote detailed JSON -> {args.write_json}")


if __name__ == "__main__":
    main()
