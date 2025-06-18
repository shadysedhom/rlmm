#!/usr/bin/env python3
"""Convert Kaggle Binance BTC-perp 250 ms CSV into newline-JSON snapshots.

Usage
-----
python engine/convert_kaggle_binance.py \
    --csv data/raw/kaggle/BTCUSDT-250ms.csv \
    --symbol BTCUSDT \
    --out_dir data/raw/binance_kaggle

Creates   data/raw/binance_kaggle/BTCUSDT_depth20_YYYYMMDD.jsonl.gz
Each line:
{ "ts": 1673222400.123456, "bids": [[p1,q1],...,], "asks": [[p1,q1],...]}  
compatible with our playback engine.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

CHUNK_ROWS = 500_000  # tune for your RAM
DEPTH = 10  # dataset has 10 bid & 10 ask levels (depth 20 total book)


def build_column_names() -> List[str]:
    """Return the canonical column list *excluding* the dummy index kolumn.

    After we pass `usecols=range(1, 43)` to `pd.read_csv`, the dummy index
    (column 0 in the raw dump) is discarded, so the first real field is the
    millisecond timestamp.
    """
    cols: List[str] = ["ts_ms", "datetime"]
    for lvl in range(1, DEPTH + 1):
        cols += [f"bid_price_{lvl}", f"bid_size_{lvl}"]
    for lvl in range(1, DEPTH + 1):
        cols += [f"ask_price_{lvl}", f"ask_size_{lvl}"]
    return cols


def snap_from_row(row) -> dict:
    """Convert a pandas row to our canonical snapshot dict."""
    return {
        # convert milliseconds -> fractional seconds
        "ts": row["ts_ms"] / 1_000.0,
        "bids": [[row[f"bid_price_{i}"], row[f"bid_size_{i}"]] for i in range(1, DEPTH + 1)],
        "asks": [[row[f"ask_price_{i}"], row[f"ask_size_{i}"]] for i in range(1, DEPTH + 1)],
    }


def write_chunk(df: pd.DataFrame, out_dir: Path, symbol: str):
    """Append rows grouped by YYYYMMDD into gzipped JSONL files."""
    # Derive YYYYMMDD from the millisecond epoch timestamp – avoids reliance
    # on the (sometimes corrupted) human-readable datetime string included in
    # the dump.
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.strftime("%Y%m%d")
    for date, grp in df.groupby("date"):
        out_path = out_dir / f"{symbol}_depth{DEPTH}_{date}.jsonl.gz"
        with gzip.open(out_path, "at") as gz:
            for _, row in grp.iterrows():
                gz.write(json.dumps(snap_from_row(row)) + "\n")


def convert(csv_path: Path, out_dir: Path, symbol: str):
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    expected_cols = 2 + DEPTH * 4  # ts_ms, datetime + 20 price/size pairs *2

    reader = pd.read_csv(
        csv_path,
        header=None,
        chunksize=CHUNK_ROWS,
        skiprows=1,            # drop the numeric header row (0,1,2,…)
        usecols=range(1, 43),  # drop dummy row-index column; keep 42 fields
        names=build_column_names(),
    )

    for chunk in tqdm(reader, desc="Converting chunks"):
        if chunk.shape[1] != expected_cols:
            raise ValueError(
                f"Unexpected column count {chunk.shape[1]} (expected {expected_cols})."
            )

        # No further column fix-ups needed; optional drop of datetime string
        chunk = chunk.drop(columns=["datetime"])
        write_chunk(chunk, out_dir, symbol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Kaggle Binance CSV to JSONL snapshots")
    ap.add_argument("--csv", type=Path, required=True, help="Path to raw Kaggle CSV file")
    ap.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol (used in file names)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/raw/binance_kaggle"), help="Destination directory")
    args = ap.parse_args()

    try:
        convert(args.csv, args.out_dir, args.symbol.upper())
    except KeyboardInterrupt:
        sys.exit("Interrupted") 