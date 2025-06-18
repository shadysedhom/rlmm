#!/usr/bin/env python3
"""Simple playback engine that streams historical snapshots to a trading strategy
and accounts for naive fills.

Usage
-----
python engine/playback.py --data_dir data/converted/binance \
                          --strategy mid_spread \
                          --speed 0  # 0 = as fast as possible; else realtime factor

The playback reads *.jsonl.gz files produced by convert_kaggle_binance.py.
It instantiates the chosen strategy from strategies/ and prints P&L every N sec.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import time
import sys
from pathlib import Path
from typing import Generator, List, Dict, Any

# Ensure project root (two levels up) is on import path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class SnapshotLoader:
    """Yield snapshots dicts sorted by timestamp from all files in a directory."""

    def __init__(self, data_dir: Path):
        self.files: List[Path] = sorted(data_dir.glob("*.jsonl.gz"))
        if not self.files:
            raise FileNotFoundError(f"No *.jsonl.gz files found in {data_dir}")

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        for fp in self.files:
            with gzip.open(fp, "rt") as gz:
                for line in gz:
                    yield json.loads(line)


def load_strategy(name: str):
    """Dynamically import strategies.<name>.Strategy class."""
    module = importlib.import_module(f"strategies.{name}")
    return module.Strategy  # type: ignore[attr-defined]


class BookSimulator:
    """Very naive matching engine for best-quote fills."""

    def __init__(self):
        self.inventory: float = 0.0
        self.cash: float = 0.0

    def process(self, snapshot: Dict[str, Any], quote: Dict[str, float]):
        """Update positions given market snapshot and our outstanding quote.

        quote = {"bid_price": float, "ask_price": float, "size": float}
        """
        best_bid = snapshot["bids"][0][0]
        best_ask = snapshot["asks"][0][0]
        size = quote["size"]

        # Fill buy side if our bid crosses current best ask
        if quote["bid_price"] >= best_ask:
            self.inventory += size
            self.cash -= quote["bid_price"] * size

        # Fill sell side if our ask crosses current best bid
        if quote["ask_price"] <= best_bid:
            self.inventory -= size
            self.cash += quote["ask_price"] * size

    def mark_to_market(self, mid: float) -> float:
        return self.cash + self.inventory * mid


def main():
    parser = argparse.ArgumentParser(description="Playback historical order-book snapshots")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--strategy", default="mid_spread", help="Strategy module under strategies/")
    parser.add_argument("--speed", type=float, default=0.0, help="0=as fast as possible; else realtime divisor (e.g., 10=10x faster)")
    parser.add_argument("--print_every", type=int, default=1000, help="How many snapshots between P&L prints")
    args = parser.parse_args()

    loader = SnapshotLoader(args.data_dir)
    StratCls = load_strategy(args.strategy)
    strat = StratCls()
    book = BookSimulator()

    prev_ts: float | None = None
    for i, snap in enumerate(loader, start=1):
        ts = snap["ts"]
        mid = (snap["bids"][0][0] + snap["asks"][0][0]) / 2.0

        # Optional real-time throttling
        if args.speed > 0 and prev_ts is not None:
            delay = (ts - prev_ts) / args.speed
            if delay > 0:
                time.sleep(delay)
        prev_ts = ts

        quote = strat.on_snapshot(snap)
        book.process(snap, quote)

        if i % args.print_every == 0:
            pnl = book.mark_to_market(mid)
            print(f"[{i}] ts={ts:.3f}  inventory={book.inventory:.4f}  PnL={pnl:.2f}")


if __name__ == "__main__":
    main() 