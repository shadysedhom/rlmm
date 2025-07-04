"""Depth utilities for N-level order-book snapshots.

This module is completely *data-structure* focused; it contains **no** trading logic.  
Keeping it separate prevents `book.py` from growing even larger and lets strategies
import depth analytics without touching the simulator internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

PriceLevel = Tuple[float, float]  # (price, qty)


@dataclass
class DepthLadder:
    """Wrapper around 10-level Binance depth arrays.

    Attributes
    ----------
    bids / asks : list[PriceLevel]
        Each element is (price, quantity).  Prices are assumed sorted
        bids: descending, asks: ascending.
    ts : float
        Epoch-seconds timestamp of the snapshot.
    """

    bids: List[PriceLevel]
    asks: List[PriceLevel]
    ts: float | None = None
    _cum_bid: np.ndarray = field(init=False, repr=False)
    _cum_ask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Sanity — keep original order (bids desc, asks asc) but store cumulative qty for O(1) queries
        bid_qty = np.array([q for _, q in self.bids], dtype=float)
        ask_qty = np.array([q for _, q in self.asks], dtype=float)
        self._cum_bid = np.cumsum(bid_qty)
        self._cum_ask = np.cumsum(ask_qty)

    # ------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------
    def best_bid(self) -> float:
        return self.bids[0][0]

    def best_ask(self) -> float:
        return self.asks[0][0]

    def mid(self) -> float:
        return (self.best_bid() + self.best_ask()) / 2.0

    # ------------------------------------------------------------
    # Aggregate metrics over N levels
    # ------------------------------------------------------------
    def cum_volume(self, side: str, levels: int) -> float:
        """Cumulative quantity up to *levels* (1-indexed)."""
        levels = max(1, min(levels, len(self.bids)))  # protect bounds; bids & asks len equal
        if side == "buy":
            return float(self._cum_bid[levels - 1])
        elif side == "sell":
            return float(self._cum_ask[levels - 1])
        else:
            raise ValueError("side must be 'buy' or 'sell'")

    def imbalance(self, levels: int = 5) -> float:
        """Signed depth imbalance ∈ [-1, 1].

        +1  = all volume on bid side,  ‑1 = all on ask side.
        """
        bid_vol = self.cum_volume("buy", levels)
        ask_vol = self.cum_volume("sell", levels)
        if bid_vol + ask_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    # ------------------------------------------------------------
    # Micro-price — liquidity-weighted mid
    # ------------------------------------------------------------
    def micro_price(self, levels: int = 5) -> float:
        """Liquidity-weighted price used by many HFT desks."""
        levels = max(1, min(levels, len(self.bids)))
        bid_p = np.array([p for p, _ in self.bids[:levels]], dtype=float)
        ask_p = np.array([p for p, _ in self.asks[:levels]], dtype=float)
        bid_w = self._cum_bid[:levels]
        ask_w = self._cum_ask[:levels]
        return float((bid_p @ bid_w + ask_p @ ask_w) / (bid_w.sum() + ask_w.sum())) 