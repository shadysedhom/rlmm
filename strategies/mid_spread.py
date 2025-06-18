"""Naïve mid-spread quoting strategy.

Quotes at mid ± half the top-of-book spread with fixed size.
"""
from __future__ import annotations

from typing import Dict, Any


class Strategy:
    def __init__(self, size: float = 0.001):
        self.size = size

    def on_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        best_bid = snapshot["bids"][0][0]
        best_ask = snapshot["asks"][0][0]
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        return {
            "bid_price": mid - spread / 2,
            "ask_price": mid + spread / 2,
            "size": self.size,
        } 