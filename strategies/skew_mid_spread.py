"""Inventory-skewed mid-spread maker.

Price quotes are shifted ("skewed") away from the risky side so that
inventory mean-reverts toward zero.

bid = mid − spread/2 − γ·inventory
ask = mid + spread/2 − γ·inventory

A positive inventory (long) pushes the bid price down (less eager to buy) and the ask price down (more eager to sell) so the
strategy preferentially sells to achieve delta-neutrality.  A negative inventory (short) does the
opposite, pushing the bid price up (more eager to buy) and the ask price up (less eager to sell).

Only price skew is implemented here; both sides quote the same fixed
size because the current BookSimulator API supports a single size
parameter.  Size asymmetry can be added once the simulator tracks
per-side order sizes.
"""
from __future__ import annotations

from typing import Dict, Any


class Strategy:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        size: float = 0.1,
        gamma: float = 0.1,
        max_inventory: float = 100.0,
    ):
        """Create a skewed mid-spread strategy.

        Args:
            size: Base order size in BTC (applied to both sides).
            gamma: Price skew coefficient in *price units* per BTC of
                inventory.  For BTC-USDT with 0.1 tick, gamma=0.1 means
                each 1 BTC of inventory shifts quotes by one tick.
            max_inventory: Maximum inventory size to consider for position sizing.
        """
        self.size = size
        self.gamma = gamma
        self.max_inventory = max_inventory

    # ------------------------------------------------------------------
    # Public API expected by engine.playback
    # ------------------------------------------------------------------
    def on_snapshot(
        self,
        snapshot: Dict[str, Any],
        inventory: float = 0.0,
    ) -> Dict[str, float]:
        """Generate quotes based on current order book snapshot and inventory."""
        mid = snapshot["mid"]
        spread = snapshot["asks"][0][0] - snapshot["bids"][0][0]
        
        # Inventory-based position sizing: reduce size when inventory is large
        # This creates more realistic volatility as large positions are harder to manage
        inventory_factor = max(0.1, 1.0 - abs(inventory) / self.max_inventory)
        adjusted_size = self.size * inventory_factor
        
        # Skew quotes based on inventory to encourage mean reversion
        skew = self.gamma * inventory * spread
        
        bid_price = mid - spread/2 + skew
        ask_price = mid + spread/2 + skew
        
        return {
            "bid_price": bid_price,
            "ask_price": ask_price,
            "size": adjusted_size
        } 