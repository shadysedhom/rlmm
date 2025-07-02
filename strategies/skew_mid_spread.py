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
        use_ticks: bool = False,
        clamp_ticks: int = 2,
    ):
        """Create a skewed mid-spread strategy.

        Args:
            size: Base order size in BTC (applied to both sides).
            gamma: Price skew coefficient in *price units* per BTC of
                inventory.  For BTC-USDT with 0.1 tick, gamma=0.1 means
                each 1 BTC of inventory shifts quotes by one tick.
            max_inventory: Maximum inventory size to consider for position sizing.
            use_ticks: If True, interpret gamma in ticks/BTC
            clamp_ticks: minimum distance heavy side must stay behind touch
        """
        self.size = size
        self.gamma = gamma
        self.max_inventory = max_inventory
        self.use_ticks = use_ticks
        self.clamp_ticks = clamp_ticks

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
        best_bid = snapshot["bids"][0][0]
        best_ask = snapshot["asks"][0][0]
        spread = best_ask - best_bid

        # Derive tick size from first two bid levels if possible
        if len(snapshot["bids"]) >= 2:
            tick = round(snapshot["bids"][0][0] - snapshot["bids"][1][0], 8)
            if tick <= 0:
                tick = 0.01  # fallback
        else:
            tick = 0.01

        # Position sizing depending on inventory magnitude
        inventory_factor = max(0.1, 1.0 - abs(inventory) / self.max_inventory)
        adjusted_size = self.size * inventory_factor

        # ---------------- Skew -----------------
        if self.use_ticks:
            skew_px = self.gamma * inventory * tick
        else:
            skew_px = self.gamma * inventory * spread

        bid_price = mid - spread / 2 - skew_px
        ask_price = mid + spread / 2 - skew_px

        # Round to tick grid
        bid_price = round(bid_price / tick) * tick
        ask_price = round(ask_price / tick) * tick

        # --------------- Clamp heavy side ---------------
        if inventory > 0:  # long → heavy side is bid
            min_bid = best_bid - self.clamp_ticks * tick
            if bid_price >= min_bid:
                bid_price = min_bid
        elif inventory < 0:  # short → heavy side is ask
            min_ask = best_ask + self.clamp_ticks * tick
            if ask_price <= min_ask:
                ask_price = min_ask

        # Ensure we never cross
        if bid_price >= ask_price:
            # Widen by one tick to maintain proper order
            bid_price = ask_price - tick

        return {
            "bid_price": bid_price,
            "ask_price": ask_price,
            "size": adjusted_size,
        } 