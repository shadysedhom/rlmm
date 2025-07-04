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
from collections import deque
import numpy as np


class Strategy:  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        size: float = 0.1,
        gamma: float = 0.1,
        max_inventory: float = 100.0,
        use_ticks: bool = False,
        clamp_ticks: int = 2,
        gamma_imb: float = 0.0,
        # --- new volatility-adaptive params ---
        k_vol: float = 1.0,
        vol_window: int = 20,
        # --- depth-gradient params ---
        gradient_threshold: float = 1.5,
        gradient_penalty_ticks: int = 1,
        gradient_size_factor: float = 0.7,
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
            gamma_imb: Price skew coefficient in ticks/BTC per unit depth imbalance [-1,1]
            k_vol: Volatility coefficient
            vol_window: Window size for realised volatility calculation
            gradient_threshold: Threshold for depth gradient adjustment
            gradient_penalty_ticks: Penalty for depth gradient adjustment
            gradient_size_factor: Size factor for depth gradient adjustment
        """
        self.base_size = size  # keep original for reference
        self.gamma = gamma
        self.max_inventory = max_inventory
        self.use_ticks = use_ticks
        self.clamp_ticks = clamp_ticks
        self.gamma_imb = gamma_imb  # ticks of skew per unit depth imbalance [-1,1]

        # --- new parameters ---
        self.k_vol = k_vol
        self.vol_window = max(2, int(vol_window))
        self.gradient_threshold = gradient_threshold
        self.gradient_penalty_ticks = gradient_penalty_ticks
        self.gradient_size_factor = gradient_size_factor

        # rolling window of recent micro-prices for realised vol calc
        self._price_window: deque[float] = deque(maxlen=self.vol_window)

    # ------------------------------------------------------------------
    # Public API expected by engine.playback
    # ------------------------------------------------------------------
    def on_snapshot(
        self,
        snapshot: Dict[str, Any],
        inventory: float = 0.0,
    ) -> Dict[str, float]:
        """Generate quotes based on current order book snapshot and inventory."""
        # Snapshots created by playback include explicit 'mid'.  Unit tests may omit it.
        if "mid" in snapshot:
            mid = snapshot["mid"]
        else:
            mid = (snapshot["bids"][0][0] + snapshot["asks"][0][0]) / 2.0
        best_bid = snapshot["bids"][0][0]
        best_ask = snapshot["asks"][0][0]

        # Derive tick size from first two bid levels if possible
        if len(snapshot["bids"]) >= 2:
            tick = round(snapshot["bids"][0][0] - snapshot["bids"][1][0], 8)
            if tick <= 0:
                tick = 0.01  # fallback
        else:
            tick = 0.01

        # Position sizing depending on inventory magnitude (base before gradient adjustments)
        inventory_factor = max(0.1, 1.0 - abs(inventory) / self.max_inventory)
        adjusted_size = self.base_size * inventory_factor

        # ------------- Fair value & volatility -------------
        if "depth" in snapshot:
            micro_price = snapshot["depth"].micro_price(levels=5)
        else:
            micro_price = (best_bid + best_ask) / 2.0

        # Maintain rolling window and compute realised vol (stdev)
        self._price_window.append(micro_price)
        if len(self._price_window) > 1:
            vol = float(np.std(self._price_window))
        else:
            vol = tick  # fallback minimal vol

        half_spread = max(tick, self.k_vol * vol)

        # ---------------- Skew -----------------
        if self.use_ticks:
            skew_px = self.gamma * inventory * tick
        else:
            skew_px = self.gamma * inventory * (best_ask - best_bid)

        bid_price = micro_price - half_spread - skew_px
        ask_price = micro_price + half_spread - skew_px

        # ---------------- Depth-imbalance skew ----------------
        if self.gamma_imb != 0.0 and "depth" in snapshot:
            try:
                imb = snapshot["depth"].imbalance(levels=5)  # ∈ [-1,1]
                price_adjust = self.gamma_imb * imb * tick
                bid_price += price_adjust
                ask_price += price_adjust
            except Exception:
                # Gracefully ignore if depth helper missing/malformed
                pass

        # --------------- Depth gradient check ---------------
        if "depth" in snapshot:
            depth = snapshot["depth"]

            # Bid side gradient
            shallow_bid = depth.cum_volume("buy", 2)
            deep_bid = max(1e-9, depth.cum_volume("buy", 5) - shallow_bid)
            gradient_bid = shallow_bid / deep_bid

            # Ask side gradient
            shallow_ask = depth.cum_volume("sell", 2)
            deep_ask = max(1e-9, depth.cum_volume("sell", 5) - shallow_ask)
            gradient_ask = shallow_ask / deep_ask

            # If book thins quickly behind the touch, quote more conservatively
            if gradient_bid > self.gradient_threshold:
                bid_price -= self.gradient_penalty_ticks * tick
                adjusted_size *= self.gradient_size_factor

            if gradient_ask > self.gradient_threshold:
                ask_price += self.gradient_penalty_ticks * tick
                adjusted_size *= self.gradient_size_factor

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

        # Round to tick grid after all adjustments
        bid_price = round(bid_price / tick) * tick
        ask_price = round(ask_price / tick) * tick

        return {
            "bid_price": bid_price,
            "ask_price": ask_price,
            "size": adjusted_size,
        } 