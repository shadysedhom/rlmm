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
        size: float | None = None,
        size_base: float | None = None,
        gamma: float = 0.1,
        max_inventory: float = 100.0,
        use_ticks: bool = False,
        clamp_ticks: int = 2,
        gamma_imb: float = 0.0,
        # --- volatility-adaptive spread params ---
        k_vol: float = 1.0,
        vol_window: int = 20,
        # --- depth-gradient params ---
        gradient_threshold: float = 1.5,
        gradient_penalty_ticks: int = 1,
        gradient_size_factor: float = 0.7,
        # --- adaptive gamma params ---
        gamma_inv_scale: float = 1.0,   # α: how strongly gamma scales with |inventory| / max_inventory
        gamma_vol_scale: float = 1.0,   # β: how strongly gamma scales with realised vol / vol_ref
        gamma_cap_factor: float = 3.0,  # cap: γ_dynamic ≤ γ_base * cap_factor
        vol_ref: float | None = None,   # reference vol; if None will be estimated on first window
        # --- dynamic sizing params ---
        size_min_factor: float = 0.2,   # lower bound fraction of base_size
        size_exp: float = 1.0,          # exponent for non-linear sizing (1 = linear)
        # --- fixed spread padding ---
        extra_spread_ticks: int = 0,    # add N ticks to half-spread regardless of volatility
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
            gamma_inv_scale: α: how strongly gamma scales with |inventory| / max_inventory
            gamma_vol_scale: β: how strongly gamma scales with realised vol / vol_ref
            gamma_cap_factor: cap: γ_dynamic ≤ γ_base * cap_factor
            vol_ref: reference vol; if None will be estimated on first window
            size_min_factor: lower bound fraction of base_size
            size_exp: exponent for non-linear sizing (1 = linear)
            extra_spread_ticks: fixed number of ticks added to half-spread
        """
        if size is not None:
            self.base_size = size
        elif size_base is not None:
            self.base_size = size_base
        else:
            self.base_size = 0.1
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

        # adaptive gamma parameters
        self.gamma_inv_scale = gamma_inv_scale
        self.gamma_vol_scale = gamma_vol_scale
        self.gamma_cap_factor = max(1.0, gamma_cap_factor)
        self.vol_ref = vol_ref  # set later if None

        # sizing params
        self.size_min_factor = max(0.0, min(size_min_factor, 1.0))
        self.size_exp = max(0.1, size_exp)

        # fixed spread padding
        self.extra_spread_ticks = max(0, int(extra_spread_ticks))

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

        # -------- Dynamic position sizing --------
        inv_ratio = min(1.0, abs(inventory) / self.max_inventory)
        inventory_factor = max(
            self.size_min_factor,
            (1.0 - inv_ratio) ** self.size_exp,
        )
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

        # Establish reference vol if not provided (use median of first full window)
        if self.vol_ref is None and len(self._price_window) == self.vol_window:
            self.vol_ref = float(np.median(self._price_window)) * 1e-6 + vol  # avoid zero

        half_spread = max(tick, self.k_vol * vol) + self.extra_spread_ticks * tick

        # ---------------- Adaptive gamma -----------------
        gamma_dynamic = self.gamma

        # inventory scaling
        if self.gamma_inv_scale != 0 and self.max_inventory > 0:
            gamma_dynamic *= 1 + self.gamma_inv_scale * abs(inventory) / self.max_inventory

        # volatility scaling
        if self.gamma_vol_scale != 0 and self.vol_ref:
            gamma_dynamic *= 1 + self.gamma_vol_scale * (vol / self.vol_ref)

        # cap gamma to avoid extreme widening
        gamma_dynamic = min(gamma_dynamic, self.gamma * self.gamma_cap_factor)

        # ---------------- Skew -----------------
        if self.use_ticks:
            skew_px = gamma_dynamic * inventory * tick
        else:
            skew_px = gamma_dynamic * inventory * (best_ask - best_bid)

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