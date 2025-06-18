#!/usr/bin/env python3
"""Minimal order-book simulator with queue-depletion fills.

Designed for Level-2 snapshot data (price, visible size).
Only supports one active order per side for simplicity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import random
import numpy as np

from engine import logger


@dataclass
class Order:
    side: str  # "buy" or "sell"
    price: float
    size: float
    queue_ahead: float  # visible size ahead of us at placement
    placed_ts: float
    filled: float = 0.0
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def is_filled(self) -> bool:
        return self.filled >= self.size - 1e-9  # float tolerance


class BookSimulator:
    """Track one quote per side and settle fills via queue depletion."""

    def __init__(
        self,
        tick_size: float,
        cancel_if_behind: bool = True,
        max_ticks_away: int = 3,
        maker_rebate: float = 0.0001,  # +0.01% realistic Binance VIP0
        taker_fee: float = 0.0004,     # -0.04%
        latency_mean_ms: float = 20.0,
        latency_std_ms: float = 5.0,
        min_keep_filled_frac: float = 0.25,
        max_inventory: float = 0.01,
        order_rejection_rate: float = 0.001,  # 0.1% of orders rejected
        partial_fill_rate: float = 0.05,      # 5% of fills are partial
        execution_slippage_bps: float = 0.5,  # 0.5 bps average slippage
        queue_position_uncertainty: float = 0.1,  # 10% uncertainty in queue position
        # Queue competition parameters
        min_competitors: int = 1,             # Minimum competing orders per price level
        max_competitors: int = 5,             # Maximum competing orders per price level
        competitor_order_size: float = 0.05,  # Average size of competing orders (BTC)
        # API and infrastructure parameters
        api_rate_limit_orders_per_sec: int = 10,  # Binance limit: 10 orders/second
        api_rate_limit_orders_per_10min: int = 100000,  # Binance limit: 100k orders/10min
        maintenance_window_probability: float = 0.0001,  # 0.01% chance of maintenance
        network_congestion_probability: float = 0.001,   # 0.1% chance of network issues
    ):
        self.tick = tick_size
        self.cancel_if_behind = cancel_if_behind
        self.max_ticks_away = max_ticks_away
        self.maker_rebate = maker_rebate
        self.taker_fee = taker_fee
        self.lat_mean = latency_mean_ms / 1000.0
        self.lat_std = latency_std_ms / 1000.0
        self.min_keep_filled_frac = min_keep_filled_frac
        self.max_inventory = abs(max_inventory)
        self.order_rejection_rate = order_rejection_rate
        self.partial_fill_rate = partial_fill_rate
        self.execution_slippage_bps = execution_slippage_bps
        self.queue_position_uncertainty = queue_position_uncertainty
        # Competition parameters
        self.min_competitors = min_competitors
        self.max_competitors = max_competitors
        self.competitor_order_size = competitor_order_size
        # API and infrastructure parameters
        self.api_rate_limit_orders_per_sec = api_rate_limit_orders_per_sec
        self.api_rate_limit_orders_per_10min = api_rate_limit_orders_per_10min
        self.maintenance_window_probability = maintenance_window_probability
        self.network_congestion_probability = network_congestion_probability
        
        # Order rate tracking
        self.orders_this_second = 0
        self.orders_this_10min = 0
        self.last_second_ts = None
        self.last_10min_ts = None
        
        # Dynamic competition intensity (set in update)
        self.competition_intensity = 0.5
        
        self.active_orders: List[Order] = []
        self.filled_orders: List[Order] = []  # Keep track of filled orders for validation
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.fills_count: int = 0

        # Orders waiting for activation due to latency
        self.pending_orders: List[dict] = []  # each dict keys: side, price, size, activate_ts
        
        # PnL attribution
        self.pnl_components = {
            "inventory_moves": 0.0,
            "fees": 0.0,
            "rebates": 0.0,
            "adverse_selection": 0.0,
            "funding_rates": 0.0,  # New component for perpetual futures funding
        }
        
        # Funding rate tracking for perpetual futures
        self.last_funding_ts = None
        self.funding_interval = 8 * 3600  # 8 hours in seconds
        self.funding_rate_mean = 0.0001   # 0.01% per 8h period (typical)
        self.funding_rate_std = 0.0002    # Volatility in funding rates
        
        # Last snapshot for validation
        self.last_snapshot = None
        
        logger.info("BookSimulator initialized", 
                   tick_size=tick_size,
                   maker_rebate=maker_rebate,
                   taker_fee=taker_fee,
                   latency_mean_ms=latency_mean_ms*1000,
                   latency_std_ms=latency_std_ms*1000,
                   order_rejection_rate=order_rejection_rate,
                   partial_fill_rate=partial_fill_rate,
                   execution_slippage_bps=execution_slippage_bps,
                   competition_intensity=self.competition_intensity,
                   min_competitors=min_competitors,
                   max_competitors=max_competitors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def place_quotes(self, snapshot: dict, bid_price: float, ask_price: float, size: float):
        """Refresh quotes while keeping partially filled orders that remain at the same price.

        For each side we allow at most one active order (existing design).  If an order on
        that side is still live and its price equals the desired new price we leave it in
        place to preserve time-priority.  Otherwise we cancel/replace that side.
        """
        # First, promote any pending orders whose latency has elapsed using this snapshot
        self._activate_pending(snapshot)

        # Separate existing orders by side for easy lookup
        buy_ord = next((o for o in self.active_orders if o.side == "buy" and not o.is_filled()), None)
        sell_ord = next((o for o in self.active_orders if o.side == "sell" and not o.is_filled()), None)

        new_active: List[Order] = []

        # BUY SIDE
        if buy_ord is not None and abs(buy_ord.price - bid_price) < 1e-9:
            # Keep existing order · preserves queue position
            new_active.append(buy_ord)
            logger.debug("Keeping buy order", 
                        order_id=buy_ord.order_id, 
                        price=buy_ord.price, 
                        size=buy_ord.size,
                        filled=buy_ord.filled,
                        queue_ahead=buy_ord.queue_ahead)
        else:
            # Cancel (implicitly drop) and post fresh order
            if buy_ord is not None:
                logger.info("Canceling buy order", 
                           order_id=buy_ord.order_id, 
                           price=buy_ord.price, 
                           size=buy_ord.size,
                           filled=buy_ord.filled,
                           queue_ahead=buy_ord.queue_ahead,
                           event=logger.OrderEvent.CANCELED)
            new_active.append(self._submit_order("buy", bid_price, size, snapshot))

        # SELL SIDE
        if sell_ord is not None and abs(sell_ord.price - ask_price) < 1e-9:
            new_active.append(sell_ord)
            logger.debug("Keeping sell order", 
                        order_id=sell_ord.order_id, 
                        price=sell_ord.price, 
                        size=sell_ord.size,
                        filled=sell_ord.filled,
                        queue_ahead=sell_ord.queue_ahead)
        else:
            # Cancel (implicitly drop) and post fresh order
            if sell_ord is not None:
                logger.info("Canceling sell order", 
                           order_id=sell_ord.order_id, 
                           price=sell_ord.price, 
                           size=sell_ord.size,
                           filled=sell_ord.filled,
                           queue_ahead=sell_ord.queue_ahead,
                           event=logger.OrderEvent.CANCELED)
            new_active.append(self._submit_order("sell", ask_price, size, snapshot))

        self.active_orders = new_active
        self._validate_state(snapshot)

    def update(self, snapshot: dict):
        """Advance simulator by one snapshot; update queue positions & fills."""
        # Store snapshot for validation
        self.last_snapshot = snapshot
        
        # Apply funding rates for perpetual futures (every 8 hours)
        self._apply_funding_rates(snapshot["ts"])
        
        # Visible sizes for quick lookup {price: size}
        book_sizes: Dict[float, float] = {p: q for p, q in snapshot["bids"]}
        book_sizes.update({p: q for p, q in snapshot["asks"]})

        still_live: List[Order] = []
        for order in self.active_orders:
            # --------------------------------------------------
            # 0) Stale-quote risk: if mid has moved such that our
            #    resting quote is now marketable, execute it IMMEDIATELY
            #    against the opposite touch; treat as TAKER.
            # --------------------------------------------------
            best_bid = snapshot["bids"][0][0]
            best_ask = snapshot["asks"][0][0]

            crossed = (
                (order.side == "buy" and order.price >= best_ask - 1e-9) or
                (order.side == "sell" and order.price <= best_bid + 1e-9)
            )

            if crossed and not order.is_filled():
                remaining_qty = order.size - order.filled
                if remaining_qty > 0:
                    trade_price = best_ask if order.side == "buy" else best_bid
                    self._settle_fill(
                        order.side,
                        remaining_qty,
                        trade_price,
                        snapshot["ts"],  # placed_ts == current_ts → taker
                        snapshot["ts"],
                        best_bid,
                        best_ask,
                        order_id=order.order_id,
                    )
                    order.filled = order.size  # fully filled

                    # Record in filled orders list for accounting consistency
                    if order not in self.filled_orders:
                        self.filled_orders.append(order)

            # After potential stale-quote fill, skip further processing if filled
            if order.is_filled():
                continue

            # --------------------------------------------------
            # 1) Queue-depletion logic (existing)
            # --------------------------------------------------
            prev_visible = order.queue_ahead + (order.size - order.filled)
            curr_visible = book_sizes.get(order.price, 0.0)
            # Net change in visible size at our price level
            change = curr_visible - prev_visible

            if change > 0:
                # New volume appeared ahead of us → we are pushed back in queue
                order.queue_ahead += change
                logger.debug(
                    "Queue size increased – competitors entered ahead",
                    order_id=order.order_id,
                    added_volume=change,
                    new_queue_ahead=order.queue_ahead,
                )

            delta = max(-change, 0.0)  # positive amount traded/cancelled ahead since last snapshot

            if delta > 0:
                # 1) Volume that hit the queue *ahead* of us only moves our position forward
                volume_ahead = min(delta, order.queue_ahead)
                order.queue_ahead -= volume_ahead

                # 2) Remaining delta volume interacts with our own order => executed qty
                executed_against_us = max(0.0, delta - volume_ahead)
                remaining_qty = order.size - order.filled

                if executed_against_us > 0 and remaining_qty > 0:
                    # Add queue position uncertainty (we don't know exactly where we are)
                    uncertainty_factor = 1.0 + random.uniform(-self.queue_position_uncertainty, self.queue_position_uncertainty)
                    adjusted_queue_ahead = max(0, order.queue_ahead * uncertainty_factor)
                    
                    # Calculate fill probability based on queue depth
                    # Deeper queue = lower probability of being filled
                    queue_depth_factor = max(0.1, 1.0 - (adjusted_queue_ahead / (order.size * 10)))
                    fill_probability = min(1.0, queue_depth_factor)

                    # Randomly decide if we get filled based on probability
                    if random.random() < fill_probability:
                        # Simulate partial fills
                        if random.random() < self.partial_fill_rate:
                            # Partial fill - only fill a portion of what's available
                            partial_fill_ratio = random.uniform(0.3, 0.8)
                            fill_qty = min(executed_against_us * partial_fill_ratio, remaining_qty)
                        else:
                            fill_qty = min(executed_against_us, remaining_qty)
                            
                        order.filled += fill_qty
                        order.queue_ahead = 0  # Reset queue position after fill

                        # Track fill event count only when actual qty was executed
                        if fill_qty > 0:
                            self.fills_count += 1

                        # Add execution slippage (price improvement/deterioration)
                        slippage_bps = random.gauss(0, self.execution_slippage_bps / 10000.0)
                        execution_price = order.price * (1 + slippage_bps)

                        self._settle_fill(
                            order.side,
                            fill_qty,
                            execution_price,  # Use slippage-adjusted price
                            order.placed_ts,
                            snapshot["ts"],
                            snapshot["bids"][0][0],
                            snapshot["asks"][0][0],
                            order_id=order.order_id,
                        )
                    else:
                        # Order was skipped - move queue position forward but don't fill
                        order.queue_ahead = max(0, order.queue_ahead - executed_against_us)
            # If queue ahead depleted, mark full fill via maker path
            if order.queue_ahead <= 0:
                fill_qty = order.size - order.filled
                if fill_qty > 0:
                    best_bid = snapshot["bids"][0][0]
                    best_ask = snapshot["asks"][0][0]
                    self._settle_fill(
                        order.side,
                        fill_qty,
                        order.price,
                        order.placed_ts,
                        snapshot["ts"],
                        best_bid,
                        best_ask,
                        order.order_id,
                    )
                order.filled = order.size
                # Add to filled orders list for validation
                if order not in self.filled_orders:
                    self.filled_orders.append(order)
            else:
                # Optional: cancel if too deep relative to best price
                best_bid = snapshot["bids"][0][0]
                best_ask = snapshot["asks"][0][0]
                if self.cancel_if_behind:
                    drift_ticks = (
                        (best_bid - order.price) / self.tick
                        if order.side == "buy"
                        else (order.price - best_ask) / self.tick
                    )
                    if drift_ticks > self.max_ticks_away:
                        fill_frac = order.filled / order.size if order.size > 0 else 1.0
                        # Cancel only if little of our order already filled (to simulate being stuck when mostly through queue)
                        if fill_frac < self.min_keep_filled_frac:
                            logger.info("Canceling stale order", 
                                       order_id=order.order_id,
                                       side=order.side,
                                       price=order.price,
                                       drift_ticks=drift_ticks,
                                       max_ticks_away=self.max_ticks_away,
                                       fill_fraction=fill_frac,
                                       event=logger.OrderEvent.CANCELED)
                            continue  # drop order
                still_live.append(order)

        self.active_orders = still_live
        self._validate_state(snapshot)

        # After processing normal queue logic, enforce inventory limit via taker trades if necessary
        self._liquidate_if_needed(snapshot)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_state(self, snapshot: dict) -> None:
        """Validate the internal state of the book simulator."""
        try:
            # 1. Validate inventory consistency
            expected_inventory = sum(o.filled for o in self.filled_orders if o.side == "buy") - \
                                sum(o.filled for o in self.filled_orders if o.side == "sell")
            if abs(self.inventory - expected_inventory) > 1e-9:
                logger.error("Inventory inconsistency detected", 
                            actual_inventory=self.inventory,
                            expected_inventory=expected_inventory)
                
            # 2. Validate no duplicate orders
            order_ids = [o.order_id for o in self.active_orders]
            if len(order_ids) != len(set(order_ids)):
                logger.error("Duplicate order IDs detected", order_ids=order_ids)
                
            # 3. Validate queue positions are non-negative
            for order in self.active_orders:
                if order.queue_ahead < 0:
                    logger.error("Negative queue position detected", 
                                order_id=order.order_id,
                                queue_ahead=order.queue_ahead)
                    
            # 4. Validate filled amounts don't exceed order size
            for order in self.active_orders + self.filled_orders:
                if order.filled > order.size + 1e-9:
                    logger.error("Order filled more than size", 
                                order_id=order.order_id,
                                filled=order.filled,
                                size=order.size)
                    
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _submit_order(self, side: str, price: float, size: float, snapshot: dict) -> Order:
        """Create order with latency. If latency <= 0, activate immediately."""
        
        # Check for maintenance windows and network congestion
        if random.random() < self.maintenance_window_probability:
            logger.warning("Order rejected due to maintenance window", 
                          side=side, price=price, size=size,
                          event=logger.OrderEvent.REJECTED)
            return Order(side, price, 0.0, 0.0, snapshot["ts"], 0.0, "MAINTENANCE")
            
        if random.random() < self.network_congestion_probability:
            logger.warning("Order rejected due to network congestion", 
                          side=side, price=price, size=size,
                          event=logger.OrderEvent.REJECTED)
            return Order(side, price, 0.0, 0.0, snapshot["ts"], 0.0, "CONGESTION")
        
        # Check API rate limits
        current_ts = snapshot["ts"]
        
        # Reset second counter if needed
        if self.last_second_ts is None or current_ts - self.last_second_ts >= 1.0:
            self.orders_this_second = 0
            self.last_second_ts = current_ts
            
        # Reset 10-minute counter if needed
        if self.last_10min_ts is None or current_ts - self.last_10min_ts >= 600.0:
            self.orders_this_10min = 0
            self.last_10min_ts = current_ts
            
        # Check rate limits
        if self.orders_this_second >= self.api_rate_limit_orders_per_sec:
            logger.warning("Order rejected due to per-second rate limit", 
                          side=side, price=price, size=size,
                          event=logger.OrderEvent.REJECTED)
            return Order(side, price, 0.0, 0.0, snapshot["ts"], 0.0, "RATE_LIMIT_SEC")
            
        if self.orders_this_10min >= self.api_rate_limit_orders_per_10min:
            logger.warning("Order rejected due to 10-minute rate limit", 
                          side=side, price=price, size=size,
                          event=logger.OrderEvent.REJECTED)
            return Order(side, price, 0.0, 0.0, snapshot["ts"], 0.0, "RATE_LIMIT_10MIN")
        
        # Increment counters
        self.orders_this_second += 1
        self.orders_this_10min += 1
        
        # Simulate order rejection (exchange infrastructure issues, invalid orders, etc.)
        if random.random() < self.order_rejection_rate:
            logger.warning("Order rejected by exchange", 
                          side=side, price=price, size=size,
                          event=logger.OrderEvent.REJECTED)
            # Return a dummy order that will be ignored
            return Order(side, price, 0.0, 0.0, snapshot["ts"], 0.0, "REJECTED")
        
        delay = max(0.0, random.gauss(self.lat_mean, self.lat_std))
        if delay < 1e-6:
            # Immediate activation
            order = self._add_order(side, price, size, snapshot)
            logger.info("Order placed and activated immediately", 
                       order_id=order.order_id,
                       side=side,
                       price=price,
                       size=size,
                       queue_ahead=order.queue_ahead,
                       event=logger.OrderEvent.PLACED)
            return order
            
        activate_ts = snapshot["ts"] + delay
        order_id = str(uuid.uuid4())[:8]
        self.pending_orders.append({
            "side": side,
            "price": price,
            "size": size,
            "activate_ts": activate_ts,
            "order_id": order_id,
        })
        logger.info("Order placed with latency", 
                   order_id=order_id,
                   side=side,
                   price=price,
                   size=size,
                   delay_ms=delay*1000,
                   activate_ts=activate_ts,
                   event=logger.OrderEvent.PLACED)
                   
        # Return a placeholder Order object with zero size so caller can ignore
        return Order(side, price, 0.0, 0.0, snapshot["ts"], 0.0, order_id)

    def _activate_pending(self, snapshot: dict):
        now = snapshot["ts"]
        still_pending = []
        for p in self.pending_orders:
            if p["activate_ts"] <= now:
                side = p["side"]
                price = p["price"]
                size = p["size"]
                order_id = p["order_id"]

                best_bid = snapshot["bids"][0][0]
                best_ask = snapshot["asks"][0][0]

                # If price crosses, treat as immediate taker fill (latency slippage)
                crossed = (side == "buy" and price >= best_ask - 1e-9) or (
                    side == "sell" and price <= best_bid + 1e-9
                )

                if crossed:
                    trade_price = best_ask if side == "buy" else best_bid
                    self._settle_fill(
                        side,
                        size,
                        trade_price,
                        snapshot["ts"],  # placed == arrival, taker
                        snapshot["ts"],
                        best_bid,
                        best_ask,
                        order_id=order_id,
                    )

                    # Record synthetic filled order for validation
                    synthetic = Order(
                        side=side,
                        price=trade_price,
                        size=size,
                        queue_ahead=0.0,
                        placed_ts=snapshot["ts"],
                        filled=size,
                        order_id=order_id,
                    )
                    self.filled_orders.append(synthetic)

                    logger.info(
                        "Latency slippage – order crossed spread and executed as taker",
                        order_id=order_id,
                        side=side,
                        sent_price=price,
                        executed_price=trade_price,
                        size=size,
                        event=logger.OrderEvent.FORCED_LIQ,
                    )
                else:
                    order = self._add_order(side, price, size, snapshot, order_id)
                    logger.info(
                        "Pending order activated",
                        order_id=order.order_id,
                        side=order.side,
                        price=order.price,
                        size=order.size,
                        queue_ahead=order.queue_ahead,
                        event=logger.OrderEvent.ACTIVATED,
                    )
            else:
                still_pending.append(p)
        self.pending_orders = still_pending

    def _add_order(self, side: str, price: float, size: float, snapshot: dict, order_id: str = None) -> Order:
        # Visible size at this price level right now (before we add ourselves)
        visible = 0.0
        levels = snapshot["bids"] if side == "buy" else snapshot["asks"]
        for p, q in levels:
            if abs(p - price) < 1e-9:
                visible = q
                break
        
        # Calculate competing orders that will be placed ahead of us
        competing_size = self._calculate_competing_orders(price, size)
        
        # Our queue position = visible size + competing orders
        queue_ahead = visible + competing_size
                
        if order_id is None:
            order_id = str(uuid.uuid4())[:8]
            
        order = Order(
            side=side,
            price=price,
            size=size,
            queue_ahead=queue_ahead,
            placed_ts=snapshot["ts"],
            order_id=order_id,
        )
        self.active_orders.append(order)
        return order

    def _settle_fill(
        self,
        order_side: str,
        qty: float,
        price: float,
        placed_ts: float,
        current_ts: float,
        best_bid: float,
        best_ask: float,
        order_id: str,
    ):  # noqa: E501
        # Track total number of fills (each fill event counts once regardless of qty)
        self.fills_count += 1
        notional = qty * price
        # Determine maker vs taker
        is_taker = abs(current_ts - placed_ts) < 1e-9

        # Base cash/inventory movement
        if order_side == "buy":
            self.inventory += qty
            self.cash -= notional
            logger.info("Buy order filled", 
                       order_id=order_id,
                       price=price,
                       qty=qty,
                       notional=notional,
                       is_taker=is_taker,
                       new_inventory=self.inventory,
                       event=logger.OrderEvent.FILLED)
        else:
            self.inventory -= qty
            self.cash += notional
            logger.info("Sell order filled", 
                       order_id=order_id,
                       price=price,
                       qty=qty,
                       notional=notional,
                       is_taker=is_taker,
                       new_inventory=self.inventory,
                       event=logger.OrderEvent.FILLED)

        # Apply fees/rebates
        if is_taker:
            fee = notional * self.taker_fee
            self.cash -= fee
            self.pnl_components["fees"] -= fee
            logger.debug("Taker fee applied", 
                        order_id=order_id,
                        fee=fee,
                        event=logger.PnLEvent.FEE)
        else:
            # Maker rebate
            rebate = notional * self.maker_rebate
            self.cash += rebate
            self.pnl_components["rebates"] += rebate
            logger.debug("Maker rebate applied", 
                        order_id=order_id,
                        rebate=rebate,
                        event=logger.PnLEvent.REBATE)
            
            # Note: We removed immediate adverse selection marking
            # Real market makers hold positions and manage inventory over time
            # Adverse selection only applies when forced to liquidate

    # ------------------------------------------------------------------
    # Inventory liquidation helper
    # ------------------------------------------------------------------
    def _liquidate_if_needed(self, snapshot: dict):
        """If inventory exceeds ±max_inventory, aggressively trade at the touch to flatten."""
        if self.max_inventory <= 0:
            return

        # Liquidate in chunks (at most max_inventory per slice) so we don't move the full
        # position in one trade – mimics splitting market-orders.
        while abs(self.inventory) > self.max_inventory + 1e-9:
            # Excess amount to bring us back inside the band
            excess = abs(self.inventory) - self.max_inventory
            qty = min(excess, self.max_inventory)  # chunk size

            side = "sell" if self.inventory > 0 else "buy"

            best_bid = snapshot["bids"][0][0]
            best_ask = snapshot["asks"][0][0]
            price = best_bid if side == "sell" else best_ask

            # Unique ID to avoid duplicates in validation
            liq_id = "LIQ" + str(uuid.uuid4())[:6]

            # Taker trade (placed_ts == current_ts)
            self._settle_fill(
                side,
                qty,
                price,
                snapshot["ts"],
                snapshot["ts"],
                best_bid,
                best_ask,
                order_id=liq_id,
            )

            # Apply adverse selection cost for forced liquidation
            # This represents the cost of being forced to trade when we don't want to
            if side == "sell":  # We're selling, so adverse if price goes up
                adverse_cost = (best_ask - best_bid) * qty * 0.5  # Half spread as adverse selection
            else:  # We're buying, so adverse if price goes down
                adverse_cost = (best_ask - best_bid) * qty * 0.5  # Half spread as adverse selection
            
            self.cash -= adverse_cost
            self.pnl_components["adverse_selection"] -= adverse_cost
            logger.debug("Forced liquidation adverse selection", 
                        side=side,
                        qty=qty,
                        adverse_cost=adverse_cost,
                        event=logger.PnLEvent.ADVERSE_SELECTION)

            # Record synthetic filled order so validation sees it
            synthetic = Order(
                side=side,
                price=price,
                size=qty,
                queue_ahead=0.0,
                placed_ts=snapshot["ts"],
                filled=qty,
                order_id=liq_id,
            )
            self.filled_orders.append(synthetic)

            logger.info(
                "Forced liquidation executed",
                side=side,
                qty=qty,
                price=price,
                remaining_inventory=self.inventory,
                event=logger.OrderEvent.FORCED_LIQ,
            )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def mark_to_market(self, mid_price: float) -> float:
        """Calculate mark-to-market P&L and update inventory value component."""
        mtm = self.cash + self.inventory * mid_price
        logger.debug("Mark-to-market update", 
                    cash=self.cash,
                    inventory=self.inventory,
                    mid_price=mid_price,
                    mtm_value=mtm,
                    event=logger.PnLEvent.MARK_TO_MARKET)
        return mtm
        
    def get_pnl_attribution(self) -> Dict[str, float]:
        """Return detailed P&L attribution by component."""
        return self.pnl_components

    # ------------------------------------------------------------------
    # Competition modeling
    # ------------------------------------------------------------------
    def _calculate_competing_orders(self, price_level: float, our_order_size: float) -> float:
        """Calculate the size of competing orders that will be placed ahead of us in the queue.
        
        This models other market makers competing for the same opportunities.
        Competition intensity varies based on market conditions and our order size.
        """
        # Base number of competitors based on competition intensity
        base_competitors = self.min_competitors + int(
            (self.max_competitors - self.min_competitors) * self.competition_intensity
        )
        
        # Add some randomness to make it more realistic
        num_competitors = random.randint(
            max(0, base_competitors - 1), 
            min(self.max_competitors, base_competitors + 1)
        )
        
        # Calculate total competing order size
        # Larger orders attract more competition
        size_multiplier = 1.0 + (our_order_size / self.competitor_order_size) * 0.5
        competing_size = num_competitors * self.competitor_order_size * size_multiplier
        
        # Add some randomness to competing order sizes
        competing_size *= random.uniform(0.8, 1.2)
        
        logger.debug("Competing orders calculated", 
                    price_level=price_level,
                    our_order_size=our_order_size,
                    num_competitors=num_competitors,
                    competing_size=competing_size,
                    competition_intensity=self.competition_intensity)
        
        return competing_size

    def _apply_funding_rates(self, current_ts: float):
        """Apply funding rates for perpetual futures (every 8 hours) based on current inventory position."""
        if self.last_funding_ts is None:
            # First time - initialize and don't apply funding yet
            self.last_funding_ts = current_ts
            return
            
        if current_ts - self.last_funding_ts >= self.funding_interval:
            # Calculate funding rate based on current inventory position
            funding_rate = random.gauss(self.funding_rate_mean, self.funding_rate_std)
            
            # Apply funding rate to inventory (simplified calculation)
            funding_cost = self.inventory * funding_rate
            self.inventory += funding_cost
            
            # Track funding cost in P&L
            self.pnl_components["funding_rates"] -= abs(funding_cost)
            
            # Update last funding timestamp
            self.last_funding_ts = current_ts
            
            logger.debug("Funding rates applied", 
                        funding_rate=funding_rate,
                        funding_cost=funding_cost,
                        new_inventory=self.inventory,
                        event=logger.PnLEvent.FUNDING_RATE)
