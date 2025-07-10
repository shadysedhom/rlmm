#!/usr/bin/env python3
"""Simple playback engine that streams historical snapshots to a trading strategy
and accounts for naive fills.

Example Entry:
[5000]   ts=1673303913.792  inv=-0.5050  PnL=-0.21  active=2
  │        │                 │            │          └ how many live orders
  │        │                 │            └ mark-to-market P&L in USDT
  │        │                 └ signed inventory in BTC (+ long, – short)
  │        └ epoch-seconds timestamp of that snapshot
  └ snapshot counter (printed every 5 000)

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
import gzip, json, time, importlib, sys, os
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional
import statistics
import logging
import math
import numpy as np
from collections import defaultdict

# Ensure project root (two levels up) is on import path when executed directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine import logger
from engine.book import BookSimulator


class SnapshotLoader:
    """Yield snapshots dicts sorted by timestamp from all files in a directory."""

    def __init__(self, data_dir: Path):
        self.files: List[Path] = sorted(data_dir.glob("*.jsonl.gz"))
        if not self.files:
            raise FileNotFoundError(f"No *.jsonl.gz files found in {data_dir}")
        logger.info("SnapshotLoader initialized", file_count=len(self.files), files=[f.name for f in self.files])

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        for fp in self.files:
            logger.info(f"Processing file", file=fp.name)
            with gzip.open(fp, "rt") as gz:
                for line in gz:
                    snap = json.loads(line)
                    # Attach DepthLadder helper for richer analytics (non-breaking addition)
                    try:
                        from engine.depth import DepthLadder  # local import to avoid cycle if depth imports playback
                        snap["depth"] = DepthLadder(bids=snap["bids"], asks=snap["asks"], ts=snap["ts"])
                    except Exception as _e:
                        # Fallback gracefully if structure mismatched
                        pass
                    yield snap


def load_strategy(name: str):
    """Dynamically import strategies.<n>.Strategy class."""
    try:
        module = importlib.import_module(f"strategies.{name}")
        logger.info(f"Strategy loaded", strategy=name)
        return module.Strategy  # type: ignore[attr-defined]
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load strategy", strategy=name, error=str(e))
        raise


def validate_snapshot(snap: Dict[str, Any], prev_snap: Optional[Dict[str, Any]] = None) -> bool:
    """Validate snapshot data integrity."""
    try:
        # Basic structure checks
        assert "ts" in snap, "Missing timestamp"
        assert "bids" in snap, "Missing bids"
        assert "asks" in snap, "Missing asks"
        
        # Check bid/ask arrays
        assert len(snap["bids"]) > 0, "Empty bids"
        assert len(snap["asks"]) > 0, "Empty asks"
        
        # Check price ordering
        bid_prices = [p for p, _ in snap["bids"]]
        ask_prices = [p for p, _ in snap["asks"]]
        
        assert all(bid_prices[i] >= bid_prices[i+1] for i in range(len(bid_prices)-1)), "Bids not in descending order"
        assert all(ask_prices[i] <= ask_prices[i+1] for i in range(len(ask_prices)-1)), "Asks not in ascending order"
        
        # Check for crossed book
        assert bid_prices[0] < ask_prices[0], "Crossed book detected"
        
        # Check for reasonable timestamp if we have previous snapshot
        if prev_snap:
            assert snap["ts"] >= prev_snap["ts"], "Timestamp regression"
            
        return True
    except AssertionError as e:
        logger.error("Snapshot validation failed", error=str(e), snapshot=snap)
        return False


def main():
    parser = argparse.ArgumentParser(description="Playback historical order-book snapshots")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--strategy", default="mid_spread", help="Strategy module under strategies/")
    parser.add_argument("--speed", type=float, default=0.0, help="0=as fast as possible; else realtime divisor (e.g., 10=10x faster)")
    parser.add_argument("--print_every", type=int, default=1000, help="How many snapshots between P&L prints")
    parser.add_argument("--hold_time", type=float, default=0.5, help="Minimum time (seconds) a quote must rest before it can be replaced")
    parser.add_argument("--latency_mean_ms", type=float, default=100.0, help="Mean network/order latency in milliseconds")
    parser.add_argument("--latency_std_ms", type=float, default=30.0, help="Std dev of latency in ms")
    parser.add_argument("--maker_fee", type=float, default=0.0002, help="Maker fee rate (positive cost, negative rebate)")
    parser.add_argument("--taker_fee", type=float, default=0.0005, help="Taker fee rate (e.g., 0.0005 for 5 bps)")
    parser.add_argument("--max_inventory", type=float, default=0.01, help="Absolute BTC inventory threshold before forced taker liquidation")
    parser.add_argument("--capital_base", type=float, default=10000.0, help="Nominal capital (USDT) to normalise P&L for return calculations")
    parser.add_argument("--sharpe_horizon", type=float, default=60.0, help="Seconds per return bucket for Sharpe ratio (e.g., 60 = 1-minute Sharpe)")
    parser.add_argument("--strategy_kwargs", type=str, default="", help="Comma-separated strategy kwargs, e.g. 'gamma=0.05,size=0.002'")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log_file", type=str, help="Log to file (in addition to console)")
    parser.add_argument("--json_logs", action="store_true", help="Output logs in JSON format")
    parser.add_argument("--output_dir", type=str, help="Directory to save results and metrics")
    parser.add_argument("--max_snapshots", type=int, default=0, help="Maximum number of snapshots to process (0 = no limit)")
    parser.add_argument("--log_throttle", type=int, default=0, help="Only log every N snapshots to reduce verbosity (0 = log all snapshots)")
    parser.add_argument("--print_interval", type=int, default=5000, help="Print summary every N snapshots (quant-friendly output)")
    parser.add_argument("--order_ttl", type=float, default=0.8, help="Maximum lifetime in seconds for a resting quote before cancel (0 = disabled)")
    # Competition parameters
    parser.add_argument("--competition_intensity", type=float, default=0.9, help="Competition intensity (0.0 = none, 1.0 = high)")
    parser.add_argument("--min_competitors", type=int, default=3, help="Minimum competing orders per price level")
    parser.add_argument("--max_competitors", type=int, default=8, help="Maximum competing orders per price level")
    parser.add_argument("--competitor_order_size", type=float, default=0.05, help="Average size of competing orders (BTC)")
    parser.add_argument("--funding_interval", type=float, default=8.0, help="Funding interval in hours (default 8). Use smaller for short simulations")
    # --- new hedging parameters ---
    parser.add_argument("--hedge_fraction", type=float, default=1.0, help="Fraction of inventory to hedge when taking out inventory (0-1)")
    parser.add_argument("--inventory_band", type=float, default=0.0, help="No hedging while |inventory| is below this value (BTC)")
    parser.add_argument("--momentum_filter", action="store_true", help="Enable momentum filter to skip hedge when price moves in our favour")
    parser.add_argument("--momentum_lookback", type=int, default=1, help="Number of snapshots to look back for momentum test (>=1)")
    parser.add_argument("--max_ticks_away", type=int, default=3, help="Cancel/replace order if it drifts more than this many ticks behind the touch")
    parser.add_argument("--no_cancel_if_behind", action="store_true", help="Disable automatic cancellation of orders that drift behind touch")
    parser.add_argument("--post_only", action="store_true", help="Enable post-only (protect-on-cross) logic; orders that would cross are canceled")
    # Sharpe denominator selection
    parser.add_argument("--sharpe_denominator", type=str, default="capital", choices=["capital", "max_inventory", "notional"], help="What to divide returns by when computing Sharpe: capital (capital_base), max_inventory (max_inventory*mid), or notional (absolute inventory notional per bucket)")
    args = parser.parse_args()
    
    # Set up logging
    logger.setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        json_format=args.json_logs,
        console=True
    )
    
    logger.info("Playback started", args=vars(args))

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Output directory created", output_dir=args.output_dir)

    loader = SnapshotLoader(args.data_dir)

    # Detect tick size from first snapshot
    first_snap = next(iter(loader))
    bid_prices = [p for p, _ in first_snap["bids"][:5]]
    tick = min(round(bid_prices[i] - bid_prices[i + 1], 8) for i in range(len(bid_prices) - 1))
    logger.info("Tick size detected", tick_size=tick)

    # Reset loader iterator (simple: recreate)
    loader = SnapshotLoader(args.data_dir)

    # Parse strategy keyword arguments
    kw: Dict[str, Any] = {}
    if args.strategy_kwargs:
        for kv in args.strategy_kwargs.split(","):
            if not kv:
                continue
            key, val = kv.split("=")
            # Naively cast to float if possible, else leave as str
            try:
                kw[key] = float(val)
            except ValueError:
                kw[key] = val

    try:
        StratCls = load_strategy(args.strategy)
        strat = StratCls(**kw)
        logger.info("Strategy instantiated", strategy=args.strategy, kwargs=kw)
    except Exception as e:
        logger.critical("Failed to instantiate strategy", error=str(e))
        return

    book = BookSimulator(
        tick_size=tick,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        latency_mean_ms=args.latency_mean_ms,
        latency_std_ms=args.latency_std_ms,
        max_inventory=args.max_inventory,
        min_competitors=args.min_competitors,
        max_competitors=args.max_competitors,
        competitor_order_size=args.competitor_order_size,
        competition_intensity=args.competition_intensity,
        cancel_if_behind=not args.no_cancel_if_behind,
        post_only=args.post_only,
        max_ticks_away=args.max_ticks_away,
        capital_base=args.capital_base,
        order_ttl_sec=args.order_ttl if args.order_ttl > 0 else None,
        # Funding parameters
        # Convert hours to seconds so we can tweak in short backtests
        funding_interval_sec=int(args.funding_interval * 3600),
        # Hedging parameters
        hedge_fraction=args.hedge_fraction,
        inventory_band=args.inventory_band,
        momentum_filter=args.momentum_filter,
        momentum_lookback=args.momentum_lookback,
    )

    prev_ts: float | None = None
    prev_snap: Dict[str, Any] | None = None
    last_update_ts: float | None = None  # when quotes were last replaced
    pnl_series: List[float] = []
    inv_series: List[float] = []
    ts_series: List[float] = []
    
    # Track metrics for each snapshot
    metrics_log = []
    
    # Throttle logging to prevent console overflow
    log_throttle_counter = 0

    for i, snap in enumerate(loader, start=1):
        # Check if we've reached the maximum number of snapshots
        if args.max_snapshots > 0 and i > args.max_snapshots:
            logger.info(f"Reached maximum number of snapshots ({args.max_snapshots}), stopping")
            break
            
        # Determine if we should log for this snapshot
        should_log = (args.log_throttle == 0) or (log_throttle_counter % args.log_throttle == 0)
        log_throttle_counter += 1
        
        # Temporarily adjust log level if throttling
        if not should_log:
            original_level = logger.logger.level  # Access the actual logger's level
            logger.logger.setLevel(logging.WARNING)  # Only log warnings and above
        
        # Validate snapshot
        if not validate_snapshot(snap, prev_snap):
            logger.warning("Skipping invalid snapshot", snapshot_idx=i)
            continue
            
        prev_snap = snap
        ts = snap["ts"]
        mid = (snap["bids"][0][0] + snap["asks"][0][0]) / 2.0
        snap["mid"] = mid  # inject mid for adverse-selection logic

        # ---------------- Funding countdown ----------------
        try:
            snap["t_to_funding"] = book.time_to_next_funding(ts)
        except AttributeError:
            # Backward-compat: older BookSimulator without this helper
            pass

        # Optional real-time throttling
        if args.speed > 0 and prev_ts is not None:
            delay = (ts - prev_ts) / args.speed
            if delay > 0:
                time.sleep(delay)
        prev_ts = ts

        # Always advance simulator first (fills existing quotes)
        book.update(snap)

        # Decide whether we are allowed to refresh quotes
        if (last_update_ts is None) or (ts - last_update_ts >= args.hold_time):
            try:
                quote = strat.on_snapshot(snap, book.inventory)
                logger.debug("Strategy generated quote", 
                           bid_price=quote["bid_price"], 
                           ask_price=quote["ask_price"], 
                           size=quote["size"])
                
                # Validate quote
                if quote["bid_price"] >= quote["ask_price"]:
                    logger.error("Invalid quote: bid >= ask", bid=quote["bid_price"], ask=quote["ask_price"])
                else:
                    book.place_quotes(snap, quote["bid_price"], quote["ask_price"], quote["size"])
                    last_update_ts = ts
            except Exception as e:
                logger.error("Strategy error", error=str(e))

        # Mark-to-market after update; record each snapshot for accurate stats
        portfolio_value = book.mark_to_market(mid)
        pnl_current = portfolio_value - args.capital_base  # Calculate actual P&L
        pnl_series.append(portfolio_value)  # Store absolute portfolio value for drawdown
        inv_series.append(book.inventory)
        ts_series.append(ts)
        
        # Record metrics for this snapshot
        metrics_log.append({
            "snapshot": i,
            "ts": ts,
            "mid": mid,
            "inventory": book.inventory,
            "pnl": pnl_current,  # Store actual P&L
            "portfolio_value": portfolio_value,  # Store absolute value
            "active_orders": len(book.active_orders),
            "fill_count": book.fills_count
        })

        # Print summary at regular intervals (quant-friendly output)
        if i % args.print_interval == 0:
            logger.info(
                f"[{i}] ts={ts:.3f} inv={book.inventory:.4f} PnL={pnl_current:.2f} active={len(book.active_orders)} fills={book.fills_count}"
            )
            
        # Print more detailed info if requested via print_every
        if i % args.print_every == 0 and args.print_every != args.print_interval:
            logger.debug(
                f"Snapshot {i}: ts={ts:.3f}, mid={mid:.2f}, inventory={book.inventory:.6f}, PnL={pnl_current:.4f}"
            )
            
        # Restore original log level if throttling
        if not should_log:
            logger.logger.setLevel(original_level)

    # --------------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------------
    if pnl_series:
        import math
        import numpy as np
        from collections import defaultdict

        if ts_series and len(ts_series) > 1:
            logger.info("Computing performance metrics...")
            
            # Calculate total P&L and final inventory
            final_portfolio_value = pnl_series[-1] if pnl_series else args.capital_base
            final_pnl = final_portfolio_value - args.capital_base  # Calculate actual P&L
            final_inv = inv_series[-1] if inv_series else 0.0
            
            logger.info(f"Final portfolio value: {final_portfolio_value:.4f} USDT")
            logger.info(f"Final P&L: {final_pnl:.4f} USDT")
            logger.info(f"Final inventory: {final_inv:.6f} BTC")
            logger.info(f"Total fills: {book.fills_count}")
        
            # Calculate max drawdown - now working with absolute portfolio values
            max_drawdown = 0.0
            peak = args.capital_base  # Start from initial capital
            for pnl in pnl_series:
                # pnl is now absolute portfolio value (cash + inventory * mid_price)
                current_value = pnl  # No need to add capital_base again
                if current_value > peak:
                    peak = current_value
                drawdown = peak - current_value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            logger.info(f"Maximum drawdown: {max_drawdown:.4f} USDT ({max_drawdown/args.capital_base*100:.4f}% of capital)")
        
        # ------------------ Sharpe Ratio (Industry Standard) ------------------
        # Industry standard: Only annualize if we have sufficient data (>30 days)
        # Otherwise report non-annualized Sharpe ratio
        if len(pnl_series) > 10 and len(ts_series) > 1:  # Need minimum data points
            # Calculate simple returns from portfolio values
            portfolio_returns = np.diff(pnl_series) / args.capital_base
            
            if len(portfolio_returns) > 1 and np.std(portfolio_returns) > 0:
                mean_return = np.mean(portfolio_returns)
                std_return = np.std(portfolio_returns)
                risk_free_rate = 0.0  # Assume 0% risk-free rate
                
                # Calculate actual simulation duration
                simulation_duration_seconds = ts_series[-1] - ts_series[0]
                simulation_duration_days = simulation_duration_seconds / (24 * 3600)
                
                # Industry standard: Only annualize if we have >30 days of data
                if simulation_duration_days >= 30:
                    # Standard annualization for sufficient data
                    returns_per_year = len(portfolio_returns) / (simulation_duration_seconds / (365.25 * 24 * 3600))
                    annualized_volatility = std_return * math.sqrt(returns_per_year)
                    
                    total_return = final_pnl / args.capital_base
                    annualized_return = total_return / (simulation_duration_days / 365.25)
                    
                    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
                    
                    logger.info(f"Sharpe ratio (annualized): {sharpe:.4f}")
                    logger.info(f"Simulation duration: {simulation_duration_days:.1f} days")
                    logger.info(f"Annualized return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
                    logger.info(f"Annualized volatility: {annualized_volatility:.4f} ({annualized_volatility*100:.2f}%)")
                else:
                    # Non-annualized Sharpe for short periods (industry standard)
                    sharpe = (mean_return - risk_free_rate) / std_return
                    total_return = final_pnl / args.capital_base
                    
                    logger.info(f"Sharpe ratio (non-annualized): {sharpe:.4f}")
                    logger.info(f"Simulation duration: {simulation_duration_days:.1f} days (too short for annualization)")
                    logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%)")
                    logger.info(f"Return volatility: {std_return:.6f} ({std_return*100:.4f}%)")
                    logger.info(f"Note: Industry standard requires >30 days for meaningful annualized Sharpe")
            else:
                logger.info("Sharpe ratio: undefined (insufficient return variation)")
        else:
            logger.info("Sharpe ratio: undefined (insufficient data points)")
        
        # Get P&L attribution
        pnl_attribution = book.get_pnl_attribution()
        for component, value in pnl_attribution.items():
            logger.info(f"P&L {component}: {value:.4f} USDT")
        
        # Save metrics if output directory specified
        if args.output_dir:
            metrics_file = os.path.join(args.output_dir, f"{args.strategy}_metrics.json")
            
            # Create metrics dictionary with all calculated values
            additional_metrics = book.get_additional_metrics()
            metrics_dict = {
                "final_pnl": final_pnl,
                "final_inventory": final_inv,
                "fill_count": book.fills_count,
                "max_drawdown": max_drawdown,
                "pnl_attribution": pnl_attribution,
                "additional_metrics": additional_metrics,
                "strategy": args.strategy,
                "strategy_kwargs": kw,
                "snapshots_processed": len(pnl_series)
            }
            
            # Add Sharpe ratio if calculated
            if 'sharpe' in locals():
                metrics_dict["sharpe_ratio"] = sharpe
                metrics_dict["capital_base"] = args.capital_base
                metrics_dict["sharpe_horizon_seconds"] = args.sharpe_horizon
            
            with open(metrics_file, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")
            
            # Save detailed log if requested
            if args.log_file:
                logger.info(f"Detailed log saved to {args.log_file}")

        # Print additional metrics for fill rate, latency, queue, etc.
        additional_metrics = book.get_additional_metrics()
        logger.info("--- Additional Model Metrics ---")
        for k, v in additional_metrics.items():
            logger.info(f"{k}: {v}")


if __name__ == "__main__":
    main() 