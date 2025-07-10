#!/usr/bin/env python3
"""
Run the market-making simulator with appropriate parameters.

This script provides a convenient interface for running the simulator with
different parameters and logging configurations.
"""
import argparse
import subprocess
import os
import sys
import json
from pathlib import Path

# Optional YAML support
try:
    import yaml  # type: ignore
except ImportError:  # Fallback when PyYAML not installed
    yaml = None

def main():
    parser = argparse.ArgumentParser(description="Run market-making simulator with appropriate parameters")
    parser.add_argument("--strategy", default="skew_mid_spread", help="Strategy to use (mid_spread or skew_mid_spread)")
    parser.add_argument("--max_snapshots", type=int, default=10000, help="Maximum number of snapshots to process")
    parser.add_argument("--log_level", default="INFO", help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log_throttle", type=int, default=10, help="Only log every N snapshots to reduce verbosity (0 = log all snapshots)")
    parser.add_argument("--print_interval", type=int, default=5000, help="Print summary every N snapshots (quant-friendly output)")
    parser.add_argument("--output_dir", default="./results", help="Directory to save results")
    parser.add_argument("--log_file", help="Path to log file (optional)")
    parser.add_argument("--json_logs", action="store_true", help="Output logs in JSON format")
    parser.add_argument("--strategy_kwargs", default="", help="Strategy parameters (e.g. 'gamma=0.1,size=0.001')")
    parser.add_argument("--config", help="Path to YAML or JSON config file with parameters")
    parser.add_argument("--data_dir", default="data/converted/binance", help="Directory containing order book data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (sets log_throttle=0)")
    parser.add_argument("--quiet", action="store_true", help="Run silently and only show the final summary")
    parser.add_argument("--order_ttl", type=float, default=0.8, help="Maximum lifetime of a resting quote in seconds (0 to disable)")
    parser.add_argument("--max_inventory", type=float, default=0.01, help="Absolute BTC inventory limit before forced liquidation")
    parser.add_argument("--latency_mean_ms", type=float, default=100.0, help="Mean network/order latency in milliseconds")
    parser.add_argument("--latency_std_ms", type=float, default=30.0, help="Std-dev of latency in milliseconds")
    parser.add_argument("--competition_intensity", type=float, default=0.9, help="Queue competition intensity (0 none, 1 high)")
    parser.add_argument("--min_competitors", type=int, default=3, help="Minimum competing orders per price level")
    parser.add_argument("--max_competitors", type=int, default=8, help="Maximum competing orders per price level")
    parser.add_argument("--competitor_order_size", type=float, default=0.05, help="Average size (BTC) of competing orders")
    parser.add_argument("--funding_interval", type=float, default=8.0, help="Funding interval in hours (perpetual futures)")
    parser.add_argument("--maker_fee", type=float, default=0.0002, help="Maker fee rate (positive cost, negative rebate)")
    parser.add_argument("--taker_fee", type=float, default=0.0005, help="Taker fee rate in decimal (e.g., 0.0005 = 5 bps)")
    parser.add_argument("--hedge_fraction", type=float, default=1.0, help="Fraction of inventory to hedge when taking out inventory (0-1, 1=full)")
    parser.add_argument("--inventory_band", type=float, default=0.0, help="Skip hedging while |inventory| ≤ band (BTC)")
    parser.add_argument("--momentum_filter", action="store_true", help="Enable momentum filter to skip hedging when price moves favourably")
    parser.add_argument("--momentum_lookback", type=int, default=1, help="Number of snapshots to look back for momentum check (>=1)")
    parser.add_argument("--max_ticks_away", type=int, default=3, help="Cancel order if it drifts more than N ticks behind best bid/ask")
    parser.add_argument("--no_cancel_if_behind", action="store_true", help="Disable cancel_if_behind logic in playback")
    parser.add_argument("--post_only", action="store_true", help="Enable post-only protect-on-cross in playback")
    parser.add_argument("--hold_time", type=float, default=0.5, help="Minimum seconds a quote must rest before it can be replaced")
    parser.add_argument("--sharpe_denominator", type=str, default="capital", choices=["capital", "max_inventory", "notional"], help="Denominator for Sharpe calc (capital, max_inventory, notional)")
    parser.add_argument("--sharpe_horizon", type=float, default=60.0, help="Seconds per return bucket for Sharpe ratio (e.g., 3600 = 1-hour Sharpe)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default log file if not provided
    if not args.log_file and args.output_dir:
        args.log_file = os.path.join(args.output_dir, f"{args.strategy}_simulation.log")
    
    # Override log_throttle if verbose mode is enabled
    if args.verbose:
        args.log_throttle = 0
        
    # Override log_level if quiet mode is enabled
    if args.quiet:
        args.log_level = "CRITICAL"
        # Redirect stdout to null to suppress all output except the final summary
        original_stdout = sys.stdout
        null_output = open(os.devnull, 'w')
        sys.stdout = null_output
    
    # ------------------------------------------------------------------
    # Config file overrides (highest precedence: CLI > config > defaults)
    # ------------------------------------------------------------------
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Config file {cfg_path} not found", file=sys.stderr)
            sys.exit(1)

        try:
            if cfg_path.suffix.lower() == ".json":
                with cfg_path.open() as f:
                    cfg_dict = json.load(f)
            else:
                if yaml is None:
                    raise ImportError("PyYAML is required for YAML config files. Install with 'pip install pyyaml'.")
                with cfg_path.open() as f:
                    cfg_dict = yaml.safe_load(f)
            if not isinstance(cfg_dict, dict):
                raise ValueError("Config root must be a mapping/dictionary")
        except Exception as e:
            print(f"Failed to load config: {e}", file=sys.stderr)
            sys.exit(1)

        # Apply config values to args if they are not set via CLI
        for key, val in cfg_dict.items():
            if hasattr(args, key):
                current = getattr(args, key)
                default_val = parser.get_default(key)
                # Skip override if the user supplied a CLI value different from the default
                if current != default_val:
                    continue
                setattr(args, key, val)

        # Handle strategy_kwargs if provided as dict in config
        if isinstance(args.strategy_kwargs, dict):
            # Convert to comma-separated k=v string expected downstream
            args.strategy_kwargs = ",".join(f"{k}={v}" for k, v in args.strategy_kwargs.items())
    
    # Build command to run playback.py
    cmd = [
        "python", "engine/playback.py",
        "--data_dir", args.data_dir,
        "--strategy", args.strategy,
        "--max_snapshots", str(args.max_snapshots),
        "--log_level", args.log_level,
        "--log_throttle", str(args.log_throttle),
        "--print_interval", str(args.print_interval),
        "--output_dir", args.output_dir
    ]
    
    # Add optional arguments
    if args.log_file:
        cmd.extend(["--log_file", args.log_file])
    
    if args.json_logs:
        cmd.append("--json_logs")
    
    if args.strategy_kwargs:
        cmd.extend(["--strategy_kwargs", args.strategy_kwargs])
    
    # Forward playback parameters
    cmd.extend(["--order_ttl", str(args.order_ttl)])
    cmd.extend(["--max_inventory", str(args.max_inventory)])
    cmd.extend(["--latency_mean_ms", str(args.latency_mean_ms)])
    cmd.extend(["--latency_std_ms", str(args.latency_std_ms)])
    cmd.extend(["--competition_intensity", str(args.competition_intensity)])
    cmd.extend(["--min_competitors", str(args.min_competitors)])
    cmd.extend(["--max_competitors", str(args.max_competitors)])
    cmd.extend(["--competitor_order_size", str(args.competitor_order_size)])
    cmd.extend(["--funding_interval", str(args.funding_interval)])
    cmd.extend(["--maker_fee", str(args.maker_fee)])
    cmd.extend(["--taker_fee", str(args.taker_fee)])
    
    # Forward hedging parameters
    cmd.extend(["--hedge_fraction", str(args.hedge_fraction)])
    cmd.extend(["--inventory_band", str(args.inventory_band)])
    if args.momentum_filter:
        cmd.append("--momentum_filter")
    cmd.extend(["--momentum_lookback", str(args.momentum_lookback)])
    cmd.extend(["--hold_time", str(args.hold_time)])
    if args.no_cancel_if_behind:
        cmd.append("--no_cancel_if_behind")
    
    # Forward max_ticks_away parameter
    cmd.extend(["--max_ticks_away", str(args.max_ticks_away)])
    
    # Forward post_only parameter
    if args.post_only:
        cmd.append("--post_only")
    cmd.extend(["--sharpe_denominator", args.sharpe_denominator])
    cmd.extend(["--sharpe_horizon", str(args.sharpe_horizon)])
    
    # Print command for reference (unless in quiet mode)
    if not args.quiet:
        print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd, capture_output=args.quiet)
    
    # Restore stdout if in quiet mode
    if args.quiet:
        sys.stdout = original_stdout
    
    print(f"\nSimulation complete!")
    print(f"Results saved to {args.output_dir}")
    if args.log_file:
        print(f"Log file: {args.log_file}")
    
    # If metrics file exists, print a summary
    metrics_file = os.path.join(args.output_dir, f"{args.strategy}_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            print("\nPerformance Summary:")
            print(f"Final P&L: {metrics.get('final_pnl', 0):.4f} USDT")
            print(f"Final Inventory: {metrics.get('final_inventory', 0):.6f} BTC")
            print(f"Total Fills: {metrics.get('fill_count', 0)}")
            
            # Print P&L attribution if available
            if 'pnl_attribution' in metrics:
                print("\nP&L Attribution:")
                for component, value in metrics['pnl_attribution'].items():
                    print(f"  {component}: {value:.4f} USDT")

            # Print Sharpe ratio if calculated
            if 'sharpe_ratio' in metrics:
                print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.4f} (window {metrics.get('sharpe_horizon_seconds', '?')}s)")

            # Print additional model metrics if available
            if 'additional_metrics' in metrics:
                print("\nAdditional Model Metrics:")
                for k, v in metrics['additional_metrics'].items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")
        except Exception as e:
            print(f"Could not read metrics file: {e}")

if __name__ == "__main__":
    main() 