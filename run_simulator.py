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
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run market-making simulator with appropriate parameters")
    parser.add_argument("--strategy", default="mid_spread", help="Strategy to use (mid_spread or skew_mid_spread)")
    parser.add_argument("--max_snapshots", type=int, default=10000, help="Maximum number of snapshots to process")
    parser.add_argument("--log_level", default="INFO", help="Logging level (TRACE, DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log_throttle", type=int, default=10, help="Only log every N snapshots to reduce verbosity (0 = log all snapshots)")
    parser.add_argument("--print_interval", type=int, default=5000, help="Print summary every N snapshots (quant-friendly output)")
    parser.add_argument("--output_dir", default="./results", help="Directory to save results")
    parser.add_argument("--log_file", help="Path to log file (optional)")
    parser.add_argument("--json_logs", action="store_true", help="Output logs in JSON format")
    parser.add_argument("--strategy_kwargs", default="", help="Strategy parameters (e.g. 'gamma=0.1,size=0.001')")
    parser.add_argument("--data_dir", default="data/converted/binance", help="Directory containing order book data")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (sets log_throttle=0)")
    parser.add_argument("--quiet", action="store_true", help="Run silently and only show the final summary")
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
            import json
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
        except Exception as e:
            print(f"Could not read metrics file: {e}")

if __name__ == "__main__":
    main() 