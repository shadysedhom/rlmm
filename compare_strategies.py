#!/usr/bin/env python3
"""
Compare multiple market-making strategies and parameter combinations.

This script runs the simulator with different strategies and parameters,
collects the results, and presents them in a tabular format for easy comparison.
"""
import argparse
import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any

def run_simulation(
    strategy: str,
    max_snapshots: int = 5000,
    strategy_kwargs: str = "",
    data_dir: str = "data/converted/binance",
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """Run a simulation with the given parameters and return the metrics."""
    
    # Create a unique output directory for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"{strategy}_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        "python", "run_simulator.py",
        "--strategy", strategy,
        "--max_snapshots", str(max_snapshots),
        "--quiet",
        "--output_dir", run_output_dir
    ]
    
    if strategy_kwargs:
        cmd.extend(["--strategy_kwargs", strategy_kwargs])
    
    # Run the simulation
    print(f"Running {strategy} with params: {strategy_kwargs or 'default'}")
    subprocess.run(cmd, check=True)
    
    # Read the metrics file
    metrics_file = os.path.join(run_output_dir, f"{strategy}_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        print(f"Warning: No metrics file found at {metrics_file}")
        return {}

def format_table(results: List[Dict[str, Any]]) -> str:
    """Format the results as a markdown table."""
    if not results:
        return "No results to display."
    
    # Define the columns
    columns = [
        "Strategy", "Parameters", "P&L", "Inventory", "Fills", 
        "Rebates", "Adverse Selection"
    ]
    
    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for result in results:
        widths["Strategy"] = max(widths["Strategy"], len(result.get("strategy", "")))
        widths["Parameters"] = max(widths["Parameters"], len(str(result.get("strategy_kwargs", ""))))
        widths["P&L"] = max(widths["P&L"], len(f"{result.get('final_pnl', 0):.4f}"))
        widths["Inventory"] = max(widths["Inventory"], len(f"{result.get('final_inventory', 0):.6f}"))
        widths["Fills"] = max(widths["Fills"], len(str(result.get("fill_count", 0))))
        
        if "pnl_attribution" in result:
            widths["Rebates"] = max(widths["Rebates"], 
                                   len(f"{result['pnl_attribution'].get('rebates', 0):.4f}"))
            widths["Adverse Selection"] = max(widths["Adverse Selection"], 
                                            len(f"{result['pnl_attribution'].get('adverse_selection', 0):.4f}"))
    
    # Create the header
    header = " | ".join(f"{col:{widths[col]}}" for col in columns)
    separator = "-|-".join("-" * widths[col] for col in columns)
    
    # Create the rows
    rows = []
    for result in results:
        row = [
            f"{result.get('strategy', ''):{widths['Strategy']}}",
            f"{str(result.get('strategy_kwargs', '')):{widths['Parameters']}}",
            f"{result.get('final_pnl', 0):.4f}".rjust(widths["P&L"]),
            f"{result.get('final_inventory', 0):.6f}".rjust(widths["Inventory"]),
            f"{result.get('fill_count', 0)}".rjust(widths["Fills"])
        ]
        
        if "pnl_attribution" in result:
            row.append(f"{result['pnl_attribution'].get('rebates', 0):.4f}".rjust(widths["Rebates"]))
            row.append(f"{result['pnl_attribution'].get('adverse_selection', 0):.4f}".rjust(widths["Adverse Selection"]))
        else:
            row.append(" " * widths["Rebates"])
            row.append(" " * widths["Adverse Selection"])
        
        rows.append(" | ".join(row))
    
    # Combine everything
    return f"{header}\n{separator}\n" + "\n".join(rows)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple market-making strategies")
    parser.add_argument("--max_snapshots", type=int, default=5000, 
                        help="Maximum number of snapshots to process for each simulation")
    parser.add_argument("--output_dir", default="./results/comparison", 
                        help="Directory to save all results")
    parser.add_argument("--data_dir", default="data/converted/binance", 
                        help="Directory containing order book data")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define the strategies and parameters to compare
    comparisons = [
        {"strategy": "mid_spread", "strategy_kwargs": ""},
        {"strategy": "skew_mid_spread", "strategy_kwargs": "gamma=0.1,size=0.001"},
        {"strategy": "skew_mid_spread", "strategy_kwargs": "gamma=0.2,size=0.001"},
        {"strategy": "skew_mid_spread", "strategy_kwargs": "gamma=0.1,size=0.002"},
        {"strategy": "skew_mid_spread", "strategy_kwargs": "gamma=0.05,size=0.001"}
    ]
    
    # Run the simulations and collect results
    results = []
    for config in comparisons:
        metrics = run_simulation(
            strategy=config["strategy"],
            max_snapshots=args.max_snapshots,
            strategy_kwargs=config["strategy_kwargs"],
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Add the strategy and parameters to the metrics
        metrics["strategy"] = config["strategy"]
        metrics["strategy_kwargs"] = config["strategy_kwargs"]
        results.append(metrics)
    
    # Format and display the results
    table = format_table(results)
    print("\nResults Comparison:")
    print(table)
    
    # Save the results to a file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"comparison_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("Strategy Comparison Results\n")
        f.write("=========================\n\n")
        f.write(table)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 