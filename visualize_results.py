#!/usr/bin/env python3
"""
Visualize market-making simulation results.

This script reads the metrics files from simulation runs and creates
visualizations to help analyze and compare strategy performance.
"""
import argparse
import os
import json
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional

def load_metrics(directory: str) -> Dict[str, Any]:
    """Load metrics from a directory containing simulation results."""
    metrics_files = glob.glob(os.path.join(directory, "*_metrics.json"))
    if not metrics_files:
        print(f"No metrics files found in {directory}")
        return {}
    
    # Use the first metrics file found
    with open(metrics_files[0], 'r') as f:
        return json.load(f)

def load_log(directory: str) -> List[Dict[str, Any]]:
    """Load the detailed log file if available and parse it."""
    log_files = glob.glob(os.path.join(directory, "*.log"))
    if not log_files:
        print(f"No log files found in {directory}")
        return []
    
    # Try to parse the log file for time series data
    events = []
    try:
        with open(log_files[0], 'r') as f:
            for line in f:
                # Look for lines with summary metrics
                if "[INFO]" in line and "[" in line and "]" in line and "ts=" in line and "inv=" in line and "PnL=" in line:
                    # Extract the snapshot number which is inside square brackets
                    # Example: [INFO] [5000] ts=1673303913.792 inv=-0.0020 PnL=8.09 active=0 fills=2404
                    snapshot_match = re.search(r'\[INFO\].*?\[(\d+)\]', line)
                    if not snapshot_match:
                        continue
                        
                    snapshot_num = int(snapshot_match.group(1))
                    
                    # Extract timestamp, inventory, PnL using regex
                    ts_match = re.search(r'ts=([\d\.]+)', line)
                    inv_match = re.search(r'inv=([-\d\.]+)', line)
                    pnl_match = re.search(r'PnL=([-\d\.]+)', line)
                    
                    if ts_match and inv_match and pnl_match:
                        ts = float(ts_match.group(1))
                        inv = float(inv_match.group(1))
                        pnl = float(pnl_match.group(1))
                        
                        events.append({
                            "snapshot": snapshot_num,
                            "timestamp": ts,
                            "inventory": inv,
                            "pnl": pnl
                        })
    except Exception as e:
        print(f"Error parsing log file: {e}")
    
    return events

def plot_pnl_comparison(results_dirs: List[str], labels: Optional[List[str]] = None):
    """Plot P&L comparison between different strategies."""
    if not labels:
        labels = [os.path.basename(d) for d in results_dirs]
    
    plt.figure(figsize=(12, 6))
    
    for i, directory in enumerate(results_dirs):
        events = load_log(directory)
        if events:
            snapshots = [e["snapshot"] for e in events]
            pnls = [e["pnl"] for e in events]
            plt.plot(snapshots, pnls, label=labels[i])
    
    plt.title("P&L Comparison")
    plt.xlabel("Snapshot")
    plt.ylabel("P&L (USDT)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_inventory_comparison(results_dirs: List[str], labels: Optional[List[str]] = None):
    """Plot inventory comparison between different strategies."""
    if not labels:
        labels = [os.path.basename(d) for d in results_dirs]
    
    plt.figure(figsize=(12, 6))
    
    for i, directory in enumerate(results_dirs):
        events = load_log(directory)
        if events:
            snapshots = [e["snapshot"] for e in events]
            inventories = [e["inventory"] for e in events]
            plt.plot(snapshots, inventories, label=labels[i])
    
    plt.title("Inventory Comparison")
    plt.xlabel("Snapshot")
    plt.ylabel("Inventory (BTC)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_pnl_attribution(results_dirs: List[str], labels: Optional[List[str]] = None):
    """Plot P&L attribution breakdown for each strategy."""
    if not labels:
        labels = [os.path.basename(d) for d in results_dirs]
    
    plt.figure(figsize=(12, 6))
    
    components = ['rebates', 'adverse_selection', 'inventory_moves', 'fees']
    colors = ['green', 'red', 'blue', 'orange', 'purple']
    
    x = np.arange(len(labels))
    width = 0.15
    
    for i, component in enumerate(components):
        values = []
        for directory in results_dirs:
            metrics = load_metrics(directory)
            if metrics and 'pnl_attribution' in metrics and component in metrics['pnl_attribution']:
                values.append(metrics['pnl_attribution'][component])
            else:
                values.append(0)
        
        plt.bar(x + i*width - width*2, values, width, label=component, color=colors[i])
    
    plt.title("P&L Attribution Breakdown")
    plt.xlabel("Strategy")
    plt.ylabel("P&L Component (USDT)")
    plt.xticks(x, labels)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf()

def plot_summary_metrics(results_dirs: List[str], labels: Optional[List[str]] = None):
    """Plot summary metrics for each strategy."""
    if not labels:
        labels = [os.path.basename(d) for d in results_dirs]
    
    # Collect metrics
    pnls = []
    fill_counts = []
    sharpes = []
    drawdowns = []
    
    for directory in results_dirs:
        metrics = load_metrics(directory)
        pnls.append(metrics.get('final_pnl', 0))
        fill_counts.append(metrics.get('fill_count', 0))
        sharpes.append(metrics.get('sharpe_ratio', 0))
        drawdowns.append(metrics.get('max_drawdown', 0))
    
    # Create figure with four subplots (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot P&L
    ax1.bar(labels, pnls, color='blue')
    ax1.set_title("Final P&L")
    ax1.set_ylabel("P&L (USDT)")
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot fill count
    ax2.bar(labels, fill_counts, color='green')
    ax2.set_title("Total Fills")
    ax2.set_ylabel("Number of Fills")
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot Sharpe ratio
    ax3.bar(labels, sharpes, color='purple')
    ax3.set_title("Sharpe Ratio (Annualized)")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot Max Drawdown
    ax4.bar(labels, drawdowns, color='red')
    ax4.set_title("Maximum Drawdown")
    ax4.set_ylabel("Drawdown (USDT)")
    ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize market-making simulation results")
    parser.add_argument("--results_dirs", nargs='+', required=True, 
                        help="Directories containing simulation results")
    parser.add_argument("--labels", nargs='+', 
                        help="Labels for each results directory (default: directory names)")
    parser.add_argument("--output_dir", default="./results/visualizations", 
                        help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use provided labels or directory names
    labels = args.labels if args.labels else [os.path.basename(d) for d in args.results_dirs]
    if len(labels) != len(args.results_dirs):
        print("Warning: Number of labels doesn't match number of directories. Using directory names.")
        labels = [os.path.basename(d) for d in args.results_dirs]
    
    # Create visualizations
    print("Creating P&L comparison plot...")
    pnl_fig = plot_pnl_comparison(args.results_dirs, labels)
    pnl_fig.savefig(os.path.join(args.output_dir, "pnl_comparison.png"))
    
    print("Creating inventory comparison plot...")
    inv_fig = plot_inventory_comparison(args.results_dirs, labels)
    inv_fig.savefig(os.path.join(args.output_dir, "inventory_comparison.png"))
    
    print("Creating P&L attribution plot...")
    attr_fig = plot_pnl_attribution(args.results_dirs, labels)
    attr_fig.savefig(os.path.join(args.output_dir, "pnl_attribution.png"))
    
    print("Creating summary metrics plot...")
    summary_fig = plot_summary_metrics(args.results_dirs, labels)
    summary_fig.savefig(os.path.join(args.output_dir, "summary_metrics.png"))
    
    print(f"Visualizations saved to {args.output_dir}")
    
    # Show plots if running interactively
    plt.show()

if __name__ == "__main__":
    main() 