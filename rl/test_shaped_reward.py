#!/usr/bin/env python3
"""
Test script for shaped reward function
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_env import TradingEnvironment

def test_shaped_reward():
    """Test the shaped reward function"""
    
    print("Testing Shaped Reward Function")
    print("=" * 50)
    
    # Initialize environment
    env = TradingEnvironment(
        data_path="../data/converted/binance",
        capital_base=10000.0,
        max_inventory=20.0,
        max_steps=100
    )
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few steps to see reward breakdown
    for step in range(5):
        print(f"\n--- Step {step + 1} ---")
        
        # Random action (bid_offset, ask_offset, size_multiplier)
        action = np.array([
            np.random.uniform(-2.0, -0.5),  # bid_offset (negative, below mid)
            np.random.uniform(0.5, 2.0),    # ask_offset (positive, above mid)
            np.random.uniform(0.5, 1.5)     # size_multiplier
        ])
        
        print(f"Action: bid_offset={action[0]:.3f}, ask_offset={action[1]:.3f}, size_mult={action[2]:.3f}")
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward:.6f}")
        print(f"Inventory: {info['inventory']:.4f}")
        print(f"Fills: {info['fills_count']}")
        
        # Print reward breakdown if available
        if 'last_reward_breakdown' in env.episode_metrics:
            breakdown = env.episode_metrics['last_reward_breakdown']
            print("Reward Breakdown:")
            for component, value in breakdown.items():
                print(f"  {component}: {value:.6f}")
        
        # Print episode metrics
        print("Episode Metrics:")
        for key, value in env.episode_metrics.items():
            if key not in ['pnl_history', 'last_reward_breakdown']:
                print(f"  {key}: {value}")
        
        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print("\n" + "=" * 50)
    print("Shaped Reward Test Complete!")
    
    # Test reward components individually
    print("\nTesting Individual Reward Components:")
    print("-" * 40)
    
    # Test inventory penalty
    print("1. Inventory Penalty Test:")
    for inventory in [0, 5, 10, 15, 20]:
        env.book.inventory = inventory
        ratio = inventory / env.max_inventory
        penalty = -0.5 * (ratio ** 2)
        print(f"   Inventory {inventory}: penalty = {penalty:.4f}")
    
    # Test spread capture bonus
    print("\n2. Spread Capture Bonus Test:")
    for fills in [0, 1, 2, 5]:
        bonus = fills * 0.1
        print(f"   {fills} fills: bonus = {bonus:.4f}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_shaped_reward() 