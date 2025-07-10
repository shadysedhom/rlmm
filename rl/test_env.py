"""
Test script for the Trading RL Environment
"""

import numpy as np
from trading_env import TradingEnvironment

def test_environment():
    """Test basic environment functionality"""
    
    print("Testing Trading RL Environment...")
    
    # Create environment
    env = TradingEnvironment(
        data_path="../data/converted/binance",
        capital_base=10000.0,
        max_inventory=20.0,
        max_steps=100,  # Short episode for testing
        tick_size=0.1
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test random actions
    print("\nTesting random actions...")
    total_reward = 0.0
    
    for step in range(10):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        print(f"Step {step}: Action={action}, Reward={reward:.4f}, "
              f"Inventory={info['inventory']:.4f}, Terminated={terminated}")
        
        if terminated or truncated:
            print("Episode ended early")
            break
    
    print(f"\nTotal reward: {total_reward:.4f}")
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_environment() 