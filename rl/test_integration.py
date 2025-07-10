#!/usr/bin/env python3
"""
Quick integration test for RL training pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import TradingEnvironment

def test_integration():
    """Test the basic integration between environment and PPO"""
    
    print("Testing RL Environment Integration")
    print("=" * 50)
    
    # Create environment
    env = TradingEnvironment(
        data_path="../data/converted/binance",
        capital_base=10000.0,
        max_inventory=20.0,
        max_steps=50,  # Short test
        tick_size=0.1
    )
    
    print("✓ Environment created successfully")
    
    # Test environment reset
    obs, info = env.reset()
    print(f"✓ Environment reset: obs shape = {obs.shape}")
    
    # Test random actions
    print("\nTesting random actions...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward = {reward:.6f}, inventory = {info['inventory']:.4f}")
        
        if terminated or truncated:
            print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    print("✓ Random actions work correctly")
    
    # Test PPO initialization
    print("\nTesting PPO integration...")
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=3e-4,
        n_steps=64,  # Small for testing
        batch_size=32,
        verbose=0
    )
    
    print("✓ PPO model created successfully")
    
    # Test short training
    print("\nTesting short training run...")
    model.learn(total_timesteps=200, progress_bar=False)
    print("✓ Short training completed")
    
    # Test prediction
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"✓ Model prediction: action = {action}")
    
    # Test evaluation
    print("\nTesting evaluation...")
    obs = vec_env.reset()
    total_reward = 0
    
    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        
        if done[0]:
            break
    
    print(f"✓ Evaluation completed: total reward = {total_reward:.6f}")
    
    print("\n" + "=" * 50)
    print("✅ All integration tests passed!")
    print("Ready for full training pipeline.")

if __name__ == "__main__":
    test_integration() 