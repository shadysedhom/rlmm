#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    print("Testing imports...")
    
    try:
        from trading_env import TradingEnvironment
        print("✓ TradingEnvironment imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TradingEnvironment: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✓ PPO imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PPO: {e}")
        return False
    
    try:
        from engine.book import BookSimulator
        print("✓ BookSimulator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import BookSimulator: {e}")
        return False
    
    try:
        from engine.playback import SnapshotLoader
        print("✓ SnapshotLoader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SnapshotLoader: {e}")
        return False
    
    try:
        from strategies.skew_mid_spread import Strategy as SkewMidSpreadStrategy
        print("✓ SkewMidSpreadStrategy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SkewMidSpreadStrategy: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 