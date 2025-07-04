#!/usr/bin/env python3
"""Unit tests for strategy implementations."""
import sys
from pathlib import Path
import unittest

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from strategies.mid_spread import Strategy as MidSpreadStrategy
from strategies.skew_mid_spread import Strategy as SkewMidSpreadStrategy


class TestMidSpreadStrategy(unittest.TestCase):
    """Test cases for the MidSpreadStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = MidSpreadStrategy(size=0.5)
        
        # Create a basic snapshot for testing
        self.snapshot = {
            "ts": 1000.0,
            "bids": [(10000.0, 1.0), (9999.9, 2.0), (9999.8, 3.0)],
            "asks": [(10000.1, 1.5), (10000.2, 2.5), (10000.3, 3.5)],
        }

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        self.assertEqual(self.strategy.size, 0.5)

    def test_on_snapshot(self):
        """Test quote generation."""
        quote = self.strategy.on_snapshot(self.snapshot)
        
        # Expected values
        expected_mid = (10000.0 + 10000.1) / 2  # 10000.05
        expected_spread = 10000.1 - 10000.0  # 0.1
        expected_bid = expected_mid - expected_spread / 2  # 10000.0
        expected_ask = expected_mid + expected_spread / 2  # 10000.1
        
        self.assertEqual(quote["bid_price"], expected_bid)
        self.assertEqual(quote["ask_price"], expected_ask)
        self.assertEqual(quote["size"], 0.5)
        
    def test_inventory_ignored(self):
        """Test that inventory parameter is ignored."""
        # Generate quotes with different inventory values
        quote1 = self.strategy.on_snapshot(self.snapshot, inventory=0.0)
        quote2 = self.strategy.on_snapshot(self.snapshot, inventory=1.0)
        quote3 = self.strategy.on_snapshot(self.snapshot, inventory=-1.0)
        
        # All quotes should be identical since mid_spread ignores inventory
        self.assertEqual(quote1, quote2)
        self.assertEqual(quote1, quote3)


class TestSkewMidSpreadStrategy(unittest.TestCase):
    """Test cases for the SkewMidSpreadStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = SkewMidSpreadStrategy(size=0.5, gamma=0.1, clamp_ticks=0)
        
        # Create a basic snapshot for testing
        self.snapshot = {
            "ts": 1000.0,
            "bids": [(10000.0, 1.0), (9999.9, 2.0), (9999.8, 3.0)],
            "asks": [(10000.1, 1.5), (10000.2, 2.5), (10000.3, 3.5)],
        }

    def test_initialization(self):
        """Test that strategy initializes correctly."""
        self.assertEqual(self.strategy.size, 0.5)
        self.assertEqual(self.strategy.gamma, 0.1)

    def test_zero_inventory(self):
        """Test quote generation with zero inventory."""
        quote = self.strategy.on_snapshot(self.snapshot, inventory=0.0)
        
        # With zero inventory, should match mid_spread
        expected_mid = (10000.0 + 10000.1) / 2  # 10000.05
        expected_spread = 10000.1 - 10000.0  # 0.1
        expected_bid = expected_mid - expected_spread / 2  # 10000.0
        expected_ask = expected_mid + expected_spread / 2  # 10000.1
        
        self.assertAlmostEqual(quote["bid_price"], expected_bid, places=6)
        self.assertAlmostEqual(quote["ask_price"], expected_ask, places=6)
        self.assertEqual(quote["size"], 0.5)
        
    def test_positive_inventory(self):
        """Test quote generation with positive inventory."""
        quote = self.strategy.on_snapshot(self.snapshot, inventory=1.0)
        
        # With positive inventory, bid should be lower and ask should be lower
        expected_mid = (10000.0 + 10000.1) / 2  # 10000.05
        expected_spread = 10000.1 - 10000.0  # 0.1
        
        # Skew calculation: gamma * inventory
        skew = 0.1 * 1.0  # 0.1
        
        expected_bid = expected_mid - expected_spread / 2 - skew  # 9999.9
        expected_ask = expected_mid + expected_spread / 2 - skew  # 10000.0
        
        self.assertAlmostEqual(quote["bid_price"], expected_bid, places=6)
        self.assertAlmostEqual(quote["ask_price"], expected_ask, places=6)
        self.assertEqual(quote["size"], 0.5)
        
    def test_negative_inventory(self):
        """Test quote generation with negative inventory."""
        quote = self.strategy.on_snapshot(self.snapshot, inventory=-1.0)
        
        # With negative inventory, bid should be higher and ask should be higher
        expected_mid = (10000.0 + 10000.1) / 2  # 10000.05
        expected_spread = 10000.1 - 10000.0  # 0.1
        
        # Skew calculation: gamma * inventory
        skew = 0.1 * (-1.0)  # -0.1
        
        expected_bid = expected_mid - expected_spread / 2 - skew  # 10000.1
        expected_ask = expected_mid + expected_spread / 2 - skew  # 10000.2
        
        self.assertAlmostEqual(quote["bid_price"], expected_bid, places=6)
        self.assertAlmostEqual(quote["ask_price"], expected_ask, places=6)
        self.assertEqual(quote["size"], 0.5)
        
    def test_different_gamma_values(self):
        """Test that different gamma values produce different skews."""
        # Create strategies with different gamma values
        strategy1 = SkewMidSpreadStrategy(size=0.5, gamma=0.1, clamp_ticks=0)
        strategy2 = SkewMidSpreadStrategy(size=0.5, gamma=0.2, clamp_ticks=0)
        
        # Generate quotes with same inventory
        quote1 = strategy1.on_snapshot(self.snapshot, inventory=1.0)
        quote2 = strategy2.on_snapshot(self.snapshot, inventory=1.0)
        
        # Strategy with higher gamma should have more skew
        self.assertLess(quote2["bid_price"], quote1["bid_price"])
        self.assertLess(quote2["ask_price"], quote1["ask_price"])


if __name__ == "__main__":
    unittest.main() 