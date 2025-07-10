"""
Reinforcement Learning Environment for Market Making

This environment wraps the existing trading infrastructure to enable
RL agents to learn optimal quoting strategies.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys
import os

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.book import BookSimulator
from engine.playback import SnapshotLoader

class TradingEnvironment(gym.Env):
    """
    Market Making RL Environment
    
    State Space:
    - Current inventory (normalized)
    - Mid price (normalized)
    - Bid-ask spread
    - Market imbalance
    - Recent P&L
    - Volatility estimate
    
    Action Space:
    - Bid offset from mid (continuous)
    - Ask offset from mid (continuous)
    - Quote size multiplier (continuous)
    """
    
    def __init__(self, 
                 data_path: str = "data/converted/binance",
                 capital_base: float = 10000.0,
                 max_inventory: float = 20.0,
                 max_steps: int = 200,  # Reduced for faster training
                 tick_size: float = 0.1):
        
        super().__init__()
        
        # Environment parameters
        self.data_path = data_path
        self.capital_base = capital_base
        self.max_inventory = max_inventory
        self.max_steps = max_steps
        self.tick_size = tick_size
        
        # Trading fees (default values matching baseline)
        self.maker_fee = -0.0001  # Rebate for providing liquidity
        self.taker_fee = 0.0004   # Fee for taking liquidity
        
        # OPTIMIZATION: Load data once during initialization
        print("Loading market data (one time)...")
        self.snapshot_loader = SnapshotLoader(Path(self.data_path))
        self.all_snapshots = list(self.snapshot_loader)
        print(f"Loaded {len(self.all_snapshots)} snapshots")
        
        if len(self.all_snapshots) < self.max_steps:
            raise ValueError(f"Not enough snapshots ({len(self.all_snapshots)}) for episode length ({self.max_steps})")
        
        # State tracking
        self.current_step = 0
        self.episode_pnl = 0.0
        self.initial_mid = None
        self.prev_portfolio_value = capital_base
        
        # Current episode data
        self.snapshots = []
        self.current_snapshot_idx = 0
        
        # Book simulator
        self.book = None
        
        # Define action space: [bid_offset, ask_offset, size_multiplier]
        # bid_offset, ask_offset: -50 to +50 ticks from mid
        # size_multiplier: 0.1 to 2.0 (10% to 200% of base size)
        self.action_space = spaces.Box(
            low=np.array([-50.0, -50.0, 0.1]),
            high=np.array([50.0, 50.0, 2.0]),
            dtype=np.float32
        )
        
        # Define observation space
        # [inventory_ratio, mid_price_norm, spread_norm, imbalance, pnl_norm, volatility]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Performance tracking
        self.episode_metrics = {
            'total_pnl': 0.0,
            'spread_capture': 0.0,
            'inventory_penalty': 0.0,
            'transaction_costs': 0.0,
            'max_inventory': 0.0,
            'fill_count': 0
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Select a random starting point for the episode
        max_start = len(self.all_snapshots) - self.max_steps
        self.current_snapshot_idx = np.random.randint(0, max_start)
        
        # Initialize book simulator
        self.book = BookSimulator(
            tick_size=self.tick_size,
            capital_base=self.capital_base,
            max_inventory=self.max_inventory,
            latency_mean_ms=50.0,
            latency_std_ms=10.0,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee
        )
        
        # Reset tracking variables
        self.current_step = 0
        self.episode_pnl = 0.0
        self.prev_portfolio_value = self.capital_base
        
        # Get initial state
        current_snapshot = self.all_snapshots[self.current_snapshot_idx]
        self.initial_mid = (current_snapshot["bids"][0][0] + current_snapshot["asks"][0][0]) / 2.0
        
        # Reset metrics
        self.episode_metrics = {
            'total_pnl': 0.0,
            'spread_capture': 0.0,
            'inventory_penalty': 0.0,
            'transaction_costs': 0.0,
            'max_inventory': 0.0,
            'fill_count': 0,
            'prev_fill_count': 0,
            'prev_fees': 0.0,
            'pnl_history': [],
            'last_reward_breakdown': {}
        }
        
        observation = self._get_observation(current_snapshot)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Get current snapshot
        current_snapshot = self.all_snapshots[self.current_snapshot_idx]
        
        # Extract action components
        bid_offset, ask_offset, size_multiplier = action
        
        # Calculate mid price
        mid_price = (current_snapshot["bids"][0][0] + current_snapshot["asks"][0][0]) / 2.0
        
        # Calculate quote prices
        bid_price = mid_price + (bid_offset * self.tick_size)
        ask_price = mid_price + (ask_offset * self.tick_size)
        
        # Ensure valid quote (bid < ask)
        if bid_price >= ask_price:
            # Penalize invalid quotes
            reward = -10.0
            observation = self._get_observation(current_snapshot)
            info = self._get_info()
            terminated = False
            truncated = False
            return observation, reward, terminated, truncated, info
        
        # Calculate quote size
        base_size = 0.5  # Base size in BTC
        quote_size = base_size * size_multiplier
        
        # Update book simulator
        self.book.update(current_snapshot)
        
        # Place quotes
        try:
            self.book.place_quotes(current_snapshot, bid_price, ask_price, quote_size)
        except Exception as e:
            # Penalize execution errors
            reward = -5.0
            observation = self._get_observation(current_snapshot)
            info = self._get_info()
            terminated = False
            truncated = False
            return observation, reward, terminated, truncated, info
        
        # Calculate reward
        reward = self._calculate_reward(current_snapshot)
        
        # Update step counter
        self.current_step += 1
        self.current_snapshot_idx += 1
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get next observation
        if not terminated and not truncated:
            next_snapshot = self.all_snapshots[self.current_snapshot_idx]
            observation = self._get_observation(next_snapshot)
        else:
            observation = self._get_observation(current_snapshot)
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """Convert market snapshot to observation vector"""
        
        # Calculate mid price
        mid_price = (snapshot["bids"][0][0] + snapshot["asks"][0][0]) / 2.0
        
        # Normalize inventory (-1 to 1)
        inventory_ratio = self.book.inventory / self.max_inventory if self.book else 0.0
        inventory_ratio = np.clip(inventory_ratio, -1.0, 1.0)
        
        # Normalize mid price (relative to initial)
        mid_price_norm = (mid_price - self.initial_mid) / self.initial_mid if self.initial_mid else 0.0
        mid_price_norm = np.clip(mid_price_norm, -1.0, 1.0)
        
        # Calculate spread (normalized)
        spread = snapshot["asks"][0][0] - snapshot["bids"][0][0]
        spread_norm = np.clip(spread / (mid_price * 0.01), 0.0, 1.0)  # Normalize by 1% of mid
        
        # Calculate market imbalance
        bid_size = snapshot["bids"][0][1]
        ask_size = snapshot["asks"][0][1]
        total_size = bid_size + ask_size
        imbalance = (bid_size - ask_size) / total_size if total_size > 0 else 0.0
        
        # Calculate recent P&L (normalized)
        current_portfolio_value = self.book.mark_to_market(mid_price) if self.book else self.capital_base
        recent_pnl = current_portfolio_value - self.prev_portfolio_value
        pnl_norm = np.clip(recent_pnl / (self.capital_base * 0.01), -1.0, 1.0)  # Normalize by 1% of capital
        self.prev_portfolio_value = current_portfolio_value
        
        # Simple volatility estimate (placeholder)
        volatility = 0.5  # Could implement rolling volatility calculation
        
        observation = np.array([
            inventory_ratio,
            mid_price_norm,
            spread_norm,
            imbalance,
            pnl_norm,
            volatility
        ], dtype=np.float32)
        
        return observation
    
    def _calculate_reward(self, snapshot: Dict[str, Any]) -> float:
        """
        Calculate shaped reward for market making RL
        
        Components:
        1. P&L change (primary signal)
        2. Inventory penalty (risk management)
        3. Spread capture bonus (encourage market making)
        4. Transaction cost penalty (efficiency)
        5. Fill rate bonus (activity incentive)
        6. Quote quality bonus (tight spreads)
        """
        
        if not self.book:
            return 0.0
        
        # Get current portfolio value
        mid_price = (snapshot["bids"][0][0] + snapshot["asks"][0][0]) / 2.0
        current_portfolio_value = self.book.mark_to_market(mid_price)
        
        # === 1. P&L Component (Primary Signal) ===
        pnl_change = current_portfolio_value - self.prev_portfolio_value
        pnl_component = pnl_change
        
        # === 2. Inventory Risk Penalty ===
        # Quadratic penalty to strongly discourage large inventory
        inventory_ratio = self.book.inventory / self.max_inventory
        inventory_component = -0.5 * (inventory_ratio ** 2)
        
        # === 3. Spread Capture Bonus ===
        # Reward for successful fills (market making behavior)
        fills_this_step = self.book.fills_count - self.episode_metrics.get('prev_fill_count', 0)
        spread_capture_component = fills_this_step * 0.1  # Small bonus per fill
        
        # === 4. Transaction Cost Penalty ===
        # Get P&L attribution to extract fees
        try:
            pnl_attribution = self.book.get_pnl_attribution()
            fees_this_step = pnl_attribution.get('fees', 0.0) - self.episode_metrics.get('prev_fees', 0.0)
            transaction_cost_component = fees_this_step * 2.0  # Amplify cost penalty
        except:
            transaction_cost_component = 0.0
        
        # === 5. Quote Quality Bonus ===
        # Reward for posting competitive quotes (tight spreads)
        market_spread = snapshot["asks"][0][0] - snapshot["bids"][0][0]
        quote_quality_component = 0.0
        
        # === 6. Risk-Adjusted Return Component ===
        # Penalize volatility of returns (encourage consistency)
        volatility_penalty = 0.0
        if len(self.episode_metrics.get('pnl_history', [])) > 5:
            recent_pnls = self.episode_metrics['pnl_history'][-5:]
            pnl_volatility = np.std(recent_pnls) if len(recent_pnls) > 1 else 0.0
            volatility_penalty = -0.01 * pnl_volatility
        
        # === Combine All Components ===
        total_reward = (
            pnl_component +                    # Primary: profit/loss
            inventory_component +              # Risk: inventory control
            spread_capture_component +         # Activity: successful fills
            transaction_cost_component +       # Efficiency: minimize costs
            quote_quality_component +          # Quality: competitive quotes
            volatility_penalty                 # Consistency: stable returns
        )
        
        # === Update Tracking Variables ===
        self.episode_metrics['prev_fill_count'] = self.book.fills_count
        try:
            pnl_attribution = self.book.get_pnl_attribution()
            self.episode_metrics['prev_fees'] = pnl_attribution.get('fees', 0.0)
        except:
            self.episode_metrics['prev_fees'] = 0.0
        
        # Track P&L history for volatility calculation
        if 'pnl_history' not in self.episode_metrics:
            self.episode_metrics['pnl_history'] = []
        self.episode_metrics['pnl_history'].append(pnl_change)
        
        # Keep only recent history
        if len(self.episode_metrics['pnl_history']) > 20:
            self.episode_metrics['pnl_history'] = self.episode_metrics['pnl_history'][-20:]
        
        # === Update Episode Metrics ===
        self.episode_metrics['total_pnl'] = current_portfolio_value - self.capital_base
        self.episode_metrics['inventory_penalty'] += inventory_component
        self.episode_metrics['spread_capture'] += spread_capture_component
        self.episode_metrics['transaction_costs'] += transaction_cost_component
        self.episode_metrics['max_inventory'] = max(self.episode_metrics['max_inventory'], abs(self.book.inventory))
        self.episode_metrics['fill_count'] = self.book.fills_count
        
        # === Reward Components for Analysis ===
        self.episode_metrics['last_reward_breakdown'] = {
            'pnl': pnl_component,
            'inventory': inventory_component,
            'spread_capture': spread_capture_component,
            'transaction_costs': transaction_cost_component,
            'quote_quality': quote_quality_component,
            'volatility': volatility_penalty,
            'total': total_reward
        }
        
        return total_reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate early"""
        
        if not self.book:
            return False
        
        # Terminate if inventory becomes too large
        if abs(self.book.inventory) > self.max_inventory * 1.5:
            return True
        
        # Terminate if losses are too large
        current_portfolio_value = self.book.mark_to_market(
            (self.all_snapshots[self.current_snapshot_idx]["bids"][0][0] + 
             self.all_snapshots[self.current_snapshot_idx]["asks"][0][0]) / 2.0
        )
        
        if current_portfolio_value < self.capital_base * 0.8:  # 20% loss
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for monitoring"""
        
        info = {
            'step': self.current_step,
            'inventory': self.book.inventory if self.book else 0.0,
            'fills_count': self.book.fills_count if self.book else 0,
            'metrics': self.episode_metrics.copy()
        }
        
        return info
    
    def render(self, mode='human'):
        """Render environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Inventory: {self.book.inventory:.4f}, "
                  f"P&L: {self.episode_metrics['total_pnl']:.2f}")
    
    def close(self):
        """Clean up resources"""
        pass 