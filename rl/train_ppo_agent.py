#!/usr/bin/env python3
"""
PPO Agent Training Script for Market Making RL

This script trains a PPO agent using the TradingEnvironment and compares
performance against the baseline skew_mid_spread strategy.
"""

import os
import sys
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
import json
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
from trading_env import TradingEnvironment
from strategies.skew_mid_spread import Strategy as SkewMidSpreadStrategy
from engine.playback import SnapshotLoader
from engine.book import BookSimulator

class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_metrics = []
        
    def _on_step(self) -> bool:
        # Log training progress
        if self.n_calls % self.eval_freq == 0:
            if len(self.locals.get('infos', [])) > 0:
                info = self.locals['infos'][0]
                if 'episode' in info:
                    episode_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    
                    self.episode_rewards.append(episode_reward)
                    
                    if self.verbose > 0:
                        print(f"Step {self.n_calls}: Episode reward: {episode_reward:.4f}, "
                              f"Length: {episode_length}")
        
        return True

def create_training_environment(config: Dict) -> TradingEnvironment:
    """Create and configure the training environment"""
    
    env = TradingEnvironment(
        data_path=config['data_path'],
        capital_base=config['capital_base'],
        max_inventory=config['max_inventory'],
        max_steps=config['max_steps'],
        tick_size=config['tick_size']
    )
    
    return env

def train_ppo_agent(config: Dict) -> PPO:
    """Train PPO agent with the given configuration"""
    
    print("Creating training environment...")
    env = create_training_environment(config)
    
    # Wrap environment for monitoring
    env = Monitor(env, filename=f"results/rl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards
    if config.get('normalize_env', True):
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    print("Initializing PPO agent...")
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        max_grad_norm=config['max_grad_norm'],
        verbose=1,
        tensorboard_log=f"results/tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create callback
    callback = TradingCallback(eval_freq=config['eval_freq'])
    
    print(f"Starting training for {config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callback,
        progress_bar=True
    )
    
    # Save the trained model
    model_path = f"results/ppo_market_maker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return model, vec_env

def evaluate_agent(model: PPO, vec_env, config: Dict, num_episodes: int = 10) -> Dict:
    """Evaluate the trained agent"""
    
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        obs = vec_env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            episode_reward += reward[0]  # Extract scalar from array
            episode_steps += 1
            
            if done[0]:  # Check done status from array
                break
        
        episode_rewards.append(episode_reward)
        
        # Extract episode metrics from info
        if len(info) > 0 and 'metrics' in info[0]:
            episode_metrics.append(info[0]['metrics'].copy())
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Steps = {episode_steps}")
    
    # Calculate statistics
    eval_stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_metrics': episode_metrics
    }
    
    return eval_stats

def run_baseline_comparison(config: Dict, rl_snapshots: List = None) -> Dict:
    """Run baseline skew_mid_spread strategy for comparison"""
    
    print("=" * 60)
    print("STARTING BASELINE STRATEGY COMPARISON")
    print("=" * 60)
    
    # Load configuration for baseline
    print("1. Loading baseline configuration from configs/default.yaml...")
    try:
        with open('configs/default.yaml', 'r') as f:
            baseline_config = yaml.safe_load(f)
        print("   ✓ Configuration loaded successfully")
        print(f"   ✓ Maker fee: {baseline_config['maker_fee']}")
        print(f"   ✓ Taker fee: {baseline_config['taker_fee']}")
    except Exception as e:
        print(f"   ✗ ERROR loading config: {e}")
        raise
    
    # OPTIMIZATION: Reuse snapshots from RL environment if available
    if rl_snapshots is not None:
        print("2. Reusing snapshots from RL environment (FAST!)...")
        snapshots = rl_snapshots[:config['max_steps']]
        print(f"   ✓ Using {len(snapshots)} pre-loaded snapshots")
    else:
        # Fallback: Initialize snapshot loader
        print("2. Initializing snapshot loader...")
        try:
            snapshot_loader = SnapshotLoader(Path(config['data_path']))
            print(f"   ✓ SnapshotLoader created for path: {config['data_path']}")
        except Exception as e:
            print(f"   ✗ ERROR creating SnapshotLoader: {e}")
            raise
        
        print(f"6. Loading snapshots (limited to {config['max_steps']} steps)...")
        try:
            snapshots = list(snapshot_loader)[:config['max_steps']]
            print(f"   ✓ Loaded {len(snapshots)} snapshots successfully")
        except Exception as e:
            print(f"   ✗ ERROR loading snapshots: {e}")
            raise
    
    print("3. Initializing book simulator...")
    try:
        book_simulator = BookSimulator(
            tick_size=config['tick_size'],
            capital_base=config['capital_base'],
            max_inventory=config['max_inventory'],
            maker_fee=baseline_config['maker_fee'],
            taker_fee=baseline_config['taker_fee']
        )
        print(f"   ✓ BookSimulator created with capital_base: {config['capital_base']}")
    except Exception as e:
        print(f"   ✗ ERROR creating BookSimulator: {e}")
        raise
    
    # Extract strategy parameters from config
    print("4. Extracting strategy parameters...")
    try:
        strategy_kwargs = baseline_config['strategy_kwargs']
        print(f"   ✓ Found {len(strategy_kwargs)} strategy parameters")
        print(f"   ✓ Key parameters: gamma={strategy_kwargs.get('gamma')}, size_base={strategy_kwargs.get('size_base')}")
    except Exception as e:
        print(f"   ✗ ERROR extracting strategy kwargs: {e}")
        raise
    
    print("5. Initializing SkewMidSpread strategy...")
    try:
        strategy = SkewMidSpreadStrategy(
            size_base=strategy_kwargs['size_base'],
            gamma=strategy_kwargs['gamma'],
            max_inventory=config['max_inventory'],
            k_vol=strategy_kwargs['k_vol'],
            extra_spread_ticks=strategy_kwargs['extra_spread_ticks'],
            use_ticks=strategy_kwargs['use_ticks'],
            clamp_ticks=strategy_kwargs['clamp_ticks'],
            gamma_imb=strategy_kwargs['gamma_imb'],
            gamma_inv_scale=strategy_kwargs['gamma_inv_scale'],
            gamma_vol_scale=strategy_kwargs['gamma_vol_scale'],
            gamma_cap_factor=strategy_kwargs['gamma_cap_factor'],
            size_min_factor=strategy_kwargs['size_min_factor'],
            size_exp=strategy_kwargs['size_exp'],
            gradient_threshold=strategy_kwargs['gradient_threshold'],
            gradient_penalty_ticks=strategy_kwargs['gradient_penalty_ticks']
        )
        print("   ✓ SkewMidSpread strategy initialized successfully")
    except Exception as e:
        print(f"   ✗ ERROR initializing strategy: {e}")
        raise
    
    print(f"   ✓ First snapshot timestamp: {snapshots[0]['ts'] if snapshots else 'N/A'}")
    print(f"   ✓ Last snapshot timestamp: {snapshots[-1]['ts'] if snapshots else 'N/A'}")
    
    print("7. Running baseline simulation...")
    print(f"   Target steps: {len(snapshots)}")
    print(f"   Progress updates every 20 steps")
    
    try:
        for i, snapshot in enumerate(snapshots):
            if i % 20 == 0:  # More frequent progress updates
                print(f"   Progress: {i}/{len(snapshots)} ({i/len(snapshots)*100:.1f}%) - Inventory: {book_simulator.inventory:.4f}")
                
            # Update book with new snapshot
            book_simulator.update(snapshot)
            
            # Get strategy quotes
            quotes = strategy.on_snapshot(snapshot, book_simulator.inventory)
            
            # Place quotes if valid
            if quotes and 'bid_price' in quotes and 'ask_price' in quotes:
                book_simulator.place_quotes(snapshot, quotes['bid_price'], quotes['ask_price'], quotes['size'])
                
        print(f"   ✓ Completed all {len(snapshots)} simulation steps")
            
    except Exception as e:
        print(f"   ✗ ERROR during simulation at step {i}: {e}")
        raise
    
    # Get final metrics
    print("8. Computing final metrics...")
    try:
        final_pnl_attribution = book_simulator.get_pnl_attribution()
        final_portfolio_value = book_simulator.mark_to_market(snapshots[-1]["bids"][0][0])
        print(f"   ✓ Final portfolio value: {final_portfolio_value:.2f}")
        print(f"   ✓ P&L attribution computed: {len(final_pnl_attribution)} components")
    except Exception as e:
        print(f"   ✗ ERROR computing final metrics: {e}")
        raise
    
    baseline_stats = {
        'final_pnl': final_portfolio_value - config['capital_base'],
        'pnl_attribution': final_pnl_attribution,
        'fill_count': book_simulator.fills_count,
        'max_inventory': abs(book_simulator.inventory)  # Use current inventory instead of history
    }
    
    print("=" * 60)
    print("BASELINE COMPARISON COMPLETED SUCCESSFULLY!")
    print(f"Final P&L: {baseline_stats['final_pnl']:.2f}")
    print(f"Fill Count: {baseline_stats['fill_count']}")
    print(f"Max Inventory: {baseline_stats['max_inventory']:.4f}")
    print("=" * 60)
    
    return baseline_stats

def create_performance_report(rl_stats: Dict, baseline_stats: Dict, config: Dict) -> str:
    """Create a comprehensive performance report"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"results/rl_vs_baseline_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REINFORCEMENT LEARNING vs BASELINE COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {config['name']}\n\n")
        
        # RL Agent Performance
        f.write("REINFORCEMENT LEARNING AGENT PERFORMANCE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Mean Episode Reward: {rl_stats['mean_reward']:.6f}\n")
        f.write(f"Std Episode Reward:  {rl_stats['std_reward']:.6f}\n")
        f.write(f"Min Episode Reward:  {rl_stats['min_reward']:.6f}\n")
        f.write(f"Max Episode Reward:  {rl_stats['max_reward']:.6f}\n")
        
        # Calculate average metrics across episodes
        if rl_stats['episode_metrics']:
            avg_metrics = {}
            for key in rl_stats['episode_metrics'][0].keys():
                if key not in ['pnl_history', 'last_reward_breakdown']:
                    values = [ep[key] for ep in rl_stats['episode_metrics'] if key in ep]
                    if values:
                        avg_metrics[key] = np.mean(values)
            
            f.write(f"Average Total P&L:     {avg_metrics.get('total_pnl', 0):.2f}\n")
            f.write(f"Average Fill Count:    {avg_metrics.get('fill_count', 0):.1f}\n")
            f.write(f"Average Max Inventory: {avg_metrics.get('max_inventory', 0):.2f}\n")
        
        f.write("\n")
        
        # Baseline Performance
        f.write("BASELINE STRATEGY PERFORMANCE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Final P&L:      {baseline_stats['final_pnl']:.2f}\n")
        f.write(f"Fill Count:     {baseline_stats['fill_count']}\n")
        f.write(f"Max Inventory:  {baseline_stats['max_inventory']:.2f}\n")
        
        f.write("\nP&L Attribution:\n")
        for component, value in baseline_stats['pnl_attribution'].items():
            f.write(f"  {component}: {value:.2f}\n")
        
        f.write("\n")
        
        # Comparison
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 50 + "\n")
        
        if rl_stats['episode_metrics']:
            avg_rl_pnl = np.mean([ep.get('total_pnl', 0) for ep in rl_stats['episode_metrics']])
            pnl_improvement = ((avg_rl_pnl - baseline_stats['final_pnl']) / abs(baseline_stats['final_pnl'])) * 100 if baseline_stats['final_pnl'] != 0 else 0
            
            f.write(f"RL Agent Avg P&L:    {avg_rl_pnl:.2f}\n")
            f.write(f"Baseline P&L:        {baseline_stats['final_pnl']:.2f}\n")
            f.write(f"Improvement:         {pnl_improvement:.1f}%\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    return report_path

def main():
    """Main training and evaluation pipeline"""
    
    # Training configuration
    config = {
        'name': 'PPO_MarketMaker_v1_fast',
        'data_path': 'data/converted/binance',
        'capital_base': 10000.0,
        'max_inventory': 20.0,
        'max_steps': 200,  # Reduced for faster episodes
        'tick_size': 0.1,
        
        # PPO hyperparameters (smaller for faster training)
        'learning_rate': 3e-4,
        'n_steps': 512,    # Reduced from 2048
        'batch_size': 32,  # Reduced from 64
        'n_epochs': 4,     # Reduced from 10
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Training parameters (reduced for faster testing)
        'total_timesteps': 20000,  # Reduced from 100000
        'eval_freq': 500,          # Reduced from 1000
        'normalize_env': True
    }
    
    print("Starting PPO Market Making Training Pipeline")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train the agent
    model, vec_env = train_ppo_agent(config)
    
    # Evaluate the agent
    rl_stats = evaluate_agent(model, vec_env, config, num_episodes=10)
    
    # Run baseline comparison (reuse snapshots from RL environment for speed)
    try:
        # Extract snapshots from the RL environment through wrapper layers
        # vec_env is VecNormalize -> DummyVecEnv -> Monitor -> TradingEnvironment
        if hasattr(vec_env, 'venv'):  # VecNormalize wrapper
            underlying_vec_env = vec_env.venv
        else:
            underlying_vec_env = vec_env
            
        # Now get the actual environment from DummyVecEnv
        actual_env = underlying_vec_env.envs[0]  # DummyVecEnv stores envs in a list
        
        # If it's wrapped by Monitor, get the underlying env
        if hasattr(actual_env, 'env'):  # Monitor wrapper
            trading_env = actual_env.env
        else:
            trading_env = actual_env
            
        # Now we should have the actual TradingEnvironment
        if hasattr(trading_env, 'all_snapshots'):
            env_snapshots = trading_env.all_snapshots
            print(f"   ✓ Successfully accessed {len(env_snapshots)} pre-loaded snapshots!")
            baseline_stats = run_baseline_comparison(config, rl_snapshots=env_snapshots)
        else:
            print(f"   ⚠ TradingEnvironment doesn't have all_snapshots attribute")
            baseline_stats = run_baseline_comparison(config, rl_snapshots=None)
    except Exception as e:
        # Fallback if we can't access the snapshots
        print(f"   ⚠ Could not access RL snapshots ({e}), loading fresh...")
        baseline_stats = run_baseline_comparison(config, rl_snapshots=None)
    
    # Create performance report
    report_path = create_performance_report(rl_stats, baseline_stats, config)
    
    print(f"\nTraining completed!")
    print(f"Performance report saved to: {report_path}")
    print(f"RL Agent Mean Reward: {rl_stats['mean_reward']:.6f}")
    print(f"Baseline P&L: {baseline_stats['final_pnl']:.2f}")
    
    # Save results
    results = {
        'config': config,
        'rl_stats': rl_stats,
        'baseline_stats': baseline_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = f"results/rl_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {results_path}")

if __name__ == "__main__":
    main() 