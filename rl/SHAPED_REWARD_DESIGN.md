# Shaped Reward Function for Market Making RL

## Overview
The shaped reward function is designed to guide the RL agent toward profitable market making behavior by combining multiple reward components that capture different aspects of trading performance.

## Reward Components

### 1. P&L Component (Primary Signal)
- **Purpose**: Direct profit/loss feedback
- **Formula**: `pnl_change = current_portfolio_value - prev_portfolio_value`
- **Weight**: 1.0 (baseline)
- **Rationale**: This is the ultimate objective - making money

### 2. Inventory Risk Penalty
- **Purpose**: Discourage large inventory positions
- **Formula**: `inventory_component = -0.5 * (inventory_ratio^2)`
- **Weight**: -0.5
- **Rationale**: Quadratic penalty strongly discourages large positions, encouraging delta-neutral market making

### 3. Spread Capture Bonus
- **Purpose**: Reward successful fills (market making activity)
- **Formula**: `spread_capture_component = fills_this_step * 0.1`
- **Weight**: 0.1 per fill
- **Rationale**: Encourages active market making and liquidity provision

### 4. Transaction Cost Penalty
- **Purpose**: Minimize trading fees and slippage
- **Formula**: `transaction_cost_component = fees_this_step * 2.0`
- **Weight**: 2.0x amplification
- **Rationale**: Heavily penalize expensive taker trades, encourage maker fills

### 5. Quote Quality Bonus
- **Purpose**: Reward competitive spreads
- **Formula**: Currently placeholder (0.0)
- **Rationale**: Could reward quotes that are tighter than market spread

### 6. Volatility Penalty
- **Purpose**: Encourage consistent returns
- **Formula**: `volatility_penalty = -0.01 * std(recent_pnls)`
- **Weight**: -0.01
- **Rationale**: Penalize erratic P&L patterns, encourage stable performance

## Total Reward Formula

```python
total_reward = (
    pnl_component +                    # Primary: profit/loss
    inventory_component +              # Risk: inventory control  
    spread_capture_component +         # Activity: successful fills
    transaction_cost_component +       # Efficiency: minimize costs
    quote_quality_component +          # Quality: competitive quotes
    volatility_penalty                 # Consistency: stable returns
)
```

## Key Design Principles

1. **P&L Dominance**: P&L change is the primary signal, other components are secondary
2. **Risk Management**: Quadratic inventory penalty prevents dangerous positions
3. **Market Making Behavior**: Bonuses for fills and competitive quotes
4. **Cost Awareness**: Heavy penalties for transaction costs
5. **Consistency**: Slight penalty for volatile returns

## Tracking and Analysis

The reward function tracks detailed metrics for analysis:
- Individual component values
- Episode-level aggregates
- P&L history for volatility calculation
- Fill counts and fee tracking

## Expected Behavior

The shaped reward should encourage the agent to:
- Stay near inventory neutral
- Capture bid-ask spreads through maker fills
- Avoid expensive taker trades
- Maintain consistent performance
- Actively provide liquidity

## Future Enhancements

1. **Adaptive Weights**: Adjust component weights based on market conditions
2. **Quote Quality**: Implement actual quote competitiveness measurement
3. **Market Impact**: Add penalty for moving the market
4. **Risk-Adjusted Returns**: More sophisticated risk metrics (Sharpe ratio, VaR)
5. **Regime Detection**: Different reward structures for different market regimes 