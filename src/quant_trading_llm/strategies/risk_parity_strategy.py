"""
Risk parity trading strategy for portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import cvxpy as cp

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig


class RiskParityStrategy(BaseStrategy):
    """Risk parity strategy for portfolio allocation."""
    
    def __init__(self, config: StrategyConfig = None):
        super().__init__("RiskParityStrategy", config)
        self.config.lookback_period = 252  # 1 year for risk calculations
        self.config.rebalance_frequency = "weekly"
        self.target_risk_contribution = 0.1  # Equal risk contribution per asset
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk and return metrics for risk parity."""
        df = data.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Risk metrics
        df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        df['volatility_60d'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
        df['volatility_252d'] = df['returns'].rolling(window=252).std() * np.sqrt(252)
        
        # Return metrics
        df['return_20d'] = df['returns'].rolling(window=20).mean() * 252
        df['return_60d'] = df['returns'].rolling(window=60).mean() * 252
        df['return_252d'] = df['returns'].rolling(window=252).mean() * 252
        
        # Sharpe ratio
        df['sharpe_252d'] = np.where(
            df['volatility_252d'] > 0,
            df['return_252d'] / df['volatility_252d'],
            0
        )
        
        # Maximum drawdown
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['running_max'] = df['cumulative_returns'].expanding().max()
        df['drawdown'] = (df['cumulative_returns'] - df['running_max']) / df['running_max']
        df['max_drawdown_252d'] = df['drawdown'].rolling(window=252).min()
        
        # Correlation with market (using price as proxy)
        df['correlation_market'] = df['returns'].rolling(window=252).corr(df['returns'])
        
        return df
    
    async def generate_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[TradingSignal]:
        """Generate risk parity allocation signals."""
        # This is a portfolio-level strategy, so we handle it differently
        # For now, return empty signals for individual symbols
        return []
    
    def calculate_risk_parity_weights(
        self,
        returns_df: pd.DataFrame,
        risk_target: float = 0.1
    ) -> Dict[str, float]:
        """
        Calculate risk parity weights for portfolio assets.
        
        Args:
            returns_df: DataFrame with returns for each asset
            risk_target: Target portfolio risk level
            
        Returns:
            Dictionary with optimal weights for each asset
        """
        if returns_df.empty:
            return {}
        
        assets = returns_df.columns
        n_assets = len(assets)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov().values
        
        # Risk parity optimization
        try:
            # Define optimization variables
            weights = cp.Variable(n_assets)
            
            # Portfolio variance
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            portfolio_volatility = cp.sqrt(portfolio_variance)
            
            # Risk contribution for each asset
            marginal_contrib = cp.multiply(cov_matrix @ weights, weights)
            risk_contrib = cp.multiply(marginal_contrib, 1 / portfolio_volatility)
            
            # Objective: minimize deviation from equal risk contribution
            target_risk = risk_target / n_assets
            objective = cp.Minimize(
                cp.sum_squares(risk_contrib - target_risk)
            )
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Fully invested
                weights >= 0,  # Long only
                weights <= 0.3,  # Maximum 30% per asset
                portfolio_volatility <= risk_target  # Risk constraint
            ]
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                weights_dict = {
                    assets[i]: float(weights.value[i])
                    for i in range(n_assets)
                    if weights.value[i] > 0.001  # Filter out tiny weights
                }
                return weights_dict
            else:
                logger.warning(f"Risk parity optimization failed: {problem.status}")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            return {}
    
    def calculate_portfolio_risk_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics."""
        try:
            # Convert weights to array
            assets = list(weights.keys())
            weights_array = np.array([weights[asset] for asset in assets])
            
            # Calculate portfolio returns
            portfolio_returns = returns_df[assets] @ weights_array
            
            # Risk metrics
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            portfolio_return = portfolio_returns.mean() * 252
            
            # Individual asset risk contributions
            cov_matrix = returns_df[assets].cov().values
            marginal_risk = cov_matrix @ weights_array
            risk_contrib = weights_array * marginal_risk / portfolio_volatility
            
            # Diversification ratio
            weighted_vol = np.sum(weights_array * returns_df[assets].std() * np.sqrt(252))
            diversification_ratio = weighted_vol / portfolio_volatility
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'portfolio_return': portfolio_return,
                'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
                'diversification_ratio': diversification_ratio,
                'risk_contributions': dict(zip(assets, risk_contrib)),
                'effective_number_assets': 1 / np.sum(weights_array ** 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        rebalancing_threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        Determine rebalancing trades based on deviation from target weights.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            rebalancing_threshold: Minimum deviation to trigger rebalancing
            
        Returns:
            Dictionary with rebalancing trades
        """
        trades = {}
        
        # All assets in either current or target
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        
        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            
            # Check if deviation exceeds threshold
            if abs(current - target) > rebalancing_threshold:
                trades[asset] = target - current
        
        return trades
    
    def calculate_risk_budget_allocation(
        self,
        asset_returns: Dict[str, pd.Series],
        risk_budgets: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate allocation based on risk budgets.
        
        Args:
            asset_returns: Dictionary with returns for each asset
            risk_budgets: Dictionary with risk budget for each asset
            
        Returns:
            Dictionary with optimal allocation
        """
        try:
            # Create returns DataFrame
            returns_df = pd.DataFrame(asset_returns).dropna()
            assets = list(asset_returns.keys())
            
            # Risk budgets
            budgets = np.array([risk_budgets.get(asset, 0.1) for asset in assets])
            
            # Covariance matrix
            cov_matrix = returns_df.cov().values
            
            # Define optimization
            weights = cp.Variable(len(assets))
            
            # Portfolio volatility
            portfolio_variance = cp.quad_form(weights, cov_matrix)
            portfolio_volatility = cp.sqrt(portfolio_variance)
            
            # Risk contribution
            marginal_contrib = cp.multiply(cov_matrix @ weights, weights)
            risk_contrib = cp.multiply(marginal_contrib, 1 / portfolio_volatility)
            
            # Objective: minimize deviation from risk budgets
            objective = cp.Minimize(
                cp.sum_squares(risk_contrib - budgets)
            )
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= 0.3
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                return {
                    assets[i]: float(weights.value[i])
                    for i in range(len(assets))
                    if weights.value[i] > 0.001
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating risk budget allocation: {e}")
            return {}
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get risk parity-specific parameters."""
        base_params = super().get_parameters()
        risk_parity_params = {
            'target_risk_contribution': self.target_risk_contribution,
            'rebalancing_frequency': self.config.rebalance_frequency,
            'max_position_size': 0.3,
            'min_position_size': 0.01,
            'risk_target': 0.1
        }
        base_params.update(risk_parity_params)
        return base_params