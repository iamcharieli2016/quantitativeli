"""
Risk management system for the quantitative trading platform.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from ..config import get_config
from ..data.database import db_manager


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio."""
    
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float = 0.0
    correlation_market: float = 0.0
    
    @property
    def risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0
        
        # Position size risk
        if abs(self.position_size) > 0.2:  # >20% of portfolio
            score += 30
        elif abs(self.position_size) > 0.1:  # >10% of portfolio
            score += 15
        
        # Drawdown risk
        if self.max_drawdown < -0.1:  # >10% drawdown
            score += 25
        elif self.max_drawdown < -0.05:  # >5% drawdown
            score += 15
        
        # Volatility risk
        if self.volatility > 0.3:  # >30% annual volatility
            score += 20
        elif self.volatility > 0.2:  # >20% annual volatility
            score += 10
        
        # VaR risk
        if abs(self.var_95) > 0.05:  # >5% daily VaR
            score += 25
        elif abs(self.var_95) > 0.03:  # >3% daily VaR
            score += 15
        
        return min(score, 100)


class RiskLimits:
    """Risk limits configuration."""
    
    def __init__(self):
        self.config = get_config()
        self.limits = {
            'max_position_size': self.config.trading.max_position_size,
            'max_portfolio_risk': 0.15,  # 15% max portfolio risk
            'max_single_asset_risk': 0.05,  # 5% max risk per asset
            'max_daily_loss': self.config.trading.max_daily_loss,
            'max_drawdown': self.config.trading.max_drawdown,
            'max_leverage': 2.0,
            'min_liquidity_ratio': 0.1,  # 10% cash buffer
            'max_correlation': 0.8,  # Max correlation between positions
            'var_confidence': 0.95,
            'var_horizon_days': 1
        }
    
    def update_limits(self, new_limits: Dict[str, float]) -> None:
        """Update risk limits."""
        self.limits.update(new_limits)
        logger.info("Risk limits updated")
    
    def get_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return self.limits.copy()


class RiskManager:
    """Central risk management system."""
    
    def __init__(self):
        self.config = get_config()
        self.limits = RiskLimits()
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics_cache: Dict[str, RiskMetrics] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.utcnow()
    
    def add_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        timestamp: datetime = None
    ) -> bool:
        """Add or update a position."""
        try:
            timestamp = timestamp or datetime.utcnow()
            
            if symbol in self.positions:
                # Update existing position
                existing = self.positions[symbol]
                total_size = existing['size'] + size
                
                if abs(total_size) < 1e-6:  # Position closed
                    del self.positions[symbol]
                    logger.info(f"Position closed: {symbol}")
                    return True
                
                # Calculate new average entry price
                total_value = (existing['size'] * existing['entry_price']) + (size * entry_price)
                new_entry_price = total_value / total_size
                
                self.positions[symbol] = {
                    'size': total_size,
                    'entry_price': new_entry_price,
                    'last_update': timestamp
                }
            else:
                # New position
                self.positions[symbol] = {
                    'size': size,
                    'entry_price': entry_price,
                    'last_update': timestamp
                }
            
            logger.info(f"Position updated: {symbol} size={size}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """Remove a position."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Position removed: {symbol}")
            return True
        return False
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position details."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions."""
        return self.positions.copy()
    
    def calculate_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in prices:
                total_value += position['size'] * prices[symbol]
        return total_value
    
    def calculate_portfolio_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio weights."""
        total_value = self.calculate_portfolio_value(prices)
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            if symbol in prices:
                value = position['size'] * prices[symbol]
                weights[symbol] = value / total_value
        
        return weights
    
    def check_position_limits(self, symbol: str, proposed_size: float, prices: Dict[str, float]) -> Tuple[bool, str]:
        """Check if proposed position size is within limits."""
        
        # Calculate portfolio value
        total_value = self.calculate_portfolio_value(prices)
        if total_value == 0:
            return True, "Portfolio empty, position allowed"
        
        # Check individual position size
        proposed_value = abs(proposed_size * prices.get(symbol, 0))
        max_position_value = total_value * self.limits.limits['max_position_size']
        
        if proposed_value > max_position_value:
            return False, f"Position size exceeds limit: {proposed_value:.2f} > {max_position_value:.2f}"
        
        # Check total portfolio risk
        current_weights = self.calculate_portfolio_weights(prices)
        
        # Simulate new weights with proposed position
        new_weights = current_weights.copy()
        if symbol in new_weights:
            new_weights[symbol] = (self.positions.get(symbol, {}).get('size', 0) + proposed_size) * prices[symbol] / total_value
        else:
            new_weights[symbol] = proposed_size * prices[symbol] / total_value
        
        # Check concentration risk
        max_weight = max(new_weights.values())
        if max_weight > self.limits.limits['max_position_size']:
            return False, f"Weight limit exceeded: {max_weight:.2f} > {self.limits.limits['max_position_size']}"
        
        return True, "Position within limits"
    
    def check_daily_limits(self, daily_pnl: float) -> Tuple[bool, str]:
        """Check if daily PnL is within limits."""
        self.daily_pnl += daily_pnl
        self.daily_trades += 1
        
        # Check daily loss limit
        if self.daily_pnl < -self.limits.limits['max_daily_loss']:
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f} < {-self.limits.limits['max_daily_loss']}"
        
        return True, "Daily limits OK"
    
    def calculate_risk_metrics(
        self,
        symbol: str,
        prices: Dict[str, float],
        price_history: pd.DataFrame = None
    ) -> RiskMetrics:
        """Calculate risk metrics for a position."""
        position = self.positions.get(symbol)
        if not position or symbol not in prices:
            return None
        
        current_price = prices[symbol]
        position_size = position['size']
        entry_price = position['entry_price']
        
        # Basic risk calculations
        unrealized_pnl = position_size * (current_price - entry_price)
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        # VaR calculation (simplified)
        if price_history is not None and len(price_history) > 20:
            returns = price_history['close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5) * abs(position_size * current_price)
            var_99 = np.percentile(returns, 1) * abs(position_size * current_price)
            expected_shortfall = np.mean(returns[returns < np.percentile(returns, 5)]) * abs(position_size * current_price)
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming 0 risk-free rate)
            sharpe = returns.mean() * 252 / volatility if volatility > 0 else 0
        else:
            var_95 = var_99 = expected_shortfall = 0
            volatility = max_drawdown = sharpe = 0
        
        # Calculate portfolio value for position sizing
        total_value = self.calculate_portfolio_value(prices)
        position_value = abs(position_size * current_price)
        position_weight = position_value / total_value if total_value > 0 else 0
        
        return RiskMetrics(
            symbol=symbol,
            position_size=position_weight,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            volatility=volatility
        )
    
    def get_portfolio_risk_summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary."""
        if not self.positions:
            return {'total_value': 0, 'positions': 0, 'risk_score': 0}
        
        total_value = self.calculate_portfolio_value(prices)
        weights = self.calculate_portfolio_weights(prices)
        
        # Calculate individual risk metrics
        risk_metrics = {}
        total_var_95 = 0
        total_var_99 = 0
        
        for symbol in self.positions.keys():
            if symbol in prices:
                # Get price history for VaR calculation
                # In a real implementation, this would fetch historical data
                metrics = self.calculate_risk_metrics(symbol, prices)
                if metrics:
                    risk_metrics[symbol] = metrics
                    total_var_95 += metrics.var_95
                    total_var_99 += metrics.var_99
        
        # Portfolio-level statistics
        max_position = max(weights.values()) if weights else 0
        concentration_risk = max_position / sum(weights.values()) if weights else 0
        
        # Calculate overall risk score
        avg_risk_score = np.mean([m.risk_score for m in risk_metrics.values()]) if risk_metrics else 0
        
        return {
            'total_value': total_value,
            'positions': len(self.positions),
            'risk_metrics': risk_metrics,
            'portfolio_var_95': total_var_95,
            'portfolio_var_99': total_var_99,
            'max_position_weight': max_position,
            'concentration_risk': concentration_risk,
            'average_risk_score': avg_risk_score,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades
        }
    
    def check_risk_alerts(self, prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for risk alerts and warnings."""
        alerts = []
        
        # Check daily loss limit
        if self.daily_pnl < -self.limits.limits['max_daily_loss']:
            alerts.append({
                'type': 'DAILY_LOSS_LIMIT',
                'severity': 'CRITICAL',
                'message': f"Daily loss limit exceeded: {self.daily_pnl:.2f}",
                'value': self.daily_pnl,
                'limit': -self.limits.limits['max_daily_loss']
            })
        
        # Check portfolio concentration
        weights = self.calculate_portfolio_weights(prices)
        max_weight = max(weights.values()) if weights else 0
        
        if max_weight > self.limits.limits['max_position_size']:
            max_symbol = max(weights, key=weights.get)
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'severity': 'HIGH',
                'message': f"High concentration in {max_symbol}: {max_weight:.2f}",
                'value': max_weight,
                'limit': self.limits.limits['max_position_size']
            })
        
        # Check individual position risk
        for symbol, position in self.positions.items():
            if symbol in prices:
                # Check drawdown
                current_price = prices[symbol]
                drawdown = (current_price - position['entry_price']) / position['entry_price']
                
                if drawdown < -self.limits.limits['max_drawdown']:
                    alerts.append({
                        'type': 'DRAWDOWN_LIMIT',
                        'severity': 'MEDIUM',
                        'message': f"High drawdown in {symbol}: {drawdown:.2%}",
                        'value': drawdown,
                        'limit': -self.limits.limits['max_drawdown']
                    })
        
        return alerts
    
    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at market open)."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.utcnow()
        logger.info("Daily risk limits reset")
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return self.limits.get_limits()
    
    def update_risk_limits(self, new_limits: Dict[str, float]) -> None:
        """Update risk limits."""
        self.limits.update_limits(new_limits)
    
    def export_risk_report(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """Export comprehensive risk report."""
        summary = self.get_portfolio_risk_summary(prices)
        alerts = self.check_risk_alerts(prices)
        
        return {
            'generated_at': datetime.utcnow().isoformat(),
            'portfolio_summary': summary,
            'risk_alerts': alerts,
            'risk_limits': self.get_risk_limits(),
            'positions': self.get_all_positions()
        }


# Global risk manager instance
risk_manager = RiskManager()