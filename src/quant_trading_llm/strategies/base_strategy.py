"""
Base strategy class for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class TradingSignal:
    """Represents a trading signal."""
    symbol: str
    signal_type: SignalType
    price: float
    quantity: float
    confidence: float  # 0-1
    timestamp: datetime
    strategy_name: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    lookback_period: int = 20
    risk_tolerance: float = 0.02  # 2% max daily loss
    position_size: float = 0.1    # 10% of portfolio per position
    stop_loss: float = 0.05       # 5% stop loss
    take_profit: float = 0.1      # 10% take profit
    max_positions: int = 10
    min_confidence: float = 0.6   # 60% minimum confidence
    rebalance_frequency: str = "daily"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lookback_period': self.lookback_period,
            'risk_tolerance': self.risk_tolerance,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_positions': self.max_positions,
            'min_confidence': self.min_confidence,
            'rebalance_frequency': self.rebalance_frequency
        }


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, config: StrategyConfig = None):
        self.name = name
        self.config = config or StrategyConfig()
        self.is_initialized = False
        self.performance_metrics = {}
    
    @abstractmethod
    async def generate_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[TradingSignal]:
        """
        Generate trading signals for the given symbol and data.
        
        Args:
            symbol: Asset symbol
            data: Market data DataFrame
            current_position: Current position information
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            DataFrame with calculated indicators
        """
        pass
    
    def validate_signal(
        self,
        signal: TradingSignal,
        current_position: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate a trading signal based on strategy rules.
        
        Args:
            signal: Trading signal to validate
            current_position: Current position information
            
        Returns:
            True if signal is valid, False otherwise
        """
        # Check minimum confidence
        if signal.confidence < self.config.min_confidence:
            return False
        
        # Check position sizing
        if signal.quantity <= 0:
            return False
        
        # Check if we already have a position in the same direction
        if current_position and current_position.get('side') == signal.signal_type.value:
            return False
        
        return True
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        risk_per_trade: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on portfolio value and risk parameters.
        
        Args:
            portfolio_value: Total portfolio value
            risk_per_trade: Risk per trade as percentage of portfolio
            
        Returns:
            Position size in currency
        """
        if risk_per_trade is None:
            risk_per_trade = self.config.position_size
        
        return portfolio_value * risk_per_trade
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate risk metrics for the strategy.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of risk metrics
        """
        if returns.empty:
            return {}
        
        metrics = {}
        
        # Basic statistics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = returns.mean() * 252
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        if metrics['annualized_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        # Win rate
        if len(returns) > 0:
            metrics['win_rate'] = (returns > 0).sum() / len(returns)
        else:
            metrics['win_rate'] = 0
        
        # Calmar ratio
        if abs(metrics['max_drawdown']) > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
        
        return metrics
    
    def backtest(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Historical market data
            initial_capital: Starting capital
            commission: Commission per trade
            
        Returns:
            Backtest results
        """
        try:
            # Calculate indicators
            data_with_indicators = self.calculate_indicators(data)
            
            # Initialize backtest variables
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = [initial_capital]
            
            # Generate signals for each day
            for i in range(len(data_with_indicators)):
                current_data = data_with_indicators.iloc[:i+1]
                
                if len(current_data) < self.config.lookback_period:
                    continue
                
                # Generate signals (simplified for backtesting)
                signals = asyncio.run(self.generate_signals(
                    symbol=data_with_indicators.index.name or "ASSET",
                    data=current_data
                ))
                
                # Execute trades based on signals
                current_price = current_data.iloc[-1]['close']
                
                for signal in signals:
                    if signal.signal_type == SignalType.BUY and position == 0:
                        # Buy signal
                        position_size = self.calculate_position_size(capital)
                        shares = int(position_size / current_price)
                        cost = shares * current_price * (1 + commission)
                        
                        if cost <= capital:
                            position = shares
                            capital -= cost
                            trades.append({
                                'timestamp': current_data.index[-1],
                                'type': 'BUY',
                                'price': current_price,
                                'shares': shares,
                                'cost': cost
                            })
                    
                    elif signal.signal_type == SignalType.SELL and position > 0:
                        # Sell signal
                        revenue = position * current_price * (1 - commission)
                        capital += revenue
                        
                        trades.append({
                            'timestamp': current_data.index[-1],
                            'type': 'SELL',
                            'price': current_price,
                            'shares': position,
                            'revenue': revenue
                        })
                        
                        position = 0
                
                # Calculate portfolio value
                portfolio_value = capital + (position * current_price)
                equity_curve.append(portfolio_value)
            
            # Calculate returns
            returns = pd.Series(equity_curve).pct_change().dropna()
            
            # Calculate performance metrics
            risk_metrics = self.calculate_risk_metrics(returns)
            
            return {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'final_value': equity_curve[-1],
                'total_return': (equity_curve[-1] - initial_capital) / initial_capital,
                'risk_metrics': risk_metrics,
                'trades': trades,
                'trade_count': len(trades),
                'equity_curve': equity_curve
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {'error': str(e)}
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'name': self.name,
            'config': self.config.to_dict(),
            'is_initialized': self.is_initialized
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters."""
        if 'config' in parameters:
            for key, value in parameters['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the strategy with given parameters."""
        try:
            self.set_parameters(kwargs)
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing strategy {self.name}: {e}")
            return False