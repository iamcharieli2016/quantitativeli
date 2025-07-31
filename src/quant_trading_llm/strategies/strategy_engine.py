"""
Strategy engine for managing and executing trading strategies.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from ..config import get_config
from ..data.market_data import market_data_provider
from ..data.database import db_manager
from .base_strategy import BaseStrategy, TradingSignal, SignalType
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .llm_enhanced_strategy import LLMEnhancedStrategy
from .risk_parity_strategy import RiskParityStrategy


class StrategyEngine:
    """Central engine for managing and executing trading strategies."""
    
    def __init__(self):
        self.config = get_config()
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[TradingSignal] = []
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize available trading strategies."""
        try:
            # Register built-in strategies
            self.register_strategy(MomentumStrategy())
            self.register_strategy(MeanReversionStrategy())
            self.register_strategy(LLMEnhancedStrategy())
            self.register_strategy(RiskParityStrategy())
            
            logger.info("Strategy engine initialized with built-in strategies")
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
    
    def register_strategy(self, strategy: BaseStrategy) -> bool:
        """Register a new trading strategy."""
        try:
            self.strategies[strategy.name] = strategy
            logger.info(f"Registered strategy: {strategy.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering strategy {strategy.name}: {e}")
            return False
    
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """List all available strategies."""
        return list(self.strategies.keys())
    
    async def generate_signals(
        self,
        symbols: List[str],
        strategy_name: str = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[TradingSignal]]:
        """
        Generate trading signals for given symbols.
        
        Args:
            symbols: List of asset symbols
            strategy_name: Specific strategy to use (None for all)
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary mapping symbols to their signals
        """
        signals = {}
        
        # Determine which strategies to use
        strategies_to_use = []
        if strategy_name:
            if strategy_name in self.strategies:
                strategies_to_use = [self.strategies[strategy_name]]
            else:
                logger.error(f"Strategy {strategy_name} not found")
                return signals
        else:
            strategies_to_use = list(self.strategies.values())
        
        # Generate signals for each symbol and strategy
        tasks = []
        for symbol in symbols:
            for strategy in strategies_to_use:
                task = self._generate_strategy_signals(symbol, strategy, start_date, end_date)
                tasks.append((symbol, strategy.name, task))
        
        # Execute all tasks
        results = {}
        for symbol, strategy_name, task in tasks:
            try:
                strategy_signals = await task
                if symbol not in results:
                    results[symbol] = []
                results[symbol].extend(strategy_signals)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol} with {strategy_name}: {e}")
        
        return results
    
    async def _generate_strategy_signals(
        self,
        symbol: str,
        strategy: BaseStrategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[TradingSignal]:
        """Generate signals for a specific strategy and symbol."""
        try:
            # Get market data
            data = await market_data_provider.get_stock_data(
                symbol=symbol,
                period="1y",
                interval="1d"
            )
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return []
            
            # Filter by date range if provided
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Get current position
            current_position = self.active_positions.get(symbol)
            
            # Generate signals
            signals = await strategy.generate_signals(symbol, data, current_position)
            
            # Filter valid signals
            valid_signals = [
                signal for signal in signals
                if strategy.validate_signal(signal, current_position)
            ]
            
            # Log signals
            for signal in valid_signals:
                logger.info(
                    f"Generated {signal.signal_type.value} signal for {symbol} "
                    f"using {strategy.name} at ${signal.price:.2f} "
                    f"(confidence: {signal.confidence:.2f})"
                )
            
            return valid_signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def update_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Update active positions."""
        self.active_positions = positions.copy()
        logger.info(f"Updated {len(positions)} active positions")
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        return self.active_positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active positions."""
        return self.active_positions.copy()
    
    async def backtest_strategy(
        self,
        symbols: List[str],
        strategy_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Backtest a strategy on historical data.
        
        Args:
            symbols: List of symbols to backtest
            strategy_name: Name of strategy to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Strategy {strategy_name} not found")
            return {'error': f'Strategy {strategy_name} not found'}
        
        try:
            all_results = {}
            total_trades = 0
            total_return = 0
            
            for symbol in symbols:
                # Get historical data
                data = await market_data_provider.get_stock_data(
                    symbol=symbol,
                    period="max",
                    interval="1d"
                )
                
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Filter by date range
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
                
                if len(data) < strategy.config.lookback_period:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Run backtest
                results = strategy.backtest(data, initial_capital)
                results['symbol'] = symbol
                
                all_results[symbol] = results
                total_trades += results.get('trade_count', 0)
                total_return += results.get('total_return', 0)
            
            # Aggregate results
            portfolio_return = total_return / len(symbols) if symbols else 0
            
            return {
                'strategy_name': strategy_name,
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'individual_results': all_results,
                'portfolio_return': portfolio_return,
                'total_trades': total_trades,
                'symbols_tested': len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy_name}: {e}")
            return {'error': str(e)}
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """Get performance metrics for a strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {'error': f'Strategy {strategy_name} not found'}
        
        return {
            'strategy_name': strategy_name,
            'parameters': strategy.get_parameters(),
            'is_initialized': strategy.is_initialized,
            'signals_generated': len([s for s in self.signal_history if s.strategy_name == strategy_name])
        }
    
    def get_all_performances(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all strategies."""
        return {
            name: self.get_strategy_performance(name)
            for name in self.strategies.keys()
        }
    
    def save_signals_to_database(self, signals: List[TradingSignal]) -> None:
        """Save generated signals to database for analysis."""
        try:
            for signal in signals:
                # Save technical indicators if available
                if signal.metadata:
                    from ..data.database import db_manager
                    db_manager.save_technical_indicator(
                        symbol=signal.symbol,
                        indicator_name=f"{signal.strategy_name}_signal",
                        timestamp=signal.timestamp,
                        value=float(signal.signal_type.value == 'BUY'),
                        parameters=signal.metadata,
                        signal=signal.signal_type.value.lower(),
                        confidence=signal.confidence
                    )
            
            logger.info(f"Saved {len(signals)} signals to database")
        except Exception as e:
            logger.error(f"Error saving signals to database: {e}")
    
    def get_signal_history(
        self,
        symbol: str = None,
        strategy_name: str = None,
        limit: int = 100
    ) -> List[TradingSignal]:
        """Get historical signals."""
        signals = self.signal_history
        
        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        
        if strategy_name:
            signals = [s for s in signals if s.strategy_name == strategy_name]
        
        return signals[-limit:] if limit else signals
    
    async def optimize_strategy(
        self,
        symbol: str,
        strategy_name: str,
        parameter_ranges: Dict[str, tuple] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters for a symbol.
        
        Args:
            symbol: Asset symbol
            strategy_name: Strategy name
            parameter_ranges: Parameter ranges for optimization
            
        Returns:
            Optimization results
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {'error': f'Strategy {strategy_name} not found'}
        
        # Placeholder for optimization logic
        # In a real implementation, this would use grid search, genetic algorithms, etc.
        
        return {
            'symbol': symbol,
            'strategy_name': strategy_name,
            'best_parameters': strategy.get_parameters(),
            'best_performance': 0.0,
            'optimization_method': 'grid_search',
            'note': 'Optimization not implemented in this version'
        }
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {'error': f'Strategy {strategy_name} not found'}
        
        return {
            'strategy_name': strategy_name,
            'config': strategy.config.to_dict(),
            'description': self._get_strategy_description(strategy_name)
        }
    
    def _get_strategy_description(self, strategy_name: str) -> str:
        """Get description for a strategy."""
        descriptions = {
            'MomentumStrategy': 'Follows price momentum trends using technical indicators',
            'MeanReversionStrategy': 'Buys oversold and sells overbought conditions',
            'LLMEnhancedStrategy': 'Uses AI insights combined with technical analysis',
            'RiskParityStrategy': 'Balances risk across multiple assets'
        }
        return descriptions.get(strategy_name, 'No description available')
    
    def reset_engine(self) -> None:
        """Reset the strategy engine to initial state."""
        self.active_positions.clear()
        self.signal_history.clear()
        self.strategies.clear()
        self._initialize_strategies()
        logger.info("Strategy engine reset")


# Global strategy engine instance
strategy_engine = StrategyEngine()