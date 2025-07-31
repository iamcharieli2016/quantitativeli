"""
Backtesting engine for quantitative trading strategies.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger

from ..config import get_config
from ..data.database import db_manager
from ..strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
from ..risk.risk_manager import risk_manager


@dataclass
class BacktestResult:
    """Results from a backtest."""
    
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade_return: float
    largest_win: float
    largest_loss: float
    equity_curve: List[float]
    trades: List[Dict[str, Any]]
    daily_returns: List[float]
    monthly_returns: List[float]
    yearly_returns: List[float]
    
    @property
    def calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        return self.annualized_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0
    
    @property
    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        negative_returns = [r for r in self.daily_returns if r < 0]
        if not negative_returns:
            return self.annualized_return / 0.01  # Avoid division by zero
        downside_dev = np.std(negative_returns) * np.sqrt(252)
        return self.annualized_return / downside_dev


class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies."""
    
    def __init__(self):
        self.config = get_config()
        self.is_running = False
        self.results_cache: Dict[str, BacktestResult] = {}
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ) -> BacktestResult:
        """
        Run backtest for a strategy on a single symbol.
        
        Args:
            strategy: Trading strategy to test
            symbol: Asset symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            commission: Commission per trade
            slippage: Slippage per trade
            
        Returns:
            Complete backtest results
        """
        try:
            logger.info(
                f"Starting backtest: {strategy.name} on {symbol} "
                f"from {start_date} to {end_date}"
            )
            
            # Get historical data
            from ..data.market_data import market_data_provider
            data = await market_data_provider.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            if data.empty or len(data) < strategy.config.lookback_period:
                logger.warning(f"Insufficient data for {symbol}")
                return self._empty_result(strategy.name, symbol, start_date, end_date)
            
            # Run backtest simulation
            result = await self._simulate_trades(
                strategy, symbol, data, initial_capital, commission, slippage
            )
            
            # Cache result
            cache_key = f"{strategy.name}_{symbol}_{start_date}_{end_date}"
            self.results_cache[cache_key] = result
            
            logger.info(
                f"Backtest completed: {result.total_return:.2%} return, "
                f"{result.total_trades} trades, Sharpe: {result.sharpe_ratio:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return self._empty_result(strategy.name, symbol, start_date, end_date)
    
    async def _simulate_trades(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pd.DataFrame,
        initial_capital: float,
        commission: float,
        slippage: float
    ) -> BacktestResult:
        """Simulate trades for the backtest."""
        
        # Initialize backtest state
        capital = initial_capital
        position = 0
        cash = initial_capital
        equity_curve = [initial_capital]
        trades = []
        daily_returns = []
        
        # Track position
        current_position = None
        
        # Process each day
        for i in range(strategy.config.lookback_period, len(data)):
            current_data = data.iloc[:i+1]
            current_date = current_data.index[-1]
            current_price = current_data.iloc[-1]['close']
            
            # Generate signals
            signals = await strategy.generate_signals(symbol, current_data, current_position)
            
            # Execute trades based on signals
            for signal in signals:
                if signal.signal_type == SignalType.BUY:
                    # Calculate position size
                    position_size = strategy.calculate_position_size(cash)
                    shares = int(position_size / (current_price * (1 + commission + slippage)))
                    
                    if shares > 0 and cash >= shares * current_price * (1 + commission + slippage):
                        # Execute buy
                        cost = shares * current_price * (1 + commission + slippage)
                        position += shares
                        cash -= cost
                        
                        trades.append({
                            'date': current_date,
                            'type': 'BUY',
                            'price': current_price,
                            'shares': shares,
                            'value': cost,
                            'cash': cash
                        })
                        
                        current_position = {
                            'side': 'BUY',
                            'quantity': position,
                            'entry_price': current_price
                        }
                
                elif signal.signal_type == SignalType.SELL and position > 0:
                    # Execute sell
                    revenue = position * current_price * (1 - commission - slippage)
                    cash += revenue
                    
                    trades.append({
                        'date': current_date,
                        'type': 'SELL',
                        'price': current_price,
                        'shares': position,
                        'value': revenue,
                        'cash': cash
                    })
                    
                    position = 0
                    current_position = None
            
            # Calculate portfolio value
            portfolio_value = cash + (position * current_price)
            equity_curve.append(portfolio_value)
            
            # Calculate daily returns
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
        
        # Calculate final metrics
        final_capital = cash + (position * current_price)
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Calculate performance metrics
        if daily_returns:
            daily_returns_array = np.array(daily_returns)
            volatility = np.std(daily_returns_array) * np.sqrt(252)
            annualized_return = np.mean(daily_returns_array) * 252
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Max drawdown
            cumulative_returns = np.array(equity_curve) / initial_capital
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Trade statistics
            winning_trades = [t for t in trades if t['type'] == 'SELL' and t['value'] > t.get('previous_buy_value', 0)]
            losing_trades = [t for t in trades if t['type'] == 'SELL' and t['value'] <= t.get('previous_buy_value', 0)]
            
            total_trades = len(trades)
            win_rate = len(winning_trades) / max(len(trades) // 2, 1)
            
            # Calculate trade returns
            trade_returns = []
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            for sell_trade in sell_trades:
                # Find corresponding buy trade
                buy_trade = max([t for t in buy_trades if t['date'] < sell_trade['date']], 
                              key=lambda x: x['date'], default=None)
                if buy_trade:
                    trade_return = (sell_trade['value'] - buy_trade['value']) / buy_trade['value']
                    trade_returns.append(trade_return)
                    sell_trade['previous_buy_value'] = buy_trade['value']
            
            average_trade_return = np.mean(trade_returns) if trade_returns else 0
            largest_win = max(trade_returns) if trade_returns else 0
            largest_loss = min(trade_returns) if trade_returns else 0
            profit_factor = len(winning_trades) / max(len(losing_trades), 1)
        else:
            volatility = annualized_return = sharpe_ratio = max_drawdown = 0
            total_trades = winning_trades = losing_trades = 0
            average_trade_return = largest_win = largest_loss = profit_factor = 0
        
        # Calculate monthly and yearly returns
        monthly_returns = self._calculate_periodic_returns(equity_curve, data.index, 'M')
        yearly_returns = self._calculate_periodic_returns(equity_curve, data.index, 'Y')
        
        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            average_trade_return=average_trade_return,
            largest_win=largest_win,
            largest_loss=largest_loss,
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns
        )
    
    def _calculate_periodic_returns(
        self,
        equity_curve: List[float],
        dates: pd.DatetimeIndex,
        period: str
    ) -> List[float]:
        """Calculate returns for specified periods."""
        if len(equity_curve) < 2:
            return []
        
        df = pd.DataFrame({'equity': equity_curve}, index=dates)
        
        # Resample to specified period
        if period == 'M':
            period_returns = df['equity'].resample('M').last().pct_change().dropna()
        elif period == 'Y':
            period_returns = df['equity'].resample('Y').last().pct_change().dropna()
        else:
            return []
        
        return period_returns.tolist()
    
    async def run_portfolio_backtest(
        self,
        strategies: List[BaseStrategy],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Run backtest for multiple strategies and symbols.
        
        Args:
            strategies: List of strategies to test
            symbols: List of symbols to test
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital per symbol
            
        Returns:
            Portfolio backtest results
        """
        individual_results = {}
        
        # Run backtests for each strategy-symbol combination
        tasks = []
        for strategy in strategies:
            for symbol in symbols:
                task = self.run_backtest(
                    strategy, symbol, start_date, end_date, initial_capital
                )
                tasks.append((strategy.name, symbol, task))
        
        # Collect results
        for strategy_name, symbol, task in tasks:
            try:
                result = await task
                if strategy_name not in individual_results:
                    individual_results[strategy_name] = {}
                individual_results[strategy_name][symbol] = result
            except Exception as e:
                logger.error(f"Error in portfolio backtest for {strategy_name}-{symbol}: {e}")
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(individual_results)
        
        return {
            'individual_results': individual_results,
            'portfolio_metrics': portfolio_metrics,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'strategies': [s.name for s in strategies],
            'symbols': symbols
        }
    
    def _calculate_portfolio_metrics(self, results: Dict[str, Dict[str, BacktestResult]]) -> Dict[str, float]:
        """Calculate portfolio-level metrics."""
        all_results = []
        for strategy_results in results.values():
            for result in strategy_results.values():
                all_results.append(result)
        
        if not all_results:
            return {}
        
        # Calculate average metrics
        avg_return = np.mean([r.total_return for r in all_results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
        avg_drawdown = np.mean([r.max_drawdown for r in all_results])
        
        # Calculate equal-weighted portfolio return
        portfolio_returns = []
        for i in range(max(len(r.equity_curve) for r in all_results)):
            daily_values = []
            for result in all_results:
                if i < len(result.equity_curve):
                    daily_values.append(result.equity_curve[i])
                else:
                    daily_values.append(result.equity_curve[-1])
            
            portfolio_value = np.mean(daily_values)
            portfolio_returns.append(portfolio_value)
        
        # Calculate portfolio metrics
        portfolio_returns_pct = []
        for i in range(1, len(portfolio_returns)):
            portfolio_returns_pct.append((portfolio_returns[i] - portfolio_returns[i-1]) / portfolio_returns[i-1])
        
        if portfolio_returns_pct:
            portfolio_volatility = np.std(portfolio_returns_pct) * np.sqrt(252)
            portfolio_sharpe = (avg_return * 252) / portfolio_volatility if portfolio_volatility > 0 else 0
        else:
            portfolio_volatility = portfolio_sharpe = 0
        
        return {
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'average_drawdown': avg_drawdown,
            'portfolio_return': (portfolio_returns[-1] - portfolio_returns[0]) / portfolio_returns[0] if portfolio_returns else 0,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_sharpe': portfolio_sharpe
        }
    
    def _empty_result(
        self,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Return empty backtest result."""
        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=0,
            final_capital=0,
            total_return=0,
            annualized_return=0,
            volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_trade_return=0,
            largest_win=0,
            largest_loss=0,
            equity_curve=[],
            trades=[],
            daily_returns=[],
            monthly_returns=[],
            yearly_returns=[]
        )
    
    def get_backtest_summary(self, result: BacktestResult) -> Dict[str, Any]:
        """Get summary of backtest results."""
        return {
            'strategy': result.strategy_name,
            'symbol': result.symbol,
            'period': f"{result.start_date.date()} to {result.end_date.date()}",
            'total_return': f"{result.total_return:.2%}",
            'annualized_return': f"{result.annualized_return:.2%}",
            'volatility': f"{result.volatility:.2%}",
            'sharpe_ratio': f"{result.sharpe_ratio:.2f}",
            'max_drawdown': f"{result.max_drawdown:.2%}",
            'win_rate': f"{result.win_rate:.2%}",
            'total_trades': result.total_trades,
            'calmar_ratio': f"{result.calmar_ratio:.2f}",
            'sortino_ratio': f"{result.sortino_ratio:.2f}"
        }


# Global backtest engine instance
backtest_engine = BacktestEngine()