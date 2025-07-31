"""
Mean reversion trading strategy.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import talib
from datetime import datetime

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy based on statistical arbitrage."""
    
    def __init__(self, config: StrategyConfig = None):
        super().__init__("MeanReversionStrategy", config)
        self.config.lookback_period = 50  # Longer lookback for mean reversion
        self.config.stop_loss = 0.05
        self.config.take_profit = 0.08
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion indicators."""
        df = data.copy()
        
        # Moving averages
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Z-score calculation
        df['price_zscore'] = (df['close'] - df['sma_20']) / df['close'].rolling(window=20).std()
        
        # RSI for oversold/overbought conditions
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Standard deviation
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Distance from mean
        df['distance_from_mean'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        
        # Mean reversion score
        df['reversion_score'] = np.where(
            df['price_zscore'] > 1.5, -df['price_zscore'],
            np.where(df['price_zscore'] < -1.5, -df['price_zscore'], 0)
        )
        
        # Support and resistance levels
        df['support_level'] = talib.MIN(df['low'], timeperiod=10)
        df['resistance_level'] = talib.MAX(df['high'], timeperiod=10)
        
        return df
    
    async def generate_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[TradingSignal]:
        """Generate mean reversion trading signals."""
        signals = []
        
        if len(data) < self.config.lookback_period:
            return signals
        
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]
        
        # Entry conditions for LONG (oversold)
        long_conditions = [
            latest['price_zscore'] < -2.0,  # Significantly below mean
            latest['rsi'] < 30,  # Oversold
            latest['williams_r'] < -80,  # Oversold
            latest['close'] < latest['bb_lower'],  # Below lower Bollinger Band
            latest['distance_from_mean'] < -5.0,  # More than 5% below mean
            latest['volatility'] > 0  # Ensure some volatility
        ]
        
        # Entry conditions for SHORT (overbought)
        short_conditions = [
            latest['price_zscore'] > 2.0,  # Significantly above mean
            latest['rsi'] > 70,  # Overbought
            latest['williams_r'] > -20,  # Overbought
            latest['close'] > latest['bb_upper'],  # Above upper Bollinger Band
            latest['distance_from_mean'] > 5.0,  # More than 5% above mean
            latest['volatility'] > 0  # Ensure some volatility
        ]
        
        # Exit conditions
        exit_conditions = []
        if current_position:
            position_side = current_position.get('side', 'HOLD')
            entry_price = current_position.get('entry_price', 0)
            current_price = latest['close']
            
            # Stop loss and take profit
            if position_side == 'BUY':
                stop_loss_price = entry_price * (1 - self.config.stop_loss)
                take_profit_price = entry_price * (1 + self.config.take_profit)
                
                if current_price <= stop_loss_price:
                    exit_conditions.append('stop_loss')
                elif current_price >= take_profit_price:
                    exit_conditions.append('take_profit')
                elif latest['price_zscore'] > -0.5:  # Reverted to mean
                    exit_conditions.append('mean_reversion')
            
            elif position_side == 'SELL':
                stop_loss_price = entry_price * (1 + self.config.stop_loss)
                take_profit_price = entry_price * (1 - self.config.take_profit)
                
                if current_price >= stop_loss_price:
                    exit_conditions.append('stop_loss')
                elif current_price <= take_profit_price:
                    exit_conditions.append('take_profit')
                elif latest['price_zscore'] < 0.5:  # Reverted to mean
                    exit_conditions.append('mean_reversion')
        
        timestamp = datetime.utcnow()
        
        # Entry signals
        if all(long_conditions) and not current_position:
            confidence = min(1.0, abs(latest['price_zscore']) / 3.0)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=latest['close'],
                quantity=1.0,
                confidence=confidence,
                timestamp=timestamp,
                strategy_name=self.name,
                metadata={
                    'price_zscore': latest['price_zscore'],
                    'rsi': latest['rsi'],
                    'williams_r': latest['williams_r'],
                    'distance_from_mean': latest['distance_from_mean'],
                    'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                }
            ))
        
        elif all(short_conditions) and not current_position:
            confidence = min(1.0, abs(latest['price_zscore']) / 3.0)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=latest['close'],
                quantity=1.0,
                confidence=confidence,
                timestamp=timestamp,
                strategy_name=self.name,
                metadata={
                    'price_zscore': latest['price_zscore'],
                    'rsi': latest['rsi'],
                    'williams_r': latest['williams_r'],
                    'distance_from_mean': latest['distance_from_mean'],
                    'bb_position': (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                }
            ))
        
        # Exit signals
        if exit_conditions and current_position:
            signal_type = SignalType.SELL if current_position.get('side') == 'BUY' else SignalType.BUY
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                price=latest['close'],
                quantity=current_position.get('quantity', 0),
                confidence=0.9,  # High confidence for exits
                timestamp=timestamp,
                strategy_name=self.name,
                metadata={
                    'exit_reason': exit_conditions[0],
                    'current_zscore': latest['price_zscore'],
                    'current_rsi': latest['rsi']
                }
            ))
        
        return signals
    
    def calculate_reversion_probability(self, data: pd.DataFrame) -> float:
        """Calculate probability of mean reversion."""
        df = self.calculate_indicators(data)
        if len(df) < 1:
            return 0.0
        
        latest = df.iloc[-1]
        
        # Factors contributing to mean reversion
        zscore_factor = abs(latest['price_zscore']) / 3.0
        rsi_factor = abs(latest['rsi'] - 50) / 50.0
        bb_factor = min(abs(latest['distance_from_mean']) / 5.0, 1.0)
        
        # Combined reversion probability
        reversion_prob = (zscore_factor * 0.4 + rsi_factor * 0.3 + bb_factor * 0.3)
        
        return float(np.clip(reversion_prob, 0, 1))
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get mean reversion-specific parameters."""
        base_params = super().get_parameters()
        reversion_params = {
            'zscore_threshold': 2.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'mean_reversion_threshold': 5.0
        }
        base_params.update(reversion_params)
        return base_params
    
    def calculate_mean_distance(self, data: pd.DataFrame) -> float:
        """Calculate current distance from mean."""
        df = self.calculate_indicators(data)
        if len(df) < 1:
            return 0.0
        
        latest = df.iloc[-1]
        return float(latest['distance_from_mean'])