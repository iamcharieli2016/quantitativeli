"""
Momentum-based trading strategy.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import talib
from datetime import datetime

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig


class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy based on price and volume momentum."""
    
    def __init__(self, config: StrategyConfig = None):
        super().__init__("MomentumStrategy", config)
        self.config.lookback_period = 20  # Override default for momentum
        self.config.stop_loss = 0.08  # Wider stop for momentum
        self.config.take_profit = 0.15  # Larger profit target
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        df = data.copy()
        
        # Price momentum indicators
        df['returns'] = df['close'].pct_change()
        df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Volume momentum
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Rate of Change (ROC)
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # ADX for trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Momentum score (composite indicator)
        df['momentum_score'] = (
            (df['momentum_5d'] * 0.4 +
             df['momentum_10d'] * 0.3 +
             df['momentum_20d'] * 0.2 +
             df['roc'] * 0.1) * 100
        )
        
        return df
    
    async def generate_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[TradingSignal]:
        """Generate momentum-based trading signals."""
        signals = []
        
        if len(data) < self.config.lookback_period:
            return signals
        
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]
        
        # Entry conditions for LONG
        long_conditions = [
            latest['momentum_score'] > 2.0,  # Strong upward momentum
            latest['rsi'] < 70,  # Not overbought
            latest['macd'] > latest['macd_signal'],  # MACD bullish crossover
            latest['volume_ratio'] > 1.2,  # Above average volume
            latest['adx'] > 25,  # Strong trend
            latest['momentum_5d'] > 0  # Short-term positive
        ]
        
        # Entry conditions for SHORT
        short_conditions = [
            latest['momentum_score'] < -2.0,  # Strong downward momentum
            latest['rsi'] > 30,  # Not oversold
            latest['macd'] < latest['macd_signal'],  # MACD bearish crossover
            latest['volume_ratio'] > 1.2,  # Above average volume
            latest['adx'] > 25,  # Strong trend
            latest['momentum_5d'] < 0  # Short-term negative
        ]
        
        # Exit conditions
        exit_conditions = []
        if current_position:
            position_side = current_position.get('side', 'HOLD')
            entry_price = current_position.get('entry_price', 0)
            current_price = latest['close']
            
            # Stop loss
            if position_side == 'BUY':
                stop_loss_price = entry_price * (1 - self.config.stop_loss)
                if current_price <= stop_loss_price:
                    exit_conditions.append('stop_loss')
                
                # Take profit
                take_profit_price = entry_price * (1 + self.config.take_profit)
                if current_price >= take_profit_price:
                    exit_conditions.append('take_profit')
                
                # Momentum reversal
                if latest['momentum_score'] < -1.0:
                    exit_conditions.append('momentum_reversal')
            
            elif position_side == 'SELL':
                stop_loss_price = entry_price * (1 + self.config.stop_loss)
                if current_price >= stop_loss_price:
                    exit_conditions.append('stop_loss')
                
                take_profit_price = entry_price * (1 - self.config.take_profit)
                if current_price <= take_profit_price:
                    exit_conditions.append('take_profit')
                
                # Momentum reversal
                if latest['momentum_score'] > 1.0:
                    exit_conditions.append('momentum_reversal')
        
        # Generate signals
        timestamp = datetime.utcnow()
        
        # Entry signals
        if all(long_conditions) and not current_position:
            confidence = min(1.0, latest['momentum_score'] / 10.0)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=latest['close'],
                quantity=1.0,  # Will be calculated by position sizing
                confidence=confidence,
                timestamp=timestamp,
                strategy_name=self.name,
                metadata={
                    'momentum_score': latest['momentum_score'],
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'volume_ratio': latest['volume_ratio'],
                    'adx': latest['adx']
                }
            ))
        
        elif all(short_conditions) and not current_position:
            confidence = min(1.0, abs(latest['momentum_score']) / 10.0)
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=latest['close'],
                quantity=1.0,
                confidence=confidence,
                timestamp=timestamp,
                strategy_name=self.name,
                metadata={
                    'momentum_score': latest['momentum_score'],
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'volume_ratio': latest['volume_ratio'],
                    'adx': latest['adx']
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
                confidence=0.8,  # High confidence for exits
                timestamp=timestamp,
                strategy_name=self.name,
                metadata={
                    'exit_reason': exit_conditions[0],
                    'current_momentum': latest['momentum_score'],
                    'current_rsi': latest['rsi']
                }
            ))
        
        return signals
    
    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get momentum-specific parameters."""
        base_params = super().get_parameters()
        momentum_params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_threshold': 25,
            'volume_threshold': 1.2,
            'momentum_threshold': 2.0
        }
        base_params.update(momentum_params)
        return base_params
    
    def set_strategy_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set momentum-specific parameters."""
        super().set_parameters(parameters)
        # Momentum-specific parameter handling would go here
        pass
    
    def calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall momentum strength."""
        df = self.calculate_indicators(data)
        if len(df) < 1:
            return 0.0
        
        latest = df.iloc[-1]
        
        # Weighted momentum score
        momentum_strength = (
            latest['momentum_score'] * 0.3 +
            (latest['rsi'] - 50) / 50 * 0.2 +
            latest['macd_hist'] * 0.3 +
            (latest['adx'] - 25) / 25 * 0.2
        )
        
        return float(np.clip(momentum_strength, -10, 10))