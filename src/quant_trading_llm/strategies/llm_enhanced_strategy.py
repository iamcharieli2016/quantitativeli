"""
LLM-enhanced trading strategy combining technical analysis with AI insights.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import talib
from datetime import datetime

from .base_strategy import BaseStrategy, TradingSignal, SignalType, StrategyConfig
from ..llm.llm_analyzer import llm_analyzer
from ..data.database import db_manager


class LLMEnhancedStrategy(BaseStrategy):
    """Strategy that combines technical analysis with LLM insights."""
    
    def __init__(self, config: StrategyConfig = None):
        super().__init__("LLMEnhancedStrategy", config)
        self.config.lookback_period = 30
        self.config.min_confidence = 0.7  # Higher confidence for AI-enhanced signals
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for LLM analysis."""
        df = data.copy()
        
        # Basic technical indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # RSI and momentum
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df
    
    async def generate_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[TradingSignal]:
        """Generate signals using LLM-enhanced analysis."""
        signals = []
        
        if len(data) < self.config.lookback_period:
            return signals
        
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]
        
        # Prepare data for LLM analysis
        price_data = {
            'current_price': float(latest['close']),
            'price_change_24h': float(df['close'].pct_change().iloc[-1] * 100),
            'price_change_7d': float((latest['close'] / df['close'].iloc[-7] - 1) * 100),
            'volume_24h': float(latest['volume']),
            'market_cap': float(latest['close'] * latest['volume'] / 0.01)  # Estimate
        }
        
        technical_indicators = {
            'rsi': float(latest['rsi']),
            'macd': float(latest['macd']),
            'macd_signal': float(latest['macd_signal']),
            'sma_20': float(latest['sma_20']),
            'sma_50': float(latest['sma_50']),
            'bb_upper': float(latest['bb_upper']),
            'bb_lower': float(latest['bb_lower']),
            'atr': float(latest['atr'])
        }
        
        # Get market sentiment
        sentiment_data = await self._get_market_sentiment(symbol)
        
        # Use LLM for analysis
        if llm_analyzer.is_available():
            llm_analysis = await llm_analyzer.analyze_market_data(
                symbol=symbol,
                price_data=price_data,
                technical_indicators=technical_indicators,
                market_sentiment=sentiment_data,
                news_headlines=[]  # Could be populated with real news
            )
            
            # Generate signal based on LLM recommendation
            signal = self._create_signal_from_llm(
                symbol, latest['close'], llm_analysis, current_position
            )
            
            if signal:
                signals.append(signal)
        
        # Fallback to technical analysis if LLM unavailable
        else:
            technical_signals = await self._generate_technical_signals(
                symbol, df, current_position
            )
            signals.extend(technical_signals)
        
        return signals
    
    async def _get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get market sentiment data."""
        try:
            from ..llm.sentiment_analyzer import sentiment_analyzer
            sentiment = await sentiment_analyzer.get_aggregated_sentiment(symbol, 7)
            return {
                'sentiment': sentiment.get('sentiment', 'neutral'),
                'score': sentiment.get('sentiment_score', 0),
                'fear_greed': 50  # Placeholder
            }
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {'sentiment': 'neutral', 'score': 0, 'fear_greed': 50}
    
    def _create_signal_from_llm(
        self,
        symbol: str,
        current_price: float,
        llm_analysis: Dict[str, Any],
        current_position: Optional[Dict[str, Any]]
    ) -> Optional[TradingSignal]:
        """Create trading signal from LLM analysis."""
        try:
            recommendation = llm_analysis.get('recommendation', '').upper()
            
            if recommendation == 'BUY' and not current_position:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=current_price,
                    quantity=1.0,
                    confidence=llm_analysis.get('confidence', 0.5) / 100.0,
                    timestamp=datetime.utcnow(),
                    strategy_name=self.name,
                    metadata={
                        'llm_recommendation': recommendation,
                        'target_price': llm_analysis.get('target_price', current_price * 1.1),
                        'stop_loss': llm_analysis.get('stop_loss', current_price * 0.95),
                        'reasoning': llm_analysis.get('reasoning', '')
                    }
                )
            
            elif recommendation == 'SELL' and not current_position:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=current_price,
                    quantity=1.0,
                    confidence=llm_analysis.get('confidence', 0.5) / 100.0,
                    timestamp=datetime.utcnow(),
                    strategy_name=self.name,
                    metadata={
                        'llm_recommendation': recommendation,
                        'target_price': llm_analysis.get('target_price', current_price * 0.9),
                        'stop_loss': llm_analysis.get('stop_loss', current_price * 1.05),
                        'reasoning': llm_analysis.get('reasoning', '')
                    }
                )
            
            elif recommendation == 'HOLD' and current_position:
                # Check if we should exit based on LLM analysis
                if llm_analysis.get('confidence', 0) < 50:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.CLOSE,
                        price=current_price,
                        quantity=current_position.get('quantity', 0),
                        confidence=0.7,
                        timestamp=datetime.utcnow(),
                        strategy_name=self.name,
                        metadata={'exit_reason': 'llm_low_confidence'}
                    )
        
        except Exception as e:
            logger.error(f"Error creating signal from LLM: {e}")
        
        return None
    
    async def _generate_technical_signals(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_position: Optional[Dict[str, Any]]
    ) -> List[TradingSignal]:
        """Generate fallback technical signals."""
        signals = []
        latest = data.iloc[-1]
        
        # Simple technical rules
        bullish_conditions = [
            latest['close'] > latest['sma_20'],
            latest['sma_20'] > latest['sma_50'],
            latest['rsi'] < 70,
            latest['macd'] > latest['macd_signal']
        ]
        
        bearish_conditions = [
            latest['close'] < latest['sma_20'],
            latest['sma_20'] < latest['sma_50'],
            latest['rsi'] > 30,
            latest['macd'] < latest['macd_signal']
        ]
        
        if all(bullish_conditions) and not current_position:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                price=latest['close'],
                quantity=1.0,
                confidence=0.6,
                timestamp=datetime.utcnow(),
                strategy_name=self.name,
                metadata={'signal_source': 'technical_fallback'}
            ))
        
        elif all(bearish_conditions) and not current_position:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                price=latest['close'],
                quantity=1.0,
                confidence=0.6,
                timestamp=datetime.utcnow(),
                strategy_name=self.name,
                metadata={'signal_source': 'technical_fallback'}
            ))
        
        return signals