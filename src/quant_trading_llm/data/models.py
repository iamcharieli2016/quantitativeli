"""
Data models for market data.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class MarketData(Base):
    """Base market data model."""
    
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    source = Column(String(50), nullable=False)  # yfinance, binance, etc.
    data_type = Column(String(20), nullable=False)  # price, volume, ohlcv, etc.
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_source_type', 'source', 'data_type'),
    )


class PriceData(Base):
    """Price data model for OHLCV data."""
    
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=True)
    interval = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d, etc.
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_symbol_interval_timestamp', 'symbol', 'interval', 'timestamp'),
    )


class VolumeData(Base):
    """Volume data model."""
    
    __tablename__ = "volume_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    volume = Column(Float, nullable=False)
    quote_volume = Column(Float, nullable=True)
    trade_count = Column(Integer, nullable=True)
    taker_buy_base_volume = Column(Float, nullable=True)
    taker_buy_quote_volume = Column(Float, nullable=True)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class TechnicalIndicator(Base):
    """Technical indicator data model."""
    
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    indicator_name = Column(String(50), nullable=False, index=True)
    indicator_value = Column(Float, nullable=False)
    parameters = Column(JSON, default={})  # Indicator parameters
    signal = Column(String(20), nullable=True)  # buy, sell, hold
    confidence = Column(Float, nullable=True)  # Signal confidence 0-1
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index('idx_symbol_indicator_timestamp', 'symbol', 'indicator_name', 'timestamp'),
    )


class MarketSentiment(Base):
    """Market sentiment data from LLM analysis."""
    
    __tablename__ = "market_sentiment"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    sentiment_label = Column(String(20), nullable=False)  # bearish, neutral, bullish
    confidence = Column(Float, nullable=False)  # 0-1
    reasoning = Column(String(1000), nullable=True)
    news_summary = Column(String(2000), nullable=True)
    llm_model = Column(String(50), nullable=False)
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class TradingPair(Base):
    """Trading pair configuration."""
    
    __tablename__ = "trading_pairs"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    base_currency = Column(String(10), nullable=False)
    quote_currency = Column(String(10), nullable=False)
    exchange = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    min_trade_size = Column(Float, nullable=False)
    max_trade_size = Column(Float, nullable=True)
    tick_size = Column(Float, nullable=False)
    step_size = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())