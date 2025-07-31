"""
Quantitative Trading System with LLM Integration

A production-ready quantitative trading system that combines traditional
quantitative strategies with Large Language Model (LLM) insights for
enhanced market analysis and decision making.
"""

__version__ = "1.0.0"
__author__ = "Fenghua"
__email__ = "fenghua@example.com"

from .config import Config
from .data.market_data import MarketDataProvider
from .llm.llm_analyzer import LLMAnalyzer
from .strategies.strategy_engine import StrategyEngine
from .risk.risk_manager import RiskManager
from .trading.trading_engine import TradingEngine

__all__ = [
    "Config",
    "MarketDataProvider", 
    "LLMAnalyzer",
    "StrategyEngine",
    "RiskManager",
    "TradingEngine",
]