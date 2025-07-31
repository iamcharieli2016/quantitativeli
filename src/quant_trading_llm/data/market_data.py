"""
Market data provider for fetching and managing financial data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import yfinance as yf
import ccxt
from loguru import logger

from ..config import get_config
from .database import db_manager


class MarketDataProvider:
    """Unified market data provider supporting multiple sources."""
    
    def __init__(self):
        self.config = get_config()
        self.exchanges = {}
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """Initialize cryptocurrency exchanges."""
        try:
            # Initialize Binance exchange
            if self.config.data.binance_api_key and self.config.data.binance_secret:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.config.data.binance_api_key,
                    'secret': self.config.data.binance_secret,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
            else:
                # Use public API if no credentials provided
                self.exchanges['binance'] = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
            
            # Initialize Coinbase Pro
            self.exchanges['coinbasepro'] = ccxt.coinbasepro({
                'enableRateLimit': True
            })
            
            logger.info("Exchanges initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
            else:
                data = ticker.history(
                    period=period,
                    interval=interval
                )
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add metadata
            data['symbol'] = symbol
            data['interval'] = interval
            data['source'] = 'yfinance'
            
            # Save to database
            await asyncio.to_thread(
                db_manager.save_price_data, symbol, data, 'yfinance'
            )
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_crypto_data(
        self,
        symbol: str,
        exchange: str = "binance",
        timeframe: str = "1d",
        limit: int = 365,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get cryptocurrency data from exchanges.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
            exchange: Exchange name ('binance', 'coinbasepro')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
            limit: Number of records to fetch
            since: Start date for data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if exchange not in self.exchanges:
                logger.error(f"Exchange {exchange} not initialized")
                return pd.DataFrame()
            
            ex = self.exchanges[exchange]
            
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'limit': limit
            }
            
            if since:
                params['since'] = int(since.timestamp() * 1000)
            
            # Fetch data from exchange
            ohlcv = await asyncio.to_thread(ex.fetch_ohlcv, **params)
            
            if not ohlcv:
                logger.warning(f"No data found for {symbol} on {exchange}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add metadata
            df['symbol'] = symbol
            df['interval'] = timeframe
            df['source'] = exchange
            
            # Save to database
            await asyncio.to_thread(
                db_manager.save_price_data, symbol, df, exchange
            )
            
            logger.info(f"Fetched {len(df)} records for {symbol} from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol} from {exchange}: {e}")
            return pd.DataFrame()
    
    async def get_realtime_price(
        self,
        symbol: str,
        source: str = "yfinance"
    ) -> Dict[str, float]:
        """
        Get real-time price data.
        
        Args:
            symbol: Asset symbol
            source: Data source ('yfinance', 'binance', 'coinbasepro')
            
        Returns:
            Dictionary with current price data
        """
        try:
            if source == "yfinance":
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if data.empty:
                    return {}
                
                latest = data.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': float(latest['Close']),
                    'volume': float(latest['Volume']),
                    'timestamp': latest.name,
                    'change': float((latest['Close'] - latest['Open']) / latest['Open'] * 100),
                    'source': 'yfinance'
                }
            
            elif source in self.exchanges:
                ex = self.exchanges[source]
                ticker = await asyncio.to_thread(ex.fetch_ticker, symbol)
                
                return {
                    'symbol': symbol,
                    'price': float(ticker['last']),
                    'bid': float(ticker['bid']),
                    'ask': float(ticker['ask']),
                    'volume': float(ticker['quoteVolume'] or 0),
                    'timestamp': datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    'change': float(ticker['percentage'] or 0),
                    'source': source
                }
            
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return {}
    
    async def get_market_info(self, symbol: str, source: str = "yfinance") -> Dict[str, Any]:
        """
        Get market information for a symbol.
        
        Args:
            symbol: Asset symbol
            source: Data source
            
        Returns:
            Dictionary with market information
        """
        try:
            if source == "yfinance":
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                return {
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'beta': info.get('beta', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'source': 'yfinance'
                }
            
            elif source in self.exchanges:
                ex = self.exchanges[source]
                markets = await asyncio.to_thread(ex.fetch_markets)
                
                market_info = next((m for m in markets if m['symbol'] == symbol), None)
                if market_info:
                    return {
                        'symbol': symbol,
                        'name': market_info.get('id', symbol),
                        'base': market_info.get('base', ''),
                        'quote': market_info.get('quote', ''),
                        'active': market_info.get('active', True),
                        'precision': market_info.get('precision', {}),
                        'limits': market_info.get('limits', {}),
                        'source': source
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching market info for {symbol}: {e}")
            return {}
    
    async def get_multiple_assets(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        source: str = "yfinance"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple assets concurrently.
        
        Args:
            symbols: List of asset symbols
            period: Period to fetch
            interval: Data interval
            source: Data source
            
        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        tasks = []
        
        for symbol in symbols:
            if source == "yfinance":
                task = self.get_stock_data(symbol, period, interval)
            else:
                task = self.get_crypto_data(symbol, source, interval)
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                data = await task
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def get_available_symbols(self, source: str = "yfinance") -> List[str]:
        """
        Get available symbols for a data source.
        
        Args:
            source: Data source
            
        Returns:
            List of available symbols
        """
        try:
            if source in self.exchanges:
                ex = self.exchanges[source]
                markets = ex.load_markets()
                return list(markets.keys())
            
            # For yfinance, return common stock symbols
            return [
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'SPY', 'QQQ', 'IWM', 'VTI', 'VYM', 'TLT', 'GLD', 'USO'
            ]
            
        except Exception as e:
            logger.error(f"Error getting available symbols for {source}: {e}")
            return []


# Global market data provider instance
market_data_provider = MarketDataProvider()