"""
Database management and connection handling.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, and_, desc
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import pandas as pd
from loguru import logger

from ..config import get_config
from .models import Base, PriceData, MarketData, TechnicalIndicator, MarketSentiment


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.config = get_config()
        self.engine = create_engine(
            self.config.database.connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=self.config.debug
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def save_price_data(self, symbol: str, data: pd.DataFrame, source: str = "yfinance") -> int:
        """Save OHLCV price data to database."""
        if data.empty:
            return 0
            
        count = 0
        with self.get_session() as session:
            for timestamp, row in data.iterrows():
                # Convert timestamp to datetime if it's not already
                if isinstance(timestamp, str):
                    timestamp = pd.to_datetime(timestamp)
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp)
                
                # Check if record already exists
                existing = session.query(PriceData).filter(
                    and_(
                        PriceData.symbol == symbol,
                        PriceData.timestamp == timestamp,
                        PriceData.interval == row.get("interval", "1d")
                    )
                ).first()
                
                if not existing:
                    price_data = PriceData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(row.get("Open", row.get("open", 0))),
                        high_price=float(row.get("High", row.get("high", 0))),
                        low_price=float(row.get("Low", row.get("low", 0))),
                        close_price=float(row.get("Close", row.get("close", 0))),
                        volume=float(row.get("Volume", row.get("volume", 0))),
                        adjusted_close=float(row.get("Adj Close", row.get("adj_close", row.get("close", 0)))),
                        interval=row.get("interval", "1d"),
                        source=source
                    )
                    session.add(price_data)
                    count += 1
            
            session.commit()
            logger.info(f"Saved {count} price records for {symbol}")
            return count
    
    def get_price_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get price data from database."""
        with self.get_session() as session:
            query = session.query(PriceData).filter(
                and_(
                    PriceData.symbol == symbol,
                    PriceData.interval == interval
                )
            )
            
            if start_date:
                query = query.filter(PriceData.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceData.timestamp <= end_date)
                
            query = query.order_by(PriceData.timestamp)
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame([
                {
                    "timestamp": r.timestamp,
                    "open": r.open_price,
                    "high": r.high_price,
                    "low": r.low_price,
                    "close": r.close_price,
                    "volume": r.volume,
                    "adj_close": r.adjusted_close,
                    "source": r.source
                }
                for r in results
            ])
            
            df.set_index("timestamp", inplace=True)
            return df
    
    def save_technical_indicator(
        self,
        symbol: str,
        indicator_name: str,
        timestamp: datetime,
        value: float,
        parameters: Dict[str, Any] = None,
        signal: Optional[str] = None,
        confidence: Optional[float] = None,
        source: str = "system"
    ):
        """Save technical indicator data."""
        with self.get_session() as session:
            indicator = TechnicalIndicator(
                symbol=symbol,
                indicator_name=indicator_name,
                timestamp=timestamp,
                indicator_value=value,
                parameters=parameters or {},
                signal=signal,
                confidence=confidence,
                source=source
            )
            session.add(indicator)
            session.commit()
    
    def get_technical_indicators(
        self,
        symbol: str,
        indicator_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get technical indicator data."""
        with self.get_session() as session:
            query = session.query(TechnicalIndicator).filter(
                and_(
                    TechnicalIndicator.symbol == symbol,
                    TechnicalIndicator.indicator_name == indicator_name
                )
            )
            
            if start_date:
                query = query.filter(TechnicalIndicator.timestamp >= start_date)
            if end_date:
                query = query.filter(TechnicalIndicator.timestamp <= end_date)
                
            query = query.order_by(TechnicalIndicator.timestamp)
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame([
                {
                    "timestamp": r.timestamp,
                    "value": r.indicator_value,
                    "parameters": r.parameters,
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "source": r.source
                }
                for r in results
            ])
            
            df.set_index("timestamp", inplace=True)
            return df
    
    def save_market_sentiment(
        self,
        symbol: str,
        sentiment_score: float,
        sentiment_label: str,
        confidence: float,
        reasoning: Optional[str] = None,
        news_summary: Optional[str] = None,
        llm_model: str = "unknown",
        source: str = "llm"
    ):
        """Save market sentiment data from LLM analysis."""
        with self.get_session() as session:
            sentiment = MarketSentiment(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label,
                confidence=confidence,
                reasoning=reasoning,
                news_summary=news_summary,
                llm_model=llm_model,
                source=source
            )
            session.add(sentiment)
            session.commit()
    
    def get_market_sentiment(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get market sentiment data."""
        with self.get_session() as session:
            query = session.query(MarketSentiment).filter(
                MarketSentiment.symbol == symbol
            )
            
            if start_date:
                query = query.filter(MarketSentiment.timestamp >= start_date)
            if end_date:
                query = query.filter(MarketSentiment.timestamp <= end_date)
                
            query = query.order_by(desc(MarketSentiment.timestamp))
            results = query.limit(100).all()  # Limit to recent 100 records
            
            if not results:
                return pd.DataFrame()
            
            df = pd.DataFrame([
                {
                    "timestamp": r.timestamp,
                    "sentiment_score": r.sentiment_score,
                    "sentiment_label": r.sentiment_label,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "news_summary": r.news_summary,
                    "llm_model": r.llm_model,
                    "source": r.source
                }
                for r in results
            ])
            
            df.set_index("timestamp", inplace=True)
            return df
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.get_session() as session:
            # Clean up price data
            deleted_prices = session.query(PriceData).filter(
                PriceData.timestamp < cutoff_date
            ).delete()
            
            # Clean up market data
            deleted_market = session.query(MarketData).filter(
                MarketData.timestamp < cutoff_date
            ).delete()
            
            # Clean up technical indicators
            deleted_indicators = session.query(TechnicalIndicator).filter(
                TechnicalIndicator.timestamp < cutoff_date
            ).delete()
            
            # Clean up market sentiment (keep for shorter period)
            sentiment_cutoff = datetime.utcnow() - timedelta(days=30)
            deleted_sentiment = session.query(MarketSentiment).filter(
                MarketSentiment.timestamp < sentiment_cutoff
            ).delete()
            
            session.commit()
            
            logger.info(
                f"Cleaned up old data: {deleted_prices} prices, "
                f"{deleted_market} market records, {deleted_indicators} indicators, "
                f"{deleted_sentiment} sentiment records"
            )


# Global database manager instance
db_manager = DatabaseManager()