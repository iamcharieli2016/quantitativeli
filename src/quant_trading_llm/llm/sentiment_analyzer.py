"""
Sentiment analyzer for market sentiment analysis.
"""

import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import aiohttp
from loguru import logger

from ..config import get_config
from ..data.database import db_manager


class SentimentAnalyzer:
    """Market sentiment analyzer using multiple data sources."""
    
    def __init__(self):
        self.config = get_config()
    
    async def analyze_news_sentiment(
        self,
        symbol: str,
        news_text: str,
        source: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles.
        
        Args:
            symbol: Asset symbol
            news_text: News text content
            source: News source type
            
        Returns:
            Sentiment analysis results
        """
        try:
            # Use TextBlob for basic sentiment analysis
            blob = TextBlob(news_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Map polarity to sentiment categories
            if polarity > 0.1:
                sentiment = "bullish"
            elif polarity < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            # Calculate confidence based on subjectivity and text length
            confidence = min(0.9, max(0.1, 1 - subjectivity + (len(news_text) / 1000)))
            
            result = {
                "symbol": symbol,
                "sentiment": sentiment,
                "sentiment_score": polarity,
                "confidence": confidence,
                "subjectivity": subjectivity,
                "source": source,
                "timestamp": datetime.utcnow()
            }
            
            # Save to database
            db_manager.save_market_sentiment(
                symbol=symbol,
                sentiment_score=polarity,
                sentiment_label=sentiment,
                confidence=confidence,
                news_summary=news_text[:500],
                source=source
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return {
                "error": str(e),
                "sentiment": "neutral",
                "sentiment_score": 0,
                "confidence": 0
            }
    
    async def analyze_social_sentiment(
        self,
        symbol: str,
        social_posts: List[str],
        platform: str = "twitter"
    ) -> Dict[str, Any]:
        """
        Analyze sentiment from social media posts.
        
        Args:
            symbol: Asset symbol
            social_posts: List of social media posts
            platform: Social media platform
            
        Returns:
            Aggregated sentiment analysis
        """
        try:
            if not social_posts:
                return {
                    "sentiment": "neutral",
                    "sentiment_score": 0,
                    "confidence": 0,
                    "post_count": 0
                }
            
            sentiments = []
            confidences = []
            
            for post in social_posts:
                blob = TextBlob(post)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                sentiments.append(polarity)
                confidences.append(1 - subjectivity)
            
            # Calculate weighted average
            weights = np.array(confidences)
            if weights.sum() > 0:
                avg_sentiment = np.average(sentiments, weights=weights)
                avg_confidence = np.mean(confidences)
            else:
                avg_sentiment = np.mean(sentiments)
                avg_confidence = 0.5
            
            # Map to sentiment categories
            if avg_sentiment > 0.05:
                sentiment = "bullish"
            elif avg_sentiment < -0.05:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            result = {
                "symbol": symbol,
                "sentiment": sentiment,
                "sentiment_score": float(avg_sentiment),
                "confidence": float(avg_confidence),
                "post_count": len(social_posts),
                "platform": platform,
                "timestamp": datetime.utcnow()
            }
            
            # Save to database
            db_manager.save_market_sentiment(
                symbol=symbol,
                sentiment_score=float(avg_sentiment),
                sentiment_label=sentiment,
                confidence=float(avg_confidence),
                source=platform
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment for {symbol}: {e}")
            return {
                "error": str(e),
                "sentiment": "neutral",
                "sentiment_score": 0,
                "confidence": 0
            }
    
    async def get_aggregated_sentiment(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment from multiple sources.
        
        Args:
            symbol: Asset symbol
            lookback_days: Number of days to look back
            
        Returns:
            Aggregated sentiment data
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=lookback_days)
            
            # Get historical sentiment data
            sentiment_df = db_manager.get_market_sentiment(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if sentiment_df.empty:
                return {
                    "sentiment": "neutral",
                    "sentiment_score": 0,
                    "confidence": 0,
                    "data_points": 0,
                    "trend": "flat"
                }
            
            # Calculate weighted average
            weights = sentiment_df['confidence']
            if weights.sum() > 0:
                avg_sentiment = np.average(sentiment_df['sentiment_score'], weights=weights)
                avg_confidence = np.mean(sentiment_df['confidence'])
            else:
                avg_sentiment = np.mean(sentiment_df['sentiment_score'])
                avg_confidence = 0.5
            
            # Calculate sentiment trend
            if len(sentiment_df) > 1:
                recent_sentiment = sentiment_df.tail(3)['sentiment_score'].mean()
                older_sentiment = sentiment_df.head(3)['sentiment_score'].mean()
                
                if recent_sentiment > older_sentiment + 0.1:
                    trend = "improving"
                elif recent_sentiment < older_sentiment - 0.1:
                    trend = "deteriorating"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Map to sentiment categories
            if avg_sentiment > 0.1:
                sentiment = "bullish"
            elif avg_sentiment < -0.1:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            
            return {
                "symbol": symbol,
                "sentiment": sentiment,
                "sentiment_score": float(avg_sentiment),
                "confidence": float(avg_confidence),
                "data_points": len(sentiment_df),
                "trend": trend,
                "sentiment_volatility": float(sentiment_df['sentiment_score'].std()),
                "latest_sentiment": float(sentiment_df.iloc[-1]['sentiment_score']),
                "latest_confidence": float(sentiment_df.iloc[-1]['confidence'])
            }
            
        except Exception as e:
            logger.error(f"Error getting aggregated sentiment for {symbol}: {e}")
            return {
                "error": str(e),
                "sentiment": "neutral",
                "sentiment_score": 0,
                "confidence": 0
            }
    
    async def analyze_market_momentum(
        self,
        symbol: str,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze market momentum based on price and sentiment.
        
        Args:
            symbol: Asset symbol
            price_data: Price data DataFrame
            
        Returns:
            Momentum analysis results
        """
        try:
            if price_data.empty:
                return {"error": "No price data available"}
            
            # Get sentiment data
            sentiment_data = await self.get_aggregated_sentiment(symbol)
            
            # Calculate price momentum
            returns = price_data['close'].pct_change().dropna()
            
            # Calculate momentum indicators
            momentum_5d = (price_data['close'].iloc[-1] / price_data['close'].iloc[-6] - 1) * 100
            momentum_20d = (price_data['close'].iloc[-1] / price_data['close'].iloc[-21] - 1) * 100
            
            # Calculate volatility
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Combine price and sentiment momentum
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            
            # Calculate combined momentum score
            price_momentum = np.sign(momentum_5d) * min(abs(momentum_5d) / 5, 1)
            combined_momentum = 0.7 * price_momentum + 0.3 * sentiment_score
            
            # Determine momentum category
            if combined_momentum > 0.5:
                momentum_category = "strong_upward"
            elif combined_momentum > 0.1:
                momentum_category = "weak_upward"
            elif combined_momentum < -0.5:
                momentum_category = "strong_downward"
            elif combined_momentum < -0.1:
                momentum_category = "weak_downward"
            else:
                momentum_category = "neutral"
            
            return {
                "symbol": symbol,
                "momentum_category": momentum_category,
                "combined_momentum": float(combined_momentum),
                "price_momentum_5d": float(momentum_5d),
                "price_momentum_20d": float(momentum_20d),
                "volatility": float(volatility),
                "sentiment_contribution": float(sentiment_score),
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market momentum for {symbol}: {e}")
            return {
                "error": str(e),
                "momentum_category": "neutral",
                "combined_momentum": 0
            }
    
    async def get_market_sentiment_summary(
        self,
        symbols: List[str],
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Get market sentiment summary for multiple symbols.
        
        Args:
            symbols: List of symbols to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Market sentiment summary
        """
        tasks = []
        for symbol in symbols:
            task = self.get_aggregated_sentiment(symbol, lookback_days)
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                sentiment = await task
                results[symbol] = sentiment
            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        # Calculate market-wide sentiment
        valid_sentiments = [r for r in results.values() if 'sentiment_score' in r]
        if valid_sentiments:
            avg_market_sentiment = np.mean([r['sentiment_score'] for r in valid_sentiments])
            bullish_count = sum(1 for r in valid_sentiments if r['sentiment_score'] > 0.1)
            bearish_count = sum(1 for r in valid_sentiments if r['sentiment_score'] < -0.1)
            neutral_count = len(valid_sentiments) - bullish_count - bearish_count
            
            market_summary = {
                "market_sentiment_score": float(avg_market_sentiment),
                "bullish_symbols": bullish_count,
                "bearish_symbols": bearish_count,
                "neutral_symbols": neutral_count,
                "total_symbols": len(valid_sentiments)
            }
        else:
            market_summary = {
                "market_sentiment_score": 0,
                "bullish_symbols": 0,
                "bearish_symbols": 0,
                "neutral_symbols": 0,
                "total_symbols": 0
            }
        
        return {
            "individual_sentiments": results,
            "market_summary": market_summary,
            "timestamp": datetime.utcnow()
        }


# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()