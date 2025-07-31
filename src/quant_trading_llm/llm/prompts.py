"""
Prompt templates for LLM market analysis.
"""

from typing import Dict, Any, List
from datetime import datetime


class PromptTemplates:
    """Collection of prompt templates for market analysis."""
    
    @staticmethod
    def market_analysis(
        symbol: str,
        price_data: Dict[str, Any],
        technical_indicators: Dict[str, float],
        market_sentiment: Dict[str, Any],
        news_headlines: List[str]
    ) -> str:
        """Generate prompt for comprehensive market analysis."""
        
        return f"""
        You are an expert quantitative trader with deep knowledge of financial markets, 
        technical analysis, and market psychology. Analyze the following market data 
        for {symbol} and provide a comprehensive trading recommendation.

        MARKET DATA:
        - Current Price: ${price_data.get('current_price', 0):.2f}
        - Price Change (24h): {price_data.get('price_change_24h', 0):.2f}%
        - Volume (24h): {price_data.get('volume_24h', 0):,.0f}
        - Market Cap: ${price_data.get('market_cap', 0):,.0f}

        TECHNICAL INDICATORS:
        - RSI (14): {technical_indicators.get('rsi', 0):.2f}
        - MACD: {technical_indicators.get('macd', 0):.4f}
        - MACD Signal: {technical_indicators.get('macd_signal', 0):.4f}
        - 20-day SMA: ${technical_indicators.get('sma_20', 0):.2f}
        - 50-day SMA: ${technical_indicators.get('sma_50', 0):.2f}
        - Bollinger Upper: ${technical_indicators.get('bb_upper', 0):.2f}
        - Bollinger Lower: ${technical_indicators.get('bb_lower', 0):.2f}

        MARKET SENTIMENT:
        - Overall Sentiment: {market_sentiment.get('sentiment', 'Neutral')}
        - Sentiment Score: {market_sentiment.get('score', 0):.2f}
        - Fear & Greed Index: {market_sentiment.get('fear_greed', 50)}

        RECENT NEWS HEADLINES:
        {chr(10).join(f"- {headline}" for headline in news_headlines[-5:])}

        Based on this analysis, provide:
        1. A clear trading recommendation (BUY, SELL, or HOLD) with confidence level (0-100%)
        2. Key technical levels (support and resistance)
        3. Risk factors and potential catalysts
        4. Recommended position size and stop-loss levels
        5. Time horizon for the trade (short-term, medium-term, long-term)

        Format your response as JSON with the following structure:
        {{
            "recommendation": "BUY|SELL|HOLD",
            "confidence": 0-100,
            "target_price": float,
            "stop_loss": float,
            "position_size": float (as percentage of portfolio),
            "time_horizon": "short-term|medium-term|long-term",
            "key_levels": {{
                "support": [float],
                "resistance": [float]
            }},
            "risk_factors": [string],
            "catalysts": [string],
            "reasoning": string
        }}
        """
    
    @staticmethod
    def sentiment_analysis(
        symbol: str,
        news_text: str,
        social_media_mentions: List[str],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate prompt for sentiment analysis."""
        
        return f"""
        Analyze the market sentiment for {symbol} based on the following information:

        NEWS AND MARKET CONTEXT:
        {news_text}

        SOCIAL MEDIA MENTIONS:
        {chr(10).join(f"- {mention}" for mention in social_media_mentions[:10])}

        MARKET CONTEXT:
        - Current Price: ${market_context.get('price', 0):.2f}
        - Price Change (7d): {market_context.get('price_change_7d', 0):.2f}%
        - Volume Trend: {market_context.get('volume_trend', 'Neutral')}
        - Market Cap Rank: {market_context.get('market_cap_rank', 'Unknown')}

        Provide a comprehensive sentiment analysis considering:
        1. Overall market sentiment (bullish, bearish, or neutral)
        2. Sentiment score (-1 to 1, where -1 is extremely bearish, 1 is extremely bullish)
        3. Key sentiment drivers
        4. Risk of sentiment reversal
        5. Impact of recent news and events

        Format your response as JSON:
        {{
            "sentiment": "bullish|bearish|neutral",
            "sentiment_score": -1.0 to 1.0,
            "confidence": 0-100,
            "key_drivers": [string],
            "risk_factors": [string],
            "news_impact": string,
            "social_sentiment": string,
            "summary": string
        }}
        """
    
    @staticmethod
    def risk_assessment(
        portfolio: Dict[str, Any],
        market_conditions: Dict[str, Any],
        correlation_data: Dict[str, float]
    ) -> str:
        """Generate prompt for risk assessment."""
        
        return f"""
        Perform a comprehensive risk assessment for the following portfolio:

        PORTFOLIO COMPOSITION:
        Total Value: ${portfolio.get('total_value', 0):,.2f}
        Cash Position: ${portfolio.get('cash', 0):,.2f} ({portfolio.get('cash_percentage', 0):.1f}%)
        
        Holdings:
        {chr(10).join(f"- {holding['symbol']}: ${holding['value']:,.2f} ({holding['percentage']:.1f}%)" 
                     for holding in portfolio.get('holdings', []))}

        MARKET CONDITIONS:
        - VIX: {market_conditions.get('vix', 0):.2f}
        - Market Volatility: {market_conditions.get('market_volatility', 'Low')}
        - Economic Indicators: {market_conditions.get('economic_indicators', 'Stable')}
        - Geopolitical Risk: {market_conditions.get('geopolitical_risk', 'Low')}

        CORRELATION DATA:
        {chr(10).join(f"- {pair}: {corr:.2f}" for pair, corr in correlation_data.items())}

        Assess the following risk factors:
        1. Portfolio concentration risk
        2. Market volatility impact
        3. Correlation risk between holdings
        4. Liquidity risk
        5. Tail risk scenarios
        6. Maximum drawdown potential

        Provide recommendations for risk mitigation.

        Format your response as JSON:
        {{
            "overall_risk": "low|medium|high|extreme",
            "risk_score": 0-100,
            "var_95": float,
            "max_drawdown": float,
            "concentration_risk": string,
            "correlation_risk": string,
            "liquidity_risk": string,
            "tail_risk_scenarios": [string],
            "recommendations": [string],
            "hedge_strategies": [string]
        }}
        """
    
    @staticmethod
    def portfolio_optimization(
        current_portfolio: Dict[str, Any],
        market_outlook: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> str:
        """Generate prompt for portfolio optimization."""
        
        return f"""
        Optimize the following portfolio based on current market outlook:

        CURRENT PORTFOLIO:
        {chr(10).join(f"- {asset['symbol']}: {asset['weight']:.1f}% (expected_return: {asset['expected_return']:.1f}%, 
                       volatility: {asset['volatility']:.1f}%)" 
                     for asset in current_portfolio.get('assets', []))}

        MARKET OUTLOOK:
        - Economic Cycle: {market_outlook.get('economic_cycle', 'Neutral')}
        - Interest Rate Trend: {market_outlook.get('interest_rates', 'Stable')}
        - Inflation Expectations: {market_outlook.get('inflation', 'Moderate')}
        - Sector Performance: {market_outlook.get('sector_performance', 'Mixed')}

        CONSTRAINTS:
        - Maximum single position: {constraints.get('max_position', 10)}%
        - Minimum cash buffer: {constraints.get('min_cash', 5)}%
        - Risk tolerance: {constraints.get('risk_tolerance', 'Moderate')}
        - Time horizon: {constraints.get('time_horizon', 'Medium')}

        Provide an optimized allocation considering:
        1. Risk-return optimization
        2. Diversification benefits
        3. Transaction costs
        4. Tax implications
        5. Rebalancing frequency

        Format your response as JSON:
        {{
            "optimized_allocation": {{
                "symbol": weight_percentage
            }},
            "expected_return": float,
            "expected_volatility": float,
            "sharpe_ratio": float,
            "rebalancing_frequency": string,
            "rationale": string,
            "implementation_steps": [string]
        }}
        """