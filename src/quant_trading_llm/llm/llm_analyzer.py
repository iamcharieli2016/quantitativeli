"""
LLM analyzer for market analysis and trading decisions.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import openai
import anthropic
from loguru import logger

from ..config import get_config
from .prompts import PromptTemplates


class LLMAnalyzer:
    """Unified LLM analyzer supporting multiple providers."""
    
    def __init__(self):
        self.config = get_config()
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM API clients."""
        try:
            if self.config.llm.openai_api_key:
                self.openai_client = openai.OpenAI(
                    api_key=self.config.llm.openai_api_key
                )
                logger.info("OpenAI client initialized")
            
            if self.config.llm.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config.llm.anthropic_api_key
                )
                logger.info("Anthropic client initialized")
            
            if not self.openai_client and not self.anthropic_client:
                logger.warning("No LLM API keys provided. LLM features will be disabled.")
        
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {e}")
    
    async def _call_openai(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model or self.config.llm.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trader. Analyze the market data and provide structured JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.config.llm.temperature,
                max_tokens=max_tokens or self.config.llm.max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_anthropic(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=model or self.config.llm.anthropic_model,
                max_tokens=max_tokens or self.config.llm.max_tokens,
                temperature=temperature or self.config.llm.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            # Extract JSON from response if it's wrapped in markdown
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.rfind("```")
                content = content[json_start:json_end].strip()
            
            return json.loads(content)
        
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def analyze_market_data(
        self,
        symbol: str,
        price_data: Dict[str, Any],
        technical_indicators: Dict[str, float],
        market_sentiment: Dict[str, Any],
        news_headlines: List[str],
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Analyze market data and provide trading recommendations.
        
        Args:
            symbol: Asset symbol
            price_data: Current price information
            technical_indicators: Technical indicator values
            market_sentiment: Market sentiment data
            news_headlines: Recent news headlines
            provider: LLM provider to use
            
        Returns:
            Trading recommendation with analysis
        """
        prompt = PromptTemplates.market_analysis(
            symbol=symbol,
            price_data=price_data,
            technical_indicators=technical_indicators,
            market_sentiment=market_sentiment,
            news_headlines=news_headlines
        )
        
        try:
            if provider == "openai" and self.openai_client:
                return await self._call_openai(prompt)
            elif provider == "anthropic" and self.anthropic_client:
                return await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Provider {provider} not available")
        
        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 0,
                "reasoning": f"Analysis error: {str(e)}",
                "error": True
            }
    
    async def analyze_sentiment(
        self,
        symbol: str,
        news_text: str,
        social_media_mentions: List[str],
        market_context: Dict[str, Any],
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment from news and social media.
        
        Args:
            symbol: Asset symbol
            news_text: Combined news text
            social_media_mentions: Social media mentions
            market_context: Market context data
            provider: LLM provider to use
            
        Returns:
            Sentiment analysis results
        """
        prompt = PromptTemplates.sentiment_analysis(
            symbol=symbol,
            news_text=news_text,
            social_media_mentions=social_media_mentions,
            market_context=market_context
        )
        
        try:
            if provider == "openai" and self.openai_client:
                return await self._call_openai(prompt)
            elif provider == "anthropic" and self.anthropic_client:
                return await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Provider {provider} not available")
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0,
                "confidence": 0,
                "summary": f"Sentiment analysis error: {str(e)}",
                "error": True
            }
    
    async def assess_risk(
        self,
        portfolio: Dict[str, Any],
        market_conditions: Dict[str, Any],
        correlation_data: Dict[str, float],
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Assess portfolio risk.
        
        Args:
            portfolio: Portfolio composition
            market_conditions: Current market conditions
            correlation_data: Asset correlation data
            provider: LLM provider to use
            
        Returns:
            Risk assessment results
        """
        prompt = PromptTemplates.risk_assessment(
            portfolio=portfolio,
            market_conditions=market_conditions,
            correlation_data=correlation_data
        )
        
        try:
            if provider == "openai" and self.openai_client:
                return await self._call_openai(prompt)
            elif provider == "anthropic" and self.anthropic_client:
                return await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Provider {provider} not available")
        
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {
                "overall_risk": "medium",
                "risk_score": 50,
                "recommendations": [f"Risk assessment error: {str(e)}"],
                "error": True
            }
    
    async def optimize_portfolio(
        self,
        current_portfolio: Dict[str, Any],
        market_outlook: Dict[str, Any],
        constraints: Dict[str, Any],
        provider: str = "openai"
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.
        
        Args:
            current_portfolio: Current portfolio
            market_outlook: Market outlook
            constraints: Investment constraints
            provider: LLM provider to use
            
        Returns:
            Portfolio optimization results
        """
        prompt = PromptTemplates.portfolio_optimization(
            current_portfolio=current_portfolio,
            market_outlook=market_outlook,
            constraints=constraints
        )
        
        try:
            if provider == "openai" and self.openai_client:
                return await self._call_openai(prompt)
            elif provider == "anthropic" and self.anthropic_client:
                return await self._call_anthropic(prompt)
            else:
                raise ValueError(f"Provider {provider} not available")
        
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {
                "optimized_allocation": {},
                "expected_return": 0,
                "expected_volatility": 0,
                "rationale": f"Portfolio optimization error: {str(e)}",
                "error": True
            }
    
    async def batch_analyze(
        self,
        symbols: List[str],
        analysis_type: str = "market_analysis",
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple symbols concurrently.
        
        Args:
            symbols: List of symbols to analyze
            analysis_type: Type of analysis to perform
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dictionary mapping symbols to their analysis results
        """
        tasks = []
        
        for symbol in symbols:
            if analysis_type == "market_analysis":
                task = self.analyze_market_data(
                    symbol=symbol,
                    price_data=kwargs.get('price_data', {}).get(symbol, {}),
                    technical_indicators=kwargs.get('technical_indicators', {}).get(symbol, {}),
                    market_sentiment=kwargs.get('market_sentiment', {}).get(symbol, {}),
                    news_headlines=kwargs.get('news_headlines', {}).get(symbol, [])
                )
            elif analysis_type == "sentiment":
                task = self.analyze_sentiment(
                    symbol=symbol,
                    news_text=kwargs.get('news_text', {}).get(symbol, ""),
                    social_media_mentions=kwargs.get('social_media_mentions', {}).get(symbol, []),
                    market_context=kwargs.get('market_context', {}).get(symbol, {})
                )
            else:
                continue
            
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                result = await task
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        
        return results
    
    def is_available(self, provider: str = None) -> bool:
        """Check if LLM services are available."""
        if provider:
            if provider == "openai":
                return self.openai_client is not None
            elif provider == "anthropic":
                return self.anthropic_client is not None
        
        return self.openai_client is not None or self.anthropic_client is not None


# Global LLM analyzer instance
llm_analyzer = LLMAnalyzer()