"""
Configuration management for the quantitative trading system.
"""

import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field, validator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="quant_trading", env="DB_NAME")
    username: str = Field(default="quant_user", env="DB_USERNAME")
    password: str = Field(default="", env="DB_PASSWORD")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis configuration for caching and message queue."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def connection_string(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class LLMConfig(BaseSettings):
    """LLM provider configuration."""
    
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v


class TradingConfig(BaseSettings):
    """Trading system configuration."""
    
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")  # 10% of portfolio
    max_daily_loss: float = Field(default=0.02, env="MAX_DAILY_LOSS")  # 2% of portfolio
    max_drawdown: float = Field(default=0.05, env="MAX_DRAWDOWN")  # 5% of portfolio
    commission: float = Field(default=0.001, env="COMMISSION")  # 0.1% per trade
    slippage: float = Field(default=0.0005, env="SLIPPAGE")  # 0.05% slippage
    
    @validator("max_position_size", "max_daily_loss", "max_drawdown")
    def validate_percentages(cls, v):
        if not 0 < v < 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v


class DataConfig(BaseSettings):
    """Data sources configuration."""
    
    yfinance_enabled: bool = Field(default=True, env="YFINANCE_ENABLED")
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret: Optional[str] = Field(default=None, env="BINANCE_SECRET")
    data_retention_days: int = Field(default=365, env="DATA_RETENTION_DAYS")
    update_interval_minutes: int = Field(default=5, env="DATA_UPDATE_INTERVAL")


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8>}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        env="LOG_FORMAT"
    )
    file_path: str = Field(default="logs/quant_trading.log", env="LOG_FILE_PATH")
    rotation: str = Field(default="1 day", env="LOG_ROTATION")
    retention: str = Field(default="30 days", env="LOG_RETENTION")


class Config(BaseSettings):
    """Main configuration class."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Celery Configuration
    celery_broker_url: Optional[str] = Field(default=None, env="CELERY_BROKER_URL")
    celery_result_backend: Optional[str] = Field(default=None, env="CELERY_RESULT_BACKEND")
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator("api_workers")
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("API workers must be at least 1")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    config = Config()
    return config