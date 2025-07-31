# Quantitative Trading with LLM Integration

A production-ready quantitative trading system that integrates Large Language Models (LLMs) for advanced market analysis and automated trading strategies.

## 🚀 Features

### Core Trading Engine
- **Multi-strategy Support**: Momentum, mean reversion, LLM-enhanced, and risk parity strategies
- **Real-time Data**: Yahoo Finance and cryptocurrency exchange integration
- **Portfolio Management**: Comprehensive position tracking and performance analysis
- **Risk Management**: VaR calculations, position sizing, and automated risk controls

### LLM Integration
- **Market Analysis**: AI-powered sentiment and fundamental analysis
- **Strategy Enhancement**: LLM-assisted signal generation and validation
- **Risk Assessment**: AI-driven risk factor identification
- **Multi-provider Support**: OpenAI GPT-4 and Anthropic Claude integration

### Backtesting Framework
- **Historical Analysis**: Comprehensive backtesting with performance metrics
- **Strategy Optimization**: Parameter tuning and strategy comparison
- **Risk Simulation**: Monte Carlo and stress testing
- **Performance Analytics**: Sharpe ratio, max drawdown, win rate calculations

### Live Trading
- **Paper Trading**: Safe simulation with real market data
- **Live Execution**: Real order placement and execution
- **Position Management**: Real-time P&L tracking and rebalancing
- **Emergency Controls**: Automated stop-loss and risk limits

## 🛠️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   LLM Analysis  │    │   Strategies    │
│                 │    │                 │    │                 │
│ • Yahoo Finance │───→│ • Sentiment     │───→│ • Momentum      │
│ • Binance       │    │ • Fundamentals  │    │ • Mean Reversion│
│ • Real-time     │    │ • Risk Factors  │    │ • LLM-Enhanced  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Risk Manager  │
                    │                 │
                    │ • VaR Calc      │
                    │ • Position Size │
                    │ • Stop Loss     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Trading Engine  │
                    │                 │
                    │ • Live Trading  │
                    │ • Paper Trading │
                    │ • Order Mgmt    │
                    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for development)
- API keys for LLM providers (OpenAI/Anthropic)

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd quant-trading-llm
```

2. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Deploy with Docker**
```bash
./scripts/deploy.sh
```

4. **Access services**
- Main Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## 🎯 Usage

### Backtesting Strategies

```python
from quant_trading_llm.backtest import BacktestEngine
from quant_trading_llm.strategies import MomentumStrategy

# Initialize backtest engine
engine = BacktestEngine()

# Add strategy
strategy = MomentumStrategy()
engine.add_strategy(strategy, symbols=['AAPL', 'GOOGL'], initial_capital=10000)

# Run backtest
results = engine.run_backtest(start_date='2023-01-01', end_date='2023-12-31')
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

### Live Trading

```python
from quant_trading_llm.trading import trading_engine
from quant_trading_llm.strategies import LLMEnhancedStrategy

# Start trading engine
await trading_engine.start_trading()

# Add strategies
await trading_engine.add_strategy(
    strategy_name="llm_enhanced",
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    initial_capital=50000
)

# Monitor positions
positions = trading_engine.get_all_positions()
print(positions)
```

### LLM Analysis

```python
from quant_trading_llm.llm import LLMAnalyzer

analyzer = LLMAnalyzer()

# Market sentiment analysis
sentiment = await analyzer.analyze_market_sentiment('AAPL')
print(f"Sentiment Score: {sentiment.score}")
print(f"Recommendation: {sentiment.recommendation}")

# Risk assessment
risk_factors = await analyzer.assess_risk_factors(['AAPL', 'TSLA'])
for factor in risk_factors:
    print(f"{factor.symbol}: {factor.risk_level}")
```

## 📊 API Endpoints

### Trading API
- `POST /api/v1/trading/start` - Start live trading
- `POST /api/v1/trading/stop` - Stop live trading
- `GET /api/v1/positions` - Get current positions
- `GET /api/v1/orders` - Get order history
- `POST /api/v1/orders` - Place new order

### Strategy API
- `GET /api/v1/strategies` - List available strategies
- `POST /api/v1/strategies/{name}/activate` - Activate strategy
- `DELETE /api/v1/strategies/{name}/deactivate` - Deactivate strategy
- `GET /api/v1/backtest` - Run backtest

### Analysis API
- `POST /api/v1/analysis/sentiment` - Analyze market sentiment
- `POST /api/v1/analysis/risk` - Risk assessment
- `GET /api/v1/analysis/signals` - Get trading signals

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `PAPER_TRADING` | Enable paper trading mode | `true` |
| `INITIAL_CAPITAL` | Initial trading capital | `100000` |
| `COMMISSION` | Trading commission rate | `0.001` |
| `STOP_LOSS_PCT` | Default stop loss percentage | `0.02` |

### Strategy Configuration

Each strategy can be configured via YAML files in the `config/strategies/` directory:

```yaml
# config/strategies/momentum.yml
name: "momentum"
lookback_period: 20
momentum_threshold: 0.05
position_size: 0.1
stop_loss: 0.02
take_profit: 0.05
```

## 📈 Monitoring

### Prometheus Metrics
- `trading_positions_total` - Total number of positions
- `trading_pnl_total` - Total unrealized P&L
- `trading_orders_total` - Total orders executed
- `trading_risk_var_95` - 95% Value at Risk
- `llm_analysis_duration_seconds` - LLM analysis duration

### Grafana Dashboards
- **Trading Overview**: Real-time portfolio performance
- **Strategy Performance**: Individual strategy metrics
- **Risk Monitoring**: Risk metrics and alerts
- **LLM Analysis**: AI analysis results and performance

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Strategy Tests
```bash
pytest tests/strategies/
```

### Backtest Validation
```bash
python scripts/validate_backtests.py
```

## 🚨 Security

### Best Practices
- All API keys stored securely in environment variables
- Database connections use SSL/TLS
- Rate limiting on all endpoints
- Input validation and sanitization
- Secure session management

### Risk Management
- Position size limits
- Stop-loss orders
- Maximum drawdown limits
- Portfolio concentration limits
- Real-time risk monitoring

## 🔄 Deployment

### Development
```bash
docker-compose up -d
```

### Production
```bash
./scripts/deploy.sh
```

### Scaling
```bash
docker-compose up -d --scale trading-app=3
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs)
- [Strategy Development Guide](docs/strategy_development.md)
- [Risk Management Guide](docs/risk_management.md)
- [LLM Integration Guide](docs/llm_integration.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk and is not suitable for every investor. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making investment decisions.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/quant-trading-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/quant-trading-llm/discussions)
- **Documentation**: [Wiki](https://github.com/your-org/quant-trading-llm/wiki)

---

**Built with ❤️ for the quantitative trading community**
