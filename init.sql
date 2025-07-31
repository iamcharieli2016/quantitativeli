-- Initialize database for quantitative trading system

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema
CREATE SCHEMA IF NOT EXISTS trading;

-- Market data tables
CREATE TABLE IF NOT EXISTS trading.market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(15,8),
    high_price DECIMAL(15,8),
    low_price DECIMAL(15,8),
    close_price DECIMAL(15,8),
    volume BIGINT,
    adjusted_close DECIMAL(15,8),
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timestamp, source)
);

-- Price data table (real-time)
CREATE TABLE IF NOT EXISTS trading.price_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(15,8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    volume BIGINT,
    bid DECIMAL(15,8),
    ask DECIMAL(15,8),
    source VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Technical indicators
CREATE TABLE IF NOT EXISTS trading.technical_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(15,8),
    parameters JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, indicator_name, timestamp)
);

-- Sentiment data
CREATE TABLE IF NOT EXISTS trading.sentiment_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    sentiment_score DECIMAL(5,4),
    sentiment_label VARCHAR(20),
    confidence DECIMAL(5,4),
    source VARCHAR(50),
    text_content TEXT,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading positions
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15,8) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    current_price DECIMAL(15,8),
    unrealized_pnl DECIMAL(15,8),
    unrealized_pnl_pct DECIMAL(5,4),
    strategy VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading orders
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
    quantity DECIMAL(15,8) NOT NULL,
    price DECIMAL(15,8),
    stop_price DECIMAL(15,8),
    status VARCHAR(20) NOT NULL,
    filled_quantity DECIMAL(15,8) DEFAULT 0,
    filled_price DECIMAL(15,8),
    commission DECIMAL(10,4) DEFAULT 0,
    strategy VARCHAR(50),
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS trading.strategy_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    date DATE NOT NULL,
    total_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    win_rate DECIMAL(5,4),
    num_trades INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(strategy_name, symbol, date)
);

-- Risk metrics
CREATE TABLE IF NOT EXISTS trading.risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    portfolio_value DECIMAL(15,8),
    cash_balance DECIMAL(15,8),
    total_exposure DECIMAL(15,8),
    var_95 DECIMAL(15,8),
    var_99 DECIMAL(15,8),
    max_position_weight DECIMAL(5,4),
    diversification_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- LLM analysis results
CREATE TABLE IF NOT EXISTS trading.llm_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    analysis_type VARCHAR(50),
    prompt TEXT,
    response TEXT,
    sentiment_score DECIMAL(5,4),
    recommendation VARCHAR(20),
    confidence DECIMAL(5,4),
    model VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON trading.market_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_price_data_symbol_timestamp ON trading.price_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_name ON trading.technical_indicators(symbol, indicator_name);
CREATE INDEX IF NOT EXISTS idx_sentiment_data_symbol_timestamp ON trading.sentiment_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading.positions(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading.orders(status);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_name_date ON trading.strategy_performance(strategy_name, date);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON trading.risk_metrics(timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW trading.latest_prices AS (
    SELECT DISTINCT ON (symbol) 
        symbol,
        price,
        timestamp,
        source
    FROM trading.price_data
    ORDER BY symbol, timestamp DESC
);

CREATE OR REPLACE VIEW trading.portfolio_summary AS (
    SELECT 
        symbol,
        SUM(quantity) as total_quantity,
        AVG(entry_price) as avg_entry_price,
        SUM(unrealized_pnl) as total_unrealized_pnl,
        strategy,
        MAX(timestamp) as last_update
    FROM trading.positions
    GROUP BY symbol, strategy
);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA trading TO quant_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO quant_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO quant_user;

-- Insert sample data for testing
INSERT INTO trading.market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume, source) VALUES
('AAPL', NOW() - INTERVAL '1 day', 150.00, 155.00, 149.50, 152.50, 1000000, 'Yahoo'),
('GOOGL', NOW() - INTERVAL '1 day', 2700.00, 2750.00, 2695.00, 2730.00, 500000, 'Yahoo'),
('MSFT', NOW() - INTERVAL '1 day', 380.00, 385.00, 378.00, 382.50, 800000, 'Yahoo')
ON CONFLICT DO NOTHING;