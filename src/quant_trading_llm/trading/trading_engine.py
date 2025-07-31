"""
Live trading engine for quantitative strategies.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from ..config import get_config
from ..strategies.strategy_engine import strategy_engine
from ..risk.risk_manager import risk_manager
from .order_manager import OrderManager
from .portfolio_manager import PortfolioManager


class TradingEngine:
    """Live trading engine for executing quantitative strategies."""
    
    def __init__(self):
        self.config = get_config()
        self.is_running = False
        self.order_manager = OrderManager()
        self.portfolio_manager = PortfolioManager()
        self.active_strategies: Dict[str, Dict[str, Any]] = {}
        self.last_update = None
    
    async def start_trading(self) -> bool:
        """Start the live trading engine."""
        try:
            if self.is_running:
                logger.warning("Trading engine already running")
                return False
            
            logger.info("Starting live trading engine...")
            
            # Initialize components
            await self.order_manager.initialize()
            await self.portfolio_manager.initialize()
            
            self.is_running = True
            logger.info("Live trading engine started successfully")
            
            # Start main trading loop
            asyncio.create_task(self._main_trading_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            return False
    
    async def stop_trading(self) -> bool:
        """Stop the live trading engine."""
        try:
            if not self.is_running:
                logger.warning("Trading engine not running")
                return False
            
            logger.info("Stopping live trading engine...")
            
            # Close all positions safely
            await self._close_all_positions()
            
            self.is_running = False
            logger.info("Live trading engine stopped")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            return False
    
    async def add_strategy(
        self,
        strategy_name: str,
        symbols: List[str],
        initial_capital: float = 10000.0
    ) -> bool:
        """Add a strategy to the live trading engine."""
        try:
            strategy = strategy_engine.get_strategy(strategy_name)
            if not strategy:
                logger.error(f"Strategy {strategy_name} not found")
                return False
            
            self.active_strategies[strategy_name] = {
                'symbols': symbols,
                'initial_capital': initial_capital,
                'allocated_capital': initial_capital,
                'current_positions': {},
                'last_signal_time': None
            }
            
            logger.info(
                f"Added strategy {strategy_name} for symbols: {symbols} "
                f"with capital: ${initial_capital:,.2f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False
    
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy from the live trading engine."""
        try:
            if strategy_name not in self.active_strategies:
                logger.warning(f"Strategy {strategy_name} not active")
                return False
            
            # Close all positions for this strategy
            await self._close_strategy_positions(strategy_name)
            
            del self.active_strategies[strategy_name]
            logger.info(f"Removed strategy {strategy_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing strategy: {e}")
            return False
    
    async def _main_trading_loop(self):
        """Main trading loop that runs continuously."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Update portfolio positions
                await self.portfolio_manager.update_positions()
                
                # Generate and execute signals for active strategies
                for strategy_name, config in self.active_strategies.items():
                    await self._execute_strategy_signals(
                        strategy_name, config, current_time
                    )
                
                # Risk monitoring
                await self._monitor_risks()
                
                self.last_update = current_time
                
                # Wait for next cycle (adjust based on trading frequency)
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _execute_strategy_signals(
        self,
        strategy_name: str,
        config: Dict[str, Any],
        current_time: datetime
    ):
        """Execute signals for a specific strategy."""
        try:
            strategy = strategy_engine.get_strategy(strategy_name)
            if not strategy:
                return
            
            # Get current market data for symbols
            from ..data.market_data import market_data_provider
            
            for symbol in config['symbols']:
                # Get latest market data
                data = await market_data_provider.get_stock_data(
                    symbol=symbol,
                    period="1mo",
                    interval="1d"
                )
                
                if data.empty:
                    continue
                
                # Get current position
                current_position = config['current_positions'].get(symbol)
                
                # Generate signals
                signals = await strategy.generate_signals(symbol, data, current_position)
                
                for signal in signals:
                    if self._should_execute_signal(signal, config):
                        await self._execute_trade(signal, strategy_name, config)
                        
        except Exception as e:
            logger.error(f"Error executing strategy signals: {e}")
    
    def _should_execute_signal(self, signal, config: Dict[str, Any]) -> bool:
        """Check if signal should be executed based on risk and other factors."""
        try:
            # Check risk limits
            if not risk_manager.is_running:
                return False
            
            # Check signal confidence
            if signal.confidence < strategy.config.min_confidence:
                return False
            
            # Check if we have sufficient capital
            allocated_capital = config['allocated_capital']
            if allocated_capital <= 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal execution: {e}")
            return False
    
    async def _execute_trade(self, signal, strategy_name: str, config: Dict[str, Any]):
        """Execute a single trade based on signal."""
        try:
            # Get current price
            from ..data.market_data import market_data_provider
            current_price = await market_data_provider.get_realtime_price(signal.symbol)
            
            if not current_price:
                logger.warning(f"Could not get price for {signal.symbol}")
                return
            
            price = current_price['price']
            
            # Calculate position size
            strategy = strategy_engine.get_strategy(strategy_name)
            position_size = strategy.calculate_position_size(config['allocated_capital'])
            
            # Execute order
            order_result = await self.order_manager.place_order(
                symbol=signal.symbol,
                side=signal.signal_type.value,
                quantity=position_size / price,
                price=price,
                strategy=strategy_name
            )
            
            if order_result['success']:
                # Update position tracking
                if signal.signal_type.value == 'BUY':
                    config['current_positions'][signal.symbol] = {
                        'quantity': position_size / price,
                        'entry_price': price,
                        'timestamp': datetime.utcnow()
                    }
                elif signal.signal_type.value == 'SELL':
                    config['current_positions'].pop(signal.symbol, None)
                
                logger.info(
                    f"Executed {signal.signal_type.value} for {signal.symbol} "
                    f"at ${price:.2f} (strategy: {strategy_name})"
                )
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _monitor_risks(self):
        """Monitor risk levels and take action if necessary."""
        try:
            # Get current positions
            positions = self.portfolio_manager.get_positions()
            
            # Get current prices for risk calculation
            from ..data.market_data import market_data_provider
            prices = {}
            for symbol in positions.keys():
                price_data = await market_data_provider.get_realtime_price(symbol)
                if price_data:
                    prices[symbol] = price_data['price']
            
            # Check risk alerts
            alerts = risk_manager.check_risk_alerts(prices)
            
            for alert in alerts:
                if alert['severity'] == 'CRITICAL':
                    logger.critical(f"Risk alert: {alert['message']}")
                    await self._emergency_stop()
                elif alert['severity'] == 'HIGH':
                    logger.warning(f"Risk alert: {alert['message']}")
                    await self._reduce_exposure()
            
        except Exception as e:
            logger.error(f"Error monitoring risks: {e}")
    
    async def _emergency_stop(self):
        """Emergency stop all trading activities."""
        logger.critical("Emergency stop triggered - closing all positions")
        await self.stop_trading()
    
    async def _reduce_exposure(self):
        """Reduce portfolio exposure due to risk alerts."""
        logger.warning("Reducing portfolio exposure due to risk alerts")
        # Implement gradual position reduction
        pass
    
    async def _close_all_positions(self):
        """Close all open positions."""
        try:
            positions = self.portfolio_manager.get_positions()
            
            for symbol, position in positions.items():
                if position['quantity'] > 0:
                    await self.order_manager.close_position(symbol)
            
            logger.info("All positions closed")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    async def _close_strategy_positions(self, strategy_name: str):
        """Close all positions for a specific strategy."""
        try:
            if strategy_name not in self.active_strategies:
                return
            
            config = self.active_strategies[strategy_name]
            
            for symbol, position in config['current_positions'].items():
                await self.order_manager.close_position(symbol)
            
            config['current_positions'].clear()
            logger.info(f"Closed all positions for strategy {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error closing strategy positions: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading engine status."""
        return {
            'is_running': self.is_running,
            'active_strategies': len(self.active_strategies),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'order_manager_status': self.order_manager.get_status(),
            'portfolio_status': self.portfolio_manager.get_status()
        }
    
    def get_strategy_status(self, strategy_name: str) -> Dict[str, Any]:
        """Get status for a specific strategy."""
        if strategy_name not in self.active_strategies:
            return {'error': 'Strategy not active'}
        
        config = self.active_strategies[strategy_name]
        return {
            'strategy_name': strategy_name,
            'symbols': config['symbols'],
            'allocated_capital': config['allocated_capital'],
            'current_positions': config['current_positions'],
            'last_signal_time': config['last_signal_time']
        }
    
    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all current positions across all strategies."""
        all_positions = {}
        
        for strategy_name, config in self.active_strategies.items():
            for symbol, position in config['current_positions'].items():
                if symbol not in all_positions:
                    all_positions[symbol] = []
                all_positions[symbol].append({
                    'strategy': strategy_name,
                    'position': position
                })
        
        return all_positions


# Global trading engine instance
trading_engine = TradingEngine()