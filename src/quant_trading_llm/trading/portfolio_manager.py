"""
Portfolio management system for live trading.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger

from ..config import get_config
from ..data.database import db_manager


@dataclass
class Position:
    """Represents a portfolio position."""
    
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    last_update: datetime
    strategy: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PortfolioManager:
    """Manages portfolio positions and performance tracking."""
    
    def __init__(self):
        self.config = get_config()
        self.positions: Dict[str, Position] = {}
        self.cash_balance: float = 0.0
        self.total_value: float = 0.0
        self.is_initialized = False
        self.last_update = None
    
    async def initialize(self, initial_cash: float = 100000.0) -> bool:
        """Initialize portfolio manager."""
        try:
            logger.info(f"Initializing portfolio manager with ${initial_cash:,.2f}")
            
            self.cash_balance = initial_cash
            self.total_value = initial_cash
            self.is_initialized = True
            
            logger.info("Portfolio manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio manager: {e}")
            return False
    
    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str = ""
    ) -> bool:
        """Add or update a position."""
        try:
            if not self.is_initialized:
                return False
            
            if symbol in self.positions:
                # Update existing position
                existing = self.positions[symbol]
                total_quantity = existing.quantity + quantity
                
                if abs(total_quantity) < 1e-6:  # Position closed
                    del self.positions[symbol]
                    logger.info(f"Position closed: {symbol}")
                    return True
                
                # Calculate new average entry price
                total_value = (existing.quantity * existing.entry_price) + (quantity * price)
                new_entry_price = total_value / total_quantity
                
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=total_quantity,
                    entry_price=new_entry_price,
                    current_price=price,
                    market_value=total_quantity * price,
                    unrealized_pnl=total_quantity * (price - new_entry_price),
                    unrealized_pnl_pct=(price - new_entry_price) / new_entry_price * 100,
                    last_update=datetime.utcnow(),
                    strategy=strategy
                )
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                    last_update=datetime.utcnow(),
                    strategy=strategy
                )
            
            logger.info(
                f"Position updated: {symbol} {quantity:,.4f} @ ${price:.2f} "
                f"(strategy: {strategy})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """Remove a position."""
        try:
            if symbol in self.positions:
                del self.positions[symbol]
                logger.info(f"Position removed: {symbol}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing position: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
    
    def get_positions_by_strategy(self, strategy: str) -> Dict[str, Position]:
        """Get positions for a specific strategy."""
        return {
            symbol: position 
            for symbol, position in self.positions.items()
            if position.strategy == strategy
        }
    
    async def update_positions(self) -> bool:
        """Update all positions with current market prices."""
        try:
            if not self.is_initialized:
                return False
            
            from ..data.market_data import market_data_provider
            
            for symbol, position in self.positions.items():
                try:
                    # Get current market price
                    market_data = await market_data_provider.get_realtime_price(symbol)
                    
                    if market_data and 'price' in market_data:
                        current_price = market_data['price']
                        
                        # Update position
                        position.current_price = current_price
                        position.market_value = position.quantity * current_price
                        position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
                        position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                        position.last_update = datetime.utcnow()
                        
                except Exception as e:
                    logger.error(f"Error updating position for {symbol}: {e}")
            
            self.last_update = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return False
    
    def update_cash_balance(self, amount: float) -> bool:
        """Update cash balance."""
        try:
            if not self.is_initialized:
                return False
            
            self.cash_balance += amount
            return True
            
        except Exception as e:
            logger.error(f"Error updating cash balance: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        try:
            positions_value = sum(pos.market_value for pos in self.positions.values())
            return self.cash_balance + positions_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """Calculate portfolio weights."""
        try:
            total_value = self.get_portfolio_value()
            if total_value == 0:
                return {}
            
            weights = {}
            for symbol, position in self.positions.items():
                weights[symbol] = position.market_value / total_value
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating portfolio weights: {e}")
            return {}
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            total_value = self.get_portfolio_value()
            positions_value = sum(pos.market_value for pos in self.positions.values())
            
            # Calculate total PnL
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Get positions by strategy
            positions_by_strategy = {}
            for symbol, position in self.positions.items():
                if position.strategy not in positions_by_strategy:
                    positions_by_strategy[position.strategy] = []
                positions_by_strategy[position.strategy].append(position.to_dict())
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cash_balance': self.cash_balance,
                'positions_value': positions_value,
                'total_value': total_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'number_of_positions': len(self.positions),
                'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
                'positions_by_strategy': positions_by_strategy,
                'portfolio_weights': self.get_portfolio_weights()
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        try:
            if not self.positions:
                return {}
            
            total_value = self.get_portfolio_value()
            if total_value == 0:
                return {}
            
            # Basic metrics
            positions = list(self.positions.values())
            
            # Calculate weighted averages
            weights = [pos.market_value / total_value for pos in positions]
            entry_prices = [pos.entry_price for pos in positions]
            current_prices = [pos.current_price for pos in positions]
            
            # Portfolio return (simplified)
            portfolio_return = sum(w * p.unrealized_pnl_pct / 100 for w, p in zip(weights, positions))
            
            # Position concentration
            max_weight = max(weights) if weights else 0
            
            # Number of positions
            diversification_score = min(len(positions) / 20, 1.0)  # Normalize to 0-1
            
            return {
                'portfolio_return': portfolio_return,
                'max_position_weight': max_weight,
                'diversification_score': diversification_score,
                'number_of_positions': len(positions),
                'cash_ratio': self.cash_balance / total_value,
                'positions_ratio': (total_value - self.cash_balance) / total_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def export_portfolio_report(self) -> Dict[str, Any]:
        """Export comprehensive portfolio report."""
        try:
            summary = self.get_portfolio_summary()
            metrics = self.get_performance_metrics()
            
            return {
                'report_type': 'portfolio_summary',
                'generated_at': datetime.utcnow().isoformat(),
                'summary': summary,
                'metrics': metrics,
                'is_paper_trading': self.config.trading.paper_trading
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio report: {e}")
            return {}
    
    def reset_portfolio(self, initial_cash: float = 100000.0) -> bool:
        """Reset portfolio to initial state."""
        try:
            self.positions.clear()
            self.cash_balance = initial_cash
            self.total_value = initial_cash
            self.last_update = None
            
            logger.info(f"Portfolio reset with ${initial_cash:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting portfolio: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get portfolio manager status."""
        return {
            'is_initialized': self.is_initialized,
            'positions': len(self.positions),
            'cash_balance': self.cash_balance,
            'total_value': self.get_portfolio_value(),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }