"""
Order management system for live trading.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from ..config import get_config
from ..data.database import db_manager


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Order:
    """Represents a trading order."""
    
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    timestamp: datetime = None
    strategy: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_cancelled(self) -> bool:
        return self.status == OrderStatus.CANCELLED
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'side': self.side.value,
            'order_type': self.order_type.value,
            'status': self.status.value
        }


class OrderManager:
    """Manages trading orders and execution."""
    
    def __init__(self):
        self.config = get_config()
        self.orders: Dict[str, Order] = {}
        self.order_id_counter = 0
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the order manager."""
        try:
            logger.info("Initializing order manager...")
            self.is_initialized = True
            logger.info("Order manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing order manager: {e}")
            return False
    
    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_id_counter += 1
        return f"ORD_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self.order_id_counter:06d}"
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        strategy: str = "",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol: Asset symbol
            side: BUY or SELL
            quantity: Order quantity
            price: Order price (for limit/stop orders)
            order_type: MARKET, LIMIT, STOP, STOP_LIMIT
            strategy: Strategy name placing the order
            metadata: Additional metadata
            
        Returns:
            Order execution result
        """
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'Order manager not initialized'}
            
            # Validate inputs
            if not symbol or quantity <= 0:
                return {'success': False, 'error': 'Invalid symbol or quantity'}
            
            # Create order
            order_id = self.generate_order_id()
            order = Order(
                id=order_id,
                symbol=symbol,
                side=OrderSide(side.upper()),
                order_type=OrderType(order_type.upper()),
                quantity=quantity,
                price=price,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            # Add to orders
            self.orders[order_id] = order
            
            # Simulate order execution (in real implementation, integrate with broker)
            execution_result = await self._simulate_order_execution(order)
            
            if execution_result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = quantity
                order.filled_price = execution_result['filled_price']
                order.commission = execution_result['commission']
            else:
                order.status = OrderStatus.REJECTED
            
            # Log order
            logger.info(
                f"Order {order_id}: {side} {quantity} {symbol} "
                f"at {execution_result.get('filled_price', price)} - "
                f"Status: {order.status.value}"
            )
            
            return {
                'success': execution_result['success'],
                'order_id': order_id,
                'order': order.to_dict(),
                'filled_quantity': order.filled_quantity,
                'filled_price': order.filled_price,
                'commission': order.commission
            }
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _simulate_order_execution(self, order: Order) -> Dict[str, Any]:
        """Simulate order execution (replace with real broker integration)."""
        try:
            # Get current market price
            from ..data.market_data import market_data_provider
            market_data = await market_data_provider.get_realtime_price(order.symbol)
            
            if not market_data:
                return {'success': False, 'error': 'Could not get market data'}
            
            current_price = market_data['price']
            
            # Determine filled price based on order type
            if order.order_type == OrderType.MARKET:
                filled_price = current_price * (1 + 0.0001)  # Small slippage
            elif order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and current_price <= order.price) or \
                   (order.side == OrderSide.SELL and current_price >= order.price):
                    filled_price = order.price
                else:
                    return {'success': False, 'error': 'Limit price not reached'}
            else:
                filled_price = current_price
            
            # Calculate commission
            commission = abs(order.quantity * filled_price * self.config.trading.commission)
            
            return {
                'success': True,
                'filled_price': filled_price,
                'commission': commission
            }
            
        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        try:
            if order_id not in self.orders:
                return {'success': False, 'error': 'Order not found'}
            
            order = self.orders[order_id]
            
            if order.is_filled or order.is_cancelled:
                return {'success': False, 'error': 'Order already filled or cancelled'}
            
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order {order_id} cancelled")
            
            return {'success': True, 'order_id': order_id}
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details."""
        order = self.orders.get(order_id)
        return order.to_dict() if order else None
    
    def get_all_orders(self) -> List[Dict[str, Any]]:
        """Get all orders."""
        return [order.to_dict() for order in self.orders.values()]
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        return [
            order.to_dict() 
            for order in self.orders.values() 
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
    
    def get_filled_orders(self) -> List[Dict[str, Any]]:
        """Get all filled orders."""
        return [
            order.to_dict() 
            for order in self.orders.values() 
            if order.status == OrderStatus.FILLED
        ]
    
    def get_orders_by_strategy(self, strategy: str) -> List[Dict[str, Any]]:
        """Get orders for a specific strategy."""
        return [
            order.to_dict() 
            for order in self.orders.values() 
            if order.strategy == strategy
        ]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get orders for a specific symbol."""
        return [
            order.to_dict() 
            for order in self.orders.values() 
            if order.symbol == symbol
        ]
    
    async def close_position(
        self,
        symbol: str,
        strategy: str = ""
    ) -> Dict[str, Any]:
        """Close position for a symbol."""
        try:
            # Get current market price
            from ..data.market_data import market_data_provider
            market_data = await market_data_provider.get_realtime_price(symbol)
            
            if not market_data:
                return {'success': False, 'error': 'Could not get market data'}
            
            # Place closing order
            result = await self.place_order(
                symbol=symbol,
                side="SELL",
                quantity=1.0,  # This would be actual position size
                order_type="MARKET",
                strategy=strategy
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get order manager status."""
        return {
            'is_initialized': self.is_initialized,
            'total_orders': len(self.orders),
            'open_orders': len(self.get_open_orders()),
            'filled_orders': len(self.get_filled_orders())
        }
    
    def clear_orders(self, status: OrderStatus = None) -> int:
        """Clear orders based on status."""
        if status is None:
            count = len(self.orders)
            self.orders.clear()
            return count
        
        removed = 0
        orders_to_remove = [
            order_id for order_id, order in self.orders.items()
            if order.status == status
        ]
        
        for order_id in orders_to_remove:
            del self.orders[order_id]
            removed += 1
        
        return removed