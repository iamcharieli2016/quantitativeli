"""
Portfolio optimization utilities for risk management and allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cvxpy as cp
from datetime import datetime, timedelta
from loguru import logger

from ..config import get_config


class PortfolioOptimizer:
    """Advanced portfolio optimization using various techniques."""
    
    def __init__(self):
        self.config = get_config()
    
    def optimize_mean_variance(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_tolerance: float = 1.0,
        constraints: Dict[str, float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Mean-variance optimization.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_tolerance: Risk tolerance parameter
            constraints: Additional constraints
            
        Returns:
            Optimal weights and optimization results
        """
        try:
            n_assets = len(expected_returns)
            weights = cp.Variable(n_assets)
            
            # Objective: maximize risk-adjusted return
            portfolio_return = expected_returns @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_variance)
            
            # Constraints
            constraints_list = [
                cp.sum(weights) == 1,
                weights >= 0  # Long only
            ]
            
            if constraints:
                if 'max_weight' in constraints:
                    constraints_list.append(weights <= constraints['max_weight'])
                if 'min_weight' in constraints:
                    constraints_list.append(weights >= constraints['min_weight'])
            
            # Solve optimization
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                results = {
                    'status': 'optimal',
                    'expected_return': float(expected_returns @ optimal_weights),
                    'portfolio_variance': float(portfolio_variance.value),
                    'sharpe_ratio': float(expected_returns @ optimal_weights / np.sqrt(portfolio_variance.value))
                }
                
                return optimal_weights, results
            else:
                logger.warning(f"Mean-variance optimization failed: {problem.status}")
                return np.array([]), {'status': str(problem.status)}
                
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {e}")
            return np.array([]), {'error': str(e)}
    
    def optimize_risk_parity(
        self,
        covariance_matrix: np.ndarray,
        target_risk: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Risk parity optimization with optional target risk.
        
        Args:
            covariance_matrix: Covariance matrix
            target_risk: Target portfolio risk (optional)
            
        Returns:
            Optimal risk parity weights and results
        """
        try:
            n_assets = covariance_matrix.shape[0]
            weights = cp.Variable(n_assets)
            
            # Portfolio variance
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            portfolio_volatility = cp.sqrt(portfolio_variance)
            
            # Risk contribution
            marginal_contrib = cp.multiply(covariance_matrix @ weights, weights)
            risk_contrib = cp.multiply(marginal_contrib, 1 / portfolio_volatility)
            
            # Objective: minimize deviation from equal risk contribution
            target_contrib = 1.0 / n_assets
            objective = cp.Minimize(
                cp.sum_squares(risk_contrib - target_contrib)
            )
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0.01,  # Minimum 1% allocation
                weights <= 0.3   # Maximum 30% allocation
            ]
            
            if target_risk:
                constraints.append(portfolio_volatility <= target_risk)
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                results = {
                    'status': 'optimal',
                    'portfolio_volatility': float(portfolio_volatility.value),
                    'diversification_ratio': self._calculate_diversification_ratio(
                        optimal_weights, covariance_matrix
                    ),
                    'effective_number_assets': 1 / np.sum(optimal_weights ** 2)
                }
                
                return optimal_weights, results
            else:
                logger.warning(f"Risk parity optimization failed: {problem.status}")
                return np.array([]), {'status': str(problem.status)}
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return np.array([]), {'error': str(e)}
    
    def optimize_maximum_diversification(
        self,
        covariance_matrix: np.ndarray,
        expected_returns: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Maximum diversification portfolio optimization.
        
        Args:
            covariance_matrix: Covariance matrix
            expected_returns: Expected returns (optional)
            
        Returns:
            Optimal diversification weights and results
        """
        try:
            n_assets = covariance_matrix.shape[0]
            weights = cp.Variable(n_assets)
            
            # Calculate portfolio volatility and weighted sum of volatilities
            volatilities = np.sqrt(np.diag(covariance_matrix))
            portfolio_volatility = cp.sqrt(cp.quad_form(weights, covariance_matrix))
            weighted_volatility = volatilities @ weights
            
            # Objective: maximize diversification ratio
            objective = cp.Maximize(weighted_volatility / portfolio_volatility)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= 0.3
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                diversification_ratio = float(weighted_volatility.value / portfolio_volatility.value)
                
                results = {
                    'status': 'optimal',
                    'diversification_ratio': diversification_ratio,
                    'portfolio_volatility': float(portfolio_volatility.value)
                }
                
                return optimal_weights, results
            else:
                logger.warning(f"Maximum diversification optimization failed: {problem.status}")
                return np.array([]), {'status': str(problem.status)}
                
        except Exception as e:
            logger.error(f"Error in maximum diversification optimization: {e}")
            return np.array([]), {'error': str(e)}
    
    def optimize_black_litterman(
        self,
        market_weights: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 2.5,
        confidence: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Black-Litterman portfolio optimization.
        
        Args:
            market_weights: Market capitalization weights
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            confidence: Confidence in views
            
        Returns:
            Optimal Black-Litterman weights and results
        """
        try:
            n_assets = len(market_weights)
            
            # Calculate implied returns (equilibrium returns)
            implied_returns = risk_aversion * covariance_matrix @ market_weights
            
            # For simplicity, use market weights as optimal
            # In practice, incorporate investor views
            optimal_weights = market_weights
            
            results = {
                'status': 'optimal',
                'implied_returns': implied_returns,
                'sharpe_ratio': float(implied_returns @ optimal_weights / np.sqrt(optimal_weights @ covariance_matrix @ optimal_weights))
            }
            
            return optimal_weights, results
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            return np.array([]), {'error': str(e)}
    
    def optimize_robust_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        uncertainty_level: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Robust portfolio optimization with uncertainty.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            uncertainty_level: Level of parameter uncertainty
            
        Returns:
            Robust optimal weights and results
        """
        try:
            n_assets = len(expected_returns)
            weights = cp.Variable(n_assets)
            
            # Robust objective: worst-case optimization
            portfolio_return = expected_returns @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            
            # Uncertainty-adjusted objective
            uncertainty_penalty = uncertainty_level * cp.norm(weights, 1)
            objective = cp.Maximize(
                portfolio_return - uncertainty_penalty - 0.5 * portfolio_variance
            )
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,
                weights >= 0,
                weights <= 0.3
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                results = {
                    'status': 'optimal',
                    'expected_return': float(expected_returns @ optimal_weights),
                    'portfolio_variance': float(portfolio_variance.value),
                    'uncertainty_penalty': float(uncertainty_penalty.value)
                }
                
                return optimal_weights, results
            else:
                logger.warning(f"Robust optimization failed: {problem.status}")
                return np.array([]), {'status': str(problem.status)}
                
        except Exception as e:
            logger.error(f"Error in robust optimization: {e}")
            return np.array([]), {'error': str(e)}
    
    def calculate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Calculate efficient frontier points.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            num_points: Number of frontier points
            
        Returns:
            Returns, volatilities, and weights for efficient frontier
        """
        try:
            n_assets = len(expected_returns)
            returns_range = np.linspace(
                expected_returns.min(),
                expected_returns.max(),
                num_points
            )
            
            efficient_returns = []
            efficient_volatilities = []
            efficient_weights = []
            
            for target_return in returns_range:
                weights = cp.Variable(n_assets)
                
                # Objective: minimize variance for target return
                objective = cp.Minimize(cp.quad_form(weights, covariance_matrix))
                
                constraints = [
                    expected_returns @ weights == target_return,
                    cp.sum(weights) == 1,
                    weights >= 0,
                    weights <= 0.3
                ]
                
                problem = cp.Problem(objective, constraints)
                problem.solve()
                
                if problem.status == cp.OPTIMAL:
                    efficient_returns.append(float(target_return))
                    efficient_volatilities.append(float(np.sqrt(cp.quad_form(weights, covariance_matrix).value)))
                    efficient_weights.append(weights.value)
            
            return np.array(efficient_returns), np.array(efficient_volatilities), efficient_weights
            
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return np.array([]), np.array([]), []
    
    def _calculate_diversification_ratio(
        self,
        weights: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> float:
        """Calculate diversification ratio."""
        try:
            volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_vol = np.sum(weights * volatilities)
            portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
            
            return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 1.0
    
    def optimize_with_transaction_costs(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: np.ndarray,
        transaction_cost: float = 0.001,
        constraints: Dict[str, float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Portfolio optimization with transaction costs.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            current_weights: Current portfolio weights
            transaction_cost: Transaction cost per trade
            constraints: Additional constraints
            
        Returns:
            Optimal weights considering transaction costs
        """
        try:
            n_assets = len(expected_returns)
            weights = cp.Variable(n_assets)
            
            # Transaction cost (L1 norm of changes)
            transaction_cost_term = transaction_cost * cp.norm(weights - current_weights, 1)
            
            # Objective
            portfolio_return = expected_returns @ weights
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            objective = cp.Maximize(
                portfolio_return - 0.5 * portfolio_variance - transaction_cost_term
            )
            
            # Constraints
            constraints_list = [
                cp.sum(weights) == 1,
                weights >= 0
            ]
            
            if constraints:
                if 'max_weight' in constraints:
                    constraints_list.append(weights <= constraints['max_weight'])
            
            # Solve
            problem = cp.Problem(objective, constraints_list)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                results = {
                    'status': 'optimal',
                    'expected_return': float(expected_returns @ optimal_weights),
                    'portfolio_variance': float(portfolio_variance.value),
                    'transaction_cost': float(transaction_cost_term.value)
                }
                
                return optimal_weights, results
            else:
                logger.warning(f"Optimization with transaction costs failed: {problem.status}")
                return np.array([]), {'status': str(problem.status)}
                
        except Exception as e:
            logger.error(f"Error in transaction cost optimization: {e}")
            return np.array([]), {'error': str(e)}


# Global optimizer instance
portfolio_optimizer = PortfolioOptimizer()