"""
Value at Risk (VaR) calculation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import warnings
from loguru import logger


class VaRCalculator:
    """Value at Risk calculation using multiple methods."""
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
    
    def calculate_historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0,
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate VaR using historical simulation.
        
        Args:
            returns: Historical returns array
            portfolio_value: Portfolio value
            confidence_level: Confidence level (e.g., 0.95)
            holding_period: Holding period in days
            
        Returns:
            VaR value and statistics
        """
        try:
            if len(returns) < 30:
                warnings.warn("Insufficient data for reliable VaR calculation")
                return 0.0, {'error': 'Insufficient data'}
            
            # Scale returns for holding period
            scaled_returns = returns * np.sqrt(holding_period)
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_return = np.percentile(scaled_returns, var_percentile)
            var_amount = abs(var_return) * portfolio_value
            
            # Additional statistics
            stats = {
                'var_return': float(var_return),
                'var_amount': float(var_amount),
                'confidence_level': confidence_level,
                'holding_period': holding_period,
                'sample_size': len(returns),
                'method': 'historical'
            }
            
            return var_amount, stats
            
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            return 0.0, {'error': str(e)}
    
    def calculate_parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        distribution: str = 'normal'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate VaR using parametric method.
        
        Args:
            returns: Historical returns array
            portfolio_value: Portfolio value
            confidence_level: Confidence level
            holding_period: Holding period in days
            distribution: Distribution type ('normal', 't', 'skewed_t')
            
        Returns:
            VaR value and statistics
        """
        try:
            if len(returns) < 30:
                return 0.0, {'error': 'Insufficient data'}
            
            # Calculate distribution parameters
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Scale for holding period
            scaled_mean = mean_return * holding_period
            scaled_vol = volatility * np.sqrt(holding_period)
            
            # Calculate VaR based on distribution
            if distribution == 'normal':
                z_score = stats.norm.ppf(1 - confidence_level)
                var_return = scaled_mean + z_score * scaled_vol
                
            elif distribution == 't':
                # Fit t-distribution
                params = stats.t.fit(returns)
                df, loc, scale = params
                t_score = stats.t.ppf(1 - confidence_level, df)
                var_return = scaled_mean + t_score * scaled_vol
                
            else:  # normal as fallback
                z_score = stats.norm.ppf(1 - confidence_level)
                var_return = scaled_mean + z_score * scaled_vol
            
            var_amount = abs(var_return) * portfolio_value
            
            stats_dict = {
                'var_return': float(var_return),
                'var_amount': float(var_amount),
                'confidence_level': confidence_level,
                'holding_period': holding_period,
                'mean_return': float(mean_return),
                'volatility': float(volatility),
                'distribution': distribution,
                'method': 'parametric'
            }
            
            return var_amount, stats_dict
            
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            return 0.0, {'error': str(e)}
    
    def calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        simulations: int = 10000
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            returns: Historical returns array
            portfolio_value: Portfolio value
            confidence_level: Confidence level
            holding_period: Holding period in days
            simulations: Number of Monte Carlo simulations
            
        Returns:
            VaR value and statistics
        """
        try:
            if len(returns) < 30:
                return 0.0, {'error': 'Insufficient data'}
            
            # Fit distribution parameters
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            # Monte Carlo simulation
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(
                mean_return * holding_period,
                volatility * np.sqrt(holding_period),
                simulations
            )
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_return = np.percentile(simulated_returns, var_percentile)
            var_amount = abs(var_return) * portfolio_value
            
            # Additional statistics
            stats = {
                'var_return': float(var_return),
                'var_amount': float(var_amount),
                'confidence_level': confidence_level,
                'holding_period': holding_period,
                'simulations': simulations,
                'mean_return': float(mean_return),
                'volatility': float(volatility),
                'method': 'monte_carlo'
            }
            
            return var_amount, stats
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0, {'error': str(e)}
    
    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        portfolio_value: float = 1.0,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Historical returns array
            portfolio_value: Portfolio value
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            Expected shortfall value and statistics
        """
        try:
            if len(returns) < 30:
                return 0.0, {'error': 'Insufficient data'}
            
            if method == 'historical':
                # Historical ES
                var_percentile = (1 - confidence_level) * 100
                var_return = np.percentile(returns, var_percentile)
                
                # Calculate average of returns worse than VaR
                tail_returns = returns[returns <= var_return]
                if len(tail_returns) > 0:
                    es_return = np.mean(tail_returns)
                else:
                    es_return = var_return
                
            elif method == 'parametric':
                # Parametric ES (normal distribution)
                mean_return = np.mean(returns)
                volatility = np.std(returns)
                
                alpha = 1 - confidence_level
                z_score = stats.norm.ppf(alpha)
                phi_z = stats.norm.pdf(z_score)
                
                es_return = mean_return - volatility * phi_z / alpha
                
            else:
                return 0.0, {'error': 'Invalid method'}
            
            es_amount = abs(es_return) * portfolio_value
            
            stats_dict = {
                'es_return': float(es_return),
                'es_amount': float(es_amount),
                'confidence_level': confidence_level,
                'method': method,
                'var_return': float(np.percentile(returns, (1 - confidence_level) * 100))
            }
            
            return es_amount, stats_dict
            
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return 0.0, {'error': str(e)}
    
    def calculate_component_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate component VaR for portfolio positions.
        
        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights array
            confidence_level: Confidence level
            
        Returns:
            Component VaR for each asset
        """
        try:
            if returns.empty or len(weights) != returns.shape[1]:
                return {}
            
            # Portfolio returns
            portfolio_returns = returns.values @ weights
            
            # Portfolio VaR
            var_percentile = (1 - confidence_level) * 100
            portfolio_var_return = np.percentile(portfolio_returns, var_percentile)
            
            # Calculate marginal VaR
            covariance_matrix = returns.cov().values
            portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights)
            marginal_var = (portfolio_var_return / portfolio_volatility) * (covariance_matrix @ weights)
            
            # Component VaR
            component_var = weights * marginal_var
            
            return {
                returns.columns[i]: float(component_var[i])
                for i in range(len(returns.columns))
            }
            
        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return {}
    
    def calculate_stress_test_var(
        self,
        returns: np.ndarray,
        stress_scenarios: Dict[str, Dict[str, float]],
        portfolio_value: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate VaR under stress test scenarios.
        
        Args:
            returns: Historical returns array
            stress_scenarios: Dictionary of stress scenarios
            portfolio_value: Portfolio value
            
        Returns:
            VaR values under stress scenarios
        """
        try:
            stress_results = {}
            
            for scenario_name, scenario_params in stress_scenarios.items():
                # Apply stress parameters to returns
                stressed_returns = returns.copy()
                
                if 'volatility_shock' in scenario_params:
                    shock_factor = scenario_params['volatility_shock']
                    stressed_returns *= shock_factor
                
                if 'correlation_shock' in scenario_params:
                    # Simplified correlation shock (would need full correlation matrix)
                    stressed_returns *= scenario_params['correlation_shock']
                
                # Calculate VaR under stress
                var_amount, _ = self.calculate_historical_var(
                    stressed_returns,
                    portfolio_value,
                    confidence_level=0.99
                )
                
                stress_results[scenario_name] = var_amount
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error calculating stress test VaR: {e}")
            return {}
    
    def calculate_portfolio_var(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level VaR.
        
        Args:
            returns_df: DataFrame with asset returns
            weights: Portfolio weights
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            Portfolio VaR and component VaR
        """
        try:
            # Portfolio returns
            portfolio_returns = returns_df.values @ weights
            
            # Calculate VaR based on method
            if method == 'historical':
                var_amount, stats = self.calculate_historical_var(
                    portfolio_returns,
                    portfolio_value=1.0,
                    confidence_level=confidence_level
                )
            elif method == 'parametric':
                var_amount, stats = self.calculate_parametric_var(
                    portfolio_returns,
                    portfolio_value=1.0,
                    confidence_level=confidence_level
                )
            elif method == 'monte_carlo':
                var_amount, stats = self.calculate_monte_carlo_var(
                    portfolio_returns,
                    portfolio_value=1.0,
                    confidence_level=confidence_level
                )
            else:
                return {'error': 'Invalid method'}
            
            # Component VaR
            component_var = self.calculate_component_var(
                returns_df, weights, confidence_level
            )
            
            return {
                'portfolio_var': var_amount,
                'component_var': component_var,
                'method': method,
                'confidence_level': confidence_level,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return {'error': str(e)}
    
    def generate_var_report(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 1.0
    ) -> Dict[str, Any]:
        """Generate comprehensive VaR report."""
        try:
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'portfolio_value': portfolio_value,
                'assets': list(returns_df.columns),
                'weights': weights.tolist(),
                'var_results': {}
            }
            
            # Calculate VaR for different confidence levels and methods
            for confidence in self.confidence_levels:
                report['var_results'][f'var_{int(confidence*100)}'] = {}
                
                for method in ['historical', 'parametric', 'monte_carlo']:
                    try:
                        var_result = self.calculate_portfolio_var(
                            returns_df, weights, confidence, method
                        )
                        report['var_results'][f'var_{int(confidence*100)}'][method] = var_result
                    except Exception as e:
                        report['var_results'][f'var_{int(confidence*100)}'][method] = {'error': str(e)}
            
            # Expected shortfall
            for confidence in self.confidence_levels:
                es_amount, es_stats = self.calculate_expected_shortfall(
                    returns_df.values @ weights,
                    portfolio_value,
                    confidence
                )
                report['var_results'][f'es_{int(confidence*100)}'] = {
                    'amount': es_amount,
                    'statistics': es_stats
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating VaR report: {e}")
            return {'error': str(e)}