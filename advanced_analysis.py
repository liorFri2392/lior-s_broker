#!/usr/bin/env python3
"""
Advanced Analysis Module - Ultimate Broker Features
Includes: Candlestick patterns, statistical models, bond analysis, yield optimization
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedAnalyzer:
    """Advanced statistical and technical analysis for ultimate broker functionality."""
    
    def __init__(self):
        pass
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns for technical analysis."""
        if data.empty or len(data) < 3:
            return []
        
        patterns = []
        closes = data['Close'].values
        opens = data['Open'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        # Get last 5 candles for pattern detection
        recent_data = data.tail(5)
        
        # Bullish patterns
        if len(recent_data) >= 3:
            # Hammer pattern
            last = recent_data.iloc[-1]
            body = abs(last['Close'] - last['Open'])
            lower_shadow = min(last['Open'], last['Close']) - last['Low']
            upper_shadow = last['High'] - max(last['Open'], last['Close'])
            
            if lower_shadow > 2 * body and upper_shadow < body and last['Close'] > last['Open']:
                patterns.append({
                    "pattern": "HAMMER",
                    "signal": "BULLISH",
                    "strength": "STRONG",
                    "description": "Potential reversal to the upside"
                })
            
            # Engulfing pattern
            if len(recent_data) >= 2:
                prev = recent_data.iloc[-2]
                if (prev['Close'] < prev['Open'] and  # Previous bearish
                    last['Close'] > last['Open'] and  # Current bullish
                    last['Open'] < prev['Close'] and
                    last['Close'] > prev['Open']):
                    patterns.append({
                        "pattern": "BULLISH_ENGULFING",
                        "signal": "BULLISH",
                        "strength": "STRONG",
                        "description": "Bullish reversal pattern"
                    })
            
            # Bearish patterns
            if lower_shadow < body and upper_shadow > 2 * body and last['Close'] < last['Open']:
                patterns.append({
                    "pattern": "HANGING_MAN",
                    "signal": "BEARISH",
                    "strength": "MODERATE",
                    "description": "Potential reversal to the downside"
                })
        
        return patterns
    
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate from 3-month Treasury yield (^IRX).
        Falls back to 4.5% if unavailable."""
        try:
            irx = yf.Ticker("^IRX")
            hist = irx.history(period="5d")
            if hist is not None and not hist.empty and 'Close' in hist.columns:
                # ^IRX is quoted as percentage (e.g. 4.5 means 4.5%)
                rate = float(hist['Close'].iloc[-1]) / 100.0
                if 0 < rate < 0.15:  # Sanity: between 0% and 15%
                    return rate
        except Exception as e:
            logger.debug(f"Failed to fetch risk-free rate: {e}")
        return 0.045  # Default fallback

    def calculate_statistical_forecast(self, data: pd.DataFrame, periods: int = 60) -> Dict:
        """
        Calculate statistical forecast using Geometric Brownian Motion (GBM).
        
        Uses log returns (stationary) instead of raw price levels (non-stationary).
        GBM: S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
        where mu = drift (annualized mean log return), sigma = annualized volatility.
        Monte Carlo simulation provides confidence intervals.
        """
        if data.empty or len(data) < 30:
            return {}
        
        try:
            closes = data['Close'].values
            if len(closes) < 30:
                return {}
            
            current_price = closes[-1]
            if current_price <= 0:
                return {}
            
            # Use LOG RETURNS (stationary, proper for financial time series)
            log_returns = np.diff(np.log(closes))
            
            # Annualized parameters
            mu_daily = np.mean(log_returns)
            sigma_daily = np.std(log_returns, ddof=1)
            mu_annual = mu_daily * 252
            sigma_annual = sigma_daily * np.sqrt(252)
            
            # Time horizon in years
            t = periods / 252.0
            
            # GBM expected price: E[S(t)] = S(0) * exp(mu * t)
            # GBM median price: S(0) * exp((mu - sigma^2/2) * t)
            drift = mu_annual - 0.5 * sigma_annual ** 2
            expected_price = current_price * np.exp(mu_annual * t)
            median_price = current_price * np.exp(drift * t)
            
            # Monte Carlo simulation for confidence intervals (1000 paths)
            n_simulations = 1000
            np.random.seed(42)  # Reproducible
            random_shocks = np.random.normal(0, 1, n_simulations)
            simulated_log_returns = drift * t + sigma_annual * np.sqrt(t) * random_shocks
            simulated_prices = current_price * np.exp(simulated_log_returns)
            
            # Confidence intervals from simulation
            ci_5 = float(np.percentile(simulated_prices, 5))
            ci_25 = float(np.percentile(simulated_prices, 25))
            ci_75 = float(np.percentile(simulated_prices, 75))
            ci_95 = float(np.percentile(simulated_prices, 95))
            
            # Expected returns (using median for "expected_return_polynomial" for backward compat)
            expected_return_median = (median_price / current_price - 1) * 100
            expected_return_mean = (expected_price / current_price - 1) * 100
            
            # Linear regression on log prices for trend-based forecast (backward compat)
            log_closes = np.log(closes)
            X = np.arange(len(closes)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X, log_closes)
            log_forecast = lr.predict(np.array([[len(closes) + periods]]))[0]
            linear_forecast = np.exp(log_forecast)
            expected_return_linear = (linear_forecast / current_price - 1) * 100
            
            # Cap at realistic range
            max_realistic_return = 200
            min_realistic_return = -80
            expected_return_linear = max(min_realistic_return, min(max_realistic_return, expected_return_linear))
            expected_return_median = max(min_realistic_return, min(max_realistic_return, expected_return_median))
            
            return {
                "current_price": float(current_price),
                "forecast_linear": float(linear_forecast),
                "forecast_polynomial": float(median_price),  # Backward compat: use GBM median
                "expected_return_linear": float(expected_return_linear),
                "expected_return_polynomial": float(expected_return_median),  # Backward compat
                "expected_return_mean": float(expected_return_mean),
                "confidence_interval": float(ci_95 - ci_5),
                "confidence_interval_95": (float(ci_5), float(ci_95)),
                "confidence_interval_50": (float(ci_25), float(ci_75)),
                "forecast_periods": periods,
                "annualized_drift": float(mu_annual * 100),
                "annualized_volatility": float(sigma_annual * 100),
                "method": "GBM_MonteCarlo"
            }
        except Exception as e:
            logger.warning(f"Failed to calculate statistical forecast: {e}")
            return {}
    
    def _calculate_temporal_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate proper temporal max drawdown (peak-to-trough in chronological order)."""
        if len(prices) < 2:
            return 0.0
        cumulative_max = np.maximum.accumulate(prices)
        drawdowns = (prices - cumulative_max) / cumulative_max
        return float(np.min(drawdowns) * 100)

    def analyze_bonds(self, ticker: str) -> Dict:
        """Advanced bond analysis focusing on yield and risk."""
        try:
            bond = yf.Ticker(ticker)
            info = bond.info
            
            analysis = {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "yield_analysis": {},
                "risk_metrics": {},
                "recommendation": "NEUTRAL"
            }
            
            # Get historical data
            data = bond.history(period="2y")
            if data.empty:
                return analysis
            
            # Calculate yield metrics
            returns = data['Close'].pct_change().dropna()
            annual_yield = returns.mean() * 252 * 100
            
            # Current yield (if available)
            current_yield = info.get("yield", 0) * 100 if info.get("yield") else annual_yield
            
            # Yield stability
            yield_volatility = returns.std() * np.sqrt(252) * 100
            
            # Dynamic risk-free rate
            risk_free_rate = self.get_risk_free_rate()
            excess_return = (annual_yield / 100) - risk_free_rate
            risk_adjusted_yield = (excess_return / (yield_volatility / 100)) if yield_volatility > 0 else 0
            
            analysis["yield_analysis"] = {
                "current_yield": current_yield,
                "annual_yield": annual_yield,
                "yield_volatility": yield_volatility,
                "risk_adjusted_yield": risk_adjusted_yield,
                "risk_free_rate_used": risk_free_rate * 100
            }
            
            # Risk metrics — proper temporal max drawdown
            max_drawdown = self._calculate_temporal_max_drawdown(data['Close'].values)
            price_stability = max(0.0, 1 - (yield_volatility / 100))  # Clamp >= 0
            
            analysis["risk_metrics"] = {
                "max_drawdown": max_drawdown,
                "price_stability": price_stability,
                "volatility": yield_volatility
            }
            
            # Recommendation based on yield and risk
            score = 50
            if risk_adjusted_yield > 2:
                score += 20
                analysis["recommendation"] = "STRONG BUY"
            elif risk_adjusted_yield > 1:
                score += 10
                analysis["recommendation"] = "BUY"
            elif risk_adjusted_yield < -1:
                score -= 20
                analysis["recommendation"] = "AVOID"
            
            if current_yield > 5 and yield_volatility < 10:
                score += 15
            elif current_yield < 2:
                score -= 10
            
            analysis["score"] = max(0, min(100, score))
            
            return analysis
            
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def optimize_mid_term_yield(self, candidates: List[Dict], target_years: int = 3) -> List[Dict]:
        """Optimize for highest yield in mid-term (3-5 years) using statistical models."""
        if not candidates:
            return []
        
        optimized = []
        
        for candidate in candidates:
            ticker = candidate.get("ticker")
            if not ticker:
                continue
            
            try:
                stock = yf.Ticker(ticker)
                # Get enough data for analysis
                data = stock.history(period=f"{target_years+1}y")
                
                if data.empty or len(data) < 60:  # Minimum 60 days
                    continue
                
                # Calculate historical returns
                returns = data['Close'].pct_change().dropna()
                if len(returns) < 30:
                    continue
                
                # Statistical projections
                forecast = self.calculate_statistical_forecast(data, periods=target_years * 252)
                
                # Expected mid-term return
                if forecast and forecast.get("expected_return_polynomial") is not None:
                    expected_return = forecast.get("expected_return_polynomial", 0)
                else:
                    # Fallback to historical average annualized
                    expected_return = returns.mean() * 252 * 100 if len(returns) > 0 else 0
                
                # Risk-adjusted return
                volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
                sharpe_ratio = (expected_return / 100) / (volatility / 100) if volatility > 0 else 0
                
                # Calculate probability of positive return
                positive_days = (returns > 0).sum() if len(returns) > 0 else 0
                win_rate = (positive_days / len(returns)) * 100 if len(returns) > 0 else 50
                
                # Score based on yield optimization
                score = expected_return  # Base score is expected return
                score += sharpe_ratio * 10  # Bonus for risk-adjusted returns
                score += (win_rate - 50) * 0.5  # Bonus for consistency
                
                candidate["mid_term_analysis"] = {
                    "expected_return": expected_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "win_rate": win_rate,
                    "forecast": forecast,
                    "optimization_score": score
                }
                
                optimized.append(candidate)
                
            except Exception as e:
                logger.debug(f"Failed to optimize candidate {candidate.get('ticker', 'unknown')}: {e}")
                continue
        
        # Sort by optimization score (highest yield potential)
        optimized.sort(key=lambda x: x.get("mid_term_analysis", {}).get("optimization_score", 0), reverse=True)
        
        return optimized
    
    def calculate_portfolio_optimization(self, holdings: List[Dict], target_return: float = 0.10) -> Dict:
        """
        Calculate optimal portfolio allocation using Modern Portfolio Theory.
        
        Performs actual mean-variance optimization:
        1. Monte Carlo simulation to map the efficient frontier
        2. Maximum Sharpe Ratio portfolio (tangency portfolio)
        3. Minimum Variance portfolio
        4. Optimal weights for each asset
        """
        if not holdings or len(holdings) < 2:
            return {}
        
        # Get historical returns for all holdings
        returns_data = {}
        for holding in holdings:
            ticker = holding.get("ticker")
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="2y")
                if not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    returns_data[ticker] = returns
            except Exception as e:
                logger.debug(f"Failed to get returns data for {ticker}: {e}")
                continue
        
        if len(returns_data) < 2:
            return {}
        
        # Create returns matrix aligned by date (proper join)
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            return {}
        
        n_assets = len(returns_df.columns)
        tickers = list(returns_df.columns)
        
        # Annualized parameters
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Dynamic risk-free rate
        risk_free_rate = self.get_risk_free_rate()
        
        # Individual asset metrics
        portfolio_metrics = {}
        for ticker in tickers:
            mean_ret = mean_returns[ticker]
            std_ret = np.sqrt(cov_matrix.loc[ticker, ticker])
            sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0
            
            # Sortino ratio (downside deviation only)
            daily_returns = returns_df[ticker]
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else std_ret
            sortino = (mean_ret - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            portfolio_metrics[ticker] = {
                "expected_return": float(mean_ret * 100),
                "volatility": float(std_ret * 100),
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino)
            }
        
        # Monte Carlo optimization: find max Sharpe and min variance portfolios
        n_simulations = 5000
        results = np.zeros((n_simulations, 3 + n_assets))  # return, vol, sharpe, weights...
        
        for i in range(n_simulations):
            # Random weights (sum to 1, no shorting)
            weights = np.random.dirichlet(np.ones(n_assets))
            
            # Portfolio return and volatility
            port_return = np.dot(weights, mean_returns.values)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            port_sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
            
            results[i, 0] = port_return
            results[i, 1] = port_vol
            results[i, 2] = port_sharpe
            results[i, 3:] = weights
        
        # Maximum Sharpe Ratio portfolio
        max_sharpe_idx = np.argmax(results[:, 2])
        max_sharpe_weights = results[max_sharpe_idx, 3:]
        max_sharpe_return = results[max_sharpe_idx, 0]
        max_sharpe_vol = results[max_sharpe_idx, 1]
        max_sharpe_ratio = results[max_sharpe_idx, 2]
        
        # Minimum Variance portfolio
        min_vol_idx = np.argmin(results[:, 1])
        min_vol_weights = results[min_vol_idx, 3:]
        min_vol_return = results[min_vol_idx, 0]
        min_vol_vol = results[min_vol_idx, 1]
        
        # Current weights (from holdings)
        total_value = sum(h.get("current_value", h.get("quantity", 0) * h.get("current_price", 0)) for h in holdings)
        current_weights = {}
        for h in holdings:
            t = h.get("ticker")
            if t in tickers:
                val = h.get("current_value", h.get("quantity", 0) * h.get("current_price", 0))
                current_weights[t] = val / total_value if total_value > 0 else 1.0 / n_assets
        
        # Current portfolio metrics
        cw = np.array([current_weights.get(t, 0) for t in tickers])
        if np.sum(cw) > 0:
            cw = cw / np.sum(cw)  # Normalize
            current_return = float(np.dot(cw, mean_returns.values))
            current_vol = float(np.sqrt(np.dot(cw.T, np.dot(cov_matrix.values, cw))))
            current_sharpe = float((current_return - risk_free_rate) / current_vol) if current_vol > 0 else 0
        else:
            current_return = 0
            current_vol = 0
            current_sharpe = 0
        
        # Build optimal weights dict
        optimal_weights = {tickers[i]: float(max_sharpe_weights[i]) for i in range(n_assets)}
        min_var_weights_dict = {tickers[i]: float(min_vol_weights[i]) for i in range(n_assets)}
        
        # Correlation matrix
        correlation_matrix = returns_df.corr()
        
        return {
            "portfolio_metrics": portfolio_metrics,
            "correlation_matrix": correlation_matrix.to_dict(),
            "risk_free_rate": float(risk_free_rate * 100),
            "current_portfolio": {
                "weights": {t: float(w) for t, w in current_weights.items()},
                "expected_return": float(current_return * 100),
                "volatility": float(current_vol * 100),
                "sharpe_ratio": float(current_sharpe)
            },
            "max_sharpe_portfolio": {
                "weights": optimal_weights,
                "expected_return": float(max_sharpe_return * 100),
                "volatility": float(max_sharpe_vol * 100),
                "sharpe_ratio": float(max_sharpe_ratio)
            },
            "min_variance_portfolio": {
                "weights": min_var_weights_dict,
                "expected_return": float(min_vol_return * 100),
                "volatility": float(min_vol_vol * 100)
            },
            "recommendation": self._generate_optimization_recommendation(
                current_weights, optimal_weights, tickers, current_sharpe, max_sharpe_ratio
            )
        }
    
    def _generate_optimization_recommendation(self, current_weights: Dict, optimal_weights: Dict, 
                                                tickers: List[str], current_sharpe: float, 
                                                optimal_sharpe: float) -> str:
        """Generate actionable recommendation from optimization results."""
        if optimal_sharpe <= current_sharpe * 1.05:
            return "Portfolio is near-optimal. No significant rebalancing needed."
        
        changes = []
        for t in tickers:
            curr = current_weights.get(t, 0)
            opt = optimal_weights.get(t, 0)
            diff = opt - curr
            if abs(diff) > 0.05:  # Only flag >5% differences
                direction = "Increase" if diff > 0 else "Decrease"
                changes.append(f"{direction} {t} from {curr*100:.1f}% to {opt*100:.1f}%")
        
        if changes:
            return f"Rebalance to improve Sharpe from {current_sharpe:.2f} to {optimal_sharpe:.2f}: " + "; ".join(changes)
        return "Portfolio allocation is close to optimal."

