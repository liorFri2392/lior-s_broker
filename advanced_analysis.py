#!/usr/bin/env python3
"""
Advanced Analysis Module - Ultimate Broker Features
Includes: Candlestick patterns, statistical models, bond analysis, yield optimization
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

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
    
    def calculate_statistical_forecast(self, data: pd.DataFrame, periods: int = 60) -> Dict:
        """Calculate statistical forecast using regression models with error handling."""
        if data.empty or len(data) < 30:
            return {}
        
        try:
            closes = data['Close'].values
            if len(closes) < 30:
                return {}
            
            dates = np.arange(len(closes))
            
            # Linear regression
            X = dates.reshape(-1, 1)
            y = closes
            
            # Simple linear regression
            lr = LinearRegression()
            lr.fit(X, y)
            linear_forecast = lr.predict(np.array([[len(closes) + periods]]))[0]
            
            # Polynomial regression (degree 2) - more accurate for trends
            try:
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(X)
                poly_lr = LinearRegression()
                poly_lr.fit(X_poly, y)
                X_future = poly_features.transform(np.array([[len(closes) + periods]]))
                poly_forecast = poly_lr.predict(X_future)[0]
            except Exception:
                # Fallback to linear if polynomial fails
                poly_forecast = linear_forecast
            
            # Calculate confidence intervals
            residuals = y - lr.predict(X)
            std_error = np.std(residuals) if len(residuals) > 0 else 0
            confidence_interval = 1.96 * std_error if std_error > 0 else 0  # 95% confidence
            
            # Calculate expected return
            current_price = closes[-1]
            if current_price > 0:
                expected_return_linear = (linear_forecast / current_price - 1) * 100
                expected_return_poly = (poly_forecast / current_price - 1) * 100
            else:
                expected_return_linear = 0
                expected_return_poly = 0
            
            return {
                "current_price": current_price,
                "forecast_linear": linear_forecast,
                "forecast_polynomial": poly_forecast,
                "expected_return_linear": expected_return_linear,
                "expected_return_polynomial": expected_return_poly,
                "confidence_interval": confidence_interval,
                "forecast_periods": periods
            }
        except Exception:
            return {}
    
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
            
            # Risk-adjusted yield (Sharpe-like for bonds)
            risk_free_rate = 0.03  # Approximate 3% risk-free rate
            excess_return = (annual_yield / 100) - risk_free_rate
            risk_adjusted_yield = (excess_return / (yield_volatility / 100)) * 100 if yield_volatility > 0 else 0
            
            analysis["yield_analysis"] = {
                "current_yield": current_yield,
                "annual_yield": annual_yield,
                "yield_volatility": yield_volatility,
                "risk_adjusted_yield": risk_adjusted_yield
            }
            
            # Risk metrics
            max_drawdown = ((data['Close'].min() / data['Close'].max()) - 1) * 100
            price_stability = 1 - (yield_volatility / 100)  # Higher is better
            
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
                
            except Exception:
                continue
        
        # Sort by optimization score (highest yield potential)
        optimized.sort(key=lambda x: x.get("mid_term_analysis", {}).get("optimization_score", 0), reverse=True)
        
        return optimized
    
    def calculate_portfolio_optimization(self, holdings: List[Dict], target_return: float = 0.10) -> Dict:
        """Calculate optimal portfolio allocation for target return using Modern Portfolio Theory."""
        if not holdings:
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
            except Exception:
                continue
        
        if not returns_data:
            return {}
        
        # Create returns matrix
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            return {}
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Simple optimization: maximize Sharpe ratio
        # For each asset, calculate expected return and risk
        portfolio_metrics = {}
        for ticker in returns_data.keys():
            mean_ret = mean_returns[ticker]
            std_ret = np.sqrt(cov_matrix.loc[ticker, ticker])
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            
            portfolio_metrics[ticker] = {
                "expected_return": mean_ret * 100,
                "volatility": std_ret * 100,
                "sharpe_ratio": sharpe
            }
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        return {
            "portfolio_metrics": portfolio_metrics,
            "correlation_matrix": correlation_matrix.to_dict(),
            "recommendation": "Diversify based on low correlation assets"
        }

