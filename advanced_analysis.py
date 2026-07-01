#!/usr/bin/env python3
"""
Advanced Analysis Module - Ultimate Broker Features
Includes: Candlestick patterns, statistical models, bond analysis, yield optimization
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Tuple
import market_data
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

        Delegates to the shared cached market-data layer, which memoizes the
        ^IRX fetch (and persists it across runs) so we don't re-download a
        scalar on every call across the codebase. Falls back to 4.5% if
        unavailable."""
        return market_data.get_risk_free_rate()

    def calculate_statistical_forecast(self, data: pd.DataFrame, periods: int = 60) -> Dict:
        """
        Statistical forecast under Geometric Brownian Motion (GBM).

        Model: log S_t = log S_0 + (μ − σ²/2)·t + σ·W_t,  W_t ~ N(0, t).
        Hence log-returns are i.i.d. Normal with mean (μ − σ²/2) per unit time
        and std σ. The MLE of the log-drift m := μ − σ²/2 is the sample mean
        of the log-returns; σ̂ is their sample std.

            E[S_t]      = S_0 · exp(μ · t)           = S_0 · exp((m + ½σ²)·t)
            Median[S_t] = S_0 · exp(m · t)

        Quantiles are closed-form (LogNormal), so we use them analytically
        instead of running Monte Carlo over a quantity we know exactly.
        """
        if data.empty or len(data) < 30:
            return {}

        try:
            closes = data['Close'].to_numpy(dtype=float)
            closes = closes[np.isfinite(closes) & (closes > 0)]
            if len(closes) < 30:
                return {}

            current_price = float(closes[-1])

            # Log-returns are stationary; sample mean estimates the GBM log-drift m.
            log_returns = np.diff(np.log(closes))

            m_daily = float(np.mean(log_returns))                 # m = μ − σ²/2 (daily)
            sigma_daily = float(np.std(log_returns, ddof=1))
            m_annual = m_daily * 252.0
            sigma_annual = sigma_daily * np.sqrt(252.0)
            mu_annual = m_annual + 0.5 * sigma_annual ** 2        # Itô correction → arithmetic drift

            # Horizon in years
            t = periods / 252.0
            log_var = (sigma_annual ** 2) * t                     # Var[log S_t/S_0]
            log_sd = np.sqrt(log_var)

            # Closed-form LogNormal moments
            expected_price = current_price * np.exp(mu_annual * t)        # E[S_t]
            median_price = current_price * np.exp(m_annual * t)           # Median

            # Closed-form quantiles via the inverse CDF of the LogNormal
            # log S_t ~ N(log S_0 + m·t, σ²·t)
            from scipy.stats import norm
            def lognormal_quantile(p: float) -> float:
                return float(current_price * np.exp(m_annual * t + norm.ppf(p) * log_sd))

            ci_5 = lognormal_quantile(0.05)
            ci_25 = lognormal_quantile(0.25)
            ci_75 = lognormal_quantile(0.75)
            ci_95 = lognormal_quantile(0.95)

            # Linear regression on log-prices (descriptive trend only — note that this
            # is a trend-stationary model, structurally inconsistent with GBM; kept for
            # backward compatibility of the JSON shape).
            log_closes = np.log(closes)
            X = np.arange(len(closes)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(X, log_closes)
            log_forecast = float(lr.predict(np.array([[len(closes) + periods]]))[0])
            linear_forecast = float(np.exp(log_forecast))
            expected_return_linear = (linear_forecast / current_price - 1.0) * 100.0

            expected_return_median = (median_price / current_price - 1.0) * 100.0
            expected_return_mean = (expected_price / current_price - 1.0) * 100.0

            return {
                "current_price": current_price,
                "forecast_linear": linear_forecast,
                "forecast_polynomial": float(median_price),     # backward-compat key
                "expected_return_linear": float(expected_return_linear),
                "expected_return_polynomial": float(expected_return_median),  # backward-compat
                "expected_return_mean": float(expected_return_mean),
                "confidence_interval": float(ci_95 - ci_5),
                "confidence_interval_95": (ci_5, ci_95),
                "confidence_interval_50": (ci_25, ci_75),
                "forecast_periods": periods,
                "annualized_log_drift": float(m_annual * 100),
                "annualized_drift": float(mu_annual * 100),
                "annualized_volatility": float(sigma_annual * 100),
                "method": "GBM_closed_form_LogNormal"
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
            info = market_data.get_info(ticker)

            analysis = {
                "ticker": ticker,
                "name": info.get("longName", ticker),
                "yield_analysis": {},
                "risk_metrics": {},
                "recommendation": "NEUTRAL"
            }

            # Use adjusted close so dividends/coupon distributions are reflected
            data = market_data.get_history(ticker, period="2y", auto_adjust=True)
            if data.empty:
                return analysis

            # Total-return series (price changes already reinvest distributions
            # because auto_adjust=True). For bond ETFs this is the most honest
            # proxy for realized yield: geometric, not arithmetic.
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 30:
                return analysis

            # Geometric annualization avoids Jensen's-gap bias of mean*252
            total_return = float((1.0 + returns).prod() - 1.0)
            years = len(returns) / 252.0
            annual_total_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

            # Current distribution yield (if reported by yfinance) — this is the
            # true coupon/dividend yield, distinct from price total-return above.
            distribution_yield = info.get("yield")
            current_yield = (distribution_yield * 100) if distribution_yield else annual_total_return * 100
            annual_yield = annual_total_return * 100      # report total return as "annual yield"

            # Volatility of daily returns, annualized
            yield_volatility = float(returns.std() * np.sqrt(252) * 100)

            # Sharpe-style risk-adjusted total return
            risk_free_rate = self.get_risk_free_rate()
            excess_return = annual_total_return - risk_free_rate
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

        # Fetch all candidate histories in ONE batched (and cached) request
        # instead of re-downloading each ticker sequentially.
        period = f"{target_years+1}y"
        cand_tickers = [c.get("ticker") for c in candidates if c.get("ticker")]
        histories = market_data.get_histories(cand_tickers, period=period)

        for candidate in candidates:
            ticker = candidate.get("ticker")
            if not ticker:
                continue

            try:
                # Served from the batched cache (no per-ticker re-download)
                data = histories.get(ticker)
                if data is None:
                    data = market_data.get_history(ticker, period=period)

                if data is None or data.empty or len(data) < 60:  # Minimum 60 days
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
    
    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """Euclidean projection of v onto the probability simplex {w ≥ 0, Σw = 1}.

        Implements the O(n log n) algorithm of Wang & Carreira-Perpiñán (2013).
        We project the unconstrained analytical solution onto the long-only
        simplex; this is the standard treatment for no-short-selling MPT and
        avoids degenerate solutions when Σ⁻¹(μ − rf·1) has negative entries.
        """
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho_candidates = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
        if rho_candidates.size == 0:
            # Fallback to equal weights if no valid pivot (degenerate input).
            return np.ones(n) / n
        rho = rho_candidates[-1]
        theta = cssv[rho] / (rho + 1)
        return np.maximum(v - theta, 0.0)

    def calculate_portfolio_optimization(self, holdings: List[Dict], target_return: float = 0.10) -> Dict:
        """
        Mean-variance optimization (Markowitz) — closed-form analytical solution.

        Tangency (max-Sharpe) portfolio with risk-free asset:
            w* ∝ Σ⁻¹ (μ − r_f · 1),  normalized to Σ w = 1.

        Minimum-variance portfolio (no return constraint):
            w* ∝ Σ⁻¹ · 1,  normalized to Σ w = 1.

        Σ is estimated with Ledoit–Wolf shrinkage toward a scaled-identity
        target. Raw sample covariance is poorly conditioned with T ~ 500 days
        and 5–20 assets (Markowitz "garbage in, garbage out"); shrinkage gives
        a well-conditioned Σ̂ with proven MSE-optimality.

        Both analytical solutions can produce negative weights when Σ⁻¹ μ has
        sign-mixed entries; we project to the long-only simplex (no shorting)
        which matches the prior MC behavior and the realistic constraint set.
        """
        if not holdings or len(holdings) < 2:
            return {}

        # Get historical returns for all holdings in ONE batched (cached) request
        opt_tickers = [h.get("ticker") for h in holdings if h.get("ticker")]
        histories = market_data.get_histories(opt_tickers, period="2y", auto_adjust=True)
        returns_data = {}
        for ticker in opt_tickers:
            data = histories.get(ticker)
            if data is not None and not data.empty and 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()
                returns_data[ticker] = returns

        if len(returns_data) < 2:
            return {}

        # Align by date (inner join) — required for a meaningful Σ̂
        returns_df = pd.DataFrame(returns_data).dropna()
        if len(returns_df) < 30:
            return {}

        n_assets = len(returns_df.columns)
        tickers = list(returns_df.columns)

        # Annualized expected return (arithmetic, for portfolio-variance math)
        mean_returns = returns_df.mean() * 252
        mu_vec = mean_returns.values

        # Ledoit–Wolf shrinkage covariance (annualized)
        try:
            lw = LedoitWolf().fit(returns_df.values)
            cov_daily = lw.covariance_
        except Exception as e:
            logger.debug(f"LedoitWolf failed, falling back to sample covariance: {e}")
            cov_daily = returns_df.cov().values
        cov_matrix_arr = cov_daily * 252.0
        cov_matrix = pd.DataFrame(cov_matrix_arr, index=tickers, columns=tickers)

        # Regularize before inversion (tiny diagonal load handles edge cases)
        eps = 1e-10 * np.trace(cov_matrix_arr) / max(n_assets, 1)
        sigma_reg = cov_matrix_arr + eps * np.eye(n_assets)
        try:
            sigma_inv = np.linalg.inv(sigma_reg)
        except np.linalg.LinAlgError:
            sigma_inv = np.linalg.pinv(sigma_reg)

        # Dynamic risk-free rate
        risk_free_rate = self.get_risk_free_rate()

        # Individual asset metrics
        portfolio_metrics = {}
        for ticker in tickers:
            mean_ret = mean_returns[ticker]
            std_ret = float(np.sqrt(cov_matrix.loc[ticker, ticker]))
            sharpe = (mean_ret - risk_free_rate) / std_ret if std_ret > 0 else 0.0

            # Canonical Sortino: target semi-deviation
            # TSD = √( (1/N) · Σ min(R − MAR, 0)² )  (denominator N, not N_negatives)
            daily_returns = returns_df[ticker]
            mar_daily = risk_free_rate / 252.0
            downside = np.minimum(daily_returns - mar_daily, 0.0)
            target_semi_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(252))
            sortino = (mean_ret - risk_free_rate) / target_semi_dev if target_semi_dev > 0 else 0.0

            portfolio_metrics[ticker] = {
                "expected_return": float(mean_ret * 100),
                "volatility": float(std_ret * 100),
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino)
            }

        ones = np.ones(n_assets)

        # ---- Tangency (max-Sharpe) portfolio ---------------------------------
        excess = mu_vec - risk_free_rate * ones
        w_tan_raw = sigma_inv @ excess
        sum_w = w_tan_raw.sum()
        # Only normalize when the tangency sum is strictly positive. When excess
        # returns are net-negative, sum_w <= 0 and dividing by it FLIPS every
        # sign, producing a nonsensical "max-Sharpe" portfolio. Fall back to the
        # minimum-variance solution (Σ⁻¹·1) in that degenerate regime.
        if sum_w > 0 and np.isfinite(sum_w):
            w_tan = w_tan_raw / sum_w
        else:
            w_mv_fallback = sigma_inv @ ones
            sum_mv_fallback = w_mv_fallback.sum()
            if sum_mv_fallback > 0 and np.isfinite(sum_mv_fallback):
                w_tan = w_mv_fallback / sum_mv_fallback
            else:
                w_tan = ones / n_assets
        # Project onto long-only simplex (no shorting, matches prior behaviour)
        max_sharpe_weights = self._project_to_simplex(w_tan)

        # ---- Minimum-variance portfolio --------------------------------------
        w_mv_raw = sigma_inv @ ones
        sum_mv = w_mv_raw.sum()
        if sum_mv != 0 and np.isfinite(sum_mv):
            w_mv = w_mv_raw / sum_mv
        else:
            w_mv = ones / n_assets
        min_vol_weights = self._project_to_simplex(w_mv)

        def _port_stats(w: np.ndarray):
            r = float(w @ mu_vec)
            v = float(np.sqrt(w @ cov_matrix_arr @ w))
            s = (r - risk_free_rate) / v if v > 0 else 0.0
            return r, v, s

        max_sharpe_return, max_sharpe_vol, max_sharpe_ratio = _port_stats(max_sharpe_weights)
        min_vol_return, min_vol_vol, _ = _port_stats(min_vol_weights)

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

