#!/usr/bin/env python3
"""
Risk Management Module - Automatic stop-loss, take-profit, and risk monitoring
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import market_data
import portfolio_io
import logging

logger = logging.getLogger(__name__)


def fit_garch_1_1(returns: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Maximum-likelihood fit of GARCH(1,1) to a 1-D returns series.

    GARCH(1,1):
        r_t      = μ + ε_t,         ε_t = σ_t · z_t,   z_t ~ N(0, 1)
        σ²_t     = ω + α·ε²_{t-1} + β·σ²_{t-1}
    with stationarity α + β < 1 and positivity ω > 0, α ≥ 0, β ≥ 0.

    Returns
    -------
    params : dict with mu, omega, alpha, beta, unconditional_variance,
             sigma2_now (current filtered conditional variance, σ²_T),
             sigma2_next (one-step-ahead forecast, σ²_{T+1}),
             persistence (α+β), converged (bool).
    sigma2 : array of length len(returns) — filtered conditional variances.
    """
    from scipy.optimize import minimize

    r = np.asarray(returns, dtype=float)
    n = r.shape[0]
    if n < 50:
        raise ValueError("GARCH fit requires at least 50 observations")

    sample_var = float(np.var(r, ddof=1))
    sample_mean = float(np.mean(r))

    def _filter(theta: np.ndarray):
        mu, omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.9999:
            return None
        eps = r - mu
        sigma2 = np.empty(n, dtype=float)
        # Initialize with unconditional variance under the parameter values
        unconditional = omega / (1.0 - alpha - beta)
        sigma2[0] = max(unconditional, 1e-12)
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            if not np.isfinite(sigma2[t]) or sigma2[t] <= 0:
                return None
        return eps, sigma2

    def _neg_log_lik(theta):
        out = _filter(theta)
        if out is None:
            return 1e10
        eps, sigma2 = out
        # Gaussian log-lik: −½ Σ ( log(2π σ²) + ε²/σ² )
        return float(0.5 * np.sum(np.log(2.0 * np.pi * sigma2) + (eps ** 2) / sigma2))

    # Sensible starting values: typical equity GARCH(1,1) lands near
    # (ω, α, β) = (γ·σ²·(1−α−β), 0.05, 0.92) with low γ. Try a few seeds and
    # take the best.
    seeds = [
        (sample_mean, 0.1 * sample_var, 0.05, 0.90),
        (sample_mean, 0.05 * sample_var, 0.10, 0.85),
        (sample_mean, 0.2 * sample_var, 0.15, 0.70),
    ]
    best = None
    for x0 in seeds:
        try:
            res = minimize(
                _neg_log_lik, x0,
                method='L-BFGS-B',
                bounds=[(None, None), (1e-12, None), (0.0, 0.999), (0.0, 0.999)],
                options={'maxiter': 200},
            )
            if res.success or res.fun < 1e9:
                if best is None or res.fun < best.fun:
                    best = res
        except Exception:
            continue
    if best is None:
        raise RuntimeError("GARCH(1,1) optimizer failed to converge from any starting point")

    mu, omega, alpha, beta = best.x
    eps, sigma2 = _filter(best.x)
    sigma2_now = float(sigma2[-1])
    # One-step-ahead forecast
    sigma2_next = float(omega + alpha * eps[-1] ** 2 + beta * sigma2_now)
    unconditional = float(omega / (1.0 - alpha - beta)) if (alpha + beta) < 1 else float('inf')

    return (
        {
            "mu": float(mu),
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(alpha + beta),
            "unconditional_variance": unconditional,
            "sigma2_now": sigma2_now,
            "sigma2_next": sigma2_next,
            "converged": bool(best.success),
            "log_likelihood": float(-best.fun),
            "n_obs": int(n),
        },
        sigma2,
    )


class RiskManager:
    """Manage portfolio risk with stop-loss, take-profit, and alerts."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.stop_loss_percent = 10.0  # Default 10% stop-loss
        self.take_profit_percent = 20.0  # Default 20% take-profit
        self.max_position_size = 0.15  # Max 15% per position
        self.max_sector_exposure = 0.30  # Max 30% per sector
    
    def load_portfolio(self) -> Dict:
        """Load portfolio from JSON file."""
        return portfolio_io.load_portfolio(self.portfolio_file)
    
    def check_stop_loss_take_profit(self, portfolio: Dict = None) -> List[Dict]:
        """
        Check if any holdings hit stop-loss or take-profit levels.

        Requires a tracked cost basis. Will NOT fall back to last_price (which
        is the *current* market price, not the purchase price) — that fallback
        previously caused bogus alerts whenever portfolio.json was stale
        relative to the live quote.

        Cost-basis field: 'cost_basis' (canonical); 'purchase_price' (legacy
        alias) is also accepted.

        Returns:
            List of actions. Holdings without a tracked cost basis are
            silently skipped; the count is exposed via skipped_no_cost_basis.
        """
        if portfolio is None:
            portfolio = self.load_portfolio()

        actions = []
        skipped = []
        holdings = portfolio.get("holdings", [])

        # Fetch all live prices in ONE batched (cached) request instead of a
        # separate yf.Ticker call per holding inside the loop.
        live_tickers = [
            h.get("ticker") for h in holdings
            if h.get("ticker")
            and (h.get("cost_basis") or h.get("purchase_price"))
            and h.get("last_price", 0) != 0
        ]
        live_prices = {}
        if live_tickers:
            try:
                live_prices, _, _ = market_data.get_prices(live_tickers)
            except Exception as e:
                logger.debug(f"Could not batch-fetch live prices: {e}")

        for holding in holdings:
            ticker = holding.get("ticker")
            quantity = holding.get("quantity", 0)
            cost_basis = holding.get("cost_basis") or holding.get("purchase_price")
            current_price_stored = holding.get("last_price", 0)

            if not cost_basis or cost_basis <= 0:
                skipped.append(ticker)
                continue
            if current_price_stored == 0:
                continue

            # Use the batched live price, falling back to stored last_price.
            current_price = live_prices.get(ticker, current_price_stored)

            return_pct = (current_price / cost_basis - 1) * 100

            if return_pct <= -self.stop_loss_percent:
                actions.append({
                    "action": "SELL",
                    "ticker": ticker,
                    "reason": f"Stop-loss triggered: {return_pct:.2f}% loss (threshold: -{self.stop_loss_percent}%)",
                    "priority": "CRITICAL",
                    "quantity": quantity,
                    "current_price": current_price,
                    "purchase_price": cost_basis,
                    "return_pct": return_pct,
                })
            elif return_pct >= self.take_profit_percent:
                actions.append({
                    "action": "SELL_PARTIAL",
                    "ticker": ticker,
                    "reason": f"Take-profit reached: {return_pct:.2f}% gain (threshold: {self.take_profit_percent}%)",
                    "priority": "HIGH",
                    "quantity": int(quantity * 0.5),
                    "current_price": current_price,
                    "purchase_price": cost_basis,
                    "return_pct": return_pct,
                })

        if skipped:
            logger.info(
                f"Stop-loss/take-profit skipped for {len(skipped)} holdings with no "
                f"cost_basis tracked: {', '.join(skipped[:10])}"
                + ("…" if len(skipped) > 10 else "")
            )
            # Stash for the print path
            self._skipped_no_cost_basis = skipped
        else:
            self._skipped_no_cost_basis = []

        return actions
    
    def check_position_sizes(self, portfolio: Dict = None) -> List[Dict]:
        """Check if any positions are too large (concentration risk)."""
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        warnings = []
        holdings = portfolio.get("holdings", [])
        
        # Calculate total value
        total_value = portfolio.get("total_value", 0)
        if total_value == 0:
            total_value = sum(h.get("current_value", 0) for h in holdings)
            total_value += portfolio.get("cash", 0)
        
        if total_value == 0:
            return warnings
        
        # Check each position
        for holding in holdings:
            value = holding.get("current_value", 0)
            weight = (value / total_value) * 100 if total_value > 0 else 0
            
            if weight > self.max_position_size * 100:
                warnings.append({
                    "type": "POSITION_SIZE",
                    "ticker": holding.get("ticker"),
                    "current_weight": weight,
                    "max_weight": self.max_position_size * 100,
                    "recommendation": f"Reduce {holding.get('ticker')} from {weight:.1f}% to {self.max_position_size * 100:.1f}%"
                })
        
        return warnings
    
    def check_sector_exposure(self, portfolio: Dict = None) -> List[Dict]:
        """Check sector concentration (requires sector mapping)."""
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        warnings = []
        # This would require sector mapping - simplified version
        # In production, you'd use a sector classification API
        
        return warnings
    
    def _get_portfolio_returns(self, portfolio: Dict, period: str = "1y") -> Optional[pd.Series]:
        """Calculate portfolio-level daily returns using actual historical data and weights."""
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return None
        
        total_value = portfolio.get("total_value", 0)
        if total_value == 0:
            total_value = sum(h.get("current_value", 0) for h in holdings)
        if total_value == 0:
            return None
        
        returns_data = {}
        weights = {}

        # Fetch every holding's history in ONE batched (cached) request rather
        # than a sequential yf.Ticker.history() per holding.
        ret_tickers = [
            h.get("ticker") for h in holdings
            if h.get("ticker") and h.get("current_value", 0) > 0
        ]
        histories = market_data.get_histories(ret_tickers, period=period)

        for holding in holdings:
            ticker = holding.get("ticker")
            value = holding.get("current_value", 0)
            if not ticker or value <= 0:
                continue

            data = histories.get(ticker)
            if data is not None and not data.empty and 'Close' in data.columns:
                returns_data[ticker] = data['Close'].pct_change().dropna()
                weights[ticker] = value / total_value

        if not returns_data:
            return None
        
        # Align by date (inner join)
        returns_df = pd.DataFrame(returns_data).dropna()
        if len(returns_df) < 20:
            return None
        
        # Calculate weighted portfolio returns
        weight_array = np.array([weights.get(t, 0) for t in returns_df.columns])
        weight_array = weight_array / weight_array.sum()  # Normalize
        portfolio_returns = returns_df.values @ weight_array
        
        return pd.Series(portfolio_returns, index=returns_df.index)
    
    def calculate_var_cvar(self, portfolio: Dict = None, confidence_levels: List[float] = None,
                           portfolio_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (Expected Shortfall).

        Methods:
        1. Historical VaR: Based on actual return distribution
        2. Parametric VaR: Assumes normal distribution
        3. CVaR (Expected Shortfall): Average loss beyond VaR threshold

        Args:
            portfolio: Portfolio data
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
            portfolio_returns: Pre-computed portfolio daily returns. When the
                caller (e.g. calculate_portfolio_risk_metrics) already computed
                them, pass them in to avoid a redundant _get_portfolio_returns()
                call (which re-fetches every holding's history).

        Returns:
            Dict with VaR and CVaR metrics at each confidence level
        """
        if portfolio is None:
            portfolio = self.load_portfolio()

        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        total_value = portfolio.get("total_value", 0)
        if total_value == 0:
            total_value = sum(h.get("current_value", 0) for h in portfolio.get("holdings", []))

        if portfolio_returns is None:
            portfolio_returns = self._get_portfolio_returns(portfolio)
        if portfolio_returns is None or len(portfolio_returns) < 20:
            return {"error": "Insufficient data for VaR calculation"}
        
        results = {}
        
        from scipy import stats as _stats
        mu = float(portfolio_returns.mean())
        sigma = float(portfolio_returns.std(ddof=1))

        # Fit a Student-t to the empirical returns once (heavy-tailed model).
        # Daily equity returns have excess kurtosis 3–15; Gaussian VaR
        # systematically understates tail loss at 99% by 20–50%.
        try:
            t_df, t_loc, t_scale = _stats.t.fit(portfolio_returns.values)
        except Exception:
            t_df, t_loc, t_scale = float('inf'), mu, sigma

        # GARCH(1,1) conditional volatility. Equity returns exhibit volatility
        # clustering — calm periods have low σ_t and storms have high σ_t. A
        # constant-σ VaR overstates risk during calm regimes and understates
        # it during turbulent ones. The one-step-ahead conditional sigma
        # adapts to the current regime.
        garch_info = None
        garch_sigma_next = None
        try:
            garch_params, _ = fit_garch_1_1(portfolio_returns.values)
            garch_info = garch_params
            garch_sigma_next = float(np.sqrt(garch_params["sigma2_next"]))
        except Exception as e:
            logger.debug(f"GARCH fit skipped: {e}")

        for cl in confidence_levels:
            alpha = 1 - cl
            cl_key = f"{int(cl * 100)}%"

            # 1. Historical VaR — quantile of empirical return distribution
            historical_var = float(np.percentile(portfolio_returns, alpha * 100))
            historical_var_dollar = float(historical_var * total_value)

            # 2a. Parametric Gaussian VaR (for comparison; usually optimistic)
            z_score = _stats.norm.ppf(alpha)
            parametric_var = float(mu + z_score * sigma)
            parametric_var_dollar = float(parametric_var * total_value)

            # 2c. GARCH(1,1) one-step-ahead conditional Gaussian VaR.
            # σ_{t+1|t} reflects current regime; can be very different from σ̂.
            if garch_sigma_next is not None and garch_info is not None:
                garch_var = float(garch_info["mu"] + z_score * garch_sigma_next)
                garch_var_dollar = float(garch_var * total_value)
            else:
                garch_var = parametric_var
                garch_var_dollar = parametric_var_dollar

            # 2b. Parametric Student-t VaR (heavy-tailed; closer to reality)
            try:
                t_var = float(_stats.t.ppf(alpha, df=t_df, loc=t_loc, scale=t_scale))
            except Exception:
                t_var = parametric_var
            t_var_dollar = float(t_var * total_value)

            # 3. Historical CVaR (Expected Shortfall) — average of left-tail losses.
            # Note: with cl=0.95 and ~250 days this averages ~12 observations;
            # the estimate has large SE. Treat as indicative, not precise.
            tail_returns = portfolio_returns[portfolio_returns <= historical_var]
            if len(tail_returns) > 0:
                cvar = float(tail_returns.mean())
                cvar_dollar = float(cvar * total_value)
            else:
                cvar = historical_var
                cvar_dollar = historical_var_dollar

            # 3b. Closed-form Student-t CVaR. For X ~ t_ν(loc, scale),
            # ES_α(X) = loc - scale · ( (ν + τ²)/(ν − 1) ) · f_ν(τ) / α,
            # where τ = t_inv(α; ν) (standard) and f_ν is the standard-t pdf.
            try:
                if t_df > 1 and np.isfinite(t_df):
                    tau = _stats.t.ppf(alpha, df=t_df)
                    pdf_tau = _stats.t.pdf(tau, df=t_df)
                    es_t = float(t_loc - t_scale * ((t_df + tau ** 2) / (t_df - 1)) * pdf_tau / alpha)
                else:
                    es_t = t_var
            except Exception:
                es_t = t_var
            es_t_dollar = float(es_t * total_value)

            results[cl_key] = {
                "historical_var_daily": round(historical_var * 100, 3),
                "historical_var_dollar": round(abs(historical_var_dollar), 2),
                "parametric_var_daily": round(parametric_var * 100, 3),
                "parametric_var_dollar": round(abs(parametric_var_dollar), 2),
                "student_t_var_daily": round(t_var * 100, 3),
                "student_t_var_dollar": round(abs(t_var_dollar), 2),
                "student_t_df": round(float(t_df), 2) if np.isfinite(t_df) else None,
                "garch_var_daily": round(garch_var * 100, 3),
                "garch_var_dollar": round(abs(garch_var_dollar), 2),
                "cvar_daily": round(cvar * 100, 3),
                "cvar_dollar": round(abs(cvar_dollar), 2),
                "student_t_cvar_daily": round(es_t * 100, 3),
                "student_t_cvar_dollar": round(abs(es_t_dollar), 2),
                "interpretation": f"At {cl_key} confidence, historical daily loss is bounded by "
                                  f"${abs(historical_var_dollar):,.2f}; Student-t (df≈{t_df:.1f}) VaR "
                                  f"is ${abs(t_var_dollar):,.2f}; GARCH conditional VaR is "
                                  f"${abs(garch_var_dollar):,.2f}. Conditional shortfall ≈ "
                                  f"${abs(cvar_dollar):,.2f} (hist) / ${abs(es_t_dollar):,.2f} (t)."
            }
        
        # Annualized VaR via the square-root-of-time rule.
        # NOTE: this rule is only exact for i.i.d. Gaussian returns. Real
        # equity series exhibit volatility clustering (GARCH effects) and
        # heavy tails; the actual annual tail loss is typically 10–30%
        # larger than √252-scaled daily VaR. We surface this as a label.
        best_cl = confidence_levels[0]
        cl_key = f"{int(best_cl * 100)}%"
        if cl_key in results:
            daily_var = results[cl_key]["historical_var_daily"] / 100
            annual_var = daily_var * np.sqrt(252)
            results["annualized_var"] = {
                "confidence": cl_key,
                "annual_var_pct": round(annual_var * 100, 2),
                "annual_var_dollar": round(abs(annual_var * total_value), 2),
                "scaling": "sqrt-of-time (i.i.d. assumption — likely understates fat-tailed annual risk)"
            }
        
        results["portfolio_value"] = total_value
        results["data_points"] = len(portfolio_returns)
        if garch_info is not None:
            results["garch_1_1"] = {
                "alpha": round(garch_info["alpha"], 4),
                "beta": round(garch_info["beta"], 4),
                "persistence": round(garch_info["persistence"], 4),
                "sigma_now_daily_pct": round(float(np.sqrt(garch_info["sigma2_now"]) * 100), 3),
                "sigma_next_daily_pct": round(float(garch_sigma_next * 100), 3) if garch_sigma_next else None,
                "unconditional_sigma_daily_pct": round(float(np.sqrt(garch_info["unconditional_variance"]) * 100), 3)
                    if np.isfinite(garch_info["unconditional_variance"]) else None,
                "n_obs": garch_info["n_obs"],
                "interpretation": "If σ_next > σ_uncond, the portfolio is in an elevated-vol regime; "
                                  "constant-σ VaR understates current risk. Vice versa for calm regimes."
            }

        return results

    def calculate_portfolio_risk_metrics(self, portfolio: Dict = None) -> Dict:
        """Calculate overall portfolio risk metrics including VaR/CVaR."""
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return {}
        
        # Get current prices and calculate metrics
        total_value = portfolio.get("total_value", 0)
        if total_value == 0:
            total_value = sum(h.get("current_value", 0) for h in holdings)
        
        if total_value == 0:
            return {}
        
        # Concentration (Herfindahl–Hirschman Index). HHI ∈ [1/n, 1].
        # The reciprocal 1/HHI is the canonical "effective number of holdings"
        # — interpretable as the number of equally-weighted positions that
        # would produce the same concentration. A 5-asset equal-weight
        # portfolio has HHI=0.20, effective_n=5.0. The previous metric
        # `1 − HHI` is bounded above by `1 − 1/n` and is misleading for
        # small portfolios.
        weights = [h.get("current_value", 0) / total_value for h in holdings if total_value > 0]
        concentration = sum(w**2 for w in weights)
        effective_n = (1.0 / concentration) if concentration > 0 else 0.0

        # Actual portfolio-level volatility, max drawdown, Sharpe from historical data
        portfolio_returns = self._get_portfolio_returns(portfolio)

        portfolio_volatility = None
        portfolio_max_drawdown = None
        portfolio_sharpe = None

        if portfolio_returns is not None and len(portfolio_returns) > 20:
            portfolio_volatility = float(portfolio_returns.std(ddof=1) * np.sqrt(252) * 100)

            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            portfolio_max_drawdown = float(drawdown.min() * 100)

            try:
                from advanced_analysis import AdvancedAnalyzer
                rf = AdvancedAnalyzer().get_risk_free_rate()
            except Exception:
                rf = 0.045
            daily_rf = rf / 252
            excess = portfolio_returns - daily_rf
            # Use std of EXCESS returns in the Sharpe denominator (strict definition).
            excess_std = float(excess.std(ddof=1))
            if excess_std > 0:
                portfolio_sharpe = float((excess.mean() * 252) / (excess_std * np.sqrt(252)))

        result = {
            "total_positions": len(holdings),
            "concentration_index": concentration,            # HHI
            "effective_number_holdings": round(effective_n, 2),  # 1 / HHI — canonical
            "diversification_score": 1 - concentration,      # legacy (Gini-Simpson), kept for back-compat
            "max_position_weight": max(weights) * 100 if weights else 0,
            "cash_percent": (portfolio.get("cash", 0) / total_value * 100) if total_value > 0 else 0
        }
        
        if portfolio_volatility is not None:
            result["portfolio_volatility"] = round(portfolio_volatility, 2)
        if portfolio_max_drawdown is not None:
            result["portfolio_max_drawdown"] = round(portfolio_max_drawdown, 2)
        if portfolio_sharpe is not None:
            result["portfolio_sharpe"] = round(portfolio_sharpe, 3)
        
        # Add VaR/CVaR — reuse the portfolio_returns we already computed above
        # so _get_portfolio_returns (and its batched history fetch) runs once
        # per report instead of three times.
        try:
            var_metrics = self.calculate_var_cvar(portfolio, portfolio_returns=portfolio_returns)
            if "error" not in var_metrics:
                result["var_cvar"] = var_metrics
        except Exception as e:
            logger.debug(f"Failed to calculate VaR/CVaR: {e}")
        
        return result
    
    def get_risk_alerts(self, portfolio: Dict = None) -> Dict:
        """Get all risk alerts and recommendations."""
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        alerts = {
            "stop_loss_take_profit": self.check_stop_loss_take_profit(portfolio),
            "position_sizes": self.check_position_sizes(portfolio),
            "sector_exposure": self.check_sector_exposure(portfolio),
            "risk_metrics": self.calculate_portfolio_risk_metrics(portfolio)
        }
        
        # Count critical alerts
        critical_count = sum(1 for a in alerts["stop_loss_take_profit"] if a.get("priority") == "CRITICAL")
        high_count = sum(1 for a in alerts["stop_loss_take_profit"] if a.get("priority") == "HIGH")
        
        alerts["summary"] = {
            "critical_alerts": critical_count,
            "high_priority_alerts": high_count,
            "total_warnings": len(alerts["position_sizes"]) + len(alerts["sector_exposure"])
        }
        
        return alerts
    
    def print_risk_report(self, portfolio: Dict = None):
        """Print comprehensive risk report."""
        alerts = self.get_risk_alerts(portfolio)
        
        print("\n" + "=" * 60)
        print("RISK MANAGEMENT REPORT")
        print("=" * 60 + "\n")
        
        # Summary
        summary = alerts["summary"]
        print(f"📊 Risk Summary:")
        print(f"   Critical Alerts: {summary['critical_alerts']}")
        print(f"   High Priority Alerts: {summary['high_priority_alerts']}")
        print(f"   Warnings: {summary['total_warnings']}\n")
        
        # Stop-loss / Take-profit
        if alerts["stop_loss_take_profit"]:
            print("⚠️  Stop-Loss / Take-Profit Actions:")
            for action in alerts["stop_loss_take_profit"]:
                print(f"   [{action['priority']}] {action['action']}: {action['ticker']}")
                print(f"      {action['reason']}")
                print(f"      Return: {action['return_pct']:.2f}%")
                print(f"      Recommended: Sell {action['quantity']} shares @ ${action['current_price']:.2f}\n")
        else:
            print("✅ No stop-loss or take-profit triggers\n")

        # Disclose any holdings skipped due to missing cost_basis. The previous
        # implementation silently fell back to last_price (the *current* price)
        # and fired bogus alerts; we now skip and tell the user instead.
        skipped = getattr(self, "_skipped_no_cost_basis", []) or []
        if skipped:
            print(f"ℹ️  Skipped stop-loss/take-profit for {len(skipped)} holdings without a tracked")
            print(f"   cost_basis: {', '.join(skipped[:10])}" + ("…" if len(skipped) > 10 else ""))
            print(f"   Run `make backfill-cost-basis` to set cost_basis = last_price for them.\n")
        
        # Position sizes
        if alerts["position_sizes"]:
            print("⚠️  Position Size Warnings:")
            for warning in alerts["position_sizes"]:
                print(f"   {warning['recommendation']}\n")
        else:
            print("✅ All positions within size limits\n")
        
        # Risk metrics
        metrics = alerts["risk_metrics"]
        if metrics:
            print("📈 Portfolio Risk Metrics:")
            print(f"   Total Positions: {metrics.get('total_positions', 0)}")
            print(f"   Concentration Index (HHI): {metrics.get('concentration_index', 0):.3f}")
            print(f"   Effective # Holdings (1/HHI): {metrics.get('effective_number_holdings', 0):.2f}")
            print(f"   Diversification Score (1-HHI): {metrics.get('diversification_score', 0):.2f}")
            print(f"   Largest Position: {metrics.get('max_position_weight', 0):.1f}%")
            print(f"   Cash: {metrics.get('cash_percent', 0):.1f}%")
            
            if metrics.get('portfolio_volatility') is not None:
                print(f"   Portfolio Volatility: {metrics['portfolio_volatility']:.2f}%")
            if metrics.get('portfolio_max_drawdown') is not None:
                print(f"   Max Drawdown: {metrics['portfolio_max_drawdown']:.2f}%")
            if metrics.get('portfolio_sharpe') is not None:
                print(f"   Portfolio Sharpe Ratio: {metrics['portfolio_sharpe']:.3f}")
            print()
            
            # VaR/CVaR
            var_cvar = metrics.get('var_cvar', {})
            if var_cvar and "error" not in var_cvar:
                print("📉 Value at Risk (VaR) / Expected Shortfall (CVaR):")
                for cl_key in ["95%", "99%"]:
                    if cl_key in var_cvar:
                        vc = var_cvar[cl_key]
                        print(f"   {cl_key} Confidence:")
                        print(f"      Historical VaR:        {vc['historical_var_daily']:>7.3f}%  (${vc['historical_var_dollar']:>10,.2f}/day)")
                        print(f"      Gaussian VaR:          {vc['parametric_var_daily']:>7.3f}%  (${vc['parametric_var_dollar']:>10,.2f}/day)")
                        if 'student_t_var_daily' in vc:
                            df_str = f"df≈{vc['student_t_df']:.1f}" if vc.get('student_t_df') else "df=∞"
                            print(f"      Student-t VaR ({df_str:>7}): {vc['student_t_var_daily']:>7.3f}%  (${vc['student_t_var_dollar']:>10,.2f}/day)")
                        if 'garch_var_daily' in vc:
                            print(f"      GARCH(1,1) VaR:        {vc['garch_var_daily']:>7.3f}%  (${vc['garch_var_dollar']:>10,.2f}/day)  ← regime-aware")
                        print(f"      Historical CVaR:       {vc['cvar_daily']:>7.3f}%  (${vc['cvar_dollar']:>10,.2f}/day)")
                        if 'student_t_cvar_daily' in vc:
                            print(f"      Student-t CVaR:        {vc['student_t_cvar_daily']:>7.3f}%  (${vc['student_t_cvar_dollar']:>10,.2f}/day)")

                ann = var_cvar.get("annualized_var")
                if ann:
                    print(f"   Annualized VaR ({ann['confidence']}): {ann['annual_var_pct']:.2f}% (${ann['annual_var_dollar']:,.2f})")
                    note = ann.get('scaling')
                    if note:
                        print(f"      [{note}]")

                garch = var_cvar.get("garch_1_1")
                if garch:
                    print("\n   GARCH(1,1) volatility regime:")
                    print(f"      α={garch['alpha']:.3f}  β={garch['beta']:.3f}  persistence={garch['persistence']:.3f}")
                    print(f"      σ_now = {garch['sigma_now_daily_pct']:.2f}%/day  vs  σ_uncond = {garch['unconditional_sigma_daily_pct']:.2f}%/day")
                    if garch.get('sigma_next_daily_pct') is not None:
                        print(f"      σ_next (1-day-ahead) = {garch['sigma_next_daily_pct']:.2f}%/day")
                    print(f"      {garch['interpretation']}")
                print()
        
        print("=" * 60 + "\n")

def backfill_cost_basis(portfolio_file: str = "portfolio.json", confirm: bool = True) -> int:
    """Set cost_basis = last_price for every holding that doesn't have one.

    This is a one-time migration for portfolios that pre-date cost-basis
    tracking. The assumption it makes is: "assume each existing holding was
    purchased at its most-recent stored last_price". That's not the true
    historical cost basis, but it gives stop-loss/take-profit a sane baseline
    going forward, and is opt-in.

    Returns the number of holdings updated.
    """
    with open(portfolio_file, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)

    needs = [
        h for h in portfolio.get("holdings", [])
        if not (h.get("cost_basis") or h.get("purchase_price"))
        and h.get("last_price", 0) > 0
    ]
    if not needs:
        print("All holdings already have a cost_basis tracked. Nothing to do.")
        return 0

    print(f"Will set cost_basis = last_price for {len(needs)} holdings:")
    for h in needs:
        print(f"  {h['ticker']:>6}  cost_basis ← ${h['last_price']:.2f}  (qty {h.get('quantity',0)})")

    if confirm:
        ans = input("\nProceed? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted; portfolio.json not modified.")
            return 0

    for h in needs:
        h["cost_basis"] = round(float(h["last_price"]), 4)

    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Updated {len(needs)} holdings in {portfolio_file}.")
    return len(needs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill-cost-basis":
        # Non-interactive when --yes is passed (useful for CI/scripts).
        confirm = "--yes" not in sys.argv
        backfill_cost_basis(confirm=confirm)
    else:
        manager = RiskManager()
        manager.print_risk_report()

