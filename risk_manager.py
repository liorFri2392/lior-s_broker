#!/usr/bin/env python3
"""
Risk Management Module - Automatic stop-loss, take-profit, and risk monitoring
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

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
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"holdings": [], "cash": 0}
    
    def check_stop_loss_take_profit(self, portfolio: Dict = None) -> List[Dict]:
        """
        Check if any holdings hit stop-loss or take-profit levels.
        
        Returns:
            List of actions needed (SELL recommendations)
        """
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        actions = []
        holdings = portfolio.get("holdings", [])
        
        for holding in holdings:
            ticker = holding.get("ticker")
            quantity = holding.get("quantity", 0)
            purchase_price = holding.get("purchase_price") or holding.get("last_price", 0)
            current_price = holding.get("last_price", 0)
            
            if purchase_price == 0 or current_price == 0:
                continue
            
            # Get real-time price
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
            except Exception as e:
                logger.debug(f"Could not get current price for {ticker}: {e}")
                continue
            
            # Calculate return
            return_pct = (current_price / purchase_price - 1) * 100
            
            # Check stop-loss
            if return_pct <= -self.stop_loss_percent:
                actions.append({
                    "action": "SELL",
                    "ticker": ticker,
                    "reason": f"Stop-loss triggered: {return_pct:.2f}% loss (threshold: -{self.stop_loss_percent}%)",
                    "priority": "CRITICAL",
                    "quantity": quantity,
                    "current_price": current_price,
                    "purchase_price": purchase_price,
                    "return_pct": return_pct
                })
            
            # Check take-profit
            elif return_pct >= self.take_profit_percent:
                actions.append({
                    "action": "SELL_PARTIAL",
                    "ticker": ticker,
                    "reason": f"Take-profit reached: {return_pct:.2f}% gain (threshold: {self.take_profit_percent}%)",
                    "priority": "HIGH",
                    "quantity": int(quantity * 0.5),  # Sell 50% to lock profits
                    "current_price": current_price,
                    "purchase_price": purchase_price,
                    "return_pct": return_pct
                })
        
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
        
        for holding in holdings:
            ticker = holding.get("ticker")
            value = holding.get("current_value", 0)
            if not ticker or value <= 0:
                continue
            
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period)
                if data is not None and not data.empty and 'Close' in data.columns:
                    returns_data[ticker] = data['Close'].pct_change().dropna()
                    weights[ticker] = value / total_value
            except Exception as e:
                logger.debug(f"Failed to get returns for {ticker}: {e}")
                continue
        
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
    
    def calculate_var_cvar(self, portfolio: Dict = None, confidence_levels: List[float] = None) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
        
        Methods:
        1. Historical VaR: Based on actual return distribution
        2. Parametric VaR: Assumes normal distribution
        3. CVaR (Expected Shortfall): Average loss beyond VaR threshold
        
        Args:
            portfolio: Portfolio data
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
        
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
        
        portfolio_returns = self._get_portfolio_returns(portfolio)
        if portfolio_returns is None or len(portfolio_returns) < 20:
            return {"error": "Insufficient data for VaR calculation"}
        
        results = {}
        
        for cl in confidence_levels:
            alpha = 1 - cl
            cl_key = f"{int(cl * 100)}%"
            
            # 1. Historical VaR
            historical_var = float(np.percentile(portfolio_returns, alpha * 100))
            historical_var_dollar = float(historical_var * total_value)
            
            # 2. Parametric VaR (Normal distribution assumption)
            from scipy import stats
            mu = float(portfolio_returns.mean())
            sigma = float(portfolio_returns.std())
            z_score = stats.norm.ppf(alpha)
            parametric_var = float(mu + z_score * sigma)
            parametric_var_dollar = float(parametric_var * total_value)
            
            # 3. CVaR (Expected Shortfall) — average of losses beyond VaR
            tail_returns = portfolio_returns[portfolio_returns <= historical_var]
            if len(tail_returns) > 0:
                cvar = float(tail_returns.mean())
                cvar_dollar = float(cvar * total_value)
            else:
                cvar = historical_var
                cvar_dollar = historical_var_dollar
            
            results[cl_key] = {
                "historical_var_daily": round(historical_var * 100, 3),
                "historical_var_dollar": round(abs(historical_var_dollar), 2),
                "parametric_var_daily": round(parametric_var * 100, 3),
                "parametric_var_dollar": round(abs(parametric_var_dollar), 2),
                "cvar_daily": round(cvar * 100, 3),
                "cvar_dollar": round(abs(cvar_dollar), 2),
                "interpretation": f"At {cl_key} confidence, daily loss should not exceed ${abs(historical_var_dollar):,.2f}. "
                                  f"If it does, expected loss is ${abs(cvar_dollar):,.2f}."
            }
        
        # Annualized VaR (approximate using sqrt of time)
        best_cl = confidence_levels[0]
        cl_key = f"{int(best_cl * 100)}%"
        if cl_key in results:
            daily_var = results[cl_key]["historical_var_daily"] / 100
            annual_var = daily_var * np.sqrt(252)
            results["annualized_var"] = {
                "confidence": cl_key,
                "annual_var_pct": round(annual_var * 100, 2),
                "annual_var_dollar": round(abs(annual_var * total_value), 2)
            }
        
        results["portfolio_value"] = total_value
        results["data_points"] = len(portfolio_returns)
        
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
        
        # Calculate concentration (Herfindahl index)
        weights = [h.get("current_value", 0) / total_value for h in holdings if total_value > 0]
        concentration = sum(w**2 for w in weights)
        
        # Calculate actual portfolio volatility and max drawdown from historical data
        portfolio_returns = self._get_portfolio_returns(portfolio)
        
        portfolio_volatility = None
        portfolio_max_drawdown = None
        portfolio_sharpe = None
        
        if portfolio_returns is not None and len(portfolio_returns) > 20:
            portfolio_volatility = float(portfolio_returns.std() * np.sqrt(252) * 100)
            
            # Temporal max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            portfolio_max_drawdown = float(drawdown.min() * 100)
            
            # Portfolio Sharpe
            try:
                from advanced_analysis import AdvancedAnalyzer
                rf = AdvancedAnalyzer().get_risk_free_rate()
            except Exception:
                rf = 0.045
            daily_rf = rf / 252
            excess = portfolio_returns - daily_rf
            if portfolio_returns.std() > 0:
                portfolio_sharpe = float((excess.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)))
        
        result = {
            "total_positions": len(holdings),
            "concentration_index": concentration,
            "diversification_score": 1 - concentration,
            "max_position_weight": max(weights) * 100 if weights else 0,
            "cash_percent": (portfolio.get("cash", 0) / total_value * 100) if total_value > 0 else 0
        }
        
        if portfolio_volatility is not None:
            result["portfolio_volatility"] = round(portfolio_volatility, 2)
        if portfolio_max_drawdown is not None:
            result["portfolio_max_drawdown"] = round(portfolio_max_drawdown, 2)
        if portfolio_sharpe is not None:
            result["portfolio_sharpe"] = round(portfolio_sharpe, 3)
        
        # Add VaR/CVaR
        try:
            var_metrics = self.calculate_var_cvar(portfolio)
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
            print(f"   Concentration Index: {metrics.get('concentration_index', 0):.3f}")
            print(f"   Diversification Score: {metrics.get('diversification_score', 0):.2f}")
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
                        print(f"      Historical VaR: {vc['historical_var_daily']:.3f}% (${vc['historical_var_dollar']:,.2f}/day)")
                        print(f"      CVaR (Exp. Shortfall): {vc['cvar_daily']:.3f}% (${vc['cvar_dollar']:,.2f}/day)")
                
                ann = var_cvar.get("annualized_var")
                if ann:
                    print(f"   Annualized VaR ({ann['confidence']}): {ann['annual_var_pct']:.2f}% (${ann['annual_var_dollar']:,.2f})")
                print()
        
        print("=" * 60 + "\n")

if __name__ == "__main__":
    manager = RiskManager()
    manager.print_risk_report()

