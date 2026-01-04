#!/usr/bin/env python3
"""
Risk Management Module - Automatic stop-loss, take-profit, and risk monitoring
"""

import json
import os
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
    
    def calculate_portfolio_risk_metrics(self, portfolio: Dict = None) -> Dict:
        """Calculate overall portfolio risk metrics."""
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return {}
        
        # Get current prices and calculate metrics
        total_value = portfolio.get("total_value", 0)
        if total_value == 0:
            total_value = sum(h.get("current_value", 0) for h in holdings)
        
        # Calculate concentration (Herfindahl index)
        weights = [h.get("current_value", 0) / total_value for h in holdings if total_value > 0]
        concentration = sum(w**2 for w in weights)
        
        # Calculate weighted average volatility (simplified)
        # In production, would calculate actual volatility from historical data
        
        return {
            "total_positions": len(holdings),
            "concentration_index": concentration,
            "diversification_score": 1 - concentration,
            "max_position_weight": max(weights) * 100 if weights else 0,
            "cash_percent": (portfolio.get("cash", 0) / total_value * 100) if total_value > 0 else 0
        }
    
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
            print(f"   Cash: {metrics.get('cash_percent', 0):.1f}%\n")
        
        print("=" * 60 + "\n")

if __name__ == "__main__":
    manager = RiskManager()
    manager.print_risk_report()

