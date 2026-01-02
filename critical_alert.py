#!/usr/bin/env python3
"""
Critical Alert System - Detects urgent portfolio actions
Runs deep analysis to find critical buy/sell opportunities
"""

import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional
from portfolio_analyzer import PortfolioAnalyzer
from deposit_advisor import DepositAdvisor
from email_notifier import EmailNotifier
import yfinance as yf

logger = logging.getLogger(__name__)

class CriticalAlertSystem:
    """Detects and alerts on critical portfolio opportunities."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.analyzer = PortfolioAnalyzer(portfolio_file)
        self.advisor = DepositAdvisor(portfolio_file)
        self.critical_threshold = 75  # Score threshold for critical buy
        self.urgent_sell_threshold = 25  # Score threshold for urgent sell
    
    def is_market_trading_day(self) -> bool:
        """Check if today is a US market trading day (excludes holidays)."""
        try:
            # Use SPY as market indicator
            spy = yf.Ticker("SPY")
            today = datetime.now().date()
            
            # Try to get today's data
            hist = spy.history(period="5d")
            if not hist.empty:
                last_date = hist.index[-1].date()
                # If last trading day is today or yesterday, market is likely open
                days_diff = (today - last_date).days
                return days_diff <= 1
            
            # Fallback: check if it's a weekday (Mon-Fri)
            weekday = today.weekday()  # 0=Monday, 6=Sunday
            return weekday < 5  # Monday to Friday
            
        except Exception as e:
            logger.warning(f"Could not determine market status: {e}")
            # Fallback to weekday check
            weekday = datetime.now().weekday()
            return weekday < 5
    
    def check_critical_opportunities(self) -> Dict:
        """Run comprehensive analysis to find critical opportunities."""
        print("=" * 60)
        print("CRITICAL ALERT SYSTEM - Deep Analysis")
        print("=" * 60)
        
        # Check if market is trading
        is_trading = self.is_market_trading_day()
        market_status, market_message = self.analyzer.is_market_open()
        
        print(f"\n📊 Market Status: {market_message}")
        print(f"   Trading Day: {'✅ Yes' if is_trading else '❌ No (Holiday/Weekend)'}")
        print()
        
        if not is_trading:
            return {
                "critical_items": [],
                "has_critical": False,
                "message": "Not a trading day - no analysis performed"
            }
        
        critical_items = []
        
        # 1. Analyze current portfolio for urgent sells
        print("Analyzing current portfolio for urgent actions...")
        portfolio_analysis = self.analyzer.analyze()
        
        if portfolio_analysis:
            holdings_analysis = portfolio_analysis.get("holdings_analysis", [])
            portfolio_metrics = portfolio_analysis.get("portfolio_metrics", {})
            
            # Check for urgent sells (STRONG SELL, very low scores)
            for holding in holdings_analysis:
                score = holding.get("recommendation_score", 50)
                recommendation = holding.get("recommendation", "HOLD")
                ticker = holding.get("ticker", "")
                current_value = holding.get("current_value", 0)
                
                if recommendation == "STRONG SELL" or score < self.urgent_sell_threshold:
                    critical_items.append({
                        "type": "SELL",
                        "ticker": ticker,
                        "priority": "CRITICAL",
                        "reason": f"STRONG SELL signal - Score: {score:.1f}/100. Risk of significant loss.",
                        "amount": current_value * 0.5,  # Recommend selling 50%
                        "shares": int(holding.get("quantity", 0) * 0.5),
                        "score": score,
                        "current_price": holding.get("current_price", 0)
                    })
            
            # Check for rebalancing needs (over-concentration)
            rebalancing = portfolio_analysis.get("rebalancing", {})
            if rebalancing.get("needed") and rebalancing.get("recommendations"):
                for rec in rebalancing.get("recommendations", []):
                    if rec.get("action") == "SELL":
                        critical_items.append({
                            "type": "SELL",
                            "ticker": rec.get("ticker", ""),
                            "priority": "HIGH",
                            "reason": f"Over-concentration: {rec.get('current_weight', 0)*100:.1f}% of portfolio. Diversification needed.",
                            "amount": rec.get("reduce_amount_usd", 0),
                            "shares": rec.get("reduce_shares", 0),
                            "score": next((h.get("recommendation_score", 50) for h in holdings_analysis if h.get("ticker") == rec.get("ticker")), 50)
                        })
        
        # 2. Scan for critical buy opportunities (high-yield ETFs)
        print("\nScanning for critical buy opportunities...")
        critical_buys = self.scan_critical_buy_opportunities()
        critical_items.extend(critical_buys)
        
        # 3. Check for market anomalies or extreme opportunities
        print("\nChecking for market anomalies...")
        anomalies = self.check_market_anomalies()
        critical_items.extend(anomalies)
        
        # Sort by priority (CRITICAL > HIGH > MEDIUM)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        critical_items.sort(key=lambda x: priority_order.get(x.get("priority", "MEDIUM"), 3))
        
        return {
            "critical_items": critical_items,
            "has_critical": len(critical_items) > 0,
            "portfolio_value": portfolio_analysis.get("portfolio_metrics", {}).get("total_value", 0) if portfolio_analysis else 0,
            "market_status": market_message,
            "timestamp": datetime.now().isoformat()
        }
    
    def scan_critical_buy_opportunities(self) -> List[Dict]:
        """Scan for ETFs with exceptional buy opportunities, considering current portfolio."""
        critical_buys = []
        
        # Get current portfolio holdings to avoid recommending what we already have
        portfolio = self.analyzer.load_portfolio()
        current_holdings = [h.get("ticker", "").upper() for h in portfolio.get("holdings", [])]
        portfolio_value = portfolio.get("total_value", 0)
        cash_available = portfolio.get("cash", 0)
        
        logger.info(f"Current portfolio has {len(current_holdings)} holdings: {current_holdings}")
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Cash available: ${cash_available:,.2f}")
        
        # Focus on high-potential categories
        high_potential_categories = [
            "AI_AND_ROBOTICS", "TECHNOLOGY", "SEMICONDUCTORS", "CRYPTO", "CLEAN_ENERGY",
            "HEALTHCARE", "FINANCIAL", "GROWTH", "MOMENTUM"
        ]
        
        # Analyze top ETFs from these categories
        analyzed_etfs = []
        for category in high_potential_categories:
            if category in self.advisor.ETF_CATEGORIES:
                etfs = self.advisor.ETF_CATEGORIES[category][:3]  # Top 3 from each category
                for etf in etfs:
                    etf_upper = etf.upper()
                    # Skip if already in portfolio (unless it's a great opportunity to increase)
                    if etf_upper not in analyzed_etfs:
                        analyzed_etfs.append(etf)
        
        print(f"   Analyzing {len(analyzed_etfs)} high-potential ETFs (excluding current holdings)...")
        
        # Analyze in batches
        for i, etf in enumerate(analyzed_etfs[:20]):  # Limit to 20 for performance
            try:
                etf_upper = etf.upper()
                
                # Skip if already in portfolio (focus on diversification)
                if etf_upper in current_holdings:
                    logger.debug(f"Skipping {etf} - already in portfolio")
                    continue
                
                analysis = self.advisor.analyze_etf(etf, verbose=False)
                score = analysis.get("score", 0)
                
                # Check for critical buy signals
                if score >= self.critical_threshold:
                    expected_return = analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0)
                    reasons = analysis.get("reasons", [])
                    
                    # Only flag as critical if:
                    # 1. Very high score (>=75)
                    # 2. Strong expected return (>15%)
                    # 3. Positive technical indicators
                    if expected_return > 15 or score >= 85:
                        # Calculate recommended amount based on available cash
                        recommended_amount = min(1000, cash_available * 0.1) if cash_available > 0 else 1000
                        
                        critical_buys.append({
                            "type": "BUY",
                            "ticker": etf,
                            "priority": "CRITICAL" if score >= 85 else "HIGH",
                            "reason": f"Exceptional opportunity - Score: {score}/100. Expected 3yr return: {expected_return:.1f}%. Not in current portfolio.",
                            "amount": recommended_amount,
                            "expected_return": expected_return,
                            "score": score,
                            "details": "; ".join(reasons[:3]),  # Top 3 reasons
                            "diversification": "New holding - adds diversification"
                        })
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"   Progress: {i + 1}/{min(len(analyzed_etfs), 20)} ETFs analyzed...")
                    
            except Exception as e:
                logger.debug(f"Failed to analyze ETF {etf}: {e}")
                continue  # Skip failed analyses
        
        return critical_buys
    
    def check_market_anomalies(self) -> List[Dict]:
        """Check for market anomalies that require immediate attention."""
        anomalies = []
        
        try:
            # Check SPY for extreme movements
            spy = yf.Ticker("SPY")
            hist = spy.history(period="5d")
            
            if not hist.empty and len(hist) >= 2:
                recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                
                # Extreme market drop (>3% in one day)
                if recent_return < -3:
                    anomalies.append({
                        "type": "ACTION",
                        "title": "Market Drop Alert",
                        "priority": "HIGH",
                        "reason": f"SPY dropped {abs(recent_return):.2f}% - Consider defensive positions",
                        "details": "Market experiencing significant decline. Review portfolio for risk exposure."
                    })
                
                # Extreme market surge (>3% in one day)
                elif recent_return > 3:
                    anomalies.append({
                        "type": "ACTION",
                        "title": "Market Surge Alert",
                        "priority": "MEDIUM",
                        "reason": f"SPY surged {recent_return:.2f}% - Consider taking profits",
                        "details": "Market experiencing significant gains. Consider rebalancing to lock in profits."
                    })
        except Exception as e:
            logger.warning(f"Failed to check market anomalies: {e}")
            pass
        
        return anomalies
    
    def send_alerts(self, results: Dict) -> bool:
        """Send email alerts if critical items found."""
        if not results.get("has_critical"):
            print("\n✅ No critical actions required at this time.")
            return False
        
        try:
            notifier = EmailNotifier()
            
            critical_items = results.get("critical_items", [])
            portfolio_value = results.get("portfolio_value", 0)
            
            # Count by type
            buy_count = sum(1 for item in critical_items if item.get("type") == "BUY")
            sell_count = sum(1 for item in critical_items if item.get("type") == "SELL")
            
            subject = f"URGENT: {buy_count} Buy{'s' if buy_count != 1 else ''}, {sell_count} Sell{'s' if sell_count != 1 else ''} Required"
            
            success = notifier.send_critical_alert(subject, critical_items, portfolio_value)
            
            if success:
                print(f"\n✅ Critical alert email sent with {len(critical_items)} urgent action(s)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}", exc_info=True)
            print(f"\n❌ Error sending alerts: {e}")
            print("   Make sure EMAIL_SENDER and EMAIL_PASSWORD are set in environment variables")
            return False
    
    def run(self) -> Dict:
        """Main function to run critical alert check."""
        results = self.check_critical_opportunities()
        
        if results.get("has_critical"):
            print(f"\n⚠️  Found {len(results['critical_items'])} critical action(s) requiring immediate attention:")
            for item in results["critical_items"]:
                print(f"\n   [{item.get('priority', 'MEDIUM')}] {item.get('type', 'ACTION')}: {item.get('ticker', item.get('title', 'N/A'))}")
                print(f"      Reason: {item.get('reason', 'N/A')}")
            
            # Send email alerts
            self.send_alerts(results)
        else:
            print("\n✅ No critical actions required. Portfolio is in good shape.")
            print("   (No email sent - only critical alerts trigger emails)")
        
        return results

if __name__ == "__main__":
    alert_system = CriticalAlertSystem()
    results = alert_system.run()
    
    # Exit with code 0 if no critical items, 1 if critical items found
    sys.exit(0 if not results.get("has_critical") else 1)

