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
        # Only if portfolio is significantly unbalanced and cash available
        portfolio = self.analyzer.load_portfolio()
        cash_available = portfolio.get("cash", 0)
        
        # Check if portfolio needs balancing
        if portfolio_analysis:
            balance_check = self.analyzer.check_80_20_balance(
                portfolio_metrics,
                holdings_analysis
            )
            
            needs_balancing = not balance_check.get("is_balanced", False)
            bonds_percent = balance_check.get("bonds_percent", 0)
        else:
            needs_balancing = True  # If no analysis, assume needs balancing
            bonds_percent = 0
        
        # Only scan for buys if:
        # 1. Portfolio is unbalanced (needs bonds or stocks)
        # 2. Have sufficient cash (>$500)
        if needs_balancing and cash_available > 500:
            print("\nScanning for critical buy opportunities...")
            critical_buys = self.scan_critical_buy_opportunities()
            
            # Filter to only most critical (limit to top 3-5)
            # Prioritize bonds if portfolio lacks bonds
            if bonds_percent < 15:
                bond_buys = [b for b in critical_buys if b.get("category") == "BONDS"]
                other_buys = [b for b in critical_buys if b.get("category") != "BONDS"]
                # Top 2 bonds + top 2 others (max 4 total)
                critical_buys = bond_buys[:2] + other_buys[:2]
            else:
                critical_buys = critical_buys[:3]  # Top 3 overall
            
            critical_items.extend(critical_buys)
        else:
            if cash_available <= 500:
                print("\n⚠️  Insufficient cash for new purchases (${:.2f})".format(cash_available))
            elif not needs_balancing:
                print("\n✅ Portfolio is balanced - no critical buy opportunities needed")
        
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
        """
        Scan for critical buy opportunities following 80/20 Balanced Growth Strategy.
        Prioritizes Core ETFs and Bonds over high-risk trends.
        """
        critical_buys = []
        
        # Get current portfolio holdings to avoid recommending what we already have
        portfolio = self.analyzer.load_portfolio()
        current_holdings = [h.get("ticker", "").upper() for h in portfolio.get("holdings", [])]
        portfolio_value = portfolio.get("total_value", 0)
        cash_available = portfolio.get("cash", 0)
        
        logger.info(f"Current portfolio has {len(current_holdings)} holdings: {current_holdings}")
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Cash available: ${cash_available:,.2f}")
        
        # 80/20 Strategy: Focus on Core, Satellite, and Bonds
        # Exclude high-risk categories (leveraged, crypto)
        excluded_categories = ["LEVERAGED_2X", "LEVERAGED_3X", "LEVERAGED_INVERSE", "CRYPTO"]
        
        # Core ETFs (essential for portfolio stability)
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        
        # Satellite ETFs (growth, but not too risky)
        satellite_categories = [
            "US_SMALL_CAP", "TECHNOLOGY", "HEALTHCARE", "EMERGING_MARKETS"
        ]
        
        # Bond ETFs (protection)
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
        
        # Collect ETFs to analyze
        analyzed_etfs = []
        
        # Add Core ETFs (highest priority)
        for etf in core_etfs:
            if etf.upper() not in analyzed_etfs:
                analyzed_etfs.append(etf)
        
        # Add Satellite ETFs from safe categories
        for category in satellite_categories:
            if category in self.advisor.ETF_CATEGORIES:
                etfs = self.advisor.ETF_CATEGORIES[category][:2]  # Top 2 from each category
                for etf in etfs:
                    etf_upper = etf.upper()
                    # Skip excluded categories
                    is_excluded = any(etf_upper in self.advisor.ETF_CATEGORIES.get(cat, []) 
                                     for cat in excluded_categories)
                    if not is_excluded and etf_upper not in analyzed_etfs:
                        analyzed_etfs.append(etf)
        
        # Add Bond ETFs
        for etf in bond_etfs:
            if etf.upper() not in analyzed_etfs:
                analyzed_etfs.append(etf)
        
        print(f"   Analyzing {len(analyzed_etfs)} ETFs (Core, Satellite, Bonds) following 80/20 strategy...")
        print("   (Excluding leveraged ETFs and crypto for balanced risk)")
        
        # Analyze in batches
        for i, etf in enumerate(analyzed_etfs[:25]):  # Limit to 25 for performance
            try:
                etf_upper = etf.upper()
                
                # Skip if already in portfolio (unless it's Core or Bonds - those we might want to increase)
                is_core = etf_upper in [e.upper() for e in core_etfs]
                is_bond = etf_upper in [e.upper() for e in bond_etfs]
                
                if etf_upper in current_holdings and not (is_core or is_bond):
                    logger.debug(f"Skipping {etf} - already in portfolio")
                    continue
                
                analysis = self.advisor.analyze_etf(etf, verbose=False)
                score = analysis.get("score", 0)
                
                # Check if already in portfolio with sufficient allocation
                holding = next((h for h in portfolio.get("holdings", []) if h.get("ticker", "").upper() == etf_upper), None)
                if holding:
                    holding_value = holding.get("current_value", 0)
                    holding_weight = (holding_value / portfolio_value * 100) if portfolio_value > 0 else 0
                    
                    # Skip if already has sufficient allocation
                    if is_core and holding_weight >= 15:  # Core should be 15-20% each
                        logger.debug(f"Skipping {etf} - already has {holding_weight:.1f}% allocation")
                        continue
                    elif is_bond and holding_weight >= 10:  # Bonds should be 10-15% each
                        logger.debug(f"Skipping {etf} - already has {holding_weight:.1f}% allocation")
                        continue
                    elif not (is_core or is_bond) and holding_weight >= 5:  # Satellite should be 5-10% each
                        logger.debug(f"Skipping {etf} - already has {holding_weight:.1f}% allocation")
                        continue
                
                # Check if portfolio actually needs this (80/20 balance check)
                # Calculate current allocation
                bonds_value = sum(h.get("current_value", 0) for h in portfolio.get("holdings", []) 
                                if h.get("ticker", "").upper() in [b.upper() for b in bond_etfs])
                bonds_percent = (bonds_value / portfolio_value * 100) if portfolio_value > 0 else 0
                
                # Only recommend bonds if portfolio has < 20% bonds
                if is_bond and bonds_percent >= 20:
                    logger.debug(f"Skipping {etf} - portfolio already has {bonds_percent:.1f}% bonds")
                    continue
                
                # Only recommend Core/Satellite if portfolio has < 80% stocks
                stocks_value = portfolio_value - bonds_value
                stocks_percent = (stocks_value / portfolio_value * 100) if portfolio_value > 0 else 100
                if (is_core or not is_bond) and stocks_percent >= 80:
                    # Still allow if bonds are needed (to balance)
                    if bonds_percent < 20:
                        pass  # Allow to balance
                    else:
                        logger.debug(f"Skipping {etf} - portfolio already has {stocks_percent:.1f}% stocks")
                        continue
                
                # Boost score for Core and Bonds (they're essential) - but cap at 100
                if is_core:
                    score = min(100, score + 15)
                    analysis["reasons"].append("Core holding - essential for portfolio stability")
                elif is_bond:
                    score = min(100, score + 20)
                    analysis["reasons"].append("Bond holding - essential for portfolio protection")
                
                # Ensure score doesn't exceed 100
                score = min(100, score)
                
                # Check for critical buy signals (lower threshold for Core/Bonds)
                threshold = 70 if (is_core or is_bond) else self.critical_threshold
                
                if score >= threshold:
                    expected_return = analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0)
                    reasons = analysis.get("reasons", [])
                    
                    # Filter out unrealistic returns (>50% is not realistic for 3 years)
                    if expected_return > 50 or expected_return < -50:
                        logger.debug(f"Skipping {etf} - unrealistic expected return: {expected_return:.1f}%")
                        continue
                    
                    # For Core and Bonds, lower the expected return requirement
                    min_return = 5 if (is_core or is_bond) else 10  # Lowered from 10/15
                    
                    # Only flag as critical if:
                    # 1. High score (>=threshold)
                    # 2. Reasonable expected return (filtered unrealistic ones)
                    # 3. Not leveraged (we exclude those)
                    # 4. Have cash available
                    is_leveraged = analysis.get("is_leveraged", False)
                    
                    if is_leveraged:
                        # Skip leveraged ETFs - not suitable for 80/20 strategy
                        logger.debug(f"Skipping {etf} - leveraged ETF, not suitable for balanced strategy")
                        continue
                    
                    # Check if we have cash available
                    if cash_available < 100:
                        logger.debug(f"Skipping {etf} - insufficient cash (${cash_available:.2f})")
                        continue
                    
                    # Only recommend if expected return is reasonable OR score is very high
                    if (expected_return > min_return and expected_return <= 50) or score >= 85:
                        leverage_mult = analysis.get("leverage_multiplier", 1.0)
                        
                        # Calculate recommended amount based on category and available cash
                        if is_core:
                            # Core gets higher allocation, but not more than available cash
                            recommended_amount = min(1500, cash_available * 0.3, cash_available * 0.5)
                            category_note = "Core holding - increase allocation"
                        elif is_bond:
                            # Bonds get moderate allocation, prioritize if portfolio lacks bonds
                            if bonds_percent < 15:
                                recommended_amount = min(1000, cash_available * 0.4, cash_available * 0.6)
                            else:
                                recommended_amount = min(1000, cash_available * 0.2, cash_available * 0.3)
                            category_note = "Bond holding - add protection"
                        else:
                            # Satellite gets smaller allocation
                            recommended_amount = min(800, cash_available * 0.15, cash_available * 0.25)
                            category_note = "Satellite holding - growth opportunity"
                        
                        # Ensure we don't recommend more than available cash
                        recommended_amount = min(recommended_amount, cash_available)
                        
                        if recommended_amount < 50:  # Skip if amount too small
                            logger.debug(f"Skipping {etf} - recommended amount too small (${recommended_amount:.2f})")
                            continue
                        
                        # Cap expected return display at 50% to avoid confusion
                        display_return = min(expected_return, 50) if expected_return > 0 else expected_return
                        
                        critical_buys.append({
                            "type": "BUY",
                            "ticker": etf,
                            "priority": "CRITICAL" if (score >= 85 or (is_bond and bonds_percent < 10)) else "HIGH",
                            "reason": f"{category_note} - Score: {score}/100. Expected 3yr return: {display_return:.1f}%.",
                            "amount": recommended_amount,
                            "expected_return": display_return,
                            "score": score,
                            "details": "; ".join(reasons[:3]),  # Top 3 reasons
                            "diversification": "Balanced 80/20 strategy",
                            "is_leveraged": False,
                            "leverage_multiplier": 1.0,
                            "category": "CORE" if is_core else ("BONDS" if is_bond else "SATELLITE")
                        })
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"   Progress: {i + 1}/{min(len(analyzed_etfs), 25)} ETFs analyzed...")
                    
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

