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
            if hist is not None and not hist.empty:
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
    
    def detect_emerging_trends(self) -> List[Dict]:
        """
        Detect emerging trends across all categories.
        This automatically identifies new hot sectors and trends.
        """
        excluded_categories = ["LEVERAGED_2X", "LEVERAGED_3X", "LEVERAGED_INVERSE", "CRYPTO"]
        emerging_trends = self.advisor.detect_emerging_trends(excluded_categories)
        
        # Filter to only very strong trends for critical alerts
        strong_trends = [t for t in emerging_trends if t.get("trend") == "STRONG_UPTREND" and t.get("avg_momentum", 0) > 8]
        
        return strong_trends
    
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
        
        # 3. Detect emerging trends and recommend replacing weak holdings
        print("\n🔍 Detecting emerging trends and hot sectors...")
        emerging_trends = self.detect_emerging_trends()
        if emerging_trends:
            print(f"   Found {len(emerging_trends)} emerging trends with strong momentum!")
            
            # Get current holdings for comparison
            holdings_analysis = portfolio_analysis.get("holdings_analysis", []) if portfolio_analysis else []
            portfolio_metrics = portfolio_analysis.get("portfolio_metrics", {}) if portfolio_analysis else {}
            total_value = portfolio_metrics.get("total_value", 0)
            
            for trend in emerging_trends[:3]:  # Top 3 emerging trends
                category = trend.get("category", "")
                momentum = trend.get("avg_momentum", 0)
                return_pct = trend.get("avg_return", 0)
                trend_etfs = trend.get("etfs", [])
                
                # Check if we should recommend replacing weak holdings with hot trend ETFs
                if trend_etfs and holdings_analysis:
                    # Find weak holdings (score < 50 or STRONG SELL)
                    weak_holdings = [
                        h for h in holdings_analysis 
                        if h.get("recommendation_score", 50) < 50 
                        or h.get("recommendation", "") == "STRONG SELL"
                    ]
                    
                    # Analyze the best ETF from the hot trend
                    best_trend_etf = None
                    best_trend_score = 0
                    for etf_ticker in trend_etfs[:2]:  # Check top 2 from trend
                        try:
                            etf_analysis = self.advisor.analyze_etf(etf_ticker, verbose=False)
                            etf_score = etf_analysis.get("score", 0)
                            if etf_score > best_trend_score:
                                best_trend_score = etf_score
                                best_trend_etf = {
                                    "ticker": etf_ticker,
                                    "score": etf_score,
                                    "analysis": etf_analysis,
                                    "category": category,
                                    "momentum": momentum,
                                    "return": return_pct
                                }
                        except Exception as e:
                            logger.debug(f"Failed to analyze trend ETF {etf_ticker}: {e}")
                            continue
                    
                    # If we found a strong trend ETF and have weak holdings, recommend replacement
                    if best_trend_etf and best_trend_etf["score"] >= 70 and weak_holdings:
                        # Find the weakest holding to replace
                        weakest_holding = min(weak_holdings, key=lambda h: h.get("recommendation_score", 0))
                        weakest_score = weakest_holding.get("recommendation_score", 0)
                        weakest_ticker = weakest_holding.get("ticker", "")
                        weakest_value = weakest_holding.get("current_value", 0)
                        
                        # Only recommend if the trend ETF is significantly better (at least 20 points difference)
                        if best_trend_etf["score"] >= weakest_score + 20:
                            shares_to_sell = int(weakest_holding.get("quantity", 0) * 0.5)  # Sell 50% of weak holding
                            sell_amount = shares_to_sell * weakest_holding.get("current_price", 0)
                            
                            if sell_amount > 100:  # Only if meaningful amount
                                # Calculate buy shares and amount based on current price of the trend ETF
                                trend_analysis = best_trend_etf.get("analysis", {})
                                buy_price = trend_analysis.get("current_price", 0)
                                remaining_cash = 0
                                
                                if buy_price > 0:
                                    buy_shares = int(sell_amount / buy_price)
                                    buy_amount = buy_shares * buy_price
                                    remaining_cash = sell_amount - buy_amount
                                    
                                    # If we have significant remaining cash (>$50), try to buy one more share
                                    if remaining_cash > 50 and remaining_cash >= buy_price:
                                        buy_shares += 1
                                        buy_amount = buy_shares * buy_price
                                        remaining_cash = sell_amount - buy_amount
                                else:
                                    # Fallback: try to get price from yfinance
                                    try:
                                        import yfinance as yf
                                        stock = yf.Ticker(best_trend_etf["ticker"])
                                        hist = stock.history(period="1d")
                                        if hist is not None and not hist.empty and 'Close' in hist.columns:
                                            buy_price = float(hist['Close'].iloc[-1])
                                            buy_shares = int(sell_amount / buy_price)
                                            buy_amount = buy_shares * buy_price
                                            remaining_cash = sell_amount - buy_amount
                                            
                                            # If we have significant remaining cash (>$50), try to buy one more share
                                            if remaining_cash > 50 and remaining_cash >= buy_price:
                                                buy_shares += 1
                                                buy_amount = buy_shares * buy_price
                                                remaining_cash = sell_amount - buy_amount
                                        else:
                                            buy_price = 0
                                            buy_shares = 0
                                            buy_amount = 0
                                            remaining_cash = 0
                                    except Exception:
                                        buy_price = 0
                                        buy_shares = 0
                                        buy_amount = 0
                                        remaining_cash = 0
                                
                                # Only recommend REPLACE if we can actually buy at least 1 share
                                if buy_shares > 0 and buy_price > 0:
                                    critical_items.append({
                                        "type": "REPLACE",
                                        "priority": "HIGH",
                                        "sell_ticker": weakest_ticker,
                                        "sell_score": weakest_score,
                                        "sell_amount": sell_amount,
                                        "sell_shares": shares_to_sell,
                                        "buy_ticker": best_trend_etf["ticker"],
                                        "buy_price": buy_price,
                                        "buy_shares": buy_shares,
                                        "buy_amount": buy_amount,
                                        "remaining_cash": remaining_cash,
                                        "buy_score": best_trend_etf["score"],
                                        "buy_category": category,
                                        "reason": f"🔄 REPLACE: Sell {weakest_ticker} (Score: {weakest_score:.1f}/100) → Buy {best_trend_etf['ticker']} from 🔥 {category} trend (Score: {best_trend_etf['score']:.1f}/100, {momentum:.1f}% momentum)",
                                        "momentum": momentum,
                                        "return": return_pct,
                                        "details": f"Replace weak holding with hot trend: {category} showing {momentum:.1f}% momentum and {return_pct:.1f}% return"
                                    })
                                    print(f"   🔄 REPLACE: {weakest_ticker} (Score: {weakest_score:.1f}) → {best_trend_etf['ticker']} from {category} (Score: {best_trend_etf['score']:.1f})")
                                    continue  # Skip adding as separate EMERGING_TREND if we already added REPLACE
                                else:
                                    # Can't buy enough shares, skip this replacement
                                    logger.debug(f"Skipping REPLACE {weakest_ticker} → {best_trend_etf['ticker']}: sell_amount ${sell_amount:.2f} insufficient for buy_price ${buy_price:.2f}")
                
                # Only add emerging trend alerts if:
                # 1. We have cash available (>$100) to potentially buy, OR
                # 2. We already have other specific recommendations (BUY/SELL/REPLACE)
                has_specific_actions = any(item.get("type") in ["BUY", "SELL", "REPLACE"] for item in critical_items)
                
                # Check if we have enough cash to buy at least 1 share of the cheapest ETF in the trend
                has_cash_for_trend = False
                if cash_available > 100 and trend_etfs:
                    # Try to get price of cheapest ETF in trend
                    try:
                        import yfinance as yf
                        min_price = float('inf')
                        for etf_ticker in trend_etfs[:3]:  # Check first 3 ETFs
                            try:
                                stock = yf.Ticker(etf_ticker)
                                hist = stock.history(period="1d")
                                if hist is not None and not hist.empty and 'Close' in hist.columns:
                                    price = float(hist['Close'].iloc[-1])
                                    min_price = min(min_price, price)
                            except Exception:
                                continue
                        
                        # If we found a price and have enough cash, we can potentially buy
                        if min_price != float('inf') and cash_available >= min_price:
                            has_cash_for_trend = True
                        elif min_price != float('inf'):
                            logger.debug(f"Trend {category}: cash ${cash_available:.2f} insufficient for cheapest ETF (${min_price:.2f})")
                    except Exception as e:
                        logger.debug(f"Failed to check trend ETF prices: {e}")
                        # Fallback: use simple cash check
                        has_cash_for_trend = cash_available > 100
                elif cash_available > 100:
                    # Fallback: if we can't check prices, use simple threshold
                    has_cash_for_trend = True
                
                if has_cash_for_trend or has_specific_actions:
                    # If no replacement recommended, add as regular emerging trend alert
                    critical_items.append({
                        "type": "EMERGING_TREND",
                        "category": category,
                        "priority": "HIGH",
                        "reason": f"🔥 EMERGING TREND: {category} showing strong momentum ({momentum:.1f}% in 20 days, {return_pct:.1f}% in 6mo)",
                        "etfs": trend_etfs,
                        "momentum": momentum,
                        "return": return_pct,
                        "score": trend.get("score", 50)
                    })
                    print(f"   🔥 {category}: {momentum:.1f}% momentum, {return_pct:.1f}% return - ETFs: {', '.join(trend_etfs)}")
                else:
                    print(f"   🔥 {category}: {momentum:.1f}% momentum detected, but insufficient cash (${cash_available:.2f}) - skipping alert")
        else:
            print("   No strong emerging trends detected at this time")
        
        # 4. Deep analysis: Compare ALL existing holdings with market alternatives
        print("\n🔍 Deep Analysis: Comparing existing holdings with market alternatives...")
        if portfolio_analysis and holdings_analysis:
            replacement_opportunities = self.find_better_alternatives(holdings_analysis, portfolio_metrics)
            
            # Also check for over-concentration and hot sector opportunities (same logic as analyze)
            concentration_opportunities = self.analyzer.find_concentration_opportunities(holdings_analysis, portfolio_metrics)
            if concentration_opportunities:
                # Convert to critical_alert format
                for opp in concentration_opportunities:
                    replacement_opportunities.append({
                        "type": "REPLACE",
                        "priority": "HIGH" if opp.get("category") == "DIVERSIFICATION" else "MEDIUM",
                        "sell_ticker": opp.get("sell_ticker", ""),
                        "sell_score": opp.get("sell_score", 50),
                        "sell_amount": opp.get("sell_amount", 0),
                        "sell_shares": opp.get("sell_shares", 0),
                        "sell_category": opp.get("category", "SATELLITE"),
                        "buy_ticker": opp.get("buy_ticker", ""),
                        "buy_price": opp.get("buy_price", 0),
                        "buy_shares": opp.get("buy_shares", 0),
                        "buy_amount": opp.get("buy_amount", 0),
                        "remaining_cash": opp.get("sell_amount", 0) - opp.get("buy_amount", 0),
                        "buy_score": opp.get("buy_score", 0),
                        "buy_category": opp.get("category", "SATELLITE"),
                        "score_improvement": opp.get("score_improvement", 0),
                        "reason": f"🔄 OPTIMIZE: {opp.get('sell_ticker', '')} → {opp.get('buy_ticker', '')} - {opp.get('reason', '')}",
                        "details": f"Replace {opp.get('replace_percentage', 0)*100:.0f}% of {opp.get('sell_ticker', '')} with {opp.get('buy_ticker', '')}. Expected return: {opp.get('expected_return', 0):.1f}% vs {opp.get('current_return', 0):.1f}%",
                        "strategy": "80/20 Balanced Growth"
                    })
            
            if replacement_opportunities:
                # Filter out duplicate buy recommendations (same ETF recommended multiple times)
                # Keep only the best replacement opportunity for each buy_ticker
                seen_buy_tickers = {}
                filtered_opportunities = []
                for opp in replacement_opportunities:
                    buy_ticker = opp.get("buy_ticker", "").upper()
                    if buy_ticker not in seen_buy_tickers:
                        seen_buy_tickers[buy_ticker] = opp
                        filtered_opportunities.append(opp)
                    else:
                        # Keep the one with higher priority or better score improvement
                        existing = seen_buy_tickers[buy_ticker]
                        existing_priority = existing.get("priority", "MEDIUM")
                        new_priority = opp.get("priority", "MEDIUM")
                        priority_order = {"HIGH": 0, "MEDIUM": 1}
                        if priority_order.get(new_priority, 2) < priority_order.get(existing_priority, 2):
                            # New one has higher priority, replace
                            seen_buy_tickers[buy_ticker] = opp
                            filtered_opportunities = [o for o in filtered_opportunities if o != existing]
                            filtered_opportunities.append(opp)
                        elif opp.get("score_improvement", 0) > existing.get("score_improvement", 0):
                            # New one has better score improvement, replace
                            seen_buy_tickers[buy_ticker] = opp
                            filtered_opportunities = [o for o in filtered_opportunities if o != existing]
                            filtered_opportunities.append(opp)
                
                if filtered_opportunities:
                    print(f"   Found {len(filtered_opportunities)} better alternatives for existing holdings!")
                    if len(replacement_opportunities) > len(filtered_opportunities):
                        print(f"   (Filtered out {len(replacement_opportunities) - len(filtered_opportunities)} duplicate recommendations)")
                    critical_items.extend(filtered_opportunities)
                else:
                    print("   ✅ All existing holdings are performing well - no better alternatives found")
            else:
                print("   ✅ All existing holdings are performing well - no better alternatives found")
        
        # 5. Check for market anomalies or extreme opportunities
        print("\nChecking for market anomalies...")
        anomalies = self.check_market_anomalies()
        critical_items.extend(anomalies)
        
        # Sort by priority (CRITICAL > HIGH > MEDIUM)
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        critical_items.sort(key=lambda x: priority_order.get(x.get("priority", "MEDIUM"), 3))
        
        # has_critical should be True only if there are specific actionable recommendations
        # (BUY/SELL/REPLACE), not just trends
        specific_action_types = ["BUY", "SELL", "REPLACE"]
        has_specific_actions = any(item.get("type") in specific_action_types for item in critical_items)
        
        return {
            "critical_items": critical_items,
            "has_critical": has_specific_actions,  # Only true if specific actions exist
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
        # Note: Crypto is excluded by default for balanced risk profile
        # To enable crypto, remove "CRYPTO" from excluded_categories (not recommended for young families)
        excluded_categories = ["LEVERAGED_2X", "LEVERAGED_3X", "LEVERAGED_INVERSE", "CRYPTO"]
        
        # Core ETFs (essential for portfolio stability)
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        
        # Satellite ETFs (growth, but not too risky)
        # Expanded to cover more opportunities while maintaining balanced risk
        satellite_categories = [
            # Core Satellite (essential diversification)
            "US_SMALL_CAP", "TECHNOLOGY", "HEALTHCARE", "EMERGING_MARKETS",
            # High-growth trends (but not leveraged)
            "AI_AND_ROBOTICS", "QUANTUM_COMPUTING", "SEMICONDUCTORS", "CLOUD_COMPUTING", "CYBERSECURITY",
            "ELECTRIC_VEHICLES", "CLEAN_ENERGY",
            # Defensive growth
            "REAL_ESTATE", "INFRASTRUCTURE",
            # Investment styles
            "DIVIDEND", "GROWTH", "VALUE",
            # Sector diversification
            "FINANCIAL", "ENERGY", "CONSUMER",
            # Thematic trends
            "ESG", "BIOTECH"
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
                etfs = self.advisor.ETF_CATEGORIES[category][:3]  # Top 3 from each category (increased from 2)
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
        # Increased limit to 60 to cover more opportunities (was 25)
        for i, etf in enumerate(analyzed_etfs[:60]):  # Limit to 60 for comprehensive coverage
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
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{min(len(analyzed_etfs), 60)} ETFs analyzed...")
                    
            except Exception as e:
                logger.debug(f"Failed to analyze ETF {etf}: {e}")
                continue  # Skip failed analyses
        
        return critical_buys
    
    def find_better_alternatives(self, holdings_analysis: List[Dict], portfolio_metrics: Dict) -> List[Dict]:
        """
        Deep analysis: Compare ALL existing holdings with market alternatives.
        Finds better ETFs even if current holdings are not weak.
        This ensures portfolio is always optimized according to 80/20 strategy.
        """
        replacement_opportunities = []
        
        # Get current portfolio
        portfolio = self.analyzer.load_portfolio()
        current_tickers = [h.get("ticker", "").upper() for h in portfolio.get("holdings", [])]
        total_value = portfolio_metrics.get("total_value", 0)
        
        # Define categories for comparison
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
        
        # Get all satellite categories from advisor
        satellite_categories = [
            "US_SMALL_CAP", "TECHNOLOGY", "HEALTHCARE", "EMERGING_MARKETS",
            "AI_AND_ROBOTICS", "QUANTUM_COMPUTING", "SEMICONDUCTORS", "CLOUD_COMPUTING", "CYBERSECURITY",
            "ELECTRIC_VEHICLES", "CLEAN_ENERGY", "REAL_ESTATE", "INFRASTRUCTURE",
            "DIVIDEND", "GROWTH", "VALUE", "FINANCIAL", "ENERGY", "CONSUMER", "ESG", "BIOTECH"
        ]
        
        print(f"   Analyzing {len(holdings_analysis)} existing holdings for better alternatives...")
        
        for holding in holdings_analysis:
            ticker = holding.get("ticker", "").upper()
            current_score = holding.get("recommendation_score", 50)
            current_value = holding.get("current_value", 0)
            current_weight = (current_value / total_value * 100) if total_value > 0 else 0
            
            # Skip if holding is too small (<2% of portfolio) - not worth replacing
            if current_weight < 2:
                continue
            
            # Determine category
            is_core = ticker in [e.upper() for e in core_etfs]
            is_bond = ticker in [e.upper() for e in bond_etfs]
            
            # Find candidates in same category
            candidates = []
            category_name = ""
            
            if is_core:
                candidates = [e for e in core_etfs if e.upper() != ticker and e.upper() not in current_tickers]
                category_name = "CORE"
            elif is_bond:
                candidates = [e for e in bond_etfs if e.upper() != ticker and e.upper() not in current_tickers]
                category_name = "BONDS"
            else:
                # For satellite ETFs, search in all satellite categories
                category_name = "SATELLITE"
                for cat in satellite_categories:
                    if cat in self.advisor.ETF_CATEGORIES:
                        cat_etfs = self.advisor.ETF_CATEGORIES[cat]
                        # Add top 2 from each category that we don't already have
                        for etf in cat_etfs[:2]:
                            if etf.upper() != ticker and etf.upper() not in current_tickers:
                                if etf.upper() not in [c.upper() for c in candidates]:
                                    candidates.append(etf)
            
            # Analyze candidates to find better alternatives
            best_alternative = None
            best_score = 0
            best_analysis = None
            
            for candidate in candidates[:5]:  # Check top 5 candidates
                try:
                    candidate_analysis = self.advisor.analyze_etf(candidate, verbose=False)
                    candidate_score = candidate_analysis.get("score", 0)
                    
                    # Boost score for Core and Bonds (they're essential for 80/20 strategy)
                    if is_core:
                        candidate_score = min(100, candidate_score + 15)
                    elif is_bond:
                        candidate_score = min(100, candidate_score + 20)
                    
                    # Check if significantly better (at least 15 points higher for moderate replacement)
                    # Or at least 25 points higher for strong replacement
                    score_diff = candidate_score - current_score
                    
                    if score_diff >= 15 and candidate_score > best_score:
                        # Additional checks: ensure alternative has good fundamentals
                        expected_return = candidate_analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0)
                        
                        # Filter unrealistic returns
                        if -50 <= expected_return <= 50:
                            best_score = candidate_score
                            best_alternative = candidate
                            best_analysis = candidate_analysis
                            
                except Exception as e:
                    logger.debug(f"Failed to analyze alternative {candidate}: {e}")
                    continue
            
            # If found significantly better alternative, recommend replacement
            if best_alternative and best_analysis:
                score_diff = best_score - current_score
                priority = "HIGH" if score_diff >= 25 else "MEDIUM"
                
                # Only recommend if meaningful improvement
                if score_diff >= 15:
                    # Recommend replacing 30-50% depending on score difference
                    replace_percentage = 0.5 if score_diff >= 25 else 0.3
                    shares_to_sell = int(holding.get("quantity", 0) * replace_percentage)
                    sell_amount = shares_to_sell * holding.get("current_price", 0)
                    
                    if sell_amount > 100:  # Only if meaningful amount
                        expected_return = best_analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0)
                        current_return = holding.get("mid_term_forecast", {}).get("expected_3yr_return", 0) if isinstance(holding.get("mid_term_forecast"), dict) else 0
                        
                        # Calculate buy shares and amount based on current price of the new ETF
                        # Use ALL money from sale to buy new ETF (maximize allocation)
                        buy_price = best_analysis.get("current_price", 0)
                        remaining_cash = 0
                        
                        if buy_price > 0:
                            # Calculate how many shares we can buy with the sell amount
                            buy_shares = int(sell_amount / buy_price)
                            buy_amount = buy_shares * buy_price
                            remaining_cash = sell_amount - buy_amount
                            
                            # If we have significant remaining cash (>$50), try to buy one more share
                            # But only if we have enough money for the additional share
                            if remaining_cash > 50 and remaining_cash >= buy_price:
                                buy_shares += 1
                                buy_amount = buy_shares * buy_price
                                remaining_cash = sell_amount - buy_amount
                        else:
                            # Fallback: try to get price from yfinance
                            try:
                                import yfinance as yf
                                stock = yf.Ticker(best_alternative)
                                hist = stock.history(period="1d")
                                if hist is not None and not hist.empty and 'Close' in hist.columns:
                                    buy_price = float(hist['Close'].iloc[-1])
                                    buy_shares = int(sell_amount / buy_price)
                                    buy_amount = buy_shares * buy_price
                                    remaining_cash = sell_amount - buy_amount
                                    
                                    # If we have significant remaining cash (>$50), try to buy one more share
                                    # But only if we have enough money for the additional share
                                    if remaining_cash > 50 and remaining_cash >= buy_price:
                                        buy_shares += 1
                                        buy_amount = buy_shares * buy_price
                                        remaining_cash = sell_amount - buy_amount
                                else:
                                    buy_price = 0
                                    buy_shares = 0
                                    buy_amount = 0
                                    remaining_cash = 0
                            except Exception:
                                buy_price = 0
                                buy_shares = 0
                                buy_amount = 0
                                remaining_cash = 0
                        
                        replacement_opportunities.append({
                            "type": "REPLACE",
                            "priority": priority,
                            "sell_ticker": ticker,
                            "sell_score": current_score,
                            "sell_amount": sell_amount,
                            "sell_shares": shares_to_sell,
                            "sell_category": category_name,
                            "buy_ticker": best_alternative,
                            "buy_price": buy_price,
                            "buy_shares": buy_shares,
                            "buy_amount": buy_amount,
                            "remaining_cash": remaining_cash,
                            "buy_score": best_score,
                            "buy_category": category_name,
                            "score_improvement": score_diff,
                            "reason": f"🔄 OPTIMIZE: {ticker} (Score: {current_score:.1f}/100) → {best_alternative} (Score: {best_score:.1f}/100, +{score_diff:.1f} points) - Better performance for {category_name} allocation",
                            "details": f"Replace {replace_percentage*100:.0f}% of {ticker} with {best_alternative} to improve {category_name} allocation. Expected return: {expected_return:.1f}% vs {current_return:.1f}%",
                            "strategy": "80/20 Balanced Growth"
                        })
                        print(f"   🔄 {ticker} (Score: {current_score:.1f}) → {best_alternative} (Score: {best_score:.1f}, +{score_diff:.1f})")
        
        return replacement_opportunities
    
    def check_market_anomalies(self) -> List[Dict]:
        """Check for market anomalies that require immediate attention."""
        anomalies = []
        
        try:
            # Check SPY for extreme movements
            spy = yf.Ticker("SPY")
            hist = spy.history(period="5d")
            
            if hist is not None and not hist.empty and 'Close' in hist.columns and len(hist) >= 2:
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
            replace_count = sum(1 for item in critical_items if item.get("type") == "REPLACE")
            trend_count = sum(1 for item in critical_items if item.get("type") == "EMERGING_TREND")
            
            # Only send email if there are specific actionable recommendations (BUY/SELL/REPLACE)
            # Trends alone are not actionable without cash or specific recommendations
            has_specific_actions = buy_count > 0 or sell_count > 0 or replace_count > 0
            
            if not has_specific_actions:
                print("\n✅ No specific actionable recommendations (only trends detected, but no cash or weak holdings to replace).")
                print("   Email not sent - trends are informational only when no actions are available.")
                return False
            
            # Build subject line (only for specific actions)
            actions = []
            if replace_count > 0:
                actions.append(f"{replace_count} Replace{'s' if replace_count != 1 else ''}")
            if buy_count > 0:
                actions.append(f"{buy_count} Buy{'s' if buy_count != 1 else ''}")
            if sell_count > 0:
                actions.append(f"{sell_count} Sell{'s' if sell_count != 1 else ''}")
            
            subject = f"URGENT: {', '.join(actions)} Required" if actions else "Portfolio Alert"
            
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
        
        critical_items = results.get("critical_items", [])
        specific_action_types = ["BUY", "SELL", "REPLACE"]
        specific_actions = [item for item in critical_items if item.get("type") in specific_action_types]
        trends_only = [item for item in critical_items if item.get("type") == "EMERGING_TREND"]
        
        if results.get("has_critical") and specific_actions:
            print(f"\n⚠️  Found {len(specific_actions)} critical action(s) requiring immediate attention:")
            for item in specific_actions:
                print(f"\n   [{item.get('priority', 'MEDIUM')}] {item.get('type', 'ACTION')}: {item.get('ticker', item.get('title', 'N/A'))}")
                print(f"      Reason: {item.get('reason', 'N/A')}")
            
            # Also show trends if any (as informational)
            if trends_only:
                print(f"\n📊 Also detected {len(trends_only)} hot trend(s) (included in email):")
                for item in trends_only:
                    print(f"   🔥 {item.get('category', 'N/A')}: {item.get('momentum', 0):.1f}% momentum")
            
            # Send email alerts
            self.send_alerts(results)
        elif trends_only:
            print(f"\n📊 Detected {len(trends_only)} hot trend(s), but no specific actionable recommendations:")
            for item in trends_only:
                print(f"   🔥 {item.get('category', 'N/A')}: {item.get('momentum', 0):.1f}% momentum - ETFs: {', '.join(item.get('etfs', []))}")
            print("\n   ℹ️  No email sent - trends are informational only when:")
            print("      • No cash available for purchases (<$100)")
            print("      • No weak holdings to replace with trend ETFs")
            print("      • Portfolio is balanced and performing well")
        else:
            print("\n✅ No critical actions required. Portfolio is in good shape.")
            print("   (No email sent - only critical alerts trigger emails)")
        
        return results

if __name__ == "__main__":
    alert_system = CriticalAlertSystem()
    results = alert_system.run()
    
    # Exit with code 0 if no critical items, 1 if critical items found
    sys.exit(0 if not results.get("has_critical") else 1)

