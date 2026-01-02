#!/usr/bin/env python3
"""
Deposit Advisor - Recommends ETF purchases based on deposit amount and current portfolio.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from portfolio_analyzer import PortfolioAnalyzer
from advanced_analysis import AdvancedAnalyzer

class DepositAdvisor:
    """Advises on ETF purchases when depositing funds."""
    
    # Popular ETF categories and tickers - organized by trends and industries
    ETF_CATEGORIES = {
        # Core Market Exposure
        "US_LARGE_CAP": ["SPY", "VOO", "IVV", "QQQ"],
        "US_SMALL_CAP": ["IWM", "VB", "IJR"],
        "INTERNATIONAL": ["VEA", "VXUS", "EFA", "EEM"],
        "EMERGING_MARKETS": ["VWO", "EEM", "IEMG"],
        
        # Technology & Innovation (High Growth Trends)
        "TECHNOLOGY": ["XLK", "VGT", "FTEC", "QQQ"],
        "AI_AND_ROBOTICS": ["BOTZ", "ROBO", "AIQ", "IRBO", "CHAT"],
        "SEMICONDUCTORS": ["SOXX", "SMH", "XSD"],
        "CLOUD_COMPUTING": ["WCLD", "SKYY", "CLOU"],
        "CYBERSECURITY": ["HACK", "CIBR", "BUG"],
        "ELECTRIC_VEHICLES": ["DRIV", "IDRV", "KARS"],
        "CLEAN_ENERGY": ["ICLN", "QCLN", "PBW"],
        
        # Healthcare & Biotech (Defensive + Growth)
        "HEALTHCARE": ["XLV", "VHT", "IBB"],
        "BIOTECH": ["IBB", "XBI", "BBH"],
        "GENOMICS": ["GNOM", "ARKG", "HELX"],
        
        # Financial Services
        "FINANCIAL": ["XLF", "VFH", "IYF"],
        "FINTECH": ["FINX", "IPAY", "ARKF"],
        
        # Energy & Commodities
        "ENERGY": ["XLE", "VDE", "IYE"],
        "OIL_GAS": ["XOP", "XES", "IEZ"],
        "GOLD": ["GLD", "IAU", "SGOL"],
        "SILVER": ["SLV", "SIVR", "PSLV"],
        "COMMODITIES": ["DBC", "GSG", "PDBC"],
        
        # Consumer & Retail
        "CONSUMER": ["XLY", "VCR", "IYC"],
        "CONSUMER_STAPLES": ["XLP", "VDC", "IYK"],
        "E_COMMERCE": ["IBUY", "ONLN", "CLIX"],
        
        # Real Estate & Infrastructure
        "REAL_ESTATE": ["VNQ", "SCHH", "IYR"],
        "INFRASTRUCTURE": ["IFRA", "PAVE", "IGF"],
        
        # Fixed Income
        "BONDS": ["BND", "AGG", "TLT"],
        "HIGH_YIELD": ["HYG", "JNK", "SHYG"],
        "TIPS": ["TIP", "SCHP", "VTIP"],
        
        # Digital Assets
        "CRYPTO": ["GBTC", "BITO", "ETHE"],
        "BLOCKCHAIN": ["BLOK", "LEGR", "KOIN"],
        
        # Investment Styles
        "DIVIDEND": ["VYM", "SCHD", "DVY"],
        "GROWTH": ["VUG", "IVW", "IWF"],
        "VALUE": ["VTV", "IVE", "IWD"],
        "MOMENTUM": ["MTUM", "QMOM", "DWAS"],
        
        # Thematic & Future Trends
        "ESG": ["ESG", "ESGD", "SUSL"],
        "WATER": ["PHO", "CGW", "FIW"],
        "AGRICULTURE": ["DBA", "MOO", "VEGI"],
        "DEFENSE": ["ITA", "XAR", "PPA"],
        "AEROSPACE": ["ITA", "XAR", "PPA"]
    }
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.analyzer = PortfolioAnalyzer(portfolio_file)
        self.advanced_analyzer = AdvancedAnalyzer()
        self.exchange_rate_usd_ils = 3.7  # Approximate, should be fetched from API
    
    def load_portfolio(self) -> Dict:
        """Load portfolio from JSON file."""
        return self.analyzer.load_portfolio()
    
    def get_exchange_rate(self) -> float:
        """Get current USD/ILS exchange rate."""
        try:
            # Try to get real-time exchange rate from yfinance
            usd_ils = yf.Ticker("USDILS=X")
            hist = usd_ils.history(period="1d")
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1])
                self.exchange_rate_usd_ils = rate
                return rate
            else:
                # Fallback to approximate rate
                return self.exchange_rate_usd_ils
        except Exception as e:
            print(f"Warning: Could not fetch exchange rate, using default: {e}")
            return self.exchange_rate_usd_ils
    
    def analyze_industry_trends(self, category: str) -> Dict:
        """Analyze industry trends for a category by comparing all ETFs in that category."""
        if category not in self.ETF_CATEGORIES:
            return {"trend": "UNKNOWN", "score": 50, "reason": "Category not found"}
        
        etfs = self.ETF_CATEGORIES[category]
        category_returns = []
        category_momentums = []
        
        for etf in etfs[:3]:  # Analyze top 3 ETFs in category
            try:
                stock = yf.Ticker(etf)
                data = stock.history(period="6mo")
                if not data.empty and len(data) > 20:
                    # Calculate 6-month return
                    return_6m = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    category_returns.append(return_6m)
                    
                    # Calculate recent momentum (last 20 days)
                    momentum = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100 if len(data) >= 20 else 0
                    category_momentums.append(momentum)
            except Exception:
                continue
        
        if not category_returns:
            return {"trend": "UNKNOWN", "score": 50, "reason": "Insufficient data"}
        
        avg_return = np.mean(category_returns)
        avg_momentum = np.mean(category_momentums)
        
        # Determine trend
        if avg_return > 15 and avg_momentum > 5:
            trend = "STRONG_UPTREND"
            score = 75
            reason = f"Strong industry performance: {avg_return:.1f}% return, {avg_momentum:.1f}% momentum"
        elif avg_return > 5 and avg_momentum > 0:
            trend = "UPTREND"
            score = 65
            reason = f"Positive industry trend: {avg_return:.1f}% return"
        elif avg_return < -10 or avg_momentum < -5:
            trend = "DOWNTREND"
            score = 35
            reason = f"Weak industry performance: {avg_return:.1f}% return"
        else:
            trend = "NEUTRAL"
            score = 50
            reason = f"Neutral industry trend: {avg_return:.1f}% return"
        
        return {
            "trend": trend,
            "score": score,
            "reason": reason,
            "avg_return": avg_return,
            "avg_momentum": avg_momentum
        }
    
    def analyze_etf(self, ticker: str) -> Dict:
        """Analyze an ETF for investment potential with industry trend analysis."""
        print(f"Analyzing ETF: {ticker}...")
        
        analysis = {
            "ticker": ticker,
            "score": 0,
            "current_price": 0,
            "recommendation": "NEUTRAL",
            "reasons": [],
            "industry_trend": {}
        }
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get basic info
            analysis["name"] = info.get("longName", ticker)
            analysis["category"] = info.get("category", "Unknown")
            analysis["expense_ratio"] = info.get("annualReportExpenseRatio", 0) * 100
            
            # Get price - check market status first
            market_status, market_message = self.analyzer.is_market_open()
            
            # Try to get real-time price if market is open
            price_found = False
            if market_status:
                try:
                    price = float(stock.fast_info.get('lastPrice', 0))
                    if price > 0:
                        analysis["current_price"] = price
                        price_found = True
                except Exception:
                    pass  # Fall through to historical data
            
            # Fallback to historical data (last close) if real-time price not found
            if not price_found:
                hist = stock.history(period="1d")
                if not hist.empty:
                    analysis["current_price"] = float(hist['Close'].iloc[-1])
                else:
                    try:
                        analysis["current_price"] = float(stock.fast_info.get('lastPrice', 0))
                    except Exception:
                        return analysis
            
            # Get historical data for analysis
            data = stock.history(period="1y")
            if data.empty:
                return analysis
            
            # Calculate metrics
            returns = data['Close'].pct_change().dropna()
            
            # Score calculation
            score = 50  # Base score
            
            # 1. Performance (30 points)
            annual_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
            if annual_return > 20:
                score += 20
                analysis["reasons"].append(f"Strong annual return: {annual_return:.1f}%")
            elif annual_return > 10:
                score += 10
                analysis["reasons"].append(f"Good annual return: {annual_return:.1f}%")
            elif annual_return < -10:
                score -= 15
                analysis["reasons"].append(f"Negative annual return: {annual_return:.1f}%")
            
            # 2. Volatility (20 points)
            volatility = returns.std() * np.sqrt(252) * 100
            if volatility < 15:
                score += 10
                analysis["reasons"].append(f"Low volatility: {volatility:.1f}%")
            elif volatility > 30:
                score -= 10
                analysis["reasons"].append(f"High volatility: {volatility:.1f}%")
            
            # 3. Momentum (20 points)
            recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100 if len(data) >= 20 else 0
            if recent_return > 5:
                score += 15
                analysis["reasons"].append(f"Positive momentum: {recent_return:.1f}%")
            elif recent_return < -5:
                score -= 15
                analysis["reasons"].append(f"Negative momentum: {recent_return:.1f}%")
            
            # 4. Expense ratio (10 points)
            if analysis["expense_ratio"] < 0.1:
                score += 10
                analysis["reasons"].append(f"Low expense ratio: {analysis['expense_ratio']:.2f}%")
            elif analysis["expense_ratio"] > 0.5:
                score -= 5
                analysis["reasons"].append(f"High expense ratio: {analysis['expense_ratio']:.2f}%")
            
            # 5. Volume/Liquidity (10 points)
            avg_volume = data['Volume'].mean()
            if avg_volume > 1_000_000:
                score += 10
                analysis["reasons"].append("High liquidity")
            elif avg_volume < 100_000:
                score -= 5
                analysis["reasons"].append("Low liquidity")
            
            # 6. Technical indicators (10 points)
            sma_20 = data['Close'].tail(20).mean()
            sma_50 = data['Close'].tail(min(50, len(data))).mean()
            current_price = data['Close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                score += 10
                analysis["reasons"].append("Bullish trend (price above moving averages)")
            elif current_price < sma_20 < sma_50:
                score -= 10
                analysis["reasons"].append("Bearish trend (price below moving averages)")
            
            # 7. Industry trend analysis (15 points)
            etf_category = None
            for cat, etfs in self.ETF_CATEGORIES.items():
                if ticker in etfs:
                    etf_category = cat
                    break
            
            if etf_category:
                industry_trend = self.analyze_industry_trends(etf_category)
                analysis["industry_trend"] = industry_trend
                
                # Add industry trend to score
                if industry_trend["trend"] == "STRONG_UPTREND":
                    score += 15
                    analysis["reasons"].append(f"🔥 Hot industry: {industry_trend['reason']}")
                elif industry_trend["trend"] == "UPTREND":
                    score += 10
                    analysis["reasons"].append(f"📈 Growing industry: {industry_trend['reason']}")
                elif industry_trend["trend"] == "DOWNTREND":
                    score -= 10
                    analysis["reasons"].append(f"📉 Declining industry: {industry_trend['reason']}")
            
            # 8. Mid-term yield optimization (20 points) - ULTIMATE BROKER FEATURE
            try:
                mid_term_analysis = self.advanced_analyzer.optimize_mid_term_yield([{"ticker": ticker}], target_years=3)
                if mid_term_analysis and len(mid_term_analysis) > 0:
                    mt_analysis = mid_term_analysis[0].get("mid_term_analysis", {})
                    expected_return = mt_analysis.get("expected_return", 0)
                    optimization_score = mt_analysis.get("optimization_score", 0)
                    
                    analysis["mid_term_yield"] = {
                        "expected_3yr_return": expected_return,
                        "optimization_score": optimization_score,
                        "sharpe_ratio": mt_analysis.get("sharpe_ratio", 0),
                        "win_rate": mt_analysis.get("win_rate", 50)
                    }
                    
                    # Add to score based on mid-term yield potential
                    if expected_return > 15:
                        score += 20
                        analysis["reasons"].append(f"🎯 Excellent mid-term yield potential: {expected_return:.1f}% (3yr forecast)")
                    elif expected_return > 10:
                        score += 15
                        analysis["reasons"].append(f"📊 Strong mid-term yield potential: {expected_return:.1f}% (3yr forecast)")
                    elif expected_return > 5:
                        score += 10
                        analysis["reasons"].append(f"✅ Good mid-term yield potential: {expected_return:.1f}% (3yr forecast)")
            except Exception:
                pass
            
            # 9. Candlestick patterns (5 points)
            try:
                patterns = self.advanced_analyzer.detect_candlestick_patterns(data)
                if patterns:
                    analysis["candlestick_patterns"] = patterns
                    bullish = [p for p in patterns if p.get('signal') == 'BULLISH']
                    if bullish:
                        score += 5
                        analysis["reasons"].append(f"📈 Bullish candlestick pattern detected: {bullish[0].get('pattern')}")
            except Exception:
                pass
            
            # 10. Bond analysis (if applicable)
            if etf_category in ["BONDS", "HIGH_YIELD", "TIPS"]:
                try:
                    bond_analysis = self.advanced_analyzer.analyze_bonds(ticker)
                    if bond_analysis and "yield_analysis" in bond_analysis:
                        analysis["bond_analysis"] = bond_analysis
                        current_yield = bond_analysis["yield_analysis"].get("current_yield", 0)
                        risk_adj_yield = bond_analysis["yield_analysis"].get("risk_adjusted_yield", 0)
                        
                        if risk_adj_yield > 2:
                            score += 15
                            analysis["reasons"].append(f"💰 Excellent bond yield: {current_yield:.2f}% (risk-adjusted: {risk_adj_yield:.2f})")
                        elif risk_adj_yield > 1:
                            score += 10
                            analysis["reasons"].append(f"💵 Good bond yield: {current_yield:.2f}%")
                except Exception:
                    pass
            
            analysis["score"] = max(0, min(100, score))
            analysis["annual_return"] = annual_return
            analysis["volatility"] = volatility
            analysis["momentum"] = recent_return
            
            if analysis["score"] >= 70:
                analysis["recommendation"] = "STRONG BUY"
            elif analysis["score"] >= 60:
                analysis["recommendation"] = "BUY"
            elif analysis["score"] >= 40:
                analysis["recommendation"] = "HOLD"
            elif analysis["score"] >= 30:
                analysis["recommendation"] = "AVOID"
            else:
                analysis["recommendation"] = "STRONG AVOID"
                
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            analysis["reasons"].append(f"Error in analysis: {str(e)}")
        
        return analysis
    
    def get_portfolio_diversification(self, portfolio: Dict) -> Dict:
        """Analyze current portfolio diversification."""
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return {"categories": {}, "gaps": list(self.ETF_CATEGORIES.keys())}
        
        categories = {}
        for holding in holdings:
            ticker = holding["ticker"]
            # Try to determine category
            for cat, etfs in self.ETF_CATEGORIES.items():
                if ticker in etfs:
                    categories[cat] = categories.get(cat, 0) + holding.get("current_value", 0)
        
        # Find gaps
        gaps = [cat for cat in self.ETF_CATEGORIES.keys() if cat not in categories]
        
        return {"categories": categories, "gaps": gaps}
    
    def recommend_etfs(self, deposit_amount_ils: float, portfolio: Dict) -> List[Dict]:
        """Recommend ETFs to buy based on deposit amount and current portfolio."""
        print(f"\nAnalyzing deposit of ₪{deposit_amount_ils:,.2f}...")
        
        # Convert to USD
        exchange_rate = self.get_exchange_rate()
        deposit_amount_usd = deposit_amount_ils / exchange_rate
        
        print(f"Deposit amount in USD: ${deposit_amount_usd:,.2f}")
        
        # Analyze current portfolio
        diversification = self.get_portfolio_diversification(portfolio)
        current_holdings = [h["ticker"] for h in portfolio.get("holdings", [])]
        
        # Analyze all potential ETFs
        all_etfs = []
        for category, etfs in self.ETF_CATEGORIES.items():
            for etf in etfs:
                if etf not in all_etfs:
                    all_etfs.append(etf)
        
        print(f"\nAnalyzing {len(all_etfs)} potential ETFs...")
        etf_analyses = []
        
        for etf in all_etfs:
            analysis = self.analyze_etf(etf)
            if analysis["current_price"] > 0:  # Only include if we got valid data
                # Bonus for diversification
                etf_category = None
                for cat, etfs in self.ETF_CATEGORIES.items():
                    if etf in etfs:
                        etf_category = cat
                        break
                
                if etf_category in diversification["gaps"]:
                    analysis["score"] += 15  # Bonus for filling gaps
                    analysis["reasons"].append("Fills diversification gap")
                
                # Small penalty if already holding
                if etf in current_holdings:
                    analysis["score"] -= 5
                    analysis["reasons"].append("Already in portfolio (consider increasing)")
                
                etf_analyses.append(analysis)
        
        # Optimize for mid-term yield (3-5 years) - ULTIMATE BROKER FEATURE
        print("\n🔬 Optimizing for mid-term yield (3-5 years) using statistical models...")
        optimized_etfs = self.advanced_analyzer.optimize_mid_term_yield(etf_analyses, target_years=3)
        
        # Combine score with mid-term yield optimization
        for etf in etf_analyses:
            ticker = etf["ticker"]
            # Find matching optimized analysis
            optimized = next((o for o in optimized_etfs if o.get("ticker") == ticker), None)
            if optimized and "mid_term_analysis" in optimized:
                opt_score = optimized["mid_term_analysis"].get("optimization_score", 0)
                # Boost score by 30% if mid-term yield is excellent
                if opt_score > 20:
                    etf["score"] = min(100, etf["score"] + 10)
                    etf["mid_term_boost"] = True
        
        # Sort by score (now includes mid-term optimization)
        etf_analyses.sort(key=lambda x: x["score"], reverse=True)
        
        # Select top recommendations
        recommendations = []
        remaining_amount = deposit_amount_usd
        
        # Strategy: Mix of top performers and diversification
        top_etfs = etf_analyses[:10]  # Top 10 by score
        
        # Allocate funds
        allocations = []
        
        # 1. Top 3 ETFs get 60% of funds
        top_3 = top_etfs[:3]
        for i, etf in enumerate(top_3):
            allocation_pct = 0.25 if i == 0 else 0.175  # 25% + 17.5% + 17.5% = 60%
            amount = deposit_amount_usd * allocation_pct
            shares = int(amount / etf["current_price"]) if etf["current_price"] > 0 else 0
            if shares > 0:
                allocations.append({
                    "ticker": etf["ticker"],
                    "name": etf.get("name", etf["ticker"]),
                    "allocation_amount": amount,
                    "allocation_percent": allocation_pct * 100,
                    "shares": shares,
                    "price": etf["current_price"],
                    "score": etf["score"],
                    "recommendation": etf["recommendation"],
                    "reasons": etf["reasons"],
                    "action": "NEW" if etf["ticker"] not in current_holdings else "INCREASE"
                })
                remaining_amount -= shares * etf["current_price"]
        
        # 2. Diversification picks get 30% of funds
        diversification_picks = [e for e in etf_analyses if e["ticker"] not in [a["ticker"] for a in allocations]][:3]
        for i, etf in enumerate(diversification_picks[:2]):
            if remaining_amount > 100:  # Only if enough left
                allocation_pct = 0.15
                amount = min(deposit_amount_usd * allocation_pct, remaining_amount)
                shares = int(amount / etf["current_price"]) if etf["current_price"] > 0 else 0
                if shares > 0:
                    allocations.append({
                        "ticker": etf["ticker"],
                        "name": etf.get("name", etf["ticker"]),
                        "allocation_amount": amount,
                        "allocation_percent": (amount / deposit_amount_usd) * 100,
                        "shares": shares,
                        "price": etf["current_price"],
                        "score": etf["score"],
                        "recommendation": etf["recommendation"],
                        "reasons": etf["reasons"],
                        "action": "NEW" if etf["ticker"] not in current_holdings else "INCREASE"
                    })
                    remaining_amount -= shares * etf["current_price"]
        
        # 3. Remaining goes to top pick
        if remaining_amount > 50 and allocations:
            top_pick = allocations[0]
            additional_shares = int(remaining_amount / top_pick["price"])
            if additional_shares > 0:
                allocations[0]["shares"] += additional_shares
                allocations[0]["allocation_amount"] += additional_shares * top_pick["price"]
                allocations[0]["allocation_percent"] = (allocations[0]["allocation_amount"] / deposit_amount_usd) * 100
        
        return allocations
    
    def update_portfolio_with_purchases(self, portfolio: Dict, recommendations: List[Dict], deposit_amount_usd: float):
        """Update portfolio.json with new purchases."""
        exchange_rate = self.get_exchange_rate()
        total_spent_usd = sum(rec['allocation_amount'] for rec in recommendations)
        
        # Ensure portfolio has required fields
        if "currency" not in portfolio:
            portfolio["currency"] = "USD"
        if "note" not in portfolio:
            portfolio["note"] = "All prices and values are in USD. Cash and portfolio values shown in ILS in the app are converted from USD."
        
        # Update cash (add deposit, subtract what was spent)
        current_cash_usd = portfolio.get("cash", 0)
        new_cash_usd = current_cash_usd + deposit_amount_usd - total_spent_usd
        portfolio["cash"] = round(new_cash_usd, 2)
        
        # Initialize holdings if not exists
        if "holdings" not in portfolio:
            portfolio["holdings"] = []
        
        # Update or add holdings
        for rec in recommendations:
            ticker = rec['ticker']
            shares_to_add = rec['shares']
            current_price = rec['price']
            
            # Check if holding already exists
            existing_holding = None
            for i, holding in enumerate(portfolio["holdings"]):
                if holding.get("ticker") == ticker:
                    existing_holding = i
                    break
            
            if existing_holding is not None:
                # Update existing holding
                portfolio["holdings"][existing_holding]["quantity"] += shares_to_add
                portfolio["holdings"][existing_holding]["last_price"] = current_price
                portfolio["holdings"][existing_holding]["current_value"] = (
                    portfolio["holdings"][existing_holding]["quantity"] * current_price
                )
            else:
                # Add new holding
                new_holding = {
                    "ticker": ticker,
                    "quantity": shares_to_add,
                    "last_price": current_price,
                    "current_value": shares_to_add * current_price
                }
                portfolio["holdings"].append(new_holding)
        
        # Save updated portfolio
        portfolio["last_updated"] = datetime.now().isoformat()
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Portfolio updated successfully!")
        print(f"   Total spent: ${total_spent_usd:,.2f} (₪{total_spent_usd * exchange_rate:,.2f})")
        print(f"   Remaining cash: ${new_cash_usd:,.2f} (₪{new_cash_usd * exchange_rate:,.2f})")
        print(f"   Portfolio saved to {self.portfolio_file}\n")
    
    def ask_confirmation(self) -> bool:
        """Ask user for confirmation."""
        while True:
            response = input("\nDid you execute these purchases? (yes/no): ").strip().lower()
            if response in ['yes', 'y', 'כן', 'י']:
                return True
            elif response in ['no', 'n', 'לא', 'ל']:
                return False
            else:
                print("Please enter 'yes' or 'no' (כן/לא)")
    
    def advise(self, deposit_amount_ils: float):
        """Main advisory function."""
        print("=" * 60)
        print("DEPOSIT ADVISORY SYSTEM")
        print("=" * 60)
        
        # Check and display market status
        market_status, market_message = self.analyzer.is_market_open()
        print(f"\n📊 Market Status: {market_message}")
        if market_status:
            print("   ⚡ Using REAL-TIME prices for recommendations")
        else:
            print("   📅 Using LAST CLOSE prices for recommendations")
        print()
        
        # Load portfolio
        portfolio = self.load_portfolio()
        
        if not os.path.exists(self.portfolio_file):
            print("Warning: Portfolio file not found. Creating new portfolio structure.")
            portfolio = {
                "currency": "USD",
                "cash": 0,
                "holdings": [],
                "last_updated": None,
                "total_value": 0
            }
        
        # Get recommendations
        recommendations = self.recommend_etfs(deposit_amount_ils, portfolio)
        
        # Print recommendations
        exchange_rate = self.get_exchange_rate()
        deposit_amount_usd = deposit_amount_ils / exchange_rate
        self.print_recommendations(deposit_amount_ils, recommendations, portfolio)
        
        # Ask for confirmation
        if recommendations:
            confirmed = self.ask_confirmation()
            
            if confirmed:
                # Update portfolio with purchases
                self.update_portfolio_with_purchases(portfolio, recommendations, deposit_amount_usd)
            else:
                print("\n❌ Portfolio not updated. No changes were made.\n")
        else:
            print("\nNo recommendations to execute.\n")
        
        return recommendations
    
    def print_recommendations(self, deposit_amount_ils: float, recommendations: List[Dict], portfolio: Dict):
        """Print recommendations in a formatted way."""
        print("\n" + "=" * 60)
        print("INVESTMENT RECOMMENDATIONS")
        print("=" * 60)
        
        exchange_rate = self.get_exchange_rate()
        deposit_amount_usd = deposit_amount_ils / exchange_rate
        
        # Get market status
        market_status, market_message = self.analyzer.is_market_open()
        
        print(f"\nDeposit Amount: ₪{deposit_amount_ils:,.2f} (${deposit_amount_usd:,.2f})")
        print(f"Exchange Rate: {exchange_rate} ILS/USD")
        print(f"📊 {market_message}")
        
        if not recommendations:
            print("\nNo recommendations available at this time.")
            return
        
        print("\n" + "-" * 60)
        print("RECOMMENDED PURCHASES (All prices and amounts in USD)")
        print("-" * 60)
        print("\n⚠️  NOTE: All purchases are executed in USD. Amounts shown in ILS are for reference only.")
        
        total_allocated = 0
        for i, rec in enumerate(recommendations, 1):
            allocation_ils = rec['allocation_amount'] * exchange_rate
            print(f"\n{i}. {rec['ticker']} - {rec['name']}")
            print(f"   Action: {rec['action']}")
            print(f"   Recommendation: {rec['recommendation']} (Score: {rec['score']:.1f}/100)")
            print(f"   BUY: {rec['shares']} shares")
            print(f"   Price per share: ${rec['price']:.2f}")
            print(f"   Total cost: ${rec['allocation_amount']:,.2f} (₪{allocation_ils:,.2f})")
            print(f"   Allocation: {rec['allocation_percent']:.1f}% of deposit")
            print(f"   Reasons:")
            for reason in rec['reasons'][:3]:  # Show top 3 reasons
                print(f"     • {reason}")
            
            total_allocated += rec['allocation_amount']
        
        total_allocated_ils = total_allocated * exchange_rate
        remaining_ils = (deposit_amount_usd - total_allocated) * exchange_rate
        
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        print(f"Total Allocated: ${total_allocated:,.2f} (₪{total_allocated_ils:,.2f})")
        print(f"Remaining: ${deposit_amount_usd - total_allocated:,.2f} (₪{remaining_ils:,.2f})")
        
        print("\n" + "=" * 60)
        print("Recommendations completed!")
        print("=" * 60)
        print("\n⚠️  After executing the purchases, you will be asked to confirm.")
        print("   If you confirm, the portfolio will be updated automatically.\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deposit_advisor.py <deposit_amount_ils>")
        sys.exit(1)
    
    try:
        deposit_amount = float(sys.argv[1])
        advisor = DepositAdvisor()
        advisor.advise(deposit_amount)
    except ValueError:
        print("Error: Please provide a valid deposit amount in ILS")
        sys.exit(1)

