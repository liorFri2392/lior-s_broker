#!/usr/bin/env python3
"""
Deposit Advisor - Recommends ETF purchases based on deposit amount and current portfolio.
"""

import json
import os
import sys
import logging
import subprocess
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from portfolio_analyzer import PortfolioAnalyzer
from advanced_analysis import AdvancedAnalyzer

# Try to import requests for GitHub API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Try to import PyNaCl for encryption (required for GitHub Secrets API)
try:
    from nacl import encoding, public
    HAS_PYNACL = True
except ImportError:
    HAS_PYNACL = False

logger = logging.getLogger(__name__)

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
        "AI_AND_ROBOTICS": ["BOTZ", "ROBO", "AIQ", "CHAT"],  # IRBO removed - possibly delisted
        "QUANTUM_COMPUTING": ["QUBT", "QTUM"],  # Quantum computing - emerging technology
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
        # WARNING: Crypto is HIGH RISK - Not recommended for young families with mortgages
        # Only consider if you have high risk tolerance and can afford to lose the investment
        # Bitcoin ETFs: BITO (Bitcoin futures), GBTC (Grayscale Bitcoin Trust)
        "CRYPTO": ["GBTC", "BITO", "ETHE", "IBIT", "FBTC"],  # Bitcoin ETFs (high volatility)
        "BLOCKCHAIN": ["BLOK", "LEGR", "KOIN"],  # Blockchain technology (less risky than direct crypto)
        
        # Investment Styles
        "DIVIDEND": ["VYM", "SCHD", "DVY"],
        "GROWTH": ["VUG", "IVW", "IWF"],
        "VALUE": ["VTV", "IVE", "IWD"],
        "MOMENTUM": ["MTUM", "QMOM", "DWAS"],
        
        # Leveraged ETFs (HIGH RISK - For experienced investors only)
        "LEVERAGED_2X": ["SSO", "QLD", "UWM", "EFO"],  # 2x leverage
        "LEVERAGED_3X": ["TQQQ", "SPXL", "UPRO", "TNA", "FAS", "CURE"],  # 3x leverage (very high risk)
        "LEVERAGED_INVERSE": ["SQQQ", "SPXS", "SPXU", "TZA", "FAZ"],  # Inverse leveraged (bearish)
        
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
            except Exception as e:
                logger.debug(f"Failed to analyze ETF {etf} in category {category}: {e}")
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
    
    def analyze_etf(self, ticker: str, verbose: bool = False) -> Dict:
        """
        Analyze an ETF for investment potential with industry trend analysis.
        
        Note: This function can analyze ANY ETF ticker, even if not in ETF_CATEGORIES.
        If you discover a new ETF, you can analyze it directly by ticker.
        The system will automatically fetch data and calculate scores.
        """
        if verbose:
            print(f"Analyzing ETF: {ticker}...")
        
        analysis = {
            "ticker": ticker,
            "score": 0,
            "current_price": 0,
            "recommendation": "NEUTRAL",
            "reasons": [],
            "industry_trend": {},
            "is_leveraged": False,
            "leverage_multiplier": 1.0,
            "risk_level": "NORMAL"
        }
        
        # Check if ETF is leveraged
        leveraged_2x = ["SSO", "QLD", "UWM", "EFO"]
        leveraged_3x = ["TQQQ", "SPXL", "UPRO", "TNA", "FAS", "CURE", "SOXL", "LABU", "TECL"]
        leveraged_inverse = ["SQQQ", "SPXS", "SPXU", "TZA", "FAZ", "SOXS", "LABD", "TECS"]
        
        if ticker.upper() in leveraged_2x:
            analysis["is_leveraged"] = True
            analysis["leverage_multiplier"] = 2.0
            analysis["risk_level"] = "HIGH"
        elif ticker.upper() in leveraged_3x:
            analysis["is_leveraged"] = True
            analysis["leverage_multiplier"] = 3.0
            analysis["risk_level"] = "VERY_HIGH"
        elif ticker.upper() in leveraged_inverse:
            analysis["is_leveraged"] = True
            analysis["leverage_multiplier"] = -3.0  # Negative for inverse
            analysis["risk_level"] = "VERY_HIGH"
        
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
                except Exception as e:
                    logger.debug(f"Failed to get real-time price for {ticker}: {e}")
                    pass  # Fall through to historical data
            
            # Fallback to historical data (last close) if real-time price not found
            if not price_found:
                try:
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        analysis["current_price"] = float(hist['Close'].iloc[-1])
                    else:
                        # Try longer period if 1d fails (for delisted ETFs)
                        hist = stock.history(period="5d")
                        if not hist.empty:
                            analysis["current_price"] = float(hist['Close'].iloc[-1])
                        else:
                            try:
                                analysis["current_price"] = float(stock.fast_info.get('lastPrice', 0))
                                if analysis["current_price"] == 0:
                                    logger.warning(f"{ticker}: Possibly delisted - no price data found")
                                    return analysis
                            except Exception:
                                logger.warning(f"{ticker}: Possibly delisted - no price data available")
                                return analysis  # Return early if no price data
                except Exception as e:
                    logger.warning(f"{ticker}: Failed to get price data - possibly delisted: {e}")
                    return analysis  # Return early if can't get price
            
            # Get historical data for analysis
            try:
                data = stock.history(period="1y")
                if data.empty:
                    # Try shorter period if 1y fails
                    data = stock.history(period="6mo")
                    if data.empty:
                        logger.warning(f"{ticker}: No historical data available - possibly delisted")
                        return analysis
            except Exception as e:
                logger.warning(f"{ticker}: Failed to get historical data - possibly delisted: {e}")
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
            
            # 2. Volatility (20 points) - Adjusted for leveraged ETFs
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Leveraged ETFs have inherently higher volatility - adjust expectations
            if analysis["is_leveraged"]:
                leverage = abs(analysis["leverage_multiplier"])
                # For leveraged ETFs, multiply volatility threshold by leverage
                volatility_threshold_low = 15 * leverage
                volatility_threshold_high = 30 * leverage
                
                if volatility < volatility_threshold_low:
                    score += 10
                    analysis["reasons"].append(f"Relatively low volatility for {leverage}x leveraged ETF: {volatility:.1f}%")
                elif volatility > volatility_threshold_high:
                    score -= 15  # More penalty for high volatility in leveraged ETFs
                    analysis["reasons"].append(f"⚠️ Very high volatility for {leverage}x leveraged ETF: {volatility:.1f}%")
                else:
                    analysis["reasons"].append(f"Expected high volatility for {leverage}x leveraged ETF: {volatility:.1f}%")
            else:
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
            
            # 8. Statistical forecast for mid-term yield (will be optimized later in batch)
            # Note: Full optimization happens in recommend_etfs() for efficiency
            try:
                forecast = self.advanced_analyzer.calculate_statistical_forecast(data, periods=252*3)
                if forecast and forecast.get("expected_return_polynomial") is not None:
                    expected_return = forecast.get("expected_return_polynomial", 0)
                    analysis["mid_term_forecast"] = {
                        "expected_3yr_return": expected_return,
                        "forecast_price": forecast.get("forecast_polynomial", 0)
                    }
                    
                    # Preliminary score boost (full optimization in batch later)
                    if expected_return > 15:
                        score += 15
                        analysis["reasons"].append(f"🎯 Excellent mid-term yield potential: {expected_return:.1f}% (3yr forecast)")
                    elif expected_return > 10:
                        score += 10
                        analysis["reasons"].append(f"📊 Strong mid-term yield potential: {expected_return:.1f}% (3yr forecast)")
            except Exception as e:
                logger.debug(f"Failed to calculate statistical forecast for {ticker}: {e}")
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
            except Exception as e:
                logger.debug(f"Failed to detect candlestick patterns for {ticker}: {e}")
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
                except Exception as e:
                    logger.debug(f"Failed to analyze bonds for {ticker}: {e}")
                    pass
            
            # Adjust score for leveraged ETFs - reduce score due to high risk
            if analysis["is_leveraged"]:
                leverage = abs(analysis["leverage_multiplier"])
                risk_penalty = 10 * leverage  # Penalty: 10 points for 2x, 20 for 3x
                score -= risk_penalty
                analysis["reasons"].append(f"⚠️ LEVERAGED ETF: {leverage}x leverage - High risk! Score reduced by {risk_penalty} points")
                if analysis["leverage_multiplier"] < 0:
                    analysis["reasons"].append(f"⚠️ INVERSE ETF: Moves opposite to market - Very risky!")
            
            analysis["score"] = max(0, min(100, score))
            analysis["annual_return"] = annual_return
            analysis["volatility"] = volatility
            analysis["momentum"] = recent_return
            
            # Adjust recommendation thresholds for leveraged ETFs
            if analysis["is_leveraged"]:
                # Higher thresholds for leveraged ETFs due to risk
                if analysis["score"] >= 80:
                    analysis["recommendation"] = "STRONG BUY (HIGH RISK)"
                elif analysis["score"] >= 70:
                    analysis["recommendation"] = "BUY (HIGH RISK)"
                elif analysis["score"] >= 50:
                    analysis["recommendation"] = "HOLD (HIGH RISK)"
                elif analysis["score"] >= 40:
                    analysis["recommendation"] = "AVOID (HIGH RISK)"
                else:
                    analysis["recommendation"] = "STRONG AVOID (HIGH RISK)"
            else:
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
            logger.error(f"Error analyzing {ticker}: {e}")
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
        """
        Recommend ETFs based on 80/20 Balanced Growth Strategy:
        - 80% Stocks (50% Core + 30% Satellite)
        - 20% Bonds (Protection)
        Optimized for family with mortgage - balanced risk/return.
        """
        print(f"\nAnalyzing deposit of ₪{deposit_amount_ils:,.2f}...")
        print("📊 Strategy: 80/20 Balanced Growth (80% Stocks, 20% Bonds)")
        print("   Optimized for family stability with growth potential\n")
        
        # Convert to USD
        exchange_rate = self.get_exchange_rate()
        deposit_amount_usd = deposit_amount_ils / exchange_rate
        
        print(f"Deposit amount in USD: ${deposit_amount_usd:,.2f}")
        
        # Analyze current portfolio to understand current allocation
        current_holdings = [h["ticker"] for h in portfolio.get("holdings", [])]
        portfolio_value = sum(h.get("current_value", 0) for h in portfolio.get("holdings", []))
        portfolio_value += portfolio.get("cash", 0)
        
        # Calculate current portfolio allocation
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        
        current_bonds_value = sum(h.get("current_value", 0) for h in portfolio.get("holdings", []) 
                                 if h.get("ticker", "").upper() in [b.upper() for b in bond_etfs])
        current_stocks_value = portfolio_value - current_bonds_value
        
        current_bonds_percent = (current_bonds_value / portfolio_value * 100) if portfolio_value > 0 else 0
        current_stocks_percent = (current_stocks_value / portfolio_value * 100) if portfolio_value > 0 else 100
        
        # Calculate what the portfolio will look like after deposit
        total_after_deposit = portfolio_value + deposit_amount_usd
        
        # Calculate target allocation for entire portfolio (80/20)
        target_stocks_total = total_after_deposit * 0.80
        target_bonds_total = total_after_deposit * 0.20
        
        # Calculate how much we need to add to reach target
        needed_stocks = max(0, target_stocks_total - current_stocks_value)
        needed_bonds = max(0, target_bonds_total - current_bonds_value)
        
        # Adjust deposit allocation based on what's needed
        # If portfolio is already heavy on stocks, allocate more of deposit to bonds
        if current_stocks_percent > 80:
            # Portfolio has too many stocks - prioritize bonds in deposit
            bonds_target = min(deposit_amount_usd, needed_bonds)
            stocks_target = deposit_amount_usd - bonds_target
        elif current_bonds_percent > 20:
            # Portfolio has too many bonds - prioritize stocks in deposit
            stocks_target = min(deposit_amount_usd, needed_stocks)
            bonds_target = deposit_amount_usd - stocks_target
        else:
            # Portfolio is relatively balanced - use standard 80/20 split
            stocks_target = deposit_amount_usd * 0.80
            bonds_target = deposit_amount_usd * 0.20
        
        # Ensure we don't exceed what's needed
        stocks_target = min(stocks_target, needed_stocks) if needed_stocks > 0 else stocks_target
        bonds_target = min(bonds_target, needed_bonds) if needed_bonds > 0 else bonds_target
        
        # If one target is zero but we have deposit, allocate to the other
        if stocks_target == 0 and bonds_target == 0:
            # Both targets met - use standard split
            stocks_target = deposit_amount_usd * 0.80
            bonds_target = deposit_amount_usd * 0.20
        elif stocks_target == 0:
            bonds_target = deposit_amount_usd
        elif bonds_target == 0:
            stocks_target = deposit_amount_usd
        
        # Ensure we use the full deposit amount
        total_allocated = stocks_target + bonds_target
        if total_allocated < deposit_amount_usd * 0.99:  # Allow small rounding differences
            # Distribute remaining amount proportionally
            remaining = deposit_amount_usd - total_allocated
            if stocks_target > 0 and bonds_target > 0:
                stocks_target += remaining * (stocks_target / total_allocated)
                bonds_target += remaining * (bonds_target / total_allocated)
            elif stocks_target > 0:
                stocks_target += remaining
            else:
                bonds_target += remaining
        
        print(f"📊 Current Portfolio: {current_stocks_percent:.1f}% Stocks, {current_bonds_percent:.1f}% Bonds")
        print(f"📊 After Deposit Target: 80% Stocks, 20% Bonds")
        if current_stocks_percent > 80 or current_bonds_percent > 20:
            print(f"⚠️  Portfolio is unbalanced - adjusting deposit allocation to rebalance")
            print(f"   Deposit allocation: ${stocks_target:,.2f} Stocks ({(stocks_target/deposit_amount_usd*100):.1f}%), ${bonds_target:,.2f} Bonds ({(bonds_target/deposit_amount_usd*100):.1f}%)")
        else:
            print(f"   Deposit allocation: ${stocks_target:,.2f} Stocks ({(stocks_target/deposit_amount_usd*100):.1f}%), ${bonds_target:,.2f} Bonds ({(bonds_target/deposit_amount_usd*100):.1f}%)")
        print()
        
        # Define Core ETFs (stable, broad market)
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]  # US Large Cap + International
        # Define Satellite ETFs (growth, trends - but not too risky)
        # Expanded to include more categories for better coverage
        satellite_etfs = [
            # Core Satellite (essential diversification)
            "IWM", "VB", "XLK", "VGT", "VWO", "EEM", "XLV", "VHT",  # Small Cap, Tech, Emerging, Healthcare
            # High-growth trends
            "BOTZ", "ROBO", "QUBT", "QTUM",  # AI/Robotics, Quantum Computing
            "SOXX", "SMH", "WCLD", "SKYY",  # Semiconductors, Cloud
            "HACK", "CIBR", "ICLN", "QCLN",  # Cybersecurity, Clean Energy
            "DRIV", "IDRV",  # Electric Vehicles
            # Defensive growth
            "VNQ", "SCHH", "IFRA", "PAVE",  # Real Estate, Infrastructure
            # Investment styles
            "VYM", "SCHD", "VUG", "IVW", "VTV", "IVE",  # Dividend, Growth, Value
            # Sector diversification
            "XLF", "VFH", "XLE", "VDE", "XLY", "VCR"  # Financial, Energy, Consumer
        ]
        # Define Bonds (protection)
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]  # US Bonds + Inflation Protection
        
        # Exclude high-risk categories
        excluded_categories = ["LEVERAGED_2X", "LEVERAGED_3X", "LEVERAGED_INVERSE", "CRYPTO"]
        excluded_tickers = []
        for cat in excluded_categories:
            if cat in self.ETF_CATEGORIES:
                excluded_tickers.extend(self.ETF_CATEGORIES[cat])
        
        print(f"🔍 Analyzing Core, Satellite, and Bond ETFs...")
        print("   (Excluding leveraged ETFs and crypto for balanced risk)\n")
        
        # Analyze Core ETFs
        core_analyses = []
        for etf in core_etfs:
            if etf not in excluded_tickers:
                analysis = self.analyze_etf(etf, verbose=False)
                if analysis["current_price"] > 0:
                    # Boost score for Core ETFs (they're essential)
                    analysis["score"] += 20
                    analysis["reasons"].append("Core holding - essential for portfolio stability")
                    if etf in current_holdings:
                        analysis["reasons"].append("Already in portfolio - consider increasing")
                    core_analyses.append(analysis)
        
        # Analyze Satellite ETFs
        satellite_analyses = []
        for etf in satellite_etfs:
            if etf not in excluded_tickers:
                analysis = self.analyze_etf(etf, verbose=False)
                if analysis["current_price"] > 0:
                    # Small boost for diversification
                    if etf not in current_holdings:
                        analysis["score"] += 10
                        analysis["reasons"].append("Satellite holding - adds growth potential")
                    satellite_analyses.append(analysis)
        
        # Analyze Bond ETFs
        bond_analyses = []
        for etf in bond_etfs:
            if etf not in excluded_tickers:
                analysis = self.analyze_etf(etf, verbose=False)
                if analysis["current_price"] > 0:
                    # Boost score for Bonds (essential for protection)
                    analysis["score"] += 25
                    analysis["reasons"].append("Bond holding - essential for portfolio protection")
                    if etf in current_holdings:
                        analysis["reasons"].append("Already in portfolio - consider increasing")
                    bond_analyses.append(analysis)
        
        # Optimize for mid-term yield
        print("🔬 Optimizing for mid-term yield (3-5 years) using statistical models...")
        all_analyses = core_analyses + satellite_analyses + bond_analyses
        optimized_etfs = self.advanced_analyzer.optimize_mid_term_yield(all_analyses, target_years=3)
        
        # Apply optimization boost
        for etf in all_analyses:
            ticker = etf["ticker"]
            optimized = next((o for o in optimized_etfs if o.get("ticker") == ticker), None)
            if optimized and "mid_term_analysis" in optimized:
                opt_score = optimized["mid_term_analysis"].get("optimization_score", 0)
                if opt_score > 20:
                    etf["score"] = min(100, etf["score"] + 10)
                    etf["mid_term_boost"] = True
        
        # Sort by score
        core_analyses.sort(key=lambda x: x["score"], reverse=True)
        satellite_analyses.sort(key=lambda x: x["score"], reverse=True)
        bond_analyses.sort(key=lambda x: x["score"], reverse=True)
        
        print("✅ Analysis complete! Generating 80/20 balanced recommendations...\n")
        
        # Build allocations following 80/20 strategy
        allocations = []
        remaining_stocks = stocks_target
        remaining_bonds = bonds_target
        
        # 1. Core Stocks (50% of total = 62.5% of stocks allocation)
        core_target = stocks_target * 0.625  # 50% of total portfolio
        core_allocated = 0
        for etf in core_analyses[:2]:  # Top 2 Core ETFs
            if core_allocated < core_target:
                amount = min(core_target * 0.5, core_target - core_allocated)  # Split between top 2
                shares = int(amount / etf["current_price"]) if etf["current_price"] > 0 else 0
                if shares > 0:
                    actual_amount = shares * etf["current_price"]
                    allocations.append({
                        "ticker": etf["ticker"],
                        "name": etf.get("name", etf["ticker"]),
                        "allocation_amount": actual_amount,
                        "allocation_percent": (actual_amount / deposit_amount_usd) * 100,
                        "shares": shares,
                        "price": etf["current_price"],
                        "score": etf["score"],
                        "recommendation": etf["recommendation"],
                        "reasons": etf["reasons"],
                        "action": "NEW" if etf["ticker"] not in current_holdings else "INCREASE",
                        "category": "CORE"
                    })
                    core_allocated += actual_amount
                    remaining_stocks -= actual_amount
        
        # 2. Satellite Stocks (30% of total = 37.5% of stocks allocation)
        satellite_target = stocks_target * 0.375  # 30% of total portfolio
        satellite_allocated = 0
        for etf in satellite_analyses[:3]:  # Top 3 Satellite ETFs
            if satellite_allocated < satellite_target:
                amount = min(satellite_target / 3, satellite_target - satellite_allocated)
                shares = int(amount / etf["current_price"]) if etf["current_price"] > 0 else 0
                if shares > 0:
                    actual_amount = shares * etf["current_price"]
                    allocations.append({
                        "ticker": etf["ticker"],
                        "name": etf.get("name", etf["ticker"]),
                        "allocation_amount": actual_amount,
                        "allocation_percent": (actual_amount / deposit_amount_usd) * 100,
                        "shares": shares,
                        "price": etf["current_price"],
                        "score": etf["score"],
                        "recommendation": etf["recommendation"],
                        "reasons": etf["reasons"],
                        "action": "NEW" if etf["ticker"] not in current_holdings else "INCREASE",
                        "category": "SATELLITE"
                    })
                    satellite_allocated += actual_amount
                    remaining_stocks -= actual_amount
        
        # 3. Bonds (25% of total)
        bond_allocated = 0
        for etf in bond_analyses[:2]:  # Top 2 Bond ETFs
            if bond_allocated < bonds_target:
                amount = min(bonds_target * 0.6, bonds_target - bond_allocated)  # 60% to first, 40% to second
                if bond_allocated > 0:
                    amount = bonds_target - bond_allocated  # Remaining to second
                shares = int(amount / etf["current_price"]) if etf["current_price"] > 0 else 0
                if shares > 0:
                    actual_amount = shares * etf["current_price"]
                    allocations.append({
                        "ticker": etf["ticker"],
                        "name": etf.get("name", etf["ticker"]),
                        "allocation_amount": actual_amount,
                        "allocation_percent": (actual_amount / deposit_amount_usd) * 100,
                        "shares": shares,
                        "price": etf["current_price"],
                        "score": etf["score"],
                        "recommendation": etf["recommendation"],
                        "reasons": etf["reasons"],
                        "action": "NEW" if etf["ticker"] not in current_holdings else "INCREASE",
                        "category": "BONDS"
                    })
                    bond_allocated += actual_amount
                    remaining_bonds -= actual_amount
        
        # 4. Allocate any remaining to Core (safest option)
        remaining_total = remaining_stocks + remaining_bonds
        if remaining_total > 50 and core_analyses:
            top_core = core_analyses[0]
            shares = int(remaining_total / top_core["current_price"]) if top_core["current_price"] > 0 else 0
            if shares > 0:
                actual_amount = shares * top_core["current_price"]
                # Check if already in allocations
                existing = next((a for a in allocations if a["ticker"] == top_core["ticker"]), None)
                if existing:
                    existing["shares"] += shares
                    existing["allocation_amount"] += actual_amount
                    existing["allocation_percent"] = (existing["allocation_amount"] / deposit_amount_usd) * 100
                else:
                    allocations.append({
                        "ticker": top_core["ticker"],
                        "name": top_core.get("name", top_core["ticker"]),
                        "allocation_amount": actual_amount,
                        "allocation_percent": (actual_amount / deposit_amount_usd) * 100,
                        "shares": shares,
                        "price": top_core["current_price"],
                        "score": top_core["score"],
                        "recommendation": top_core["recommendation"],
                        "reasons": top_core["reasons"],
                        "action": "NEW" if top_core["ticker"] not in current_holdings else "INCREASE",
                        "category": "CORE"
                    })
        
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
        
        # Try to update GitHub secret automatically
        self._try_update_github_secret(portfolio)
        
        print(f"\n✅ Portfolio updated successfully!")
        print(f"   Total spent: ${total_spent_usd:,.2f} (₪{total_spent_usd * exchange_rate:,.2f})")
        print(f"   Remaining cash: ${new_cash_usd:,.2f} (₪{new_cash_usd * exchange_rate:,.2f})")
        print(f"   Portfolio saved to {self.portfolio_file}\n")
    
    def _try_update_github_secret(self, portfolio: Dict):
        """Try to update GitHub secret automatically (silently, no errors if fails)."""
        # Method 1: Try GitHub CLI first (easiest)
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                # GitHub CLI is available, try to update secret
                portfolio_json_str = json.dumps(portfolio, ensure_ascii=False, indent=2)
                process = subprocess.Popen(
                    ["gh", "secret", "set", "PORTFOLIO_JSON", "--repo", "liorFri2392/lior-s_broker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=portfolio_json_str, timeout=10)
                if process.returncode == 0:
                    logger.info("✅ GitHub secret updated automatically (via GitHub CLI)")
                    print("   ✅ GitHub secret updated automatically!")
                    return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # GitHub CLI not available, try API method
        
        # Method 2: Try GitHub API with token (if requests and PyNaCl available)
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token and HAS_REQUESTS and HAS_PYNACL:
            try:
                if self._update_secret_via_api(github_token, portfolio):
                    print("   ✅ GitHub secret updated automatically!")
                    return True
            except Exception as e:
                logger.debug(f"GitHub API update failed: {e}")
        
        # If both methods fail, silently return False
        return False
    
    def _update_secret_via_api(self, token: str, portfolio: Dict) -> bool:
        """Update GitHub secret using GitHub API with token."""
        repo_owner = "liorFri2392"
        repo_name = "lior-s_broker"
        secret_name = "PORTFOLIO_JSON"
        
        # Get public key
        public_key_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/secrets/public-key"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            response = requests.get(public_key_url, headers=headers, timeout=10)
            response.raise_for_status()
            public_key_data = response.json()
            public_key = public_key_data["key"]
            key_id = public_key_data["key_id"]
            
            # Encrypt the secret using public key
            portfolio_json_str = json.dumps(portfolio, ensure_ascii=False, indent=2)
            public_key_obj = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
            sealed_box = public.SealedBox(public_key_obj)
            encrypted = sealed_box.encrypt(portfolio_json_str.encode("utf-8"))
            encrypted_value = base64.b64encode(encrypted).decode("utf-8")
            
            # Update the secret
            secret_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/secrets/{secret_name}"
            payload = {
                "encrypted_value": encrypted_value,
                "key_id": key_id
            }
            
            response = requests.put(secret_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("✅ GitHub secret updated automatically (via GitHub API)")
            return True
            
        except Exception as e:
            logger.debug(f"GitHub API secret update failed: {e}")
            return False
    
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
        print("\n📊 Strategy: 80/20 Balanced Growth")
        print("   • 80% Stocks (50% Core + 30% Satellite)")
        print("   • 20% Bonds (Protection)")
        print("   • Optimized for family stability with growth potential")
        print("\n⚠️  NOTE: All purchases are executed in USD. Amounts shown in ILS are for reference only.")
        
        # Calculate and show allocation breakdown
        core_total = sum(r['allocation_amount'] for r in recommendations if r.get('category') == 'CORE')
        satellite_total = sum(r['allocation_amount'] for r in recommendations if r.get('category') == 'SATELLITE')
        bonds_total = sum(r['allocation_amount'] for r in recommendations if r.get('category') == 'BONDS')
        
        if core_total + satellite_total + bonds_total > 0:
            print(f"\n📈 Allocation Breakdown:")
            if core_total > 0:
                print(f"   • Core Stocks: ${core_total:,.2f} ({core_total/deposit_amount_usd*100:.1f}%)")
            if satellite_total > 0:
                print(f"   • Satellite Stocks: ${satellite_total:,.2f} ({satellite_total/deposit_amount_usd*100:.1f}%)")
            if bonds_total > 0:
                print(f"   • Bonds: ${bonds_total:,.2f} ({bonds_total/deposit_amount_usd*100:.1f}%)")
        
        # Check for leveraged ETFs in recommendations
        leveraged_count = sum(1 for rec in recommendations if any(lev in rec.get('ticker', '').upper() 
            for lev in ['TQQQ', 'SPXL', 'UPRO', 'TNA', 'FAS', 'CURE', 'SOXL', 'LABU', 'TECL', 
                       'SQQQ', 'SPXS', 'SPXU', 'TZA', 'FAZ', 'SOXS', 'LABD', 'TECS', 'SSO', 'QLD', 'UWM', 'EFO']))
        
        if leveraged_count > 0:
            print("\n" + "=" * 60)
            print("🚨 WARNING: LEVERAGED ETFs DETECTED 🚨")
            print("=" * 60)
            print("⚠️  Leveraged ETFs are EXTREMELY RISKY:")
            print("   • Losses can be 2x-3x the underlying index")
            print("   • Very high volatility - can lose 50%+ in days")
            print("   • Not suitable for beginners or long-term holding")
            print("   • Only for experienced investors who understand the risks")
            print("   • Consider limiting leveraged ETFs to <5% of portfolio")
            print("=" * 60 + "\n")
        
        total_allocated = 0
        for i, rec in enumerate(recommendations, 1):
            allocation_ils = rec['allocation_amount'] * exchange_rate
            ticker_upper = rec['ticker'].upper()
            is_leveraged = any(lev in ticker_upper for lev in ['TQQQ', 'SPXL', 'UPRO', 'TNA', 'FAS', 'CURE', 
                                                               'SOXL', 'LABU', 'TECL', 'SQQQ', 'SPXS', 'SPXU', 
                                                               'TZA', 'FAZ', 'SOXS', 'LABD', 'TECS', 'SSO', 'QLD', 'UWM', 'EFO'])
            
            print(f"\n{i}. {rec['ticker']} - {rec['name']}")
            category = rec.get('category', '')
            if category == 'CORE':
                print(f"   📊 Category: CORE (Essential for portfolio stability)")
            elif category == 'BONDS':
                print(f"   🛡️  Category: BONDS (Portfolio protection)")
            elif category == 'SATELLITE':
                print(f"   🚀 Category: SATELLITE (Growth opportunity)")
            if is_leveraged:
                print(f"   🚨 LEVERAGED ETF - EXTREME RISK 🚨")
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
        print("   If you confirm, the portfolio will be updated automatically.")
        print("\n💡 REMINDER: Update GitHub Secret")
        print("   After updating your portfolio, update the GitHub secret so critical alerts use the latest data:")
        print("   Run: make update-secret")
        print("   Or go to: https://github.com/liorFri2392/lior-s_broker/settings/secrets/actions\n")

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

