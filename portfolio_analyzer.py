#!/usr/bin/env python3
"""
Portfolio Analyzer - Advanced Investment Portfolio Analysis System
Analyzes portfolio holdings, provides recommendations, and suggests rebalancing.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import requests
from newsapi import NewsApiClient
from advanced_analysis import AdvancedAnalyzer
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """Advanced portfolio analysis system with news, trends, and statistical analysis."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.exchange_rate_usd_ils = 3.7  # Default, will be updated
        self.exchange_rate_cache_time = None  # Cache for exchange rate
        self.price_cache = {}  # Cache for prices: {ticker: (price, timestamp)}
        self.market_data_cache = {}  # Cache for market data: {ticker: (data, timestamp)}
        self.news_cache = {}  # Cache for news: {ticker: (sentiment, timestamp)}
        self.cache_timeout = timedelta(minutes=5)  # Cache timeout
        self.advanced_analyzer = AdvancedAnalyzer()  # Advanced analysis module
    
    def get_exchange_rate(self) -> float:
        """Get current USD/ILS exchange rate with caching."""
        now = datetime.now()
        
        # Check cache first
        if (self.exchange_rate_cache_time and 
            now - self.exchange_rate_cache_time < self.cache_timeout):
            return self.exchange_rate_usd_ils
        
        # Fetch new rate
        try:
            usd_ils = yf.Ticker("USDILS=X")
            hist = usd_ils.history(period="1d")
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1])
                self.exchange_rate_usd_ils = rate
                self.exchange_rate_cache_time = now
                return rate
            return self.exchange_rate_usd_ils
        except Exception:
            return self.exchange_rate_usd_ils
        
    def load_portfolio(self) -> Dict:
        """Load portfolio from JSON file."""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "cash": 0,
            "holdings": [],
            "last_updated": None,
            "total_value": 0
        }
    
    def save_portfolio(self, portfolio: Dict):
        """Save portfolio to JSON file locally."""
        portfolio["last_updated"] = datetime.now().isoformat()
        portfolio_path = os.path.abspath(self.portfolio_file)
        with open(portfolio_path, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)
    
    def is_market_open(self) -> Tuple[bool, str]:
        """Check if US stock market (NYSE/NASDAQ) is currently open."""
        try:
            # Get current time in Eastern Time (ET)
            now_utc = datetime.now(timezone.utc)
            # Convert to ET (UTC-5 or UTC-4 depending on DST)
            # Simple approximation: ET is UTC-5
            et_offset = timedelta(hours=5)
            now_et = now_utc - et_offset
            
            # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = now_et.weekday() < 5
            
            if is_weekday and market_open <= now_et <= market_close:
                return True, "Market is OPEN - Prices are real-time"
            else:
                if not is_weekday:
                    return False, "Market is CLOSED - Weekend (Prices from last close)"
                else:
                    return False, "Market is CLOSED - After hours (Prices from last close)"
        except Exception:
            return None, "Unable to determine market status"
    
    def get_current_prices(self, tickers: List[str]) -> Tuple[Dict[str, float], Optional[bool], str]:
        """Get current prices for tickers with caching and parallel fetching.
        Returns: (prices_dict, market_status, market_message)
        - prices_dict: Dictionary of ticker -> price
        - market_status: True if market is open, False if closed, None if unknown
        - market_message: Human-readable market status message"""
        prices = {}
        uncached_tickers = []
        market_status, market_message = self.is_market_open()
        
        # Check cache first
        now = datetime.now()
        for ticker in tickers:
            if ticker in self.price_cache:
                cached_data, cached_time = self.price_cache[ticker]
                # If market is open, use shorter cache (1 minute), otherwise 5 minutes
                cache_timeout = timedelta(minutes=1) if market_status else self.cache_timeout
                if now - cached_time < cache_timeout:
                    prices[ticker] = cached_data
                else:
                    uncached_tickers.append(ticker)
            else:
                uncached_tickers.append(ticker)
        
        # Fetch uncached prices in parallel
        if uncached_tickers:
            def fetch_price(ticker):
                try:
                    stock = yf.Ticker(ticker)
                    
                    # Try to get real-time price first (if market is open)
                    if market_status:
                        try:
                            # Use fast_info for real-time data
                            price = float(stock.fast_info.get('lastPrice', 0))
                            if price > 0:
                                self.price_cache[ticker] = (price, now)
                                return ticker, price, True  # True = real-time
                        except Exception:
                            pass
                    
                    # Fallback to historical data (last close)
                    info = stock.history(period="1d")
                    if not info.empty:
                        price = float(info['Close'].iloc[-1])
                        self.price_cache[ticker] = (price, now)
                        return ticker, price, False  # False = last close
                    else:
                        try:
                            price = float(stock.fast_info.get('lastPrice', 0))
                            if price > 0:
                                self.price_cache[ticker] = (price, now)
                                return ticker, price, False
                        except Exception:
                            return ticker, None, False
                except Exception as e:
                    print(f"Warning: Could not fetch price for {ticker}: {e}")
                    return ticker, None, False
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(fetch_price, ticker) for ticker in uncached_tickers]
                for future in as_completed(futures):
                    ticker, price, is_realtime = future.result()
                    if price is not None:
                        prices[ticker] = price
        
        return prices, market_status, market_message
    
    def get_market_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical market data for analysis with caching."""
        now = datetime.now()
        cache_key = f"{ticker}_{period}"
        
        # Check cache first
        if cache_key in self.market_data_cache:
            cached_data, cached_time = self.market_data_cache[cache_key]
            if now - cached_time < self.cache_timeout:
                return cached_data
        
        # Fetch new data
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                self.market_data_cache[cache_key] = (data, now)
            return data
        except Exception:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        if data.empty or len(data) < 20:
            return {}
        
        indicators = {}
        closes = data['Close']
        
        # Moving averages
        indicators['sma_20'] = closes.tail(20).mean()
        indicators['sma_50'] = closes.tail(min(50, len(closes))).mean()
        indicators['sma_200'] = closes.tail(min(200, len(closes))).mean()
        
        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
        
        # Volatility
        returns = closes.pct_change()
        indicators['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Momentum
        indicators['momentum'] = (closes.iloc[-1] / closes.iloc[-20] - 1) * 100 if len(closes) >= 20 else 0
        
        # Sharpe ratio (simplified)
        if returns.std() > 0:
            indicators['sharpe'] = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            indicators['sharpe'] = 0
        
        # Beta (market correlation) - simplified
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period=period)
            if not spy_data.empty and len(spy_data) == len(data):
                spy_returns = spy_data['Close'].pct_change().dropna()
                asset_returns = returns.dropna()
                if len(spy_returns) == len(asset_returns) and len(asset_returns) > 0:
                    covariance = np.cov(asset_returns, spy_returns)[0][1]
                    spy_variance = np.var(spy_returns)
                    if spy_variance > 0:
                        indicators['beta'] = covariance / spy_variance
        except Exception:
            indicators['beta'] = 1.0  # Default beta
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        indicators['max_drawdown'] = drawdown.min() * 100
        
        # Trend analysis
        if len(closes) >= 50:
            short_ma = closes.tail(20).mean()
            long_ma = closes.tail(50).mean()
            if short_ma > long_ma:
                indicators['trend'] = 'BULLISH'
            elif short_ma < long_ma:
                indicators['trend'] = 'BEARISH'
            else:
                indicators['trend'] = 'NEUTRAL'
        else:
            indicators['trend'] = 'NEUTRAL'
        
        # Candlestick patterns (advanced)
        try:
            candlestick_patterns = self.advanced_analyzer.detect_candlestick_patterns(data)
            if candlestick_patterns:
                indicators['candlestick_patterns'] = candlestick_patterns
                # Add signal strength based on patterns
                bullish_patterns = [p for p in candlestick_patterns if p.get('signal') == 'BULLISH']
                bearish_patterns = [p for p in candlestick_patterns if p.get('signal') == 'BEARISH']
                if bullish_patterns:
                    indicators['pattern_signal'] = 'BULLISH'
                elif bearish_patterns:
                    indicators['pattern_signal'] = 'BEARISH'
        except Exception:
            pass
        
        # Statistical forecast (mid-term yield optimization)
        try:
            forecast = self.advanced_analyzer.calculate_statistical_forecast(data, periods=252*3)  # 3 years
            if forecast:
                indicators['forecast'] = forecast
        except Exception:
            pass
        
        return indicators
    
    def get_news_sentiment(self, ticker: str) -> Dict:
        """Get news sentiment for a ticker with advanced analysis and caching."""
        now = datetime.now()
        
        # Check cache first
        if ticker in self.news_cache:
            cached_sentiment, cached_time = self.news_cache[ticker]
            if now - cached_time < self.cache_timeout:
                return cached_sentiment
        
        sentiment = {
            "score": 50,  # Neutral base
            "articles_count": 0,
            "recent_news": [],
            "sentiment_analysis": "NEUTRAL"
        }
        
        try:
            # Try to get news from yfinance (free, no API key needed)
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news:
                sentiment["articles_count"] = len(news)
                sentiment["recent_news"] = [
                    {
                        "title": article.get('title', ''),
                        "source": article.get('publisher', 'Unknown'),
                        "published": datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat() if article.get('providerPublishTime') else 'Unknown'
                    }
                    for article in news[:5]
                ]
                
                # Simple sentiment analysis based on titles
                positive_words = ['gain', 'rise', 'up', 'growth', 'profit', 'beat', 'strong', 'bullish', 'surge']
                negative_words = ['fall', 'drop', 'down', 'loss', 'miss', 'weak', 'bearish', 'decline', 'crash']
                
                positive_count = sum(1 for article in news if any(word in article.get('title', '').lower() for word in positive_words))
                negative_count = sum(1 for article in news if any(word in article.get('title', '').lower() for word in negative_words))
                
                if positive_count > negative_count:
                    sentiment["score"] = 50 + min(positive_count * 5, 30)
                    sentiment["sentiment_analysis"] = "POSITIVE"
                elif negative_count > positive_count:
                    sentiment["score"] = 50 - min(negative_count * 5, 30)
                    sentiment["sentiment_analysis"] = "NEGATIVE"
                else:
                    sentiment["sentiment_analysis"] = "NEUTRAL"
            
            # Also try NewsAPI if key is available
            if self.news_api_key:
                try:
                    newsapi = NewsApiClient(api_key=self.news_api_key)
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    company_name = info.get('longName', ticker)
                    
                    articles = newsapi.get_everything(
                        q=f"{company_name} OR {ticker}",
                        language='en',
                        sort_by='relevancy',
                        page_size=10
                    )
                    
                    if articles['status'] == 'ok' and articles['articles']:
                        sentiment["articles_count"] += len(articles['articles'])
                        sentiment["recent_news"].extend([
                            {
                                "title": article['title'],
                                "source": article['source']['name'],
                                "published": article['publishedAt']
                            }
                            for article in articles['articles'][:3]
                        ])
                except Exception:
                    pass
        except Exception:
            pass  # Silent fail, use default sentiment
        
        # Cache the result
        self.news_cache[ticker] = (sentiment, now)
        return sentiment
    
    def analyze_holding(self, ticker: str, quantity: float, current_price: float) -> Dict:
        """Comprehensive analysis of a single holding."""
        print(f"Analyzing {ticker}...")
        
        analysis = {
            "ticker": ticker,
            "quantity": quantity,
            "current_price": current_price,
            "current_value": quantity * current_price,
            "technical_indicators": {},
            "news_sentiment": {},
            "recommendation": "HOLD",
            "recommendation_score": 0
        }
        
        # Get market data
        data = self.get_market_data(ticker)
        if not data.empty:
            analysis["technical_indicators"] = self.calculate_technical_indicators(data)
        
        # Get news sentiment
        analysis["news_sentiment"] = self.get_news_sentiment(ticker)
        
        # Calculate recommendation score
        score = 50  # Neutral base
        
        if analysis["technical_indicators"]:
            ti = analysis["technical_indicators"]
            
            # RSI analysis
            if ti.get('rsi', 50) < 30:
                score += 15  # Oversold, potential buy
            elif ti.get('rsi', 50) > 70:
                score -= 15  # Overbought, potential sell
            
            # Momentum analysis
            if ti.get('momentum', 0) > 5:
                score += 10
            elif ti.get('momentum', 0) < -5:
                score -= 10
            
            # Sharpe ratio
            if ti.get('sharpe', 0) > 1:
                score += 10
            elif ti.get('sharpe', 0) < 0:
                score -= 10
        
        # News sentiment (enhanced)
        news_sentiment = analysis["news_sentiment"]
        if news_sentiment.get("sentiment_analysis") == "POSITIVE":
            score += 15
        elif news_sentiment.get("sentiment_analysis") == "NEGATIVE":
            score -= 15
        
        if news_sentiment["score"] > 60:
            score += 5
        elif news_sentiment["score"] < 40:
            score -= 5
        
        # Industry trend analysis (if available from deposit_advisor)
        try:
            from deposit_advisor import DepositAdvisor
            advisor = DepositAdvisor(self.portfolio_file)
            
            # Find category for this ticker
            etf_category = None
            for cat, etfs in advisor.ETF_CATEGORIES.items():
                if ticker in etfs:
                    etf_category = cat
                    break
            
            if etf_category:
                industry_trend = advisor.analyze_industry_trends(etf_category)
                analysis["industry_trend"] = industry_trend
                
                # Add industry trend to score
                if industry_trend.get("trend") == "STRONG_UPTREND":
                    score += 10
                elif industry_trend.get("trend") == "UPTREND":
                    score += 5
                elif industry_trend.get("trend") == "DOWNTREND":
                    score -= 5
        except Exception:
            pass  # Industry trend analysis is optional
        
        analysis["recommendation_score"] = max(0, min(100, score))
        
        if score >= 70:
            analysis["recommendation"] = "STRONG BUY"
        elif score >= 60:
            analysis["recommendation"] = "BUY"
        elif score >= 40:
            analysis["recommendation"] = "HOLD"
        elif score >= 30:
            analysis["recommendation"] = "SELL"
        else:
            analysis["recommendation"] = "STRONG SELL"
        
        return analysis
    
    def calculate_portfolio_metrics(self, portfolio: Dict, analyses: List[Dict]) -> Dict:
        """Calculate overall portfolio metrics."""
        total_value = portfolio.get("cash", 0)
        holdings_value = sum(a["current_value"] for a in analyses)
        total_value += holdings_value
        
        # Calculate weights
        weights = {}
        for analysis in analyses:
            if total_value > 0:
                weights[analysis["ticker"]] = analysis["current_value"] / total_value
        
        # Portfolio diversification score
        diversification_score = 1 - sum(w**2 for w in weights.values())  # Herfindahl index
        
        # Average recommendation score
        avg_score = np.mean([a["recommendation_score"] for a in analyses]) if analyses else 50
        
        return {
            "total_value": total_value,
            "cash": portfolio.get("cash", 0),
            "holdings_value": holdings_value,
            "diversification_score": diversification_score,
            "average_recommendation_score": avg_score,
            "weights": weights
        }
    
    def find_best_etfs_to_buy(self, amount_usd: float, current_holdings: List[str], exclude_tickers: List[str] = None) -> List[Dict]:
        """Find best ETFs to buy for diversification - uses all categories from DepositAdvisor."""
        from deposit_advisor import DepositAdvisor
        
        exclude_tickers = exclude_tickers or []
        advisor = DepositAdvisor(self.portfolio_file)
        
        # Get all ETFs from all categories (including Gold, Silver, Crypto, AI, etc.)
        all_etfs = []
        for category, etfs in advisor.ETF_CATEGORIES.items():
            for etf in etfs:
                if etf not in all_etfs:
                    all_etfs.append(etf)
        
        # Filter out current holdings and excluded
        diversification_etfs = [
            etf for etf in all_etfs 
            if etf not in current_holdings and etf not in exclude_tickers
        ]
        
        # Filter out current holdings and excluded
        candidate_etfs = [
            etf for etf in diversification_etfs 
            if etf not in current_holdings and etf not in exclude_tickers
        ]
        
        # Analyze top candidates in parallel
        def analyze_etf_candidate(etf):
            try:
                stock = yf.Ticker(etf)
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    shares = int(amount_usd / price / 3)  # Divide by 3 for 3 recommendations
                    if shares > 0:
                        # Simple scoring
                        score = 60  # Base score for diversification
                        info = stock.info
                        if info.get('annualReportExpenseRatio', 1) < 0.001:
                            score += 10
                        
                        return {
                            "ticker": etf,
                            "name": info.get('longName', etf),
                            "shares": shares,
                            "price": price,
                            "allocation_amount": shares * price,
                            "score": score,
                            "recommendation": "BUY",
                            "reasons": ["Diversification", "Low expense ratio" if info.get('annualReportExpenseRatio', 1) < 0.001 else "Good diversification"]
                        }
            except Exception:
                pass
            return None
        
        recommendations = []
        with ThreadPoolExecutor(max_workers=min(5, len(candidate_etfs[:10]))) as executor:
            futures = [executor.submit(analyze_etf_candidate, etf) for etf in candidate_etfs[:10]]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    recommendations.append(result)
        
        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]
    
    def check_rebalancing(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """Determine if rebalancing is needed."""
        rebalancing = {
            "needed": False,
            "reason": "",
            "recommendations": [],
            "buy_recommendations": []
        }
        
        weights = portfolio_metrics["weights"]
        total_holdings = len(weights)
        total_value_usd = portfolio_metrics["holdings_value"]
        current_tickers = [a["ticker"] for a in analyses]
        total_sell_amount = 0
        
        # Check for over-concentration
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.4:  # More than 40% in one holding
            rebalancing["needed"] = True
            ticker = max(weights, key=weights.get)
            current_value_usd = total_value_usd * max_weight
            target_value_usd = total_value_usd * 0.25
            reduce_amount_usd = current_value_usd - target_value_usd
            total_sell_amount += reduce_amount_usd
            
            # Find the analysis for this ticker to get current price
            ticker_analysis = next((a for a in analyses if a["ticker"] == ticker), None)
            if ticker_analysis and ticker_analysis["current_price"] > 0:
                reduce_shares = int(reduce_amount_usd / ticker_analysis["current_price"])
                rebalancing["reason"] = f"Over-concentration: {ticker} is {max_weight*100:.1f}% of portfolio"
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "current_weight": max_weight,
                    "target_weight": 0.25,
                    "reduce_amount_usd": reduce_amount_usd,
                    "reduce_shares": reduce_shares,
                    "current_price_usd": ticker_analysis["current_price"],
                    "reason": f"Diversification - reduce by ${reduce_amount_usd:,.2f} ({reduce_shares} shares)"
                })
            else:
                rebalancing["reason"] = f"Over-concentration: {ticker} is {max_weight*100:.1f}% of portfolio"
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "current_weight": max_weight,
                    "target_weight": 0.25,
                    "reason": "Diversification"
                })
        
        # Check for poor diversification
        if portfolio_metrics["diversification_score"] < 0.5 and total_holdings < 5:
            rebalancing["needed"] = True
            if not rebalancing["reason"]:
                rebalancing["reason"] = "Low diversification - consider adding more holdings"
        
        # Check for underperforming holdings
        for analysis in analyses:
            if analysis["recommendation_score"] < 30:
                rebalancing["needed"] = True
                if not rebalancing["reason"]:
                    rebalancing["reason"] = f"Underperforming holding: {analysis['ticker']}"
                
                # Calculate sell amount
                sell_value_usd = analysis["current_value"]
                sell_shares = analysis["quantity"]
                total_sell_amount += sell_value_usd
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": analysis["ticker"],
                    "sell_amount_usd": sell_value_usd,
                    "sell_shares": sell_shares,
                    "current_price_usd": analysis["current_price"],
                    "reason": f"Low recommendation score: {analysis['recommendation_score']:.1f} - Consider selling {sell_shares} shares (${sell_value_usd:,.2f})"
                })
        
        # If we're selling, recommend what to buy instead
        if total_sell_amount > 0 and rebalancing["recommendations"]:
            sell_tickers = [r["ticker"] for r in rebalancing["recommendations"] if r["action"] == "SELL"]
            buy_recs = self.find_best_etfs_to_buy(total_sell_amount, current_tickers, exclude_tickers=sell_tickers)
            rebalancing["buy_recommendations"] = buy_recs
        
        return rebalancing
    
    def analyze(self) -> Dict:
        """Main analysis function."""
        print("=" * 60)
        print("Portfolio Analysis Starting...")
        print("=" * 60)
        
        # Load portfolio
        portfolio = self.load_portfolio()
        
        if not portfolio.get("holdings"):
            print("No holdings found in portfolio. Please add holdings first.")
            return {}
        
        # Get current prices and market status
        tickers = [h["ticker"] for h in portfolio["holdings"]]
        prices, market_status, market_message = self.get_current_prices(tickers)
        
        # Display market status
        print(f"\n📊 Market Status: {market_message}")
        if market_status:
            print("   ⚡ Using REAL-TIME prices")
        else:
            print("   📅 Using LAST CLOSE prices")
        print()
        
        # Analyze each holding in parallel for better performance
        analyses = []
        holdings_to_analyze = [
            (h["ticker"], h["quantity"], prices.get(h["ticker"], h.get("last_price", 0)))
            for h in portfolio["holdings"]
            if prices.get(h["ticker"], h.get("last_price", 0)) > 0
        ]
        
        if holdings_to_analyze:
            with ThreadPoolExecutor(max_workers=min(5, len(holdings_to_analyze))) as executor:
                futures = {
                    executor.submit(self.analyze_holding, ticker, quantity, price): (ticker, quantity, price)
                    for ticker, quantity, price in holdings_to_analyze
                }
                
                for future in as_completed(futures):
                    try:
                        analysis = future.result()
                        analyses.append(analysis)
                    except Exception as e:
                        ticker, _, _ = futures[future]
                        print(f"Error analyzing {ticker}: {e}")
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio, analyses)
        
        # Check rebalancing
        rebalancing = self.check_rebalancing(portfolio_metrics, analyses)
        
        # Update portfolio with current values
        portfolio["total_value"] = portfolio_metrics["total_value"]
        for i, analysis in enumerate(analyses):
            if i < len(portfolio["holdings"]):
                portfolio["holdings"][i]["last_price"] = analysis["current_price"]
                portfolio["holdings"][i]["current_value"] = analysis["current_value"]
        
        # Save updated portfolio
        self.save_portfolio(portfolio)
        
        # Compile results
        results = {
            "portfolio_metrics": portfolio_metrics,
            "holdings_analysis": analyses,
            "rebalancing": rebalancing,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print results
        self.print_analysis_results(results)
        
        # Ask for confirmation if rebalancing is needed
        if rebalancing["needed"] and (rebalancing["recommendations"] or rebalancing["buy_recommendations"]):
            confirmed = self.ask_rebalancing_confirmation()
            if confirmed:
                self.update_portfolio_from_rebalancing(portfolio, rebalancing, analyses)
                print("\n✅ Portfolio updated successfully based on rebalancing actions!\n")
            else:
                print("\n❌ Portfolio not updated. No changes were made.\n")
        
        return results
    
    def ask_rebalancing_confirmation(self) -> bool:
        """Ask user for confirmation to execute rebalancing."""
        while True:
            response = input("\nDid you execute the rebalancing actions (sell/buy)? (yes/no): ").strip().lower()
            if response in ['yes', 'y', 'כן', 'י']:
                return True
            elif response in ['no', 'n', 'לא', 'ל']:
                return False
            else:
                print("Please enter 'yes' or 'no' (כן/לא)")
    
    def update_portfolio_from_rebalancing(self, portfolio: Dict, rebalancing: Dict, analyses: List[Dict]):
        """Update portfolio.json based on rebalancing recommendations."""
        exchange_rate = self.get_exchange_rate()
        
        # Process SELL actions
        for rec in rebalancing.get("recommendations", []):
            if rec["action"] == "SELL":
                ticker = rec["ticker"]
                shares_to_sell = rec.get("reduce_shares", rec.get("sell_shares", 0))
                
                # Find and update the holding
                for holding in portfolio.get("holdings", []):
                    if holding["ticker"] == ticker:
                        current_quantity = holding.get("quantity", 0)
                        new_quantity = max(0, current_quantity - shares_to_sell)
                        
                        if new_quantity > 0:
                            # Update quantity and value
                            holding["quantity"] = new_quantity
                            current_price = rec.get("current_price_usd", holding.get("last_price", 0))
                            holding["last_price"] = current_price
                            holding["current_value"] = new_quantity * current_price
                        else:
                            # Remove holding if quantity becomes 0
                            portfolio["holdings"].remove(holding)
                        
                        # Add to cash
                        sell_amount = shares_to_sell * rec.get("current_price_usd", holding.get("last_price", 0))
                        portfolio["cash"] = portfolio.get("cash", 0) + sell_amount
                        break
        
        # Process BUY actions
        for rec in rebalancing.get("buy_recommendations", []):
            ticker = rec["ticker"]
            shares_to_buy = rec.get("shares", 0)
            price = rec.get("price", 0)
            buy_amount = rec.get("allocation_amount", shares_to_buy * price)
            
            if shares_to_buy > 0 and price > 0:
                # Check if holding already exists
                existing_holding = None
                for holding in portfolio.get("holdings", []):
                    if holding["ticker"] == ticker:
                        existing_holding = holding
                        break
                
                if existing_holding:
                    # Update existing holding
                    existing_holding["quantity"] += shares_to_buy
                    existing_holding["last_price"] = price
                    existing_holding["current_value"] = existing_holding["quantity"] * price
                else:
                    # Add new holding
                    new_holding = {
                        "ticker": ticker,
                        "quantity": shares_to_buy,
                        "last_price": price,
                        "current_value": shares_to_buy * price
                    }
                    portfolio.setdefault("holdings", []).append(new_holding)
                
                # Subtract from cash
                portfolio["cash"] = max(0, portfolio.get("cash", 0) - buy_amount)
        
        # Recalculate total value
        total_value = portfolio.get("cash", 0)
        for holding in portfolio.get("holdings", []):
            total_value += holding.get("current_value", 0)
        portfolio["total_value"] = total_value
        
        # Save updated portfolio
        self.save_portfolio(portfolio)
        
        # Print summary
        print("\n" + "-" * 60)
        print("PORTFOLIO UPDATE SUMMARY")
        print("-" * 60)
        print(f"Cash: ${portfolio.get('cash', 0):,.2f} (₪{portfolio.get('cash', 0) * exchange_rate:,.2f})")
        print(f"Total Holdings: {len(portfolio.get('holdings', []))}")
        print(f"Total Portfolio Value: ${total_value:,.2f} (₪{total_value * exchange_rate:,.2f})")
        portfolio_path = os.path.abspath(self.portfolio_file)
        print(f"\n✅ Portfolio saved locally to: {portfolio_path}")
    
    def print_analysis_results(self, results: Dict):
        """Print analysis results in a formatted way."""
        exchange_rate = self.get_exchange_rate()
        
        print("\n" + "=" * 60)
        print("PORTFOLIO ANALYSIS RESULTS")
        print("=" * 60)
        
        metrics = results["portfolio_metrics"]
        total_value_ils = metrics['total_value'] * exchange_rate
        cash_ils = metrics['cash'] * exchange_rate
        holdings_value_ils = metrics['holdings_value'] * exchange_rate
        
        print(f"\nTotal Portfolio Value: ₪{total_value_ils:,.2f} (${metrics['total_value']:,.2f})")
        print(f"Cash: ₪{cash_ils:,.2f} (${metrics['cash']:,.2f})")
        print(f"Holdings Value: ₪{holdings_value_ils:,.2f} (${metrics['holdings_value']:,.2f})")
        print(f"Exchange Rate: {exchange_rate} ILS/USD")
        print(f"Diversification Score: {metrics['diversification_score']:.2f} (1.0 = perfect diversification)")
        print(f"Average Recommendation Score: {metrics['average_recommendation_score']:.1f}/100")
        
        print("\n" + "-" * 60)
        print("HOLDINGS ANALYSIS (All prices and values in USD)")
        print("-" * 60)
        
        for analysis in results["holdings_analysis"]:
            print(f"\n{analysis['ticker']}:")
            print(f"  Quantity: {analysis['quantity']} shares")
            print(f"  Current Price: ${analysis['current_price']:.2f} per share")
            print(f"  Current Value: ${analysis['current_value']:,.2f}")
            print(f"  Weight: {results['portfolio_metrics']['weights'].get(analysis['ticker'], 0)*100:.1f}%")
            print(f"  Recommendation: {analysis['recommendation']} (Score: {analysis['recommendation_score']:.1f}/100)")
            
            if analysis["technical_indicators"]:
                ti = analysis["technical_indicators"]
                print(f"  RSI: {ti.get('rsi', 0):.1f}")
                print(f"  Momentum: {ti.get('momentum', 0):.2f}%")
                print(f"  Volatility: {ti.get('volatility', 0):.2f}%")
                print(f"  Trend: {ti.get('trend', 'NEUTRAL')}")
                if 'beta' in ti:
                    print(f"  Beta: {ti.get('beta', 1.0):.2f}")
                if 'max_drawdown' in ti:
                    print(f"  Max Drawdown: {ti.get('max_drawdown', 0):.2f}%")
        
        print("\n" + "=" * 60)
        print("REBALANCING SUMMARY")
        print("=" * 60)
        
        rebalancing = results["rebalancing"]
        exchange_rate = self.get_exchange_rate()
        
        if rebalancing["needed"]:
            print("⚠️  REBALANCING IS RECOMMENDED")
            print(f"Reason: {rebalancing['reason']}\n")
            
            # Create summary table
            print("┌" + "─" * 78 + "┐")
            print("│" + " " * 20 + "ACTION SUMMARY" + " " * 43 + "│")
            print("├" + "─" * 78 + "┤")
            
            # HOLD section
            hold_items = [a for a in results["holdings_analysis"] 
                         if a["ticker"] not in [r["ticker"] for r in rebalancing["recommendations"]]]
            if hold_items:
                print("│ ✅ HOLD (Keep as is):" + " " * 57 + "│")
                for item in hold_items:
                    value_ils = item["current_value"] * exchange_rate
                    print(f"│   • {item['ticker']:6s} - {item['quantity']:3d} shares × ${item['current_price']:7.2f} = ${item['current_value']:8,.2f} (₪{value_ils:8,.2f})" + " " * 5 + "│")
            
            # SELL section
            if rebalancing["recommendations"]:
                print("│" + "─" * 78 + "│")
                print("│ 🔴 SELL:" + " " * 70 + "│")
                total_sell = 0
                for rec in rebalancing["recommendations"]:
                    if 'reduce_amount_usd' in rec:
                        amount = rec['reduce_amount_usd']
                        shares = rec['reduce_shares']
                        price = rec['current_price_usd']
                    elif 'sell_amount_usd' in rec:
                        amount = rec['sell_amount_usd']
                        shares = rec['sell_shares']
                        price = rec['current_price_usd']
                    else:
                        continue
                    total_sell += amount
                    amount_ils = amount * exchange_rate
                    print(f"│   • {rec['ticker']:6s} - Sell {shares:3d} shares × ${price:7.2f} = ${amount:8,.2f} (₪{amount_ils:8,.2f})" + " " * 5 + "│")
                
                if total_sell > 0:
                    total_sell_ils = total_sell * exchange_rate
                    print("│" + "─" * 78 + "│")
                    print(f"│   Total to sell: ${total_sell:10,.2f} (₪{total_sell_ils:10,.2f})" + " " * 44 + "│")
            
            # BUY section
            if rebalancing["buy_recommendations"]:
                print("│" + "─" * 78 + "│")
                print("│ 🟢 BUY (Recommended replacements):" + " " * 42 + "│")
                total_buy = 0
                for rec in rebalancing["buy_recommendations"]:
                    amount = rec.get('allocation_amount', 0)
                    shares = rec.get('shares', 0)
                    price = rec.get('price', 0)
                    if amount > 0 and shares > 0:
                        total_buy += amount
                        amount_ils = amount * exchange_rate
                        name = rec.get('name', rec['ticker'])[:30]
                        print(f"│   • {rec['ticker']:6s} - Buy {shares:3d} shares × ${price:7.2f} = ${amount:8,.2f} (₪{amount_ils:8,.2f})" + " " * 5 + "│")
                        print(f"│     {name}" + " " * (78 - len(name) - 5) + "│")
                        if rec.get('reasons'):
                            top_reason = rec['reasons'][0][:65]
                            print(f"│     Reason: {top_reason}" + " " * (78 - len(top_reason) - 12) + "│")
                
                if total_buy > 0:
                    total_buy_ils = total_buy * exchange_rate
                    print("│" + "─" * 78 + "│")
                    print(f"│   Total to buy:  ${total_buy:10,.2f} (₪{total_buy_ils:10,.2f})" + " " * 44 + "│")
            
            print("└" + "─" * 78 + "┘")
        else:
            print("✅ Portfolio is well-balanced. No rebalancing needed at this time.")
        
        print("\n" + "=" * 60)
        print(f"Analysis completed at: {results['timestamp']}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer()
    analyzer.analyze()

