#!/usr/bin/env python3
"""
Portfolio Analyzer - Advanced Investment Portfolio Analysis System
Analyzes portfolio holdings, provides recommendations, and suggests rebalancing.
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from advanced_analysis import AdvancedAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.cache_file = ".cache.json"  # Persistent cache file
        self.advanced_analyzer = AdvancedAnalyzer()  # Advanced analysis module
        self._load_cache()  # Load persistent cache on init
    
    def _load_cache(self):
        """Load persistent cache from file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # Load price cache (only valid entries)
                    now = datetime.now()
                    price_cache_data = cache_data.get('price_cache', {})
                    if price_cache_data and isinstance(price_cache_data, dict):
                        for ticker, data in price_cache_data.items():
                            if data and isinstance(data, dict) and 'timestamp' in data and 'price' in data:
                                try:
                                    cached_time = datetime.fromisoformat(data['timestamp'])
                                    if now - cached_time < self.cache_timeout:
                                        self.price_cache[ticker] = (data['price'], cached_time)
                                except (ValueError, KeyError) as e:
                                    logger.debug(f"Failed to load cache entry for {ticker}: {e}")
                                    continue
                    # Load exchange rate cache
                    ex_data = cache_data.get('exchange_rate')
                    if ex_data and isinstance(ex_data, dict) and 'timestamp' in ex_data and 'rate' in ex_data:
                        try:
                            cached_time = datetime.fromisoformat(ex_data['timestamp'])
                            if now - cached_time < self.cache_timeout:
                                self.exchange_rate_usd_ils = ex_data['rate']
                                self.exchange_rate_cache_time = cached_time
                        except (ValueError, KeyError) as e:
                            logger.debug(f"Failed to load exchange rate cache: {e}")
                    logger.info(f"Loaded cache from {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to persistent file."""
        try:
            cache_data = {
                'price_cache': {
                    ticker: {
                        'price': price,
                        'timestamp': timestamp.isoformat()
                    }
                    for ticker, (price, timestamp) in self.price_cache.items()
                },
                'exchange_rate': {
                    'rate': self.exchange_rate_usd_ils,
                    'timestamp': self.exchange_rate_cache_time.isoformat() if self.exchange_rate_cache_time else None
                } if self.exchange_rate_cache_time else None
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
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
                self._save_cache()  # Save to persistent cache
                return rate
            return self.exchange_rate_usd_ils
        except Exception as e:
            logger.warning(f"Failed to fetch exchange rate: {e}")
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
        except Exception as e:
            logger.warning(f"Failed to determine market status: {e}")
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
                                self._save_cache()  # Save to persistent cache
                                return ticker, price, True  # True = real-time
                        except Exception as e:
                            logger.debug(f"Failed to get real-time price for {ticker}: {e}")
                            pass
                    
                    # Fallback to historical data (last close)
                    info = stock.history(period="1d")
                    if not info.empty:
                        price = float(info['Close'].iloc[-1])
                        self.price_cache[ticker] = (price, now)
                        self._save_cache()  # Save to persistent cache
                        return ticker, price, False  # False = last close
                    else:
                        try:
                            price = float(stock.fast_info.get('lastPrice', 0))
                            if price > 0:
                                self.price_cache[ticker] = (price, now)
                                self._save_cache()  # Save to persistent cache
                                return ticker, price, False
                        except Exception as e:
                            logger.debug(f"Failed to get fallback price for {ticker}: {e}")
                            return ticker, None, False
                except Exception as e:
                    logger.warning(f"Could not fetch price for {ticker}: {e}")
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
        except Exception as e:
            logger.warning(f"Failed to fetch market data for {ticker}: {e}")
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
            # Use same period as data
            data_period = "1y" if len(data) < 252 else "2y" if len(data) < 504 else "5y"
            spy_data = spy.history(period=data_period)
            if not spy_data.empty:
                # Align data lengths
                min_len = min(len(data), len(spy_data))
                spy_returns = spy_data['Close'].pct_change().dropna().tail(min_len)
                asset_returns = returns.dropna().tail(min_len)
                if len(spy_returns) == len(asset_returns) and len(asset_returns) > 10:
                    covariance = np.cov(asset_returns, spy_returns)[0][1]
                    spy_variance = np.var(spy_returns)
                    if spy_variance > 0:
                        indicators['beta'] = covariance / spy_variance
                    else:
                        indicators['beta'] = 1.0
                else:
                    indicators['beta'] = 1.0
            else:
                indicators['beta'] = 1.0
        except Exception as e:
            logger.debug(f"Failed to calculate beta: {e}")
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
        except Exception as e:
            logger.debug(f"Failed to detect candlestick patterns: {e}")
            pass
        
        # Statistical forecast (mid-term yield optimization)
        try:
            forecast = self.advanced_analyzer.calculate_statistical_forecast(data, periods=252*3)  # 3 years
            if forecast:
                indicators['forecast'] = forecast
        except Exception as e:
            logger.debug(f"Failed to calculate statistical forecast: {e}")
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
                except Exception as e:
                    logger.debug(f"Failed to fetch NewsAPI data: {e}")
                    pass
        except Exception as e:
            logger.debug(f"Failed to fetch news sentiment for {ticker}: {e}")
            pass  # Silent fail, use default sentiment
        
        # Cache the result
        self.news_cache[ticker] = (sentiment, now)
        return sentiment
    
    def analyze_holding(self, ticker: str, quantity: float, current_price: float, verbose: bool = True, advisor = None) -> Dict:
        """Comprehensive analysis of a single holding."""
        if verbose:
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
        etf_category = None
        if advisor:  # Use provided advisor instead of creating new one
            try:
                # Find category for this ticker
                for cat, etfs in advisor.ETF_CATEGORIES.items():
                    if ticker in etfs:
                        etf_category = cat
                        break
                
                if etf_category:
                    industry_trend = advisor.analyze_industry_trends(etf_category)
                    analysis["industry_trend"] = industry_trend
                    
                    # Add industry trend to score
                    if industry_trend.get("trend") == "STRONG_UPTREND":
                        score += 15
                    elif industry_trend.get("trend") == "UPTREND":
                        score += 10
                    elif industry_trend.get("trend") == "DOWNTREND":
                        score -= 10
            except Exception as e:
                logger.debug(f"Failed to analyze industry trend: {e}")
                pass  # Industry trend analysis is optional
        
        # Statistical forecast (mid-term yield) - from technical indicators
        if analysis["technical_indicators"] and "forecast" in analysis["technical_indicators"]:
            forecast = analysis["technical_indicators"]["forecast"]
            if forecast and forecast.get("expected_return_polynomial") is not None:
                expected_return = forecast.get("expected_return_polynomial", 0)
                analysis["mid_term_forecast"] = {
                    "expected_3yr_return": expected_return,
                    "forecast_price": forecast.get("forecast_polynomial", 0)
                }
                
                # Add forecast to score
                if expected_return > 15:
                    score += 15
                elif expected_return > 10:
                    score += 10
                elif expected_return < -10:
                    score -= 10
        
        # Candlestick patterns - from technical indicators
        if analysis["technical_indicators"] and "candlestick_patterns" in analysis["technical_indicators"]:
            patterns = analysis["technical_indicators"]["candlestick_patterns"]
            if patterns:
                bullish = [p for p in patterns if p.get('signal') == 'BULLISH']
                bearish = [p for p in patterns if p.get('signal') == 'BEARISH']
                if bullish:
                    score += 5
                elif bearish:
                    score -= 5
        
        # Bond analysis (if applicable)
        if etf_category in ["BONDS", "HIGH_YIELD", "TIPS"]:
            try:
                bond_analysis = self.advanced_analyzer.analyze_bonds(ticker)
                if bond_analysis and "yield_analysis" in bond_analysis:
                    analysis["bond_analysis"] = bond_analysis
                    current_yield = bond_analysis["yield_analysis"].get("current_yield", 0)
                    yield_volatility = bond_analysis["yield_analysis"].get("yield_volatility", 0)
                    
                    # Risk-adjusted yield (simplified)
                    risk_adj_yield = current_yield - (yield_volatility / 2) if yield_volatility > 0 else current_yield
                    
                    if risk_adj_yield > 2:
                        score += 15
                    elif risk_adj_yield > 1:
                        score += 10
                    elif risk_adj_yield < 0.5:
                        score -= 10
            except Exception as e:
                logger.debug(f"Failed to analyze bonds: {e}")
                pass
        
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
        """
        Find best ETFs to buy following 75/25 Balanced Growth Strategy.
        Prioritizes Core ETFs and Bonds over high-risk trends.
        """
        from deposit_advisor import DepositAdvisor
        
        exclude_tickers = exclude_tickers or []
        advisor = DepositAdvisor(self.portfolio_file)
        
        # Define Core ETFs (priority)
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        # Define Satellite ETFs (safe growth)
        satellite_etfs = ["IWM", "VB", "XLK", "VGT", "VWO", "EEM", "XLV", "VHT"]
        # Define Bond ETFs (protection)
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
        
        # Exclude high-risk categories
        excluded_categories = ["LEVERAGED_2X", "LEVERAGED_3X", "LEVERAGED_INVERSE", "CRYPTO"]
        excluded_from_categories = []
        for cat in excluded_categories:
            if cat in advisor.ETF_CATEGORIES:
                excluded_from_categories.extend(advisor.ETF_CATEGORIES[cat])
        
        # Build candidate list prioritizing Core and Bonds
        candidate_etfs = []
        
        # Add Core ETFs first (highest priority)
        for etf in core_etfs:
            if etf not in current_holdings and etf not in exclude_tickers:
                candidate_etfs.append(etf)
        
        # Add Bond ETFs (high priority for protection)
        for etf in bond_etfs:
            if etf not in current_holdings and etf not in exclude_tickers:
                candidate_etfs.append(etf)
        
        # Add Satellite ETFs (moderate priority)
        for etf in satellite_etfs:
            if (etf not in current_holdings and 
                etf not in exclude_tickers and 
                etf not in candidate_etfs and
                etf not in excluded_from_categories):
                candidate_etfs.append(etf)
        
        # Analyze candidates in parallel
        def analyze_etf_candidate(etf):
            try:
                stock = yf.Ticker(etf)
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    shares = int(amount_usd / price / 3)  # Divide by 3 for 3 recommendations
                    if shares > 0:
                        # Scoring based on category
                        score = 60  # Base score
                        info = stock.info
                        
                        # Boost for Core and Bonds
                        if etf.upper() in [e.upper() for e in core_etfs]:
                            score += 20
                            category = "CORE"
                        elif etf.upper() in [e.upper() for e in bond_etfs]:
                            score += 25
                            category = "BONDS"
                        else:
                            category = "SATELLITE"
                        
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
                            "reasons": [
                                f"{category} holding" if category else "Diversification",
                                "Low expense ratio" if info.get('annualReportExpenseRatio', 1) < 0.001 else "Good diversification"
                            ],
                            "category": category
                        }
            except Exception as e:
                logger.debug(f"Failed to analyze ETF candidate {etf}: {e}")
                pass
            return None
        
        recommendations = []
        # Limit to top candidates (prioritize Core and Bonds)
        candidates_to_analyze = candidate_etfs[:10]
        with ThreadPoolExecutor(max_workers=min(5, len(candidates_to_analyze))) as executor:
            futures = [executor.submit(analyze_etf_candidate, etf) for etf in candidates_to_analyze]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    recommendations.append(result)
        
        # Sort by score (Core and Bonds will be prioritized)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]
    
    def check_75_25_balance(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """
        Check if portfolio follows 75/25 Balanced Growth Strategy:
        - 75% Stocks (50% Core + 25% Satellite)
        - 25% Bonds
        Returns recommendations to achieve proper balance.
        """
        balance_check = {
            "is_balanced": False,
            "stocks_percent": 0,
            "bonds_percent": 0,
            "core_percent": 0,
            "satellite_percent": 0,
            "recommendations": []
        }
        
        # Define Core ETFs (SPY, VOO, VXUS, etc.)
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        # Define Bond ETFs
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
        
        total_value = portfolio_metrics["total_value"]
        if total_value == 0:
            return balance_check
        
        # Calculate current allocation
        stocks_value = 0
        bonds_value = 0
        core_value = 0
        satellite_value = 0
        
        for analysis in analyses:
            ticker = analysis["ticker"]
            value = analysis["current_value"]
            
            if ticker.upper() in [e.upper() for e in bond_etfs]:
                bonds_value += value
            else:
                stocks_value += value
                if ticker.upper() in [e.upper() for e in core_etfs]:
                    core_value += value
                else:
                    satellite_value += value
        
        # Calculate percentages
        stocks_percent = (stocks_value / total_value) * 100 if total_value > 0 else 0
        bonds_percent = (bonds_value / total_value) * 100 if total_value > 0 else 0
        core_percent = (core_value / total_value) * 100 if total_value > 0 else 0
        satellite_percent = (satellite_value / total_value) * 100 if total_value > 0 else 0
        
        balance_check["stocks_percent"] = stocks_percent
        balance_check["bonds_percent"] = bonds_percent
        balance_check["core_percent"] = core_percent
        balance_check["satellite_percent"] = satellite_percent
        
        # Check if balanced (with tolerance of ±5%)
        target_stocks = 75
        target_bonds = 25
        target_core = 50
        target_satellite = 25
        
        tolerance = 5
        is_balanced = (
            abs(stocks_percent - target_stocks) <= tolerance and
            abs(bonds_percent - target_bonds) <= tolerance
        )
        
        balance_check["is_balanced"] = is_balanced
        
        # Generate recommendations if not balanced
        if not is_balanced:
            recommendations = []
            
            # Check if need more bonds
            if bonds_percent < target_bonds - tolerance:
                needed_bonds = total_value * (target_bonds - bonds_percent) / 100
                recommendations.append({
                    "action": "BUY_BONDS",
                    "amount": needed_bonds,
                    "reason": f"Portfolio has {bonds_percent:.1f}% bonds, target is {target_bonds}%. Need ${needed_bonds:,.2f} in bonds for protection."
                })
            
            # Check if need more core stocks
            if core_percent < target_core - tolerance:
                needed_core = total_value * (target_core - core_percent) / 100
                recommendations.append({
                    "action": "BUY_CORE",
                    "amount": needed_core,
                    "reason": f"Portfolio has {core_percent:.1f}% core stocks, target is {target_core}%. Need ${needed_core:,.2f} in core ETFs (SPY, VXUS)."
                })
            
            # Check if too many bonds
            if bonds_percent > target_bonds + tolerance:
                excess_bonds = total_value * (bonds_percent - target_bonds) / 100
                recommendations.append({
                    "action": "REDUCE_BONDS",
                    "amount": excess_bonds,
                    "reason": f"Portfolio has {bonds_percent:.1f}% bonds, target is {target_bonds}%. Consider reducing by ${excess_bonds:,.2f} to increase growth."
                })
            
            # Check if too many stocks
            if stocks_percent > target_stocks + tolerance:
                excess_stocks = total_value * (stocks_percent - target_stocks) / 100
                recommendations.append({
                    "action": "REDUCE_STOCKS",
                    "amount": excess_stocks,
                    "reason": f"Portfolio has {stocks_percent:.1f}% stocks, target is {target_stocks}%. Consider reducing by ${excess_stocks:,.2f} and adding bonds for protection."
                })
            
            balance_check["recommendations"] = recommendations
        
        return balance_check
    
    def _generate_concrete_75_25_recommendations(
        self,
        balance_info: Dict,
        portfolio_metrics: Dict,
        holdings_analysis: List[Dict],
        exchange_rate: float
    ) -> List[Dict]:
        """Generate concrete buy recommendations to achieve 75/25 balance."""
        recommendations = []
        
        bonds_percent = balance_info.get("bonds_percent", 0)
        stocks_percent = balance_info.get("stocks_percent", 0)
        core_percent = balance_info.get("core_percent", 0)
        satellite_percent = balance_info.get("satellite_percent", 0)
        
        total_value = portfolio_metrics.get("total_value", 0)
        cash_available = portfolio_metrics.get("cash", 0)
        
        if total_value == 0:
            return recommendations
        
        # Define target ETFs
        core_etfs = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
        bond_etfs = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
        
        # Check what we need
        target_bonds = total_value * 0.25
        current_bonds = total_value * (bonds_percent / 100)
        needed_bonds = max(0, target_bonds - current_bonds)
        
        target_core = total_value * 0.50
        current_core = total_value * (core_percent / 100)
        needed_core = max(0, target_core - current_core)
        
        target_satellite = total_value * 0.25
        current_satellite = total_value * (satellite_percent / 100)
        needed_satellite = max(0, target_satellite - current_satellite)
        
        # Get current holdings
        current_holdings = [h.get("ticker", "").upper() for h in holdings_analysis]
        
        # Recommend bonds if needed (priority)
        if needed_bonds > 100 and cash_available > 100:
            # Try to get price for bond ETFs
            import yfinance as yf
            for bond_etf in bond_etfs:
                if bond_etf.upper() not in current_holdings or needed_bonds > 500:
                    try:
                        stock = yf.Ticker(bond_etf)
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            price = float(hist['Close'].iloc[-1])
                            # Recommend up to needed amount or available cash
                            amount_to_use = min(needed_bonds, cash_available * 0.5)
                            shares = int(amount_to_use / price) if price > 0 else 0
                            if shares > 0:
                                actual_amount = shares * price
                                recommendations.append({
                                    "ticker": bond_etf,
                                    "shares": shares,
                                    "price": price,
                                    "amount": actual_amount,
                                    "amount_ils": actual_amount * exchange_rate,
                                    "reason": f"Add bonds to reach 25% target (currently {bonds_percent:.1f}%)"
                                })
                                needed_bonds -= actual_amount
                                if needed_bonds <= 0 or len(recommendations) >= 2:
                                    break
                    except Exception as e:
                        logger.debug(f"Failed to get price for {bond_etf}: {e}")
                        continue
        
        # Recommend core if needed
        if needed_core > 100 and cash_available > 100:
            import yfinance as yf
            for core_etf in core_etfs:
                if core_etf.upper() not in current_holdings or needed_core > 500:
                    try:
                        stock = yf.Ticker(core_etf)
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            price = float(hist['Close'].iloc[-1])
                            amount_to_use = min(needed_core, cash_available * 0.3)
                            shares = int(amount_to_use / price) if price > 0 else 0
                            if shares > 0:
                                actual_amount = shares * price
                                recommendations.append({
                                    "ticker": core_etf,
                                    "shares": shares,
                                    "price": price,
                                    "amount": actual_amount,
                                    "amount_ils": actual_amount * exchange_rate,
                                    "reason": f"Increase core holdings to reach 50% target (currently {core_percent:.1f}%)"
                                })
                                needed_core -= actual_amount
                                if needed_core <= 0 or len(recommendations) >= 4:
                                    break
                    except Exception as e:
                        logger.debug(f"Failed to get price for {core_etf}: {e}")
                        continue
        
        return recommendations
    
    def check_rebalancing(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """Determine if rebalancing is needed, including 75/25 balance check."""
        rebalancing = {
            "needed": False,
            "reason": "",
            "recommendations": [],
            "buy_recommendations": [],
            "balance_75_25": {}
        }
        
        # First check 75/25 balance
        balance_check = self.check_75_25_balance(portfolio_metrics, analyses)
        rebalancing["balance_75_25"] = balance_check
        
        if not balance_check["is_balanced"]:
            rebalancing["needed"] = True
            balance_reasons = [r["reason"] for r in balance_check["recommendations"]]
            if not rebalancing["reason"]:
                rebalancing["reason"] = f"Portfolio not balanced (75/25): {balance_reasons[0] if balance_reasons else 'Needs rebalancing'}"
        
        weights = portfolio_metrics["weights"]
        total_holdings = len(weights)
        total_value_usd = portfolio_metrics["holdings_value"]
        current_tickers = [a["ticker"] for a in analyses]
        total_sell_amount = 0
        
        # Check for over-concentration - BUT only if performance is poor
        # Logic: Performance first, concentration is only an upper limit
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.4:  # More than 40% in one holding
            ticker = max(weights, key=weights.get)
            ticker_analysis = next((a for a in analyses if a["ticker"] == ticker), None)
            
            # Only recommend selling if:
            # 1. Concentration is VERY high (>50%) OR
            # 2. Performance is poor (score < 50)
            should_reduce = False
            if max_weight > 0.5:  # Very high concentration (>50%)
                should_reduce = True
                reason_text = f"Very high concentration: {ticker} is {max_weight*100:.1f}% of portfolio (risk limit)"
            elif ticker_analysis and ticker_analysis.get("recommendation_score", 50) < 50:
                # High concentration + poor performance = sell
                should_reduce = True
                reason_text = f"Over-concentration + poor performance: {ticker} is {max_weight*100:.1f}% with score {ticker_analysis.get('recommendation_score', 0):.1f}/100"
            elif ticker_analysis and ticker_analysis.get("recommendation_score", 50) >= 70:
                # High concentration BUT good performance = keep it (just note it)
                should_reduce = False
                rebalancing["reason"] = f"Note: {ticker} is {max_weight*100:.1f}% of portfolio but performing well (Score: {ticker_analysis.get('recommendation_score', 0):.1f}/100) - keeping due to strong performance"
            
            if should_reduce:
                rebalancing["needed"] = True
                current_value_usd = total_value_usd * max_weight
                target_value_usd = total_value_usd * 0.30  # Reduce to 30% (not 25%) to be less aggressive
                reduce_amount_usd = current_value_usd - target_value_usd
                total_sell_amount += reduce_amount_usd
                
                if ticker_analysis and ticker_analysis["current_price"] > 0:
                    reduce_shares = int(reduce_amount_usd / ticker_analysis["current_price"])
                    if not rebalancing["reason"]:
                        rebalancing["reason"] = reason_text
                    rebalancing["recommendations"].append({
                        "action": "SELL",
                        "ticker": ticker,
                        "current_weight": max_weight,
                        "target_weight": 0.30,
                        "reduce_amount_usd": reduce_amount_usd,
                        "reduce_shares": reduce_shares,
                        "current_price_usd": ticker_analysis["current_price"],
                        "reason": f"{reason_text} - reduce by ${reduce_amount_usd:,.2f} ({reduce_shares} shares)"
                    })
                else:
                    if not rebalancing["reason"]:
                        rebalancing["reason"] = reason_text
                    rebalancing["recommendations"].append({
                        "action": "SELL",
                        "ticker": ticker,
                        "current_weight": max_weight,
                        "target_weight": 0.30,
                        "reason": reason_text
                    })
        
        # Check for poor diversification
        if portfolio_metrics["diversification_score"] < 0.5 and total_holdings < 5:
            rebalancing["needed"] = True
            if not rebalancing["reason"]:
                rebalancing["reason"] = "Low diversification - consider adding more holdings"
        
        # PRIORITY 1: Check for underperforming holdings (STRONG SELL or SELL recommendations)
        # Performance-based selling takes priority over concentration
        for analysis in analyses:
            score = analysis.get("recommendation_score", 50)
            recommendation = analysis.get("recommendation", "HOLD")
            ticker = analysis["ticker"]
            current_weight = weights.get(ticker, 0)
            
            # Sell if performance is poor, regardless of concentration
            if recommendation == "STRONG SELL" or score < 30:
                rebalancing["needed"] = True
                if not rebalancing["reason"]:
                    rebalancing["reason"] = f"Underperforming holding: {ticker} (Score: {score:.1f}/100) - Performance-based sell"
                
                # More aggressive selling for very poor performance
                sell_percentage = 0.75 if recommendation == "STRONG SELL" or score < 20 else 0.5
                sell_value_usd = analysis["current_value"] * sell_percentage
                sell_shares = int(analysis["quantity"] * sell_percentage)
                total_sell_amount += sell_value_usd
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "sell_amount_usd": sell_value_usd,
                    "sell_shares": sell_shares,
                    "current_price_usd": analysis["current_price"],
                    "current_weight": current_weight,
                    "reason": f"{recommendation} - Score: {score:.1f}/100. Poor performance - selling {sell_percentage*100:.0f}% ({sell_shares} shares, ${sell_value_usd:,.2f})"
                })
            elif recommendation == "SELL" or (30 <= score < 40):
                # Moderate underperformance - sell smaller portion
                rebalancing["needed"] = True
                if not rebalancing["reason"]:
                    rebalancing["reason"] = f"Underperforming holding: {ticker} (Score: {score:.1f}/100)"
                
                sell_percentage = 0.25
                sell_value_usd = analysis["current_value"] * sell_percentage
                sell_shares = int(analysis["quantity"] * sell_percentage)
                total_sell_amount += sell_value_usd
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "sell_amount_usd": sell_value_usd,
                    "sell_shares": sell_shares,
                    "current_price_usd": analysis["current_price"],
                    "current_weight": current_weight,
                    "reason": f"{recommendation} - Score: {score:.1f}/100. Moderate underperformance - selling {sell_percentage*100:.0f}% ({sell_shares} shares, ${sell_value_usd:,.2f})"
                })
        
        # Check for holdings with very low scores but not yet SELL (warning threshold)
        low_score_holdings = [a for a in analyses if 30 <= a["recommendation_score"] < 40]
        if low_score_holdings and not rebalancing["needed"]:
            # Don't force rebalancing, but note it in the reason
            rebalancing["reason"] = f"Some holdings have low scores: {', '.join([a['ticker'] for a in low_score_holdings])}"
        
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
        
        # Show cache usage info
        cached_prices = sum(1 for t in tickers if t in self.price_cache and 
                           datetime.now() - self.price_cache[t][1] < (timedelta(minutes=1) if market_status else self.cache_timeout))
        if cached_prices > 0:
            print(f"   💾 Using cached prices for {cached_prices}/{len(tickers)} tickers (cache: {'1 min' if market_status else '5 min'})")
        print()
        
        # Create DepositAdvisor once (to avoid loading cache multiple times)
        try:
            from deposit_advisor import DepositAdvisor
            advisor = DepositAdvisor(self.portfolio_file)
        except Exception as e:
            logger.debug(f"Could not create DepositAdvisor: {e}")
            advisor = None
        
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
                        executor.submit(self.analyze_holding, ticker, quantity, price, verbose=True, advisor=advisor): (ticker, quantity, price)
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
        
        # Add concrete 75/25 recommendations if needed
        balance_info = rebalancing.get("balance_75_25", {})
        if balance_info and not balance_info.get("is_balanced", False):
            exchange_rate = self.get_exchange_rate()
            concrete_recs = self._generate_concrete_75_25_recommendations(
                balance_info, portfolio_metrics, analyses, exchange_rate
            )
            # Add to buy_recommendations if not already there
            if concrete_recs and not rebalancing.get("buy_recommendations"):
                rebalancing["buy_recommendations"] = []
            for rec in concrete_recs:
                # Check if not already in buy_recommendations
                if not any(r.get("ticker") == rec["ticker"] for r in rebalancing.get("buy_recommendations", [])):
                    rebalancing["buy_recommendations"].append({
                        "ticker": rec["ticker"],
                        "shares": rec["shares"],
                        "price": rec["price"],
                        "allocation_amount": rec["amount"],
                        "name": rec["ticker"],
                        "reasons": [rec["reason"]]
                    })
        
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
        
        # Show 75/25 balance status
        balance_info = results["rebalancing"].get("balance_75_25", {})
        if balance_info:
            print("\n" + "-" * 60)
            print("75/25 BALANCED GROWTH STRATEGY STATUS")
            print("-" * 60)
            stocks_pct = balance_info.get("stocks_percent", 0)
            bonds_pct = balance_info.get("bonds_percent", 0)
            core_pct = balance_info.get("core_percent", 0)
            satellite_pct = balance_info.get("satellite_percent", 0)
            is_balanced = balance_info.get("is_balanced", False)
            
            status_icon = "✅" if is_balanced else "⚠️"
            print(f"{status_icon} Stocks: {stocks_pct:.1f}% (Target: 75%)")
            print(f"   ├─ Core: {core_pct:.1f}% (Target: 50%)")
            print(f"   └─ Satellite: {satellite_pct:.1f}% (Target: 25%)")
            print(f"{status_icon} Bonds: {bonds_pct:.1f}% (Target: 25%)")
            
            if not is_balanced:
                recommendations = balance_info.get("recommendations", [])
                if recommendations:
                    print(f"\n📋 Recommendations to achieve 75/25 balance:")
                    for rec in recommendations:
                        print(f"   • {rec.get('reason', 'N/A')}")
                    
                    # Generate concrete buy recommendations
                    print(f"\n💡 CONCRETE BUY RECOMMENDATIONS:")
                    concrete_recs = self._generate_concrete_75_25_recommendations(
                        balance_info, metrics, results["holdings_analysis"], exchange_rate
                    )
                    if concrete_recs:
                        for rec in concrete_recs:
                            print(f"   🟢 BUY: {rec['ticker']} - {rec['shares']} shares × ${rec['price']:.2f} = ${rec['amount']:,.2f} (₪{rec['amount_ils']:,.2f})")
                            print(f"      Reason: {rec['reason']}")
        
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
        
        # Remind user to update GitHub secret if portfolio changed
        if rebalancing.get("needed") or any(a.get("recommendation") in ["BUY", "SELL"] for a in results["holdings_analysis"]):
            print("💡 REMINDER: Update GitHub Secret")
            print("   Your portfolio has changed. Update the secret so GitHub Actions uses the latest data:")
            print("   Run: make update-secret")
            print("   Or go to: https://github.com/liorFri2392/lior-s_broker/settings/secrets/actions\n")

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer()
    analyzer.analyze()

