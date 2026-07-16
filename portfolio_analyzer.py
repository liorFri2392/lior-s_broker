#!/usr/bin/env python3
"""
Portfolio Analyzer - Advanced Investment Portfolio Analysis System
Analyzes portfolio holdings, provides recommendations, and suggests rebalancing.
"""

import json
import os
import sys
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from advanced_analysis import AdvancedAnalyzer
from tax_analyzer import TaxAnalyzer
import allocation
import market_data
import etf_universe
import ledger
import portfolio_io
import github_secret
import scoring
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

    # 80/20 strategy ETF universes - the single source of truth lives in
    # etf_universe.py; re-exposed as class attributes for the many self.* uses.
    CORE_ETFS = etf_universe.CORE_ETFS
    BOND_ETFS = etf_universe.BOND_ETFS
    SATELLITE_ETFS = etf_universe.SATELLITE_ETFS

    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        # Price / market-data / FX caching is now owned by the process-wide
        # market_data singleton (batched, thread-safe, DST-correct, persists to
        # .cache.json). Only news caching remains local (not provided there).
        self.news_cache = {}  # Cache for news: {ticker: (sentiment, timestamp)}
        self.cache_timeout = timedelta(hours=4)   # Cache when market closed: 4 hours for stability
        self.cache_timeout_market_open = timedelta(minutes=60)  # When market open: 60 min to reduce noise
        self.rebalancing_cooldown_days = 30  # Don't recommend rebalancing for underperformers more than once per 30 days
        self.advanced_analyzer = AdvancedAnalyzer()  # Advanced analysis module

    @staticmethod
    def normalize_ils_per_usd(raw_rate: float) -> float:
        """Return how many ILS equal 1 USD (delegates to the shared market layer)."""
        return market_data.market.normalize_ils_per_usd(raw_rate)

    def get_exchange_rate(self) -> float:
        """Get ILS per 1 USD (for USD→ILS multiply, for ILS→USD divide)."""
        return market_data.get_exchange_rate()

    def load_portfolio(self) -> Dict:
        """Load portfolio from JSON file."""
        return portfolio_io.load_portfolio(self.portfolio_file)
    
    def save_portfolio(self, portfolio: Dict, sync_github_secret: bool = True):
        """Save portfolio to JSON file locally."""
        portfolio["last_updated"] = datetime.now().isoformat()
        portfolio_path = os.path.abspath(self.portfolio_file)
        with open(portfolio_path, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)
        
        if sync_github_secret:
            # Try to update GitHub secret automatically (if GitHub CLI is available)
            self._try_update_github_secret(portfolio)

    def refresh_portfolio_prices(self, verbose: bool = False, sync_github_secret: bool = False) -> Dict:
        """Refresh last_price and current_value from live market data (no trade prompts)."""
        portfolio = self.load_portfolio()
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return portfolio

        tickers = [h["ticker"] for h in holdings]
        prices, market_status, market_message = self.get_current_prices(tickers)

        if verbose:
            print(f"\n🔄 Refreshing portfolio prices...")
            print(f"   📊 {market_message}")

        total_value = portfolio.get("cash", 0)
        refreshed = 0
        for holding in holdings:
            ticker = holding["ticker"]
            # Prefer the freshly fetched price; only fall back to the stored
            # last_price when there is no positive live quote (a valid 0.0 or
            # None must not silently reuse a stale price via ``or``).
            fetched = prices.get(ticker)
            price = fetched if (fetched is not None and fetched > 0) else holding.get("last_price", 0)
            if price <= 0:
                total_value += holding.get("current_value", 0)
                continue
            holding["last_price"] = price
            holding["current_value"] = holding.get("quantity", 0) * price
            total_value += holding["current_value"]
            refreshed += 1

        portfolio["total_value"] = total_value
        self.save_portfolio(portfolio, sync_github_secret=sync_github_secret)

        if verbose:
            print(f"   ✅ Updated {refreshed}/{len(holdings)} holdings — portfolio value: ${total_value:,.2f}\n")

        return portfolio
    
    def _try_update_github_secret(self, portfolio: Dict):
        """Best-effort sync of the portfolio to the GitHub secret (shared impl)."""
        return github_secret.update_portfolio_secret(portfolio)

    def is_market_open(self) -> Tuple[bool, str]:
        """Check if US stock market (NYSE/NASDAQ) is currently open (DST-correct)."""
        return market_data.is_market_open()

    def get_current_prices(self, tickers: List[str]) -> Tuple[Dict[str, float], Optional[bool], str]:
        """Get current prices for tickers.
        Delegates to the shared market layer, which batches (one yf.download for
        all uncached tickers), caches, and persists thread-safely.
        Returns: (prices_dict, market_status, market_message)"""
        return market_data.get_prices(tickers)

    def get_market_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical market data for analysis (cached by the shared layer)."""
        return market_data.get_history(ticker, period=period)

    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        if data is None or data.empty or len(data) < 20:
            return {}
        
        if 'Close' not in data.columns:
            return {}
        
        indicators = {}
        closes = data['Close']

        # Rolling moving averages (use the LAST rolling value, not nested .tail().mean()
        # which produced different windows than .rolling and was harder to reason about).
        sma_20_series = closes.rolling(window=20).mean()
        sma_50_series = closes.rolling(window=min(50, len(closes))).mean()
        sma_200_series = closes.rolling(window=min(200, len(closes))).mean()
        indicators['sma_20'] = float(sma_20_series.iloc[-1]) if not pd.isna(sma_20_series.iloc[-1]) else float(closes.tail(20).mean())
        indicators['sma_50'] = float(sma_50_series.iloc[-1]) if not pd.isna(sma_50_series.iloc[-1]) else float(closes.tail(min(50, len(closes))).mean())
        indicators['sma_200'] = float(sma_200_series.iloc[-1]) if not pd.isna(sma_200_series.iloc[-1]) else float(closes.tail(min(200, len(closes))).mean())

        # Wilder's RSI (the original definition used by Bloomberg/TradingView/MT4).
        # Wilder smoothing is an EWMA with alpha = 1/14. Equivalent in pandas:
        # ewm(alpha=1/period, adjust=False). The simple-rolling-mean variant
        # (a.k.a. "Cutler's RSI") diverges by 5–15% from Wilder over the cycle.
        period = 14
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        if pd.isna(avg_loss.iloc[-1]) or pd.isna(avg_gain.iloc[-1]):
            indicators['rsi'] = 50.0
        elif avg_loss.iloc[-1] < 1e-12:
            indicators['rsi'] = 100.0
        else:
            rs_last = float(avg_gain.iloc[-1] / avg_loss.iloc[-1])
            indicators['rsi'] = float(100.0 - (100.0 / (1.0 + rs_last)))

        # Volatility (annualized). Drop the leading NaN from pct_change explicitly.
        returns = closes.pct_change().dropna()
        if len(returns) == 0:
            return {}
        indicators['volatility'] = float(returns.std() * np.sqrt(252) * 100)

        # Momentum (20-day price return)
        indicators['momentum'] = float((closes.iloc[-1] / closes.iloc[-20] - 1) * 100) if len(closes) >= 20 else 0.0

        # Sharpe ratio (annualized; dynamic risk-free rate)
        try:
            risk_free_rate = self.advanced_analyzer.get_risk_free_rate()
        except Exception:
            risk_free_rate = 0.045
        daily_rf = risk_free_rate / 252.0

        excess_returns = returns - daily_rf
        excess_std = float(excess_returns.std())
        if excess_std > 0:
            indicators['sharpe'] = float((excess_returns.mean() * 252) / (excess_std * np.sqrt(252)))
        else:
            indicators['sharpe'] = 0.0
        indicators['risk_free_rate'] = float(risk_free_rate * 100)

        # Sortino: target semi-deviation (MAR = daily_rf), normalized by N total observations.
        # TSD = sqrt( mean( min(R - MAR, 0)^2 ) )  — the canonical formula.
        downside_dev = np.minimum(returns - daily_rf, 0.0)
        target_semi_dev = float(np.sqrt(np.mean(downside_dev ** 2)) * np.sqrt(252))
        if target_semi_dev > 0:
            indicators['sortino'] = float(((returns.mean() - daily_rf) * 252) / target_semi_dev)
        else:
            indicators['sortino'] = indicators['sharpe']

        # Beta vs SPY. Use consistent ddof and return NaN on failure so downstream
        # consumers can gate explicitly instead of seeing a silent default of 1.0.
        try:
            data_period = "1y" if len(data) < 252 else "2y" if len(data) < 504 else "5y"
            # Cached via the shared layer so SPY is downloaded at most once per
            # period/TTL instead of once per holding being analyzed.
            spy_data = market_data.get_history("SPY", period=data_period, auto_adjust=True)
            if spy_data is not None and not spy_data.empty and 'Close' in spy_data.columns:
                asset_ret_df = returns.to_frame(name='asset')
                spy_ret_df = spy_data['Close'].pct_change().dropna().to_frame(name='spy')
                aligned = asset_ret_df.join(spy_ret_df, how='inner').dropna()

                if len(aligned) > 10:
                    # Both with ddof=1 (consistent unbiased estimators)
                    covariance = float(aligned[['asset', 'spy']].cov().iloc[0, 1])
                    spy_variance = float(aligned['spy'].var(ddof=1))
                    indicators['beta'] = float(covariance / spy_variance) if spy_variance > 0 else float('nan')
                else:
                    indicators['beta'] = float('nan')
            else:
                indicators['beta'] = float('nan')
        except Exception as e:
            logger.debug(f"Failed to calculate beta: {e}")
            indicators['beta'] = float('nan')

        # Maximum Drawdown. Must avoid NaN propagation from the leading pct_change NaN
        # (which would otherwise turn the cumprod into all-NaN).
        cumulative = (1.0 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        if len(drawdown) > 0 and not drawdown.isna().all():
            indicators['max_drawdown'] = float(drawdown.min() * 100)
        else:
            indicators['max_drawdown'] = 0.0

        # Trend analysis: compare current SMA20 vs current SMA50 directly (point
        # comparison of two rolling means at the same date — the conventional
        # MA-crossover convention). The previous code compared .tail(20).mean()
        # to .tail(50).mean(), which are heavily overlapping samples and biased
        # toward calling NEUTRAL.
        if len(closes) >= 50:
            sma20_now = float(sma_20_series.iloc[-1])
            sma50_now = float(sma_50_series.iloc[-1])
            if sma20_now > sma50_now:
                indicators['trend'] = 'BULLISH'
            elif sma20_now < sma50_now:
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
            # News via the shared cached market layer (free, no API key needed).
            news = market_data.get_news(ticker)

            def _fields(article: Dict) -> Tuple[str, str, str]:
                """Title/source/published across yfinance schemas: old flat
                (title/publisher/providerPublishTime) and new nested content
                (content.title/content.provider.displayName/content.pubDate)."""
                content = article.get('content') if isinstance(article.get('content'), dict) else {}
                title = article.get('title') or content.get('title') or ''
                provider = content.get('provider') if isinstance(content.get('provider'), dict) else {}
                source = article.get('publisher') or provider.get('displayName') or 'Unknown'
                published = 'Unknown'
                if article.get('providerPublishTime'):
                    published = datetime.fromtimestamp(article['providerPublishTime']).isoformat()
                elif content.get('pubDate'):
                    published = str(content['pubDate'])
                return title, source, published

            if news:
                parsed = [_fields(a) for a in news]
                sentiment["articles_count"] = len(news)
                sentiment["recent_news"] = [
                    {"title": title, "source": source, "published": published}
                    for title, source, published in parsed[:5]
                ]

                # Simple sentiment analysis based on titles
                positive_words = ['gain', 'rise', 'up', 'growth', 'profit', 'beat', 'strong', 'bullish', 'surge']
                negative_words = ['fall', 'drop', 'down', 'loss', 'miss', 'weak', 'bearish', 'decline', 'crash']

                titles = [title.lower() for title, _, _ in parsed]
                positive_count = sum(1 for t in titles if any(word in t for word in positive_words))
                negative_count = sum(1 for t in titles if any(word in t for word in negative_words))
                
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
                    info = market_data.get_info(ticker)
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
        if data is not None and not data.empty:
            analysis["technical_indicators"] = self.calculate_technical_indicators(data)
        
        # Get news sentiment
        analysis["news_sentiment"] = self.get_news_sentiment(ticker)
        
        # === NORMALIZED SCORING (consistent with deposit_advisor) ===
        sub_scores = {}
        
        if analysis["technical_indicators"]:
            ti = analysis["technical_indicators"]
            
            # RSI sub-score: map oversold (buy signal) / overbought (sell signal)
            rsi = ti.get('rsi', 50)
            # For existing holdings: oversold = higher score (recovery potential)
            # overbought = lower score (profit-taking)
            rsi_norm = scoring.clamp01((100 - rsi) / 100.0)  # Invert: low RSI = high score
            sub_scores['rsi'] = rsi_norm

            # Momentum sub-score
            momentum = ti.get('momentum', 0)
            momentum_norm = scoring.momentum_score(momentum)
            sub_scores['momentum'] = momentum_norm

            # Sharpe sub-score
            sharpe = ti.get('sharpe', 0)
            sharpe_norm = scoring.sharpe_score(sharpe)
            sub_scores['sharpe'] = sharpe_norm

            # Sortino sub-score (if available)
            sortino = ti.get('sortino', sharpe)
            sortino_norm = scoring.sharpe_score(sortino)
            sub_scores['sortino'] = sortino_norm

            # Volatility sub-score (lower is better)
            vol = ti.get('volatility', 20)
            vol_norm = scoring.volatility_score(vol)
            sub_scores['volatility'] = vol_norm
            
            # Trend sub-score
            trend = ti.get('trend', 'NEUTRAL')
            trend_norm = 0.75 if trend == 'BULLISH' else (0.25 if trend == 'BEARISH' else 0.5)
            sub_scores['trend'] = trend_norm
        
        # News sentiment sub-score (normalized)
        news_sentiment = analysis["news_sentiment"]
        sentiment_score = news_sentiment.get("score", 50)
        sentiment_norm = scoring.clamp01(sentiment_score / 100.0)
        sub_scores['sentiment'] = sentiment_norm
        
        # Weighted combination
        # Risk-adjusted (Sharpe+Sortino): 30%, Momentum: 20%, Volatility: 15%,
        # Trend: 15%, Sentiment: 10%, RSI: 10%
        holding_weights = {
            'sharpe': 0.15, 'sortino': 0.15,
            'momentum': 0.20, 'volatility': 0.15,
            'trend': 0.15, 'sentiment': 0.10, 'rsi': 0.10
        }
        
        weighted = scoring.weighted_score(sub_scores, holding_weights)
        score = weighted * 80  # Map [0,1] -> [0,80], reserve 20 pts for refinements
        
        # === REFINEMENT FACTORS (up to ±20 points total) ===
        
        # Industry trend analysis (up to ±5 points)
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
                    
                    if industry_trend.get("trend") == "STRONG_UPTREND":
                        score += 5
                    elif industry_trend.get("trend") == "UPTREND":
                        score += 3
                    elif industry_trend.get("trend") == "DOWNTREND":
                        score -= 5
            except Exception as e:
                logger.debug(f"Failed to analyze industry trend: {e}")
                pass  # Industry trend analysis is optional
        
        # Statistical forecast (up to ±5 points)
        if analysis["technical_indicators"] and "forecast" in analysis["technical_indicators"]:
            forecast = analysis["technical_indicators"]["forecast"]
            if forecast and forecast.get("expected_return_polynomial") is not None:
                expected_return = forecast.get("expected_return_polynomial", 0)
                analysis["mid_term_forecast"] = {
                    "expected_3yr_return": expected_return,
                    "forecast_price": forecast.get("forecast_polynomial", 0)
                }
                
                if expected_return > 15:
                    score += 5
                elif expected_return > 10:
                    score += 3
                elif expected_return < -10:
                    score -= 5
        
        # Candlestick patterns (up to ±2 points)
        if analysis["technical_indicators"] and "candlestick_patterns" in analysis["technical_indicators"]:
            patterns = analysis["technical_indicators"]["candlestick_patterns"]
            if patterns:
                bullish = [p for p in patterns if p.get('signal') == 'BULLISH']
                bearish = [p for p in patterns if p.get('signal') == 'BEARISH']
                if bullish:
                    score += 2
                elif bearish:
                    score -= 2
        
        # Bond analysis (up to ±5 points, if applicable)
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
                        score += 5
                    elif risk_adj_yield > 1:
                        score += 3
                    elif risk_adj_yield < 0.5:
                        score -= 5
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
        scored = [a["recommendation_score"] for a in analyses
                  if a.get("recommendation_score") is not None]
        avg_score = np.mean(scored) if scored else None  # None in fast mode
        
        # Calculate cumulative return since start (baseline tracking)
        baseline_value = portfolio.get("baseline_value")
        start_date = portfolio.get("start_date")
        cumulative_return = None
        cumulative_return_pct = None
        days_since_start = None
        hours_since_start = None
        minutes_since_start = None
        
        if baseline_value and baseline_value > 0:
            cumulative_return = total_value - baseline_value
            cumulative_return_pct = (cumulative_return / baseline_value) * 100
            
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    time_diff = datetime.now(timezone.utc) - start_dt
                    days_since_start = time_diff.days
                    # If same day, calculate hours/minutes
                    if days_since_start == 0:
                        hours_since_start = int(time_diff.total_seconds() / 3600)
                        if hours_since_start < 1:
                            minutes_since_start = int(time_diff.total_seconds() / 60)
                except Exception as e:
                    logger.debug(f"Failed to parse start_date: {e}")
        
        # True (deposit-adjusted) performance from the ledger. The baseline
        # "cumulative return" above counts deposits as gains; this one doesn't.
        true_performance = ledger.performance(portfolio, total_value)

        return {
            "total_value": total_value,
            "cash": portfolio.get("cash", 0),
            "holdings_value": holdings_value,
            "diversification_score": diversification_score,
            "average_recommendation_score": avg_score,
            "weights": weights,
            "baseline_value": baseline_value,
            "cumulative_return": cumulative_return,
            "cumulative_return_pct": cumulative_return_pct,
            "days_since_start": days_since_start,
            "hours_since_start": hours_since_start,
            "minutes_since_start": minutes_since_start,
            "start_date": start_date,
            "true_performance": true_performance,
        }
    
    def check_80_20_balance(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """
        Check the portfolio against the model strategy defined in allocation.py
        (single source of truth for both this report and `make deposit`).
        Classification: allocation groups (so VXUS/VEA/IEFA/VWO/EEM are all CORE);
        targets: allocation.category_targets() (currently 85/15 - 70 core /
        15 satellite / 15 bonds). Returns recommendations to achieve balance.
        """
        cat_targets = allocation.category_targets()
        target_core = cat_targets.get("CORE", 0) * 100
        target_satellite = cat_targets.get("SATELLITE", 0) * 100
        target_bonds = cat_targets.get("BONDS", 0) * 100
        target_stocks = target_core + target_satellite

        balance_check = {
            "is_balanced": False,
            "stocks_percent": 0,
            "bonds_percent": 0,
            "core_percent": 0,
            "satellite_percent": 0,
            "targets": {
                "stocks": target_stocks, "bonds": target_bonds,
                "core": target_core, "satellite": target_satellite,
            },
            "recommendations": []
        }

        total_value = portfolio_metrics["total_value"]
        if total_value == 0:
            return balance_check

        # Calculate current allocation via the shared allocation groups.
        # Unclassified tickers count as satellite (equity risk, no target).
        stocks_value = 0
        bonds_value = 0
        core_value = 0
        satellite_value = 0

        for analysis in analyses:
            ticker = analysis["ticker"]
            value = analysis["current_value"]
            group_key = allocation.classify(ticker)
            category = allocation.GROUP_BY_KEY[group_key]["category"] if group_key else "SATELLITE"

            if category == "BONDS":
                bonds_value += value
            else:
                stocks_value += value
                if category == "CORE":
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
                    "reason": f"Portfolio has {bonds_percent:.1f}% bonds, target is {target_bonds:.0f}%. Need ${needed_bonds:,.2f} in bonds for protection."
                })
            
            # Check if need more core stocks
            if core_percent < target_core - tolerance:
                needed_core = total_value * (target_core - core_percent) / 100
                recommendations.append({
                    "action": "BUY_CORE",
                    "amount": needed_core,
                    "reason": f"Portfolio has {core_percent:.1f}% core stocks, target is {target_core:.0f}%. Need ${needed_core:,.2f} in core ETFs (SPY, VXUS)."
                })
            
            # Check if too many bonds
            if bonds_percent > target_bonds + tolerance:
                excess_bonds = total_value * (bonds_percent - target_bonds) / 100
                recommendations.append({
                    "action": "REDUCE_BONDS",
                    "amount": excess_bonds,
                    "reason": f"Portfolio has {bonds_percent:.1f}% bonds, target is {target_bonds:.0f}%. Consider reducing by ${excess_bonds:,.2f} to increase growth."
                })
            
            # Check if too many stocks
            if stocks_percent > target_stocks + tolerance:
                excess_stocks = total_value * (stocks_percent - target_stocks) / 100
                recommendations.append({
                    "action": "REDUCE_STOCKS",
                    "amount": excess_stocks,
                    "reason": f"Portfolio has {stocks_percent:.1f}% stocks, target is {target_stocks:.0f}%. Consider reducing by ${excess_stocks:,.2f} and adding bonds for protection."
                })
            
            balance_check["recommendations"] = recommendations
        
        return balance_check
    
    def _build_rebalance_plan(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """Two concrete ways to get back to target, for the report:

        Option A (deposits only, no tax): how many months of deposits at the
        recent pace close the underweight gap.
        Option B (trade now): allocation.rebalance_plan sells overweight groups
        down to target (dust first) and reallocates proceeds - each sell
        annotated with an estimated capital-gains tax where cost basis is known.
        """
        portfolio = self.load_portfolio()
        holdings = portfolio.get("holdings", [])
        cash = float(portfolio.get("cash", 0) or 0)
        prices = {a["ticker"].upper(): float(a.get("current_price", 0) or 0)
                  for a in analyses}
        # Every group member needs a price so buys can pick equivalents.
        needed = {t.upper() for g in allocation.TARGET_GROUPS for t in g["tickers"]}
        missing = [t for t in needed if prices.get(t, 0) <= 0]
        if missing:
            fetched, _, _ = market_data.get_prices(missing)
            prices.update({k.upper(): v for k, v in (fetched or {}).items()})

        plan = allocation.rebalance_plan(holdings, cash, prices)

        # Tax annotation for Option B sells (None = unknown, no cost basis).
        cost_basis = {(h.get("ticker") or "").upper():
                      (h.get("cost_basis") or h.get("purchase_price"))
                      for h in holdings}
        ta = TaxAnalyzer(self.portfolio_file)
        total_tax = 0.0
        tax_known = True
        for s in plan["sells"]:
            cb = cost_basis.get(s["ticker"])
            tax = ta.estimate_sale_tax_usd(cb, s["price"], s["shares"]) if cb else None
            s["est_tax_usd"] = tax
            if tax is None:
                tax_known = False
            else:
                total_tax += tax
        plan["est_total_tax_usd"] = round(total_tax, 2)
        plan["tax_fully_known"] = tax_known

        # Option A: months to close the gap at the recent deposit pace.
        monthly = ledger.average_monthly_deposit(portfolio)
        plan["monthly_deposit_usd"] = monthly
        plan["months_to_target"] = (
            round(plan["underweight_usd"] / monthly, 1)
            if monthly and monthly > 0 else None
        )
        return plan

    def check_rebalancing(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """Determine if rebalancing is needed, including 80/20 balance check."""
        rebalancing = {
            "needed": False,
            "reason": "",
            "recommendations": [],
            "buy_recommendations": [],
            "balance_80_20": {}
        }
        
        # First check 80/20 balance
        balance_check = self.check_80_20_balance(portfolio_metrics, analyses)
        rebalancing["balance_80_20"] = balance_check
        
        if not balance_check["is_balanced"]:
            rebalancing["needed"] = True
            balance_reasons = [r["reason"] for r in balance_check["recommendations"]]
            if not rebalancing["reason"]:
                rebalancing["reason"] = f"Portfolio off target: {balance_reasons[0] if balance_reasons else 'Needs rebalancing'}"
            rebalancing["plan"] = self._build_rebalance_plan(portfolio_metrics, analyses)

        weights = portfolio_metrics["weights"]
        total_holdings = len(weights)
        total_value_usd = portfolio_metrics["holdings_value"]
        current_tickers = [a["ticker"] for a in analyses]
        total_sell_amount = 0
        
        # Over-concentration is a pure RISK rule, not a performance call:
        # >50% in one holding always triggers a reduce; 40-50% is only noted.
        # (Score-based carve-outs were removed - a good recent score is not a
        # reason to hold half the portfolio in one ticker, and a bad one is
        # not a reason to churn.)
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.4:  # More than 40% in one holding
            ticker = max(weights, key=weights.get)
            ticker_analysis = next((a for a in analyses if a["ticker"] == ticker), None)

            should_reduce = max_weight > 0.5
            reason_text = f"Very high concentration: {ticker} is {max_weight*100:.1f}% of portfolio (risk limit)"
            if not should_reduce and not rebalancing["reason"]:
                rebalancing["reason"] = (f"Note: {ticker} is {max_weight*100:.1f}% of portfolio - "
                                         f"watch concentration (reduce trigger is 50%)")

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
        
        # NOTE (deliberate design): score-based "underperformer" sells, the
        # better-alternative swap scans, and legacy sell-stocks-to-buy-bonds
        # logic were REMOVED. Monthly technical scores on a multi-decade
        # passive strategy generate taxable churn, not returns. Sells happen
        # for exactly two reasons: rebalancing back to target (the plan above)
        # and genuine risk events (stop-loss, extreme concentration).

        # If the only issue is target drift (no performance/concentration sells),
        # expose the trade-now plan (Option B) through the standard confirm-apply
        # flow so answering "yes, I executed these" updates the portfolio.
        plan = rebalancing.get("plan")
        if plan and not rebalancing["recommendations"] and not rebalancing["buy_recommendations"]:
            for s in plan["sells"]:
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": s["ticker"],
                    "sell_shares": s["shares"],
                    "sell_amount_usd": s["amount"],
                    "current_price_usd": s["price"],
                    "reason": f"Rebalance: {s['group']} over target",
                })
            for b in plan["buys"]:
                rebalancing["buy_recommendations"].append({
                    "ticker": b["ticker"],
                    "shares": b["shares"],
                    "price": b["price"],
                    "allocation_amount": b["amount"],
                    "name": b["ticker"],
                    "reasons": [f"Rebalance: fill {b['group']} toward target"],
                })

        return rebalancing
    
    def analyze(self, record_run: bool = True) -> Dict:
        """Main analysis function.

        ``record_run`` controls whether ``last_analyze_run_date`` is stamped at the
        END of the run. It used to be stamped here at the start, which made the
        30-day rebalancing cooldown see "today" on every run and therefore be
        permanently active (check_rebalancing reloads the portfolio from disk).
        Automated callers (critical_alert / CI) pass record_run=False so the
        30-day "run make analyze" reminder still reflects conscious user runs.
        """
        print("=" * 60)
        print("Portfolio Analysis Starting...")
        print("=" * 60)
        # (The emerging-trends scan was removed from analyze: trend information
        # only influences the bounded satellite tilt inside `make deposit`.)

        # Load portfolio
        portfolio = self.load_portfolio()
        
        if not portfolio.get("holdings"):
            print("No holdings found in portfolio. Please add holdings first.")
            return {}
        
        # Get current prices and market status
        tickers = [h["ticker"] for h in portfolio["holdings"]]
        prices, market_status, market_message = self.get_current_prices(tickers)
        
        # Initialize baseline tracking if not exists (first time using the app)
        if "baseline_value" not in portfolio or portfolio.get("baseline_value") is None:
            # Calculate current total value as baseline
            baseline_value = portfolio.get("cash", 0)
            for h in portfolio["holdings"]:
                price = prices.get(h["ticker"], h.get("last_price", 0))
                baseline_value += h["quantity"] * price

            portfolio["baseline_value"] = baseline_value
            portfolio["start_date"] = datetime.now(timezone.utc).isoformat()
            # Intermediate save - the single GitHub-secret sync happens once
            # at the end of analyze() (each sync is a subprocess call).
            self.save_portfolio(portfolio, sync_github_secret=False)
            logger.info(f"📊 Initialized baseline tracking: ${baseline_value:,.2f} on {portfolio['start_date']}")

        # Seed the deposit/trade ledger if missing so true (deposit-adjusted)
        # returns can be tracked from now on.
        current_total = portfolio.get("cash", 0) + sum(
            h["quantity"] * prices.get(h["ticker"], h.get("last_price", 0))
            for h in portfolio["holdings"]
        )
        if ledger.ensure_ledger(portfolio, current_total):
            self.save_portfolio(portfolio, sync_github_secret=False)
        
        # Display market status
        print(f"\n📊 Market Status: {market_message}")
        if market_status:
            print("   ⚡ Using REAL-TIME prices")
        else:
            print("   📅 Using LAST CLOSE prices")
        
        # Price caching is handled by the shared market_data layer.
        print(f"   💾 Prices served via shared cache (window: {'60 min' if market_status else '4 hr'})")
        print()
        
        analyses = []
        holdings_to_analyze = [
            (h["ticker"], h["quantity"], prices.get(h["ticker"], h.get("last_price", 0)))
            for h in portfolio["holdings"]
            if prices.get(h["ticker"], h.get("last_price", 0)) > 0
        ]

        # FAST by default: allocation, drift plan, and true returns only need
        # prices and values. Per-holding technical scores (1y history + news
        # per ticker - the slow part) are informational-only now that trades
        # never come from scores, so they're computed only on request.
        verbose_holdings = os.environ.get("VERBOSE_HOLDINGS", "").strip().lower() in ("1", "true", "yes")

        if holdings_to_analyze and not verbose_holdings:
            for ticker, quantity, price in holdings_to_analyze:
                analyses.append({
                    "ticker": ticker,
                    "quantity": quantity,
                    "current_price": price,
                    "current_value": quantity * price,
                    "technical_indicators": {},
                    "recommendation": "HOLD",
                    "recommendation_score": None,  # not computed in fast mode
                })
        elif holdings_to_analyze:
            # Full mode (VERBOSE_HOLDINGS=1): compute indicators/scores too.
            try:
                from deposit_advisor import DepositAdvisor
                advisor = DepositAdvisor(self.portfolio_file)
            except Exception as e:
                logger.debug(f"Could not create DepositAdvisor: {e}")
                advisor = None
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
        
        # Check rebalancing (drift plan + risk guards; trade recommendations
        # come solely from the rebalance plan wired inside check_rebalancing)
        rebalancing = self.check_rebalancing(portfolio_metrics, analyses)

        # Update portfolio with current values.
        # analyses is reordered by the ThreadPool (as_completed) and filtered
        # (zero-price holdings dropped), so we must match by ticker - a positional
        # analyses[i] -> holdings[i] write would assign prices to the wrong holding.
        portfolio["total_value"] = portfolio_metrics["total_value"]
        analysis_by_ticker = {a["ticker"]: a for a in analyses}
        for holding in portfolio["holdings"]:
            analysis = analysis_by_ticker.get(holding["ticker"])
            if analysis:
                holding["last_price"] = analysis["current_price"]
                holding["current_value"] = analysis["current_value"]
        
        # Save updated portfolio (sync deferred to the single end-of-run sync)
        self.save_portfolio(portfolio, sync_github_secret=False)

        # (The "better ETF alternatives" and hot-sector replacement scans were
        # removed deliberately: swapping funds on score differences generates
        # taxable churn, not returns, for a passive target-weight strategy.)

        # Compile results
        results = {
            "portfolio_metrics": portfolio_metrics,
            "holdings_analysis": analyses,
            "rebalancing": rebalancing,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print results
        self.print_analysis_results(results)

        # Read-only mode: never ask to update portfolio (CI, make analyze-preview, or no TTY)
        read_only = (
            os.environ.get("ANALYZE_READONLY", "").strip().lower() in ("1", "true", "yes")
            or not sys.stdin.isatty()
        )

        if read_only:
            print("\n📋 Read-only mode: portfolio was not modified. Run 'make analyze' (without read-only) to update after you execute trades.\n")
            return results

        # Ask for confirmation if rebalancing is needed
        if rebalancing["needed"] and (rebalancing["recommendations"] or rebalancing["buy_recommendations"]):
            confirmed = self.ask_rebalancing_confirmation()
            if confirmed:
                self.update_portfolio_from_rebalancing(portfolio, rebalancing, analyses)
                print("\n✅ Portfolio updated successfully based on rebalancing actions!\n")
            else:
                print("\n❌ Portfolio not updated. No changes were made.\n")

        # Stamp the run date now that the analysis (and its cooldown checks,
        # which read the PREVIOUS stamp) has completed, then do the SINGLE
        # save+GitHub-secret sync for the whole interactive run. Read-only/CI
        # runs return above without stamping or syncing.
        if record_run:
            portfolio["last_analyze_run_date"] = datetime.now().strftime("%Y-%m-%d")
        self.save_portfolio(portfolio, sync_github_secret=True)

        return results
    
    def ask_rebalancing_confirmation(self) -> bool:
        """Ask user for confirmation: did you already execute these exact trades in your broker?"""
        print("\n⚠️  Answer YES only if you already did these exact sell/buy trades in your broker.")
        print("   Answer NO to keep portfolio.json unchanged (recommendations are for reference only).")
        while True:
            response = input("\nDid you execute these exact rebalancing trades in your broker? (yes/no): ").strip().lower()
            if response in ['yes', 'y', 'כן', 'י']:
                return True
            elif response in ['no', 'n', 'לא', 'ל']:
                return False
            else:
                print("Please enter 'yes' or 'no' (כן/לא)")
    
    def _apply_buy(self, portfolio: Dict, ticker: str, shares: int, price: float):
        """Add ``shares`` of ``ticker`` at ``price`` to the portfolio, merging into
        an existing holding (weighted-average cost basis) or creating a new one."""
        if shares <= 0 or price <= 0:
            return
        existing_holding = None
        for holding in portfolio.get("holdings", []):
            if holding["ticker"] == ticker:
                existing_holding = holding
                break

        if existing_holding:
            old_qty = existing_holding.get("quantity", 0)
            old_cb = (existing_holding.get("cost_basis")
                      or existing_holding.get("purchase_price")
                      or existing_holding.get("last_price", price))
            new_qty = old_qty + shares
            if new_qty > 0:
                existing_holding["cost_basis"] = round(
                    (old_qty * old_cb + shares * price) / new_qty, 4
                )
            existing_holding["quantity"] = new_qty
            existing_holding["last_price"] = price
            existing_holding["current_value"] = new_qty * price
        else:
            portfolio.setdefault("holdings", []).append({
                "ticker": ticker,
                "quantity": shares,
                "cost_basis": round(price, 4),
                "last_price": price,
                "current_value": shares * price,
            })
        ledger.record_trade(portfolio, "buy", ticker, shares, price)

    def _apply_sell(self, portfolio: Dict, ticker: str, shares: int, price: float = 0) -> Tuple[float, int]:
        """Reduce/remove ``shares`` of ``ticker`` from the portfolio. ``price`` (if
        > 0) updates last_price on the remaining position. Returns
        ``(used_price, shares_actually_sold)`` — the sold count is clamped to the
        held quantity so callers never credit cash for shares that don't exist."""
        for holding in portfolio.get("holdings", []):
            if holding["ticker"] == ticker:
                current_quantity = holding.get("quantity", 0)
                sold = min(max(shares, 0), current_quantity)
                new_quantity = current_quantity - sold
                used_price = price if price > 0 else holding.get("last_price", 0)
                if new_quantity > 0:
                    holding["quantity"] = new_quantity
                    holding["last_price"] = used_price
                    holding["current_value"] = new_quantity * used_price
                else:
                    portfolio["holdings"].remove(holding)
                if sold > 0 and used_price > 0:
                    ledger.record_trade(portfolio, "sell", ticker, sold, used_price)
                return used_price, sold
        return price, 0

    def update_portfolio_from_rebalancing(self, portfolio: Dict, rebalancing: Dict, analyses: List[Dict]):
        """Update portfolio.json based on rebalancing recommendations."""
        exchange_rate = self.get_exchange_rate()

        # Process SELL actions
        for rec in rebalancing.get("recommendations", []):
            if rec["action"] == "SELL":
                ticker = rec["ticker"]
                shares_to_sell = rec.get("reduce_shares", rec.get("sell_shares", 0))
                used_price, sold = self._apply_sell(portfolio, ticker, shares_to_sell, rec.get("current_price_usd", 0))
                # Add proceeds to cash (only for shares actually held/sold)
                portfolio["cash"] = portfolio.get("cash", 0) + sold * used_price

        # Process BUY actions
        for rec in rebalancing.get("buy_recommendations", []):
            ticker = rec["ticker"]
            shares_to_buy = rec.get("shares", 0)
            price = rec.get("price", 0)
            buy_amount = rec.get("allocation_amount", shares_to_buy * price)

            if shares_to_buy > 0 and price > 0:
                self._apply_buy(portfolio, ticker, shares_to_buy, price)
                # Subtract from cash
                portfolio["cash"] = max(0, portfolio.get("cash", 0) - buy_amount)

        # Reconcile with the broker's real remaining cash - actual fills
        # differ from scan prices, so tracked cash drifts unless corrected.
        actual = ledger.ask_actual_cash(portfolio.get("cash", 0), exchange_rate)
        if actual is not None:
            delta = ledger.reconcile_cash(portfolio, actual,
                                          note="post-rebalancing broker reconciliation")
            if delta:
                print(f"   🔧 Cash reconciled to ${portfolio['cash']:,.2f} "
                      f"(execution drift {delta:+,.2f} USD logged)")

        # Recalculate total value
        total_value = portfolio.get("cash", 0)
        for holding in portfolio.get("holdings", []):
            total_value += holding.get("current_value", 0)
        portfolio["total_value"] = total_value

        # Record rebalancing date so we don't recommend again too soon (cooldown)
        portfolio["last_rebalancing_date"] = datetime.now().strftime("%Y-%m-%d")

        # Save updated portfolio
        self.save_portfolio(portfolio, sync_github_secret=False)  # analyze() syncs once at end
        
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
        print(f"Exchange Rate: 1 USD = {exchange_rate:.4f} ILS  (1 ILS = {1/exchange_rate:.4f} USD)")
        print(f"Diversification Score: {metrics['diversification_score']:.2f} (1.0 = perfect diversification)")
        if metrics.get('average_recommendation_score') is not None:
            print(f"Average Recommendation Score: {metrics['average_recommendation_score']:.1f}/100")
        
        # Display cumulative return since start
        if metrics.get("baseline_value") and metrics.get("baseline_value") > 0:
            baseline = metrics["baseline_value"]
            cumulative_return = metrics.get("cumulative_return")
            cumulative_return_pct = metrics.get("cumulative_return_pct")
            days_since_start = metrics.get("days_since_start")
            start_date = metrics.get("start_date")
            
            if cumulative_return is not None and cumulative_return_pct is not None:
                baseline_ils = baseline * exchange_rate
                cumulative_return_ils = cumulative_return * exchange_rate
                
                # Format start date nicely
                start_date_str = "N/A"
                if start_date:
                    try:
                        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                        start_date_str = start_dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                
                print("\n" + "-" * 60)
                print("📈 VALUE GROWTH SINCE START (includes your deposits - NOT investment return)")
                print("-" * 60)
                print(f"Baseline Value: ₪{baseline_ils:,.2f} (${baseline:,.2f})")
                print(f"Start Date: {start_date_str}")
                
                # Format period - show minutes/hours if same day, otherwise days/months/years
                hours_since_start = metrics.get("hours_since_start")
                minutes_since_start = metrics.get("minutes_since_start")
                
                if days_since_start is not None:
                    months = days_since_start / 30.44
                    if days_since_start < 30:
                        period_str = f"{days_since_start} days"
                    elif days_since_start < 365:
                        period_str = f"{months:.1f} months ({days_since_start} days)"
                    else:
                        years = days_since_start / 365.25
                        period_str = f"{years:.1f} years ({days_since_start} days)"
                    print(f"Period: {period_str}")
                elif hours_since_start is not None and hours_since_start > 0:
                    print(f"Period: {hours_since_start} hour{'s' if hours_since_start != 1 else ''}")
                elif minutes_since_start is not None and minutes_since_start > 0:
                    print(f"Period: {minutes_since_start} minute{'s' if minutes_since_start != 1 else ''}")
                else:
                    print(f"Period: Just started")
                
                # Color code the return
                if cumulative_return_pct >= 0:
                    return_icon = "📈"
                    return_color = ""
                else:
                    return_icon = "📉"
                    return_color = ""
                
                print(f"{return_icon} Value Growth: {return_color}₪{cumulative_return_ils:+,.2f} (${cumulative_return:+,.2f})")
                print(f"{return_icon} Growth Percentage: {return_color}{cumulative_return_pct:+.2f}% (deposits inflate this number)")

        # True performance: gains net of deposits, from the transaction ledger.
        perf = metrics.get("true_performance")
        if perf:
            gain_ils = perf["gain_usd"] * exchange_rate
            invested_ils = perf["net_invested_usd"] * exchange_rate
            icon = "📈" if perf["gain_usd"] >= 0 else "📉"
            print("\n" + "-" * 60)
            print("💵 TRUE INVESTMENT PERFORMANCE (net of deposits)")
            print("-" * 60)
            print(f"Tracking Since: {perf['ledger_start_date']} ({perf['days']} days)")
            print(f"Money Put In: ₪{invested_ils:,.2f} (${perf['net_invested_usd']:,.2f})")
            print(f"{icon} Investment Gain: ₪{gain_ils:+,.2f} (${perf['gain_usd']:+,.2f})  [{perf['gain_pct']:+.2f}%]")
            if perf.get("xirr_pct") is not None:
                print(f"📊 Money-Weighted Annual Return (XIRR): {perf['xirr_pct']:+.2f}%")
            else:
                print("📊 Money-Weighted Annual Return (XIRR): available after 30 days of tracking")
        
        # Show strategy balance status (targets come from allocation.py -
        # the same source `make deposit` uses).
        balance_info = results["rebalancing"].get("balance_80_20", {})
        if balance_info:
            tgts = balance_info.get("targets", {})
            t_stocks = tgts.get("stocks", 85)
            t_core = tgts.get("core", 70)
            t_sat = tgts.get("satellite", 15)
            t_bonds = tgts.get("bonds", 15)
            print("\n" + "-" * 60)
            print(f"STRATEGY STATUS ({allocation.strategy_summary()})")
            print("-" * 60)
            stocks_pct = balance_info.get("stocks_percent", 0)
            bonds_pct = balance_info.get("bonds_percent", 0)
            core_pct = balance_info.get("core_percent", 0)
            satellite_pct = balance_info.get("satellite_percent", 0)
            is_balanced = balance_info.get("is_balanced", False)

            status_icon = "✅" if is_balanced else "⚠️"
            print(f"{status_icon} Stocks: {stocks_pct:.1f}% (Target: {t_stocks:.0f}%)")
            print(f"   ├─ Core: {core_pct:.1f}% (Target: {t_core:.0f}%)")
            print(f"   └─ Satellite: {satellite_pct:.1f}% (Target: {t_sat:.0f}%)")
            print(f"{status_icon} Bonds: {bonds_pct:.1f}% (Target: {t_bonds:.0f}%)")

            if not is_balanced:
                for rec in balance_info.get("recommendations", []):
                    print(f"   • {rec.get('reason', 'N/A')}")

                plan = results["rebalancing"].get("plan")
                if plan:
                    print("\n🧭 REBALANCING PLAN - two ways to get back to target:")

                    # Option A: keep depositing (no sales, no tax).
                    monthly = plan.get("monthly_deposit_usd")
                    months = plan.get("months_to_target")
                    print("\n   Option A - deposits only (no selling, no tax):")
                    if months is not None:
                        print(f"      Keep depositing ~${monthly:,.0f}/month (your recent pace);")
                        print(f"      the ${plan['underweight_usd']:,.0f} underweight gap closes in ~{months:.0f} months.")
                    else:
                        print(f"      Future deposits will close the ${plan['underweight_usd']:,.0f} gap"
                              f" (no deposit history yet to estimate how fast).")

                    # Option B: trade now.
                    print("\n   Option B - trade now (sell overweight -> buy underweight):")
                    if not plan["sells"]:
                        print("      No sensible sells found - use Option A.")
                    else:
                        for s in plan["sells"]:
                            tax = s.get("est_tax_usd")
                            tax_note = (f", est. tax ${tax:,.0f}" if tax
                                        else (", no tax (no gain)" if tax == 0.0
                                              else ", tax unknown (no cost basis)"))
                            print(f"      🔴 SELL {s['shares']} x {s['ticker']:<5} = ${s['amount']:>9,.2f}"
                                  f"  [{s['group']}{tax_note}]")
                        for b in plan["buys"]:
                            print(f"      🟢 BUY  {b['shares']} x {b['ticker']:<5} = ${b['amount']:>9,.2f}"
                                  f"  [{b['group']}]")
                        tax_line = f"${plan['est_total_tax_usd']:,.0f}"
                        if not plan.get("tax_fully_known"):
                            tax_line += " + unknown (some positions lack cost basis - run 'make backfill-cost-basis')"
                        print(f"      Est. capital-gains tax: {tax_line}")
                        print("      💡 Option A avoids all tax; Option B fixes the drift today.")
        
        print("\n" + "-" * 76)
        print("HOLDINGS (USD, sorted by value - set VERBOSE_HOLDINGS=1 for full details)")
        print("-" * 76)
        # No BUY/SELL signal column - per-holding technical scores are
        # informational only; trades come from the rebalance plan and risk
        # events, never from monthly score wiggles.
        print(f"{'TICKER':<7}{'GROUP':<16}{'QTY':>5}{'PRICE':>10}{'VALUE':>12}{'WT%':>6}{'SCORE':>7}")
        verbose_holdings = os.environ.get("VERBOSE_HOLDINGS", "").strip().lower() in ("1", "true", "yes")
        sorted_holdings = sorted(results["holdings_analysis"],
                                 key=lambda a: -a.get("current_value", 0))
        for analysis in sorted_holdings:
            weight = results['portfolio_metrics']['weights'].get(analysis['ticker'], 0) * 100
            group = allocation.classify(analysis['ticker']) or "-"
            score = analysis.get('recommendation_score')
            score_str = f"{score:>7.1f}" if score is not None else f"{'-':>7}"
            print(f"{analysis['ticker']:<7}{group:<16}{analysis['quantity']:>5}"
                  f"{analysis['current_price']:>10,.2f}{analysis['current_value']:>12,.2f}"
                  f"{weight:>6.1f}{score_str}")
            if verbose_holdings and analysis["technical_indicators"]:
                ti = analysis["technical_indicators"]
                beta_val = ti.get('beta')
                beta_str = "N/A" if beta_val is None or (isinstance(beta_val, float) and np.isnan(beta_val)) else f"{beta_val:.2f}"
                print(f"       RSI {ti.get('rsi', 0):.0f} | momentum {ti.get('momentum', 0):+.1f}% | "
                      f"vol {ti.get('volatility', 0):.1f}% | {ti.get('trend', 'NEUTRAL')} | "
                      f"beta {beta_str} | maxDD {ti.get('max_drawdown', 0):.1f}%")
        
        print("\n" + "=" * 60)
        print("REBALANCING SUMMARY")
        print("=" * 60)
        
        rebalancing = results["rebalancing"]
        # Reuse the exchange_rate fetched at the top of this method.

        if rebalancing["needed"]:
            print("⚠️  REBALANCING IS RECOMMENDED")
            print(f"Reason: {rebalancing['reason']}")

            # Actions only - everything not listed here is a HOLD.
            total_sell = 0.0
            if rebalancing["recommendations"]:
                print("\n🔴 SELL:")
                for rec in rebalancing["recommendations"]:
                    if 'reduce_amount_usd' in rec:
                        amount, shares, price = rec['reduce_amount_usd'], rec['reduce_shares'], rec['current_price_usd']
                    elif 'sell_amount_usd' in rec:
                        amount, shares, price = rec['sell_amount_usd'], rec['sell_shares'], rec['current_price_usd']
                    else:
                        continue
                    total_sell += amount
                    print(f"   • {rec['ticker']:<6} {shares} x ${price:,.2f} = ${amount:,.2f} (₪{amount * exchange_rate:,.2f})")
                if total_sell > 0:
                    print(f"   Total to sell: ${total_sell:,.2f} (₪{total_sell * exchange_rate:,.2f})")

            total_buy = 0.0
            if rebalancing["buy_recommendations"]:
                print("\n🟢 BUY:")
                for rec in rebalancing["buy_recommendations"]:
                    amount = rec.get('allocation_amount', 0)
                    shares = rec.get('shares', 0)
                    price = rec.get('price', 0)
                    if amount > 0 and shares > 0:
                        total_buy += amount
                        print(f"   • {rec['ticker']:<6} {shares} x ${price:,.2f} = ${amount:,.2f} (₪{amount * exchange_rate:,.2f})")
                        if rec.get('reasons'):
                            print(f"     {rec['reasons'][0]}")
                if total_buy > 0:
                    print(f"   Total to buy: ${total_buy:,.2f} (₪{total_buy * exchange_rate:,.2f})")

            print("\n   (all other holdings: HOLD)")
        else:
            print("✅ Portfolio is well-balanced. No rebalancing needed at this time.")
        

        print("\n" + "=" * 60)
        print(f"Analysis completed at: {results['timestamp']}")
        print("=" * 60 + "\n")
        # (No update-secret reminder: interactive analyze runs sync the GitHub
        # secret automatically at the end of the run.)

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer()
    analyzer.analyze()

