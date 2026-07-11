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

    def _run_update_secret(self) -> None:
        """Run make update-secret so GitHub secret is updated (runs every time user confirms 'yes')."""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                [sys.executable, os.path.join(script_dir, "update_github_secret.py")],
                cwd=script_dir,
                timeout=15,
                capture_output=False,
            )
            if result.returncode == 0:
                print("   ✅ GitHub secret updated automatically!")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.debug(f"update-secret run failed: {e}")
            print("   ⚠️  Run 'make update-secret' manually to sync the GitHub secret.")
    
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
        avg_score = np.mean([a["recommendation_score"] for a in analyses]) if analyses else 50
        
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
    
    def find_best_etfs_to_buy(self, amount_usd: float, current_holdings: List[str], exclude_tickers: List[str] = None) -> List[Dict]:
        """
        Find best ETFs to buy following 80/20 Balanced Growth Strategy.
        Prioritizes Core ETFs and Bonds over high-risk trends.
        """
        from deposit_advisor import DepositAdvisor
        
        exclude_tickers = exclude_tickers or []
        advisor = DepositAdvisor(self.portfolio_file)

        core_etfs = self.CORE_ETFS
        # Expanded satellite universe (broader than SATELLITE_ETFS) for candidate search.
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
        bond_etfs = self.BOND_ETFS

        # Exclude high-risk categories
        excluded_categories = etf_universe.EXCLUDED_CATEGORIES
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
        
        core_upper = {e.upper() for e in core_etfs}
        bond_upper = {e.upper() for e in bond_etfs}

        # Limit to top candidates (prioritize Core and Bonds), then fetch all
        # their prices in ONE batched request via the shared layer.
        candidates_to_analyze = candidate_etfs[:10]
        candidate_prices, _, _ = market_data.get_prices(candidates_to_analyze)

        def analyze_etf_candidate(etf):
            try:
                price = candidate_prices.get(etf)
                if not price or price <= 0:
                    return None
                shares = int(amount_usd / price / 3)  # Divide by 3 for 3 recommendations
                if shares > 0:
                    # Scoring based on category
                    score = 60  # Base score
                    info = market_data.get_info(etf)

                    # Boost for Core and Bonds
                    if etf.upper() in core_upper:
                        score += 20
                        category = "CORE"
                    elif etf.upper() in bond_upper:
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
        with ThreadPoolExecutor(max_workers=min(5, len(candidates_to_analyze))) as executor:
            futures = [executor.submit(analyze_etf_candidate, etf) for etf in candidates_to_analyze]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    recommendations.append(result)
        
        # Sort by score (Core and Bonds will be prioritized)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:3]
    
    def check_80_20_balance(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """
        Check if portfolio follows 80/20 Balanced Growth Strategy:
        - 80% Stocks (50% Core + 30% Satellite)
        - 20% Bonds
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
        
        core_upper = {e.upper() for e in self.CORE_ETFS}
        bond_upper = {e.upper() for e in self.BOND_ETFS}

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

            if ticker.upper() in bond_upper:
                bonds_value += value
            else:
                stocks_value += value
                if ticker.upper() in core_upper:
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
        target_stocks = 80
        target_bonds = 20
        target_core = 50
        target_satellite = 30
        
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
    
    def _generate_concrete_80_20_recommendations(
        self,
        balance_info: Dict,
        portfolio_metrics: Dict,
        holdings_analysis: List[Dict],
        exchange_rate: float
    ) -> List[Dict]:
        """Generate concrete buy recommendations to achieve 80/20 balance."""
        recommendations = []
        
        bonds_percent = balance_info.get("bonds_percent", 0)
        stocks_percent = balance_info.get("stocks_percent", 0)
        core_percent = balance_info.get("core_percent", 0)
        satellite_percent = balance_info.get("satellite_percent", 0)
        
        total_value = portfolio_metrics.get("total_value", 0)
        cash_available = portfolio_metrics.get("cash", 0)
        
        if total_value == 0:
            return recommendations
        
        core_etfs = self.CORE_ETFS
        bond_etfs = self.BOND_ETFS

        # Check what we need
        target_bonds = total_value * 0.20
        current_bonds = total_value * (bonds_percent / 100)
        needed_bonds = max(0, target_bonds - current_bonds)

        target_core = total_value * 0.50
        current_core = total_value * (core_percent / 100)
        needed_core = max(0, target_core - current_core)

        target_satellite = total_value * 0.30
        current_satellite = total_value * (satellite_percent / 100)
        needed_satellite = max(0, target_satellite - current_satellite)

        # Get current holdings
        current_holdings = [h.get("ticker", "").upper() for h in holdings_analysis]

        # Batch-fetch all candidate prices up front (one request for bonds + core).
        candidate_prices, _, _ = market_data.get_prices(bond_etfs + core_etfs)

        # Recommend bonds if needed (priority)
        if needed_bonds > 100:
            remaining_needed = needed_bonds
            for bond_etf in bond_etfs:
                if remaining_needed <= 0 or len(recommendations) >= 3:
                    break
                if bond_etf.upper() not in current_holdings or needed_bonds > 500:
                    price = candidate_prices.get(bond_etf)
                    if price and price > 0:
                        # Recommend based on what's needed, but note cash limitation
                        amount_to_recommend = min(remaining_needed, cash_available) if cash_available > 0 else remaining_needed
                        shares = int(amount_to_recommend / price)
                        if shares > 0:
                            actual_amount = shares * price
                            reason = f"Add bonds to reach 20% target (currently {bonds_percent:.1f}%)"
                            if cash_available < needed_bonds:
                                reason += f" | Need ${needed_bonds:,.2f} total, but only ${cash_available:,.2f} cash available"
                            recommendations.append({
                                "ticker": bond_etf,
                                "shares": shares,
                                "price": price,
                                "amount": actual_amount,
                                "amount_ils": actual_amount * exchange_rate,
                                "reason": reason,
                                "needed_total": needed_bonds,
                                "cash_available": cash_available
                            })
                            remaining_needed -= actual_amount

        # Recommend core if needed (only if we have cash left after bonds)
        remaining_cash = cash_available - sum(r.get("amount", 0) for r in recommendations)
        if needed_core > 100 and remaining_cash > 100:
            remaining_needed = needed_core
            for core_etf in core_etfs:
                if remaining_needed <= 0 or len(recommendations) >= 5:
                    break
                if core_etf.upper() not in current_holdings or needed_core > 500:
                    price = candidate_prices.get(core_etf)
                    if price and price > 0:
                        amount_to_recommend = min(remaining_needed, remaining_cash)
                        shares = int(amount_to_recommend / price)
                        if shares > 0:
                            actual_amount = shares * price
                            reason = f"Increase core holdings to reach 50% target (currently {core_percent:.1f}%)"
                            if remaining_cash < needed_core:
                                reason += f" | Need ${needed_core:,.2f} total, but only ${remaining_cash:,.2f} cash available"
                            recommendations.append({
                                "ticker": core_etf,
                                "shares": shares,
                                "price": price,
                                "amount": actual_amount,
                                "amount_ils": actual_amount * exchange_rate,
                                "reason": reason,
                                "needed_total": needed_core,
                                "cash_available": remaining_cash
                            })
                            remaining_needed -= actual_amount
                            remaining_cash -= actual_amount

        return recommendations
    
    def _generate_bond_recommendations(self, amount_usd: float, current_tickers: List[str], exchange_rate: float) -> List[Dict]:
        """Generate bond ETF recommendations for a specific amount."""
        bond_etfs = self.BOND_ETFS
        recommendations = []

        current_upper = {t.upper() for t in current_tickers}
        # One batched price fetch for all bond ETFs.
        bond_prices, _, _ = market_data.get_prices(bond_etfs)
        remaining_amount = amount_usd

        for bond_etf in bond_etfs:
            if remaining_amount <= 0 or len(recommendations) >= 3:
                break

            if bond_etf.upper() not in current_upper:
                price = bond_prices.get(bond_etf)
                if price and price > 0:
                    shares = int(remaining_amount / price)
                    if shares > 0:
                        actual_amount = shares * price
                        recommendations.append({
                            "ticker": bond_etf,
                            "shares": shares,
                            "price": price,
                            "allocation_amount": actual_amount,
                            "name": bond_etf,
                            "reasons": [f"Rebalancing: Buy bonds to reach 25% target"]
                        })
                        remaining_amount -= actual_amount

        return recommendations
    
    def find_better_alternatives(self, holdings_analysis: List[Dict], portfolio_metrics: Dict) -> List[Dict]:
        """
        Deep analysis: Compare ALL existing holdings with market alternatives.
        Finds better ETFs even if current holdings are not weak.
        This ensures portfolio is always optimized according to 80/20 strategy.
        """
        from deposit_advisor import DepositAdvisor
        
        replacement_opportunities = []
        advisor = DepositAdvisor(self.portfolio_file)
        
        # Get current portfolio
        portfolio = self.load_portfolio()
        current_tickers = [h.get("ticker", "").upper() for h in portfolio.get("holdings", [])]
        total_value = portfolio_metrics.get("total_value", 0)

        # Cost-basis lookup so replacement recs can estimate the tax hit of selling.
        cost_basis_by_ticker = {
            h.get("ticker", "").upper(): (h.get("cost_basis") or h.get("purchase_price"))
            for h in portfolio.get("holdings", [])
        }
        # exchange_rate is irrelevant for the USD tax estimate; pass 1.0 to skip
        # the live FX fetch in TaxAnalyzer.__init__.
        tax_analyzer = TaxAnalyzer(self.portfolio_file, exchange_rate=1.0)

        # Define categories for comparison
        core_etfs = self.CORE_ETFS
        bond_etfs = self.BOND_ETFS
        satellite_categories = etf_universe.SATELLITE_CATEGORIES

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
                    if cat in advisor.ETF_CATEGORIES:
                        cat_etfs = advisor.ETF_CATEGORIES[cat]
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
                    candidate_analysis = advisor.analyze_etf(candidate, verbose=False)
                    candidate_score = candidate_analysis.get("score", 0)
                    
                    # Boost score for Core and Bonds (they're essential for 80/20 strategy)
                    if is_core:
                        candidate_score = min(100, candidate_score + 15)
                    elif is_bond:
                        candidate_score = min(100, candidate_score + 20)
                    
                    # Check if significantly better (at least 15 points higher)
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
                
                # Only recommend if meaningful improvement
                if score_diff >= 15:
                    # Recommend replacing 30-50% depending on score difference
                    replace_percentage = 0.5 if score_diff >= 25 else 0.3
                    shares_to_sell = int(holding.get("quantity", 0) * replace_percentage)
                    sell_amount = shares_to_sell * holding.get("current_price", 0)
                    
                    if sell_amount > 100:  # Only if meaningful amount
                        expected_return = best_analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0)
                        current_return = holding.get("mid_term_forecast", {}).get("expected_3yr_return", 0) if isinstance(holding.get("mid_term_forecast"), dict) else 0
                        
                        # Calculate buy shares
                        buy_price = best_analysis.get("current_price", 0)
                        if buy_price > 0:
                            buy_shares = int(sell_amount / buy_price)
                            buy_amount = buy_shares * buy_price
                        else:
                            buy_shares = 0
                            buy_amount = 0
                        
                        # Only recommend if we can buy at least 1 share
                        if buy_shares > 0:
                            # Estimate the realized capital-gains tax of selling
                            # this portion, so the human weighs it against the
                            # score gain (a taxable swap can be net-negative).
                            est_tax = tax_analyzer.estimate_sale_tax_usd(
                                cost_basis_by_ticker.get(ticker),
                                holding.get("current_price", 0),
                                shares_to_sell,
                            )
                            if est_tax is None:
                                tax_note = ("⚠️ No cost basis tracked — tax impact unknown; "
                                            "selling a gain triggers ~25% capital-gains tax.")
                            elif est_tax > 0:
                                tax_note = (f"⚠️ Selling triggers ~${est_tax:,.0f} capital-gains "
                                            f"tax (25%); the score gain must justify it.")
                            else:
                                tax_note = "No capital-gains tax (no gain on the portion sold)."
                            replacement_opportunities.append({
                            "sell_ticker": ticker,
                            "sell_score": current_score,
                            "sell_shares": shares_to_sell,
                            "sell_amount": sell_amount,
                            "buy_ticker": best_alternative,
                            "buy_score": best_score,
                            "buy_shares": buy_shares,
                            "buy_amount": buy_amount,
                            "buy_price": buy_price,
                            "score_improvement": score_diff,
                            "category": category_name,
                            "expected_return": expected_return,
                            "current_return": current_return,
                            "replace_percentage": replace_percentage,
                            "estimated_sale_tax_usd": est_tax,
                            "tax_note": tax_note,
                        })
        
        return replacement_opportunities
    
    def find_concentration_opportunities(self, holdings_analysis: List[Dict], portfolio_metrics: Dict, emerging_trends: List[Dict] = None) -> List[Dict]:
        """
        Find replacement opportunities based on:
        1. Over-concentration (ETF > 30% of portfolio)
        2. Hot sectors (emerging trends with strong momentum)
        3. Portfolio balance (Core too low, Satellite too high)

        ``emerging_trends`` may be passed in (computed once by the caller) to
        avoid recomputing the expensive trend sweep; if None it is computed here.
        """
        from deposit_advisor import DepositAdvisor

        replacement_opportunities = []
        advisor = DepositAdvisor(self.portfolio_file)

        # Get current portfolio
        portfolio = self.load_portfolio()
        current_tickers = [h.get("ticker", "").upper() for h in portfolio.get("holdings", [])]
        total_value = portfolio_metrics.get("total_value", 0)

        # Get emerging trends (reuse caller's computation when provided)
        if emerging_trends is None:
            excluded_categories = etf_universe.EXCLUDED_CATEGORIES
            emerging_trends = advisor.detect_emerging_trends(excluded_categories)
        hot_sectors = {t.get("category", ""): t for t in emerging_trends[:5]}  # Top 5 hot sectors

        # Calculate current portfolio balance
        core_etfs = self.CORE_ETFS
        bond_etfs = self.BOND_ETFS

        core_value = 0
        satellite_value = 0
        bonds_value = 0
        
        for holding in holdings_analysis:
            ticker = holding.get("ticker", "").upper()
            value = holding.get("current_value", 0)
            
            if ticker in [e.upper() for e in core_etfs]:
                core_value += value
            elif ticker in [e.upper() for e in bond_etfs]:
                bonds_value += value
            else:
                satellite_value += value
        
        core_pct = (core_value / total_value * 100) if total_value > 0 else 0
        satellite_pct = (satellite_value / total_value * 100) if total_value > 0 else 0
        
        # Check each holding for over-concentration or opportunities
        for holding in holdings_analysis:
            ticker = holding.get("ticker", "").upper()
            current_score = holding.get("recommendation_score", 50)
            current_value = holding.get("current_value", 0)
            current_weight = (current_value / total_value * 100) if total_value > 0 else 0
            
            # Skip if too small
            if current_weight < 2:
                continue
            
            is_core = ticker in [e.upper() for e in core_etfs]
            is_bond = ticker in [e.upper() for e in bond_etfs]
            
            # 1. Check for over-concentration (>30% of portfolio)
            if current_weight > 30:
                # Find best alternative from hot sectors or Core
                best_alternative = None
                best_score = 0
                best_analysis = None
                reason = ""
                
                # Priority: If Core is low, recommend Core ETFs
                if core_pct < 45:
                    for core_etf in core_etfs:
                        if core_etf.upper() not in current_tickers:
                            try:
                                analysis = advisor.analyze_etf(core_etf, verbose=False)
                                score = analysis.get("score", 0)
                                if score > best_score:
                                    best_score = score
                                    best_alternative = core_etf
                                    best_analysis = analysis
                                    reason = f"Over-concentration ({current_weight:.1f}%) + Low Core ({core_pct:.1f}%)"
                            except Exception:
                                continue
                
                # If no Core alternative or Core is OK, check hot sectors
                if not best_alternative or core_pct >= 45:
                    for category, trend_data in hot_sectors.items():
                        if category in advisor.ETF_CATEGORIES:
                            trend_etfs = advisor.ETF_CATEGORIES[category]
                            momentum = trend_data.get("avg_momentum", 0)
                            
                            # Only consider very hot sectors (momentum > 10%)
                            if momentum > 10:
                                for etf in trend_etfs[:2]:
                                    if etf.upper() not in current_tickers:
                                        try:
                                            analysis = advisor.analyze_etf(etf, verbose=False)
                                            score = analysis.get("score", 0)
                                            # Boost score for hot sectors
                                            score = min(100, score + 10)
                                            if score > best_score:
                                                best_score = score
                                                best_alternative = etf
                                                best_analysis = analysis
                                                reason = f"Over-concentration ({current_weight:.1f}%) + Hot sector ({category}, {momentum:.1f}% momentum)"
                                        except Exception:
                                            continue
                
                # If found alternative, recommend replacement
                if best_alternative and best_analysis:
                    # Recommend replacing 30-50% depending on concentration
                    replace_percentage = 0.5 if current_weight > 40 else 0.3
                    shares_to_sell = int(holding.get("quantity", 0) * replace_percentage)
                    sell_amount = shares_to_sell * holding.get("current_price", 0)
                    
                    if sell_amount > 100:
                        buy_price = best_analysis.get("current_price", 0)
                        if buy_price > 0:
                            buy_shares = int(sell_amount / buy_price)
                            buy_amount = buy_shares * buy_price
                        else:
                            buy_shares = 0
                            buy_amount = 0
                        
                        if buy_shares > 0:
                            replacement_opportunities.append({
                                "sell_ticker": ticker,
                                "sell_score": current_score,
                                "sell_shares": shares_to_sell,
                                "sell_amount": sell_amount,
                                "buy_ticker": best_alternative,
                                "buy_score": best_score,
                                "buy_shares": buy_shares,
                                "buy_amount": buy_amount,
                                "buy_price": buy_price,
                                "score_improvement": best_score - current_score,
                                "category": "DIVERSIFICATION",
                                "expected_return": best_analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0),
                                "current_return": holding.get("mid_term_forecast", {}).get("expected_3yr_return", 0) if isinstance(holding.get("mid_term_forecast"), dict) else 0,
                                "replace_percentage": replace_percentage,
                                "reason": reason
                            })
            
            # 2. Check if Satellite is too high and Core is too low
            elif not is_core and not is_bond and satellite_pct > 35 and core_pct < 45:
                # Recommend replacing some Satellite with Core
                for core_etf in core_etfs:
                    if core_etf.upper() not in current_tickers:
                        try:
                            analysis = advisor.analyze_etf(core_etf, verbose=False)
                            score = analysis.get("score", 0)
                            # Boost score for Core when Core is low
                            score = min(100, score + 15)
                            
                            # Only recommend if meaningful improvement
                            if score >= current_score + 10:
                                replace_percentage = 0.3
                                shares_to_sell = int(holding.get("quantity", 0) * replace_percentage)
                                sell_amount = shares_to_sell * holding.get("current_price", 0)
                                
                                if sell_amount > 100:
                                    buy_price = analysis.get("current_price", 0)
                                    if buy_price > 0:
                                        buy_shares = int(sell_amount / buy_price)
                                        buy_amount = buy_shares * buy_price
                                    else:
                                        buy_shares = 0
                                        buy_amount = 0
                                    
                                    if buy_shares > 0:
                                        replacement_opportunities.append({
                                            "sell_ticker": ticker,
                                            "sell_score": current_score,
                                            "sell_shares": shares_to_sell,
                                            "sell_amount": sell_amount,
                                            "buy_ticker": core_etf,
                                            "buy_score": score,
                                            "buy_shares": buy_shares,
                                            "buy_amount": buy_amount,
                                            "buy_price": buy_price,
                                            "score_improvement": score - current_score,
                                            "category": "CORE",
                                            "expected_return": analysis.get("mid_term_forecast", {}).get("expected_3yr_return", 0),
                                            "current_return": holding.get("mid_term_forecast", {}).get("expected_3yr_return", 0) if isinstance(holding.get("mid_term_forecast"), dict) else 0,
                                            "replace_percentage": replace_percentage,
                                            "reason": f"Satellite too high ({satellite_pct:.1f}%) + Core too low ({core_pct:.1f}%) - Rebalance to 80/20"
                                        })
                                        break  # Only one Core recommendation per Satellite holding
                        except Exception:
                            continue
        
        return replacement_opportunities
    
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
                rebalancing["reason"] = f"Portfolio not balanced (80/20): {balance_reasons[0] if balance_reasons else 'Needs rebalancing'}"
        
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
        
        # PRIORITY 1: Check for underperforming holdings and compare with better alternatives
        # Performance-based selling takes priority over concentration
        # Also check if there are better ETF alternatives available
        
        # First, identify underperforming holdings
        underperforming_holdings = []
        for analysis in analyses:
            score = analysis.get("recommendation_score", 50)
            recommendation = analysis.get("recommendation", "HOLD")
            ticker = analysis["ticker"]
            
            if recommendation == "STRONG SELL" or score < 30:
                underperforming_holdings.append({
                    "ticker": ticker,
                    "score": score,
                    "recommendation": recommendation,
                    "analysis": analysis
                })
            elif recommendation == "SELL" or (30 <= score < 40):
                underperforming_holdings.append({
                    "ticker": ticker,
                    "score": score,
                    "recommendation": recommendation,
                    "analysis": analysis,
                    "moderate": True
                })
        
        # Rebalancing cooldown: if last rebalance OR last make analyze was < 30 days ago,
        # only recommend for really unique opportunities (severe underperformers: STRONG SELL / score < 25)
        portfolio = self.load_portfolio()
        today = datetime.now().date()
        within_cooldown = False
        for date_key in ("last_rebalancing_date", "last_analyze_run_date"):
            last_d_str = portfolio.get(date_key)
            if not last_d_str:
                continue
            try:
                last_d = datetime.strptime(last_d_str, "%Y-%m-%d").date()
                if (today - last_d).days < self.rebalancing_cooldown_days:
                    within_cooldown = True
                    break
            except (ValueError, TypeError):
                pass
        if within_cooldown:
            underperforming_holdings = [
                u for u in underperforming_holdings
                if u.get("recommendation") == "STRONG SELL" or u.get("score", 50) < 25
            ]
        
        # For each underperforming holding, try to find better alternatives
        if underperforming_holdings:
            from deposit_advisor import DepositAdvisor
            advisor = DepositAdvisor(self.portfolio_file)
            
            # Define categories for comparison
            core_etfs = self.CORE_ETFS
            satellite_etfs = self.SATELLITE_ETFS
            bond_etfs = self.BOND_ETFS
            
            for underperformer in underperforming_holdings:
                ticker = underperformer["ticker"]
                current_score = underperformer["score"]
                analysis = underperformer["analysis"]
                current_weight = weights.get(ticker, 0)
                is_moderate = underperformer.get("moderate", False)
                
                # Determine category of underperforming ETF
                ticker_upper = ticker.upper()
                is_core = ticker_upper in [e.upper() for e in core_etfs]
                is_bond = ticker_upper in [e.upper() for e in bond_etfs]
                is_satellite = ticker_upper in [e.upper() for e in satellite_etfs]
                
                # Find better alternatives in the same category
                better_alternatives = []
                candidates = []
                
                if is_core:
                    candidates = [e for e in core_etfs if e.upper() != ticker_upper and e.upper() not in current_tickers]
                elif is_bond:
                    candidates = [e for e in bond_etfs if e.upper() != ticker_upper and e.upper() not in current_tickers]
                elif is_satellite:
                    candidates = [e for e in satellite_etfs if e.upper() != ticker_upper and e.upper() not in current_tickers]
                
                # Analyze candidates to find better alternatives
                for candidate in candidates[:3]:  # Check top 3 alternatives
                    try:
                        candidate_analysis = advisor.analyze_etf(candidate, verbose=False)
                        candidate_score = candidate_analysis.get("score", 0)
                        
                        # If alternative is significantly better (at least 20 points higher)
                        if candidate_score > current_score + 20:
                            better_alternatives.append({
                                "ticker": candidate,
                                "score": candidate_score,
                                "name": candidate_analysis.get("name", candidate),
                                "current_price": candidate_analysis.get("current_price", 0),
                                "score_difference": candidate_score - current_score
                            })
                    except Exception as e:
                        logger.debug(f"Failed to analyze alternative {candidate}: {e}")
                        continue
                
                # Sort alternatives by score difference
                better_alternatives.sort(key=lambda x: x["score_difference"], reverse=True)
                
                # Generate sell recommendation with replacement suggestion
                rebalancing["needed"] = True
                if is_moderate:
                    sell_percentage = 0.25
                else:
                    sell_percentage = 0.75
                
                sell_value_usd = analysis["current_value"] * sell_percentage
                sell_shares = int(analysis["quantity"] * sell_percentage)
                # If fractional shares round to 0 (e.g. 25% of 1 share), recommend selling 1 share so the action is valid
                if sell_shares == 0 and analysis["quantity"] >= 1 and sell_value_usd > 0:
                    sell_shares = 1
                    sell_value_usd = analysis["current_price"] * sell_shares
                total_sell_amount += sell_value_usd
                
                reason = f"{underperformer['recommendation']} - Score: {current_score:.1f}/100. Poor performance"
                if better_alternatives:
                    best_alternative = better_alternatives[0]
                    reason += f" | Better alternative: {best_alternative['ticker']} (Score: {best_alternative['score']:.1f}/100, +{best_alternative['score_difference']:.1f} points)"
                    # Add buy recommendation for the better alternative
                    if best_alternative["current_price"] > 0:
                        buy_shares = max(1, int(sell_value_usd / best_alternative["current_price"]))
                        buy_amount = buy_shares * best_alternative["current_price"]
                        rebalancing["buy_recommendations"].append({
                            "ticker": best_alternative["ticker"],
                            "name": best_alternative["name"],
                            "shares": buy_shares,
                            "price": best_alternative["current_price"],
                            "amount": buy_amount,
                            "allocation_amount": buy_amount,
                            "reason": f"Replace {ticker} (Score: {current_score:.1f}) with {best_alternative['ticker']} (Score: {best_alternative['score']:.1f}) - Better performance"
                        })
                else:
                    reason += " - No better alternatives found in same category"
                
                if not rebalancing["reason"]:
                    rebalancing["reason"] = f"Underperforming holding: {ticker} (Score: {current_score:.1f}/100)"
                
                rebalancing["recommendations"].append({
                    "action": "SELL",
                    "ticker": ticker,
                    "sell_amount_usd": sell_value_usd,
                    "sell_shares": sell_shares,
                    "current_price_usd": analysis["current_price"],
                    "current_weight": current_weight,
                    "current_score": current_score,
                    "better_alternatives": better_alternatives[:2],  # Top 2 alternatives
                    "reason": f"{reason} - selling {sell_percentage*100:.0f}% ({sell_shares} shares, ${sell_value_usd:,.2f})"
                })
        
        # Check for holdings with very low scores but not yet SELL (warning threshold)
        low_score_holdings = [a for a in analyses if 30 <= a["recommendation_score"] < 40]
        if low_score_holdings and not rebalancing["needed"]:
            # Don't force rebalancing, but note it in the reason
            rebalancing["reason"] = f"Some holdings have low scores: {', '.join([a['ticker'] for a in low_score_holdings])}"
        
        # PRIORITY 2: Rebalance for 80/20 balance by selling stocks to buy bonds
        # BUT: Consider tax implications and alternatives
        if not balance_check["is_balanced"]:
            bonds_percent = balance_check.get("bonds_percent", 0)
            stocks_percent = balance_check.get("stocks_percent", 0)
            cash_available = portfolio_metrics.get("cash", 0)
            
            # If we have too many stocks (>80%) and too few bonds (<20%)
            if stocks_percent > 80 and bonds_percent < 20:
                target_stocks = 80
                target_bonds = 20
                excess_stocks = stocks_percent - target_stocks
                needed_bonds = target_bonds - bonds_percent
                
                # Calculate how much to rebalance
                total_value = portfolio_metrics["total_value"]
                needed_bonds_value = total_value * (needed_bonds / 100)
                
                # STRATEGY DECISION: Should we sell or wait for deposit?
                # Option 1: If we have significant cash, prefer using cash over selling
                # Option 2: If we need to sell, prioritize tax-loss harvesting
                
                # Check if we can use cash instead (if cash covers >50% of needed bonds)
                if cash_available > needed_bonds_value * 0.5:
                    # Prefer using cash - add note but don't force selling
                    if not rebalancing["reason"]:
                        rebalancing["reason"] = f"Portfolio not balanced (80/20): Need ${needed_bonds_value:,.2f} in bonds. Consider using ${cash_available:,.2f} cash first, or wait for deposit to avoid tax on sales."
                else:
                    # Need to sell - but be smart about it
                    rebalance_percentage = min(excess_stocks, needed_bonds) / 100
                    
                    if rebalance_percentage > 0.05:  # Only if difference is >5%
                        rebalance_amount = total_value * rebalance_percentage
                        
                        # Load portfolio to check purchase dates for tax analysis
                        portfolio = self.load_portfolio()
                        from tax_analyzer import TaxAnalyzer
                        tax_analyzer = TaxAnalyzer()
                        # Use the live FX rate so tax ILS figures are accurate.
                        tax_exchange_rate = market_data.get_exchange_rate()
                        tax_analyzer.exchange_rate_usd_ils = tax_exchange_rate

                        core_upper = {c.upper() for c in self.CORE_ETFS}
                        bond_upper = {b.upper() for b in self.BOND_ETFS}

                        # Analyze stocks for tax implications
                        stocks_to_sell = []
                        for analysis in analyses:
                            ticker = analysis["ticker"]
                            if ticker.upper() not in bond_upper:
                                is_core = ticker.upper() in core_upper

                                # Find holding info for tax calculation
                                holding = next((h for h in portfolio.get("holdings", []) if h.get("ticker") == ticker), None)
                                # Ensure purchase_date is not None
                                purchase_date = None
                                if holding:
                                    purchase_date = holding.get("purchase_date")
                                if not purchase_date:
                                    purchase_date = portfolio.get("last_updated")
                                if not purchase_date:
                                    purchase_date = datetime.now().isoformat()
                                
                                # Use last_price as cost_basis if no cost_basis recorded (assume bought at last known price)
                                cost_basis = holding.get("cost_basis", holding.get("last_price", analysis["current_price"])) if holding else analysis["current_price"]
                                
                                # Calculate tax impact
                                tax_calc = tax_analyzer.calculate_capital_gains_tax(
                                    purchase_price=cost_basis,
                                    sale_price=analysis["current_price"],
                                    quantity=analysis["quantity"],
                                    purchase_date=purchase_date,
                                    exchange_rate=tax_exchange_rate
                                )
                                
                                # Calculate gain/loss
                                gain_loss = tax_calc.get("total_gain_ils", 0)
                                tax_cost = tax_calc.get("capital_gains_tax_ils", 0)
                                is_loss = gain_loss < 0
                                is_long_term = tax_calc.get("is_long_term", False)
                                
                                stocks_to_sell.append({
                                    "ticker": ticker,
                                    "analysis": analysis,
                                    "is_core": is_core,
                                    "weight": weights.get(ticker, 0),
                                    "gain_loss": gain_loss,
                                    "tax_cost": tax_cost,
                                    "is_loss": is_loss,
                                    "is_long_term": is_long_term,
                                    "tax_impact": tax_cost / analysis["current_value"] if analysis["current_value"] > 0 else 0
                                })
                        
                        # Sort by tax efficiency: losses first, then long-term gains, then short-term gains
                        # Within each group: satellite first, then by tax impact (lower is better)
                        stocks_to_sell.sort(key=lambda x: (
                            not x["is_loss"],  # Losses first (True < False, so losses come first)
                            not x["is_long_term"],  # Long-term before short-term
                            x["is_core"],  # Satellite before core
                            x["tax_impact"]  # Lower tax impact first
                        ))
                        
                        # Sell stocks to get money for bonds
                        remaining_to_sell = rebalance_amount
                        total_tax_cost = 0
                        for stock_info in stocks_to_sell:
                            if remaining_to_sell <= 0:
                                break
                            
                            ticker = stock_info["ticker"]
                            analysis = stock_info["analysis"]
                            current_value = analysis["current_value"]
                            
                            # Sell up to the needed amount, but max 30% of this holding
                            sell_amount = min(remaining_to_sell, current_value * 0.30)
                            if sell_amount > 100:  # Only if meaningful amount
                                sell_shares = int(sell_amount / analysis["current_price"]) if analysis["current_price"] > 0 else 0
                                if sell_shares > 0:
                                    # Check if already in sell recommendations (avoid duplicates)
                                    if not any(r.get("ticker") == ticker and r.get("action") == "SELL" 
                                              for r in rebalancing["recommendations"]):
                                        rebalancing["needed"] = True
                                        total_sell_amount += sell_amount
                                        
                                        # Calculate proportional tax for this sale
                                        proportional_tax = stock_info["tax_cost"] * (sell_amount / current_value) if current_value > 0 else 0
                                        total_tax_cost += proportional_tax
                                        
                                        # Build reason with tax info
                                        reason = f"Rebalancing: Sell {sell_shares} shares (${sell_amount:,.2f}) to buy bonds"
                                        if stock_info["is_loss"]:
                                            reason += f" | Tax benefit: {abs(stock_info['gain_loss']):,.0f} ILS loss"
                                        elif stock_info["tax_cost"] > 0:
                                            reason += f" | Tax cost: ~{proportional_tax:,.0f} ILS"
                                        if stock_info["is_long_term"]:
                                            reason += " | Long-term (lower tax)"
                                        
                                        rebalancing["recommendations"].append({
                                            "action": "SELL",
                                            "ticker": ticker,
                                            "sell_amount_usd": sell_amount,
                                            "sell_shares": sell_shares,
                                            "current_price_usd": analysis["current_price"],
                                            "current_weight": stock_info["weight"],
                                            "reason": reason,
                                            "tax_cost_ils": proportional_tax,
                                            "is_loss": stock_info["is_loss"]
                                        })
                                        remaining_to_sell -= sell_amount
                        
                        # Add tax warning to reason
                        if total_tax_cost > 0:
                            rebalancing["reason"] += f" | Estimated tax cost: ~{total_tax_cost:,.0f} ILS. Consider waiting for deposit instead."
                        
                        # Now recommend bonds to buy with the proceeds
                        if total_sell_amount > 0:
                            bond_recs = self._generate_bond_recommendations(total_sell_amount, current_tickers, exchange_rate=self.get_exchange_rate())
                            if bond_recs:
                                rebalancing["buy_recommendations"].extend(bond_recs)
        
        # If we're selling for other reasons (performance, concentration), recommend what to buy instead
        if total_sell_amount > 0 and rebalancing["recommendations"]:
            sell_tickers = [r["ticker"] for r in rebalancing["recommendations"] if r["action"] == "SELL"]
            # Only add buy recommendations if not already added for rebalancing
            if not rebalancing["buy_recommendations"]:
                buy_recs = self.find_best_etfs_to_buy(total_sell_amount, current_tickers, exclude_tickers=sell_tickers)
                rebalancing["buy_recommendations"] = buy_recs
        
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
        
        # Check for emerging trends first (NEW - automatically detects hot sectors)
        print("\n🔍 Checking for emerging trends and hot sectors...")
        from deposit_advisor import DepositAdvisor
        advisor = DepositAdvisor(self.portfolio_file)
        excluded_categories = etf_universe.EXCLUDED_CATEGORIES
        emerging_trends = advisor.detect_emerging_trends(excluded_categories)
        if emerging_trends:
            print(f"   ✅ Found {len(emerging_trends)} emerging trends with strong momentum:")
            for trend in emerging_trends[:3]:  # Show top 3
                category = trend.get("category", "")
                momentum = trend.get("avg_momentum", 0)
                return_pct = trend.get("avg_return", 0)
                print(f"      🔥 {category}: {momentum:.1f}% momentum, {return_pct:.1f}% return (6mo)")
        else:
            print("   ℹ️  No strong emerging trends detected at this time")
        
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
            self.save_portfolio(portfolio)
            logger.info(f"📊 Initialized baseline tracking: ${baseline_value:,.2f} on {portfolio['start_date']}")

        # Seed the deposit/trade ledger if missing so true (deposit-adjusted)
        # returns can be tracked from now on.
        current_total = portfolio.get("cash", 0) + sum(
            h["quantity"] * prices.get(h["ticker"], h.get("last_price", 0))
            for h in portfolio["holdings"]
        )
        if ledger.ensure_ledger(portfolio, current_total):
            self.save_portfolio(portfolio)
        
        # Display market status
        print(f"\n📊 Market Status: {market_message}")
        if market_status:
            print("   ⚡ Using REAL-TIME prices")
        else:
            print("   📅 Using LAST CLOSE prices")
        
        # Price caching is handled by the shared market_data layer.
        print(f"   💾 Prices served via shared cache (window: {'60 min' if market_status else '4 hr'})")
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
        
        # Add concrete 80/20 recommendations if needed
        balance_info = rebalancing.get("balance_80_20", {})
        if balance_info and not balance_info.get("is_balanced", False):
            exchange_rate = self.get_exchange_rate()
            concrete_recs = self._generate_concrete_80_20_recommendations(
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
        
        # Save updated portfolio
        self.save_portfolio(portfolio)
        
        # Check for better alternatives (like critical_alert does)
        print("\n🔍 Checking for better ETF alternatives...")
        replacement_opportunities = self.find_better_alternatives(analyses, portfolio_metrics)
        
        # Also check for over-concentration and hot sector opportunities
        # (reuse the emerging_trends computed at the top of analyze() - no recompute).
        concentration_opportunities = self.find_concentration_opportunities(analyses, portfolio_metrics, emerging_trends=emerging_trends)
        if concentration_opportunities:
            replacement_opportunities.extend(concentration_opportunities)
        
        if replacement_opportunities:
            print(f"   ✅ Found {len(replacement_opportunities)} better alternatives!")
            rebalancing["replacement_opportunities"] = replacement_opportunities
        else:
            print("   ✅ All holdings are performing well - no better alternatives found")
        
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

        # Stamp the run date now that the analysis (and its cooldown checks,
        # which read the PREVIOUS stamp) has completed. Read-only/CI runs never
        # stamp, so the 30-day reminder tracks conscious interactive runs only.
        if record_run and not read_only:
            portfolio["last_analyze_run_date"] = datetime.now().strftime("%Y-%m-%d")
            self.save_portfolio(portfolio)

        if read_only:
            print("\n📋 Read-only mode: portfolio was not modified. Run 'make analyze' (without read-only) to update after you execute trades.\n")
            return results
        
        # Ask for confirmation if rebalancing is needed
        if rebalancing["needed"] and (rebalancing["recommendations"] or rebalancing["buy_recommendations"]):
            confirmed = self.ask_rebalancing_confirmation()
            if confirmed:
                self.update_portfolio_from_rebalancing(portfolio, rebalancing, analyses)
                print("\n✅ Portfolio updated successfully based on rebalancing actions!\n")
                self._run_update_secret()
            else:
                print("\n❌ Portfolio not updated. No changes were made.\n")
        
        # Ask for confirmation if replacement opportunities exist
        if rebalancing.get("replacement_opportunities"):
            confirmed = self.ask_replacement_confirmation()
            if confirmed:
                self.update_portfolio_from_replacements(portfolio, rebalancing["replacement_opportunities"], analyses)
                print("\n✅ Portfolio updated successfully based on replacement recommendations!\n")
                self._run_update_secret()
            else:
                print("\n❌ Portfolio not updated. No changes were made.\n")
        
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
    
    def ask_replacement_confirmation(self) -> bool:
        """Ask user: did you already execute these replacement trades in your broker?"""
        print("\n⚠️  Answer YES only if you already did these exact replacement trades in your broker.")
        while True:
            response = input("\nDid you execute these exact replacement trades in your broker? (yes/no): ").strip().lower()
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

        # Recalculate total value
        total_value = portfolio.get("cash", 0)
        for holding in portfolio.get("holdings", []):
            total_value += holding.get("current_value", 0)
        portfolio["total_value"] = total_value
        
        # Record rebalancing date so we don't recommend again too soon (cooldown)
        portfolio["last_rebalancing_date"] = datetime.now().strftime("%Y-%m-%d")
        
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
    
    def update_portfolio_from_replacements(self, portfolio: Dict, replacement_opportunities: List[Dict], analyses: List[Dict]):
        """Update portfolio.json based on replacement recommendations."""
        exchange_rate = self.get_exchange_rate()
        
        # Process each replacement opportunity
        for opp in replacement_opportunities:
            sell_ticker = opp["sell_ticker"]
            buy_ticker = opp["buy_ticker"]
            shares_to_sell = opp["sell_shares"]
            shares_to_buy = opp["buy_shares"]
            sell_amount = opp["sell_amount"]
            buy_price = opp["buy_price"]
            buy_amount = opp["buy_amount"]
            remaining_cash = sell_amount - buy_amount
            
            # Get current price for sell ticker from analyses
            sell_price = 0
            for analysis in analyses:
                if analysis["ticker"] == sell_ticker:
                    sell_price = analysis.get("current_price", 0)
                    break

            # SELL: Reduce or remove the old holding
            used_price, sold = self._apply_sell(portfolio, sell_ticker, shares_to_sell, sell_price)
            if sold < shares_to_sell:
                # Fewer shares held than the recommendation assumed - shrink the
                # proceeds so cash reflects reality, not the plan.
                sell_amount = sold * used_price
                remaining_cash = sell_amount - buy_amount

            # BUY: Add or update the new holding
            self._apply_buy(portfolio, buy_ticker, shares_to_buy, buy_price)

            # Add remaining cash from the transaction
            if remaining_cash > 0:
                portfolio["cash"] = portfolio.get("cash", 0) + remaining_cash
        
        # Recalculate total value
        total_value = portfolio.get("cash", 0)
        for holding in portfolio.get("holdings", []):
            total_value += holding.get("current_value", 0)
        portfolio["total_value"] = total_value
        
        # Save updated portfolio
        self.save_portfolio(portfolio)
        
        # Print summary
        print("\n" + "-" * 60)
        print("PORTFOLIO UPDATE SUMMARY (Replacements)")
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
        
        # Show 80/20 balance status
        balance_info = results["rebalancing"].get("balance_80_20", {})
        if balance_info:
            print("\n" + "-" * 60)
            print("80/20 BALANCED GROWTH STRATEGY STATUS")
            print("-" * 60)
            stocks_pct = balance_info.get("stocks_percent", 0)
            bonds_pct = balance_info.get("bonds_percent", 0)
            core_pct = balance_info.get("core_percent", 0)
            satellite_pct = balance_info.get("satellite_percent", 0)
            is_balanced = balance_info.get("is_balanced", False)
            
            status_icon = "✅" if is_balanced else "⚠️"
            print(f"{status_icon} Stocks: {stocks_pct:.1f}% (Target: 80%)")
            print(f"   ├─ Core: {core_pct:.1f}% (Target: 50%)")
            print(f"   └─ Satellite: {satellite_pct:.1f}% (Target: 30%)")
            print(f"{status_icon} Bonds: {bonds_pct:.1f}% (Target: 20%)")
            
            if not is_balanced:
                recommendations = balance_info.get("recommendations", [])
                if recommendations:
                    print(f"\n📋 Recommendations to achieve 80/20 balance:")
                    for rec in recommendations:
                        print(f"   • {rec.get('reason', 'N/A')}")
                    
                    # Generate concrete buy recommendations
                    print(f"\n💡 CONCRETE BUY RECOMMENDATIONS:")
                    concrete_recs = self._generate_concrete_80_20_recommendations(
                        balance_info, metrics, results["holdings_analysis"], exchange_rate
                    )
                    if concrete_recs:
                        total_needed = sum(r.get('needed_total', 0) for r in concrete_recs if r.get('needed_total'))
                        total_recommended = sum(r.get('amount', 0) for r in concrete_recs)
                        if total_needed > 0 and total_needed > total_recommended:
                            print(f"\n   ⚠️  Total needed: ${total_needed:,.2f}, but only ${total_recommended:,.2f} recommended (limited by available cash)")
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
                    beta_val = ti.get('beta')
                    if beta_val is None or (isinstance(beta_val, float) and np.isnan(beta_val)):
                        print("  Beta: N/A")
                    else:
                        print(f"  Beta: {beta_val:.2f}")
                if 'max_drawdown' in ti:
                    print(f"  Max Drawdown: {ti.get('max_drawdown', 0):.2f}%")
        
        print("\n" + "=" * 60)
        print("REBALANCING SUMMARY")
        print("=" * 60)
        
        rebalancing = results["rebalancing"]
        # Reuse the exchange_rate fetched at the top of this method.

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
        
        # Show replacement opportunities if any
        if rebalancing.get("replacement_opportunities"):
            print("\n" + "=" * 60)
            print("🔄 REPLACEMENT OPPORTUNITIES (Better Alternatives Found)")
            print("=" * 60)
            # Bug fix: portfolio_metrics has no "exchange_rate" key, so the old
            # .get(..., 3.15) always used a stale hardcoded rate. Use the live rate
            # already fetched at the top of this method.

            for opp in rebalancing["replacement_opportunities"]:
                sell_ticker = opp["sell_ticker"]
                buy_ticker = opp["buy_ticker"]
                score_diff = opp["score_improvement"]
                sell_score = opp["sell_score"]
                buy_score = opp["buy_score"]
                sell_shares = opp["sell_shares"]
                sell_amount = opp["sell_amount"]
                buy_shares = opp["buy_shares"]
                buy_amount = opp["buy_amount"]
                buy_price = opp["buy_price"]
                category = opp["category"]
                replace_pct = opp["replace_percentage"] * 100
                
                priority = "HIGH" if score_diff >= 25 else "MEDIUM"
                priority_icon = "🔴" if priority == "HIGH" else "🟡"
                
                print(f"\n{priority_icon} [{priority}] OPTIMIZE: {sell_ticker} → {buy_ticker}")
                print(f"   Category: {category}")
                print(f"   Score Improvement: {sell_score:.1f}/100 → {buy_score:.1f}/100 (+{score_diff:.1f} points)")
                print(f"   Recommendation: Replace {replace_pct:.0f}% of {sell_ticker} with {buy_ticker}")
                print(f"   🔴 SELL: {sell_shares} shares of {sell_ticker} = ${sell_amount:,.2f} (₪{sell_amount * exchange_rate:,.2f})")
                print(f"   🟢 BUY: {buy_shares} shares of {buy_ticker} @ ${buy_price:.2f} = ${buy_amount:,.2f} (₪{buy_amount * exchange_rate:,.2f})")
                tax_note = opp.get("tax_note")
                if tax_note:
                    print(f"   🧾 {tax_note}")
                if sell_amount - buy_amount > 5:
                    remaining = sell_amount - buy_amount
                    print(f"   💰 Remaining Cash: ${remaining:,.2f} (₪{remaining * exchange_rate:,.2f})")
            
            print("\n" + "=" * 60)
        
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

