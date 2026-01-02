#!/usr/bin/env python3
"""
Portfolio Analyzer - Advanced Investment Portfolio Analysis System
Analyzes portfolio holdings, provides recommendations, and suggests rebalancing.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import requests
from newsapi import NewsApiClient
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """Advanced portfolio analysis system with news, trends, and statistical analysis."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self.exchange_rate_usd_ils = 3.7  # Default, will be updated
    
    def get_exchange_rate(self) -> float:
        """Get current USD/ILS exchange rate."""
        try:
            usd_ils = yf.Ticker("USDILS=X")
            hist = usd_ils.history(period="1d")
            if not hist.empty:
                rate = float(hist['Close'].iloc[-1])
                self.exchange_rate_usd_ils = rate
                return rate
            return self.exchange_rate_usd_ils
        except Exception as e:
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
        """Save portfolio to JSON file."""
        portfolio["last_updated"] = datetime.now().isoformat()
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(portfolio, f, indent=2, ensure_ascii=False)
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for tickers."""
        prices = {}
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                info = stock.history(period="1d")
                if not info.empty:
                    prices[ticker] = float(info['Close'].iloc[-1])
                else:
                    # Try to get info from fast_info
                    try:
                        prices[ticker] = float(stock.fast_info['lastPrice'])
                    except:
                        print(f"Warning: Could not fetch price for {ticker}")
        except Exception as e:
            print(f"Error fetching prices: {e}")
        return prices
    
    def get_market_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical market data for analysis."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching market data for {ticker}: {e}")
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
        
        return indicators
    
    def get_news_sentiment(self, ticker: str) -> Dict:
        """Get news sentiment for a ticker."""
        sentiment = {
            "score": 0,
            "articles_count": 0,
            "recent_news": []
        }
        
        if not self.news_api_key:
            return sentiment
        
        try:
            newsapi = NewsApiClient(api_key=self.news_api_key)
            # Get company name from ticker
            stock = yf.Ticker(ticker)
            info = stock.info
            company_name = info.get('longName', ticker)
            
            # Search for news
            articles = newsapi.get_everything(
                q=f"{company_name} OR {ticker}",
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            if articles['status'] == 'ok':
                sentiment["articles_count"] = len(articles['articles'])
                sentiment["recent_news"] = [
                    {
                        "title": article['title'],
                        "source": article['source']['name'],
                        "published": article['publishedAt']
                    }
                    for article in articles['articles'][:5]
                ]
                # Simple sentiment scoring (can be enhanced with NLP)
                sentiment["score"] = min(len(articles['articles']) * 10, 100)
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
        
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
        
        # News sentiment
        if analysis["news_sentiment"]["score"] > 50:
            score += 10
        elif analysis["news_sentiment"]["score"] < 20:
            score -= 10
        
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
    
    def check_rebalancing(self, portfolio_metrics: Dict, analyses: List[Dict]) -> Dict:
        """Determine if rebalancing is needed."""
        rebalancing = {
            "needed": False,
            "reason": "",
            "recommendations": []
        }
        
        weights = portfolio_metrics["weights"]
        total_holdings = len(weights)
        total_value_usd = portfolio_metrics["holdings_value"]
        
        # Check for over-concentration
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.4:  # More than 40% in one holding
            rebalancing["needed"] = True
            ticker = max(weights, key=weights.get)
            current_value_usd = total_value_usd * max_weight
            target_value_usd = total_value_usd * 0.25
            reduce_amount_usd = current_value_usd - target_value_usd
            
            # Find the analysis for this ticker to get current price
            ticker_analysis = next((a for a in analyses if a["ticker"] == ticker), None)
            if ticker_analysis and ticker_analysis["current_price"] > 0:
                reduce_shares = int(reduce_amount_usd / ticker_analysis["current_price"])
                rebalancing["reason"] = f"Over-concentration: {ticker} is {max_weight*100:.1f}% of portfolio"
                rebalancing["recommendations"].append({
                    "action": "REDUCE",
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
                    "action": "REDUCE",
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
                rebalancing["recommendations"].append({
                    "action": "CONSIDER_SELL",
                    "ticker": analysis["ticker"],
                    "sell_amount_usd": sell_value_usd,
                    "sell_shares": sell_shares,
                    "current_price_usd": analysis["current_price"],
                    "reason": f"Low recommendation score: {analysis['recommendation_score']:.1f} - Consider selling {sell_shares} shares (${sell_value_usd:,.2f})"
                })
        
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
        
        # Get current prices
        tickers = [h["ticker"] for h in portfolio["holdings"]]
        prices = self.get_current_prices(tickers)
        
        # Analyze each holding
        analyses = []
        for holding in portfolio["holdings"]:
            ticker = holding["ticker"]
            quantity = holding["quantity"]
            current_price = prices.get(ticker, holding.get("last_price", 0))
            
            if current_price > 0:
                analysis = self.analyze_holding(ticker, quantity, current_price)
                analyses.append(analysis)
        
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
        
        return results
    
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
        
        print("\n" + "-" * 60)
        print("REBALANCING RECOMMENDATION")
        print("-" * 60)
        
        rebalancing = results["rebalancing"]
        if rebalancing["needed"]:
            print("⚠️  REBALANCING IS RECOMMENDED")
            print(f"Reason: {rebalancing['reason']}")
            if rebalancing["recommendations"]:
                print("\nSpecific Recommendations (All amounts in USD):")
                for rec in rebalancing["recommendations"]:
                    print(f"\n  {rec['action']}: {rec['ticker']}")
                    if 'reduce_amount_usd' in rec:
                        print(f"    Sell: {rec['reduce_shares']} shares at ${rec['current_price_usd']:.2f} = ${rec['reduce_amount_usd']:,.2f}")
                    elif 'sell_amount_usd' in rec:
                        print(f"    Sell: {rec['sell_shares']} shares at ${rec['current_price_usd']:.2f} = ${rec['sell_amount_usd']:,.2f}")
                    print(f"    Reason: {rec['reason']}")
        else:
            print("✅ Portfolio is well-balanced. No rebalancing needed at this time.")
        
        print("\n" + "=" * 60)
        print(f"Analysis completed at: {results['timestamp']}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer()
    analyzer.analyze()

