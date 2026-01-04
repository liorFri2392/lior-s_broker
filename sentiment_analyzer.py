#!/usr/bin/env python3
"""
Advanced Sentiment Analyzer - Improved sentiment analysis using multiple methods
"""

import re
from typing import Dict, List, Optional
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial news and data."""
    
    def __init__(self):
        # Enhanced word lists
        self.positive_words = [
            'gain', 'rise', 'up', 'growth', 'profit', 'beat', 'strong', 'bullish', 'surge',
            'rally', 'soar', 'jump', 'climb', 'advance', 'boost', 'outperform', 'exceed',
            'breakthrough', 'milestone', 'record', 'high', 'momentum', 'optimistic', 'positive'
        ]
        
        self.negative_words = [
            'fall', 'drop', 'down', 'loss', 'miss', 'weak', 'bearish', 'decline', 'crash',
            'plunge', 'slump', 'dive', 'tumble', 'collapse', 'concern', 'worry', 'risk',
            'uncertainty', 'volatile', 'pressure', 'challenge', 'headwind', 'disappoint'
        ]
        
        # Financial-specific terms
        self.financial_positive = [
            'earnings beat', 'revenue growth', 'margin expansion', 'guidance raise',
            'upgrade', 'buy rating', 'outperform', 'strong fundamentals'
        ]
        
        self.financial_negative = [
            'earnings miss', 'revenue decline', 'margin compression', 'guidance cut',
            'downgrade', 'sell rating', 'underperform', 'weak fundamentals'
        ]
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using enhanced word matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dict with sentiment score and analysis
        """
        if not text:
            return {"score": 50, "sentiment": "NEUTRAL", "confidence": 0}
        
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Check for financial-specific terms (weighted higher)
        financial_positive = sum(1 for term in self.financial_positive if term in text_lower)
        financial_negative = sum(1 for term in self.financial_negative if term in text_lower)
        
        # Calculate score (0-100, 50 is neutral)
        base_score = 50
        
        # Word-based sentiment
        word_sentiment = (positive_count - negative_count) * 3
        base_score += word_sentiment
        
        # Financial term sentiment (weighted 2x)
        financial_sentiment = (financial_positive - financial_negative) * 6
        base_score += financial_sentiment
        
        # Clamp to 0-100
        score = max(0, min(100, base_score))
        
        # Determine sentiment
        if score >= 65:
            sentiment = "POSITIVE"
        elif score <= 35:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        # Calculate confidence based on word count
        total_words = positive_count + negative_count + financial_positive + financial_negative
        confidence = min(100, total_words * 10)
        
        return {
            "score": score,
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_signals": positive_count + financial_positive,
            "negative_signals": negative_count + financial_negative
        }
    
    def analyze_news_sentiment(self, ticker: str, max_articles: int = 10) -> Dict:
        """
        Analyze sentiment from news articles for a ticker.
        
        Args:
            ticker: Stock/ETF ticker
            max_articles: Maximum number of articles to analyze
        
        Returns:
            Dict with aggregated sentiment
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return {
                    "ticker": ticker,
                    "score": 50,
                    "sentiment": "NEUTRAL",
                    "articles_analyzed": 0,
                    "confidence": 0
                }
            
            # Analyze each article
            sentiments = []
            for article in news[:max_articles]:
                title = article.get('title', '')
                summary = article.get('summary', '')
                text = f"{title} {summary}"
                
                sentiment = self.analyze_text_sentiment(text)
                sentiments.append(sentiment)
            
            if not sentiments:
                return {
                    "ticker": ticker,
                    "score": 50,
                    "sentiment": "NEUTRAL",
                    "articles_analyzed": 0,
                    "confidence": 0
                }
            
            # Aggregate sentiment
            avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
            avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
            
            # Count sentiment distribution
            positive_count = sum(1 for s in sentiments if s['sentiment'] == 'POSITIVE')
            negative_count = sum(1 for s in sentiments if s['sentiment'] == 'NEGATIVE')
            
            # Determine overall sentiment
            if avg_score >= 60:
                overall_sentiment = "POSITIVE"
            elif avg_score <= 40:
                overall_sentiment = "NEGATIVE"
            else:
                overall_sentiment = "NEUTRAL"
            
            return {
                "ticker": ticker,
                "score": avg_score,
                "sentiment": overall_sentiment,
                "articles_analyzed": len(sentiments),
                "confidence": avg_confidence,
                "positive_articles": positive_count,
                "negative_articles": negative_count,
                "neutral_articles": len(sentiments) - positive_count - negative_count
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze news sentiment for {ticker}: {e}")
            return {
                "ticker": ticker,
                "score": 50,
                "sentiment": "NEUTRAL",
                "error": str(e)
            }
    
    def analyze_multiple_tickers(self, tickers: List[str]) -> Dict:
        """Analyze sentiment for multiple tickers."""
        results = {}
        
        for ticker in tickers:
            results[ticker] = self.analyze_news_sentiment(ticker)
        
        # Calculate portfolio-level sentiment
        scores = [r['score'] for r in results.values() if 'score' in r]
        if scores:
            portfolio_sentiment = sum(scores) / len(scores)
        else:
            portfolio_sentiment = 50
        
        return {
            "individual_sentiments": results,
            "portfolio_sentiment": portfolio_sentiment,
            "portfolio_sentiment_label": "POSITIVE" if portfolio_sentiment >= 60 else ("NEGATIVE" if portfolio_sentiment <= 40 else "NEUTRAL")
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test
    result = analyzer.analyze_news_sentiment("SPY")
    print(f"Sentiment for SPY: {result}")

