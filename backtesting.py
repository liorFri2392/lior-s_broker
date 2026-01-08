#!/usr/bin/env python3
"""
Backtesting Module - Test portfolio strategies on historical data
Simulates trading strategies and analyzes performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import json
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """Backtest trading strategies on historical data."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
    
    def backtest_strategy(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        strategy: str = "buy_and_hold",
        rebalance_frequency: str = "monthly",
        allocation: Dict[str, float] = None
    ) -> Dict:
        """
        Backtest a strategy on historical data.
        
        Args:
            tickers: List of tickers to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy: Strategy type ("buy_and_hold", "rebalance", "momentum")
            rebalance_frequency: "daily", "weekly", "monthly", "quarterly"
            allocation: Dict of {ticker: weight} for rebalancing
        
        Returns:
            Dict with performance metrics
        """
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {strategy.upper()} Strategy")
        print(f"{'='*60}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Tickers: {', '.join(tickers)}\n")
        
        # Get historical data
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    data[ticker] = hist
                    print(f"✅ Loaded {len(hist)} days of data for {ticker}")
                else:
                    print(f"❌ No data for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to get data for {ticker}: {e}")
                continue
        
        if not data:
            return {"error": "No data available for backtesting"}
        
        # Align dates
        common_dates = None
        for ticker, df in data.items():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        if len(common_dates) == 0:
            return {"error": "No common dates found"}
        
        # Run strategy
        if strategy == "buy_and_hold":
            results = self._backtest_buy_and_hold(data, common_dates, allocation)
        elif strategy == "rebalance":
            results = self._backtest_rebalance(data, common_dates, rebalance_frequency, allocation)
        elif strategy == "momentum":
            results = self._backtest_momentum(data, common_dates)
        else:
            return {"error": f"Unknown strategy: {strategy}"}
        
        # Calculate metrics
        metrics = self._calculate_metrics(results, start_date, end_date)
        results.update(metrics)
        
        return results
    
    def _backtest_buy_and_hold(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
        allocation: Dict[str, float] = None
    ) -> Dict:
        """Backtest buy and hold strategy."""
        tickers = list(data.keys())
        
        # Equal allocation if not specified
        if allocation is None:
            allocation = {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        # Calculate initial positions
        positions = {}
        prices_start = {}
        for ticker in tickers:
            if ticker in data and len(data[ticker]) > 0:
                prices_start[ticker] = data[ticker]['Close'].iloc[0]
                if prices_start[ticker] > 0:
                    shares = int((self.initial_capital * allocation.get(ticker, 0)) / prices_start[ticker])
                    positions[ticker] = shares
        
        # Track portfolio value over time
        portfolio_values = []
        daily_returns = []
        
        for date in dates:
            portfolio_value = 0
            for ticker, shares in positions.items():
                if ticker in data and date in data[ticker].index:
                    price = data[ticker].loc[date, 'Close']
                    portfolio_value += shares * price
            
            portfolio_values.append(portfolio_value)
            if len(portfolio_values) > 1:
                daily_return = (portfolio_value / portfolio_values[-2] - 1) * 100
                daily_returns.append(daily_return)
        
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        return {
            "strategy": "buy_and_hold",
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "positions": positions,
            "dates": dates.tolist()
        }
    
    def _backtest_rebalance(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
        frequency: str,
        allocation: Dict[str, float] = None
    ) -> Dict:
        """Backtest rebalancing strategy."""
        tickers = list(data.keys())
        
        if allocation is None:
            allocation = {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        # Determine rebalance dates
        if frequency == "daily":
            rebalance_dates = dates
        elif frequency == "weekly":
            rebalance_dates = dates[dates.weekday == 0]  # Mondays
        elif frequency == "monthly":
            rebalance_dates = dates[dates.day == 1]  # First of month
        elif frequency == "quarterly":
            rebalance_dates = dates[(dates.month % 3 == 1) & (dates.day == 1)]
        else:
            rebalance_dates = dates
        
        portfolio_value = self.initial_capital
        portfolio_values = [portfolio_value]
        daily_returns = []
        positions = {}
        
        for i, date in enumerate(dates):
            # Rebalance if needed
            if date in rebalance_dates or i == 0:
                # Calculate new positions based on allocation
                for ticker in tickers:
                    if ticker in data and date in data[ticker].index:
                        price = data[ticker].loc[date, 'Close']
                        if price > 0:
                            target_value = portfolio_value * allocation.get(ticker, 0)
                            positions[ticker] = int(target_value / price)
            
            # Calculate current portfolio value
            current_value = 0
            for ticker, shares in positions.items():
                if ticker in data and date in data[ticker].index:
                    price = data[ticker].loc[date, 'Close']
                    current_value += shares * price
            
            portfolio_value = current_value
            portfolio_values.append(portfolio_value)
            
            if len(portfolio_values) > 1:
                daily_return = (portfolio_value / portfolio_values[-2] - 1) * 100
                daily_returns.append(daily_return)
        
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        return {
            "strategy": "rebalance",
            "rebalance_frequency": frequency,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "dates": dates.tolist()
        }
    
    def _backtest_momentum(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex
    ) -> Dict:
        """Backtest momentum strategy - buy winners, sell losers."""
        tickers = list(data.keys())
        
        portfolio_value = self.initial_capital
        portfolio_values = [portfolio_value]
        daily_returns = []
        positions = {}
        lookback_period = 20  # 20 days for momentum
        
        for i, date in enumerate(dates):
            if i < lookback_period:
                continue
            
            # Calculate momentum for each ticker
            momentum_scores = {}
            for ticker in tickers:
                if ticker in data and date in data[ticker].index:
                    try:
                        past_date_idx = data[ticker].index.get_loc(date) - lookback_period
                        if past_date_idx >= 0:
                            past_price = data[ticker]['Close'].iloc[past_date_idx]
                            current_price = data[ticker].loc[date, 'Close']
                            momentum = (current_price / past_price - 1) * 100
                            momentum_scores[ticker] = momentum
                    except (KeyError, IndexError):
                        continue
            
            # Buy top momentum tickers
            if momentum_scores:
                sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                top_tickers = [t[0] for t in sorted_tickers[:3]]  # Top 3
                
                # Equal allocation to top tickers
                allocation_per_ticker = 1.0 / len(top_tickers)
                positions = {}
                for ticker in top_tickers:
                    if ticker in data and date in data[ticker].index:
                        price = data[ticker].loc[date, 'Close']
                        if price > 0:
                            target_value = portfolio_value * allocation_per_ticker
                            positions[ticker] = int(target_value / price)
            
            # Calculate current portfolio value
            current_value = 0
            for ticker, shares in positions.items():
                if ticker in data and date in data[ticker].index:
                    price = data[ticker].loc[date, 'Close']
                    current_value += shares * price
            
            portfolio_value = current_value
            portfolio_values.append(portfolio_value)
            
            if len(portfolio_values) > 1:
                daily_return = (portfolio_value / portfolio_values[-2] - 1) * 100
                daily_returns.append(daily_return)
        
        final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        return {
            "strategy": "momentum",
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "dates": dates.tolist()
        }
    
    def _calculate_metrics(self, results: Dict, start_date: str, end_date: str) -> Dict:
        """Calculate performance metrics."""
        if "daily_returns" not in results or not results["daily_returns"]:
            return {}
        
        returns = np.array(results["daily_returns"])
        
        # Annualized return
        days = len(returns)
        years = days / 252
        if years > 0:
            total_return = results.get("total_return", 0)
            annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 3% risk-free rate)
        risk_free_rate = 3.0
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Max drawdown
        portfolio_values = results.get("portfolio_values", [])
        if portfolio_values:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak * 100
            max_drawdown = np.min(drawdown)
        else:
            max_drawdown = 0
        
        # Win rate
        positive_days = (returns > 0).sum()
        win_rate = (positive_days / len(returns)) * 100 if len(returns) > 0 else 0
        
        # Best and worst day
        best_day = np.max(returns) if len(returns) > 0 else 0
        worst_day = np.min(returns) if len(returns) > 0 else 0
        
        return {
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "best_day": best_day,
            "worst_day": worst_day,
            "total_days": days
        }
    
    def print_results(self, results: Dict):
        """Print backtesting results in a formatted way."""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print(f"\n{'='*60}")
        print("BACKTESTING RESULTS")
        print(f"{'='*60}\n")
        
        print(f"Strategy: {results.get('strategy', 'N/A').upper()}")
        if 'rebalance_frequency' in results:
            print(f"Rebalance Frequency: {results['rebalance_frequency']}")
        
        print(f"\n💰 Performance:")
        print(f"   Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        print(f"   Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"   Total Return: {results.get('total_return', 0):.2f}%")
        print(f"   Annualized Return: {results.get('annualized_return', 0):.2f}%")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Volatility: {results.get('volatility', 0):.2f}%")
        print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"   Win Rate: {results.get('win_rate', 0):.1f}%")
        
        print(f"\n📈 Daily Performance:")
        print(f"   Best Day: {results.get('best_day', 0):.2f}%")
        print(f"   Worst Day: {results.get('worst_day', 0):.2f}%")
        print(f"   Total Trading Days: {results.get('total_days', 0)}")
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Example usage
    backtester = Backtester(initial_capital=10000)
    
    # Test buy and hold
    results = backtester.backtest_strategy(
        tickers=["SPY", "VXUS", "BND"],
        start_date="2020-01-01",
        end_date="2023-12-31",
        strategy="buy_and_hold",
        allocation={"SPY": 0.5, "VWO": 0.3, "BND": 0.2}
    )
    
    backtester.print_results(results)

