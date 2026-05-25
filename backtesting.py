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

    def __init__(
        self,
        initial_capital: float = 10000,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 5.0,
    ):
        """
        commission_per_trade: flat $ commission per buy/sell (most US brokers
            now charge $0; IB/Schwab/Fidelity/Robinhood ≈ 0).
        slippage_bps: round-trip slippage in basis points (5 bps = 0.05% per
            trade is reasonable for liquid ETFs; less liquid funds 10–25 bps).
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = float(commission_per_trade)
        self.slippage_bps = float(slippage_bps)
        self.results = {}

    @staticmethod
    def _trade_cost(notional: float, slippage_bps: float, commission: float) -> float:
        """Cost of executing a trade with given notional value.

        slippage_bps is applied to the absolute traded notional (one-side).
        """
        return abs(notional) * (slippage_bps / 10_000.0) + commission

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
        
        # Get historical data with auto_adjust=True so 'Close' is the dividend-
        # and split-adjusted total-return series. Without this, dividend-paying
        # ETFs (SPY ~1.5%/yr, BND ~3%/yr) understate total return by 5–10% per
        # year of backtest.
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
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
        
        cash = float(self.initial_capital)
        positions = {}
        prices_start = {}
        total_costs = 0.0
        for ticker in tickers:
            if ticker in data and len(data[ticker]) > 0:
                prices_start[ticker] = data[ticker]['Close'].iloc[0]
                if prices_start[ticker] > 0:
                    target_notional = self.initial_capital * allocation.get(ticker, 0)
                    shares = int(target_notional / prices_start[ticker])
                    positions[ticker] = shares
                    actual_notional = shares * prices_start[ticker]
                    cost = self._trade_cost(actual_notional, self.slippage_bps,
                                            self.commission_per_trade if shares > 0 else 0.0)
                    cash -= actual_notional + cost
                    total_costs += cost

        portfolio_values = []
        daily_returns = []

        for date in dates:
            mark = 0.0
            for ticker, shares in positions.items():
                if ticker in data and date in data[ticker].index:
                    mark += shares * data[ticker].loc[date, 'Close']
            portfolio_value = cash + mark
            portfolio_values.append(portfolio_value)
            if len(portfolio_values) > 1 and portfolio_values[-2] > 0:
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
            "dates": dates.tolist(),
            "total_trading_costs": round(total_costs, 2),
            "rebalances_executed": 1,
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
        
        # Determine rebalance dates. The previous version filtered by `day == 1`,
        # which misses ~35% of months because the calendar 1st is usually a
        # weekend/holiday and equity markets are closed. We instead take the
        # FIRST trading day of each period — which is what a real rebalancing
        # schedule does.
        dates_series = pd.Series(dates, index=dates)
        if frequency == "daily":
            rebalance_dates = dates
        elif frequency == "weekly":
            rebalance_dates = dates_series.groupby(dates.to_period('W')).head(1).index
        elif frequency == "monthly":
            rebalance_dates = dates_series.groupby(dates.to_period('M')).head(1).index
        elif frequency == "quarterly":
            rebalance_dates = dates_series.groupby(dates.to_period('Q')).head(1).index
        else:
            rebalance_dates = dates
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
        
        # Track cash + share positions explicitly. Every trade debits cash by
        # |Δshares|·price for shares purchased (or credits for sales), AND by
        # the trading cost (slippage + commission). The portfolio value is
        # always cash + Σ positions·price.
        cash = float(self.initial_capital)
        positions = {ticker: 0 for ticker in tickers}
        portfolio_values = []
        daily_returns = []
        total_costs = 0.0
        rebalances_executed = 0

        for i, date in enumerate(dates):
            current_prices = {
                t: data[t].loc[date, 'Close']
                for t in tickers if t in data and date in data[t].index
            }

            if (date in rebalance_dates or i == 0) and current_prices:
                # Current portfolio value at pre-trade prices
                mtm = cash + sum(positions[t] * current_prices.get(t, 0) for t in tickers)

                new_positions = {}
                for t in tickers:
                    p = current_prices.get(t, 0)
                    if p > 0:
                        target_val = mtm * allocation.get(t, 0)
                        new_positions[t] = int(target_val / p)
                    else:
                        new_positions[t] = positions.get(t, 0)

                rebal_cost = 0.0
                trades_executed = 0
                net_cash_change = 0.0
                for t in tickers:
                    delta = new_positions[t] - positions.get(t, 0)
                    if delta != 0 and current_prices.get(t, 0) > 0:
                        notional = delta * current_prices[t]
                        rebal_cost += self._trade_cost(notional, self.slippage_bps, self.commission_per_trade)
                        net_cash_change -= notional      # buying debits cash; selling credits
                        trades_executed += 1
                cash = cash + net_cash_change - rebal_cost
                positions = new_positions
                total_costs += rebal_cost
                if trades_executed > 0:
                    rebalances_executed += 1

            portfolio_value = cash + sum(positions[t] * current_prices.get(t, 0) for t in tickers)
            portfolio_values.append(portfolio_value)
            if len(portfolio_values) > 1 and portfolio_values[-2] > 0:
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
            "dates": dates.tolist(),
            "total_trading_costs": round(total_costs, 2),
            "rebalances_executed": rebalances_executed,
        }
    
    def _backtest_momentum(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex
    ) -> Dict:
        """Backtest momentum strategy - buy winners, sell losers."""
        tickers = list(data.keys())
        
        cash = float(self.initial_capital)
        positions = {t: 0 for t in tickers}
        portfolio_values = []
        daily_returns = []
        total_costs = 0.0
        rebalances_executed = 0
        lookback_period = 20

        for i, date in enumerate(dates):
            if i < lookback_period:
                continue

            current_prices = {
                t: data[t].loc[date, 'Close']
                for t in tickers if t in data and date in data[t].index
            }

            momentum_scores = {}
            for ticker in tickers:
                if ticker in data and date in data[ticker].index:
                    try:
                        past_idx = data[ticker].index.get_loc(date) - lookback_period
                        if past_idx >= 0:
                            past_price = data[ticker]['Close'].iloc[past_idx]
                            cur_price = current_prices.get(ticker, 0)
                            if past_price > 0 and cur_price > 0:
                                momentum_scores[ticker] = (cur_price / past_price - 1) * 100
                    except (KeyError, IndexError):
                        continue

            if momentum_scores:
                mtm = cash + sum(positions[t] * current_prices.get(t, 0) for t in tickers)
                sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
                top_tickers = [t[0] for t in sorted_tickers[:3]]
                alloc_each = 1.0 / len(top_tickers)

                new_positions = {t: 0 for t in tickers}
                for ticker in top_tickers:
                    p = current_prices.get(ticker, 0)
                    if p > 0:
                        new_positions[ticker] = int((mtm * alloc_each) / p)

                rebal_cost = 0.0
                trades_executed = 0
                net_cash_change = 0.0
                for t in tickers:
                    delta = new_positions[t] - positions.get(t, 0)
                    if delta != 0 and current_prices.get(t, 0) > 0:
                        notional = delta * current_prices[t]
                        rebal_cost += self._trade_cost(notional, self.slippage_bps, self.commission_per_trade)
                        net_cash_change -= notional
                        trades_executed += 1
                cash = cash + net_cash_change - rebal_cost
                positions = new_positions
                total_costs += rebal_cost
                if trades_executed > 0:
                    rebalances_executed += 1

            portfolio_value = cash + sum(positions[t] * current_prices.get(t, 0) for t in tickers)
            portfolio_values.append(portfolio_value)
            if len(portfolio_values) > 1 and portfolio_values[-2] > 0:
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
            "dates": dates.tolist(),
            "total_trading_costs": round(total_costs, 2),
            "rebalances_executed": rebalances_executed,
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

        # Sharpe ratio with dynamic risk-free rate from 3-month T-bill yield
        # (^IRX), consistent with the rest of the codebase.
        try:
            from advanced_analysis import AdvancedAnalyzer
            risk_free_rate = AdvancedAnalyzer().get_risk_free_rate() * 100.0
        except Exception:
            risk_free_rate = 4.5  # Fallback consistent with advanced_analysis default
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

        if 'total_trading_costs' in results:
            print(f"\n💸 Costs:")
            print(f"   Total Trading Costs: ${results['total_trading_costs']:,.2f}")
            print(f"   Rebalances Executed: {results.get('rebalances_executed', 0)}")
            cost_drag = results['total_trading_costs'] / max(results.get('initial_capital', 1), 1) * 100
            print(f"   Cost Drag (vs initial capital): {cost_drag:.2f}%")
        
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

