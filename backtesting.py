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
import market_data
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
        allocation: Dict[str, float] = None,
        monthly_deposit: float = 0.0,
    ) -> Dict:
        """
        Backtest a strategy on historical data.

        Args:
            tickers: List of tickers to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            strategy: Strategy type ("buy_and_hold", "rebalance", "momentum",
                "monthly_deposit" - DCA using the app's actual gap-fill allocator)
            rebalance_frequency: "daily", "weekly", "monthly", "quarterly"
            allocation: Dict of {ticker: weight} for rebalancing
            monthly_deposit: USD added on the first trading day of each month
                (monthly_deposit strategy only)

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
        #
        # Backtests need an explicit start/end window, which the shared
        # market_data cache keys on `period` rather than date ranges — so we
        # issue ONE batched yf.download for the whole ticker set (instead of a
        # sequential yf.Ticker.history() per ticker) and slice per ticker.
        data = {}
        try:
            raw = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.warning(f"Batched download failed: {e}")
            raw = None

        for ticker in tickers:
            hist = pd.DataFrame()
            try:
                if raw is not None and not raw.empty:
                    if len(tickers) == 1:
                        hist = raw.dropna(how="all")
                    elif isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                        hist = raw[ticker].dropna(how="all")
            except Exception:
                hist = pd.DataFrame()

            if hist is None or hist.empty:
                # Fallback to a per-ticker fetch for odd/delisted symbols.
                try:
                    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
                except Exception as e:
                    logger.warning(f"Failed to get data for {ticker}: {e}")
                    hist = pd.DataFrame()

            if hist is not None and not hist.empty:
                data[ticker] = hist
                print(f"✅ Loaded {len(hist)} days of data for {ticker}")
            else:
                print(f"❌ No data for {ticker}")

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

        # Build ONE aligned close-price matrix (dates × tickers) up front. All
        # strategies do their daily mark-to-market against this via vectorized
        # matrix ops instead of per-cell .loc lookups inside a Python loop.
        prices = pd.concat(
            {t: data[t]['Close'] for t in data},
            axis=1,
        ).reindex(common_dates).ffill()

        # Run strategy
        if strategy == "buy_and_hold":
            results = self._backtest_buy_and_hold(data, common_dates, allocation, prices)
        elif strategy == "rebalance":
            results = self._backtest_rebalance(data, common_dates, rebalance_frequency, allocation, prices)
        elif strategy == "momentum":
            results = self._backtest_momentum(data, common_dates, prices)
        elif strategy == "monthly_deposit":
            results = self._backtest_monthly_deposit(common_dates, prices, monthly_deposit)
        else:
            return {"error": f"Unknown strategy: {strategy}"}

        # Calculate metrics
        metrics = self._calculate_metrics(results, start_date, end_date)
        results.update(metrics)

        # For DCA, the lump-sum annualization of total_return is meaningless;
        # the money-weighted (XIRR) figure computed by the strategy wins.
        if "money_weighted_annual_return" in results:
            results["annualized_return"] = results["money_weighted_annual_return"]

        return results

    def _backtest_monthly_deposit(
        self,
        dates: pd.DatetimeIndex,
        prices: pd.DataFrame,
        monthly_deposit: float,
    ) -> Dict:
        """DCA backtest of the app's ACTUAL strategy: on the first trading day
        of each month, deposit cash and allocate it with the same target-weight
        gap-filling engine `make deposit` uses (allocation.gap_fill_allocate).

        Returns deposit-adjusted performance: total_return is gain over money
        put in, and money_weighted_annual_return is the XIRR of the flows.
        """
        import allocation as alloc
        import ledger

        tickers = list(prices.columns)
        price_mat = prices.to_numpy(dtype=float)
        dates_series = pd.Series(dates, index=dates)
        deposit_dates = set(pd.DatetimeIndex(
            dates_series.groupby(dates.to_period('M')).head(1).index
        ))

        cash = float(self.initial_capital)
        positions: Dict[str, int] = {t: 0 for t in tickers}
        portfolio_values: List[float] = []
        daily_returns: List[float] = []
        flows: List[Dict] = []
        total_costs = 0.0
        deposits_made = 0
        prev_value = None
        first_deposit = True

        for i, date in enumerate(dates):
            row = price_mat[i]
            day_prices = {t: float(row[j]) for j, t in enumerate(tickers)
                          if np.isfinite(row[j]) and row[j] > 0}

            flow = 0.0
            if date in deposit_dates:
                if first_deposit:
                    flow = float(self.initial_capital)
                    first_deposit = False
                else:
                    cash += monthly_deposit
                    flow = float(monthly_deposit)
                    deposits_made += 1
                flows.append({"date": pd.Timestamp(date).to_pydatetime(), "amount": flow})

                if len(tickers) == 1:
                    # Single-ticker benchmark mode: all-in DCA (the gap-fill
                    # engine would cap one ticker at its group's target weight).
                    t = tickers[0]
                    p = day_prices.get(t, 0)
                    shares = int(cash // p) if p > 0 else 0
                    if shares > 0:
                        notional = shares * p
                        cost = self._trade_cost(notional, self.slippage_bps,
                                                self.commission_per_trade)
                        cash -= notional + cost
                        total_costs += cost
                        positions[t] += shares
                else:
                    holdings = [
                        {"ticker": t, "quantity": q,
                         "last_price": day_prices.get(t, 0),
                         "current_value": q * day_prices.get(t, 0)}
                        for t, q in positions.items() if q > 0
                    ]
                    for b in alloc.gap_fill_allocate(holdings, cash, day_prices):
                        cost = self._trade_cost(b["amount"], self.slippage_bps,
                                                self.commission_per_trade)
                        cash -= b["amount"] + cost
                        total_costs += cost
                        positions[b["ticker"]] = positions.get(b["ticker"], 0) + b["shares"]

            value = cash + sum(q * day_prices.get(t, 0) for t, q in positions.items())
            # Deposit-adjusted daily return: strip the same-day inflow so a
            # deposit doesn't register as a +% "market" day.
            if prev_value is not None and prev_value > 0:
                daily_returns.append(((value - flow) / prev_value - 1) * 100)
            portfolio_values.append(value)
            prev_value = value

        final_value = portfolio_values[-1] if portfolio_values else float(self.initial_capital)
        net_invested = float(self.initial_capital) + deposits_made * float(monthly_deposit)
        total_return = (final_value / net_invested - 1) * 100 if net_invested > 0 else 0.0

        xirr_pct = None
        if flows and len(dates) > 0:
            r = ledger._xirr(flows, final_value, pd.Timestamp(dates[-1]).to_pydatetime())
            if r is not None:
                xirr_pct = r * 100.0

        results = {
            "strategy": "monthly_deposit",
            "initial_capital": self.initial_capital,
            "monthly_deposit": monthly_deposit,
            "deposits_made": deposits_made,
            "net_invested": round(net_invested, 2),
            "final_value": final_value,
            "total_return": total_return,  # gain over money put in
            "portfolio_values": portfolio_values,
            "daily_returns": daily_returns,
            "positions": {t: q for t, q in positions.items() if q > 0},
            "dates": dates.tolist(),
            "total_trading_costs": round(total_costs, 2),
            "rebalances_executed": deposits_made + 1,
        }
        if xirr_pct is not None:
            results["money_weighted_annual_return"] = xirr_pct
        return results
    
    def _backtest_buy_and_hold(
        self,
        data: Dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
        allocation: Dict[str, float] = None,
        prices: pd.DataFrame = None
    ) -> Dict:
        """Backtest buy and hold strategy (vectorized mark-to-market)."""
        tickers = list(data.keys())

        # Equal allocation if not specified
        if allocation is None:
            allocation = {ticker: 1.0 / len(tickers) for ticker in tickers}

        if prices is None:
            prices = pd.concat({t: data[t]['Close'] for t in data}, axis=1).reindex(dates).ffill()
        prices = prices[tickers]

        cash = float(self.initial_capital)
        total_costs = 0.0
        prices_start = prices.iloc[0]
        shares_vec = np.zeros(len(tickers))
        for j, ticker in enumerate(tickers):
            p0 = float(prices_start[ticker])
            if p0 > 0:
                target_notional = self.initial_capital * allocation.get(ticker, 0)
                shares = int(target_notional / p0)
                shares_vec[j] = shares
                actual_notional = shares * p0
                cost = self._trade_cost(actual_notional, self.slippage_bps,
                                        self.commission_per_trade if shares > 0 else 0.0)
                cash -= actual_notional + cost
                total_costs += cost

        # Vectorized daily mark-to-market: (dates × tickers) @ shares + cash
        marks = prices.to_numpy(dtype=float) @ shares_vec
        portfolio_values_arr = marks + cash
        portfolio_values = portfolio_values_arr.tolist()

        # Daily % returns where the prior day's value was positive
        daily_returns = []
        if len(portfolio_values_arr) > 1:
            prev = portfolio_values_arr[:-1]
            curr = portfolio_values_arr[1:]
            mask = prev > 0
            rets = np.where(mask, (curr / prev - 1) * 100, 0.0)
            daily_returns = rets[mask].tolist()

        positions = {ticker: int(shares_vec[j]) for j, ticker in enumerate(tickers)}
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
        allocation: Dict[str, float] = None,
        prices: pd.DataFrame = None
    ) -> Dict:
        """Backtest rebalancing strategy."""
        tickers = list(data.keys())

        if allocation is None:
            allocation = {ticker: 1.0 / len(tickers) for ticker in tickers}

        # Aligned price matrix → positional row lookups instead of per-cell
        # .loc inside the daily loop.
        if prices is None:
            prices = pd.concat({t: data[t]['Close'] for t in data}, axis=1).reindex(dates).ffill()
        prices = prices[tickers]
        price_mat = prices.to_numpy(dtype=float)

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
        rebalance_set = set(rebalance_dates)

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
            row = price_mat[i]
            current_prices = {
                t: row[j]
                for j, t in enumerate(tickers) if np.isfinite(row[j])
            }

            if (date in rebalance_set or i == 0) and current_prices:
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
        dates: pd.DatetimeIndex,
        prices: pd.DataFrame = None
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

        # Aligned price matrix + fully vectorized momentum: the lookback return
        # for every (date, ticker) is prices / prices.shift(lookback) - 1.
        if prices is None:
            prices = pd.concat({t: data[t]['Close'] for t in data}, axis=1).reindex(dates).ffill()
        prices = prices[tickers]
        price_mat = prices.to_numpy(dtype=float)
        momentum_mat = (prices / prices.shift(lookback_period) - 1.0).to_numpy(dtype=float) * 100.0

        for i, date in enumerate(dates):
            if i < lookback_period:
                continue

            row = price_mat[i]
            current_prices = {
                t: row[j]
                for j, t in enumerate(tickers) if np.isfinite(row[j])
            }

            mom_row = momentum_mat[i]
            momentum_scores = {
                t: mom_row[j]
                for j, t in enumerate(tickers)
                if np.isfinite(mom_row[j]) and np.isfinite(row[j]) and row[j] > 0
            }

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

        # Annualized return. Base the elapsed time on actual CALENDAR span
        # (start→end), not the count of daily-return observations — momentum
        # drops its lookback warm-up days, which would otherwise shrink the
        # denominator and inflate the annualized figure.
        days = len(returns)
        try:
            span_days = (datetime.strptime(end_date, "%Y-%m-%d")
                         - datetime.strptime(start_date, "%Y-%m-%d")).days
            years = span_days / 365.25 if span_days > 0 else days / 252
        except Exception:
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
        if results.get('strategy') == 'monthly_deposit':
            print(f"   Monthly Deposit: ${results.get('monthly_deposit', 0):,.2f}"
                  f" x {results.get('deposits_made', 0)} deposits")
            print(f"   Money Put In: ${results.get('net_invested', 0):,.2f}")
        print(f"   Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"   Total Return: {results.get('total_return', 0):.2f}%"
              + (" (gain over money put in)" if results.get('strategy') == 'monthly_deposit' else ""))
        if 'money_weighted_annual_return' in results:
            print(f"   Money-Weighted Annual Return (XIRR): {results['money_weighted_annual_return']:.2f}%")
        else:
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
    # Backtest the app's ACTUAL strategy: monthly deposits allocated by the
    # gap-fill engine over the model portfolio, vs a 100% S&P DCA benchmark.
    import allocation as alloc

    START, END = "2019-01-01", "2024-12-31"
    INITIAL, MONTHLY = 1000, 2000

    model_tickers = sorted({g["tickers"][0] for g in alloc.TARGET_GROUPS})

    backtester = Backtester(initial_capital=INITIAL)
    results = backtester.backtest_strategy(
        tickers=model_tickers,
        start_date=START,
        end_date=END,
        strategy="monthly_deposit",
        monthly_deposit=MONTHLY,
    )
    backtester.print_results(results)

    print("\n--- Benchmark: same deposits, all-in 100% SPY DCA ---")
    bench = Backtester(initial_capital=INITIAL)
    bench_results = bench.backtest_strategy(
        tickers=["SPY"],
        start_date=START,
        end_date=END,
        strategy="monthly_deposit",
        monthly_deposit=MONTHLY,
    )
    bench.print_results(bench_results)

