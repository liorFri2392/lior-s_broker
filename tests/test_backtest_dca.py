"""Offline test for the monthly_deposit (DCA gap-fill) backtest strategy.

Exercises _backtest_monthly_deposit directly with a synthetic price matrix -
no network, no yf.download.
"""
import numpy as np
import pandas as pd
import pytest

from backtesting import Backtester


def _flat_prices(tickers, price=100.0, days=260, start="2024-01-02"):
    dates = pd.bdate_range(start=start, periods=days)
    return pd.DataFrame({t: np.full(days, price) for t in tickers}, index=dates)


def test_dca_flat_market_returns_zero():
    """Flat prices, zero costs: final value == money put in, XIRR ~ 0."""
    bt = Backtester(initial_capital=1000, commission_per_trade=0.0, slippage_bps=0.0)
    prices = _flat_prices(["SPY"], price=100.0)
    res = bt._backtest_monthly_deposit(prices.index, prices, monthly_deposit=500.0)
    assert res["net_invested"] == 1000 + res["deposits_made"] * 500
    assert res["final_value"] == pytest.approx(res["net_invested"])
    assert res["total_return"] == pytest.approx(0.0)
    if "money_weighted_annual_return" in res:
        assert abs(res["money_weighted_annual_return"]) < 0.05


def test_dca_deposits_do_not_appear_as_market_returns():
    """Deposit days must not register as +% daily returns."""
    bt = Backtester(initial_capital=1000, commission_per_trade=0.0, slippage_bps=0.0)
    prices = _flat_prices(["SPY"], price=50.0)
    res = bt._backtest_monthly_deposit(prices.index, prices, monthly_deposit=1000.0)
    # Flat market -> every deposit-adjusted daily return is exactly 0.
    assert max(abs(r) for r in res["daily_returns"]) < 1e-9


def test_dca_multi_ticker_uses_gap_fill_targets():
    """With the full model universe, money spreads across groups instead of
    piling into one ticker."""
    import allocation as alloc
    tickers = sorted({g["tickers"][0] for g in alloc.TARGET_GROUPS})
    bt = Backtester(initial_capital=5000, commission_per_trade=0.0, slippage_bps=0.0)
    prices = _flat_prices(tickers, price=50.0, days=300)
    res = bt._backtest_monthly_deposit(prices.index, prices, monthly_deposit=2000.0)
    positions = res["positions"]
    assert len(positions) >= 8  # most groups received money
    # US_CORE (30% target) must hold ~2.5x the value of BONDS_AGG (12% target).
    total = sum(positions.values()) * 50.0
    us_core = positions.get("SPY", 0) * 50.0
    assert us_core / total == pytest.approx(0.30, abs=0.05)


def test_dca_costs_reduce_value():
    bt_free = Backtester(initial_capital=1000, commission_per_trade=0.0, slippage_bps=0.0)
    bt_cost = Backtester(initial_capital=1000, commission_per_trade=1.0, slippage_bps=25.0)
    prices = _flat_prices(["SPY"], price=100.0)
    free = bt_free._backtest_monthly_deposit(prices.index, prices, monthly_deposit=500.0)
    cost = bt_cost._backtest_monthly_deposit(prices.index, prices, monthly_deposit=500.0)
    assert cost["final_value"] < free["final_value"]
    assert cost["total_trading_costs"] > 0
