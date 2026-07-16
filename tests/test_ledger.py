"""Offline tests for ledger.py - deposit tracking and money-weighted returns."""
from datetime import datetime

import pytest

import ledger


def test_opening_only_one_year_return():
    p = {"transactions": [{"type": "opening", "date": "2025-07-11", "value_usd": 10000}]}
    perf = ledger.performance(p, 10700.0, now=datetime(2026, 7, 11))
    assert perf["net_invested_usd"] == 10000.0
    assert perf["gain_usd"] == 700.0
    assert abs(perf["xirr_pct"] - 7.0) < 0.1


def test_deposits_are_not_gains():
    """The core regression: monthly deposits must not show up as return."""
    p = {"transactions": [
        {"type": "opening", "date": "2026-01-01", "value_usd": 9000},
        {"type": "deposit", "date": "2026-02-01", "amount_usd": 3000},
        {"type": "deposit", "date": "2026-03-01", "amount_usd": 3000},
        {"type": "deposit", "date": "2026-04-01", "amount_usd": 3000},
    ]}
    # Flat market: value == money put in -> zero gain, ~zero XIRR.
    perf = ledger.performance(p, 18000.0, now=datetime(2026, 7, 1))
    assert abs(perf["gain_usd"]) < 1e-6
    assert abs(perf["gain_pct"]) < 1e-6
    assert abs(perf["xirr_pct"]) < 0.05


def test_ensure_ledger_seeds_once():
    p = {}
    assert ledger.ensure_ledger(p, 5000.0) is True
    assert ledger.ensure_ledger(p, 9999.0) is False  # already seeded
    openings = [t for t in p["transactions"] if t["type"] == "opening"]
    assert len(openings) == 1
    assert openings[0]["value_usd"] == 5000.0


def test_ensure_ledger_rejects_nonpositive():
    assert ledger.ensure_ledger({}, 0) is False
    assert ledger.ensure_ledger({}, -5) is False


def test_record_deposit_and_trade():
    p = {}
    ledger.ensure_ledger(p, 1000.0)
    ledger.record_deposit(p, 250.0, amount_ils=920.0, exchange_rate=3.68)
    ledger.record_trade(p, "buy", "VOO", 2, 100.0)
    ledger.record_trade(p, "sell", "XYZ", 0, 100.0)  # ignored: zero shares
    types = [t["type"] for t in p["transactions"]]
    assert types == ["opening", "deposit", "buy"]
    dep = p["transactions"][1]
    assert dep["amount_usd"] == 250.0 and dep["ils_per_usd"] == 3.68


def test_performance_none_without_ledger():
    assert ledger.performance({}, 1000.0) is None
    assert ledger.performance({"transactions": []}, 1000.0) is None


def test_average_monthly_deposit():
    p = {"transactions": [
        {"type": "opening", "date": "2026-01-01", "value_usd": 1000},
        {"type": "deposit", "date": "2026-02-01", "amount_usd": 600},
        {"type": "deposit", "date": "2026-03-01", "amount_usd": 600},
        {"type": "deposit", "date": "2026-04-01", "amount_usd": 600},
    ]}
    avg = ledger.average_monthly_deposit(p, now=datetime(2026, 5, 1))
    # 1800 over ~2.9 months since first deposit -> ~615/month
    assert 500 < avg < 700
    assert ledger.average_monthly_deposit({}) is None
    # A same-week burst is not extrapolated (min 1 month window).
    burst = {"transactions": [
        {"type": "deposit", "date": "2026-06-01", "amount_usd": 1000},
        {"type": "deposit", "date": "2026-06-03", "amount_usd": 1000},
    ]}
    assert ledger.average_monthly_deposit(burst, now=datetime(2026, 6, 5)) == 2000.0


def test_parse_cash_input():
    assert ledger.parse_cash_input("133.5", 3.0) == 133.5
    assert ledger.parse_cash_input("400 ils", 3.2) == 125.0
    assert ledger.parse_cash_input("₪400", 3.2) == 125.0
    assert ledger.parse_cash_input("1,200 nis", 3.0) == 400.0
    assert ledger.parse_cash_input("$99", 3.0) == 99.0
    assert ledger.parse_cash_input("", 3.0) is None
    assert ledger.parse_cash_input("abc", 3.0) is None
    assert ledger.parse_cash_input("-5", 3.0) is None


def test_reconcile_cash_logs_adjustment_without_touching_flows():
    p = {"transactions": [{"type": "opening", "date": "2026-01-01", "value_usd": 1000}],
         "cash": 796.94}
    delta = ledger.reconcile_cash(p, 133.22)
    assert delta == pytest.approx(-663.72)
    assert p["cash"] == 133.22
    adj = [t for t in p["transactions"] if t["type"] == "adjustment"]
    assert len(adj) == 1 and adj[0]["amount_usd"] == pytest.approx(-663.72)
    # Adjustments must NOT change net invested (drift = reduced gain, not a withdrawal).
    perf = ledger.performance(p, 5000.0, now=datetime(2026, 7, 1))
    assert perf["net_invested_usd"] == 1000.0
    # No-op when already matching.
    assert ledger.reconcile_cash(p, 133.22) == 0.0
    assert len([t for t in p["transactions"] if t["type"] == "adjustment"]) == 1


def test_short_period_has_no_xirr():
    p = {"transactions": [{"type": "opening", "date": "2026-07-01", "value_usd": 1000}]}
    perf = ledger.performance(p, 1050.0, now=datetime(2026, 7, 10))
    assert perf["gain_usd"] == 50.0
    assert perf["xirr_pct"] is None  # < 30 days: annualizing would be noise
