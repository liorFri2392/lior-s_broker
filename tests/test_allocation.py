"""Offline tests for allocation.py - gap-filling deposit allocation."""
import pytest

import allocation


PRICES = {
    "SPY": 745.0, "VOO": 550.0, "IVV": 560.0,
    "VEA": 70.0, "VXUS": 84.0, "IEFA": 96.0,
    "VWO": 59.0, "EEM": 66.0,
    "IWM": 293.0, "VB": 230.0,
    "VTV": 217.0, "IVW": 136.0, "SCHD": 32.0,
    "XLK": 181.0, "VGT": 115.0,
    "XLV": 150.0, "XBI": 163.0,
    "XLE": 55.0, "VDE": 157.0,
    "PAVE": 56.0, "IFRA": 61.0,
    "BND": 72.0, "AGG": 98.0,
    "VTIP": 49.0, "TIP": 108.0, "SCHP": 26.0,
}


def _holding(ticker, value, qty=10):
    return {"ticker": ticker, "quantity": qty,
            "last_price": value / qty, "current_value": value}


def test_targets_sum_to_one():
    assert sum(g["target"] for g in allocation.TARGET_GROUPS) == pytest.approx(1.0)


def test_classify_equivalents_share_a_group():
    assert allocation.classify("SPY") == allocation.classify("VOO") == "US_CORE"
    assert allocation.classify("BND") == allocation.classify("AGG") == "BONDS_AGG"
    assert allocation.classify("TIP") == allocation.classify("VTIP") == "TIPS"
    assert allocation.classify("VEA") == allocation.classify("IEFA")
    assert allocation.classify("UNKNOWN_TICKER") is None


def test_choose_instrument_prefers_largest_held():
    holdings = [_holding("VTIP", 2830), _holding("TIP", 108), _holding("SCHP", 445)]
    assert allocation.choose_instrument("TIPS", holdings) == "VTIP"
    # Empty group -> default (first listed).
    assert allocation.choose_instrument("US_SMALL_CAP", []) == "IWM"


def test_gap_fill_sends_money_to_biggest_gap():
    # Portfolio massively short on US_CORE (the real portfolio's disease).
    holdings = [
        _holding("SPY", 745, qty=1),
        _holding("VEA", 3728),
        _holding("VWO", 1124),
        _holding("BND", 1464),
        _holding("VTIP", 3384),
    ]
    buys = allocation.gap_fill_allocate(holdings, 3000.0, PRICES)
    by_ticker = {b["ticker"]: b for b in buys}
    # The dominant gap is US_CORE -> most of the deposit goes to SPY.
    assert "SPY" in by_ticker
    assert by_ticker["SPY"]["amount"] == max(b["amount"] for b in buys)
    # Never exceeds budget.
    assert sum(b["amount"] for b in buys) <= 3000.0 + 1e-9


def test_gap_fill_tops_up_held_ticker_not_twin():
    # AGG held, BND not: bond money must go to AGG (no new near-duplicate).
    holdings = [_holding("AGG", 500), _holding("SPY", 20000)]
    buys = allocation.gap_fill_allocate(holdings, 2000.0, PRICES)
    tickers = {b["ticker"] for b in buys}
    assert "AGG" in tickers and "BND" not in tickers


def test_gap_fill_no_buys_above_target():
    # Everything exactly at target -> gaps only from the new budget itself;
    # no group may end meaningfully above target.
    total = 100_000.0
    holdings = []
    for g in allocation.TARGET_GROUPS:
        t = g["tickers"][0]
        holdings.append(_holding(t, g["target"] * total))
    buys = allocation.gap_fill_allocate(holdings, 1000.0, PRICES)
    values = allocation.group_values(holdings)
    for b in buys:
        values[b["group"]] += b["amount"]
    total_after = sum(values.values()) + (1000.0 - sum(b["amount"] for b in buys))
    for g in allocation.TARGET_GROUPS:
        # Allow half a share of overshoot (the algorithm's stated tolerance).
        max_price = max(PRICES.get(t, 0) for t in g["tickers"])
        assert values[g["key"]] <= g["target"] * total_after + max_price


def test_gap_fill_converges_over_monthly_deposits():
    """The core property: repeated deposits move the portfolio TOWARD the
    targets and never invent new tickers when a group member is held."""
    holdings = [
        _holding("SPY", 745, qty=1), _holding("VXUS", 3546), _holding("VEA", 3728),
        _holding("VWO", 1124), _holding("IWM", 293), _holding("VTV", 218),
        _holding("XLK", 726), _holding("XLV", 300), _holding("XLE", 1668),
        _holding("PAVE", 280), _holding("BND", 582), _holding("AGG", 882),
        _holding("VTIP", 2830), _holding("TIP", 108), _holding("SCHP", 446),
    ]
    start_tickers = {h["ticker"] for h in holdings}

    def us_core_drift():
        values = allocation.group_values(holdings)
        total = sum(values.values())
        return abs(values["US_CORE"] / total - 0.30)

    drift_before = us_core_drift()
    for _ in range(12):  # a year of ~$2000 monthly deposits, flat prices
        buys = allocation.gap_fill_allocate(holdings, 2000.0, PRICES)
        for b in buys:
            for h in holdings:
                if h["ticker"] == b["ticker"]:
                    h["quantity"] += b["shares"]
                    h["current_value"] += b["amount"]
                    break
            else:
                holdings.append(_holding(b["ticker"], b["amount"], qty=b["shares"]))
    drift_after = us_core_drift()

    assert drift_after < drift_before  # converging, not fragmenting
    assert drift_after < 0.05          # and close to target within a year
    # Every group had a held member, so NO new tickers should ever appear.
    assert {h["ticker"] for h in holdings} == start_tickers


def test_gap_fill_skips_unpriced_instruments():
    holdings = [_holding("SPY", 100)]
    buys = allocation.gap_fill_allocate(holdings, 5000.0, {"SPY": 745.0})
    # Only SPY has a price -> only SPY may be bought; nothing invented.
    assert all(b["ticker"] == "SPY" for b in buys)


def test_gap_fill_zero_or_negative_budget():
    assert allocation.gap_fill_allocate([], 0, PRICES) == []
    assert allocation.gap_fill_allocate([], -50, PRICES) == []


def test_other_group_never_receives_money():
    holdings = [_holding("QQQ_WEIRD", 5000)]  # unclassified
    buys = allocation.gap_fill_allocate(holdings, 1000.0, PRICES)
    assert all(b["ticker"] != "QQQ_WEIRD" for b in buys)
