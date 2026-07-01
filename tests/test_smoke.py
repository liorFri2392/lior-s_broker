"""Offline regression net for lior's_broker.

These tests pin behaviour that the deduplication refactor must preserve:
constant ETF universes, portfolio load/save, share-sizing math, tax math, FX
normalization, and that the high-level flows run end-to-end on synthetic data.
"""
import json
import math

import pytest


# --------------------------------------------------------------------------- #
# ETF universe constants — must survive being moved into a shared module.
# --------------------------------------------------------------------------- #
def test_analyzer_etf_universes():
    from portfolio_analyzer import PortfolioAnalyzer
    assert PortfolioAnalyzer.CORE_ETFS == ["SPY", "VOO", "IVV", "VXUS", "VEA"]
    assert PortfolioAnalyzer.BOND_ETFS == ["BND", "AGG", "TIP", "SCHP", "VTIP"]
    assert PortfolioAnalyzer.SATELLITE_ETFS == [
        "IWM", "VB", "XLK", "VGT", "VWO", "EEM", "XLV", "VHT",
    ]


def test_leveraged_and_satellite_constants():
    import deposit_advisor as da
    assert da.LEVERAGED_2X == ["SSO", "QLD", "UWM", "EFO"]
    assert da.LEVERAGED_3X == [
        "TQQQ", "SPXL", "UPRO", "TNA", "FAS", "CURE", "SOXL", "LABU", "TECL",
    ]
    assert da.LEVERAGED_INVERSE == [
        "SQQQ", "SPXS", "SPXU", "TZA", "FAZ", "SOXS", "LABD", "TECS",
    ]
    assert da.ALL_LEVERAGED_TICKERS == (
        set(da.LEVERAGED_2X) | set(da.LEVERAGED_3X) | set(da.LEVERAGED_INVERSE)
    )
    assert "TECHNOLOGY" in da.SATELLITE_CATEGORIES
    assert "HEALTHCARE" in da.SATELLITE_CATEGORIES


def test_satellite_categories_shared_across_modules():
    """critical_alert must use the *same* satellite list as deposit_advisor."""
    import deposit_advisor as da
    import critical_alert as ca
    assert ca.SATELLITE_CATEGORIES == da.SATELLITE_CATEGORIES


# --------------------------------------------------------------------------- #
# load_portfolio — behaviour must be identical across all four owners.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("factory", [
    lambda pf: __import__("portfolio_analyzer").PortfolioAnalyzer(pf),
    lambda pf: __import__("risk_manager").RiskManager(pf),
    lambda pf: __import__("tax_analyzer").TaxAnalyzer(pf, exchange_rate=3.7),
])
def test_load_portfolio_roundtrip(factory, portfolio_file, sample_portfolio):
    obj = factory(portfolio_file)
    loaded = obj.load_portfolio()
    assert loaded["cash"] == sample_portfolio["cash"]
    assert [h["ticker"] for h in loaded["holdings"]] == ["SPY", "XLV", "BND"]


def test_load_portfolio_missing_file_defaults(tmp_path):
    from risk_manager import RiskManager
    rm = RiskManager(str(tmp_path / "nope.json"))
    loaded = rm.load_portfolio()
    assert loaded["holdings"] == []
    assert loaded["cash"] == 0


def test_deposit_advisor_load_portfolio(offline_market, portfolio_file):
    from deposit_advisor import DepositAdvisor
    da = DepositAdvisor(portfolio_file)
    loaded = da.load_portfolio()
    assert len(loaded["holdings"]) == 3


# --------------------------------------------------------------------------- #
# Share-sizing math (_shares_for_amount).
# --------------------------------------------------------------------------- #
def test_shares_for_amount_basic(portfolio_file):
    from critical_alert import CriticalAlertSystem
    cas = CriticalAlertSystem(portfolio_file)
    shares, price, remaining = cas._shares_for_amount("SPY", 1000, known_price=100)
    assert (shares, price) == (10, 100)
    assert remaining == 0


def test_shares_for_amount_leftover_below_price(portfolio_file):
    from critical_alert import CriticalAlertSystem
    cas = CriticalAlertSystem(portfolio_file)
    # $1080 @ $100 -> 10 shares, $80 left. The leftover is < price, so no extra
    # share is bought (whole-share flooring always leaves remainder < price).
    shares, price, remaining = cas._shares_for_amount("SPY", 1080, known_price=100)
    assert shares == 10
    assert remaining == pytest.approx(80.0)


def test_shares_for_amount_no_price(portfolio_file, monkeypatch):
    import market_data
    # No known price and none available from the market layer -> buys nothing.
    monkeypatch.setattr(market_data, "get_price", lambda t: None)
    from critical_alert import CriticalAlertSystem
    cas = CriticalAlertSystem(portfolio_file)
    assert cas._shares_for_amount("SPY", 1000, known_price=0) == (0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# FX normalization (pure).
# --------------------------------------------------------------------------- #
def test_normalize_ils_per_usd():
    from market_data import MarketData
    assert MarketData.normalize_ils_per_usd(3.7) == 3.7
    # A sub-1 quote (USD per ILS) is inverted to ILS per USD.
    assert MarketData.normalize_ils_per_usd(0.29) == pytest.approx(1 / 0.29)
    # Non-positive falls back to the default (~2.94).
    assert MarketData.normalize_ils_per_usd(0) > 1


# --------------------------------------------------------------------------- #
# Tax math (pure).
# --------------------------------------------------------------------------- #
def test_capital_gains_tax_short_term_gain():
    from tax_analyzer import TaxAnalyzer
    ta = TaxAnalyzer(exchange_rate=3.7)
    ta.LONG_TERM_REDUCTION = 0.0  # isolate from the unverified reduction
    res = ta.calculate_capital_gains_tax(
        purchase_price=100, sale_price=150, quantity=10,
        purchase_date="2024-01-01T00:00:00", sale_date="2024-06-01T00:00:00",
    )
    # gain = 50*10 = $500 -> 500*3.7 ILS -> 25% tax
    assert res["capital_gains_tax_ils"] == pytest.approx(500 * 3.7 * 0.25)


def test_capital_gains_tax_loss_no_tax():
    from tax_analyzer import TaxAnalyzer
    ta = TaxAnalyzer(exchange_rate=3.7)
    res = ta.calculate_capital_gains_tax(
        purchase_price=150, sale_price=100, quantity=10,
        purchase_date="2024-01-01T00:00:00", sale_date="2024-06-01T00:00:00",
    )
    assert res["capital_gains_tax_ils"] == 0


def test_capital_gains_tax_mixed_timezone_dates():
    """Regression: a 'Z'-suffixed purchase date must not crash against a naive sale."""
    from tax_analyzer import TaxAnalyzer
    ta = TaxAnalyzer(exchange_rate=3.7)
    res = ta.calculate_capital_gains_tax(
        purchase_price=100, sale_price=120, quantity=1,
        purchase_date="2022-01-01T00:00:00Z",  # tz-aware
        sale_date="2024-06-01T00:00:00",       # naive
    )
    assert res["capital_gains_tax_ils"] >= 0


# --------------------------------------------------------------------------- #
# High-level flows on synthetic data (offline).
# --------------------------------------------------------------------------- #
def test_refresh_portfolio_prices_offline(offline_market, portfolio_file):
    from portfolio_analyzer import PortfolioAnalyzer
    pa = PortfolioAnalyzer(portfolio_file)
    result = pa.refresh_portfolio_prices(verbose=False, sync_github_secret=False)
    # cash 500 + 3 holdings * (qty * $100)
    expected = 500 + (9 + 6 + 5) * 100.0
    assert result["total_value"] == pytest.approx(expected)
    for h in result["holdings"]:
        assert h["last_price"] == 100.0


def test_analyze_holding_offline(offline_market, portfolio_file):
    from portfolio_analyzer import PortfolioAnalyzer
    pa = PortfolioAnalyzer(portfolio_file)
    result = pa.analyze_holding("SPY", 9, 100.0, verbose=False)
    assert isinstance(result, dict)
    assert result.get("ticker") == "SPY"


def test_risk_alerts_offline(offline_market, portfolio_file):
    from risk_manager import RiskManager
    rm = RiskManager(portfolio_file)
    alerts = rm.get_risk_alerts()
    assert isinstance(alerts, (list, dict))


def test_deposit_advisor_analyze_etf_offline(offline_market, portfolio_file):
    from deposit_advisor import DepositAdvisor
    da = DepositAdvisor(portfolio_file)
    analysis = da.analyze_etf("SPY")
    assert isinstance(analysis, dict)
    score = analysis.get("score", analysis.get("total_score"))
    if score is not None:
        assert not math.isnan(float(score))


# --------------------------------------------------------------------------- #
# Golden master — the scoring engines must produce byte-identical scores on
# fixed synthetic data. This guards the extraction of shared scoring helpers:
# refactors that change these numbers would silently change real recommendations.
# --------------------------------------------------------------------------- #
GOLDEN_ETF_SCORES = {"SPY": 53.8308726551, "BND": 55.2031900678, "XLK": 66.2719374239}
GOLDEN_HOLDING_SCORES = {"SPY": 49.2763033514, "BND": 47.2972730977, "XLK": 59.8033473956}


@pytest.mark.parametrize("ticker,expected", GOLDEN_ETF_SCORES.items())
def test_golden_analyze_etf_score(offline_market, portfolio_file, ticker, expected):
    from deposit_advisor import DepositAdvisor
    da = DepositAdvisor(portfolio_file)
    assert da.analyze_etf(ticker)["score"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("ticker,expected", GOLDEN_HOLDING_SCORES.items())
def test_golden_analyze_holding_score(offline_market, portfolio_file, ticker, expected):
    from portfolio_analyzer import PortfolioAnalyzer
    pa = PortfolioAnalyzer(portfolio_file)
    score = pa.analyze_holding(ticker, 10, 100.0, verbose=False)["recommendation_score"]
    assert score == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# GitHub secret sync (extracted shared module) — mocked, no CLI/network.
# --------------------------------------------------------------------------- #
def test_github_secret_cli_success(monkeypatch):
    import github_secret as gs

    class _Ver:
        returncode = 0

    class _Proc:
        returncode = 0
        def communicate(self, input=None, timeout=None):
            return ("", "")

    monkeypatch.setattr(gs.subprocess, "run", lambda *a, **k: _Ver())
    monkeypatch.setattr(gs.subprocess, "Popen", lambda *a, **k: _Proc())
    assert gs.update_portfolio_secret({"cash": 1}) is True


def test_github_secret_no_cli_no_token(monkeypatch):
    import github_secret as gs

    def _boom(*a, **k):
        raise FileNotFoundError("gh not installed")

    monkeypatch.setattr(gs.subprocess, "run", _boom)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # No CLI and no token -> best-effort returns False, never raises.
    assert gs.update_portfolio_secret({"cash": 1}) is False


# --------------------------------------------------------------------------- #
# Import smoke — every module imports without side effects hitting the network.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", [
    "market_data", "portfolio_analyzer", "deposit_advisor", "critical_alert",
    "risk_manager", "tax_analyzer", "tax_report", "backtesting",
    "advanced_analysis", "ml_predictor", "email_notifier",
])
def test_module_imports(mod):
    __import__(mod)
