"""Shared pytest fixtures.

The whole suite runs OFFLINE: the ``market_data`` layer (the single gateway to
yfinance) is monkeypatched with deterministic synthetic data so tests never hit
the network and never flake. This lets the suite act as a regression net around
the deduplication refactor without depending on live market conditions.
"""
import hashlib
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Make the project importable when pytest is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import market_data  # noqa: E402


def synthetic_history(ticker: str = "TEST", days: int = 300, start: float = 80.0) -> pd.DataFrame:
    """A deterministic, gently rising OHLCV frame usable by every indicator.

    Deterministic per-ticker (seed derived from the symbol) so results are
    stable across runs but differ between tickers.
    """
    # Stable seed (Python's str hash is salted per-process; use md5 so the
    # synthetic series — and thus the golden-master scores — are reproducible).
    seed = int.from_bytes(hashlib.md5(ticker.encode()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=datetime(2024, 1, 1), periods=days, freq="B")
    drift = np.linspace(0, 0.4, days)
    noise = rng.normal(0, 0.01, days).cumsum()
    close = start * (1 + drift + noise)
    close = np.clip(close, 1.0, None)
    frame = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, days),
        },
        index=idx,
    )
    return frame


@pytest.fixture
def offline_market(monkeypatch):
    """Patch the market_data singleton + module functions with synthetic data."""
    price = 100.0

    def _history(ticker, period="1y", auto_adjust=True):
        return synthetic_history(ticker)

    def _histories(tickers, period="1y", auto_adjust=True):
        return {t: synthetic_history(t) for t in tickers}

    def _prices(tickers):
        return {t: price for t in tickers}, False, "Market is CLOSED (test)"

    def _price(ticker):
        return price

    patches = {
        "get_history": _history,
        "get_histories": _histories,
        "get_prices": _prices,
        "get_price": _price,
        "get_info": lambda t: {},
        "get_news": lambda t: [],
        "get_exchange_rate": lambda: 3.7,
        "get_risk_free_rate": lambda: 0.045,
        "is_market_open": lambda: (False, "Market is CLOSED (test)"),
        "prefetch": lambda tickers, period="1y": None,
    }
    for name, fn in patches.items():
        monkeypatch.setattr(market_data, name, fn, raising=True)
        # Keep the singleton in lock-step for callers that use market_data.market.*
        if hasattr(market_data.market, name):
            monkeypatch.setattr(market_data.market, name, fn, raising=True)
    return patches


@pytest.fixture
def sample_portfolio():
    return {
        "currency": "USD",
        "cash": 500.0,
        "holdings": [
            {"ticker": "SPY", "quantity": 9, "last_price": 100.0,
             "current_value": 900.0, "cost_basis": 80.0},
            {"ticker": "XLV", "quantity": 6, "last_price": 100.0,
             "current_value": 600.0, "cost_basis": 120.0},
            {"ticker": "BND", "quantity": 5, "last_price": 100.0,
             "current_value": 500.0, "cost_basis": 100.0},
        ],
        "last_updated": None,
        "total_value": 0,
    }


@pytest.fixture
def portfolio_file(tmp_path, sample_portfolio):
    import json
    p = tmp_path / "portfolio.json"
    p.write_text(json.dumps(sample_portfolio), encoding="utf-8")
    return str(p)
