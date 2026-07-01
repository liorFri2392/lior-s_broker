#!/usr/bin/env python3
"""
market_data.py - Shared, cached market-data access layer.

Single source of truth for all yfinance access across the project. Every module
should import the process-wide singleton ``market`` (or the module-level helper
functions) instead of calling ``yfinance`` directly. This gives us:

- one in-memory cache shared by every consumer in a run (the same ticker is
  fetched at most once per TTL window, no matter how many modules ask for it);
- batched downloads via ``yf.download`` instead of N sequential ``Ticker.history``
  round-trips;
- a single persistent ``.cache.json`` (prices, FX, risk-free rate) that survives
  between runs;
- DST-correct US market-hours detection;
- a memoized risk-free rate and exchange rate.

Thread-safe: caches are guarded by a re-entrant lock and the persistent file is
written atomically, so concurrent ``ThreadPoolExecutor`` fetches can't corrupt it.
"""

import json
import logging
import math
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback if tzdata missing
    _ET = None

logger = logging.getLogger(__name__)

# Cache TTLs (kept consistent with the original PortfolioAnalyzer behaviour).
_TTL_MARKET_CLOSED = timedelta(hours=4)
_TTL_MARKET_OPEN = timedelta(minutes=60)
_TTL_FX = timedelta(minutes=30)
_TTL_RISK_FREE = timedelta(hours=12)
_TTL_INFO = timedelta(hours=12)

_DEFAULT_FX = 1.0 / 0.34  # ILS per USD when 1 ILS = 0.34 USD (~2.94)
_DEFAULT_RISK_FREE = 0.045

# A single-fetch price move beyond this vs the last known price is treated as a
# likely data glitch and logged (a real ETF cannot move 60% between fetches).
_MAX_PRICE_JUMP = 0.60


class MarketData:
    """Process-wide cached gateway to market data."""

    def __init__(self, cache_file: str = ".cache.json"):
        self.cache_file = cache_file
        self._lock = threading.RLock()

        # In-memory caches.
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._history_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._info_cache: Dict[str, Tuple[dict, datetime]] = {}
        self._news_cache: Dict[str, Tuple[list, datetime]] = {}

        self._fx_rate: float = _DEFAULT_FX
        self._fx_time: Optional[datetime] = None
        self._risk_free: float = _DEFAULT_RISK_FREE
        self._risk_free_time: Optional[datetime] = None

        # Memoized market status for a single run (cheap, but avoids recompute spam).
        self._market_status_cache: Optional[Tuple[bool, str, datetime]] = None

        self._load_cache()

    # ------------------------------------------------------------------ #
    # Persistent cache (.cache.json) - prices + FX + risk-free scalars.
    # ------------------------------------------------------------------ #
    def _load_cache(self) -> None:
        try:
            if not os.path.exists(self.cache_file):
                return
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            now = datetime.now()

            for ticker, entry in (data.get("price_cache") or {}).items():
                try:
                    ts = datetime.fromisoformat(entry["timestamp"])
                    if now - ts < _TTL_MARKET_CLOSED:
                        self._price_cache[ticker] = (float(entry["price"]), ts)
                except (ValueError, KeyError, TypeError):
                    continue

            fx = data.get("exchange_rate")
            if isinstance(fx, dict) and fx.get("rate") and fx.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(fx["timestamp"])
                    if now - ts < _TTL_MARKET_CLOSED:
                        self._fx_rate = float(fx["rate"])
                        self._fx_time = ts
                except (ValueError, TypeError):
                    pass

            rf = data.get("risk_free")
            if isinstance(rf, dict) and rf.get("rate") and rf.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(rf["timestamp"])
                    if now - ts < _TTL_RISK_FREE:
                        self._risk_free = float(rf["rate"])
                        self._risk_free_time = ts
                except (ValueError, TypeError):
                    pass

            logger.info(f"Loaded market cache from {self.cache_file}")
        except Exception as e:  # noqa: BLE001 - cache is best-effort
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Atomically persist scalar caches. Safe to call from any thread."""
        with self._lock:
            try:
                payload = {
                    "price_cache": {
                        t: {"price": p, "timestamp": ts.isoformat()}
                        for t, (p, ts) in self._price_cache.items()
                    },
                    "exchange_rate": (
                        {"rate": self._fx_rate, "timestamp": self._fx_time.isoformat()}
                        if self._fx_time
                        else None
                    ),
                    "risk_free": (
                        {"rate": self._risk_free, "timestamp": self._risk_free_time.isoformat()}
                        if self._risk_free_time
                        else None
                    ),
                }
                tmp = f"{self.cache_file}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, self.cache_file)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------ #
    # Market hours (DST-correct).
    # ------------------------------------------------------------------ #
    def is_market_open(self) -> Tuple[Optional[bool], str]:
        """Return (is_open, message). Uses America/New_York so DST is handled."""
        with self._lock:
            if self._market_status_cache:
                is_open, msg, ts = self._market_status_cache
                if datetime.now() - ts < timedelta(minutes=1):
                    return is_open, msg
        try:
            if _ET is not None:
                now_et = datetime.now(_ET)
            else:  # crude fallback, no DST
                now_et = datetime.now(timezone.utc) - timedelta(hours=5)
            open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            is_weekday = now_et.weekday() < 5
            if is_weekday and open_t <= now_et <= close_t:
                result = (True, "Market is OPEN - Prices are real-time")
            elif not is_weekday:
                result = (False, "Market is CLOSED - Weekend (Prices from last close)")
            else:
                result = (False, "Market is CLOSED - After hours (Prices from last close)")
            with self._lock:
                self._market_status_cache = (result[0], result[1], datetime.now())
            return result
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to determine market status: {e}")
            return None, "Unable to determine market status"

    def _history_ttl(self) -> timedelta:
        is_open, _ = self.is_market_open()
        return _TTL_MARKET_OPEN if is_open else _TTL_MARKET_CLOSED

    # ------------------------------------------------------------------ #
    # Historical data - single + batched, cached.
    # ------------------------------------------------------------------ #
    def get_history(self, ticker: str, period: str = "1y", auto_adjust: bool = True) -> pd.DataFrame:
        """Cached single-ticker history. Returns an empty DataFrame on failure."""
        key = f"{ticker}|{period}|{int(auto_adjust)}"
        now = datetime.now()
        ttl = self._history_ttl()
        with self._lock:
            hit = self._history_cache.get(key)
            if hit and now - hit[1] < ttl:
                return hit[0]
        try:
            data = yf.Ticker(ticker).history(period=period, auto_adjust=auto_adjust)
            if data is None or data.empty:
                return pd.DataFrame()
            with self._lock:
                self._history_cache[key] = (data, now)
            return data
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch history for {ticker}: {e}")
            return pd.DataFrame()

    def get_histories(
        self, tickers: List[str], period: str = "1y", auto_adjust: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Batched multi-ticker history via a single ``yf.download`` round-trip.

        Already-cached tickers are served from cache; only the misses are
        downloaded, and they go out in one batched request.
        """
        tickers = [t for t in dict.fromkeys(tickers) if t]  # dedupe, keep order
        result: Dict[str, pd.DataFrame] = {}
        now = datetime.now()
        ttl = self._history_ttl()
        misses: List[str] = []

        with self._lock:
            for t in tickers:
                key = f"{t}|{period}|{int(auto_adjust)}"
                hit = self._history_cache.get(key)
                if hit and now - hit[1] < ttl:
                    result[t] = hit[0]
                else:
                    misses.append(t)

        if not misses:
            return result

        try:
            raw = yf.download(
                misses,
                period=period,
                auto_adjust=auto_adjust,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Batched download failed ({len(misses)} tickers): {e}")
            raw = None

        for t in misses:
            df = pd.DataFrame()
            try:
                if raw is None or raw.empty:
                    df = pd.DataFrame()
                elif len(misses) == 1:
                    df = raw.dropna(how="all")
                elif isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0):
                    df = raw[t].dropna(how="all")
            except Exception:  # noqa: BLE001
                df = pd.DataFrame()

            if df is None or df.empty:
                # Fall back to a per-ticker fetch (handles delisted/odd symbols).
                df = self.get_history(t, period=period, auto_adjust=auto_adjust)

            if df is not None and not df.empty:
                with self._lock:
                    self._history_cache[f"{t}|{period}|{int(auto_adjust)}"] = (df, now)
                result[t] = df
            else:
                result[t] = pd.DataFrame()

        return result

    def prefetch(self, tickers: List[str], period: str = "1y") -> None:
        """Warm the cache for a known universe of tickers in one batched call."""
        self.get_histories(tickers, period=period)

    # ------------------------------------------------------------------ #
    # Prices - single + batched, cached.
    # ------------------------------------------------------------------ #
    def get_price(self, ticker: str) -> Optional[float]:
        prices, _, _ = self.get_prices([ticker])
        return prices.get(ticker)

    def get_prices(
        self, tickers: List[str]
    ) -> Tuple[Dict[str, float], Optional[bool], str]:
        """Return (prices, market_status, message). Cached + batched."""
        tickers = [t for t in dict.fromkeys(tickers) if t]
        market_status, market_message = self.is_market_open()
        ttl = _TTL_MARKET_OPEN if market_status else _TTL_MARKET_CLOSED
        now = datetime.now()
        prices: Dict[str, float] = {}
        misses: List[str] = []

        with self._lock:
            for t in tickers:
                hit = self._price_cache.get(t)
                if hit and now - hit[1] < ttl:
                    prices[t] = hit[0]
                else:
                    misses.append(t)

        if misses:
            histories = self.get_histories(misses, period="1d")
            for t in misses:
                df = histories.get(t)
                try:
                    if df is not None and not df.empty and "Close" in df.columns:
                        price = float(df["Close"].iloc[-1])
                        # Reject bad data: non-finite (NaN/inf) or non-positive.
                        if not math.isfinite(price) or price <= 0:
                            continue
                        # Flag implausible single-fetch jumps vs the last known
                        # price — almost always a data glitch, not a real move.
                        prev = self._price_cache.get(t)
                        if prev and prev[0] > 0:
                            change = abs(price / prev[0] - 1)
                            if change > _MAX_PRICE_JUMP:
                                logger.warning(
                                    f"{t}: suspicious price jump {prev[0]:.2f} -> "
                                    f"{price:.2f} ({change*100:.0f}%). Using new value; "
                                    f"verify before trading."
                                )
                        prices[t] = price
                        with self._lock:
                            self._price_cache[t] = (price, now)
                except Exception:  # noqa: BLE001
                    continue
            self._save_cache()

        return prices, market_status, market_message

    # ------------------------------------------------------------------ #
    # Ticker .info (slow; cached aggressively).
    # ------------------------------------------------------------------ #
    def get_info(self, ticker: str) -> dict:
        now = datetime.now()
        with self._lock:
            hit = self._info_cache.get(ticker)
            if hit and now - hit[1] < _TTL_INFO:
                return hit[0]
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to fetch info for {ticker}: {e}")
            info = {}
        with self._lock:
            self._info_cache[ticker] = (info, now)
        return info

    # ------------------------------------------------------------------ #
    # News headlines - cached (avoids re-fetching the same ticker's news
    # across the several analysis passes in one run).
    # ------------------------------------------------------------------ #
    def get_news(self, ticker: str) -> list:
        now = datetime.now()
        with self._lock:
            hit = self._news_cache.get(ticker)
            if hit and now - hit[1] < _TTL_INFO:
                return hit[0]
        try:
            news = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to fetch news for {ticker}: {e}")
            news = []
        with self._lock:
            self._news_cache[ticker] = (news, now)
        return news

    # ------------------------------------------------------------------ #
    # Risk-free rate (^IRX) - memoized.
    # ------------------------------------------------------------------ #
    def get_risk_free_rate(self) -> float:
        with self._lock:
            if self._risk_free_time and datetime.now() - self._risk_free_time < _TTL_RISK_FREE:
                return self._risk_free
        rate = _DEFAULT_RISK_FREE
        try:
            hist = self.get_history("^IRX", period="5d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                candidate = float(hist["Close"].iloc[-1]) / 100.0
                if 0 < candidate < 0.15:
                    rate = candidate
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to fetch risk-free rate: {e}")
        with self._lock:
            self._risk_free = rate
            self._risk_free_time = datetime.now()
        self._save_cache()
        return rate

    # ------------------------------------------------------------------ #
    # FX (ILS per USD) - memoized + env override.
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_ils_per_usd(raw_rate: float) -> float:
        """Return how many ILS equal 1 USD (invert sub-1 USD-per-ILS quotes)."""
        if raw_rate <= 0:
            return _DEFAULT_FX
        if raw_rate < 1.0:
            return 1.0 / raw_rate
        return raw_rate

    def get_exchange_rate(self) -> float:
        override = os.getenv("ILS_PER_USD") or os.getenv("EXCHANGE_RATE_ILS_PER_USD")
        if override:
            try:
                rate = self.normalize_ils_per_usd(float(override))
                with self._lock:
                    self._fx_rate = rate
                return rate
            except ValueError:
                logger.warning(f"Invalid ILS_PER_USD override: {override!r}")

        with self._lock:
            if self._fx_time and datetime.now() - self._fx_time < _TTL_FX:
                return self._fx_rate

        for symbol in ("ILSUSD=X", "USDILS=X"):
            try:
                hist = self.get_history(symbol, period="1d")
                if hist is not None and not hist.empty:
                    raw = float(hist["Close"].iloc[-1])
                    rate = self.normalize_ils_per_usd(raw)
                    with self._lock:
                        self._fx_rate = rate
                        self._fx_time = datetime.now()
                    self._save_cache()
                    return rate
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Failed to fetch {symbol}: {e}")

        logger.warning("Failed to fetch exchange rate, using cached/default")
        return self._fx_rate

    def clear(self) -> None:
        with self._lock:
            self._price_cache.clear()
            self._history_cache.clear()
            self._info_cache.clear()
            self._market_status_cache = None


# ---------------------------------------------------------------------- #
# Process-wide singleton + module-level convenience helpers.
# ---------------------------------------------------------------------- #
market = MarketData()

def get_history(ticker: str, period: str = "1y", auto_adjust: bool = True) -> pd.DataFrame:
    return market.get_history(ticker, period=period, auto_adjust=auto_adjust)

def get_histories(tickers: List[str], period: str = "1y", auto_adjust: bool = True) -> Dict[str, pd.DataFrame]:
    return market.get_histories(tickers, period=period, auto_adjust=auto_adjust)

def get_price(ticker: str) -> Optional[float]:
    return market.get_price(ticker)

def get_prices(tickers: List[str]) -> Tuple[Dict[str, float], Optional[bool], str]:
    return market.get_prices(tickers)

def get_info(ticker: str) -> dict:
    return market.get_info(ticker)

def get_news(ticker: str) -> list:
    return market.get_news(ticker)

def get_risk_free_rate() -> float:
    return market.get_risk_free_rate()

def get_exchange_rate() -> float:
    return market.get_exchange_rate()

def is_market_open() -> Tuple[Optional[bool], str]:
    return market.is_market_open()

def prefetch(tickers: List[str], period: str = "1y") -> None:
    market.prefetch(tickers, period=period)
