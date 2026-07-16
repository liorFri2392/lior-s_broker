"""ledger.py - Deposit/trade ledger and honest performance math.

The old "cumulative return" compared today's total value against a fixed
``baseline_value`` that was never adjusted for deposits, so every monthly
deposit was reported as investment gain. This module fixes that:

- Every deposit (and, for auditability, every recorded buy/sell) is appended
  to ``portfolio["transactions"]``.
- Performance is computed from investor cash flows: an opening balance, the
  dated deposits after it, and today's value. That yields the true dollar
  gain (value minus money put in) and a money-weighted annual return (XIRR).

For an existing portfolio with no transaction history the ledger is seeded
with today's value as the opening balance - returns are then measured
honestly from "now" instead of pretending past deposits were gains.

All amounts are USD, matching portfolio.json.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DATE_FMT = "%Y-%m-%d"


def _today() -> str:
    return datetime.now(timezone.utc).strftime(_DATE_FMT)


def _parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(str(s)[:10], _DATE_FMT)
    except (ValueError, TypeError):
        return None


def ensure_ledger(portfolio: Dict, current_total_value: float) -> bool:
    """Seed the ledger with an opening balance if none exists.

    Returns True if the portfolio was modified (caller should save).
    """
    txs = portfolio.get("transactions")
    if isinstance(txs, list) and any(t.get("type") == "opening" for t in txs):
        return False
    if current_total_value is None or current_total_value <= 0:
        return False
    portfolio["transactions"] = [{
        "type": "opening",
        "date": _today(),
        "value_usd": round(float(current_total_value), 2),
        "note": "Ledger start - value of the portfolio when deposit/return tracking began.",
    }] + [t for t in (txs or []) if t.get("type") != "opening"]
    logger.info("Ledger initialized with opening balance $%.2f", current_total_value)
    return True


def record_deposit(portfolio: Dict, amount_usd: float,
                   amount_ils: Optional[float] = None,
                   exchange_rate: Optional[float] = None) -> None:
    """Append a cash deposit to the ledger (call before saving portfolio.json)."""
    if amount_usd is None or amount_usd <= 0:
        return
    tx = {"type": "deposit", "date": _today(), "amount_usd": round(float(amount_usd), 2)}
    if amount_ils:
        tx["amount_ils"] = round(float(amount_ils), 2)
    if exchange_rate:
        tx["ils_per_usd"] = round(float(exchange_rate), 4)
    portfolio.setdefault("transactions", []).append(tx)


def record_trade(portfolio: Dict, side: str, ticker: str, shares: float, price_usd: float) -> None:
    """Append an executed buy/sell to the ledger (audit trail; trades are
    internal cash<->shares moves so they don't affect the return math)."""
    if not ticker or shares <= 0 or price_usd <= 0:
        return
    portfolio.setdefault("transactions", []).append({
        "type": side,  # "buy" | "sell"
        "date": _today(),
        "ticker": ticker,
        "shares": shares,
        "price_usd": round(float(price_usd), 4),
        "amount_usd": round(float(shares) * float(price_usd), 2),
    })


def _external_cash_flows(portfolio: Dict) -> List[Dict]:
    """Opening balance + deposits/withdrawals, sorted by date. Empty if no ledger."""
    flows = []
    for t in portfolio.get("transactions", []) or []:
        d = _parse_date(t.get("date"))
        if d is None:
            continue
        if t.get("type") == "opening" and t.get("value_usd", 0) > 0:
            flows.append({"date": d, "amount": float(t["value_usd"])})
        elif t.get("type") == "deposit" and t.get("amount_usd", 0) > 0:
            flows.append({"date": d, "amount": float(t["amount_usd"])})
        elif t.get("type") == "withdrawal" and t.get("amount_usd", 0) > 0:
            flows.append({"date": d, "amount": -float(t["amount_usd"])})
    flows.sort(key=lambda f: f["date"])
    return flows


def _xirr(flows: List[Dict], current_value: float, now: datetime) -> Optional[float]:
    """Money-weighted annual return via bisection on NPV.

    ``flows`` are money the investor put IN (positive) / took OUT (negative);
    ``current_value`` is what it is worth now. Returns a fraction (0.07 = 7%/yr)
    or None when it cannot be determined.
    """
    if not flows or current_value <= 0:
        return None
    t0 = flows[0]["date"]
    span_years = max((now - t0).days, 1) / 365.25

    def npv(rate: float) -> float:
        # current_value minus every flow grown at ``rate`` from its date to today.
        grown = 0.0
        for f in flows:
            yrs = (f["date"] - t0).days / 365.25
            grown += f["amount"] * (1.0 + rate) ** (span_years - yrs)
        return current_value - grown

    # npv is monotonically decreasing in rate (all net flows are positive money
    # in), so bisection over a wide bracket finds the unique root.
    lo, hi = -0.9999, 100.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo < 0 or f_hi > 0:
        return None  # no sign change in bracket (degenerate flows)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            break
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def parse_cash_input(raw: str, ils_per_usd: float) -> Optional[float]:
    """Parse a user-entered cash amount. Plain numbers are USD; an ILS marker
    ('ils', 'nis', '₪', 'שח') converts at ``ils_per_usd``. Empty/invalid -> None."""
    s = (raw or "").strip().lower().replace(",", "")
    if not s:
        return None
    is_ils = any(tok in s for tok in ("₪", "ils", "nis", "שח", 'ש"ח'))
    for tok in ("₪", "ils", "nis", "שח", 'ש"ח', "usd", "$"):
        s = s.replace(tok, "")
    try:
        val = float(s.strip())
    except ValueError:
        return None
    if val < 0:
        return None
    return round(val / ils_per_usd, 2) if is_ils else round(val, 2)


def ask_actual_cash(expected_usd: float, ils_per_usd: float) -> Optional[float]:
    """Interactive prompt for the broker's real remaining cash (None = keep
    the computed figure / non-interactive session)."""
    import sys
    if not sys.stdin.isatty():
        return None
    raw = input(f"   Broker's ACTUAL remaining cash "
                f"[Enter = keep ${expected_usd:,.2f}; e.g. '133.5' USD or '400 ils']: ")
    return parse_cash_input(raw, ils_per_usd)


def reconcile_cash(portfolio: Dict, actual_cash_usd: float,
                   note: str = "broker cash reconciliation") -> float:
    """Set tracked cash to the broker's actual figure; return the delta.

    The difference (usually real fills costing more/less than scan prices) is
    logged as an ``adjustment`` transaction for the audit trail. Adjustments
    are deliberately NOT external cash flows: the money was really deposited,
    so execution drift must show up as reduced gain, not reduced investment.
    """
    tracked = float(portfolio.get("cash", 0) or 0)
    delta = round(float(actual_cash_usd) - tracked, 2)
    if abs(delta) < 0.01:
        return 0.0
    portfolio["cash"] = round(float(actual_cash_usd), 2)
    portfolio.setdefault("transactions", []).append({
        "type": "adjustment",
        "date": _today(),
        "amount_usd": delta,
        "note": note,
    })
    return delta


def average_monthly_deposit(portfolio: Dict,
                            now: Optional[datetime] = None) -> Optional[float]:
    """Average USD deposited per month, from the ledger's deposit history.

    Total deposits divided by the months elapsed since the first deposit
    (minimum one month so a burst of early deposits isn't extrapolated).
    None when no deposits are recorded yet.
    """
    deposits = [(d, t) for t in portfolio.get("transactions", []) or []
                if t.get("type") == "deposit" and t.get("amount_usd", 0) > 0
                and (d := _parse_date(t.get("date"))) is not None]
    if not deposits:
        return None
    now = now or datetime.now()
    first = min(d for d, _ in deposits)
    months = max((now - first).days / 30.44, 1.0)
    total = sum(t["amount_usd"] for _, t in deposits)
    return round(total / months, 2)


def performance(portfolio: Dict, current_total_value: float,
                now: Optional[datetime] = None) -> Optional[Dict]:
    """True performance from the ledger, or None when no ledger exists.

    Returns dict with:
      ledger_start_date  - ISO date the ledger began
      net_invested_usd   - opening balance + deposits - withdrawals
      gain_usd           - current value minus money put in (the real P&L)
      gain_pct           - gain as % of net invested (simple, not annualized)
      xirr_pct           - money-weighted annual return %, or None if
                           undeterminable / period too short (<30 days)
      days               - days since ledger start
    """
    flows = _external_cash_flows(portfolio)
    if not flows:
        return None
    now = now or datetime.now()
    net_invested = sum(f["amount"] for f in flows)
    if net_invested <= 0:
        return None
    gain = current_total_value - net_invested
    days = max((now - flows[0]["date"]).days, 0)
    xirr_pct = None
    if days >= 30:
        r = _xirr(flows, current_total_value, now)
        if r is not None:
            xirr_pct = r * 100.0
    return {
        "ledger_start_date": flows[0]["date"].strftime(_DATE_FMT),
        "net_invested_usd": round(net_invested, 2),
        "gain_usd": round(gain, 2),
        "gain_pct": round(gain / net_invested * 100.0, 2),
        "xirr_pct": round(xirr_pct, 2) if xirr_pct is not None else None,
        "days": days,
    }
