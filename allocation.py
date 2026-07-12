"""allocation.py - Target-weight gap-filling for monthly deposits.

The old deposit flow picked buys by ranking momentum-ish scores each month,
gave a bonus to tickers NOT already held, and split fixed per-slot budgets by
whole-share flooring. Over repeated deposits that fragments the portfolio
(28 positions, near-duplicate funds, share price deciding allocation) instead
of converging it.

This module replaces that with the boring, convergent mechanism:

1. Every ticker is classified into an equivalence GROUP (SPY==VOO==IVV,
   BND==AGG, TIP==SCHP==VTIP, VEA==IEFA, XLK==VGT==...). Near-duplicates are
   one bucket, so deposits top up what you own instead of buying a twin.
2. Each group has a TARGET weight. Targets implement the app's stated 80/20
   strategy (50% core / 30% satellite / 20% bonds) at group granularity.
3. Each deposit is allocated greedily, one share at a time, to the group with
   the largest remaining dollar gap below its target. Whole-share flooring
   stops being an allocator: an expensive share simply waits until its group's
   gap is large enough to afford it, instead of being skipped forever.
4. Within a group, money goes to the held ticker with the most value
   (consolidation); a new ticker is bought only when the group is empty.

Pure functions, no I/O - the DepositAdvisor supplies prices and holdings.
"""
from typing import Dict, List, Optional

# Group definitions. ``tickers`` are ordered: the FIRST is the default
# instrument when the group is not yet held. Targets sum to 1.0:
#   Core 50 = US_CORE 30 + INTL_DEVELOPED 14 + EMERGING 6
#   Satellite 30 = SMALL_CAP 5 + US_STYLE 5 + TECH 8 + HEALTHCARE 4 + ENERGY 4 + INFRASTRUCTURE 4
#   Bonds 20 = BONDS_AGG 12 + TIPS 8
TARGET_GROUPS: List[Dict] = [
    {"key": "US_CORE", "category": "CORE", "target": 0.30,
     "tickers": ["SPY", "VOO", "IVV"]},
    {"key": "INTL_DEVELOPED", "category": "CORE", "target": 0.14,
     "tickers": ["VEA", "VXUS", "IEFA"]},
    {"key": "EMERGING", "category": "CORE", "target": 0.06,
     "tickers": ["VWO", "EEM"]},
    {"key": "US_SMALL_CAP", "category": "SATELLITE", "target": 0.05,
     "tickers": ["IWM", "VB"]},
    {"key": "US_STYLE", "category": "SATELLITE", "target": 0.05,
     "tickers": ["VTV", "IVW", "SCHD"]},
    {"key": "TECH", "category": "SATELLITE", "target": 0.08,
     "tickers": ["XLK", "VGT", "SMH", "HACK", "ROBO", "QTUM", "DRIV"]},
    {"key": "HEALTHCARE", "category": "SATELLITE", "target": 0.04,
     "tickers": ["XLV", "VHT", "XBI"]},
    {"key": "ENERGY", "category": "SATELLITE", "target": 0.04,
     "tickers": ["XLE", "VDE", "ICLN"]},
    {"key": "INFRASTRUCTURE", "category": "SATELLITE", "target": 0.04,
     "tickers": ["PAVE", "IFRA"]},
    {"key": "BONDS_AGG", "category": "BONDS", "target": 0.12,
     "tickers": ["BND", "AGG"]},
    {"key": "TIPS", "category": "BONDS", "target": 0.08,
     "tickers": ["VTIP", "TIP", "SCHP"]},
]

GROUP_BY_KEY: Dict[str, Dict] = {g["key"]: g for g in TARGET_GROUPS}
_TICKER_TO_GROUP: Dict[str, str] = {
    t: g["key"] for g in TARGET_GROUPS for t in g["tickers"]
}

assert abs(sum(g["target"] for g in TARGET_GROUPS) - 1.0) < 1e-9


def classify(ticker: str) -> Optional[str]:
    """Group key for a ticker, or None for tickers outside the model
    (they keep their value weight but never receive new money)."""
    return _TICKER_TO_GROUP.get((ticker or "").upper())


def group_values(holdings: List[Dict]) -> Dict[str, float]:
    """Sum current_value per group key; unclassified tickers under '_OTHER'."""
    values: Dict[str, float] = {g["key"]: 0.0 for g in TARGET_GROUPS}
    values["_OTHER"] = 0.0
    for h in holdings:
        key = classify(h.get("ticker", "")) or "_OTHER"
        values[key] += float(h.get("current_value", 0) or 0)
    return values


# Bounded trend tilt: trends may reweight SATELLITE groups by at most this
# factor around their base target (0.5x..1.5x), and choose which ticker ENTERS
# an empty satellite group - but they never touch core/bond targets, never grow
# the satellite budget as a whole, and never add a duplicate to a held group.
SATELLITE_TILT_STRENGTH = 0.5
_TILT_MIN, _TILT_MAX = 0.5, 1.5


def choose_instrument(group_key: str, holdings: List[Dict],
                      momentum_by_ticker: Optional[Dict[str, float]] = None) -> str:
    """The ticker new money in this group should buy.

    If a group member is already held, keep consolidating into the largest one
    (one ticker per group - no fragmentation). Only for an EMPTY group may
    momentum pick which member to enter; otherwise the group's default.
    """
    group = GROUP_BY_KEY[group_key]
    members = [t.upper() for t in group["tickers"]]
    held = [h for h in holdings
            if (h.get("ticker") or "").upper() in set(members)
            and float(h.get("quantity", 0) or 0) > 0]
    if held:
        best = max(held, key=lambda h: float(h.get("current_value", 0) or 0))
        return best["ticker"].upper()
    if momentum_by_ticker and group["category"] == "SATELLITE":
        ranked = [(momentum_by_ticker.get(m), m) for m in members
                  if momentum_by_ticker.get(m) is not None]
        if ranked:
            return max(ranked)[1]
    return group["tickers"][0]


def tilt_satellite_targets(momentum_by_group: Dict[str, float],
                           strength: float = SATELLITE_TILT_STRENGTH) -> Dict[str, float]:
    """Reweight the SATELLITE group targets by relative momentum, bounded.

    Core and bond targets are untouched, the satellite total is preserved
    exactly (so 80/20 and 50/30/20 hold), and each satellite group moves at
    most _TILT_MIN.._TILT_MAX around its base weight. Returns a full
    {group_key: target} dict usable as ``gap_fill_allocate(targets=...)``.
    """
    out = {g["key"]: g["target"] for g in TARGET_GROUPS}
    sats = [g for g in TARGET_GROUPS if g["category"] == "SATELLITE"]
    if not momentum_by_group or not sats:
        return out

    base = {g["key"]: g["target"] for g in sats}
    sat_total = sum(base.values())
    m = {k: float(momentum_by_group.get(k, 0.0) or 0.0) for k in base}
    vals = list(m.values())
    mean = sum(vals) / len(vals)
    half_spread = (max(vals) - min(vals)) / 2.0 or 1.0

    raw = {}
    for k in base:
        z = (m[k] - mean) / half_spread          # ~[-1, 1]
        factor = min(_TILT_MAX, max(_TILT_MIN, 1.0 + strength * z))
        raw[k] = base[k] * factor
    scale = sat_total / sum(raw.values()) if sum(raw.values()) > 0 else 1.0
    for k in base:
        out[k] = raw[k] * scale
    return out


def gap_fill_allocate(holdings: List[Dict], budget_usd: float,
                      prices: Dict[str, float],
                      targets: Optional[Dict[str, float]] = None,
                      momentum_by_ticker: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Allocate ``budget_usd`` across the target groups, one share at a time,
    always to the group with the largest remaining dollar gap.

    ``prices`` must contain a positive price for each group's chosen
    instrument; groups with a missing/invalid price are skipped (never guessed).
    ``targets`` optionally overrides per-group target weights (e.g. from
    ``tilt_satellite_targets``); ``momentum_by_ticker`` optionally lets trend
    pick which ticker enters an empty satellite group.

    Returns aggregated buys:
      [{ticker, shares, price, amount, group, category, is_new}], largest first.
    """
    if budget_usd is None or budget_usd <= 0:
        return []

    tgt = {g["key"]: g["target"] for g in TARGET_GROUPS}
    if targets:
        tgt.update({k: v for k, v in targets.items() if k in tgt})

    values = group_values(holdings)
    total_after = sum(values.values()) + budget_usd
    held_tickers = {(h.get("ticker") or "").upper() for h in holdings
                    if float(h.get("quantity", 0) or 0) > 0}

    instruments: Dict[str, Dict] = {}
    for g in TARGET_GROUPS:
        ticker = choose_instrument(g["key"], holdings, momentum_by_ticker)
        price = float(prices.get(ticker, 0) or 0)
        if price <= 0:
            continue
        instruments[g["key"]] = {
            "ticker": ticker,
            "price": price,
            "is_new": ticker not in held_tickers,
        }

    remaining = float(budget_usd)
    bought: Dict[str, Dict] = {}
    while True:
        best_key, best_gap = None, 0.0
        for key, inst in instruments.items():
            if inst["price"] > remaining:
                continue
            gap = tgt[key] * total_after - values[key]
            # Buy only while the group stays at-or-below target after the share.
            if gap >= inst["price"] * 0.5 and gap > best_gap:
                best_key, best_gap = key, gap
        if best_key is None:
            break
        inst = instruments[best_key]
        values[best_key] += inst["price"]
        remaining -= inst["price"]
        rec = bought.setdefault(best_key, {
            "ticker": inst["ticker"],
            "shares": 0,
            "price": inst["price"],
            "amount": 0.0,
            "group": best_key,
            "category": GROUP_BY_KEY[best_key]["category"],
            "is_new": inst["is_new"],
        })
        rec["shares"] += 1
        rec["amount"] = round(rec["shares"] * inst["price"], 2)

    return sorted(bought.values(), key=lambda r: -r["amount"])


def allocation_report(holdings: List[Dict], cash_usd: float = 0.0) -> List[Dict]:
    """Current vs target weight per group - for display.

    Returns [{group, category, target_pct, current_pct, gap_usd, instrument}].
    """
    values = group_values(holdings)
    total = sum(values.values()) + max(cash_usd or 0, 0)
    if total <= 0:
        return []
    rows = []
    for g in TARGET_GROUPS:
        current = values[g["key"]]
        rows.append({
            "group": g["key"],
            "category": g["category"],
            "target_pct": g["target"] * 100.0,
            "current_pct": current / total * 100.0,
            "gap_usd": round(g["target"] * total - current, 2),
            "instrument": choose_instrument(g["key"], holdings),
        })
    if values["_OTHER"] > 0:
        rows.append({
            "group": "_OTHER (no new money)",
            "category": "OTHER",
            "target_pct": 0.0,
            "current_pct": values["_OTHER"] / total * 100.0,
            "gap_usd": 0.0,
            "instrument": "-",
        })
    return rows
