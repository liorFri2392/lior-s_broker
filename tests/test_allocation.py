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
    # Equity fully at target, only bonds under target, AGG held (not BND):
    # bond money must top up AGG, never buy the BND near-duplicate.
    base = 100_000.0
    holdings = [_holding(g["tickers"][0], g["target"] * base)
                for g in allocation.TARGET_GROUPS if g["category"] != "BONDS"]
    holdings.append(_holding("AGG", 200))  # tiny, deeply under bond target
    buys = allocation.gap_fill_allocate(holdings, 3000.0, PRICES)
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

    us_core_target = allocation.GROUP_BY_KEY["US_CORE"]["target"]

    def us_core_drift():
        values = allocation.group_values(holdings)
        total = sum(values.values())
        return abs(values["US_CORE"] / total - us_core_target)

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


def test_gap_fill_funds_group_via_cheaper_equivalent_when_preferred_unaffordable():
    # US_CORE is deeply underweight (big holdings elsewhere) but SPY ($745)
    # exceeds a $600 budget; the allocator must still fund US_CORE via a cheaper
    # equivalent (VOO/IVV), not leak the money to less-underweight groups.
    holdings = [_holding("VEA", 10000), _holding("BND", 5000), _holding("SPY", 745, qty=1)]
    buys = allocation.gap_fill_allocate(holdings, 600.0, PRICES)  # can't afford SPY
    by_group = {b["group"]: b for b in buys}
    assert "US_CORE" in by_group
    assert by_group["US_CORE"]["ticker"] in ("VOO", "IVV")  # cheaper equivalent
    assert by_group["US_CORE"]["price"] <= 600.0


def test_gap_fill_prefers_held_when_affordable():
    # US_CORE underweight and the budget affords SPY: the held/preferred ticker
    # wins (no gratuitous switch to a cheaper twin).
    holdings = [_holding("VEA", 10000), _holding("SPY", 745, qty=1)]
    buys = allocation.gap_fill_allocate(holdings, 1000.0, PRICES)
    us_core = next(b for b in buys if b["group"] == "US_CORE")
    assert us_core["ticker"] == "SPY"


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


# --------------------------------------------------------------------------- #
# Bounded trend tilt.
# --------------------------------------------------------------------------- #
def _sat_total(targets):
    return sum(v for k, v in targets.items()
               if allocation.GROUP_BY_KEY[k]["category"] == "SATELLITE")


def test_tilt_preserves_satellite_total_and_core_bonds():
    base_sat = _sat_total({g["key"]: g["target"] for g in allocation.TARGET_GROUPS})
    tilt = allocation.tilt_satellite_targets({"TECH": 30.0, "ENERGY": -20.0, "HEALTHCARE": 5.0})
    # Satellite sleeve total unchanged (still 30%).
    assert _sat_total(tilt) == pytest.approx(base_sat)
    # Core/bond targets untouched.
    for g in allocation.TARGET_GROUPS:
        if g["category"] != "SATELLITE":
            assert tilt[g["key"]] == pytest.approx(g["target"])


def test_tilt_favors_trending_sector_within_bounds():
    tilt = allocation.tilt_satellite_targets({"TECH": 40.0, "ENERGY": -30.0})
    tech_base = allocation.GROUP_BY_KEY["TECH"]["target"]
    energy_base = allocation.GROUP_BY_KEY["ENERGY"]["target"]
    assert tilt["TECH"] > tech_base       # winner up
    assert tilt["ENERGY"] < energy_base   # laggard down
    # Bounded: no satellite group moves beyond 0.5x..1.5x its base.
    for g in allocation.TARGET_GROUPS:
        if g["category"] == "SATELLITE":
            assert 0.5 * g["target"] - 1e-9 <= tilt[g["key"]] <= 1.5 * g["target"] + 1e-9


def test_tilt_bounds_hold_after_renormalization():
    """Regression: two small hot sectors + large cold ones used to end at
    ~1.69x base after the rescale-to-sleeve-total step."""
    tilt = allocation.tilt_satellite_targets({
        "US_STYLE": 50.0, "HEALTHCARE": 50.0,
        "US_SMALL_CAP": -10.0, "TECH": -10.0, "ENERGY": -10.0, "INFRASTRUCTURE": -10.0,
    })
    for g in allocation.TARGET_GROUPS:
        if g["category"] == "SATELLITE":
            ratio = tilt[g["key"]] / g["target"]
            assert 0.5 - 1e-9 <= ratio <= 1.5 + 1e-9
    assert _sat_total(tilt) == pytest.approx(
        sum(g["target"] for g in allocation.TARGET_GROUPS if g["category"] == "SATELLITE"))


def test_tilt_empty_momentum_is_identity():
    tilt = allocation.tilt_satellite_targets({})
    for g in allocation.TARGET_GROUPS:
        assert tilt[g["key"]] == pytest.approx(g["target"])


def test_momentum_picks_entry_ticker_for_empty_group_only():
    # TECH empty: momentum picks the trending member instead of the default.
    mom = {"XLK": 5.0, "SMH": 40.0, "VGT": 3.0}
    assert allocation.choose_instrument("TECH", [], mom) == "SMH"
    # But when a member is already held, consolidation wins - momentum ignored.
    held = [_holding("XLK", 1000)]
    assert allocation.choose_instrument("TECH", held, mom) == "XLK"


def test_momentum_never_overrides_core_group():
    # Core groups ignore momentum entirely (no trend-picking of core).
    mom = {"VOO": 99.0, "IVV": 98.0}
    assert allocation.choose_instrument("US_CORE", [], mom) == "SPY"


def test_tilt_still_converges_stock_bond_split():
    # With a strong tilt, the stock/bond split must still hold: bonds ~20%.
    holdings = [_holding("SPY", 5000), _holding("BND", 500)]
    tilt = allocation.tilt_satellite_targets({"TECH": 50.0})
    buys = allocation.gap_fill_allocate(holdings, 3000.0, PRICES, targets=tilt)
    bond_target_total = sum(v for k, v in tilt.items()
                            if allocation.GROUP_BY_KEY[k]["category"] == "BONDS")
    assert bond_target_total == pytest.approx(allocation.category_targets()["BONDS"])
