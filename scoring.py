"""scoring.py - Shared scoring primitives.

``portfolio_analyzer.analyze_holding`` (scores existing holdings) and
``deposit_advisor.analyze_etf`` (scores buy candidates) are intentionally
*different* models — they weigh different features (a holding is judged on
RSI / Sortino / news sentiment; a candidate on Sharpe-lower-bound / expense
ratio / liquidity). But both build a ``{name: 0..1}`` sub-score dict, normalize
each feature the same way, and combine with the same weighted sum. Those
normalization formulas were copy-pasted into both; they live here now so the two
engines share one definition of "normalize a Sharpe / momentum / volatility into
[0, 1]" without being forced into one model.

Every function here is pure and preserves the original arithmetic exactly.
"""
from typing import Dict


def clamp01(x: float) -> float:
    """Clamp a value into the [0, 1] range."""
    return max(0.0, min(1.0, x))


def sharpe_score(sharpe: float) -> float:
    """Normalize a Sharpe (or Sortino) ratio to [0, 1]: maps [-1, 3] -> [0, 1]."""
    return clamp01((sharpe + 1) / 4.0)


def momentum_score(momentum_pct: float) -> float:
    """Normalize a momentum percentage to [0, 1]: maps [-15, 15] -> [0, 1]."""
    return clamp01((momentum_pct + 15) / 30.0)


def volatility_score(volatility_pct: float) -> float:
    """Normalize volatility to [0, 1] (lower is better): maps [5, 40] -> [1, 0]."""
    return clamp01(1.0 - (volatility_pct - 5) / 35.0)


def weighted_score(sub_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted sum of sub-scores; missing sub-scores default to a neutral 0.5."""
    return sum(sub_scores.get(k, 0.5) * w for k, w in weights.items())
