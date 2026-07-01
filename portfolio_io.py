"""portfolio_io.py - Shared portfolio.json loading.

The identical ``load_portfolio`` body was previously copy-pasted into four
classes. It now lives here; each class delegates to ``load_portfolio``.
"""
import json
import os
from typing import Dict


def default_portfolio() -> Dict:
    """The canonical empty portfolio returned when no file exists."""
    return {"cash": 0, "holdings": [], "last_updated": None, "total_value": 0}


def load_portfolio(portfolio_file: str) -> Dict:
    """Load a portfolio from ``portfolio_file``; return an empty default if absent."""
    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return default_portfolio()
