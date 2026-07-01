"""etf_universe.py - Single source of truth for the ETF classifications.

These lists were previously duplicated across ``portfolio_analyzer``,
``deposit_advisor`` and ``critical_alert``. They now live here and are imported
everywhere so the 80/20 core/satellite/bond strategy and the leveraged-ETF
exclusions stay consistent across every module.
"""
from typing import List, Set

# 80/20 strategy ETF universes.
CORE_ETFS: List[str] = ["SPY", "VOO", "IVV", "VXUS", "VEA"]
BOND_ETFS: List[str] = ["BND", "AGG", "TIP", "SCHP", "VTIP"]
SATELLITE_ETFS: List[str] = ["IWM", "VB", "XLK", "VGT", "VWO", "EEM", "XLV", "VHT"]

# Leveraged / inverse ETFs - excluded from the balanced strategy.
LEVERAGED_2X: List[str] = ["SSO", "QLD", "UWM", "EFO"]
LEVERAGED_3X: List[str] = ["TQQQ", "SPXL", "UPRO", "TNA", "FAS", "CURE", "SOXL", "LABU", "TECL"]
LEVERAGED_INVERSE: List[str] = ["SQQQ", "SPXS", "SPXU", "TZA", "FAZ", "SOXS", "LABD", "TECS"]
ALL_LEVERAGED_TICKERS: Set[str] = set(LEVERAGED_2X) | set(LEVERAGED_3X) | set(LEVERAGED_INVERSE)

# Satellite ETF categories used by the critical-alert / deposit scan.
SATELLITE_CATEGORIES: List[str] = [
    "US_SMALL_CAP", "TECHNOLOGY", "HEALTHCARE", "EMERGING_MARKETS",
    "AI_AND_ROBOTICS", "QUANTUM_COMPUTING", "SEMICONDUCTORS", "CLOUD_COMPUTING", "CYBERSECURITY",
    "ELECTRIC_VEHICLES", "CLEAN_ENERGY", "REAL_ESTATE", "INFRASTRUCTURE",
    "DIVIDEND", "GROWTH", "VALUE", "FINANCIAL", "ENERGY", "CONSUMER", "ESG", "BIOTECH",
]

# ETF_CATEGORIES keys that are never eligible for the balanced strategy.
EXCLUDED_CATEGORIES: List[str] = ["LEVERAGED_2X", "LEVERAGED_3X", "LEVERAGED_INVERSE", "CRYPTO"]
