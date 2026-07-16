#!/usr/bin/env python3
"""
Critical Alert System - Detects urgent portfolio actions
Runs deep analysis to find critical buy/sell opportunities
"""

import json
import math
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from portfolio_analyzer import PortfolioAnalyzer
from email_notifier import EmailNotifier
from etf_universe import SATELLITE_CATEGORIES, EXCLUDED_CATEGORIES
import allocation
import market_data

logger = logging.getLogger(__name__)

class CriticalAlertSystem:
    """Detects and alerts on critical portfolio opportunities."""
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.analyzer = PortfolioAnalyzer(portfolio_file)
        self.critical_threshold = 75  # Score threshold for critical buy
        self.urgent_sell_threshold = 25  # Score threshold for urgent sell
        self.review_cooldown_days = 30  # Align with rebalance: fewer urgent emails soon after analyze/rebalance

    def _shares_for_amount(self, ticker: str, amount: float, known_price: float = 0) -> Tuple[int, float, float]:
        """
        Given a dollar ``amount`` to spend on ``ticker``, return
        (shares, price, remaining_cash). If ``known_price`` is not supplied,
        fetch it from the shared cached market-data layer.

        Buys as many whole shares as ``amount`` allows; if >$50 is left over and
        it covers one more share, buys that extra share. Returns (0, price, 0)
        when no price is available or not even one share is affordable.
        """
        price = float(known_price or 0)
        if price <= 0:
            price = float(market_data.get_price(ticker) or 0)
        if price <= 0:
            return 0, 0.0, 0.0

        shares = int(amount / price)
        spent = shares * price
        remaining = amount - spent
        # If we have significant remaining cash (>$50) and it covers another
        # share, buy one more.
        if remaining > 50 and remaining >= price:
            shares += 1
            spent = shares * price
            remaining = amount - spent
        if shares <= 0:
            return 0, price, 0.0
        return shares, price, remaining

    def _within_review_cooldown(self, portfolio: Dict) -> bool:
        """True if last analyze or rebalance was within review_cooldown_days."""
        today = datetime.now().date()
        for key in ("last_analyze_run_date", "last_rebalancing_date"):
            s = portfolio.get(key)
            if not s:
                continue
            try:
                d = datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
                if (today - d).days < self.review_cooldown_days:
                    return True
            except (ValueError, TypeError):
                continue
        return False
    
    def _refine_critical_items(
        self,
        items: List[Dict],
        portfolio: Dict,
        total_value: float,
    ) -> Tuple[List[Dict], bool]:
        """
        Dedupe SELL by ticker, fix 0-share math, drop empty concentration rows,
        and during cooldown skip urgent sells for tiny positions unless score is very low.
        Returns (refined_items, any_critical_priority).
        """
        within_cd = self._within_review_cooldown(portfolio)
        holdings_by_ticker = {h["ticker"].upper(): h for h in portfolio.get("holdings", [])}
        priority_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        
        non_sell: List[Dict] = []
        sells: List[Dict] = []
        for item in items:
            if item.get("type") != "SELL":
                non_sell.append(item)
                continue
            reason = item.get("reason", "")
            # Drop bogus concentration lines (0 shares, ~0 amount)
            if "Over-concentration" in reason and item.get("shares", 0) == 0 and item.get("amount", 0) < 1:
                continue
            t = (item.get("ticker") or "").upper()
            h = holdings_by_ticker.get(t, {})
            qty = int(h.get("quantity", 0) or 0)
            price = float(item.get("current_price") or h.get("last_price", 0) or 0)
            pos_value = float(h.get("current_value", 0) or (qty * price if price else 0))
            weight = (pos_value / total_value) if total_value and total_value > 0 else 0
            score = float(item.get("score", 50))
            
            if "STRONG SELL" in reason or "Risk of significant loss" in reason:
                if within_cd and weight < 0.01 and score >= 15:
                    continue
                if qty > 0 and price > 0:
                    sh = max(1, round(qty * 0.5))
                    item["shares"] = sh
                    item["amount"] = round(sh * price, 2)
            elif qty > 0 and price > 0 and item.get("shares", 0) == 0 and item.get("amount", 0) > 0:
                item["shares"] = max(1, int(item["amount"] / price))
                item["amount"] = round(item["shares"] * price, 2)
            
            sells.append(item)
        
        # One SELL per ticker: keep best priority (CRITICAL over HIGH); prefer STRONG SELL over concentration
        best_by_ticker: Dict[str, Dict] = {}
        for item in sells:
            t = (item.get("ticker") or "").upper()
            if not t:
                continue
            cur = best_by_ticker.get(t)
            if cur is None:
                best_by_ticker[t] = item
                continue
            r_new = priority_rank.get(item.get("priority", "MEDIUM"), 3)
            r_old = priority_rank.get(cur.get("priority", "MEDIUM"), 3)
            if r_new < r_old:
                best_by_ticker[t] = item
            elif r_new == r_old and "STRONG SELL" in item.get("reason", "") and "STRONG SELL" not in cur.get("reason", ""):
                best_by_ticker[t] = item
        
        merged_sells = list(best_by_ticker.values())

        # A ticker may be sold by at most ONE item. REPLACE entries whose
        # sell_ticker already has a SELL (or an earlier REPLACE) are dropped -
        # otherwise apply-time decrements the same position twice and credits
        # cash for shares that were already sold.
        sell_tickers = set(best_by_ticker.keys())
        seen_replace_sells: set = set()
        deduped_non_sell: List[Dict] = []
        for item in non_sell:
            if item.get("type") == "REPLACE":
                st = (item.get("sell_ticker") or "").upper()
                if st and (st in sell_tickers or st in seen_replace_sells):
                    continue
                if st:
                    seen_replace_sells.add(st)
            deduped_non_sell.append(item)

        merged = deduped_non_sell + merged_sells
        any_critical = any(
            i.get("type") == "SELL" and i.get("priority") == "CRITICAL"
            for i in merged_sells
        )
        return merged, any_critical
    
    def is_market_trading_day(self) -> bool:
        """Check if today is a US market trading day (excludes holidays)."""
        try:
            # Use SPY as market indicator (cached; reused by check_market_anomalies).
            today = datetime.now().date()

            # Try to get today's data
            hist = market_data.get_history("SPY", period="5d")
            if hist is not None and not hist.empty:
                last_date = hist.index[-1].date()
                # If last trading day is today or yesterday, market is likely open
                days_diff = (today - last_date).days
                return days_diff <= 1
            
            # Fallback: check if it's a weekday (Mon-Fri)
            weekday = today.weekday()  # 0=Monday, 6=Sunday
            return weekday < 5  # Monday to Friday
            
        except Exception as e:
            logger.warning(f"Could not determine market status: {e}")
            # Fallback to weekday check
            weekday = datetime.now().weekday()
            return weekday < 5
    
    def check_critical_opportunities(self) -> Dict:
        """Genuine-emergency scan - deliberately NO scoring and NO churn.

        The old version ran the full scoring engine, scanned 60 buy candidates,
        hunted "better alternatives" and hot-sector replacements - machinery
        that recommended taxable fund-switching on noisy monthly signals. The
        daily alert now covers exactly what warrants an email:
          1. stop-loss / take-profit triggers (cost-basis based),
          2. extreme single-position concentration (>50%),
          3. strategy drift beyond 10 percentage points (informational plan),
          4. market anomaly (extreme SPY one-day move).
        """
        print("=" * 60)
        print("CRITICAL ALERT SYSTEM - Risk & Drift Scan")
        print("=" * 60)

        if not self.is_market_trading_day():
            return {
                "critical_items": [],
                "has_critical": False,
                "message": "Not a trading day - no analysis performed"
            }

        print("Refreshing prices...")
        portfolio = self.analyzer.refresh_portfolio_prices(verbose=False, sync_github_secret=False)
        holdings = portfolio.get("holdings", [])
        cash = float(portfolio.get("cash", 0) or 0)
        total_value = cash + sum(float(h.get("current_value", 0) or 0) for h in holdings)
        critical_items: List[Dict] = []

        # 1. Stop-loss / take-profit triggers (genuine risk events).
        try:
            from risk_manager import RiskManager
            rm = RiskManager(self.portfolio_file)
            for act in rm.check_stop_loss_take_profit(portfolio):
                qty = int(act.get("quantity", 0) or 0)
                price = float(act.get("current_price", 0) or 0)
                if qty <= 0 or price <= 0:
                    continue
                critical_items.append({
                    "type": "SELL",
                    "ticker": act.get("ticker", ""),
                    "priority": act.get("priority", "HIGH"),
                    "reason": act.get("reason", "Risk trigger"),
                    "amount": round(qty * price, 2),
                    "shares": qty,
                    "score": 0 if act.get("priority") == "CRITICAL" else 50,
                    "current_price": price,
                    "quantity": qty,
                })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Risk-manager stop-loss check failed: {e}")

        # 2. Extreme concentration (>50% in one position) - pure risk rule.
        if total_value > 0:
            for h in holdings:
                weight = float(h.get("current_value", 0) or 0) / total_value
                if weight > 0.5:
                    price = float(h.get("last_price", 0) or 0)
                    qty = int(h.get("quantity", 0) or 0)
                    excess_shares = max(1, int(qty * (weight - 0.35) / weight)) if qty else 0
                    critical_items.append({
                        "type": "SELL",
                        "ticker": h.get("ticker", ""),
                        "priority": "HIGH",
                        "reason": f"Over-concentration: {weight*100:.1f}% of portfolio in one position (risk limit 50%)",
                        "amount": round(excess_shares * price, 2),
                        "shares": excess_shares,
                        "score": 50,
                        "current_price": price,
                        "quantity": qty,
                    })

        # 3. Strategy drift beyond 10 percentage points -> informational item.
        try:
            analyses = [{"ticker": h.get("ticker", ""),
                         "current_value": float(h.get("current_value", 0) or 0)}
                        for h in holdings]
            balance = self.analyzer.check_80_20_balance({"total_value": total_value}, analyses)
            tgts = balance.get("targets", {})
            drift = max(
                abs(balance.get("stocks_percent", 0) - tgts.get("stocks", 85)),
                abs(balance.get("bonds_percent", 0) - tgts.get("bonds", 15)),
                abs(balance.get("core_percent", 0) - tgts.get("core", 70)),
            )
            if drift >= 10 and not self._within_review_cooldown(portfolio):
                critical_items.append({
                    "type": "REBALANCE",
                    "priority": "MEDIUM",
                    "reason": (f"Allocation drift of {drift:.0f} percentage points from target "
                               f"(core {balance.get('core_percent', 0):.0f}% vs {tgts.get('core', 70):.0f}%, "
                               f"bonds {balance.get('bonds_percent', 0):.0f}% vs {tgts.get('bonds', 15):.0f}%)"),
                    "details": "Your regular deposits close this gap tax-free over time.",
                })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Drift check failed: {e}")

        # 4. Market anomalies (extreme one-day SPY moves).
        critical_items.extend(self.check_market_anomalies())

        critical_items, any_critical = self._refine_critical_items(
            critical_items, portfolio, total_value
        )

        return {
            "critical_items": critical_items,
            "has_critical": bool(critical_items),
            "any_critical_sell": any_critical,
            "message": f"Found {len(critical_items)} alert(s)" if critical_items else "No alerts",
        }

    def check_market_anomalies(self) -> List[Dict]:
        """Check for market anomalies that require immediate attention."""
        anomalies = []
        
        try:
            # Check SPY for extreme movements (cached; usually a hit from
            # is_market_trading_day's earlier 5d fetch).
            hist = market_data.get_history("SPY", period="5d")

            if hist is not None and not hist.empty and 'Close' in hist.columns and len(hist) >= 2:
                recent_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                
                # Extreme market drop (>3% in one day)
                if recent_return < -3:
                    anomalies.append({
                        "type": "ACTION",
                        "title": "Market Drop Alert",
                        "priority": "HIGH",
                        "reason": f"SPY dropped {abs(recent_return):.2f}% - Consider defensive positions",
                        "details": "Market experiencing significant decline. Review portfolio for risk exposure."
                    })
                
                # Extreme market surge (>3% in one day)
                elif recent_return > 3:
                    anomalies.append({
                        "type": "ACTION",
                        "title": "Market Surge Alert",
                        "priority": "MEDIUM",
                        "reason": f"SPY surged {recent_return:.2f}% - Consider taking profits",
                        "details": "Market experiencing significant gains. Consider rebalancing to lock in profits."
                    })
        except Exception as e:
            logger.warning(f"Failed to check market anomalies: {e}")
            pass
        
        return anomalies
    
    def apply_recommendations_to_portfolio(self, critical_items: List[Dict], auto_apply: bool = True) -> bool:
        """
        Apply recommendations from critical_items to portfolio.json.
        
        Args:
            critical_items: List of recommendation dictionaries
            auto_apply: If True, apply automatically. If False, ask for confirmation.
                      Default is True - automatically apply when email is sent.
        
        Returns:
            True if portfolio was updated, False otherwise
        """
        # Filter only actionable items (REPLACE, SELL, BUY)
        actionable_items = [item for item in critical_items if item.get("type") in ["REPLACE", "SELL", "BUY"]]
        
        if not actionable_items:
            return False
        
        # Load current portfolio
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                portfolio = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            print(f"❌ Error loading portfolio: {e}")
            return False
        
        # Ask for confirmation only if explicitly disabled
        if not auto_apply:
            print("\n" + "="*60)
            print("APPLY RECOMMENDATIONS TO PORTFOLIO?")
            print("="*60)
            print(f"Found {len(actionable_items)} actionable recommendation(s):")
            for item in actionable_items:
                action_type = item.get("type")
                if action_type == "REPLACE":
                    print(f"  🔄 REPLACE: {item.get('sell_ticker')} → {item.get('buy_ticker')}")
                elif action_type == "SELL":
                    print(f"  🔴 SELL: {item.get('ticker')} ({item.get('shares', 0)} shares)")
                elif action_type == "BUY":
                    print(f"  🟢 BUY: {item.get('ticker')} (${item.get('amount', 0):,.2f})")
            
            response = input("\nApply these recommendations to portfolio.json? (yes/no): ").strip().lower()
            if response not in ['yes', 'y', 'כן']:
                print("❌ Portfolio update cancelled.")
                return False
        
        # Apply each recommendation
        print("\n📝 Auto-applying recommendations to portfolio...")
        
        for item in actionable_items:
            action_type = item.get("type")
            
            if action_type == "REPLACE":
                # Handle REPLACE: Sell one ticker, buy another
                sell_ticker = item.get("sell_ticker")
                buy_ticker = item.get("buy_ticker")
                sell_shares = item.get("sell_shares", 0)
                sell_amount = item.get("sell_amount", 0)
                buy_shares = item.get("buy_shares", 0)
                buy_price = item.get("buy_price", 0)
                buy_amount = item.get("buy_amount", 0)
                remaining_cash = item.get("remaining_cash", 0)
                
                # Find and update sell_ticker
                sell_holding = None
                for holding in portfolio.get("holdings", []):
                    if holding.get("ticker") == sell_ticker:
                        sell_holding = holding
                        break
                
                if sell_holding:
                    old_qty = sell_holding.get("quantity", 0)
                    new_qty = max(0, old_qty - sell_shares)
                    sell_holding["quantity"] = new_qty
                    if new_qty > 0:
                        sell_holding["current_value"] = new_qty * sell_holding.get("last_price", 0)
                    else:
                        sell_holding["current_value"] = 0
                    print(f"  ✅ SELL: {sell_ticker} - {old_qty} → {new_qty} shares")
                else:
                    print(f"  ⚠️  Warning: {sell_ticker} not found in portfolio")
                
                # Find or create buy_ticker
                buy_holding = None
                for holding in portfolio.get("holdings", []):
                    if holding.get("ticker") == buy_ticker:
                        buy_holding = holding
                        break
                
                if buy_holding:
                    old_qty = buy_holding.get("quantity", 0)
                    old_cb = (buy_holding.get("cost_basis")
                              or buy_holding.get("purchase_price")
                              or buy_holding.get("last_price", buy_price))
                    new_qty = old_qty + buy_shares
                    if new_qty > 0:
                        buy_holding["cost_basis"] = round(
                            (old_qty * old_cb + buy_shares * buy_price) / new_qty, 4
                        )
                    buy_holding["quantity"] = new_qty
                    buy_holding["last_price"] = buy_price
                    buy_holding["current_value"] = new_qty * buy_price
                    print(f"  ✅ BUY: {buy_ticker} - {old_qty} → {new_qty} shares @ ${buy_price:.2f}")
                else:
                    new_holding = {
                        "ticker": buy_ticker,
                        "quantity": buy_shares,
                        "cost_basis": round(buy_price, 4),
                        "last_price": buy_price,
                        "current_value": buy_shares * buy_price,
                    }
                    portfolio.setdefault("holdings", []).append(new_holding)
                    print(f"  ✅ BUY: {buy_ticker} - NEW holding: {buy_shares} shares @ ${buy_price:.2f}")
                
                # Update cash: add sell_amount, subtract buy_amount
                portfolio["cash"] = portfolio.get("cash", 0) + sell_amount - buy_amount
                
            elif action_type == "SELL":
                # Handle SELL
                ticker = item.get("ticker")
                shares_to_sell = item.get("shares", 0)
                sell_amount = item.get("amount", 0)
                
                sell_holding = None
                for holding in portfolio.get("holdings", []):
                    if holding.get("ticker") == ticker:
                        sell_holding = holding
                        break
                
                if sell_holding:
                    old_qty = sell_holding.get("quantity", 0)
                    new_qty = max(0, old_qty - shares_to_sell)
                    sell_holding["quantity"] = new_qty
                    if new_qty > 0:
                        sell_holding["current_value"] = new_qty * sell_holding.get("last_price", 0)
                    else:
                        sell_holding["current_value"] = 0
                    portfolio["cash"] = portfolio.get("cash", 0) + sell_amount
                    print(f"  ✅ SELL: {ticker} - {old_qty} → {new_qty} shares (+${sell_amount:.2f} cash)")
                else:
                    print(f"  ⚠️  Warning: {ticker} not found in portfolio")
            
            elif action_type == "BUY":
                # Handle BUY
                ticker = item.get("ticker")
                buy_amount = item.get("amount", 0)
                # Try to get shares and price from item
                buy_shares = item.get("shares", 0)
                buy_price = item.get("price", 0)
                
                if buy_shares == 0 and buy_price > 0:
                    buy_shares = int(buy_amount / buy_price)
                elif buy_price == 0 and buy_shares > 0:
                    buy_price = buy_amount / buy_shares if buy_shares > 0 else 0
                
                if buy_shares > 0 and buy_price > 0:
                    buy_holding = None
                    for holding in portfolio.get("holdings", []):
                        if holding.get("ticker") == ticker:
                            buy_holding = holding
                            break
                    
                    if buy_holding:
                        old_qty = buy_holding.get("quantity", 0)
                        buy_holding["quantity"] = old_qty + buy_shares
                        buy_holding["last_price"] = buy_price
                        buy_holding["current_value"] = buy_holding["quantity"] * buy_price
                        print(f"  ✅ BUY: {ticker} - {old_qty} → {buy_holding['quantity']} shares @ ${buy_price:.2f}")
                    else:
                        # Create new holding (cost_basis = price just paid, so
                        # downstream cost-basis/tax math has a value to work with).
                        new_holding = {
                            "ticker": ticker,
                            "quantity": buy_shares,
                            "cost_basis": round(buy_price, 4),
                            "last_price": buy_price,
                            "current_value": buy_shares * buy_price
                        }
                        portfolio.setdefault("holdings", []).append(new_holding)
                        print(f"  ✅ BUY: {ticker} - NEW holding: {buy_shares} shares @ ${buy_price:.2f}")
                    
                    portfolio["cash"] = max(0, portfolio.get("cash", 0) - buy_amount)
        
        # Remove holdings with 0 quantity
        portfolio["holdings"] = [h for h in portfolio.get("holdings", []) if h.get("quantity", 0) > 0]
        
        # Recalculate total value
        total_value = portfolio.get("cash", 0)
        for holding in portfolio.get("holdings", []):
            total_value += holding.get("current_value", 0)
        portfolio["total_value"] = total_value
        
        # Update timestamp
        portfolio["last_updated"] = datetime.now().isoformat()
        
        # Save portfolio
        try:
            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Portfolio updated and saved to {self.portfolio_file}")
            print(f"   Total Portfolio Value: ${total_value:,.2f}")
            print(f"   Cash: ${portfolio.get('cash', 0):,.2f}")
            
            # Try to update GitHub secret
            try:
                import subprocess
                portfolio_json_str = json.dumps(portfolio, ensure_ascii=False, indent=2)
                process = subprocess.Popen(
                    ["gh", "secret", "set", "PORTFOLIO_JSON", "--repo", "liorFri2392/lior-s_broker"],
                    stdin=subprocess.PIPE,
                    text=True
                )
                process.communicate(input=portfolio_json_str)
                if process.returncode == 0:
                    print("✅ GitHub secret updated successfully!")
                else:
                    print("⚠️  GitHub secret update failed (run 'make update-secret' manually)")
            except Exception as e:
                logger.debug(f"Failed to update GitHub secret: {e}")
                print("⚠️  GitHub secret update skipped (run 'make update-secret' manually)")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
            print(f"❌ Error saving portfolio: {e}")
            return False
    
    def send_alerts(self, results: Dict) -> bool:
        """Send email alerts if critical items found."""
        if not results.get("has_critical"):
            print("\n✅ No critical actions required at this time.")
            return False
        
        try:
            notifier = EmailNotifier()
            
            critical_items = results.get("critical_items", [])
            portfolio_value = results.get("portfolio_value", 0)
            portfolio_metrics = results.get("portfolio_metrics", {})
            
            # Count by type
            buy_count = sum(1 for item in critical_items if item.get("type") == "BUY")
            sell_count = sum(1 for item in critical_items if item.get("type") == "SELL")
            replace_count = sum(1 for item in critical_items if item.get("type") == "REPLACE")
            info_count = sum(1 for item in critical_items if item.get("type") in ("REBALANCE", "ACTION"))
            trend_count = sum(1 for item in critical_items if item.get("type") == "EMERGING_TREND")
            
            # Only send email if there are specific actionable recommendations (BUY/SELL/REPLACE)
            # Trends alone are not actionable without cash or specific recommendations
            has_specific_actions = buy_count > 0 or sell_count > 0 or replace_count > 0 or info_count > 0
            
            if not has_specific_actions:
                print("\n✅ No specific actionable recommendations (only trends detected, but no cash or weak holdings to replace).")
                print("   Email not sent - trends are informational only when no actions are available.")
                return False
            
            # Build subject line (only for specific actions)
            actions = []
            if replace_count > 0:
                actions.append(f"{replace_count} Replace{'s' if replace_count != 1 else ''}")
            if buy_count > 0:
                actions.append(f"{buy_count} Buy{'s' if buy_count != 1 else ''}")
            if sell_count > 0:
                actions.append(f"{sell_count} Sell{'s' if sell_count != 1 else ''}")
            
            within_cd = results.get("within_review_cooldown", False)
            any_crit = results.get("any_critical_sell", True)
            if within_cd and not any_crit:
                subject = f"Review: {', '.join(actions)} suggested" if actions else "Portfolio review"
            else:
                subject = f"URGENT: {', '.join(actions)} Required" if actions else "Portfolio Alert"
            
            success = notifier.send_critical_alert(
                subject,
                critical_items,
                portfolio_value,
                portfolio_metrics,
                email_options={
                    "within_review_cooldown": within_cd,
                    "any_critical_sell": any_crit,
                },
            )
            
            if success:
                print(f"\n✅ Critical alert email sent with {len(critical_items)} urgent action(s)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}", exc_info=True)
            print(f"\n❌ Error sending alerts: {e}")
            print("   Make sure EMAIL_SENDER and EMAIL_PASSWORD are set in environment variables")
            return False
    
    def run(self) -> Dict:
        """Main function to run critical alert check."""
        results = self.check_critical_opportunities()
        
        critical_items = results.get("critical_items", [])
        specific_action_types = ["BUY", "SELL", "REPLACE", "REBALANCE", "ACTION"]
        specific_actions = [item for item in critical_items if item.get("type") in specific_action_types]
        trends_only = [item for item in critical_items if item.get("type") == "EMERGING_TREND"]
        
        if results.get("has_critical") and specific_actions:
            print(f"\n⚠️  Found {len(specific_actions)} critical action(s) requiring immediate attention:")
            for item in specific_actions:
                print(f"\n   [{item.get('priority', 'MEDIUM')}] {item.get('type', 'ACTION')}: {item.get('ticker', item.get('title', 'N/A'))}")
                print(f"      Reason: {item.get('reason', 'N/A')}")
            
            # Also show trends if any (as informational)
            if trends_only:
                print(f"\n📊 Also detected {len(trends_only)} hot trend(s) (included in email):")
                for item in trends_only:
                    print(f"   🔥 {item.get('category', 'N/A')}: {item.get('momentum', 0):.1f}% momentum")
            
            # Send email alerts
            email_sent = self.send_alerts(results)

            # Recommendations are NOT applied automatically: portfolio.json must
            # reflect trades actually executed at the broker, not suggestions.
            # (The old auto_apply-after-email recorded phantom trades daily in CI.)
            # Opt back in explicitly with ALERTS_AUTO_APPLY=1 if you really want it.
            auto_apply_env = os.environ.get("ALERTS_AUTO_APPLY", "").strip().lower() in ("1", "true", "yes")
            if email_sent and specific_actions and auto_apply_env:
                print("\n" + "="*60)
                print("AUTO-UPDATING PORTFOLIO FROM EMAIL RECOMMENDATIONS (ALERTS_AUTO_APPLY=1)")
                print("="*60)
                self.apply_recommendations_to_portfolio(specific_actions, auto_apply=True)
            elif email_sent and specific_actions:
                print("\nℹ️  Portfolio NOT modified. After you execute trades at your broker,")
                print("   run 'make analyze' and confirm them there.")
        elif trends_only:
            print(f"\n📊 Detected {len(trends_only)} hot trend(s), but no specific actionable recommendations:")
            for item in trends_only:
                print(f"   🔥 {item.get('category', 'N/A')}: {item.get('momentum', 0):.1f}% momentum - ETFs: {', '.join(item.get('etfs', []))}")
            print("\n   ℹ️  No email sent - trends are informational only when:")
            print("      • No cash available for purchases (<$100)")
            print("      • No weak holdings to replace with trend ETFs")
            print("      • Portfolio is balanced and performing well")
        else:
            print("\n✅ No critical actions required. Portfolio is in good shape.")
            print("   (No email sent - only critical alerts trigger emails)")
        
        return results

if __name__ == "__main__":
    alert_system = CriticalAlertSystem()
    results = alert_system.run()

    # Finding alerts is a SUCCESSFUL run - exit 0 either way. (The old
    # exit-1-on-alerts forced continue-on-error in CI, which also masked real
    # crashes; now a non-zero exit always means the pipeline itself broke.)
    sys.exit(0)

