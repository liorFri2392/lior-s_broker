#!/usr/bin/env python3
"""
Tax Analyzer - Calculate tax implications for Israeli investors in US markets
Handles capital gains, dividends, and tax optimization
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TaxAnalyzer:
    """Analyze tax implications for portfolio transactions."""
    
    # Israeli tax rates (2024)
    CAPITAL_GAINS_TAX = 0.25  # 25% on capital gains
    DIVIDEND_TAX = 0.25  # 25% on dividends
    US_WITHHOLDING_TAX = 0.15  # 15% US withholding (with W-8BEN form)
    
    # Tax exemptions
    ANNUAL_EXEMPTION_ILS = 0  # No exemption for capital gains (as of 2024)
    LONG_TERM_REDUCTION = 0.20  # 20% reduction for long-term (>2 years) - check current law
    
    def __init__(self, portfolio_file: str = "portfolio.json"):
        self.portfolio_file = portfolio_file
        self.exchange_rate_usd_ils = 3.7  # Will be updated from market
    
    def load_portfolio(self) -> Dict:
        """Load portfolio from JSON file."""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"holdings": [], "cash": 0}
    
    def calculate_capital_gains_tax(
        self,
        purchase_price: float,
        sale_price: float,
        quantity: int,
        purchase_date: str,
        sale_date: str = None,
        exchange_rate: float = None
    ) -> Dict:
        """
        Calculate capital gains tax for a sale.
        
        Args:
            purchase_price: Price per share when purchased (USD)
            sale_price: Price per share when sold (USD)
            quantity: Number of shares
            purchase_date: Purchase date (ISO format)
            sale_date: Sale date (default: today)
            exchange_rate: USD/ILS exchange rate
        
        Returns:
            Dict with tax calculations
        """
        if sale_date is None:
            sale_date = datetime.now().isoformat()
        
        if exchange_rate is None:
            exchange_rate = self.exchange_rate_usd_ils
        
        # Calculate gain/loss
        gain_per_share = sale_price - purchase_price
        total_gain_usd = gain_per_share * quantity
        total_gain_ils = total_gain_usd * exchange_rate
        
        # Check if long-term (>2 years)
        purchase_dt = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
        sale_dt = datetime.fromisoformat(sale_date.replace('Z', '+00:00'))
        holding_period = (sale_dt - purchase_dt).days
        
        is_long_term = holding_period > 730  # 2 years
        
        # Calculate tax
        if total_gain_ils > 0:
            # Capital gains tax
            if is_long_term:
                # Long-term reduction (check current Israeli law)
                taxable_gain = total_gain_ils * (1 - self.LONG_TERM_REDUCTION)
            else:
                taxable_gain = total_gain_ils
            
            # Apply exemption if applicable
            taxable_gain = max(0, taxable_gain - self.ANNUAL_EXEMPTION_ILS)
            
            capital_gains_tax = taxable_gain * self.CAPITAL_GAINS_TAX
        else:
            # Loss - can offset gains
            capital_gains_tax = 0
            taxable_gain = 0
        
        # Net after tax
        net_proceeds_usd = (sale_price * quantity) - (capital_gains_tax / exchange_rate)
        net_proceeds_ils = sale_price * quantity * exchange_rate - capital_gains_tax
        
        return {
            "purchase_price_usd": purchase_price,
            "sale_price_usd": sale_price,
            "quantity": quantity,
            "total_gain_usd": total_gain_usd,
            "total_gain_ils": total_gain_ils,
            "holding_period_days": holding_period,
            "is_long_term": is_long_term,
            "taxable_gain_ils": taxable_gain,
            "capital_gains_tax_ils": capital_gains_tax,
            "net_proceeds_usd": net_proceeds_usd,
            "net_proceeds_ils": net_proceeds_ils,
            "effective_tax_rate": (capital_gains_tax / total_gain_ils * 100) if total_gain_ils > 0 else 0
        }
    
    def calculate_dividend_tax(
        self,
        dividend_amount_usd: float,
        exchange_rate: float = None
    ) -> Dict:
        """
        Calculate tax on dividends.
        
        Args:
            dividend_amount_usd: Dividend amount in USD
            exchange_rate: USD/ILS exchange rate
        
        Returns:
            Dict with tax calculations
        """
        if exchange_rate is None:
            exchange_rate = self.exchange_rate_usd_ils
        
        dividend_ils = dividend_amount_usd * exchange_rate
        
        # US withholding tax (if W-8BEN form filed)
        us_withholding_usd = dividend_amount_usd * self.US_WITHHOLDING_TAX
        us_withholding_ils = us_withholding_usd * exchange_rate
        
        # Remaining dividend after US tax
        remaining_dividend_usd = dividend_amount_usd - us_withholding_usd
        remaining_dividend_ils = remaining_dividend_ils = remaining_dividend_usd * exchange_rate
        
        # Israeli tax on remaining dividend
        israeli_tax_ils = remaining_dividend_ils * self.DIVIDEND_TAX
        
        # Total tax
        total_tax_ils = us_withholding_ils + israeli_tax_ils
        
        # Net dividend
        net_dividend_usd = dividend_amount_usd - (total_tax_ils / exchange_rate)
        net_dividend_ils = dividend_ils - total_tax_ils
        
        return {
            "dividend_amount_usd": dividend_amount_usd,
            "dividend_amount_ils": dividend_ils,
            "us_withholding_tax_ils": us_withholding_ils,
            "israeli_tax_ils": israeli_tax_ils,
            "total_tax_ils": total_tax_ils,
            "net_dividend_usd": net_dividend_usd,
            "net_dividend_ils": net_dividend_ils,
            "effective_tax_rate": (total_tax_ils / dividend_ils * 100) if dividend_ils > 0 else 0
        }
    
    def analyze_portfolio_tax_impact(
        self,
        portfolio: Dict = None,
        proposed_sales: List[Dict] = None
    ) -> Dict:
        """
        Analyze tax impact of proposed portfolio changes.
        
        Args:
            portfolio: Current portfolio
            proposed_sales: List of proposed sales with {ticker, quantity, sale_price}
        
        Returns:
            Dict with tax analysis
        """
        if portfolio is None:
            portfolio = self.load_portfolio()
        
        if proposed_sales is None:
            proposed_sales = []
        
        holdings = portfolio.get("holdings", [])
        tax_analysis = {
            "total_capital_gains_tax": 0,
            "total_losses": 0,
            "sales_analysis": [],
            "recommendations": []
        }
        
        for sale in proposed_sales:
            ticker = sale.get("ticker")
            quantity = sale.get("quantity", 0)
            sale_price = sale.get("sale_price", 0)
            
            # Find holding
            holding = next((h for h in holdings if h.get("ticker") == ticker), None)
            if not holding:
                continue
            
            purchase_price = holding.get("purchase_price") or holding.get("last_price", 0)
            purchase_date = holding.get("purchase_date") or portfolio.get("last_updated", datetime.now().isoformat())
            
            # Calculate tax
            tax_calc = self.calculate_capital_gains_tax(
                purchase_price=purchase_price,
                sale_price=sale_price,
                quantity=quantity,
                purchase_date=purchase_date
            )
            
            tax_analysis["sales_analysis"].append({
                "ticker": ticker,
                **tax_calc
            })
            
            if tax_calc["total_gain_ils"] > 0:
                tax_analysis["total_capital_gains_tax"] += tax_calc["capital_gains_tax_ils"]
            else:
                tax_analysis["total_losses"] += abs(tax_calc["total_gain_ils"])
        
        # Tax optimization recommendations
        if tax_analysis["total_losses"] > 0:
            tax_analysis["recommendations"].append(
                f"Consider offsetting gains with losses: {tax_analysis['total_losses']:,.2f} ILS available"
            )
        
        # Check for long-term holdings
        long_term_holdings = []
        for holding in holdings:
            purchase_date = holding.get("purchase_date") or portfolio.get("last_updated")
            if purchase_date:
                purchase_dt = datetime.fromisoformat(purchase_date.replace('Z', '+00:00'))
                days_held = (datetime.now() - purchase_dt).days
                if days_held > 730:
                    long_term_holdings.append(holding.get("ticker"))
        
        if long_term_holdings:
            tax_analysis["recommendations"].append(
                f"Long-term holdings (>2 years) eligible for tax reduction: {', '.join(long_term_holdings)}"
            )
        
        return tax_analysis
    
    def print_tax_report(self, tax_analysis: Dict):
        """Print tax analysis report."""
        print("\n" + "=" * 60)
        print("TAX ANALYSIS REPORT")
        print("=" * 60 + "\n")
        
        if tax_analysis["sales_analysis"]:
            print("💰 Capital Gains Tax Analysis:")
            for sale in tax_analysis["sales_analysis"]:
                print(f"\n   {sale['ticker']}:")
                print(f"      Gain: ${sale['total_gain_usd']:,.2f} ({sale['total_gain_ils']:,.2f} ILS)")
                print(f"      Tax: {sale['capital_gains_tax_ils']:,.2f} ILS ({sale['effective_tax_rate']:.1f}%)")
                print(f"      Net Proceeds: {sale['net_proceeds_ils']:,.2f} ILS")
                if sale['is_long_term']:
                    print(f"      ✅ Long-term holding - tax reduction applied")
            
            print(f"\n   Total Capital Gains Tax: {tax_analysis['total_capital_gains_tax']:,.2f} ILS")
            if tax_analysis['total_losses'] > 0:
                print(f"   Available Losses for Offset: {tax_analysis['total_losses']:,.2f} ILS")
        
        if tax_analysis["recommendations"]:
            print("\n💡 Tax Optimization Recommendations:")
            for rec in tax_analysis["recommendations"]:
                print(f"   • {rec}")
        
        print("\n" + "=" * 60 + "\n")
        print("⚠️  NOTE: Tax calculations are estimates. Consult a tax advisor for accurate calculations.")
        print("   Israeli tax laws may change. Verify current rates and exemptions.\n")

if __name__ == "__main__":
    analyzer = TaxAnalyzer()
    
    # Example
    tax_calc = analyzer.calculate_capital_gains_tax(
        purchase_price=100,
        sale_price=150,
        quantity=10,
        purchase_date="2022-01-01T00:00:00"
    )
    
    print(f"Tax calculation: {tax_calc}")

