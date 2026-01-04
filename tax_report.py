#!/usr/bin/env python3
"""
Tax Report Script - Generate comprehensive tax report for portfolio
"""

from tax_analyzer import TaxAnalyzer
import json
import sys
from datetime import datetime

def main():
    analyzer = TaxAnalyzer()
    portfolio = analyzer.load_portfolio()
    
    print("\n" + "=" * 60)
    print("PORTFOLIO TAX ANALYSIS")
    print("=" * 60 + "\n")
    
    holdings = portfolio.get("holdings", [])
    if not holdings:
        print("❌ No holdings found in portfolio.")
        print("   Add holdings to portfolio.json first.\n")
        return
    
    print(f"📊 Portfolio Overview:")
    print(f"   Total Holdings: {len(holdings)}")
    print(f"   Portfolio Value: ${portfolio.get('total_value', 0):,.2f}")
    print(f"   Cash: ${portfolio.get('cash', 0):,.2f}\n")
    
    # Analyze each holding for potential tax
    print("💰 Tax Analysis by Holding:\n")
    
    total_potential_tax = 0
    long_term_holdings = []
    
    for holding in holdings:
        ticker = holding.get("ticker")
        quantity = holding.get("quantity", 0)
        current_price = holding.get("last_price", 0)
        purchase_price = holding.get("purchase_price") or current_price
        purchase_date = holding.get("purchase_date") or portfolio.get("last_updated", datetime.now().isoformat())
        
        if purchase_price == 0 or current_price == 0:
            continue
        
        # Calculate potential tax if sold now
        tax_calc = analyzer.calculate_capital_gains_tax(
            purchase_price=purchase_price,
            sale_price=current_price,
            quantity=quantity,
            purchase_date=purchase_date
        )
        
        gain_ils = tax_calc.get("total_gain_ils", 0)
        tax_ils = tax_calc.get("capital_gains_tax_ils", 0)
        
        if gain_ils > 0:
            total_potential_tax += tax_ils
        
        # Check if long-term
        holding_period = tax_calc.get("holding_period_days", 0)
        if holding_period > 730:
            long_term_holdings.append(ticker)
        
        print(f"   {ticker}:")
        print(f"      Quantity: {quantity} shares")
        print(f"      Purchase Price: ${purchase_price:.2f}")
        print(f"      Current Price: ${current_price:.2f}")
        print(f"      Gain/Loss: ${tax_calc.get('total_gain_usd', 0):,.2f} ({gain_ils:,.2f} ILS)")
        
        if gain_ils > 0:
            print(f"      Potential Tax: {tax_ils:,.2f} ILS ({tax_calc.get('effective_tax_rate', 0):.1f}%)")
            if tax_calc.get("is_long_term"):
                print(f"      ✅ Long-term holding - tax reduction applied")
        elif gain_ils < 0:
            print(f"      💡 Loss - can offset gains")
        
        print()
    
    # Summary
    print("=" * 60)
    print("TAX SUMMARY")
    print("=" * 60 + "\n")
    
    if total_potential_tax > 0:
        print(f"💰 Total Potential Tax (if all sold now): {total_potential_tax:,.2f} ILS")
    else:
        print("💰 No capital gains tax if all holdings sold now")
    
    if long_term_holdings:
        print(f"\n✅ Long-term Holdings (>2 years) - Eligible for tax reduction:")
        for ticker in long_term_holdings:
            print(f"   • {ticker}")
    
    print("\n💡 Tax Optimization Tips:")
    print("   • Hold positions >2 years for tax reduction")
    print("   • Offset gains with losses")
    print("   • Consider selling losers before year-end")
    print("   • Consult tax advisor for accurate calculations")
    
    print("\n" + "=" * 60)
    print("⚠️  NOTE: Calculations are estimates. Consult a tax advisor.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

