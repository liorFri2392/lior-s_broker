#!/usr/bin/env python3
"""
Update GitHub Secret - Automatically update PORTFOLIO_JSON secret
This keeps the GitHub Actions workflow in sync with your local portfolio

Two methods:
1. Using GitHub CLI (gh) - easiest if installed
2. Manual copy-paste - always works
"""

import json
import sys
from pathlib import Path

import github_secret

def update_github_secret():
    """Update PORTFOLIO_JSON secret in GitHub repository."""

    # Read portfolio.json
    portfolio_file = Path("portfolio.json")
    if not portfolio_file.exists():
        print("❌ Error: portfolio.json not found!")
        print("   Run 'make analyze' first to create/update portfolio.json")
        return False

    with open(portfolio_file, 'r', encoding='utf-8') as f:
        portfolio_data = json.load(f)

    portfolio_json_str = json.dumps(portfolio_data, ensure_ascii=False, indent=2)

    # Try the automatic paths (GitHub CLI, then REST API) via the shared helper.
    if github_secret.update_portfolio_secret(portfolio_data, verbose=True):
        print("✅ Successfully updated PORTFOLIO_JSON secret in GitHub!")
        return True
    print("⚠️  Automatic update unavailable, showing manual method...")

    # Manual method - show content to copy
    print("\n" + "="*60)
    print("UPDATE GITHUB SECRET - Manual Method")
    print("="*60)
    print("\n📋 Steps to update PORTFOLIO_JSON secret:")
    print("1. Go to: https://github.com/liorFri2392/lior-s_broker/settings/secrets/actions")
    print("2. Click on PORTFOLIO_JSON secret")
    print("3. Click 'Update'")
    print("4. Copy the content below and paste it:")
    print("\n" + "-"*60)
    print(portfolio_json_str)
    print("-"*60 + "\n")
    
    # Also save to file for easy access
    output_file = "portfolio_for_github_secret.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(portfolio_json_str)
    print(f"✅ Portfolio data also saved to: {output_file}")
    print(f"   You can open this file and copy its contents\n")
    
    print("💡 Tip: Install GitHub CLI for automatic updates:")
    print("   brew install gh  # macOS")
    print("   Then run: gh auth login")
    print("   Then: make update-secret (will work automatically)\n")
    
    return True

if __name__ == "__main__":
    success = update_github_secret()
    sys.exit(0 if success else 1)
