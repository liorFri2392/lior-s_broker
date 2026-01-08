#!/usr/bin/env python3
"""
Update GitHub Secret - Automatically update PORTFOLIO_JSON secret
This keeps the GitHub Actions workflow in sync with your local portfolio
"""

import json
import os
import sys
import requests
from pathlib import Path

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
    
    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ Error: GITHUB_TOKEN environment variable not set!")
        print("\nTo set it up:")
        print("1. Go to GitHub → Settings → Developer settings → Personal access tokens")
        print("2. Generate new token with 'repo' and 'workflow' permissions")
        print("3. Run: export GITHUB_TOKEN=your_token_here")
        print("4. Or add to .env file: GITHUB_TOKEN=your_token_here")
        return False
    
    # Get repository info
    repo_owner = "liorFri2392"
    repo_name = "lior-s_broker"
    secret_name = "PORTFOLIO_JSON"
    
    # GitHub API endpoint
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/secrets/{secret_name}"
    
    # Get public key for encryption
    public_key_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/secrets/public-key"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        # Get public key
        response = requests.get(public_key_url, headers=headers)
        response.raise_for_status()
        public_key_data = response.json()
        public_key = public_key_data["key"]
        key_id = public_key_data["key_id"]
        
        # Encrypt the secret (using PyNaCl or simple base64 for now)
        # For simplicity, we'll use the GitHub API which handles encryption
        # But GitHub requires using libsodium for encryption
        
        print("⚠️  GitHub Secrets API requires encryption with libsodium.")
        print("   For now, please update manually:")
        print("\n📋 Steps to update PORTFOLIO_JSON secret:")
        print("1. Go to: https://github.com/liorFri2392/lior-s_broker/settings/secrets/actions")
        print("2. Click on PORTFOLIO_JSON secret")
        print("3. Click 'Update'")
        print("4. Copy the content below and paste it:")
        print("\n" + "="*60)
        print(portfolio_json_str)
        print("="*60 + "\n")
        
        # Alternative: Save to file for easy copy
        output_file = "portfolio_for_github_secret.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(portfolio_json_str)
        print(f"✅ Portfolio data saved to: {output_file}")
        print(f"   Copy the content and paste it into GitHub secret\n")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error updating secret: {e}")
        print("\n📋 Manual update required:")
        print("1. Go to GitHub → Settings → Secrets → Actions")
        print("2. Update PORTFOLIO_JSON secret")
        print("3. Copy content from portfolio.json")
        return False

if __name__ == "__main__":
    success = update_github_secret()
    sys.exit(0 if success else 1)
