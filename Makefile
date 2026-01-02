.PHONY: analyze deposit setup install

# Default target
all: help

help:
	@echo "Available commands:"
	@echo "  make analyze    - Analyze current portfolio and provide recommendations"
	@echo "  make deposit    - Get recommendations for depositing funds"
	@echo "  make install    - Install required Python packages"
	@echo "  make setup      - Initial setup (install packages and create portfolio.json)"

install:
	@echo "Installing required packages..."
	pip3 install -r requirements.txt

setup: install
	@echo "Setting up initial portfolio structure..."
	@if [ ! -f portfolio.json ]; then \
		echo '{"currency": "USD", "note": "All prices and values are in USD. Cash and portfolio values shown in ILS in the app are converted from USD.", "cash": 46.7, "holdings": [{"ticker": "XLV", "quantity": 6, "last_price": 154.8, "current_value": 928.8}, {"ticker": "SPY", "quantity": 9, "last_price": 681.92, "current_value": 6137.28}], "last_updated": null, "total_value": 0}' > portfolio.json; \
		echo "Created portfolio.json with initial holdings"; \
	else \
		echo "portfolio.json already exists"; \
	fi

analyze:
	@echo "Running portfolio analysis..."
	python3 portfolio_analyzer.py

deposit:
	@echo "Deposit Advisory System"
	@read -p "Enter deposit amount in ILS (₪): " amount; \
	python3 deposit_advisor.py $$amount

