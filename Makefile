.PHONY: analyze deposit setup install venv clean

# Virtual environment directory
VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3

# Default target
all: help

help:
	@echo "Available commands:"
	@echo "  make venv       - Create virtual environment"
	@echo "  make install    - Install required Python packages"
	@echo "  make setup      - Initial setup (create venv, install packages, create portfolio.json)"
	@echo "  make analyze    - Analyze current portfolio and provide recommendations"
	@echo "  make deposit    - Get recommendations for depositing funds"
	@echo "  make clean      - Remove virtual environment"

venv:
	@echo "Creating virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists."; \
	fi

install: venv
	@echo "Installing required packages..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Packages installed successfully."

setup: install
	@echo "Setting up initial portfolio structure..."
	@if [ ! -f portfolio.json ]; then \
		echo '{"currency": "USD", "note": "All prices and values are in USD. Cash and portfolio values shown in ILS in the app are converted from USD.", "cash": 46.7, "holdings": [{"ticker": "XLV", "quantity": 6, "last_price": 154.8, "current_value": 928.8}, {"ticker": "SPY", "quantity": 9, "last_price": 681.92, "current_value": 6137.28}], "last_updated": null, "total_value": 0}' > portfolio.json; \
		echo "Created portfolio.json with initial holdings"; \
	else \
		echo "portfolio.json already exists"; \
	fi
	@echo ""
	@echo "✅ Setup complete! You can now use 'make analyze' or 'make deposit'"

analyze: venv
	@echo "Running portfolio analysis..."
	@$(PYTHON) portfolio_analyzer.py

deposit: venv
	@echo "Deposit Advisory System"
	@read -p "Enter deposit amount in ILS (₪): " amount; \
	$(PYTHON) deposit_advisor.py $$amount

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Virtual environment removed."

