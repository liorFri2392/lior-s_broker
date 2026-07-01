.PHONY: analyze deposit refresh-prices setup install venv clean backtest risk tax alerts backfill-cost-basis test

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
	@echo "  make analyze    - Analyze portfolio; asks to update portfolio.json if you executed the recommended trades"
	@echo "  make analyze-preview - Same as analyze but never updates portfolio (read-only)"
	@echo "  make deposit    - Refresh prices + live USD/ILS rate, then deposit recommendations"
	@echo "  make refresh-prices - Update portfolio.json prices from market (no trade prompts)"
	@echo "  make alerts     - Check for critical portfolio actions (requires .env with email credentials)"
	@echo "  make backtest    - Run backtesting on historical data"
	@echo "  make risk       - Check portfolio risk and get risk management report"
	@echo "  make tax        - Analyze tax implications for portfolio"
	@echo "  make update-secret - Update GitHub secret with current portfolio (for workflows)"
	@echo "  make clean      - Remove virtual environment"

venv:
	@echo "Creating virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists."; \
	fi

# Stamp file: deps are reinstalled only when requirements.txt changes.
DEPS_STAMP = $(VENV)/.deps-installed

$(DEPS_STAMP): requirements.txt | venv
	@echo "Installing required packages..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@touch $(DEPS_STAMP)
	@echo "Packages installed successfully."

install: $(DEPS_STAMP)

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

analyze: install
	@echo "Running portfolio analysis..."
	@$(PYTHON) portfolio_analyzer.py

analyze-preview: install
	@echo "Running portfolio analysis (read-only - portfolio.json will not be changed)..."
	@ANALYZE_READONLY=1 $(PYTHON) portfolio_analyzer.py

refresh-prices: install
	@echo "Refreshing portfolio prices..."
	@$(PYTHON) -c "from portfolio_analyzer import PortfolioAnalyzer; PortfolioAnalyzer().refresh_portfolio_prices(verbose=True)"

deposit: install
	@echo "Deposit Advisory System"
	@read -p "Enter deposit amount in ILS (₪): " amount; \
	$(PYTHON) deposit_advisor.py $$amount

alerts: install
	@echo "Checking for critical portfolio actions..."
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "   Please create .env file with EMAIL_SENDER and EMAIL_PASSWORD"; \
		echo "   You can copy .env.example to .env and fill in your values"; \
		exit 1; \
	fi
	@$(PYTHON) critical_alert.py

backtest: install
	@echo "Running backtesting analysis..."
	@$(PYTHON) backtesting.py

risk: install
	@echo "Checking portfolio risk..."
	@$(PYTHON) -c "from risk_manager import RiskManager; rm = RiskManager(); rm.print_risk_report()"

backfill-cost-basis: install
	@echo "Setting cost_basis = last_price for holdings without a tracked cost basis."
	@echo "This is a one-time migration so stop-loss / take-profit alerts work."
	@$(PYTHON) risk_manager.py backfill-cost-basis

tax: install
	@echo "Analyzing tax implications for current portfolio..."
	@$(PYTHON) tax_report.py

test: install
	@echo "Running offline test suite..."
	@$(PYTHON) -m pytest -q

update-secret: install
	@echo "Updating GitHub secret with current portfolio..."
	@$(PYTHON) update_github_secret.py

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Virtual environment removed."

