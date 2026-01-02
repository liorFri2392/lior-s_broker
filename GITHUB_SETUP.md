# GitHub Actions Setup - Critical Alerts

## Overview
This repository includes a GitHub Actions workflow that runs daily (Monday-Friday) during market hours to check for critical portfolio actions and send email alerts.

## Setup Instructions

### 1. Configure GitHub Secrets

Go to your repository on GitHub:
1. Navigate to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add the following secrets:

#### Required Secrets:

- **`EMAIL_SENDER`**: Your Gmail address (e.g., `liorfried2334@gmail.com`)
- **`EMAIL_PASSWORD`**: Your Gmail App Password (not your regular password!)
  - To generate: Google Account → Security → 2-Step Verification → App passwords
  - Create app password for "Mail" and use the generated password
- **`EMAIL_RECIPIENT`** (optional): Recipient email address (defaults to EMAIL_SENDER if not set)
- **`PORTFOLIO_JSON`**: Your portfolio.json file content as a JSON string
  - Copy the entire content of your `portfolio.json` file
  - Paste it as the secret value

### 2. Gmail App Password Setup

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Enable **2-Step Verification** if not already enabled
3. Go to **App passwords**
4. Select **Mail** and **Other (Custom name)**
5. Enter "Portfolio Alerts" as the name
6. Click **Generate**
7. Copy the 16-character password (spaces are ignored)
8. Use this password as `EMAIL_PASSWORD` secret

### 3. Workflow Schedule

The workflow runs automatically:
- **Days**: Monday through Friday (weekdays only)
- **Times**: 
  - 10:00 AM EST (15:00 UTC)
  - 2:00 PM EST (19:00 UTC)

You can also trigger it manually:
- Go to **Actions** tab → **Daily Critical Portfolio Alerts** → **Run workflow**

### 4. Local Testing

To test the alerts locally:

```bash
# 1. Create .env file (copy from .env.example)
cp .env.example .env

# 2. Edit .env and add your credentials
# EMAIL_SENDER=your_email@gmail.com
# EMAIL_PASSWORD=your_app_password
# EMAIL_RECIPIENT=your_email@gmail.com

# 3. Run the alert check
make alerts
```

## What Triggers Alerts?

The system sends email alerts for:

1. **Urgent Sells**:
   - Holdings with recommendation score < 25
   - STRONG SELL recommendations
   - Risk of significant loss

2. **Critical Buys**:
   - ETFs with score ≥ 75
   - Expected 3-year return > 15%
   - Exceptional opportunities

3. **Over-concentration**:
   - Single holding > 40% of portfolio
   - Diversification needed

4. **Market Anomalies**:
   - SPY drop > 3% in one day
   - SPY surge > 3% in one day

## Email Format

Alerts include:
- **Priority level** (CRITICAL, HIGH, MEDIUM)
- **Action type** (BUY, SELL, ACTION)
- **Detailed reasons** and recommendations
- **Expected returns** and scores
- **Specific amounts** and share quantities

## Troubleshooting

### Workflow fails to send emails:
- Check that all secrets are set correctly
- Verify Gmail App Password is correct
- Check workflow logs in Actions tab

### No alerts received:
- Check spam folder
- Verify EMAIL_RECIPIENT is set correctly
- Check that portfolio.json secret is valid JSON

### Market day detection issues:
- Workflow checks if market is trading
- Skips weekends and holidays automatically
- Uses SPY as market indicator

## Security Notes

- **Never commit** `.env` file to repository (already in .gitignore)
- **Never commit** `portfolio.json` to repository (already in .gitignore)
- Use **App Passwords** for Gmail, not your regular password
- GitHub Secrets are encrypted and only accessible to workflows

