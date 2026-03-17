#!/usr/bin/env python3
"""
Analyze Reminder - Check if 30 days passed since last make analyze, send email reminder.
Used by GitHub Actions; reads portfolio.json (created from PORTFOLIO_JSON secret).
"""

import json
import os
import sys
from datetime import datetime

REMINDER_DAYS = 30
PORTFOLIO_FILE = os.environ.get("PORTFOLIO_FILE", "portfolio.json")


def main():
    if not os.path.exists(PORTFOLIO_FILE):
        print(f"No {PORTFOLIO_FILE} found, skipping reminder.")
        return 0

    with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
        portfolio = json.load(f)

    last_run = portfolio.get("last_analyze_run_date")
    if not last_run:
        # Never recorded – treat as old and send reminder
        send_reminder(
            "Run make analyze – no run date recorded",
            "Your portfolio has no last_analyze_run_date. Run: make analyze"
        )
        return 0

    try:
        last_d = datetime.strptime(last_run, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid last_analyze_run_date: {last_run}")
        return 0

    days_ago = (datetime.now().date() - last_d).days
    if days_ago < REMINDER_DAYS:
        print(f"Last analyze was {days_ago} days ago (< {REMINDER_DAYS}). No reminder sent.")
        return 0

    send_reminder(
        f"📊 Reminder: Run make analyze ({days_ago} days since last run)",
        f"It's been {days_ago} days since you last ran portfolio analysis.\n\n"
        f"Run locally: make analyze\n\n"
        f"Then update the GitHub secret: make update-secret"
    )
    return 0


def send_reminder(subject: str, body: str) -> None:
    if not os.environ.get("EMAIL_SENDER") or not os.environ.get("EMAIL_PASSWORD"):
        print("EMAIL_SENDER/EMAIL_PASSWORD not set; skipping email.")
        return
    try:
        from email_notifier import EmailNotifier
        notifier = EmailNotifier()
        if notifier.send_simple_reminder(subject, body):
            print("✅ Analyze reminder email sent.")
        else:
            print("⚠️ Reminder not sent (email failed).")
    except Exception as e:
        print(f"⚠️ Could not send reminder: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
