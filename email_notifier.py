#!/usr/bin/env python3
"""
Email Notifier Module - Sends critical alerts via Gmail SMTP
"""

import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict
from datetime import datetime

# Try to load .env file if available (optional - works with environment variables too)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    # dotenv not installed - will use environment variables directly
    pass

logger = logging.getLogger(__name__)

class EmailNotifier:
    """Send email notifications for critical portfolio actions."""
    
    def __init__(self):
        self.sender_email = os.getenv("EMAIL_SENDER", "")
        self.sender_password = os.getenv("EMAIL_PASSWORD", "")
        self.recipient_email = os.getenv("EMAIL_RECIPIENT", self.sender_email)  # Default to sender if not set
        
        if not self.sender_email or not self.sender_password:
            logger.error("EMAIL_SENDER and EMAIL_PASSWORD must be set in environment variables")
            raise ValueError("EMAIL_SENDER and EMAIL_PASSWORD must be set in .env file or environment variables")
    
    def send_critical_alert(self, subject: str, critical_items: List[Dict], portfolio_value: float = None) -> bool:
        """Send critical alert email with urgent actions."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"🚨 CRITICAL ALERT: {subject}"
            
            # Build email body
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .critical {{ color: #d32f2f; font-weight: bold; }}
                    .action {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                    .buy {{ background-color: #d4edda; padding: 10px; margin: 10px 0; border-left: 4px solid #28a745; }}
                    .sell {{ background-color: #f8d7da; padding: 10px; margin: 10px 0; border-left: 4px solid #dc3545; }}
                    .info {{ background-color: #d1ecf1; padding: 10px; margin: 10px 0; border-left: 4px solid #17a2b8; }}
                    h2 {{ color: #d32f2f; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1 class="critical">🚨 CRITICAL PORTFOLIO ALERT</h1>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            if portfolio_value:
                body += f'<p><strong>Portfolio Value:</strong> ${portfolio_value:,.2f}</p>'
            
            body += "<h2>⚠️ URGENT ACTIONS REQUIRED:</h2>"
            
            for item in critical_items:
                action_type = item.get("type", "ACTION")
                priority = item.get("priority", "HIGH")
                
                if action_type == "BUY":
                    is_leveraged = item.get('is_leveraged', False)
                    leverage_warning = ""
                    if is_leveraged:
                        leverage = abs(item.get('leverage_multiplier', 1.0))
                        leverage_warning = f"""
                        <div style="background-color: #ffebee; padding: 15px; margin: 10px 0; border-left: 5px solid #f44336;">
                            <h4 style="color: #c62828; margin-top: 0;">🚨 LEVERAGED ETF WARNING 🚨</h4>
                            <p style="color: #c62828; font-weight: bold;">This is a {leverage}x leveraged ETF - EXTREMELY HIGH RISK!</p>
                            <ul style="color: #c62828;">
                                <li>Losses can be {leverage}x the underlying index</li>
                                <li>Very high volatility - can lose 50%+ in days</li>
                                <li>Not suitable for beginners</li>
                                <li>Consider limiting to &lt;5% of portfolio</li>
                            </ul>
                        </div>
                        """
                    body += f"""
                    <div class="buy">
                        <h3>🟢 BUY NOW: {item.get('ticker', 'N/A')}</h3>
                        {leverage_warning}
                        <p><strong>Reason:</strong> {item.get('reason', 'N/A')}</p>
                        <p><strong>Priority:</strong> {priority}</p>
                        <p><strong>Recommended Amount:</strong> ${item.get('amount', 0):,.2f}</p>
                        <p><strong>Expected Return:</strong> {item.get('expected_return', 0):.1f}%</p>
                        <p><strong>Score:</strong> {item.get('score', 0)}/100</p>
                    </div>
                    """
                elif action_type == "SELL":
                    body += f"""
                    <div class="sell">
                        <h3>🔴 SELL NOW: {item.get('ticker', 'N/A')}</h3>
                        <p><strong>Reason:</strong> {item.get('reason', 'N/A')}</p>
                        <p><strong>Priority:</strong> {priority}</p>
                        <p><strong>Recommended Amount:</strong> ${item.get('amount', 0):,.2f}</p>
                        <p><strong>Shares to Sell:</strong> {item.get('shares', 0)}</p>
                        <p><strong>Current Score:</strong> {item.get('score', 0)}/100</p>
                    </div>
                    """
                elif action_type == "REPLACE":
                    # Check if it's an optimization (better alternative) or trend replacement
                    is_optimization = "OPTIMIZE" in item.get('reason', '') or item.get('strategy') is not None
                    
                    if is_optimization:
                        body += f"""
                        <div class="action" style="background-color: #e8f5e9; border-left: 4px solid #4caf50;">
                            <h3>🔄 OPTIMIZE: {item.get('sell_ticker', 'N/A')} → {item.get('buy_ticker', 'N/A')}</h3>
                            <p><strong>Reason:</strong> {item.get('reason', 'N/A')}</p>
                            <p><strong>Priority:</strong> {priority}</p>
                            <p><strong>Strategy:</strong> {item.get('strategy', '80/20 Balanced Growth')}</p>
                            <div style="margin: 10px 0;">
                                <p><strong>🔴 SELL:</strong> {item.get('sell_ticker', 'N/A')} ({item.get('sell_category', 'N/A')})</p>
                                <ul>
                                    <li>Score: {item.get('sell_score', 0):.1f}/100</li>
                                    <li>Amount: ${item.get('sell_amount', 0):,.2f}</li>
                                    <li>Shares: {item.get('sell_shares', 0)}</li>
                                </ul>
                            </div>
                            <div style="margin: 10px 0;">
                                <p><strong>🟢 BUY:</strong> {item.get('buy_ticker', 'N/A')} ({item.get('buy_category', 'N/A')})</p>
                                <ul>
                                    <li>Score: {item.get('buy_score', 0):.1f}/100</li>
                                    <li>Score Improvement: +{item.get('score_improvement', 0):.1f} points</li>
                                    <li>Price: ${item.get('buy_price', 0):,.2f}</li>
                                    <li>Shares to Buy: {item.get('buy_shares', 0)}</li>
                                    <li>Amount: ${item.get('buy_amount', 0):,.2f}</li>
                                    {f"<li><strong>Remaining Cash:</strong> ${item.get('remaining_cash', 0):,.2f}</li>" if item.get('remaining_cash', 0) > 5 else ""}
                                </ul>
                            </div>
                            <p><strong>Details:</strong> {item.get('details', 'Optimize portfolio allocation with better performing ETF')}</p>
                        </div>
                        """
                    else:
                        body += f"""
                        <div class="action" style="background-color: #fff3e0; border-left: 4px solid #ff9800;">
                            <h3>🔄 REPLACE: {item.get('sell_ticker', 'N/A')} → {item.get('buy_ticker', 'N/A')}</h3>
                            <p><strong>Reason:</strong> {item.get('reason', 'N/A')}</p>
                            <p><strong>Priority:</strong> {priority}</p>
                            <div style="margin: 10px 0;">
                                <p><strong>🔴 SELL:</strong> {item.get('sell_ticker', 'N/A')}</p>
                                <ul>
                                    <li>Score: {item.get('sell_score', 0):.1f}/100 (Weak performance)</li>
                                    <li>Amount: ${item.get('sell_amount', 0):,.2f}</li>
                                    <li>Shares: {item.get('sell_shares', 0)}</li>
                                </ul>
                            </div>
                            <div style="margin: 10px 0;">
                                <p><strong>🟢 BUY:</strong> {item.get('buy_ticker', 'N/A')} (🔥 {item.get('buy_category', 'Trend')})</p>
                                <ul>
                                    <li>Score: {item.get('buy_score', 0):.1f}/100 (Strong performance)</li>
                                    <li>Price: ${item.get('buy_price', 0):,.2f}</li>
                                    <li>Shares to Buy: {item.get('buy_shares', 0)}</li>
                                    <li>Amount: ${item.get('buy_amount', 0):,.2f}</li>
                                    {f"<li><strong>Remaining Cash:</strong> ${item.get('remaining_cash', 0):,.2f}</li>" if item.get('remaining_cash', 0) > 5 else ""}
                                    <li>Momentum: {item.get('momentum', 0):.1f}% (20 days)</li>
                                    <li>Return: {item.get('return', 0):.1f}% (6 months)</li>
                                </ul>
                            </div>
                            <p><strong>Details:</strong> {item.get('details', 'Replace weak holding with hot trend ETF')}</p>
                        </div>
                        """
                elif action_type == "EMERGING_TREND":
                    etfs_list = ", ".join(item.get('etfs', []))
                    body += f"""
                    <div class="info" style="background-color: #fff3e0; border-left: 4px solid #ff9800;">
                        <h3>🔥 EMERGING TREND: {item.get('category', 'N/A')}</h3>
                        <p><strong>Reason:</strong> {item.get('reason', 'N/A')}</p>
                        <p><strong>Priority:</strong> {priority}</p>
                        <p><strong>Momentum:</strong> {item.get('momentum', 0):.1f}% (20 days)</p>
                        <p><strong>Return:</strong> {item.get('return', 0):.1f}% (6 months)</p>
                        <p><strong>ETFs:</strong> {etfs_list}</p>
                        <p><em>Consider adding these ETFs to your portfolio if you have cash available.</em></p>
                    </div>
                    """
                else:
                    body += f"""
                    <div class="action">
                        <h3>⚠️ ACTION REQUIRED: {item.get('title', 'N/A')}</h3>
                        <p><strong>Reason:</strong> {item.get('reason', 'N/A')}</p>
                        <p><strong>Priority:</strong> {priority}</p>
                        <p>{item.get('details', '')}</p>
                    </div>
                    """
            
            body += """
                <hr>
                <p><em>This is an automated alert from your Portfolio Analyzer.</em></p>
                <p><em>Please review your portfolio and take appropriate action.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email via Gmail SMTP
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Critical alert email sent successfully to {self.recipient_email}")
            print(f"✅ Critical alert email sent successfully to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending critical alert email: {e}", exc_info=True)
            print(f"❌ Error sending email: {e}")
            return False
    
    def send_daily_summary(self, summary: Dict) -> bool:
        """Send daily summary email (non-critical)."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"📊 Daily Portfolio Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
            <html>
            <body>
                <h2>Daily Portfolio Summary</h2>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Portfolio Value:</strong> ${summary.get('total_value', 0):,.2f}</p>
                <p><strong>Status:</strong> {summary.get('status', 'No critical actions needed')}</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Daily summary email sent successfully to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending daily summary email: {e}", exc_info=True)
            print(f"❌ Error sending daily summary: {e}")
            return False

