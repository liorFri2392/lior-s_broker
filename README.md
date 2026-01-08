# ליאור'ס ברוקר - הברוקר האולטימטיבי 🚀

מערכת ניתוח תיק השקעות מתקדמת ומתוחכמת - הברוקר האולטימטיבי עם ניתוח סטטיסטי, טכני, ומודלים מתמטיים.

## תכונות מתקדמות - הברוקר האולטימטיבי

### ניתוח מתקדם
- **ניתוח תיק מעמיק**: אינדיקטורים טכניים (RSI, Momentum, Sharpe, Beta), ניתוח תרשימי נרות, מגמות
- **מודלים סטטיסטיים**: רגרסיה ליניארית ופולינומית, תחזיות 3-5 שנים, אופטימיזציה לתשואה בטווח הבינוני
- **ניתוח טרנדים**: זיהוי תעשיות "חמות", השוואה בין מגזרים, ניתוח מומנטום
- **ניתוח חדשות וסנטימנט**: ניתוח חדשות בזמן אמת, סנטימנט חיובי/שלילי

### ניתוח אג"ח ותשואות
- **ניתוח אג"ח מתקדם**: תשואות, סיכונים, תשואה מתואמת סיכון
- **אופטימיזציה לתשואה**: התמקדות בתשואה הגבוהה ביותר בטווח הבינוני (3-5 שנים)
- **ניתוח היסטורי**: מודלים מבוססי היסטוריה, ניתוח מצבים נוכחיים

### ניתוח טכני מתקדם
- **תרשימי נרות**: זיהוי דפוסים (Hammer, Engulfing, וכו')
- **אינדיקטורים**: RSI, Moving Averages, Volatility, Momentum, Sharpe Ratio, Beta, Max Drawdown
- **ניתוח מגמות**: BULLISH/BEARISH/NEUTRAL, זיהוי נקודות כניסה/יציאה

### Machine Learning מתקדם 🚀
- **LSTM Neural Networks**: תחזיות מחירים מדויקות עם deep learning
- **תחזיות לטווח ארוך**: מודלים מתקדמים לתחזית 30-90 יום קדימה
- **אופטימיזציה מבוססת ML**: שימוש ב-machine learning לאופטימיזציה של תיק

### Backtesting 📊
- **סימולציות היסטוריות**: בדיקת אסטרטגיות על נתונים היסטוריים
- **אסטרטגיות מרובות**: Buy & Hold, Rebalancing, Momentum
- **מדדי ביצועים**: Sharpe Ratio, Max Drawdown, Win Rate, Annualized Returns

### ניהול סיכונים אוטומטי 🛡️
- **Stop-Loss אוטומטי**: התראות על הפסדים מעל סף מסוים
- **Take-Profit אוטומטי**: המלצות למכירה כשמגיעים ליעד רווח
- **ניטור ריכוז**: התראות על ריכוז יתר בתיק
- **מדדי סיכון**: ניתוח תנודתיות, קורלציות, וסיכון כולל

### ניתוח מיסים 💰
- **חישוב מס רווחי הון**: חישוב מדויק של מס על מכירות
- **ניתוח דיבידנדים**: חישוב מס על דיבידנדים (מס אמריקאי + ישראלי)
- **אופטימיזציה מס**: המלצות להפחתת נטל מס
- **ניתוח תקופות החזקה**: זיהוי החזקות ארוכות טווח להטבות מס

### ניתוח Sentiment משופר 📰
- **ניתוח חדשות מתקדם**: זיהוי sentiment משופר עם מילות מפתח פיננסיות
- **ניתוח מרובה מקורות**: אגרגציה של sentiment מכמה מקורות
- **ניתוח תיק כולל**: sentiment ברמת תיק ולא רק מניה בודדת

### ייעוץ חכם
- **המלצות איזון מחדש**: זיהוי אוטומטי + המלצות מה למכור ומה לקנות במקום
- **ייעוץ הפקדות**: ניתוח מאות ETF מכל המגזרים, המלצות מבוססות על 80/20 Balanced Growth
- **ניתוח מגוון**: זיהוי פערים, המלצות על פיזור אופטימלי

### תכונות נוספות
- **מחירים בזמן אמת**: זיהוי אם השוק פתוח, מחירים בזמן אמת או סגירה אחרונה
- **Caching חכם**: אופטימיזציה של ביצועים, פחות קריאות API
- **Parallel Processing**: ניתוח מקבילי למהירות מקסימלית

## התקנה

```bash
make setup
```

או באופן ידני:

```bash
pip install -r requirements.txt
```

## הגדרה ראשונית - חשוב! ⚠️

**בפעם הראשונה, אתה חייב להגדיר את `portfolio.json` עם התיק שלך!**

### איך להגדיר את portfolio.json:

1. **צור את הקובץ `portfolio.json`** (או השתמש ב-`make setup` שיוצר אותו אוטומטית)

2. **עדכן את התיק שלך** - הוסף את כל ההחזקות שלך:

```json
{
  "currency": "USD",
  "note": "All prices and values are in USD. Cash and portfolio values shown in ILS in the app are converted from USD.",
  "cash": 46.7,
  "holdings": [
    {
      "ticker": "XLV",
      "quantity": 6,
      "last_price": 154.8,
      "current_value": 928.8
    },
    {
      "ticker": "SPY",
      "quantity": 9,
      "last_price": 681.92,
      "current_value": 6137.28
    }
  ],
  "last_updated": null,
  "total_value": 0
}
```

3. **פורמט:**
   - `cash`: מזומן בדולרים (USD)
   - `holdings`: רשימת החזקות
     - `ticker`: סמל המניה/ETF (לדוגמה: "SPY", "XLV")
     - `quantity`: כמות מניות
     - `last_price`: מחיר אחרון (יועדכן אוטומטית)
     - `current_value`: שווי נוכחי (יועדכן אוטומטית)

4. **הערות:**
   - כל המחירים והערכים ב-USD
   - המערכת תעדכן אוטומטית את המחירים והערכים כשתהריץ `make analyze`
   - אם אין לך החזקות עדיין, השאר `holdings` כרשימה ריקה: `[]`

## שימוש

### ניתוח תיק נוכחי

```bash
make analyze
```

פקודה זו:
- קוראת את התיק הנוכחי מקובץ `portfolio.json`
- מעדכנת מחירים נוכחיים
- מבצעת ניתוח מעמיק של כל החזקה
- מספקת המלצות על איזון מחדש
- מעדכנת את קובץ `portfolio.json` עם נתונים עדכניים

### ייעוץ הפקדה

```bash
make deposit
```

פקודה זו:
- מבקשת ממך להכניס סכום הפקדה בשקלים
- מנתחת את התיק הנוכחי
- בודקת מאות ETF פוטנציאליים
- מספקת המלצות מפורטות על אילו ETF לקנות (80/20 Balanced Growth Strategy)
- מציינת האם לקנות חדשים או להגדיל החזקות קיימות

### Backtesting - בדיקת אסטרטגיות

```bash
make backtest
```

או ישירות ב-Python:

```python
from backtesting import Backtester

backtester = Backtester(initial_capital=10000)
results = backtester.backtest_strategy(
    tickers=["SPY", "VXUS", "BND"],
    start_date="2020-01-01",
    end_date="2023-12-31",
    strategy="buy_and_hold",
    allocation={"SPY": 0.5, "VWO": 0.3, "BND": 0.2}
)
backtester.print_results(results)
```

### ניהול סיכונים

```bash
make risk
```

או ישירות ב-Python:

```python
from risk_manager import RiskManager

manager = RiskManager()
alerts = manager.get_risk_alerts()
manager.print_risk_report()
```

### ניתוח מיסים

```bash
make tax
```

או ישירות ב-Python:

```python
from tax_analyzer import TaxAnalyzer

analyzer = TaxAnalyzer()
tax_calc = analyzer.calculate_capital_gains_tax(
    purchase_price=100,
    sale_price=150,
    quantity=10,
    purchase_date="2022-01-01T00:00:00"
)
analyzer.print_tax_report({"sales_analysis": [tax_calc]})
```

### Machine Learning - תחזיות LSTM

```python
from ml_predictor import MLPredictor

predictor = MLPredictor()
result = predictor.predict_with_lstm("SPY", periods=30)
print(f"Predicted return: {result['expected_return']:.2f}%")
```

### ניתוח Sentiment משופר

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_news_sentiment("SPY")
print(f"Sentiment: {result['sentiment']} (Score: {result['score']})")
```

## מבנה קבצים

- `portfolio_analyzer.py` - מודול ניתוח התיק הראשי
- `deposit_advisor.py` - מודול ייעוץ הפקדות
- `portfolio.json` - קובץ JSON המכיל את התיק הנוכחי
- `Makefile` - פקודות make
- `requirements.txt` - תלויות Python

## פורמט portfolio.json

```json
{
  "cash": 172.74,
  "holdings": [
    {
      "ticker": "XLV",
      "quantity": 6,
      "last_price": 154.8,
      "current_value": 928.8
    }
  ],
  "last_updated": "2024-01-01T12:00:00",
  "total_value": 7250.08
}
```

## משתני סביבה (אופציונלי)

לשימוש מלא בתכונות (חדשות, API נוספות), הוסף קובץ `.env`:

```
NEWS_API_KEY=your_news_api_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
```

## הערות

- המערכת משתמשת ב-yfinance לקבלת נתוני שוק
- ניתוח מבוסס על אינדיקטורים טכניים, סנטימנט, ומגמות
- המלצות הן להתייחסות בלבד ואינן מהוות ייעוץ השקעות מקצועי

