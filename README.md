# ליאור'ס ברוקר - הברוקר האולטימטיבי 🚀

מערכת ניתוח תיק השקעות מתקדמת ומתוחכמת - הברוקר האולטימטיבי עם ניתוח סטטיסטי, טכני, ומודלים מתמטיים.

## תכונות מתקדמות - הברוקר האולטימטיבי

### ניתוח מתקדם
- **ניתוח תיק מעמיק**: אינדיקטורים טכניים (RSI, Momentum, Sharpe, Beta), ניתוח תרשימי נרות, מגמות
- **מודלים סטטיסטיים**: רגרסיה ליניארית ופולינומית, תחזיות 3-5 שנים, אופטימיזציה לתשואה בטווח הבינוני
- **ניתוח טרנדים**: זיהוי תעשיות "חמות", השוואה בין מגזרים, ניתוח מומנטום

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
- **חישוב מס רווחי הון**: 25% שטוח (אין הנחת החזקה ארוכה בישראל)
- **ניתוח דיבידנדים**: ניכוי מס אמריקאי 25% (אמנה) + זיכוי מס זר בישראל

### פילוסופיה: אסטרטגיה אחת, בלי החלפות 🎯
המערכת **לא** ממליצה להחליף קרנות על בסיס ציונים/מומנטום — החלפות כאלה
מייצרות מס ועמלות, לא תשואה. מכירה קורית משתי סיבות בלבד:
1. **איזון מחדש ליעד** (תוכנית Option A/B בדוח)
2. **אירוע סיכון אמיתי** (stop-loss, ריכוז קיצוני מעל 50%)
מגמות משפיעות רק על ההטיה התחומה בתוך סל הלוויין (15%) בהפקדות.

### ייעוץ חכם
- **תוכנית איזון מחדש**: Option A (הפקדות בלבד, ללא מס, עם הערכת חודשים)
  או Option B (מכירה/קנייה עכשיו עם הערכת מס לכל מכירה)
- **ייעוץ הפקדות**: הקצאת gap-fill מתכנסת ליעד 85/15 (ליבה רחבה + סל לוויין עם trend-tilt)

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
- מסווגת כל החזקה לקבוצת-שקילות (SPY=VOO=IVV, BND=AGG, TIP=SCHP=VTIP וכו')
- משווה את המשקל הנוכחי של כל קבוצה מול היעד (85/15 — 70 ליבה / 15 לוויין / 15 אג"ח)
- מקצה כל דולר לפער הגדול ביותר מתחת ליעד — תמיד מגדילה החזקה קיימת בקבוצה לפני קניית טיקר חדש
- עודף שלא מספיק למניה שלמה נשאר כמזומן; בהפקדה הבאה תישאל אם לכלול אותו
  (ברירת מחדל: מקצים רק את סכום ההפקדה; `DEPOSIT_SWEEP_CASH=1` לכלול תמיד)

**Bounded trend tilt (ברירת מחדל):** מומנטום 3 חודשים מטה את *סל הלוויין* לעבר
סקטורים חמים (למשל TECH מ-8% ל-11.6%) ובוחר איזה טיקר להיכנס לקבוצת-לוויין
ריקה — אך תמיד בתוך תקרות: לעולם לא נוגע בליבה/אג"ח, לא מגדיל את תקציב הלוויין
מעבר ל-30%, ולא מוסיף כפילות לסקטור שכבר מוחזק. לכיבוי: `DEPOSIT_TREND_TILT=0`.

(המנגנון הישן מבוסס-ציונים זמין עם `DEPOSIT_ADVISOR_LEGACY=1`)

### מעקב תשואה אמיתית 📒

כל הפקדה וכל עסקה נרשמות ב-`transactions` בתוך `portfolio.json`.
הדוח של `make analyze` מציג:
- **TRUE INVESTMENT PERFORMANCE** — רווח נטו בניכוי הכסף שהופקד + תשואה שנתית משוקללת-כסף (XIRR)
- שורת "צמיחת ערך" הישנה מסומנת במפורש ככוללת את ההפקדות שלך (היא איננה תשואה)

### Backtesting - בדיקת אסטרטגיות

```bash
make backtest
```

`make backtest` מריץ עכשיו את האסטרטגיה **האמיתית** של האפליקציה: הפקדות
חודשיות שמוקצות ע"י מנוע ה-gap-fill, מול בנצ'מרק DCA של 100% SPY.
התוצאות מוצגות בניכוי הפקדות (XIRR), לא כתשואת סכום-חד-פעמי.

או ישירות ב-Python:

```python
from backtesting import Backtester

backtester = Backtester(initial_capital=1000)
results = backtester.backtest_strategy(
    tickers=["SPY", "VEA", "VWO", "IWM", "VTV", "XLK", "XLV", "XLE", "PAVE", "BND", "VTIP"],
    start_date="2019-01-01",
    end_date="2024-12-31",
    strategy="monthly_deposit",
    monthly_deposit=2000,
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

### Machine Learning - תחזיות LSTM (אופציונלי)

דורש התקנת tensorflow: `pip install -r requirements-ml.txt` (המודול אינו
בשימוש בהמלצות — נשמר לניסויים בלבד).

```python
from ml_predictor import MLPredictor

predictor = MLPredictor()
result = predictor.predict_with_lstm("SPY", periods=30)
print(f"Predicted return: {result['expected_return']:.2f}%")
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

