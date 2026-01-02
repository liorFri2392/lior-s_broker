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

### ייעוץ חכם
- **המלצות איזון מחדש**: זיהוי אוטומטי + המלצות מה למכור ומה לקנות במקום
- **ייעוץ הפקדות**: ניתוח מאות ETF מכל המגזרים, המלצות מבוססות על טרנדים
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
- מספקת המלצות מפורטות על אילו ETF לקנות
- מציינת האם לקנות חדשים או להגדיל החזקות קיימות

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

