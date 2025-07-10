**Intraday Data Defined**
Intraday stock data refers to the real-time (or near-real-time) price and volume information for a security, captured at frequent intervals throughout a single trading day (e.g. every 1 minute, 5 minutes, etc.), rather than just end-of-day summaries. It provides granular insights into price movements and trading activity within the session ([intrinio.com][1], [portaracqg.com][2]).

---

## 1. Manual Download via Yahoo Finance Website

1. **Navigate to the ticker page** on Yahoo Finance (e.g. [https://finance.yahoo.com/quote/AAPL](https://finance.yahoo.com/quote/AAPL)).
2. Click **“Historical Data”**.
3. Under **“Time Period”**, set the dates (for intraday, usually choose one day).
4. Under **“Frequency”**, select an intraday interval (e.g. “1 minute”, “5 minutes”).
5. Click **“Apply”**, then **“Download”** to get a CSV of the OHLCV bars for that day ([help.yahoo.com][3]).

---

## 2. Programmatic Retrieval with Python’s `yfinance`

You can use the open-source `yfinance` library to pull intraday bars directly into pandas:

```python
import yfinance as yf

# 1. Create a Ticker object
ticker = yf.Ticker("AAPL")

# 2. Download 1-day of 1-minute bars
intraday = ticker.history(period="1d", interval="1m")

print(intraday.head())
```

* **period**

  * `"1d"` for one trading day (you can also use `"5d"`, `"7d"`, etc.).
* **interval**

  * `"1m"`, `"2m"`, `"5m"`, `"15m"`, `"30m"`, `"60m"`, … up to `"1d"` ([algotrading101.com][4]).

**Note:**

* High-resolution intraday (interval < 1 d) is typically only available for the most recent \~60 days, and 1 minute bars only for the last \~7 days due to Yahoo Finance’s data limits ([algotrading101.com][4]).
* For longer historical intraday coverage, you may need to cache data regularly or use a paid data vendor.

---

### Summary

| Method                  | Pros                               | Cons                                             |
| ----------------------- | ---------------------------------- | ------------------------------------------------ |
| **Manual CSV Download** | Quick, no coding                   | One ticker/day at a time; manual steps           |
| **`yfinance` Library**  | Scriptable; integrates with pandas | Limited look-back window for high-freq intervals |

Pick the approach that best fits your workflow—manual for ad-hoc checks and `yfinance` for automated pipelines.

[1]: https://intrinio.com/blog/how-to-get-intraday-stock-data-your-helpful-guide?utm_source=chatgpt.com "Intraday Stock Data: Definition, Uses, Benefits & How to Access"
[2]: https://portaracqg.com/2023/11/16/what-is-intraday-data-and-why-is-it-important/?utm_source=chatgpt.com "What is Intraday Data & Why is it Important? - PortaraCQG"
[3]: https://help.yahoo.com/kb/SLN2311.html?utm_source=chatgpt.com "Download historical data in Yahoo Finance"
[4]: https://algotrading101.com/learn/yfinance-guide/?utm_source=chatgpt.com "yfinance Library - A Complete Guide - AlgoTrading101 Blog"
