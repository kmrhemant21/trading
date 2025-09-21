Awesome—let’s wire your **30-day-low bounce + volume surge** idea into a **live, Groww-API trading script** (ready for NSE cash segment, delivery CNC). It does: login, fetch daily candles, scan signals, size positions with ATR risk, and (optionally) place entry + protective SL orders via Groww.

I’m giving you a single Python file you can drop into your repo. It uses Groww’s **official Python SDK** (`pip install growwapi`) and follows their latest docs for auth, historical candles, and order placement. ([Groww][1])

---

# `groww_strategy.py`

```python
"""
Groww API Strategy – 30D-Low Bounce with Volume Surge
-----------------------------------------------------
Entry (Daily timeframe):
  - Yesterday's Low == 30-day rolling minimum (i.e., yesterday printed the 30D low)
  - Today's Volume > 1.8 * Yesterday's Volume
  - Today's Low > Yesterday's Low
  - Today's Open > ₹20
  - Today's High  < 30-day rolling maximum (not a breakout)

Risk & Orders:
  - Position sizing by fixed risk % of capital using ATR(14)
  - Entry: MARKET CNC (delivery)
  - Protective Stop: SL-M below entry at ATR_MULT_SL * ATR
  - Optional Target: Limit at ATR_MULT_TP * ATR above entry (commented)

Requires: pip install growwapi pandas numpy pyotp
Docs used: Groww SDK intro/auth, historical candles, orders, annexures.
"""

from __future__ import annotations
import os, time, math, uuid, datetime as dt
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# --- Groww SDK ---
# Docs: https://groww.in/trade-api/docs/python-sdk  (pip install growwapi)
from growwapi import GrowwAPI   # 
import pyotp                   # for TOTP login (recommended by Groww)  

# ============ CONFIG ============
# Universe (NSE cash symbols, no .NS suffix here per Groww "trading_symbol")
TICKERS: List[str] = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "LT", "ITC", "ICICIBANK", "HINDUNILVR"
]

# Capital & risk
STARTING_CAPITAL_RUPEES: float = 3_00_000.0
RISK_PCT_PER_TRADE: float = 0.005         # 0.5% of capital at risk per position
ATR_LEN: int = 14
ATR_MULT_SL: float = 1.5
ATR_MULT_TP: float = 2.0

# Entry filter params
VOL_MULTIPLIER: float = 1.8
MIN_OPEN_PRICE: float = 20.0
LOOKBACK_DAYS: int = 30     # for 30D low/high checks
MIN_HISTORY_DAYS: int = 80  # pull some buffer for ATR etc.

# Order settings
EXCHANGE = "NSE"
SEGMENT  = "CASH"
PRODUCT  = "CNC"            # delivery
VALIDITY = "DAY"
ORDER_TYPE_ENTRY = "MARKET" # entry at market
ORDER_TYPE_SL    = "SL_M"   # protective stop as stop-loss market  
PLACE_TARGET     = False    # optionally place a target LIMIT order (commented below)

# Safety
MAX_NOTIONAL_PER_TRADE: float = 75_000.0  # cap notional per trade
ROUND_LOT: int = 1                         # equity lot (usually 1)

# Execution mode
DRY_RUN: bool = True  # True = no live orders, just prints the plan

# --- Auth (choose ONE flow) ---
# Flow A (recommended, no expiry): API key + TOTP from authenticator app
API_KEY   = os.getenv("GROWW_API_KEY",   "YOUR_API_KEY")
TOTP_SEED = os.getenv("GROWW_TOTP_SEED", "YOUR_TOTP_SECRET")  # base32

# Flow B (daily-approval flow): api_key + secret (uncomment if you prefer)
# API_SECRET = os.getenv("GROWW_API_SECRET", "YOUR_API_SECRET")  # daily approval needed

# =================================


def now_ist() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))


def groww_login() -> GrowwAPI:
    """
    Log in using Groww's recommended flows.
    Docs: Introduction -> Authentication (TOTP flow & API key/secret).  
    """
    # Flow A: API key + TOTP (no daily expiry)
    if API_KEY and TOTP_SEED and TOTP_SEED != "YOUR_TOTP_SECRET":
        totp = pyotp.TOTP(TOTP_SEED).now()
        access_token = GrowwAPI.get_access_token(api_key=API_KEY, totp=totp)  # 
        return GrowwAPI(access_token)

    # Flow B: API key + secret (expires daily at ~6AM)
    # access_token = GrowwAPI.get_access_token(api_key=API_KEY, secret=API_SECRET)  # 
    # return GrowwAPI(access_token)

    raise RuntimeError("Set GROWW_API_KEY and GROWW_TOTP_SEED env vars (or use secret flow).")


def fetch_daily_candles(g: GrowwAPI, symbol: str, days: int) -> pd.DataFrame:
    """
    Use Groww Historical Data API to fetch daily candles (1D interval).
    Returns DataFrame with columns: [open, high, low, close, volume] and tz-aware index.
    Docs: Historical Data -> get_historical_candle_data (interval_in_minutes=1440).  
    """
    end = now_ist()
    start = end - dt.timedelta(days=days + 5)  # small buffer

    # Groww expects "YYYY-MM-DD HH:mm:ss" in IST or epoch ms (docs show both).  
    start_s = start.strftime("%Y-%m-%d %H:%M:%S")
    end_s   = end.strftime("%Y-%m-%d %H:%M:%S")

    resp = g.get_historical_candle_data(
        trading_symbol=symbol,
        exchange=g.EXCHANGE_NSE,
        segment=g.SEGMENT_CASH,
        start_time=start_s,
        end_time=end_s,
        interval_in_minutes=1440
    )  # returns { "candles": [[epoch, o,h,l,c,v], ...], ... }  

    candles = resp.get("candles", [])
    if not candles:
        raise RuntimeError(f"No candles for {symbol}")

    arr = np.array(candles, dtype=float)
    # epoch seconds -> UTC; we’ll localize to IST for clarity
    ts = pd.to_datetime(arr[:, 0], unit="s", utc=True).tz_convert("Asia/Kolkata")
    df = pd.DataFrame({
        "Open":  arr[:, 1],
        "High":  arr[:, 2],
        "Low":   arr[:, 3],
        "Close": arr[:, 4],
        "Volume": arr[:, 5].astype("int64")
    }, index=ts)
    # Keep only trading days and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def ta_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def signal_today(df: pd.DataFrame) -> bool:
    """
    Implements the screen on the *last available daily bar* (assumes EOD).
    """
    if len(df) < max(LOOKBACK_DAYS + 2, ATR_LEN + 5):
        return False

    low_y = df["Low"].shift(1)
    vol_y = df["Volume"].shift(1)
    lowest30_y = df["Low"].rolling(LOOKBACK_DAYS).min().shift(1)
    highest30_t = df["High"].rolling(LOOKBACK_DAYS).max()

    cond = (
        (low_y.eq(lowest30_y)) &                      # yesterday made 30D low
        (df["Volume"] > vol_y * VOL_MULTIPLIER) &     # volume surge
        (df["Low"] > low_y) &                         # higher low today
        (df["Open"] > MIN_OPEN_PRICE) &               # price sanity
        (df["High"] < highest30_t)                    # not a 30D breakout
    )
    # We want the last row's truth value
    return bool(cond.iloc[-1])


def size_position(df: pd.DataFrame, capital: float) -> Tuple[int, float, float]:
    """
    Returns (qty, sl_price_delta, tp_price_delta) using ATR-based risk.
    """
    atr = ta_atr(df, ATR_LEN).iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        return 0, 0.0, 0.0

    px = float(df["Close"].iloc[-1])
    risk_rupees = capital * RISK_PCT_PER_TRADE
    # risk per share ~ ATR_MULT_SL * ATR
    risk_per_share = ATR_MULT_SL * atr
    qty = int(max(1, math.floor(risk_rupees / max(1e-6, risk_per_share))))
    # cap notional
    qty = min(qty, int(MAX_NOTIONAL_PER_TRADE // max(1.0, px)))
    # round
    qty = (qty // ROUND_LOT) * ROUND_LOT
    if qty <= 0:
        return 0, 0.0, 0.0
    return qty, ATR_MULT_SL * atr, ATR_MULT_TP * atr


def place_entry_and_sl(g: GrowwAPI, symbol: str, last_close: float, qty: int, sl_delta: float, tp_delta: float) -> Dict:
    """
    Places MARKET buy and a protective SL-M (stop-loss market).
    Groww order fields per SDK Orders doc.  
    """
    ref = f"STR-{uuid.uuid4().hex[:10].upper()}"
    print(f"Placing BUY MARKET for {symbol} x{qty} (ref={ref})")

    entry = g.place_order(
        trading_symbol=symbol,
        quantity=qty,
        validity=g.VALIDITY_DAY,
        exchange=g.EXCHANGE_NSE,
        segment=g.SEGMENT_CASH,
        product=g.PRODUCT_CNC,
        order_type=g.ORDER_TYPE_MARKET,
        transaction_type=g.TRANSACTION_TYPE_BUY,
        order_reference_id=ref
    )  # returns groww_order_id & status  

    # Stop-loss market (trigger below entry)
    sl_trigger = round(last_close - sl_delta, 2)
    print(f"Placing protective SL-M at {sl_trigger} for {symbol}")

    sl = g.place_order(
        trading_symbol=symbol,
        quantity=qty,
        validity=g.VALIDITY_DAY,
        exchange=g.EXCHANGE_NSE,
        segment=g.SEGMENT_CASH,
        product=g.PRODUCT_CNC,
        order_type=g.ORDER_TYPE_SL_M,  # Annexures -> Order Type SL_M  
        transaction_type=g.TRANSACTION_TYPE_SELL,
        trigger_price=sl_trigger,
        order_reference_id=ref + "-SL"
    )

    # Optional target LIMIT
    # tp_price = round(last_close + tp_delta, 2)
    # print(f"Placing target LIMIT at {tp_price} for {symbol}")
    # tp = g.place_order(
    #     trading_symbol=symbol,
    #     quantity=qty,
    #     validity=g.VALIDITY_DAY,
    #     exchange=g.EXCHANGE_NSE,
    #     segment=g.SEGMENT_CASH,
    #     product=g.PRODUCT_CNC,
    #     order_type=g.ORDER_TYPE_LIMIT,
    #     transaction_type=g.TRANSACTION_TYPE_SELL,
    #     price=tp_price,
    #     order_reference_id=ref + "-TP"
    # )

    return {"entry": entry, "sl": sl}  # , "tp": tp


def run_once():
    g = groww_login()
    print(f"[{now_ist()}] Logged in to Groww.")

    results = []
    for sym in TICKERS:
        try:
            df = fetch_daily_candles(g, sym, max(LOOKBACK_DAYS, MIN_HISTORY_DAYS))
            if not signal_today(df):
                results.append({"Symbol": sym, "Signal": False})
                continue

            qty, sl_d, tp_d = size_position(df, STARTING_CAPITAL_RUPEES)
            last_close = float(df["Close"].iloc[-1])

            if qty <= 0:
                results.append({"Symbol": sym, "Signal": True, "Reason": "Qty=0 by risk caps"})
                continue

            plan = {
                "Symbol": sym, "Signal": True,
                "LastClose": last_close, "Qty": qty,
                "SL_Trigger": round(last_close - sl_d, 2),
                "TP_Limit": round(last_close + tp_d, 2)
            }

            if DRY_RUN:
                print("[DRY RUN]", plan)
                results.append(plan)
            else:
                resp = place_entry_and_sl(g, sym, last_close, qty, sl_d, tp_d)
                results.append({**plan, "OrderResp": resp})

            # Respect rate-limits a bit (Orders: 15/s; Live/Non-trading: 10–20/s)  
            time.sleep(0.15)

        except Exception as e:
            results.append({"Symbol": sym, "Error": str(e)})

    out = pd.DataFrame(results)
    print("\n=== RUN SUMMARY ===")
    print(out.to_string(index=False))
    # Save CSV
    stamp = now_ist().strftime("%Y%m%d_%H%M%S")
    out.to_csv(f"groww_strategy_{stamp}.csv", index=False)


if __name__ == "__main__":
    run_once()
```

---

## How to use (quick start)

1. **Install & set keys**

```bash
pip install --upgrade growwapi pandas numpy pyotp
export GROWW_API_KEY="your_api_key"
export GROWW_TOTP_SEED="base32_totp_seed_from_groww"
python groww_strategy.py
```

Auth flows & rate limits are per Groww docs (TOTP recommended; order/live/non-trading limits documented). ([Groww][1])

2. **Dry-run first**
   The script defaults to `DRY_RUN=True` (prints the plan and writes a CSV). Flip to `False` to send live orders.

3. **What it places**

* **Entry:** BUY **MARKET** (CNC) on NSE cash.
* **Protective stop:** **SL-M** SELL with trigger `(close – 1.5×ATR14)`. (Order-type constants per Annexures.) ([Groww][2])
* (Optional) **Target LIMIT** at `(close + 2×ATR14)`—left commented; enable if you want OCO-style management.

4. **Data source**
   Historical candles use Groww’s **Historical Data API** (`get_historical_candle_data`, `interval_in_minutes=1440`), which returns `[epoch, o, h, l, c, v]`. Script localizes timestamps to **IST**. ([Groww][3])

5. **Editing the universe & risk**

* Edit `TICKERS` to your watchlist.
* Tune `RISK_PCT_PER_TRADE`, `ATR_MULT_SL`, `ATR_MULT_TP`, and `MAX_NOTIONAL_PER_TRADE`.

---

## Notes & extensions

* **Order management & status:** You can poll `get_order_status` / `get_order_list` to reconcile fills (IDs & schemas shown in the Orders page). ([Groww][4])
* **Live execution:** To act intraday (e.g., on 5-min candles), switch `interval_in_minutes` to 5 and adapt the signal. (Mind the historical max windows per interval.) ([Groww][3])
* **Feeds/WebSocket:** Subscribing to **GrowwFeed** can stream LTP and order updates for tighter control. ([Groww][5])
* **Segments/Order types:** Supported values for exchange/segment/order types/products are listed in **Annexures** (used above). ([Groww][2])

If you want this turned into a pair of Jupyter notebooks (scanner + backtester) or wired to your existing **Groww/GCP** infra with cron & logs, say the word and I’ll split it cleanly.

[1]: https://groww.in/trade-api/docs/python-sdk "Introduction - Groww API"
[2]: https://groww.in/trade-api/docs/python-sdk/annexures "Annexures - Groww API"
[3]: https://groww.in/trade-api/docs/python-sdk/historical-data "Historical Data - Groww API"
[4]: https://groww.in/trade-api/docs/python-sdk/orders "Orders - Groww API"
[5]: https://groww.in/trade-api/docs/python-sdk/feed?utm_source=chatgpt.com "Feed - Groww API"
