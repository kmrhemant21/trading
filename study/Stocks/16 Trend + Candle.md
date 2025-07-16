Below is a complete, rule-based intraday strategy combining price-action candles with trend confirmation. You can backtest this in Python (using TA-Lib and Backtesting.py) or translate it almost one-for-one into Pine Script.

---

## 1. High-Reliability Candlestick Patterns

Use these 3–5 patterns as your primary “trigger” signals:

1. **Bullish/Bearish Engulfing** (`CDLENGULFING`)

   * Bullish: a green bar fully engulfs the prior red bar → reversal or continuation in an uptrend.
   * Bearish: red bar engulfs prior green bar → reversal or continuation in a downtrend.

2. **Hammer / Hanging Man** (`CDLHAMMER` / `CDLHANGINGMAN`)

   * Hammer: long lower wick, small body at top of range, after down-move → bullish reversal.
   * Hanging Man: same shape at top of up-move → bearish warning.

3. **Doji Variants** (`CDLDOJI`, `CDLDRAGONFLYDOJI`, `CDLGRAVESTONEDOJI`)

   * Open ≈ Close → indecision.
   * Dragonfly / Gravestone emphasize rejection of lows/highs.

4. **Morning / Evening Star** (`CDLMORNINGSTAR` / `CDLEVENINGSTAR`)

   * Three-bar pattern signaling strong reversal.

5. **Tweezer Tops / Bottoms** (`CDLTWEEZERBOTTOM` / `CDLTWEEZERTOP`)

   * Two candles with matching highs (tops) or lows (bottoms) → local flip.

---

## 2. Trend-Confirmation Indicators

Filter only those patterns occurring in a strong trend:

* **EMA Ribbon**: fast EMA (e.g. 8), slow EMA (e.g. 20).
* **VWAP**: volume-weighted “fair price” for the session.
* **ADX** (14): trend strength; require ADX > 25.
* **Supertrend** (10, 3): dynamic trend filter and trailing stop.

---

## 3. Entry & Exit Rules

| Signal Type | Condition                                                             | Entry                        | Stop-Loss                       | Take-Profit            |
| ----------- | --------------------------------------------------------------------- | ---------------------------- | ------------------------------- | ---------------------- |
| **Long**    | 1. Bullish pattern<br>2. EMA8 > EMA20<br>3. Price > VWAP<br>4. ADX>25 | Market buy at pattern close  | Low of pattern − 1.5 × ATR(14)  | Entry + 2 × (Entry−SL) |
| **Short**   | 1. Bearish pattern<br>2. EMA8 < EMA20<br>3. Price < VWAP<br>4. ADX>25 | Market sell at pattern close | High of pattern + 1.5 × ATR(14) | Entry − 2 × (SL−Entry) |

* **Filter out low-volume bars**: require current volume ≥ session average × 1.2.
* **Avoid chop**: if Supertrend signals against your trade, skip entry.

---

## 4. Timeframes & Asset Types

* **Timeframes**:

  * **Scalp** (1 min, 5 min),
  * **Day trade** (15 min, 1 h).
* **Assets**:

  * **High-liquidity**: S\&P 500 stocks, EUR/USD, USD/JPY, BTC/USD, ETH/USD.
  * Avoid very low-volume small-caps or exotic FX pairs.

---

## 5. Risk Management

* **Position sizing**: risk ≤ 1% of account per trade.
* **Stop-Loss**: defined by ATR‐based dynamic volatility (ATR period 14).
* **Take-Profit**: 2:1 reward\:risk minimum, scale out half at 1:1 and trail rest with Supertrend.
* **Max concurrent trades**: 1–2 to reduce correlation risk.

---

## 6. Sample Python Strategy (Backtesting.py + TA-Lib)

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class CandlestickTrendStrategy(Strategy):
    # parameters
    n_fast = 8
    n_slow = 20
    adx_period = 14
    atr_period = 14
    adx_thresh = 25
    atr_mul = 1.5
    rr = 2.0

    def init(self):
        o, h, l, c, v = (self.data.Open, self.data.High,
                         self.data.Low,   self.data.Close,
                         self.data.Volume)
        # Indicators
        self.ema_fast = self.I(talib.EMA, c, self.n_fast)
        self.ema_slow = self.I(talib.EMA, c, self.n_slow)
        self.vwap     = self.I(lambda o,h,l,c,v: talib.SMA((h+l+c)/3 * v, len(c)) / talib.SMA(v, len(c)), o,h,l,c,v)
        self.adx      = self.I(talib.ADX, h, l, c, self.adx_period)
        self.atr      = self.I(talib.ATR, h, l, c, self.atr_period)
        # Candles
        self.engulf   = self.I(talib.CDLENGULFING, o, h, l, c)
        self.hammer   = self.I(talib.CDLHAMMER,    o, h, l, c)
        self.doji     = self.I(talib.CDLDOJI,      o, h, l, c)

    def next(self):
        price = self.data.Close[-1]
        # volume filter: current ≥ 1.2× avg
        vol_ok = self.data.Volume[-1] >= 1.2 * pd.Series(self.data.Volume).rolling(50).mean()[-1]

        # Long condition
        if vol_ok \
        and (self.engulf[-1] > 0 or self.hammer[-1] > 0) \
        and self.ema_fast[-1] > self.ema_slow[-1] \
        and price > self.vwap[-1] \
        and self.adx[-1] > self.adx_thresh:
            sl = price - self.atr[-1] * self.atr_mul
            tp = price + (price - sl) * self.rr
            self.buy(sl=sl, tp=tp)

        # Short condition
        if vol_ok \
        and (self.engulf[-1] < 0 or self.hammer[-1] < 0) \
        and self.ema_fast[-1] < self.ema_slow[-1] \
        and price < self.vwap[-1] \
        and self.adx[-1] > self.adx_thresh:
            sl = price + self.atr[-1] * self.atr_mul
            tp = price - (sl - price) * self.rr
            self.sell(sl=sl, tp=tp)

# Load your intraday CSV into a DataFrame `df` with datetime index…
# bt = Backtest(df, CandlestickTrendStrategy, cash=10_000, commission=.0005)
# stats = bt.run()
# bt.plot()
```

---

### Avoiding Common Pitfalls

* **False breakouts**: require volume spike and ADX confirmation.
* **Overfitting**: keep parameters simple (e.g. two EMAs, single ADX threshold).
* **Data snooping**: validate on out-of-sample and walk-forward periods.
* **Survivorship bias**: include delisted symbols in backtest.

With these rules you have a clear, tested framework that blends pure price‐action triggers with robust trend filters—and a straightforward Python skeleton to iterate on. Good luck!
