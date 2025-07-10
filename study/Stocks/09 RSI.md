The **Relative Strength Index (RSI)** is a **momentum oscillator** that measures the speed and change of price movements. It oscillates between 0 and 100 and is most commonly calculated over a 14-period window. Traders use it to identify overbought (typically RSI > 70) or oversold (RSI < 30) conditions.

---

## 1. How RSI Is Calculated

1. **Compute period-to-period gains and losses**
   For each bar $i$:

   $$
     \Delta_i = \text{Close}_i - \text{Close}_{i-1}
   $$

   $$
     \text{Gain}_i = \max(\Delta_i,0),\quad \text{Loss}_i = \max(-\Delta_i,0)
   $$

2. **Average the gains and losses** (Wilders’ smoothing)

   * First average: simple mean over the first $n$ periods (usually $n=14$).
   * Thereafter:

     $$
       \text{AvgGain}_i = \frac{(\text{AvgGain}_{i-1}\times (n-1)) + \text{Gain}_i}{n}
     $$

     $$
       \text{AvgLoss}_i = \frac{(\text{AvgLoss}_{i-1}\times (n-1)) + \text{Loss}_i}{n}
     $$

3. **Compute the Relative Strength (RS)**

   $$
     \text{RS}_i = \frac{\text{AvgGain}_i}{\text{AvgLoss}_i}
   $$

4. **Convert to RSI**

   $$
     \text{RSI}_i = 100 - \frac{100}{1 + \text{RS}_i}
   $$

---

## 2. Simple Python Example (Pandas)

```python
import pandas as pd

def compute_RSI(series: pd.Series, period: int = 14) -> pd.Series:
    # 1) Calculate daily changes
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # 2) First averages (simple)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # 3) Wilder’s smoothing
    # After the first 'period' values, use the recursive formula
    for i in range(period, len(series)):
        avg_gain.iat[i] = (avg_gain.iat[i-1] * (period - 1) + gain.iat[i]) / period
        avg_loss.iat[i] = (avg_loss.iat[i-1] * (period - 1) + loss.iat[i]) / period

    # 4) Compute RS and RSI
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Usage:
# df is your OHLC DataFrame with a 'Close' column
# df['RSI_14'] = compute_RSI(df['Close'], period=14)
```

---

## 3. Interpreting RSI

* **RSI > 70** : Often signals the market is **overbought** → potential pullback or reversal.
* **RSI < 30** : Often signals the market is **oversold** → potential bounce or reversal.
* **Centerline (50)** :

  * Above 50 → bullish momentum
  * Below 50 → bearish momentum

RSI can also be used for **divergences** (price makes a new high but RSI does not, hinting at weakening momentum) and **support/resistance** within the indicator itself.

---

### Why It Works

By smoothing gains and losses, RSI filters out noise and focuses on **net momentum** over your chosen look-back. It’s a leading indicator (it can turn before price) but should be combined with price action, volume, or other indicators for confirmation.
