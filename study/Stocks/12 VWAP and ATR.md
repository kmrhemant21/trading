**VWAP (Volume-Weighted Average Price)**
VWAP is the average price at which a security has traded throughout the day, weighted by volume. It’s calculated intraday (and resets each trading day) as:

$$
\text{VWAP}_t = \frac{\sum_{i=1}^{t}(P_i \times V_i)}{\sum_{i=1}^{t}V_i}
$$

* $P_i$ = typical price of trade $i$ (often the traded price)
* $V_i$ = volume of trade $i$
* The summations run from market open up to time $t$.

**Why use VWAP?**

* **Benchmarking:** Institutions compare their execution price to VWAP to gauge “good” fill prices.
* **Trend indicator:** On intraday charts, price above VWAP suggests bullish bias; below VWAP suggests bearish bias.
* **Dynamic support/resistance:** Traders often watch VWAP as an intraday magnet for price.

---

**ATR (Average True Range) Volatility**
ATR is a measure of how much an asset typically moves over a given period, capturing both intraday range and gaps. First defined by Welles Wilder:

1. **True Range (TR):** For each period $i$:

   $$
   \text{TR}_i = \max\bigl(H_i - L_i,\;|H_i - C_{i-1}|,\;|L_i - C_{i-1}|\bigr)
   $$

   * $H_i$, $L_i$ = high and low of the current period
   * $C_{i-1}$ = close of the previous period

2. **ATR over $n$ periods** (commonly $n=14$):

   $$
   \text{ATR}_n = \frac{1}{n}\sum_{i=1}^{n}\text{TR}_i
   $$

   (Or via Wilder’s smoothing: $\text{ATR}_t = \frac{(\text{ATR}_{t-1}\times(n-1) + \text{TR}_t)}{n}$.)

**Why use ATR?**

* **Volatility gauge:** Higher ATR means larger average moves; low ATR means consolidation.
* **Stop-loss placement:** Traders often set stops at a multiple of ATR (e.g., 1× or 1.5× ATR) to adapt to current volatility.
* **Position sizing:** Risk can be scaled by ATR so that each trade risks a similar dollar amount despite changing volatility.

---

### Quick Comparison

| Indicator | What It Measures | Calculation Basis             | Primary Use                      |
| --------- | ---------------- | ----------------------------- | -------------------------------- |
| **VWAP**  | Average price    | Cumulative price × volume     | Execution benchmark, trend guide |
| **ATR**   | Volatility       | Moving average of true ranges | Stops, position sizing           |

By combining VWAP (price bias) with ATR (volatility), you can time entries when price direction and volatility regime align with your risk parameters.
