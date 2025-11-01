# 🧠 DETAILED EXPLANATION — FUTURES AND OPTIONS (F&O)

---

## 🔹 1. What are Derivatives?

**Derivatives** are financial instruments whose value is *derived* from an underlying asset — like **stocks, indices, commodities, or currencies**.

So, if the underlying moves, the derivative value also changes.

In India, derivatives are traded on **NSE** and **BSE**, mostly under the **F&O Segment** (Futures & Options).

### 📈 Common underlying assets:

* Stock derivatives: RELIANCE, TCS, INFY, etc.
* Index derivatives: NIFTY 50, BANKNIFTY, FINNIFTY.
* Commodities: Gold, Crude Oil.
* Currencies: USD-INR, EUR-INR.

---

## ⚙️ 2. FUTURES CONTRACTS – In Depth

A **Futures Contract** is a *standardized agreement* between two parties to buy or sell a specific quantity of an asset at a fixed price on a future date.

Both parties are *obligated* to fulfill the contract on expiry.

### 🧮 Example:

* You buy **Reliance November Futures** at ₹2,520.
* Lot size = 250 shares.
* Contract Value = ₹2,520 × 250 = ₹6,30,000.

You **don’t pay ₹6.3 lakh**, but a **margin** of ~20%:
→ ₹1,26,000 margin lets you control ₹6.3L exposure.

---

### ⚖️ 2.1 Mark-to-Market (MTM)

Every day, the exchange settles the difference between today’s futures price and yesterday’s.

If the price moves in your favor → profit credited.
If not → loss debited.

Example:

* Buy at ₹2,520
* Day 1 close ₹2,540 → +₹20 × 250 = ₹5,000 credited
* Day 2 close ₹2,510 → −₹30 × 250 = ₹7,500 debited

This daily MTM ensures no counterparty risk.

---

### 📆 2.2 Expiry

* Stock futures usually have **1-month expiry**.
* Expire on **last Thursday** of each month.
* You can trade **current month (near)**, **next month (mid)**, and **far month** contracts.

---

### 💡 2.3 Advantages of Futures

* High leverage → big exposure with small capital.
* Transparent and standardized.
* Can be used for **hedging** (locking prices).

### ⚠️ 2.4 Risks

* Losses can exceed margin (unlimited downside).
* Daily MTM margin calls.
* Short-term instruments (expire monthly).

---

## ⚙️ 3. OPTIONS CONTRACTS – In Depth

An **Option** gives the holder the *right, but not obligation*, to buy or sell an asset at a fixed **strike price** before or on expiry.

Two types:

* **Call Option (CE)** → Right to Buy.
* **Put Option (PE)** → Right to Sell.

The option **buyer** pays a **premium** to the **seller (writer)**.

---

### 🧩 3.1 Option Terminology

| Term                       | Meaning                                            |
| -------------------------- | -------------------------------------------------- |
| **Strike Price**           | Fixed price at which the asset can be bought/sold. |
| **Premium**                | Price paid by option buyer.                        |
| **Expiry Date**            | Date when option contract ends.                    |
| **Lot Size**               | Number of underlying shares in one contract.       |
| **In The Money (ITM)**     | Option has intrinsic value.                        |
| **At The Money (ATM)**     | Spot = Strike.                                     |
| **Out of The Money (OTM)** | No intrinsic value yet.                            |

---

### 💰 3.2 Example – Call Option

**Reliance = ₹2,500**, Buy **2500 Call Option** (Nov) @ ₹50.

Lot = 250 shares.

| Scenario | Spot   | Intrinsic Value | Profit/Loss                       |
| -------- | ------ | --------------- | --------------------------------- |
| Goes up  | ₹2,600 | ₹100            | (100 − 50) × 250 = ₹12,500 profit |
| Falls    | ₹2,450 | ₹0              | −50 × 250 = ₹12,500 loss          |

✅ **Max Loss = Premium (₹12,500)**
✅ **Max Profit = Unlimited**

---

### 💰 3.3 Example – Put Option

**Reliance = ₹2,500**, Buy **2500 Put Option** (Nov) @ ₹40.

| Scenario | Spot   | Intrinsic Value | Profit/Loss                       |
| -------- | ------ | --------------- | --------------------------------- |
| Falls    | ₹2,400 | ₹100            | (100 − 40) × 250 = ₹15,000 profit |
| Rises    | ₹2,550 | ₹0              | −40 × 250 = ₹10,000 loss          |

✅ **Put buyers profit when prices fall.**

---

### ⚖️ 3.4 Option Writer (Seller)

If you **sell a call**, you collect the premium upfront but face unlimited loss if price rises.

| Scenario               | Action          | Result                              |
| ---------------------- | --------------- | ----------------------------------- |
| You sell 2500 CE @ ₹50 | Receive ₹12,500 | If stock > ₹2,550, you start losing |

Hence, writers need **high margin** and **hedging**.

---

## 📊 4. FUTURES vs OPTIONS – Key Differences

| Feature    | Futures              | Options                              |
| ---------- | -------------------- | ------------------------------------ |
| Rights     | Obligation           | Right, not obligation                |
| Margin     | Required             | Only for sellers                     |
| Premium    | No                   | Buyer pays                           |
| Risk       | Unlimited            | Limited (buyer)                      |
| Reward     | Unlimited            | Unlimited (Call), High (Put)         |
| Leverage   | High                 | Moderate                             |
| Settlement | Daily MTM            | On expiry                            |
| Use Case   | Speculation, hedging | Hedging, speculation, income writing |

---

## ⚙️ 5. Option Pricing (How Premium is Decided)

The option **premium** has two parts:

1. **Intrinsic Value** – Profit if exercised now.
   → For Call: Max(Spot − Strike, 0)
   → For Put: Max(Strike − Spot, 0)

2. **Time Value** – Value of time till expiry (chance of profit).

As expiry nears, **Time Value decays** (called **Theta Decay**).

---

## ⚗️ 6. Option Greeks – Sensitivity Measures

| Greek     | Measures                                        | Interpretation                                     |
| --------- | ----------------------------------------------- | -------------------------------------------------- |
| **Delta** | Rate of change of option price with stock price | ΔCall = +ve (0→1), ΔPut = −ve (0→−1)               |
| **Gamma** | Rate of change of Delta                         | High Gamma = large change in Delta                 |
| **Theta** | Time decay                                      | Negative for buyers (option loses value with time) |
| **Vega**  | Sensitivity to volatility                       | High Vega = more affected by IV changes            |
| **Rho**   | Sensitivity to interest rates                   | Minor effect in equities                           |

🧠 Example:
If NIFTY Call has **Delta = 0.6**, and NIFTY rises by 100 points → Option rises by ≈60 points.

---

## 💼 7. How Margins Work in F&O

### 7.1 Futures Margin

* **SPAN Margin:** Covers worst-case movement.
* **Exposure Margin:** Additional safety buffer.
* Total margin ~15–25% of contract value.

Example:
Reliance Futures (₹6.3L value) → margin ₹1.2L → leverage ≈ 5×.

### 7.2 Option Margin

* **Buyer:** Pays premium only.
* **Seller:** Needs full margin (like futures) due to unlimited risk.

---

## 🧩 8. Option Strategies – Combining Calls and Puts

| Strategy             | View                   | Structure                            | Payoff               |
| -------------------- | ---------------------- | ------------------------------------ | -------------------- |
| **Covered Call**     | Mildly Bullish         | Hold stock + Sell Call               | Earn premium income  |
| **Protective Put**   | Hedge downside         | Hold stock + Buy Put                 | Limited downside     |
| **Bull Call Spread** | Moderately Bullish     | Buy lower strike call, sell higher   | Limited profit/loss  |
| **Bear Put Spread**  | Moderately Bearish     | Buy higher strike put, sell lower    | Limited profit/loss  |
| **Straddle**         | Expect high volatility | Buy same strike Call + Put           | Profit if large move |
| **Iron Condor**      | Expect low volatility  | Sell OTM call & put, buy farther OTM | Earn stable income   |

---

## 📈 9. Payoff Examples (Text Diagrams)

### Call Buyer

```
          /
         /
--------/
Loss ->│
       │
       └───→ Stock Price
```

(Loss limited to premium, profit unlimited)

### Call Seller

```
\         
 \        
  \-------
Loss unlimited │
               └───→ Stock Price
```

### Put Buyer

```
   /
  /
 /--------
Loss ->│
       │
       └───→ Stock Price
```

---

## ⚠️ 10. Common Mistakes New Traders Make

1. Trading options without understanding time decay.
2. Holding OTM options till expiry → 100% premium loss.
3. Not using stop-loss or hedging positions.
4. Over-leveraging futures.
5. Ignoring volatility and event risk.

---

## 🧠 11. Why Professionals Use F&O

| Purpose         | Use                                                        |
| --------------- | ---------------------------------------------------------- |
| **Hedging**     | Protect portfolio (e.g., Buy Put on NIFTY to hedge stocks) |
| **Arbitrage**   | Exploit price differences (spot vs futures)                |
| **Speculation** | Take directional bets with leverage                        |
| **Income**      | Sell covered options for monthly income                    |

---

## 📅 12. Example: NIFTY Futures

* NIFTY Spot = 22,500
* NIFTY Futures = 22,550
* Margin = ₹1.5L
* If NIFTY rises to 22,800 → profit ₹250 × 50 = ₹12,500
* If falls to 22,300 → loss ₹250 × 50 = ₹12,500

Thus, 1% move in NIFTY = ~8–10% move on margin capital.

---

## 🧾 13. Option Chain Analysis

On NSE, the **Option Chain** shows for each strike:

* LTP (Last traded price)
* OI (Open interest)
* IV (Implied volatility)
* Change in OI
* Premiums

By analyzing OI build-up, traders infer **support/resistance** and **market sentiment**.

---

## 🧮 14. Practical Realities

| Concept                 | Meaning                                                       |
| ----------------------- | ------------------------------------------------------------- |
| **Physical Settlement** | Stocks are delivered if held till expiry.                     |
| **Cash Settlement**     | Index options/futures are settled in cash.                    |
| **Hedging Ratio**       | Optimal proportion of futures/options to hedge spot exposure. |
| **Volatility Crush**    | IV drops post-event, reducing option premiums.                |

---

## 📘 15. Summary Table

| Aspect           | Futures                      | Options                                 |
| ---------------- | ---------------------------- | --------------------------------------- |
| Directional View | Clear bullish/bearish        | Can profit from volatility or direction |
| Risk             | Unlimited                    | Limited for buyers                      |
| Margin           | High                         | Low for buyers                          |
| Expiry Impact    | None                         | Strong (Theta decay)                    |
| Ideal For        | Experienced traders, hedgers | Beginners, hedgers, volatility traders  |

---

## 🏁 16. Takeaway

**Futures** = High leverage, high risk, linear profit/loss.
**Options** = Non-linear payoffs, risk-defined strategies, time-sensitive.

Mastering both enables:

* Hedging portfolio risk
* Building structured strategies
* Creating consistent income (with proper risk control)

---

Would you like me to now **generate a full illustrated PDF** version of this expanded detailed explanation — including **Greeks charts, payoff diagrams, and example tables** for download?
