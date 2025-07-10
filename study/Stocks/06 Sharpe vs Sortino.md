**Sharpe Ratio** and **Sortino Ratio** are two of the most widely used **risk-adjusted performance** metrics in finance. They both attempt to tell you how much return you’re getting *per unit of risk*, but they measure “risk” slightly differently.

---

## Sharpe Ratio

* **Definition**

  $$
    \text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
  $$

  where

  * $R_p$ = average portfolio return
  * $R_f$ = risk-free rate (e.g. yield on a short-term Treasury)
  * $\sigma_p$ = standard deviation of portfolio returns

* **Interpretation**

  * Tells you how many units of total volatility you’re “paid” for.
  * **Higher is better**: >1 is good, >2 is very good, >3 is excellent.
  * Applies **penalty** for *all* volatility—both upside and downside swings.

* **Use Cases**

  * Great for comparing strategies that have roughly **symmetric** returns.
  * Easy to compute and broadly understood.

---

## Sortino Ratio

* **Definition**

  $$
    \text{Sortino} = \frac{R_p - R_f}{\sigma_{\text{down}}}
  $$

  where

  * $\sigma_{\text{down}} =$ the standard deviation of **negative** returns only (downside deviation).

* **Interpretation**

  * Focuses *only* on “bad” volatility—penalizes the strategy for downside moves, but **ignores** upside volatility.
  * **Higher is better**; often larger than the Sharpe for the same return series, because $\sigma_{\text{down}} \le \sigma_p$.

* **Use Cases**

  * Preferred when you care mostly about **protecting from negative returns** rather than penalizing big positive jumps.
  * More meaningful for **asymmetric** return profiles or strategies with occasional large gains.

---

## Key Differences

| Feature            | Sharpe Ratio                          | Sortino Ratio                               |
| ------------------ | ------------------------------------- | ------------------------------------------- |
| Denominator        | Total standard deviation ($\sigma_p$) | Downside deviation ($\sigma_{\text{down}}$) |
| Penalty on Upside? | Yes (penalizes all volatility)        | No (only penalizes negative moves)          |
| Sensitivity        | Sensitive to both good & bad swings   | Sensitive only to bad swings                |
| Typical Use        | Broad comparisons                     | Downside‐focused risk assessment            |

---

## Which Should You Use?

* **Sharpe** is a good *baseline*—if two strategies have similar Sharpe, you know they’re getting paid well for their total risk.
* **Sortino** is better when you want to emphasize **capital preservation** and not “penalize” your strategy for big winners.

In practice, many quants report **both**. If your strategy has few big upside outliers but small frequent losses, you’ll often see a **higher Sortino** than Sharpe, reflecting that the “good” volatility shouldn’t count against you.
