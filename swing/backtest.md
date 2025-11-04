Short answer: **start simple → then layer on complexity**.

## What I recommend (practical path)

1. **Phase 1 — Validate the edge per ticker (discovery)**

* Use **`cash_sharing=False` (per-ticker mode)**.
* Equal “bucket” (e.g., ₹1L) per stock; one trade/day/stock.
* Aggregate results across tickers **after** the run.
* Why: isolates the ORB+retest logic from portfolio effects, makes it obvious which symbols actually have edge, and avoids optimizer bias early.

2. **Phase 2 — Reality check with capital constraint**

* Switch to **`cash_sharing=True` (shared pool)** to reflect your real deployable capital (e.g., ₹10L).
* Add practical constraints:

  * **max_concurrent_positions** (e.g., 5–10)
  * **min notional per trade** (skip tiny fills)
  * **allow_partial=True + lock_cash=True**
  * Priority via **`call_seq`** or pre-scaling sizes at the timestamp.
* Why: shows how fills, cash contention, and fees change outcomes.

3. **Phase 3 — Sizing alpha (optional, only if it earns its keep)**

* Add **VOLAR tilt** (ranking → weights with floor/cap, e.g., 2%–50%) on days with multiple signals.
* Keep it **simple & stable**:

  * 60–120 trading-day lookback
  * Shift by 1 day (no look-ahead)
  * Recompute daily; don’t hyper-opt
* Benchmark it vs equal-weight shared pool. Keep only if it **improves net expectancy / reduces DD** after costs.

---

## When to choose what

| Scenario                                                                | Use                                                       |
| ----------------------------------------------------------------------- | --------------------------------------------------------- |
| You’re still proving the signal works, want clean attribution by symbol | **Per-ticker mode**, aggregate later                      |
| You’re preparing for live with a fixed wallet (e.g., ₹10L)              | **Shared pool** + caps (max positions, min notional)      |
| You get too many simultaneous signals and want a principled tilt        | **Shared pool + VOLAR weights** (with floor/cap)          |
| Capital is abundant and you’ll split evenly                             | **Per-ticker** is fine; size each stock to a fixed bucket |

---

## Guardrails that matter in the real world

* **Liquidity & slippage**: enforce min ADV/impact; skip illiquid names/intraday gaps.
* **Concurrency cap**: e.g., max 8 open positions keeps focus and cuts fee drag.
* **Sector/stock caps**: avoid overloading one sector; max 1–2 names per sector.
* **Turnover control**: ORB can be active—watch costs; your Groww model already helps.
* **Walk-forward**: retrain/refresh VOLAR window OOS (e.g., 3y wf: 2y train/1y test).
* **Stability tests**: vary `sl_rupees/tp_rupees`, entry window, and see if rankings still help.

---

## Defaults I’d start with

* **Phase 1 (baseline)**
  `portfolio_mode="per_ticker"`, `capital_per_stock=₹1,00,000`, **no VOLAR**.

* **Phase 2 (deploy model)**
  `portfolio_mode="shared"`, `total_capital_shared=₹10,00,000`, `allow_partial=True`, `lock_cash=True`, **max_concurrent=8** (implement by limiting fills at each timestamp), **min_notional=₹50,000**.

* **Phase 3 (tilt)**
  `use_volar_weights=True`, `volar_lookback_days=60`, `volar_min_w=0.05`, `volar_max_w=0.35`. Keep equal-weight as an A/B benchmark.

---

### Bottom line

* **Yes**: Start **per-ticker** and aggregate to prove the signal.
* **Then**: Move to **shared pool** to mimic real capital constraints.
* **Finally**: Try **VOLAR weighting**—keep it only if it improves **net** results with your brokerage model, lower drawdowns, or both.
