Absolutely ✅ — here’s your **final, comprehensive developer documentation** for
`nse_option_chain_sentiment_batch_final.py` — now with **explicit bullish vs. bearish logic explained in detail**, including intuitive interpretations of PCR, OI, ΔOI, and equilibrium zones.

You can save this as `nse_option_chain_sentiment_batch_final.md`.

---

# 📘 Developer Documentation — NSE Option Chain Sentiment Analyzer

---

## 🧩 1️⃣ Purpose

This Python script performs **option chain sentiment analysis** directly from **NSE India** for multiple symbols.
It automatically determines the **nearest expiry**, analyzes **Open Interest (OI)** and **Change in OI (ΔOI)** patterns, identifies **support/resistance zones**, and classifies the market tone as **Bullish, Bearish, or Neutral**.

It outputs:

* Human-readable summaries in the **terminal**
* A structured **summary.csv** (fresh each run)

---

## ⚙️ 2️⃣ Configuration Overview

| Variable            | Description                                            |
| ------------------- | ------------------------------------------------------ |
| `SYMBOLS_FILE`      | List of tickers (`symbols.txt`)                        |
| `OUTPUT_DIR`        | Output folder (default: `option_chain_outputs`)        |
| `EXPIRY_PREF`       | Expiry selector (`auto`, `weekly`, `monthly`, or date) |
| `TOTAL_RETRIES`     | Retry count for HTTP requests                          |
| `BACKOFF_FACTOR`    | Delay multiplier for retries                           |
| `REQUEST_SLEEP_SEC` | Delay between NSE API calls                            |

---

## 🌐 3️⃣ NSE Data Retrieval

The script fetches live option chain data using official NSE APIs:

| Segment  | Endpoint                                                             |
| -------- | -------------------------------------------------------------------- |
| Indices  | `https://www.nseindia.com/api/option-chain-indices?symbol={symbol}`  |
| Equities | `https://www.nseindia.com/api/option-chain-equities?symbol={symbol}` |

To avoid blocking:

* Bootstraps a **session** via `https://www.nseindia.com/`
* Uses realistic headers & cookies
* Retries automatically with exponential backoff

---

## 📅 4️⃣ Expiry Handling Logic

The API provides a list of `expiryDates`.
`choose_exp()` selects one based on `EXPIRY_PREF`:

| Mode           | Behavior                                  |
| -------------- | ----------------------------------------- |
| `"auto"`       | Nearest upcoming expiry                   |
| `"weekly"`     | Nearest weekly expiry (not last Thursday) |
| `"monthly"`    | Last Thursday of the month                |
| `"YYYY-MM-DD"` | Specific expiry if available              |

---

## 🧮 5️⃣ Data Processing Pipeline

Each symbol passes through the following steps:

1. **Fetch JSON** from NSE API
2. **Flatten JSON → DataFrame** with one row per strike:

   ```
   strike, expiry, ce_oi, ce_coi, pe_oi, pe_coi
   ```
3. **Filter** to chosen expiry
4. **Compute aggregates**:

   * Total Call OI (`Σ ce_oi`)
   * Total Put OI (`Σ pe_oi`)
   * ΔCall OI (`Σ ce_coi`)
   * ΔPut OI (`Σ pe_coi`)
   * PCR = Total Put OI / Total Call OI
5. **Find Top-5 Resistances & Supports**:

   * Resistance → highest **Call OI**
   * Support → highest **Put OI**
6. **ΔOI hotspots**:

   * Max Call COI → Fresh resistance zone
   * Max Put COI → Fresh support zone
7. **Equilibrium detection**:

   * If top CE and PE OI are at the same strike → Equilibrium Zone
8. **Closest Gaps**:

   * 2–3 smallest strike differences within Top-5 sets
9. **Sentiment derivation** using PCR + ΔOI logic
10. **Output** → pretty console block + `summary.csv`

---

## 📈 6️⃣ Bullish & Bearish Logic (Core Sentiment Engine)

This is the **heart of the analysis**, combining both **Put-Call Ratio (PCR)** and **Change in OI (ΔOI)** signals.

---

### 🔹 A) PCR Interpretation

| PCR Range     | Sentiment   | Market Psychology                                                                        |
| ------------- | ----------- | ---------------------------------------------------------------------------------------- |
| **> 1.2**     | **Bullish** | More Put OI → traders writing Puts expecting price to hold above → support building      |
| **0.8 – 1.2** | **Neutral** | Balanced activity → sideways / indecisive phase                                          |
| **< 0.8**     | **Bearish** | More Call OI → traders writing Calls expecting price to stay below → resistance building |

**Example:**

> PCR = 0.50 → 2× more Calls than Puts → **Bearish tone**

---

### 🔹 B) Change in OI (ΔOI) Interpretation

| ΔOI Pattern            | Meaning                                                             | Sentiment |
| ---------------------- | ------------------------------------------------------------------- | --------- |
| **ΔPutOI > ΔCallOI**   | More fresh Put writing → confidence in supports → **Bullish tilt**  |           |
| **ΔCallOI > ΔPutOI**   | More fresh Call writing → rising overhead supply → **Bearish tilt** |           |
| **Both small / equal** | Lack of conviction → **Neutral**                                    |           |

**Example:**

> ΔCallOI = +5,000, ΔPutOI = +1,000 → Bears more active → **Bearish**

---

### 🔹 C) Combined Scoring Logic

| PCR tilt       | ΔOI tilt                | Final Sentiment |
| -------------- | ----------------------- | --------------- |
| Bullish        | Bullish                 | **Bullish**     |
| Bearish        | Bearish                 | **Bearish**     |
| Opposite tilts | **Neutral / Mild Bias** |                 |
| Both neutral   | **Neutral**             |                 |

**Pseudocode:**

```python
score = 0
if pcr < 0.8: score -= 1
elif pcr > 1.2: score += 1

if delta_put_oi > delta_call_oi: score += 1
elif delta_call_oi > delta_put_oi: score -= 1

sentiment = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
```

**Human-readable explanation (example):**

```
PCR 0.57 (<0.80) → bearish tilt; ΔCallOI 618 > ΔPutOI 264 → bearish tilt
```

---

### 🔹 D) Support & Resistance (OI Structure)

| Signal               | Interpretation                                                             | Market Bias |
| -------------------- | -------------------------------------------------------------------------- | ----------- |
| **Call OI rising**   | More traders writing Calls → Expect price won’t rise → **Bearish ceiling** |             |
| **Put OI rising**    | More traders writing Puts → Expect price won’t fall → **Bullish floor**    |             |
| **Call OI > Put OI** | Market expects downside resistance to dominate                             | Bearish     |
| **Put OI > Call OI** | Market expects support to hold                                             | Bullish     |

---

### 🔹 E) Equilibrium Zone Logic

If **max Call OI** and **max Put OI** are at the **same strike**:

→ **Equilibrium Zone**
= “Battlefield” between bulls & bears.

* Often leads to **sideways movement**
* Near expiry, price tends to **pin** to this strike (max-pain effect)

Example:

> Resistance (max CE OI): 1000
> Support (max PE OI): 1000
> → Equilibrium detected — **neutral to range-bound sentiment**

---

## 🧾 7️⃣ Example Interpretation (HDFCBANK)

| Metric    | Value       | Signal | Meaning                                  |
| --------- | ----------- | ------ | ---------------------------------------- |
| PCR       | 0.78        | < 0.8  | Bearish tilt                             |
| ΔCallOI   | +5,747      | ↑      | Call writers active → Resistance forming |
| ΔPutOI    | +3,466      | ↓      | Support weaker                           |
| Top CE OI | 1000        | —      | Strong ceiling at 1000                   |
| Top PE OI | 1000        | —      | Same strike → Equilibrium Zone           |
| Sentiment | **Bearish** | —      | Bears dominating                         |

---

## 📊 8️⃣ Output Columns (`summary.csv`)

| Column                           | Description                            |
| -------------------------------- | -------------------------------------- |
| symbol                           | Stock/index symbol                     |
| segment                          | `indices` / `equities`                 |
| expiry                           | Expiry date chosen                     |
| as_of                            | Timestamp from NSE feed                |
| pcr                              | Put/Call Ratio                         |
| total_call_oi / total_put_oi     | Aggregate OI                           |
| delta_call_oi / delta_put_oi     | Intraday change in OI                  |
| top5_resistances / top5_supports | Strike:OI pairs                        |
| res_gaps / sup_gaps              | Smallest internal gaps in Top-5 levels |
| ce_hotspot / pe_hotspot          | Strikes with highest ΔOI               |
| equilibrium / equilibrium_strike | True if both sides at same strike      |
| sentiment                        | Bullish / Bearish / Neutral            |
| rationale                        | Text reason (PCR & ΔOI)                |
| report_text                      | Full pretty block                      |
| status                           | “ok” or “error: …”                     |

---

## 🧱 9️⃣ File Layout

```
project/
│
├── nse_option_chain_sentiment_batch_final.py
├── symbols.txt
└── option_chain_outputs/
    └── summary.csv
```

---

## 🧭 10️⃣ Running the Script

**Install dependencies**

```bash
pip install requests pandas urllib3
```

**Run**

```bash
python nse_option_chain_sentiment_batch_final.py
```

**Output**

```
option_chain_outputs/summary.csv
```

Each run overwrites the previous output.

---

## 🧠 11️⃣ Sentiment Quick Reference

| Pattern                 | Interpretation                              | Sentiment                      |
| ----------------------- | ------------------------------------------- | ------------------------------ |
| **Call OI ↑, Put OI ↓** | Resistance strengthening, support weakening | **Bearish**                    |
| **Put OI ↑, Call OI ↓** | Support strengthening, resistance weakening | **Bullish**                    |
| **Both OI ↑**           | Writers active both sides (range-bound)     | **Neutral**                    |
| **Both OI ↓**           | Position unwinding                          | **Trend reversal / Uncertain** |

---

## 🧩 12️⃣ Example Output (360ONE)

```
========================================================================
Symbol: 360ONE | Segment: equities | Expiry: 25-Nov-2025
As Of : 31-Oct-2025 15:30:00
------------------------------------------------------------------------
Total Call OI: 2,002
Total Put  OI: 1,147
PCR        : 0.57
ΔCall OI   : 618
ΔPut  OI   : 264
------------------------------------------------------------------------
Top-5 Resistances (CE OI): 1200:619; 1180:398; 1160:222; 1140:166; 1100:165
Top-5 Supports    (PE OI): 1100:250; 1000:177; 1140:141; 1020:110; 1120:85
Flow (ΔOI) CE hotspot:  1180  | ΔOI: 269
Flow (ΔOI) PE hotspot:  1000  | ΔOI: 103
Closest resistance gaps: 20; 20; 20
Closest support gaps   : 20; 20; 20
------------------------------------------------------------------------
Sentiment: Bearish
Why      : PCR 0.57 (<0.80) → bearish tilt; ΔCallOI 618 > ΔPutOI 264 → bearish tilt
========================================================================
```

**Interpretation:**

* PCR < 0.8 → Bearish
* Call writers active → strong resistance at 1200–1180
* Weak Put buildup → minimal support
  → Market bias: **Bearish**

---

## 🧩 13️⃣ Troubleshooting

| Issue                      | Possible Cause       | Fix                                 |
| -------------------------- | -------------------- | ----------------------------------- |
| HTTP 429 / 503             | NSE rate limit       | Increase `REQUEST_SLEEP_SEC` to ≥1s |
| “error: No expiry dates”   | API temporary outage | Retry after a few minutes           |
| PCR shows `inf` or `NaN`   | Missing CE data      | Ignore / log warning                |
| Empty supports/resistances | Illiquid symbol      | Skip or verify F&O eligibility      |

---

## 📈 14️⃣ Extensibility

You can extend this script easily:

| Add-on                 | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| **Max Pain**           | Compute strike with minimum total payout (for expiry pinning) |
| **Underlying Price**   | Add live LTP from `quote-equity` API for context              |
| **Visualization**      | Plot CE/PE OI ladders or PCR charts                           |
| **Alerts**             | Send Telegram/Slack messages on sentiment flip                |
| **Historical Storage** | Append daily runs to a DB (for sentiment trend analysis)      |

---

## ⚡ 15️⃣ Performance Tips

* Default delay = 1s per request → safe for ~100–150 symbols/run.
* Run **after 3:30 PM IST** for stable, EOD data.
* Use **cron jobs** or **Task Scheduler** for automation.

---

## 🧭 16️⃣ Summary

| Component       | Description                                              |
| --------------- | -------------------------------------------------------- |
| Data Source     | NSE India public API                                     |
| Output          | CSV + terminal summary                                   |
| Bias Model      | PCR + ΔOI + OI Structure                                 |
| Primary Signals | Resistance/Support buildup, OI hotspots, equilibrium     |
| Final Labels    | Bullish / Neutral / Bearish                              |
| Designed For    | F&O traders, quantitative analysts, sentiment dashboards |

---

## 📚 References

* **NSE Official Data Portal:** [https://www.nseindia.com/option-chain](https://www.nseindia.com/option-chain)
* **PCR & OI Concepts:**

  * NISM Series VIII – Equity Derivatives
  * CBOE Option Analytics Primer
  * NSE Academy Technical Analysis Module

---

✅ **In short:**

> The script converts raw NSE option chain data into actionable sentiment insights — highlighting where traders are placing their bets and how support/resistance is evolving intraday.

---

Would you like me to create a **flowchart diagram (data flow & logic)** version of this documentation (for README or Confluence)? It can visually show:

```
symbols.txt → NSE API → DataFrame → PCR/OI analysis → sentiment engine → summary.csv
```
