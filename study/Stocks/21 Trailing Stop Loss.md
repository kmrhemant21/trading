### 🧩 **Exit Condition Explained**

| Parameter                  | Meaning                                                                                                                                | Example in your setup                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Stop Loss (SL)**         | The loss level (in **points** or **%**) at which your position will be automatically closed to prevent further losses.                 | **1 point** — if the stock moves 1 point against your position, the trade will exit.         |
| **Target Profit (TP)**     | The profit level (in **points** or **%**) at which your trade will book profit and exit automatically.                                 | **4 points** — if the price moves 4 points in your favor, trade exits with profit.           |
| **Trailing SL (optional)** | A **dynamic stop loss** that moves along with the price when the trade is in profit. It locks in profits as the price moves favorably. | **1 point** — once the price moves 1 point in your favor, the SL starts trailing by 1 point. |
| **TPSL Type**              | Defines whether the SL and TP are measured in **absolute points** or **percentage**.                                                   | **Points (pts)** — values like 1 and 4 mean exactly ₹1 and ₹4 per share movement.            |

---

### ⚙️ **How it works in your configuration**

* **Initial stop loss:** 1 point
* **Target profit:** 4 points
* **Trailing stop loss:** 1 point

#### Suppose:

* You enter a **long trade** at ₹100.
* **SL = 99**, **TP = 104** initially.

Now, as the price rises:

1. When price hits ₹101, the **trailing SL activates** → it moves from ₹99 → ₹100.
2. If price goes to ₹102 → SL moves to ₹101.
3. If price drops from ₹102 to ₹101 → trade exits at ₹101 (locking profit).
4. If price instead continues to ₹104 → TP hit, trade exits with 4-point profit.

---

### 💡 Notes:

* If you choose **percentage mode**, these would mean 1% SL and 4% TP.
* **Trailing SL** is especially useful in volatile markets — it lets you ride profits while minimizing risk.
* These parameters apply **per stock** (each trade managed independently).