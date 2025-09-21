# Supertrend & EMA Swing Trading Strategy Backtester

This project provides a complete Python script to backtest a disciplined, multi-indicator swing trading strategy on a list of stocks. It uses a combination of trend and momentum indicators for entry signals and a sophisticated 4-tier exit system to manage risk and profits.

The backtester is fully customizable and outputs a detailed trade-by-trade log to a CSV file for in-depth analysis.

-----

## 📈 The Trading Strategy

The core of this project is a **Trend-Momentum Swing Strategy**. It is designed to enter trades at the beginning of a confirmed uptrend and uses a strict set of rules to exit, balancing profit-taking with capital protection.

### **Entry Logic**

A trade is initiated on the **first day** a stock enters a **"Confirmed Uptrend State"**. This state is defined by two conditions being met simultaneously:

1.  **Trend Confirmation:** The **Supertrend (10, 3)** indicator must be bullish (i.e., the price is above the Supertrend line).
2.  **Momentum Confirmation:** The **fast 9-period EMA** must be above the **slow 15-period EMA**.

This dual confirmation ensures we only trade stocks that have both a solid underlying trend and positive short-term momentum.

### **Exit Logic**

The exit strategy is a 4-tier system. The first condition to be met will trigger the exit. This creates a disciplined approach to every trade.

1.  **Hard Stop-Loss (HSL):** A fixed percentage (e.g., **-5%**) below the entry price. This is the maximum acceptable loss on any single trade.
2.  **Trailing Stop-Loss (TSL):** A dynamic percentage (e.g., **10%**) that follows the peak price of the trade. This is designed to lock in profits as a trade moves in our favor.
3.  **Time-Based Stop:** A maximum holding period (e.g., **10 trading days / 2 weeks**). This prevents capital from being stuck in trades that are not performing.
4.  **Supertrend Stop (SL):** The original indicator-based stop. If the trend reverses and the Supertrend flips to bearish, the trade is closed.

-----

## 📂 Project Files

  * `backtest_multi_stock.py`: The main script. You run this file to perform the backtest. All strategy parameters (tickers, dates, stop-loss percentages) are configured here.
  * `custom_indicators.py`: A helper file containing the Python functions to calculate the EMA and Supertrend indicators from scratch. It must be in the same directory as the main script.
  * `backtest_results_advanced.csv`: The output file. After a backtest is run, this CSV file is generated, containing a detailed log of every trade executed.

-----

## 🛠️ How to Use

#### **1. Prerequisites**

Make sure you have the required Python libraries installed. If not, open your terminal and run:

```bash
pip install pandas yfinance
```

#### **2. Setup**

Place both `backtest_multi_stock.py` and `custom_indicators.py` in the same project folder.

#### **3. Configuration**

Open `backtest_multi_stock.py` in a code editor. You can customize the following sections at the top of the file:

  * `TICKER_LIST`: Add or remove the stock symbols you want to test.
  * `START_DATE` and `END_DATE`: Define the backtesting period.
  * Indicator settings (`ST_LENGTH`, `EMA_FAST_LENGTH`, etc.).

You can also tune the exit strategy parameters in the `if __name__ == "__main__":` block at the bottom of the script:

```python
executed_trades = run_backtest(
    stock_df, 
    stock_ticker, 
    trailing_stop_loss_pct=10.0,
    hard_stop_loss_pct=5.0,
    max_holding_days=10 
)
```

#### **4. Run the Backtest**

Navigate to the project folder in your terminal and run the script:

```bash
python backtest_multi_stock.py
```

#### **5. Analyze the Results**

The script will print a summary of the overall performance in the console. For a detailed breakdown, open the generated `backtest_results_advanced.csv` file in a spreadsheet program like Excel or Google Sheets. The `exit_reason` column is particularly useful for understanding *how* your trades are ending (e.g., hitting your profit target vs. being stopped out).