```
import random
import json

# Define some sample tickers and patterns
tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NFLX', 'META', 'IBM', 'JPM', 'BAC']
timeframes = ['daily', 'weekly', 'monthly']
indicators = ['14-period RSI', '5×20 EMA crossover', '20-period Bollinger Bands', 'MACD (12,26,9)', '50-day SMA']
strategies = ['momentum', 'mean-reversion', 'breakout', 'trend-following']

def generate_prompt_completion():
    choice = random.choice(['indicator', 'trend', 'signal', 'portfolio'])
    ticker = random.choice(tickers)
    if choice == 'indicator':
        ind = random.choice(indicators)
        tf = random.choice(timeframes)
        prompt = f"Analyze the {ind} for {ticker} on the {tf} timeframe and suggest an action."
        # Generate synthetic indicator values
        if 'RSI' in ind:
            val = round(random.uniform(10, 90), 1)
            comp = f"The {ind} for {ticker} on the {tf} timeframe is {val}. "
            if val > 70:
                comp += "This indicates overbought conditions; consider taking profits or waiting for a pullback."
            elif val < 30:
                comp += "This indicates oversold conditions; consider a buying opportunity."
            else:
                comp += "This is in a neutral zone; hold current positions."
        elif 'EMA' in ind:
            short = round(random.uniform(50, 150), 2)
            long = round(short - random.uniform(-5, 5), 2)
            comp = (f"The short-term EMA is {short} and the long-term EMA is {long}. "
                    + ("Bullish crossover detected; consider entering a long position."
                       if short > long else "Bearish crossover detected; consider exiting long positions or shorting."))
        elif 'Bollinger' in ind:
            mid = round(random.uniform(100, 200), 2)
            upper = round(mid + random.uniform(1, 5), 2)
            lower = round(mid - random.uniform(1, 5), 2)
            comp = (f"{ticker}'s price is near the bands: upper={upper}, middle={mid}, lower={lower}. "
                    + "A breakout above upper band suggests momentum; a move below lower band suggests mean reversion.")
        elif 'MACD' in ind:
            macd = round(random.uniform(-5, 5), 2)
            signal = round(macd + random.uniform(-1, 1), 2)
            comp = (f"MACD line is {macd} and signal line is {signal}. "
                    + ("MACD above signal line; bullish bias." if macd > signal else "MACD below signal line; bearish bias."))
        else:  # SMA
            sma = round(random.uniform(100, 300), 2)
            price = round(sma + random.uniform(-10, 10), 2)
            comp = (f"The current price is {price} and the 50-day SMA is {sma}. "
                    + ("Positive trend; price above SMA." if price > sma else "Negative trend; price below SMA."))
    elif choice == 'trend':
        strat = random.choice(strategies)
        prompt = f"Describe the current {strat} trend for {ticker} using multiple indicators."
        comp = f"{ticker} is showing a {strat} trend: price above SMA and EMAs, RSI at {round(random.uniform(40, 60),1)}, volume trending upward."
    elif choice == 'signal':
        strat = random.choice(strategies)
        prompt = f"Generate a {strat} trade signal for {ticker} with entry and exit criteria."
        entry = round(random.uniform(100, 200), 2)
        exit_p = entry + random.uniform(2, 10)
        stop = entry - random.uniform(2, 10)
        comp = (f"Enter long at {entry}, set take-profit at {round(exit_p,2)}, stop-loss at {round(stop,2)} "
                + "based on strategy and risk management.")
    else:  # portfolio
        holdings = random.sample(tickers, k=3)
        prompt = f"Summarize the portfolio performance for {', '.join(holdings)} over the past month."
        returns = [round(random.uniform(-5, 15), 1) for _ in holdings]
        comp = " | ".join(f"{t}: {r}%" for t, r in zip(holdings, returns))
        comp = f"Over the past month, " + comp + ". Overall volatility moderate; consider rebalancing."
    return {"prompt": prompt, "completion": comp}

# Generate 2000 entries
entries = [generate_prompt_completion() for _ in range(2000)]

# Save to JSONL
file_path = '/mnt/data/stock_bot_data.jsonl'
with open(file_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

file_path
```

```
import random
import json

# Define some sample tickers and patterns
tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NFLX', 'META', 'IBM', 'JPM', 'BAC']
timeframes = ['daily', 'weekly', 'monthly']
indicators = ['14-period RSI', '5×20 EMA crossover', '20-period Bollinger Bands', 'MACD (12,26,9)', '50-day SMA']
strategies = ['momentum', 'mean-reversion', 'breakout', 'trend-following']

def generate_prompt_completion():
    choice = random.choice(['indicator', 'trend', 'signal', 'portfolio'])
    ticker = random.choice(tickers)
    if choice == 'indicator':
        ind = random.choice(indicators)
        tf = random.choice(timeframes)
        prompt = f"Analyze the {ind} for {ticker} on the {tf} timeframe and suggest an action."
        # Generate synthetic indicator values
        if 'RSI' in ind:
            val = round(random.uniform(10, 90), 1)
            comp = f"The {ind} for {ticker} on the {tf} timeframe is {val}. "
            if val > 70:
                comp += "This indicates overbought conditions; consider taking profits or waiting for a pullback."
            elif val < 30:
                comp += "This indicates oversold conditions; consider a buying opportunity."
            else:
                comp += "This is in a neutral zone; hold current positions."
        elif 'EMA' in ind:
            short = round(random.uniform(50, 150), 2)
            long = round(short - random.uniform(-5, 5), 2)
            comp = (f"The short-term EMA is {short} and the long-term EMA is {long}. "
                    + ("Bullish crossover detected; consider entering a long position."
                       if short > long else "Bearish crossover detected; consider exiting long positions or shorting."))
        elif 'Bollinger' in ind:
            mid = round(random.uniform(100, 200), 2)
            upper = round(mid + random.uniform(1, 5), 2)
            lower = round(mid - random.uniform(1, 5), 2)
            comp = (f"{ticker}'s price is near the bands: upper={upper}, middle={mid}, lower={lower}. "
                    + "A breakout above upper band suggests momentum; a move below lower band suggests mean reversion.")
        elif 'MACD' in ind:
            macd = round(random.uniform(-5, 5), 2)
            signal = round(macd + random.uniform(-1, 1), 2)
            comp = (f"MACD line is {macd} and signal line is {signal}. "
                    + ("MACD above signal line; bullish bias." if macd > signal else "MACD below signal line; bearish bias."))
        else:  # SMA
            sma = round(random.uniform(100, 300), 2)
            price = round(sma + random.uniform(-10, 10), 2)
            comp = (f"The current price is {price} and the 50-day SMA is {sma}. "
                    + ("Positive trend; price above SMA." if price > sma else "Negative trend; price below SMA."))
    elif choice == 'trend':
        strat = random.choice(strategies)
        prompt = f"Describe the current {strat} trend for {ticker} using multiple indicators."
        comp = f"{ticker} is showing a {strat} trend: price above SMA and EMAs, RSI at {round(random.uniform(40, 60),1)}, volume trending upward."
    elif choice == 'signal':
        strat = random.choice(strategies)
        prompt = f"Generate a {strat} trade signal for {ticker} with entry and exit criteria."
        entry = round(random.uniform(100, 200), 2)
        exit_p = round(entry + random.uniform(2, 10),2)
        stop = round(entry - random.uniform(2, 10),2)
        comp = (f"Enter long at {entry}, set take-profit at {exit_p}, stop-loss at {stop} based on strategy and risk management.")
    else:  # portfolio
        holdings = random.sample(tickers, k=3)
        prompt = f"Summarize the portfolio performance for {', '.join(holdings)} over the past month."
        returns = [round(random.uniform(-5, 15), 1) for _ in holdings]
        comp = " | ".join(f"{t}: {r}%" for t, r in zip(holdings, returns))
        comp = f"Over the past month, " + comp + ". Overall volatility moderate; consider rebalancing."

    return {"prompt": prompt, "completion": comp}

# Generate 2000 entries
entries = [generate_prompt_completion() for _ in range(2000)]

# Save to JSONL
file_path = '/mnt/data/stock_bot_data.jsonl'
with open(file_path, 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')

# Preview first 5 entries
import pandas as pd
df_preview = pd.DataFrame(entries[:5])
import ace_tools as tools; tools.display_dataframe_to_user(name="Preview of Generated stock_bot_data.jsonl", dataframe=df_preview)

file_path

```