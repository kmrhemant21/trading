# File: custom_indicators.py (Corrected Version)

import pandas as pd

def calculate_ema(df, length):
    """Calculates the Exponential Moving Average (EMA)."""
    return df['Close'].ewm(span=length, adjust=False).mean()

def calculate_atr(df, length):
    """Calculates the Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift(1))
    low_prev_close = abs(df['Low'] - df['Close'].shift(1))
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def calculate_supertrend(df, atr_length, multiplier):
    """
    Calculates the Supertrend indicator with the corrected logic.
    """
    # Calculate ATR and basic bands
    df['atr'] = calculate_atr(df, atr_length)
    df['upper_band'] = ((df['High'] + df['Low']) / 2) + (multiplier * df['atr'])
    df['lower_band'] = ((df['High'] + df['Low']) / 2) - (multiplier * df['atr'])
    
    # Initialize Supertrend columns
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 1 # Default to uptrend

    # The core iterative logic for correct Supertrend calculation
    for i in range(1, len(df)):
        curr_idx = df.index[i]
        prev_idx = df.index[i-1]

        # If previous trend was up
        if df.loc[prev_idx, 'supertrend_direction'] == 1:
            # The new supertrend line is the max of the current lower band or the PREVIOUS supertrend line
            df.loc[curr_idx, 'supertrend'] = max(df.loc[curr_idx, 'lower_band'], df.loc[prev_idx, 'supertrend'])
            
            # Check for trend flip: if close crosses BELOW the new supertrend line
            if df.loc[curr_idx, 'Close'] < df.loc[curr_idx, 'supertrend']:
                df.loc[curr_idx, 'supertrend_direction'] = -1
                # On flip, the new line becomes the current UPPER band
                df.loc[curr_idx, 'supertrend'] = df.loc[curr_idx, 'upper_band']
            else:
                df.loc[curr_idx, 'supertrend_direction'] = 1

        # If previous trend was down
        else:
            # The new supertrend line is the min of the current upper band or the PREVIOUS supertrend line
            df.loc[curr_idx, 'supertrend'] = min(df.loc[curr_idx, 'upper_band'], df.loc[prev_idx, 'supertrend'])

            # Check for trend flip: if close crosses ABOVE the new supertrend line
            if df.loc[curr_idx, 'Close'] > df.loc[curr_idx, 'supertrend']:
                df.loc[curr_idx, 'supertrend_direction'] = 1
                # On flip, the new line becomes the current LOWER band
                df.loc[curr_idx, 'supertrend'] = df.loc[curr_idx, 'lower_band']
            else:
                df.loc[curr_idx, 'supertrend_direction'] = -1

    # Clean up intermediate columns
    df.drop(['atr', 'upper_band', 'lower_band'], axis=1, inplace=True)
    return df