#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math, warnings, logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    import matplotlib.pyplot as plt
except Exception:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("candle_confirm_mvo_v1")

# =========================
# CONFIG
# =========================
@dataclass
class Config:
    # Data
    start_date: str = "2015-01-01"
    end_date: str   = "2025-01-01"
    static_symbols: Optional[List[str]] = None
    static_symbols_path: Optional[str] = None
    cache_dir: str = "cache"
    out_dir: str   = "outputs"
    plot: bool     = True

    # Trend filters
    ema_fast: int = 10
    ema_slow: int = 20
    ema_htf:  int = 200  # long-term trend
    use_htf_trend: bool = True  # require Close > EMA200

    # Exits
    stop_loss_pct: float = 0.05
    target_pct: float    = 0.10

    # Portfolio
    apply_fees: bool    = True
    initial_capital: float = 500_000.0
    max_concurrent_positions: int = 5
    deploy_cash_frac: float = 0.25
    top_k_daily: int = 300

    # Ranking & filters
    benchmark_try: Tuple[str,...] = ("^CNX500","^CRSLDX","^NSE500","^NIFTY500","^BSE500","^NSEI")
    volar_lookback: int = 252
    filter_52w_window: int = 252
    within_pct_of_52w_high: float = 0.50

    # Liquidity guards (optional)
    enable_basic_liquidity: bool = False
    min_price_inr: float = 50.0
    min_avg_vol_20d: float = 50_000.0

CFG = Config()

# =========================
# FEES (Groww-like)
# =========================
APPLY_FEES = True

def calc_fees(turnover_buy: float, turnover_sell: float) -> float:
    if not APPLY_FEES:
        return 0.0
    BROKER_PCT = 0.001
    BROKER_MIN = 5.0
    BROKER_CAP = 20.0
    STT_PCT = 0.001
    STAMP_BUY_PCT = 0.00015
    EXCH_PCT = 0.0000297
    SEBI_PCT = 0.000001
    IPFT_PCT = 0.000001
    GST_PCT = 0.18
    DP_SELL = 20.0 if turnover_sell >= 100 else 0.0

    def _broker(turnover):
        if turnover <= 0: return 0.0
        fee = turnover * BROKER_PCT
        return max(BROKER_MIN, min(fee, BROKER_CAP))

    br_buy  = _broker(turnover_buy)
    br_sell = _broker(turnover_sell)
    stt   = STT_PCT * (turnover_buy + turnover_sell)
    stamp = STAMP_BUY_PCT * turnover_buy
    exch  = EXCH_PCT * (turnover_buy + turnover_sell)
    sebi  = SEBI_PCT * (turnover_buy + turnover_sell)
    ipft  = IPFT_PCT * (turnover_buy + turnover_sell)
    dp    = DP_SELL
    gst_base = br_buy + br_sell + dp + exch + sebi + ipft
    gst   = GST_PCT * gst_base
    return float((br_buy + br_sell) + stt + stamp + exch + sebi + ipft + dp + gst)

# =========================
# Helpers
# =========================
def ensure_dirs(*paths): [os.makedirs(p, exist_ok=True) for p in paths]

def today_str():
    return pd.Timestamp.today(tz="Asia/Kolkata").strftime("%Y-%m-%d")

def load_static_symbols(static_symbols: Optional[List[str]], static_symbols_path: Optional[str]) -> List[str]:
    syms: List[str] = []
    if static_symbols and len(static_symbols) > 0:
        syms = list(static_symbols)
    elif static_symbols_path and os.path.exists(static_symbols_path):
        with open(static_symbols_path, "r") as f:
            syms = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Provide CFG.static_symbols=[...] or CFG.static_symbols_path file.")
    out = []
    for s in syms:
        s = s.strip().upper()
        if not s.endswith(".NS"): s = f"{s}.NS"
        out.append(s)
    uniq = []
    seen = set()
    for s in out:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq

def fetch_prices(tickers: List[str], start: str, end: Optional[str], cache_dir: str) -> Dict[str, pd.DataFrame]:
    ensure_dirs(cache_dir)
    data = {}
    end = end or today_str()
    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker.replace('^','_')}.parquet")
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if len(df) and pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d") >= end:
                    data[ticker] = df; continue
            except Exception:
                pass
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, multi_level_index=False)
            if df is None or df.empty: continue
            df = df.rename(columns=str.title)[['Open','High','Low','Close','Volume']].dropna()
            df.index.name = "date"
            df.to_parquet(cache_path)
            data[ticker] = df
        except Exception:
            continue
    return data

# =========================
# Indicators (pure Python)
# =========================
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

def adx_plus_minus_di(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, prev_close)
    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()

    plus_di  = 100 * (plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    return adx_series, plus_di, minus_di

# =========================
# Candlestick pattern detectors (bullish only)
# =========================
# All functions return a boolean pd.Series aligned to df.index
def _body(o, c): return (c - o).abs()
def _is_bull(o, c): return c > o
def _is_bear(o, c): return c < o

def patt_bull_engulfing(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    op, cp = o.shift(1), c.shift(1)
    cond = _is_bear(op, cp) & _is_bull(o, c) & (c >= op) & (o <= cp)
    return cond.fillna(False)

def patt_piercing(df: pd.DataFrame) -> pd.Series:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    op, hp, lp, cp = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    mid_prev = (op + cp) / 2.0
    cond = _is_bear(op, cp) & _is_bull(o, c) \
           & (o < lp) \
           & (c > mid_prev) & (c < op)
    return cond.fillna(False)

def patt_morning_star(df: pd.DataFrame) -> pd.Series:
    # 3-candle: bear -> small -> strong bull closing into first candle body (>50%)
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]
    o1, c1 = o.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)

    bear1 = _is_bear(o2, c2)
    small2 = (_body(o1, c1) <= (_body(o2, c2) * 0.6))  # small-ish
    bull3 = _is_bull(o, c)

    # Close of 3rd candle retraces >50% into candle 1 body
    mid1 = (o2 + c2) / 2.0
    retrace = c > mid1

    cond = bear1 & small2 & bull3 & retrace
    return cond.fillna(False)

def patt_harami_bull(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]
    o1, c1 = o.shift(1), c.shift(1)
    # Previous bearish, current small body entirely within previous body
    prev_bear = _is_bear(o1, c1)
    body_small = (_body(o, c) <= _body(o1, c1) * 0.75)
    inside = (h <= o1.clip(upper=h)) & (l >= c1.clip(lower=l))  # h<=prev open and l>=prev close for bearish prev
    # Allow bullish or doji small body
    cond = prev_bear & body_small & inside & (c >= o)  # bullish/neutral close
    return cond.fillna(False)

def patt_harami_cross_bull(df: pd.DataFrame, doji_pct=0.1) -> pd.Series:
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]
    o1, c1 = o.shift(1), c.shift(1)
    rng = (h - l).replace(0, np.nan)
    doji = (_body(o, c) <= (rng * doji_pct))
    prev_bear = _is_bear(o1, c1)
    inside = (h <= o1) & (l >= c1)
    cond = prev_bear & doji & inside
    return cond.fillna(False)

def patt_hammer(df: pd.DataFrame, shadow_mult=2.0) -> pd.Series:
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]
    body = _body(o, c)
    lower_shadow = (o.clip(lower=c) - l).abs() if (o>=c).all() else (np.minimum(o, c) - l).abs()
    upper_shadow = (h - np.maximum(o, c)).abs()
    cond = (lower_shadow >= shadow_mult * body) & (upper_shadow <= body) & (c >= o)  # bullish close
    return cond.fillna(False)

def patt_inverted_hammer(df: pd.DataFrame, shadow_mult=2.0) -> pd.Series:
    o, c, h, l = df["Open"], df["Close"], df["High"], df["Low"]
    body = _body(o, c)
    upper_shadow = (h - np.maximum(o, c)).abs()
    lower_shadow = (np.minimum(o, c) - l).abs()
    cond = (upper_shadow >= shadow_mult * body) & (lower_shadow <= body) & (c >= o)
    return cond.fillna(False)

# Unified bullish pattern set
BULLISH_PATTERNS = [
    "ENGULFING",
    "PIERCING",
    "MORNING_STAR",
    "HARAMI",
    "HARAMI_CROSS",
    "HAMMER",
    "INVERTED_HAMMER",
]

def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    # Trend EMAs
    out["ema_fast"] = ema(out["Close"], cfg.ema_fast)
    out["ema_slow"] = ema(out["Close"], cfg.ema_slow)
    out["ema_htf"]  = ema(out["Close"], cfg.ema_htf)

    # Confirmations
    out["rsi"] = rsi(out["Close"], 14)
    macd_line, macd_sig, macd_hist = macd(out["Close"], 12, 26, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    out["sma50"] = sma(out["Close"], 50)
    out["bb_mid"] = sma(out["Close"], 20)
    adxv, pdi, ndi = adx_plus_minus_di(out["High"], out["Low"], out["Close"], 14)
    out["adx"] = adxv
    out["+di"] = pdi
    out["-di"] = ndi

    # Liquidity helpers
    out["avg_vol_20"] = out["Volume"].rolling(20).mean()

    # 52w high for ranking filter later
    out["high_52w"] = out["Close"].rolling(252).max()

    # ----- Candlestick pattern masks -----
    patt_map = {
        "ENGULFING":       patt_bull_engulfing(out),
        "PIERCING":        patt_piercing(out),
        "MORNING_STAR":    patt_morning_star(out),
        "HARAMI":          patt_harami_bull(out),
        "HARAMI_CROSS":    patt_harami_cross_bull(out),
        "HAMMER":          patt_hammer(out),
        "INVERTED_HAMMER": patt_inverted_hammer(out),
    }
    # Any bullish pattern today
    patt_any = None
    for name in BULLISH_PATTERNS:
        patt_any = patt_map[name] if patt_any is None else (patt_any | patt_map[name])
    out["bullish_pattern"] = patt_any.fillna(False)

    return out.dropna()

def basic_liquidity_ok(row: pd.Series, cfg: Config) -> bool:
    if not cfg.enable_basic_liquidity:
        return True
    if row["Close"] < cfg.min_price_inr: return False
    if row["avg_vol_20"] < cfg.min_avg_vol_20d: return False
    return True

def simulate_ticker(ticker: str, df: pd.DataFrame, cfg: Config):
    d = compute_indicators(df, cfg).copy()
    # NOTE: added "pattern" so we can carry the exact bullish pattern name to the portfolio stage
    cols = ["ticker","side","date","price","shares","reason","signal_reason","score",
            "pattern","rsi","adx","+di","-di","macd_line","macd_signal",
            "ema_fast","ema_slow","ema_htf","sma50","bb_mid","close","high_52w"]
    if d.empty:
        return pd.DataFrame(columns=cols), pd.Series(dtype=float)

    # --- Entry conditions ---
    trend_ok   = (d["ema_fast"] > d["ema_slow"])
    htf_ok     = (d["Close"] > d["ema_htf"]) if cfg.use_htf_trend else pd.Series(True, index=d.index)

    # Rebuild the same pattern map here to know exactly which pattern fired at dt
    patt_map = {
        "ENGULFING":       patt_bull_engulfing(d),
        "PIERCING":        patt_piercing(d),
        "MORNING_STAR":    patt_morning_star(d),
        "HARAMI":          patt_harami_bull(d),
        "HARAMI_CROSS":    patt_harami_cross_bull(d),
        "HAMMER":          patt_hammer(d),
        "INVERTED_HAMMER": patt_inverted_hammer(d),
    }
    pattern_ok = None
    for nm, ser in patt_map.items():
        pattern_ok = ser if pattern_ok is None else (pattern_ok | ser)
    pattern_ok = pattern_ok.fillna(False)

    # Confirmation: any ONE true
    conf_rsi    = d["rsi"] > 50.0
    conf_macd   = d["macd_line"] > d["macd_signal"]
    conf_adx    = (d["adx"] > 20.0) & (d["+di"] > d["-di"])
    conf_sma50  = d["Close"] > d["sma50"]
    conf_bbmid  = d["Close"] > d["bb_mid"]
    confirmation_any = conf_rsi | conf_macd | conf_adx | conf_sma50 | conf_bbmid

    entry_signal = pattern_ok & trend_ok & htf_ok & confirmation_any
    exit_ind = (d["ema_fast"] < d["ema_slow"])  # trend invalidation

    in_pos = False
    entry_px = stop_px = tgt_px = 0.0
    trades = []

    idx = list(d.index)
    for i in range(len(idx)-1):
        dt, nxt = idx[i], idx[i+1]
        row, nxt_row = d.loc[dt], d.loc[nxt]

        if not in_pos:
            if entry_signal.loc[dt]:
                # Which bullish pattern(s) fired exactly today?
                patterns_detected = [nm for nm, ser in patt_map.items() if bool(ser.loc[dt])]
                patt_str = " + ".join(patterns_detected) if patterns_detected else "BullishPattern"

                # Which confirmations passed?
                confs = []
                if conf_rsi.loc[dt]:    confs.append("RSI>50")
                if conf_macd.loc[dt]:   confs.append("MACD>Signal")
                if conf_adx.loc[dt]:    confs.append("ADX>20 & +DI>-DI")
                if conf_sma50.loc[dt]:  confs.append("Close>SMA50")
                if conf_bbmid.loc[dt]:  confs.append("Close>BBmid")
                sig_reason = ", ".join(confs) if confs else "none"

                px = float(nxt_row["Open"])  # enter next open
                trades.append({
                    "ticker": ticker, "side": "BUY", "date": nxt,
                    "price": px, "shares": 0,
                    # reason is a short tag here; portfolio stage will build a verbose "Entry: ..." string
                    "reason": "candidate",
                    "signal_reason": sig_reason,
                    "score": float(1000.0*((row["ema_fast"]-row["ema_slow"])/row["Close"])) + float(row["adx"]) - float(row["rsi"]),
                    "pattern": patt_str,
                    "rsi": float(row["rsi"]), "adx": float(row["adx"]), "+di": float(row["+di"]), "-di": float(row["-di"]),
                    "macd_line": float(row["macd_line"]), "macd_signal": float(row["macd_signal"]),
                    "ema_fast": float(row["ema_fast"]), "ema_slow": float(row["ema_slow"]), "ema_htf": float(row["ema_htf"]),
                    "sma50": float(row["sma50"]), "bb_mid": float(row["bb_mid"]),
                    "close": float(row["Close"]), "high_52w": float(row["high_52w"])
                })
                in_pos = True
                entry_px = px
                stop_px = entry_px * (1 - cfg.stop_loss_pct)
                tgt_px  = entry_px * (1 + cfg.target_pct)

        else:
            hit = None
            exec_date = nxt
            exit_detail = ""
            if nxt_row["Low"] <= stop_px and nxt_row["High"] >= tgt_px:
                hit = "target"
                exec_price = float(tgt_px)
                exit_detail = f"TakeProfit hit: High {float(nxt_row['High']):.2f} ≥ TP {tgt_px:.2f} (from entry {entry_px:.2f})"
            elif nxt_row["Low"] <= stop_px:
                hit = "stop"
                exec_price = float(stop_px)
                exit_detail = f"StopLoss hit: Low {float(nxt_row['Low']):.2f} ≤ SL {stop_px:.2f} (from entry {entry_px:.2f})"
            elif nxt_row["High"] >= tgt_px:
                hit = "target"
                exec_price = float(tgt_px)
                exit_detail = f"TakeProfit hit: High {float(nxt_row['High']):.2f} ≥ TP {tgt_px:.2f} (from entry {entry_px:.2f})"
            elif exit_ind.loc[dt]:
                hit = "indicator_exit"
                exec_price = float(nxt_row["Open"])
                exit_detail = f"Trend invalidation: EMA{cfg.ema_fast}<EMA{cfg.ema_slow} at close {float(row['Close']):.2f}"

            if hit is not None:
                trades.append({
                    "ticker": ticker, "side": "SELL", "date": exec_date,
                    "price": float(exec_price), "shares": 0,
                    "reason": hit,                  # 'stop' | 'target' | 'indicator_exit'
                    "signal_reason": exit_detail,   # human-readable explanation
                    "score": np.nan,
                    "pattern": "",                  # not applicable on exit
                    "rsi": float(row["rsi"]), "adx": float(row["adx"]), "+di": float(row["+di"]), "-di": float(row["-di"]),
                    "macd_line": float(row["macd_line"]), "macd_signal": float(row["macd_signal"]),
                    "ema_fast": float(row["ema_fast"]), "ema_slow": float(row["ema_slow"]), "ema_htf": float(row["ema_htf"]),
                    "sma50": float(row["sma50"]), "bb_mid": float(row["bb_mid"]),
                    "close": float(row["Close"]), "high_52w": float(row["high_52w"])
                })
                in_pos = False
                entry_px = stop_px = tgt_px = 0.0

    if in_pos:
        last_dt = d.index[-1]; row = d.loc[last_dt]
        trades.append({
            "ticker": ticker, "side": "SELL", "date": last_dt,
            "price": float(row["Close"]), "shares": 0,
            "reason": "final_close",
            "signal_reason": "Final close: end of data",
            "score": np.nan,
            "pattern": "",
            "rsi": float(row["rsi"]), "adx": float(row["adx"]), "+di": float(row["+di"]), "-di": float(row["-di"]),
            "macd_line": float(row["macd_line"]), "macd_signal": float(row["macd_signal"]),
            "ema_fast": float(row["ema_fast"]), "ema_slow": float(row["ema_slow"]), "ema_htf": float(row["ema_htf"]),
            "sma50": float(row["sma50"]), "bb_mid": float(row["bb_mid"]),
            "close": float(row["Close"]), "high_52w": float(row["high_52w"])
        })

    return pd.DataFrame(trades, columns=cols), pd.Series(dtype=float)


# =========================
# Ranking, portfolio, metrics (unchanged core)
# =========================
def pick_benchmark(benchmarks: Tuple[str,...], start: str, end: Optional[str], cache_dir: str) -> Tuple[str, pd.DataFrame]:
    for t in benchmarks:
        data = fetch_prices([t], start, end, cache_dir)
        df = data.get(t)
        if df is not None and not df.empty:
            log.info("Using benchmark: %s", t)
            return t, df
    idx = pd.date_range(start=start, end=end or today_str(), freq="B")
    df = pd.DataFrame({"Close": np.ones(len(idx))}, index=idx)
    log.warning("No benchmark found; using synthetic flat series.")
    return "SYNTH_BENCH", df

def compute_volar_scores(end_dt: pd.Timestamp, tickers: List[str], data_map: Dict[str,pd.DataFrame], bench_df: pd.DataFrame, lookback: int) -> Dict[str, float]:
    scores = {}
    bser = bench_df["Close"].loc[:end_dt].pct_change().dropna().iloc[-lookback:]
    for t in tickers:
        df = data_map.get(t)
        if df is None or df.empty:
            scores[t] = 0.0; continue
        if end_dt not in df.index:
            df = df[df.index <= end_dt]
            if df.empty:
                scores[t] = 0.0; continue
        r = df["Close"].loc[:end_dt].pct_change().dropna().iloc[-lookback:]
        common = pd.concat([r, bser], axis=1, keys=["s","b"]).dropna()
        if common.shape[0] < max(20, int(0.4*lookback)):
            scores[t] = 0.0; continue
        excess = common["s"] - common["b"]
        vol = common["s"].std(ddof=0)
        scores[t] = 0.0 if vol <= 1e-8 else float((excess.mean() / vol) * math.sqrt(252.0))
    return scores

def markowitz_long_only(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    n = len(mu); eps = 1e-6
    Sigma = Sigma + eps*np.eye(n)

    def solve_lambda(lmbd: float, active_mask=None):
        if active_mask is None:
            A = np.block([[2*lmbd*Sigma, np.ones((n,1))],[np.ones((1,n)), np.zeros((1,1))]])
            b = np.concatenate([mu, np.array([1.0])])
            try: w = np.linalg.solve(A, b)[:n]
            except np.linalg.LinAlgError: w = np.full(n, 1.0/n)
            return w
        else:
            idx = np.where(active_mask)[0]
            if len(idx)==0: return np.full(n, 1.0/n)
            S = Sigma[np.ix_(idx, idx)]
            o = np.ones(len(idx)); m = mu[idx]
            A = np.block([[2*lmbd*S, o[:,None]],[o[None,:], np.zeros((1,1))]])
            b = np.concatenate([m, np.array([1.0])])
            try: w_sub = np.linalg.solve(A, b)[:len(idx)]
            except np.linalg.LinAlgError: w_sub = np.full(len(idx), 1.0/len(idx))
            w = np.zeros(n); w[idx] = w_sub; return w

    best_w = np.full(n, 1.0/n); best_sr = -1e9
    for lmbd in np.logspace(-3, 3, 31):
        active = np.ones(n, dtype=bool); w = None
        for _ in range(n):
            w = solve_lambda(lmbd, active_mask=active)
            if not (w < 0).any(): break
            active[np.argmin(w)] = False
        if w is None: continue
        w = np.clip(w, 0, None); 
        if w.sum() <= 0: continue
        w = w / w.sum()
        mu_p = float(mu @ w); vol_p = float(np.sqrt(w @ Sigma @ w))
        if vol_p <= 1e-8: continue
        sr = mu_p / vol_p
        if sr > best_sr: best_sr, best_w = sr, w.copy()
    return best_w

def aggregate_and_apply(all_trades: pd.DataFrame, data_map: Dict[str, pd.DataFrame], bench_df: pd.DataFrame, cfg: Config):
    if all_trades.empty:
        return all_trades, pd.Series(dtype=float), {}

    side_order = {"BUY": 0, "SELL": 1}
    all_trades = (all_trades
        .assign(_sorder=all_trades["side"].map(side_order))
        .sort_values(by=["date", "_sorder"], kind="stable")
        .drop(columns=["_sorder"])
        .reset_index(drop=True)
    )
    all_trades["date"] = pd.to_datetime(all_trades["date"])

    equity_curve = []
    dates = sorted(all_trades["date"].unique().tolist())
    cash = cfg.initial_capital
    open_positions = {}
    completed_legs = []

    global APPLY_FEES
    APPLY_FEES = cfg.apply_fees

    def _get_close_on(tkr, dt):
        df = data_map.get(tkr)
        if df is None or df.empty:
            return np.nan
        if dt in df.index:
            return float(df.loc[dt, "Close"])
        prev = df[df.index <= dt]
        if prev.empty:
            return np.nan
        return float(prev["Close"].iloc[-1])

    if dates:
        seed_date = pd.to_datetime(dates[0]) - pd.Timedelta(days=1)
        equity_curve.append((seed_date, float(cash)))

    for dt in dates:
        day_trades = all_trades[all_trades["date"] == dt].copy()

        # ---- SELL first (with readable exit reason) ----
        for _, tr in day_trades[day_trades["side"] == "SELL"].iterrows():
            tkr = tr["ticker"]
            price = float(tr["price"])
            pos = open_positions.get(tkr)
            if pos is None:
                continue
            shares = int(pos["shares"])
            turnover_sell = shares * price
            fee = calc_fees(0.0, turnover_sell)
            pnl = (price - pos["entry_px"]) * shares
            cash += (turnover_sell - fee)
            realized = pnl - fee - pos.get("buy_fee", 0.0)

            # Make the exit reason human-readable
            base = str(tr.get("reason", "")).lower()
            detail = str(tr.get("signal_reason", "")).strip()
            if base == "stop":
                exit_text = f"Exit: StopLoss hit — {detail}" if detail else "Exit: StopLoss hit"
            elif base == "target":
                exit_text = f"Exit: TakeProfit hit — {detail}" if detail else "Exit: TakeProfit hit"
            elif base == "indicator_exit":
                exit_text = f"Exit: Trend invalidation — {detail}" if detail else "Exit: Trend invalidation (EMA fast<slow)"
            elif base == "final_close":
                exit_text = "Exit: Final close — end of data"
            else:
                exit_text = f"Exit: {tr.get('reason','')}"

            completed_legs.append({
                "ticker": tkr, "side": "SELL", "date": dt, "price": price,
                "shares": shares, "reason": exit_text,
                "turnover": turnover_sell, "fees_inr": fee, "pnl_inr": realized,
                "rsi": tr.get("rsi", np.nan), "adx": tr.get("adx", np.nan),
                "+di": tr.get("+di", np.nan), "-di": tr.get("-di", np.nan),
                "macd_line": tr.get("macd_line", np.nan), "macd_signal": tr.get("macd_signal", np.nan),
                "ema_fast": tr.get("ema_fast", np.nan), "ema_slow": tr.get("ema_slow", np.nan), "ema_htf": tr.get("ema_htf", np.nan),
                "sma50": tr.get("sma50", np.nan), "bb_mid": tr.get("bb_mid", np.nan),
                "close": tr.get("close", np.nan), "high_52w": tr.get("high_52w", np.nan),
                "volar": tr.get("volar", np.nan), "mvo_weight": np.nan, "alloc_inr": np.nan
            })
            log.info("Exit %-12s px=%8.2f shares=%6d :: %s | net=%.2f cash=%.2f",
                     tkr, price, shares, exit_text, realized, cash)
            del open_positions[tkr]

        # ---- BUY candidates today ----
        buys_today = day_trades[day_trades["side"] == "BUY"].copy()
        # 52w filter
        if not buys_today.empty:
            keep = []
            for _, rr in buys_today.iterrows():
                df = data_map.get(rr["ticker"])
                if df is None or df.empty or dt not in df.index:
                    continue
                close = float(df.loc[dt, "Close"])
                hist = df["Close"].loc[:dt]
                window = hist.iloc[-cfg.filter_52w_window:] if len(hist)>=cfg.filter_52w_window else hist
                high_52w = float(window.max())
                if high_52w>0 and close >= cfg.within_pct_of_52w_high * high_52w:
                    keep.append(rr)
            buys_today = pd.DataFrame(keep) if keep else pd.DataFrame(columns=buys_today.columns)

        # Exclude already-held tickers
        if not buys_today.empty:
            buys_today = buys_today[~buys_today["ticker"].isin(open_positions.keys())]

        # VOLAR ranking
        if not buys_today.empty:
            tickers = buys_today["ticker"].tolist()
            volar_scores = compute_volar_scores(dt, tickers, data_map, bench_df, cfg.volar_lookback)
            buys_today["volar"] = buys_today["ticker"].map(volar_scores)
            buys_today = buys_today.sort_values("volar", ascending=False).reset_index(drop=True)

        slots = cfg.max_concurrent_positions - len(open_positions)
        selected = pd.DataFrame(columns=buys_today.columns)
        if slots > 0 and not buys_today.empty:
            selected = buys_today.head(min(cfg.top_k_daily, slots)).copy()

        if not selected.empty:
            log.info("Selected %d BUY candidates on %s:", selected.shape[0], dt.date())
            for i, rr in selected.reset_index(drop=True).iterrows():
                log.info("  %-12s volar=%6.2f rank=%d px=%8.2f", rr["ticker"], rr.get("volar",0.0), i+1, rr["price"])

            # MVO sizing
            names = selected["ticker"].tolist()
            rets = []
            for t in names:
                df = data_map.get(t)
                ser = df["Close"].loc[:dt].pct_change().dropna().iloc[-cfg.volar_lookback:]
                rets.append(ser)
            R = pd.concat(rets, axis=1); R.columns = names; R = R.dropna()
            if R.empty or R.shape[0] < max(20, int(0.4*cfg.volar_lookback)) or R.shape[1] == 0:
                weights = np.full(len(names), 1.0/len(names))
            else:
                mu = R.mean().values; Sigma = R.cov().values
                weights = markowitz_long_only(mu, Sigma)

            deploy_cash = max(0.0, float(cash)) * float(cfg.deploy_cash_frac)
            if deploy_cash <= 0:
                log.info("No deployable cash (cap=%.0f%%) on %s", 100*cfg.deploy_cash_frac, dt.date())
            else:
                alloc = (weights / weights.sum()) * deploy_cash if weights.sum()>0 else np.full(len(names), deploy_cash/len(names))
                rank_map = {row["ticker"]: (idx+1) for idx, (_, row) in enumerate(selected.iterrows())}
                for w_amt, t in zip(alloc, names):
                    df_t = data_map[t]
                    price = float(df_t.loc[dt, "Close"] if dt in df_t.index else df_t["Close"].loc[:dt].iloc[-1])
                    shares = int(math.floor(w_amt / price))
                    if shares <= 0:
                        log.info("Skip BUY %-12s (alloc %.2f too small)", t, w_amt); continue
                    turn = shares * price
                    fee = calc_fees(turn, 0.0)
                    total_cost = turn + fee
                    if total_cost > cash:
                        shares = int(math.floor((cash - fee) / price))
                        if shares <= 0:
                            log.info("Skip BUY %-12s due to cash/fees", t); continue
                        turn = shares * price; total_cost = turn + fee
                    cash -= total_cost
                    open_positions[t] = {"entry_date": dt, "entry_px": price, "shares": shares, "buy_fee": fee, "entry_reason": "entry"}

                    row_sel = selected[selected["ticker"]==t].iloc[0]
                    patt_str = str(row_sel.get("pattern","BullishPattern"))
                    sigs = str(row_sel.get("signal_reason","")).strip()
                    sigs_fmt = f"[{sigs}]" if sigs else "[none]"
                    volar_val = float(row_sel.get("volar", np.nan))
                    rank_pos = rank_map.get(t, np.nan)
                    high_52w = float(row_sel.get("high_52w", np.nan))
                    close_val = float(row_sel.get("close", np.nan))
                    pct_52w = (close_val / high_52w) if (high_52w and high_52w>0) else np.nan
                    mvo_weight_today = (w_amt / deploy_cash) if deploy_cash > 0 else 0.0

                    # *** READABLE ENTRY REASON ***
                    reason_text = (
                        f"Entry: {patt_str} + EMA{cfg.ema_fast}>EMA{cfg.ema_slow} + Close>EMA{cfg.ema_htf}; "
                        f"Conf={sigs_fmt}; 52w%={pct_52w:.1%} (≥ {cfg.within_pct_of_52w_high:.0%}); "
                        f"VOLAR rank {int(rank_pos)}/{len(names)} (VOLAR={volar_val:.2f}); "
                        f"MVO weight={mvo_weight_today:.1%} of capped cash"
                    )

                    completed_legs.append({
                        "ticker": t, "side": "BUY", "date": dt, "price": price,
                        "shares": shares, "reason": reason_text,
                        "turnover": turn, "fees_inr": fee, "pnl_inr": 0.0,
                        "rsi": float(row_sel.get("rsi", np.nan)), "adx": float(row_sel.get("adx", np.nan)),
                        "+di": float(row_sel.get("+di", np.nan)), "-di": float(row_sel.get("-di", np.nan)),
                        "macd_line": float(row_sel.get("macd_line", np.nan)), "macd_signal": float(row_sel.get("macd_signal", np.nan)),
                        "ema_fast": float(row_sel.get("ema_fast", np.nan)), "ema_slow": float(row_sel.get("ema_slow", np.nan)), "ema_htf": float(row_sel.get("ema_htf", np.nan)),
                        "sma50": float(row_sel.get("sma50", np.nan)), "bb_mid": float(row_sel.get("bb_mid", np.nan)),
                        "close": close_val, "high_52w": high_52w,
                        "volar": volar_val, "mvo_weight": float(mvo_weight_today), "alloc_inr": float(w_amt)
                    })
                    log.info("BUY %-12s px=%8.2f sh=%6d fee=%.2f cash=%.2f :: %s",
                             t, price, shares, fee, cash, reason_text)

        # MTM valuation
        mtm = 0.0
        for _tkr, pos in open_positions.items():
            px = _get_close_on(_tkr, dt)
            if not np.isnan(px):
                mtm += pos["shares"] * px
        total_equity = cash + mtm
        equity_curve.append((dt, float(total_equity)))

    eq_ser = pd.Series([e for _, e in equity_curve], index=[d for d, _ in equity_curve])
    legs_df = pd.DataFrame(completed_legs).sort_values(["date", "ticker", "side"]).reset_index(drop=True)

    # Roundtrips (these already carry the improved BUY/SELL "reason")
    roundtrips = []
    by_tkr_open = {}
    for _, leg in legs_df.iterrows():
        tkr = leg["ticker"]
        if leg["side"] == "BUY":
            by_tkr_open[tkr] = leg
        else:
            buy = by_tkr_open.pop(tkr, None)
            if buy is None:
                continue
            fees_total = float(buy.get("fees_inr", 0.0) + leg.get("fees_inr", 0.0))
            gross_pnl = (leg["price"] - buy["price"]) * buy["shares"]
            net_pnl   = gross_pnl - fees_total
            ret_pct   = (leg["price"] / buy["price"] - 1.0) * 100.0
            days_held = (pd.to_datetime(leg["date"]) - pd.to_datetime(buy["date"])).days
            roundtrips.append({
                "ticker": tkr,
                "entry_date": pd.to_datetime(buy["date"]),
                "entry_price": float(buy["price"]),
                "exit_date": pd.to_datetime(leg["date"]),
                "exit_price": float(leg["price"]),
                "days_held": int(days_held),
                "shares": int(buy["shares"]),
                "entry_reason": buy.get("reason",""),
                "exit_reason": leg.get("reason",""),
                "gross_pnl_inr": float(gross_pnl),
                "fees_total_inr": float(fees_total),
                "net_pnl_inr": float(net_pnl),
                "return_pct": float(ret_pct),
                # carry some entry context
                "rsi_entry": float(buy.get("rsi", np.nan)),
                "adx_entry": float(buy.get("adx", np.nan)),
                "ema_fast_entry": float(buy.get("ema_fast", np.nan)),
                "ema_slow_entry": float(buy.get("ema_slow", np.nan)),
                "ema_htf_entry": float(buy.get("ema_htf", np.nan)),
                "sma50_entry": float(buy.get("sma50", np.nan)),
                "bb_mid_entry": float(buy.get("bb_mid", np.nan)),
                "volar_entry": float(buy.get("volar", np.nan)),
                "mvo_weight_entry": float(buy.get("mvo_weight", np.nan)),
                "alloc_inr_entry": float(buy.get("alloc_inr", np.nan))
            })
    trips_df = pd.DataFrame(roundtrips).sort_values(["entry_date","ticker"]).reset_index(drop=True)

    metrics = compute_metrics(eq_ser, legs_df)
    return legs_df, trips_df, eq_ser, metrics


def compute_metrics(equity: pd.Series, legs_df: pd.DataFrame):
    out = {}
    if equity is None or equity.empty: return out
    eq = equity.dropna()
    daily_ret = eq.pct_change().fillna(0.0)

    days = (eq.index[-1] - eq.index[0]).days or 1
    years = days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1 if years > 0 else 0.0

    sharpe = (daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(252)) if daily_ret.std(ddof=0) > 0 else 0.0

    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    max_dd = dd.min()

    wins = 0
    n_sells = legs_df[legs_df["side"] == "SELL"].shape[0] if legs_df is not None and not legs_df.empty else 0
    for _, r in legs_df[legs_df["side"] == "SELL"].iterrows():
        if float(r.get("pnl_inr", 0.0)) > 0: wins += 1
    win_rate = (wins / n_sells) * 100.0 if n_sells > 0 else 0.0

    out.update({
        "start_equity_inr": float(eq.iloc[0]),
        "final_equity_inr": float(eq.iloc[-1]),
        "cagr_pct": float(cagr * 100),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100),
        "win_rate_pct": float(win_rate),
        "n_trades": int(n_sells),
    })
    return out

def plot_equity(equity: pd.Series, out_path: str):
    if equity is None or equity.empty: return
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(equity.index, equity.values)
        plt.title("Equity Curve")
        plt.xlabel("Date"); plt.ylabel("Equity (INR)")
        plt.tight_layout(); plt.savefig(out_path); plt.close()
    except Exception:
        pass

def backtest(cfg: Config):
    ensure_dirs(cfg.cache_dir, cfg.out_dir)
    log.info("Universe: loading static symbols...")
    symbols = load_static_symbols(cfg.static_symbols, cfg.static_symbols_path)
    log.info("Loaded %d symbols.", len(symbols))

    log.info("Data: fetching OHLCV from yfinance (adjusted)...")
    data_map = fetch_prices(symbols, cfg.start_date, cfg.end_date, cfg.cache_dir)
    log.info("Downloaded %d symbols with data.", len(data_map))

    bench_tkr, bench_df = pick_benchmark(cfg.benchmark_try, cfg.start_date, cfg.end_date, cfg.cache_dir)
    log.info("Benchmark selected: %s", bench_tkr)

    log.info("Signals: scanning bullish candlesticks + confirmations + trend filters...")
    all_trades = []
    for i, tkr in enumerate(symbols, 1):
        df = data_map.get(tkr)
        if df is None or df.empty: continue
        tr, _ = simulate_ticker(tkr, df, cfg)
        if not tr.empty: all_trades.append(tr)
        if i % 50 == 0:
            log.info("  processed %d/%d tickers...", i, len(symbols))

    if not all_trades:
        log.warning("No signals generated; check your thresholds or timeframe.")
        return None, None, None, {}
    all_trades = pd.concat(all_trades, ignore_index=True)

    log.info("Portfolio: cap daily deploy to %.0f%% cash; 52w>=%.0f%% high; top-%d by VOLAᵣ; MVO; max %d positions.",
             cfg.deploy_cash_frac*100, cfg.within_pct_of_52w_high*100, cfg.top_k_daily, cfg.max_concurrent_positions)
    legs_df, trips_df, equity, metrics = aggregate_and_apply(all_trades, data_map, bench_df, cfg)

    stamp = pd.Timestamp.today(tz="Asia/Kolkata").strftime("%Y%m%d_%H%M%S")
    legs_path = os.path.join(cfg.out_dir, f"trades_legs_{stamp}.csv")
    trips_path = os.path.join(cfg.out_dir, f"trades_roundtrips_{stamp}.csv")
    equity_path = os.path.join(cfg.out_dir, f"equity_{stamp}.csv")
    metrics_path = os.path.join(cfg.out_dir, f"metrics_{stamp}.json")
    eq_plot_path = os.path.join(cfg.out_dir, f"equity_{stamp}.png")

    if legs_df is not None: legs_df.to_csv(legs_path, index=False)
    if trips_df is not None: trips_df.to_csv(trips_path, index=False)
    if equity is not None: pd.DataFrame({"date": equity.index, "equity": equity.values}).to_csv(equity_path, index=False)
    with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)
    if cfg.plot and equity is not None: plot_equity(equity, eq_plot_path)

    log.info("=== METRICS ===\n%s", json.dumps(metrics, indent=2))
    log.info("Files written:\n  %s\n  %s\n  %s\n  %s", legs_path, trips_path, equity_path, metrics_path)
    if cfg.plot: log.info("  %s", eq_plot_path)

def main():
    global APPLY_FEES
    APPLY_FEES = bool(CFG.apply_fees)

    # Example universe (yours). You can switch to a file via CFG.static_symbols_path.
    CFG.static_symbols = ['360ONE.NS','3MINDIA.NS','AARTIIND.NS','ABB.NS','ACC.NS','ADANIENT.NS','ADANIPORTS.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AXISBANK.NS','BAJAJ-AUTO.NS','BAJFINANCE.NS','BHARTIARTL.NS','BPCL.NS','BRITANNIA.NS','CIPLA.NS','COALINDIA.NS','COFORGE.NS','DRREDDY.NS','EICHERMOT.NS','GRASIM.NS','HCLTECH.NS','HDFCBANK.NS','HINDALCO.NS','HINDUNILVR.NS','ICICIBANK.NS','INFY.NS','ITC.NS','JSWSTEEL.NS','KOTAKBANK.NS','LT.NS','MARUTI.NS','NESTLEIND.NS','NTPC.NS','ONGC.NS','POWERGRID.NS','RELIANCE.NS','SBIN.NS','SUNPHARMA.NS','TATAMOTORS.NS','TATASTEEL.NS','TCS.NS','TECHM.NS','TITAN.NS','ULTRACEMCO.NS','WIPRO.NS']
    # Or: CFG.static_symbols_path = "nifty500.txt"

    backtest(CFG)

if __name__ == "__main__":
    main()
