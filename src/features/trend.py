"""
trend.py — Maximal Trend & Momentum Feature Module
Institutional-Grade Research Version

This module implements:
- Rolling utility functions
- Manual implementations of SMA, EMA
- Framework for WMA, HMA, TEMA, KAMA (implemented in later chunks)
- Parameter sweeps across multiple windows
- Foundational trend feature wrappers

Chunk 1 Contents:
1. Rolling utilities
2. SMA (Simple Moving Average)
3. EMA (Exponential Moving Average)
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. ROLLING UTILITIES
# ============================================================

def rolling_window(a: np.ndarray, window: int):
    """
    Efficient rolling window view of a 1D numpy array.
    Equivalent to pandas.Series.rolling(...).values but faster for bulk ops.
    """
    if window < 1:
        raise ValueError("Window must be >= 1")
    if window > a.shape[0]:
        return np.array([])

    shape = (a.shape[0] - window + 1, window)
    strides = (a.strides[0], a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_mean(a: np.ndarray, window: int):
    """
    Fast rolling mean using rolling_window.
    """
    if window > len(a):
        return np.full_like(a, np.nan)
    rw = rolling_window(a, window)
    means = rw.mean(axis=1)
    return np.concatenate([np.full(window - 1, np.nan), means])


def rolling_std(a: np.ndarray, window: int):
    """
    Fast rolling standard deviation.
    """
    if window > len(a):
        return np.full_like(a, np.nan)
    rw = rolling_window(a, window)
    stds = rw.std(axis=1)
    return np.concatenate([np.full(window - 1, np.nan), stds])


# ============================================================
# 2. SIMPLE MOVING AVERAGE (SMA)
# ============================================================

def sma(arr: np.ndarray, window: int):
    """
    Manual SMA implementation for maximal transparency.
    """
    if window <= 0:
        raise ValueError("SMA window must be > 0")
    return rolling_mean(arr, window)


def add_sma_features(df: pd.DataFrame, windows):
    """
    Adds SMA features for a list of windows.
    """
    close = df["Close"].values
    for w in windows:
        df[f"SMA_{w}"] = sma(close, w)
    return df


# ============================================================
# 3. EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================================

def ema(arr: np.ndarray, window: int):
    """
    Manual EMA using recursive smoothing:
        EMA_t = EMA_{t-1} + alpha * (price_t - EMA_{t-1})
    alpha = 2 / (window + 1)
    """
    if window <= 0:
        raise ValueError("EMA window must be > 0")

    alpha = 2 / (window + 1)
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]

    for t in range(1, len(arr)):
        out[t] = out[t - 1] + alpha * (arr[t] - out[t - 1])
    
    return out


def add_ema_features(df: pd.DataFrame, windows):
    """
    Adds EMA features for a list of windows.
    """
    close = df["Close"].values
    for w in windows:
        df[f"EMA_{w}"] = ema(close, w)
    return df


# ============================================================
# 4. Trend Feature Aggregator (PARTIAL — completed in chunk 6)
# ============================================================

def add_trend_features(df: pd.DataFrame):
    """
    Dispatcher for trend features.
    In this chunk, only SMA/EMA windows are included.
    Additional trend features added in later chunks.
    """

    MA_WINDOWS = [3, 5, 8, 10, 12, 20, 21, 34, 50, 89, 100, 144, 200]

    df = add_sma_features(df, MA_WINDOWS)
    df = add_ema_features(df, MA_WINDOWS)

    # Remaining features added across future chunks:
    # - WMA / HMA / TEMA / KAMA
    # - Oscillators (RSI, Stochastics, CMO, ROC, TRIX, etc.)
    # - Trend slopes & curvature
    # - Trend diagnostics

    return df
# ============================================================
# 4. WEIGHTED MOVING AVERAGE (WMA)
# ============================================================

def wma(arr: np.ndarray, window: int):
    """
    Weighted Moving Average (manual implementation)
    Weights: 1, 2, ..., window (linear weights)
    """
    if window <= 0:
        raise ValueError("WMA window must be > 0")

    out = np.full_like(arr, np.nan, dtype=float)
    weights = np.arange(1, window + 1)

    for t in range(window - 1, len(arr)):
        segment = arr[t - window + 1 : t + 1]
        out[t] = np.dot(segment, weights) / weights.sum()

    return out


def add_wma_features(df: pd.DataFrame, windows):
    """
    Add WMA features for multiple windows.
    """
    close = df["Close"].values
    for w in windows:
        df[f"WMA_{w}"] = wma(close, w)
    return df


# ============================================================
# 5. HULL MOVING AVERAGE (HMA)
# ============================================================

def hma(arr: np.ndarray, window: int):
    """
    Hull Moving Average (manual implementation)
    
    Steps:
    1) WMA1 = WMA(arr, window / 2)
    2) WMA2 = WMA(arr, window)
    3) raw = 2*WMA1 - WMA2
    4) HMA = WMA(raw, sqrt(window))
    """
    if window <= 0:
        raise ValueError("HMA window must be > 0")

    w1 = int(window / 2)
    w2 = int(np.sqrt(window))

    wma_half = wma(arr, w1)
    wma_full = wma(arr, window)
    raw = 2 * wma_half - wma_full

    return wma(raw, w2)


def add_hma_features(df: pd.DataFrame, windows):
    """
    Add HMA features.
    """
    close = df["Close"].values
    for w in windows:
        df[f"HMA_{w}"] = hma(close, w)
    return df


# ============================================================
# 6. TRIPLE EXPONENTIAL MOVING AVERAGE (TEMA)
# ============================================================

def tema(arr: np.ndarray, window: int):
    """
    Manual TEMA computation:
        TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where:
        EMA1 = EMA(arr, window)
        EMA2 = EMA(EMA1, window)
        EMA3 = EMA(EMA2, window)
    """
    ema1 = ema(arr, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    return 3 * ema1 - 3 * ema2 + ema3


def add_tema_features(df: pd.DataFrame, windows):
    """
    Add TEMA feature set.
    """
    close = df["Close"].values
    for w in windows:
        df[f"TEMA_{w}"] = tema(close, w)
    return df


# ============================================================
# 7. KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
# ============================================================

def kama(arr: np.ndarray, window: int, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (manual implementation)

    ER = Efficiency Ratio
        ER = |price_t - price_{t-window}| / sum(|price_i - price_{i-1}|)

    Smoothing constant SC:
        fastSC = (2/(fast+1))^2
        slowSC = (2/(slow+1))^2
        SC = ER*(fastSC - slowSC) + slowSC
    """
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)

    fastSC = (2 / (fast + 1)) ** 2
    slowSC = (2 / (slow + 1)) ** 2

    for t in range(window, n):
        change = abs(arr[t] - arr[t - window])

        volatility = np.sum(np.abs(arr[t - window + 1 : t + 1] - arr[t - window : t]))
        ER = change / volatility if volatility != 0 else 0

        SC = ER * (fastSC - slowSC) + slowSC

        if np.isnan(out[t - 1]):
            out[t] = arr[t]
        else:
            out[t] = out[t - 1] + SC * (arr[t] - out[t - 1])

    return out


def add_kama_features(df: pd.DataFrame, windows):
    """
    Add KAMA features across multiple windows.
    """
    close = df["Close"].values
    for w in windows:
        df[f"KAMA_{w}"] = kama(close, w)
    return df


# ============================================================
# 8. Extend Trend Dispatcher
# ============================================================

def add_trend_features(df: pd.DataFrame):
    """
    Updated dispatcher including:
    - SMA
    - EMA
    - WMA
    - HMA
    - TEMA
    - KAMA

    Oscillators, slopes, diagnostics added in future chunks.
    """

    MA_WINDOWS = [3, 5, 8, 10, 12, 20, 21, 34, 50, 89, 100, 144, 200]

    df = add_sma_features(df, MA_WINDOWS)
    df = add_ema_features(df, MA_WINDOWS)
    df = add_wma_features(df, MA_WINDOWS)
    df = add_hma_features(df, MA_WINDOWS)
    df = add_tema_features(df, MA_WINDOWS)
    df = add_kama_features(df, MA_WINDOWS)

    return df


# ============================================================
# 9. MOMENTUM INDICATORS & OSCILLATORS
# ============================================================

# ------------------------------------------------------------
# RSI (Relative Strength Index)
# ------------------------------------------------------------

def rsi(arr: np.ndarray, window: int = 14):
    """
    Manual RSI implementation:
    RS = avg_gain / avg_loss
    RSI = 100 - 100 / (1 + RS)
    """
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(arr)
    avg_loss = np.zeros_like(arr)

    # Seed initial averages
    avg_gain[window] = gains[:window].mean()
    avg_loss[window] = losses[:window].mean()

    # Recursive computation
    for t in range(window + 1, len(arr)):
        avg_gain[t] = (avg_gain[t - 1] * (window - 1) + gains[t - 1]) / window
        avg_loss[t] = (avg_loss[t - 1] * (window - 1) + losses[t - 1]) / window

    rs = avg_gain / (avg_loss + 1e-10)
    rsi_vals = 100 - (100 / (1 + rs))
    rsi_vals[: window] = np.nan
    return rsi_vals


def add_rsi_features(df: pd.DataFrame, windows=[7, 14, 21]):
    arr = df["Close"].values
    for w in windows:
        df[f"RSI_{w}"] = rsi(arr, w)
    return df


# ------------------------------------------------------------
# Stochastic Oscillator
# ------------------------------------------------------------

def stochastic(df: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    lowest_low = df["Low"].rolling(window).min()
    highest_high = df["High"].rolling(window).max()
    percent_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low + 1e-10)
    percent_d = percent_k.rolling(smooth_d).mean()
    return percent_k, percent_d


def add_stochastic_features(df: pd.DataFrame, windows=[14, 21]):
    for w in windows:
        k, d = stochastic(df, w)
        df[f"StochK_{w}"] = k
        df[f"StochD_{w}"] = d
    return df


# ------------------------------------------------------------
# Rate of Change (ROC)
# ------------------------------------------------------------

def roc(arr: np.ndarray, window: int):
    """
    ROC = (price_t / price_{t-window} - 1) * 100
    """
    out = (arr / np.concatenate([np.full(window, np.nan), arr[:-window]])) - 1
    return out * 100


def add_roc_features(df: pd.DataFrame, windows=[5, 10, 20]):
    arr = df["Close"].values
    for w in windows:
        df[f"ROC_{w}"] = roc(arr, w)
    return df


# ------------------------------------------------------------
# Williams %R
# ------------------------------------------------------------

def williams_r(df: pd.DataFrame, window: int = 14):
    highest_high = df["High"].rolling(window).max()
    lowest_low = df["Low"].rolling(window).min()
    return -100 * (highest_high - df["Close"]) / (highest_high - lowest_low + 1e-10)


def add_williams_features(df: pd.DataFrame, windows=[14, 28]):
    for w in windows:
        df[f"WilliamsR_{w}"] = williams_r(df, w)
    return df


# ------------------------------------------------------------
# CMO (Chande Momentum Oscillator)
# ------------------------------------------------------------

def cmo(arr: np.ndarray, window: int = 14):
    deltas = np.diff(arr)
    up = np.where(deltas > 0, deltas, 0).sum()
    down = np.where(deltas < 0, -deltas, 0).sum()
    return 100 * (up - down) / (up + down + 1e-10)


def add_cmo_features(df: pd.DataFrame, windows=[14, 21]):
    arr = df["Close"].values
    for w in windows:
        df[f"CMO_{w}"] = pd.Series(arr).diff().rolling(w).apply(
            lambda s: 100 * (s[s > 0].sum() - (-s[s < 0]).sum()) / 
                      (s[s > 0].sum() + (-s[s < 0]).sum() + 1e-10),
            raw=False
        )
    return df


# ------------------------------------------------------------
# TRIX (Triple Smoothed EMA Momentum)
# ------------------------------------------------------------

def trix(arr: np.ndarray, window: int = 15):
    ema1 = ema(arr, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    trix_line = (ema3 - np.concatenate([[np.nan], ema3[:-1]])) / (ema3 + 1e-10)
    return trix_line * 100


def add_trix_features(df: pd.DataFrame, windows=[15, 30]):
    arr = df["Close"].values
    for w in windows:
        df[f"TRIX_{w}"] = trix(arr, w)
    return df


# ------------------------------------------------------------
# Ultimate Oscillator
# ------------------------------------------------------------

def ultimate_oscillator(df: pd.DataFrame, short=7, mid=14, long=28):
    """Ultimate Oscillator (Larry Williams).

    Buying Pressure (BP) = Close - min(Low, PrevClose)
    True Range (TR)     = max(High, PrevClose) - min(Low, PrevClose)

    This implementation is written defensively to avoid accidental list/shift
    mistakes and to work on any DataFrame with the standard OHLC columns.
    """

    prev_close = df["Close"].shift(1)
    min_low_prev = pd.concat([df["Low"], prev_close], axis=1).min(axis=1)
    max_high_prev = pd.concat([df["High"], prev_close], axis=1).max(axis=1)

    bp = df["Close"] - min_low_prev
    tr = max_high_prev - min_low_prev

    avg7 = bp.rolling(short).sum() / (tr.rolling(short).sum() + 1e-10)
    avg14 = bp.rolling(mid).sum() / (tr.rolling(mid).sum() + 1e-10)
    avg28 = bp.rolling(long).sum() / (tr.rolling(long).sum() + 1e-10)

    return 100 * (4*avg7 + 2*avg14 + avg28) / 7


def add_ultimate_features(df: pd.DataFrame):
    df["Ultimate"] = ultimate_oscillator(df)
    return df


# ------------------------------------------------------------
# Awesome Oscillator
# ------------------------------------------------------------

def awesome_oscillator(df: pd.DataFrame):
    median = (df["High"] + df["Low"]) / 2
    sma5 = median.rolling(5).mean()
    sma34 = median.rolling(34).mean()
    return sma5 - sma34


def add_awesome_features(df: pd.DataFrame):
    df["AO"] = awesome_oscillator(df)
    return df


# ------------------------------------------------------------
# True Strength Index (TSI)
# ------------------------------------------------------------

def tsi(arr: np.ndarray, r=25, s=13):
    momentum = np.diff(arr)
    abs_momentum = np.abs(momentum)

    ema1 = ema(momentum, r)
    ema2 = ema(ema1, s)

    abs_ema1 = ema(abs_momentum, r)
    abs_ema2 = ema(abs_ema1, s)

    return 100 * (ema2 / (abs_ema2 + 1e-10))


def add_tsi_features(df: pd.DataFrame):
    arr = df["Close"].values
    tsi_vals = np.concatenate([[np.nan], tsi(arr)])
    df["TSI"] = tsi_vals
    return df


# ------------------------------------------------------------
# Momentum Aggregator (partial)
# ------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame):
    """
    Adds all oscillator and momentum indicators.
    """
    df = add_rsi_features(df)
    df = add_stochastic_features(df)
    df = add_roc_features(df)
    df = add_williams_features(df)
    df = add_cmo_features(df)
    df = add_trix_features(df)
    df = add_ultimate_features(df)
    df = add_awesome_features(df)
    df = add_tsi_features(df)

    return df

# ============================================================
# 10. TREND SLOPES, CURVATURE, & REGRESSION-BASED FEATURES
# ============================================================

from numpy.linalg import lstsq


# ------------------------------------------------------------
# Rolling Linear Regression Slope
# ------------------------------------------------------------

def rolling_regression_slope(arr: np.ndarray, window: int):
    """
    Computes slope of a rolling linear regression:
    
        price_t ~ a + b*t + error

    Where b = trend slope.
    """
    out = np.full(len(arr), np.nan)
    t_idx = np.arange(window)

    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1 : i + 1]
        x = t_idx - t_idx.mean()
        y_c = y - y.mean()

        # slope = sum(x*y) / sum(x^2)
        denom = np.sum(x * x)
        if denom == 0:
            continue
        b = np.sum(x * y_c) / denom
        out[i] = b

    return out


def add_regression_slope_features(df: pd.DataFrame, windows=[10, 20, 50, 100]):
    arr = df["Close"].values
    for w in windows:
        df[f"RegSlope_{w}"] = rolling_regression_slope(arr, w)
    return df


# ------------------------------------------------------------
# Rolling Regression R² — Trend Quality
# ------------------------------------------------------------

def rolling_regression_r2(arr: np.ndarray, window: int):
    """
    R² measures how well a linear trend explains price movement.
    A high R² = strong, clean trend.
    """
    out = np.full(len(arr), np.nan)
    t_idx = np.arange(window)

    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1 : i + 1]
        x = t_idx - t_idx.mean()
        y_c = y - y.mean()

        slope = np.sum(x * y_c) / (np.sum(x * x) + 1e-10)
        intercept = y.mean()

        y_hat = intercept + slope * x
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        r2 = 1 - ss_res / (ss_tot + 1e-10)
        out[i] = r2

    return out


def add_regression_r2_features(df: pd.DataFrame, windows=[20, 50, 100]):
    arr = df["Close"].values
    for w in windows:
        df[f"RegR2_{w}"] = rolling_regression_r2(arr, w)
    return df


# ------------------------------------------------------------
# Trend Curvature (2nd Derivative Approximation)
# ------------------------------------------------------------

def trend_curvature(arr: np.ndarray, window: int):
    """
    Approximate curvature using second derivative of rolling SMA or EMA.
    Measures turning behavior.
    """
    sma_vals = rolling_mean(arr, window)
    # Second difference
    curvature = np.concatenate([[np.nan], [np.nan], sma_vals[2:] - 2*sma_vals[1:-1] + sma_vals[:-2]])
    return curvature


def add_curvature_features(df: pd.DataFrame, windows=[10, 20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"Curv_{w}"] = trend_curvature(arr, w)
    return df


# ------------------------------------------------------------
# Trend Acceleration (Derivative of Trend Slope)
# ------------------------------------------------------------

def trend_acceleration(arr: np.ndarray, window: int):
    """
    First derivative of regression slope.
    Positive acceleration = increasing trend strength.
    """
    slope = rolling_regression_slope(arr, window)
    accel = np.concatenate([[np.nan], np.diff(slope)])
    return accel


def add_trend_acceleration_features(df: pd.DataFrame, windows=[10, 20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"Accel_{w}"] = trend_acceleration(arr, w)
    return df


# ------------------------------------------------------------
# Detrended Price Oscillator (DPO)
# ------------------------------------------------------------

def dpo(arr: np.ndarray, window: int):
    """
    DPO removes trend to highlight cycles:
        DPO = price - SMA(price, window shifted)
    """
    shift = int(window / 2) + 1
    sma_vals = rolling_mean(arr, window)
    shifted = np.concatenate([np.full(shift, np.nan), sma_vals[:-shift]])
    return arr - shifted


def add_dpo_features(df: pd.DataFrame, windows=[10, 20, 30]):
    arr = df["Close"].values
    for w in windows:
        df[f"DPO_{w}"] = dpo(arr, w)
    return df


# ------------------------------------------------------------
# Trend Persistence (Directional Consistency)
# ------------------------------------------------------------

def trend_persistence(arr: np.ndarray, window: int):
    """
    Counts how many of the last `window` returns had the same sign:
    e.g., 18/20 positive => persistent uptrend
    """
    rets = np.diff(arr)
    signs = np.where(rets > 0, 1, -1)
    out = np.full(len(arr), np.nan)

    for i in range(window, len(arr)):
        segment = signs[i - window : i]
        pos = np.sum(segment > 0)
        neg = np.sum(segment < 0)
        out[i] = (pos - neg) / window  # normalized persistence

    return out


def add_trend_persistence_features(df: pd.DataFrame, windows=[10, 20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"TrendPersist_{w}"] = trend_persistence(arr, w)
    return df


# ------------------------------------------------------------
# Noise-to-Signal Ratio (Trend Strength)
# ------------------------------------------------------------

def noise_to_signal_ratio(arr: np.ndarray, window: int):
    """
    Noise = absolute price changes
    Signal = absolute net change
    A high NSR means messy market with no clean trend.
    """
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        segment = arr[i - window : i]
        net_change = abs(segment[-1] - segment[0])
        noise = np.sum(np.abs(np.diff(segment)))
        out[i] = noise / (net_change + 1e-10)
    return out


def add_nsr_features(df: pd.DataFrame, windows=[10, 20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"NSR_{w}"] = noise_to_signal_ratio(arr, w)
    return df


# ------------------------------------------------------------
# Volatility-Adjusted Trend Slope
# ------------------------------------------------------------

def vol_adjusted_slope(df: pd.DataFrame, window: int):
    """
    Trend slope normalized by realized volatility.
    Helps compare slope strength across different vol regimes.
    """
    slope = rolling_regression_slope(df["Close"].values, window)
    vol = df["Close"].pct_change().rolling(window).std() * np.sqrt(252)
    return slope / (vol + 1e-10)


def add_vol_adj_slope_features(df: pd.DataFrame, windows=[20, 50, 100]):
    for w in windows:
        df[f"VASlope_{w}"] = vol_adjusted_slope(df, w)
    return df


# ------------------------------------------------------------
# Trend Strength Index (TSI2 — custom research metric)
# ------------------------------------------------------------

def trend_strength_index(arr: np.ndarray, slope_window: int, vol_window: int):
    slope = rolling_regression_slope(arr, slope_window)
    vols = np.std(np.diff(arr[-vol_window:])) if len(arr) > vol_window else np.nan
    return slope / (vols + 1e-10)


def add_trend_strength_features(df: pd.DataFrame):
    arr = df["Close"].values
    df["TSI2_20"] = trend_strength_index(arr, slope_window=20, vol_window=20)
    df["TSI2_50"] = trend_strength_index(arr, slope_window=50, vol_window=50)
    return df


# ------------------------------------------------------------
# Trend Aggregator (partial)
# ------------------------------------------------------------

def add_trend_geometry_features(df: pd.DataFrame):
    """
    Adds geometric/mechanical trend descriptors.
    """
    df = add_regression_slope_features(df)
    df = add_regression_r2_features(df)
    df = add_curvature_features(df)
    df = add_trend_acceleration_features(df)
    df = add_dpo_features(df)
    df = add_trend_persistence_features(df)
    df = add_nsr_features(df)
    df = add_vol_adj_slope_features(df)
    df = add_trend_strength_features(df)
    
    return df

# ============================================================
# 11. TREND DIAGNOSTICS, REVERSALS, COMPRESSION, EXHAUSTION
# ============================================================


# ------------------------------------------------------------
# Trend Reversal: Local Maxima / Minima
# ------------------------------------------------------------

def local_extrema(arr: np.ndarray, window: int = 5):
    """
    Local maxima = arr[t] is the highest point in the window around t.
    Local minima = arr[t] is the lowest point in the window around t.
    Used for trend reversal detection.
    """
    out_max = np.full(len(arr), 0)
    out_min = np.full(len(arr), 0)

    for i in range(window, len(arr) - window):
        segment = arr[i - window : i + window + 1]
        if arr[i] == np.max(segment):
            out_max[i] = 1
        if arr[i] == np.min(segment):
            out_min[i] = 1

    return out_max, out_min


def add_extrema_features(df: pd.DataFrame, windows=[3, 5, 10]):
    arr = df["Close"].values
    for w in windows:
        maxes, mins = local_extrema(arr, w)
        df[f"LocalMax_{w}"] = maxes
        df[f"LocalMin_{w}"] = mins
    return df


# ------------------------------------------------------------
# Trend Flip Detector (Slope-Based)
# ------------------------------------------------------------

def slope_flip_detector(arr: np.ndarray, window: int = 20):
    """
    Detects when regression slope changes from + to - or vice versa.
    +1 = flipped upward
    -1 = flipped downward
    """
    slope = rolling_regression_slope(arr, window)
    flip = np.full(len(arr), 0)
    for t in range(1, len(arr)):
        if slope[t - 1] < 0 and slope[t] > 0:
            flip[t] = 1
        elif slope[t - 1] > 0 and slope[t] < 0:
            flip[t] = -1
    return flip


def add_slope_flip_features(df: pd.DataFrame, windows=[10, 20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"SlopeFlip_{w}"] = slope_flip_detector(arr, w)
    return df


# ------------------------------------------------------------
# Trend Exhaustion: Momentum Divergence
# ------------------------------------------------------------

def momentum_divergence(df: pd.DataFrame, window: int = 20):
    """
    Price makes new highs but momentum doesn't (bearish),
    or price makes new lows but momentum doesn't (bullish).
    """
    arr = df["Close"].values
    rsi_vals = rsi(arr, window)

    price_high = pd.Series(arr).rolling(window).max()
    price_low = pd.Series(arr).rolling(window).min()
    rsi_high = pd.Series(rsi_vals).rolling(window).max()
    rsi_low = pd.Series(rsi_vals).rolling(window).min()

    bearish_div = ((arr >= price_high) & (rsi_vals < rsi_high.shift(1))).astype(int)
    bullish_div = ((arr <= price_low) & (rsi_vals > rsi_low.shift(1))).astype(int)

    return bearish_div, bullish_div


def add_divergence_features(df: pd.DataFrame, windows=[14, 20]):
    for w in windows:
        bear, bull = momentum_divergence(df, w)
        df[f"BearDiv_{w}"] = bear
        df[f"BullDiv_{w}"] = bull
    return df


# ------------------------------------------------------------
# Trend Exhaustion: Price Extension
# ------------------------------------------------------------

def price_extension(arr: np.ndarray, window: int = 20):
    """
    Measures how extended price is relative to a rolling mean.
    High extension => trend exhaustion risk.
    """
    sma_vals = rolling_mean(arr, window)
    extension = (arr - sma_vals) / (sma_vals + 1e-10)
    return extension


def add_price_extension_features(df: pd.DataFrame, windows=[10, 20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"Extension_{w}"] = price_extension(arr, w)
    return df


# ------------------------------------------------------------
# Compression / Squeeze Detection
# ------------------------------------------------------------

def bollinger_band_width(df: pd.DataFrame, window: int = 20):
    """
    BB Width = (Upper - Lower) / Middle
    Narrow BB width indicates volatility compression.
    """
    sma_vals = df["Close"].rolling(window).mean()
    std_vals = df["Close"].rolling(window).std()

    upper = sma_vals + 2 * std_vals
    lower = sma_vals - 2 * std_vals
    width = (upper - lower) / (sma_vals + 1e-10)
    return width


def atr_compression(df: pd.DataFrame, window: int = 14):
    """
    ATR compression: ATR normalized by rolling range.
    Low ATR => volatility squeeze.
    """
    atr = df["TrueRange"].rolling(window).mean()
    price_range = df["High"].rolling(window).max() - df["Low"].rolling(window).min()
    return atr / (price_range + 1e-10)


def add_compression_features(df: pd.DataFrame):
    df["BBWidth_20"] = bollinger_band_width(df, 20)
    df["ATRCompress_14"] = atr_compression(df, 14)
    return df


# ------------------------------------------------------------
# Trend Deviation: Distance from Regression Line
# ------------------------------------------------------------

def regression_deviation(arr: np.ndarray, window: int = 50):
    """
    Measures deviation from rolling regression fit.
    Large deviations imply instability or overextension.
    """
    out = np.full(len(arr), np.nan)
    t_idx = np.arange(window)

    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1 : i + 1]
        x = t_idx - t_idx.mean()
        y_c = y - y.mean()

        slope = np.sum(x * y_c) / (np.sum(x * x) + 1e-10)
        intercept = y.mean()

        y_hat = intercept + slope * x
        out[i] = arr[i] - y_hat[-1]

    return out


def add_regression_deviation_features(df: pd.DataFrame, windows=[20, 50]):
    arr = df["Close"].values
    for w in windows:
        df[f"RegDev_{w}"] = regression_deviation(arr, w)
    return df


# ------------------------------------------------------------
# Trend Compression Index (custom research metric)
# ------------------------------------------------------------

def trend_compression_index(df: pd.DataFrame, window: int = 20):
    """
    Trend Compression = ratio of:
        rolling std of close / ATR(20)
    Low ratio => quiet, compressed environment.
    """
    price_std = df["Close"].rolling(window).std()
    atr = df["TrueRange"].rolling(20).mean()
    return price_std / (atr + 1e-10)


def add_tci_features(df: pd.DataFrame):
    df["TCI_20"] = trend_compression_index(df, 20)
    df["TCI_50"] = trend_compression_index(df, 50)
    return df


# ------------------------------------------------------------
# FINAL TREND FEATURE AGGREGATOR
# ------------------------------------------------------------

def add_trend_features(df: pd.DataFrame):
    """
    Final full trend feature generator including:
    
    Chunk 1:
    - SMA, EMA
    
    Chunk 2:
    - WMA, HMA, TEMA, KAMA
    
    Chunk 3:
    - RSI, Stochastics, CMO, TRIX, ROC, WilliamsR
    - Ultimate Oscillator, Awesome Oscillator, TSI
    
    Chunk 4:
    - Regression slopes, R2, curvature, acceleration
    - Persistence, DPO, NSR, vol-adjusted slope
    - Trend strength metrics
    
    Chunk 5:
    - Reversal detectors (local maxima/minima & slope flips)
    - Momentum/price divergence
    - Extension, compression indicators
    - Regression deviation
    - Trend compression index
    """

    # Existing features from previous chunks
    MA_WINDOWS = [3,5,8,10,12,20,21,34,50,89,100,144,200]

    df = add_sma_features(df, MA_WINDOWS)
    df = add_ema_features(df, MA_WINDOWS)
    df = add_wma_features(df, MA_WINDOWS)
    df = add_hma_features(df, MA_WINDOWS)
    df = add_tema_features(df, MA_WINDOWS)
    df = add_kama_features(df, MA_WINDOWS)

    df = add_momentum_features(df)
    df = add_trend_geometry_features(df)

    df = add_extrema_features(df)
    df = add_slope_flip_features(df)
    df = add_divergence_features(df)
    df = add_price_extension_features(df)
    df = add_compression_features(df)
    df = add_regression_deviation_features(df)
    df = add_tci_features(df)

    return df
