"""
fractals.py — Market Fractal Structure Module
Institutional-Grade Version

Chunk 1:
    - Bill Williams fractals (5-bar)
    - Multi-window fractals (3, 5, 7)
    - Fractal strength metrics
    - Fractal trendlines (support/resistance from fractals)
    - Fractal break signals
    - Fractal volatility compression
"""

import numpy as np
import pandas as pd

def williams_fractals(df: pd.DataFrame):
    """
    Returns:
        fractal_high (array)
        fractal_low (array)
    """
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    fractal_high = np.zeros(n)
    fractal_low = np.zeros(n)

    for i in range(2, n-2):

        # Fractal High
        if (high[i] > high[i-1] > high[i-2]) and (high[i] > high[i+1] > high[i+2]):
            fractal_high[i] = 1

        # Fractal Low
        if (low[i] < low[i-1] < low[i-2]) and (low[i] < low[i+1] < low[i+2]):
            fractal_low[i] = 1

    return fractal_high, fractal_low

def adaptive_fractals(df: pd.DataFrame, window=3):
    """
    window must be odd: 3, 5, 7, ...
    center index = window // 2
    """

    if window % 2 == 0:
        raise ValueError("window must be odd")

    k = window // 2
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    fractal_high = np.zeros(n)
    fractal_low = np.zeros(n)

    for i in range(k, n-k):
        seg_high = high[i-k:i+k+1]
        seg_low = low[i-k:i+k+1]

        if high[i] == max(seg_high):
            fractal_high[i] = 1
        if low[i] == min(seg_low):
            fractal_low[i] = 1

    return fractal_high, fractal_low

def fractal_strength(df: pd.DataFrame, fractal_high, fractal_low, window=20):
    """
    Stronger fractals occur further from mid-range.
    """

    highs = df["High"]
    lows = df["Low"]

    range_high = highs.rolling(window).max()
    range_low = lows.rolling(window).min()
    mid = (range_high + range_low) / 2

    strength_hi = (highs - mid) / (range_high - range_low + 1e-10)
    strength_lo = (mid - lows) / (range_high - range_low + 1e-10)

    strength_hi = strength_hi * fractal_high
    strength_lo = strength_lo * fractal_low

    return strength_hi, strength_lo

def fractal_trendlines(df: pd.DataFrame, fractal_high, fractal_low, lookback=50):
    """
    Creates dynamic support/resistance from fractals.
    """

    highs = df["High"]
    lows = df["Low"]

    res = highs.where(fractal_high == 1).rolling(lookback).max()
    sup = lows.where(fractal_low == 1).rolling(lookback).min()

    return sup, res

def fractal_breaks(df: pd.DataFrame, sup, res):
    close = df["Close"]

    breakout = (close > res.shift(1)).astype(int)
    breakdown = (close < sup.shift(1)).astype(int)

    return breakout, breakdown

def fractal_compression(fractal_high, fractal_low, window=20):
    """
    Measures structural compression via decreasing time between fractals.
    """

    idx_hi = np.where(fractal_high == 1)[0]
    idx_lo = np.where(fractal_low == 1)[0]

    # Time since last fractal
    comp_hi = np.full(len(fractal_high), np.nan)
    comp_lo = np.full(len(fractal_low), np.nan)

    for i in range(1, len(idx_hi)):
        comp_hi[idx_hi[i]] = idx_hi[i] - idx_hi[i-1]

    for i in range(1, len(idx_lo)):
        comp_lo[idx_lo[i]] = idx_lo[i] - idx_lo[i-1]

    # Rolling compression score
    comp = pd.Series(np.nan_to_num(
        pd.Series(comp_hi).rolling(window).mean() +
        pd.Series(comp_lo).rolling(window).mean()
    ))

    return comp

def add_fractal_features(df: pd.DataFrame):
    """
    Full fractal feature suite:
        - Bill Williams fractals
        - Adaptive fractals (3,5,7)
        - Fractal strength
        - Fractal-based SR trendlines
        - Fractal break signals
        - Fractal compression
    """

    # Base fractals
    fhi5, flo5 = williams_fractals(df)

    df["FractalHigh_5"] = fhi5
    df["FractalLow_5"] = flo5

    # Adaptive fractals
    for w in [3, 5, 7]:
        fhi, flo = adaptive_fractals(df, window=w)
        df[f"FractalHigh_{w}"] = fhi
        df[f"FractalLow_{w}"] = flo

        hi_strength, lo_strength = fractal_strength(df, fhi, flo)
        df[f"FracStrengthHigh_{w}"] = hi_strength
        df[f"FracStrengthLow_{w}"] = lo_strength

    # Trendlines
    sup, res = fractal_trendlines(df, fhi5, flo5)
    df["FractalSupport"] = sup
    df["FractalResistance"] = res

    # Breakouts
    bo, bd = fractal_breaks(df, sup, res)
    df["FractalBreakout"] = bo
    df["FractalBreakdown"] = bd

    # Compression
    df["FractalCompression"] = fractal_compression(fhi5, flo5)

    return df
#end of fractals.py