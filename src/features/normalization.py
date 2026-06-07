"""
normalization.py — Maximal Feature Normalization Suite
Institutional-Grade Version

Chunk 1:
    - Standard z-score
    - Robust z-score (MAD)
    - Min-max scaling
    - Rank scaling
    - Winsorization
    - Vol-adjusted normalization
    - Regime-aware normalization
"""

import numpy as np
import pandas as pd

def zscore(series):
    mean = series.mean()
    std = series.std() + 1e-10
    return (series - mean) / std

def mad(series):
    return np.median(np.abs(series - np.median(series))) + 1e-10

def robust_zscore(series):
    median = np.median(series)
    mad_val = mad(series)
    return (series - median) / mad_val

def minmax_scale(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-10)

def rank_scale(series):
    """
    Converts values to percent ranks in [0, 1].
    """
    return series.rank(pct=True)

def winsorize(series, lower=0.01, upper=0.99):
    """
    Clips series between given percentiles.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)

def vol_adjust(series, window=20):
    vol = series.rolling(window).std() + 1e-10
    return series / vol

def regime_aware_normalize(series, regime_series):
    """
    Applies different normalizations depending on volatility regime.
    
    Assumes regime_series ∈ {0,1,2,3} from vol classification.
    """
    normalized = np.zeros(len(series))
    s = series.copy().reset_index(drop=True)
    r = regime_series.reset_index(drop=True)

    for i in range(len(s)):
        if r[i] == 0:  # low vol
            normalized[i] = (s[i] - s.mean()) / (s.std() + 1e-10)

        elif r[i] == 1:  # normal vol
            normalized[i] = (s[i] - s.mean()) / (s.std() + 1e-10)

        elif r[i] == 2:  # high vol
            med = np.median(s)
            mad_val = mad(s)
            normalized[i] = (s[i] - med) / mad_val

        elif r[i] == 3:  # extreme vol
            # Winsorize first
            ws = winsorize(s)
            med = np.median(ws)
            mad_val = mad(ws)
            normalized[i] = (ws[i] - med) / mad_val

        else:
            normalized[i] = np.nan

    return pd.Series(normalized, index=series.index)

def normalize_feature(df, col, method="zscore", regime_col=None):
    """
    Normalize a single feature in df using chosen method.
    """

    s = df[col]

    if method == "zscore":
        return zscore(s)

    if method == "robust":
        return robust_zscore(s)

    if method == "minmax":
        return minmax_scale(s)

    if method == "rank":
        return rank_scale(s)

    if method == "winsor":
        return winsorize(s)

    if method == "vol_adjust":
        return vol_adjust(s)

    if method == "regime_aware":
        if regime_col is None:
            raise ValueError("regime_col required for regime-aware normalization")
        return regime_aware_normalize(s, df[regime_col])

    raise ValueError(f"Unknown normalization method: {method}")

def apply_normalization(df: pd.DataFrame, cols, method="zscore", regime_col=None):
    for col in cols:
        df[f"{col}_{method}"] = normalize_feature(df, col, method, regime_col)
    return df
# end of normalization.py