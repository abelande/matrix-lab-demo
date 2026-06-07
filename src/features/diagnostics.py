"""
diagnostics.py — Maximal Diagnostic Suite
Institutional-Grade Version

Chunk 1:
    - Feature drift detection
    - Rolling variance & mean stability
    - Autocorrelation diagnostics
    - Signal-to-noise ratio (SNR)
    - Multicollinearity detection (VIF)
    - Constant & zero variance checks
    - Leakage diagnostics
"""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

def detect_constant_features(df: pd.DataFrame):
    """
    Returns columns with zero variance or constant values.
    """
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    return constant_cols

def compute_feature_drift(series, window=200):
    """
    Feature drift = difference between rolling mean and overall mean.
    """
    overall_mean = series.mean()
    roll_mean = series.rolling(window).mean()

    drift = roll_mean - overall_mean
    return drift

def compute_variance_drift(series, window=200):
    overall_std = series.std()
    roll_std = series.rolling(window).std()
    return roll_std - overall_std

def autocorrelation(series, lag=1):
    """
    Computes autocorrelation at lag k.
    """
    return series.autocorr(lag=lag)

def rolling_autocorr(series, window=200, lag=1):
    out = np.zeros(len(series))
    for i in range(window, len(series)):
        out[i] = series.iloc[i-window:i].autocorr(lag=lag)
    return out

def signal_to_noise(series):
    """
    SNR = |mean| / std
    """
    return abs(series.mean()) / (series.std() + 1e-10)

def rolling_snr(series, window=200):
    out = np.zeros(len(series))
    for i in range(window, len(series)):
        window_slice = series.iloc[i-window:i]
        out[i] = signal_to_noise(window_slice)
    return out

def compute_vif(df: pd.DataFrame, cols):
    """
    Computes VIF for selected feature columns.
    """
    X = df[cols].fillna(0).values
    vifs = {}
    for i, col in enumerate(cols):
        vifs[col] = vif(X, i)
    return vifs

def leakage_test(df: pd.DataFrame, feature, target="Close", horizon=1):
    """
    Checks correlation between feature and future returns.
    If extremely high, the feature may leak future information.
    """
    future_ret = df[target].pct_change().shift(-horizon)
    aligned = df[feature]

    corr = aligned.corr(future_ret)

    return corr

def detect_lookahead_leak(series, future_window=5):
    """
    Detects if series leads price changes in an unrealistic way.
    """
    forward_change = series.shift(-future_window) - series
    return forward_change

def run_feature_diagnostics(df: pd.DataFrame, feature_cols, target="Close"):
    """
    Runs full diagnostic suite and returns:
        - constant feature list
        - drift metrics
        - variance drift
        - autocorrelation
        - SNR
        - VIF
        - leakage correlations
    """

    results = {
        "constant_features": [],
        "drift": {},
        "variance_drift": {},
        "autocorr": {},
        "snr": {},
        "vif": {},
        "leakage": {}
    }

    # Constant features
    results["constant_features"] = detect_constant_features(df[feature_cols])

    # Drift metrics
    for col in feature_cols:
        results["drift"][col] = compute_feature_drift(df[col])
        results["variance_drift"][col] = compute_variance_drift(df[col])
        results["autocorr"][col] = autocorrelation(df[col])
        results["snr"][col] = signal_to_noise(df[col])
        results["leakage"][col] = leakage_test(df, col, target=target)

    # VIF calculations
    try:
        results["vif"] = compute_vif(df, feature_cols)
    except Exception:
        results["vif"] = {col: np.nan for col in feature_cols}

    return results
#end of diagnostics.py