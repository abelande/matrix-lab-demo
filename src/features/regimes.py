"""
regimes.py — Maximal Regime Feature Module
Institutional-Grade Version

Chunk 1:
    - Volatility regimes (4-state)
    - Trend regimes (3-state)
    - Combined trend-volatility regimes
    - Regime persistence
    - Regime transition signals
"""

import numpy as np
import pandas as pd

# Support both package-style imports (e.g., features.regimes) and standalone
# usage where these modules live in the same folder.
try:  # pragma: no cover
    from .trend import rolling_regression_slope
    from .volatility import ewma_volatility
except ImportError:  # pragma: no cover
    # When loaded in a non-package context (e.g., via importlib by path),
    # ensure this directory is importable.
    import os
    import sys
    _HERE = os.path.dirname(__file__)
    if _HERE and _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from trend import rolling_regression_slope
    from volatility import ewma_volatility


# ============================================================
# 1. VOLATILITY REGIME CLASSIFICATION (4-STATE)
# ============================================================

def volatility_regime_classification(df: pd.DataFrame, window=20):
    """
    Vol regime classification using EWMA volatility.

    States:
        0 = low vol
        1 = normal vol
        2 = high vol
        3 = extreme vol
    """
    ew = ewma_volatility(df, lambda_=0.94)

    vol = pd.Series(ew)
    q1 = vol.quantile(0.25)
    q2 = vol.quantile(0.50)
    q3 = vol.quantile(0.75)

    regime = np.zeros(len(df))

    regime[vol > q1] = 1
    regime[vol > q2] = 2
    regime[vol > q3] = 3

    return regime


def add_vol_regime_class(df: pd.DataFrame):
    df["VolRegClass_4"] = volatility_regime_classification(df)
    return df


# ============================================================
# 2. TREND REGIME CLASSIFICATION (3-STATE)
# ============================================================

def trend_regime_classification(df: pd.DataFrame, window=50):
    """
    Trend regime classification using regression slope.

    States:
        -1 = downtrend
         0 = neutral / balanced
         1 = uptrend
    """
    slope = rolling_regression_slope(df["Close"].values, window)

    # Normalize slope for regime classification
    s = pd.Series(slope)
    z = (s - s.mean()) / (s.std() + 1e-10)

    regime = np.zeros(len(df))

    regime[z > 0.5] = 1
    regime[z < -0.5] = -1

    return regime


def add_trend_regime_class(df: pd.DataFrame):
    df["TrendRegClass_3"] = trend_regime_classification(df)
    return df


# ============================================================
# 3. COMBINED VOLATILITY–TREND REGIME MATRIX
# ============================================================

def combined_regime_matrix(df: pd.DataFrame):
    """
    Combined state encoding:

    (TrendRegClass_3, VolRegClass_4)

    Encoded as a single integer:
        state = trend * 10 + vol

    Examples:
        Trend +1, Vol 0 →  10
        Trend -1, Vol 3 → -13
    """
    trend = df["TrendRegClass_3"]
    vol = df["VolRegClass_4"]

    return trend * 10 + vol


def add_combined_regime(df: pd.DataFrame):
    df["Regime_Combo"] = combined_regime_matrix(df)
    return df


# ============================================================
# 4. REGIME PERSISTENCE METRICS
# ============================================================

def regime_persistence(series: pd.Series, window=20):
    """
    Measures how long the regime has persisted.

    Persistence = number of consecutive days in same regime.
    """
    persistence = np.zeros(len(series))
    count = 0

    persistence[0] = 1

    for i in range(1, len(series)):
        if series.iloc[i] == series.iloc[i-1]:
            count += 1
        else:
            count = 1
        persistence[i] = count

    return persistence


def add_regime_persistence(df: pd.DataFrame):
    df["RegimePersistence"] = regime_persistence(df["Regime_Combo"])
    return df


# ============================================================
# 5. REGIME TRANSITION SIGNALS
# ============================================================

def regime_transitions(df: pd.DataFrame):
    """
    Detects regime shifts.
    Returns:
        +1 if regime shifts upward (towards high vol / uptrend)
        -1 if regime shifts downward
         0 otherwise
    """
    reg = df["Regime_Combo"]
    transitions = np.zeros(len(df))

    for i in range(1, len(df)):
        if reg.iloc[i] > reg.iloc[i-1]:
            transitions[i] = 1
        elif reg.iloc[i] < reg.iloc[i-1]:
            transitions[i] = -1

    return transitions


def add_regime_transition_features(df: pd.DataFrame):
    df["RegimeShift"] = regime_transitions(df)
    return df


# ============================================================
# 6. MASTER DISPATCHER (Chunk 1)
# ============================================================

def add_regime_features(df: pd.DataFrame):
    """
    Full deterministic regime framework:
        - Volatility regime (4 states)
        - Trend regime (3 states)
        - Combined regime matrix
        - Regime persistence
        - Regime transition signals
    """
    df = add_vol_regime_class(df)
    df = add_trend_regime_class(df)
    df = add_combined_regime(df)
    df = add_regime_persistence(df)
    df = add_regime_transition_features(df)

    return df
