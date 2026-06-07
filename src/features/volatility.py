"""
volatility.py — Maximal Volatility Feature Module
Institutional-Grade Version

Chunk 1:
    - True Range
    - Realized volatility estimators:
        • Parkinson
        • Garman-Klass
        • Rogers-Satchell
        • Yang-Zhang
    - Rolling variance / stdev
    - Vol-of-vol
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. TRUE RANGE (Foundation for Volatility)
# ============================================================

def true_range(df: pd.DataFrame):
    """
    True Range = max(
        High - Low,
        |High - PrevClose|,
        |Low - PrevClose|
    )
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def add_true_range(df: pd.DataFrame):
    df["TrueRange"] = true_range(df)
    return df


# ============================================================
# 2. PARKINSON VOLATILITY (High-Low Range)
# ============================================================

def parkinson_vol(df: pd.DataFrame, window=20):
    """
    Parkinson volatility uses only high-low price:
        sigma^2 = (1/(4 ln2)) * average( (ln(H/L))^2 )
    """
    hl = np.log(df["High"] / df["Low"]) ** 2
    return (hl.rolling(window).mean() / (4 * np.log(2))) ** 0.5


def add_parkinson_vol(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"Parkinson_{w}"] = parkinson_vol(df, window=w)
    return df


# ============================================================
# 3. GARMAN-KLASS VOLATILITY
# ============================================================

def garman_klass_vol(df: pd.DataFrame, window=20):
    """
    GK volatility incorporates open-high-low-close:
    
    sigma^2 = 0.5*(ln(H/L))^2 - (2ln2 - 1)*(ln(C/O))^2
    """
    log_hl = np.log(df["High"] / df["Low"]) ** 2
    log_co = np.log(df["Close"] / df["Open"]) ** 2
    var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    var[var < 0] = 0  # numerical safety
    return var.rolling(window).mean() ** 0.5


def add_gk_vol(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"GK_{w}"] = garman_klass_vol(df, window=w)
    return df


# ============================================================
# 4. ROGERS-SATCHELL VOLATILITY
# ============================================================

def rogers_satchell_vol(df: pd.DataFrame, window=20):
    """
    Rogers-Satchell models volatility in trending markets.

    RS = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """
    term1 = np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"])
    term2 = np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"])
    rs = (term1 + term2).rolling(window).mean()
    return np.sqrt(np.maximum(rs, 0))


def add_rs_vol(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"RS_{w}"] = rogers_satchell_vol(df, window=w)
    return df


# ============================================================
# 5. YANG-ZHANG VOLATILITY (Most Accurate Estimator)
# ============================================================

def yang_zhang_vol(df: pd.DataFrame, window=20):
    """
    Yang-Zhang combines:
        • overnight returns
        • Rogers-Satchell (intra-day)
        • open-to-close volatility
    Considered one of the best estimators for equity microstructure.
    """
    log_oc = np.log(df["Close"] / df["Open"])
    log_oo = np.log(df["Open"] / df["Close"].shift(1))

    sigma_oc = log_oc.rolling(window).var()
    sigma_oo = log_oo.rolling(window).var()
    rs = rogers_satchell_vol(df, window=window) ** 2

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz = sigma_oo + k * sigma_oc + (1 - k) * rs
    return np.sqrt(np.maximum(yz, 0))


def add_yz_vol(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"YZ_{w}"] = yang_zhang_vol(df, window=w)
    return df


# ============================================================
# 6. Rolling Variance, Rolling Stdev, Vol-of-Vol
# ============================================================

def add_basic_vol_metrics(df: pd.DataFrame, windows=[10, 20, 50]):
    returns = df["Close"].pct_change()

    for w in windows:
        df[f"Vol_{w}"] = returns.rolling(w).std()
        df[f"Var_{w}"] = returns.rolling(w).var()
        df[f"VolOfVol_{w}"] = df[f"Vol_{w}"].rolling(w).std()

    return df


# ============================================================
# COMPATIBILITY WRAPPERS
# ============================================================

def add_basic_vol_features(df: pd.DataFrame):
    """Compatibility wrapper used by the unified pipeline.

    Earlier versions of the repo referred to a single entrypoint named
    `add_basic_vol_features`. In this snapshot, the basic vol features are
    broken into multiple helpers. This wrapper composes a sensible default
    sequence while remaining safe if some prerequisite columns are missing.
    """
    df = df.copy()

    # True range & ATR-like components (if OHLC exists).
    if {"High", "Low", "Close"}.issubset(df.columns):
        df = add_true_range(df)

    # Common realized vol estimators.
    df = add_basic_vol_metrics(df)

    # Higher-frequency realized vol features (based on returns).
    df = add_realized_vol_features(df)

    # EWMA volatility estimate.
    df = add_ewma_vol(df)

    return df


# ============================================================
# 7. Master Dispatcher (Chunk 1)
# ============================================================

def add_realized_vol_features(df: pd.DataFrame):
    """
    Realized volatility suite (chunk 1):
        - True Range
        - Parkinson
        - Garman-Klass
        - Rogers-Satchell
        - Yang-Zhang
        - Rolling variance & stdev
        - Vol-of-Vol
    """

    df = add_true_range(df)
    df = add_parkinson_vol(df)
    df = add_gk_vol(df)
    df = add_rs_vol(df)
    df = add_yz_vol(df)
    df = add_basic_vol_metrics(df)

    return df

# ============================================================
# 8. EXPONENTIAL WEIGHTED MOVING AVERAGE (EWMA) VOLATILITY
# ============================================================

def ewma_volatility(df: pd.DataFrame, lambda_=0.94):
    """
    RiskMetrics EWMA volatility:
        sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2
    """
    returns = df["Close"].pct_change().fillna(0)
    ewma = np.zeros(len(returns))
    ewma[0] = returns.iloc[0] ** 2

    for t in range(1, len(returns)):
        ewma[t] = lambda_ * ewma[t-1] + (1 - lambda_) * returns.iloc[t] ** 2

    return np.sqrt(ewma)


def add_ewma_vol(df: pd.DataFrame):
    df["EWMA_94"] = ewma_volatility(df, lambda_=0.94)
    df["EWMA_97"] = ewma_volatility(df, lambda_=0.97)
    df["EWMA_99"] = ewma_volatility(df, lambda_=0.99)
    return df


# ============================================================
# 9. VOLATILITY CLUSTERING
# ============================================================

def volatility_clustering(df: pd.DataFrame, window=20):
    """
    Clustering ratio:
        cluster = mean(|r_t|) / std(|r_t|)
    High values → volatility clustering regime.
    """
    abs_ret = df["Close"].pct_change().abs()
    ma = abs_ret.rolling(window).mean()
    sd = abs_ret.rolling(window).std()
    return ma / (sd + 1e-10)


def add_vol_clustering(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"VolCluster_{w}"] = volatility_clustering(df, w)
    return df


# ------------------------------------------------------------
# Shock–Recovery Indicator
# ------------------------------------------------------------

def shock_recovery_indicator(df: pd.DataFrame, window=20):
    """
    Measures recovery speed after volatility spikes.

    SR = (current vol - rolling min vol) / (rolling max vol - rolling min vol)
    → Approaches 1 in recovery regime
    """
    vol = df["Close"].pct_change().rolling(window).std()
    rolling_max = vol.rolling(window).max()
    rolling_min = vol.rolling(window).min()
    return (vol - rolling_min) / (rolling_max - rolling_min + 1e-10)


def add_shock_recovery(df: pd.DataFrame):
    df["ShockRec_20"] = shock_recovery_indicator(df, 20)
    df["ShockRec_50"] = shock_recovery_indicator(df, 50)
    return df


# ============================================================
# 10. UPSIDE / DOWNSIDE VOLATILITY
# ============================================================

def upside_downside_vol(df: pd.DataFrame, window=20):
    """
    Separates volatility into:
        - upside vol (returns > 0)
        - downside vol (returns < 0)
    """
    rets = df["Close"].pct_change()

    upside = rets.where(rets > 0, 0)
    downside = rets.where(rets < 0, 0)

    up_vol = upside.rolling(window).std()
    down_vol = downside.rolling(window).std()
    return up_vol, down_vol


def add_ud_vol(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        up, down = upside_downside_vol(df, w)
        df[f"UpsideVol_{w}"] = up
        df[f"DownsideVol_{w}"] = down
        df[f"UDRatio_{w}"] = up / (down + 1e-10)
    return df


# ============================================================
# 11. REALIZED VARIANCE & BIPOWER VARIATION
# ============================================================

def realized_variance(df: pd.DataFrame):
    returns = df["Close"].pct_change()
    return np.square(returns).fillna(0)


def realized_bipower_variation(df: pd.DataFrame, window=20):
    """
    Bipower Variation (BV):
        BV = (pi/2)*sum(|r_t|*|r_{t-1}|)

    Robust against jumps, tracks continuous volatility.
    """
    rets = df["Close"].pct_change().abs()
    bv = (np.pi / 2) * (rets * rets.shift(1)).rolling(window).sum()
    return bv


def add_bv_features(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"BV_{w}"] = realized_bipower_variation(df, window=w)
    return df


# ============================================================
# 12. BNS JUMP DETECTION (Barndorff-Nielsen & Shephard)
# ============================================================

def bns_jump_detection(df: pd.DataFrame, window=20):
    """
    BNS Jump component:
        Jump = RV - BV
    Normalized jump index:
        Jump / BV
    """
    rv = realized_variance(df).rolling(window).sum()
    bv = realized_bipower_variation(df, window)

    jump_component = rv - bv
    jump_component[jump_component < 0] = 0  # no negative jumps

    jump_index = jump_component / (bv + 1e-10)
    return jump_component, jump_index


def add_bns_jump_features(df: pd.DataFrame, windows=[20, 50]):
    for w in windows:
        jump, jump_idx = bns_jump_detection(df, w)
        df[f"JumpComp_{w}"] = jump
        df[f"JumpIdx_{w}"] = jump_idx
    return df


# ============================================================
# 13. VOLATILITY REGIME SIGNAL (simple precursor to HMM)
# ============================================================

def volatility_regime_signal(df: pd.DataFrame, window=20):
    """
    Creates a smoothed volatility signal suitable for downstream HMM.
    
    SmoothVol = rolling_mean(return^2, window)
    """
    rets = df["Close"].pct_change().fillna(0)
    smooth = (rets ** 2).rolling(window).mean()
    return smooth


def add_vol_regime_signal(df: pd.DataFrame):
    df["VolRegime_20"] = volatility_regime_signal(df, 20)
    df["VolRegime_50"] = volatility_regime_signal(df, 50)
    return df


# ============================================================
# 14. MASTER DISPATCHER (Chunk 2)
# ============================================================

def add_advanced_vol_features(df: pd.DataFrame):
    """
    Adds advanced volatility analytics:
        - EWMA volatility (0.94, 0.97, 0.99)
        - Volatility clustering
        - Shock-recovery indicator
        - Upside / downside volatility
        - Bipower Variation (BV)
        - BNS Jump components
        - Volatility regime signal
    """
    df = add_ewma_vol(df)
    df = add_vol_clustering(df)
    df = add_shock_recovery(df)
    df = add_ud_vol(df)
    df = add_bv_features(df)
    df = add_bns_jump_features(df)
    df = add_vol_regime_signal(df)

    return df

# ============================================================
# 15. TAIL RISK METRICS
# ============================================================

def tail_volatility(df: pd.DataFrame, window=20, percentile=0.1):
    """
    Computes downside and upside tail volatility.
    Downside tail = std of returns in lower percentile.
    Upside tail   = std of returns in upper percentile.
    """
    rets = df["Close"].pct_change()

    # percentiles
    lower_cut = rets.quantile(percentile)
    upper_cut = rets.quantile(1 - percentile)

    downside = rets.where(rets < lower_cut).rolling(window).std()
    upside = rets.where(rets > upper_cut).rolling(window).std()

    return downside, upside


def add_tail_vol(df: pd.DataFrame, windows=[20, 50], percentile=0.1):
    for w in windows:
        down, up = tail_volatility(df, window=w, percentile=percentile)
        df[f"TailDownVol_{w}"] = down
        df[f"TailUpVol_{w}"] = up
        df[f"TailRatio_{w}"] = up / (down + 1e-10)
    return df


# ============================================================
# 16. EXTREME MOVE DETECTOR (X-MOVE)
# ============================================================

def extreme_move_indicator(df: pd.DataFrame, z_threshold=3):
    """
    Flags extreme moves where |z-score of returns| > threshold.
    """
    rets = df["Close"].pct_change()
    z = (rets - rets.mean()) / (rets.std() + 1e-10)
    return (z.abs() > z_threshold).astype(int)


def add_extreme_move_feature(df: pd.DataFrame, z_threshold=3):
    df["XMove"] = extreme_move_indicator(df, z_threshold)
    return df


# ============================================================
# 17. VOLATILITY SKEW (Directional Asymmetry)
# ============================================================

def volatility_skew(df: pd.DataFrame, window=20):
    """
    Volatility skew = (upside vol - downside vol) / total vol
    """
    rets = df["Close"].pct_change()

    upside = rets.where(rets > 0).rolling(window).std()
    downside = rets.where(rets < 0).rolling(window).std()

    total = rets.rolling(window).std()

    skew = (upside - downside) / (total + 1e-10)
    return skew


def add_vol_skew(df: pd.DataFrame, windows=[20, 50]):
    for w in windows:
        df[f"VolSkew_{w}"] = volatility_skew(df, w)
    return df


# ============================================================
# 18. VOLATILITY KURTOSIS (TAIL FATNESS)
# ============================================================

def volatility_kurtosis(df: pd.DataFrame, window=20):
    """
    Returns rolling kurtosis of returns.
    High kurtosis => fat tails => crash risk.
    """
    return df["Close"].pct_change().rolling(window).kurt()


def add_vol_kurtosis(df: pd.DataFrame, windows=[20, 50]):
    for w in windows:
        df[f"VolKurt_{w}"] = volatility_kurtosis(df, w)
    return df


# ============================================================
# 19. VOLATILITY CONVEXITY
# ============================================================

def volatility_convexity(df: pd.DataFrame, window=20):
    """
    Vol convexity approximates the curvature of volatility.
    Convex markets are usually stressed or transitioning.
    """
    vol = df["Close"].pct_change().rolling(window).std()
    convexity = vol.diff().diff()
    return convexity


def add_vol_convexity(df: pd.DataFrame, windows=[20, 50]):
    for w in windows:
        df[f"VolConvex_{w}"] = volatility_convexity(df, w)
    return df


# ============================================================
# 20. VOLATILITY REGIME TRANSITION METRICS
# ============================================================

def volatility_state_shift(df: pd.DataFrame, window=50):
    """
    Detects transitions between vol regimes by examining
    changes in smoothed (EWMA) volatility levels.
    
    Returns +1 on shift to high-vol regime
            -1 on shift to low-vol regime
             0 otherwise
    """
    ewma = ewma_volatility(df, lambda_=0.94)
    shift = np.zeros(len(df))

    for i in range(1, len(df)):
        if ewma[i-1] < ewma[i] and ewma[i] > ewma.mean():
            shift[i] = 1
        elif ewma[i-1] > ewma[i] and ewma[i] < ewma.mean():
            shift[i] = -1

    return shift


def add_vol_regime_shift(df: pd.DataFrame):
    df["VolShift_50"] = volatility_state_shift(df, 50)
    return df


# ============================================================
# 21. VOLATILITY TRANSFORMATION FEATURES
# ============================================================

def add_vol_transforms(df: pd.DataFrame, windows=[20, 50]):
    """
    Adds:
        - log-vol
        - z-scored volatility
        - volatility ratios
    """
    for w in windows:
        vol = df["Close"].pct_change().rolling(w).std()

        df[f"LogVol_{w}"] = np.log(vol + 1e-10)
        df[f"ZVol_{w}"] = (vol - vol.mean()) / (vol.std() + 1e-10)
        df[f"VolRatio_{w}"] = vol / (df["EWMA_94"] + 1e-10)

    return df

# ============================================================
# 22. MASTER DISPATCHER (Chunk 3)
# ============================================================

def add_tail_and_shape_features(df: pd.DataFrame):
    """
    Adds:
        - Tail volatility (downside/upside)
        - Extreme move indicator
        - Volatility skew
        - Volatility kurtosis
        - Volatility convexity
        - Vol regime shifts
        - Log-vol, Z-vol, vol ratios
    """
    df = add_tail_vol(df)
    df = add_extreme_move_feature(df)
    df = add_vol_skew(df)
    df = add_vol_kurtosis(df)
    df = add_vol_convexity(df)
    df = add_vol_regime_shift(df)
    df = add_vol_transforms(df)

    return df
