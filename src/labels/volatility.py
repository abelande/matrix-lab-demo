"""
volatility.py — Maximal Institutional Volatility Estimators
Used in event creation, triple-barrier labeling, meta-labeling, and regime labels.
"""

import numpy as np
import pandas as pd

def ewma_volatility(close: pd.Series, span: int = 100):
    """
    Exponentially weighted volatility of returns.
    Used as default dynamic barrier scale in triple-barrier labeling.
    """
    returns = close.pct_change()
    vol = returns.ewm(span=span).std()
    return vol

def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 30):
    """
    Parkinson volatility:
        σ² = (1/(4 ln 2)) * mean[(ln(H/L))²]
    """
    hl = np.log(high / low) ** 2
    factor = 1.0 / (4 * np.log(2))
    return (factor * hl.rolling(window).mean()).apply(np.sqrt)

def garman_klass_vol(open_: pd.Series,
                     high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     window: int = 30):
    """
    Garman–Klass volatility estimator.
    """
    log_hl = np.log(high / low)
    log_oc = np.log(close / open_)

    rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_oc ** 2
    return rs.rolling(window).mean().apply(np.sqrt)

def rogers_satchell_vol(open_: pd.Series,
                        high: pd.Series,
                        low: pd.Series,
                        close: pd.Series,
                        window: int = 30):
    """
    Rogers–Satchell volatility estimator.
    """
    term1 = np.log(high / close) * np.log(high / open_)
    term2 = np.log(low / close) * np.log(low / open_)
    rs = term1 + term2
    return rs.rolling(window).mean().apply(np.sqrt)

def yang_zhang_vol(open_: pd.Series,
                   high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   window: int = 30):
    """
    Yang–Zhang volatility — most accurate of classical estimators.
    """

    # Overnight return
    r_o = np.log(open_ / close.shift(1))

    # Open–Close
    r_c = np.log(close / open_)

    # Rogers–Satchell term
    rs = rogers_satchell_vol(open_, high, low, close, window=1) ** 2

    # Compute windows
    sigma_o = r_o.rolling(window).var()
    sigma_c = r_c.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()

    return np.sqrt(sigma_o + sigma_c + sigma_rs)

def return_volatility(close: pd.Series, window: int = 30):
    return close.pct_change().rolling(window).std()

def volatility_regime(vol: pd.Series, bins: int = 4):
    """
    Discretizes volatility into regimes.
    Example output: 0,1,2,3
    """
    return pd.qcut(vol, bins, labels=False, duplicates="drop")

def compute_volatility(df: pd.DataFrame,
                       method: str = "ewma",
                       span: int = 100,
                       window: int = 30):
    """
    General-purpose dispatcher for volatility estimation.
    """

    if method == "ewma":
        return ewma_volatility(df["Close"], span=span)

    if method == "parkinson":
        return parkinson_vol(df["High"], df["Low"], window=window)

    if method == "gk":
        return garman_klass_vol(df["Open"], df["High"], df["Low"], df["Close"], window)

    if method == "rs":
        return rogers_satchell_vol(df["Open"], df["High"], df["Low"], df["Close"], window)

    if method == "yz":
        return yang_zhang_vol(df["Open"], df["High"], df["Low"], df["Close"], window)

    if method == "return":
        return return_volatility(df["Close"], window)

    raise ValueError(f"Unknown volatility method: {method}")
#end of volatility.py