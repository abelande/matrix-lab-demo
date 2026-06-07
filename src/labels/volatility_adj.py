"""
volatility_adj.py — Volatility-Adjusted Labels & Returns
Maximal Institutional Version

Chunk 1:
    - Volatility-normalized forward returns
    - z-scored returns
    - Risk-adjusted directional labels
    - Signal-to-Volatility Ratio (SVR) targets
    - Drawdown-adjusted returns (risk-sensitive targets)
"""

import numpy as np
import pandas as pd
from typing import List, Union

from .forward_returns import forward_return
from .volatility import ewma_volatility, return_volatility

def vol_normalized_return(close: pd.Series,
                          horizon: int = 20,
                          vol_window: int = 20,
                          log: bool = False):
    """
    forward_return / rolling_volatility
    Creates risk-adjusted target.
    """

    ret = forward_return(close, horizon=horizon, log=log)
    vol = return_volatility(close, window=vol_window)

    adj = ret / (vol + 1e-12)
    return adj

def zscored_forward_return(close: pd.Series,
                           horizon: int = 20,
                           lookback: int = 100):
    """
    Converts forward returns into standard score:
        z = (fwd - mean) / std
    """

    ret = forward_return(close, horizon=horizon)
    mean = ret.rolling(lookback).mean()
    std = ret.rolling(lookback).std()

    return (ret - mean) / (std + 1e-12)

def risk_adjusted_direction(close: pd.Series,
                            horizon: int = 20,
                            vol_window: int = 20,
                            threshold: float = 0.5):
    """
    risk_adj = forward_return / volatility
    Then convert to up/down label using threshold.

    threshold = minimum normalized return required for a directional call
    """

    adj = vol_normalized_return(close, horizon, vol_window)

    label = np.where(adj > threshold, 1,
            np.where(adj < -threshold, -1, 0))

    return pd.Series(label, index=close.index)

def svr_target(close: pd.Series,
               horizon: int = 20,
               vol_window: int = 20):
    """
    Signal-to-Volatility Ratio:
        SVR = forward_return / volatility
    """

    ret = forward_return(close, horizon)
    vol = return_volatility(close, vol_window)

    return ret / (vol + 1e-12)

def drawdown_adjusted_return(close: pd.Series,
                             horizon: int = 20):
    """
    Penalizes returns by interim drawdowns inside the horizon.
    """

    ret = forward_return(close, horizon)
    dd = (close.expanding().max() - close) / close

    # normalize drawdown to same scale as returns
    dd_norm = dd.rolling(horizon).max()

    return ret - dd_norm

def build_vol_adjusted_label(close: pd.Series,
                             method: str = "svr",
                             horizon: int = 20,
                             vol_window: int = 20,
                             threshold: float = 0.5):
    """
    Available methods:
        - 'svr'       : signal-to-vol ratio
        - 'norm'      : vol-normalized forward return
        - 'zscore'    : z-scored forward return
        - 'risk_dir'  : thresholded risk-adjusted direction
        - 'dd_adj'    : drawdown-adjusted return
    """

    if method == "svr":
        return svr_target(close, horizon, vol_window)

    if method == "norm":
        return vol_normalized_return(close, horizon, vol_window)

    if method == "zscore":
        return zscored_forward_return(close, horizon)

    if method == "risk_dir":
        return risk_adjusted_direction(close, horizon, vol_window, threshold)

    if method == "dd_adj":
        return drawdown_adjusted_return(close, horizon)

    raise ValueError(f"Unknown volatility-adjusted method: {method}")
#end of volatility_adj.py