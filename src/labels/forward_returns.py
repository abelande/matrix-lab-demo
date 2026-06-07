"""
forward_returns.py — Forward Return Target Builder
Maximal Institutional Version

Chunk 1:
    - Single-horizon forward return computation
    - Multiple-horizon stacked forward returns
    - Log-return vs simple return options
    - Excess returns (vs. benchmark or risk-free)
    - Return alignment utilities
"""

import numpy as np
import pandas as pd
from typing import List, Union

def forward_return(close: pd.Series,
                   horizon: int,
                   log: bool = False):
    """
    Computes forward simple or log returns.

    horizon:
        number of bars forward (not seconds)
        e.g., horizon=20 → 20-bar forward return

    log:
        False = simple return
        True  = log return

    Returns:
        pd.Series aligned with index of 'close'
    """

    close_fwd = close.shift(-horizon)

    if log:
        ret = np.log(close_fwd / close)
    else:
        ret = (close_fwd / close) - 1

    return ret

def multi_forward_returns(close: pd.Series,
                          horizons: List[int],
                          log: bool = False):
    """
    Computes forward returns for multiple horizons.

    Returns:
        DataFrame with columns:
            fwd_1, fwd_5, fwd_10, ...
    """

    df = pd.DataFrame(index=close.index)

    for h in horizons:
        df[f"fwd_{h}"] = forward_return(close, horizon=h, log=log)

    return df

def excess_forward_return(close: pd.Series,
                          benchmark: pd.Series,
                          horizon: int,
                          log: bool = False):
    """
    Computes forward returns minus benchmark forward returns.
    """

    ret_asset = forward_return(close, horizon=horizon, log=log)
    ret_bench = forward_return(benchmark, horizon=horizon, log=log)

    return ret_asset - ret_bench

def rolling_forward_returns(close: pd.Series,
                            max_horizon: int = 60,
                            log: bool = False):
    """
    Computes all forward returns from 1 to max_horizon.
    Useful for visual diagnostics and alpha horizon discovery.

    Returns:
        DataFrame with columns:
            fwd_1, fwd_2, ..., fwd_max_horizon
    """

    df = pd.DataFrame(index=close.index)

    for h in range(1, max_horizon + 1):
        df[f"fwd_{h}"] = forward_return(close, horizon=h, log=log)

    return df

def directional_forward_return(close: pd.Series,
                               horizon: int):
    """
    Direction-only target:
        +1 = price rises over horizon
        -1 = price falls
         0 = unchanged (rare)
    """

    ret = forward_return(close, horizon=horizon)
    return np.sign(ret).fillna(0)

def trim_forward_returns(df: pd.DataFrame,
                         max_horizon: int):
    """
    Removes the last 'max_horizon' rows where no future data exists.
    """

    if max_horizon <= 0:
        return df

    return df.iloc[:-max_horizon]
#end of forward_returns.py