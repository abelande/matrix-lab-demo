"""
multi_horizon.py — Multi-Horizon Label Builder
Maximal Institutional Version

Chunk 1:
    - Horizon matrix construction
    - Multi-horizon directional labels
    - Multi-horizon log returns
    - Horizon stacking utilities
    - Safe alignment tools
"""

import numpy as np
import pandas as pd
from typing import List, Union

from .forward_returns import forward_return, trim_forward_returns

def build_horizon_return_matrix(close: pd.Series,
                                horizons: List[int],
                                log: bool = False):
    """
    Constructs a DataFrame of forward returns for multiple horizons.

    Output columns:
        ret_h1, ret_h5, ret_h10, ...
    """

    df = pd.DataFrame(index=close.index)

    for h in horizons:
        df[f"ret_h{h}"] = forward_return(close, h, log=log)

    # Remove rows without full horizon coverage
    df = trim_forward_returns(df, max(horizons))

    return df

def build_horizon_log_return_matrix(close: pd.Series,
                                    horizons: List[int]):
    """
    Multi-horizon log return builder.
    """

    return build_horizon_return_matrix(close, horizons, log=True)

def build_horizon_direction_matrix(close: pd.Series,
                                   horizons: List[int]):
    """
    Multi-class directional labels across horizons.
    """

    df = pd.DataFrame(index=close.index)

    for h in horizons:
        df[f"dir_h{h}"] = np.sign(forward_return(close, h)).fillna(0)

    df = trim_forward_returns(df, max(horizons))
    return df

def stack_horizon_targets(close: pd.Series,
                          horizons: List[int],
                          log: bool = False):
    """
    Creates a stacked (samples × horizons) matrix
    suitable for DL models with multi-output heads.

    Returns:
        np.ndarray shape (N, H)
    """

    df = build_horizon_return_matrix(close, horizons, log=log)

    # Convert to numpy target matrix
    return df.values, df.index, df.columns

def validate_horizons(close: pd.Series, horizons: List[int]):
    """
    Ensures all horizons are <= length of dataset.
    """

    max_h = max(horizons)

    if max_h >= len(close):
        raise ValueError(f"Horizon {max_h} too large for dataset of length {len(close)}")

def build_multi_horizon_targets(close: pd.Series,
                                horizons: List[int],
                                log: bool = False,
                                directional: bool = False):
    """
    Unified high-level builder:

        directional=False → return targets
        directional=True  → directional classification targets
    """

    validate_horizons(close, horizons)

    if directional:
        df = build_horizon_direction_matrix(close, horizons)
    else:
        df = build_horizon_return_matrix(close, horizons, log=log)

    return df
#end of multi_horizon.py