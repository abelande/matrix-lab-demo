from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def embargo_bars(n_obs: int, embargo: Union[int, float]) -> int:
    """Convert embargo spec to bars.
    - int: bars
    - float: fraction of n_obs, must be in [0,1)
    """
    if isinstance(embargo, float):
        if embargo < 0 or embargo >= 1:
            raise ValueError("embargo fraction must be in [0, 1).")
        return int(np.ceil(n_obs * embargo))
    if isinstance(embargo, int):
        if embargo < 0:
            raise ValueError("embargo bars must be >= 0.")
        return embargo
    raise TypeError("embargo must be int (bars) or float (fraction).")


def contiguous_blocks(sorted_idx: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start, end) inclusive blocks from sorted indices."""
    if sorted_idx.size == 0:
        return []
    blocks: List[Tuple[int, int]] = []
    start = prev = int(sorted_idx[0])
    for x in sorted_idx[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            blocks.append((start, prev))
            start = prev = x
    blocks.append((start, prev))
    return blocks


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_obs: int,
    embargo: Union[int, float],
    *,
    after_each_block: bool = True,
) -> np.ndarray:
    """Remove embargo window(s) after test indices from training indices.

    Supports non-contiguous test sets:
      - if after_each_block=True, embargo after each contiguous test block
      - else embargo only after the final test index
    """
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    e = embargo_bars(n_obs, embargo)
    if e <= 0 or test_idx.size == 0:
        return train_idx

    test_sorted = np.unique(np.sort(test_idx))
    embargo_indices = set()

    if after_each_block:
        for _, end in contiguous_blocks(test_sorted):
            for j in range(end + 1, min(n_obs, end + 1 + e)):
                embargo_indices.add(j)
    else:
        end = int(test_sorted.max())
        for j in range(end + 1, min(n_obs, end + 1 + e)):
            embargo_indices.add(j)

    if not embargo_indices:
        return train_idx

    embargo_indices = np.fromiter(embargo_indices, dtype=int)
    embargo_set = set(embargo_indices.tolist())
    out = np.array([i for i in train_idx if i not in embargo_set], dtype=int)
    return out


def adaptive_embargo_mask(
    index: pd.DatetimeIndex,
    test_idx: np.ndarray,
    vol: pd.Series,
    method: str = "vol_scaled",
    scale: float = 2.0,
    window: int = 50,
) -> np.ndarray:
    """Return boolean mask marking embargo rows around the test set.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Datetime index for the full dataset.
    test_idx : array-like of int
        Integer positions for the test set.
    vol : pd.Series
        Volatility series indexed by `index`.
    method : str
        Embargo sizing method. Currently supports "vol_scaled".
    scale : float
        Scaling factor used by the sizing method.
    window : int
        Rolling window size used for volatility normalization.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing.")

    n_obs = len(index)
    if n_obs == 0:
        return np.zeros(0, dtype=bool)

    test_idx = np.asarray(test_idx, dtype=int)
    if test_idx.size == 0:
        return np.zeros(n_obs, dtype=bool)
    if test_idx.min() < 0 or test_idx.max() >= n_obs:
        raise IndexError("test_idx contains positions outside index range.")

    if window <= 0:
        raise ValueError("window must be positive.")
    if scale < 0:
        raise ValueError("scale must be non-negative.")

    vol = pd.Series(vol, index=vol.index if isinstance(vol, pd.Series) else index)
    vol = vol.reindex(index)
    if vol.isna().all():
        raise ValueError("vol series is all NaN after reindexing to index.")
    if vol.isna().any():
        vol = vol.ffill().bfill()

    if method != "vol_scaled":
        raise ValueError(f"Unknown method: {method}")

    typical = vol.rolling(window=window, min_periods=1).median()
    typical = typical.replace(0, np.nan)
    ratio = (vol / typical).fillna(0.0)
    embargo_len = np.ceil(scale * ratio).astype(int).to_numpy()
    embargo_len = np.maximum(embargo_len, 0)

    test_sorted = np.unique(np.sort(test_idx))
    mask = np.zeros(n_obs, dtype=bool)
    for start, end in contiguous_blocks(test_sorted):
        block_len = int(embargo_len[start : end + 1].max())
        if block_len <= 0:
            continue
        left = max(0, start - block_len)
        right = min(n_obs - 1, end + block_len)
        mask[left : right + 1] = True

    mask[test_sorted] = False
    return mask
