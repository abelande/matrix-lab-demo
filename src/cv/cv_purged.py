"""
purged_cv.py — Purged & Embargoed Cross-Validation utilities
Inspired by Marcos López de Prado (Advances in Financial Machine Learning)

Core idea:
- Labels often span time (event starts at t0 and ends at t1).
- If a training label overlaps a test label in time, training is "contaminated".
- Purging removes overlapping training samples.
- Embargo removes a safety buffer after the test set to reduce leakage.

This module supports:
- PurgedKFold (sklearn-compatible)
- purged_cv_splits convenience generator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import BaseCrossValidator
except Exception:  # sklearn optional
    BaseCrossValidator = object

# Set up debug file logger
_debug_log_path = Path(__file__).resolve().parents[2] / "artifacts" / "debugs" / "cv_purged_debug.log"
_debug_log_path.parent.mkdir(parents=True, exist_ok=True)
_debug_logger = logging.getLogger("cv_purged_debug")
_debug_logger.setLevel(logging.DEBUG)
_debug_logger.propagate = False
if not _debug_logger.handlers:
    _fh = logging.FileHandler(_debug_log_path, mode="a")
    _fh.setFormatter(logging.Formatter("%(asctime)s — %(message)s"))
    _debug_logger.addHandler(_fh)


IndexLike = Union[pd.Index, pd.Series, np.ndarray]


def _to_index(x: IndexLike) -> pd.Index:
    if isinstance(x, pd.Index):
        return x
    if isinstance(x, pd.DataFrame):
        return x.index
    if isinstance(x, pd.Series):
        return x.index
    return pd.Index(np.asarray(x))


def assert_time_index(index: pd.Index) -> None:
    """
    Assert that index is a valid DatetimeIndex for time series cross-validation.

    Performs strict validation checks and raises errors if any fail.
    Use this for input validation in CV splitters.

    Checks:
    - Index is a DatetimeIndex
    - Not empty
    - No duplicate timestamps
    - Chronologically sorted

    Parameters
    ----------
    index : pd.Index
        Index to validate

    Raises
    ------
    TypeError
        If index is not a DatetimeIndex
    ValueError
        If index is empty, has duplicates, or is not sorted

    Examples
    --------
    >>> assert_time_index(df.index)  # Validates, raises if invalid
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"Index must be a DatetimeIndex, got {type(index).__name__}. "
            "Convert with: df.index = pd.to_datetime(df.index)"
        )

    if len(index) == 0:
        raise ValueError("Index is empty.")

    if index.duplicated().any():
        n_dupes = index.duplicated().sum()
        raise ValueError(
            f"Index contains {n_dupes} duplicate timestamps. "
            "Remove with: df = df[~df.index.duplicated(keep='first')]"
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            "Index is not chronologically sorted. "
            "Sort with: df.sort_index(inplace=True)"
        )


def validate_datetime_index(index: pd.Index, sort: bool = True) -> pd.Index:
    """
    Validate and optionally sort a DatetimeIndex for time series data.

    Checks:
    - Index is a DatetimeIndex
    - No duplicate timestamps
    - Chronologically sorted (or sorts if sort=True)
    - Not empty

    Parameters
    ----------
    index : pd.Index
        Index to validate
    sort : bool, default=True
        If True, return sorted index. If False, raise error if unsorted.

    Returns
    -------
    pd.Index
        Validated (and possibly sorted) DatetimeIndex

    Raises
    ------
    TypeError
        If index is not a DatetimeIndex
    ValueError
        If index has duplicates, is empty, or is unsorted (when sort=False)
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"Index must be a DatetimeIndex, got {type(index).__name__}. "
            "Convert with: df.index = pd.to_datetime(df.index)"
        )

    if len(index) == 0:
        raise ValueError("Index is empty.")

    if index.duplicated().any():
        n_dupes = index.duplicated().sum()
        raise ValueError(
            f"Index contains {n_dupes} duplicate timestamps. "
            "Remove with: df = df[~df.index.duplicated(keep='first')]"
        )

    if not index.is_monotonic_increasing:
        if sort:
            # Return sorted index
            return index.sort_values()
        else:
            raise ValueError(
                "Index is not chronologically sorted. "
                "Sort with: df.sort_index(inplace=True)"
            )

    return index


def validate_t1(index: pd.Index, t1: pd.Series) -> pd.Series:
    """
    t1: Series indexed by t0 (event start), with values = event end time
    """
    if not isinstance(t1, pd.Series):
        raise TypeError("t1 must be a pandas Series indexed by event start times (t0).")
    if not isinstance(t1.index, pd.Index):
        raise TypeError("t1.index must be a pandas Index.")

    # Align to index, but keep only index entries present in t1 if provided that way
    t1 = t1.reindex(index)

    # If t1 has NaNs, treat as instantaneous labels ending at t0
    t1 = t1.fillna(pd.Series(index=index, data=index))

    return t1


def _compute_embargo_bars(n_obs: int, embargo: Union[int, float]) -> int:
    """
    embargo:
      - int: number of bars/rows to embargo
      - float in [0,1): fraction of dataset length to embargo
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


def _purge_train_indices(
    index: pd.Index,
    t1: pd.Series,
    test_indices: np.ndarray,
) -> np.ndarray:
    """
    Remove training points whose (t0, t1) overlaps any test event span.
    Overlap definition:
      train t0 <= test t1 AND train t1 >= test t0
    """
    _debug_logger.debug(f"_purge START: t1 type={type(t1)}, t1 shape={getattr(t1, 'shape', 'no shape')}")
    test_t0 = index[test_indices]
    test_t1 = t1.iloc[test_indices]

    _debug_logger.debug(f"_purge EARLY: test_t0 type={type(test_t0)}, len={len(test_t0)}")
    _debug_logger.debug(f"_purge EARLY: test_t1 type={type(test_t1)}, shape={getattr(test_t1, 'shape', 'no shape')}")

    # Test span bounds (vector)
    test_start = test_t0.min()
    test_end = test_t1.max()

    _debug_logger.debug(f"_purge EARLY: test_start={test_start}, type={type(test_start)}, shape={getattr(test_start, 'shape', 'scalar')}")

    # Any train event overlapping the test span gets purged
    train_t0 = index
    # Use .array instead of .values to preserve timezone info
    train_t1 = t1.array

    _debug_logger.debug(f"_purge: index len={len(index)}, t1 len={len(t1)}, train_t1 len={len(train_t1)}")
    _debug_logger.debug(f"_purge: train_t0 type={type(train_t0)}, train_t1 type={type(train_t1)}")
    _debug_logger.debug(f"_purge: test_start type={type(test_start)}, test_end type={type(test_end)}")

    # Isolate the comparison to find which one fails
    try:
        result1 = train_t0 <= test_end
        _debug_logger.debug(f"_purge: (train_t0 <= test_end) shape={result1.shape}")
    except Exception as e:
        _debug_logger.debug(f"_purge: train_t0 <= test_end FAILED: {e}")

    try:
        result2 = train_t1 >= test_start
        _debug_logger.debug(f"_purge: (train_t1 >= test_start) shape={result2.shape}")
    except Exception as e:
        _debug_logger.debug(f"_purge: train_t1 >= test_start FAILED: {e}")

    overlap = (train_t0 <= test_end) & (train_t1 >= test_start)
    keep_mask = ~overlap
    return np.where(keep_mask)[0]



def _apply_embargo(
    n_obs: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    embargo_bars: int,
) -> np.ndarray:
    """
    Remove the embargo window immediately AFTER the test set from training indices.
    """
    if embargo_bars <= 0:
        return train_indices

    test_end = test_indices.max()
    embargo_start = test_end + 1
    embargo_end = min(n_obs, test_end + 1 + embargo_bars)

    embargo_range = np.arange(embargo_start, embargo_end, dtype=int)
    if embargo_range.size == 0:
        return train_indices

    embargo_set = set(embargo_range.tolist())
    return np.array([i for i in train_indices if i not in embargo_set], dtype=int)


def purged_split(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    test_idx: np.ndarray,
    t1: Optional[pd.Series] = None,
    embargo: Union[int, float] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-split purging and embargo helper.

    Given a dataset and test indices, returns purged train indices.
    Useful when you already have a train/test split defined and want to apply
    purging and embargo logic.

    Parameters
    ----------
    X : DataFrame/Series/ndarray
        Dataset with index for temporal alignment.
    test_idx : np.ndarray
        Positional indices for the test set.
    t1 : pd.Series, optional
        Event end times indexed by t0. If None, labels are instantaneous.
    embargo : int|float, default=0
        int -> number of bars to embargo after test set
        float -> fraction of dataset length to embargo

    Returns
    -------
    train_idx : np.ndarray
        Purged and embargoed training indices (positional).
    test_idx : np.ndarray
        Test indices (unchanged, returned for convenience).

    Example
    -------
    >>> train_idx, test_idx = purged_split(X, test_idx=test_indices, embargo=0.01)
    >>> X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """
    index = _to_index(X)
    n_obs = len(index)

    # Validate test indices
    test_idx = np.asarray(test_idx, dtype=int)
    if test_idx.max() >= n_obs or test_idx.min() < 0:
        raise ValueError(f"test_idx contains invalid indices for dataset of length {n_obs}")

    # Handle t1
    if t1 is None:
        t1 = pd.Series(index=index, data=index)
    else:
        t1 = validate_t1(index, t1)

    # Compute embargo bars
    embargo_bars = _compute_embargo_bars(n_obs, embargo)

    # Start with all indices as potential training set
    train_idx = np.arange(0, n_obs, dtype=int)

    # Purge overlapping training samples
    train_idx = _purge_train_indices(index=index, t1=t1, test_indices=test_idx)

    # Apply embargo
    train_idx = _apply_embargo(
        n_obs=n_obs,
        train_indices=train_idx,
        test_indices=test_idx,
        embargo_bars=embargo_bars,
    )

    # Remove test indices from train if still present
    train_set = set(train_idx.tolist())
    train_set.difference_update(test_idx.tolist())
    train_idx = np.array(sorted(train_set), dtype=int)

    return train_idx, test_idx


@dataclass
class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold CV with embargo.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        Event end times, indexed by t0 aligned to X index.
        If None, labels are treated as instantaneous (t1 == t0).
    embargo : int|float
        int -> embargo bars
        float -> embargo fraction of dataset length
    """

    n_splits: int = 5
    t1: Optional[pd.Series] = None
    embargo: Union[int, float] = 0

    def split(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        index = _to_index(X)
        n_obs = len(index)

        t1 = self.t1
        _debug_logger.debug(f"PurgedKFold.split: self.t1 length={len(self.t1) if self.t1 is not None else 'None'}")
        if t1 is None:
            # instantaneous labels
            t1 = pd.Series(index=index, data=index)
        else:
            _debug_logger.debug(f"PurgedKFold.split: Before validate_t1 - t1 length={len(t1)}, index length={len(index)}")
            t1 = validate_t1(index, t1)
            _debug_logger.debug(f"PurgedKFold.split: After validate_t1 - t1 length={len(t1)}")

        embargo_bars = _compute_embargo_bars(n_obs, self.embargo)

        # standard k-fold partition by index position
        fold_sizes = np.full(self.n_splits, n_obs // self.n_splits, dtype=int)
        fold_sizes[: n_obs % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = np.arange(start, stop, dtype=int)

            # start with all indices, then purge overlaps, then embargo
            train_idx = np.arange(0, n_obs, dtype=int)
            train_idx = _purge_train_indices(index=index, t1=t1, test_indices=test_idx)
            train_idx = _apply_embargo(
                n_obs=n_obs,
                train_indices=train_idx,
                test_indices=test_idx,
                embargo_bars=embargo_bars,
            )

            # also remove test indices if still present
            train_set = set(train_idx.tolist())
            train_set.difference_update(test_idx.tolist())
            train_idx = np.array(sorted(train_set), dtype=int)

            yield train_idx, test_idx
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def purged_cv_splits(
    df_or_X: Union[pd.DataFrame, pd.Series, np.ndarray],
    n_splits: int = 5,
    embargo: Union[int, float] = 0,
    t1: Optional[pd.Series] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience generator for existing notebook usage.

    Parameters
    ----------
    df_or_X : DataFrame/Series/ndarray
        Must have an index if using event times.
    n_splits : int
    embargo : int|float
        int -> bars
        float -> fraction of dataset
    t1 : Series or None
        Event end times; if None labels treated instantaneous.

    Yields
    ------
    train_idx, test_idx : np.ndarray, np.ndarray
        Positional indices suitable for iloc.
    """
    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo=embargo)
    yield from cv.split(df_or_X)
