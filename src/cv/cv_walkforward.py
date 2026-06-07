from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, Union

import numpy as np
import pandas as pd

from .cv_purged import purged_split, assert_time_index

try:
    from sklearn.model_selection import BaseCrossValidator
except Exception:  # sklearn optional
    BaseCrossValidator = object


@dataclass
class RollingWindowCV(BaseCrossValidator):
    """Classic rolling walk-forward CV with optional purge/embargo."""
    train_size: int
    test_size: int
    event_end_times: Optional[pd.Series] = None
    embargo_pct: Union[int, float] = 0.0
    step: Optional[int] = None

    def split(
        self,
        X: Union[pd.DataFrame, pd.Series],
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = X.index
        assert_time_index(idx)
        n = len(idx)
        step = self.step or self.test_size

        start = 0
        while True:
            train_start = start
            train_end = start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > n:
                break

            test_idx = np.arange(test_start, test_end, dtype=int)
            train_idx = np.arange(train_start, train_end, dtype=int)

            if self.event_end_times is not None or self.embargo_pct:
                train_idx, _ = purged_split(
                    X,
                    test_idx,
                    t1=self.event_end_times,
                    embargo=self.embargo_pct,
                )

                # Restrict to intended train window
                win_set = set(np.arange(train_start, train_end, dtype=int).tolist())
                train_idx = np.array([i for i in train_idx if i in win_set], dtype=int)

            yield train_idx, test_idx
            start += step

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is None:
            return 0
        n = len(X)
        step = self.step or self.test_size
        window = self.train_size + self.test_size
        if step <= 0 or n < window:
            return 0
        return 1 + (n - window) // step


@dataclass
class WalkForwardSplit:
    """Walk-forward split generator (optionally purged/embargoed via t1)."""
    train_size: int
    test_size: int
    step: Optional[int] = None
    embargo: Union[int, float] = 0
    t1: Optional[pd.Series] = None
    expanding: bool = False

    def split(self, X: Union[pd.DataFrame, pd.Series]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = X.index
        assert_time_index(idx)
        n = len(idx)
        step = self.step or self.test_size

        start = 0
        while True:
            if self.expanding:
                train_start = 0
                train_end = start + self.train_size
            else:
                train_start = start
                train_end = start + self.train_size

            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > n:
                break

            test_idx = np.arange(test_start, test_end, dtype=int)
            train_idx = np.arange(train_start, train_end, dtype=int)

            # If leakage-aware t1 provided, purge/embargo against test
            if self.t1 is not None or self.embargo:
                train_idx, _ = purged_split(X, test_idx, t1=self.t1, embargo=self.embargo)

                # Restrict to intended train window when not expanding
                if not self.expanding:
                    win_set = set(np.arange(train_start, train_end, dtype=int).tolist())
                    train_idx = np.array([i for i in train_idx if i in win_set], dtype=int)

            yield train_idx, test_idx
            start += step


@dataclass
class RollingWindowSplit(WalkForwardSplit):
    """Alias for non-expanding walk-forward."""
    expanding: bool = False


@dataclass
class ExpandingWindowSplit(WalkForwardSplit):
    """Alias for expanding window CV."""
    expanding: bool = True
