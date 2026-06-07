from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cv_purged import assert_time_index, validate_t1, _purge_train_indices as purged_train_indices
from .cv_adaptive_embargo import apply_embargo

try:
    from sklearn.model_selection import BaseCrossValidator
except Exception:  # sklearn optional
    BaseCrossValidator = object


IndexLike = Union[pd.Index, pd.Series, np.ndarray]


def _to_index(x: IndexLike) -> pd.Index:
    if isinstance(x, pd.Index):
        return x
    if isinstance(x, pd.DataFrame):
        return x.index
    if isinstance(x, pd.Series):
        return x.index
    return pd.Index(np.asarray(x))

@dataclass
class CombinatorialPurgedKFold(BaseCrossValidator):
    """Combinatorial Purged Cross-Validation (CPCV).

    Split the dataset into N contiguous folds (by time). For each split, choose K folds as test,
    the rest as train, then apply purging and embargo using event end-times.

    Overlap definition (purging):
      train event i overlaps test event j iff t0_i <= t1_j and t1_i >= t0_j.

    Parameters
    ----------
    n_folds : int
        Total number of base folds (N).
    n_test_folds : int
        Number of folds selected as test in each split (K).
    t1 : Series | None
        Event end-times indexed by t0. If None, labels are instantaneous (t1 == t0).
    embargo : int|float
        int -> embargo bars, float -> fraction of dataset length.
    """
    n_folds: int = 10
    n_test_folds: int = 2
    embargo: Union[int, float] = 0
    t1: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be >= 2.")
        if self.n_test_folds < 1:
            raise ValueError("n_test_folds must be >= 1.")
        if self.n_test_folds >= self.n_folds:
            raise ValueError("n_test_folds must be < n_folds.")

    def split(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        index = _to_index(X)
        assert_time_index(index)
        t1a = validate_t1(index, self.t1)
        n = len(index)

        # Base fold boundaries (contiguous)
        fold_sizes = np.full(self.n_folds, n // self.n_folds, dtype=int)
        fold_sizes[: n % self.n_folds] += 1

        boundaries = []
        cur = 0
        for fs in fold_sizes:
            boundaries.append((cur, cur + fs))  # [start, stop)
            cur += fs

        fold_indices = [np.arange(s, e, dtype=int) for (s, e) in boundaries]

        for test_folds in combinations(range(self.n_folds), self.n_test_folds):
            test_idx = np.concatenate([fold_indices[i] for i in test_folds])
            train_idx = purged_train_indices(index, t1a, test_idx)
            train_idx = apply_embargo(
                train_idx=train_idx,
                test_idx=test_idx,
                n_obs=n,
                embargo=self.embargo,
                after_each_block=True,
            )
            test_set = set(test_idx.tolist())
            train_idx = np.array([i for i in train_idx if i not in test_set], dtype=int)
            yield train_idx, np.sort(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return math.comb(self.n_folds, self.n_test_folds)


def combinatorial_cv_splits(
    X: Union[pd.DataFrame, pd.Series, np.ndarray],
    *,
    n_folds: int = 10,
    n_test_folds: int = 2,
    t1: Optional[pd.Series] = None,
    embargo: Union[int, float] = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Convenience generator for CPCV splits (purging + embargo)."""
    yield from CombinatorialPurgedCV(
        n_folds=n_folds,
        n_test_folds=n_test_folds,
        t1=t1,
        embargo=embargo,
    ).split(X)
