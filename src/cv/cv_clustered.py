from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cv_adaptive_embargo import apply_embargo, contiguous_blocks

IndexLike = Union[pd.Index, pd.Series, np.ndarray]


def _to_index(x: IndexLike) -> pd.Index:
    if isinstance(x, pd.Index):
        return x
    if isinstance(x, pd.DataFrame):
        return x.index
    if isinstance(x, pd.Series):
        return x.index
    return pd.Index(np.asarray(x))


def _assert_time_index(index: pd.Index) -> None:
    if not isinstance(index, pd.Index):
        raise TypeError("index must be a pandas Index.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing.")


def _validate_event_end_times(index: pd.Index, t1: Optional[pd.Series]) -> Optional[pd.Series]:
    if t1 is None:
        return None
    if not isinstance(t1, pd.Series):
        raise TypeError("event_end_times must be a pandas Series indexed by event start times.")
    t1 = t1.reindex(index)
    if t1.isna().any():
        t1 = t1.fillna(pd.Series(index=index, data=index))
    return t1


def _make_blocks(n_obs: int, n_splits: int, blocks_per_split: int, min_block_size: int) -> List[Tuple[int, int]]:
    if n_obs <= 0:
        return []
    if blocks_per_split < 1:
        raise ValueError("blocks_per_split must be >= 1.")
    if min_block_size < 1:
        raise ValueError("min_block_size must be >= 1.")
    target_blocks = max(n_splits, n_splits * blocks_per_split)
    block_size = max(min_block_size, int(np.ceil(n_obs / target_blocks)))
    blocks = []
    for start in range(0, n_obs, block_size):
        stop = min(n_obs, start + block_size)
        blocks.append((start, stop))  # [start, stop)
    return blocks


def _block_cluster_counts(
    clusters: pd.Series,
    blocks: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    cat = pd.Categorical(clusters)
    codes = np.asarray(cat.codes, dtype=int)
    n_clusters = len(cat.categories)

    counts = np.zeros((len(blocks), n_clusters), dtype=int)
    sizes = np.zeros(len(blocks), dtype=int)
    for i, (start, stop) in enumerate(blocks):
        block_codes = codes[start:stop]
        sizes[i] = stop - start
        if block_codes.size:
            counts[i] = np.bincount(block_codes, minlength=n_clusters)
    return counts, sizes


def _assign_blocks_to_folds(
    counts: np.ndarray,
    sizes: np.ndarray,
    n_splits: int,
) -> List[List[int]]:
    n_blocks = counts.shape[0]
    n_clusters = counts.shape[1]
    target = counts.sum(axis=0) / max(n_splits, 1)
    target_size = sizes.sum() / max(n_splits, 1)

    fold_counts = np.zeros((n_splits, n_clusters), dtype=float)
    fold_sizes = np.zeros(n_splits, dtype=float)
    fold_blocks: List[List[int]] = [[] for _ in range(n_splits)]

    for b in range(n_blocks):
        if b < n_splits:
            k = b
        else:
            best_k = 0
            best_score = None
            for k in range(n_splits):
                new_counts = fold_counts[k] + counts[b]
                new_size = fold_sizes[k] + sizes[b]
                score = float(np.sum((new_counts - target) ** 2) + (new_size - target_size) ** 2)
                if best_score is None or score < best_score:
                    best_score = score
                    best_k = k
            k = best_k

        fold_counts[k] += counts[b]
        fold_sizes[k] += sizes[b]
        fold_blocks[k].append(b)
    return fold_blocks


def _purged_train_indices(
    index: pd.Index,
    t1: pd.Series,
    test_idx: np.ndarray,
) -> np.ndarray:
    n_obs = len(index)
    if test_idx.size == 0:
        return np.arange(n_obs, dtype=int)

    mask = np.ones(n_obs, dtype=bool)
    test_sorted = np.unique(np.sort(test_idx))
    for start, end in contiguous_blocks(test_sorted):
        t0 = index[start]
        t1_block = t1.iloc[start : end + 1].max()
        # Use .array instead of .values to preserve timezone info
        overlap = (index <= t1_block) & (t1.array >= t0)
        mask &= ~overlap
    return np.where(mask)[0]


@dataclass
class ClusteredCV:
    """Regime-aware CV splitter using contiguous blocks and cluster balancing.

    Parameters
    ----------
    clusters : array-like or pd.Series
        Cluster/regime labels aligned to X.index.
    n_splits : int
        Number of folds.
    event_end_times : pd.Series | None
        Event end times for purging (t1). If None, no purging.
    embargo_pct : int|float
        Embargo bars (int) or fraction of dataset (float in [0,1)).
    blocks_per_split : int
        Approximate number of contiguous blocks per split to balance clusters.
    min_block_size : int
        Minimum size of each contiguous block.
    """

    clusters: Union[pd.Series, Iterable]
    n_splits: int = 5
    event_end_times: Optional[pd.Series] = None
    embargo_pct: Union[int, float] = 0.0
    blocks_per_split: int = 4
    min_block_size: int = 1

    def split(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        index = _to_index(X)
        _assert_time_index(index)
        n_obs = len(index)
        if n_obs == 0:
            return
        if self.n_splits > n_obs:
            raise ValueError("n_splits cannot exceed number of observations.")

        # Align clusters to index
        if isinstance(self.clusters, pd.Series):
            clusters = self.clusters.reindex(index)
        else:
            clusters = pd.Series(self.clusters, index=index)
        if clusters.isna().any():
            raise ValueError("clusters contain NaN after aligning to X.index.")

        t1 = _validate_event_end_times(index, self.event_end_times)

        blocks = _make_blocks(n_obs, self.n_splits, self.blocks_per_split, self.min_block_size)
        counts, sizes = _block_cluster_counts(clusters, blocks)
        fold_blocks = _assign_blocks_to_folds(counts, sizes, self.n_splits)

        for fold in range(self.n_splits):
            test_blocks = fold_blocks[fold]
            test_idx_list: List[np.ndarray] = []
            for b in test_blocks:
                start, stop = blocks[b]
                test_idx_list.append(np.arange(start, stop, dtype=int))
            test_idx = np.concatenate(test_idx_list) if test_idx_list else np.array([], dtype=int)

            train_idx = np.arange(n_obs, dtype=int)
            if t1 is not None and test_idx.size:
                train_idx = _purged_train_indices(index, t1, test_idx)

            if test_idx.size and self.embargo_pct:
                train_idx = apply_embargo(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    n_obs=n_obs,
                    embargo=self.embargo_pct,
                    after_each_block=True,
                )

            # Remove test indices from train (safety)
            test_set = set(test_idx.tolist())
            train_idx = np.array([i for i in train_idx if i not in test_set], dtype=int)

            yield train_idx, np.sort(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
