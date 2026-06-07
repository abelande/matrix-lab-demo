from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Literal

import numpy as np
import pandas as pd

from .cv_purged import assert_time_index

@dataclass
class GroupedTimeSplit:
    """Group-aware time split to prevent leakage within the same group.

    Example: group='D' for day, 'W' for week, 'M' for month using pandas period frequency.
    """
    n_splits: int = 5
    group_freq: Literal["D", "W", "M"] = "D"

    def split(self, X: pd.DataFrame | pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idx = X.index
        assert_time_index(idx)

        groups = idx.to_period(self.group_freq)
        unique_groups = pd.Index(groups.unique()).sort_values()
        n_groups = len(unique_groups)

        fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
        fold_sizes[: n_groups % self.n_splits] += 1

        cur = 0
        for fs in fold_sizes:
            test_groups = unique_groups[cur : cur + fs]
            test_mask = groups.isin(test_groups)
            test_idx = np.where(test_mask.values)[0]
            train_idx = np.where(~test_mask.values)[0]
            yield train_idx.astype(int), test_idx.astype(int)
            cur += fs


class CVVisualizer:
    """Simple CV timeline visualizer for train/test splits."""

    def __init__(self, index: pd.Index):
        assert_time_index(index)
        self.index = index

    def timeline(self, train_idx: np.ndarray, test_idx: np.ndarray, title: str = "..."):
        """Plot where train vs test fall along the full index timeline.

        This is a convenience visualization and should not affect model correctness.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            # Graceful fallback if matplotlib isn't available.
            return None

        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        x = np.arange(len(self.index))
        fig, ax = plt.subplots(figsize=(10, 2.5))

        ax.scatter(x[train_idx], np.zeros_like(train_idx), s=10, c="#1f77b4", label="train")
        ax.scatter(x[test_idx], np.ones_like(test_idx), s=10, c="#ff7f0e", label="test")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["train", "test"])
        ax.set_xlabel("index position")
        ax.set_title(title)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        return ax
