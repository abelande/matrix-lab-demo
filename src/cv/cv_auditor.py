from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cv_purged import assert_time_index, validate_t1 as _validate_t1_imported


def validate_t1(index: pd.Index, t1: Optional[pd.Series]) -> pd.Series:
    """Wrapper for validate_t1 that handles None."""
    if t1 is None:
        return pd.Series(index=index, data=index)
    return _validate_t1_imported(index, t1)


def overlap_mask(index: pd.Index, t1: pd.Series, test_indices: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask indicating which observations overlap with test period.
    An observation overlaps if its event window [t0, t1] intersects [test_start, test_end].
    """
    test_start = index[test_indices].min()
    test_end = index[test_indices].max()

    t0 = index
    t1_vals = t1.array

    # Overlap: t0 <= test_end AND t1 >= test_start
    overlap = (t0 <= test_end) & (t1_vals >= test_start)
    return overlap

def split_summary(
    X: pd.DataFrame | pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    *,
    t1: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Summarize fold sizes and purge counts."""
    idx = X.index
    assert_time_index(idx)
    t1a = validate_t1(idx, t1)
    rows = []
    n = len(idx)

    for k, (tr, te) in enumerate(splits):
        mask = overlap_mask(idx, t1a, te)
        purged = int(mask.sum())
        rows.append({
            "fold": k,
            "train_n": int(len(tr)),
            "test_n": int(len(te)),
            "purged_n": purged,
            "purged_frac": purged / n,
        })
    return pd.DataFrame(rows)


def leakage_check(
    X: pd.DataFrame | pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    *,
    t1: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """Confirm no train sample overlaps any test event in each fold."""
    idx = X.index
    assert_time_index(idx)
    t1a = validate_t1(idx, t1)

    results = []
    ok = True
    for k, (tr, te) in enumerate(splits):
        tr_set = set(map(int, tr))
        # Anything overlapping test should have been removed
        mask = overlap_mask(idx, t1a, te)
        overlapping_idx = set(np.where(mask)[0].tolist())
        leak = len(tr_set.intersection(overlapping_idx)) > 0
        ok = ok and (not leak)
        results.append({"fold": k, "leak": leak})
    return {"ok": ok, "folds": results}


class CVLeakageAuditor:
    """Audit CV splits for leakage and basic time window sanity."""

    def __init__(
        self,
        X: pd.DataFrame | pd.Series,
        event_end_times: Optional[pd.Series],
        embargo_masks: Optional[Union[Iterable[np.ndarray], Dict[int, np.ndarray]]] = None,
        *,
        y: Optional[pd.Series] = None,
    ) -> None:
        self.X = X
        self.y = y
        self.event_end_times = event_end_times
        self.embargo_masks = embargo_masks

        assert_time_index(self.X.index)
        if self.y is not None and isinstance(self.y, pd.Series):
            if not self.y.index.equals(self.X.index):
                self.y = self.y.reindex(self.X.index)

    def audit(self, cv_splitter) -> pd.DataFrame:
        """Iterate CV splits and return a leakage report DataFrame."""
        idx = self.X.index
        t1a = validate_t1(idx, self.event_end_times) if self.event_end_times is not None else None

        splits = self._get_splits(cv_splitter)
        rows: List[Dict[str, Any]] = []
        n = len(idx)

        for fold, (tr, te) in enumerate(splits):
            tr = np.asarray(tr, dtype=int)
            te = np.asarray(te, dtype=int)

            row: Dict[str, Any] = {
                "fold": fold,
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "train_min_time": idx[tr].min() if len(tr) else pd.NaT,
                "train_max_time": idx[tr].max() if len(tr) else pd.NaT,
                "test_min_time": idx[te].min() if len(te) else pd.NaT,
                "test_max_time": idx[te].max() if len(te) else pd.NaT,
                # Always-present leak columns so downstream report-building never
                # KeyErrors when t1/embargo are absent for a run; real values below override.
                "overlap_leak_n": 0, "overlap_leak": False,
                "embargo_leak_n": 0, "embargo_leak": False,
            }

            if t1a is not None:
                purge_mask = overlap_mask(idx, t1a, te)
                purge_idx = set(np.where(purge_mask)[0].tolist())
                tr_set = set(tr.tolist())
                overlap_leak_n = len(tr_set.intersection(purge_idx))
                row.update({
                    "overlap_window_n": int(purge_mask.sum()),
                    "overlap_window_frac": float(purge_mask.sum()) / n if n else 0.0,
                    "overlap_leak_n": int(overlap_leak_n),
                    "overlap_leak": overlap_leak_n > 0,
                })

            emb_mask = self._get_embargo_mask(fold, te)
            if emb_mask is not None:
                emb_mask = self._normalize_mask(emb_mask, idx)
                emb_idx = set(np.where(emb_mask)[0].tolist())
                tr_set = set(tr.tolist())
                emb_leak_n = len(tr_set.intersection(emb_idx))
                row.update({
                    "embargo_window_n": int(emb_mask.sum()),
                    "embargo_window_frac": float(emb_mask.sum()) / n if n else 0.0,
                    "embargo_leak_n": int(emb_leak_n),
                    "embargo_leak": emb_leak_n > 0,
                })

            if self.y is not None:
                y_tr = self.y.iloc[tr]
                y_te = self.y.iloc[te]
                row.update({
                    "train_class_counts": y_tr.value_counts(dropna=False).to_dict(),
                    "test_class_counts": y_te.value_counts(dropna=False).to_dict(),
                    "train_class_frac": (y_tr.value_counts(normalize=True, dropna=False)).to_dict(),
                    "test_class_frac": (y_te.value_counts(normalize=True, dropna=False)).to_dict(),
                })

            rows.append(row)

        df = pd.DataFrame(rows)
        # Guarantee leak columns exist even when a splitter yields no folds (empty df
        # has no columns), so downstream report-building never KeyErrors.
        for _col, _default in (("overlap_leak", False), ("overlap_leak_n", 0),
                               ("overlap_window_n", 0), ("overlap_window_frac", 0.0),
                               ("embargo_leak", False), ("embargo_leak_n", 0),
                               ("embargo_window_n", 0), ("embargo_window_frac", 0.0)):
            if _col not in df.columns:
                df[_col] = _default
        return df

    def _get_splits(self, cv_splitter) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.y is not None:
            try:
                return list(cv_splitter.split(self.X, self.y))
            except TypeError:
                return list(cv_splitter.split(self.X))
        return list(cv_splitter.split(self.X))

    def _get_embargo_mask(
        self,
        fold: int,
        test_idx: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.embargo_masks is None:
            return None
        if isinstance(self.embargo_masks, dict):
            return self.embargo_masks.get(fold)
        if isinstance(self.embargo_masks, list) or isinstance(self.embargo_masks, tuple):
            if fold < len(self.embargo_masks):
                return self.embargo_masks[fold]
            return None
        if callable(self.embargo_masks):
            return self.embargo_masks(test_idx, fold)
        return None

    @staticmethod
    def _normalize_mask(mask: np.ndarray, index: pd.Index) -> np.ndarray:
        if isinstance(mask, pd.Series):
            if not mask.index.equals(index):
                mask = mask.reindex(index)
            return mask.fillna(False).to_numpy(dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != len(index):
            raise ValueError("embargo mask length must match X index length.")
        return mask
