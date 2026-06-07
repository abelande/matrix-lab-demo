import numpy as np
import pandas as pd

from cv.cv_walkforward import RollingWindowCV


def _make_X(n: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"x": np.arange(n)}, index=idx)


def test_rolling_window_cv_basic_splits():
    X = _make_X(10)
    cv = RollingWindowCV(train_size=4, test_size=2)

    splits = list(cv.split(X))
    expected = [
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
        (np.array([2, 3, 4, 5]), np.array([6, 7])),
        (np.array([4, 5, 6, 7]), np.array([8, 9])),
    ]

    assert len(splits) == len(expected)
    for (train_idx, test_idx), (exp_train, exp_test) in zip(splits, expected):
        assert np.array_equal(train_idx, exp_train)
        assert np.array_equal(test_idx, exp_test)


def test_rolling_window_cv_get_n_splits():
    X = _make_X(10)
    cv = RollingWindowCV(train_size=4, test_size=2)
    assert cv.get_n_splits(X) == 3

    cv = RollingWindowCV(train_size=4, test_size=2, step=3)
    assert cv.get_n_splits(X) == 2

    cv = RollingWindowCV(train_size=6, test_size=6)
    assert cv.get_n_splits(X) == 0
