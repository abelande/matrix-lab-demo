"""
chaos.py — Maximal Chaos & Complexity Analytics
Institutional-Grade Version

Chunk 1:
    - Hurst exponent
    - Fractal dimension
    - Detrended Fluctuation Analysis (DFA)
    - Approximate Entropy (ApEn)
    - Sample Entropy (SampEn)
    - Lyapunov exponent
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq

def hurst_exponent(series, min_window=10, max_window=200):
    """
    Computes Hurst exponent using log(R/S) regression.
    """

    # Use log-returns; guard against non-positive values.
    s = pd.Series(series).astype(float)
    s = s.where(s > 0)
    s = np.log(s).diff().dropna()
    if len(s) < max_window:
        return np.nan

    window_sizes = np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        num=20
    ).astype(int)

    tau = []
    lag_vec = []

    for w in window_sizes:
        if w >= len(s):
            continue
        rs = (s.rolling(w).max() - s.rolling(w).min()) / (s.rolling(w).std() + 1e-10)
        tau.append(np.nanmean(rs.values))
        lag_vec.append(w)

    tau = np.array(tau)
    lag_vec = np.array(lag_vec)

    log_tau = np.log(tau + 1e-10)
    log_lag = np.log(lag_vec + 1e-10)

    # Regression for slope (Hurst exponent)
    coef, _, _, _ = lstsq(
        np.vstack([log_lag, np.ones(len(log_lag))]).T,
        log_tau,
        rcond=None
    )

    # lstsq returns [slope, intercept]
    return float(coef[0])

def higuchi_fd(series, kmax=10):
    """
    Computes Higuchi Fractal Dimension (HFD)
    """
    x = series.values
    N = len(x)

    L = np.zeros(kmax)
    k_vals = np.arange(1, kmax + 1)

    for k in k_vals:
        Lk = np.zeros(k)
        for m in range(k):
            idx = np.arange(m, N, k)
            diffs = np.abs(np.diff(x[idx]))
            if len(diffs) == 0:
                continue
            Lm = np.sum(diffs) * (N - 1) / (len(diffs) * k)
            Lk[m] = Lm
        L[k - 1] = Lk.mean()

    logL = np.log(L + 1e-10)
    logk = np.log(1.0 / k_vals)

    coef, _, _, _ = lstsq(
        np.vstack([logk, np.ones(len(logk))]).T,
        logL,
        rcond=None
    )

    # fractal dimension = 1 - slope
    return float(1 - coef[0])

def detrended_fluctuation_analysis(series, window_sizes=[10, 20, 50, 100]):
    """
    Computes Detrended Fluctuation Analysis (α exponent)
    """
    x = series.values
    y = np.cumsum(x - np.mean(x))
    F = []

    for win in window_sizes:
        if win >= len(x):
            F.append(np.nan)
            continue

        rms_vals = []
        for i in range(0, len(x) - win, win):
            segment = y[i:i+win]
            t = np.arange(win)
            coeffs = np.polyfit(t, segment, 1)
            detrended = segment - np.polyval(coeffs, t)
            rms_vals.append(np.sqrt(np.mean(detrended**2)))
        F.append(np.mean(rms_vals))

    window_sizes = np.asarray(window_sizes, dtype=float)
    F = np.array(F, dtype=float)
    logF = np.log(F + 1e-10)
    logS = np.log(window_sizes + 1e-10)

    coef, _, _, _ = lstsq(
        np.vstack([logS, np.ones(len(logS))]).T,
        logF,
        rcond=None
    )

    return float(coef[0])   # DFA exponent

def approximate_entropy(U, m=2, r=0.2):
    """
    Pincus Approximate Entropy (ApEn)
    """
    U = np.array(U)

    def _phi(m):
        X = np.array([U[i:i+m] for i in range(len(U)-m+1)])
        C = np.sum(
            np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r,
            axis=0
        ) / (len(U) - m + 1)
        return np.sum(np.log(C + 1e-10)) / (len(U) - m + 1)

    return _phi(m) - _phi(m+1)

def sample_entropy(U, m=2, r=0.2):
    U = np.array(U)
    N = len(U)

    def _count(m):
        X = np.array([U[i:i+m] for i in range(N - m)])
        C = np.sum(
            np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r,
            axis=0
        )
        return np.sum(C) - (N - m)

    B = _count(m)
    A = _count(m + 1)

    return -np.log((A + 1e-10) / (B + 1e-10))

def lyapunov_exponent(series, delay=1, dimension=2, window=200):
    """
    Simple Rosenstein method for Lyapunov exponent.
    """
    x = series.values
    N = len(x)

    if N < window + delay * dimension:
        return np.nan

    # Construct embedding vectors
    M = N - (dimension - 1) * delay
    Y = np.zeros((M, dimension))
    for i in range(dimension):
        Y[:, i] = x[i*delay : i*delay + M]

    # Distances between nearest neighbors
    d0 = np.zeros(M)

    for i in range(M):
        # exclude very close indices
        dists = np.linalg.norm(Y - Y[i], axis=1)
        dists[i] = np.inf 
        j = np.argmin(dists)
        d0[i] = dists[j]

    # Growth of separation
    log_divergence = np.log(d0 + 1e-10)
    L = np.mean(log_divergence[:window]) / window
    return L

def add_chaos_features(df: pd.DataFrame):
    """
    Adds:
        - Hurst exponent
        - Fractal dimension
        - DFA exponent
        - Approximate entropy
        - Sample entropy
        - Lyapunov exponent
    """
    close = df["Close"].copy()

    df["Hurst"] = hurst_exponent(close)
    df["FracDim"] = higuchi_fd(close)
    df["DFA"] = detrended_fluctuation_analysis(close)
    df["ApEn"] = approximate_entropy(close)
    df["SampEn"] = sample_entropy(close)
    df["Lyap"] = lyapunov_exponent(close)

    return df
#end of chaos.py