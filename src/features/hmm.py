"""
hmm.py — Maximal HMM Regime Modeling Module
Institutional-Grade Version

Chunk 1:
    - Feature preparation for HMM (trend & volatility embeddings)
    - Standardization utilities
    - Gaussian HMM model wrapper
    - Fit, predict states, predict probabilities
"""

import numpy as np
import pandas as pd

# Support both package-style imports (e.g., features.hmm) and standalone
# usage where these modules live in the same folder.
try:  # pragma: no cover
    from .volatility import ewma_volatility
    from .trend import rolling_regression_slope
except ImportError:  # pragma: no cover
    # When loaded in a non-package context (e.g., via importlib by path),
    # ensure this directory is importable.
    import os
    import sys
    _HERE = os.path.dirname(__file__)
    if _HERE and _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from volatility import ewma_volatility
    from trend import rolling_regression_slope

# If hmmlearn is unavailable, we provide a fallback warning.
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    GaussianHMM = None

def prepare_hmm_features(df: pd.DataFrame, vol_window=20, slope_window=20):
    """
    Constructs a (T x F) matrix of stationary-ish features 
    appropriate for Hidden Markov Models.

    Features:
        - ewma_vol (volatility level)
        - abs_ret (return magnitude)
        - slope (trend velocity)
    """

    close = df["Close"]

    ewma_vol = ewma_volatility(df, lambda_=0.94)
    abs_ret = close.pct_change().abs().fillna(0)

    slope = rolling_regression_slope(close.values, slope_window)
    slope = pd.Series(slope).fillna(0)

    X = np.column_stack([
        ewma_vol,
        abs_ret.values,
        slope.values
    ])

    return X

def standardize_features(X):
    """
    Standardizes input features for HMM:
        X_std = (X - mean) / std
    """
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) + 1e-10
    return (X - mean) / std, mean, std

def fit_hmm_model(X, n_states=3, covariance_type="full", random_state=42):
    """
    Fits a Gaussian HMM to the feature matrix X.

    Returns:
        - fitted model
        - smoothed state probabilities (gamma)
        - most likely state sequence (viterbi)
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn not installed. Install via pip install hmmlearn")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        random_state=random_state,
        n_iter=200
    )

    model.fit(X)

    # Smoothed probabilities (posterior state probabilities)
    posteriors = model.predict_proba(X)

    # Most likely states (Viterbi path)
    states = model.predict(X)

    return model, states, posteriors

def add_hmm_regimes(df: pd.DataFrame, n_states=3):
    """
    Adds HMM regime features to df:
        - HMM_State: most likely state
        - HMM_P_i: probability of each hidden state
    """

    # 1) Create feature matrix
    X = prepare_hmm_features(df)

    # 2) Standardize for HMM stability
    X_std, mean, std = standardize_features(X)

    # 3) Fit HMM
    model, states, posteriors = fit_hmm_model(X_std, n_states=n_states)

    # 4) Assign outputs
    df["HMM_State"] = states

    # Create probability columns
    for i in range(n_states):
        df[f"HMM_P{i}"] = posteriors[:, i]

    return df, model

# ============================================================
# 5. TRANSITION MATRIX & REGIME DURATION ANALYSIS
# ============================================================

def extract_transition_matrix(model):
    """
    Extracts the state transition matrix (n_states x n_states)
    from a fitted GaussianHMM model.
    """
    return model.transmat_


def expected_regime_durations(transition_matrix):
    """
    Expected duration for each regime:
        E[D_i] = 1 / (1 - P_ii)

    Where P_ii is the probability of remaining in regime i.
    """
    durations = []
    for i in range(transition_matrix.shape[0]):
        p_stay = transition_matrix[i, i]
        if p_stay < 1:
            durations.append(1 / (1 - p_stay))
        else:
            durations.append(np.inf)
    return np.array(durations)


# ============================================================
# 6. REGIME PERSISTENCE & SWITCHING RISK
# ============================================================

def compute_persistence_index(states):
    """
    Persistence index measures how stable a regime is over time.
    """
    persistence = np.zeros(len(states))
    count = 1

    for i in range(1, len(states)):
        if states[i] == states[i-1]:
            count += 1
        else:
            count = 1
        persistence[i] = count

    return persistence


def compute_switching_risk(states):
    """
    High switching risk if states change frequently:
        SwitchingRisk_t = 1 if state_t != state_{t-1}
    """
    risk = np.zeros(len(states))
    for i in range(1, len(states)):
        risk[i] = 1 if states[i] != states[i-1] else 0
    return risk


# ============================================================
# 7. SHOCK PROBABILITY (FAST TRANSITIONS)
# ============================================================

def compute_shock_probability(posteriors, threshold=0.3):
    """
    Shock = posterior for dominant state drops sharply.

    ShockProb_t = 1 if max(posterior_{t}) - max(posterior_{t-1}) < -threshold
    """
    max_p = posteriors.max(axis=1)
    shock = np.zeros(len(max_p))

    for i in range(1, len(max_p)):
        if max_p[i] - max_p[i-1] < -threshold:
            shock[i] = 1

    return shock


# ============================================================
# 8. PROBABILITY-WEIGHTED EMBEDDING (for ML models)
# ============================================================

def probability_weighted_embedding(posteriors):
    """
    Creates a smoothed feature representation of regimes:
        Embed_t = weighted sum of state vectors.

    If n_states = 3, embedding dims = 3.
    """
    return posteriors  # Direct embedding as P(state_i)


# ============================================================
# 9. REGIME VOLATILITY (State Uncertainty Over Time)
# ============================================================

def regime_uncertainty(posteriors, window=20):
    """
    Computes uncertainty based on entropy of posterior distribution.
    Higher entropy = less confidence in regime classification.
    """
    entropy = -np.sum(posteriors * np.log(posteriors + 1e-10), axis=1)
    return pd.Series(entropy).rolling(window).mean()


# ============================================================
# 10. MULTI-HMM ENSEMBLE SUPPORT
# ============================================================

def fit_multiple_hmms(X_std, n_states_list=[2, 3, 4]):
    """
    Fits multiple HMMs with different numbers of hidden states.
    Returns:
        - dict of models
        - dict of states
        - dict of posterior probabilities
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn not installed.")

    models = {}
    states = {}
    probs = {}

    for n in n_states_list:
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            random_state=42,
            n_iter=200
        )

        model.fit(X_std)

        models[n] = model
        states[n] = model.predict(X_std)
        probs[n] = model.predict_proba(X_std)

    return models, states, probs


# ============================================================
# 11. ADVANCED DISPATCHER
# ============================================================

def add_hmm_advanced_features(df: pd.DataFrame, model, states, posteriors):
    """
    Adds advanced HMM-derived regime features:
        - Transition matrix entries
        - Expected durations
        - Persistence index
        - Switching risk
        - Shock probability
        - Regime uncertainty (entropy)
        - Probability-weighted embedding
    """

    # ---- Transition matrix & expected duration ----
    P = extract_transition_matrix(model)
    durations = expected_regime_durations(P)

    for i in range(len(durations)):
        df[f"HMM_RegimeDuration_{i}"] = durations[i]

    # ---- Persistence & switching ----
    df["HMM_Persistence"] = compute_persistence_index(states)
    df["HMM_SwitchRisk"] = compute_switching_risk(states)

    # ---- Shock probability ----
    df["HMM_ShockProb"] = compute_shock_probability(posteriors)

    # ---- Probabilistic embedding ----
    embed = probability_weighted_embedding(posteriors)
    for i in range(embed.shape[1]):
        df[f"HMM_Embed_{i}"] = embed[:, i]

    # ---- Regime uncertainty ----
    df["HMM_Uncertainty"] = regime_uncertainty(posteriors)

    return df


# ============================================================
# 12. MASTER DISPATCHER (Chunk 2)
# ============================================================

def add_hmm_features(df: pd.DataFrame, n_states=3):
    """
    Full HMM regime enhancement pipeline:
        1) Prepare features
        2) Standardize
        3) Fit HMM
        4) Add base states & probabilities
        5) Add advanced regime analytics
    """
    # ---- Prepare inputs ----
    X = prepare_hmm_features(df)
    X_std, mean, std = standardize_features(X)

    # ---- Fit model ----
    model, states, posteriors = fit_hmm_model(X_std, n_states=n_states)

    # ---- Basic outputs ----
    df["HMM_State"] = states
    for i in range(n_states):
        df[f"HMM_P{i}"] = posteriors[:, i]

    # ---- Advanced features ----
    df = add_hmm_advanced_features(df, model, states, posteriors)

    return df, model
#end of hmm.py