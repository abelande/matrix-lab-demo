"""
clustering.py — Maximal Unsupervised Regime Module
Institutional-Grade Version

Chunk 1:
    - Feature preparation for clustering
    - PCA compression
    - KMeans clustering
    - Cluster assignment
    - Cluster distance metrics
    - Stability scores
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Support both package-style imports (e.g., features.clustering) and standalone
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

def prepare_cluster_features(df: pd.DataFrame, vol_window=20, slope_window=20):
    """
    Creates feature matrix for clustering.

    Features:
        - ewma volatility
        - abs returns
        - regression slope
        - daily range
        - normalized deviation from rolling median
    """
    close = df["Close"]

    ew_vol = ewma_volatility(df, lambda_=0.94)
    abs_ret = close.pct_change().abs().fillna(0)

    slope = rolling_regression_slope(close.values, slope_window)
    slope = pd.Series(slope).fillna(0)

    rng = (df["High"] - df["Low"]).fillna(0)

    median_dev = (close - close.rolling(50).median()) / (close.rolling(50).std() + 1e-10)

    X = np.column_stack([
        ew_vol,
        abs_ret.values,
        slope.values,
        rng.values,
        median_dev.values
    ])

    return X

def scale_features(X):
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0) + 1e-10
    return (X - mean) / std, mean, std

def apply_pca(X, n_components=3):
    """
    PCA for geometric structure of regimes.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def fit_kmeans(X, n_clusters=4, random_state=42):
    """
    Fits a KMeans cluster model and returns cluster labels and distances.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)

    # Distance to cluster center (anomaly metric)
    distances = km.transform(X).min(axis=1)

    return km, labels, distances

def cluster_stability(labels):
    """
    Stability = number of consecutive periods in same cluster.
    """
    stab = np.zeros(len(labels))
    count = 1

    stab[0] = 1

    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            count += 1
        else:
            count = 1
        stab[i] = count

    return stab

def add_cluster_regimes(df: pd.DataFrame, n_clusters=4, pca_components=3):
    """
    Full unsupervised clustering pipeline:
        1) Prepare stationar-ish features
        2) Scale them
        3) Apply PCA compression
        4) Run KMeans clustering
        5) Add cluster labels + distances + stability
    """
    # ---- Step 1: raw feature prep ----
    X = prepare_cluster_features(df)

    # ---- Step 2: standardize ----
    X_scaled, mean, std = scale_features(X)

    # ---- Step 3: PCA ----
    X_pca, pca = apply_pca(X_scaled, n_components=pca_components)

    # ---- Step 4: KMeans ----
    km, labels, distances = fit_kmeans(X_pca, n_clusters=n_clusters)

    # ---- Save to df ----
    df["ClusterLabel"] = labels
    df["ClusterDist"] = distances
    df["ClusterStability"] = cluster_stability(labels)

    # ---- Optional: store PCA components in df ----
    for i in range(pca_components):
        df[f"PCA_{i}"] = X_pca[:, i]

    return df, km, pca

# ============================================================
# 7. GAUSSIAN MIXTURE MODEL CLUSTERING (SOFT PROBABILITIES)
# ============================================================

from sklearn.mixture import GaussianMixture

def fit_gmm(X, n_clusters=4, random_state=42):
    """
    Fits a Gaussian Mixture Model and returns:
        - hard labels
        - soft probabilities (responsibilities)
        - distance metric (negative log-likelihood)
    """
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state
    )

    gmm.fit(X)

    # Hard cluster assignments
    labels = gmm.predict(X)

    # Soft probabilities (P(cluster | X_t))
    probs = gmm.predict_proba(X)

    # Distance metric (anomaly score)
    nll = -gmm.score_samples(X)

    return gmm, labels, probs, nll

def compute_transition_matrix(labels, n_clusters):
    """
    Computes empirical cluster transition probabilities.
    """
    P = np.zeros((n_clusters, n_clusters))

    for t in range(1, len(labels)):
        i = labels[t-1]
        j = labels[t]
        P[i, j] += 1

    # Normalize rows
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)

    return P

def compute_cluster_drift(labels, window=20):
    """
    Drift_t = probability of changing clusters within a rolling window.
    """
    drift = np.zeros(len(labels))

    for t in range(window, len(labels)):
        window_labels = labels[t-window:t]
        changes = np.sum(window_labels[1:] != window_labels[:-1])
        drift[t] = changes / window

    return drift

def cluster_fingerprints(df, labels, n_clusters):
    """
    Computes summary statistics for each cluster:
        - mean volatility
        - mean slope
        - mean returns
        - mean range
    """

    fingerprints = {}

    for c in range(n_clusters):
        idx = labels == c
        subset = df.loc[idx]

        fingerprints[c] = {
            "Volatility": subset["Close"].pct_change().rolling(20).std().mean(),
            "Slope": subset["Close"].pct_change().mean(),
            "Range": (subset["High"] - subset["Low"]).mean(),
            "Return": subset["Close"].pct_change().mean()
        }

    return fingerprints

def hybrid_anomaly_score(kmeans_dist, gmm_nll, alpha=0.5):
    """
    Hybrid anomaly score:
        score = alpha * normalized KMeans dist
              + (1-alpha) * normalized GMM NLL
    """
    kd = (kmeans_dist - np.mean(kmeans_dist)) / (np.std(kmeans_dist) + 1e-10)
    gn = (gmm_nll - np.mean(gmm_nll)) / (np.std(gmm_nll) + 1e-10)

    return alpha * kd + (1 - alpha) * gn

def ensemble_clustering(df, X_scaled, pca_components=3, n_clusters=4):
    """
    Runs both KMeans and GMM clustering on PCA embeddings.
    Produces:
        - KMeans labels & distances
        - GMM labels & probabilities
        - Ensemble anomaly score
        - Multi-regime consistency metrics
    """

    # Apply PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)

    # KMeans
    km, km_labels, km_dist = fit_kmeans(X_pca, n_clusters=n_clusters)

    # GMM
    gmm, gmm_labels, gmm_probs, gmm_nll = fit_gmm(X_pca, n_clusters=n_clusters)

    # Ensemble anomaly score
    anomaly = hybrid_anomaly_score(km_dist, gmm_nll)

    # Consistency (agreement between KMeans and GMM)
    consistency = (km_labels == gmm_labels).astype(int)

    return {
        "pca": pca,
        "X_pca": X_pca,
        "km": km,
        "km_labels": km_labels,
        "km_dist": km_dist,
        "gmm": gmm,
        "gmm_labels": gmm_labels,
        "gmm_probs": gmm_probs,
        "gmm_nll": gmm_nll,
        "anomaly": anomaly,
        "consistency": consistency
    }

def add_advanced_cluster_features(df: pd.DataFrame, n_clusters=4, pca_components=3):
    """
    Full advanced pipeline:
        - Prepare + scale features
        - PCA
        - KMeans clustering
        - GMM clustering (soft probabilities)
        - Cluster transition matrix
        - Drift metrics
        - Cluster fingerprints
        - Ensemble anomaly score
        - Regime consistency
    """

    # Features
    X = prepare_cluster_features(df)
    X_scaled, mean, std = scale_features(X)

    # Ensemble system
    results = ensemble_clustering(df, X_scaled, pca_components, n_clusters)

    # Save PCA components to df
    for i in range(pca_components):
        df[f"PCA_{i}"] = results["X_pca"][:, i]

    # Save KMeans
    df["KM_Label"] = results["km_labels"]
    df["KM_Dist"] = results["km_dist"]

    # Save GMM
    df["GMM_Label"] = results["gmm_labels"]
    for i in range(n_clusters):
        df[f"GMM_Prob_{i}"] = results["gmm_probs"][:, i]
    df["GMM_NLL"] = results["gmm_nll"]

    # Ensemble anomaly & consistency
    df["ClusterAnomaly"] = results["anomaly"]
    df["ClusterConsistency"] = results["consistency"]

    # Cluster transition matrix
    P = compute_transition_matrix(results["km_labels"], n_clusters)
    for i in range(n_clusters):
        for j in range(n_clusters):
            df[f"KM_P_{i}_{j}"] = P[i, j]

    # Drift
    df["ClusterDrift"] = compute_cluster_drift(results["km_labels"])

    # Fingerprints
    fp = cluster_fingerprints(df, results["km_labels"], n_clusters)
    # (Fingerprints are dicts; not added to df because they are metadata)

    return df, results
# End of clustering.py