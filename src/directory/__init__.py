"""
myquantlab - Quantitative Finance Research Lab
===============================================

A toolkit for regime detection, cross-validation, and ML-based labeling
for financial time series.

Modules:
    features    - Feature engineering (HMM, chaos, fractals, volatility, etc.)
    labels      - Target labeling (triple barrier, forward returns, meta-labeling)
    cv          - Cross-validation (purged, combinatorial, walk-forward)
"""

__version__ = "0.1.0"

# ============================================================
# Features
# ============================================================
from features.hmm import add_hmm_features
from features.chaos import hurst_exponent, approximate_entropy, sample_entropy
from features.fractals import williams_fractals, adaptive_fractals
from features.volatility import ewma_volatility, return_volatility
from features.trend import rolling_regression_slope
from features.regimes import add_trend_regime_class
from features.clustering import add_cluster_regimes
from features.seasonality import add_seasonality_features
from features.structure import add_structure_features
from features.pipeline import build_feature_pipeline

# ============================================================
# Labels
# ============================================================
from labels.triple_barrier import triple_barrier_labeling
from labels.forward_returns import forward_return, trim_forward_returns
from labels.volatility_adj import volatility_adjusted_labels
from labels.meta_labeling import meta_labeling
from labels.multi_horizon import multi_horizon_labels

# ============================================================
# Cross-Validation
# ============================================================
from cv.cv_purged import purged_split
from cv.cv_walkforward import RollingWindowCV
from cv.cv_combinatorial import CombinatorialPurgedCV
from cv.cv_clustered import ClusteredCV
from cv.cv_adaptive_embargo import apply_embargo

# ============================================================
# Public API
# ============================================================
__all__ = [
    # Features
    "add_hmm_features",
    "hurst_exponent",
    "approximate_entropy",
    "sample_entropy",
    "williams_fractals",
    "adaptive_fractals",
    "ewma_volatility",
    "return_volatility",
    "rolling_regression_slope",
    "add_trend_regime_class",
    "add_cluster_regimes",
    "add_seasonality_features",
    "add_structure_features",
    "build_feature_pipeline",
    # Labels
    "triple_barrier_labeling",
    "forward_return",
    "trim_forward_returns",
    "volatility_adjusted_labels",
    "meta_labeling",
    "multi_horizon_labels",
    # CV
    "purged_split",
    "RollingWindowCV",
    "CombinatorialPurgedCV",
    "ClusteredCV",
    "apply_embargo",
]