"""
pipeline.py — Unified Research Pipeline
Maximal Institutional-Grade Version

Chunk 1:
    - PipelineConfig dataclass
    - Data integrity checks
    - Precompute OHLC returns, ATR, TR
    - Run all feature modules in correct order
    - Align & clean final dataset
    - Produce feature matrix X ready for modeling
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Import all feature modules.
# These modules may either live inside a package (features.*) or in the same
# folder as this file (as provided in this project snapshot). We support both.
try:  # pragma: no cover
    from features.trend import add_trend_features
    from features.volatility import (
        add_basic_vol_features,
        add_advanced_vol_features,
        add_tail_and_shape_features,
    )
    from features.structure import (
        add_structure_features,
        add_structure_features_chunk2,
    )
    from features.seasonality import add_seasonality_features
    from features.regimes import add_regime_features
    from features.hmm import add_hmm_features
    from features.clustering import add_cluster_regimes, add_advanced_cluster_features
    from features.chaos import add_chaos_features
    from features.normalization import apply_normalization
except ModuleNotFoundError:  # pragma: no cover
    # When loaded outside a package (e.g., via importlib by path), make sure
    # this directory is importable so that `import trend`, etc. works.
    import os
    import sys
    _HERE = os.path.dirname(__file__)
    if _HERE and _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    from trend import add_trend_features
    from volatility import (
        add_basic_vol_features,
        add_advanced_vol_features,
        add_tail_and_shape_features,
    )
    from structure import add_structure_features, add_structure_features_chunk2
    from seasonality import add_seasonality_features
    from regimes import add_regime_features
    from hmm import add_hmm_features
    from clustering import add_cluster_regimes, add_advanced_cluster_features
    from chaos import add_chaos_features
    from normalization import apply_normalization

@dataclass
class PipelineConfig:
    # Core feature groups
    trend: bool = True
    volatility_basic: bool = True
    volatility_adv: bool = True
    volatility_shape: bool = True
    structure_basic: bool = True
    structure_adv: bool = True
    seasonality: bool = True
    regimes: bool = True
    hmm: bool = False       # Optional (slow)
    clustering: bool = False
    clustering_adv: bool = False
    chaos: bool = True

    # Normalization
    normalize: bool = True
    normalization_method: str = "zscore"

    # Data cleaning
    drop_nans: bool = True
    drop_inf: bool = True

    # Robustness: prevent the pipeline from collapsing to an empty DataFrame
    # when some long-horizon features are mostly NaN on shorter samples.
    drop_all_nan_cols: bool = True
    max_nan_col_frac: float = 0.60  # drop feature cols with >60% NaN
    min_row_non_nan_frac: float = 0.70  # min non-NaN fraction across features
    impute_nans: bool = True
    impute_method: str = "median"  # median|mean|ffill

    # Logging
    verbose: bool = True

    # Registration for metadata
    enabled_modules: list = field(default_factory=list)

def compute_prelim_features(df: pd.DataFrame):
    """
    Adds:
        - returns
        - log returns
        - true range
        - ATR
    """
    df["Return"] = df["Close"].pct_change()
    df["LogRet"] = np.log(df["Close"]).diff()

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))

    df["TrueRange"] = np.nanmax(
        np.vstack([high_low, high_close, low_close]),
        axis=0
    )

    df["ATR_14"] = df["TrueRange"].rolling(14).mean()
    df["VolRet"] = df["Return"].rolling(20).std()

    return df

def run_feature_module(df, config, module_fn, module_name):
    if config.verbose:
        print(f"• Running module: {module_name}")
    try:
        df = module_fn(df)
        config.enabled_modules.append(module_name)
    except Exception as e:
        print(f"Module {module_name} failed: {e}")
    return df

def build_features(df: pd.DataFrame, config: PipelineConfig):
    """
    Main unified feature builder.
    Applies all enabled modules in dependency order.
    """

    df = df.copy()

    # Step 0 – Precomputations
    df = compute_prelim_features(df)

    # ---- Trend ----
    if config.trend:
        df = run_feature_module(df, config, add_trend_features, "trend")

    # ---- Volatility ----
    if config.volatility_basic:
        df = run_feature_module(df, config, add_basic_vol_features, "vol_basic")

    if config.volatility_adv:
        df = run_feature_module(df, config, add_advanced_vol_features, "vol_advanced")

    if config.volatility_shape:
        df = run_feature_module(df, config, add_tail_and_shape_features, "vol_shape")

    # ---- Market Structure ----
    if config.structure_basic:
        df = run_feature_module(df, config, add_structure_features, "structure_basic")

    if config.structure_adv:
        df = run_feature_module(df, config, add_structure_features_chunk2, "structure_advanced")

    # ---- Seasonality ----
    if config.seasonality:
        df = run_feature_module(df, config, add_seasonality_features, "seasonality")

    # ---- Deterministic Regimes ----
    if config.regimes:
        df = run_feature_module(df, config, add_regime_features, "regimes")

    # ---- HMM (optional, slow) ----
    if config.hmm:
        df, _ = add_hmm_features(df)
        config.enabled_modules.append("hmm")

    # ---- Clustering (optional) ----
    if config.clustering:
        df, _, _ = add_cluster_regimes(df)
        config.enabled_modules.append("cluster_basic")

    if config.clustering_adv:
        df, _ = add_advanced_cluster_features(df)
        config.enabled_modules.append("cluster_advanced")

    # ---- Chaos & Complexity ----
    if config.chaos:
        df = run_feature_module(df, config, add_chaos_features, "chaos")

    # ---- Normalization ----
    if config.normalize:
        feature_cols = [
            col for col in df.columns
            if col not in ["Open", "High", "Low", "Close", "Volume"]
        ]
        df = apply_normalization(df, feature_cols, method=config.normalization_method)

    # ---- Cleaning ----
    exclude = ["Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in df.columns if c not in exclude]

    if config.drop_inf:
        df = df.replace([np.inf, -np.inf], np.nan)

    # Drop feature columns that are mostly NaN (common on short samples when many
    # long-horizon indicators are enabled).
    if config.drop_all_nan_cols and feature_cols:
        nan_frac = df[feature_cols].isna().mean()
        drop_cols = nan_frac[(nan_frac >= 1.0) | (nan_frac > config.max_nan_col_frac)].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            feature_cols = [c for c in feature_cols if c not in drop_cols]

    # Row filtering and imputation.
    if feature_cols:
        # Keep rows that have "enough" observed data.
        row_non_nan_frac = 1.0 - df[feature_cols].isna().mean(axis=1)
        df = df.loc[row_non_nan_frac >= config.min_row_non_nan_frac].copy()

        if config.impute_nans:
            if config.impute_method == "ffill":
                df[feature_cols] = df[feature_cols].ffill()
            elif config.impute_method == "mean":
                df[feature_cols] = df[feature_cols].apply(lambda s: s.fillna(s.mean()) if pd.api.types.is_numeric_dtype(s) else s)
            else:  # median (default)
                df[feature_cols] = df[feature_cols].apply(lambda s: s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s)

    # Final drop (only if requested and imputation is disabled)
    if config.drop_nans and not config.impute_nans:
        df = df.dropna()

    return df

def build_feature_matrix(df: pd.DataFrame, config: PipelineConfig):
    """
    Runs build_features() and returns:
        - X: feature matrix
        - df: full dataframe with metadata
    """
    df = build_features(df, config)

    # Select features (exclude raw OHLC)
    exclude = ["Open", "High", "Low", "Close", "Volume"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()

    return X, df, feature_cols

# ============================================================
# 6. LABEL INTEGRATION (TRIPLE-BARRIER, META-LABELS, TREND LABELS)
# ============================================================

try:  # pragma: no cover
    from labels import (
        triple_barrier, meta_labeling,
    )
except ModuleNotFoundError:  # pragma: no cover
    # In case the project is structured as a package.
    from .labels import (
        triple_barrier,
        meta_labeling,
    )

def add_labels(df: pd.DataFrame,
               method="triple_barrier",
               horizon=20,
               pt_sl=(1, 1),
               min_ret=0.002):
    """
    Adds y labels to df.
    method:
        - 'triple_barrier'
        - 'meta'
        - 'trend'
    """

    if method == "triple_barrier":
        df["label"] = triple_barrier(
            df,
            horizon=horizon,
            pt_sl=pt_sl,
            min_ret=min_ret
        )

    elif method == "meta":
        df["label"] = meta_labeling(
            df,
            horizon=horizon,
            pt_sl=pt_sl
        )

    elif method == "trend":
        df["label"] = trend(df, horizon=horizon)

    else:
        raise ValueError(f"Unknown label method: {method}")

    return df

try:  # pragma: no cover
    from cv.cv_purged import PurgedKFold
except ModuleNotFoundError:  # pragma: no cover
    from cv.cv_purged import PurgedKFold

def get_purged_cv(n_splits=5, embargo=10):
    """
    Returns a PurgedKFold splitter with embargoing.
    """
    return PurgedKFold(
        n_splits=n_splits,
        embargo=embargo
    )

def assign_split_groups(df: pd.DataFrame, mode="time"):
    """
    Assigns group labels for group-aware CV splitting.
    """

    if mode == "time":
        # Each sample gets its chronological index
        return np.arange(len(df))

    if mode == "regime":
        return df["Regime_Combo"].values

    if mode == "hmm":
        return df["HMM_State"].values

    if mode == "cluster":
        return df["ClusterLabel"].values

    raise ValueError(f"Unknown group mode: {mode}")

def build_dataset(df: pd.DataFrame,
                  feature_cols,
                  label_col="label",
                  group_mode="time"):

    X = df[feature_cols].copy().values
    y = df[label_col].copy().values
    groups = assign_split_groups(df, mode=group_mode)

    return X, y, groups

def generate_cv_splits(df, X, y, groups,
                       n_splits=5,
                       embargo=10,
                       shuffle=False):

    cv = get_purged_cv(n_splits=n_splits, embargo=embargo)

    splits = []
    for train_idx, test_idx in cv.split(X, y, groups):
        splits.append((train_idx, test_idx))

    return splits

def run_full_pipeline(df: pd.DataFrame,
                      config: PipelineConfig,
                      label_method="triple_barrier",
                      horizon=20,
                      pt_sl=(1, 1),
                      embargo=10,
                      group_mode="time"):

    # ---- Feature construction ----
    X_df, full_df, feature_cols = build_feature_matrix(df, config)

    # ---- Add labels ----
    full_df = add_labels(full_df,
                         method=label_method,
                         horizon=horizon,
                         pt_sl=pt_sl)

    # ---- Build dataset ----
    X, y, groups = build_dataset(
        full_df,
        feature_cols=feature_cols,
        group_mode=group_mode
    )

    # ---- CV SPLITS ----
    splits = generate_cv_splits(
        full_df,
        X, y, groups,
        n_splits=5,
        embargo=embargo
    )

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "splits": splits,
        "df": full_df,
        "features": feature_cols
    }

def inference_pipeline(df: pd.DataFrame,
                       config: PipelineConfig,
                       feature_cols):
    """
    Applies feature pipeline in inference mode.
    No labels, no CV splitting.
    """

    X_df, full_df, f_cols = build_feature_matrix(df, config)

    # Enforce feature order
    X = X_df[feature_cols].copy().values

    return X, full_df
#end of pipeline.py