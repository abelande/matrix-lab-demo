"""
meta_labeling.py — Meta-Label Generation Engine
Maximal Institutional Version

Chunk 1:
    - Meta-label theory (De Prado)
    - Extract event outcomes from triple-barrier engine
    - Align primary predictions with event outcomes
    - Compute meta-labels
    - Provide meta-data for training secondary models
"""

import numpy as np
import pandas as pd

from .events import validate_time_index
from .triple_barrier import triple_barrier_labeling

def align_primary_signals(primary_signals: pd.Series,
                          events: pd.DataFrame):
    """
    Aligns primary model predictions with event index.
    
    primary_signals:
        Series indexed by timestamps with values:
            +1 = predicted long
            -1 = predicted short
             0 = no trade

    events:
        DataFrame with event timestamps as index (t0)

    Returns:
        Series aligned with events.index containing primary signals at t0.
    """

    return primary_signals.reindex(events.index).fillna(0)
    
# 1  if primary model predicted correct direction  
# -1 if primary model predicted wrong direction  
# 0  if primary model did not predict → ignored in training
    

def compute_meta_label(event_labels: pd.Series,
                       primary_sides: pd.Series):
    """
    Computes meta labels:

    For each event:
        primary_side = ±1 or 0
        event_label  = outcome of triple-barrier labeling

    meta = 1  if label * primary_side > 0
    meta = -1 if label * primary_side < 0
    meta = 0  if primary_side == 0 (ignored)
    """

    aligned_label = event_labels.reindex(primary_sides.index).fillna(0)

    # If primary model didn't predict signal, meta label = 0
    mask_trade = primary_sides != 0

    meta = pd.Series(0, index=primary_sides.index, dtype=float)

    # correct direction
    meta.loc[mask_trade & ((aligned_label * primary_sides) > 0)] = 1

    # incorrect direction
    meta.loc[mask_trade & ((aligned_label * primary_sides) < 0)] = -1

    return meta

def build_meta_features(events_with_outcomes: pd.DataFrame):
    """
    Constructs features for meta-label model:

    Possible features (all optional for later model training):
        - event_vol
        - pt, sl
        - ret
        - time_to_barrier
        - time_to_vertical_barrier
    """

    df = events_with_outcomes.copy()

    df["event_vol"] = df["vol"]

    df["time_to_barrier"] = (df["t_end"] - df.index).dt.total_seconds()

    df["time_to_vertical"] = (df["t1"] - df.index).dt.total_seconds()
    # ret, pt/sl already in dataframe

    return df

def meta_labeling(df: pd.DataFrame,
                  primary_signals: pd.Series,
                  pt_sl=(1, 1),
                  min_ret=0.001,
                  horizon=86400,
                  vol_span=100):
    """
    Full meta-labeling pipeline:
        1. Compute triple-barrier labels
        2. Extract event table + outcomes
        3. Align primary signals with events
        4. Compute meta labels
        5. Build meta features for the meta model

    Returns:
        meta_df: DataFrame with meta labels and event features
    """

    validate_time_index(df)

    # Step 1: triple-barrier labeling
    df_labeled, events, outcomes = triple_barrier_labeling(
        df,
        pt_sl=pt_sl,
        min_ret=min_ret,
        horizon=horizon,
        vol_span=vol_span
    )

    # Step 2: align primary signals to event times
    primary_sides = align_primary_signals(primary_signals, events)

    # Step 3: extract event labels from outcomes
    event_labels = outcomes["label"]

    # Step 4: compute meta labels
    meta = compute_meta_label(event_labels, primary_sides)

    # Step 5: build meta feature set for training
    meta_features = build_meta_features(outcomes)

    # Combine everything
    meta_df = meta_features.copy()
    meta_df["primary_side"] = primary_sides
    meta_df["meta_label"] = meta

    return meta_df
#end of meta_labeling.py