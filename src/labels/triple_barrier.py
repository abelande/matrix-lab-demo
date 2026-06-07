"""
triple_barrier.py — Core Triple-Barrier Labeling Engine
Maximal Institutional Version

Chunk 1:
    - Event-driven label generation
    - Integration with events.py & barriers.py
    - Volatility-based PT/SL scaling
    - Triple-barrier label output
    - Metadata for meta-labeling and diagnostics
"""

import numpy as np
import pandas as pd

from .events import (
    validate_time_index,
    get_basic_events,
    build_event_table,
)

from .barriers import (
    compute_barrier_outcomes
)

from .volatility import ewma_volatility

def generate_triple_barrier_events(df: pd.DataFrame,
                                   pt_sl=(1, 1),
                                   min_ret=0.001,
                                   horizon=86400,     # 1 day default (seconds)
                                   vol_span=100):
    """
    Step 1:
        - Validate time index
        - Compute volatility
        - Select events
        - Build event table with PT/SL based on vol

    Returns:
        events: DataFrame indexed by t0 with pt/sl/t1/vol
    """

    validate_time_index(df)

    close = df["Close"]
    vol = ewma_volatility(close, span=vol_span)

    # Select t0 events where vol > noise threshold
    t_events = get_basic_events(df, min_ret=min_ret, vol_series=vol)

    # Build event table
    events = build_event_table(
        df,
        t_events=t_events,
        vol=vol,
        horizon=horizon,
        pt_sl=pt_sl
    )

    return events

def apply_triple_barrier(df: pd.DataFrame,
                         events: pd.DataFrame):
    """
    Step 2:
        Apply PT/SL/vertical barrier detection per event.

    Returns:
        event_outcomes DataFrame with:
            - label
            - t_end (hit time or vertical barrier)
            - ret (realized return)
            - pt, sl (thresholds)
            - t1 (vertical barrier)
    """
    close = df["Close"]

    # Compute barriers
    outcomes = compute_barrier_outcomes(close, events)

    # Merge with event metadata
    merged = events.join(outcomes, how="left")

    return merged

def attach_labels_to_df(df: pd.DataFrame,
                        event_outcomes: pd.DataFrame):
    """
    Step 3:
        Merge outcomes back into original df.

    Output columns added to df:
        label
        t1
        t_end
        ret
        pt
        sl
        vol
    """

    # Expand event_outcomes into df index
    aligned = df.join(event_outcomes, how="left")

    # Extract only the label column for training
    aligned["label"] = aligned["label"].fillna(0)

    return aligned

def triple_barrier_labeling(df: pd.DataFrame,
                            pt_sl=(1, 1),
                            min_ret=0.001,
                            horizon=86400,
                            vol_span=100):
    """
    Full triple-barrier labeling workflow:
        1. Event generation
        2. Apply triple-barrier logic
        3. Merge labels into df

    Returns:
        df_with_labels
        events
        event_outcomes
    """

    # Step 1: Event creation
    events = generate_triple_barrier_events(
        df,
        pt_sl=pt_sl,
        min_ret=min_ret,
        horizon=horizon,
        vol_span=vol_span
    )

    # Step 2: Barrier application
    outcomes = apply_triple_barrier(df, events)

    # Step 3: Add results to df
    df_labeled = attach_labels_to_df(df, outcomes)

    return df_labeled, events, outcomes

def apply_side(df: pd.DataFrame, side: pd.Series = None):
    """
    Allows directional labels:
        side = +1 → long
        side = -1 → short
        side = 0 → neutral (no trade)

    If side is None:
        all events assumed direction-neutral.

    You can pass any model-predicted side as well.
    """

    if side is None:
        df["side"] = 1  # default long-only
    else:
        df["side"] = side.reindex(df.index).fillna(0)

    # directional label = label * side
    df["dir_label"] = df["label"] * df["side"]

    return df
#end of tripple_barrier.py
