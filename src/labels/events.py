"""
events.py — Event Creation Engine for Labeling
Maximal Institutional Version

Chunk 1:
    - Basic event selection
    - Volatility-threshold event filtering
    - Vertical barrier assignment
    - Time index alignment & validation
"""

import numpy as np
import pandas as pd
from typing import Union

def validate_time_index(df: pd.DataFrame):
    """
    Ensures dataframe index is:
        - DatetimeIndex
        - Sorted
        - Unique
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted in ascending order")

    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique")

def get_basic_events(df: pd.DataFrame,
                     min_ret: float = 0.001,
                     vol_series: pd.Series = None):
    """
    Basic event selection:
        - Use volatility series if provided
        - Otherwise use |returns| > min_ret threshold
    """

    validate_time_index(df)

    close = df["Close"]

    if vol_series is None:
        ret = close.pct_change().abs()
        t_events = ret[ret > min_ret].index
    else:
        t_events = vol_series[vol_series > min_ret].index

    return pd.Index(t_events)

def get_vertical_barriers(df: pd.DataFrame,
                          t_events: pd.Index,
                          horizon: Union[int, pd.Timedelta]):
    """
    horizon:
        - if int → seconds forward
        - if Timedelta → applied directly

    Returns:
        pd.Series mapping t0 → t1 (vertical barrier time)
    """

    validate_time_index(df)
    close_index = df.index

    if isinstance(horizon, int):
        horizon = pd.Timedelta(seconds=horizon)

    t1 = {}

    for t0 in t_events:
        t_end = t0 + horizon
        loc = close_index.searchsorted(t_end)

        if loc < len(close_index):
            t1[t0] = close_index[loc]
        else:
            t1[t0] = pd.NaT

    return pd.Series(t1)

def filter_events_with_valid_barriers(t1: pd.Series):
    """
    Removes events whose horizon exceeds dataset end.
    """
    return t1.dropna().index

def build_event_table(df: pd.DataFrame,
                      t_events: pd.Index,
                      vol: pd.Series,
                      horizon: Union[int, pd.Timedelta],
                      pt_sl=(1, 1)):
    """
    Returns DataFrame of events:
        - t1 vertical barrier
        - pt (profit-take threshold)
        - sl (stop-loss threshold)
    """

    # 1. Vertical barrier
    t1 = get_vertical_barriers(df, t_events, horizon)
    valid_events = filter_events_with_valid_barriers(t1)

    # 2. Extract vol for valid events
    vol_ev = vol.loc[valid_events]

    # 3. Compute PT/SL as multiples of volatility
    pt = pt_sl[0] * vol_ev
    sl = -pt_sl[1] * vol_ev

    events = pd.DataFrame({
        "t1": t1.loc[valid_events],
        "vol": vol_ev,
        "pt": pt,
        "sl": sl
    })

    return events

def filter_event_spacing(events: pd.DataFrame, min_distance: int = None):
    """
    Ensures events are spaced by at least min_distance rows.
    """
    if min_distance is None:
        return events

    idx = events.index
    keep = [idx[0]]

    for i in range(1, len(idx)):
        if (idx[i] - keep[-1]).seconds >= min_distance:
            keep.append(idx[i])

    return events.loc[keep]

def filter_overlapping_events(events: pd.DataFrame):
    """
    Removes overlapping events by ensuring:
        t0_next >= t1_previous
    """
    if len(events) == 0:
        return events

    events_sorted = events.sort_index()
    keep = [events_sorted.index[0]]

    last_end = events_sorted.iloc[0]["t1"]

    for i in range(1, len(events_sorted)):
        t0_next = events_sorted.index[i]
        if t0_next >= last_end:
            keep.append(t0_next)
            last_end = events_sorted.loc[t0_next]["t1"]

    return events_sorted.loc[keep]
# end of events.py