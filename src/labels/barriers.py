"""
barriers.py — PT/SL Barrier Detection for Triple-Barrier Labeling
Maximal Institutional Version

Chunk 1:
    - Price path extraction
    - PT/SL barrier checking
    - Barrier hit time detection
    - Vertical barrier fallback
    - Return computation
"""

import numpy as np
import pandas as pd

def get_price_path(close: pd.Series, t0, t1):
    """
    Returns the price path from t0 to t1 inclusive.
    """
    return close.loc[t0:t1]

def apply_barriers_to_event(close: pd.Series,
                            t0,
                            t1,
                            pt,
                            sl):
    """
    Applies PT/SL barriers to a single event.
    Returns:
        label: +1, -1, or 0
        end_time: timestamp of barrier touch or t1
        ret: realized return at barrier or vertical barrier
    """

    # Extract the price path
    price_path = get_price_path(close, t0, t1)

    # Return series relative to initial price
    initial_price = price_path.iloc[0]
    returns = price_path / initial_price - 1

    # PT hit?
    pt_hits = returns[returns >= pt]
    if len(pt_hits) > 0:
        end_time = pt_hits.index[0]
        ret = returns.loc[end_time]
        label = 1
        return label, end_time, ret

    # SL hit?
    sl_hits = returns[returns <= sl]
    if len(sl_hits) > 0:
        end_time = sl_hits.index[0]
        ret = returns.loc[end_time]
        label = -1
        return label, end_time, ret

    # Neither → use vertical barrier return
    end_time = t1
    ret = returns.iloc[-1]
    label = np.sign(ret)  # could be 0  
    return label, end_time, ret

def apply_barriers(close: pd.Series,
                   events: pd.DataFrame):
    """
    Applies PT/SL barriers to all events.

    events contains:
        - index = t0
        - t1
        - pt
        - sl

    Returns DataFrame:
        label
        t_end (barrier hit or t1)
        ret
    """

    results = []

    for t0, row in events.iterrows():
        t1 = row["t1"]
        pt = row["pt"]
        sl = row["sl"]

        if pd.isna(t1):
            continue

        label, end_time, ret = apply_barriers_to_event(
            close,
            t0,
            t1,
            pt,
            sl
        )

        results.append({
            "t0": t0,
            "t1": t1,
            "t_end": end_time,
            "label": label,
            "ret": ret,
            "pt": pt,
            "sl": sl
        })

    if not results:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["t1", "t_end", "label", "ret", "pt", "sl"]).rename_axis("t0")

    results_df = pd.DataFrame(results).set_index("t0")
    return results_df

def safe_return_calc(price_series: pd.Series):
    """
    Computes returns robustly against zeros or corrupted data.
    """
    p0 = price_series.iloc[0]
    if p0 == 0 or np.isnan(p0):
        return np.nan * price_series

    return price_series / p0 - 1

def compute_barrier_outcomes(close: pd.Series,
                             events: pd.DataFrame):
    """
    Wrapper for barrier outcomes.
    Returns only label and t_end for triple-barrier integration.
    """

    out = apply_barriers(close, events)

    return out[["label", "t_end", "ret"]]
#end of barriers.py