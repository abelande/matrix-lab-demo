"""
labels.py — Maximal Label Engineering Module
Institutional-Grade Version

Chunk 1:
    - Event creation (vertical barrier)
    - Volatility-adjusted dynamic PT/SL barriers
    - Triple-barrier label generation
    - Side/Direction-neutral labeling
    - Meta-label compatibility outputs
"""
from features.regimes import add_trend_regime_class
from features.hmm import add_hmm_features
from features.clustering import add_cluster_regimes
import numpy as np
import pandas as pd

def daily_volatility(close, span=100):
    """
    Exponentially Weighted Volatility estimator (EWMA).
    """
    returns = close.pct_change()
    vol = returns.ewm(span=span).std()
    return vol

def get_vertical_barriers(close, timestamps, horizon_seconds):
    """
    Assigns a vertical barrier timestamp:
        t + horizon
    """
    vertical_barriers = {}

    for t in timestamps:
        end_time = t + pd.Timedelta(seconds=horizon_seconds)
        # Find next timestamp in the index
        idx = close.index.searchsorted(end_time)
        if idx < len(close):
            vertical_barriers[t] = close.index[idx]
        else:
            vertical_barriers[t] = np.nan

    return pd.Series(vertical_barriers)

def get_events(close,
               volatility,
               pt_sl=(1, 1),
               horizon_seconds=60*60*24,  # 1 day default in seconds
               min_ret=0.001):
    """
    Creates events with:
        - t0 (start time)
        - vertical barrier (t1)
        - target PT & SL thresholds
    """

    # Filter out tiny returns
    t_events = volatility[volatility > min_ret].index

    # Vertical barriers
    t1 = get_vertical_barriers(close, t_events, horizon_seconds)

    # Dynamic PT/SL barriers
    pts = pt_sl[0] * volatility.loc[t_events]
    sls = -pt_sl[1] * volatility.loc[t_events]

    events = pd.DataFrame({
        "t1": t1,
        "pt": pts,
        "sl": sls
    })

    return events.dropna()

def apply_barriers(close, events):
    """
    For each event:
        - Check if price hits PT or SL before t1.
        - If neither hits → use return at t1.
    """

    out = []

    for t0, row in events.iterrows():
        t1 = row["t1"]
        pt = row["pt"]
        sl = row["sl"]

        if pd.isna(t1):
            continue

        # Price path between t0 and t1
        prices = close.loc[t0:t1]

        # Returns relative to t0 price
        returns = prices / prices.iloc[0] - 1

        # PT hit
        pt_hit = returns[returns > pt]
        # SL hit
        sl_hit = returns[returns < sl]

        # Outcome:
        if len(pt_hit) > 0:
            label = 1
            end_time = pt_hit.index[0]

        elif len(sl_hit) > 0:
            label = -1
            end_time = sl_hit.index[0]

        else:
            # Use return at vertical barrier
            final_ret = returns.iloc[-1]
            label = np.sign(final_ret)
            end_time = t1

        out.append({
            "t0": t0,
            "t1": t1,
            "label": label,
            "end_time": end_time
        })

    return pd.DataFrame(out).set_index("t0")

def apply_triple_barrier(df,
                         horizon_seconds=60*60*24,
                         pt_sl=(1, 1),
                         min_ret=0.001):

    close = df["Close"]
    vol = daily_volatility(close)

    # 1. Events
    events = get_events(
        close,
        volatility=vol,
        pt_sl=pt_sl,
        horizon_seconds=horizon_seconds,
        min_ret=min_ret
    )

    # 2. Label assignment
    labels = apply_barriers(close, events)

    # Result aligned with df
    df = df.join(labels["label"], how="left")

    return df["label"]

def prepare_meta_labels(df, primary_model_preds, horizon_seconds=60*60*24):
    """
    Primary model gives entry signals.
    Meta model learns which entries are worth taking.
    """

    close = df["Close"]
    vol = daily_volatility(close)

    events = get_events(close, vol, horizon_seconds=horizon_seconds)

    labels = apply_barriers(close, events)

    pred = primary_model_preds.reindex(labels.index).fillna(0)

    meta_y = pd.Series(0, index=labels.index, dtype="int8")
    mask = pred != 0
    meta_y.loc[mask] = np.where(labels.loc[mask, "label"] == pred.loc[mask], 1, -1)
   
    return meta_y

def apply_trend_label(df, horizon_bars=20):
    """
    Label = sign of return over fixed horizon.
    """
    ret = df["Close"].pct_change(horizon_bars).shift(-horizon_bars)
    return np.sign(ret)

# ============================================================
# 8. META-LABELING ENGINE (PER LÓPEZ DE PRADO)
# ============================================================

def meta_label_events(base_events, base_predictions):
    """
    Constructs meta-labels:
        base_predictions ∈ {1, -1, 0}

    Meta label:
        1  → primary model direction was correct
        -1 → incorrect
        0  → no trade (ignored)
    """
    df_meta = base_events.copy()
    df_meta["primary"] = base_predictions.reindex(df_meta.index).fillna(0)

    df_meta["meta_label"] = 0
    mask_trades = df_meta["primary"] != 0

    df_meta.loc[mask_trades, "meta_label"] = (
        df_meta.loc[mask_trades, "label"] == df_meta.loc[mask_trades, "primary"]
    ).astype(int)

    df_meta.loc[mask_trades & (df_meta["meta_label"] == 0), "meta_label"] = -1

    return df_meta["meta_label"]

def apply_meta_labeling(df,
                        primary_preds,
                        horizon_seconds=60*60*24,
                        pt_sl=(1,1),
                        min_ret=0.001):
    """
    Full pipeline:
        1. create triple-barrier events
        2. apply barriers → event labels
        3. meta-label on top of primary predictions
    """

    close = df["Close"]
    vol = daily_volatility(close)

    events = get_events(
        close,
        volatility=vol,
        pt_sl=pt_sl,
        horizon_seconds=horizon_seconds,
        min_ret=min_ret
    )

    event_labels = apply_barriers(close, events) #has "label" column
    events_full = events.join(event_labels["label"], how="inner")
    pred = primary_preds.reindex(events_full.index).fillna(0) 
    return meta_label_events(events_full, pred)

def apply_vol_adjusted_trend_label(df, horizon_bars=20, vol_window=20):
    """
    Directional label scaled by volatility.
    Intuition:
        ret_adj = return / volatility
        label = sign(ret_adj)
    """
    returns = df["Close"].pct_change(horizon_bars).shift(-horizon_bars)
    vol = df["Close"].pct_change().rolling(vol_window).std()

    adj = returns / (vol + 1e-10)
    return np.sign(adj)

def apply_regime_conditioned_label(df, horizon_bars=20):
   if "TrendRegClass_3" not in df.columns:
    df = add_trend_regime_class(df)
    """
    Creates a label that measures trend but only within a given regime.
    Different regimes → different trend strength.

    Returns:
        label ∈ {-2, -1, 0, 1, 2}
    """
    ret = df["Close"].pct_change(horizon_bars).shift(-horizon_bars)
    direction = np.sign(ret)

    # Trend regime: -1,0,+1
    trend_reg = df["TrendRegClass_3"]

    return direction * (trend_reg.replace(0,1))  # amplify trend inside trend regimes

def apply_hmm_weighted_label(df, horizon_bars=20): 
   if "HMM_Uncertainty" not in df.columns: #auto-generate if missing
    df, _ = add_hmm_features(df)
    """
    Weighted directional returns using HMM state probabilities.
    Good when regime changes modify reward structure.
    """

    ret = df["Close"].pct_change(horizon_bars).shift(-horizon_bars)
    direction = np.sign(ret).fillna(0)

    # Weighted by state stability (uncertainty = entropy)
    if "HMM_Uncertainty" in df.columns:
        unc = df["HMM_Uncertainty"]
    else:
        unc = 0

    weights = 1 / (1 + unc)  # lower weight in high uncertainty

    return direction * weights

def apply_cluster_conditioned_label(df, horizon_bars=20):
   if "ClusterStability" not in df.columns: #auto-generate if missing
    df, _, _ = add_cluster_regimes(df)
    """
    Uses cluster regimes to condition trend label.
    Some clusters are trend-friendly, others are chop.
    """
    ret = df["Close"].pct_change(horizon_bars).shift(-horizon_bars)
    direction = np.sign(ret).fillna(0)

    # Cluster stability (from clustering module)
    if "ClusterStability" in df.columns:
        stab = df["ClusterStability"]
        stab = stab / (stab.max() + 1e-10)
    else:
        stab = 1

    return direction * stab

def ensemble_labels(*labels, weights=None):
    """
    Combines multiple labels:
        label_ensemble = weighted_sign(sum_i weights_i * labels_i)
    """

    L = np.column_stack(labels)

    if weights is None:
        weights = np.ones(L.shape[1])

    combined = np.dot(L, weights)

    # sign of weighted sum = final label
    return np.sign(combined)

def build_labels(
    df,
    method="triple",
    horizon_seconds=60*60*24,
    horizon_bars=20,
    pt_sl=(1, 1),
    primary_preds=None,
    min_ret=0.001,
):
    if method == "triple":
        return apply_triple_barrier(
            df,
            horizon_seconds=horizon_seconds,
            pt_sl=pt_sl,
            min_ret=min_ret,
        )

    elif method == "meta":
        if primary_preds is None:
            raise ValueError("primary_preds required for meta labeling")
        return apply_meta_labeling(
            df,
            primary_preds=primary_preds,
            horizon_seconds=horizon_seconds,
            pt_sl=pt_sl,
            min_ret=min_ret,
        )

    elif method == "trend":
        return apply_trend_label(df, horizon_bars=horizon_bars)

    elif method == "trend_vol":
        return apply_vol_adjusted_trend_label(df, horizon_bars=horizon_bars)

    elif method == "trend_regime":
        return apply_regime_conditioned_label(df, horizon_bars=horizon_bars)

    elif method == "trend_hmm":
        return apply_hmm_weighted_label(df, horizon_bars=horizon_bars)

    elif method == "trend_cluster":
        return apply_cluster_conditioned_label(df, horizon_bars=horizon_bars)

    else:
        raise ValueError(f"Unknown label method: {method}")
