"""
structure.py — Maximal Market Structure Feature Module
Institutional-Grade Version

Chunk 1:
    - Swing High / Swing Low detection
    - Multi-level support/resistance levels
    - Support/resistance strength scoring
    - Price position in range
    - Range metrics (rolling, session)
    - Candle structure analytics foundation
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. SWING HIGH / SWING LOW DETECTION
# ============================================================

def detect_swing_points(df: pd.DataFrame, window=3):
    """
    Detects swing highs/lows:
        Swing High: High[t] is highest among t-window ... t+window
        Swing Low:  Low[t] is lowest among t-window ... t+window
    Returns:
        swing_high (array of 0/1)
        swing_low (array of 0/1)
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)

    swing_high = np.zeros(n)
    swing_low = np.zeros(n)

    for i in range(window, n - window):
        seg_high = highs[i - window : i + window + 1]
        seg_low = lows[i - window : i + window + 1]

        if highs[i] == max(seg_high):
            swing_high[i] = 1
        if lows[i] == min(seg_low):
            swing_low[i] = 1

    return swing_high, swing_low


def add_swing_features(df: pd.DataFrame, windows=[2, 3, 5]):
    for w in windows:
        high, low = detect_swing_points(df, window=w)
        df[f"SwingHigh_{w}"] = high
        df[f"SwingLow_{w}"] = low
    return df


# ============================================================
# 2. SUPPORT & RESISTANCE LEVELS (ROLLING)
# ============================================================

def rolling_support_resistance(df: pd.DataFrame, window=20):
    """
    Support = rolling minimum low
    Resistance = rolling maximum high
    """
    support = df["Low"].rolling(window).min()
    resistance = df["High"].rolling(window).max()
    return support, resistance


def add_sr_levels(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        sup, res = rolling_support_resistance(df, w)
        df[f"Support_{w}"] = sup
        df[f"Resistance_{w}"] = res
    return df


# ============================================================
# 3. SUPPORT/RESISTANCE STRENGTH SCORING
# ============================================================

def sr_strength(df: pd.DataFrame, window=20, tolerance=0.001):
    """
    Strength is measured by:
        - How many times price touched or respected support/resistance
        - How recently touches occurred
    """
    lows = df["Low"]
    highs = df["High"]
    sup, res = rolling_support_resistance(df, window)

    support_touches = ((lows - sup).abs() <= tolerance * df["Close"]).astype(int)
    resistance_touches = ((highs - res).abs() <= tolerance * df["Close"]).astype(int)

    strength_sup = support_touches.rolling(window).sum()
    strength_res = resistance_touches.rolling(window).sum()

    return strength_sup, strength_res


def add_sr_strength(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        s_sup, s_res = sr_strength(df, window=w)
        df[f"SupportStrength_{w}"] = s_sup
        df[f"ResistanceStrength_{w}"] = s_res
    return df


# ============================================================
# 4. PRICE POSITION WITHIN RANGE
# ============================================================

def price_position_within_range(df: pd.DataFrame, window=20):
    """
    Computes price position:
        0 = bottom of range
        1 = top of range
    Helps understand breakout/mean reversion environments.
    """
    range_low = df["Low"].rolling(window).min()
    range_high = df["High"].rolling(window).max()

    return (df["Close"] - range_low) / (range_high - range_low + 1e-10)


def add_price_position(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        df[f"PosInRange_{w}"] = price_position_within_range(df, w)
    return df


# ============================================================
# 5. RANGE METRICS (ROLLING & SESSION)
# ============================================================

def rolling_range(df: pd.DataFrame, window=20):
    """
    Rolling range = high - low
    """
    return df["High"].rolling(window).max() - df["Low"].rolling(window).min()


def add_rolling_range(df: pd.DataFrame, windows=[5, 10, 20, 50]):
    for w in windows:
        df[f"Range_{w}"] = rolling_range(df, w)
    return df


def session_range(df: pd.DataFrame):
    """
    Computes daily/session high-low range.
    Assumes df has a datetime index.
    """
    daily_high = df["High"].resample("1D").max()
    daily_low = df["Low"].resample("1D").min()
    session = (daily_high - daily_low).reindex(df.index, method="ffill")
    return session


def add_session_range(df: pd.DataFrame):
    if isinstance(df.index, pd.DatetimeIndex):
        df["SessionRange"] = session_range(df)
    else:
        df["SessionRange"] = np.nan
    return df


# ============================================================
# 6. CANDLE STRUCTURE FOUNDATION
# ============================================================

def candle_body(df: pd.DataFrame):
    return df["Close"] - df["Open"]


def candle_range(df: pd.DataFrame):
    return df["High"] - df["Low"]


def upper_wick(df: pd.DataFrame):
    return df["High"] - df[["Close", "Open"]].max(axis=1)


def lower_wick(df: pd.DataFrame):
    return df[["Close", "Open"]].min(axis=1) - df["Low"]


def add_candle_structure(df: pd.DataFrame):
    df["Body"] = candle_body(df)
    df["Range"] = candle_range(df)
    df["UpperWick"] = upper_wick(df)
    df["LowerWick"] = lower_wick(df)
    df["BodyToRange"] = df["Body"] / (df["Range"] + 1e-10)
    df["WickRatio"] = (df["UpperWick"] + df["LowerWick"]) / (df["Range"] + 1e-10)
    return df


# ============================================================
# 7. STRUCTURE DISPATCHER (CHUNK 1)
# ============================================================

def add_structure_features(df: pd.DataFrame):
    """
    Chunk 1 Market Structure:
        - Swing highs/lows
        - Support/resistance
        - SR strength
        - Price in range
        - Range metrics
        - Candle geometry
    """

    df = add_swing_features(df)
    df = add_sr_levels(df)
    df = add_sr_strength(df)
    df = add_price_position(df)
    df = add_rolling_range(df)
    df = add_session_range(df)
    df = add_candle_structure(df)

    return df

# ============================================================
# 8. BREAKOUTS & BREAKDOWNS
# ============================================================

def breakout_detection(df: pd.DataFrame, window=20):
    """
    Breakout = Close crosses above rolling resistance.
    Breakdown = Close crosses below rolling support.
    """
    support = df["Low"].rolling(window).min()
    resistance = df["High"].rolling(window).max()

    close = df["Close"]

    breakout = ((close.shift(1) <= resistance.shift(1)) & (close > resistance)).astype(int)
    breakdown = ((close.shift(1) >= support.shift(1)) & (close < support)).astype(int)

    return breakout, breakdown


def add_breakout_features(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        bo, bd = breakout_detection(df, window=w)
        df[f"Breakout_{w}"] = bo
        df[f"Breakdown_{w}"] = bd
    return df


# ============================================================
# 9. FAILED BREAKOUTS (LIQUIDITY GRABS)
# ============================================================

def liquidity_grab_detector(df: pd.DataFrame, window=20):
    """
    Detects failed breakout/breakdown:
    - Breaks above resistance but closes back below (bull trap)
    - Breaks below support but closes back above (bear trap)
    """
    support = df["Low"].rolling(window).min()
    resistance = df["High"].rolling(window).max()
    close = df["Close"]

    # Bull Trap: wick breaks above but candle closes below resistance
    wick_high = df["High"] > resistance
    close_below = close < resistance
    bull_trap = (wick_high & close_below).astype(int)

    # Bear Trap: wick breaks below but candle closes above support
    wick_low = df["Low"] < support
    close_above = close > support
    bear_trap = (wick_low & close_above).astype(int)

    return bull_trap, bear_trap


def add_liquidity_grab_features(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        bull, bear = liquidity_grab_detector(df, window=w)
        df[f"BullTrap_{w}"] = bull
        df[f"BearTrap_{w}"] = bear
    return df


# ============================================================
# 10. RANGE COMPRESSION & EXPANSION SIGNALS
# ============================================================

def compression_signal(df: pd.DataFrame, window=20):
    """
    Compression = rolling range / ATR equivalent is small.
    Expansion = sudden increase in range.
    """
    rolling_rng = df["High"].rolling(window).max() - df["Low"].rolling(window).min()
    atr = df["TrueRange"].rolling(window).mean()

    # Compression: narrow range relative to ATR
    compress = (rolling_rng / (atr + 1e-10)).rolling(window).mean()

    # Expansion: breakout volatility
    expand = df["TrueRange"] / (df["TrueRange"].rolling(window).mean() + 1e-10)

    return compress, expand


def add_compression_expansion(df: pd.DataFrame, windows=[10, 20, 50]):
    for w in windows:
        comp, exp = compression_signal(df, window=w)
        df[f"Compression_{w}"] = comp
        df[f"Expansion_{w}"] = exp
    return df


# ============================================================
# 11. CANDLE IMBALANCE STRUCTURE
# ============================================================

def imbalance_metrics(df: pd.DataFrame):
    """
    Candle imbalance:
        - Body direction
        - Upper/lower wick dominance
        - Body dominance over range
    """
    body = df["Body"]
    range_ = df["Range"]

    upper = df["UpperWick"]
    lower = df["LowerWick"]

    df["WickImbalance"] = (upper - lower) / (range_ + 1e-10)
    df["BodyImbalance"] = body / (range_ + 1e-10)
    df["CandleDirection"] = np.sign(body)

    return df


def add_imbalance_features(df: pd.DataFrame):
    return imbalance_metrics(df)


# ============================================================
# 12. IMPULSE–RESPONSE STRUCTURE
# ============================================================

def impulse_response(df: pd.DataFrame, window=5):
    """
    Models the relationship between strong directional impulse
    and subsequent reaction.

    Impulse = large body candle
    Response = retracement/continuation after impulse candle
    """
    body = df["Body"]
    range_ = df["Range"]

    impulse = (abs(body) > range_.rolling(window).mean()).astype(int)

    # Response: returns after impulse
    future_ret = df["Close"].pct_change().shift(-1)

    continuation = ((impulse == 1) & (np.sign(body) == np.sign(future_ret))).astype(int)
    reversal = ((impulse == 1) & (np.sign(body) != np.sign(future_ret))).astype(int)

    return impulse, continuation, reversal


def add_impulse_response_features(df: pd.DataFrame, windows=[5, 10]):
    for w in windows:
        imp, cont, rev = impulse_response(df, window=w)
        df[f"Impulse_{w}"] = imp
        df[f"Continuation_{w}"] = cont
        df[f"Reversal_{w}"] = rev
    return df


# ============================================================
# 13. REJECTION SIGNATURES (WICK-DRIVEN)
# ============================================================

def rejection_signals(df: pd.DataFrame, wick_multiplier=2):
    """
    Rejection candles signal strong reversals:
        - Long upper wick compared to body → bearish rejection
        - Long lower wick compared to body → bullish rejection
    """
    body = abs(df["Body"])
    upper = df["UpperWick"]
    lower = df["LowerWick"]

    bullish_reject = ((lower > wick_multiplier * body) & (df["Close"] > df["Open"])).astype(int)
    bearish_reject = ((upper > wick_multiplier * body) & (df["Close"] < df["Open"])).astype(int)

    return bullish_reject, bearish_reject


def add_rejection_features(df: pd.DataFrame):
    bull, bear = rejection_signals(df)
    df["BullReject"] = bull
    df["BearReject"] = bear
    return df


# ============================================================
# 14. STRUCTURE REGIME LABELS
# ============================================================

def structure_regime(df: pd.DataFrame, window=20):
    """
    Regimes:
        +1 = trending (range expanding)
         0 = balanced (range stable)
        -1 = compressing (range narrowing)
    """
    rng = df["Range"].rolling(window).mean()
    trend = rng.diff()

    regime = np.zeros(len(df))
    regime[trend > 0] = 1
    regime[trend < 0] = -1
    return regime


def add_structure_regime(df: pd.DataFrame):
    df["StructureRegime_20"] = structure_regime(df, 20)
    df["StructureRegime_50"] = structure_regime(df, 50)
    return df


# ============================================================
# 15. STRUCTURE DISPATCHER (CHUNK 2)
# ============================================================

def add_structure_features_chunk2(df: pd.DataFrame):
    """
    Chunk 2 Market Structure:
        - Breakouts & breakdowns
        - Liquidity grabs (failed breakouts)
        - Range compression/expansion
        - Candle imbalance
        - Impulse–response loops
        - Wick rejection signals
        - Structure regimes
    """

    df = add_breakout_features(df)
    df = add_liquidity_grab_features(df)
    df = add_compression_expansion(df)
    df = add_imbalance_features(df)
    df = add_impulse_response_features(df)
    df = add_rejection_features(df)
    df = add_structure_regime(df)

    return df
