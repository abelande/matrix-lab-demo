"""
feature_descriptions.py — Feature Name Lookup & Description Table

Provides human-readable descriptions for quantitative features
used in SHAP analysis and model interpretation.
"""

import numpy as np
import pandas as pd

# ============================================================
# Feature Description Lookup Dictionary
# ============================================================

FEATURE_DESCRIPTIONS = {
    # Regression / Trend
    "RegR2": ("Regression R²", "Goodness of linear trend fit (1=perfect trend, 0=no trend)"),
    "RegSlope": ("Regression Slope", "Direction/steepness of linear trend"),
    "VASlope": ("Volatility-Adjusted Slope", "Trend slope normalized by volatility"),

    # Moving Averages
    "EMA": ("Exponential Moving Average", "Price smoothing with recent bias"),
    "WMA": ("Weighted Moving Average", "Price smoothing with linear weights"),
    "TEMA": ("Triple EMA", "Fast-responding smoothed price"),
    "KAMA": ("Kaufman Adaptive MA", "Volatility-adaptive trend filter"),
    "SMA": ("Simple Moving Average", "Equal-weighted price average"),

    # Momentum / Strength
    "RSI": ("Relative Strength Index", "Overbought/oversold momentum (0-100)"),
    "TSI": ("True Strength Index", "Double-smoothed momentum oscillator"),
    "TSI2": ("True Strength Index v2", "Alternative TSI calculation"),
    "TCI": ("Trend Confirmation Index", "Trend strength confirmation"),
    "MOM": ("Momentum", "Price change over N periods"),
    "ROC": ("Rate of Change", "Percentage price change"),

    # Volatility
    "Vol": ("Volatility", "Price dispersion / risk measure"),
    "EWMA": ("EWMA Volatility", "Exponentially weighted volatility"),
    "VolCluster": ("Volatility Cluster", "Volatility regime persistence"),
    "VolRegime": ("Volatility Regime", "Current vol state (low/normal/high/extreme)"),
    "ATR": ("Average True Range", "Volatility based on high-low-close"),
    "BB": ("Bollinger Band", "Volatility bands around moving average"),

    # Support / Resistance
    "ResistanceStrength": ("Resistance Strength", "Strength of overhead price resistance"),
    "SupportStrength": ("Support Strength", "Strength of price floor support"),
    "LocalMax": ("Local Maximum", "Recent peak detection"),
    "LocalMin": ("Local Minimum", "Recent trough detection"),

    # Price Action
    "Return": ("Return", "Price change over period"),
    "LogRet": ("Log Return", "Logarithmic price change"),
    "returns": ("Returns", "Simple returns"),
    "logret": ("Log Returns", "Log returns"),
    "UpperWick": ("Upper Wick", "Rejection from highs (selling pressure)"),
    "LowerWick": ("Lower Wick", "Rejection from lows (buying pressure)"),
    "BodySize": ("Candle Body Size", "Open-close range"),

    # Time / Session
    "H": ("Hour", "Hour of day"),
    "DOW": ("Day of Week", "Day of week (0=Mon, 4=Fri)"),
    "SessionRange": ("Session Range", "High-low range for session"),
    "DOWReturn": ("Day-of-Week Return", "Returns by weekday"),

    # Regime
    "Regime_Combo": ("Combined Regime", "Trend + Volatility regime state"),
    "VolRegClass": ("Volatility Regime Class", "Volatility state classification"),
    "TrendRegClass": ("Trend Regime Class", "Trend state classification"),
    "RegimePersistence": ("Regime Persistence", "Duration in current regime"),
    "RegimeShift": ("Regime Shift", "Regime transition signal"),

    # Other Indicators
    "DPO": ("Detrended Price Oscillator", "Price deviation from trend"),
    "UDRatio": ("Up/Down Ratio", "Ratio of up vs down moves"),
    "Expansion": ("Range Expansion", "Volatility breakout measure"),
    "Compression": ("Range Compression", "Volatility squeeze measure"),
    "JumpIdx": ("Jump Index", "Sudden price movement detection"),
    "SwingHigh": ("Swing High", "Recent swing high level"),
    "SwingLow": ("Swing Low", "Recent swing low level"),
    "MACD": ("MACD", "Trend-following momentum indicator"),
    "ADX": ("Average Directional Index", "Trend strength (not direction)"),
    "CCI": ("Commodity Channel Index", "Deviation from statistical mean"),
    "OBV": ("On-Balance Volume", "Cumulative volume flow"),
    "VWAP": ("Volume-Weighted Avg Price", "Average price weighted by volume"),
}


# ============================================================
# Project-Specific Features (myquantlab)
# ============================================================

PROJECT_FEATURES = {
    # Moving Averages (Additional)
    "HMA": ("Hull Moving Average", "Low-lag smoothed price with reduced delay"),

    # Momentum / Oscillators
    "AO": ("Awesome Oscillator", "Momentum using 34/5 period SMA difference"),
    "TRIX": ("Triple Smoothed EMA", "Rate of change of triple-smoothed EMA"),
    "Ultimate": ("Ultimate Oscillator", "Multi-timeframe momentum oscillator"),
    "WilliamsR": ("Williams %R", "Overbought/oversold oscillator (0 to -100)"),
    "StochK": ("Stochastic %K", "Fast stochastic oscillator"),
    "StochD": ("Stochastic %D", "Slow stochastic (smoothed %K)"),

    # Trend Analysis
    "Curv": ("Curvature", "Second derivative of price trend"),
    "Accel": ("Acceleration", "Rate of change of slope"),
    "TrendPersist": ("Trend Persistence", "Duration of current trend direction"),
    "NSR": ("Noise-to-Signal Ratio", "Trend clarity measure"),
    "SlopeFlip": ("Slope Flip", "Trend direction change detection"),
    "RegDev": ("Regression Deviation", "Price deviation from regression line"),

    # Volatility (Extended)
    "Var": ("Variance", "Squared price dispersion"),
    "VolOfVol": ("Volatility of Volatility", "Second-order volatility"),
    "Parkinson": ("Parkinson Volatility", "High-low range-based volatility"),
    "GK": ("Garman-Klass Volatility", "OHLC-based volatility estimator"),
    "RS": ("Rogers-Satchell Volatility", "Drift-independent volatility"),
    "YZ": ("Yang-Zhang Volatility", "Combined overnight/intraday volatility"),
    "VolKurt": ("Volatility Kurtosis", "Fat-tailedness of volatility distribution"),
    "VolConvex": ("Volatility Convexity", "Curvature of volatility term structure"),
    "VolShift": ("Volatility Shift", "Change in volatility regime"),
    "LogVol": ("Log Volatility", "Natural log of volatility"),
    "ZVol": ("Z-Score Volatility", "Standardized volatility"),
    "VolRatio": ("Volatility Ratio", "Ratio of short/long term volatility"),
    "VolRet": ("Volatility Return", "Return adjusted by volatility"),
    "TrueRange": ("True Range", "Max of high-low, |high-prev_close|, |low-prev_close|"),
    "ATRCompress": ("ATR Compression", "Volatility squeeze based on ATR"),
    "BBWidth": ("Bollinger Band Width", "Distance between upper/lower bands"),
    "ShockRec": ("Shock Recovery", "Speed of recovery from volatility spike"),
    "UpsideVol": ("Upside Volatility", "Volatility of positive returns only"),
    "DownsideVol": ("Downside Volatility", "Volatility of negative returns only"),
    "BV": ("Bipower Variation", "Jump-robust volatility estimate"),
    "JumpComp": ("Jump Component", "Magnitude of price jumps"),
    "XMove": ("Extreme Move", "Large price movement indicator"),

    # Support / Resistance (Extended)
    "Support": ("Support Level", "Nearest support price level"),
    "Resistance": ("Resistance Level", "Nearest resistance price level"),
    "PosInRange": ("Position in Range", "Where price sits between support/resistance"),
    "Breakout": ("Breakout", "Price breaking above resistance"),
    "Breakdown": ("Breakdown", "Price breaking below support"),
    "BullTrap": ("Bull Trap", "Failed breakout reversal"),
    "BearTrap": ("Bear Trap", "Failed breakdown reversal"),
    "Extension": ("Extension", "Price extended from mean"),

    # Price Action / Candlestick
    "Body": ("Candle Body", "Open-to-close range"),
    "Range": ("Candle Range", "High-to-low range"),
    "BodyToRange": ("Body to Range Ratio", "Body size relative to total range"),
    "WickRatio": ("Wick Ratio", "Upper wick vs lower wick"),
    "WickImbalance": ("Wick Imbalance", "Asymmetry between wicks"),
    "BodyImbalance": ("Body Imbalance", "Consecutive body direction bias"),
    "CandleDirection": ("Candle Direction", "Bullish (+1) or bearish (-1) candle"),
    "Impulse": ("Impulse Move", "Strong directional price movement"),
    "Continuation": ("Continuation Pattern", "Trend continuation signal"),
    "Reversal": ("Reversal Pattern", "Trend reversal signal"),
    "BullReject": ("Bullish Rejection", "Rejection from lows (buying pressure)"),
    "BearReject": ("Bearish Rejection", "Rejection from highs (selling pressure)"),
    "StructureRegime": ("Structure Regime", "Market structure state"),

    # Time Features
    "DayOfWeek": ("Day of Week", "Numeric day (0=Mon, 6=Sun)"),
    "DayOfMonth": ("Day of Month", "Day number in month (1-31)"),
    "Month": ("Month", "Month number (1-12)"),
    "Quarter": ("Quarter", "Quarter number (1-4)"),
    "Hour": ("Hour", "Hour of day (0-23)"),
    "Minute": ("Minute", "Minute of hour (0-59)"),
    "MonthReturn": ("Monthly Return Pattern", "Historical returns for this month"),
    "TurnOfMonth": ("Turn of Month", "Near month-end/start indicator"),
    "EOM": ("End of Month", "End of month indicator"),
    "EOQ": ("End of Quarter", "End of quarter indicator"),
    "DaysToHoliday": ("Days to Holiday", "Trading days until next holiday"),
    "DaysFromHoliday": ("Days from Holiday", "Trading days since last holiday"),

    # Complexity / Fractal Features
    "Hurst": ("Hurst Exponent", "Trend persistence measure (>0.5=trending, <0.5=mean-reverting)"),
    "FracDim": ("Fractal Dimension", "Price path complexity measure"),
    "DFA": ("Detrended Fluctuation Analysis", "Long-range correlation measure"),
    "ApEn": ("Approximate Entropy", "Time series regularity/predictability"),
    "SampEn": ("Sample Entropy", "Improved entropy measure"),
    "Lyap": ("Lyapunov Exponent", "Chaos/sensitivity to initial conditions"),

    # Labels / Targets
    "t1": ("Event End Time", "Triple barrier event horizon"),
    "dir_h5": ("Direction Horizon 5", "Forward return direction (5-period)"),
    "dir_h10": ("Direction Horizon 10", "Forward return direction (10-period)"),
    "dir_h20": ("Direction Horizon 20", "Forward return direction (20-period)"),
    "dir_h50": ("Direction Horizon 50", "Forward return direction (50-period)"),
    "dir_h100": ("Direction Horizon 100", "Forward return direction (100-period)"),
}

# Merge project features into main dictionary
FEATURE_DESCRIPTIONS.update(PROJECT_FEATURES)


# ============================================================
# Helper Functions
# ============================================================

def parse_feature_name(feature: str) -> tuple:
    """
    Parse feature name into base, window, and modifiers.

    Returns:
        (base_name, window, is_zscore, is_pct)
    """
    parts = feature.replace("_zscore", "").replace("_pct", "").split("_")

    base_parts = []
    window = None
    for p in parts:
        if p.isdigit():
            window = p
        else:
            base_parts.append(p)

    base = "_".join(base_parts) if base_parts else feature
    is_zscore = "_zscore" in feature
    is_pct = "_pct" in feature

    return base, window, is_zscore, is_pct


def get_feature_description(feature: str) -> tuple:
    """
    Get full name and description for a feature.

    Returns:
        (full_name, description)
    """
    base, window, is_zscore, is_pct = parse_feature_name(feature)

    if base in FEATURE_DESCRIPTIONS:
        full_name, description = FEATURE_DESCRIPTIONS[base]
    else:
        full_name = base
        description = "—"

    if window:
        full_name = f"{full_name} ({window}-period)"

    if is_zscore:
        full_name += " [z-score]"
        description += " (standardized)"
    if is_pct:
        full_name += " [%]"

    return full_name, description


def print_top_features_table(features, importances=None, top_n=None):
    """
    Print formatted table of top features with descriptions.

    Args:
        features: List or array of feature names
        importances: Optional array of importance values
        top_n: Optional limit (if None, prints all)
    """
    if top_n:
        features = features[:top_n]
        if importances is not None:
            importances = importances[:top_n]

    print("\n" + "=" * 110)
    if importances is not None:
        print(f"{'Rank':<5} {'Feature':<30} {'Importance':<12} {'Full Name':<30} {'Description':<30}")
    else:
        print(f"{'Rank':<5} {'Feature':<30} {'Full Name':<35} {'Description':<35}")
    print("=" * 110)

    for i, feat in enumerate(features, 1):
        full_name, desc = get_feature_description(feat)
        full_name = full_name[:33] + ".." if len(full_name) > 35 else full_name
        desc = desc[:33] + ".." if len(desc) > 35 else desc

        if importances is not None:
            imp = importances[i-1]
            print(f"{i:<5} {feat:<30} {imp:<12.6f} {full_name:<30} {desc:<30}")
        else:
            print(f"{i:<5} {feat:<30} {full_name:<35} {desc:<35}")

    print("=" * 110 + "\n")


def get_top_features_df(feature_names, importances, top_n=25):
    """
    Return a DataFrame of top features with descriptions.

    Args:
        feature_names: Array/list of all feature names
        importances: Array of importance values (same order as feature_names)
        top_n: Number of top features to return

    Returns:
        pd.DataFrame with columns: Feature, Importance, Full_Name, Description
    """
    top_idx = np.argsort(importances)[-top_n:][::-1]

    rows = []
    for rank, idx in enumerate(top_idx, 1):
        feat = feature_names[idx]
        full_name, desc = get_feature_description(feat)
        rows.append({
            "Rank": rank,
            "Feature": feat,
            "Importance": importances[idx],
            "Full_Name": full_name,
            "Description": desc
        })

    return pd.DataFrame(rows)
# ============================================================
# Section 12 SHAP Analysis Guide Table
# ============================================================
def print_section12_guide():
    """Print Section 12 SHAP Analysis guide table."""
    print("=" * 100)
    print("  SECTION 12 — Model Interpretation & Explainability Guide")
    print("=" * 100)
    print(f"{'Part':<25} {'Purpose':<45} {'Output':<30}")
    print("-" * 100)
    
    guide = [
        ("1. Fit Final Model", "Retrain model on full data for SHAP (not CV splits)", "Fitted final_model"),
        ("2. SHAP Setup", "Create TreeExplainer, compute SHAP for 2000 samples", "shap_values (2000, 666, 3)"),
        ("3a. Summary (Class 1)", "Beeswarm: feature importance + direction for positive class", "Plot: dots colored by value"),
        ("3b. Summary (All)", "Compare importance across all 3 classes", "Plot: multi-class comparison"),
        ("4. Bar Chart", "Simple ranked importance (no direction)", "Plot: horizontal bars"),
        ("5. Dependence Plots", "How each top feature's SHAP varies with its value", "Plot: scatter (high→high/low SHAP)"),
        ("6. Permutation Importance", "Model-agnostic importance (shuffle & measure drop)", "Plot: bar chart, validates SHAP"),
        ("7. Regime Analysis", "Feature importance changes across market regimes", "Plot: heatmap, Table: top 25"),
        ("8. Local Explanation", "Explain ONE specific prediction", "Plots: waterfall, decision, 3D"),
        ("9. Save Artifacts", "Persist importance scores for later use", "JSON file saved"),
        ("10. Indian Run", "Full pairwise interaction analysis (Option A + QME)", "Plots: R², Cohen's d heatmaps"),
    ]
    
    for part, purpose, output in guide:
        # Truncate if too long
        purpose = purpose[:43] + ".." if len(purpose) > 45 else purpose
        output = output[:28] + ".." if len(output) > 30 else output
        print(f"{part:<25} {purpose:<45} {output:<30}")
    
    print("=" * 100)
    print()