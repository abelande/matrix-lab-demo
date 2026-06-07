"""
Signal Extensions Module
========================
Advanced signal processing utilities for the RGNB trading system.

Extensions:
1. Mean Reversion - Detect oversold/overbought bounce opportunities
2. Conviction Scoring - Compute bull-bear probability spreads
3. Divergence Detection - Identify model vs price disagreements
4. Risk Sizing - Dynamic position sizing based on opposing probability
5. Regime Filtering - Filter signals based on regime and probability constraints

Usage:
    from extensions.signal_extensions import (
        generate_reversion_signals,
        compute_conviction_metrics,
        detect_divergences,
        compute_risk_adjusted_size,
        apply_regime_filter,
        run_all_extensions
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EXTENSION 1: Mean Reversion Signal Generator
# =============================================================================

def generate_reversion_signals(
    df: pd.DataFrame,
    price_col: str = "price",
    lookback: int = 10,
    bear_extreme: float = 0.75,
    bull_extreme: float = 0.75,
    price_move_thresh: float = 0.005,
    plot: bool = False
) -> pd.DataFrame:
    """
    Generate mean reversion signals when probability extremes diverge from price.

    Logic:
    - LONG REVERSION: P(bear) extremely high but price hasn't dropped much -> oversold bounce
    - SHORT REVERSION: P(bull) extremely high but price hasn't risen much -> overbought fade

    Args:
        df: DataFrame with ensemble_prob_bull, ensemble_prob_bear, and price
        price_col: Column name for price
        lookback: Bars to measure price change over
        bear_extreme: Threshold for "extremely bearish" probability
        bull_extreme: Threshold for "extremely bullish" probability
        price_move_thresh: Minimum price % move to confirm direction
        plot: Whether to generate visualization

    Returns:
        DataFrame with reversion signals and diagnostics
    """
    result = df.copy()

    # Calculate recent price movement
    result["price_pct_change"] = result[price_col].pct_change(lookback)

    # Detect extremes
    result["bear_extreme"] = result["ensemble_prob_bear"] > bear_extreme
    result["bull_extreme"] = result["ensemble_prob_bull"] > bull_extreme

    # Price hasn't confirmed the move
    result["price_not_dropped"] = result["price_pct_change"] > -price_move_thresh
    result["price_not_risen"] = result["price_pct_change"] < price_move_thresh

    # Reversion signals
    result["reversion_long"] = (result["bear_extreme"] & result["price_not_dropped"]).astype(int)
    result["reversion_short"] = (result["bull_extreme"] & result["price_not_risen"]).astype(int)

    # Combined signal
    result["reversion_signal"] = result["reversion_long"] - result["reversion_short"]

    # Reversion strength
    result["reversion_strength"] = np.where(
        result["reversion_signal"] == 1,
        result["ensemble_prob_bear"] + result["price_pct_change"],
        np.where(
            result["reversion_signal"] == -1,
            result["ensemble_prob_bull"] - result["price_pct_change"],
            0
        )
    )

    # Summary stats
    n_long = (result["reversion_signal"] == 1).sum()
    n_short = (result["reversion_signal"] == -1).sum()
    logger.info(f"Mean Reversion: {n_long} long reversions, {n_short} short reversions")

    if plot and price_col in result.columns:
        _plot_reversion_signals(result, price_col)

    return result


def _plot_reversion_signals(df: pd.DataFrame, price_col: str):
    """Plot reversion signals on price chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[price_col], label="Price", alpha=0.7)

    long_rev = df[df["reversion_signal"] == 1]
    short_rev = df[df["reversion_signal"] == -1]

    ax.scatter(long_rev.index, long_rev[price_col], marker="^", color="green",
               s=50, label=f"Long Reversion (n={len(long_rev)})", zorder=5)
    ax.scatter(short_rev.index, short_rev[price_col], marker="v", color="red",
               s=50, label=f"Short Reversion (n={len(short_rev)})", zorder=5)

    ax.set_title("Mean Reversion Signals")
    ax.set_ylabel("Price")
    ax.legend()
    plt.show()


# =============================================================================
# EXTENSION 2: Conviction Scoring
# =============================================================================

def compute_conviction_metrics(
    df: pd.DataFrame,
    print_summary: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """
    Compute conviction metrics: P(bull) - P(bear) for each model and ensemble.

    Args:
        df: DataFrame with model probabilities
        print_summary: Whether to print analysis summary
        plot: Whether to generate visualization

    Returns:
        DataFrame with conviction scores
    """
    result = df.copy()

    # Model-level convictions
    models = ["rf", "xgb", "lgb", "logit", "mlp"]
    conviction_cols = []

    for m in models:
        bull_col = f"{m}_prob_bull"
        bear_col = f"{m}_prob_bear"
        if bull_col in result.columns and bear_col in result.columns:
            result[f"{m}_conviction"] = result[bull_col] - result[bear_col]
            conviction_cols.append(f"{m}_conviction")

    # Ensemble conviction
    if "ensemble_prob_bull" in result.columns and "ensemble_prob_bear" in result.columns:
        result["ensemble_conviction"] = result["ensemble_prob_bull"] - result["ensemble_prob_bear"]

    # Model agreement
    if conviction_cols:
        result["models_bullish"] = (result[conviction_cols] > 0).sum(axis=1)
        result["models_bearish"] = (result[conviction_cols] < 0).sum(axis=1)
        result["conviction_agreement"] = result[["models_bullish", "models_bearish"]].max(axis=1) / len(conviction_cols)

    # Conviction zones
    result["conviction_zone"] = pd.cut(
        result.get("ensemble_conviction", result[conviction_cols[0]] if conviction_cols else pd.Series(0, index=result.index)),
        bins=[-1, -0.3, -0.1, 0.1, 0.3, 1],
        labels=["Strong Bear", "Moderate Bear", "Neutral", "Moderate Bull", "Strong Bull"]
    )

    if print_summary:
        _print_conviction_summary(result)

    if plot:
        _plot_conviction(result)

    return result


def _print_conviction_summary(df: pd.DataFrame):
    """Print conviction analysis summary."""
    print("=" * 60)
    print("  CONVICTION SCORING ANALYSIS")
    print("=" * 60)

    if "ensemble_conviction" in df.columns:
        conv = df["ensemble_conviction"]
        print(f"\nEnsemble Conviction Stats:")
        print(f"  Mean:   {conv.mean():+.4f}")
        print(f"  Std:    {conv.std():.4f}")
        print(f"  Min:    {conv.min():+.4f}")
        print(f"  Max:    {conv.max():+.4f}")

    print(f"\nConviction Zone Distribution:")
    zone_dist = df["conviction_zone"].value_counts().sort_index()
    for zone, count in zone_dist.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {str(zone):15s}: {count:4d} ({pct:5.1f}%) {bar}")

    if "conviction_agreement" in df.columns:
        print(f"\nModel Agreement: {df['conviction_agreement'].mean():.1%} avg")

    print("=" * 60)


def _plot_conviction(df: pd.DataFrame):
    """Plot conviction distribution and time series."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    conv_col = "ensemble_conviction" if "ensemble_conviction" in df.columns else None
    if conv_col is None:
        return

    axes[0].hist(df[conv_col], bins=50, alpha=0.7, edgecolor="black")
    axes[0].axvline(0, color="black", linestyle="-", linewidth=2)
    axes[0].axvline(-0.3, color="red", linestyle="--", alpha=0.5)
    axes[0].axvline(0.3, color="green", linestyle="--", alpha=0.5)
    axes[0].set_title("Conviction Distribution")
    axes[0].set_xlabel("Conviction (P(bull) - P(bear))")

    axes[1].plot(df[conv_col], alpha=0.7)
    axes[1].axhline(0, color="black", linestyle="-")
    axes[1].fill_between(df.index, df[conv_col], 0,
                         where=df[conv_col] > 0, alpha=0.3, color="green")
    axes[1].fill_between(df.index, df[conv_col], 0,
                         where=df[conv_col] < 0, alpha=0.3, color="red")
    axes[1].set_title("Conviction Over Time")
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXTENSION 3: Divergence Detection
# =============================================================================

def detect_divergences(
    df: pd.DataFrame,
    price_col: str = "price",
    lookback: int = 5,
    prob_thresh: float = 0.55,
    return_thresh: float = 0.003,
    print_analysis: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """
    Detect divergences between model predictions and price behavior.

    Types:
    - BEARISH DIVERGENCE: Model says bearish but price rising -> price may drop OR model wrong
    - BULLISH DIVERGENCE: Model says bullish but price falling -> price may rise OR model wrong

    Args:
        df: DataFrame with signals and probabilities
        price_col: Column name for price
        lookback: Bars to measure price change over
        prob_thresh: Probability threshold to consider a directional prediction
        return_thresh: Minimum price % change to consider directional move
        print_analysis: Whether to print analysis
        plot: Whether to generate visualization

    Returns:
        DataFrame with divergence flags
    """
    result = df.copy()

    # Price behavior
    result["price_return"] = result[price_col].pct_change(lookback)
    result["price_rising"] = result["price_return"] > return_thresh
    result["price_falling"] = result["price_return"] < -return_thresh

    # Model predictions
    result["model_bullish"] = result["ensemble_prob_bull"] > prob_thresh
    result["model_bearish"] = result["ensemble_prob_bear"] > prob_thresh

    # Divergences
    result["bearish_divergence"] = result["model_bearish"] & result["price_rising"]
    result["bullish_divergence"] = result["model_bullish"] & result["price_falling"]

    # Divergence intensity
    result["divergence_intensity"] = np.where(
        result["bearish_divergence"],
        result["ensemble_prob_bear"] * result["price_return"],
        np.where(
            result["bullish_divergence"],
            result["ensemble_prob_bull"] * (-result["price_return"]),
            0
        )
    )

    if print_analysis:
        _print_divergence_analysis(result)

    if plot:
        _plot_divergences(result, price_col)

    return result


def _print_divergence_analysis(df: pd.DataFrame):
    """Print divergence analysis with interpretation."""
    bearish_div = df[df["bearish_divergence"]]
    bullish_div = df[df["bullish_divergence"]]

    print("=" * 70)
    print("  DIVERGENCE DETECTION ANALYSIS")
    print("=" * 70)

    print(f"\n📉 BEARISH DIVERGENCES (Model↓, Price↑): {len(bearish_div)}")
    if len(bearish_div) > 0 and "returns" in df.columns:
        next_ret = df.loc[bearish_div.index, "returns"].shift(-1)
        model_correct = (next_ret < 0).sum()
        print(f"   Model correct (price dropped): {model_correct}/{len(bearish_div)} ({model_correct/len(bearish_div):.1%})")

    print(f"\n📈 BULLISH DIVERGENCES (Model↑, Price↓): {len(bullish_div)}")
    if len(bullish_div) > 0 and "returns" in df.columns:
        next_ret = df.loc[bullish_div.index, "returns"].shift(-1)
        model_correct = (next_ret > 0).sum()
        print(f"   Model correct (price rose): {model_correct}/{len(bullish_div)} ({model_correct/len(bullish_div):.1%})")

    total_divs = len(bearish_div) + len(bullish_div)
    div_rate = total_divs / len(df) * 100
    print(f"\n⚠️  Overall Divergence Rate: {div_rate:.1f}%")
    print("=" * 70)


def _plot_divergences(df: pd.DataFrame, price_col: str):
    """Plot divergences on price chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[price_col], label="Price", alpha=0.7)

    bear_div = df[df["bearish_divergence"]]
    bull_div = df[df["bullish_divergence"]]

    ax.scatter(bear_div.index, bear_div[price_col], marker="o", color="orange",
               s=30, label=f"Bearish Div (n={len(bear_div)})", alpha=0.7)
    ax.scatter(bull_div.index, bull_div[price_col], marker="o", color="purple",
               s=30, label=f"Bullish Div (n={len(bull_div)})", alpha=0.7)

    ax.set_title("Divergence Detection")
    ax.legend()
    plt.show()


# =============================================================================
# EXTENSION 4: Risk Sizing
# =============================================================================

def compute_risk_adjusted_size(
    df: pd.DataFrame,
    base_size: float = 1.0,
    opposing_penalty: float = 0.5,
    conviction_bonus: float = 0.3,
    min_size: float = 0.1,
    max_size: float = 2.0,
    print_analysis: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """
    Compute risk-adjusted position sizes based on opposing probabilities.

    Logic:
    - LONG: Size reduced when P(bear) is high
    - SHORT: Size reduced when P(bull) is high
    - BONUS: Size increased when conviction is strong

    Args:
        df: DataFrame with ensemble probabilities and signal
        base_size: Starting position size (1.0 = full size)
        opposing_penalty: Size reduction per unit of opposing prob
        conviction_bonus: Size increase for strong conviction
        min_size: Minimum position size
        max_size: Maximum position size
        print_analysis: Whether to print analysis
        plot: Whether to generate visualization

    Returns:
        DataFrame with position_size and sized_signal columns
    """
    result = df.copy()

    # Ensure signal exists
    if "signal" not in result.columns:
        result["signal"] = np.sign(result.get("ensemble_conviction", 0))

    # Opposing probability
    result["opposing_prob"] = np.where(
        result["signal"] == 1,
        result["ensemble_prob_bear"],
        np.where(
            result["signal"] == -1,
            result["ensemble_prob_bull"],
            0.5
        )
    )

    # Conviction strength
    conv_col = "ensemble_conviction" if "ensemble_conviction" in result.columns else None
    if conv_col:
        result["conviction_strength"] = result[conv_col].abs()
    else:
        result["conviction_strength"] = 0

    # Size adjustments
    result["opposing_penalty"] = result["opposing_prob"] * opposing_penalty
    result["conviction_adjustment"] = np.where(
        result["conviction_strength"] > 0.4,
        (result["conviction_strength"] - 0.4) * conviction_bonus,
        0
    )

    # Final position size
    result["position_size"] = base_size - result["opposing_penalty"] + result["conviction_adjustment"]
    result["position_size"] = result["position_size"].clip(min_size, max_size)

    # Sized signal
    result["sized_signal"] = result["signal"] * result["position_size"]

    if print_analysis:
        _print_risk_sizing_analysis(result)

    if plot:
        _plot_risk_sizing(result)

    return result


def _print_risk_sizing_analysis(df: pd.DataFrame):
    """Print risk sizing analysis."""
    print("=" * 60)
    print("  RISK SIZING ANALYSIS")
    print("=" * 60)

    ps = df["position_size"]
    print(f"\nPosition Size Distribution:")
    print(f"  Mean: {ps.mean():.2f}x  |  Std: {ps.std():.2f}x")
    print(f"  Min:  {ps.min():.2f}x  |  Max: {ps.max():.2f}x")

    # Size buckets
    buckets = pd.cut(ps, bins=[0, 0.5, 0.8, 1.0, 1.2, 2.0],
                     labels=["Tiny", "Small", "Normal", "Large", "Huge"])
    print(f"\nSize Buckets: {dict(buckets.value_counts().sort_index())}")
    print("=" * 60)


def _plot_risk_sizing(df: pd.DataFrame):
    """Plot position sizes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(df["position_size"], alpha=0.7)
    axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5)
    axes[0].fill_between(df.index, df["position_size"], 1.0,
                         where=df["position_size"] < 1.0, alpha=0.3, color="red")
    axes[0].fill_between(df.index, df["position_size"], 1.0,
                         where=df["position_size"] > 1.0, alpha=0.3, color="green")
    axes[0].set_ylabel("Position Size")
    axes[0].set_title("Risk-Adjusted Position Sizing")

    colors = np.where(df["sized_signal"] > 0, "green",
                      np.where(df["sized_signal"] < 0, "red", "gray"))
    axes[1].bar(df.index, df["sized_signal"], width=1, alpha=0.7, color=colors)
    axes[1].set_ylabel("Sized Signal")
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXTENSION 5: Regime Filtering
# =============================================================================

def apply_regime_filter(
    df: pd.DataFrame,
    X: Optional[pd.DataFrame] = None,
    regime_col: str = "Regime_Combo",
    long_bear_max: float = 0.4,
    short_bull_max: float = 0.4,
    regime_rules: Optional[Dict[Any, Dict[str, bool]]] = None,
    print_analysis: bool = True,
    plot: bool = False
) -> pd.DataFrame:
    """
    Filter signals based on regime and probability constraints.

    Default Rules:
    - LONG signals: Only allowed when P(bear) < long_bear_max
    - SHORT signals: Only allowed when P(bull) < short_bull_max

    Args:
        df: DataFrame with signals and probabilities
        X: Feature DataFrame containing regime labels (if not in df)
        regime_col: Column containing regime labels
        long_bear_max: Max P(bear) allowed for long signals
        short_bull_max: Max P(bull) allowed for short signals
        regime_rules: Dict of regime-specific rules
            Example: {-10.0: {"allow_longs": False, "allow_shorts": True}}
        print_analysis: Whether to print analysis
        plot: Whether to generate visualization

    Returns:
        DataFrame with filtered_signal column
    """
    result = df.copy()

    # Get regime data
    if regime_col not in result.columns and X is not None and regime_col in X.columns:
        result = result.join(X[regime_col], how="left")

    if regime_rules is None:
        regime_rules = {}

    # Probability-based filters
    result["long_allowed"] = result["ensemble_prob_bear"] < long_bear_max
    result["short_allowed"] = result["ensemble_prob_bull"] < short_bull_max

    # Apply base filter
    result["filtered_signal"] = np.where(
        (result["signal"] == 1) & result["long_allowed"], 1,
        np.where(
            (result["signal"] == -1) & result["short_allowed"], -1, 0
        )
    )

    # Apply regime-specific overrides
    if regime_col in result.columns:
        for regime, rules in regime_rules.items():
            mask = result[regime_col] == regime
            if not rules.get("allow_longs", True):
                result.loc[mask & (result["signal"] == 1), "filtered_signal"] = 0
            if not rules.get("allow_shorts", True):
                result.loc[mask & (result["signal"] == -1), "filtered_signal"] = 0

    result["was_filtered"] = result["signal"] != result["filtered_signal"]

    if print_analysis:
        _print_regime_filter_analysis(result, regime_col)

    if plot:
        _plot_regime_filter(result)

    return result


def _print_regime_filter_analysis(df: pd.DataFrame, regime_col: str):
    """Print regime filtering analysis."""
    print("=" * 70)
    print("  REGIME-BASED SIGNAL FILTERING")
    print("=" * 70)

    total = (df["signal"] != 0).sum()
    filtered = df["was_filtered"].sum()

    print(f"\nFiltering Summary:")
    print(f"  Original: {total}  |  Filtered: {filtered}  |  Remaining: {total - filtered}")
    print(f"  Filter Rate: {filtered/max(total,1):.1%}")

    if regime_col in df.columns:
        print(f"\nBy Regime:")
        regime_stats = df.groupby(regime_col).agg({
            "signal": lambda x: (x != 0).sum(),
            "was_filtered": "sum"
        })
        print(regime_stats.to_string())

    print("=" * 70)


def _plot_regime_filter(df: pd.DataFrame):
    """Plot original vs filtered signals."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].bar(df.index, df["signal"], width=1, alpha=0.5, color="gray")
    axes[0].set_ylabel("Original")
    axes[0].set_title("Original vs Filtered Signals")

    colors = np.where(df["filtered_signal"] > 0, "green",
                      np.where(df["filtered_signal"] < 0, "red", "white"))
    axes[1].bar(df.index, df["filtered_signal"], width=1, alpha=0.7, color=colors)
    axes[1].set_ylabel("Filtered")
    plt.tight_layout()
    plt.show()


# =============================================================================
# CONVENIENCE FUNCTION: Run All Extensions
# =============================================================================

def run_all_extensions(
    df: pd.DataFrame,
    X: Optional[pd.DataFrame] = None,
    price_col: str = "price",
    regime_col: str = "Regime_Combo",
    regime_rules: Optional[Dict] = None,
    plot_all: bool = True,
    print_all: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run all signal extensions and return consolidated results.

    Args:
        df: DataFrame with ensemble probabilities, signals, and price
        X: Feature DataFrame with regime labels
        price_col: Price column name
        regime_col: Regime column name
        regime_rules: Regime-specific filtering rules
        plot_all: Generate all plots
        print_all: Print all summaries

    Returns:
        Tuple of (final_df, dict_of_intermediate_results)
    """
    results = {}

    print("\n" + "=" * 70)
    print("  RUNNING ALL SIGNAL EXTENSIONS")
    print("=" * 70 + "\n")

    # 1. Reversion
    print(">>> Extension 1: Mean Reversion")
    results["reversion"] = generate_reversion_signals(df, price_col=price_col, plot=plot_all)

    # 2. Conviction
    print("\n>>> Extension 2: Conviction Scoring")
    results["conviction"] = compute_conviction_metrics(df, print_summary=print_all, plot=plot_all)

    # 3. Divergence
    print("\n>>> Extension 3: Divergence Detection")
    results["divergence"] = detect_divergences(df, price_col=price_col,
                                                print_analysis=print_all, plot=plot_all)

    # 4. Risk Sizing
    print("\n>>> Extension 4: Risk Sizing")
    results["sizing"] = compute_risk_adjusted_size(df, print_analysis=print_all, plot=plot_all)

    # 5. Regime Filter
    print("\n>>> Extension 5: Regime Filtering")
    results["filtered"] = apply_regime_filter(df, X=X, regime_col=regime_col,
                                               regime_rules=regime_rules,
                                               print_analysis=print_all, plot=plot_all)

    # Merge key columns into final DataFrame
    final = df.copy()
    final["reversion_signal"] = results["reversion"]["reversion_signal"]
    final["conviction_zone"] = results["conviction"]["conviction_zone"]
    final["bearish_divergence"] = results["divergence"]["bearish_divergence"]
    final["bullish_divergence"] = results["divergence"]["bullish_divergence"]
    final["position_size"] = results["sizing"]["position_size"]
    final["sized_signal"] = results["sizing"]["sized_signal"]
    final["filtered_signal"] = results["filtered"]["filtered_signal"]

    print("\n" + "=" * 70)
    print("  ALL EXTENSIONS COMPLETE")
    print("=" * 70)

    return final, results


# =============================================================================
# QUICK-CALL HELPER FUNCTIONS (for notebook integration)
# =============================================================================

def quick_reversion(df, price_col="price", plot=True):
    """Quick call for mean reversion analysis."""
    return generate_reversion_signals(df, price_col=price_col, plot=plot)

def quick_conviction(df, plot=True):
    """Quick call for conviction analysis."""
    return compute_conviction_metrics(df, print_summary=True, plot=plot)

def quick_divergence(df, price_col="price", plot=True):
    """Quick call for divergence detection."""
    return detect_divergences(df, price_col=price_col, print_analysis=True, plot=plot)

def quick_sizing(df, plot=True):
    """Quick call for risk sizing."""
    return compute_risk_adjusted_size(df, print_analysis=True, plot=plot)

def quick_filter(df, X=None, regime_rules=None, plot=True):
    """Quick call for regime filtering."""
    return apply_regime_filter(df, X=X, regime_rules=regime_rules,
                               print_analysis=True, plot=plot)
