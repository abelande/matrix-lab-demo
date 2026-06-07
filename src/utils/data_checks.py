"""
Data Validation & Dynamic Checking Utilities
=============================================
Use these functions at the end of notebook cells to validate
data consistency across different dataset sizes.

Usage:
    from myquantlab.utils.data_checks import DataValidator

    validator = DataValidator()
    validator.check_section_01(df, X, y)  # At end of Section 01
    validator.check_section_06(df_predictions, model)  # At end of Section 06
    # etc.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data consistency throughout notebook pipeline."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.state = {}  # Track state across sections

    def log(self, msg: str, level: str = "info"):
        if self.verbose:
            getattr(logger, level)(msg)

    # =========================================================
    # Core Detection Functions
    # =========================================================

    @staticmethod
    def detect_classes(series: pd.Series) -> List:
        """Detect unique classes from a target series."""
        classes = sorted(series.dropna().unique().tolist())
        return classes

    @staticmethod
    def detect_prob_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect probability columns and map them to class labels.
        Returns: {'bull': 'prob_class_1.0', 'bear': 'prob_class_-1.0', ...}
        """
        prob_cols = [c for c in df.columns if c.startswith("prob_class_")]
        mapping = {}

        for col in prob_cols:
            # Extract class label from column name
            label = col.replace("prob_class_", "")

            # Identify bull/bear
            if "-1" in label:
                mapping["bear"] = col
                mapping[f"class_{label}"] = col
            elif "1" in label and "-1" not in label:
                mapping["bull"] = col
                mapping[f"class_{label}"] = col
            elif "0" in label:
                mapping["neutral"] = col
                mapping[f"class_{label}"] = col
            else:
                mapping[f"class_{label}"] = col

        mapping["all"] = prob_cols
        return mapping

    @staticmethod
    def detect_feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Group features by prefix/type."""
        groups = {
            "regime": [],
            "momentum": [],
            "volatility": [],
            "volume": [],
            "price": [],
            "lag": [],
            "other": []
        }

        for col in df.columns:
            col_lower = col.lower()
            if "regime" in col_lower:
                groups["regime"].append(col)
            elif any(x in col_lower for x in ["rsi", "macd", "momentum", "roc"]):
                groups["momentum"].append(col)
            elif any(x in col_lower for x in ["vol", "atr", "std", "bbw"]):
                groups["volatility"].append(col)
            elif "volume" in col_lower or "obv" in col_lower:
                groups["volume"].append(col)
            elif any(x in col_lower for x in ["price", "close", "open", "high", "low"]):
                groups["price"].append(col)
            elif "lag" in col_lower or col_lower.startswith("l_"):
                groups["lag"].append(col)
            else:
                groups["other"].append(col)

        return {k: v for k, v in groups.items() if v}

    # =========================================================
    # Validation Functions
    # =========================================================

    def validate_no_nan(self, df: pd.DataFrame, name: str,
                        columns: Optional[List[str]] = None) -> Tuple[bool, Dict]:
        """Check for NaN values."""
        cols = columns or df.columns.tolist()
        nan_counts = df[cols].isna().sum()
        nan_cols = nan_counts[nan_counts > 0]

        result = {
            "passed": len(nan_cols) == 0,
            "nan_columns": nan_cols.to_dict(),
            "total_nans": nan_counts.sum()
        }

        if not result["passed"]:
            self.log(f"{name}: Found NaN in {len(nan_cols)} columns", "warning")
        else:
            self.log(f"{name}: No NaN values")

        return result["passed"], result

    def validate_class_balance(self, y: pd.Series,
                                min_ratio: float = 0.1) -> Tuple[bool, Dict]:
        """Check class balance."""
        counts = y.value_counts()
        total = len(y)
        ratios = counts / total

        result = {
            "passed": ratios.min() >= min_ratio,
            "counts": counts.to_dict(),
            "ratios": ratios.to_dict(),
            "min_ratio": ratios.min(),
            "imbalance_ratio": counts.max() / counts.min()
        }

        if not result["passed"]:
            self.log(f"Class imbalance detected: min ratio {result['min_ratio']:.2%}", "warning")
        else:
            self.log(f"Class balance OK: {result['ratios']}")

        return result["passed"], result

    def validate_index_alignment(self, *dfs, names: Optional[List[str]] = None) -> Tuple[bool, Dict]:
        """Check that multiple DataFrames have aligned indices."""
        if len(dfs) < 2:
            return True, {"message": "Need at least 2 DataFrames"}

        names = names or [f"df_{i}" for i in range(len(dfs))]
        reference = dfs[0].index

        mismatches = {}
        for i, df in enumerate(dfs[1:], 1):
            if not reference.equals(df.index):
                diff = len(reference.symmetric_difference(df.index))
                mismatches[names[i]] = diff

        result = {
            "passed": len(mismatches) == 0,
            "mismatches": mismatches,
            "reference_len": len(reference)
        }

        if not result["passed"]:
            self.log(f"Index misalignment: {mismatches}", "warning")
        else:
            self.log(f"Index alignment OK: {len(reference)} rows")

        return result["passed"], result

    def validate_prob_columns(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate probability columns exist and sum to ~1."""
        prob_map = self.detect_prob_columns(df)
        prob_cols = prob_map.get("all", [])

        if not prob_cols:
            return False, {"passed": False, "error": "No prob columns found"}

        # Check if probs sum to 1
        prob_sums = df[prob_cols].sum(axis=1)
        sum_ok = np.allclose(prob_sums, 1.0, atol=0.01)

        result = {
            "passed": len(prob_cols) >= 2 and sum_ok,
            "prob_columns": prob_cols,
            "mapping": prob_map,
            "n_classes": len(prob_cols),
            "prob_sum_mean": prob_sums.mean(),
            "prob_sum_ok": sum_ok
        }

        if result["passed"]:
            self.log(f"Prob columns OK: {prob_cols}")
        else:
            self.log(f"Prob column issue: sum={prob_sums.mean():.3f}", "warning")

        return result["passed"], result

    # =========================================================
    # Section-Specific Checks
    # =========================================================

    def check_section_01(self, df: pd.DataFrame,
                         target_col: str = "target") -> Dict[str, Any]:
        """
        Section 01: Data Loading & Target Creation
        Validates: raw data loaded, target created, basic stats
        """
        self.log("=" * 50)
        self.log("SECTION 01 CHECK: Data Loading & Target")

        results = {
            "section": "01",
            "checks": {}
        }

        # Check dataframe exists and has data
        results["checks"]["has_data"] = len(df) > 0
        results["checks"]["n_rows"] = len(df)
        results["checks"]["n_cols"] = len(df.columns)

        # Check target exists
        if target_col in df.columns:
            classes = self.detect_classes(df[target_col])
            results["checks"]["target_exists"] = True
            results["checks"]["classes"] = classes
            results["checks"]["n_classes"] = len(classes)

            _, balance = self.validate_class_balance(df[target_col])
            results["checks"]["class_balance"] = balance
        else:
            results["checks"]["target_exists"] = False
            self.log(f"Target column '{target_col}' not found!", "error")

        # Check for datetime index
        results["checks"]["datetime_index"] = isinstance(df.index, pd.DatetimeIndex)

        # Store state for later sections
        self.state["section_01"] = {
            "n_rows": len(df),
            "columns": df.columns.tolist(),
            "classes": results["checks"].get("classes", [])
        }

        self._print_summary(results)
        return results

    def check_section_02(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Section 02: Feature Engineering
        Validates: features created, no leakage, alignment
        """
        self.log("=" * 50)
        self.log("SECTION 02 CHECK: Feature Engineering")

        results = {
            "section": "02",
            "checks": {}
        }

        # Feature counts
        feature_groups = self.detect_feature_groups(X)
        results["checks"]["feature_groups"] = {k: len(v) for k, v in feature_groups.items()}
        results["checks"]["total_features"] = len(X.columns)

        # NaN check
        _, nan_result = self.validate_no_nan(X, "Features")
        results["checks"]["nan_check"] = nan_result

        # Alignment
        _, align_result = self.validate_index_alignment(X, y.to_frame(), names=["X", "y"])
        results["checks"]["alignment"] = align_result

        # Class detection
        classes = self.detect_classes(y)
        results["checks"]["classes"] = classes

        self.state["section_02"] = {
            "n_features": len(X.columns),
            "feature_names": X.columns.tolist(),
            "classes": classes
        }

        self._print_summary(results)
        return results

    def check_section_06(self, df_predictions: pd.DataFrame,
                         model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Section 06: Model Training & Evaluation
        Validates: predictions exist, prob columns, classes match
        """
        self.log("=" * 50)
        self.log("SECTION 06 CHECK: Model Training")

        results = {
            "section": "06",
            "checks": {}
        }

        # Check predictions dataframe
        required_cols = ["true", "pred", "fold"]
        missing = [c for c in required_cols if c not in df_predictions.columns]
        results["checks"]["required_cols"] = {
            "present": [c for c in required_cols if c in df_predictions.columns],
            "missing": missing
        }

        # Prob columns
        _, prob_result = self.validate_prob_columns(df_predictions)
        results["checks"]["prob_columns"] = prob_result

        # Classes
        if "true" in df_predictions.columns:
            classes = self.detect_classes(df_predictions["true"])
            results["checks"]["classes"] = classes

        # Fold info
        if "fold" in df_predictions.columns:
            results["checks"]["n_folds"] = df_predictions["fold"].nunique()

        self.state["section_06"] = {
            "prob_mapping": prob_result.get("mapping", {}),
            "classes": results["checks"].get("classes", [])
        }

        self._print_summary(results)
        return results

    def check_section_07(self, df_preds_clean: pd.DataFrame) -> Dict[str, Any]:
        """
        Section 07: Model Diagnostics
        Validates: cleaned predictions ready for analysis
        """
        self.log("=" * 50)
        self.log("SECTION 07 CHECK: Diagnostics Data")

        results = {
            "section": "07",
            "checks": {}
        }

        results["checks"]["n_rows"] = len(df_preds_clean)

        # Prob columns exist
        _, prob_result = self.validate_prob_columns(df_preds_clean)
        results["checks"]["prob_columns"] = prob_result

        # No NaN in key columns
        key_cols = ["true", "pred"] + prob_result.get("prob_columns", [])[:3]
        _, nan_result = self.validate_no_nan(df_preds_clean, "Predictions",
                                              [c for c in key_cols if c in df_preds_clean.columns])
        results["checks"]["nan_check"] = nan_result

        self._print_summary(results)
        return results

    # =========================================================
    # Helper Functions
    # =========================================================

    def get_prob_cols(self, df: pd.DataFrame) -> Dict[str, str]:
        """Convenience method to get probability column mapping."""
        return self.detect_prob_columns(df)

    def get_bull_bear_cols(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Get bull and bear probability column names."""
        mapping = self.detect_prob_columns(df)
        return mapping.get("bull"), mapping.get("bear")

    def _print_summary(self, results: Dict):
        """Print validation summary."""
        self.log("-" * 40)
        self.log(f"Section {results['section']} Summary:")

        for key, value in results["checks"].items():
            if isinstance(value, dict):
                if "passed" in value:
                    status = "✓" if value["passed"] else "✗"
                    self.log(f"  {status} {key}")
            elif isinstance(value, bool):
                status = "✓" if value else "✗"
                self.log(f"  {status} {key}")
            elif isinstance(value, (int, float)):
                self.log(f"  • {key}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                self.log(f"  • {key}: {value}")


# =========================================================
# Standalone Helper Functions
# =========================================================

def get_prob_columns(df: pd.DataFrame) -> List[str]:
    """Get list of probability columns."""
    return [c for c in df.columns if c.startswith("prob_class_")]


def get_bull_col(df: pd.DataFrame) -> Optional[str]:
    """Get bull probability column name."""
    cols = get_prob_columns(df)
    matches = [c for c in cols if "1" in c and "-1" not in c]
    return matches[0] if matches else None


def get_bear_col(df: pd.DataFrame) -> Optional[str]:
    """Get bear probability column name."""
    cols = get_prob_columns(df)
    matches = [c for c in cols if "-1" in c]
    return matches[0] if matches else None


def get_classes(series: pd.Series) -> List:
    """Get sorted unique classes."""
    return sorted(series.dropna().unique().tolist())


def safe_get_col(df: pd.DataFrame, patterns: List[str],
                 default: Optional[str] = None) -> Optional[str]:
    """
    Safely get column matching any pattern.

    Usage:
        col = safe_get_col(df, ["prob_class_1", "prob_class_1.0"])
    """
    for pattern in patterns:
        if pattern in df.columns:
            return pattern
        # Try partial match
        matches = [c for c in df.columns if pattern in c]
        if matches:
            return matches[0]
    return default


def quick_check(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Quick one-liner check - print key stats."""
    if isinstance(df, pd.Series):
        print(f"{name}: {len(df)} rows, "
              f"NaN: {df.isna().sum()}")
    else:
        prob_cols = get_prob_columns(df)
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols, "
              f"{len(prob_cols)} prob cols, "
              f"NaN: {df.isna().sum().sum()}")


# =========================================================
# Extended Analysis Functions
# =========================================================

def compare_across_samples(
    df_small: pd.DataFrame,
    df_large: pd.DataFrame,
    metric: str = "accuracy",
    true_col: str = "true",
    pred_col: str = "pred"
) -> Dict[str, Any]:
    """
    Compare model performance across different dataset sizes.

    Use this to validate if patterns hold across short vs long timeframes.

    Args:
        df_small: Predictions from smaller dataset
        df_large: Predictions from larger dataset
        metric: One of 'accuracy', 'precision', 'recall', 'f1', 'all'
        true_col: Column name for true labels
        pred_col: Column name for predictions

    Returns:
        Dict with metrics for both datasets and comparison stats
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = {
        "small": {"n_samples": len(df_small)},
        "large": {"n_samples": len(df_large)},
        "comparison": {}
    }

    # Calculate metrics
    metrics_funcs = {
        "accuracy": lambda y, p: accuracy_score(y, p),
        "precision": lambda y, p: precision_score(y, p, average="weighted", zero_division=0),
        "recall": lambda y, p: recall_score(y, p, average="weighted", zero_division=0),
        "f1": lambda y, p: f1_score(y, p, average="weighted", zero_division=0)
    }

    if metric == "all":
        calc_metrics = list(metrics_funcs.keys())
    else:
        calc_metrics = [metric]

    for m in calc_metrics:
        if m in metrics_funcs:
            func = metrics_funcs[m]
            results["small"][m] = func(df_small[true_col], df_small[pred_col])
            results["large"][m] = func(df_large[true_col], df_large[pred_col])

            # Comparison: difference and ratio
            diff = results["small"][m] - results["large"][m]
            results["comparison"][f"{m}_diff"] = diff
            results["comparison"][f"{m}_stable"] = abs(diff) < 0.05  # Within 5%

    # Class distribution comparison
    small_dist = df_small[true_col].value_counts(normalize=True).to_dict()
    large_dist = df_large[true_col].value_counts(normalize=True).to_dict()
    results["small"]["class_dist"] = small_dist
    results["large"]["class_dist"] = large_dist

    # Print summary
    print("\n" + "=" * 50)
    print("CROSS-SAMPLE COMPARISON")
    print("=" * 50)
    print(f"Small dataset: {results['small']['n_samples']} samples")
    print(f"Large dataset: {results['large']['n_samples']} samples")
    print("-" * 50)

    for m in calc_metrics:
        if m in results["small"]:
            stable = "✓" if results["comparison"].get(f"{m}_stable", False) else "✗"
            print(f"{m.capitalize()}:")
            print(f"  Small: {results['small'][m]:.4f}")
            print(f"  Large: {results['large'][m]:.4f}")
            print(f"  Diff:  {results['comparison'][f'{m}_diff']:+.4f} {stable}")

    return results


def pattern_stability_check(
    predictions_short: pd.DataFrame,
    predictions_long: pd.DataFrame,
    prob_col: Optional[str] = None,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Track pattern/signal consistency across different timeframes.

    Checks if model confidence and prediction distributions are stable.

    Args:
        predictions_short: Predictions from shorter timeframe
        predictions_long: Predictions from longer timeframe
        prob_col: Probability column to analyze (auto-detected if None)
        threshold: Max acceptable difference for stability

    Returns:
        Dict with stability metrics
    """
    results = {
        "short": {"n_samples": len(predictions_short)},
        "long": {"n_samples": len(predictions_long)},
        "stability": {}
    }

    # Auto-detect prob column if not specified
    if prob_col is None:
        prob_cols = get_prob_columns(predictions_short)
        prob_col = prob_cols[0] if prob_cols else None

    # Prediction distribution
    short_pred_dist = predictions_short["pred"].value_counts(normalize=True)
    long_pred_dist = predictions_long["pred"].value_counts(normalize=True)

    results["short"]["pred_distribution"] = short_pred_dist.to_dict()
    results["long"]["pred_distribution"] = long_pred_dist.to_dict()

    # Distribution similarity (KL-divergence approximation using overlap)
    all_classes = set(short_pred_dist.index) | set(long_pred_dist.index)
    dist_diff = 0
    for cls in all_classes:
        p_short = short_pred_dist.get(cls, 0)
        p_long = long_pred_dist.get(cls, 0)
        dist_diff += abs(p_short - p_long)

    results["stability"]["pred_dist_diff"] = dist_diff
    results["stability"]["pred_dist_stable"] = dist_diff < threshold * 2

    # Probability calibration comparison
    if prob_col and prob_col in predictions_short.columns and prob_col in predictions_long.columns:
        short_prob_mean = predictions_short[prob_col].mean()
        long_prob_mean = predictions_long[prob_col].mean()
        short_prob_std = predictions_short[prob_col].std()
        long_prob_std = predictions_long[prob_col].std()

        results["short"]["prob_mean"] = short_prob_mean
        results["short"]["prob_std"] = short_prob_std
        results["long"]["prob_mean"] = long_prob_mean
        results["long"]["prob_std"] = long_prob_std

        prob_mean_diff = abs(short_prob_mean - long_prob_mean)
        results["stability"]["prob_mean_diff"] = prob_mean_diff
        results["stability"]["prob_mean_stable"] = prob_mean_diff < threshold

    # Accuracy comparison
    short_acc = (predictions_short["true"] == predictions_short["pred"]).mean()
    long_acc = (predictions_long["true"] == predictions_long["pred"]).mean()
    acc_diff = abs(short_acc - long_acc)

    results["short"]["accuracy"] = short_acc
    results["long"]["accuracy"] = long_acc
    results["stability"]["accuracy_diff"] = acc_diff
    results["stability"]["accuracy_stable"] = acc_diff < threshold

    # Overall stability score
    stability_checks = [v for k, v in results["stability"].items() if k.endswith("_stable")]
    results["stability"]["overall_stable"] = all(stability_checks)
    results["stability"]["stability_score"] = sum(stability_checks) / len(stability_checks) if stability_checks else 0

    # Print summary
    print("\n" + "=" * 50)
    print("PATTERN STABILITY CHECK")
    print("=" * 50)
    print(f"Short timeframe: {results['short']['n_samples']} samples")
    print(f"Long timeframe:  {results['long']['n_samples']} samples")
    print("-" * 50)
    print(f"Accuracy: Short={short_acc:.4f}, Long={long_acc:.4f}, Diff={acc_diff:.4f} {'✓' if acc_diff < threshold else '✗'}")
    print(f"Pred Dist Diff: {dist_diff:.4f} {'✓' if dist_diff < threshold * 2 else '✗'}")
    if "prob_mean_diff" in results["stability"]:
        print(f"Prob Mean Diff: {results['stability']['prob_mean_diff']:.4f} {'✓' if results['stability']['prob_mean_stable'] else '✗'}")
    print("-" * 50)
    print(f"Overall Stability: {'STABLE ✓' if results['stability']['overall_stable'] else 'UNSTABLE ✗'}")
    print(f"Stability Score: {results['stability']['stability_score']:.1%}")

    return results


def validate_by_regime(
    df: pd.DataFrame,
    regime_col: str = "Regime_Combo",
    target_col: str = "target",
    pred_col: Optional[str] = "pred",
    min_samples: int = 30
) -> Dict[str, Any]:
    """
    Regime-aware validation - analyze performance by market regime.

    Essential for understanding if patterns are regime-specific or universal.

    Args:
        df: DataFrame with predictions and regime labels
        regime_col: Column containing regime labels
        target_col: Column with true labels
        pred_col: Column with predictions (if None, only analyzes distribution)
        min_samples: Minimum samples per regime to include

    Returns:
        Dict with per-regime statistics
    """
    if regime_col not in df.columns:
        print(f"Warning: '{regime_col}' not found in DataFrame")
        return {"error": f"Column '{regime_col}' not found"}

    results = {
        "regimes": {},
        "summary": {}
    }

    regime_groups = df.groupby(regime_col)

    for regime, group in regime_groups:
        if len(group) < min_samples:
            continue

        regime_stats = {
            "n_samples": len(group),
            "pct_of_total": len(group) / len(df)
        }

        # Class distribution in this regime
        class_dist = group[target_col].value_counts(normalize=True).to_dict()
        regime_stats["class_distribution"] = class_dist

        # Dominant class
        regime_stats["dominant_class"] = group[target_col].mode().iloc[0] if len(group) > 0 else None

        # If predictions available
        if pred_col and pred_col in df.columns:
            accuracy = (group[target_col] == group[pred_col]).mean()
            regime_stats["accuracy"] = accuracy

            # Per-class accuracy in this regime
            for cls in group[target_col].unique():
                cls_mask = group[target_col] == cls
                if cls_mask.sum() > 0:
                    cls_acc = (group.loc[cls_mask, target_col] == group.loc[cls_mask, pred_col]).mean()
                    regime_stats[f"accuracy_class_{cls}"] = cls_acc

        # Probability stats if available
        prob_cols = get_prob_columns(group)
        if prob_cols:
            for pc in prob_cols[:3]:  # Limit to first 3
                regime_stats[f"{pc}_mean"] = group[pc].mean()
                regime_stats[f"{pc}_std"] = group[pc].std()

        results["regimes"][str(regime)] = regime_stats

    # Summary statistics
    if pred_col and pred_col in df.columns:
        accuracies = [r["accuracy"] for r in results["regimes"].values() if "accuracy" in r]
        if accuracies:
            results["summary"]["mean_accuracy"] = np.mean(accuracies)
            results["summary"]["std_accuracy"] = np.std(accuracies)
            results["summary"]["min_accuracy"] = np.min(accuracies)
            results["summary"]["max_accuracy"] = np.max(accuracies)
            results["summary"]["accuracy_range"] = np.max(accuracies) - np.min(accuracies)

    results["summary"]["n_regimes"] = len(results["regimes"])
    results["summary"]["total_samples"] = len(df)

    # Print summary
    print("\n" + "=" * 50)
    print("REGIME-AWARE VALIDATION")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Regimes analyzed: {len(results['regimes'])}")
    print("-" * 50)

    # Sort by accuracy if available
    regime_items = list(results["regimes"].items())
    if pred_col and any("accuracy" in r for _, r in regime_items):
        regime_items.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

    for regime, stats in regime_items:
        acc_str = f", Acc={stats['accuracy']:.3f}" if "accuracy" in stats else ""
        print(f"{regime}: n={stats['n_samples']} ({stats['pct_of_total']:.1%}){acc_str}")
        print(f"  Classes: {stats['class_distribution']}")

    if "accuracy_range" in results["summary"]:
        print("-" * 50)
        print(f"Accuracy Range: {results['summary']['min_accuracy']:.3f} - {results['summary']['max_accuracy']:.3f}")
        print(f"Accuracy Std:   {results['summary']['std_accuracy']:.3f}")

        if results["summary"]["accuracy_range"] > 0.15:
            print("⚠ High variance across regimes - consider regime-specific models")

    return results


def run_history_comparison(
    df_current: pd.DataFrame,
    history_dir: str = "artifacts/run_history",
    run_name: Optional[str] = None,
    max_history: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Compare current run with previous runs. Safe to call on first run.

    Saves current predictions to history and compares with most recent
    previous run if available.

    Args:
        df_current: Current predictions DataFrame (must have 'true', 'pred' columns)
        history_dir: Directory to store run history
        run_name: Optional name for this run (defaults to timestamp)
        max_history: Maximum number of historical runs to keep

    Returns:
        Comparison results dict, or None if this is the first run
    """
    from pathlib import Path
    from datetime import datetime

    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)

    # Generate run name if not provided
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find existing history files
    history_files = sorted(history_path.glob("run_*.parquet"))

    results = None

    # If we have previous runs, compare with the most recent
    if history_files:
        most_recent = history_files[-1]
        try:
            df_previous = pd.read_parquet(most_recent)

            print(f"\n{'='*50}")
            print(f"COMPARING WITH PREVIOUS RUN")
            print(f"Previous: {most_recent.name}")
            print(f"Current:  run_{run_name}")
            print(f"{'='*50}")

            # Run comparison
            results = compare_across_samples(
                df_current, df_previous,
                metric="all",
                true_col="true",
                pred_col="pred"
            )

            # Also run stability check
            stability = pattern_stability_check(df_current, df_previous)
            results["stability"] = stability

        except Exception as e:
            print(f"Warning: Could not compare with previous run: {e}")
    else:
        print(f"\n{'='*50}")
        print("FIRST RUN - No previous history to compare")
        print(f"Saving current run to history...")
        print(f"{'='*50}")

    # Save current run to history
    current_file = history_path / f"run_{run_name}.parquet"
    df_current.to_parquet(current_file)
    print(f"Saved: {current_file}")

    # Cleanup old history files if exceeding max
    history_files = sorted(history_path.glob("run_*.parquet"))
    if len(history_files) > max_history:
        for old_file in history_files[:-max_history]:
            old_file.unlink()
            print(f"Removed old: {old_file.name}")

    return results


def compare_with_saved(
    df_current: pd.DataFrame,
    saved_path: str,
    label: str = "saved"
) -> Optional[Dict[str, Any]]:
    """
    Compare current predictions with a specific saved file.

    Safe to call even if saved file doesn't exist.

    Args:
        df_current: Current predictions
        saved_path: Path to saved predictions parquet file
        label: Label for the saved dataset in output

    Returns:
        Comparison results, or None if file doesn't exist
    """
    from pathlib import Path

    saved_file = Path(saved_path)

    if not saved_file.exists():
        print(f"No saved predictions at {saved_path} - skipping comparison")
        return None

    try:
        df_saved = pd.read_parquet(saved_file)

        print(f"\n{'='*50}")
        print(f"COMPARING: Current vs {label}")
        print(f"{'='*50}")

        results = compare_across_samples(
            df_current, df_saved,
            metric="all"
        )

        return results

    except Exception as e:
        print(f"Error loading {saved_path}: {e}")
        return None
