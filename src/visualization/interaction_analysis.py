"""
interaction_analysis.py — Advanced Feature Interaction Analysis

Implements the "Indian Run" pairwise feature interaction sweep using:
- Option A: Continuous dependence via regression slope (β) and R²
- QME: Quantile bins (Q1 vs Q4) + Magnitude + Effect size (Cohen's d)

Mathematical basis:
- Pairwise conditional effect: Δᵢ|ⱼ = E[φᵢ | xⱼ high] − E[φᵢ | xⱼ low]
- Interaction matrix M where Mᵢⱼ quantifies "feature i depends on feature j"
- O(d²) combinatorial sweep over all feature pairs

Usage in notebook:
    from visualization.interaction_analysis import (
        compute_regression_interaction_matrix,
        compute_qme_interaction_matrix,
        compute_combined_interaction_analysis,
        plot_interaction_heatmap_r2,
        indian_run_analysis
    )

    # Full analysis with both methods
    results = indian_run_analysis(X_sample, shap_values, X.columns, top_n=25)

    # Or individual methods
    r2_matrix = compute_regression_interaction_matrix(X_sample, shap_values, X.columns)
    qme_matrix = compute_qme_interaction_matrix(X_sample, shap_values, X.columns)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, Optional

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================
# 1. OPTION A: REGRESSION-BASED INTERACTION (β + R²)
# ============================================================

def compute_regression_dependence(
    shap_f1: np.ndarray,
    x_f2: np.ndarray
) -> Dict[str, float]:
    """
    Compute how much φ_f1 depends on x_f2 using linear regression.

    Returns:
        dict with 'beta' (slope), 'r2' (explained variance), 'p_value'
    """
    # Remove NaN values
    mask = ~(np.isnan(shap_f1) | np.isnan(x_f2))
    if mask.sum() < 10:
        return {'beta': 0.0, 'r2': 0.0, 'p_value': 1.0, 'beta_std': 0.0}

    shap_clean = shap_f1[mask]
    x_clean = x_f2[mask]

    # Standardize x for comparable beta
    x_std = (x_clean - x_clean.mean()) / (x_clean.std() + 1e-10)

    # Linear regression: φ_1 = α + β*x_2 + ε
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_std, shap_clean)

    return {
        'beta': slope,
        'beta_std': slope / (shap_clean.std() + 1e-10),  # Standardized beta
        'r2': r_value ** 2,
        'p_value': p_value
    }


def compute_regression_interaction_matrix(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 25,
    class_idx: int = 1,
    metric: str = 'r2'
) -> pd.DataFrame:
    """
    Compute interaction matrix using regression-based dependence.

    For each pair (f1, f2): fit φ_f1 ~ x_f2 and extract R² or β.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features to include
        class_idx: Which class to analyze
        metric: 'r2' for explained variance, 'beta' for slope, 'beta_std' for standardized

    Returns:
        DataFrame with interaction strengths
    """
    shap_class = shap_values[:, :, class_idx]
    mean_imp = np.abs(shap_class).mean(axis=0)
    top_idx = np.argsort(mean_imp)[-top_n:][::-1]
    top_features = feature_names[top_idx]

    interaction_matrix = pd.DataFrame(
        index=top_features,
        columns=top_features,
        dtype=float
    )

    for f1 in top_features:
        f1_idx = feature_names.get_loc(f1)
        shap_f1 = shap_class[:, f1_idx]

        for f2 in top_features:
            if f1 == f2:
                interaction_matrix.loc[f1, f2] = 0.0
            else:
                x_f2 = X_sample[f2].values
                result = compute_regression_dependence(shap_f1, x_f2)
                interaction_matrix.loc[f1, f2] = result[metric]

    return interaction_matrix.astype(float)


# ============================================================
# 2. QME: QUANTILE + MAGNITUDE + EFFECT SIZE
# ============================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    d = (μ1 - μ2) / s_pooled
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def compute_qme_dependence(
    shap_f1: np.ndarray,
    x_f2: np.ndarray,
    use_magnitude: bool = True
) -> Dict[str, float]:
    """
    Compute QME (Quantile + Magnitude + Effect size) dependence.

    Compares SHAP values when x_f2 is in Q1 (low) vs Q4 (high).

    Args:
        shap_f1: SHAP values for feature 1
        x_f2: Raw values for feature 2
        use_magnitude: If True, use |SHAP| to avoid sign cancellation

    Returns:
        dict with 'diff', 'diff_abs', 'cohens_d', 'cohens_d_abs', 'mean_q1', 'mean_q4'
    """
    # Remove NaN
    mask = ~(np.isnan(shap_f1) | np.isnan(x_f2))
    if mask.sum() < 20:
        return {
            'diff': 0.0, 'diff_abs': 0.0,
            'cohens_d': 0.0, 'cohens_d_abs': 0.0,
            'mean_q1': 0.0, 'mean_q4': 0.0
        }

    shap_clean = shap_f1[mask]
    x_clean = x_f2[mask]

    # Compute quartiles
    q1_threshold = np.percentile(x_clean, 25)
    q4_threshold = np.percentile(x_clean, 75)

    # Split into Q1 (low) and Q4 (high)
    mask_q1 = x_clean <= q1_threshold
    mask_q4 = x_clean >= q4_threshold

    shap_q1 = shap_clean[mask_q1]
    shap_q4 = shap_clean[mask_q4]

    if len(shap_q1) < 5 or len(shap_q4) < 5:
        return {
            'diff': 0.0, 'diff_abs': 0.0,
            'cohens_d': 0.0, 'cohens_d_abs': 0.0,
            'mean_q1': 0.0, 'mean_q4': 0.0
        }

    # Raw SHAP comparison
    mean_q1 = shap_q1.mean()
    mean_q4 = shap_q4.mean()
    diff = mean_q4 - mean_q1
    cohens_d = compute_cohens_d(shap_q4, shap_q1)

    # Magnitude comparison (no sign cancellation)
    shap_q1_abs = np.abs(shap_q1)
    shap_q4_abs = np.abs(shap_q4)
    diff_abs = shap_q4_abs.mean() - shap_q1_abs.mean()
    cohens_d_abs = compute_cohens_d(shap_q4_abs, shap_q1_abs)

    return {
        'diff': diff,
        'diff_abs': diff_abs,
        'cohens_d': cohens_d,
        'cohens_d_abs': cohens_d_abs,
        'mean_q1': mean_q1,
        'mean_q4': mean_q4
    }


def compute_qme_interaction_matrix(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 25,
    class_idx: int = 1,
    metric: str = 'cohens_d'
) -> pd.DataFrame:
    """
    Compute interaction matrix using QME (Quantile + Magnitude + Effect size).

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features to include
        class_idx: Which class to analyze
        metric: 'diff', 'diff_abs', 'cohens_d', or 'cohens_d_abs'

    Returns:
        DataFrame with interaction strengths
    """
    shap_class = shap_values[:, :, class_idx]
    mean_imp = np.abs(shap_class).mean(axis=0)
    top_idx = np.argsort(mean_imp)[-top_n:][::-1]
    top_features = feature_names[top_idx]

    interaction_matrix = pd.DataFrame(
        index=top_features,
        columns=top_features,
        dtype=float
    )

    for f1 in top_features:
        f1_idx = feature_names.get_loc(f1)
        shap_f1 = shap_class[:, f1_idx]

        for f2 in top_features:
            if f1 == f2:
                interaction_matrix.loc[f1, f2] = 0.0
            else:
                x_f2 = X_sample[f2].values
                result = compute_qme_dependence(shap_f1, x_f2)
                interaction_matrix.loc[f1, f2] = result[metric]

    return interaction_matrix.astype(float)


# ============================================================
# 3. COMBINED ANALYSIS (OPTION A + QME)
# ============================================================

def compute_combined_interaction_analysis(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 25,
    class_idx: int = 1
) -> pd.DataFrame:
    """
    Compute comprehensive interaction analysis with both methods.

    Returns DataFrame with columns:
        - target, condition (feature pair)
        - r2 (Option A: explained variance)
        - beta_std (Option A: standardized slope)
        - cohens_d (QME: effect size)
        - diff (QME: raw difference Q4 - Q1)
        - interpretation

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features to include
        class_idx: Which class to analyze

    Returns:
        DataFrame with all interaction metrics
    """
    shap_class = shap_values[:, :, class_idx]
    mean_imp = np.abs(shap_class).mean(axis=0)
    top_idx = np.argsort(mean_imp)[-top_n:][::-1]
    top_features = feature_names[top_idx]

    results = []

    for f1 in top_features:
        f1_idx = feature_names.get_loc(f1)
        shap_f1 = shap_class[:, f1_idx]

        for f2 in top_features:
            if f1 == f2:
                continue

            x_f2 = X_sample[f2].values

            # Option A: Regression
            reg_result = compute_regression_dependence(shap_f1, x_f2)

            # QME: Quantile + Magnitude + Effect size
            qme_result = compute_qme_dependence(shap_f1, x_f2)

            # Interpretation
            direction = "MORE" if qme_result['diff'] > 0 else "LESS"
            strength = "strong" if abs(qme_result['cohens_d']) > 0.5 else \
                       "moderate" if abs(qme_result['cohens_d']) > 0.2 else "weak"

            results.append({
                'target': f1,
                'condition': f2,
                # Option A metrics
                'r2': reg_result['r2'],
                'beta': reg_result['beta'],
                'beta_std': reg_result['beta_std'],
                'p_value': reg_result['p_value'],
                # QME metrics
                'diff': qme_result['diff'],
                'diff_abs': qme_result['diff_abs'],
                'cohens_d': qme_result['cohens_d'],
                'cohens_d_abs': qme_result['cohens_d_abs'],
                'mean_q1': qme_result['mean_q1'],
                'mean_q4': qme_result['mean_q4'],
                # Interpretation
                'direction': direction,
                'strength': strength,
                'abs_cohens_d': abs(qme_result['cohens_d'])
            })

    df = pd.DataFrame(results)
    df = df.sort_values('abs_cohens_d', ascending=False)

    return df


# ============================================================
# 4. VISUALIZATION: R² HEATMAP
# ============================================================

def plot_interaction_heatmap_r2(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 20,
    class_idx: int = 1,
    figsize: tuple = (14, 12),
    save_path: str = None
) -> pd.DataFrame:
    """
    Plot heatmap using R² (explained variance) as interaction strength.

    R² is bounded [0, 1] and shows how predictable φ_f1 is from x_f2.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features
        class_idx: Which class to analyze
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        R² interaction matrix
    """
    r2_matrix = compute_regression_interaction_matrix(
        X_sample, shap_values, feature_names,
        top_n=top_n, class_idx=class_idx, metric='r2'
    )

    plt.figure(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            r2_matrix,
            cmap='YlOrRd',
            vmin=0, vmax=r2_matrix.values.max() * 1.1,
            annot=top_n <= 15,
            fmt='.2f' if top_n <= 15 else '',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'R² (Explained Variance)'}
        )
    else:
        plt.imshow(r2_matrix.values, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='R²')
        plt.xticks(range(len(r2_matrix.columns)), r2_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(r2_matrix.index)), r2_matrix.index)

    plt.title(f"Feature Interaction Strength (R² Method)\n"
              f"Read as: How predictable is Row's SHAP from Column's value",
              fontsize=11)
    plt.xlabel("Conditioning Feature (x)")
    plt.ylabel("Target Feature (SHAP)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return r2_matrix


def plot_interaction_heatmap_cohens_d(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 20,
    class_idx: int = 1,
    figsize: tuple = (14, 12),
    save_path: str = None
) -> pd.DataFrame:
    """
    Plot heatmap using Cohen's d as interaction strength.

    Cohen's d is standardized and comparable across feature pairs.
    Positive = more important when high, Negative = less important when high.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features
        class_idx: Which class to analyze
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Cohen's d interaction matrix
    """
    cohens_matrix = compute_qme_interaction_matrix(
        X_sample, shap_values, feature_names,
        top_n=top_n, class_idx=class_idx, metric='cohens_d'
    )

    plt.figure(figsize=figsize)

    if HAS_SEABORN:
        vmax = max(abs(cohens_matrix.values.min()), abs(cohens_matrix.values.max()))
        sns.heatmap(
            cohens_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-vmax, vmax=vmax,
            annot=top_n <= 15,
            fmt='.2f' if top_n <= 15 else '',
            square=True,
            linewidths=0.5,
            cbar_kws={'label': "Cohen's d (Effect Size)"}
        )
    else:
        plt.imshow(cohens_matrix.values, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label="Cohen's d")
        plt.xticks(range(len(cohens_matrix.columns)), cohens_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(cohens_matrix.index)), cohens_matrix.index)

    plt.title(f"Feature Interaction Strength (QME: Cohen's d)\n"
              f"Read as: Row is MORE (+) / LESS (-) important when Column is HIGH",
              fontsize=11)
    plt.xlabel("Conditioning Feature (Q1 vs Q4)")
    plt.ylabel("Target Feature (SHAP change)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    return cohens_matrix


# ============================================================
# 5. INDIAN RUN: FULL PAIRWISE ANALYSIS
# ============================================================

def indian_run_analysis(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 25,
    class_idx: int = 1,
    print_top: int = 20,
    plot_heatmaps: bool = True,
    save_dir: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Full "Indian Run" pairwise feature interaction analysis.

    Computes both Option A (R²) and QME (Cohen's d) for all feature pairs,
    generates heatmaps, and prints top interactions.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features to analyze
        class_idx: Which class to analyze
        print_top: Number of top interactions to print
        plot_heatmaps: Whether to generate heatmap plots
        save_dir: Optional directory to save outputs

    Returns:
        Dict with 'r2_matrix', 'cohens_d_matrix', 'full_analysis'
    """
    print("=" * 70)
    print("  INDIAN RUN: Pairwise Feature Interaction Analysis")
    print("=" * 70)
    print(f"  Analyzing top {top_n} features × {top_n} features = {top_n**2 - top_n} pairs")
    print(f"  Methods: Option A (R²) + QME (Cohen's d)")
    print("=" * 70 + "\n")

    # Compute combined analysis
    print("Computing interactions...")
    full_analysis = compute_combined_interaction_analysis(
        X_sample, shap_values, feature_names, top_n, class_idx
    )

    # Compute matrices
    r2_matrix = compute_regression_interaction_matrix(
        X_sample, shap_values, feature_names, top_n, class_idx, metric='r2'
    )
    cohens_matrix = compute_qme_interaction_matrix(
        X_sample, shap_values, feature_names, top_n, class_idx, metric='cohens_d'
    )

    # Print top interactions
    print("\n" + "=" * 70)
    print(f"  TOP {print_top} STRONGEST INTERACTIONS (by |Cohen's d|)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Target':<22} {'Condition':<18} {'d':<8} {'R²':<8} {'Direction':<10}")
    print("-" * 70)

    for i, row in enumerate(full_analysis.head(print_top).itertuples(), 1):
        print(f"{i:<5} {row.target:<22} {row.condition:<18} "
              f"{row.cohens_d:+.3f}   {row.r2:.3f}    {row.direction} ({row.strength})")

    print("=" * 70)
    print("Interpretation: Target is [MORE/LESS] important when Condition is HIGH")
    print("Cohen's d: |d| > 0.5 = strong, |d| > 0.2 = moderate, |d| < 0.2 = weak")
    print("=" * 70 + "\n")

    # Plot heatmaps
    if plot_heatmaps:
        print("Generating R² heatmap...")
        save_r2 = f"{save_dir}/interaction_r2_heatmap.png" if save_dir else None
        plot_interaction_heatmap_r2(
            X_sample, shap_values, feature_names, top_n, class_idx, save_path=save_r2
        )

        print("Generating Cohen's d heatmap...")
        save_d = f"{save_dir}/interaction_cohens_d_heatmap.png" if save_dir else None
        plot_interaction_heatmap_cohens_d(
            X_sample, shap_values, feature_names, top_n, class_idx, save_path=save_d
        )

    # Summary statistics
    print("\n" + "=" * 70)
    print("  SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Total pairs analyzed: {len(full_analysis)}")
    print(f"  Strong interactions (|d| > 0.5): {(full_analysis['abs_cohens_d'] > 0.5).sum()}")
    print(f"  Moderate interactions (|d| > 0.2): {(full_analysis['abs_cohens_d'] > 0.2).sum()}")
    print(f"  Max R²: {full_analysis['r2'].max():.4f}")
    print(f"  Max |Cohen's d|: {full_analysis['abs_cohens_d'].max():.4f}")
    print("=" * 70 + "\n")

    return {
        'r2_matrix': r2_matrix,
        'cohens_d_matrix': cohens_matrix,
        'full_analysis': full_analysis
    }


# ============================================================
# 6. SINGLE PAIR DETAILED ANALYSIS
# ============================================================

def analyze_feature_pair(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    feature_1: str,
    feature_2: str,
    class_idx: int = 1,
    plot: bool = True
) -> Dict:
    """
    Detailed analysis of a single feature pair.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        feature_1: Target feature (SHAP analyzed)
        feature_2: Conditioning feature (split by value)
        class_idx: Which class to analyze
        plot: Whether to show scatter plot

    Returns:
        Dict with all metrics and interpretation
    """
    shap_class = shap_values[:, :, class_idx]
    f1_idx = feature_names.get_loc(feature_1)
    shap_f1 = shap_class[:, f1_idx]
    x_f2 = X_sample[feature_2].values

    # Option A: Regression
    reg = compute_regression_dependence(shap_f1, x_f2)

    # QME
    qme = compute_qme_dependence(shap_f1, x_f2)

    # Print results
    print("\n" + "=" * 60)
    print(f"  INTERACTION ANALYSIS: {feature_1} ↔ {feature_2}")
    print("=" * 60)

    print("\n  OPTION A: Regression (φ₁ ~ x₂)")
    print(f"    β (standardized): {reg['beta_std']:+.4f}")
    print(f"    R²:               {reg['r2']:.4f}")
    print(f"    p-value:          {reg['p_value']:.4f}")

    print("\n  QME: Quantile + Magnitude + Effect Size")
    print(f"    Mean SHAP (Q1 - low):  {qme['mean_q1']:+.5f}")
    print(f"    Mean SHAP (Q4 - high): {qme['mean_q4']:+.5f}")
    print(f"    Difference (Q4 - Q1):  {qme['diff']:+.5f}")
    print(f"    Cohen's d:             {qme['cohens_d']:+.4f}")

    # Interpretation
    direction = "MORE" if qme['diff'] > 0 else "LESS"
    d_abs = abs(qme['cohens_d'])
    strength = "strong" if d_abs > 0.5 else "moderate" if d_abs > 0.2 else "weak"

    print("\n  INTERPRETATION:")
    print(f"    → {feature_1} is {direction} important when {feature_2} is HIGH")
    print(f"    → Effect strength: {strength} (|d| = {d_abs:.3f})")
    print(f"    → {reg['r2']*100:.1f}% of {feature_1}'s SHAP variance explained by {feature_2}")
    print("=" * 60 + "\n")

    # Plot
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter plot
        ax1 = axes[0]
        scatter = ax1.scatter(x_f2, shap_f1, c=shap_f1, cmap='coolwarm', alpha=0.5, s=10)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)

        # Regression line
        x_range = np.linspace(x_f2.min(), x_f2.max(), 100)
        x_std = (x_range - x_f2.mean()) / (x_f2.std() + 1e-10)
        y_pred = shap_f1.mean() + reg['beta'] * x_std
        ax1.plot(x_range, y_pred, 'k-', linewidth=2, label=f'R²={reg["r2"]:.3f}')

        ax1.set_xlabel(feature_2)
        ax1.set_ylabel(f'SHAP({feature_1})')
        ax1.set_title(f'Continuous Dependence (Option A)')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='SHAP')

        # Box plot for Q1 vs Q4
        ax2 = axes[1]
        q1_thresh = np.percentile(x_f2, 25)
        q4_thresh = np.percentile(x_f2, 75)
        shap_q1 = shap_f1[x_f2 <= q1_thresh]
        shap_q4 = shap_f1[x_f2 >= q4_thresh]

        bp = ax2.boxplot([shap_q1, shap_q4], labels=['Q1 (Low)', 'Q4 (High)'])
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax2.set_ylabel(f'SHAP({feature_1})')
        ax2.set_title(f"QME: Cohen's d = {qme['cohens_d']:+.3f}")

        plt.tight_layout()
        plt.show()

    return {
        'feature_1': feature_1,
        'feature_2': feature_2,
        **reg,
        **qme,
        'direction': direction,
        'strength': strength
    }