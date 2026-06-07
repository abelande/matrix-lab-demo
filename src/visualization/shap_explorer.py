"""
shap_explorer.py — Interactive SHAP Feature Interaction Explorer

Provides interactive 3D visualization and analysis tools for exploring
feature interactions based on SHAP values.

Usage in notebook:
    from visualization.shap_explorer import (
        create_interactive_shap_explorer,
        compute_interaction_matrix,
        plot_interaction_heatmap,
        plot_3d_interaction
    )

    # Interactive explorer (requires plotly + ipywidgets)
    explorer = create_interactive_shap_explorer(X_sample, shap_values, X.columns)
    display(explorer)

    # Static heatmap
    plot_interaction_heatmap(X_sample, shap_values, X.columns, top_n=15)

    # Single 3D plot
    plot_3d_interaction(X_sample, shap_values, X.columns, "RegR2_50", "VASlope_50")

##### REFERENCE USAGE SNIPPET #####
# ------------------------------------------------------------
# 8. Feature Interaction Analysis
# ------------------------------------------------------------
if RUN_SHAP:
    logger.info("Analyzing feature interactions...")
    
    # Static heatmap of top 15 feature interactions
    int_matrix = plot_interaction_heatmap(
        X_sample, shap_values, X.columns, 
        top_n=15, 
        class_idx=1,
        save_path=VISUAL_SAVE_DIR / "shap_interaction_heatmap.png"
    )
    
    # Quick text summary
    quick_interaction_analysis(X_sample, shap_values, X.columns, top_n=10)
    
    # Single 3D plot for specific features
    plot_3d_interaction(
        X_sample, shap_values, X.columns,
        feature_1="RegR2_50",
        feature_2="VASlope_50",
        class_idx=1
    )

# ------------------------------------------------------------
# 9. Interactive Explorer (Optional)
# ------------------------------------------------------------
if RUN_SHAP:
    logger.info("Launching interactive SHAP explorer...")
    explorer = create_interactive_shap_explorer(
        X_sample, shap_values, X.columns, class_idx=1
    )
    display(explorer)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import ipywidgets as widgets
    from ipywidgets import VBox, HBox
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ============================================================
# 1. INTERACTION MATRIX COMPUTATION
# ============================================================

def compute_interaction_matrix(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 20,
    class_idx: int = 1
) -> pd.DataFrame:
  """
    Compute pairwise interaction strength between top features.

    Interaction is measured as the difference in mean SHAP value for feature_1
    when feature_2 is above vs below its median.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features to include
        class_idx: Which class to analyze (default=1)

    Returns:
        DataFrame with interaction strengths (positive = more important when high)
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
                median_f2 = X_sample[f2].median()
                mask_high = X_sample[f2] > median_f2
                mask_low = X_sample[f2] <= median_f2

                shap_high = shap_f1[mask_high].mean() if mask_high.sum() > 0 else 0
                shap_low = shap_f1[mask_low].mean() if mask_low.sum() > 0 else 0

                interaction_matrix.loc[f1, f2] = shap_high - shap_low

    return interaction_matrix.astype(float)


# ============================================================
# 2. INTERACTION HEATMAP (STATIC)
# ============================================================

def plot_interaction_heatmap(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 15,
    class_idx: int = 1,
    figsize: tuple = (14, 12),
    cmap: str = "RdBu_r",
    save_path: str = None
) -> pd.DataFrame:
    """
    Plot heatmap of feature interaction strengths.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top features to include
        class_idx: Which class to analyze
        figsize: Figure size tuple
        cmap: Colormap name
        save_path: Optional path to save figure

    Returns:
        Interaction matrix DataFrame
    """
    # Compute interaction matrix
    int_matrix = compute_interaction_matrix(
        X_sample, shap_values, feature_names, top_n, class_idx
    )

    # Plot
    plt.figure(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            int_matrix,
            cmap=cmap,
            center=0,
            annot=top_n <= 15,  # Only annotate if not too crowded
            fmt=".3f" if top_n <= 15 else "",
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Interaction Strength\n(SHAP difference)"}
        )
    else:
        plt.imshow(int_matrix.values, cmap=cmap, aspect='auto')
        plt.colorbar(label="Interaction Strength")
        plt.xticks(range(len(int_matrix.columns)), int_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(int_matrix.index)), int_matrix.index)

    plt.title(f"Feature Interaction Strength (Top {top_n} Features)\n"
              f"Read as: Row feature importance changes when Column feature is high vs low",
              fontsize=11)
    plt.xlabel("Conditioning Feature (high vs low)")
    plt.ylabel("Target Feature (SHAP change)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    # Print strongest interactions
    print("\n" + "=" * 60)
    print("TOP 10 STRONGEST INTERACTIONS")
    print("=" * 60)

    # Flatten and sort
    interactions = []
    for f1 in int_matrix.index:
        for f2 in int_matrix.columns:
            if f1 != f2:
                val = int_matrix.loc[f1, f2]
                interactions.append((f1, f2, val, abs(val)))

    interactions.sort(key=lambda x: x[3], reverse=True)

    for i, (f1, f2, val, _) in enumerate(interactions[:10], 1):
        direction = "MORE" if val > 0 else "LESS"
        print(f"{i:2}. {f1} is {direction} important when {f2} is HIGH ({val:+.4f})")

    print("=" * 60 + "\n")

    return int_matrix


# ============================================================
# 3. STATIC 3D INTERACTION PLOT
# ============================================================

def plot_3d_interaction(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    feature_1: str,
    feature_2: str,
    class_idx: int = 1,
    figsize: tuple = (12, 8),
    save_path: str = None
):
    """
    Plot static 3D scatter of feature interaction.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        feature_1: First feature (X-axis, SHAP on Z-axis)
        feature_2: Second feature (Y-axis, conditioning)
        class_idx: Which class to analyze
        figsize: Figure size
        save_path: Optional path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    shap_class = shap_values[:, :, class_idx]
    f1_idx = feature_names.get_loc(feature_1)
    shap_f1 = shap_class[:, f1_idx]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_sample[feature_1],
        X_sample[feature_2],
        shap_f1,
        c=shap_f1,
        cmap="coolwarm",
        alpha=0.6,
        s=10
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(f"SHAP ({feature_1})")
    plt.colorbar(scatter, label="SHAP value", shrink=0.6)
    plt.title(f"Feature Interaction: {feature_1} vs {feature_2}")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

    # Print interaction stats
    median_f2 = X_sample[feature_2].median()
    shap_high = shap_f1[X_sample[feature_2] > median_f2].mean()
    shap_low = shap_f1[X_sample[feature_2] <= median_f2].mean()

    print(f"\nInteraction Analysis: {feature_1} ↔ {feature_2}")
    print(f"─" * 50)
    print(f"Mean SHAP for {feature_1}:")
    print(f"  When {feature_2} is HIGH: {shap_high:+.4f}")
    print(f"  When {feature_2} is LOW:  {shap_low:+.4f}")
    print(f"  Difference:              {shap_high - shap_low:+.4f}")

    if abs(shap_high - shap_low) > 0.001:
        direction = "MORE" if shap_high > shap_low else "LESS"
        print(f"\n→ Model relies on {feature_1} {direction} when {feature_2} is high")
    else:
        print(f"\n→ No significant interaction detected")


# ============================================================
# 4. INTERACTIVE 3D EXPLORER (REQUIRES PLOTLY)
# ============================================================

def create_interactive_shap_explorer(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    class_idx: int = 1,
    initial_features: tuple = None
):
    """
    Create interactive 3D SHAP feature interaction explorer.

    Use dropdowns to select features, rotate/zoom the 3D plot,
    and see interaction statistics update live.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        class_idx: Which class to analyze
        initial_features: Optional (feature_1, feature_2) tuple for initial view

    Returns:
        VBox widget to display in notebook

    Usage:
        explorer = create_interactive_shap_explorer(X_sample, shap_values, X.columns)
        display(explorer)
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Interactive explorer requires plotly and ipywidgets. "
            "Install with: pip install plotly ipywidgets"
        )

    # Precompute SHAP values for selected class
    shap_class = shap_values[:, :, class_idx]
    feature_list = feature_names.tolist()

    # Initial features
    if initial_features:
        init_f1, init_f2 = initial_features
    else:
        # Use top 2 most important features
        mean_imp = np.abs(shap_class).mean(axis=0)
        top_idx = np.argsort(mean_imp)[-2:][::-1]
        init_f1 = feature_names[top_idx[0]]
        init_f2 = feature_names[top_idx[1]]

    # Create figure
    fig = go.FigureWidget()

    # Add initial scatter trace
    f1_idx = feature_names.get_loc(init_f1)
    initial_shap = shap_class[:, f1_idx]

    fig.add_trace(go.Scatter3d(
        x=X_sample[init_f1],
        y=X_sample[init_f2],
        z=initial_shap,
        mode='markers',
        marker=dict(
            size=3,
            color=initial_shap,
            colorscale='RdBu_r',
            colorbar=dict(title="SHAP", thickness=15),
            opacity=0.7
        ),
        hovertemplate=(
            f"{init_f1}: %{{x:.3f}}<br>"
            f"{init_f2}: %{{y:.3f}}<br>"
            "SHAP: %{z:.4f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title=f"Interaction: {init_f1} vs {init_f2}",
        scene=dict(
            xaxis_title=init_f1,
            yaxis_title=init_f2,
            zaxis_title="SHAP value"
        ),
        width=900,
        height=650,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Stats output
    stats_output = widgets.Output()

    def update_plot(feature_1, feature_2):
        """Update plot when dropdowns change."""
        f1_idx = feature_names.get_loc(feature_1)
        shap_f1 = shap_class[:, f1_idx]

        with fig.batch_update():
            fig.data[0].x = X_sample[feature_1]
            fig.data[0].y = X_sample[feature_2]
            fig.data[0].z = shap_f1
            fig.data[0].marker.color = shap_f1
            fig.data[0].hovertemplate = (
                f"{feature_1}: %{{x:.3f}}<br>"
                f"{feature_2}: %{{y:.3f}}<br>"
                "SHAP: %{z:.4f}<extra></extra>"
            )

            fig.layout.scene.xaxis.title = feature_1
            fig.layout.scene.yaxis.title = feature_2
            fig.layout.title = f"Interaction: {feature_1} vs {feature_2}"

        # Update stats
        stats_output.clear_output()
        with stats_output:
            median_f2 = X_sample[feature_2].median()
            mask_high = X_sample[feature_2] > median_f2
            mask_low = ~mask_high

            shap_high = shap_f1[mask_high].mean()
            shap_low = shap_f1[mask_low].mean()
            diff = shap_high - shap_low

            print("═" * 55)
            print(f"  INTERACTION ANALYSIS: {feature_1} ↔ {feature_2}")
            print("═" * 55)
            print(f"  Mean SHAP for {feature_1}:")
            print(f"    When {feature_2} is HIGH:  {shap_high:+.5f}")
            print(f"    When {feature_2} is LOW:   {shap_low:+.5f}")
            print(f"    Difference:               {diff:+.5f}")
            print("─" * 55)

            if abs(diff) > 0.001:
                direction = "MORE" if diff > 0 else "LESS"
                strength = "strongly" if abs(diff) > 0.005 else "slightly"
                print(f"  → Model {strength} relies on {feature_1} {direction}")
                print(f"    when {feature_2} is high")
            else:
                print(f"  → No significant interaction detected")
            print("═" * 55)

    # Create dropdowns
    dropdown_f1 = widgets.Dropdown(
        options=feature_list,
        value=init_f1,
        description='Feature 1 (X):',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px')
    )

    dropdown_f2 = widgets.Dropdown(
        options=feature_list,
        value=init_f2,
        description='Feature 2 (Y):',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px')
    )

    def on_change(change):
        update_plot(dropdown_f1.value, dropdown_f2.value)

    dropdown_f1.observe(on_change, names='value')
    dropdown_f2.observe(on_change, names='value')

    # Initial stats
    update_plot(init_f1, init_f2)

    # Layout
    title_label = widgets.HTML(
        value="<h3>Interactive SHAP Feature Interaction Explorer</h3>"
                "<p style='color:gray'>Select features from dropdowns. "
                "Drag to rotate, scroll to zoom.</p>"
    )
    controls = HBox([dropdown_f1, dropdown_f2])

    return VBox([title_label, controls, fig, stats_output])


# ============================================================
# 5. QUICK ANALYSIS FUNCTION
# ============================================================

def quick_interaction_analysis(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: pd.Index,
    top_n: int = 10,
    class_idx: int = 1
):
    """
    Quick text-based summary of strongest feature interactions.

    Args:
        X_sample: Feature values DataFrame
        shap_values: 3D array (samples, features, classes)
        feature_names: Column names from X
        top_n: Number of top interactions to show
        class_idx: Which class to analyze
    """
    int_matrix = compute_interaction_matrix(
        X_sample, shap_values, feature_names,
        top_n=min(30, len(feature_names)),
        class_idx=class_idx
    )

    # Flatten and sort
    interactions = []
    for f1 in int_matrix.index:
        for f2 in int_matrix.columns:
            if f1 != f2:
                val = int_matrix.loc[f1, f2]
                interactions.append({
                    'target': f1,
                    'condition': f2,
                    'strength': val,
                    'abs_strength': abs(val)
                })

    df = pd.DataFrame(interactions)
    df = df.sort_values('abs_strength', ascending=False).head(top_n)

    print("\n" + "=" * 70)
    print(f"  TOP {top_n} FEATURE INTERACTIONS (Class {class_idx})")
    print("=" * 70)
    print(f"{'Rank':<5} {'Target Feature':<25} {'Condition':<20} {'Effect':<10}")
    print("-" * 70)

    for i, row in enumerate(df.itertuples(), 1):
        direction = "↑ MORE" if row.strength > 0 else "↓ LESS"
        print(f"{i:<5} {row.target:<25} {row.condition:<20} {direction} ({row.strength:+.4f})")

    print("=" * 70)
    print("Read as: 'Target is [MORE/LESS] important when Condition is HIGH'")
    print("=" * 70 + "\n")

    return df