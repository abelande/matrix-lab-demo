"""
visualization — SHAP and model interpretation visualization tools.
"""
""""" # Import from shap_explorer module
from .shap_explorer import (
    compute_interaction_matrix,
    plot_interaction_heatmap,
    plot_3d_interaction,
    create_interactive_shap_explorer,
    quick_interaction_analysis,
)
"""""
from .interaction_analysis import (
    compute_regression_interaction_matrix,
    compute_qme_interaction_matrix,
    compute_combined_interaction_analysis,
    plot_interaction_heatmap_r2,
    plot_interaction_heatmap_cohens_d,
    indian_run_analysis,
    analyze_feature_pair,
)

__all__ = [
    # shap_explorer (original median-split)
    "compute_interaction_matrix",
    "plot_interaction_heatmap",
    "plot_3d_interaction",
    "create_interactive_shap_explorer",
    "quick_interaction_analysis",
    # interaction_analysis (Option A + QME)
    "compute_regression_interaction_matrix",
    "compute_qme_interaction_matrix",
    "compute_combined_interaction_analysis",
    "plot_interaction_heatmap_r2",
    "plot_interaction_heatmap_cohens_d",
    "indian_run_analysis",
    "analyze_feature_pair",
]