"""
Signal Extensions Package
=========================
Advanced signal processing utilities for the RGNB trading system.
"""

from .signal_extensions import (
    # Core functions
    generate_reversion_signals,
    compute_conviction_metrics,
    detect_divergences,
    compute_risk_adjusted_size,
    apply_regime_filter,
    run_all_extensions,
    # Quick-call helpers
    quick_reversion,
    quick_conviction,
    quick_divergence,
    quick_sizing,
    quick_filter,
)

__all__ = [
    "generate_reversion_signals",
    "compute_conviction_metrics",
    "detect_divergences",
    "compute_risk_adjusted_size",
    "apply_regime_filter",
    "run_all_extensions",
    "quick_reversion",
    "quick_conviction",
    "quick_divergence",
    "quick_sizing",
    "quick_filter",
]
