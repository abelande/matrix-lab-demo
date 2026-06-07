# ============================================================
# SECTION 07 — Model Diagnostics & Cross-Validation Analysis
# ============================================================

logger.info("Starting Section 07 — Diagnostics & CV Analysis")

# ------------------------------------------------------------
# 0. Load and Clean Predictions
# ------------------------------------------------------------

df_preds = df_predictions.copy()
df_preds["timestamp"] = df_preds.index

# Clean predictions (drop NaN)
prob_cols = [col for col in df_preds.columns if col.startswith("prob_class_")]
df_preds_clean = df_preds.dropna(subset=["true"] + prob_cols)
logger.info(f"Cleaned predictions: {len(df_preds_clean)} rows (dropped {len(df_preds) - len(df_preds_clean)} NaN rows)")

# Dynamic class detection
actual_classes = sorted(df_preds_clean["true"].unique())
n_classes = len(actual_classes)

# Rolling window helper
def get_rolling_window(n_rows: int, target_pct: float = 0.15, min_window: int = 15) -> int:
    """
    Dynamically compute rolling window size based on data length.
    
    Args:
        n_rows: Number of rows in the dataset
        target_pct: Target window as percentage of data (default 15%)
        min_window: Minimum window size
    
    Returns:
        Window size clamped between min and max
    """
    window = int(n_rows * target_pct)
    return max(min_window, window)

# Compute once for this section
window_size = get_rolling_window(len(df_preds_clean))
logger.info(f"Using rolling window size: {window_size} (based on {len(df_preds_clean)} rows)")

# Join with regime data for later analysis
df_preds_with_regime = df_preds_clean.join(X["Regime_Combo"])


# ------------------------------------------------------------
# 1. Confusion Matrix (Across All Folds)
# ------------------------------------------------------------

logger.info("Generating overall confusion matrix...")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(df_preds_clean["true"], df_preds_clean["pred"])
actual_classes = sorted(df_preds_clean["true"].unique())
disp = ConfusionMatrixDisplay(cm, display_labels=actual_classes)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Purged CV")
plt.show()


# ------------------------------------------------------------
# 2. ROC AUC Score (dependent)
# ------------------------------------------------------------

y_true = df_preds_clean["true"]

if n_classes == 1:
    logger.warning("Only 1 class present — cannot compute ROC AUC")
    roc_auc = np.nan

elif n_classes == 2:
    # Binary classification
    pos_class = actual_classes[1]  # Second class is positive
    y_score = df_preds_clean[f"prob_class_{pos_class}"]
    roc_auc = roc_auc_score(y_true == pos_class, y_score)
    logger.info(f"Binary ROC AUC: {roc_auc:.4f}")

else:
    # Multi-class (3+ classes)
    y_probs = df_preds_clean[prob_cols].values
    roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    logger.info(f"Multi-class ROC AUC (macro): {roc_auc:.4f}")



# ------------------------------------------------------------
# 3. Precision-Recall Curve (Per Class)
# ------------------------------------------------------------

# ------------------------------------------------------------
# 3. Precision-Recall Curves (Per Class)
# ------------------------------------------------------------

logger.info("Plotting precision-recall curves...")

from sklearn.metrics import precision_recall_curve

fig, ax = plt.subplots(figsize=(10, 6))
for cls in actual_classes:
    y_binary = (df_preds_clean["true"] == cls).astype(int)
    prec, rec, _ = precision_recall_curve(y_binary, df_preds_clean[f"prob_class_{cls}"])
    ax.plot(rec, prec, label=f"Class {cls}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves by Class")
ax.legend()
plt.show()


# ------------------------------------------------------------
# 4. Rolling Error Rate
# ------------------------------------------------------------

logger.info("Computing rolling error rate...")

df_preds_clean["error"] = df_preds_clean["true"] != df_preds_clean["pred"].astype(int)

plt.figure(figsize=(14, 4))
plt.plot(df_preds_clean["error"].rolling(window_size).mean(), label=f"Rolling Error Rate ({window_size})")
plt.xlabel("Date")
plt.ylabel("Error Rate")
plt.title("Rolling Error Rate Over Time")
plt.legend()
plt.show()

# Verify rolling error is working
print(f"\nError column stats:")
print(f"  Mean: {df_preds_clean['error'].mean():.4f}")
print(f"  Sum: {df_preds_clean['error'].sum()}")
print(f"  Total rows: {len(df_preds_clean)}")
print(f"\nFirst 10 error values: {df_preds_clean['error'].head(10).tolist()}")
print(f"\nAccuracy: {(df_preds_clean['true'] == df_preds_clean['pred']).mean():.4f}")



# ------------------------------------------------------------
# 5. Error Analysis by Regime  
# ------------------------------------------------------------

logger.info("Analyzing errors by regime...")

# Re-join with regime (in case X index differs)
df_preds_with_regime = df_preds_clean.join(X["Regime_Combo"], how="left")

# Group by regime
regime_errors = df_preds_with_regime.groupby("Regime_Combo").agg(
    error_rate=("error", "mean"),
    count=("error", "count")
).sort_values("error_rate", ascending=False)

print("\nError Rate by Regime:")
print(regime_errors.round(4))


# ------------------------------------------------------------
# 6. Per-Fold Performance Summary
# ------------------------------------------------------------

logger.info("Computing per-fold performance...")

fold_results = []

if n_classes == 2:
    pos_class = actual_classes[1]
    
if n_classes >= 2:
    for fold in df_preds_clean["fold"].unique():
        fold_df = df_preds_clean[df_preds_clean["fold"] == fold]
        
        acc = (fold_df["true"] == fold_df["pred"]).mean()
        
        # Compute ROC AUC for this fold
        if n_classes == 2:
            try:
                auc = roc_auc_score(
                    fold_df["true"] == pos_class,
                    fold_df[f"prob_class_{pos_class}"]
                )
            except:
                auc = np.nan
        else:
            try:
                auc = roc_auc_score(
                    fold_df["true"],
                    fold_df[prob_cols].values,
                    multi_class='ovr',
                    average='macro'
                )
            except:
                auc = np.nan
        
        fold_results.append({
            "fold": fold,
            "accuracy": acc,
            "roc_auc": auc,
            "n_samples": len(fold_df)
        })
    
    df_fold_summary = pd.DataFrame(fold_results)
    print("\nPer-Fold Summary:")
    print(df_fold_summary.round(4))
    
    print(f"\nMean Accuracy: {df_fold_summary['accuracy'].mean():.4f} ± {df_fold_summary['accuracy'].std():.4f}")
    print(f"Mean ROC AUC:  {df_fold_summary['roc_auc'].mean():.4f} ± {df_fold_summary['roc_auc'].std():.4f}")

# ------------------------------------------------------------
# DEBUG: Check fold distribution
print(f"Folds in df_preds: {df_preds['fold'].unique()}")
print(f"Rows per fold: {df_preds['fold'].value_counts()}")
print(f"X shape: {X.shape}, df_preds shape: {df_preds.shape}")
print(f"Index overlap: {len(df_preds.index.intersection(X.index))}")

print(df_preds.columns.tolist())
print("fold" in df_preds.columns)
# ============================================================

# 7a. Feature Stability Across CV Folds
# ------------------------------------------------------------

logger.info("Computing feature stability across folds...")

# Compute correlations of predictions to each feature
feature_stability = {}
prob_cols = [c for c in df_preds.columns if c.startswith("prob_class_")]
# ------------------------------------------------------------
print("NaN count per prob column:")
print(df_preds[prob_cols].isna().sum())
print(f"\nTotal rows with any NaN prob: {df_preds[prob_cols].isna().any(axis=1).sum()} / {len(df_preds)}")
# ------------------------------------------------------------
# DEBUG: Check one fold in detail
test_fold = 0
fold_df = df_preds[df_preds["fold"] == test_fold]
fold_df = fold_df[~fold_df.index.duplicated(keep='first')]
valid_idx = fold_df.index.intersection(X.index)
print(f"\nFold {test_fold} debug:")
print(f"  Fold rows: {len(fold_df)}")
print(f"  Valid idx overlap: {len(valid_idx)}")
print(f"  Sample X values (first feature): {X.loc[valid_idx, X.columns[0]].head()}")
print(f"  Sample prob values: {fold_df.loc[valid_idx, prob_cols].max(axis=1).head()}")
# Check for NaN/constant issues
test_col = X.columns[0]
x_vals = X.loc[valid_idx, test_col].values
prob_vals = fold_df.loc[valid_idx, prob_cols].max(axis=1).values
print(f"\n  x_vals - NaN: {np.isnan(x_vals).sum()}, std: {np.nanstd(x_vals):.6f}")
print(f"  prob_vals - NaN: {np.isnan(prob_vals).sum()}, std: {np.nanstd(prob_vals):.6f}")
print(f"  Correlation: {np.corrcoef(x_vals, prob_vals)[0, 1]}")
# ------------------------------------------------------------
for col in X.columns:
    corrs = []
    for fold in df_preds["fold"].unique():
        fold_df = df_preds[df_preds["fold"] == fold]
        fold_df = fold_df[~fold_df.index.duplicated(keep='first')]  # Remove duplicate indices
        valid_idx = fold_df.index.intersection(X.index)
        if len(valid_idx) > 0:
            x_vals = X.loc[valid_idx, col].values
            prob_vals = fold_df.loc[valid_idx, prob_cols].max(axis=1).values
            corrs.append(np.corrcoef(x_vals, prob_vals)[0, 1])
    feature_stability[col] = np.nanmean(corrs)

# create df_stability from the dictionary
df_stability = pd.DataFrame.from_dict(feature_stability, orient="index", columns=["stability"])
df_stability.sort_values("stability", ascending=False, inplace=True)
# Display top stable features
stable_features = pd.Series(feature_stability).sort_values(ascending=False)
print("\nTop 10 Stable Features Across Folds:")
print(stable_features.head(10).round(4))

# ------------------------------------------------------------
# 7b. Probability Calibration Curves
# ------------------------------------------------------------

# Note: Full calibration analysis with Brier score moved to Section 07a

logger.info("Computing ECE (binned) for calibration check...")

from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 5))
if n_classes == 1:
    axes = [axes]

for idx, cls in enumerate(actual_classes):
    ax = axes[idx]
    y_binary = (df_preds_clean["true"] == cls).astype(int)
    y_prob = df_preds_clean[f"prob_class_{cls}"]
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_binary, y_prob, n_bins=10, strategy='uniform'
    )
    
    ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f"Class {cls}")
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve — Class {cls}")
    ax.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. Per-Class Rolling AUC (Stability Over Time)
# ------------------------------------------------------------

logger.info("Computing rolling AUC per class...")

# For each class, compute rolling AUC if binary data available
rolling_aucs = {}

if n_classes >= 2:
    for cls in actual_classes:
        try:
            aucs = []
            for fold in df_preds_clean["fold"].unique():
                idx = df_preds_clean[df_preds_clean["fold"] == fold].index
                valid_idx = [i for i in idx if i in df_preds_clean.index]
                if len(valid_idx) > 0 and f"prob_class_{cls}" in df_preds_clean.columns:
                    y_bin = (df_preds_clean.loc[valid_idx, "true"] == cls).astype(int)
                    y_prob = df_preds_clean.loc[valid_idx, f"prob_class_{cls}"]
                    if y_bin.nunique() == 2:
                        aucs.append(roc_auc_score(y_bin, y_prob))
            rolling_aucs[cls] = np.mean(aucs) if aucs else np.nan
        except Exception as e:
            logger.warning(f"Could not compute AUC for class {cls}: {e}")
            rolling_aucs[cls] = np.nan

    print("\nMean Rolling AUC per Class:")
    for cls, auc in rolling_aucs.items():
        print(f"  Class {cls}: {auc:.4f}" if not np.isnan(auc) else f"  Class {cls}: N/A")


# ------------------------------------------------------------
# 9. Rolling Class Probabilities
# ------------------------------------------------------------

logger.info("Plotting rolling class probabilities...")

plt.figure(figsize=(14, 5))
for cls in actual_classes:
    col = f"prob_class_{cls}"
    if col in df_preds_clean.columns:
        plt.plot(df_preds_clean[col].rolling(window_size).mean(), label=f"P(Class {cls})")
plt.xlabel("Date")
plt.ylabel("Mean Probability")
plt.title("Rolling Mean Predicted Probabilities")
plt.legend()
plt.show()

# ------------------------------------------------------------
# 10. Unified Brier Decomposition Function
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def brier_decomposition(y_true, y_prob, classes=None, class_labels=None, n_bins=10):
    """
    Unified Brier score decomposition for any number of classes.
    
    Parameters
    ----------
    y_true : array-like
        True class labels (can be any format: -1/1, 0/1, 0/1/2, etc.)
    y_prob : array-like or DataFrame
        - For binary: 1D array of probabilities for positive class
        - For multi-class: 2D array or DataFrame with shape (n_samples, n_classes)
    classes : list, optional
        Class labels in order matching prob columns. Auto-detected if None.
    class_labels : dict, optional
        Pretty names for classes, e.g., {-1: "Bearish", 1: "Bullish"}
    n_bins : int
        Number of bins for calibration grouping
    
    Returns
    -------
    dict with all computed metrics
    """
    y_true = np.asarray(y_true)
    
    # --- Detect classes and reshape probabilities ---
    if classes is None:
        classes = np.unique(y_true)
        classes = sorted(classes)
    
    n_classes = len(classes)
    n_samples = len(y_true)
    
    # Handle probability input formats
    if isinstance(y_prob, pd.DataFrame):
        y_prob_matrix = y_prob.values
    elif isinstance(y_prob, np.ndarray):
        if y_prob.ndim == 1:
            # Binary case: single probability column
            if n_classes == 2:
                y_prob_matrix = np.column_stack([1 - y_prob, y_prob])
            else:
                raise ValueError("1D y_prob only valid for binary classification")
        else:
            y_prob_matrix = y_prob
    else:
        y_prob_matrix = np.asarray(y_prob)
    
    # Validate shape
    if y_prob_matrix.shape[1] != n_classes:
        raise ValueError(f"y_prob has {y_prob_matrix.shape[1]} columns but {n_classes} classes detected")
    
    # --- One-hot encode true labels ---
    y_onehot = np.zeros((n_samples, n_classes))
    for i, cls in enumerate(classes):
        y_onehot[:, i] = (y_true == cls).astype(int)
    
    # --- Per-class decomposition ---
    class_results = {}
    
    for i, cls in enumerate(classes):
        y_binary = y_onehot[:, i]
        y_prob_cls = y_prob_matrix[:, i]
        
        y_mean = np.mean(y_binary)  # Base rate for this class
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob_cls, bins, right=True)
        bin_indices = np.clip(bin_indices, 1, n_bins) - 1
        
        reliability = 0.0
        resolution = 0.0
        
        for b in range(n_bins):
            mask = bin_indices == b
            n_k = np.sum(mask)
            
            if n_k > 0:
                p_k = np.mean(y_prob_cls[mask])
                o_k = np.mean(y_binary[mask])
                reliability += n_k * (p_k - o_k) ** 2
                resolution += n_k * (o_k - y_mean) ** 2
        
        reliability /= n_samples
        resolution /= n_samples
        uncertainty = y_mean * (1 - y_mean)
        
        brier = brier_score_loss(y_binary, y_prob_cls)
        
        label = class_labels.get(cls, str(cls)) if class_labels else str(cls)
        class_results[cls] = {
            'label': label,
            'brier': brier,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'base_rate': y_mean
        }
    
    # --- Multi-class Brier score ---
    multiclass_brier = np.mean(np.sum((y_prob_matrix - y_onehot) ** 2, axis=1))
    
    # Aggregate decomposition (weighted average across classes)
    avg_reliability = np.mean([r['reliability'] for r in class_results.values()])
    avg_resolution = np.mean([r['resolution'] for r in class_results.values()])
    avg_uncertainty = np.mean([r['uncertainty'] for r in class_results.values()])
    
    # --- Additional Uncertainty Metrics (computed on full probability matrix) ---
    
    # Prediction entropy per sample, then averaged
    p_clipped = np.clip(y_prob_matrix, 1e-10, 1 - 1e-10)
    sample_entropy = -np.sum(p_clipped * np.log2(p_clipped), axis=1)
    pred_entropy = np.mean(sample_entropy)
    max_entropy = np.log2(n_classes)  # Maximum possible entropy
    normalized_entropy = pred_entropy / max_entropy if max_entropy > 0 else 0
    
    # Confidence: max probability per sample
    confidence = np.mean(np.max(y_prob_matrix, axis=1))
    
    # Prediction variance per class, averaged
    pred_variance = np.mean(np.var(y_prob_matrix, axis=0))
    
    # Sharpness: how far from uniform distribution (1/n_classes)
    uniform_prob = 1.0 / n_classes
    sharpness = np.mean(np.abs(y_prob_matrix - uniform_prob))
    
    # Accuracy and overconfidence
    y_pred = np.array([classes[i] for i in np.argmax(y_prob_matrix, axis=1)])
    correct_mask = y_pred == y_true
    accuracy = np.mean(correct_mask)
    
    # Confidence when wrong
    if np.any(~correct_mask):
        overconfidence = np.mean(np.max(y_prob_matrix[~correct_mask], axis=1))
    else:
        overconfidence = 0.0
    
    # Confidence when right
    if np.any(correct_mask):
        correct_confidence = np.mean(np.max(y_prob_matrix[correct_mask], axis=1))
    else:
        correct_confidence = 0.0
    
    # --- Print Results ---
    print("=" * 60)
    print(f"BRIER SCORE DECOMPOSITION ({n_classes}-CLASS)")
    print("=" * 60)
    
    # Per-class results
    print("\n--- Per-Class Breakdown ---")
    print(f"{'Class':<12} {'Brier':>8} {'Reliab':>8} {'Resol':>8} {'Uncert':>8} {'Base%':>8}")
    print("-" * 60)
    
    for cls in classes:
        r = class_results[cls]
        print(f"{r['label']:<12} {r['brier']:>8.4f} {r['reliability']:>8.4f} "
              f"{r['resolution']:>8.4f} {r['uncertainty']:>8.4f} {r['base_rate']*100:>7.1f}%")
    
    print("-" * 60)
    print(f"{'Average':<12} {np.mean([r['brier'] for r in class_results.values()]):>8.4f} "
          f"{avg_reliability:>8.4f} {avg_resolution:>8.4f} {avg_uncertainty:>8.4f}")
    
    # Multi-class total
    print("\n--- Multi-Class Total ---")
    print(f"Multi-class Brier Score: {multiclass_brier:.4f}  (lower is better)")
    print(f"  = Mean squared error across all class probabilities")
    
    # Uncertainty metrics
    print("\n--- Uncertainty Metrics ---")
    print(f"Prediction Entropy:      {pred_entropy:.4f} / {max_entropy:.4f}  ({normalized_entropy*100:.1f}% of max)")
    print(f"Mean Confidence:         {confidence:.4f}  (avg max probability)")
    print(f"Prediction Variance:     {pred_variance:.4f}  (spread of estimates)")
    print(f"Sharpness:               {sharpness:.4f}  (distance from uniform)")
    
    # Calibration diagnostics
    print("\n--- Calibration Diagnostics ---")
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Confidence when RIGHT:   {correct_confidence:.4f}")
    print(f"Confidence when WRONG:   {overconfidence:.4f}")
    
    confidence_gap = overconfidence - (1 - accuracy) if overconfidence > 0 else 0
    if overconfidence > 0.7 and accuracy < 0.5:
        print(f"  ⚠ High overconfidence on errors - consider calibration")
    
    print("=" * 60)
    
    # --- Return all results ---
    return {
        'n_classes': n_classes,
        'classes': classes,
        'per_class': class_results,
        'multiclass_brier': multiclass_brier,
        'avg_reliability': avg_reliability,
        'avg_resolution': avg_resolution,
        'avg_uncertainty': avg_uncertainty,
        'pred_entropy': pred_entropy,
        'normalized_entropy': normalized_entropy,
        'confidence': confidence,
        'pred_variance': pred_variance,
        'sharpness': sharpness,
        'accuracy': accuracy,
        'correct_confidence': correct_confidence,
        'overconfidence': overconfidence
    }


# ------------------------------------------------------------
# 11. Brier Score Analysis
# ------------------------------------------------------------

logger.info("Computing Brier score decomposition...")

prob_cols = [f"prob_class_{c}" for c in actual_classes]
results = brier_decomposition(
    df_preds_clean["true"],
    df_preds_clean[prob_cols],
    classes=list(actual_classes),
    class_labels={-1.0: "Bearish", 1.0: "Bullish"}
)

logger.info("Section 07 Complete.")

# ------------------------------------------------------------
# VALIDATION CHECK — Section 07
# ------------------------------------------------------------
quick_check(df_preds_clean, "Clean Predictions")
print(f"Bull col: {get_bull_col(df_preds_clean)}, Bear col: {get_bear_col(df_preds_clean)}")

# Regime validation
if "Regime_Combo" in df_preds_with_regime.columns:
    validate_by_regime(df_preds_with_regime, regime_col="Regime_Combo", target_col="true", pred_col="pred")