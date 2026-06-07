# ============================================================
# SECTION 08 — Walkforward Simulation
# ============================================================

logger.info("Starting Section 08 — Walkforward Simulation")

# ------------------------------------------------------------
# 1. Walkforward CV Settings — Multi-Regime
# ------------------------------------------------------------

# Define ALL classes upfront for consistent probability columns
ALL_CLASSES = sorted(y.unique())  # e.g., [-1.0, 0.0, 1.0]
logger.info(f"All target classes: {ALL_CLASSES}")

# Store results for all regimes
all_regime_results = {}

for regime_name, regime_cfg in WF_CONFIGS.items():
    
    logger.info("=" * 60)
    logger.info(f"WALKFORWARD REGIME: {regime_cfg['name'].upper()}")
    logger.info(f"  Train: {regime_cfg['train_bars']} bars, Test: {regime_cfg['test_bars']} bars")
    logger.info("=" * 60)
    
    # Create splitter for this regime
    cv_walkforward = cv_walk.RollingWindowCV(
        train_size=regime_cfg["train_bars"],
        test_size=regime_cfg["test_bars"],
        step=regime_cfg["step_bars"],
        event_end_times=event_end_times
    )
    
    # ------------------------------------------------------------
    # 2. Initialize Model for Walkforward (per regime)
    # ------------------------------------------------------------
    
    wf_model = RandomForestClassifier(**MODEL_PARAMS)
    
    # ------------------------------------------------------------
    # 3. Walkforward Loop
    # ------------------------------------------------------------
    
    wf_predictions = []
    wf_metrics = []
    
    for w, (train_idx, test_idx) in enumerate(cv_walkforward.split(df_all)):
    
        logger.info(f"[{regime_name}] Window {w+1}: train={len(train_idx)}, test={len(test_idx)}")
    
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
    
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
    
        # Train model
        wf_model.fit(X_train, y_train)
    
        # Predict
        wf_probs = wf_model.predict_proba(X_test)
        wf_preds = wf_model.predict(X_test)
        
        # Get model's trained classes
        model_classes = list(wf_model.classes_)
    
        # Metrics (multiclass)
        acc = accuracy_score(y_test, wf_preds)
        f1_macro = f1_score(y_test, wf_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, wf_preds, average='weighted', zero_division=0)
        
        # Calculate ROC AUC - more lenient approach
        try:
            test_classes = set(np.unique(y_test))
            trained_classes = set(model_classes)
            
            # Only need 2+ classes in test set and model trained on them
            if len(test_classes) >= 2 and test_classes.issubset(trained_classes):
                # Build probability matrix for classes present in test
                if len(test_classes) == len(trained_classes):
                    # All classes present - use full probs
                    roc = roc_auc_score(y_test, wf_probs, multi_class='ovr', average='macro')
                else:
                    # Subset of classes - compute binary or subset ROC
                    test_classes_sorted = sorted(test_classes)
                    if len(test_classes_sorted) == 2:
                        # Binary case
                        pos_class = test_classes_sorted[-1]
                        pos_idx = model_classes.index(pos_class)
                        roc = roc_auc_score(
                            (y_test == pos_class).astype(int),
                            wf_probs[:, pos_idx]
                        )
                    else:
                        roc = np.nan
            else:
                roc = np.nan
        except Exception as e:
            logger.debug(f"ROC AUC failed for window {w}: {e}")
            roc = np.nan
    
        wf_metrics.append({
            "regime": regime_name,
            "window": w,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "roc_auc": roc,
            "n_train": len(train_idx),
            "n_test": len(test_idx)
        })
    
        # Store predictions with CONSISTENT probability columns for ALL classes
        pred_dict = {
            "timestamp": X_test.index,
            "true": y_test.values,
            "pred": wf_preds,
            "window": w
        }
        
        # Create probability columns for ALL classes (not just model's classes)
        for cls in ALL_CLASSES:
            if cls in model_classes:
                cls_idx = model_classes.index(cls)
                pred_dict[f"prob_class_{cls}"] = wf_probs[:, cls_idx]
            else:
                # Class not in model - fill with 0 probability
                pred_dict[f"prob_class_{cls}"] = np.zeros(len(y_test))
    
        wf_pred_df = pd.DataFrame(pred_dict).set_index("timestamp")
        wf_predictions.append(wf_pred_df)
    
    # ------------------------------------------------------------
    # Store regime (after all windows complete)
    # ------------------------------------------------------------
    
    df_wf_preds = pd.concat(wf_predictions).sort_index()
    df_wf_preds["regime"] = regime_name
    
    df_wf_metrics = pd.DataFrame(wf_metrics)
    
    all_regime_results[regime_name] = {
        "predictions": df_wf_preds,
        "metrics": df_wf_metrics,
    }
    
    logger.info(f"[{regime_name}] Complete — {len(df_wf_preds)} predictions, ROC valid: {df_wf_metrics['roc_auc'].notna().sum()}/{len(df_wf_metrics)}")

# === END OF REGIME LOOP ===
logger.info("Walkforward simulation complete.")

# ------------------------------------------------------------
# 4. Consolidate All Regimes
# ------------------------------------------------------------

df_all_preds = pd.concat([r["predictions"] for r in all_regime_results.values()])
df_all_metrics = pd.concat([r["metrics"] for r in all_regime_results.values()])

logger.info(f"Total predictions across regimes: {len(df_all_preds)}")

# ------------------------------------------------------------
# 5. Save Artifacts (Per-Regime + Combined)
# ------------------------------------------------------------

for regime_name, results in all_regime_results.items():
    pred_file = ROOT / "artifacts" / f"walkforward_predictions_{regime_name}.parquet"
    metrics_file = ROOT / "artifacts" / f"walkforward_metrics_{regime_name}.parquet"
    results["predictions"].to_parquet(pred_file)
    results["metrics"].to_parquet(metrics_file)
    logger.info(f"Saved {regime_name} artifacts")

# Combined
df_all_preds.to_parquet(ROOT / "artifacts" / "walkforward_predictions_all.parquet")
df_all_metrics.to_parquet(ROOT / "artifacts" / "walkforward_metrics_all.parquet")

logger.info("Saved combined walkforward artifacts.")

# ------------------------------------------------------------
# 6. Walkforward Diagnostics & Plots
# ------------------------------------------------------------

# 6A — REGIME COMPARISON (with NaN handling)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for regime_name, results in all_regime_results.items():
    df_m = results["metrics"]
    # Drop NaN for cleaner plots
    roc_valid = df_m[["window", "roc_auc"]].dropna()
    axes[0].plot(roc_valid["window"], roc_valid["roc_auc"], marker='o', label=regime_name, alpha=0.8)
    axes[1].plot(df_m["window"], df_m["f1_macro"], marker='o', label=regime_name, alpha=0.8)
    axes[2].plot(df_m["window"], df_m["accuracy"], marker='o', label=regime_name, alpha=0.8)

axes[0].set_title("ROC AUC by Regime")
axes[1].set_title("F1 Macro by Regime")
axes[2].set_title("Accuracy by Regime")

for ax in axes:
    ax.set_xlabel("Window")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ROOT / "artifacts" / "walkforward_regime_comparison.png", dpi=150)
plt.show()

# Summary table
summary = df_all_metrics.groupby("regime")[["roc_auc", "f1_macro", "accuracy"]].agg(['mean', 'count'])
print("\n" + "=" * 60)
print("REGIME COMPARISON — Mean Metrics (with valid counts)")
print("=" * 60)
print(summary.round(4))


# 6B — PER-REGIME DETAIL PLOTS
for regime_name, results in all_regime_results.items():
    
    df_wf_preds = results["predictions"].copy()
    df_wf_metrics = results["metrics"]
    
    print(f"\n{'='*60}")
    print(f"DETAILED PLOTS: {regime_name.upper()}")
    print(f"{'='*60}")
    
    # Window-level metrics (interpolate NaN for visualization)
    plt.figure(figsize=(15, 5))
    plt.plot(df_wf_metrics["window"], df_wf_metrics["roc_auc"].interpolate(), label="ROC AUC", linewidth=2)
    plt.plot(df_wf_metrics["window"], df_wf_metrics["f1_macro"], label="F1 Macro", linewidth=2)
    plt.plot(df_wf_metrics["window"], df_wf_metrics["accuracy"], label="Accuracy", linewidth=2)
    plt.title(f"[{regime_name}] Performance by Window")
    plt.xlabel("Window")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Prediction drift — use ALL_CLASSES for consistency
    wf_window = get_rolling_window(len(df_wf_preds))
    
    plt.figure(figsize=(15, 5))
    for cls in ALL_CLASSES:
        col = f"prob_class_{cls}"
        if col in df_wf_preds.columns:
            rolling_prob = df_wf_preds[col].rolling(wf_window, min_periods=1).mean()
            plt.plot(rolling_prob, label=f"P(Class {cls})", linewidth=1.5)
    plt.title(f"[{regime_name}] Prediction Drift (Rolling Mean Probability)")
    plt.ylabel("Mean Probability")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Error over time — overall + per-class
    df_wf_preds["error"] = (df_wf_preds["true"] != df_wf_preds["pred"]).astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    # Overall error rate
    axes[0].plot(df_wf_preds["error"].rolling(wf_window, min_periods=1).mean(), linewidth=1.5)
    axes[0].set_title(f"[{regime_name}] Rolling Error Rate (Overall)")
    axes[0].set_ylabel("Error Rate")
    axes[0].set_xlabel("Time")
    axes[0].grid(True, alpha=0.3)
    
    # Per-class error rate
    for cls in ALL_CLASSES:
        mask = df_wf_preds["true"] == cls
        if mask.sum() > 0:
            class_errors = df_wf_preds.loc[mask, "error"].rolling(wf_window // 3, min_periods=1).mean()
            axes[1].plot(class_errors, label=f"Class {cls}", alpha=0.8)
    axes[1].set_title(f"[{regime_name}] Rolling Error Rate (Per Class)")
    axes[1].set_ylabel("Error Rate")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 6C — Regime breakdown (using combined predictions)
df_all_preds_reg = df_all_preds.join(X["Regime_Combo"])

def compute_wf_regime_metrics(g):
    """Compute metrics for a regime group, handling edge cases."""
    metrics = {
        "accuracy": accuracy_score(g["true"], g["pred"]),
        "f1_macro": f1_score(g["true"], g["pred"], average="macro", zero_division=0),
        "count": len(g),
    }
    
    # ROC AUC with full error handling
    try:
        regime_prob_cols = [f"prob_class_{cls}" for cls in ALL_CLASSES]
        unique_classes = g["true"].nunique()
        
        prob_data = g[regime_prob_cols].values
        if np.isnan(prob_data).any():
            metrics["roc_auc"] = np.nan
        elif unique_classes < 2:
            metrics["roc_auc"] = np.nan
        elif unique_classes == len(ALL_CLASSES):
            metrics["roc_auc"] = roc_auc_score(
                g["true"], 
                prob_data,
                multi_class='ovr', 
                average='macro'
            )
        elif unique_classes == 2:
            # Binary subset
            present_classes = sorted(g["true"].unique())
            pos_class = present_classes[-1]
            metrics["roc_auc"] = roc_auc_score(
                (g["true"] == pos_class).astype(int),
                g[f"prob_class_{pos_class}"]
            )
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan
    
    return pd.Series(metrics)

wf_regime_scores = df_all_preds_reg.groupby("Regime_Combo").apply(compute_wf_regime_metrics)

logger.info(f"Walkforward regime scores:\n{wf_regime_scores}")

# Better bar chart with multiple metrics visible
fig, ax = plt.subplots(figsize=(12, 6))
wf_regime_scores[["accuracy", "f1_macro", "roc_auc"]].plot(kind="bar", ax=ax, width=0.8)
plt.title("Walkforward Performance by Market Regime")
plt.ylabel("Score")
plt.xlabel("Regime")
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Also print the table
print("\n" + "=" * 60)
print("WALKFORWARD METRICS BY MARKET REGIME")
print("=" * 60)
print(wf_regime_scores.round(4))

# Count by regime
fig, ax = plt.subplots(figsize=(10, 5))
wf_regime_scores["count"].plot(kind="bar", ax=ax, color="steelblue")
plt.title("Sample Count by Market Regime")
plt.ylabel("Count")
plt.xlabel("Regime")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Done — Section 08
# ------------------------------------------------------------

logger.info("Section 08 complete — Walkforward simulation executed and analyzed.")