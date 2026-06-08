
"""Production inference module for ensemble trading model."""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

class TradingInference:
    def __init__(self, deploy_dir: str = "."):
        self.deploy_dir = Path(deploy_dir)
        self._load_config()
        self._load_models()

    def _load_config(self):
        configs = list(self.deploy_dir.glob("config_*.json"))
        latest = sorted(configs)[-1]
        with open(latest) as f:
            self.config = json.load(f)
        self.class_idx = self.config["class_index_bullish"]
        self.thresholds = self.config["thresholds"]

    def _load_models(self):
        self.models = {}
        for name, info in self.config["models"].items():
            self.models[name] = joblib.load(self.deploy_dir / info["path"])

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Run full ensemble inference.

        Args:
            features: DataFrame with required feature columns

        Returns:
            dict with probability, signal, and component probabilities
        """
        # Base model probabilities
        probs = {
            "rf": self.models["rf"].predict_proba(features)[:, self.class_idx],
            "xgb": self.models["xgb"].predict_proba(features)[:, self.class_idx],
            "lgb": self.models["lgb"].predict_proba(features)[:, self.class_idx],
            "logit": self.models["logit"].predict_proba(features)[:, self.class_idx],
            "mlp": self.models["mlp"].predict_proba(features)[:, self.class_idx],
        }

        # Stack for blender
        stack = pd.DataFrame({f"{k}_prob": v for k, v in probs.items()})
        stack["meta_prob"] = 0.5  # Placeholder for live

        # Ensemble probability
        p_final = self.models["blender"].predict_proba(stack)[:, self.class_idx][0]

        # Signal generation
        if p_final >= self.thresholds["long"]:
            signal = 1
        elif p_final <= self.thresholds["short"]:
            signal = -1
        else:
            signal = 0

        return {
            "probability": float(p_final),
            "signal": signal,
            "signal_label": {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal],
            "components": {k: float(v[0]) for k, v in probs.items()},
            "confidence": abs(p_final - 0.5) * 2,
        }

    def check_drift(self, features: pd.DataFrame, stats_path: str = None) -> dict:
        """Check for feature drift against training distribution."""
        if stats_path is None:
            stats_files = list(self.deploy_dir.glob("feature_stats_*.json"))
            stats_path = sorted(stats_files)[-1]

        with open(stats_path) as f:
            train_stats = json.load(f)

        drift_flags = {}
        for col in features.columns:
            if col in train_stats["means"]:
                val = features[col].iloc[0]
                mean, std = train_stats["means"][col], train_stats["stds"][col]
                z_score = (val - mean) / (std + 1e-9)
                if abs(z_score) > 3:
                    drift_flags[col] = {"value": val, "z_score": z_score}

        return {"drift_detected": len(drift_flags) > 0, "flags": drift_flags}


if __name__ == "__main__":
    # Example usage
    engine = TradingInference(".")
    print(f"Loaded model version: {engine.config['version']}")
