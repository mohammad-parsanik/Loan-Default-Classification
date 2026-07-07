import json
import logging
from pathlib import Path

import joblib
import torch
import xgboost as xgb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config
from src.model.deep_sets import DeepSets

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads the full inference pipeline from an artifact directory."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pipeline(self):
        """
        Returns:
            (scaler, deep_sets_model, xgb_model, calibrator, max_loans, feature_names)
            calibrator is None when no calibrator.pkl exists in the artifact dir.
        """
        logger.info(f"Loading pipeline from {self.artifact_dir}…")

        # Metadata
        with open(self.artifact_dir / "metadata.json") as f:
            meta = json.load(f)

        feature_count = meta["feature_count"]
        max_loans     = meta["max_loans_per_customer_99th"]
        features      = meta.get("features", [])

        # Scaler
        scaler = joblib.load(self.artifact_dir / "scaler.pkl")
        logger.info("Loaded scaler pipeline.")

        # DeepSets model
        model = DeepSets(
            n_features=feature_count,
            hidden_dim=config.DEEPSETS_HIDDEN_DIM,
            embedding_dim=config.DEEPSETS_EMBED_DIM,
            dropout=0.0,   # dropout off for inference
        )
        model.load_state_dict(
            torch.load(self.artifact_dir / "deep_sets.pt", map_location=self.device)
        )
        model.to(self.device)
        model.eval()
        logger.info("Loaded DeepSets model.")

        # XGBoost
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(self.artifact_dir / "xgboost_model.json")
        logger.info("Loaded XGBoost meta-learner.")

        # Probability calibrator (optional — absent on runs without a val set)
        cal_path = self.artifact_dir / "calibrator.pkl"
        calibrator = None
        if cal_path.exists():
            calibrator = joblib.load(cal_path)
            logger.info("Loaded probability calibrator.")
        else:
            logger.warning(
                "No calibrator.pkl found — probabilities will be used raw "
                "(cost-rule decisions may be miscalibrated)."
            )

        return scaler, model, xgb_model, calibrator, max_loans, features
