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


def _build_deep_sets(hparams: dict, state_dict, device: str) -> DeepSets:
    """Reconstruct a DeepSets model for inference (dropout always forced to 0)."""
    model = DeepSets(**{**hparams, "dropout": 0.0})
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _build_xgb_from_raw(raw_bytes) -> xgb.XGBClassifier:
    """Reconstruct an XGBClassifier from Booster.save_raw() bytes."""
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(bytearray(raw_bytes))
    return xgb_model


def load_bundle(bundle_path: Path, device: str = None):
    """
    Load a single-file model_bundle.pkl produced by train_single_fold.

    Returns:
        (scaler, deep_sets_model, xgb_model, calibrator, max_loans, feature_names)
        — the same shape as ModelLoader.load_pipeline().
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model bundle from {bundle_path}…")
    bundle = joblib.load(bundle_path)

    model = _build_deep_sets(bundle["deep_sets_hparams"], bundle["deep_sets_state_dict"], device)
    xgb_model = _build_xgb_from_raw(bundle["xgb_model_raw"])
    meta = bundle["metadata"]

    return (
        bundle["scaler"], model, xgb_model, bundle["calibrator"],
        meta["max_loans_per_customer_99th"], meta.get("features", []),
    )


class ModelLoader:
    """Loads the full inference pipeline from an artifact directory, or from a
    single model_bundle.pkl file (see load_bundle)."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_pipeline(self):
        """
        Returns:
            (scaler, deep_sets_model, xgb_model, calibrator, max_loans, feature_names)
            calibrator is None when no calibrator.pkl exists in the artifact dir.
        """
        if self.artifact_dir.is_file():
            return load_bundle(self.artifact_dir, device=self.device)

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
        hparams = {
            "n_features": feature_count,
            "hidden_dim": config.DEEPSETS_HIDDEN_DIM,
            "embedding_dim": config.DEEPSETS_EMBED_DIM,
            "num_classes": config.NUM_CLASSES,
        }
        state_dict = torch.load(self.artifact_dir / "deep_sets.pt", map_location=self.device)
        model = _build_deep_sets(hparams, state_dict, self.device)
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
