"""
Load a trained model for inference and produce raw (uncalibrated) class
probabilities from portfolio instances.

Two model families are supported behind a common `Scorer` interface:
  - ArmScorer      — the deployed path (July 2026): scale → aggregate
                     features → XGBoost arm. No torch/DeepSets needed.
  - DeepSetsScorer — legacy DeepSets encoder + XGB meta-learner, kept for
                     old artifacts (config.DEEPSETS_ENABLED runs).

ModelLoader.load_pipeline() → (scorer, calibrator, features). It accepts
either an artifact DIRECTORY (fold_dir) or a single-file BUNDLE (.pkl);
the artifact kind is auto-detected.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config
from src.baselines.aggregated_xgboost import aggregate_features

logger = logging.getLogger(__name__)


# ── Scorers ───────────────────────────────────────────────────────────────────

class ArmScorer:
    """Deployed path: scale features, aggregate, run the XGBoost arm."""

    def __init__(self, scaler, arm, max_loans: int):
        self.scaler = scaler
        self.arm = arm
        self.max_loans = max_loans

    def raw_probs(self, instances: list[dict]) -> np.ndarray:
        X_scaled = self.scaler.transform([i["features"] for i in instances])
        agg_in = [
            {"features": x, "n_loans": i["n_loans"], "label": i.get("label", -1)}
            for i, x in zip(instances, X_scaled)
        ]
        X, _ = aggregate_features(agg_in)
        current_cat = np.array([i["current_cat"] for i in instances])
        return self.arm.predict_proba(X, current_cat)


class DeepSetsScorer:
    """Legacy path: DeepSets embeddings → XGBoost meta-learner."""

    def __init__(self, scaler, model, xgb_model, max_loans: int, device: str):
        self.scaler = scaler
        self.model = model
        self.xgb_model = xgb_model
        self.max_loans = max_loans
        self.device = device

    def raw_probs(self, instances: list[dict]) -> np.ndarray:
        import torch  # lazy: only the legacy path needs it
        from tqdm import tqdm

        X_scaled = self.scaler.transform([i["features"] for i in instances])
        n_features = X_scaled[0].shape[1]
        embeddings = []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(X_scaled), 512), desc="Embedding batches"):
                batch = X_scaled[i:i + 512]
                padded = np.zeros((len(batch), self.max_loans, n_features), dtype=np.float32)
                mask = np.ones((len(batch), self.max_loans), dtype=bool)
                for j, arr in enumerate(batch):
                    seq = min(len(arr), self.max_loans)
                    padded[j, :seq] = arr[:seq]
                    mask[j, :seq] = False
                emb = self.model.extract_embeddings(
                    torch.from_numpy(padded).to(self.device),
                    torch.from_numpy(mask).to(self.device),
                )
                embeddings.append(emb.cpu().numpy())
        return self.xgb_model.predict_proba(np.vstack(embeddings))


def _build_deepsets_scorer(scaler, hparams, state_dict, xgb_model, max_loans, device):
    from src.model.deep_sets import DeepSets   # lazy: legacy path only
    import torch
    model = DeepSets(**{**hparams, "dropout": 0.0})
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return DeepSetsScorer(scaler, model, xgb_model, max_loans, device)


# ── Bundles ───────────────────────────────────────────────────────────────────

def build_arm_bundle(scaler, arm, calibrator, max_loans, features) -> dict:
    """Single-file deployment artifact for the arm path (see DEPLOYMENT.md)."""
    return {
        "kind": "arm",
        "scaler": scaler,
        "arm": arm,
        "calibrator": calibrator,
        "metadata": {"max_loans_per_customer_99th": max_loans,
                     "features": features, "num_classes": config.NUM_CLASSES},
    }


def load_bundle(bundle_path: Path, device: str):
    """Load a single-file bundle (arm or legacy deepsets). → (scorer, calibrator, features)."""
    logger.info(f"Loading model bundle from {bundle_path}…")
    bundle = joblib.load(bundle_path)
    meta = bundle["metadata"]
    features = meta.get("features", [])
    max_loans = meta["max_loans_per_customer_99th"]

    if bundle.get("kind") == "arm":
        scorer = ArmScorer(bundle["scaler"], bundle["arm"], max_loans)
        return scorer, bundle.get("calibrator"), features

    # Legacy deepsets bundle
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(bytearray(bundle["xgb_model_raw"]))
    scorer = _build_deepsets_scorer(
        bundle["scaler"], bundle["deep_sets_hparams"], bundle["deep_sets_state_dict"],
        xgb_model, max_loans, device,
    )
    return scorer, bundle.get("calibrator"), features


# ── ModelLoader ───────────────────────────────────────────────────────────────

class ModelLoader:
    """Loads an inference scorer from an artifact directory or a bundle file."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        # torch is only needed for the legacy DeepSets path — don't force it
        # as a hard dependency for arm-only (torch-free) deployments.
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.device = "cpu"

    def _load_calibrator(self):
        cal_path = self.artifact_dir / "calibrator.pkl"
        if cal_path.exists():
            logger.info("Loaded probability calibrator.")
            return joblib.load(cal_path)
        logger.warning("No calibrator.pkl — probabilities used raw (queue ranking still valid).")
        return None

    def load_pipeline(self):
        """Returns (scorer, calibrator, feature_names)."""
        if self.artifact_dir.is_file():
            return load_bundle(self.artifact_dir, self.device)

        with open(self.artifact_dir / "metadata.json") as f:
            meta = json.load(f)
        features = meta.get("features", [])
        max_loans = meta["max_loans_per_customer_99th"]
        scaler = joblib.load(self.artifact_dir / "scaler.pkl")

        # Prefer the deployed arm artifact; fall back to legacy deepsets.
        arm_path = self.artifact_dir / "model_arm.pkl"
        if arm_path.exists():
            logger.info(f"Loading arm pipeline from {self.artifact_dir}…")
            scorer = ArmScorer(scaler, joblib.load(arm_path), max_loans)
            return scorer, self._load_calibrator(), features

        logger.info(f"Loading legacy DeepSets pipeline from {self.artifact_dir}…")
        import torch
        import xgboost as xgb
        hparams = {
            "n_features": meta["feature_count"],
            "hidden_dim": config.DEEPSETS_HIDDEN_DIM,
            "embedding_dim": config.DEEPSETS_EMBED_DIM,
            "num_classes": config.NUM_CLASSES,
        }
        state_dict = torch.load(self.artifact_dir / "deep_sets.pt", map_location=self.device)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(self.artifact_dir / "xgboost_model.json")
        scorer = _build_deepsets_scorer(scaler, hparams, state_dict, xgb_model, max_loans, self.device)
        return scorer, self._load_calibrator(), features
