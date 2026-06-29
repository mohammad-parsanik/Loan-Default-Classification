import torch
import xgboost as xgb
import joblib
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config
from src.model.set_transformer import SetTransformer

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_pipeline(self):
        """Loads the scaler, transformer, and XGBoost models."""
        logger.info(f"Loading models from {self.artifact_dir}...")
        
        # 1. Load Metadata
        with open(self.artifact_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        feature_count = metadata['feature_count']
        max_loans = metadata['max_loans_per_customer_99th']
        
        # 2. Load Preprocessing Pipeline
        scaler_path = self.artifact_dir / "scaler.pkl"
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler pipeline.")
        
        # 3. Load Set-Transformer
        transformer = SetTransformer(
            n_features=feature_count,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            d_feedforward=config.D_FEEDFORWARD,
            dropout=0.0  # Turn off dropout for inference
        )
        
        transformer_path = self.artifact_dir / "set_transformer.pt"
        transformer.load_state_dict(torch.load(transformer_path, map_location=self.device))
        transformer.to(self.device)
        transformer.eval()
        logger.info("Loaded Set-Transformer.")
        
        # 4. Load XGBoost
        xgb_path = self.artifact_dir / "xgboost_model.json"
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        logger.info("Loaded XGBoost Meta-Learner.")
        
        return scaler, transformer, xgb_model, max_loans
