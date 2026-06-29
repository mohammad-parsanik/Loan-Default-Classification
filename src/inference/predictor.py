import torch
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from src.data.data_loader import DataLoader
from src.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class Predictor:
    """End-to-end scoring pipeline."""
    def __init__(self, artifact_dir):
        self.loader = ModelLoader(artifact_dir)
        self.scaler, self.transformer, self.xgb_model, self.max_loans = self.loader.load_pipeline()
        self.device = self.loader.device
        self.data_loader = DataLoader()
        
    def predict(self, snapshot_date: int, output_path=None) -> pd.DataFrame:
        logger.info(f"Starting inference for snapshot {snapshot_date}")
        
        # 1. Load data from Oracle
        instances, _ = self.data_loader.load_pred_portfolios(snapshot_date, self.max_loans)
        
        if not instances:
            logger.warning("No data found to predict.")
            return pd.DataFrame()
            
        # 2. Extract features to list of arrays
        X_raw = [inst['features'] for inst in instances]
        national_codes = [inst['national_code'] for inst in instances]
        n_loans_list = [inst['n_loans'] for inst in instances]
        
        # 3. Preprocessing
        logger.info("Applying preprocessing (impute, clip, scale)...")
        X_scaled = self.scaler.transform(X_raw)
        
        # 4. Create PyTorch tensors
        logger.info("Running Set-Transformer embedding extraction...")
        # (Using a simple loop since inference batching can just be done in memory if it fits,
        # otherwise we'd use a DataLoader. For ~10k customers, doing it in batches is safer.)
        
        batch_size = 512
        all_embeddings = []
        
        self.transformer.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(X_scaled), batch_size), desc="Embedding batches"):
                batch_X = X_scaled[i:i+batch_size]
                
                # Pad batch
                n_features = batch_X[0].shape[1]
                padded = np.zeros((len(batch_X), self.max_loans, n_features), dtype=np.float32)
                mask = np.ones((len(batch_X), self.max_loans), dtype=bool)
                
                for j, inst in enumerate(batch_X):
                    seq_len = len(inst)
                    padded[j, :seq_len] = inst
                    mask[j, :seq_len] = False
                    
                t_features = torch.tensor(padded).to(self.device)
                t_mask = torch.tensor(mask).to(self.device)
                
                emb = self.transformer.extract_embeddings(t_features, t_mask)
                all_embeddings.append(emb.cpu().numpy())
                
        customer_embeddings = np.vstack(all_embeddings)
        
        # 5. XGBoost Prediction
        logger.info("Generating final risk scores...")
        probs = self.xgb_model.predict_proba(customer_embeddings)
        preds = self.xgb_model.predict(customer_embeddings)
        
        # 6. Formatting results
        results = pd.DataFrame({
            'NATIONAL_CODE': national_codes,
            'N_LOANS_IN_PORTFOLIO': n_loans_list,
            'P_NO_DELAY': probs[:, 0],
            'P_CURRENT': probs[:, 1],
            'P_PAST_DUE': probs[:, 2],
            'PREDICTED_CLASS': preds,
            'RISK_SCORE': probs[:, 2] * 2.0 + probs[:, 1] * 1.0  # Custom weighted risk score
        })
        
        # Sort by risk (highest first)
        results = results.sort_values(by='RISK_SCORE', ascending=False)
        
        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Saved predictions to {output_path}")
            
        return results
