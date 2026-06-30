"""
End-to-end scoring pipeline using the trained DeepSets + XGBoost pipeline.
"""

import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.data_loader import DataLoader
from src.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class Predictor:
    """Loads trained artifacts and scores a new snapshot."""

    def __init__(self, artifact_dir):
        from pathlib import Path
        self.loader = ModelLoader(Path(artifact_dir))
        self.scaler, self.model, self.xgb_model, self.max_loans, _ = (
            self.loader.load_pipeline()
        )
        self.device = self.loader.device
        self.data_loader = DataLoader()

    def predict(self, snapshot_date: int, output_path=None) -> pd.DataFrame:
        logger.info(f"Starting inference for snapshot {snapshot_date}")

        instances, _ = self.data_loader.load_pred_portfolios(
            snapshot_date, self.max_loans
        )
        if not instances:
            logger.warning("No data found to predict.")
            return pd.DataFrame()

        # Preprocessing
        logger.info("Applying scaler…")
        X_raw    = [inst["features"] for inst in instances]
        X_scaled = self.scaler.transform(X_raw)

        national_codes = [inst["national_code"] for inst in instances]
        n_loans_list   = [inst["n_loans"]       for inst in instances]

        # DeepSets embedding extraction in batches
        logger.info("Extracting DeepSets embeddings…")
        batch_size    = 512
        n_features    = X_scaled[0].shape[1]
        all_embeddings = []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(X_scaled), batch_size), desc="Embedding batches"):
                batch_X = X_scaled[i : i + batch_size]
                n = len(batch_X)

                padded = np.zeros((n, self.max_loans, n_features), dtype=np.float32)
                mask   = np.ones((n, self.max_loans), dtype=bool)

                for j, arr in enumerate(batch_X):
                    seq_len = min(len(arr), self.max_loans)
                    padded[j, :seq_len] = arr[:seq_len]
                    mask[j, :seq_len]   = False

                t_feat = torch.from_numpy(padded).to(self.device)
                t_mask = torch.from_numpy(mask).to(self.device)

                emb = self.model.extract_embeddings(t_feat, t_mask)
                all_embeddings.append(emb.cpu().numpy())

        customer_embeddings = np.vstack(all_embeddings)

        # XGBoost prediction
        logger.info("Generating risk scores via XGBoost…")
        probs = self.xgb_model.predict_proba(customer_embeddings)
        preds = self.xgb_model.predict(customer_embeddings)

        results = pd.DataFrame({
            "NATIONAL_CODE":        national_codes,
            "N_LOANS_IN_PORTFOLIO": n_loans_list,
            "P_NO_DELAY":           probs[:, 0],
            "P_CURRENT":            probs[:, 1],
            "P_PAST_DUE":           probs[:, 2],
            "PREDICTED_CLASS":      preds,
            # Weighted risk score: Cat-2 weight=2, Cat-1 weight=1
            "RISK_SCORE":           probs[:, 2] * 2.0 + probs[:, 1] * 1.0,
        })

        results = results.sort_values("RISK_SCORE", ascending=False).reset_index(drop=True)

        if output_path:
            results.to_csv(output_path, index=False)
            logger.info(f"Saved {len(results):,} predictions to {output_path}")

        return results
