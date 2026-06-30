import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import joblib
import torch
from tqdm import tqdm

import project_config as config
from src.data.data_explorer import explore_data
from src.data.data_loader import DataLoader
from src.data.preprocessing import create_preprocessing_pipeline
from src.data.temporal_split import split_by_time
from src.data.dataset import create_dataloaders
from src.baselines.aggregated_xgboost import AggregatedXGBoostBaseline
from src.model.set_transformer import SetTransformer
from src.model.losses import CostSensitiveFocalLoss
from src.model.trainer import TransformerTrainer
from src.model.meta_learner import XGBoostMetaLearner
from src.evaluation.visualization import plot_confusion_matrix, plot_roc_curves, plot_embeddings_umap
from src.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_pipeline():
    logger.info("=== Starting Training Pipeline ===")
    
    # Create artifact directory
    timestamp = datetime.now().strftime("%Y%M%d_%H%M%S")
    run_dir = config.ARTIFACT_DIR / timestamp
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    dl = DataLoader()
    instances, features = dl.load_train_portfolios()
    
    if not instances:
        logger.error("No instances loaded. Exiting.")
        return
        
    # 2. Split Data
    train_inst, val_inst, test_inst = split_by_time(instances)
    
    # 3. Fit Preprocessing on Train
    logger.info("Fitting preprocessing pipeline on training data...")
    preprocessor = create_preprocessing_pipeline(features, config.BINARY_FEATURES)
    
    X_train_raw = [i['features'] for i in train_inst]
    preprocessor.fit(X_train_raw)
    
    # Transform all splits
    def apply_prep(inst_list, name="data"):
        if not inst_list: return []
        logger.info(f"Extracting features for {name}...")
        X = [i['features'] for i in inst_list]
        logger.info(f"Scaling {name} features...")
        X_scaled = preprocessor.transform(X)
        for i, inst in enumerate(tqdm(inst_list, desc=f"Updating {name} features", total=len(inst_list))):
            inst['features'] = X_scaled[i]
        return inst_list
        
    train_inst = apply_prep(train_inst, "Train")
    val_inst = apply_prep(val_inst, "Val")
    test_inst = apply_prep(test_inst, "Test")
    
    # Save preprocessor
    joblib.dump(preprocessor, run_dir / "scaler.pkl")
    
    # Calculate MAX_LOANS from train set if not specified
    if config.MAX_LOANS_PER_CUSTOMER is None:
        loan_counts = [i['n_loans'] for i in train_inst]
        max_loans = int(np.percentile(loan_counts, 99))
        logger.info(f"Computed MAX_LOANS_PER_CUSTOMER (99th percentile): {max_loans}")
    else:
        max_loans = config.MAX_LOANS_PER_CUSTOMER
        
    # 4. Baseline Model
    logger.info("--- Training Baseline Model ---")
    baseline = AggregatedXGBoostBaseline()
    baseline.train(train_inst, val_inst)
    baseline_metrics = baseline.evaluate(test_inst)
    
    # 5. Dataloaders for Transformer
    train_dl, val_dl, test_dl = create_dataloaders(
        train_inst, val_inst, test_inst, 
        max_loans=max_loans, 
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # 6. Set-Transformer Training
    logger.info("--- Training Set-Transformer ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transformer = SetTransformer(
        n_features=len(features),
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_feedforward=config.D_FEEDFORWARD,
        dropout=config.DROPOUT
    )
    
    # Try torch.compile if supported
    try:
        transformer = torch.compile(transformer)
        logger.info("Enabled torch.compile for Set-Transformer")
    except:
        pass
        
    criterion = CostSensitiveFocalLoss(device=device)
    trainer = TransformerTrainer(transformer, criterion, config, device=device)
    
    trainer.train(train_dl, val_dl)
    
    # Save Transformer
    # Handle compiled model state dict correctly
    model_state = transformer._orig_mod.state_dict() if hasattr(transformer, '_orig_mod') else transformer.state_dict()
    torch.save(model_state, run_dir / "set_transformer.pt")
    
    # 7. XGBoost Meta-Learner
    logger.info("--- Training Meta-Learner ---")
    meta_learner = XGBoostMetaLearner(transformer, device=device)
    meta_learner.optimize_and_train(train_dl, val_dl, n_trials=30)  # 30 trials to save time
    
    # Save XGBoost
    meta_learner.model.save_model(run_dir / "xgboost_model.json")
    
    # 8. Evaluation
    logger.info("--- Final Evaluation ---")
    metrics = meta_learner.evaluate(test_dl)
    logger.info(f"Final Test Metrics: {metrics}")
    
    # 9. Visualizations
    # Get test embeddings and labels for UMAP
    import numpy as np
    X_test_emb, y_test = meta_learner._extract_embeddings(test_dl)
    plot_embeddings_umap(X_test_emb, y_test, save_path=plots_dir / "embedding_umap.png")
    
    # Get test predictions for plots
    y_prob = meta_learner.model.predict_proba(X_test_emb)
    y_pred = meta_learner.model.predict(X_test_emb)
    
    plot_confusion_matrix(y_test, y_pred, save_path=plots_dir / "confusion_matrix.png")
    plot_roc_curves(y_test, y_prob, save_path=plots_dir / "roc_curves.png")
    
    # Save metadata
    metadata = {
        'feature_count': len(features),
        'features': features,
        'max_loans_per_customer_99th': max_loans,
        'baseline_metrics': baseline_metrics,
        'final_metrics': metrics,
        'xgboost_params': meta_learner.best_params
    }
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"=== Pipeline Complete. Artifacts saved to {run_dir} ===")
    
def predict_pipeline(artifact_dir, snapshot_date, output_path):
    predictor = Predictor(Path(artifact_dir))
    predictor.predict(int(snapshot_date), output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Classification Pipeline")
    parser.add_argument('action', choices=['explore', 'train', 'predict'], help="Action to perform")
    parser.add_argument('--artifact_dir', type=str, help="Path to artifact directory for prediction")
    parser.add_argument('--snapshot_date', type=int, help="Snapshot date for prediction")
    parser.add_argument('--output', type=str, help="Output CSV path for prediction")
    
    args = parser.parse_args()
    
    if args.action == 'explore':
        explore_data()
    elif args.action == 'train':
        train_pipeline()
    elif args.action == 'predict':
        if not args.artifact_dir or not args.snapshot_date or not args.output:
            logger.error("Predict requires --artifact_dir, --snapshot_date, and --output")
        else:
            predict_pipeline(args.artifact_dir, args.snapshot_date, args.output)
