import torch
import numpy as np
import xgboost as xgb
import optuna
import logging
from typing import Dict
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

class XGBoostMetaLearner:
    """
    Takes frozen embeddings from the Set-Transformer and trains an XGBoost classifier.
    Includes Optuna hyperparameter optimization.
    """
    def __init__(self, transformer, device="cpu", random_state=42):
        self.transformer = transformer
        self.transformer.eval()
        self.device = device
        self.random_state = random_state
        self.best_params = None
        self.model = None
        
    def _extract_embeddings(self, dataloader) -> tuple:
        """Passes all data through the transformer to get (B, 64) embeddings."""
        all_emb = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)
                labels = batch['label']
                
                embeddings = self.transformer.extract_embeddings(features, padding_mask)
                
                all_emb.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                
        return np.vstack(all_emb), np.concatenate(all_labels)

    def optimize_and_train(self, train_dl, val_dl, n_trials=50):
        logger.info("Extracting embeddings for XGBoost training...")
        X_train, y_train = self._extract_embeddings(train_dl)
        X_val, y_val = self._extract_embeddings(val_dl)
        
        # Calculate sample weights (inverse class frequency)
        classes, counts = np.unique(y_train, return_counts=True)
        weight_dict = {c: len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weights_train = np.array([weight_dict[y] for y in y_train])
        
        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5.0),
                'random_state': self.random_state,
                'n_jobs': 4  # Limit jobs per trial
            }
            
            model = xgb.XGBClassifier(**params)
            
            # Use early stopping via callbacks in new XGBoost versions
            early_stop = xgb.callback.EarlyStopping(rounds=20, metric_name='mlogloss', data_name='validation_0')
            
            model.fit(
                X_train, y_train,
                sample_weight=sample_weights_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stop],
                verbose=False
            )
            
            preds = model.predict(X_val)
            metrics = compute_metrics(y_val, preds)
            return metrics['macro_f1']
            
        logger.info(f"Running Optuna optimization with {n_trials} trials...")
        # Reduce verbosity of Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=4)  # Parallel trials
        
        self.best_params = study.best_params
        logger.info(f"Best Optuna Macro F1: {study.best_value:.4f}")
        logger.info(f"Best Params: {self.best_params}")
        
        # Retrain on Train+Val with best params
        logger.info("Retraining on Train+Val with best parameters...")
        X_full = np.vstack((X_train, X_val))
        y_full = np.concatenate((y_train, y_val))
        
        classes, counts = np.unique(y_full, return_counts=True)
        weight_dict = {c: len(y_full) / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weights_full = np.array([weight_dict[y] for y in y_full])
        
        final_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': self.random_state,
            'n_jobs': -1  # Use all cores for final model
        }
        final_params.update(self.best_params)
        
        self.model = xgb.XGBClassifier(**final_params)
        self.model.fit(X_full, y_full, sample_weight=sample_weights_full, verbose=False)
        
        logger.info("XGBoost Meta-Learner training complete.")
        
    def evaluate(self, dataloader) -> Dict:
        """Evaluate the final XGBoost model on the given dataloader (e.g., test set)."""
        X_test, y_test = self._extract_embeddings(dataloader)
        
        probs = self.model.predict_proba(X_test)
        preds = self.model.predict(X_test)
        
        return compute_metrics(y_test, preds, probs)
