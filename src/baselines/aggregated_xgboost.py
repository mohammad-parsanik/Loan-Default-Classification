import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, cohen_kappa_score, brier_score_loss
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class AggregatedXGBoostBaseline:
    """
    Baseline model that flattens portfolios by computing min, max, mean, std
    for all features across a customer's loans.
    """
    def __init__(self, random_state=42):
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
    def _aggregate_features(self, instances: List[Dict]) -> tuple:
        """Convert list of portfolio instances to flat feature matrix."""
        X = []
        y = []
        
        for inst in instances:
            features = inst['features']  # (n_loans, n_features)
            
            # Compute stats
            f_min = np.min(features, axis=0)
            f_max = np.max(features, axis=0)
            f_mean = np.mean(features, axis=0)
            f_std = np.std(features, axis=0)
            
            # Include n_loans as a feature
            n_loans = np.array([inst['n_loans']])
            
            # Concatenate all stats
            flat_features = np.concatenate([f_min, f_max, f_mean, f_std, n_loans])
            
            X.append(flat_features)
            y.append(inst['label'])
            
        return np.vstack(X), np.array(y)
        
    def train(self, train_inst: List[Dict], val_inst: List[Dict] = None):
        logger.info("Aggregating features for baseline...")
        X_train, y_train = self._aggregate_features(train_inst)
        
        eval_set = None
        if val_inst:
            X_val, y_val = self._aggregate_features(val_inst)
            eval_set = [(X_val, y_val)]
            
        logger.info(f"Training XGBoost on {X_train.shape[1]} aggregated features...")
        
        # Calculate sample weights (inverse class frequency)
        classes, counts = np.unique(y_train, return_counts=True)
        weight_dict = {c: len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weights = np.array([weight_dict[y] for y in y_train])
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            sample_weight=sample_weights,
            verbose=False
        )
        logger.info("Baseline training complete.")
        
    def evaluate(self, test_inst: List[Dict]) -> dict:
        X_test, y_test = self._aggregate_features(test_inst)
        
        probs = self.model.predict_proba(X_test)
        preds = self.model.predict(X_test)
        
        # Metrics
        macro_f1 = f1_score(y_test, preds, average='macro')
        qwk = cohen_kappa_score(y_test, preds, weights='quadratic')
        
        # One-hot encode true labels for Brier score
        y_true_oh = np.zeros((len(y_test), 3))
        y_true_oh[np.arange(len(y_test)), y_test] = 1
        brier = np.mean([brier_score_loss(y_true_oh[:, c], probs[:, c]) for c in range(3)])
        
        metrics = {
            "macro_f1": float(macro_f1),
            "qwk": float(qwk),
            "brier_score": float(brier)
        }
        
        logger.info(f"Baseline Results: Macro F1={macro_f1:.4f}, QWK={qwk:.4f}, Brier={brier:.4f}")
        return metrics
