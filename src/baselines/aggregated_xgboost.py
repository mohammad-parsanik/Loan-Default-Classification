"""
Aggregated XGBoost baseline.

For each customer portfolio, computes min/max/mean/std across all loan features
and appends loan count → flat feature vector.  Trained directly with XGBoost
(no neural network).

Used as a comparison benchmark: the DeepSets pipeline must beat this by ≥ 3% Macro F1.
"""

import logging
from typing import Optional

import numpy as np
import xgboost as xgb

import project_config as config
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class AggregatedXGBoostBaseline:
    """
    Flattens portfolios via min/max/mean/std statistics per feature, then trains XGBoost.
    Works with instances in the standard dict format:
        {'features': (n_loans, n_features) np.float32, 'label': int, ...}
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None

    def _aggregate(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Convert portfolio instances → flat feature matrix."""
        X, y = [], []
        for inst in instances:
            f = inst["features"]  # (n_loans, n_features)
            row = np.concatenate([
                np.min(f,  axis=0),
                np.max(f,  axis=0),
                np.mean(f, axis=0),
                np.std(f,  axis=0),
                [inst["n_loans"]],
            ])
            X.append(row)
            y.append(inst["label"])
        return np.vstack(X), np.array(y, dtype=np.int32)

    @staticmethod
    def _sample_weights(y: np.ndarray) -> np.ndarray:
        """
        Inverse class-frequency weights, optionally scaled by the cost-matrix
        row sum per true class (BASELINE_COST_WEIGHTS) so training attention
        follows the business cost of misclassifying that class — same nudge
        the DeepSets model gets from its cost-sensitive loss.
        """
        classes, counts = np.unique(y, return_counts=True)
        w = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        if getattr(config, "BASELINE_COST_WEIGHTS", False):
            row_cost = np.asarray(config.COST_MATRIX).sum(axis=1)
            for c in classes:
                w[c] *= row_cost[c]
        return np.array([w[yi] for yi in y])

    def train(self, train_inst: list[dict], val_inst: Optional[list[dict]] = None) -> None:
        logger.info("Aggregating features for baseline…")
        X_train, y_train = self._aggregate(train_inst)
        sw_train = self._sample_weights(y_train)

        eval_set = None
        if val_inst:
            X_val, y_val = self._aggregate(val_inst)
            eval_set = [(X_val, y_val)]

        logger.info(f"Training XGBoost on {X_train.shape[1]} aggregated features…")
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=config.NUM_CLASSES,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            sample_weight=sw_train,
            verbose=False,
        )
        logger.info("Baseline training complete.")

    def predict_proba(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probs, y_true) for a list of instances."""
        X, y = self._aggregate(instances)
        return self.model.predict_proba(X), y

    def evaluate(self, test_inst: list[dict]) -> dict:
        probs, y_test = self.predict_proba(test_inst)
        preds = probs.argmax(axis=1)
        metrics = compute_metrics(y_test, preds, probs)
        logger.info(
            f"Baseline → Macro F1={metrics['macro_f1']:.4f}, "
            f"QWK={metrics['qwk']:.4f}, Brier={metrics.get('brier_score', '?'):.4f}"
        )
        return metrics


class BinarySevereBaseline(AggregatedXGBoostBaseline):
    """
    Binary comparator for the ranking deliverable: models the severe event
    y = (label == NUM_CLASSES-1) directly on the same aggregated features.
    Answers "does the multiclass detour cost ranking quality?" — its raw
    P(severe) is compared on recall@K/lift/PR-AUC, which are invariant to
    monotone calibration, so no calibrator is needed for the comparison.
    """

    def train(self, train_inst: list[dict], val_inst: Optional[list[dict]] = None) -> None:
        X_train, y_train = self._aggregate(train_inst)
        y_bin = (y_train == config.NUM_CLASSES - 1).astype(np.int32)
        n_pos = max(int(y_bin.sum()), 1)

        logger.info(
            f"Training binary severe-event comparator on {X_train.shape[1]} "
            f"features ({n_pos:,} positives / {len(y_bin):,})…"
        )
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_bin) - n_pos) / n_pos,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_bin, verbose=False)
        logger.info("Binary comparator training complete.")

    def severity_scores(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (P(severe), y_true_multiclass) for ranking evaluation."""
        X, y = self._aggregate(instances)
        return self.model.predict_proba(X)[:, 1], y
