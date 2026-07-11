"""
XGBoost model arms on aggregated portfolio features.

For each customer portfolio: min/max/mean/std across all loan features +
loan count → 257-feature vector. Since the Run-5 shootout these tree arms
ARE the model candidates (the DeepSets pipeline lost on every ranking
slice); see project_config.MODEL_ARMS for the lineup.

All arms share the same features and the same interface:
    train(train_inst, val_inst=None)
    predict_proba(instances) -> (probs, y_true)
Full-distribution arms return (N, NUM_CLASSES) probabilities; the binary
diagnostic arm returns (N, 2) [1-p, p] so calibration code can be reused.
"""

import logging
from typing import Optional

import numpy as np
import xgboost as xgb

import project_config as config
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Shared hyperparameters for every arm — deliberately identical so the
# comparison isolates the OBJECTIVE/decomposition, not tuning luck.
XGB_DEFAULTS = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
)


class AggregatedXGBoostBaseline:
    """
    The "multiclass" arm: one multi:softprob model over all 4 classes.
    Also the base class providing feature aggregation for the other arms.
    Works with instances in the standard dict format:
        {'features': (n_loans, n_features) np.float32, 'label': int,
         'current_cat': int, ...}
    """

    name = "multiclass"
    full_distribution = True

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
            random_state=self.random_state,
            **XGB_DEFAULTS,
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
    The "binary" arm: models the severe event y = (label == NUM_CLASSES-1)
    directly. RANKING-CEILING DIAGNOSTIC ONLY — it cannot produce the
    per-class probabilities the business needs, so it is never deployed.
    predict_proba returns (N, 2) [1-p, p] so StratifiedCalibrator applies.
    """

    name = "binary"
    full_distribution = False

    def train(self, train_inst: list[dict], val_inst: Optional[list[dict]] = None) -> None:
        X_train, y_train = self._aggregate(train_inst)
        y_bin = (y_train == config.NUM_CLASSES - 1).astype(np.int32)
        n_pos = max(int(y_bin.sum()), 1)

        logger.info(
            f"Training binary severe-event arm on {X_train.shape[1]} "
            f"features ({n_pos:,} positives / {len(y_bin):,})…"
        )
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=(len(y_bin) - n_pos) / n_pos,
            random_state=self.random_state,
            **XGB_DEFAULTS,
        )
        self.model.fit(X_train, y_bin, verbose=False)
        logger.info("Binary arm training complete.")

    def predict_proba(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Returns ((N, 2) [P(not severe), P(severe)], y_true_multiclass)."""
        X, y = self._aggregate(instances)
        p = self.model.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - p, p]), y

    def severity_scores(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (P(severe), y_true_multiclass) for ranking evaluation."""
        probs, y = self.predict_proba(instances)
        return probs[:, 1], y


class OrdinalXGBArm(AggregatedXGBoostBaseline):
    """
    The "ordinal" arm: exploits the ordered target with NUM_CLASSES-1
    cumulative binaries P(y > k). Full distribution by differencing:
      P(0) = 1 - P(y>0);  P(k) = P(y>k-1) - P(y>k);  P(3) = P(y>2)
    Cumulative probs are clamped monotone non-increasing before
    differencing (independent binaries can cross slightly).
    """

    name = "ordinal"
    full_distribution = True

    def train(self, train_inst: list[dict], val_inst: Optional[list[dict]] = None) -> None:
        X_train, y_train = self._aggregate(train_inst)
        self.models = []
        for k in range(config.NUM_CLASSES - 1):
            y_gt = (y_train > k).astype(np.int32)
            n_pos = max(int(y_gt.sum()), 1)
            logger.info(f"Ordinal arm: training P(y > {k}) "
                        f"({n_pos:,} positives / {len(y_gt):,})…")
            mdl = xgb.XGBClassifier(
                objective="binary:logistic",
                scale_pos_weight=(len(y_gt) - n_pos) / n_pos,
                random_state=self.random_state,
                **XGB_DEFAULTS,
            )
            mdl.fit(X_train, y_gt, verbose=False)
            self.models.append(mdl)
        logger.info("Ordinal arm training complete.")

    def predict_proba(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._aggregate(instances)
        cum = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        cum = np.minimum.accumulate(cum, axis=1)      # enforce P(y>0)>=P(y>1)>=P(y>2)

        probs = np.empty((len(X), config.NUM_CLASSES))
        probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, config.NUM_CLASSES - 1):
            probs[:, k] = cum[:, k - 1] - cum[:, k]
        probs[:, -1] = cum[:, -1]

        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum(axis=1, keepdims=True), y


class PerCatXGBArm(AggregatedXGBoostBaseline):
    """
    The "per_cat" arm (Mohammad's per-category-models idea): one model per
    current_cat stratum c, over that stratum's REACHABLE classes {c..3}
    (label >= current_cat by construction). Natively monotone — a cat-1
    customer gets exactly P(stay 1), P(->2), P(->3) and zero on class 0.
    Cross-stratum score comparability comes from the per-stratum calibrator
    downstream, same as every other arm.
    """

    name = "per_cat"
    full_distribution = True

    def train(self, train_inst: list[dict], val_inst: Optional[list[dict]] = None) -> None:
        self.models = {}
        for c in range(config.NUM_CLASSES - 1):       # cat 3 is trivially P(3)=1
            sub = [i for i in train_inst if i["current_cat"] == c]
            if not sub:
                logger.warning(f"Per-cat arm: no training data for stratum {c}.")
                continue
            X, y = self._aggregate(sub)
            # Labels shifted into {0 .. 3-c}; monotonicity guard for any
            # upstream violation (shouldn't happen by ETL definition)
            y_shift = np.maximum(y - c, 0)
            observed = np.unique(y_shift)
            if len(observed) < 2:
                logger.warning(f"Per-cat arm: stratum {c} has a single outcome "
                               f"({observed}) — using the point-mass fallback.")
                continue

            logger.info(f"Per-cat arm: training stratum {c} "
                        f"({len(sub):,} instances, {len(observed)} observed classes)…")
            if len(observed) == 2:
                mdl = xgb.XGBClassifier(
                    objective="binary:logistic",
                    random_state=self.random_state, **XGB_DEFAULTS,
                )
            else:
                mdl = xgb.XGBClassifier(
                    objective="multi:softprob",
                    random_state=self.random_state, **XGB_DEFAULTS,
                )
            mdl.fit(X, y_shift, verbose=False)
            self.models[c] = mdl
        logger.info("Per-cat arm training complete.")

    def predict_proba(self, instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._aggregate(instances)
        strata = np.array([i["current_cat"] for i in instances])
        probs = np.zeros((len(X), config.NUM_CLASSES))

        for c, mdl in self.models.items():
            mask = strata == c
            if not mask.any():
                continue
            p = mdl.predict_proba(X[mask])
            # Column mapping via classes_: XGB only emits columns for the
            # classes it SAW in this stratum (binary label 1 there means
            # "shifted class 1"), so map each observed class back to its
            # absolute index c + class.
            cols = np.asarray(mdl.classes_, dtype=int) + c
            probs[np.ix_(mask, cols)] = p

        # Already-severe stratum (and any stratum without a model): all
        # mass on the highest reachable class = current_cat.
        unhandled = ~np.isin(strata, list(self.models.keys()))
        if unhandled.any():
            probs[unhandled, :] = 0.0
            probs[unhandled, np.minimum(strata[unhandled], config.NUM_CLASSES - 1)] = 1.0

        probs = np.clip(probs, 0.0, None)
        return probs / probs.sum(axis=1, keepdims=True), y


ARM_BUILDERS = {
    "multiclass": AggregatedXGBoostBaseline,
    "binary":     BinarySevereBaseline,
    "ordinal":    OrdinalXGBArm,
    "per_cat":    PerCatXGBArm,
}
