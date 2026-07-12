"""
XGBoost model arms on aggregated portfolio features.

For each customer portfolio: min/max/mean/std across all loan features +
loan count → 257-feature vector. Since the Run-5 shootout these tree arms
ARE the model candidates (the DeepSets pipeline lost on every ranking
slice); see project_config.MODEL_ARMS for the lineup.

aggregate_features() converts a list of portfolio instances into (X, y)
arrays ONCE per split (train/val/test) — callers compute it a single time
and pass the arrays to every arm, rather than each arm re-deriving it from
raw instances. (The naive per-instance Python loop this replaced cost
~6.5 minutes per call at 6M instances; multiplied across several arms x
several splits that was the dominant cost of a run — see AGENT_HANDOFF.)

All arms share the same array-based interface:
    train(X_train, y_train, cat_train, X_val=None, y_val=None, cat_val=None)
    predict_proba(X, current_cat=None) -> probs
Full-distribution arms return (N, NUM_CLASSES) probabilities; the binary
diagnostic arm returns (N, 2) [1-p, p] so calibration code can be reused.
`current_cat` is only required by PerCatXGBArm; other arms ignore it.
"""

import logging
from typing import Optional

import numpy as np
import xgboost as xgb

import project_config as config

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


def aggregate_features(instances: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised min/max/mean/std/n_loans aggregation over portfolio loans.

    Flattens every instance's (n_loans, n_features) array into one big
    (total_loans, F) matrix with per-instance offsets, then computes all
    four statistics via np.reduceat — a handful of array-level passes
    instead of a Python loop calling 4-5 numpy functions per instance
    (which is what made this slow: numpy's per-call overhead dominates
    when each call only touches 1-2 rows).

    Returns (X, y) — X is (N, 4*F + 1) float32, y is (N,) int32. Uses
    inst["n_loans"] (the UNTRUNCATED loan count) for the count column,
    matching the pre-truncation semantics of the original implementation
    even though the flattened matrix only holds the truncated rows.
    """
    n = len(instances)
    if n == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int32)

    sizes = np.fromiter((i["features"].shape[0] for i in instances), dtype=np.int64, count=n)
    flat = np.concatenate([i["features"] for i in instances]).astype(np.float64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]

    mn = np.minimum.reduceat(flat, offsets, axis=0)
    mx = np.maximum.reduceat(flat, offsets, axis=0)
    sm = np.add.reduceat(flat, offsets, axis=0)
    sq = np.add.reduceat(flat * flat, offsets, axis=0)

    sizes_f = sizes.astype(np.float64)[:, None]
    mean = sm / sizes_f
    var  = np.maximum(sq / sizes_f - mean ** 2, 0.0)   # guard fp round-off
    std  = np.sqrt(var)

    n_loans = np.fromiter((i["n_loans"] for i in instances), dtype=np.float64, count=n)[:, None]
    X = np.concatenate([mn, mx, mean, std, n_loans], axis=1).astype(np.float32)
    y = np.fromiter((i["label"] for i in instances), dtype=np.int32, count=n)
    return X, y


class AggregatedXGBoostBaseline:
    """The "multiclass" arm: one multi:softprob model over all 4 classes."""

    name = "multiclass"
    full_distribution = True

    def __init__(self, random_state: int = 42, params: Optional[dict] = None):
        self.random_state = random_state
        # params overrides XGB_DEFAULTS (e.g. Optuna-tuned); None = defaults
        self.params = {**XGB_DEFAULTS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None

    @staticmethod
    def _sample_weights(y: np.ndarray) -> np.ndarray:
        """
        Inverse class-frequency weights, optionally scaled by the cost-matrix
        row sum per true class (BASELINE_COST_WEIGHTS) so training attention
        follows the business cost of misclassifying that class.
        """
        classes, counts = np.unique(y, return_counts=True)
        w = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        if getattr(config, "BASELINE_COST_WEIGHTS", False):
            row_cost = np.asarray(config.COST_MATRIX).sum(axis=1)
            for c in classes:
                w[c] *= row_cost[c]
        return np.array([w[yi] for yi in y])

    def train(self, X_train, y_train, cat_train,
              X_val=None, y_val=None, cat_val=None) -> None:
        sw_train = self._sample_weights(y_train)
        eval_set = [(X_val, y_val)] if X_val is not None and len(X_val) else None

        logger.info(f"Training XGBoost on {X_train.shape[1]} aggregated features…")
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=config.NUM_CLASSES,
            random_state=self.random_state,
            **self.params,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            sample_weight=sw_train,
            verbose=False,
        )
        logger.info("Multiclass arm training complete.")

    def predict_proba(self, X, current_cat=None) -> np.ndarray:
        return self.model.predict_proba(X)


class BinarySevereBaseline(AggregatedXGBoostBaseline):
    """
    The "binary" arm: models the severe event y = (label == NUM_CLASSES-1)
    directly. RANKING-CEILING DIAGNOSTIC ONLY — it cannot produce the
    per-class probabilities the business needs, so it is never deployed.
    predict_proba returns (N, 2) [1-p, p] so StratifiedCalibrator applies.
    """

    name = "binary"
    full_distribution = False

    def train(self, X_train, y_train, cat_train,
              X_val=None, y_val=None, cat_val=None) -> None:
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
            **self.params,
        )
        self.model.fit(X_train, y_bin, verbose=False)
        logger.info("Binary arm training complete.")

    def predict_proba(self, X, current_cat=None) -> np.ndarray:
        """Returns (N, 2) [P(not severe), P(severe)]."""
        p = self.model.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - p, p])


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

    def train(self, X_train, y_train, cat_train,
              X_val=None, y_val=None, cat_val=None) -> None:
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
                **self.params,
            )
            mdl.fit(X_train, y_gt, verbose=False)
            self.models.append(mdl)
        logger.info("Ordinal arm training complete.")

    def predict_proba(self, X, current_cat=None) -> np.ndarray:
        cum = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        cum = np.minimum.accumulate(cum, axis=1)      # enforce P(y>0)>=P(y>1)>=P(y>2)

        probs = np.empty((len(X), config.NUM_CLASSES))
        probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, config.NUM_CLASSES - 1):
            probs[:, k] = cum[:, k - 1] - cum[:, k]
        probs[:, -1] = cum[:, -1]

        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum(axis=1, keepdims=True)


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

    def train(self, X_train, y_train, cat_train,
              X_val=None, y_val=None, cat_val=None) -> None:
        self.models = {}
        for c in range(config.NUM_CLASSES - 1):       # cat 3 is trivially P(3)=1
            mask = cat_train == c
            if not mask.any():
                logger.warning(f"Per-cat arm: no training data for stratum {c}.")
                continue
            # Labels shifted into {0 .. 3-c}; monotonicity guard for any
            # upstream violation (shouldn't happen by ETL definition)
            y_shift = np.maximum(y_train[mask] - c, 0)
            observed = np.unique(y_shift)
            if len(observed) < 2:
                logger.warning(f"Per-cat arm: stratum {c} has a single outcome "
                               f"({observed}) — using the point-mass fallback.")
                continue

            logger.info(f"Per-cat arm: training stratum {c} "
                        f"({int(mask.sum()):,} instances, {len(observed)} observed classes)…")
            if len(observed) == 2:
                mdl = xgb.XGBClassifier(
                    objective="binary:logistic",
                    random_state=self.random_state, **self.params,
                )
            else:
                mdl = xgb.XGBClassifier(
                    objective="multi:softprob",
                    random_state=self.random_state, **self.params,
                )
            mdl.fit(X_train[mask], y_shift, verbose=False)
            self.models[c] = mdl
        logger.info("Per-cat arm training complete.")

    def predict_proba(self, X, current_cat=None) -> np.ndarray:
        if current_cat is None:
            raise ValueError("PerCatXGBArm.predict_proba requires current_cat")
        current_cat = np.asarray(current_cat)
        probs = np.zeros((len(X), config.NUM_CLASSES))

        for c, mdl in self.models.items():
            mask = current_cat == c
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
        unhandled = ~np.isin(current_cat, list(self.models.keys()))
        if unhandled.any():
            probs[unhandled, :] = 0.0
            probs[unhandled, np.minimum(current_cat[unhandled], config.NUM_CLASSES - 1)] = 1.0

        probs = np.clip(probs, 0.0, None)
        return probs / probs.sum(axis=1, keepdims=True)


ARM_BUILDERS = {
    "multiclass": AggregatedXGBoostBaseline,
    "binary":     BinarySevereBaseline,
    "ordinal":    OrdinalXGBArm,
    "per_cat":    PerCatXGBArm,
}


def tune_arm_params(
    arm_name: str,
    X_train, y_train, cat_train,
    X_val, y_val, cat_val,
    n_trials: int,
    random_state: int = 42,
) -> dict:
    """
    Optuna search over XGBoost hyperparameters for `arm_name`, maximising
    PR-AUC of P(severe) on the validation set — the actual deliverable
    objective (NOT macro F1; Run 5 tuned the wrong number and wasted 5.7h).

    Returns the best-params dict to pass into the arm's constructor. Only
    meaningful for full-distribution arms; the binary arm is a diagnostic
    and is not tuned. optuna is imported lazily (server-only dependency).
    """
    import optuna
    from sklearn.metrics import average_precision_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    builder = ARM_BUILDERS[arm_name]
    severe = config.NUM_CLASSES - 1
    y_val_bin = (y_val == severe).astype(np.int32)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_jobs":           -1,
        }
        arm = builder(random_state=random_state, params=params)
        arm.train(X_train, y_train, cat_train)
        p_sev = arm.predict_proba(X_val, cat_val)[:, -1]
        return average_precision_score(y_val_bin, p_sev)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Optuna[{arm_name}]: best val PR-AUC(severe) = {study.best_value:.4f}")
    logger.info(f"Optuna[{arm_name}]: best params = {study.best_params}")
    return {**study.best_params, "n_jobs": -1}


if __name__ == "__main__":
    # Self-check: aggregate_features must match the old naive per-instance
    # loop exactly (within float tolerance) — this is a pure performance
    # rewrite, not a semantics change.
    rng = np.random.default_rng(0)
    insts = [
        {"features": rng.normal(size=(k, 5)).astype(np.float32),
         "label": int(rng.integers(4)), "n_loans": k + 3}   # n_loans != truncated rows
        for k in [1, 2, 1, 2, 1]
    ]

    def _naive(instances):
        X = []
        for inst in instances:
            f = inst["features"]
            X.append(np.concatenate([f.min(0), f.max(0), f.mean(0), f.std(0), [inst["n_loans"]]]))
        return np.vstack(X).astype(np.float32)

    X_fast, y_fast = aggregate_features(insts)
    X_ref = _naive(insts)
    assert np.allclose(X_fast, X_ref, atol=1e-4), (X_fast - X_ref)
    assert y_fast.tolist() == [i["label"] for i in insts]
    assert X_fast.shape == (5, 4 * 5 + 1)

    X_empty, y_empty = aggregate_features([])
    assert len(X_empty) == 0 and len(y_empty) == 0

    print("aggregated_xgboost.py self-check OK")
