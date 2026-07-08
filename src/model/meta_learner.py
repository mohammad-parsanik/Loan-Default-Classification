"""
XGBoost meta-learner trained on DeepSets customer embeddings.

Improvements vs. original:
  - Optuna uses SQLite storage → trials survive crashes; resume automatically
  - n_jobs reduced to avoid CPU contention (XGBoost n_jobs × Optuna n_jobs)
  - Final retrain uses the best n_estimators from Optuna (no early-stopping mismatch)
  - evaluate() returns embeddings alongside metrics so run.py can reuse them
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import torch
import xgboost as xgb

import project_config as config
from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

# Suppress Optuna's per-trial INFO spam — we log best value ourselves
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostMetaLearner:
    """
    1. Freezes the DeepSets model.
    2. Extracts (N, embedding_dim) embeddings from train / val / test loaders.
    3. Runs Optuna HPO (stored in SQLite → crash-safe).
    4. Retrains on train+val using best params and best n_estimators.
    5. Evaluates on test.
    """

    def __init__(
        self,
        model,                          # frozen DeepSets
        device: str = "cpu",
        random_state: int = 42,
        study_db: Optional[Path] = None,
    ):
        self.model        = model
        self.model.eval()
        self.device       = device
        self.random_state = random_state
        self.study_db     = study_db
        self.best_params: Optional[dict] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _extract_embeddings(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        """Pass data through frozen model → (N, embed_dim) numpy array."""
        all_emb, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                features     = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                labels       = batch["label"]

                emb = self.model.extract_embeddings(features, padding_mask)
                all_emb.append(emb.cpu().numpy())
                all_labels.append(labels.numpy())

        return np.vstack(all_emb), np.concatenate(all_labels)

    # ── Class weights ─────────────────────────────────────────────────────────

    @staticmethod
    def _sample_weights(y: np.ndarray) -> np.ndarray:
        """Inverse class-frequency weights for imbalanced training."""
        classes, counts = np.unique(y, return_counts=True)
        w = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([w[yi] for yi in y])

    # ── Optuna objective ──────────────────────────────────────────────────────

    def _build_objective(self, X_train, y_train, sw_train, X_val, y_val):
        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective":        "multi:softprob",
                "num_class":        config.NUM_CLASSES,
                "eval_metric":      "mlogloss",
                "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
                "max_depth":        trial.suggest_int("max_depth", 3, 7),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "gamma":            trial.suggest_float("gamma", 0, 5.0),
                "random_state":     self.random_state,
                # Keep XGBoost single-threaded per trial; Optuna runs n_jobs=2 trials
                # in parallel → total = 2 × n_jobs_xgb cores.  With 20 cores:
                # 2 Optuna workers × 8 XGBoost threads = 16 cores, leaving 4 for OS.
                "n_jobs":           8,
            }

            early_stop = xgb.callback.EarlyStopping(
                rounds=20, metric_name="mlogloss", data_name="validation_0"
            )
            mdl = xgb.XGBClassifier(**params, callbacks=[early_stop])
            mdl.fit(
                X_train, y_train,
                sample_weight=sw_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Store the actual best iteration for later use
            trial.set_user_attr("best_iteration", mdl.best_iteration)

            preds = mdl.predict(X_val)
            return compute_metrics(y_val, preds)["macro_f1"]

        return objective

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize_and_train(
        self,
        train_dl,
        val_dl,
        n_trials: int = 30,
    ) -> None:
        logger.info("Extracting embeddings for XGBoost training…")
        X_train, y_train = self._extract_embeddings(train_dl)
        sw_train = self._sample_weights(y_train)
        
        use_val = val_dl is not None and len(val_dl) > 0

        if use_val:
            X_val, y_val = self._extract_embeddings(val_dl)

            # ── Optuna study (SQLite → crash-safe) ────────────────────────────────
            storage = None
            if self.study_db:
                storage = f"sqlite:///{self.study_db}"
                logger.info(f"Optuna storage: {storage}")

            study = optuna.create_study(
                study_name="xgb_meta",
                direction="maximize",
                storage=storage,
                load_if_exists=True,   # resume if study already exists
            )

            # How many trials remain?
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining = n_trials - completed
            if remaining <= 0:
                logger.info(f"Study already has {completed} completed trials — skipping Optuna.")
            else:
                logger.info(
                    f"Running {remaining} Optuna trials "
                    f"({completed} already completed)…"
                )
                objective = self._build_objective(X_train, y_train, sw_train, X_val, y_val)
                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=2,          # 2 parallel trials (see n_jobs comment in objective)
                    show_progress_bar=True,
                )

            self.best_params = study.best_params
            best_trial       = study.best_trial
            best_n_est       = best_trial.user_attrs.get("best_iteration", self.best_params["n_estimators"])

            logger.info(f"Best Optuna Macro F1: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")
            logger.info(f"Best n_estimators (from early stopping): {best_n_est}")

            # ── Final training set ─────────────────────────────────────────────────
            # Customer-disjoint val must stay OUT of the final model: it is
            # reused afterwards to fit the probability calibrator, which is
            # only honest on data the final model never trained on.
            if getattr(config, "VAL_SPLIT_MODE", "temporal") == "customer":
                logger.info("Retraining on Train only (val reserved for calibration)…")
                X_full, y_full, sw_full = X_train, y_train, sw_train
            else:
                logger.info("Retraining on Train+Val with best parameters…")
                X_full = np.vstack([X_train, X_val])
                y_full = np.concatenate([y_train, y_val])
                sw_full = self._sample_weights(y_full)

            final_params = {
                "objective":   "multi:softprob",
                "num_class":   config.NUM_CLASSES,
                "random_state": self.random_state,
                "n_jobs":      -1,          # use all cores for final model
                "n_estimators": best_n_est, # use the early-stopping-determined count
            }
            final_params.update({k: v for k, v in self.best_params.items() if k != "n_estimators"})
        else:
            logger.info("Validation optimization disabled. Training XGBoost with fixed default parameters on Train set.")
            X_full = X_train
            y_full = y_train
            sw_full = sw_train
            
            final_params = {
                "objective":   "multi:softprob",
                "num_class":   config.NUM_CLASSES,
                "random_state": self.random_state,
                "n_jobs":      -1,
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        self.xgb_model = xgb.XGBClassifier(**final_params)
        self.xgb_model.fit(X_full, y_full, sample_weight=sw_full, verbose=False)

        logger.info("XGBoost meta-learner training complete.")

    def evaluate(self, dataloader) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate on the given dataloader.

        Returns:
            (metrics_dict, X_embeddings, y_true, y_prob)
            — embeddings and probs are returned so run.py can use them directly
              for plots without re-extracting.
        """
        X_emb, y_true = self._extract_embeddings(dataloader)
        probs = self.xgb_model.predict_proba(X_emb)
        preds = self.xgb_model.predict(X_emb)
        return compute_metrics(y_true, preds, probs), X_emb, y_true, probs
