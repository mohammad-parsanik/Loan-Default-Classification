"""
Entry point for the Loan Default Classification pipeline.

Usage:
    python run.py explore                          # profile data (one-shot)
    python run.py train                            # full training run
    python run.py train --final                    # deployment fit: all mature snapshots, no test
    python run.py train --resume <run_dir>         # resume a crashed run
    python run.py predict --artifact_dir <dir> \\
                          [--snapshot_date <int> [<int> ...]] \\
                          [--output <path.csv>]
                          # snapshot_date defaults to project_config.PRED_SNAPSHOT_DATES,
                          # or every currently-immature snapshot if that's unset too.
                          # output defaults to <artifact_dir>/predictions/predictions_<tag>.csv

Stage checkpointing:
    Each major stage writes a sentinel file to <run_dir>/stages/<stage>.done
    so that on --resume, completed stages are skipped entirely.

Walk-Forward Validation:
    When project_config.WALK_FORWARD_ENABLED = True, the pipeline generates
    all valid (train, val, test) fold combinations from the usable snapshots
    and trains a fresh model for each fold.  Per-fold results are aggregated
    and saved to <run_dir>/walk_forward_summary.json.
    Set WALK_FORWARD_ENABLED = False to run a single static temporal split.
"""

import argparse
import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

import project_config as config
from src.data.data_explorer import explore_data
from src.data.data_loader import DataLoader
from src.data.dataset import create_dataloaders
from src.data.preprocessing import create_preprocessing_pipeline
from src.data.temporal_split import (
    split_by_time,
    split_for_final_fit,
    generate_walk_forward_folds,
    build_fold_instances,
)
from src.baselines.aggregated_xgboost import ARM_BUILDERS, aggregate_features
from src.evaluation.fold_aggregator import (
    aggregate_fold_metrics,
    log_fold_summary,
    save_fold_summary,
)
from src.evaluation.calibration import StratifiedCalibrator
from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    full_evaluation,
)
from src.evaluation.ranking import capture_curve, ranking_metrics
from src.evaluation.visualization import (
    plot_capture_curves,
    plot_confusion_matrix,
    plot_embeddings_umap,
    plot_roc_curves,
    plot_training_curves,
)
from src.inference.predictor import Predictor
from src.model.deep_sets import DeepSets
from src.model.losses import CostSensitiveFocalLoss
from src.model.trainer import TransformerTrainer

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Stage checkpointing ───────────────────────────────────────────────────────

class StageCheckpointer:
    """
    Tracks which pipeline stages have completed.
    Each completed stage writes a <stage>.done file under <run_dir>/stages/.
    On resume, stages with a .done file are skipped.
    """

    def __init__(self, run_dir: Path):
        self.stage_dir = run_dir / "stages"
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    def is_done(self, stage: str) -> bool:
        return (self.stage_dir / f"{stage}.done").exists()

    def mark_done(self, stage: str, meta: dict = None):
        p = self.stage_dir / f"{stage}.done"
        p.write_text(json.dumps({"stage": stage, "ts": _now_str(), **(meta or {})}))
        logger.info(f"[checkpoint] Stage '{stage}' complete.")

    def load(self, stage: str):
        """Load a pickled artefact saved alongside the .done sentinel."""
        p = self.stage_dir / f"{stage}.pkl"
        if p.exists():
            return joblib.load(p)
        return None

    def save(self, stage: str, obj):
        """Pickle an artefact for later resume."""
        joblib.dump(obj, self.stage_dir / f"{stage}.pkl")


# ── Timing context manager ────────────────────────────────────────────────────

@contextmanager
def timed(label: str, timing_log: dict):
    logger.info(f"▶  {label}…")
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    timing_log[label] = round(elapsed, 2)
    logger.info(f"✓  {label} completed in {elapsed:.1f}s")


def _now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Single-fold training pipeline ─────────────────────────────────────────────

def train_single_fold(
    fold_id: int,
    train_inst: list,
    val_inst: list,
    test_inst: list,
    features: list,
    fold_dir: Path,
    device: str,
    timing: dict,
) -> dict:
    """
    Run the full training pipeline (stages 4–9) for a single fold.

    Parameters
    ----------
    fold_id    : int    — fold identifier (1-based), used in log messages
    train_inst : list   — training portfolio instances
    val_inst   : list   — validation portfolio instances
    test_inst  : list   — test portfolio instances
    features   : list   — feature column names
    fold_dir   : Path   — directory for all fold-specific artifacts
    device     : str    — "cpu" or "cuda"
    timing     : dict   — shared timing log (mutated in place)

    Returns
    -------
    dict with keys:
      "baseline_metrics" : dict
      "final_metrics"    : dict
      "ds_history"       : dict
    """
    fold_label = f"Fold {fold_id:02d}"
    plots_dir  = fold_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt = StageCheckpointer(fold_dir)

    logger.info(f"\n{'─' * 60}")
    logger.info(f"  {fold_label}: {len(train_inst):,} train / "
                f"{len(val_inst):,} val / {len(test_inst):,} test")
    logger.info(f"{'─' * 60}")

    # ── Compute MAX_LOANS from this fold's training set ───────────────────────
    if config.MAX_LOANS_PER_CUSTOMER is None:
        loan_counts = [i["n_loans"] for i in train_inst]
        max_loans   = int(np.percentile(loan_counts, 99))
        logger.info(f"{fold_label} | MAX_LOANS (99th pct on train): {max_loans}")
    else:
        max_loans = config.MAX_LOANS_PER_CUSTOMER
        logger.info(f"{fold_label} | Using configured MAX_LOANS: {max_loans}")

    # Inference metadata (ModelLoader reads this at predict time)
    (fold_dir / "metadata.json").write_text(json.dumps({
        "feature_count": len(features),
        "max_loans_per_customer_99th": max_loans,
        "features": features,
    }))

    # Current-category strata for slice-level evaluation and stratified
    # calibration (order matches the dataloaders: shuffle=False for val/test)
    train_strata = np.array([i["current_cat"] for i in train_inst])
    test_strata  = np.array([i["current_cat"] for i in test_inst])
    val_strata   = np.array([i["current_cat"] for i in val_inst])

    # ── Stage: Preprocessing ──────────────────────────────────────────────────
    # Not checkpointed as a full round-trip — the transform itself (~3-4 min
    # at 6M+ instances) is cheap next to what pickling three post-transform
    # instance lists used to cost (~10-20 min just to save, as long again
    # to load back). The fitted scaler IS still persisted — it's small and
    # Predictor needs it.
    with timed(f"{fold_label}: Preprocessing", timing):
        preprocessor = create_preprocessing_pipeline(features, config.BINARY_FEATURES)

        # Fit on training loans only — never touch val/test distributions
        X_train_raw = [i["features"] for i in train_inst]
        preprocessor.fit(X_train_raw)

        for split_name, split in [("Train", train_inst), ("Val", val_inst), ("Test", test_inst)]:
            logger.info(f"  Transforming {split_name} ({len(split):,} instances)…")
            if not split:
                continue
            X_raw    = [i["features"] for i in split]
            X_scaled = preprocessor.transform(X_raw)
            for inst, x in zip(split, X_scaled):
                inst["features"] = x

    joblib.dump(preprocessor, fold_dir / "scaler.pkl")

    # ── Aggregate once, shared by every arm ───────────────────────────────────
    # aggregate_features is the expensive step (a full pass over every
    # loan); computing it here ONCE per split and passing arrays to each
    # arm avoids each of the 4 arms (5-6 counting the deployed-arm re-eval)
    # redundantly re-deriving the identical matrix from train_inst/val_inst/
    # test_inst.
    with timed(f"{fold_label}: Aggregate features", timing):
        X_train, y_train = aggregate_features(train_inst)
        X_val,   y_val    = aggregate_features(val_inst)  if val_inst  else (None, None)
        X_test,  y_test   = aggregate_features(test_inst) if test_inst else (None, None)

    # ── Stage: Model arms (XGBoost on aggregated features) ───────────────────
    # Each arm trains, calibrates on the customer-disjoint val, and is
    # evaluated through the same full_evaluation path. The binary arm is a
    # ranking-ceiling diagnostic (no class distribution → never deployed).
    arms_results: dict = {}
    arm_objects: dict = {}
    arm_calibrators: dict = {}
    arm_test_probs: dict = {}   # cached so the deployed-arm section below
                                 # doesn't re-run predict_proba redundantly

    arms_to_run = list(config.MODEL_ARMS)
    if not test_inst:
        # Final fit: nothing to compare on — train only the deployment arm.
        if config.DEPLOY_ARM == "auto":
            raise ValueError(
                "Final fit requires an explicit DEPLOY_ARM in project_config "
                "(\"auto\" needs a test set to pick a winner)."
            )
        arms_to_run = [config.DEPLOY_ARM]

    for arm_name in arms_to_run:
        stage = f"arm_{arm_name}"
        if ckpt.is_done(stage):
            logger.info(f"[skip] {fold_label} arm '{arm_name}' already complete.")
            arm, arm_cal, arm_metrics, test_probs = ckpt.load(stage)
        else:
            with timed(f"{fold_label}: Arm '{arm_name}'", timing):
                arm = ARM_BUILDERS[arm_name]()
                arm.train(X_train, y_train, train_strata, X_val, y_val, val_strata)

                arm_cal = None
                if X_val is not None and len(X_val):
                    val_probs = arm.predict_proba(X_val, val_strata)
                    # Binary arm calibrates against the binary event
                    y_fit = (y_val if arm.full_distribution
                             else (y_val == config.NUM_CLASSES - 1).astype(np.int32))
                    arm_cal = StratifiedCalibrator(
                        min_stratum_n=config.CALIBRATION_MIN_STRATUM_N
                    ).fit(val_probs, y_fit, val_strata)

                arm_metrics = {}
                test_probs = None
                if X_test is not None and len(X_test):
                    test_probs = arm.predict_proba(X_test, test_strata)
                    if arm.full_distribution:
                        arm_metrics = full_evaluation(
                            y_test, test_probs,
                            strata=test_strata, calibrator=arm_cal,
                        )
                        sev = arm_metrics.pop("_probs_cal")[:, -1]
                        arm_metrics.pop("_cost_preds", None)
                    else:
                        probs_cal = (arm_cal.transform(test_probs, test_strata)
                                     if arm_cal is not None else test_probs)
                        sev = probs_cal[:, -1]
                        arm_metrics = {"ranking": ranking_metrics(
                            y_test, sev, strata=test_strata)}

                    # Capture-curve data (carved population) for the plot
                    keep = test_strata < config.CARVE_CURRENT_CAT_GE
                    arm_metrics["capture_curve"] = capture_curve(y_test[keep], sev[keep])

            ckpt.save(stage, (arm, arm_cal, arm_metrics, test_probs))
            ckpt.mark_done(stage, {"ranking_ap": arm_metrics.get("ranking", {}).get("pr_auc")})

        arms_results[arm_name]    = arm_metrics
        arm_objects[arm_name]     = arm
        arm_calibrators[arm_name] = arm_cal
        arm_test_probs[arm_name]  = test_probs

    # ── Legacy neural arm (DeepSets encoder + XGB meta-learner) ──────────────
    ds_history = {}
    if config.DEEPSETS_ENABLED:
        ds_metrics, ds_history = _train_deepsets_legacy(
            fold_label, train_inst, val_inst, test_inst, features,
            fold_dir, plots_dir, device, timing, ckpt, max_loans,
            val_strata, test_strata, preprocessor,
        )
        arms_results["deepsets"] = ds_metrics

    # ── Deployed-arm selection ────────────────────────────────────────────────
    full_dist_arms = [n for n in arms_to_run if ARM_BUILDERS[n].full_distribution]
    if config.DEPLOY_ARM == "auto":
        deploy_name = max(
            full_dist_arms,
            key=lambda n: arms_results[n].get("ranking", {}).get("pr_auc", float("-inf")),
        )
        logger.info(
            f"{fold_label} | Deployed arm (auto, best pooled ranking AP): "
            f"'{deploy_name}' "
            f"(AP={arms_results[deploy_name].get('ranking', {}).get('pr_auc', float('nan')):.4f})"
        )
    else:
        deploy_name = config.DEPLOY_ARM
        if not ARM_BUILDERS[deploy_name].full_distribution:
            raise ValueError(
                f"DEPLOY_ARM='{deploy_name}' has no per-class distribution "
                f"(it's a ranking-only diagnostic) and cannot be deployed."
            )
        logger.info(f"{fold_label} | Deployed arm (configured): '{deploy_name}'")

    final_metrics = arms_results.get(deploy_name, {})

    # Deployment artifacts for the Predictor (aggregated-features path)
    joblib.dump(arm_objects[deploy_name], fold_dir / "model_arm.pkl")
    if arm_calibrators[deploy_name] is not None:
        joblib.dump(arm_calibrators[deploy_name], fold_dir / "calibrator.pkl")

    # ── Deployed-arm CI + plots ───────────────────────────────────────────────
    if test_inst:
        if ckpt.is_done("deployed_eval"):
            logger.info(f"[skip] {fold_label} deployed-arm evaluation already complete.")
            final_metrics = ckpt.load("deployed_eval") or final_metrics
            arms_results[deploy_name] = final_metrics
        else:
            with timed(f"{fold_label}: Deployed-arm CI + plots", timing):
                arm_cal = arm_calibrators[deploy_name]
                # Reuse the raw probs already computed for this arm in the
                # arms loop above — no need to re-run predict_proba.
                probs_test = arm_test_probs[deploy_name]
                fe = full_evaluation(
                    y_test, probs_test, strata=test_strata, calibrator=arm_cal
                )
                y_pred     = fe.pop("_cost_preds")
                y_prob_cal = fe.pop("_probs_cal")
                final_metrics["bootstrap_ci"] = bootstrap_confidence_intervals(
                    y_test, y_pred, y_prob_cal, n_iterations=500
                )
                arms_results[deploy_name] = final_metrics

                plot_confusion_matrix(y_test, y_pred, save_path=plots_dir / "confusion_matrix.png")
                plot_roc_curves(y_test, y_prob_cal,  save_path=plots_dir / "roc_curves.png")
                plot_capture_curves(
                    {n: m["capture_curve"] for n, m in arms_results.items()
                     if "capture_curve" in m},
                    save_path=plots_dir / "capture_curves.png",
                )
            ckpt.save("deployed_eval", final_metrics)
            ckpt.mark_done("deployed_eval")

        logger.info(f"{fold_label} | Deployed arm '{deploy_name}' test metrics logged below.")

        # Full metrics (minus bulky curves) for offline analysis
        (fold_dir / "arms_metrics.json").write_text(json.dumps(_to_json_safe({
            n: {k: v for k, v in m.items() if k != "capture_curve"}
            for n, m in arms_results.items()
        }), indent=2))

    if final_metrics:
        logger.info(
            f"{fold_label} DONE | deployed arm '{deploy_name}' "
            f"ranking AP: {final_metrics.get('ranking', {}).get('pr_auc', float('nan')):.4f}  "
            f"Macro F1: {final_metrics.get('macro_f1', float('nan')):.4f}"
        )
    else:
        logger.info(f"{fold_label} FINAL FIT DONE — deployment artifacts ready "
                    f"(arm '{deploy_name}').")

    return {
        "baseline_metrics": arms_results.get("multiclass", {}),
        "final_metrics":    final_metrics,
        "arms":             arms_results,
        "deployed_arm":     deploy_name,
        "ds_history":       ds_history,
    }


def _train_deepsets_legacy(
    fold_label, train_inst, val_inst, test_inst, features,
    fold_dir, plots_dir, device, timing, ckpt, max_loans,
    val_strata, test_strata, preprocessor,
):
    """
    The pre-July-10 pipeline: DeepSets encoder → XGB meta-learner →
    calibrated evaluation → legacy deployment bundle. Lost the Run-5
    ranking shootout on every slice; kept behind config.DEEPSETS_ENABLED
    for reproducibility. Returns (final_metrics, ds_history).
    """
    from src.model.meta_learner import XGBoostMetaLearner  # lazy: needs optuna

    # ── Stage: DataLoaders ────────────────────────────────────────────────────
    with timed(f"{fold_label}: Build DataLoaders", timing):
        train_dl, val_dl, test_dl = create_dataloaders(
            train_inst, val_inst, test_inst,
            max_loans=max_loans,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

    # ── Stage: DeepSets training ──────────────────────────────────────────────
    ds_history = {}
    if ckpt.is_done("deepsets"):
        logger.info(f"[skip] {fold_label} DeepSets already complete — loading checkpoint.")
        model_state, ds_history = ckpt.load("deepsets")
        model = DeepSets(
            n_features=len(features),
            hidden_dim=config.DEEPSETS_HIDDEN_DIM,
            embedding_dim=config.DEEPSETS_EMBED_DIM,
            dropout=config.DROPOUT,
            num_classes=config.NUM_CLASSES,
        )
        model.load_state_dict(model_state)
    else:
        with timed(f"{fold_label}: DeepSets training", timing):
            model = DeepSets(
                n_features=len(features),
                hidden_dim=config.DEEPSETS_HIDDEN_DIM,
                embedding_dim=config.DEEPSETS_EMBED_DIM,
                dropout=config.DROPOUT,
                num_classes=config.NUM_CLASSES,
            )

            import sys as _sys
            _can_compile = (
                not _sys.platform.startswith("win")
                or torch.cuda.is_available()
            )
            if _can_compile:
                try:
                    model = torch.compile(model)
                    logger.info("torch.compile enabled.")
                except Exception as e:
                    logger.warning(f"torch.compile unavailable: {e}")
            else:
                logger.info(
                    "torch.compile skipped (Windows CPU-only: inductor requires "
                    "MSVC cl.exe which was not found). Running in eager mode."
                )

            criterion = CostSensitiveFocalLoss(num_classes=config.NUM_CLASSES)
            trainer   = TransformerTrainer(
                model, criterion, config,
                device=device,
                checkpoint_dir=fold_dir / "checkpoints",
            )
            best_val_f1, ds_history = trainer.train(train_dl, val_dl)
            logger.info(f"{fold_label} | DeepSets best val Macro F1: {best_val_f1:.4f}")

        raw_state = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        torch.save(raw_state, fold_dir / "deep_sets.pt")
        ckpt.save("deepsets", (raw_state, ds_history))
        ckpt.mark_done("deepsets", {"best_val_f1": best_val_f1})

    if ds_history:
        plot_training_curves(ds_history, save_path=plots_dir / "training_curves.png")

    # ── Stage: XGBoost meta-learner ───────────────────────────────────────────
    if ckpt.is_done("xgboost"):
        logger.info(f"[skip] {fold_label} XGBoost already complete — loading model.")
        import xgboost as xgb
        xgb_model_obj = xgb.XGBClassifier()
        xgb_model_obj.load_model(fold_dir / "xgboost_model.json")
        meta_learner = XGBoostMetaLearner(model, device=device)
        meta_learner.xgb_model = xgb_model_obj
        meta_learner.best_params = ckpt.load("xgboost_params")
    else:
        with timed(f"{fold_label}: XGBoost meta-learner (Optuna + train)", timing):
            meta_learner = XGBoostMetaLearner(
                model,
                device=device,
                random_state=config.RANDOM_SEED,
                study_db=fold_dir / "optuna_study.db",
            )
            meta_learner.optimize_and_train(train_dl, val_dl, n_trials=30)

        meta_learner.xgb_model.save_model(fold_dir / "xgboost_model.json")
        ckpt.save("xgboost_params", meta_learner.best_params)
        ckpt.mark_done("xgboost", meta_learner.best_params or {})

    # ── Stage: Evaluation ─────────────────────────────────────────────────────
    if ckpt.is_done("evaluation"):
        logger.info(f"[skip] {fold_label} evaluation already complete.")
        final_metrics = ckpt.load("evaluation")
    else:
        with timed(f"{fold_label}: Final evaluation", timing):
            # Fit the probability calibrator on the (customer-disjoint) val
            # set — data the final XGBoost never trained on. Runs even
            # without a test set: the calibrator ships with the deployment
            # bundle, so a --final fit needs it too.
            calibrator = None
            if val_inst:
                _, _, y_val_m, val_probs_m = meta_learner.evaluate(val_dl)
                calibrator = StratifiedCalibrator(
                    min_stratum_n=config.CALIBRATION_MIN_STRATUM_N
                ).fit(val_probs_m, y_val_m, val_strata)
                joblib.dump(calibrator, fold_dir / "deepsets_calibrator.pkl")
                logger.info(f"{fold_label} | DeepSets stratified calibrator fitted on val and saved.")

            if test_inst:
                _, X_test_emb, y_test, y_prob = meta_learner.evaluate(test_dl)
                final_metrics = full_evaluation(
                    y_test, y_prob, strata=test_strata, calibrator=calibrator
                )
                # Deployed policy = expected-cost decisions on calibrated probs;
                # plots and CIs reflect it.
                y_pred     = final_metrics.pop("_cost_preds")
                y_prob_cal = final_metrics.pop("_probs_cal")

                ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob_cal, n_iterations=500)
                final_metrics["bootstrap_ci"] = ci
                logger.info(f"{fold_label} | Test Metrics: {final_metrics}")
            else:
                final_metrics = {}
                logger.info(f"{fold_label} | Final fit — no test set, evaluation skipped.")

        ckpt.save("evaluation", final_metrics)
        ckpt.mark_done("evaluation")

        if test_inst:
            # Export test embeddings for downstream SHAP analysis
            np.save(fold_dir / "test_embeddings.npy", X_test_emb)
            np.save(fold_dir / "test_labels.npy", y_test)

            with timed(f"{fold_label}: DeepSets plots", timing):
                plot_confusion_matrix(y_test, y_pred,    save_path=plots_dir / "deepsets_confusion_matrix.png")
                plot_roc_curves(y_test, y_prob,          save_path=plots_dir / "deepsets_roc_curves.png")
                plot_embeddings_umap(X_test_emb, y_test, save_path=plots_dir / "embedding_umap.png")

    # ── Single-file deployment bundle ─────────────────────────────────────────
    # Pure export convenience: packs the same trained artifacts already on
    # disk into one model_bundle.pkl, loadable standalone via
    # ModelLoader/Predictor without needing the rest of the fold directory.
    # Re-derived from `model`/`meta_learner`/`preprocessor` (always in scope
    # here regardless of whether this fold resumed from a checkpoint), plus
    # the calibrator re-read from disk since it's only set in-memory on a
    # fresh (non-resumed) evaluation run.
    bundle_raw_state = (
        model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    )
    cal_path = fold_dir / "deepsets_calibrator.pkl"
    bundle_calibrator = joblib.load(cal_path) if cal_path.exists() else None
    bundle = {
        "metadata": {
            "feature_count": len(features),
            "max_loans_per_customer_99th": max_loans,
            "features": features,
        },
        "scaler": preprocessor,
        "deep_sets_state_dict": bundle_raw_state,
        "deep_sets_hparams": {
            "n_features": len(features),
            "hidden_dim": config.DEEPSETS_HIDDEN_DIM,
            "embedding_dim": config.DEEPSETS_EMBED_DIM,
            "dropout": config.DROPOUT,
            "num_classes": config.NUM_CLASSES,
        },
        "xgb_model_raw": meta_learner.xgb_model.get_booster().save_raw(raw_format="json"),
        "calibrator": bundle_calibrator,
    }
    joblib.dump(bundle, fold_dir / "model_bundle.pkl")
    logger.info(f"{fold_label} | Legacy DeepSets bundle saved to {fold_dir / 'model_bundle.pkl'}")

    return final_metrics, ds_history


# ── Training pipeline ─────────────────────────────────────────────────────────

def train_pipeline(resume_dir: Path = None, final_fit: bool = False):
    logger.info("=" * 60)
    if final_fit:
        logger.info("  Loan Default Classification — FINAL DEPLOYMENT FIT")
        logger.info("  (all mature snapshots, no test hold-out)")
    else:
        logger.info("  Loan Default Classification — Training Pipeline")
    logger.info("=" * 60)

    # ── Global setup ──────────────────────────────────────────────────────────
    set_seeds()
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    torch.set_num_interop_threads(config.TORCH_NUM_INTEROP)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | threads: {config.TORCH_NUM_THREADS}")
    logger.info(
        f"Walk-forward mode: {'ENABLED' if config.WALK_FORWARD_ENABLED else 'DISABLED (single split)'}"
    )

    # ── Run directory ─────────────────────────────────────────────────────────
    if resume_dir and resume_dir.exists():
        run_dir = resume_dir
        logger.info(f"Resuming from {run_dir}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.ARTIFACT_DIR / (f"{ts}_final" if final_fit else ts)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

    ckpt     = StageCheckpointer(run_dir)
    timing   = {}
    pipeline_log = {"run_dir": str(run_dir), "started": _now_str(), "stages": {}}

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    # Deliberately NOT checkpointed via joblib: the NPZ cache
    # (train_portfolios_cache.npz) already makes this call fast (~30-60s).
    # Pickling 10M+ per-instance dicts here used to cost 15-30+ MINUTES to
    # save (and as long again to load back on --resume) — strictly worse
    # than just recomputing every time.
    with timed("Stage 1: Load & group data", timing):
        dl = DataLoader()
        instances, features = dl.load_train_portfolios(use_cache=True)
    if not instances:
        logger.error("No instances loaded. Exiting.")
        return
    logger.info(f"Loaded {len(instances):,} portfolio instances, {len(features)} features.")

    # ── Stage 2: Split (single) or generate folds (walk-forward) ─────────────
    fold_results: list = []

    if final_fit:
        _run_single_split(
            instances, features, run_dir, device, timing,
            fold_results, final_fit=True,
        )
        aggregated = {}
    elif config.WALK_FORWARD_ENABLED:
        # Walk-forward: generate all valid folds from usable snapshots
        if ckpt.is_done("folds_complete"):
            logger.info("[skip] All walk-forward folds already complete.")
            fold_results = ckpt.load("fold_results") or []
        else:
            with timed("Stage 2: Generate walk-forward folds", timing):
                folds = generate_walk_forward_folds(instances)

            if not folds:
                logger.warning(
                    "No valid walk-forward folds found — "
                    "falling back to single static split."
                )
                _run_single_split(
                    instances, features, run_dir, device, timing, fold_results
                )
            else:
                # Load any previously completed folds
                saved_results = ckpt.load("fold_results") or []
                completed_ids = {r["fold_id"] for r in saved_results}
                fold_results  = list(saved_results)

                for fold in folds:
                    if fold.fold_id in completed_ids:
                        logger.info(f"[skip] Fold {fold.fold_id:02d} already complete.")
                        continue

                    fold_dir = run_dir / f"fold_{fold.fold_id:02d}"
                    fold_dir.mkdir(parents=True, exist_ok=True)

                    train_inst, val_inst, test_inst = build_fold_instances(instances, fold)

                    result = train_single_fold(
                        fold_id    = fold.fold_id,
                        train_inst = train_inst,
                        val_inst   = val_inst,
                        test_inst  = test_inst,
                        features   = features,
                        fold_dir   = fold_dir,
                        device     = device,
                        timing     = timing,
                    )
                    result.update({
                        "fold_id":     fold.fold_id,
                        "train_snaps": sorted(fold.train_snaps),
                        "val_snap":    fold.val_snap,
                        "test_snap":   fold.test_snap,
                    })
                    fold_results.append(result)

                    # Save cumulative results so resume skips completed folds
                    ckpt.save("fold_results", fold_results)

                ckpt.mark_done("folds_complete")

        # ── Aggregate across all folds ────────────────────────────────────────
        if fold_results:
            aggregated = aggregate_fold_metrics(fold_results)
            log_fold_summary(aggregated)
            save_fold_summary(aggregated, run_dir / "walk_forward_summary.json")

            # Surface best fold
            best_fold = max(fold_results, key=lambda r: r["final_metrics"]["macro_f1"])
            logger.info(
                f"Best fold: Fold {best_fold['fold_id']:02d}  "
                f"(test={best_fold['test_snap']})  "
                f"Macro F1: {best_fold['final_metrics']['macro_f1']:.4f}"
            )
        else:
            aggregated = {}

    else:
        # ── Single static temporal split (original behaviour) ─────────────────
        _run_single_split(
            instances, features, run_dir, device, timing, fold_results
        )
        aggregated = {}

    # ── Stage 10: Save pipeline log ───────────────────────────────────────────
    pipeline_log["finished"] = _now_str()
    pipeline_log["timing_s"] = timing
    pipeline_log["stages"]   = {s: True for s in timing}
    pipeline_log["walk_forward_enabled"] = config.WALK_FORWARD_ENABLED

    with open(run_dir / "pipeline_log.json", "w") as f:
        json.dump(_to_json_safe(pipeline_log), f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete  →  {run_dir}")
    scored = [r for r in fold_results if r.get("final_metrics")]
    if scored:
        best = max(
            scored,
            key=lambda r: r["final_metrics"].get("ranking", {}).get("pr_auc", float("-inf")),
        )
        rk = best["final_metrics"].get("ranking", {})
        logger.info(
            f"  Deployed arm '{best.get('deployed_arm', '?')}' | "
            f"ranking AP: {rk.get('pr_auc', float('nan')):.4f}  "
            f"Macro F1: {best['final_metrics'].get('macro_f1', float('nan')):.4f}"
        )
        if aggregated:
            m = aggregated["model"]["macro_f1"]
            logger.info(f"  Mean Macro F1 (all folds): {m['mean']:.4f} ± {m['std']:.4f}")
    logger.info(f"  Total wall time: {sum(timing.values()):.0f}s")
    logger.info("=" * 60)


def _run_single_split(instances, features, run_dir, device, timing,
                      fold_results, final_fit: bool = False):
    """Helper: run the original single-split flow and append the result to fold_results."""
    # Not checkpointed — same reasoning as Stage 1: recompute (~seconds,
    # pure in-memory partitioning) is far cheaper than a joblib round-trip
    # of the instance lists.
    with timed("Stage 2: Temporal split", timing):
        split_fn = split_for_final_fit if final_fit else split_by_time
        train_inst, val_inst, test_inst = split_fn(instances)

    fold_dir = run_dir / "fold_01"
    fold_dir.mkdir(parents=True, exist_ok=True)

    result = train_single_fold(
        fold_id    = 1,
        train_inst = train_inst,
        val_inst   = val_inst,
        test_inst  = test_inst,
        features   = features,
        fold_dir   = fold_dir,
        device     = device,
        timing     = timing,
    )
    result.update({
        "fold_id":     1,
        "train_snaps": [],
        "val_snap":    None,
        "test_snap":   None,
    })
    fold_results.append(result)

    # Log summary for single split
    if not result["final_metrics"]:
        return   # final fit — nothing to compare
    _log_arms_comparison(result["arms"], result["deployed_arm"])


def _log_ranking_line(name: str, rk: dict):
    """One log line summarising a ranking-metrics dict."""
    if not rk or "pr_auc" not in rk:
        return
    parts = []
    for w in config.RANKING_REF_WINDOWS:
        block = rk.get(f"at_{w}")
        if block:
            parts.append(f"{w}: R={block['recall']:.3f} lift={block['lift']:.1f}x")
    logger.info(
        f"  {name:24s} | base_rate={rk['base_rate']:.4f}  "
        f"PR-AUC={rk['pr_auc']:.4f}  " + "  ".join(parts)
    )


def _log_arms_comparison(arms: dict, deploy_name: str):
    """All arms, pooled AND per-stratum (the Run-5 log only showed one arm's slices)."""
    logger.info("  ── Ranking (API queue): recall of future-severe in top K-hours ──")
    for name, m in arms.items():
        tag = f"{name}*" if name == deploy_name else name
        _log_ranking_line(tag, m.get("ranking"))
        for slice_name, sub in m.get("ranking", {}).get("by_current_cat", {}).items():
            _log_ranking_line(f"  {tag} [{slice_name}]", sub)
    logger.info("  (* = deployed arm; queue sorting uses its calibrated+masked P3)")

    logger.info("  ── Classification (full-distribution arms, cost-rule decisions) ──")
    for name, m in arms.items():
        cr = m.get("cost_rule")
        if not cr:
            continue
        logger.info(
            f"  {name:12s} | F1: {cr['macro_f1']:.4f}  QWK: {cr['qwk']:.4f}  "
            f"Cat-2 Rec: {cr['recall_class_2']:.4f}  "
            f"Cat-3 Rec: {cr['recall_class_3']:.4f}  Cost: {cr['avg_cost']:.4f}"
        )
        for slice_name, sm in m.get("by_current_cat", {}).items():
            logger.info(
                f"    {name} [{slice_name}, n={sm['n']:,}] | F1: {sm['macro_f1']:.4f}  "
                f"Cat-3 Rec: {sm.get('recall_class_3', float('nan')):.4f}  "
                f"Cost: {sm['avg_cost']:.4f}"
            )


def _to_json_safe(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── Predict pipeline ──────────────────────────────────────────────────────────

def predict_pipeline(artifact_dir: str, snapshot_date=None, output_path: str = None,
                     called_log: str = None):
    predictor = Predictor(Path(artifact_dir))
    predictor.predict(snapshot_date, output_path, called_log_path=called_log)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loan Default Classification Pipeline")
    parser.add_argument(
        "action", choices=["explore", "train", "predict"],
        help="Pipeline action to run.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a previous run directory to resume from.",
    )
    parser.add_argument(
        "--final", action="store_true",
        help="Deployment fit: train on ALL mature snapshots (no test "
             "hold-out), emit the model bundle. Use after evaluation runs "
             "have graded the recipe.",
    )
    parser.add_argument("--artifact_dir",  type=str)
    parser.add_argument(
        "--snapshot_date", type=int, nargs="*", default=None,
        help="Snapshot date(s) (YYYYMMDD) to score. Omit to use "
             "project_config.PRED_SNAPSHOT_DATES, or auto-select every "
             "currently-immature snapshot if that's unset too.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path. Omit to auto-name under <artifact_dir>/predictions/.",
    )
    parser.add_argument(
        "--called_log", type=str, default=None,
        help="CSV ledger of past API calls (NATIONAL_CODE, CALLED_AT). "
             "Customers called within API_DATA_TTL_DAYS are flagged "
             "RECENTLY_CALLED and skipped by the queue. Defaults to "
             "project_config.API_CALL_LOG.",
    )

    args = parser.parse_args()

    if args.action == "explore":
        explore_data()
    elif args.action == "train":
        resume = Path(args.resume) if args.resume else None
        train_pipeline(resume_dir=resume, final_fit=args.final)
    elif args.action == "predict":
        if not args.artifact_dir:
            logger.error("predict requires --artifact_dir")
            sys.exit(1)
        predict_pipeline(args.artifact_dir, args.snapshot_date, args.output,
                         called_log=args.called_log)
