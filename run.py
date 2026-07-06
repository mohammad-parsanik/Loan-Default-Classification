"""
Entry point for the Loan Default Classification pipeline.

Usage:
    python run.py explore                          # profile data (one-shot)
    python run.py train                            # full training run
    python run.py train --resume <run_dir>         # resume a crashed run
    python run.py predict --artifact_dir <dir> \\
                          --snapshot_date <int> \\
                          --output <path.csv>

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
    generate_walk_forward_folds,
    build_fold_instances,
)
from src.baselines.aggregated_xgboost import AggregatedXGBoostBaseline
from src.evaluation.fold_aggregator import (
    aggregate_fold_metrics,
    log_fold_summary,
    save_fold_summary,
)
from src.evaluation.metrics import bootstrap_confidence_intervals, compute_metrics
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_embeddings_umap,
    plot_roc_curves,
    plot_training_curves,
)
from src.inference.predictor import Predictor
from src.model.deep_sets import DeepSets
from src.model.losses import CostSensitiveFocalLoss
from src.model.meta_learner import XGBoostMetaLearner
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

    # ── Stage: Preprocessing ──────────────────────────────────────────────────
    stage_key = f"fold{fold_id:02d}_preprocessing"
    if ckpt.is_done("preprocessing"):
        logger.info(f"[skip] {fold_label} preprocessing already complete.")
        preprocessor, train_inst, val_inst, test_inst = ckpt.load("preprocessing")
    else:
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
        ckpt.save("preprocessing", (preprocessor, train_inst, val_inst, test_inst))
        ckpt.mark_done("preprocessing")

    # ── Stage: Baseline ───────────────────────────────────────────────────────
    if ckpt.is_done("baseline"):
        logger.info(f"[skip] {fold_label} baseline already complete.")
        baseline_metrics = ckpt.load("baseline")
    else:
        with timed(f"{fold_label}: Aggregated XGBoost baseline", timing):
            baseline = AggregatedXGBoostBaseline()
            baseline.train(train_inst, val_inst)
            baseline_metrics = baseline.evaluate(test_inst)
        ckpt.save("baseline", baseline_metrics)
        ckpt.mark_done("baseline", baseline_metrics)

    logger.info(
        f"{fold_label} | Baseline → Macro F1: {baseline_metrics['macro_f1']:.4f}, "
        f"QWK: {baseline_metrics['qwk']:.4f}"
    )

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
        )
        model.load_state_dict(model_state)
    else:
        with timed(f"{fold_label}: DeepSets training", timing):
            model = DeepSets(
                n_features=len(features),
                hidden_dim=config.DEEPSETS_HIDDEN_DIM,
                embedding_dim=config.DEEPSETS_EMBED_DIM,
                dropout=config.DROPOUT,
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

            criterion = CostSensitiveFocalLoss()
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
            final_metrics, X_test_emb, y_test, y_prob = meta_learner.evaluate(test_dl)
            y_pred = meta_learner.xgb_model.predict(X_test_emb)

            ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob, n_iterations=500)
            final_metrics["bootstrap_ci"] = ci

        logger.info(f"{fold_label} | Test Metrics: {final_metrics}")
        ckpt.save("evaluation", final_metrics)
        ckpt.mark_done("evaluation")

        # Export test embeddings for downstream SHAP analysis
        np.save(fold_dir / "test_embeddings.npy", X_test_emb)
        np.save(fold_dir / "test_labels.npy", y_test)

        with timed(f"{fold_label}: Plots", timing):
            plot_confusion_matrix(y_test, y_pred,    save_path=plots_dir / "confusion_matrix.png")
            plot_roc_curves(y_test, y_prob,          save_path=plots_dir / "roc_curves.png")
            plot_embeddings_umap(X_test_emb, y_test, save_path=plots_dir / "embedding_umap.png")

    logger.info(
        f"{fold_label} DONE | "
        f"Baseline Macro F1: {baseline_metrics['macro_f1']:.4f}  "
        f"Model Macro F1: {final_metrics['macro_f1']:.4f}  "
        f"Delta: {final_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}"
    )

    return {
        "baseline_metrics": baseline_metrics,
        "final_metrics":    final_metrics,
        "ds_history":       ds_history,
    }


# ── Training pipeline ─────────────────────────────────────────────────────────

def train_pipeline(resume_dir: Path = None):
    logger.info("=" * 60)
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
        run_dir = config.ARTIFACT_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

    ckpt     = StageCheckpointer(run_dir)
    timing   = {}
    pipeline_log = {"run_dir": str(run_dir), "started": _now_str(), "stages": {}}

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    if ckpt.is_done("load_data"):
        logger.info("[skip] Stage 'load_data' already complete — loading from cache.")
        instances, features = ckpt.load("load_data")
    else:
        with timed("Stage 1: Load & group data", timing):
            dl = DataLoader()
            instances, features = dl.load_train_portfolios(use_cache=True)
        if not instances:
            logger.error("No instances loaded. Exiting.")
            return
        ckpt.save("load_data", (instances, features))
        ckpt.mark_done("load_data", {"n_instances": len(instances), "n_features": len(features)})

    logger.info(f"Loaded {len(instances):,} portfolio instances, {len(features)} features.")

    # ── Stage 2: Split (single) or generate folds (walk-forward) ─────────────
    fold_results: list = []

    if config.WALK_FORWARD_ENABLED:
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
                    instances, features, run_dir, device, timing, ckpt, pipeline_log, fold_results
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
            instances, features, run_dir, device, timing, ckpt, pipeline_log, fold_results
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
    if fold_results:
        best = max(fold_results, key=lambda r: r["final_metrics"]["macro_f1"])
        logger.info(f"  Best Model Macro F1: {best['final_metrics']['macro_f1']:.4f}")
        if aggregated:
            m = aggregated["model"]["macro_f1"]
            logger.info(f"  Mean Macro F1 (all folds): {m['mean']:.4f} ± {m['std']:.4f}")
    logger.info(f"  Total wall time: {sum(timing.values()):.0f}s")
    logger.info("=" * 60)


def _run_single_split(instances, features, run_dir, device, timing, ckpt, pipeline_log, fold_results):
    """Helper: run the original single-split flow and append the result to fold_results."""
    if ckpt.is_done("split"):
        logger.info("[skip] Stage 'split' already complete.")
        train_inst, val_inst, test_inst = ckpt.load("split")
    else:
        with timed("Stage 2: Temporal split", timing):
            train_inst, val_inst, test_inst = split_by_time(instances)
        ckpt.save("split", (train_inst, val_inst, test_inst))
        ckpt.mark_done("split", {
            "n_train": len(train_inst),
            "n_val":   len(val_inst),
            "n_test":  len(test_inst),
        })

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
    bm = result["baseline_metrics"]
    fm = result["final_metrics"]
    logger.info(f"  Baseline  Macro F1: {bm['macro_f1']:.4f}")
    logger.info(f"  DeepSets+XGB Macro F1: {fm['macro_f1']:.4f}")
    logger.info(f"  Cat-2 Recall: {fm.get('recall_class_2', '?'):.4f}")


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

def predict_pipeline(artifact_dir: str, snapshot_date: int, output_path: str):
    predictor = Predictor(Path(artifact_dir))
    predictor.predict(int(snapshot_date), output_path)


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
    parser.add_argument("--artifact_dir",  type=str)
    parser.add_argument("--snapshot_date", type=int)
    parser.add_argument("--output",        type=str)

    args = parser.parse_args()

    if args.action == "explore":
        explore_data()
    elif args.action == "train":
        resume = Path(args.resume) if args.resume else None
        train_pipeline(resume_dir=resume)
    elif args.action == "predict":
        if not (args.artifact_dir and args.snapshot_date and args.output):
            logger.error("predict requires --artifact_dir, --snapshot_date, and --output")
            sys.exit(1)
        predict_pipeline(args.artifact_dir, args.snapshot_date, args.output)
