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
from src.data.temporal_split import split_by_time
from src.baselines.aggregated_xgboost import AggregatedXGBoostBaseline
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

    # ── Run directory ─────────────────────────────────────────────────────────
    if resume_dir and resume_dir.exists():
        run_dir = resume_dir
        logger.info(f"Resuming from {run_dir}")
    else:
        # Fix: %Y%m%d (lowercase m = month)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = config.ARTIFACT_DIR / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Run directory: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

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

    # ── Stage 2: Temporal split ───────────────────────────────────────────────
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

    # ── Stage 3: Compute MAX_LOANS ────────────────────────────────────────────
    if config.MAX_LOANS_PER_CUSTOMER is None:
        loan_counts = [i["n_loans"] for i in train_inst]
        max_loans   = int(np.percentile(loan_counts, 99))
        logger.info(f"Computed MAX_LOANS (99th percentile on train): {max_loans}")
    else:
        max_loans = config.MAX_LOANS_PER_CUSTOMER
        logger.info(f"Using configured MAX_LOANS: {max_loans}")

    # ── Stage 4: Preprocessing ────────────────────────────────────────────────
    if ckpt.is_done("preprocessing"):
        logger.info("[skip] Stage 'preprocessing' already complete.")
        preprocessor, train_inst, val_inst, test_inst = ckpt.load("preprocessing")
    else:
        with timed("Stage 4: Preprocessing (fit + transform)", timing):
            preprocessor = create_preprocessing_pipeline(features, config.BINARY_FEATURES)

            # Fit on training loans only
            X_train_raw = [i["features"] for i in train_inst]
            preprocessor.fit(X_train_raw)

            # Transform all splits in-place
            for split_name, split in [("Train", train_inst), ("Val", val_inst), ("Test", test_inst)]:
                logger.info(f"  Transforming {split_name} ({len(split):,} instances)…")
                X_raw    = [i["features"] for i in split]
                X_scaled = preprocessor.transform(X_raw)
                for inst, x in zip(split, X_scaled):
                    inst["features"] = x

        joblib.dump(preprocessor, run_dir / "scaler.pkl")
        ckpt.save("preprocessing", (preprocessor, train_inst, val_inst, test_inst))
        ckpt.mark_done("preprocessing")

    # ── Stage 5: Baseline ─────────────────────────────────────────────────────
    if ckpt.is_done("baseline"):
        logger.info("[skip] Stage 'baseline' already complete.")
        baseline_metrics = ckpt.load("baseline")
    else:
        with timed("Stage 5: Aggregated XGBoost baseline", timing):
            baseline = AggregatedXGBoostBaseline()
            baseline.train(train_inst, val_inst)
            baseline_metrics = baseline.evaluate(test_inst)
        ckpt.save("baseline", baseline_metrics)
        ckpt.mark_done("baseline", baseline_metrics)

    logger.info(f"Baseline → Macro F1: {baseline_metrics['macro_f1']:.4f}, QWK: {baseline_metrics['qwk']:.4f}")

    # ── Stage 6: DataLoaders ──────────────────────────────────────────────────
    with timed("Stage 6: Build DataLoaders", timing):
        train_dl, val_dl, test_dl = create_dataloaders(
            train_inst, val_inst, test_inst,
            max_loans=max_loans,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

    # ── Stage 7: DeepSets training ────────────────────────────────────────────
    if ckpt.is_done("deepsets"):
        logger.info("[skip] Stage 'deepsets' already complete — loading checkpoint.")
        model_state, ds_history = ckpt.load("deepsets")
        model = DeepSets(
            n_features=len(features),
            hidden_dim=config.DEEPSETS_HIDDEN_DIM,
            embedding_dim=config.DEEPSETS_EMBED_DIM,
            dropout=config.DROPOUT,
        )
        model.load_state_dict(model_state)
    else:
        with timed("Stage 7: DeepSets training", timing):
            model = DeepSets(
                n_features=len(features),
                hidden_dim=config.DEEPSETS_HIDDEN_DIM,
                embedding_dim=config.DEEPSETS_EMBED_DIM,
                dropout=config.DROPOUT,
            )

            # torch.compile gives a speedup on CPU via the inductor backend,
            # BUT on Windows it requires MSVC (cl.exe) which may not be present.
            # The failure is *lazy* — it crashes on the first forward pass, not
            # at compile() time, so we must gate on the platform proactively.
            import sys as _sys
            _can_compile = (
                not _sys.platform.startswith("win")   # skip on Windows without MSVC
                or torch.cuda.is_available()           # CUDA build ships its own compiler
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
                checkpoint_dir=run_dir / "checkpoints",
            )
            best_val_f1, ds_history = trainer.train(train_dl, val_dl)
            logger.info(f"DeepSets best val Macro F1: {best_val_f1:.4f}")

        # Extract underlying state dict (handles compiled model)
        raw_state = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        torch.save(raw_state, run_dir / "deep_sets.pt")
        ckpt.save("deepsets", (raw_state, ds_history))
        ckpt.mark_done("deepsets", {"best_val_f1": best_val_f1})

    # Save training curves
    if ds_history:
        plot_training_curves(ds_history, save_path=plots_dir / "training_curves.png")

    # ── Stage 8: XGBoost meta-learner ─────────────────────────────────────────
    if ckpt.is_done("xgboost"):
        logger.info("[skip] Stage 'xgboost' already complete — loading model.")
        import xgboost as xgb
        xgb_model_obj = xgb.XGBClassifier()
        xgb_model_obj.load_model(run_dir / "xgboost_model.json")
        meta_learner = XGBoostMetaLearner(model, device=device)
        meta_learner.xgb_model = xgb_model_obj
        meta_learner.best_params = ckpt.load("xgboost_params")
    else:
        with timed("Stage 8: XGBoost meta-learner (Optuna + train)", timing):
            meta_learner = XGBoostMetaLearner(
                model,
                device=device,
                random_state=config.RANDOM_SEED,
                study_db=run_dir / "optuna_study.db",
            )
            meta_learner.optimize_and_train(train_dl, val_dl, n_trials=30)

        meta_learner.xgb_model.save_model(run_dir / "xgboost_model.json")
        ckpt.save("xgboost_params", meta_learner.best_params)
        ckpt.mark_done("xgboost", meta_learner.best_params or {})

    # ── Stage 9: Evaluation ───────────────────────────────────────────────────
    if ckpt.is_done("evaluation"):
        logger.info("[skip] Stage 'evaluation' already complete.")
        final_metrics = ckpt.load("evaluation")
    else:
        with timed("Stage 9: Final evaluation", timing):
            final_metrics, X_test_emb, y_test, y_prob = meta_learner.evaluate(test_dl)
            y_pred = meta_learner.xgb_model.predict(X_test_emb)

            # Bootstrap 95% CI
            ci = bootstrap_confidence_intervals(y_test, y_pred, y_prob, n_iterations=500)
            final_metrics["bootstrap_ci"] = ci

        logger.info(f"Test Metrics: {final_metrics}")
        ckpt.save("evaluation", final_metrics)
        ckpt.mark_done("evaluation")

        # Plots
        with timed("Plots", timing):
            plot_confusion_matrix(y_test, y_pred,             save_path=plots_dir / "confusion_matrix.png")
            plot_roc_curves(y_test, y_prob,                   save_path=plots_dir / "roc_curves.png")
            plot_embeddings_umap(X_test_emb, y_test,          save_path=plots_dir / "embedding_umap.png")

    # ── Stage 10: Save metadata & pipeline log ────────────────────────────────
    pipeline_log["finished"]  = _now_str()
    pipeline_log["timing_s"]  = timing
    pipeline_log["stages"]    = {s: True for s in timing}

    metadata = {
        "feature_count":             len(features),
        "features":                  features,
        "max_loans_per_customer_99th": max_loans,
        "baseline_metrics":          baseline_metrics,
        "final_metrics":             final_metrics,
        "xgboost_params":            meta_learner.best_params,
        "training_history":          ds_history,
    }

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

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(_to_json_safe(metadata), f, indent=2)

    with open(run_dir / "pipeline_log.json", "w") as f:
        json.dump(_to_json_safe(pipeline_log), f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete  →  {run_dir}")
    logger.info(f"  Baseline  Macro F1: {baseline_metrics['macro_f1']:.4f}")
    logger.info(f"  DeepSets+XGB Macro F1: {final_metrics['macro_f1']:.4f}")
    logger.info(f"  Cat-2 Recall:          {final_metrics.get('recall_class_2', '?'):.4f}")
    logger.info(f"  Total wall time:        {sum(timing.values()):.0f}s")
    logger.info("=" * 60)


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
