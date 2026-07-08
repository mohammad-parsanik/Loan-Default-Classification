"""
explore_shap.py
===============
Standalone SHAP explorer for the Loan Default Classification project.

Loads a saved XGBoost model and pre-computed embeddings (from a training run),
then generates SHAP summary + dependence plots.

Inputs (copy from the training server's artifact directory):
  --model_path       path to the saved XGBoost model (.json or joblib .pkl)
  --embeddings_path  path to .npy test-set embeddings  (shape: N x EMBED_DIM)
  --labels_path      path to .npy test-set labels      (shape: N,)

Outputs (in --output_dir, default: explore_output/):
  shap_summary.png             -- beeswarm plot across all classes
  shap_class<N>_bar.png        -- mean |SHAP| bar chart per class (N=0,1,2)
  shap_class<N>_dependence.png -- top-feature dependence plots per class

Usage examples:
  # Basic (model and embeddings from artifact dir):
  python explore_shap.py \\
    --model_path artifacts/20260701_124055/xgb_model.json \\
    --embeddings_path artifacts/20260701_124055/test_embeddings.npy \\
    --labels_path artifacts/20260701_124055/test_labels.npy

  # Show more features in bar chart:
  python explore_shap.py ... --max_display 30

NOTE: You need to copy the artifact directory from the training server first.
      Only xgb_model.json + test_embeddings.npy + test_labels.npy are required.

How to export embeddings from the trained model (run on the server):
  # Add this snippet to run.py after Stage 8 completes, then re-run:
  import numpy as np
  emb, lbl = meta_learner._extract_embeddings(test_dataset)
  np.save(run_dir / "test_embeddings.npy", emb)
  np.save(run_dir / "test_labels.npy", lbl)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CLASS_NAMES = ["No Delay", "Current", "Past Due+", "Severe Past Due"]


# ── Model + data loading ──────────────────────────────────────────────────────

def load_xgb_model(model_path: Path):
    """Load XGBoost model from JSON or joblib pickle."""
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost is not installed.")
        sys.exit(1)

    if not model_path.exists():
        log.error(f"Model file not found: {model_path}")
        log.error("Copy the xgb_model.json from the training server's artifact directory.")
        sys.exit(1)

    model = xgb.XGBClassifier()
    if model_path.suffix == ".json":
        model.load_model(str(model_path))
        log.info(f"Loaded XGBoost model from {model_path} (JSON format).")
    else:
        try:
            import joblib
            model = joblib.load(model_path)
            log.info(f"Loaded XGBoost model from {model_path} (joblib format).")
        except Exception as e:
            log.error(f"Could not load model: {e}")
            sys.exit(1)
    return model


def load_arrays(embeddings_path: Path, labels_path: Path) -> tuple:
    for p in [embeddings_path, labels_path]:
        if not p.exists():
            log.error(f"File not found: {p}")
            sys.exit(1)
    X = np.load(embeddings_path)
    y = np.load(labels_path)
    log.info(f"Embeddings shape: {X.shape},  Labels shape: {y.shape}")
    return X, y


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def run_shap_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    max_display: int = 20,
    sample_size: int = 5000,
) -> None:
    try:
        import shap
    except ImportError:
        log.error("shap is not installed. Run: pip install shap")
        sys.exit(1)

    # Downsample if large
    if len(y) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), size=sample_size, replace=False)
        X_s, y_s = X[idx], y[idx]
        log.info(f"Sampled {sample_size:,} / {len(y):,} instances for SHAP.")
    else:
        X_s, y_s = X, y

    # Feature names: embedding dim labels
    feature_names = [f"embed_{i}" for i in range(X_s.shape[1])]

    log.info("Computing SHAP values (TreeExplainer) ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)

    # shap_values shape: (N, F, C) for multi-class XGBoost, or list of (N, F)
    if isinstance(shap_values, list):
        # older shap: list of (N, F) arrays, one per class
        sv_list = shap_values  # len = n_classes
    else:
        # newer shap: (N, F, C)
        sv_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    n_classes = len(sv_list)
    log.info(f"SHAP values computed for {n_classes} classes.")

    # ── 1. Global summary plot (multi-output beeswarm) ────────────────────────
    # Class 2 (Past Due+) and class 3 (Severe Past Due) are both actionable
    # tiers of delinquency — plot a summary for each.
    for cls, name in ((2, "Past Due+"), (3, "Severe Past Due")):
        if cls >= n_classes:
            continue
        log.info(f"Generating global SHAP summary plot for class {cls} ({name}) ...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_list[cls],
            X_s,
            feature_names = feature_names,
            max_display   = max_display,
            show          = False,
            plot_type     = "dot",
        )
        plt.title(f"SHAP Summary -- Class {cls} ({name} vs Rest)", fontweight="bold", fontsize=12)
        plt.tight_layout()
        save_path = output_dir / f"shap_summary_class{cls}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved -> {save_path.name}")

    # ── 2. Per-class mean |SHAP| bar charts ──────────────────────────────────
    for cls in range(n_classes):
        log.info(f"Generating bar chart for Class {cls} ({CLASS_NAMES[cls]}) ...")
        mean_abs_shap = np.abs(sv_list[cls]).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)[::-1][:max_display][::-1]

        fig, ax = plt.subplots(figsize=(8, max(5, max_display * 0.28)))
        y_pos = np.arange(len(sorted_idx))
        ax.barh(
            y_pos,
            mean_abs_shap[sorted_idx],
            color = ["#4C9BE8", "#F0A500", "#E84C4C"][cls],
            alpha = 0.85,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(
            f"Feature Importance (SHAP) -- OvR Class {cls}: {CLASS_NAMES[cls]}",
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_path = output_dir / f"shap_class{cls}_bar.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved -> {save_path.name}")

    # ── 3. Dependence plots for top features of each class ────────────────────
    _plot_dependence_grid(sv_list, X_s, feature_names, output_dir, n_top=6)

    log.info(f"\nAll SHAP plots saved to: {output_dir.resolve()}")


def _plot_dependence_grid(
    sv_list: list,
    X: np.ndarray,
    feature_names: list,
    output_dir: Path,
    n_top: int = 6,
) -> None:
    """
    For each class, plot a grid of dependence plots for its top N SHAP features.
    Dependence plots reveal whether the relationship between a feature and the
    SHAP value is monotonic, threshold-based, or noisy.
    """
    try:
        import shap
    except ImportError:
        return

    colors = ["#4C9BE8", "#F0A500", "#E84C4C", "#4CAF50"]
    for cls, sv in enumerate(sv_list):
        mean_abs = np.abs(sv).mean(axis=0)
        top_feats = np.argsort(mean_abs)[::-1][:n_top]

        cols = 3
        rows = (n_top + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
        axes = axes.flatten()

        for plot_i, feat_i in enumerate(top_feats):
            ax = axes[plot_i]
            ax.scatter(
                X[:, feat_i],
                sv[:, feat_i],
                c      = colors[cls],
                alpha  = 0.3,
                s      = 4,
            )
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlabel(feature_names[feat_i], fontsize=8)
            ax.set_ylabel("SHAP value", fontsize=8)
            ax.set_title(feature_names[feat_i], fontsize=9, fontweight="bold")
            ax.grid(alpha=0.2)

        # Hide unused axes
        for j in range(len(top_feats), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"SHAP Dependence Plots -- Class {cls}: {CLASS_NAMES[cls]}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        save_path = output_dir / f"shap_class{cls}_dependence.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved dependence plot -> {save_path.name}")


def print_top_features(sv_list: list, feature_names: list, top_n: int = 10) -> None:
    print("\n" + "=" * 65)
    print("  TOP SHAP FEATURES PER CLASS")
    print("=" * 65)
    for cls, sv in enumerate(sv_list):
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]
        print(f"\n  Class {cls} ({CLASS_NAMES[cls]}):")
        for rank, i in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_names[i]:30s}  mean|SHAP| = {mean_abs[i]:.5f}")
    print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SHAP analysis for the XGBoost meta-learner. Requires training artifacts."
    )
    parser.add_argument("--model_path",       type=Path, required=True,
                        help="Path to saved XGBoost model (.json or .pkl joblib).")
    parser.add_argument("--embeddings_path",  type=Path, required=True,
                        help="Path to .npy test-set embeddings (N x embed_dim).")
    parser.add_argument("--labels_path",      type=Path, required=True,
                        help="Path to .npy test-set labels (N,).")
    parser.add_argument("--output_dir",       type=Path, default=Path(__file__).parent / "explore_output",
                        help="Output directory (default: ./explore_output/).")
    parser.add_argument("--max_display",      type=int,  default=20,
                        help="Max features to show in summary/bar plots. (default: 20)")
    parser.add_argument("--sample_size",      type=int,  default=5000,
                        help="Max samples to use for SHAP computation. (default: 5000)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model  = load_xgb_model(args.model_path)
    X, y   = load_arrays(args.embeddings_path, args.labels_path)

    run_shap_analysis(
        model        = model,
        X            = X,
        y            = y,
        output_dir   = args.output_dir,
        max_display  = args.max_display,
        sample_size  = args.sample_size,
    )


if __name__ == "__main__":
    main()
