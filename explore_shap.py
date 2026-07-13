"""
explore_shap.py
===============
Standalone SHAP explorer for the Loan Default Classification project.

Two modes:

  --bundle <model_bundle.pkl> --data <snapshot.csv>   (primary, current arms)
      Loads the deployed model_arm bundle and a raw snapshot DataFrame (same
      shape as TRAIN_TABLE — see column_changes.md), aggregates it the same
      way scoring does, and runs SHAP on the real named features (MIN_/MAX_/
      MEAN_/STD_<feature> + N_LOANS). Only supports arms with a single
      xgboost model (multiclass, binary) — not ordinal/per_cat.

  --embeddings_path <.npy> --labels_path <.npy> --model_path <xgb.json>  (legacy)
      Pre-computed DeepSets embeddings from a run with DEEPSETS_ENABLED=True.
      Feature names are opaque (embed_0 ... embed_N) since the DeepSets
      encoder's dimensions have no individual meaning.

Outputs (in --output_dir, default: explore_output/):
  shap_summary_class<N>.png    -- beeswarm plot for classes 2 and 3
  shap_class<N>_bar.png        -- mean |SHAP| bar chart per class
  shap_class<N>_dependence.png -- top-feature dependence plots per class

Usage:
  # Current arm-based model (no DeepSets, no DB — data is a plain CSV):
  python explore_shap.py --bundle artifacts/<ts>_final/fold_01/model_bundle.pkl \\
                         --data snapshot_sample.csv

  # Legacy DeepSets embeddings (DEEPSETS_ENABLED=True runs only):
  python explore_shap.py --model_path artifacts/<run>/fold_01/xgboost_model.json \\
                         --embeddings_path artifacts/<run>/fold_01/test_embeddings.npy \\
                         --labels_path artifacts/<run>/fold_01/test_labels.npy
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


# ── Data loading: current arm-based path ────────────────────────────────────────

def load_from_bundle(bundle_path: Path, data_path: Path) -> tuple:
    """
    Load a model_bundle.pkl (see src/inference/model_loader.py) and a raw
    snapshot CSV, aggregate it the same way scoring does, and return
    (xgb_booster, X, y, feature_names). Only single-model arms (multiclass,
    binary) are supported — ordinal/per_cat compose several boosters and
    aren't representable as one TreeExplainer target.
    """
    import joblib
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.baselines.aggregated_xgboost import aggregate_features
    from src.data.data_loader import DataLoader

    if not bundle_path.exists():
        log.error(f"Bundle not found: {bundle_path}")
        sys.exit(1)
    if not data_path.exists():
        log.error(f"Data file not found: {data_path}")
        sys.exit(1)

    bundle = joblib.load(bundle_path)
    if bundle.get("kind") != "arm":
        log.error("This bundle is not an arm bundle (kind != 'arm'). "
                   "For legacy DeepSets bundles, use --model_path/--embeddings_path/--labels_path instead.")
        sys.exit(1)

    scaler = bundle["scaler"]
    arm = bundle["arm"]
    features = bundle["metadata"]["features"]
    max_loans = bundle["metadata"]["max_loans_per_customer_99th"]

    if getattr(arm, "model", None) is None:
        log.error(
            f"Arm '{getattr(arm, 'name', type(arm).__name__)}' has no single "
            f"trained '.model' (ordinal/per_cat arms compose several "
            f"boosters via '.models' instead) — SHAP needs a single XGBoost "
            f"model. Re-run with DEPLOY_ARM='multiclass' or 'binary'."
        )
        sys.exit(1)

    df = pd.read_csv(data_path)
    dl = DataLoader()
    instances, raw_cols = dl.process_raw_data(df, max_loans)
    if raw_cols != features:
        log.warning("Data columns differ from the bundle's training features — "
                    "proceeding, but check for a schema mismatch.")

    X_scaled = scaler.transform([i["features"] for i in instances])
    agg_in = [{"features": x, "n_loans": i["n_loans"], "label": i["label"]}
              for i, x in zip(instances, X_scaled)]
    X, y = aggregate_features(agg_in)

    agg_names = (
        [f"MIN_{f}" for f in features] + [f"MAX_{f}" for f in features]
        + [f"MEAN_{f}" for f in features] + [f"STD_{f}" for f in features]
        + ["N_LOANS"]
    )
    log.info(f"Aggregated {len(instances):,} instances into {X.shape[1]} named features "
             f"using arm '{getattr(arm, 'name', '?')}'.")
    return arm.model, X, y, agg_names


# ── Data loading: legacy DeepSets embeddings ────────────────────────────────────

def load_xgb_model(model_path: Path):
    """Load XGBoost model from JSON or joblib pickle (legacy embeddings mode)."""
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost is not installed.")
        sys.exit(1)

    if not model_path.exists():
        log.error(f"Model file not found: {model_path}")
        sys.exit(1)

    model = xgb.XGBClassifier()
    if model_path.suffix == ".json":
        model.load_model(str(model_path))
    else:
        import joblib
        model = joblib.load(model_path)
    log.info(f"Loaded XGBoost model from {model_path}.")
    return model


def load_arrays(embeddings_path: Path, labels_path: Path) -> tuple:
    for p in [embeddings_path, labels_path]:
        if not p.exists():
            log.error(f"File not found: {p}")
            sys.exit(1)
    X = np.load(embeddings_path)
    y = np.load(labels_path)
    log.info(f"Embeddings shape: {X.shape},  Labels shape: {y.shape}")
    feature_names = [f"embed_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


# ── SHAP analysis (shared by both modes) ────────────────────────────────────────

def run_shap_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    output_dir: Path,
    max_display: int = 20,
    sample_size: int = 5000,
) -> None:
    try:
        import shap
    except ImportError:
        log.error("shap is not installed. Run: pip install shap")
        sys.exit(1)

    if len(y) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(y), size=sample_size, replace=False)
        X_s, y_s = X[idx], y[idx]
        log.info(f"Sampled {sample_size:,} / {len(y):,} instances for SHAP.")
    else:
        X_s, y_s = X, y

    log.info("Computing SHAP values (TreeExplainer) ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)

    # shap_values shape: (N, F, C) for multi-class XGBoost, or list of (N, F)
    if isinstance(shap_values, list):
        sv_list = shap_values
    elif shap_values.ndim == 3:
        sv_list = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
    else:
        sv_list = [shap_values]   # binary arm: single output

    n_classes = len(sv_list)
    log.info(f"SHAP values computed for {n_classes} output(s).")

    # ── 1. Global summary plot ────────────────────────────────────────────────
    summary_targets = [(2, "Past Due+"), (3, "Severe Past Due")] if n_classes > 1 \
        else [(0, "Severe (binary)")]
    for cls, name in summary_targets:
        if cls >= n_classes:
            continue
        log.info(f"Generating global SHAP summary plot for class {cls} ({name}) ...")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_list[cls], X_s, feature_names=feature_names,
            max_display=max_display, show=False, plot_type="dot",
        )
        plt.title(f"SHAP Summary -- Class {cls} ({name} vs Rest)", fontweight="bold", fontsize=12)
        plt.tight_layout()
        save_path = output_dir / f"shap_summary_class{cls}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved -> {save_path.name}")

    # ── 2. Per-class mean |SHAP| bar charts ──────────────────────────────────
    colors = ["#4C9BE8", "#F0A500", "#E84C4C", "#4CAF50"]
    for cls in range(n_classes):
        cls_name = CLASS_NAMES[cls] if n_classes == 4 else "Severe (binary)"
        log.info(f"Generating bar chart for Class {cls} ({cls_name}) ...")
        mean_abs_shap = np.abs(sv_list[cls]).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)[::-1][:max_display][::-1]

        fig, ax = plt.subplots(figsize=(8, max(5, max_display * 0.28)))
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, mean_abs_shap[sorted_idx], color=colors[cls % len(colors)], alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Importance (SHAP) -- Class {cls}: {cls_name}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        save_path = output_dir / f"shap_class{cls}_bar.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved -> {save_path.name}")

    # ── 3. Dependence plots for top features of each class ────────────────────
    _plot_dependence_grid(sv_list, X_s, feature_names, output_dir, n_top=6)

    print_top_features(sv_list, feature_names)
    log.info(f"\nAll SHAP plots saved to: {output_dir.resolve()}")


def _plot_dependence_grid(sv_list, X, feature_names, output_dir, n_top=6):
    colors = ["#4C9BE8", "#F0A500", "#E84C4C", "#4CAF50"]
    for cls, sv in enumerate(sv_list):
        mean_abs = np.abs(sv).mean(axis=0)
        top_feats = np.argsort(mean_abs)[::-1][:n_top]

        cols = 3
        rows = (n_top + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
        axes = np.atleast_1d(axes).flatten()

        for plot_i, feat_i in enumerate(top_feats):
            ax = axes[plot_i]
            ax.scatter(X[:, feat_i], sv[:, feat_i], c=colors[cls % len(colors)], alpha=0.3, s=4)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xlabel(feature_names[feat_i], fontsize=8)
            ax.set_ylabel("SHAP value", fontsize=8)
            ax.set_title(feature_names[feat_i], fontsize=9, fontweight="bold")
            ax.grid(alpha=0.2)

        for j in range(len(top_feats), len(axes)):
            axes[j].set_visible(False)

        cls_name = CLASS_NAMES[cls] if len(sv_list) == 4 else "Severe (binary)"
        fig.suptitle(f"SHAP Dependence Plots -- Class {cls}: {cls_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        save_path = output_dir / f"shap_class{cls}_dependence.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved dependence plot -> {save_path.name}")


def print_top_features(sv_list, feature_names, top_n=10):
    print("\n" + "=" * 65)
    print("  TOP SHAP FEATURES PER CLASS")
    print("=" * 65)
    for cls, sv in enumerate(sv_list):
        cls_name = CLASS_NAMES[cls] if len(sv_list) == 4 else "Severe (binary)"
        mean_abs = np.abs(sv).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]
        print(f"\n  Class {cls} ({cls_name}):")
        for rank, i in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feature_names[i]:30s}  mean|SHAP| = {mean_abs[i]:.5f}")
    print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=None,
                        help="Path to model_bundle.pkl (current arm-based model).")
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to a raw snapshot CSV (same columns as TRAIN_TABLE). Required with --bundle.")
    parser.add_argument("--model_path", type=Path, default=None,
                        help="[legacy] Path to saved XGBoost model (.json or .pkl).")
    parser.add_argument("--embeddings_path", type=Path, default=None,
                        help="[legacy] Path to .npy DeepSets test-set embeddings.")
    parser.add_argument("--labels_path", type=Path, default=None,
                        help="[legacy] Path to .npy test-set labels.")
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "explore_output",
                        help="Output directory (default: ./explore_output/).")
    parser.add_argument("--max_display", type=int, default=20,
                        help="Max features to show in summary/bar plots. (default: 20)")
    parser.add_argument("--sample_size", type=int, default=5000,
                        help="Max samples to use for SHAP computation. (default: 5000)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.bundle:
        if not args.data:
            parser.error("--data is required with --bundle")
        model, X, y, feature_names = load_from_bundle(args.bundle, args.data)
    elif args.model_path and args.embeddings_path and args.labels_path:
        model = load_xgb_model(args.model_path)
        X, y, feature_names = load_arrays(args.embeddings_path, args.labels_path)
    else:
        parser.error("Provide either --bundle + --data, or the legacy "
                     "--model_path + --embeddings_path + --labels_path.")
        return

    run_shap_analysis(
        model=model, X=X, y=y, feature_names=feature_names,
        output_dir=args.output_dir,
        max_display=args.max_display, sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
