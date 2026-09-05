"""
explore_umap.py
===============
Standalone UMAP explorer for the Loan Default Classification project.

Two modes:
  --mode raw         Read NPZ cache, collapse portfolios to mean-loan vectors,
                     run UMAP on a stratified sample. Tests raw feature separability.

  --mode embeddings  Read a pre-computed .npy embedding file from a training run.
                     Much faster; tests separability at the model's representation level.

All UMAP hyperparameters are exposed as CLI flags so you can tune without
touching the code.

Outputs (in --output_dir):
  umap_<mode>_<timestamp>.png   -- 2D or 3D scatter coloured by class label

Usage examples:
  # Raw features (no model artifacts needed):
  python explore_umap.py --mode raw

  # Tune UMAP parameters:
  python explore_umap.py --mode raw --n_neighbors 15 --min_dist 0.05 --sample_size 50000

  # Use pre-computed embeddings from the training run:
  python explore_umap.py --mode embeddings --embeddings_path path/to/embeddings.npy --labels_path path/to/labels.npy

  # 3D plot:
  python explore_umap.py --mode raw --n_components 3
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from src.data.data_loader import load_cached_arrays  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CLASS_NAMES  = ["No Delay", "Current", "Past Due+", "Severe Past Due"]
CLASS_COLORS = ["#4C9BE8", "#F0A500", "#E84C4C", "#4CAF50"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_cache(cache_dir=None) -> tuple:
    """Load and mean-pool portfolios from the per-snapshot NPZ cache."""
    try:
        arrays, feat_cols = load_cached_arrays(cache_dir)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    features_flat = arrays["features_flat"]   # (N_loans, F)
    offsets       = arrays["offsets"]          # (N_instances+1,)
    labels        = arrays["labels"]           # (N_instances,)

    log.info(f"Loaded {len(labels):,} instances, {len(feat_cols)} features. Mean-pooling portfolios ...")
    n_instances = len(offsets) - 1
    F = features_flat.shape[1]
    X = np.empty((n_instances, F), dtype=np.float32)
    for i in range(n_instances):
        X[i] = features_flat[offsets[i] : offsets[i + 1]].mean(axis=0)

    # Replace NaN/Inf with column medians (simple safeguard for UMAP)
    col_medians = np.nanmedian(X, axis=0)
    nan_mask    = ~np.isfinite(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    return X, labels


def load_from_npy(embeddings_path: Path, labels_path: Path) -> tuple:
    """Load pre-computed embeddings and labels from .npy files."""
    if not embeddings_path.exists():
        log.error(f"Embeddings file not found: {embeddings_path}")
        sys.exit(1)
    if not labels_path.exists():
        log.error(f"Labels file not found: {labels_path}")
        sys.exit(1)

    X      = np.load(embeddings_path)
    labels = np.load(labels_path)
    log.info(f"Loaded embeddings: {X.shape}, labels: {labels.shape}")
    return X, labels


# ── Stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(
    X: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int = 42,
) -> tuple:
    """Sample `sample_size` points preserving class distribution."""
    rng = np.random.default_rng(seed)
    if len(labels) <= sample_size:
        return X, labels

    classes, counts = np.unique(labels, return_counts=True)
    class_fractions = counts / counts.sum()
    idx_list = []

    for cls, frac in zip(classes, class_fractions):
        n_cls    = max(1, int(round(sample_size * frac)))
        cls_idx  = np.where(labels == cls)[0]
        chosen   = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
        idx_list.append(chosen)

    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    log.info(f"Sampled {len(idx):,} / {len(labels):,} instances (stratified).")
    return X[idx], labels[idx]


# ── UMAP fit + plot ───────────────────────────────────────────────────────────

def run_umap_and_plot(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    n_components: int,
    mode: str,
    output_dir: Path,
    random_state: int = 42,
) -> None:
    try:
        import umap
    except ImportError:
        log.error("umap-learn is not installed. Run: pip install umap-learn")
        sys.exit(1)

    log.info(
        f"Fitting UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, "
        f"metric={metric}, n_components={n_components} on {len(labels):,} samples ..."
    )
    reducer = umap.UMAP(
        n_neighbors  = n_neighbors,
        min_dist     = min_dist,
        metric       = metric,
        n_components = n_components,
        random_state = random_state,
        low_memory   = True,
        verbose      = True,
    )
    embedding = reducer.fit_transform(X)
    log.info("UMAP fit complete.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"umap_{mode}_{ts}.png"

    if n_components == 2:
        _plot_2d(embedding, labels, save_path, mode)
    elif n_components == 3:
        _plot_3d(embedding, labels, save_path, mode)
    else:
        log.warning(f"n_components={n_components} not directly plottable; saving first 2 dims.")
        _plot_2d(embedding[:, :2], labels, save_path, mode)

    log.info(f"Saved UMAP plot -> {save_path}")


def _plot_2d(emb: np.ndarray, labels: np.ndarray, save_path: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c       = CLASS_COLORS[cls],
            label   = CLASS_NAMES[cls],
            alpha   = 0.5,
            s       = 6,
            linewidths = 0,
        )
    ax.set_title(
        f"UMAP Projection -- {mode.capitalize()} Features\n"
        f"(n={len(labels):,}  |  classes coloured by WORST_FUTURE_CAT)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    legend = ax.legend(markerscale=3, framealpha=0.8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def _plot_3d(emb: np.ndarray, labels: np.ndarray, save_path: Path, mode: str) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        ax.scatter(
            emb[mask, 0], emb[mask, 1], emb[mask, 2],
            c      = CLASS_COLORS[cls],
            label  = CLASS_NAMES[cls],
            alpha  = 0.4,
            s      = 5,
        )
    ax.set_title(f"UMAP 3D -- {mode.capitalize()} Features", fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UMAP explorer for loan default data. Tunable hyperparameters."
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["raw", "embeddings"], default="raw",
        help=(
            "raw: mean-pool portfolios from NPZ cache. "
            "embeddings: load .npy file from a training run. (default: raw)"
        ),
    )

    # Paths
    parser.add_argument("--cache_dir",       type=Path, default=None,
                        help="Directory containing NPZ cache (for --mode raw).")
    parser.add_argument("--embeddings_path", type=Path, default=None,
                        help="Path to .npy embeddings array (required for --mode embeddings).")
    parser.add_argument("--labels_path",     type=Path, default=None,
                        help="Path to .npy labels array (required for --mode embeddings).")
    parser.add_argument("--output_dir",      type=Path, default=Path(__file__).parent / "explore_output",
                        help="Output directory (default: ./explore_output/).")

    # Sampling
    parser.add_argument("--sample_size",  type=int, default=10_000,
                        help="Number of instances to pass to UMAP (stratified). (default: 10000)")
    parser.add_argument("--random_seed",  type=int, default=42)

    # UMAP hyperparameters -- all tunable from CLI
    parser.add_argument("--n_neighbors",  type=int,   default=30,
                        help="UMAP: number of neighbours. Controls local vs global structure. (default: 30)")
    parser.add_argument("--min_dist",     type=float, default=0.1,
                        help="UMAP: min distance between points in low-dim space. Smaller = tighter clusters. (default: 0.1)")
    parser.add_argument("--metric",       type=str,   default="euclidean",
                        help="UMAP: distance metric. Try 'cosine', 'manhattan', 'euclidean'. (default: euclidean)")
    parser.add_argument("--n_components", type=int,   default=2, choices=[2, 3],
                        help="UMAP output dimensionality (2 or 3). (default: 2)")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.mode == "raw":
        X, labels = load_from_cache(args.cache_dir)
    else:
        if args.embeddings_path is None or args.labels_path is None:
            parser.error("--mode embeddings requires --embeddings_path and --labels_path")
        X, labels = load_from_npy(args.embeddings_path, args.labels_path)

    # Sample
    X_samp, y_samp = stratified_sample(X, labels, args.sample_size, seed=args.random_seed)

    # Print class distribution of sample
    for cls in np.unique(y_samp):
        n = int((y_samp == cls).sum())
        pct = 100 * n / len(y_samp)
        print(f"  Class {cls} ({CLASS_NAMES[int(cls)]}): {n:,}  ({pct:.1f}%)")

    # Run UMAP + save plot
    run_umap_and_plot(
        X            = X_samp,
        labels       = y_samp,
        n_neighbors  = args.n_neighbors,
        min_dist     = args.min_dist,
        metric       = args.metric,
        n_components = args.n_components,
        mode         = args.mode,
        output_dir   = args.output_dir,
        random_state = args.random_seed,
    )

    log.info(f"Done. Output saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
