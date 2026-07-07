"""
explore_iv_woe.py
=================
Standalone Information Value (IV) & Weight of Evidence (WoE) explorer.

Strategy: One-vs-Rest (OvR) for each of the 3 classes:
  - OvR-0: "No Delay"  vs  {Current, Past Due+}
  - OvR-1: "Current"   vs  {No Delay, Past Due+}   <- hardest to separate
  - OvR-2: "Past Due+" vs  {No Delay, Current}

Inputs (no DB required -- reads from NPZ cache):
  data/train_portfolios_cache.npz  +  data/train_portfolios_cache.manifest.json

Outputs (in --output_dir, default: explore_output/):
  iv_report.csv            -- IV for every feature across all 3 OvR problems
  iv_chart.png             -- horizontal bar chart sorted by max IV across OvRs
  woe_detail_<feature>.png -- per-bin WoE bar charts for top N features

Usage examples:
  python explore_iv_woe.py
  python explore_iv_woe.py --n_bins 15 --top_n 30
  python explore_iv_woe.py --output_dir my_results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from src.data.temporal_split import filter_mature_snapshots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Class labels (must match project_config.py NUM_CLASSES order)
CLASS_NAMES = {0: "No Delay", 1: "Current", 2: "Past Due+"}


# ── Cache loading ─────────────────────────────────────────────────────────────

def load_cache(data_dir: Path):
    """
    Returns:
        features_flat  : (N_total_loans, F)  float32
        offsets        : (N_instances+1,)    int32
        labels         : (N_instances,)      int32
        snapshot_dates : (N_instances,)      object
        feat_cols      : list[str]
    """
    cache_path    = data_dir / "train_portfolios_cache.npz"
    manifest_path = data_dir / "train_portfolios_cache.manifest.json"

    if not cache_path.exists():
        log.error(f"Cache not found: {cache_path}")
        log.error("Run the training pipeline first (python run.py train) to generate the cache.")
        sys.exit(1)

    log.info(f"Loading cache from {cache_path} ...")
    with np.load(cache_path, allow_pickle=True) as npz:
        features_flat  = npz["features_flat"]
        offsets        = npz["offsets"]
        labels         = npz["labels"]
        snapshot_dates = npz["snapshot_dates"]

    with open(manifest_path) as f:
        manifest = json.load(f)
    feat_cols = manifest["feature_cols"]

    log.info(
        f"Loaded {len(labels):,} portfolio instances, "
        f"{features_flat.shape[0]:,} loans, {len(feat_cols)} features."
    )
    return features_flat, offsets, labels, snapshot_dates, feat_cols


def build_portfolio_means(features_flat: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    Collapse each portfolio (variable number of loans) to its mean loan vector.
    Returns (N_instances, F).
    """
    n_instances = len(offsets) - 1
    F = features_flat.shape[1]
    X_mean = np.empty((n_instances, F), dtype=np.float32)
    log.info(f"Collapsing {n_instances:,} portfolios to mean-loan vectors ...")
    for i in range(n_instances):
        X_mean[i] = features_flat[offsets[i] : offsets[i + 1]].mean(axis=0)
    return X_mean


def filter_matured_instances(
    X_mean: np.ndarray, labels: np.ndarray, snapshot_dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop instances from snapshots whose WORST_FUTURE_CAT label hasn't
    matured yet. On an immature snapshot the label degenerates to "worst
    category observed so far" — mixing those rows in silently mislabels
    part of the IV computation (independent of the quantile-binning fix).
    """
    mature = set(filter_mature_snapshots(np.unique(snapshot_dates)))
    mask = np.array([s in mature for s in snapshot_dates])
    n_dropped = len(mask) - mask.sum()
    if n_dropped:
        log.warning(
            f"Dropping {n_dropped:,} instance(s) from immature snapshot(s) "
            f"before IV computation."
        )
    return X_mean[mask], labels[mask]


# ── WoE / IV calculation ──────────────────────────────────────────────────────

def _make_bins(feat_valid: np.ndarray, n_bins: int):
    """
    Bin ids for IV computation, robust to skewed features.

    The original quantile-only binning collapsed to a SINGLE bin for any
    feature with >~(1 - 1/n_bins) of its mass on one value (all binary flags,
    rare-event counts, closed-loan DPDs), silently forcing IV = 0 regardless
    of predictive power. Fallbacks:
      - < 2 distinct values      → None (truly constant, IV 0 is correct)
      - <= n_bins distinct values → one bin per distinct value
      - quantile edges collapsed  → mode gets its own bin, rest quantile-binned
    """
    uniq = np.unique(feat_valid)
    if len(uniq) < 2:
        return None
    if len(uniq) <= n_bins:
        return np.searchsorted(uniq, feat_valid)

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(feat_valid, quantiles))
    if len(edges) >= 3:
        return np.digitize(feat_valid, edges[1:-1])

    # Mode-dominated: >=~80% of mass on one value ⇒ that value is the median.
    mode = np.median(feat_valid)
    rest = feat_valid[feat_valid != mode]
    sub_edges = np.unique(np.percentile(rest, quantiles))
    sub_ids = np.digitize(feat_valid, sub_edges[1:-1])
    return np.where(feat_valid == mode, 0, sub_ids + 1)


def compute_woe_iv_binary(
    feature: np.ndarray,
    binary_target: np.ndarray,
    n_bins: int = 10,
) -> tuple:
    """
    Compute WoE and IV for a single feature against a binary target.

    Args:
        feature       : (N,) float array
        binary_target : (N,) int array  (1 = "positive" class, 0 = "negative")
        n_bins        : number of quantile bins

    Returns:
        df_woe : DataFrame [bin, count, event, non_event, woe, iv_contribution]
        iv     : scalar IV for this feature
    """
    valid_mask = np.isfinite(feature)
    if valid_mask.sum() < 10:
        return pd.DataFrame(), 0.0

    feat_valid   = feature[valid_mask]
    target_valid = binary_target[valid_mask]

    bin_ids = _make_bins(feat_valid, n_bins)
    if bin_ids is None:
        return pd.DataFrame(), 0.0

    total_events     = max(int(target_valid.sum()), 1)
    total_non_events = max(int((1 - target_valid).sum()), 1)

    rows = []
    for b in np.unique(bin_ids):
        mask         = bin_ids == b
        events       = int(target_valid[mask].sum())
        n_events     = max(events, 0.5)
        n_non_events = max(mask.sum() - events, 0.5)

        dist_event     = n_events     / total_events
        dist_non_event = n_non_events / total_non_events

        woe        = float(np.log(np.clip(dist_event / dist_non_event, 1e-9, None)))
        iv_contrib = (dist_event - dist_non_event) * woe

        rows.append({
            "bin":             b,
            "count":           int(mask.sum()),
            "event":           events,
            "non_event":       int(mask.sum() - events),
            "dist_event":      dist_event,
            "dist_non_event":  dist_non_event,
            "woe":             woe,
            "iv_contribution": iv_contrib,
        })

    df_woe = pd.DataFrame(rows)
    iv = float(df_woe["iv_contribution"].sum())
    return df_woe, iv


def compute_all_ivs(
    X: np.ndarray,
    labels: np.ndarray,
    feat_cols: list,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute OvR IV for all features and all 3 classes.

    Returns DataFrame: feature, iv_ovr0, iv_ovr1, iv_ovr2, iv_max, iv_mean
    """
    n_classes = 3
    records = []

    log.info(f"Computing OvR IV for {len(feat_cols)} features x {n_classes} classes ...")
    for feat_idx, feat_name in enumerate(feat_cols):
        feat_vals = X[:, feat_idx]
        row = {"feature": feat_name}
        ivs = []
        for cls in range(n_classes):
            binary = (labels == cls).astype(np.int32)
            _, iv = compute_woe_iv_binary(feat_vals, binary, n_bins=n_bins)
            row[f"iv_ovr{cls}"] = round(iv, 5)
            ivs.append(iv)
        row["iv_max"]  = round(max(ivs), 5)
        row["iv_mean"] = round(float(np.mean(ivs)), 5)
        records.append(row)

    df = pd.DataFrame(records).sort_values("iv_max", ascending=False).reset_index(drop=True)
    return df


# ── WoE bin detail plots ──────────────────────────────────────────────────────

def plot_woe_detail(
    X: np.ndarray,
    labels: np.ndarray,
    feat_name: str,
    feat_idx: int,
    output_dir: Path,
    n_bins: int = 10,
) -> None:
    """Plot per-class OvR WoE bars for a single feature (3 subplots)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["steelblue", "darkorange", "crimson"]
    feat_vals = X[:, feat_idx]

    for cls, (ax, color) in enumerate(zip(axes, colors)):
        binary = (labels == cls).astype(np.int32)
        df_woe, iv = compute_woe_iv_binary(feat_vals, binary, n_bins=n_bins)
        if df_woe.empty:
            ax.set_title(f"OvR-{cls}: no data")
            continue
        ax.bar(range(len(df_woe)), df_woe["woe"], color=color, alpha=0.8, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"OvR-{cls}: {CLASS_NAMES[cls]} vs Rest\nIV = {iv:.4f}", fontsize=10)
        ax.set_xlabel("Quantile Bin")
        ax.set_ylabel("WoE")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"WoE by Bin -- {feat_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    safe_name = feat_name.replace("/", "_").replace("\\", "_")
    save_path = output_dir / f"woe_detail_{safe_name}.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved WoE detail -> {save_path.name}")


# ── Summary IV chart ──────────────────────────────────────────────────────────

def plot_iv_chart(df_iv: pd.DataFrame, output_dir: Path, top_n: int = 40) -> None:
    """Horizontal grouped bar chart of IV for top_n features."""
    df_plot = df_iv.head(top_n).iloc[::-1]  # flip for bottom-up

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    y = np.arange(len(df_plot))
    h = 0.25

    ax.barh(y - h, df_plot["iv_ovr0"], h * 0.9, label="OvR-0 (No Delay)",  color="steelblue",  alpha=0.85)
    ax.barh(y,     df_plot["iv_ovr1"], h * 0.9, label="OvR-1 (Current)",   color="darkorange", alpha=0.85)
    ax.barh(y + h, df_plot["iv_ovr2"], h * 0.9, label="OvR-2 (Past Due+)", color="crimson",    alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["feature"], fontsize=8)
    ax.set_xlabel("Information Value (IV)")
    ax.set_title("Feature IV by Class (One-vs-Rest) -- sorted by max IV", fontweight="bold")
    ax.legend(loc="lower right")

    # Reference lines
    for thresh, color, label in [
        (0.02, "gray",      "Weak (0.02)"),
        (0.10, "goldenrod", "Medium (0.10)"),
        (0.30, "green",     "Strong (0.30)"),
    ]:
        ax.axvline(thresh, color=color, linestyle="--", linewidth=0.8, alpha=0.8)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / "iv_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved IV chart -> {save_path}")


def print_summary(df_iv: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  IV DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(df_iv[["feature", "iv_ovr0", "iv_ovr1", "iv_ovr2", "iv_max"]].head(25).to_string(index=False))
    print("\n--- Per-class coverage ---")
    for cls in range(3):
        col     = f"iv_ovr{cls}"
        strong  = int((df_iv[col] >= 0.10).sum())
        medium  = int(((df_iv[col] >= 0.02) & (df_iv[col] < 0.10)).sum())
        useless = int((df_iv[col] < 0.02).sum())
        print(
            f"  OvR-{cls} ({CLASS_NAMES[cls]:12s}) | "
            f"Strong (>=0.10): {strong:3d}  |  Medium: {medium:3d}  |  Useless: {useless:3d}"
        )
    print("=" * 70 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute OvR IV/WoE for all features. Reads from NPZ cache."
    )
    parser.add_argument("--data_dir",    type=Path, default=Path(__file__).parent / "data",
                        help="Directory containing the NPZ cache (default: ./data/)")
    parser.add_argument("--output_dir",  type=Path, default=Path(__file__).parent / "explore_output",
                        help="Output directory (default: ./explore_output/)")
    parser.add_argument("--n_bins",      type=int,  default=10,
                        help="Number of quantile bins per feature (default: 10)")
    parser.add_argument("--top_n",       type=int,  default=20,
                        help="Number of top features to plot WoE detail for (default: 20)")
    parser.add_argument("--chart_top_n", type=int,  default=40,
                        help="Number of features in the summary bar chart (default: 40)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features_flat, offsets, labels, snapshot_dates, feat_cols = load_cache(args.data_dir)
    X_mean = build_portfolio_means(features_flat, offsets)
    X_mean, labels = filter_matured_instances(X_mean, labels, snapshot_dates)

    df_iv = compute_all_ivs(X_mean, labels, feat_cols, n_bins=args.n_bins)

    csv_path = args.output_dir / "iv_report.csv"
    df_iv.to_csv(csv_path, index=False)
    log.info(f"Saved IV report -> {csv_path}")

    print_summary(df_iv)
    plot_iv_chart(df_iv, args.output_dir, top_n=args.chart_top_n)

    log.info(f"Generating WoE detail plots for top {args.top_n} features ...")
    for _, row in df_iv.head(args.top_n).iterrows():
        feat_name = row["feature"]
        feat_idx  = feat_cols.index(feat_name)
        plot_woe_detail(X_mean, labels, feat_name, feat_idx, args.output_dir, n_bins=args.n_bins)

    log.info(f"\nDone. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
