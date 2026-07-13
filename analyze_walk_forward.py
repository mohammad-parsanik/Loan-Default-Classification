"""
analyze_walk_forward.py
========================
Reconstruct the walk-forward conclusion from saved artifacts when the
console log was lost. run.py's WF loop saves <run_dir>/stages/fold_results.pkl
after EVERY completed fold (not just at the end), so as long as folds
finished on disk, this file has everything the live log would have shown —
and more (per-arm metrics, not just the deployed one).

That file is pure metric dicts/numpy scalars (no model objects, no torch/
xgboost artifacts), so it's small and safe to copy off the server and
analyze anywhere — this script only needs numpy/pandas/matplotlib/joblib.

Usage:
  python analyze_walk_forward.py artifacts/<run_dir>
  python analyze_walk_forward.py artifacts/<run_dir>/stages/fold_results.pkl

  # If fold_results.pkl is missing/corrupted, fall back to scanning each
  # fold_XX/arms_metrics.json directly (loses test-snapshot identity, but
  # recovers the metrics):
  python analyze_walk_forward.py artifacts/<run_dir> --fallback
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

sys.path.append(str(Path(__file__).resolve().parent))
from src.evaluation.fold_aggregator import aggregate_fold_metrics, log_fold_summary


def _pooled_ap(arm_metrics: dict) -> float:
    return arm_metrics.get("ranking", {}).get("pr_auc", float("nan"))


def _cat0_ap(arm_metrics: dict) -> float:
    return (arm_metrics.get("ranking", {})
            .get("by_current_cat", {}).get("current_cat_0", {})
            .get("pr_auc", float("nan")))


def _recall_at(arm_metrics: dict, window: str) -> float:
    return arm_metrics.get("ranking", {}).get(f"at_{window}", {}).get("recall", float("nan"))


# ── Loading ──────────────────────────────────────────────────────────────────

def load_fold_results(path: Path) -> list[dict]:
    pkl_path = path if path.suffix == ".pkl" else path / "stages" / "fold_results.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"{pkl_path} not found. Pass --fallback to reconstruct from "
            f"per-fold arms_metrics.json instead (loses test-snapshot dates)."
        )
    fold_results = joblib.load(pkl_path)
    print(f"Loaded {len(fold_results)} completed fold(s) from {pkl_path}")
    return fold_results


def load_fold_results_fallback(run_dir: Path) -> list[dict]:
    """
    Reconstruct a fold_results-shaped list from each fold_XX/arms_metrics.json.
    Used only if the run-level pickle is missing/corrupted. Cannot recover
    train/val/test snapshot dates (those live only in the run-level pickle) —
    fold_id / directory order is the only ordering available.
    """
    fold_results = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        metrics_path = fold_dir / "arms_metrics.json"
        if not metrics_path.exists():
            continue
        arms = json.loads(metrics_path.read_text())
        fold_id = int(fold_dir.name.split("_")[-1])
        # Deployed arm = whichever one has bootstrap_ci (only computed for it)
        deployed = next((n for n, m in arms.items() if "bootstrap_ci" in m), None)
        fold_results.append({
            "fold_id": fold_id,
            "test_snap": None,   # not recoverable without the run-level pickle
            "val_snap": None,
            "train_snaps": [],
            "deployed_arm": deployed,
            "final_metrics": arms.get(deployed, {}),
            "arms": arms,
        })
    print(f"Reconstructed {len(fold_results)} fold(s) from arms_metrics.json "
          f"(fallback — test-snapshot dates unavailable).")
    return fold_results


# ── Analysis ─────────────────────────────────────────────────────────────────

def deployed_arm_table(fold_results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in fold_results:
        fm = r.get("final_metrics", {})
        train_snaps = sorted(r.get("train_snaps", []))
        rows.append({
            "fold_id": r["fold_id"],
            "test_snap": r.get("test_snap"),
            "val_snap": r.get("val_snap"),
            "n_train_snaps": len(train_snaps),
            "train_snaps": train_snaps,
            "deployed_arm": r.get("deployed_arm"),
            "pooled_AP": _pooled_ap(fm),
            "cat0_AP": _cat0_ap(fm),
            "R@1day": _recall_at(fm, "1_day"),
            "R@1week": _recall_at(fm, "1_week"),
            "R@1month": _recall_at(fm, "1_month"),
            "macro_f1": fm.get("macro_f1", float("nan")),
        })
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def per_fold_arm_comparison(fold_results: list[dict]) -> pd.DataFrame:
    """
    Pooled AP for EVERY trained arm, side by side, per fold — reveals
    whether an instability is arm-specific (only multiclass collapses) or
    shared (the fold/data itself is the problem, every arm struggles).
    """
    all_arms = sorted({name for r in fold_results for name in r.get("arms", {})})
    rows = []
    for r in sorted(fold_results, key=lambda r: r["fold_id"]):
        row = {"fold_id": r["fold_id"], "test_snap": r.get("test_snap"),
               "val_snap": r.get("val_snap"),
               "n_train_snaps": len(r.get("train_snaps", []))}
        for arm in all_arms:
            row[f"{arm}_AP"] = (_pooled_ap(r["arms"][arm]) if arm in r.get("arms", {})
                                else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def per_arm_stability_table(fold_results: list[dict]) -> pd.DataFrame:
    """
    For every arm that appears in ANY fold, its pooled AP across all folds
    that trained it, plus mean/std/min/max and how often it was the best
    arm that fold — extends the single-snapshot Run-6 verdict across time.
    """
    all_arms = sorted({name for r in fold_results for name in r.get("arms", {})})
    per_fold_best = []
    for r in fold_results:
        aps = {name: _pooled_ap(m) for name, m in r.get("arms", {}).items()}
        aps = {k: v for k, v in aps.items() if v == v}   # drop nan
        if aps:
            per_fold_best.append(max(aps, key=aps.get))

    rows = []
    for arm in all_arms:
        aps = [
            _pooled_ap(r["arms"][arm]) for r in fold_results
            if arm in r.get("arms", {})
        ]
        aps = [a for a in aps if a == a]
        win_rate = per_fold_best.count(arm) / len(per_fold_best) if per_fold_best else float("nan")
        rows.append({
            "arm": arm,
            "n_folds": len(aps),
            "mean_AP": np.mean(aps) if aps else float("nan"),
            "std_AP": np.std(aps) if aps else float("nan"),
            "min_AP": np.min(aps) if aps else float("nan"),
            "max_AP": np.max(aps) if aps else float("nan"),
            "fold_win_rate": win_rate,
        })
    return pd.DataFrame(rows).sort_values("mean_AP", ascending=False).reset_index(drop=True)


def filter_by_min_train_snaps(fold_results: list[dict], min_n: int) -> list[dict]:
    """
    Folds with very few training snapshots (early in the WF sequence, before
    much history existed) are a fundamentally different — and unrealistic —
    regime from production (which always trains on ALL mature snapshots).
    Averaging across them with mature-data folds understates the winning
    arm's real advantage. Use this to see the verdict restricted to folds
    with at least `min_n` training snapshots.
    """
    kept = [r for r in fold_results if len(r.get("train_snaps", [])) >= min_n]
    dropped = [r["fold_id"] for r in fold_results if r not in kept]
    if dropped:
        print(f"  (--min_train_snaps {min_n}: dropped fold(s) {dropped} "
              f"— fewer than {min_n} training snapshots)")
    return kept


def print_conclusion(deployed_df: pd.DataFrame, arm_df: pd.DataFrame,
                     arm_compare_df: pd.DataFrame = None) -> None:
    print("\n" + "=" * 78)
    print("  DEPLOYED-ARM RESULTS PER FOLD")
    print("=" * 78)
    cols = [c for c in deployed_df.columns if c != "train_snaps"]
    print(deployed_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    unstable = deployed_df[deployed_df["pooled_AP"] < 0.45]
    if len(unstable):
        print("\n  UNSTABLE FOLDS (pooled_AP < 0.45) — val_snap and exact training snapshots:")
        for _, row in unstable.iterrows():
            print(f"    fold {row['fold_id']:>2} | test={row['test_snap']} "
                  f"val={row['val_snap']} train={row['train_snaps']} "
                  f"| AP={row['pooled_AP']:.4f}")

    print("\n" + "=" * 78)
    print("  DEPLOYED-ARM STABILITY ACROSS FOLDS")
    print("=" * 78)
    for col in ["pooled_AP", "cat0_AP", "R@1week", "macro_f1"]:
        vals = deployed_df[col].dropna()
        if len(vals):
            print(f"  {col:10s}: {vals.mean():.4f} +/- {vals.std():.4f}  "
                  f"[{vals.min():.4f} - {vals.max():.4f}]  (n={len(vals)})")

    print("\n" + "=" * 78)
    print("  ARM COMPARISON ACROSS FOLDS (temporal-stability check)")
    print("=" * 78)
    print(arm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if arm_compare_df is not None and len(arm_compare_df.columns) > 4:
        print("\n" + "=" * 78)
        print("  PER-FOLD ARM COMPARISON (is instability shared or arm-specific?)")
        print("=" * 78)
        print(arm_compare_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 78)
    print("  CAVEAT: walk-forward folds assign validation as one entire calendar")
    print("  snapshot (build_fold_instances), NOT the customer-disjoint carve-out")
    print("  used by the single-split/--final path (VAL_SPLIT_MODE='customer').")
    print("  Folds with very few training snapshots are especially sensitive to")
    print("  which single month got used for calibration/early-stopping — this")
    print("  harness has not been updated to match the current split design.")
    top = arm_df.iloc[0]
    consistent = top["fold_win_rate"] >= 0.6
    print(
        f"  Verdict: '{top['arm']}' has the best mean pooled AP "
        f"({top['mean_AP']:.4f}) and wins {top['fold_win_rate']:.0%} of folds "
        f"— {'consistent with' if consistent else 'WEAKER than expected from'} "
        f"the single-snapshot Run-6 result."
    )
    if arm_df["std_AP"].max() > 0.03:
        print("  NOTE: some arm has AP std > 0.03 across folds — check the "
          "per-fold table above for a specific bad period before trusting the mean.")
    print("-" * 78 + "\n")


def plot_stability(fold_results: list[dict], deployed_df: pd.DataFrame, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    all_arms = sorted({name for r in fold_results for name in r.get("arms", {})})
    plt.figure(figsize=(10, 6))
    for arm in all_arms:
        xs, ys = [], []
        for r in sorted(fold_results, key=lambda r: r["fold_id"]):
            if arm in r.get("arms", {}):
                xs.append(r["fold_id"])
                ys.append(_pooled_ap(r["arms"][arm]))
        plt.plot(xs, ys, marker="o", label=arm)
    plt.xlabel("Fold ID (chronological test snapshot)")
    plt.ylabel("Pooled ranking AP")
    plt.title("Arm ranking AP across walk-forward folds")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved stability plot -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Run directory or fold_results.pkl path")
    parser.add_argument("--fallback", action="store_true",
                        help="Reconstruct from per-fold arms_metrics.json instead of the run-level pickle")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where to write CSVs/plot (default: alongside the input)")
    parser.add_argument("--min_train_snaps", type=int, default=3,
                        help="Also print a verdict restricted to folds with at least this "
                            "many training snapshots (default: 3). Early walk-forward folds "
                            "trained on 1-2 snapshots are a thinner-than-production regime "
                            "and can skew a naive mean across all folds — see the printed "
                            "comparison. Pass 0 to disable this second verdict.")
    args = parser.parse_args()

    run_dir = args.path if args.path.is_dir() else args.path.parent.parent
    out_dir = args.output_dir or (run_dir if run_dir.exists() else Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fallback:
        fold_results = load_fold_results_fallback(run_dir)
    else:
        try:
            fold_results = load_fold_results(args.path)
        except FileNotFoundError as e:
            print(f"{e}\nRetrying with --fallback logic automatically...")
            fold_results = load_fold_results_fallback(run_dir)

    if not fold_results:
        sys.exit("No completed folds found.")

    deployed_df = deployed_arm_table(fold_results)
    arm_df = per_arm_stability_table(fold_results)
    arm_compare_df = per_fold_arm_comparison(fold_results)
    print_conclusion(deployed_df, arm_df, arm_compare_df)

    deployed_df.to_csv(out_dir / "wf_deployed_arm_per_fold.csv", index=False)
    arm_df.to_csv(out_dir / "wf_arm_stability.csv", index=False)
    arm_compare_df.to_csv(out_dir / "wf_per_fold_arm_comparison.csv", index=False)
    plot_stability(fold_results, deployed_df, out_dir / "wf_arm_stability.png")
    print(f"\nCSVs + plot written to {out_dir.resolve()}")

    if args.min_train_snaps > 0:
        print("\n" + "#" * 78)
        print(f"  VERDICT RESTRICTED TO FOLDS WITH >= {args.min_train_snaps} TRAINING SNAPSHOTS")
        print("  (the naive all-fold mean above is skewed by thin early folds that")
        print("   are not representative of production, which trains on ALL mature")
        print("   snapshots — this is the number that actually matters for shipping)")
        print("#" * 78)
        mature = filter_by_min_train_snaps(fold_results, args.min_train_snaps)
        if mature:
            mature_deployed_df = deployed_arm_table(mature)
            mature_arm_df = per_arm_stability_table(mature)
            print(mature_deployed_df[[c for c in mature_deployed_df.columns if c != "train_snaps"]]
                  .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print()
            print(mature_arm_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            mature_deployed_df.to_csv(out_dir / "wf_mature_folds_deployed_arm.csv", index=False)
            mature_arm_df.to_csv(out_dir / "wf_mature_folds_arm_stability.csv", index=False)
        else:
            print(f"  No folds have >= {args.min_train_snaps} training snapshots.")

    # Also try the project's own aggregator/logger for a second, independently
    # -derived cross-check of the deployed-arm summary.
    try:
        aggregated = aggregate_fold_metrics(fold_results)
        log_fold_summary(aggregated)
    except Exception as e:
        print(f"(fold_aggregator cross-check skipped: {e})")


if __name__ == "__main__":
    main()
