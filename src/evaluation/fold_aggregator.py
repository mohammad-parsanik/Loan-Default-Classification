"""
Fold metric aggregation for Walk-Forward Validation.

Collects per-fold results and produces:
  - A fold-by-fold comparison table (logged + saved as JSON)
  - Aggregate statistics: mean, std, min, max across folds
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Metrics tracked across folds
_TRACKED_METRICS = [
    "macro_f1",
    "qwk",
    "accuracy",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "recall_class_3",
    "brier_score",
    "avg_cost",
]

_BASELINE_TRACKED = ["macro_f1", "qwk", "brier_score"]


def aggregate_fold_metrics(
    fold_results: List[Dict],
) -> Dict:
    """
    Aggregate per-fold metrics into summary statistics.

    Parameters
    ----------
    fold_results : list[dict]
        Each element must contain:
          - "fold_id"          : int
          - "train_snaps"      : list
          - "val_snap"         : comparable
          - "test_snap"        : comparable
          - "final_metrics"    : dict  (DeepSets+XGBoost metrics on test)
          - "baseline_metrics" : dict  (Aggregated XGBoost baseline metrics)

    Returns
    -------
    dict with keys:
      "folds"     : list[dict]  — per-fold summary rows
      "model"     : dict        — aggregate stats for DeepSets+XGBoost
      "baseline"  : dict        — aggregate stats for baseline
    """
    if not fold_results:
        return {}

    per_fold_rows = []
    for r in fold_results:
        row = {
            "fold_id":     r["fold_id"],
            "train_snaps": sorted(r["train_snaps"]),
            "val_snap":    r["val_snap"],
            "test_snap":   r["test_snap"],
        }
        fm = r.get("final_metrics", {})
        bm = r.get("baseline_metrics", {})
        for m in _TRACKED_METRICS:
            row[f"model_{m}"]    = float(fm.get(m, float("nan")))
            row[f"baseline_{m}"] = float(bm.get(m, float("nan")))
        per_fold_rows.append(row)

    def _stats(values: list) -> dict:
        arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
        if len(arr) == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "min":  float(np.min(arr)),
            "max":  float(np.max(arr)),
            "n":    int(len(arr)),
        }

    model_agg    = {m: _stats([r[f"model_{m}"]    for r in per_fold_rows]) for m in _TRACKED_METRICS}
    baseline_agg = {m: _stats([r[f"baseline_{m}"] for r in per_fold_rows]) for m in _TRACKED_METRICS}

    return {
        "n_folds":  len(fold_results),
        "folds":    per_fold_rows,
        "model":    model_agg,
        "baseline": baseline_agg,
    }


def log_fold_summary(aggregated: Dict) -> None:
    """
    Pretty-print a fold-by-fold and aggregate summary table to the logger.
    """
    if not aggregated:
        logger.warning("No fold results to summarise.")
        return

    n = aggregated["n_folds"]
    logger.info("=" * 70)
    logger.info(f"  Walk-Forward Summary  ({n} fold{'s' if n != 1 else ''})")
    logger.info("=" * 70)

    # Per-fold rows
    logger.info(f"  {'Fold':>4}  {'Test Snap':>12}  {'Baseline F1':>11}  {'Model F1':>8}  {'Delta':>7}")
    logger.info("  " + "-" * 52)
    for row in aggregated["folds"]:
        b_f1 = row.get("baseline_macro_f1", float("nan"))
        m_f1 = row.get("model_macro_f1",    float("nan"))
        delta = m_f1 - b_f1 if not (np.isnan(b_f1) or np.isnan(m_f1)) else float("nan")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   N/A"
        logger.info(
            f"  {row['fold_id']:>4}  {str(row['test_snap']):>12}  "
            f"{b_f1:>11.4f}  {m_f1:>8.4f}  {delta_str:>7}"
        )

    # Aggregate stats
    logger.info("  " + "-" * 52)
    m_stats = aggregated["model"]["macro_f1"]
    b_stats = aggregated["baseline"]["macro_f1"]
    logger.info(f"  {'Mean':>4}  {'':>12}  {b_stats['mean']:>11.4f}  {m_stats['mean']:>8.4f}")
    logger.info(f"  {'Std':>4}  {'':>12}  {b_stats['std']:>11.4f}  {m_stats['std']:>8.4f}")
    logger.info(f"  {'Min':>4}  {'':>12}  {b_stats['min']:>11.4f}  {m_stats['min']:>8.4f}")
    logger.info(f"  {'Max':>4}  {'':>12}  {b_stats['max']:>11.4f}  {m_stats['max']:>8.4f}")

    # Key metrics summary
    logger.info("  " + "-" * 52)
    logger.info("  Key metrics (mean ± std across folds):")
    for metric in ["macro_f1", "qwk", "recall_class_2", "recall_class_3", "brier_score"]:
        ms = aggregated["model"].get(metric, {})
        if ms:
            logger.info(
                f"    {metric:<20}: {ms['mean']:.4f} ± {ms['std']:.4f}  "
                f"[{ms['min']:.4f} – {ms['max']:.4f}]"
            )
    logger.info("=" * 70)


def save_fold_summary(
    aggregated: Dict,
    save_path: Path,
) -> None:
    """
    Serialise the aggregated fold results to a JSON file.

    Parameters
    ----------
    aggregated : dict
        Output from aggregate_fold_metrics().
    save_path : Path
        Destination file (e.g. run_dir / "walk_forward_summary.json").
    """
    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(save_path, "w") as f:
        json.dump(_safe(aggregated), f, indent=2)
    logger.info(f"Walk-forward summary saved → {save_path}")
