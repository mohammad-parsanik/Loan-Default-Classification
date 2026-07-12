"""
Fold metric aggregation for Walk-Forward Validation.

Summarises the DEPLOYED arm's performance across folds — the headline is
the ranking block (pooled PR-AUC of P(severe) + recall at each API budget
window), since that is the deliverable. Macro F1 is kept as a secondary
classification check. Walk-forward grades the *recipe* (mean ± std across
time); the shipped model is a separate `train --final` fit on all data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np

import project_config as config

logger = logging.getLogger(__name__)


def _fold_ranking_row(fm: dict) -> dict:
    """Flatten a fold's deployed-arm metrics into scalar columns."""
    rk = fm.get("ranking", {})
    row = {
        "ranking_ap": float(rk.get("pr_auc", float("nan"))),
        "macro_f1":   float(fm.get("macro_f1", float("nan"))),
    }
    for w in config.RANKING_REF_WINDOWS:
        block = rk.get(f"at_{w}", {})
        row[f"recall_{w}"] = float(block.get("recall", float("nan")))
    return row


def _stats(values: list) -> dict:
    arr = np.array([v for v in values if v == v], dtype=float)   # drop nan
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "n": int(len(arr))}


def aggregate_fold_metrics(fold_results: List[Dict]) -> Dict:
    """
    Aggregate per-fold deployed-arm metrics into summary statistics.

    Each fold_result needs "fold_id", "test_snap", "deployed_arm" and
    "final_metrics" (the deployed arm's full_evaluation dict).
    """
    if not fold_results:
        return {}

    metric_cols = ["ranking_ap", "macro_f1"] + [f"recall_{w}" for w in config.RANKING_REF_WINDOWS]
    per_fold_rows = []
    for r in fold_results:
        row = {
            "fold_id":      r["fold_id"],
            "test_snap":    r.get("test_snap"),
            "deployed_arm": r.get("deployed_arm"),
            **_fold_ranking_row(r.get("final_metrics", {})),
        }
        per_fold_rows.append(row)

    agg = {m: _stats([row[m] for row in per_fold_rows]) for m in metric_cols}
    return {"n_folds": len(fold_results), "folds": per_fold_rows, "aggregate": agg}


def log_fold_summary(aggregated: Dict) -> None:
    if not aggregated:
        logger.warning("No fold results to summarise.")
        return

    n = aggregated["n_folds"]
    week_key = "recall_1_week" if "recall_1_week" in aggregated["aggregate"] else None
    logger.info("=" * 70)
    logger.info(f"  Walk-Forward Summary  ({n} fold{'s' if n != 1 else ''}) — deployed arm, ranking headline")
    logger.info("=" * 70)
    logger.info(f"  {'Fold':>4}  {'Test Snap':>12}  {'Arm':>10}  {'Ranking AP':>10}  {'R@1week':>8}")
    logger.info("  " + "-" * 54)
    for row in aggregated["folds"]:
        logger.info(
            f"  {row['fold_id']:>4}  {str(row['test_snap']):>12}  "
            f"{str(row['deployed_arm']):>10}  {row['ranking_ap']:>10.4f}  "
            f"{row.get('recall_1_week', float('nan')):>8.4f}"
        )

    logger.info("  " + "-" * 54)
    logger.info("  Across folds (mean ± std [min – max]):")
    for metric, s in aggregated["aggregate"].items():
        if s["n"]:
            logger.info(
                f"    {metric:<16}: {s['mean']:.4f} ± {s['std']:.4f}  "
                f"[{s['min']:.4f} – {s['max']:.4f}]"
            )
    logger.info("=" * 70)


def save_fold_summary(aggregated: Dict, save_path: Path) -> None:
    def _safe(obj):
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(save_path, "w") as f:
        json.dump(_safe(aggregated), f, indent=2)
    logger.info(f"Walk-forward summary saved → {save_path}")
