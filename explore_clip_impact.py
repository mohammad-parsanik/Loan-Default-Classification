"""
explore_clip_impact.py
======================
Does OutlierClipper's [p1, p99] clip destroy signal?

Clipping is the one preprocessing step XGBoost cannot shrug off. Scaling and
log transforms are monotone, and trees split on order — but clipping merges
every value above p99 into a single number, so any ordering up there is gone.
That is fine when the tail is noise and harmful when the tail is the risk.

Reports, per clipped feature:
  pct_hi        share of loan-rows pinned to p99 (~1% by construction)
  n_merged      distinct values above p99 collapsed into one
  tail_span     (max - p99) / (p99 - p1) — how much range is discarded
  tail_lift     P(severe | x > p99) / P(severe) — >1 means the tail carries
                signal the clip is throwing away. This is the number to read.

With --baseline, also diffs p99 against an older cache, which is how you see
a population change (e.g. a widened contract-amount filter) move the bounds.
Copy the old cache aside BEFORE rebuilding — the path is reused.

Inputs (no DB): data/train_portfolios_cache.npz + .manifest.json
Usage:
  python explore_clip_impact.py
  python explore_clip_impact.py --baseline data/cache_700m.npz
  python explore_clip_impact.py --only REMAINING_AMNT,UPCOMING_AMNT,PAYED_OVERDUE_AMNT
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import project_config as config
from src.data.column_contract import NO_CLIP
from src.data.temporal_split import filter_mature_snapshots, register_label_horizons

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEVERE_CLASS = config.NUM_CLASSES - 1


def load_loan_rows(cache_path: Path):
    """
    Returns (features_flat, severe_flag_per_row, feat_cols), mature rows only.

    Labels are per instance; features_flat is per loan. At loan grain that is
    1:1, at portfolio grain a customer's label repeats across their loans —
    np.repeat over the offsets covers both.
    """
    manifest_path = cache_path.with_suffix(".manifest.json")
    if not cache_path.exists():
        log.error(f"Cache not found: {cache_path} — run `python run.py train` first.")
        sys.exit(1)

    with np.load(cache_path, allow_pickle=True) as npz:
        features_flat  = npz["features_flat"]
        offsets        = npz["offsets"]
        labels         = npz["labels"]
        snapshot_dates = npz["snapshot_dates"]

    with open(manifest_path) as f:
        manifest = json.load(f)
    feat_cols = manifest["feature_cols"]
    register_label_horizons(manifest.get("label_horizons", {}))

    sizes = np.diff(offsets)
    mature = set(filter_mature_snapshots(np.unique(snapshot_dates)))
    keep_inst = np.array([s in mature for s in snapshot_dates])
    if not keep_inst.all():
        log.warning(f"Dropping {(~keep_inst).sum():,} instance(s) from immature snapshots.")

    keep_rows = np.repeat(keep_inst, sizes)
    severe = np.repeat((labels == SEVERE_CLASS), sizes)[keep_rows]
    log.info(f"{cache_path.name}: {keep_rows.sum():,} loan-rows, "
             f"{len(feat_cols)} features, severe base rate {severe.mean():.4%}")
    return features_flat[keep_rows], severe, feat_cols


def clip_report(flat, severe, feat_cols, only=None) -> pd.DataFrame:
    base_rate = float(severe.mean())
    binary = set(config.BINARY_FEATURES)
    rows = []
    for i, col in enumerate(feat_cols):
        if col in binary or col in NO_CLIP:
            continue                       # clipper skips these; so do we
        if only and col not in only:
            continue
        x = flat[:, i].astype(np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        # mirror OutlierClipper.fit: ratios are clipped to [0, 1], not percentiles
        if "RATIO" in col:
            p1, p99 = 0.0, 1.0
        else:
            p1, p99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
        above = flat[:, i] > p99
        n_above = int(above.sum())
        spread = p99 - p1
        rows.append({
            "feature":   col,
            "p1":        p1,
            "p99":       p99,
            "max":       float(x.max()),
            "pct_hi":    n_above / len(x),
            "pct_lo":    float((flat[:, i] < p1).mean()),
            "n_merged":  int(len(np.unique(x[x > p99]))),
            "tail_span": float((x.max() - p99) / spread) if spread > 0 else np.nan,
            "tail_lift": float(severe[above].mean() / base_rate)
                         if n_above and base_rate > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("tail_lift", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(config.DATA_DIR / "train_portfolios_cache.npz"))
    ap.add_argument("--baseline", default=None,
                    help="older cache to diff p99 against (population-shift check)")
    ap.add_argument("--only", default=None, help="comma-separated feature subset")
    ap.add_argument("--output", default="explore_output/clip_impact.csv")
    ap.add_argument("--lift_threshold", type=float, default=1.5,
                    help="tail_lift above which a column is worth exempting")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    flat, severe, feat_cols = load_loan_rows(Path(args.cache))
    rep = clip_report(flat, severe, feat_cols, only)

    if args.baseline:
        b_flat, b_severe, b_cols = load_loan_rows(Path(args.baseline))
        base = clip_report(b_flat, b_severe, b_cols, only)[["feature", "p99", "max"]]
        rep = rep.merge(base.rename(columns={"p99": "p99_before", "max": "max_before"}),
                        on="feature", how="left")
        rep["p99_ratio"] = rep["p99"] / rep["p99_before"].replace(0, np.nan)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(rep.to_string(index=False))

    flagged = rep[rep["tail_lift"] > args.lift_threshold]
    if len(flagged):
        log.warning(
            f"{len(flagged)} feature(s) with tail_lift > {args.lift_threshold}: "
            f"{', '.join(flagged['feature'])}\n"
            "  Severe events concentrate above p99 — clipping merges that tail "
            "into one value. Consider clip:false in contract/columns.json, then "
            "A/B it on validation lift@K before committing."
        )
    else:
        log.info(f"No feature exceeds tail_lift {args.lift_threshold} — clip bounds look benign.")
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
