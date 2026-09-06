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
  head_lift     the same for x < p1, the end nothing used to measure.

Bounds (p1/p99) and everything derived from them — pct_hi/pct_lo, n_merged,
tail_span — always come from the full mature population, because that is the
population OutlierClipper fits on (run.py:262 fits the preprocessor with no
carve). --ranked_only narrows the LIFT only: which rows the severe rate is
conditioned on, never which rows set the clip. A NaN lift under --ranked_only
means no queued loan reaches that bound at all, so the clip cannot move the
ranking.

With --baseline, also diffs p99 against an older cache, which is how you see
a population change (e.g. a widened contract-amount filter) move the bounds.
Copy the old cache aside BEFORE rebuilding — the path is reused.

Inputs (no DB): data/snapshots/train_<key>/  (per-snapshot NPZ cache)
Usage:
  python explore_clip_impact.py --ranked_only
  python explore_clip_impact.py --baseline data/cache_700m.npz
  python explore_clip_impact.py --only REMAINING_AMNT,UPCOMING_AMNT,PAYED_OVERDUE_AMNT
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import project_config as config
from src.data.column_contract import NO_CLIP
from src.data.data_loader import load_cached_arrays
from src.data.temporal_split import filter_mature_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEVERE_CLASS = config.NUM_CLASSES - 1


def load_loan_rows(cache_dir=None):
    """
    Returns (features_flat, severe_flag, rankable_flag, feat_cols) per loan-row,
    mature rows only.

    Carved rows are kept and flagged rather than dropped: OutlierClipper fits on
    the whole training split (run.py:262, no carve), so the percentiles have to
    come from the same population or the report describes a clip the pipeline
    never performs. Only the lift conditions on `rankable`.

    Labels and current_cat are per instance; features_flat is per loan. At loan
    grain that is 1:1, at portfolio grain a customer's values repeat across
    their loans -- np.repeat over the offsets covers both.
    """
    try:
        arrays, feat_cols = load_cached_arrays(cache_dir)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    features_flat  = arrays["features_flat"]
    offsets        = arrays["offsets"]
    labels         = arrays["labels"]
    snapshot_dates = arrays["snapshot_dates"]
    current_cats   = arrays["current_cats"]

    sizes = np.diff(offsets)
    mature = set(filter_mature_snapshots(np.unique(snapshot_dates)))
    keep_inst = np.array([s in mature for s in snapshot_dates])
    if not keep_inst.all():
        log.warning(f"Dropping {(~keep_inst).sum():,} instance(s) from immature snapshots.")

    keep_rows = np.repeat(keep_inst, sizes)
    severe   = np.repeat((labels == SEVERE_CLASS), sizes)[keep_rows]
    rankable = np.repeat((current_cats < config.CARVE_CURRENT_CAT_GE), sizes)[keep_rows]
    log.info(f"{keep_rows.sum():,} loan-rows, "
             f"{len(feat_cols)} features, severe base rate {severe.mean():.4%}")
    return features_flat[keep_rows], severe, rankable, feat_cols


def _lift(severe, mask, base_rate):
    """P(severe | mask) / P(severe). NaN when the slice is empty -- which is the
    informative answer for a DPD column under --ranked_only: no queued loan can
    reach p99, so the clip cannot touch the ranking at all."""
    if not mask.any() or base_rate <= 0:
        return np.nan
    return float(severe[mask].mean() / base_rate)


def clip_report(flat, severe, feat_cols, only=None, lift_mask=None) -> pd.DataFrame:
    """Bounds and merge counts describe the FULL population the clipper fits on;
    `lift_mask` (--ranked_only) narrows only the rows the severe rate is
    conditioned on."""
    if lift_mask is None:
        lift_mask = np.ones(len(flat), dtype=bool)
    base_rate = float(severe[lift_mask].mean()) if lift_mask.any() else 0.0
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
        below = flat[:, i] < p1
        n_above, n_below = int(above.sum()), int(below.sum())
        spread = p99 - p1
        rows.append({
            "feature":   col,
            "p1":        p1,
            "p99":       p99,
            "max":       float(x.max()),
            "pct_hi":    n_above / len(x),
            "pct_lo":    n_below / len(x),
            "n_merged":  int(len(np.unique(x[x > p99]))),
            "tail_span": float((x.max() - p99) / spread) if spread > 0 else np.nan,
            "tail_lift": _lift(severe, above & lift_mask, base_rate),
            # The p1 side. Symmetric question, opposite end: a head_lift well
            # BELOW 1 is fine (the clean end really is clean), but one near or
            # above 1 means the clip is folding risk-bearing rows into the
            # bottom bin.
            "head_lift": _lift(severe, below & lift_mask, base_rate),
        })
    return pd.DataFrame(rows).sort_values("tail_lift", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="per-snapshot cache dir (default: the current schema's)")
    ap.add_argument("--baseline", default=None,
                    help="older cache to diff p99 against (population-shift check)")
    ap.add_argument("--only", default=None, help="comma-separated feature subset")
    ap.add_argument("--ranked_only", action="store_true",
                    help="condition tail_lift/head_lift on current_cat < CARVE_CURRENT_CAT_GE — "
                         "the population the queue actually ranks. Without it the DPD family "
                         "pins at the arithmetic ceiling 1/base_rate and measures the label "
                         "identity. Bounds and merge counts stay on the full population either "
                         "way, because that is what OutlierClipper fits on.")
    ap.add_argument("--output", default="explore_output/clip_impact.csv")
    ap.add_argument("--lift_threshold", type=float, default=1.5,
                    help="tail_lift above which a column is worth exempting")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    flat, severe, rankable, feat_cols = load_loan_rows(
        Path(args.cache) if args.cache else None)
    lift_mask = rankable if args.ranked_only else None
    if args.ranked_only:
        log.info(f"--ranked_only: scoring lift on {int(rankable.sum()):,} of "
                 f"{len(rankable):,} loan-rows (current_cat < "
                 f"{config.CARVE_CURRENT_CAT_GE}); base rate "
                 f"{severe[rankable].mean():.4%}. Bounds still from all rows.")
    rep = clip_report(flat, severe, feat_cols, only, lift_mask)

    if not args.ranked_only:
        log.warning(
            "Scoring lift over ALL mature rows. Rows with current_cat >= "
            f"{config.CARVE_CURRENT_CAT_GE} are severe by the label identity and never "
            "enter the queue, so tail_lift on any delinquency-ranking column is measuring "
            "that identity, not signal. Re-run with --ranked_only to read it."
        )

    if args.baseline:
        b_flat, b_severe, b_rankable, b_cols = load_loan_rows(Path(args.baseline))
        base = clip_report(b_flat, b_severe, b_cols, only,
                           b_rankable if args.ranked_only else None
                           )[["feature", "p99", "max"]]
        rep = rep.merge(base.rename(columns={"p99": "p99_before", "max": "max_before"}),
                        on="feature", how="left")
        rep["p99_ratio"] = rep["p99"] / rep["p99_before"].replace(0, np.nan)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(rep.to_string(index=False))

    degenerate = rep[rep["p1"] >= rep["p99"]]
    if len(degenerate):
        log.error(
            f"{len(degenerate)} feature(s) with p1 >= p99: "
            f"{', '.join(degenerate['feature'])}\n"
            "  Clipping to [p1, p99] makes these CONSTANT — the column carries no\n"
            "  information into any arm. Set clip:false in contract/columns.json."
        )

    headed = rep[rep["head_lift"] > args.lift_threshold]
    if len(headed):
        log.warning(
            f"{len(headed)} feature(s) with head_lift > {args.lift_threshold}: "
            f"{', '.join(headed['feature'])}\n"
            "  Severe events concentrate BELOW p1, so the low clip is folding\n"
            "  risk-bearing rows into the bottom bin."
        )

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
