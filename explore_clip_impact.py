"""
explore_clip_impact.py
======================
Does OutlierClipper's [p1, p99] clip destroy signal?

Clipping is the one preprocessing step XGBoost cannot shrug off. Scaling and
log transforms are monotone, and trees split on order — but clipping merges
every value above p99 into a single number, so any ordering up there is gone.
That is fine when the tail is noise and harmful when the tail is the risk.

Reports, per clipped feature:
  pct_null      share of rows NaN before imputation — see "Imputation" below
  pct_hi        share of loan-rows pinned to p99 (~1% by construction)
  pct_tied_hi   share sitting EXACTLY on p99 already
  n_merged      distinct values above p99 collapsed into one
  tail_span     (max - p99) / (p99 - p1) — how much range is discarded
  tail_lift     P(severe | x > p99) / P(severe) — the tail as a BLOCK
  tail_gradient P(severe | outer quartile of the tail) / P(severe | inner
                quartile). THIS is the number that says whether the clip
                costs anything. See "What clipping actually destroys".
  head_lift / head_gradient / pct_tied_lo   the same, for the p1 side.

With --by_current_cat, also tail_lift_cat{k} / head_lift_cat{k}: the same lift
computed INSIDE each current_cat stratum, against that stratum's own base rate.

What clipping actually destroys
-------------------------------
Not the block. After `np.clip` a split just inside the bound still isolates the
merged rows, so the model keeps "this row was beyond the bound". What it loses
is the ORDERING inside that region. So a high `tail_lift` OPENS the question and
`tail_gradient` answers it: ~1 means the region is internally flat, the clip
merges rows the model had no reason to separate, and it costs nothing.

`tail_lift` is also composition-prone, and has produced a wrong answer twice.
Over all mature rows the DPD family pins at the arithmetic ceiling 1/base_rate,
measuring the `label >= current_cat` identity — that is what --ranked_only is
for. But --ranked_only only removes cat_3: a tail made entirely of cat_2 rows
still reports cat2_rate/base_rate with zero incremental signal, which is
~11.7 at the observed rates and is exactly what Run 8 reported. Read
tail_lift_cat0 (--by_current_cat), not the pooled number.

Imputation
----------
The pipeline runs impute -> clip -> scale, so the clipper's percentiles are
taken AFTER NaNs became fill values. This script mirrors that, using
DomainAwareImputer's own fitted fill values rather than a second copy of its
rules. Dropping NaN instead — as this script used to — moves p1 on every
nullable column: `DomainAwareImputer` fills most columns with 0.0, so a column
with a >1% null rate has a real p1 of 0.0 and no head clip at all.

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
  python explore_clip_impact.py --ranked_only --by_current_cat
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
from src.data.column_contract import CLIP_BOUNDS, NO_CLIP
from src.data.data_loader import load_cached_arrays
from src.data.preprocessing import DomainAwareImputer
from src.data.temporal_split import filter_mature_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEVERE_CLASS = config.NUM_CLASSES - 1


def load_loan_rows(cache_dir=None):
    """
    Returns (features_flat, severe_flag, current_cat, null_frac, feat_cols) per
    loan-row, mature rows only, IMPUTED the way the pipeline imputes.

    Carved rows are kept and flagged rather than dropped: OutlierClipper fits on
    the whole training split (run.py:262, no carve), so the percentiles have to
    come from the same population or the report describes a clip the pipeline
    never performs. Only the lift conditions on `rankable`.

    Labels and current_cat are per instance; features_flat is per loan. At loan
    grain that is 1:1, at portfolio grain a customer's values repeat across
    their loans -- np.repeat over the offsets covers both.

    `null_frac` is measured BEFORE imputing, because it is the number that says
    whether a column's reported p1 was ever the clipper's p1.
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
    severe = np.repeat((labels == SEVERE_CLASS), sizes)[keep_rows]
    cats   = np.repeat(current_cats, sizes)[keep_rows]

    flat = features_flat[keep_rows]
    null_frac = np.isnan(flat).mean(axis=0)

    # Mirror the pipeline: impute BEFORE taking percentiles (run.py:262 fits
    # impute -> clip -> scale in that order), so the bounds reported here are
    # the bounds OutlierClipper fits. The imputer is fitted rather than
    # reimplemented -- the fill rules live in one place -- but its fill values
    # are then applied in place, because transform() would vstack a second
    # multi-GB copy of a matrix we already hold.
    imputer = DomainAwareImputer(feat_cols).fit([flat])
    for i, fill in imputer.fill_values_.items():
        nan = np.isnan(flat[:, i])
        if nan.any():
            flat[nan, i] = fill

    filled = int((null_frac > 0).sum())
    log.info(f"{keep_rows.sum():,} loan-rows, {len(feat_cols)} features, "
             f"severe base rate {severe.mean():.4%}; imputed {filled} column(s) "
             "before measuring, as the pipeline does.")
    return flat, severe, cats, null_frac, feat_cols


def _lift(severe, mask, base_rate):
    """P(severe | mask) / P(severe). NaN when the slice is empty -- which is the
    informative answer for a DPD column under --ranked_only: no queued loan can
    reach p99, so the clip cannot touch the ranking at all."""
    if not mask.any() or base_rate <= 0:
        return np.nan
    return float(severe[mask].mean() / base_rate)


#: Below this many rows beyond a bound, a quartile split is noise, not a shape.
_MIN_TAIL_ROWS = 200


def _gradient(severe, x, mask, outer_is_high: bool) -> float:
    """
    Does risk VARY across the clipped region, or is it flat?

    This is the number `tail_lift` cannot give you. Clipping does not destroy
    the block: after np.clip a split just inside the bound still isolates the
    merged rows, so "this row was beyond the bound" survives. What dies is the
    ORDERING inside the region. So the question that decides whether the clip
    costs anything is whether risk varies across it.

    Returns P(severe | outer quartile of the region) / P(severe | inner
    quartile). ~1 means internally flat -- the clip merges rows the model had no
    reason to separate, and costs nothing. Well above 1 means it is flattening a
    real ramp. NaN when the region is too small to split, or is a single value
    (an integer column with one value out there has no ordering to lose).
    """
    idx = np.flatnonzero(mask)
    if len(idx) < _MIN_TAIL_ROWS:
        return np.nan
    v = x[idx]
    lo_cut, hi_cut = np.quantile(v, [0.25, 0.75])
    if lo_cut >= hi_cut:
        return np.nan
    far, near = ((v >= hi_cut), (v <= lo_cut)) if outer_is_high else \
                ((v <= lo_cut), (v >= hi_cut))
    rate_near = float(severe[idx[near]].mean())
    if rate_near <= 0:
        return np.nan
    return float(severe[idx[far]].mean() / rate_near)


def clip_report(flat, severe, feat_cols, only=None, lift_mask=None,
                strata=None, null_frac=None) -> pd.DataFrame:
    """Bounds and merge counts describe the FULL population the clipper fits on;
    `lift_mask` (--ranked_only) narrows only the rows the severe rate is
    conditioned on. `strata` (--by_current_cat) adds the same lifts computed
    inside each queued stratum against that stratum's own base rate, which is
    what separates signal from composition."""
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
        # Imputed upstream, so this only drops +/-inf. len(x) == len(flat) in
        # the normal case, which is what makes pct_hi and the masks below share
        # a denominator -- they did not when NaNs were dropped here.
        x = flat[:, i].astype(np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        # mirror OutlierClipper.fit: a declared range beats the sample
        if col in CLIP_BOUNDS:
            p1, p99 = CLIP_BOUNDS[col]
        else:
            p1, p99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
        col_x = flat[:, i]
        above = col_x > p99
        below = col_x < p1
        n_above, n_below = int(above.sum()), int(below.sum())
        spread = p99 - p1
        row = {
            "feature":   col,
            "p1":        p1,
            "p99":       p99,
            "max":       float(x.max()),
            "pct_null":  float(null_frac[i]) if null_frac is not None else np.nan,
            "pct_hi":    n_above / len(x),
            "pct_lo":    n_below / len(x),
            # Rows already sitting ON the bound. The merged rows become
            # indistinguishable from these, so a large share here means the
            # clip costs more than ordering -- it costs the split itself.
            "pct_tied_hi": float((col_x == p99).mean()),
            "pct_tied_lo": float((col_x == p1).mean()),
            "n_merged":  int(len(np.unique(x[x > p99]))),
            # max(0) because a DECLARED bound can sit above the data (nothing
            # is discarded, so the discarded range is zero, not negative).
            "tail_span": (max(0.0, float((x.max() - p99) / spread))
                          if spread > 0 else np.nan),
            # The block. Opens the question; does not answer it.
            "tail_lift": _lift(severe, above & lift_mask, base_rate),
            # The shape inside the block. Answers it.
            "tail_gradient": _gradient(severe, col_x, above & lift_mask, True),
            # The p1 side. Symmetric question, opposite end: a head_lift well
            # BELOW 1 is fine (the clean end really is clean), but one near or
            # above 1 means the clip is folding risk-bearing rows into the
            # bottom bin.
            "head_lift": _lift(severe, below & lift_mask, base_rate),
            "head_gradient": _gradient(severe, col_x, below & lift_mask, False),
        }
        if strata is not None:
            # Within one current_cat, against that cat's own base rate. A tail
            # that is risky only because it is full of cat_2 rows scores ~1
            # here and is telling you about current_cat, which the model
            # already has as a feature.
            for k in np.unique(strata[strata < config.CARVE_CURRENT_CAT_GE]):
                in_cat = strata == k
                cat_base = float(severe[in_cat].mean()) if in_cat.any() else 0.0
                row[f"tail_lift_cat{k}"] = _lift(severe, above & in_cat, cat_base)
                row[f"head_lift_cat{k}"] = _lift(severe, below & in_cat, cat_base)
        rows.append(row)
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
    ap.add_argument("--by_current_cat", action="store_true",
                    help="also report tail_lift/head_lift INSIDE each queued "
                         "current_cat stratum, against that stratum's own base "
                         "rate. --ranked_only removes cat_3 but not cat_2, so a "
                         "tail made of cat_2 rows still reports a large pooled "
                         "lift with zero incremental signal. Read cat0.")
    ap.add_argument("--output", default="explore_output/clip_impact.csv")
    ap.add_argument("--lift_threshold", type=float, default=1.5,
                    help="tail_lift above which a column is worth a second look")
    ap.add_argument("--gradient_threshold", type=float, default=1.5,
                    help="tail_gradient above which the clip is flattening a "
                         "real ramp rather than merging equivalent rows")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    flat, severe, cats, null_frac, feat_cols = load_loan_rows(
        Path(args.cache) if args.cache else None)
    rankable = cats < config.CARVE_CURRENT_CAT_GE
    lift_mask = rankable if args.ranked_only else None
    if args.ranked_only:
        log.info(f"--ranked_only: scoring lift on {int(rankable.sum()):,} of "
                 f"{len(rankable):,} loan-rows (current_cat < "
                 f"{config.CARVE_CURRENT_CAT_GE}); base rate "
                 f"{severe[rankable].mean():.4%}. Bounds still from all rows.")
    rep = clip_report(flat, severe, feat_cols, only, lift_mask,
                      strata=cats if args.by_current_cat else None,
                      null_frac=null_frac)

    if not args.ranked_only:
        log.warning(
            "Scoring lift over ALL mature rows. Rows with current_cat >= "
            f"{config.CARVE_CURRENT_CAT_GE} are severe by the label identity and never "
            "enter the queue, so tail_lift on any delinquency-ranking column is measuring "
            "that identity, not signal. Re-run with --ranked_only to read it."
        )

    if args.baseline:
        b_flat, b_severe, b_cats, _, b_cols = load_loan_rows(Path(args.baseline))
        base = clip_report(b_flat, b_severe, b_cols, only,
                           (b_cats < config.CARVE_CURRENT_CAT_GE)
                           if args.ranked_only else None
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

    nulled = rep[rep["pct_null"] > 0.002]
    if len(nulled):
        log.info(
            f"{len(nulled)} feature(s) with a null rate above 0.2%: "
            f"{', '.join(nulled['feature'])}\n"
            "  Their fill value (usually 0.0) is part of the distribution the\n"
            "  clipper fits. Any earlier report that dropped NaN before taking\n"
            "  percentiles stated a p1 these columns never had."
        )

    # The block is risky AND has a shape inside it -- the only combination in
    # which clipping actually costs the model something.
    costly = rep[(rep["tail_lift"] > args.lift_threshold)
                 & (rep["tail_gradient"] > args.gradient_threshold)]
    flat_tail = rep[(rep["tail_lift"] > args.lift_threshold)
                    & (rep["tail_gradient"] <= args.gradient_threshold)]

    if len(costly):
        log.warning(
            f"{len(costly)} feature(s) with tail_lift > {args.lift_threshold} AND "
            f"tail_gradient > {args.gradient_threshold}: "
            f"{', '.join(costly['feature'])}\n"
            "  Risk RISES across the tail, so the clip is flattening a real ramp\n"
            "  rather than merging equivalent rows. These are the candidates for\n"
            "  clip:false — A/B on validation lift@K before committing."
        )
    if len(flat_tail):
        log.info(
            f"{len(flat_tail)} feature(s) with a high tail_lift but a FLAT "
            f"tail_gradient: {', '.join(flat_tail['feature'])}\n"
            "  The tail is risky as a block, but risk does not vary across it —\n"
            "  and the block survives clipping, because a split just inside the\n"
            "  bound still isolates it. Leave these clipped."
        )
    if not len(costly):
        log.info("No feature shows a risk ramp inside its tail — the clip merges "
                 "rows the model had no reason to separate.")

    if not args.by_current_cat:
        log.warning(
            "No --by_current_cat. --ranked_only removes cat_3 but not cat_2, so "
            "a tail made of cat_2 rows reports cat2_rate/base_rate with zero "
            "incremental signal. Re-run with --by_current_cat and read "
            "tail_lift_cat0 before acting on any pooled number above."
        )
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
