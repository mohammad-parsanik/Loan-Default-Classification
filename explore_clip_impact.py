"""
explore_clip_impact.py
======================
Does OutlierClipper's [p1, p99] clip destroy signal?

Clipping is the one preprocessing step XGBoost cannot shrug off. Scaling and
log transforms are monotone, and trees split on order — but clipping merges
every value above p99 into a single number, so any ordering up there is gone.
That is fine when the tail is noise and harmful when the tail is the risk.

Reports, per clipped feature (and per exempt one, with --include_exempt):
  pct_null      share of rows NaN before imputation — see "Imputation" below
  n_hi / pct_hi rows pinned to p99 (~1% by construction)
  pct_tied_hi   share sitting EXACTLY on p99 already
  n_merged      distinct values above p99 collapsed into one
  tail_span     (max - p99) / (p99 - p1) — how much range is discarded
  tail_lift     P(severe | x > p99) / P(severe) — the tail as a BLOCK
  tail_gradient P(severe | outer quartile of the tail) / P(severe | inner
                quartile). THIS is the number that says whether the clip
                costs anything. See "What clipping actually destroys".
  head_lift / head_gradient / pct_tied_lo   the same, for the p1 side.
  tail_verdict / head_verdict   what the clip costs at each end — none, tiny,
                inert, block, ramp, swamped. See side_verdict().

With --by_current_cat, also n_*_cat{k}, *_lift_cat{k} and *_gradient_cat{k}:
the same measurements INSIDE each queued current_cat stratum, against that
stratum's own base rate, with the row counts they rest on.

What clipping actually destroys
-------------------------------
Not the block. After `np.clip` a split just inside the bound still isolates the
merged rows, so the model keeps "this row was beyond the bound". What it loses
is the ORDERING inside that region. So a high `tail_lift` OPENS the question and
`tail_gradient` answers it: ~1 means the region is internally flat, the clip
merges rows the model had no reason to separate, and it costs nothing.

Pooled numbers are composition-prone — lift AND gradient. Over all mature rows
the DPD family pins at the arithmetic ceiling 1/base_rate, measuring the
`label >= current_cat` identity; that is what --ranked_only is for. But
--ranked_only only removes cat_3, so a pooled number still mixes cat_0, cat_1
and cat_2 rows with very different base rates. Which way the mix pushes is an
empirical question, and it has been guessed wrong: §25 first read Run 8's
pooled DPD tail_lift of ~11.7 as pure cat_2 composition, and results_9 showed
the opposite — inside cat_0 the same tails run 15-27x, i.e. recently cured
heavy delinquents, a real signal. The same mix can bend a pooled gradient
(an outer quartile that is more cat_0 than the inner one reads as a decline
even when risk rises inside every stratum). So read the per-stratum columns,
and the n_*_cat counts under them, before any pooled number.

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
#: ...and below this many severe events across the two compared quartiles. The
#: noise in a ratio of two rates is set by the event counts, not the rows: at
#: 100 events the log-ratio's standard error is ~0.2, so a gradient of 1.5
#: sits ~2 SE from flat. With ~20 events, pure noise crosses 1.5 routinely —
#: a synthetic smoke test with no signal at all flagged two "ramps" before
#: this floor existed.
_MIN_EVENTS = 100


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
    real ramp. NaN when the region is too small to split, holds too few severe
    events to estimate a shape (_MIN_EVENTS), or is a single value (an integer
    column with one value out there has no ordering to lose).
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
    if int(severe[idx[far]].sum() + severe[idx[near]].sum()) < _MIN_EVENTS:
        return np.nan
    rate_near = float(severe[idx[near]].mean())
    if rate_near <= 0:
        return np.nan
    return float(severe[idx[far]].mean() / rate_near)


def clip_report(flat, severe, feat_cols, only=None, lift_mask=None,
                strata=None, null_frac=None, include_exempt=False) -> pd.DataFrame:
    """Bounds and merge counts describe the FULL population the clipper fits on;
    `lift_mask` (--ranked_only) narrows only the rows the severe rate is
    conditioned on. `strata` (--by_current_cat) adds the same lifts AND
    gradients computed inside each queued stratum against that stratum's own
    base rate, plus the row counts they rest on — pooled numbers mix the
    strata, and both the lift and the gradient can be moved by that mix.

    `include_exempt` also reports the NO_CLIP columns, with the bounds the
    clipper WOULD fit, flagged `exempt=True` — the way to audit an exemption
    with the same measurements that would have justified it."""
    if lift_mask is None:
        lift_mask = np.ones(len(flat), dtype=bool)
    base_rate = float(severe[lift_mask].mean()) if lift_mask.any() else 0.0
    binary = set(config.BINARY_FEATURES)
    rows = []
    for i, col in enumerate(feat_cols):
        if col in binary:
            continue                       # clipper skips these; so do we
        if col in NO_CLIP and not include_exempt:
            continue
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
            "exempt":    col in NO_CLIP,
            "p1":        p1,
            "p99":       p99,
            "max":       float(x.max()),
            "pct_null":  float(null_frac[i]) if null_frac is not None else np.nan,
            "n_hi":      n_above,
            "n_lo":      n_below,
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
            # that is risky only because it is full of high-cat rows scores ~1
            # here. The gradient gets the same treatment, because a pooled
            # gradient is just as exposed to the mix: if the outer quartile of
            # a tail is more cat_0 than its inner quartile, the pooled ratio
            # falls even when risk rises inside every stratum.
            for k in np.unique(strata[strata < config.CARVE_CURRENT_CAT_GE]):
                in_cat = strata == k
                cat_base = float(severe[in_cat].mean()) if in_cat.any() else 0.0
                hi_k, lo_k = above & in_cat, below & in_cat
                row[f"n_hi_cat{k}"] = int(hi_k.sum())
                row[f"tail_lift_cat{k}"] = _lift(severe, hi_k, cat_base)
                row[f"tail_gradient_cat{k}"] = _gradient(severe, col_x, hi_k, True)
                row[f"n_lo_cat{k}"] = int(lo_k.sum())
                row[f"head_lift_cat{k}"] = _lift(severe, lo_k, cat_base)
                row[f"head_gradient_cat{k}"] = _gradient(severe, col_x, lo_k, False)
        rows.append(row)
    if not rows:                           # e.g. --only naming exempt columns
        return pd.DataFrame(columns=["feature", "exempt"])
    return pd.DataFrame(rows).sort_values("tail_lift", ascending=False)


#: The clip "swamps" a block when the rows already sitting on the bound are at
#: least this large a share relative to the rows it merges into them: the
#: merged rows are then at most 2/3 of the post-clip bin, and a split can no
#: longer isolate them. HIST_MAX_DPD_DAYS's 0 -> 1 is the case this names.
SWAMP_RATIO = 0.5


def side_verdict(row, side: str, lift_thr: float, grad_thr: float) -> str:
    """
    What clipping one end of one column costs, from a clip_report row.

      none     nothing lies beyond this bound; the clip is a no-op here
      tiny     fewer than _MIN_TAIL_ROWS rows beyond it — too few to judge,
               and too few for the clip to matter to a queue whose 1-day
               budget is thousands of rows. (TOTAL_DPD_DAYS_LAST_6M has ~40
               rows of NEGATIVE DPD below p1 = 0: a data anomaly, not a head.)
      ramp     risk varies ACROSS the region (pooled or inside some stratum):
               the clip flattens an ordering the model could have used
      swamped  the region is informative as a block, but the rows already on
               the bound outnumber enough of it that the merged block can no
               longer be split off: the clip costs the split itself
      block    informative as a block, flat inside, not swamped: the block
               survives the clip, which costs nothing. Leave it clipped.
      inert    uninformative block, flat inside

    Stratum lifts and gradients only count when their own region holds at
    least _MIN_TAIL_ROWS rows (n_hi_cat{k} / n_lo_cat{k}); a lift of 20 on a
    dozen rows is noise.

    `ramp` does not require an informative block. The old rule demanded
    tail_lift > threshold AND gradient > threshold and so missed the two
    steepest ramps in results_9 (WORST_CLOSED_LOAN_DPD 5.55,
    AVERAGE_CLOSE_LOAN_DPD 5.17), whose blocks average out near 1.4. A block
    lift is informative in either direction — a head that is unusually CLEAN
    is as much signal as one that is unusually risky.
    """
    end = "hi" if side == "tail" else "lo"
    merged = row.get(f"pct_{end}", 0.0) or 0.0
    if merged <= 0:
        return "none"
    n = row.get(f"n_{end}")
    if n is not None and np.isfinite(n) and n < _MIN_TAIL_ROWS:
        return "tiny"

    def big_enough(key):
        # "tail_lift_cat0" -> "n_hi_cat0"; pooled keys have no stratum count.
        if "_cat" not in key:
            return True
        k_n = row.get(f"n_{end}_cat" + key.rsplit("_cat", 1)[1])
        return k_n is None or not np.isfinite(k_n) or k_n >= _MIN_TAIL_ROWS

    def present(prefix):
        return [v for k, v in row.items()
                if (k == prefix or k.startswith(prefix + "_cat"))
                and isinstance(v, (int, float)) and np.isfinite(v)
                and big_enough(k)]

    grads = present(f"{side}_gradient")
    lifts = present(f"{side}_lift")
    ramp = any(g > grad_thr for g in grads)
    informative = any(v > lift_thr or v < 1 / lift_thr for v in lifts)
    swamped = informative and row.get(f"pct_tied_{end}", 0.0) >= SWAMP_RATIO * merged

    parts = [p for p, on in (("ramp", ramp), ("swamped", swamped)) if on]
    if parts:
        return "+".join(parts)
    return "block" if informative else "inert"


def add_verdicts(rep: pd.DataFrame, lift_thr: float, grad_thr: float) -> pd.DataFrame:
    rep = rep.copy()
    for side in ("tail", "head"):
        rep[f"{side}_verdict"] = [side_verdict(r, side, lift_thr, grad_thr)
                                  for r in rep.to_dict("records")]
    return rep


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
                    help="also report lift, gradient and row counts INSIDE each "
                         "queued current_cat stratum, against that stratum's own "
                         "base rate. Pooled numbers mix the strata: a tail full "
                         "of cat_2 rows reports a large lift with no incremental "
                         "signal, and a pooled gradient moves with the mix too.")
    ap.add_argument("--include_exempt", action="store_true",
                    help="also report the clip:false columns, with the bounds the "
                         "clipper WOULD fit — to audit an exemption with the same "
                         "measurements that would have justified it")
    ap.add_argument("--output", default="explore_output/clip_impact.csv")
    ap.add_argument("--lift_threshold", type=float, default=1.5,
                    help="block lift above this (or below its inverse) counts as "
                         "informative")
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
                      null_frac=null_frac, include_exempt=args.include_exempt)
    if rep.empty:
        log.error("Nothing to report — every requested column is binary or "
                  "clip:false. Pass --include_exempt to audit exempt columns.")
        return
    rep = add_verdicts(rep, args.lift_threshold, args.gradient_threshold)

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

    nulled = rep[rep["pct_null"] > 0.002]
    if len(nulled):
        log.info(
            f"{len(nulled)} feature(s) with a null rate above 0.2%: "
            f"{', '.join(nulled['feature'])}\n"
            "  Their fill value (usually 0.0) is part of the distribution the\n"
            "  clipper fits. Any earlier report that dropped NaN before taking\n"
            "  percentiles stated a p1 these columns never had."
        )

    def costs(r):
        return [s for s in ("tail", "head")
                if r[f"{s}_verdict"].startswith(("ramp", "swamped"))]

    def named(frame):
        out = []
        for r in frame.to_dict("records"):
            where = "/".join(f"{s}:{r[s + '_verdict']}" for s in costs(r))
            out.append(f"{r['feature']} ({where})" if where else r["feature"])
        return ", ".join(out)

    clipped, exempt = rep[~rep["exempt"]], rep[rep["exempt"]]
    costly = clipped[[bool(costs(r)) for r in clipped.to_dict("records")]]
    block = clipped[[not costs(r) and "block" in (r["tail_verdict"], r["head_verdict"])
                     for r in clipped.to_dict("records")]]

    if len(costly):
        log.warning(
            f"{len(costly)} clipped feature(s) where the clip costs something: "
            f"{named(costly)}\n"
            "  ramp    = risk varies across the clipped region, and the clip\n"
            "            flattens that ordering.\n"
            "  swamped = the region is informative but the merged rows can no\n"
            "            longer be split off from those already on the bound.\n"
            "  Candidates for clip:false. Read the per-stratum columns and the\n"
            "  n_*_cat counts first, then A/B on validation lift@K."
        )
    else:
        log.info("No clipped feature shows a ramp or a swamped bound — the clip "
                 "merges rows the model had no reason to separate.")
    if len(block):
        log.info(
            f"{len(block)} clipped feature(s) informative as a BLOCK but flat "
            f"inside and not swamped: {', '.join(block['feature'])}\n"
            "  A split just inside the bound still isolates the block, so the\n"
            "  clip costs nothing here. Leave these clipped."
        )
    if len(exempt):
        justified = exempt[[bool(costs(r)) for r in exempt.to_dict("records")]]
        idle = exempt[[not costs(r) for r in exempt.to_dict("records")]]
        log.info(
            f"--include_exempt: {len(exempt)} clip:false column(s) audited.\n"
            f"  Clipping WOULD cost something (exemption justified): "
            f"{named(justified) or 'none'}\n"
            f"  Clipping would cost nothing measurable (exemption harmless, not "
            f"needed): {', '.join(idle['feature']) or 'none'}"
        )

    if not args.by_current_cat:
        log.warning(
            "No --by_current_cat. Pooled lifts AND pooled gradients both move "
            "with the current_cat mix of the region they measure. Re-run with "
            "--by_current_cat before acting on any verdict above."
        )
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
