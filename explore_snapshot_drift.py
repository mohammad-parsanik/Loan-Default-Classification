"""
explore_snapshot_drift.py
=========================
Does a snapshot's content depend on how OLD it is?

The upstream installment table is hard-deleted for loans the bank is done with
(fully paid, or post-NPL), and every snapshot in the table was rebuilt from one
as-of view during the <=7B backfill. So an old snapshot was reconstructed after
more of its loans had been deleted than a recent one. Two consequences, both
invisible in any single snapshot:

  LABELS   The scope filter needs `last installment >= T+6`, which structurally
           protects against payoff-driven removal but NOT against post-NPL
           removal -- and an NPL row is exactly a `WORST_FUTURE_CAT = 3` row.
           Older snapshots should therefore look optimistically healthy.

  FEATURES The customer-level columns are computed over the customer's OTHER
           loans. If siblings were deleted, they are computed over a survivor
           subset and biased low -- worst in the oldest snapshots, absent at
           serving time (nothing has been deleted from this month's rows yet).
           That is train/serve skew in the cross-loan risk signal.

Reports per snapshot: row count, severe rate, mean portfolio size, and the mean
of each tracked feature; then Spearman rho of each series against snapshot
order. A strong positive rho on `severe_rate` AND on the customer-history
columns is the deletion signature. Genuine portfolio deterioration moves the
severe rate without dragging "how many other loans does this customer have"
along with it.

Cannot prove deletion on its own -- portfolio growth confounds the feature
means. Read it beside a cohort-persistence count, which measures the deletion
rate directly.

Inputs (no DB): data/snapshots/train_<key>/  (per-snapshot NPZ cache)
Usage:
  python explore_snapshot_drift.py
  python explore_snapshot_drift.py --features COUNT_ACTIVE_CONTRACTS,AVG_DPD_OTHER_LOANS
  python explore_snapshot_drift.py --ranked_only      # exclude the carved rows
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import project_config as config
from src.data.data_loader import load_cached_arrays
from src.data.temporal_split import filter_mature_snapshots

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEVERE_CLASS = config.NUM_CLASSES - 1

# Columns computed over the customer's OTHER / CLOSED loans. These are the
# canaries: sibling deletion drags them down, and it drags them down hardest in
# the oldest snapshots. The *_CLOSE*_LOAN_* pair is the sharpest of them --
# closed loans are precisely what gets hard-deleted.
DELETION_CANARIES = [
    "COUNT_ACTIVE_CONTRACTS",
    "COUNT_DELINQUENT_CONTRACTS",
    "AVG_DPD_OTHER_LOANS",
    "MAX_DPD_ANY_PAST_LOAN",
    "WORST_CLOSED_LOAN_DPD",
    "AVERAGE_CLOSE_LOAN_DPD",
    "CNT_RECOVERED_BEFORE",
    "HAS_RECOVERED_BEFORE",
    "HAS_EVER_BEEN_NPL",
    "PRE_UPTO30_DPD_LOANS",
    "PRE_UPTO150_DPD_LOANS",
]


def _spearman_vs_order(y: np.ndarray) -> float:
    """
    rho of y against snapshot order. The x-axis is already 0..n-1 with no ties,
    so Spearman is just Pearson on y's ranks -- no scipy needed.
    """
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 3 or np.ptp(y[ok]) == 0:
        return np.nan                     # constant series has no trend
    # Average ranks, so ties do not get an order-dependent tie-break.
    ranks = pd.Series(y[ok]).rank().to_numpy()
    order = np.arange(len(y))[ok].astype(np.float64)
    return float(np.corrcoef(ranks, order)[0, 1])


def per_snapshot_table(cache_dir=None, features=None, ranked_only=False) -> pd.DataFrame:
    arrays, feat_cols = load_cached_arrays(cache_dir)
    flat           = arrays["features_flat"]
    offsets        = arrays["offsets"]
    labels         = arrays["labels"]
    current_cats   = arrays["current_cats"]
    snapshot_dates = arrays["snapshot_dates"]
    national_codes = arrays["national_codes"]
    portfolio_n    = arrays.get("portfolio_n_loans", arrays["n_loans"])

    sizes  = np.diff(offsets)
    mature = set(filter_mature_snapshots(np.unique(snapshot_dates)))
    keep   = np.array([s in mature for s in snapshot_dates])
    if not keep.all():
        log.warning(f"Dropping {(~keep).sum():,} instance(s) from immature snapshots.")
    if ranked_only:
        carved = current_cats >= config.CARVE_CURRENT_CAT_GE
        log.info(f"--ranked_only: dropping {int((carved & keep).sum()):,} carved "
                 f"(current_cat >= {config.CARVE_CURRENT_CAT_GE}) instance(s).")
        keep &= ~carved

    wanted = features or DELETION_CANARIES
    missing = [c for c in wanted if c not in feat_cols]
    if missing:
        log.warning(f"Not in the cached feature list, skipping: {', '.join(missing)}")
    idx = {c: feat_cols.index(c) for c in wanted if c in feat_cols}

    # features_flat is per LOAN row; labels/snapshots are per INSTANCE. At loan
    # grain that is 1:1, at portfolio grain it is not -- expand either way.
    row_snap = np.repeat(snapshot_dates, sizes)
    row_keep = np.repeat(keep, sizes)

    rows = []
    for snap in sorted(np.unique(snapshot_dates[keep])):
        inst_m = keep & (snapshot_dates == snap)
        row_m  = row_keep & (row_snap == snap)
        rec = {
            "snapshot":       int(snap),
            "n_instances":    int(inst_m.sum()),
            "n_loan_rows":    int(row_m.sum()),
            "n_customers":    int(len(np.unique(national_codes[inst_m]))),
            "severe_rate":    float((labels[inst_m] == SEVERE_CLASS).mean()),
            "mean_portfolio": float(portfolio_n[inst_m].mean()),
        }
        for c, i in idx.items():
            col = flat[row_m, i].astype(np.float64)
            rec[f"mean_{c}"] = float(np.nanmean(col)) if len(col) else np.nan
        rows.append(rec)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="per-snapshot cache dir (default: the current schema's)")
    ap.add_argument("--features", default=None,
                    help=f"comma-separated override (default: {len(DELETION_CANARIES)} "
                         "customer-history canaries)")
    ap.add_argument("--ranked_only", action="store_true",
                    help="exclude current_cat >= CARVE_CURRENT_CAT_GE, matching the queue")
    ap.add_argument("--output", default="explore_output/snapshot_drift.csv")
    ap.add_argument("--rho_threshold", type=float, default=0.7,
                    help="|rho| above which a series counts as trending with snapshot age")
    args = ap.parse_args()

    feats = args.features.split(",") if args.features else None
    try:
        tab = per_snapshot_table(Path(args.cache) if args.cache else None,
                                feats, args.ranked_only)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)
    if len(tab) < 3:
        log.error(f"Only {len(tab)} mature snapshot(s) — nothing to trend.")
        sys.exit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(tab.to_string(index=False))

    trend = pd.DataFrame(
        [{"series": c, "rho_vs_snapshot_order": _spearman_vs_order(tab[c].values),
          "first": tab[c].iloc[0], "last": tab[c].iloc[-1]}
         for c in tab.columns if c != "snapshot"]
    ).sort_values("rho_vs_snapshot_order", ascending=False)
    print("\nTrend vs snapshot order (Spearman rho; +1 = rises monotonically with recency)")
    with pd.option_context("display.float_format", lambda v: f"{v:,.4g}"):
        print(trend.to_string(index=False))

    rho = dict(zip(trend.series, trend.rho_vs_snapshot_order))
    sev = rho.get("severe_rate", np.nan)
    canaries = {k: v for k, v in rho.items() if k.startswith("mean_")}
    hot = [k for k, v in canaries.items() if v > args.rho_threshold]

    log.info(f"severe_rate rho = {sev:+.3f}")
    if sev > args.rho_threshold and hot:
        log.warning(
            f"DELETION SIGNATURE: severe_rate rises with recency (rho {sev:+.3f}) AND so do "
            f"{len(hot)}/{len(canaries)} customer-history means: {', '.join(hot)}.\n"
            "  Consistent with old snapshots having been rebuilt after their loans were\n"
            "  hard-deleted upstream. Confirm with a cohort-persistence count before\n"
            "  treating the training labels as unbiased; portfolio growth alone can lift\n"
            "  the feature means, but it does not preferentially delete class-3 rows."
        )
    elif sev > args.rho_threshold:
        log.warning(
            f"severe_rate rises with recency (rho {sev:+.3f}) but the customer-history "
            "means do not follow.\n  Points AWAY from sibling deletion and toward genuine "
            "portfolio deterioration."
        )
    else:
        log.info("No monotone severe-rate trend — neither hypothesis is supported here.")
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
