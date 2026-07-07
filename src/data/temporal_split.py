"""
Temporal split for the Loan Default Classification pipeline.

Rules:
  1. Drop any snapshot whose date is within the last LABEL_HORIZON_MONTHS
     calendar months — those labels have not yet materialised.
  2. Among the remaining (usable) snapshots, ensure that:
       - test  snapshot is at least LABEL_HORIZON_MONTHS months after val
       - val   snapshot is at least LABEL_HORIZON_MONTHS months after the
         last train snapshot
     This prevents any label leakage across splits.
  3. Assignment:
       test  → newest usable snapshot
       val   → newest snapshot that is >= horizon months before test
       train → all snapshots that are >= horizon months before val

Walk-Forward (Rolling Window) Validation:
  generate_walk_forward_folds() enumerates all valid (train, val, test)
  fold combinations from the usable snapshots, respecting the same
  6-month gap constraint at every boundary.  For N usable snapshots,
  this produces multiple folds that together cover every usable time
  period as a test set at least once.
"""

import hashlib
import logging
from collections import namedtuple
from datetime import date, datetime
from typing import List, Dict, Set, Tuple, Optional

import project_config as config

logger = logging.getLogger(__name__)

# ── Named tuple for a single fold definition ──────────────────────────────────

Fold = namedtuple("Fold", ["fold_id", "train_snaps", "val_snap", "test_snap"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _snap_to_date(snap) -> date:
    """Convert an integer snapshot date (YYYYMMDD) to a date object."""
    return datetime.strptime(str(int(snap)), "%Y%m%d").date()


def _months_apart(earlier: date, later: date) -> float:
    """
    Return the (approximate) number of months from `earlier` to `later`.
    Positive when later > earlier.
    """
    return (later - earlier).days / 30.4375   # average days per month


def filter_mature_snapshots(raw_snaps) -> list:
    """
    Given raw YYYYMMDD snapshot values, return only those whose
    WORST_FUTURE_* label has had time to materialise (>= LABEL_HORIZON_MONTHS
    old). Any script that reads WORST_FUTURE_CAT/DPD — training split,
    IV/UMAP diagnostics, data profiling — must filter through this before
    trusting that column, or immature rows (whose label is really just
    "worst category observed so far") silently masquerade as real labels.
    """
    horizon = config.LABEL_HORIZON_MONTHS
    today   = date.today()

    raw_snaps  = sorted(set(raw_snaps))
    snap_dates = [_snap_to_date(s) for s in raw_snaps]

    mature  = [r for r, d in zip(raw_snaps, snap_dates) if _months_apart(d, today) >= horizon]
    dropped = [r for r, d in zip(raw_snaps, snap_dates) if _months_apart(d, today) < horizon]

    if dropped:
        logger.warning(
            f"Dropping {len(dropped)} snapshot(s) whose labels are not yet "
            f"mature (< {horizon} months old): {dropped}"
        )
    return mature


def _get_usable_snapshots(instances: List[Dict]) -> Tuple[list, list]:
    """
    From a list of instances, return (usable_raw, usable_dates) —
    snapshots that are mature (>= horizon months old).
    """
    raw_snaps = sorted(set(inst["snapshot_date"] for inst in instances))
    logger.info(f"All snapshot dates found: {raw_snaps}")

    usable_raw   = filter_mature_snapshots(raw_snaps)
    usable_dates = [_snap_to_date(s) for s in usable_raw]
    logger.info(f"Usable snapshots ({len(usable_raw)}): {usable_raw}")

    return usable_raw, usable_dates


# ── main function (single static split) ──────────────────────────────────────

def split_by_time(instances: List[Dict]) -> tuple:
    """
    Split instances chronologically into (train, val, test) sets.

    Steps
    -----
    1. Collect unique snapshot dates and sort oldest → newest.
    2. Discard snapshots that are < LABEL_HORIZON_MONTHS from today
       (their labels have not yet settled).
    3. Assign:
         - test  → newest usable snapshot
         - val   → second newest usable snapshot
         - train → all snapshots that are >= LABEL_HORIZON_MONTHS months
                   before the val snapshot.
       (Any snapshots falling in the 6-month gap between Train and Val are dropped).
    """

    horizon = config.LABEL_HORIZON_MONTHS

    usable_raw, usable_dates = _get_usable_snapshots(instances)

    if not usable_raw:
        raise ValueError(
            f"No usable snapshots remain after excluding those within the "
            f"last {horizon} months of today ({date.today()}). "
            f"Cannot build a train/val/test split."
        )

    # ── 3. Assign test, val, train ────────────────────────────────────────────
    train_snaps: Set = set()
    val_snaps:   Set = set()
    test_snaps:  Set = set()

    if len(usable_raw) == 1:
        logger.warning(
            "Only 1 usable snapshot. All data goes to train; "
            "val and test will be empty."
        )
        train_snaps = {usable_raw[0]}

    elif len(usable_raw) == 2:
        logger.warning(
            "Only 2 usable snapshots. Test = newest, Train = oldest, Val empty."
        )
        test_snaps  = {usable_raw[-1]}
        train_snaps = {usable_raw[0]}
        if _months_apart(usable_dates[0], usable_dates[1]) < horizon:
             logger.warning("Gap between train and test is less than 6 months!")

    else:
        test_raw  = usable_raw[-1]
        test_date = usable_dates[-1]

        optimize  = getattr(config, "OPTIMIZE_ON_VALIDATION", True)
        val_mode  = getattr(config, "VAL_SPLIT_MODE", "temporal")

        if optimize and val_mode == "temporal":
            val_raw   = usable_raw[-2]
            val_date  = usable_dates[-2]
            
            test_snaps = {test_raw}
            val_snaps  = {val_raw}
            
            # train candidates: >= horizon months before val
            train_candidates = [
                r for r, d in zip(usable_raw[:-2], usable_dates[:-2])
                if _months_apart(d, val_date) >= horizon
            ]
            
            gap_dropped = [
                r for r, d in zip(usable_raw[:-2], usable_dates[:-2])
                if _months_apart(d, val_date) < horizon
            ]
            
            if gap_dropped:
                 logger.warning(
                     f"Dropping {len(gap_dropped)} snapshot(s) to enforce the "
                     f"{horizon}-month gap before Val ({val_raw}): {gap_dropped}"
                 )
                 
            train_snaps = set(train_candidates)
            
            if not train_snaps:
                logger.warning(
                    f"No snapshot is at least {horizon} months before val "
                    f"snapshot {val_raw}. Train set will be empty."
                )
        else:
            test_snaps = {test_raw}
            val_snaps  = set()
            
            # train candidates: >= horizon months before test
            train_candidates = [
                r for r, d in zip(usable_raw[:-1], usable_dates[:-1])
                if _months_apart(d, test_date) >= horizon
            ]
            
            gap_dropped = [
                r for r, d in zip(usable_raw[:-1], usable_dates[:-1])
                if _months_apart(d, test_date) < horizon
            ]
            
            if gap_dropped:
                 logger.warning(
                     f"Dropping {len(gap_dropped)} snapshot(s) to enforce the "
                     f"{horizon}-month gap before Test ({test_raw}): {gap_dropped}"
                 )
                 
            train_snaps = set(train_candidates)
            
            if not train_snaps:
                logger.warning(
                    f"No snapshot is at least {horizon} months before test "
                    f"snapshot {test_raw}. Train set will be empty."
                )

    # ── 4. Build instance lists ───────────────────────────────────────────────
    train_inst = [i for i in instances if i["snapshot_date"] in train_snaps]
    val_inst   = [i for i in instances if i["snapshot_date"] in val_snaps]
    test_inst  = [i for i in instances if i["snapshot_date"] in test_snaps]

    # ── 5. Customer-disjoint in-time validation (leakage-free tuning) ─────────
    # Val labels come from the training era (>= horizon before test), so early
    # stopping / Optuna never see anything overlapping the test label window.
    if (
        getattr(config, "OPTIMIZE_ON_VALIDATION", True)
        and getattr(config, "VAL_SPLIT_MODE", "temporal") == "customer"
        and train_inst
    ):
        train_inst, val_inst = split_train_by_customer(
            train_inst, getattr(config, "CUSTOMER_VAL_FRACTION", 0.2)
        )

    logger.info("Temporal Split Configuration:")
    logger.info(f"  Train snapshots : {sorted(list(train_snaps))}")
    logger.info(f"  Val   snapshots : {sorted(list(val_snaps))}")
    logger.info(f"  Test  snapshots : {sorted(list(test_snaps))}")
    logger.info(
        f"  Instances → Train: {len(train_inst):,}, "
        f"Val: {len(val_inst):,}, Test: {len(test_inst):,}"
    )

    return train_inst, val_inst, test_inst


# ── Customer-disjoint split ───────────────────────────────────────────────────

def _customer_bucket(national_code, seed: int) -> float:
    """Stable [0, 1) bucket for a customer — md5, not hash() (seed-stable)."""
    digest = hashlib.md5(f"{seed}:{national_code}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_train_by_customer(
    train_inst: List[Dict],
    val_fraction: float,
    seed: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Partition training instances into (train, val) with DISJOINT customers:
    every snapshot-instance of a given NATIONAL_CODE lands on the same side,
    so the model is validated on customers it never saw.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    train_out, val_out = [], []
    for inst in train_inst:
        if _customer_bucket(inst["national_code"], seed) < val_fraction:
            val_out.append(inst)
        else:
            train_out.append(inst)

    logger.info(
        f"Customer-disjoint validation split: {len(train_out):,} train / "
        f"{len(val_out):,} val instances "
        f"(target val fraction {val_fraction:.0%})"
    )
    return train_out, val_out


# ── Walk-Forward fold generation ──────────────────────────────────────────────

def generate_walk_forward_folds(
    instances: List[Dict],
    min_train_snapshots: Optional[int] = None,
) -> List[Fold]:
    """
    Generate all valid walk-forward (rolling window) folds from the
    available usable snapshots.

    For each candidate (val_snap, test_snap) pair — where test_snap is the
    snapshot immediately after val_snap — the fold is valid if:
      1. test_snap is >= horizon months after val_snap.
      2. At least `min_train_snapshots` snapshots exist that are
         >= horizon months before val_snap.

    With 5 usable snapshots [S1, S2, S3, S4, S5], the algorithm produces
    every (test=Si, val=Sj) pair where i > j, both gaps are satisfied, and
    at least one training snapshot exists.  The folds are ordered so that
    the test snapshot advances forward in time (oldest test first).

    Parameters
    ----------
    instances : list[dict]
        All portfolio instances (all snapshots combined).
    min_train_snapshots : int, optional
        Minimum number of training snapshots required per fold.
        Defaults to config.MIN_TRAIN_SNAPSHOTS.

    Returns
    -------
    list[Fold]
        Ordered list of valid Fold namedtuples. Empty if no folds are valid.
    """
    if min_train_snapshots is None:
        min_train_snapshots = config.MIN_TRAIN_SNAPSHOTS

    horizon = config.LABEL_HORIZON_MONTHS
    usable_raw, usable_dates = _get_usable_snapshots(instances)

    if len(usable_raw) < 3:
        logger.warning(
            f"Walk-forward requires at least 3 usable snapshots "
            f"(found {len(usable_raw)}). Falling back to single static split."
        )
        return []

    folds: List[Fold] = []
    fold_id = 0

    # Enumerate all valid (test_idx, val_idx) pairs, test > val
    for test_idx in range(1, len(usable_raw)):
        test_snap = usable_raw[test_idx]
        test_date = usable_dates[test_idx]

        for val_idx in range(test_idx - 1, -1, -1):
            val_snap = usable_raw[val_idx]
            val_date = usable_dates[val_idx]

            # Gap 1: test must be >= horizon months after val
            gap_val_test = _months_apart(val_date, test_date)
            if gap_val_test < horizon:
                continue  # try a further-back val

            # Valid (val, test) pair — now find train candidates
            train_candidates = [
                usable_raw[i]
                for i in range(val_idx)
                if _months_apart(usable_dates[i], val_date) >= horizon
            ]

            if len(train_candidates) < min_train_snapshots:
                continue  # not enough training data for this fold

            fold_id += 1
            fold = Fold(
                fold_id   = fold_id,
                train_snaps = frozenset(train_candidates),
                val_snap    = val_snap,
                test_snap   = test_snap,
            )
            folds.append(fold)

    # Sort folds: primary by test_snap (ascending), secondary by val_snap
    folds.sort(key=lambda f: (f.test_snap, f.val_snap))
    # Re-number after sorting for cleaner logs
    folds = [
        Fold(fold_id=i + 1, train_snaps=f.train_snaps,
             val_snap=f.val_snap, test_snap=f.test_snap)
        for i, f in enumerate(folds)
    ]

    _log_fold_plan(folds)
    return folds


def _log_fold_plan(folds: List[Fold]) -> None:
    """Pretty-print the full walk-forward fold plan."""
    logger.info("=" * 60)
    logger.info(f"  Walk-Forward Fold Plan  ({len(folds)} folds)")
    logger.info("=" * 60)
    for f in folds:
        train_sorted = sorted(f.train_snaps)
        logger.info(
            f"  Fold {f.fold_id:02d} | "
            f"Train: {train_sorted} | "
            f"Val: {f.val_snap} | "
            f"Test: {f.test_snap}"
        )
    logger.info("=" * 60)


def build_fold_instances(
    instances: List[Dict],
    fold: Fold,
) -> tuple:
    """
    Partition a flat list of instances into (train_inst, val_inst, test_inst)
    for a given Fold definition.

    Parameters
    ----------
    instances : list[dict]
        All portfolio instances (all snapshots combined).
    fold : Fold
        The fold definition from generate_walk_forward_folds().

    Returns
    -------
    (train_inst, val_inst, test_inst) : tuple of list[dict]
    """
    train_inst = [i for i in instances if i["snapshot_date"] in fold.train_snaps]
    val_inst   = [i for i in instances if i["snapshot_date"] == fold.val_snap]
    test_inst  = [i for i in instances if i["snapshot_date"] == fold.test_snap]

    logger.info(
        f"Fold {fold.fold_id:02d} | "
        f"Train: {len(train_inst):,}  Val: {len(val_inst):,}  "
        f"Test: {len(test_inst):,}"
    )
    return train_inst, val_inst, test_inst
