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
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Set

import project_config as config

logger = logging.getLogger(__name__)


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


# ── main function ─────────────────────────────────────────────────────────────

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

    horizon = config.LABEL_HORIZON_MONTHS   # e.g. 6
    today   = date.today()

    # ── 1. Unique snapshots, sorted oldest → newest ───────────────────────────
    raw_snaps  = sorted(set(inst["snapshot_date"] for inst in instances))
    snap_dates = [_snap_to_date(s) for s in raw_snaps]

    logger.info(f"All snapshot dates found: {raw_snaps}")

    # ── 2. Drop immature snapshots ────────────────────────────────────────────
    usable = [
        (raw, d)
        for raw, d in zip(raw_snaps, snap_dates)
        if _months_apart(d, today) >= horizon
    ]
    dropped = [r for r, d in zip(raw_snaps, snap_dates)
               if _months_apart(d, today) < horizon]

    if dropped:
        logger.warning(
            f"Dropping {len(dropped)} snapshot(s) whose labels are not yet "
            f"mature (< {horizon} months old): {dropped}"
        )

    if not usable:
        raise ValueError(
            f"No usable snapshots remain after excluding those within the "
            f"last {horizon} months of today ({today}). "
            f"Cannot build a train/val/test split."
        )

    usable_raw   = [p[0] for p in usable]
    usable_dates = [p[1] for p in usable]
    logger.info(f"Usable snapshots ({len(usable_raw)}): {usable_raw}")

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
        # If we wanted to be strict, we'd check the gap here too, but for 2 snaps,
        # usually just one train, one test.
        if _months_apart(usable_dates[0], usable_dates[1]) < horizon:
             logger.warning("Gap between train and test is less than 6 months!")

    else:
        test_raw  = usable_raw[-1]
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

    # ── 4. Build instance lists ───────────────────────────────────────────────
    train_inst = [i for i in instances if i["snapshot_date"] in train_snaps]
    val_inst   = [i for i in instances if i["snapshot_date"] in val_snaps]
    test_inst  = [i for i in instances if i["snapshot_date"] in test_snaps]

    logger.info("Temporal Split Configuration:")
    logger.info(f"  Train snapshots : {sorted(list(train_snaps))}")
    logger.info(f"  Val   snapshots : {sorted(list(val_snaps))}")
    logger.info(f"  Test  snapshots : {sorted(list(test_snaps))}")
    logger.info(
        f"  Instances → Train: {len(train_inst):,}, "
        f"Val: {len(val_inst):,}, Test: {len(test_inst):,}"
    )

    return train_inst, val_inst, test_inst
