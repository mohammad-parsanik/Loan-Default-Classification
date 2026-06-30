"""
Temporal train/val/test split with a data-leakage safety check.

Split logic (5 snapshots):
    S1  S2  S3 │ S4  │ S5
    ───────────┼─────┼────
       TRAIN   │ VAL │ TEST

Leakage check:
    WORST_FUTURE_CAT uses a 6-month forward horizon.
    For the labels of the last training snapshot NOT to contaminate the
    test-set period, the last train snapshot must be at least
    LABEL_HORIZON_MONTHS before the first test snapshot.
    A WARNING is emitted if not — the split still proceeds so that
    development can continue while ETL is being fixed.

SNAPSHOT_DATE format:  YYYYMMDD as an integer (supports both Gregorian and
Shamsi/Persian calendar since month arithmetic is the same).
"""

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config

logger = logging.getLogger(__name__)


def _snapshot_to_year_month(snapshot_date) -> tuple[int, int]:
    """
    Parse an integer YYYYMMDD snapshot date into (year, month).
    Works for both Gregorian and Shamsi calendar dates.
    """
    d = int(snapshot_date)
    year  = d // 10000
    month = (d % 10000) // 100
    return year, month


def _month_distance(snap_a, snap_b) -> int:
    """
    Returns the number of calendar months between two YYYYMMDD snapshot dates.
    Positive if snap_b is after snap_a.
    """
    ya, ma = _snapshot_to_year_month(snap_a)
    yb, mb = _snapshot_to_year_month(snap_b)
    return (yb - ya) * 12 + (mb - ma)


def _check_leakage(
    train_snaps: list,
    test_snaps: list,
    horizon_months: int = config.LABEL_HORIZON_MONTHS,
) -> None:
    """
    Emit a WARNING if the last training snapshot is within the label horizon
    of the first test snapshot.

    The labels for snapshot T use data from (T, T + horizon_months].
    If T_last_train + horizon_months > T_first_test, the label computation
    window for training rows overlaps with the test period.
    """
    if not train_snaps or not test_snaps:
        return

    last_train = max(train_snaps)
    first_test = min(test_snaps)
    gap = _month_distance(last_train, first_test)

    logger.info(
        f"Leakage check: last train snapshot={last_train}, "
        f"first test snapshot={first_test}, gap={gap} months "
        f"(required ≥ {horizon_months})."
    )

    if gap < horizon_months:
        logger.warning(
            f"⚠️  POTENTIAL LABEL LEAKAGE DETECTED: the gap between last "
            f"train snapshot ({last_train}) and first test snapshot "
            f"({first_test}) is {gap} month(s), but WORST_FUTURE_CAT uses a "
            f"{horizon_months}-month forward horizon. Label computation windows "
            f"overlap. Re-produce ETL snapshots with a proper horizon gap before "
            f"trusting test metrics. Training will continue."
        )
    else:
        logger.info(
            f"✅ Leakage check passed: {gap}-month gap exceeds the "
            f"{horizon_months}-month label horizon."
        )


def split_by_time(instances: list[dict]) -> tuple[list, list, list]:
    """
    Split instances chronologically based on snapshot dates.

    With ≥ 5 snapshots:  all-but-last-two → train, second-to-last → val, last → test
    With 4 snapshots:    first two → train, third → val, last → test
    With 3 snapshots:    first → train, second → val, third → test
    With < 3 snapshots:  everything → train, empty val and test

    Always runs a leakage check and logs a WARNING if the gap between
    last train and first test snapshot is less than LABEL_HORIZON_MONTHS.
    """
    snapshots = sorted(set(inst["snapshot_date"] for inst in instances))
    n = len(snapshots)

    logger.info(f"Found {n} unique snapshot dates: {snapshots}")

    if n < 3:
        logger.warning(
            f"Only {n} snapshot(s) found. All instances assigned to train; "
            f"val and test will be empty."
        )
        return instances, [], []

    # Universal rule: last → test, second-to-last → val, rest → train
    train_snaps = snapshots[:-2]
    val_snaps   = [snapshots[-2]]
    test_snaps  = [snapshots[-1]]

    logger.info(
        f"Temporal split:\n"
        f"  Train: {train_snaps}\n"
        f"  Val:   {val_snaps}\n"
        f"  Test:  {test_snaps}"
    )

    # Convert to sets for O(1) lookup
    train_set = set(train_snaps)
    val_set   = set(val_snaps)
    test_set  = set(test_snaps)

    train_inst = [i for i in instances if i["snapshot_date"] in train_set]
    val_inst   = [i for i in instances if i["snapshot_date"] in val_set]
    test_inst  = [i for i in instances if i["snapshot_date"] in test_set]

    logger.info(
        f"Instance counts → Train: {len(train_inst):,}, "
        f"Val: {len(val_inst):,}, Test: {len(test_inst):,}"
    )

    # Leakage check (warning only — does not block training)
    _check_leakage(train_snaps, test_snaps)

    return train_inst, val_inst, test_inst
