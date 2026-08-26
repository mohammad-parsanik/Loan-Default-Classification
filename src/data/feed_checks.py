"""
Invariant checks on a freshly-loaded frame from the upstream feed.

These are the properties the feed contract guarantees on every row. They are
cheap (vectorised, one pass each) and they catch a bad or misaligned feed
before it reaches a model, where the same problem surfaces months later as
"the metrics moved and nobody knows why".

Violations are reported, not raised, by default: a single bad row in a 577k-row
snapshot should not abort a six-hour run, and the log line names the columns
and the count so it can be taken back upstream. Pass `strict=True` to raise.

Deliberately no cross-checking of column MEANING — that lives upstream. This
only asserts relationships the consumer already relies on.
"""

import logging

import pandas as pd

import project_config as config
from src.data.column_contract import SENTINELS

logger = logging.getLogger(__name__)

CATEGORY_COLS = ["LOAN_CATEGORY", "CATEGORY_T1", "CATEGORY_T2", "CATEGORY_T3",
                 "WORST_FUTURE_CAT", "HIST_MAX_CATEGORY"]
#: cumulative bands: a loan cannot have cleared a wider band more recently
#: than a narrower one, so the four columns are non-decreasing left to right.
DAYS_SINCE_LADDER = ["DAYS_SINCE_LAST_DPD", "DAYS_SINCE_LAST_30_DPD",
                     "DAYS_SINCE_LAST_60_DPD", "DAYS_SINCE_LAST_90_DPD"]
#: "this column has lost its sentinel" is only evidence on a frame large
#: enough that its absence is surprising.
SENTINEL_CHECK_MIN_ROWS = 1000


class FeedInvariantError(ValueError):
    """Raised by assert_feed_invariants(strict=True) when the feed is bad."""


def _has(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)


def assert_feed_invariants(df: pd.DataFrame, strict: bool = False) -> list[str]:
    """
    Check the feed's row-level invariants. Returns the list of violation
    messages (empty when clean); logs each one. Columns absent from `df` are
    skipped — prediction frames legitimately lack the label columns, and
    synthetic test frames carry only a subset.
    """
    problems: list[str] = []

    def fail(msg: str, n: int) -> None:
        problems.append(f"{msg} ({n:,} row(s))")

    # 1. WORST_FUTURE_CAT >= LOAN_CATEGORY. The forward window includes the
    #    current month and both go through the same banding function, so a
    #    loan cannot end up in a better band than it is in today. This is
    #    arithmetic, not leakage — and it is why aggregate accuracy misleads.
    if _has(df, [config.TARGET_COL, "LOAN_CATEGORY"]):
        bad = int((df[config.TARGET_COL] < df["LOAN_CATEGORY"]).sum())
        if bad:
            fail(f"{config.TARGET_COL} < LOAN_CATEGORY", bad)

    # 2. the DAYS_SINCE_LAST_* ladder is non-decreasing.
    if _has(df, DAYS_SINCE_LADDER):
        for lo, hi in zip(DAYS_SINCE_LADDER, DAYS_SINCE_LADDER[1:]):
            bad = int((df[lo] > df[hi]).sum())
            if bad:
                fail(f"{lo} > {hi}", bad)

    # 3. MONTHS_IN_CURRENT_CATEGORY is 1..6 — floor 1 because a loan that is
    #    in a category has been there at least a month; 6 is censored
    #    ("six or more"), the lag horizon rather than a data limit.
    if "MONTHS_IN_CURRENT_CATEGORY" in df.columns:
        col = df["MONTHS_IN_CURRENT_CATEGORY"]
        bad = int(((col < 1) | (col > 6)).sum())
        if bad:
            fail("MONTHS_IN_CURRENT_CATEGORY outside 1..6", bad)

    # 4. every category column is in 0..4 (the consumer collapses 3-4 into a
    #    single severe class later; the feed does not).
    for col in CATEGORY_COLS:
        if col in df.columns:
            bad = int(((df[col] < 0) | (df[col] > 4)).sum())
            if bad:
                fail(f"{col} outside 0..4", bad)

    # 5. exactly one row per LOAN_ID x SNAPSHOT_DATE. Enforced upstream at
    #    load; checked here because everything downstream assumes the grain.
    if _has(df, [config.ID_COL, config.SNAPSHOT_COL]):
        bad = int(df.duplicated([config.ID_COL, config.SNAPSHOT_COL]).sum())
        if bad:
            fail(f"duplicate ({config.ID_COL}, {config.SNAPSHOT_COL}) key", bad)

    # 6. NPL implies PRE-NPL — nested thresholds on one column, never
    #    independent flags.
    if _has(df, ["HAS_EVER_BEEN_NPL", "HAS_EVER_BEEN_PRENPL"]):
        bad = int(((df["HAS_EVER_BEEN_NPL"] == 1)
                   & (df["HAS_EVER_BEEN_PRENPL"] != 1)).sum())
        if bad:
            fail("HAS_EVER_BEEN_NPL = 1 without HAS_EVER_BEEN_PRENPL = 1", bad)

    # 7. LABEL_HORIZON_DATE is constant within a snapshot. Not a correctness
    #    requirement here (maturity is filtered per row-group anyway), but if
    #    it ever stops holding, the snapshot-keyed horizon map below is no
    #    longer the right shape and we want to know.
    if _has(df, [config.SNAPSHOT_COL, config.HORIZON_COL]):
        per_snap = df.groupby(config.SNAPSHOT_COL)[config.HORIZON_COL].nunique()
        bad = int((per_snap > 1).sum())
        if bad:
            fail(f"{config.HORIZON_COL} varies within a snapshot", bad)

    # 8. the sentinel columns still carry their sentinel. If the upstream
    #    encoding changes, a run that clips and scales them as ordinary
    #    quantities is exactly the failure the exemption exists to prevent,
    #    and it is otherwise silent. Only meaningful on a frame big enough for
    #    absence to mean something — a caller scoring 20 loans can legitimately
    #    have none, and a false alarm here trains people to ignore the check.
    if len(df) >= SENTINEL_CHECK_MIN_ROWS:
        for col, sentinel in SENTINELS.items():
            if col in df.columns and not bool((df[col] == sentinel).any()):
                problems.append(
                    f"{col} contains no {sentinel:g} sentinel in {len(df):,} rows "
                    "— the feed's encoding may have changed; re-check the "
                    "clip/scale exemptions"
                )

    if problems:
        summary = f"Feed invariant check on {len(df):,} rows found {len(problems)} problem(s):"
        if strict:
            raise FeedInvariantError(summary + "\n  - " + "\n  - ".join(problems))
        logger.warning(summary)
        for p in problems:
            logger.warning(f"  - {p}")
    else:
        logger.info(f"Feed invariant check passed on {len(df):,} rows.")
    return problems


def label_horizons(df: pd.DataFrame) -> dict:
    """
    snapshot -> LABEL_HORIZON_DATE, taken from the frame. Empty when the feed
    does not carry the column (synthetic frames, or the deprecated table).
    """
    if not _has(df, [config.SNAPSHOT_COL, config.HORIZON_COL]):
        return {}
    pairs = df[[config.SNAPSHOT_COL, config.HORIZON_COL]].dropna().drop_duplicates()
    return {int(s): int(h) for s, h in pairs.itertuples(index=False)}


if __name__ == "__main__":
    # Three rows: one clean, one delinquent, one that has never touched any
    # band (all sentinels), so a single edited cell below cannot accidentally
    # remove a sentinel from a column and trip check 8 as a side effect.
    good = pd.DataFrame({
        config.ID_COL: [1, 2, 3],
        config.SNAPSHOT_COL: [20250131] * 3,
        config.HORIZON_COL: [20250731] * 3,
        "LOAN_CATEGORY": [0, 2, 0], config.TARGET_COL: [1, 2, 0],
        "MONTHS_IN_CURRENT_CATEGORY": [1, 6, 1],
        "HAS_EVER_BEEN_NPL": [0, 1, 0], "HAS_EVER_BEEN_PRENPL": [0, 1, 0],
        "DAYS_SINCE_LAST_DPD":    [0, 10, 99999],
        "DAYS_SINCE_LAST_30_DPD": [0, 20, 99999],
        "DAYS_SINCE_LAST_60_DPD": [0, 30, 99999],
        "DAYS_SINCE_LAST_90_DPD": [0, 99999, 99999],
    })
    assert assert_feed_invariants(good) == []
    assert label_horizons(good) == {20250131: 20250731}

    # Check 8 only fires on a frame big enough for a missing sentinel to mean
    # something; below the floor it stays quiet however the column looks.
    big = pd.concat([good] * (SENTINEL_CHECK_MIN_ROWS // 3 + 1), ignore_index=True)
    big[config.ID_COL] = range(len(big))
    assert assert_feed_invariants(big) == []
    no_sentinel = big.assign(DAYS_SINCE_LAST_90_DPD=0)
    assert any("no 99999 sentinel" in f for f in assert_feed_invariants(no_sentinel))

    bad = good.copy()
    bad.loc[1, config.TARGET_COL] = 1                  # label below current cat
    bad.loc[1, "MONTHS_IN_CURRENT_CATEGORY"] = 7       # outside 1..6
    bad.loc[1, "HAS_EVER_BEEN_PRENPL"] = 0             # NPL without PRE-NPL
    bad.loc[1, "DAYS_SINCE_LAST_30_DPD"] = 5           # ladder inverted at 30 -> 60
    found = assert_feed_invariants(bad)
    assert len(found) == 4, found
    assert any("< LOAN_CATEGORY" in f for f in found)
    assert any("DAYS_SINCE_LAST_DPD > DAYS_SINCE_LAST_30_DPD" in f for f in found)
    assert any("MONTHS_IN_CURRENT_CATEGORY" in f for f in found)
    assert any("HAS_EVER_BEEN_NPL" in f for f in found)
    try:
        assert_feed_invariants(bad, strict=True)
        raise AssertionError("strict=True should have raised")
    except FeedInvariantError:
        pass
    print("feed_checks.py self-check OK")
