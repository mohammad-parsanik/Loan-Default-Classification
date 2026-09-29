"""Monthly monitoring: engine-run selection, metrics on hand-checkable data, report rendering."""

import json
import re

import numpy as np
import pandas as pd
import pytest

import project_config as config
from src.monitoring import monitor
from src.monitoring.report import build_report

ASOF = 20260928


def fake_engine(snaps, n=4000, seed=0):
    """Engine-shaped tables (OperationLog, InferenceOutput+tracking, feed) with real signal."""
    rng = np.random.default_rng(seed)
    runs, outputs, feed = [], {}, []
    for s in snaps:
        cat = rng.choice(3, n, p=[0.65, 0.25, 0.10])              # engine scores cats 0-2 only
        p3 = np.clip(rng.beta(1, 12, n) + 0.25 * (cat == 2) + 0.1 * (cat == 1), 0, 1).round(3)
        y = np.maximum(cat, np.where(rng.random(n) < p3 * 1.3,
                                     rng.choice([3, 4], n), rng.choice(3, n, p=[.8, .15, .05])))
        amt = rng.lognormal(18, 1, n)
        dpd = np.where(cat == 2, rng.integers(61, 156, n), 0)
        op = f"op-{s}"
        runs.append({"operation_id": op, "snapshot_date": s, "operation_status": "SUCCESS",
                     "operation_date": pd.Timestamp(str(s)) + pd.Timedelta(days=3),
                     "inference_failures": 0})
        outputs[op] = pd.DataFrame({
            "loan_id": np.arange(n, dtype="int64") + 10**17,
            "national_code": (np.arange(n) // 2).astype(str),
            "p3": p3, "dpd_cat": cat, "remain_of_account": amt,
            "operation_status_enrichment": np.where((amt > 5e7) & (p3 > 0.5) & (dpd > 120),
                                                    "PENDING_ENRICHMENT", "NOT_APPLICABLE")})
        horizon = int((pd.Timestamp(str(s)) + pd.DateOffset(months=6)).strftime("%Y%m%d"))
        feed.append(pd.DataFrame({"SNAPSHOT_DATE": s, "LOAN_ID": outputs[op]["loan_id"],
                                  "WORST_FUTURE_CAT": y, "LABEL_HORIZON_DATE": horizon}))
    return pd.DataFrame(runs), outputs, pd.concat(feed, ignore_index=True)


class _FakeConn:
    def __init__(self, runs, outputs, feed):
        self.runs, self.outputs, self.feed = runs, outputs, feed

    def read_sql(self, query, params=None):
        if config.INFERENCE_OPLOG_TABLE in query:
            return self.runs
        if config.INFERENCE_OUTPUT_TABLE in query:
            return self.outputs[params[0]]
        return self.feed[self.feed["SNAPSHOT_DATE"] == params[0]]


def fake_frames(snaps, n=4000, seed=0):
    return monitor.load_frames(_FakeConn(*fake_engine(snaps, n, seed)))


@pytest.fixture
def small_windows(monkeypatch):
    # "top_2" is a fixed count; "top_half" is a share: ceil(0.5 * 7) = 4 of the 7 hand loans.
    monkeypatch.setattr(config, "MONITOR_CUTOFFS", {"top_2": 2, "top_half": 0.5})
    monkeypatch.setattr(config, "MONITOR_MAIN_CUTOFF", "top_half")


def _hand_cohort():
    #            loan: 0    1    2    3    4    5    6
    preds = pd.DataFrame({
        "SNAPSHOT_DATE": 20250320, "LOAN_ID": range(7), "NATIONAL_CODE": list("abcdefg"),
        "CURRENT_CAT": [0, 0, 1, 0, 0, 0, 0],
        "RISK_SCORE":  [.9, .8, .7, .6, .5, .4, .3],
        "REMAINING_AMNT": [100, 1, 50, 1, 1, 50, 1],
        "FLAGGED": [False, True, True, False, False, False, False],
    })
    outcomes = pd.DataFrame({
        "SNAPSHOT_DATE": 20250320, "LOAN_ID": range(7),
        "WORST_FUTURE_CAT": [4, 0, 3, 0, 0, 3, 0],       # raw 4 caps to 3
        "LABEL_HORIZON_DATE": 20250920,
    })
    return preds, outcomes


def test_matured_metrics_by_hand(small_windows):
    m = monitor.compute(*_hand_cohort(), ASOF)
    c = m["matured"][20250320]
    # severe = {0, 2, 5}
    assert c["n_ranked"] == 7 and c["n_severe"] == 3
    top2, half = c["cutoffs"]["top_2"], c["cutoffs"]["top_half"]
    assert top2["k"] == 2 and half["k"] == 4
    assert top2["recall"] == pytest.approx(1 / 3)                 # top-2 = {0, 1}
    assert half["recall"] == pytest.approx(2 / 3)                 # top-4 = {0, 1, 2, 3}
    assert half["precision"] == pytest.approx(2 / 4)
    assert half["lift"] == pytest.approx((2 / 4) / (3 / 7))
    assert half["exposure_caught"] == pytest.approx(150 / 200)
    lo, hi = half["recall_ci"]
    assert lo < 2 / 3 < hi
    assert c["predicted_severe"] == pytest.approx(sum([.9, .8, .7, .6, .5, .4, .3]))
    q = c["queue_by_cat"]
    assert q[0]["recall_top_half"] == pytest.approx(1 / 2)   # cat-0 severe {0, 5}; top-4 has 0
    assert q[1]["recall_top_half"] == 1.0 and q[1]["share_of_top_half"] == pytest.approx(1 / 4)
    rule = c["enrichment_rule"]                              # flagged {1, 2}; top-2 = {0, 1}
    assert rule["n_flagged"] == 2 and rule["precision"] == pytest.approx(1 / 2)
    assert rule["recall"] == pytest.approx(1 / 3)
    assert rule["recall_top_same_size"] == pytest.approx(1 / 3)
    top = c["migration"]["top"]
    assert sum(sum(r.values()) for r in top.values()) == 4
    assert sum(sum(r.values()) for r in c["migration"]["rest"].values()) == 3


def test_immature_rows_stay_out_of_matured(small_windows):
    preds, outs = _hand_cohort()
    outs["LABEL_HORIZON_DATE"] = ASOF + 1
    m = monitor.compute(preds, outs, ASOF)
    assert not m["matured"] and 20250320 in m["interim"]
    i = m["interim"][20250320]
    assert i["deteriorated_rate"] == pytest.approx(3 / 7)         # loans 0, 2, 5
    assert i["cutoffs"]["top_2"]["deteriorated_rate"] == pytest.approx(1 / 2)
    assert i["rest"]["deteriorated_rate"] == pytest.approx(1 / 3)   # positions 5-7
    assert i["enrichment_rule"]["deteriorated_rate"] == pytest.approx(1 / 2)


def test_unmatched_loans_are_counted(small_windows):
    preds, outs = _hand_cohort()
    df, q = monitor.prepare(preds, outs[outs["LOAN_ID"] != 6], ASOF)
    assert q["unmatched_by_snapshot"] == {20250320: 1}
    assert 6 not in set(df["LOAN_ID"])


def test_pick_runs_takes_newest_usable_run_per_snapshot():
    runs = pd.DataFrame({
        "operation_id": ["a", "b", "c", "d"],
        "snapshot_date": [20260723, 20260723, 20260723, 20260823],
        "operation_status": ["SUCCESS", "PARTIAL_SUCCESS", "FAILED", "SUCCESS"],
        "operation_date": pd.to_datetime(["2026-07-25", "2026-07-28", "2026-07-30", "2026-08-25"]),
        "inference_failures": [0, 3, 0, 0],
    })
    picked = monitor.pick_runs(runs)
    assert list(picked["operation_id"]) == ["b", "d"]      # FAILED "c" is newer but unusable
    assert list(picked["n_usable_runs"]) == [2, 1]


def test_engine_columns_map_to_internal_frame():
    out = pd.DataFrame({"loan_id": [123456789012345678], "national_code": ["x"], "p3": [0.7],
                        "dpd_cat": [4], "remain_of_account": [6e7],
                        "operation_status_enrichment": ["PENDING_ENRICHMENT"]})
    p = monitor.engine_to_preds(out, 20260823)
    assert p["LOAN_ID"].item() == 123456789012345678        # 18 digits survive (no float round-trip)
    assert p["CURRENT_CAT"].item() == 3 and p["RISK_SCORE"].item() == 0.7
    assert p["FLAGGED"].item()
    out["operation_status_enrichment"] = [None]
    assert not monitor.engine_to_preds(out, 20260823)["FLAGGED"].item()


def test_cutoffs_resolve_shares_and_counts(monkeypatch):
    monkeypatch.setattr(config, "MONITOR_CUTOFFS", {"top_5pct": 0.05, "top_1000": 1000})
    assert monitor.cutoffs(100_000) == {"top_5pct": 5000, "top_1000": 1000}
    assert monitor.cutoffs(500) == {"top_5pct": 25, "top_1000": 500}   # count capped at list size
    assert monitor.cutoff_label("top_5pct") == "top 5%"
    assert monitor.cutoff_label("top_1000") == "top 1,000"


def test_psi_zero_on_identical_and_positive_on_shift():
    x = np.random.default_rng(1).random(5000)
    assert monitor.psi(x, x) == pytest.approx(0, abs=1e-9)
    assert monitor.psi(x, x ** 3) > 0.25


SNAPS = [20250321, 20250421, 20250522, 20250621, 20250722, 20250823, 20250922, 20251023,
         20251122, 20251222, 20260121, 20260220, 20260321, 20260421, 20260522, 20260621,
         20260723, 20260823]


def test_report_is_self_contained(tmp_path):
    preds, outs, runs = fake_frames(SNAPS)
    m = monitor.compute(preds, outs, ASOF, runs)
    assert m["matured"] and m["interim"]
    assert list(monitor.history_frame(m)["snapshot"]) == sorted(m["matured"])
    text = build_report(m, tmp_path).read_text()
    assert not re.search(r"(src|href)=\"https?://", text)       # works offline
    assert "data:image/png;base64," in text
    assert (tmp_path / "charts" / "trend_recall.png").exists()


def test_run_monitor_end_to_end(tmp_path):
    conn = _FakeConn(*fake_engine([20260220, 20260321, 20260723], n=3000))
    report = monitor.run_monitor(asof=ASOF, output_dir=str(tmp_path), conn=conn)
    assert report.exists()
    m = json.loads((tmp_path / str(ASOF) / "metrics.json").read_text())
    assert set(m["matured"]) == {"20260220", "20260321"} and set(m["interim"]) == {"20260723"}
    assert [r["operation_id"] for r in m["quality"]["runs"]] == ["op-20260220", "op-20260321",
                                                                 "op-20260723"]
    assert len(pd.read_csv(tmp_path / "history.csv")) == 2


def test_label_archive_keeps_highest_label_and_counts_deletions(tmp_path):
    first = pd.DataFrame({"SNAPSHOT_DATE": 20260723, "LOAN_ID": [1, 2, 3],
                          "WORST_FUTURE_CAT": [0, 4, 1], "LABEL_HORIZON_DATE": 20270123})
    _, st = monitor.reconcile_labels(first, tmp_path)
    assert st[20260723]["first_read"] and st[20260723]["n_vanished"] == 0

    # a month later: loan 2's installments were deleted (label drops 4 -> 0),
    # loan 3 is gone from the feed, loan 1 genuinely worsened, loan 4 is new
    second = pd.DataFrame({"SNAPSHOT_DATE": 20260723, "LOAN_ID": [1, 2, 4],
                           "WORST_FUTURE_CAT": [2, 0, 0], "LABEL_HORIZON_DATE": 20270123})
    out, st = monitor.reconcile_labels(second, tmp_path)
    assert st[20260723] == {"n_label_lowered": 1, "n_vanished": 1, "first_read": False}
    labels = dict(zip(out["LOAN_ID"], out["WORST_FUTURE_CAT"]))
    assert labels == {1: 2, 2: 4, 3: 1, 4: 0}
    assert set(out["LABEL_HORIZON_DATE"]) == {20270123}

    # and the archive now holds the reconciled values for next month
    _, st = monitor.reconcile_labels(second, tmp_path)
    assert st[20260723]["n_label_lowered"] == 1 and st[20260723]["n_vanished"] == 1
