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
BAND_DPD = np.array([0, 30, 100, 170, 200])     # a DPD inside each raw band 0-4


def fake_engine(snaps, n=4000, seed=0, asof=ASOF):
    """
    Engine-shaped tables (OperationLog, InferenceOutput+tracking, label ledger,
    and the monitor's outcome query) with real signal. A snapshot is FINAL
    once its horizon is 3+ days before asof, else open with the state-log
    months that have passed.
    """
    rng = np.random.default_rng(seed)
    runs, outputs, feed, ledger = [], {}, [], []
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
        horizon = pd.Timestamp(str(s)) + pd.DateOffset(months=6)
        finalized = horizon + pd.Timedelta(days=3)
        final = finalized < pd.Timestamp(str(asof))
        months = 6 if final else max(0, min(6, (pd.Timestamp(str(asof)) - pd.Timestamp(str(s))).days // 30))
        if final:
            ledger.append({"SNAPSHOT_DATE": s, "STATUS": "FINAL", "FINALIZED_AT": finalized,
                           "LOANS_UNLABELLED": 0, "LOANS_MISSING_MONTHS": 0})
        feed.append(pd.DataFrame({
            "SNAPSHOT_DATE": s, "LOAN_ID": outputs[op]["loan_id"],
            # open snapshots have seen part of the window: a lower bound on y
            "WORST_DPD_SO_FAR": BAND_DPD[y if final else np.minimum(y, cat + months // 3)],
            "MONTHS_OBSERVED": months, "MONTHS_DELETED": 0, "STATE_ROWS": months + 1,
            "FINAL_CAT": y.astype(float) if final else np.nan}))
    ledger = pd.DataFrame(ledger, columns=["SNAPSHOT_DATE", "STATUS", "FINALIZED_AT",
                                           "LOANS_UNLABELLED", "LOANS_MISSING_MONTHS"])
    return pd.DataFrame(runs), outputs, pd.concat(feed, ignore_index=True), ledger


class _FakeConn:
    def __init__(self, runs, outputs, feed, ledger):
        self.runs, self.outputs, self.feed, self.ledger = runs, outputs, feed, ledger

    def read_sql(self, query, params=None):
        if config.LABEL_LEDGER_TABLE in query:
            return self.ledger
        if config.INFERENCE_OPLOG_TABLE in query:
            return self.runs
        if config.INFERENCE_OUTPUT_TABLE in query:
            return self.outputs[params[0]]
        assert config.LABEL_STATE_TABLE in query
        return self.feed[self.feed["SNAPSHOT_DATE"] == params[-1]]


def fake_frames(snaps, n=4000, seed=0):
    return monitor.load_frames(_FakeConn(*fake_engine(snaps, n, seed)), ASOF)


@pytest.fixture
def small_windows(monkeypatch):
    # "top_2" is a fixed count; "top_half" is a share: ceil(0.5 * 7) = 4 of the 7 hand loans.
    monkeypatch.setattr(config, "MONITOR_CUTOFFS", {"top_2": 2, "top_half": 0.5})
    monkeypatch.setattr(config, "MONITOR_MAIN_CUTOFF", "top_half")


def _hand_cohort(status="FINAL"):
    #            loan: 0    1    2    3    4    5    6
    preds = pd.DataFrame({
        "SNAPSHOT_DATE": 20250320, "LOAN_ID": range(7), "NATIONAL_CODE": list("abcdefg"),
        "CURRENT_CAT": [0, 0, 1, 0, 0, 0, 0],
        "RISK_SCORE":  [.9, .8, .7, .6, .5, .4, .3],
        "REMAINING_AMNT": [100, 1, 50, 1, 1, 50, 1],
        "FLAGGED": [False, True, True, False, False, False, False],
    })
    label = np.array([4, 0, 3, 0, 0, 3, 0])               # raw 4 caps to 3
    final = status is not None
    outcomes = pd.DataFrame({
        "SNAPSHOT_DATE": 20250320, "LOAN_ID": range(7),
        "WORST_DPD_SO_FAR": BAND_DPD[label], "MONTHS_OBSERVED": 6 if final else 3,
        "MONTHS_DELETED": 0, "STATE_ROWS": 7 if final else 4,
        "FINAL_CAT": label.astype(float) if final else np.nan,
    })
    ledger = ({20250320: {"status": status, "loans_unlabelled": 0, "loans_missing_months": 0}}
              if final else {})
    return preds, outcomes, ledger


def test_band_dpd_edges():
    dpd = [0, 1, 60, 61, 155, 156, 180, 181, 400]
    assert list(monitor.band_dpd(dpd)) == [0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_matured_metrics_by_hand(small_windows):
    preds, outs, ledger = _hand_cohort()
    m = monitor.compute(preds, outs, ledger, ASOF)
    c = m["matured"][20250320]
    assert c["label_status"] == "FINAL"
    assert m["quality"]["cross_checked"] == {20250320: 7}
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


def test_open_snapshot_uses_state_log_not_null_label(small_windows):
    preds, outs, ledger = _hand_cohort(status=None)          # FINAL_CAT all NULL
    m = monitor.compute(preds, outs, ledger, ASOF)
    assert not m["matured"] and 20250320 in m["interim"]
    assert m["quality"]["unmatched_by_snapshot"] == {}       # NULL label != missing row
    i = m["interim"][20250320]
    assert i["months_observed"] == 3 and i["n_short_months"] == 0
    assert i["deteriorated_rate"] == pytest.approx(3 / 7)         # loans 0, 2, 5
    assert i["cutoffs"]["top_2"]["deteriorated_rate"] == pytest.approx(1 / 2)
    assert i["rest"]["deteriorated_rate"] == pytest.approx(1 / 3)   # positions 5-7
    assert i["enrichment_rule"]["deteriorated_rate"] == pytest.approx(1 / 2)


def test_open_snapshot_without_state_outcome_is_counted_by_cause(small_windows):
    preds, outs, ledger = _hand_cohort(status=None)
    outs.loc[[1, 2], "WORST_DPD_SO_FAR"] = np.nan
    outs.loc[1, "STATE_ROWS"] = 0                          # (b) not in the state log at all
    outs.loc[2, "MONTHS_DELETED"] = 4                      # (a) rows exist, installments deleted
    df, q = monitor.prepare(preds, outs, ledger)
    assert q["no_interim_outcome"] == {20250320: {"deleted": 1, "no_state_rows": 1}}
    assert set(df["LOAN_ID"]) == {0, 3, 4, 5, 6}


def test_just_scored_snapshot_is_left_out(small_windows):
    preds, outs, ledger = _hand_cohort(status=None)
    outs["MONTHS_OBSERVED"] = 0
    m = monitor.compute(preds, outs, ledger, ASOF)
    assert not m["interim"] and m["quality"]["no_months_yet"] == [20250320]


def test_unlabelled_loans_dropped_and_compared_with_ledger(small_windows, caplog):
    preds, outs, ledger = _hand_cohort()
    outs.loc[5, "FINAL_CAT"] = np.nan
    df, q = monitor.prepare(preds, outs, ledger)
    assert 5 not in set(df["LOAN_ID"])
    assert q["labels"][20250320]["unlabelled"] == 1
    assert "ledger says 0" in caplog.text                 # ledger expected none
    ledger[20250320]["loans_unlabelled"] = 1
    caplog.clear()
    monitor.prepare(preds, outs, ledger)
    assert "ledger says" not in caplog.text


def test_final_label_cross_check(small_windows, caplog):
    preds, outs, ledger = _hand_cohort()
    bad = outs.copy()
    bad.loc[3, "WORST_DPD_SO_FAR"] = 170                  # state log says severe, label says 0
    with pytest.raises(ValueError, match="differ"):
        monitor.prepare(preds, bad, ledger)
    monitor.prepare(preds, bad, {20250320: {**ledger[20250320], "status": "LEGACY"}})  # skipped
    with pytest.raises(ValueError, match="More than 6"):
        monitor.prepare(preds, outs.assign(MONTHS_OBSERVED=7), ledger)
    _, q = monitor.prepare(preds, outs.assign(MONTHS_OBSERVED=5), ledger)
    assert q["cross_checked"] == {20250320: 0} and "could not run" in caplog.text


def test_unmatched_loans_are_counted(small_windows):
    preds, outs, ledger = _hand_cohort()
    df, q = monitor.prepare(preds, outs[outs["LOAN_ID"] != 6], ledger)
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
    preds, outs, runs, ledger = fake_frames(SNAPS)
    for snap in SNAPS[:6]:                                 # matured before go-live
        ledger[snap]["status"] = "LEGACY"
    m = monitor.compute(preds, outs, ledger, ASOF, runs)
    assert m["matured"] and m["interim"]
    assert list(monitor.history_frame(m)["label_status"][5:7]) == ["LEGACY", "FINAL"]
    assert list(monitor.history_frame(m)["snapshot"]) == sorted(m["matured"])
    text = build_report(m, tmp_path).read_text()
    assert not re.search(r"(src|href)=\"https?://", text)       # works offline
    assert "data:image/png;base64," in text
    assert (tmp_path / "charts" / "trend_recall.png").exists()
    assert "corrected label" in text and "5 of 6 mo, open" in text


def test_run_monitor_end_to_end(tmp_path):
    conn = _FakeConn(*fake_engine([20260220, 20260321, 20260723], n=3000))
    report = monitor.run_monitor(asof=ASOF, output_dir=str(tmp_path), conn=conn)
    assert report.exists()
    m = json.loads((tmp_path / str(ASOF) / "metrics.json").read_text())
    assert set(m["matured"]) == {"20260220", "20260321"} and set(m["interim"]) == {"20260723"}
    assert [r["operation_id"] for r in m["quality"]["runs"]] == ["op-20260220", "op-20260321",
                                                                 "op-20260723"]
    assert len(pd.read_csv(tmp_path / "history.csv")) == 2


def test_asof_reads_only_what_existed_then():
    tables = fake_engine([20260220, 20260321], n=500)
    # 20260321 is finalised on 2026-09-24; its run on 2026-03-24
    preds, _, runs, ledger = monitor.load_frames(_FakeConn(*tables), 20260923)
    assert set(ledger) == {20260220}
    _, _, runs, _ = monitor.load_frames(_FakeConn(*tables), 20260323)
    assert list(runs["snapshot_date"]) == [20260220]
