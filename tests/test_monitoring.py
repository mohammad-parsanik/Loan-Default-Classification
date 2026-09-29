"""Monthly monitoring: cohort split, metrics on hand-checkable data, report rendering."""

import json
import re

import numpy as np
import pandas as pd
import pytest

import project_config as config
from src.monitoring import monitor
from src.monitoring.report import build_report

ASOF = 20260928


def fake_archive(snaps, n=4000, seed=0):
    """Prediction table + outcomes with real signal: severe odds rise with RISK_SCORE."""
    rng = np.random.default_rng(seed)
    preds, outs = [], []
    for s in snaps:
        cc = rng.choice(4, n, p=[0.6, 0.25, 0.1, 0.05])
        score = np.clip(rng.beta(1, 12, n) + 0.25 * (cc == 2) + 0.1 * (cc == 1), 0, 1).round(3)
        y = np.maximum(cc, np.where(rng.random(n) < score * 1.3, 3, rng.choice(3, n, p=[.8, .15, .05])))
        horizon = int((pd.Timestamp(str(s)) + pd.DateOffset(months=6)).strftime("%Y%m%d"))
        ids = np.arange(n)
        preds.append(pd.DataFrame({"SNAPSHOT_DATE": float(s), "LOAN_ID": ids,
                                   "NATIONAL_CODE": ids // 2, "CURRENT_CAT": cc, "RISK_SCORE": score,
                                   "RULE_FLAG": np.where(cc >= 3, "ALREADY_SEVERE", "")}))
        outs.append(pd.DataFrame({"SNAPSHOT_DATE": s, "LOAN_ID": ids, "WORST_FUTURE_CAT": y,
                                  "LABEL_HORIZON_DATE": horizon,
                                  "REMAINING_AMNT": rng.lognormal(18, 1, n)}))
    return pd.concat(preds, ignore_index=True), pd.concat(outs, ignore_index=True)


@pytest.fixture
def small_windows(monkeypatch):
    # K = 2 for "1_day", 4 for "1_week" — small enough to check by hand.
    monkeypatch.setattr(config, "API_RATE_PER_HOUR", 1)
    monkeypatch.setattr(config, "RANKING_REF_WINDOWS", {"1_day": 2, "1_week": 4})
    monkeypatch.setattr(config, "PRED_ARCHIVE_TABLE", "PRED_TABLE")


def _hand_cohort():
    #            loan: 0    1    2    3    4    5    6    7
    preds = pd.DataFrame({
        "SNAPSHOT_DATE": 20250320, "LOAN_ID": range(8), "NATIONAL_CODE": range(8),
        "CURRENT_CAT": [0, 0, 1, 0, 0, 0, 0, 3],
        "RISK_SCORE":  [.9, .8, .7, .6, .5, .4, .3, .99],
    })
    outcomes = pd.DataFrame({
        "SNAPSHOT_DATE": 20250320, "LOAN_ID": range(8),
        "WORST_FUTURE_CAT": [4, 0, 3, 0, 0, 3, 0, 4],       # raw 4 caps to 3
        "LABEL_HORIZON_DATE": 20250920,
        "REMAINING_AMNT": [100, 1, 50, 1, 1, 50, 1, 1],
    })
    return preds, outcomes


def test_matured_metrics_by_hand(small_windows):
    m = monitor.compute(*_hand_cohort(), ASOF)
    c = m["matured"][20250320]
    # loan 7 (cat 3) is carved; ranked = 7 loans, severe = {0, 2, 5}
    assert c["n_ranked"] == 7 and c["n_severe"] == 3
    assert c["at_1_day"]["recall"] == pytest.approx(1 / 3)       # top-2 = {0, 1}
    assert c["at_1_week"]["recall"] == pytest.approx(2 / 3)      # top-4 = {0, 1, 2, 3}
    assert c["at_1_week"]["exposure_caught"] == pytest.approx(150 / 200)
    lo, hi = c["at_1_week"]["recall_ci"]
    assert lo < 2 / 3 < hi
    assert c["predicted_severe"] == pytest.approx(sum([.9, .8, .7, .6, .5, .4, .3]))
    q = c["queue_by_cat"]
    assert q[0]["recall_1_week"] == pytest.approx(1 / 2)     # cat-0 severe {0, 5}; top-4 has 0
    assert q[1]["recall_1_week"] == 1.0 and q[1]["share_of_top_1_week"] == pytest.approx(1 / 4)
    top = c["migration"]["top_1_week"]
    assert sum(sum(r.values()) for r in top.values()) == 4
    assert sum(sum(r.values()) for r in c["migration"]["rest"].values()) == 3


def test_immature_rows_stay_out_of_matured(small_windows):
    preds, outs = _hand_cohort()
    outs["LABEL_HORIZON_DATE"] = ASOF + 1
    m = monitor.compute(preds, outs, ASOF)
    assert not m["matured"] and 20250320 in m["interim"]
    i = m["interim"][20250320]
    # worse than when scored: loans 0, 2, 5 (loan 7 is carved, 3 -> 3 is not "worse")
    assert i["deteriorated_rate"] == pytest.approx(3 / 7)
    assert i["top_1_day"]["deteriorated_rate"] == pytest.approx(1 / 2)
    assert i["rest"]["deteriorated_rate"] == pytest.approx(1 / 3)   # positions 5-7


def test_duplicates_and_unmatched_are_counted(small_windows):
    preds, outs = _hand_cohort()
    dup = preds.iloc[[1]].assign(RISK_SCORE=0.95)
    preds = pd.concat([preds, dup, preds.iloc[[2]]], ignore_index=True)
    outs = outs[outs["LOAN_ID"] != 6]
    df, q = monitor.prepare(preds, outs, ASOF)
    assert q["n_duplicates_dropped"] == 2
    assert q["n_duplicate_keys_with_different_scores"] == 1
    assert df.loc[df["LOAN_ID"] == 1, "RISK_SCORE"].item() == 0.95   # higher score kept
    assert q["unmatched_by_snapshot"] == {20250320: 1}
    assert 6 not in set(df["LOAN_ID"])


def test_missing_required_column_raises(small_windows):
    preds, outs = _hand_cohort()
    with pytest.raises(ValueError, match="RISK_SCORE"):
        monitor.prepare(preds.drop(columns="RISK_SCORE"), outs, ASOF)


def test_psi_zero_on_identical_and_positive_on_shift():
    x = np.random.default_rng(1).random(5000)
    assert monitor.psi(x, x) == pytest.approx(0, abs=1e-9)
    assert monitor.psi(x, x ** 3) > 0.25


def test_report_is_self_contained(tmp_path):
    snaps = [20250321, 20250421, 20250522, 20250621, 20250722, 20250823, 20250922,
             20251023, 20251122, 20251222, 20260121, 20260220, 20260321, 20260421,
             20260522, 20260621, 20260723, 20260823]
    preds, outs = fake_archive(snaps)
    m = monitor.compute(preds, outs, ASOF)
    assert m["matured"] and m["interim"]
    hist = monitor.history_frame(m)
    assert list(hist["snapshot"]) == sorted(m["matured"])
    path = build_report(m, tmp_path)
    text = path.read_text()
    assert not re.search(r"(src|href)=\"https?://", text)       # works offline
    assert "data:image/png;base64," in text
    assert (tmp_path / "charts" / "trend_recall.png").exists()


class _FakeConn:
    def __init__(self, preds, outs):
        self.preds, self.outs = preds, outs

    def read_sql(self, query, params=None):
        if params is None:
            return self.preds
        return self.outs[self.outs["SNAPSHOT_DATE"] == params[0]]


def test_run_monitor_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRED_ARCHIVE_TABLE", "PRED_TABLE")
    preds, outs = fake_archive([20260220, 20260321, 20260723], n=3000)
    report = monitor.run_monitor(asof=ASOF, output_dir=str(tmp_path), conn=_FakeConn(preds, outs))
    assert report.exists()
    m = json.loads((tmp_path / str(ASOF) / "metrics.json").read_text())
    assert set(m["matured"]) == {"20260220", "20260321"} and set(m["interim"]) == {"20260723"}
    assert len(pd.read_csv(tmp_path / "history.csv")) == 2
