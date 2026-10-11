"""
The label ledger (EDP_LABEL_FINALIZATION) decides maturity: a snapshot is
mature iff it has a ledger row, a finalised snapshot may still hold NULL labels
(dropped, never imputed), and no NPZ for a mature snapshot ever holds one.
"""

import json

import numpy as np
import pandas as pd
import pytest

import project_config as config
from src.data import temporal_split
from src.data.data_loader import DataLoader, load_cached_arrays, train_cache_dir
from src.data.column_contract import FEATURE_ORDER
from src.evaluation.metrics import full_evaluation
from tests import test_order_independence as toi
from tests.test_order_independence import _FakeConn, _contract_frame, _multi_snapshot_frame

bundle_path = toi.bundle_path   # the module-scoped fixture, shared

LOAD = dict(grain="loan", feature_order=FEATURE_ORDER)


def test_unlabelled_rows_are_dropped_and_the_vintage_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = _multi_snapshot_frame(snaps=(20250131.0,))
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(float)
    df.loc[df.index[:3], config.TARGET_COL] = np.nan        # LOANS_UNLABELLED = 3

    conn = _FakeConn(df)
    instances, _ = DataLoader(conn).load_train_portfolios(use_cache=True, **LOAD)
    assert len(instances) == len(df) - 3
    assert all(i["label"] >= 0 for i in instances)

    cache = train_cache_dir(grain="loan", feature_order=FEATURE_ORDER)
    manifest = json.loads((cache / "20250131.manifest.json").read_text())
    assert manifest["label_ledger"]["20250131"]["status"] == "FINAL"

    # A cache-only run (no DB) knows the snapshot is mature from the manifest.
    temporal_split.LABEL_LEDGER.clear()
    load_cached_arrays(cache)
    assert temporal_split.filter_mature_snapshots([20250131.0]) == [20250131.0]


def test_mature_npz_refuses_unledgered_or_unlabelled(tmp_path):
    inst = [{"national_code": "c", "snapshot_date": 20250131.0, "loan_id": 1,
             "n_loans": 1, "portfolio_n_loans": 1,
             "features": np.zeros((1, 2), np.float32), "label": 0, "current_cat": 0}]
    dl, npz = DataLoader(), tmp_path / "20250131.npz"
    meta = {"snapshot": 20250131, "mature": True}

    with pytest.raises(RuntimeError, match="not in the label ledger"):
        dl._save_cache(npz, inst, ["a", "b"], 1, "k", meta=meta)

    temporal_split.register_label_ledger({20250131: {"status": "FINAL"}})
    with pytest.raises(RuntimeError, match="unlabelled"):
        dl._save_cache(npz, [{**inst[0], "label": -1}], ["a", "b"], 1, "k", meta=meta)
    dl._save_cache(npz, inst, ["a", "b"], 1, "k", meta=meta)


def test_ranking_is_split_by_vintage_only_when_mixed():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.choice([0, 1, 3], n, p=[0.7, 0.2, 0.1])
    probs = rng.dirichlet(np.ones(config.NUM_CLASSES), n)
    strata = np.zeros(n, dtype=int)

    one = full_evaluation(y, probs, strata=strata, label_status=["FINAL"] * n)
    assert "ranking_by_label_status" not in one

    mixed = np.array(["FINAL", "LEGACY"] * (n // 2), dtype=object)
    out = full_evaluation(y, probs, strata=strata, label_status=mixed)
    by = out["ranking_by_label_status"]
    assert set(by) == {"FINAL", "LEGACY"}
    assert by["FINAL"]["n_ranked"] + by["LEGACY"]["n_ranked"] == out["ranking"]["n_ranked"]
    assert "current_cat_0" in by["FINAL"]["by_current_cat"]


def test_score_dataframe_keeps_rows_with_null_label_columns(bundle_path):
    """An open snapshot read straight from the table carries NULL labels;
    scoring must ignore them, not drop every row."""
    from src.inference.predictor import score_dataframe

    df = _contract_frame(120, 4)
    df[config.TARGET_COL] = np.nan
    df["WORST_FUTURE_DPD"] = np.nan
    queue = score_dataframe(df, bundle_path)
    assert len(queue) == len(df)
    pd.testing.assert_frame_equal(
        queue, score_dataframe(_contract_frame(120, 4, with_target=False), bundle_path))
