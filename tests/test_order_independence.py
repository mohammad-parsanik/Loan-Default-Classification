"""
Order-independence regression tests.

One idea, four applications: permute an input, assert the output is unchanged.

The failure these guard against is silent. Feature identity used to be
positional end to end — `SELECT *` decided the column order, and every
preprocessing statistic was keyed by integer index — so a reordered source
applied the wrong median, the wrong clip bounds and the wrong scaler to each
column, at identical width, with no error raised anywhere. Row order was
load-bearing in the same way: XGBoost's `subsample`/`colsample_bytree` draw by
index, and the queue's tie-break was whatever order SQL Server returned.

Each test here fails if its corresponding change is reverted.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import project_config as config
from src.data.column_contract import BINARY_FEATURES, FEATURE_ORDER
from src.data.data_loader import DataLoader


# ── fixtures ──────────────────────────────────────────────────────────────────

def _contract_frame(n: int = 400, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    """
    A frame with the real column contract's shape: all 64 features plus the
    meta columns, in contract order. Values are synthetic but typed
    plausibly — categories 0-4, binary flags 0/1, sentinels where the feed
    carries them — so the strict projection path is exercised for real.
    """
    rng = np.random.default_rng(seed)
    cat = rng.choice([0, 1, 2, 3], n, p=[0.55, 0.25, 0.12, 0.08]).astype(float)

    cols: dict = {}
    for name in FEATURE_ORDER:
        if name == "LOAN_CATEGORY":
            cols[name] = cat
        elif name == "DPD_DAYS":
            cols[name] = cat * 50 + rng.normal(0, 3, n)
        elif name in BINARY_FEATURES:
            cols[name] = rng.integers(0, 2, n).astype(float)
        elif name.startswith("DAYS_SINCE_LAST"):
            cols[name] = np.where(rng.random(n) < 0.4, 99999.0, rng.integers(0, 400, n))
        else:
            cols[name] = rng.normal(size=n)

    df = pd.DataFrame(cols)
    df.insert(0, config.ID_COL, np.arange(n))
    df.insert(1, config.SNAPSHOT_COL, 20250131.0)
    df[config.CUSTOMER_COL] = [f"cust_{seed}_{i // 2}" for i in range(n)]   # ~2 loans/customer
    df[config.CONTRACT_COL] = [f"k_{seed}_{i}" for i in range(n)]
    df[config.HORIZON_COL] = 20250731
    if with_target:
        label = np.minimum(cat + rng.choice([0, 0, 1, 2], n), 3)
        df["WORST_FUTURE_DPD"] = label * 60.0
        df[config.TARGET_COL] = label
    return df


def _instances_equal(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["national_code"] == y["national_code"]
        assert x["snapshot_date"] == y["snapshot_date"]
        assert x["loan_id"] == y["loan_id"]
        assert x["label"] == y["label"]
        assert x["current_cat"] == y["current_cat"]
        assert x["n_loans"] == y["n_loans"]
        assert x["portfolio_n_loans"] == y["portfolio_n_loans"]
        np.testing.assert_array_equal(x["features"], y["features"])


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    """A real (tiny) trained bundle: scaler + multiclass arm + calibrator."""
    import joblib

    from src.baselines.aggregated_xgboost import ARM_BUILDERS, build_features
    from src.data.preprocessing import create_preprocessing_pipeline
    from src.evaluation.calibration import StratifiedCalibrator
    from src.inference.model_loader import build_arm_bundle

    dl = DataLoader()
    train, feats = dl.process_raw_data(_contract_frame(600, 1), feature_order=FEATURE_ORDER)
    val, _ = dl.process_raw_data(_contract_frame(300, 2), feature_order=FEATURE_ORDER)

    scaler = create_preprocessing_pipeline(feats, BINARY_FEATURES)
    scaler.fit([i["features"] for i in train])
    for split in (train, val):
        for inst, x in zip(split, scaler.transform([i["features"] for i in split])):
            inst["features"] = x

    x_tr, y_tr = build_features(train)
    x_v, y_v = build_features(val)
    cat_tr = np.array([i["current_cat"] for i in train])
    cat_v = np.array([i["current_cat"] for i in val])
    arm = ARM_BUILDERS["multiclass"]()
    arm.train(x_tr, y_tr, cat_tr, x_v, y_v, cat_v)
    cal = StratifiedCalibrator(min_stratum_n=50).fit(arm.predict_proba(x_v, cat_v), y_v, cat_v)

    path = tmp_path_factory.mktemp("bundle") / "model_bundle.pkl"
    joblib.dump(build_arm_bundle(scaler, arm, cal, 2, feats), path)
    return path


# ── 1. rows ───────────────────────────────────────────────────────────────────

def test_row_permutation_yields_identical_instances():
    """
    A shuffled copy of the same rows must produce the same instances. The sort
    inside process_raw_data ends on LOAN_ID, which the feed guarantees unique
    per snapshot, so no tie can survive to inherit arrival order.
    """
    df = _contract_frame(300, 3)
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    dl = DataLoader()
    a, _ = dl.process_raw_data(df, feature_order=FEATURE_ORDER)
    b, _ = dl.process_raw_data(shuffled, feature_order=FEATURE_ORDER)
    _instances_equal(a, b)


def test_row_permutation_survives_massive_dpd_ties():
    """
    The hostile version: DPD_DAYS identical on every row, so the secondary
    sort key is useless and only the LOAN_ID tie-break can save the ordering.
    """
    df = _contract_frame(200, 4)
    df["DPD_DAYS"] = 0.0
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)

    dl = DataLoader()
    a, _ = dl.process_raw_data(df, grain="portfolio", max_loans=2,
                               feature_order=FEATURE_ORDER)
    b, _ = dl.process_raw_data(shuffled, grain="portfolio", max_loans=2,
                               feature_order=FEATURE_ORDER)
    _instances_equal(a, b)


# ── 2. columns ────────────────────────────────────────────────────────────────

def test_column_permutation_yields_identical_instances():
    """
    The M1 regression test, and the most valuable of the four. Reversed
    column order — not nudged — so a partial fix cannot pass.
    """
    df = _contract_frame(250, 5)
    reversed_cols = df[list(df.columns)[::-1]]
    assert list(reversed_cols.columns) != list(df.columns)

    dl = DataLoader()
    a, feats_a = dl.process_raw_data(df, feature_order=FEATURE_ORDER)
    b, feats_b = dl.process_raw_data(reversed_cols, feature_order=FEATURE_ORDER)
    assert feats_a == feats_b == list(FEATURE_ORDER)
    _instances_equal(a, b)


def test_column_permutation_without_explicit_order():
    """
    Even with no order supplied, the projection is a function of the column
    SET: names are sorted into contract order rather than taken as they came.
    """
    df = _contract_frame(120, 6)
    dl = DataLoader()
    a, feats_a = dl.process_raw_data(df)
    b, feats_b = dl.process_raw_data(df[list(df.columns)[::-1]])
    assert feats_a == feats_b
    _instances_equal(a, b)


# ── 3. scores and the queue ───────────────────────────────────────────────────

def test_column_permutation_yields_identical_queue(bundle_path):
    from src.inference.predictor import score_dataframe

    df = _contract_frame(200, 8, with_target=False)
    q_a = score_dataframe(df, bundle_path)
    q_b = score_dataframe(df[list(df.columns)[::-1]], bundle_path)
    pd.testing.assert_frame_equal(q_a, q_b)
    assert q_a["RISK_RANK"].notna().any()


def test_row_permutation_yields_identical_queue(bundle_path):
    """
    Including RISK_RANK. Calibrated scores tie in blocks (isotonic regression
    is a step function), so without the explicit NATIONAL_CODE/LOAN_ID
    tie-break the queue positions would follow arrival order.
    """
    from src.inference.predictor import score_dataframe

    df = _contract_frame(200, 9, with_target=False)
    q_a = score_dataframe(df, bundle_path)
    q_b = score_dataframe(df.sample(frac=1.0, random_state=3).reset_index(drop=True),
                          bundle_path)
    pd.testing.assert_frame_equal(q_a, q_b)


def test_queue_tie_break_is_actually_exercised(bundle_path):
    """
    Guard the guard: if the fixture produced no tied RISK_SCOREs, the two
    tests above would pass on a sort that has no tie-break at all.
    """
    from src.inference.predictor import score_dataframe

    q = score_dataframe(_contract_frame(200, 9, with_target=False), bundle_path)
    ranked = q[q["RULE_FLAG"] == ""]
    assert ranked["RISK_SCORE"].duplicated().any(), "no tied scores — test is vacuous"


# ── 4. missing / extra columns ────────────────────────────────────────────────

def test_missing_feature_raises_and_names_it():
    df = _contract_frame(50, 10).drop(columns=["OVERDUE_RATIO"])
    with pytest.raises(KeyError, match="OVERDUE_RATIO"):
        DataLoader().process_raw_data(df, feature_order=FEATURE_ORDER)


def test_unknown_column_is_dropped_with_a_warning(caplog):
    """
    The normal path after the feed appends a feature: an already-trained model
    keeps scoring on the 64 columns it knows.
    """
    df = _contract_frame(80, 11)
    df["JUNK_COL_72"] = 1.0

    dl = DataLoader()
    with caplog.at_level("WARNING"):
        b, feats = dl.process_raw_data(df, feature_order=FEATURE_ORDER)
    assert "JUNK_COL_72" in caplog.text
    assert "JUNK_COL_72" not in feats

    a, _ = dl.process_raw_data(df.drop(columns="JUNK_COL_72"), feature_order=FEATURE_ORDER)
    _instances_equal(a, b)


def test_appended_column_does_not_change_the_queue(bundle_path):
    from src.inference.predictor import score_dataframe

    df = _contract_frame(150, 12, with_target=False)
    q_a = score_dataframe(df, bundle_path)
    q_b = score_dataframe(df.assign(JUNK_COL_72=3.14), bundle_path)
    pd.testing.assert_frame_equal(q_a, q_b)


# ── the transformers refuse a mismatched list ─────────────────────────────────

def test_fitted_pipeline_rejects_a_permuted_feature_list():
    from src.data.preprocessing import (
        FeatureContractError,
        assert_pipeline_features,
        create_preprocessing_pipeline,
    )

    feats = list(FEATURE_ORDER)
    pipe = create_preprocessing_pipeline(feats, BINARY_FEATURES)
    pipe.fit([np.random.rand(1, len(feats)).astype(np.float32) for _ in range(40)])

    assert_pipeline_features(pipe, feats)          # the fitted list is fine
    with pytest.raises(FeatureContractError, match="DIFFERENT ORDER"):
        assert_pipeline_features(pipe, feats[::-1])


def test_sentinel_columns_are_exempt_from_clip_and_scale():
    """
    99999 means "never reached this band" and sits at the opposite end of the
    axis from 0, "in the band right now". Percentile-clipping would rewrite
    never-delinquent as cleared-long-ago; scaling would move it off the value
    the tree splits on. Both must leave the column alone.
    """
    from src.data.preprocessing import create_preprocessing_pipeline

    feats = list(FEATURE_ORDER)
    col = feats.index("DAYS_SINCE_LAST_DPD")
    rng = np.random.default_rng(0)
    # "never" is a MINORITY here, which is exactly the case where p99 sits far
    # below the sentinel and clipping is destructive rather than a no-op.
    X = [rng.integers(0, 300, (1, len(feats))).astype(np.float32) for _ in range(400)]
    for row in X[:8]:
        row[0, col] = 99999.0

    pipe = create_preprocessing_pipeline(feats, BINARY_FEATURES)
    out = pipe.fit(X).transform(X)
    assert sum(1 for row in out if row[0, col] == 99999.0) == 8


# ── the cache notices column identity ─────────────────────────────────────────

def test_cache_key_changes_with_the_feature_list():
    from src.data.data_loader import _cache_key

    base = _cache_key("x", "loan", FEATURE_ORDER)
    assert _cache_key("x", "loan", FEATURE_ORDER) == base
    assert _cache_key("x", "loan", list(FEATURE_ORDER)[::-1]) != base       # reorder
    assert _cache_key("x", "loan", list(FEATURE_ORDER) + ["NEW_72"]) != base  # addition


# ── the queue sort itself, fed out of order ───────────────────────────────────

def test_queue_is_identical_when_instances_arrive_permuted(bundle_path):
    """
    The tests above feed score_instances through process_raw_data, which has
    already put rows in canonical order — so they exercise the loader's sort,
    not the queue's own tie-break. This one hands score_instances a shuffled
    instance list directly, which is what a caller assembling instances from
    several sources does.
    """
    from src.inference.model_loader import ModelLoader
    from src.inference.predictor import score_instances

    scorer, calibrator, features = ModelLoader(bundle_path).load_pipeline()
    instances, _ = DataLoader().process_raw_data(
        _contract_frame(200, 13, with_target=False), feature_order=features
    )
    shuffled = list(np.random.default_rng(5).permutation(np.array(instances, dtype=object)))

    q_a = score_instances(instances, scorer, calibrator)
    q_b = score_instances(shuffled, scorer, calibrator)
    pd.testing.assert_frame_equal(q_a, q_b)


def test_ranking_metrics_are_identical_under_permutation():
    """
    recall@K and lift@K inherit the ordering. With ties on the score and no
    tie-break they follow input order; with LOAN_ID as the tie-break they are
    a function of the data.
    """
    from src.evaluation.ranking import ranking_metrics

    rng = np.random.default_rng(2)
    n = 500
    y = rng.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.1, 0.1])
    strata = np.zeros(n, dtype=int)
    scores = np.round(rng.random(n), 2)          # heavy ties, on purpose
    loan_id = np.arange(n)

    perm = rng.permutation(n)
    a = ranking_metrics(y, scores, strata, tie_break=loan_id)
    b = ranking_metrics(y[perm], scores[perm], strata[perm], tie_break=loan_id[perm])
    for window in config.RANKING_REF_WINDOWS:
        assert a[f"at_{window}"] == b[f"at_{window}"]


# ── per-snapshot streaming load & cache ───────────────────────────────────────

class _FakeConn:
    """Serves a multi-snapshot frame the way MSSQLConnector would."""

    def __init__(self, df: pd.DataFrame, etl_run: str = "20260901"):
        self.df = df
        self.etl_run = etl_run
        self.queries: list = []

    def get_available_snapshots(self):
        return sorted(self.df[config.SNAPSHOT_COL].unique().tolist())

    def get_label_horizons(self):
        return {int(s): int(h) for s, h in
                self.df[[config.SNAPSHOT_COL, config.HORIZON_COL]]
                .drop_duplicates().itertuples(index=False)}

    def get_etl_runs(self, limit: int = 24):
        return pd.DataFrame({"snapshot_date": [self.etl_run], "status": ["SUCCESS"],
                             "last_step": ["done"], "error_message": [None]})

    def _rows(self, snaps):
        self.queries.append(sorted(int(s) for s in snaps))
        keep = self.df[config.SNAPSHOT_COL].isin([float(d) for d in snaps])
        return self.df.loc[keep].copy()

    def load_training_data(self, snapshot_dates=None):
        return self._rows(snapshot_dates or self.get_available_snapshots())

    def load_prediction_data(self, snapshot_date=None):
        return self._rows([snapshot_date])

    def close(self):
        pass


def _multi_snapshot_frame(snaps=(20250131.0, 20250228.0, 20250331.0),
                          horizon: int = 20250731) -> pd.DataFrame:
    """Same customers in every snapshot — so customer-major and snapshot-major
    orderings genuinely differ, which is what these tests are about."""
    parts = []
    for snap in snaps:
        part = _contract_frame(120, seed=0)
        part[config.SNAPSHOT_COL] = snap
        part[config.HORIZON_COL] = horizon
        parts.append(part)
    # Shuffled on the way in: the loader's answer must not depend on it.
    return (pd.concat(parts, ignore_index=True)
              .sample(frac=1.0, random_state=7).reset_index(drop=True))


def _snapshot_major(instances: list[dict]) -> list[dict]:
    """process_raw_data's own order, re-sorted snapshot-major. `sorted` is
    stable, so the within-snapshot key (customer, DPD desc, LOAN_ID) survives."""
    return sorted(instances, key=lambda i: i["snapshot_date"])


@pytest.mark.parametrize("grain", ["loan", "portfolio"])
def test_streaming_load_matches_whole_table_load(grain):
    """
    The loader reads one snapshot at a time (a single SELECT * is ~43M rows at
    the >=7B population). The instances must be exactly those a whole-table
    load produces, in (SNAPSHOT, CUSTOMER, DPD desc, LOAN_ID) order — a
    function of the data, never of the order rows arrived in.
    """
    df = _multi_snapshot_frame()
    whole, _ = DataLoader().process_raw_data(df, grain=grain, feature_order=FEATURE_ORDER)

    conn = _FakeConn(df)
    got, _ = DataLoader(conn).load_train_portfolios(
        use_cache=False, grain=grain, feature_order=FEATURE_ORDER
    )

    # One query per snapshot, never one for the whole table.
    assert conn.queries == [[20250131], [20250228], [20250331]]
    _instances_equal(got, _snapshot_major(whole))


def test_streaming_load_is_row_order_independent():
    """Same rows, different arrival order, byte-identical instances."""
    df = _multi_snapshot_frame()
    a, _ = DataLoader(_FakeConn(df)).load_train_portfolios(
        use_cache=False, grain="loan", feature_order=FEATURE_ORDER)
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    b, _ = DataLoader(_FakeConn(shuffled)).load_train_portfolios(
        use_cache=False, grain="loan", feature_order=FEATURE_ORDER)
    _instances_equal(a, b)


def test_streaming_load_skips_immature_snapshots():
    """Immature snapshots are dropped server-side, not loaded and discarded."""
    df = _multi_snapshot_frame()
    future = df[df[config.SNAPSHOT_COL] == 20250331.0].copy()
    future[config.SNAPSHOT_COL] = 20260831.0
    future[config.HORIZON_COL] = 20270228          # horizon in the future
    df = pd.concat([df, future], ignore_index=True)

    conn = _FakeConn(df)
    instances, _ = DataLoader(conn).load_train_portfolios(
        use_cache=False, grain="loan", feature_order=FEATURE_ORDER)

    assert [20260831] not in conn.queries
    assert 20260831.0 not in {i["snapshot_date"] for i in instances}


def test_mature_snapshot_cache_is_permanent_and_incremental(tmp_path, monkeypatch):
    """
    A matured snapshot never changes upstream, so its NPZ is reused forever;
    when a new one matures only THAT snapshot is read from the DB. This is the
    whole point of caching per snapshot rather than per table.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = _multi_snapshot_frame()

    conn = _FakeConn(df)
    first, _ = DataLoader(conn).load_train_portfolios(
        use_cache=True, grain="loan", feature_order=FEATURE_ORDER)
    assert conn.queries == [[20250131], [20250228], [20250331]]

    # Second run, same snapshots: served entirely from disk.
    conn2 = _FakeConn(df)
    second, _ = DataLoader(conn2).load_train_portfolios(
        use_cache=True, grain="loan", feature_order=FEATURE_ORDER)
    assert conn2.queries == []
    _instances_equal(second, first)

    # A fourth snapshot matures: exactly one new DB read.
    extra = _contract_frame(120, seed=0)
    extra[config.SNAPSHOT_COL] = 20250430.0
    extra[config.HORIZON_COL] = 20250731
    conn3 = _FakeConn(pd.concat([df, extra], ignore_index=True))
    grown, _ = DataLoader(conn3).load_train_portfolios(
        use_cache=True, grain="loan", feature_order=FEATURE_ORDER)
    assert conn3.queries == [[20250430]]
    assert len(grown) == len(first) + 120


def test_immature_pred_cache_is_rebuilt_on_a_new_etl_run(tmp_path, monkeypatch):
    """
    The ETL rewrites the newest 7 snapshots every month, so a cached immature
    snapshot is only good for the ETL cycle that produced it. Reusing it past
    that serves last month's rows under this month's snapshot date.
    """
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = _multi_snapshot_frame(snaps=(20260831.0,), horizon=20270228)  # immature

    conn = _FakeConn(df, etl_run="20260901")
    DataLoader(conn).load_pred_portfolios(20260831, use_cache=True, grain="loan",
                                          feature_order=FEATURE_ORDER)
    assert conn.queries == [[20260831]]

    # Same ETL cycle -> cache hit.
    conn2 = _FakeConn(df, etl_run="20260901")
    DataLoader(conn2).load_pred_portfolios(20260831, use_cache=True, grain="loan",
                                           feature_order=FEATURE_ORDER)
    assert conn2.queries == []

    # Next monthly load -> the snapshot was recomputed upstream, re-read it.
    conn3 = _FakeConn(df, etl_run="20261001")
    DataLoader(conn3).load_pred_portfolios(20260831, use_cache=True, grain="loan",
                                           feature_order=FEATURE_ORDER)
    assert conn3.queries == [[20260831]]


def test_load_cached_arrays_rebases_offsets_across_snapshots(tmp_path, monkeypatch):
    """
    The diagnostics read the cache as flat arrays. Each snapshot's NPZ has its
    own offsets starting at 0, so joining them requires shifting by the running
    loan count — get that wrong and every portfolio after the first snapshot
    slices the wrong rows, silently. Portfolio grain (variable group sizes) is
    the case that can actually be wrong.
    """
    from src.data.data_loader import load_cached_arrays, train_cache_dir

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    df = _multi_snapshot_frame()

    instances, _ = DataLoader(_FakeConn(df)).load_train_portfolios(
        use_cache=True, grain="portfolio", feature_order=FEATURE_ORDER)

    arrays, feat_cols = load_cached_arrays(train_cache_dir(grain="portfolio"))
    assert feat_cols == FEATURE_ORDER
    assert len(arrays["offsets"]) == len(instances) + 1
    assert arrays["offsets"][0] == 0
    assert arrays["offsets"][-1] == arrays["features_flat"].shape[0] == len(df)

    flat, offsets = arrays["features_flat"], arrays["offsets"]
    for i, inst in enumerate(instances):
        np.testing.assert_array_equal(flat[offsets[i]:offsets[i + 1]], inst["features"])
