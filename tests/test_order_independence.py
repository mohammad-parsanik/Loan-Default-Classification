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
