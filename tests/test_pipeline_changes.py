"""
Tests for the leakage/evaluation changes (July 2026):
  - truncation sorts by current DPD_DAYS, not the label WORST_FUTURE_DPD
  - prediction path works without WORST_FUTURE_* columns
  - current_cat stratum metadata
  - customer-disjoint validation split
  - expected-cost decision rule
  - per-class isotonic calibration
  - IV binning no longer zeroes out skewed/binary features
"""

import numpy as np
import pandas as pd

import project_config as config
from src.data.data_loader import DataLoader
from src.data.temporal_split import split_train_by_customer
from src.evaluation.calibration import PerClassIsotonicCalibrator
from src.evaluation.decision import cost_decisions, expected_costs, risk_scores
from src.evaluation.metrics import compute_metrics, stratified_metrics


# ── data_loader ───────────────────────────────────────────────────────────────

def _make_df(with_target: bool = True) -> pd.DataFrame:
    """One customer, one snapshot, 3 loans (worst-future ≠ worst-current)."""
    df = pd.DataFrame({
        "LOAN_ID":         [1, 2, 3],
        "CONTRACT_NUMBER": ["c1", "c2", "c3"],
        "NATIONAL_CODE":   ["A"] * 3,
        "SNAPSHOT_DATE":   [20250101.0] * 3,
        # Loan 1: currently worst (DPD 90) but fine in the future.
        # Loan 3: currently clean but worst in the future (the leak bait).
        "DPD_DAYS":        [90.0, 30.0, 0.0],
        "LOAN_CATEGORY":   [2.0, 1.0, 0.0],
        "OTHER_FEAT":      [10.0, 20.0, 30.0],
    })
    if with_target:
        df["WORST_FUTURE_DPD"] = [0.0, 10.0, 200.0]
        df["WORST_FUTURE_CAT"] = [0.0, 1.0, 4.0]
    return df


def test_truncation_keeps_currently_worst_loans_not_future_worst():
    instances, feature_cols = DataLoader().process_raw_data(_make_df(), max_loans=2)
    assert len(instances) == 1
    inst = instances[0]

    dpd_idx = feature_cols.index("DPD_DAYS")
    kept_dpds = sorted(inst["features"][:, dpd_idx].tolist(), reverse=True)
    # Must keep the two currently-worst loans (90, 30) — NOT the future-worst
    # loan (DPD 0, WORST_FUTURE_DPD 200) the old label-sort would have kept.
    assert kept_dpds == [90.0, 30.0]

    # Label still computed over ALL loans (max future cat 4 → capped to 2)
    assert inst["label"] == config.NUM_CLASSES - 1
    # Current stratum: worst current LOAN_CATEGORY = 2
    assert inst["current_cat"] == 2
    assert inst["n_loans"] == 3


def test_prediction_table_without_labels_is_processed():
    # Old code crashed here: sort key WORST_FUTURE_DPD doesn't exist on
    # EDP_Feature_pred, and neither does TARGET_COL.
    instances, feature_cols = DataLoader().process_raw_data(
        _make_df(with_target=False), max_loans=2
    )
    assert len(instances) == 1
    assert instances[0]["label"] == -1
    assert instances[0]["current_cat"] == 2
    assert "WORST_FUTURE_DPD" not in feature_cols


# ── customer-disjoint split ───────────────────────────────────────────────────

def test_customer_split_is_disjoint_and_stable():
    rng = np.random.default_rng(0)
    codes = [f"cust_{i}" for i in range(2000)]
    # Each customer appears in 3 "snapshots"
    instances = [
        {"national_code": c, "snapshot_date": s, "label": int(rng.integers(3))}
        for c in codes for s in (1.0, 2.0, 3.0)
    ]

    train, val = split_train_by_customer(instances, val_fraction=0.2, seed=42)

    train_codes = {i["national_code"] for i in train}
    val_codes   = {i["national_code"] for i in val}
    assert not train_codes & val_codes          # fully disjoint customers
    assert 0.15 < len(val) / len(instances) < 0.25

    # Deterministic across calls (md5-based, not hash()-based)
    train2, val2 = split_train_by_customer(instances, val_fraction=0.2, seed=42)
    assert {i["national_code"] for i in val2} == val_codes


# ── decision rule ─────────────────────────────────────────────────────────────

def test_cost_decisions_flag_risky_non_modal_customers():
    probs = np.array([
        [0.45, 0.20, 0.35],   # argmax says 0, expected cost says 2
        [0.90, 0.08, 0.02],   # safely 0 either way
        [0.05, 0.15, 0.80],   # clearly 2
    ])
    assert probs.argmax(axis=1).tolist() == [0, 0, 2]
    assert cost_decisions(probs).tolist() == [2, 0, 2]

    C = np.asarray(config.COST_MATRIX)
    assert np.allclose(expected_costs(probs), probs @ C)
    # Risk score = expected cost of doing nothing (predicting 0)
    assert np.allclose(risk_scores(probs), probs @ C[:, 0])


# ── calibration ───────────────────────────────────────────────────────────────

def test_calibrator_outputs_distributions_and_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.integers(0, 3, n)
    probs = np.full((n, 3), 0.1)
    probs[np.arange(n), y] = 0.8
    flip = rng.random(n) < 0.3
    y_obs = np.where(flip, (y + 1) % 3, y)

    cal = PerClassIsotonicCalibrator().fit(probs, y_obs)
    out = cal.transform(probs)

    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all()
    top = out[np.arange(n), probs.argmax(axis=1)]
    assert abs(top.mean() - 0.7) < 0.05     # pulled from 0.8 to true 70%


# ── metrics ───────────────────────────────────────────────────────────────────

def test_stratified_metrics_and_recall_labels_on_missing_class():
    # Stratum "0" contains no true class-2 → recall indices must not shift
    y_true = np.array([0, 1, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    strata = np.array([0, 0, 0, 0, 1, 1])

    m = compute_metrics(y_true[:4], y_pred[:4])
    assert m["recall_class_2"] == 0.0       # absent class, not misindexed

    sm = stratified_metrics(y_true, y_pred, strata)
    assert sm["current_cat_0"]["n"] == 4
    assert sm["current_cat_1"]["n"] == 2
    assert sm["current_cat_1"]["recall_class_2"] == 0.5


# ── IV binning ────────────────────────────────────────────────────────────────

def test_iv_binning_handles_skewed_features():
    from explore_iv_woe import compute_woe_iv_binary

    rng = np.random.default_rng(0)
    n = 50_000

    # Predictive binary flag with only 10% positives — old code returned 0.0
    flag = (rng.random(n) < 0.10).astype(float)
    target = ((flag == 1) & (rng.random(n) < 0.6)) | ((flag == 0) & (rng.random(n) < 0.1))
    _, iv = compute_woe_iv_binary(flag, target.astype(int), n_bins=10)
    assert iv > 0.1, f"binary flag IV should be material, got {iv}"

    # Mode-dominated continuous (90% zeros), informative tail
    cont = np.where(rng.random(n) < 0.9, 0.0, rng.exponential(50, n))
    target2 = (cont > 30) & (rng.random(n) < 0.7)
    _, iv2 = compute_woe_iv_binary(cont, target2.astype(int), n_bins=10)
    assert iv2 > 0.1, f"skewed continuous IV should be material, got {iv2}"

    # Truly constant feature → IV exactly 0 (correct behaviour preserved)
    _, iv3 = compute_woe_iv_binary(np.zeros(n), target.astype(int), n_bins=10)
    assert iv3 == 0.0
