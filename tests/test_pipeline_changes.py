"""
Tests for the leakage/evaluation changes (July 2026):
  - truncation sorts by current DPD_DAYS, not the label WORST_FUTURE_DPD
  - prediction path works without WORST_FUTURE_* columns
  - current_cat stratum metadata
  - customer-disjoint validation split
  - expected-cost decision rule
  - per-class isotonic calibration
  - IV binning no longer zeroes out skewed/binary features
  - immature-snapshot rows excluded from label-derived diagnostics
  - NUM_CLASSES 3 -> 4 (raw cats 3-4 collapse into a new worst class)
  - single-file model_bundle.pkl deployment artifact round-trips correctly
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import project_config as config
from src.data.data_loader import DataLoader
from src.data.temporal_split import filter_mature_snapshots, split_train_by_customer
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

    # Label still computed over ALL loans (max future cat 4 → capped to worst class)
    assert inst["label"] == config.NUM_CLASSES - 1
    # Current stratum: worst current LOAN_CATEGORY = 2
    assert inst["current_cat"] == 2
    assert inst["n_loans"] == 3


def test_4class_capping_collapses_only_3_and_4():
    # Raw cats 0/1/2 must map 1:1; only 3 and 4 collapse into the new worst
    # class (config.NUM_CLASSES - 1 == 3).
    base = {
        "CONTRACT_NUMBER": "c1",
        "DPD_DAYS": 0.0,
        "LOAN_CATEGORY": 0.0,
        "OTHER_FEAT": 1.0,
    }
    for raw_cat, expected_label in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3)]:
        df = pd.DataFrame([{
            **base,
            "LOAN_ID": raw_cat,
            "NATIONAL_CODE": f"cust_{raw_cat}",
            "SNAPSHOT_DATE": 20250101.0,
            "WORST_FUTURE_DPD": 0.0,
            "WORST_FUTURE_CAT": float(raw_cat),
        }])
        instances, _ = DataLoader().process_raw_data(df, max_loans=1)
        assert instances[0]["label"] == expected_label, (
            f"raw cat {raw_cat} should map to label {expected_label}"
        )


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
        {"national_code": c, "snapshot_date": s, "label": int(rng.integers(config.NUM_CLASSES))}
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
        [0.45, 0.20, 0.30, 0.05],   # argmax says 0, expected cost says 2
        [0.90, 0.06, 0.03, 0.01],   # safely 0 either way
        [0.03, 0.05, 0.12, 0.80],   # clearly the worst class
    ])
    assert probs.argmax(axis=1).tolist() == [0, 0, 3]
    assert cost_decisions(probs).tolist() == [2, 0, 3]

    C = np.asarray(config.COST_MATRIX)
    assert np.allclose(expected_costs(probs), probs @ C)
    # Risk score = expected cost of doing nothing (predicting 0)
    assert np.allclose(risk_scores(probs), probs @ C[:, 0])


# ── calibration ───────────────────────────────────────────────────────────────

def test_calibrator_outputs_distributions_and_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.integers(0, config.NUM_CLASSES, n)
    off_mass = (1.0 - 0.8) / (config.NUM_CLASSES - 1)
    probs = np.full((n, config.NUM_CLASSES), off_mass)
    probs[np.arange(n), y] = 0.8
    flip = rng.random(n) < 0.3
    y_obs = np.where(flip, (y + 1) % config.NUM_CLASSES, y)

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


# ── snapshot maturity filtering ────────────────────────────────────────────────

def _yyyymmdd(d: date) -> float:
    return float(d.strftime("%Y%m%d"))


def test_filter_mature_snapshots_drops_only_immature():
    horizon = config.LABEL_HORIZON_MONTHS
    mature_snap   = _yyyymmdd(date.today() - timedelta(days=30 * (horizon + 2)))
    immature_snap = _yyyymmdd(date.today() - timedelta(days=30 * (horizon - 3)))

    result = filter_mature_snapshots([mature_snap, immature_snap])
    assert result == [mature_snap]


def test_iv_script_drops_immature_snapshot_instances():
    from explore_iv_woe import filter_matured_instances

    horizon = config.LABEL_HORIZON_MONTHS
    mature_snap   = _yyyymmdd(date.today() - timedelta(days=30 * (horizon + 2)))
    immature_snap = _yyyymmdd(date.today() - timedelta(days=30 * (horizon - 3)))

    X = np.zeros((4, 3))
    labels = np.array([0, 1, 2, 0])
    snapshot_dates = np.array([mature_snap, mature_snap, immature_snap, immature_snap])

    X_out, y_out = filter_matured_instances(X, labels, snapshot_dates)
    assert len(y_out) == 2
    assert y_out.tolist() == [0, 1]


# ── prediction path: unified onto TRAIN_TABLE ──────────────────────────────────

class _FakeConnector:
    """Duck-types the MSSQLConnector methods DataLoader needs for predict."""

    def __init__(self, available=None, pred_df=None):
        self._available = available or []
        self._pred_df = pred_df

    def get_available_snapshots(self):
        return self._available

    def load_prediction_data(self, snapshot_date=None, table=None):
        return self._pred_df


def test_load_pred_portfolios_drops_degenerate_label_columns():
    # On TRAIN_TABLE, an immature snapshot's WORST_FUTURE_* columns hold a
    # degenerate value (worst-so-far, not the real future outcome) rather
    # than being absent — must not leak in as a label.
    fake = _FakeConnector(pred_df=_make_df(with_target=True))
    instances, feature_cols = DataLoader(mssql_connector=fake).load_pred_portfolios(
        20250101, max_loans=2, use_cache=False
    )
    assert len(instances) == 1
    assert instances[0]["label"] == -1
    assert "WORST_FUTURE_DPD" not in feature_cols
    assert "WORST_FUTURE_CAT" not in feature_cols


def test_resolve_pred_snapshots_requested_found():
    fake = _FakeConnector(available=[20250101, 20250201, 20250301])
    resolved = DataLoader(mssql_connector=fake).resolve_pred_snapshots([20250201])
    assert resolved == [20250201]


def test_resolve_pred_snapshots_requested_missing_falls_back_to_immature():
    horizon = config.LABEL_HORIZON_MONTHS
    mature_snap   = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon + 2))))
    immature_snap = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon - 3))))
    fake = _FakeConnector(available=[mature_snap, immature_snap])

    resolved = DataLoader(mssql_connector=fake).resolve_pred_snapshots([99999999])
    assert resolved == [immature_snap]


def test_resolve_pred_snapshots_none_requested_selects_all_immature():
    horizon = config.LABEL_HORIZON_MONTHS
    mature_snap    = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon + 2))))
    immature_snap1 = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon - 3))))
    immature_snap2 = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon - 1))))
    fake = _FakeConnector(available=[mature_snap, immature_snap1, immature_snap2])

    resolved = DataLoader(mssql_connector=fake).resolve_pred_snapshots(None)
    assert resolved == sorted([immature_snap1, immature_snap2])


def test_resolve_pred_snapshots_no_immature_falls_back_to_latest():
    horizon = config.LABEL_HORIZON_MONTHS
    mature_snap1 = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon + 5))))
    mature_snap2 = int(_yyyymmdd(date.today() - timedelta(days=30 * (horizon + 2))))
    fake = _FakeConnector(available=[mature_snap1, mature_snap2])

    resolved = DataLoader(mssql_connector=fake).resolve_pred_snapshots(None)
    assert resolved == [mature_snap2]


# ── 4-class support: model / loss / cost matrix ────────────────────────────────

def test_deep_sets_and_loss_support_num_classes():
    import torch
    from src.model.deep_sets import DeepSets
    from src.model.losses import CostSensitiveFocalLoss

    torch.manual_seed(0)
    model = DeepSets(n_features=5, hidden_dim=8, embedding_dim=4, num_classes=config.NUM_CLASSES)
    features = torch.randn(6, 3, 5)
    mask = torch.zeros(6, 3, dtype=torch.bool)
    logits, embedding = model(features, mask)
    assert logits.shape == (6, config.NUM_CLASSES)
    assert embedding.shape == (6, 4)

    criterion = CostSensitiveFocalLoss(num_classes=config.NUM_CLASSES)
    targets = torch.randint(0, config.NUM_CLASSES, (6,))
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_cost_matrix_is_4x4_and_preserves_old_subblock():
    C = np.asarray(config.COST_MATRIX)
    assert C.shape == (4, 4)
    old = np.array([
        [0.0, 0.5, 1.0],
        [1.5, 0.0, 0.5],
        [4.0, 2.0, 0.0],
    ])
    assert np.allclose(C[:3, :3], old)


def test_compute_metrics_reports_worst_class_recall():
    y_true = np.array([0, 1, 2, 3, 3, 2])
    y_pred = np.array([0, 1, 2, 3, 2, 2])
    m = compute_metrics(y_true, y_pred)
    assert f"recall_class_{config.NUM_CLASSES - 1}" in m
    assert m[f"recall_class_{config.NUM_CLASSES - 1}"] == 0.5  # 1 of 2 true-3 recovered


# ── deployment bundle round-trip ───────────────────────────────────────────────

def test_bundle_round_trip(tmp_path):
    import joblib
    import torch
    import xgboost as xgb

    from src.data.preprocessing import create_preprocessing_pipeline
    from src.inference.model_loader import load_bundle
    from src.model.deep_sets import DeepSets

    rng = np.random.default_rng(0)
    n_features = 5
    feat_names = [f"f{i}" for i in range(n_features)]

    torch.manual_seed(0)
    model = DeepSets(n_features=n_features, hidden_dim=8, embedding_dim=4,
                      num_classes=config.NUM_CLASSES)
    model.eval()
    state_dict = model.state_dict()

    X = rng.normal(size=(50, n_features)).astype(np.float32)
    y = rng.integers(0, config.NUM_CLASSES, 50)

    # Real preprocessing pipeline (accepts list[(n_loans, F)] like production)
    scaler = create_preprocessing_pipeline(feat_names, config.BINARY_FEATURES)
    scaler.fit([row.reshape(1, -1) for row in X])

    # The legacy XGB is a META-learner on DeepSets embeddings (dim=4), not
    # raw features — fit it on embeddings so the scorer round-trips.
    feat_t = torch.from_numpy(X.reshape(50, 1, n_features))
    mask_t = torch.zeros(50, 1, dtype=torch.bool)
    with torch.no_grad():
        emb = model.extract_embeddings(feat_t, mask_t).numpy()

    xgb_model = xgb.XGBClassifier(
        n_estimators=5, max_depth=2, num_class=config.NUM_CLASSES,
        objective="multi:softprob",
    )
    xgb_model.fit(emb, y)
    xgb_raw = xgb_model.get_booster().save_raw(raw_format="json")

    probs = xgb_model.predict_proba(emb)
    calibrator = PerClassIsotonicCalibrator().fit(probs, y)

    bundle = {
        "metadata": {
            "feature_count": n_features,
            "max_loans_per_customer_99th": 2,
            "features": feat_names,
        },
        "scaler": scaler,
        "deep_sets_state_dict": state_dict,
        "deep_sets_hparams": {
            "n_features": n_features, "hidden_dim": 8, "embedding_dim": 4,
            "num_classes": config.NUM_CLASSES,
        },
        "xgb_model_raw": xgb_raw,
        "calibrator": calibrator,
    }
    bundle_path = tmp_path / "deepsets_bundle.pkl"
    joblib.dump(bundle, bundle_path)

    # New contract: load_bundle → (scorer, calibrator, features). A legacy
    # (kind-less) bundle yields a DeepSetsScorer.
    from src.inference.model_loader import DeepSetsScorer
    scorer, loaded_cal, features = load_bundle(bundle_path, device="cpu")

    assert isinstance(scorer, DeepSetsScorer)
    assert scorer.max_loans == 2
    assert features == feat_names
    assert np.allclose(loaded_cal.transform(probs), calibrator.transform(probs))

    # The legacy scorer produces a valid class distribution end-to-end
    insts = [
        {"features": rng.normal(size=(1, n_features)).astype(np.float32),
         "n_loans": 1, "current_cat": int(rng.integers(config.NUM_CLASSES))}
        for _ in range(6)
    ]
    out = scorer.raw_probs(insts)
    assert out.shape == (6, config.NUM_CLASSES)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_model_loader_dispatches_to_bundle_when_path_is_a_file(tmp_path, monkeypatch):
    """ModelLoader.load_pipeline() should route to load_bundle() for a file path."""
    from src.inference import model_loader as model_loader_mod

    sentinel = object()
    monkeypatch.setattr(model_loader_mod, "load_bundle", lambda path, device=None: sentinel)

    bundle_path = tmp_path / "model_bundle.pkl"
    bundle_path.write_bytes(b"not a real bundle, dispatch is file-based")

    loader = model_loader_mod.ModelLoader(bundle_path)
    assert loader.load_pipeline() is sentinel


# ── ranking deliverable (July 8): masking, stratified calibration, queue ───────

def test_mask_monotone_zeroes_impossible_classes():
    from src.evaluation.decision import mask_monotone

    probs = np.array([
        [0.4, 0.3, 0.2, 0.1],   # current_cat 2 → classes 0,1 impossible
        [0.4, 0.3, 0.2, 0.1],   # current_cat 0 → untouched
    ])
    out = mask_monotone(probs, np.array([2, 0]))
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert np.isclose(out[0, 2], 0.2 / 0.3) and np.isclose(out[0, 3], 0.1 / 0.3)
    assert np.allclose(out[1], probs[1])
    assert np.allclose(out.sum(axis=1), 1.0)


def test_stratified_calibrator_separates_strata():
    from src.evaluation.calibration import StratifiedCalibrator

    rng = np.random.default_rng(0)
    n = 8000
    y = rng.integers(0, 4, n)
    probs = np.full((n, 4), 0.2 / 3)
    probs[np.arange(n), y] = 0.8
    # Stratum 0 noisy (50% flips), stratum 1 clean (10% flips)
    strata = (rng.random(n) < 0.5).astype(int)
    flip = rng.random(n) < np.where(strata == 0, 0.5, 0.1)
    y_obs = np.where(flip, (y + 1) % 4, y)

    cal = StratifiedCalibrator(min_stratum_n=1000).fit(probs, y_obs, strata)
    out = cal.transform(probs, strata)
    top = out[np.arange(n), probs.argmax(axis=1)]
    assert abs(top[strata == 0].mean() - 0.5) < 0.07
    assert abs(top[strata == 1].mean() - 0.9) < 0.07

    # Below the floor for one stratum → falls back to pooled, still works
    cal2 = StratifiedCalibrator(min_stratum_n=n).fit(probs, y_obs, strata)
    out2 = cal2.transform(probs, strata)
    assert np.allclose(out2.sum(axis=1), 1.0)
    assert not cal2.per_stratum_          # everything pooled


def test_ranking_metrics_recall_lift_and_carveout(monkeypatch):
    import src.evaluation.ranking as ranking

    monkeypatch.setattr(
        config, "RANKING_REF_WINDOWS", {"w": 3 / config.API_RATE_PER_HOUR}
    )
    y = np.array([3, 3, 0, 1, 2, 0, 0, 1, 0, 3])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .15, .1])

    m = ranking.ranking_metrics(y, s)
    assert m["n_severe"] == 3
    assert np.isclose(m["at_w"]["recall"], 2 / 3)         # 2 of 3 severe in top 3
    assert np.isclose(m["at_w"]["lift"], (2 / 3) / 0.3)

    # Carve-out: already-severe (current_cat 3) leave the ranked population
    strata = np.array([3, 0, 0, 1, 2, 0, 0, 1, 0, 2])
    m2 = ranking.ranking_metrics(y, s, strata=strata)
    assert m2["n_ranked"] == 9 and m2["n_severe"] == 2
    assert "by_current_cat" in m2

    # Perfect ranking → recall hits 1.0 within n_severe calls
    monkeypatch.setattr(
        config, "RANKING_REF_WINDOWS", {"w": 2 / config.API_RATE_PER_HOUR}
    )
    y3 = np.array([3, 3, 0, 0])
    s3 = np.array([.9, .8, .2, .1])
    m3 = ranking.ranking_metrics(y3, s3)
    assert m3["at_w"]["recall"] == 1.0


def test_capture_curve_is_monotone_and_complete():
    from src.evaluation.ranking import capture_curve

    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int) * 3
    s = rng.random(500)
    cc = capture_curve(y, s, n_points=50)
    rec = np.array(cc["recall"])
    assert (np.diff(rec) >= -1e-12).all()      # non-decreasing
    assert rec[-1] == 1.0                      # full list captures everyone
    assert len(cc["hours"]) == len(rec)


def test_split_for_final_fit_uses_all_mature_snapshots():
    from src.data.temporal_split import split_for_final_fit

    horizon = config.LABEL_HORIZON_MONTHS
    mature_a   = _yyyymmdd(date.today() - timedelta(days=30 * (horizon + 8)))
    mature_b   = _yyyymmdd(date.today() - timedelta(days=30 * (horizon + 2)))
    immature   = _yyyymmdd(date.today() - timedelta(days=30 * (horizon - 3)))

    instances = [
        {"national_code": f"c{i}", "snapshot_date": snap, "label": 0, "current_cat": 0}
        for i in range(200) for snap in (mature_a, mature_b, immature)
    ]
    train, val, test = split_for_final_fit(instances)

    assert test == []
    used_snaps = {i["snapshot_date"] for i in train} | {i["snapshot_date"] for i in val}
    assert used_snaps == {mature_a, mature_b}          # immature excluded
    # Customer-disjoint val carved (OPTIMIZE_ON_VALIDATION=True in config)
    assert val and not ({i["national_code"] for i in train}
                        & {i["national_code"] for i in val})


def test_severity_scores_is_last_class_probability():
    from src.evaluation.decision import severity_scores

    p = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    assert np.allclose(severity_scores(p), [0.4, 0.1])


# ── queue rule flags: certainty band + call-ledger freshness ────────────────────

def _queue_df():
    return pd.DataFrame({
        "SNAPSHOT_DATE":  [20260601.0, 20260601.0, 20260601.0, 20260501.0, 20260601.0],
        "NATIONAL_CODE":  ["already_sev", "certain", "called", "stale_dup", "stale_dup"],
        "CURRENT_CAT":    [3, 1, 0, 0, 0],
        "RISK_SCORE":     [0.99, 0.95, 0.50, 0.40, 0.40],
    })


def test_apply_queue_flags_priorities(monkeypatch):
    from src.inference.predictor import apply_queue_flags

    monkeypatch.setattr(config, "CERTAINTY_ACT_THRESHOLD", 0.9)
    called_log = pd.DataFrame({
        "NATIONAL_CODE": ["called"],
        "CALLED_AT": [(date.today() - timedelta(days=5)).isoformat()],
    })

    flags = apply_queue_flags(_queue_df(), multi_snapshot=True, called_log=called_log)
    assert flags.tolist() == [
        "ALREADY_SEVERE",     # current_cat 3, even though score is highest
        "PREDICTED_SEVERE",   # certain enough to act without API
        "RECENTLY_CALLED",    # fresh enrichment already exists
        "SUPERSEDED",         # older snapshot row of stale_dup
        "",                   # the newest stale_dup row competes for budget
    ]


def test_apply_queue_flags_ttl_and_disabled_knob(monkeypatch):
    from src.inference.predictor import apply_queue_flags

    # Knob off (default): no PREDICTED_SEVERE even at score 0.95
    monkeypatch.setattr(config, "CERTAINTY_ACT_THRESHOLD", None)
    # Ledger entry older than the TTL: no RECENTLY_CALLED either
    old_log = pd.DataFrame({
        "NATIONAL_CODE": ["called"],
        "CALLED_AT": [
            (date.today() - timedelta(days=config.API_DATA_TTL_DAYS + 10)).isoformat()
        ],
    })

    flags = apply_queue_flags(_queue_df(), multi_snapshot=True, called_log=old_log)
    assert flags.tolist() == ["ALREADY_SEVERE", "", "", "SUPERSEDED", ""]

    # Single snapshot: no SUPERSEDED possible
    df = _queue_df().iloc[[0, 1, 2]]
    flags2 = apply_queue_flags(df, multi_snapshot=False, called_log=None)
    assert flags2.tolist() == ["ALREADY_SEVERE", "", ""]


# ── model arms (July 10 architecture switch) ────────────────────────────────────

def _arm_instances(n, seed):
    rng_l = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cat = int(rng_l.choice([0, 1, 2, 3], p=[.55, .25, .12, .08]))
        label = min(cat + int(rng_l.choice([0, 0, 1, 2])), config.NUM_CLASSES - 1)
        feats = rng_l.normal(size=(1, 6)).astype(np.float32)
        feats[0, 0] = label + rng_l.normal(0, .8)
        out.append({"national_code": f"c{seed}_{i}", "snapshot_date": 20250101.0,
                    "n_loans": 1, "features": feats, "label": label,
                    "current_cat": cat})
    return out


def _arm_arrays(n, seed):
    """Instances + the (X, y, current_cat) arrays every arm now trains on."""
    from src.baselines.aggregated_xgboost import aggregate_features
    inst = _arm_instances(n, seed)
    X, y = aggregate_features(inst)
    cat = np.array([i["current_cat"] for i in inst])
    return X, y, cat


def test_aggregate_features_matches_naive_reference():
    from src.baselines.aggregated_xgboost import aggregate_features

    insts = _arm_instances(50, 99)
    X, y = aggregate_features(insts)

    X_ref = []
    for inst in insts:
        f = inst["features"]
        X_ref.append(np.concatenate([f.min(0), f.max(0), f.mean(0), f.std(0), [inst["n_loans"]]]))
    X_ref = np.vstack(X_ref).astype(np.float32)

    assert np.allclose(X, X_ref, atol=1e-4)
    assert y.tolist() == [i["label"] for i in insts]


def test_ordinal_arm_produces_valid_distribution():
    from src.baselines.aggregated_xgboost import OrdinalXGBArm

    X_train, y_train, cat_train = _arm_arrays(1500, 1)
    X_test, y_test, cat_test = _arm_arrays(400, 2)
    arm = OrdinalXGBArm()
    arm.train(X_train, y_train, cat_train)
    probs = arm.predict_proba(X_test)

    assert probs.shape == (400, config.NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs >= 0).all()


def test_per_cat_arm_is_natively_monotone_and_handles_missing_classes():
    from src.baselines.aggregated_xgboost import PerCatXGBArm

    # Stratum 0 never reaches class 3 in training (the XGB classes_ mapping
    # regression: predicted columns must land on the classes actually seen)
    inst_train = [i for i in _arm_instances(2000, 3)
                  if not (i["current_cat"] == 0 and i["label"] == 3)]
    from src.baselines.aggregated_xgboost import aggregate_features
    X_train, y_train = aggregate_features(inst_train)
    cat_train = np.array([i["current_cat"] for i in inst_train])

    inst_test = _arm_instances(500, 4)
    X_test, y_test = aggregate_features(inst_test)
    cat_test = np.array([i["current_cat"] for i in inst_test])

    arm = PerCatXGBArm()
    arm.train(X_train, y_train, cat_train)
    probs = arm.predict_proba(X_test, cat_test)

    assert probs.shape == (500, config.NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    for c in range(1, config.NUM_CLASSES):
        # No probability mass below the current category — by construction
        assert probs[cat_test == c][:, :c].max() <= 1e-12
    # Already-severe stratum gets the point mass
    assert np.allclose(probs[cat_test == 3][:, 3], 1.0)


def test_binary_arm_two_columns_for_calibrator_reuse():
    from src.baselines.aggregated_xgboost import BinarySevereBaseline

    X_train, y_train, cat_train = _arm_arrays(1500, 5)
    X_test, y_test, cat_test = _arm_arrays(300, 6)
    arm = BinarySevereBaseline()
    arm.train(X_train, y_train, cat_train)
    probs = arm.predict_proba(X_test)

    assert probs.shape == (300, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert not arm.full_distribution        # never deployable


def test_arm_builders_registry_matches_config():
    from src.baselines.aggregated_xgboost import ARM_BUILDERS

    for name in config.MODEL_ARMS:
        assert name in ARM_BUILDERS, f"config arm '{name}' has no builder"
    deployable = [n for n in config.MODEL_ARMS if ARM_BUILDERS[n].full_distribution]
    assert deployable, "at least one full-distribution arm must be configured"


# ── Phase C: arm scorer / bundle / ranking-aware aggregator (July 11) ──────────

def _fit_arm_and_scaler(seed=1):
    """Train a multiclass arm the way run.py does; return (scaler, arm, cal, pred_raw)."""
    from src.baselines.aggregated_xgboost import aggregate_features, ARM_BUILDERS
    from src.evaluation.calibration import StratifiedCalibrator
    from src.data.preprocessing import create_preprocessing_pipeline

    train = _arm_instances(3000, seed)
    val   = _arm_instances(1000, seed + 1)
    pred  = _arm_instances(400, seed + 2)          # keeps RAW features
    feats = ["DPD_DAYS"] + [f"F{j}" for j in range(5)]

    scaler = create_preprocessing_pipeline(feats, config.BINARY_FEATURES)
    scaler.fit([i["features"] for i in train])
    for split in (train, val):
        for inst, x in zip(split, scaler.transform([i["features"] for i in split])):
            inst["features"] = x

    X_tr, y_tr = aggregate_features(train)
    X_v,  y_v  = aggregate_features(val)
    cat_tr = np.array([i["current_cat"] for i in train])
    cat_v  = np.array([i["current_cat"] for i in val])
    arm = ARM_BUILDERS["multiclass"]()
    arm.train(X_tr, y_tr, cat_tr, X_v, y_v, cat_v)
    cal = StratifiedCalibrator(min_stratum_n=200).fit(arm.predict_proba(X_v, cat_v), y_v, cat_v)
    return scaler, arm, cal, feats, pred


def test_arm_scorer_bundle_and_dir_roundtrip(tmp_path):
    import joblib
    from src.inference.model_loader import build_arm_bundle, ModelLoader, ArmScorer

    scaler, arm, cal, feats, pred = _fit_arm_and_scaler()
    ref = ArmScorer(scaler, arm, max_loans=2).raw_probs(pred)

    # Bundle round-trip
    joblib.dump(build_arm_bundle(scaler, arm, cal, 2, feats), tmp_path / "model_bundle.pkl")
    scorer_b, cal_b, feats_b = ModelLoader(tmp_path / "model_bundle.pkl").load_pipeline()
    assert np.allclose(ref, scorer_b.raw_probs(pred), atol=1e-6)
    assert feats_b == feats and cal_b is not None
    assert ref.shape == (400, config.NUM_CLASSES) and np.allclose(ref.sum(1), 1.0, atol=1e-5)

    # Directory round-trip (prefers model_arm.pkl over legacy deepsets)
    import json
    d = tmp_path / "dir"
    d.mkdir()
    joblib.dump(arm, d / "model_arm.pkl")
    joblib.dump(scaler, d / "scaler.pkl")
    joblib.dump(cal, d / "calibrator.pkl")
    (d / "metadata.json").write_text(json.dumps(
        {"feature_count": 6, "max_loans_per_customer_99th": 2, "features": feats}))
    scorer_d, _, _ = ModelLoader(d).load_pipeline()
    assert np.allclose(ref, scorer_d.raw_probs(pred), atol=1e-6)


def test_fold_aggregator_summarizes_ranking(monkeypatch):
    from src.evaluation.fold_aggregator import aggregate_fold_metrics

    monkeypatch.setattr(config, "RANKING_REF_WINDOWS", {"1_day": 24, "1_week": 168})

    def fold(fid, ap, r_week):
        return {"fold_id": fid, "test_snap": 20250000 + fid, "deployed_arm": "multiclass",
                "final_metrics": {"macro_f1": 0.6, "ranking": {
                    "pr_auc": ap,
                    "at_1_day": {"recall": 0.1},
                    "at_1_week": {"recall": r_week}}}}

    agg = aggregate_fold_metrics([fold(1, 0.55, 0.50), fold(2, 0.57, 0.52), fold(3, 0.53, 0.48)])
    assert agg["n_folds"] == 3
    assert abs(agg["aggregate"]["ranking_ap"]["mean"] - 0.55) < 1e-9
    assert agg["aggregate"]["ranking_ap"]["min"] == 0.53
    assert abs(agg["aggregate"]["recall_1_week"]["mean"] - 0.50) < 1e-9
    assert agg["aggregate"]["ranking_ap"]["n"] == 3


def test_arm_accepts_tuned_params_override():
    from src.baselines.aggregated_xgboost import AggregatedXGBoostBaseline, XGB_DEFAULTS

    arm = AggregatedXGBoostBaseline(params={"max_depth": 9, "n_estimators": 111})
    assert arm.params["max_depth"] == 9 and arm.params["n_estimators"] == 111
    assert arm.params["subsample"] == XGB_DEFAULTS["subsample"]   # unspecified keys keep defaults


# ── standalone scoring package (July 11) ────────────────────────────────────────

def _synth_score_df(n, seed, with_target=True):
    rng_l = np.random.default_rng(seed)
    cat = rng_l.choice([0, 1, 2, 3], n, p=[.55, .25, .12, .08]).astype(float)
    label = np.minimum(cat + rng_l.choice([0, 0, 1, 2], n), 3)
    d = {
        "LOAN_ID": np.arange(n), "CONTRACT_NUMBER": [f"c{seed}_{i}" for i in range(n)],
        "NATIONAL_CODE": [f"n{seed}_{i}" for i in range(n)],
        "SNAPSHOT_DATE": [20250101.0] * n,
        "DPD_DAYS": label + rng_l.normal(0, .8, n),
        "LOAN_CATEGORY": cat,
        "F1": rng_l.normal(size=n), "F2": rng_l.normal(size=n),
        "F3": rng_l.normal(size=n), "F4": rng_l.normal(size=n),
    }
    if with_target:
        d["WORST_FUTURE_DPD"] = np.zeros(n)
        d["WORST_FUTURE_CAT"] = label
    return pd.DataFrame(d)


def _fit_bundle_via_dataframe(tmp_path):
    """Train scaler+arm+calibrator the way the real pipeline does: raw df ->
    process_raw_data -> aggregate_features -> arm.train. Returns bundle_path."""
    import joblib
    from src.data.data_loader import DataLoader
    from src.baselines.aggregated_xgboost import aggregate_features, ARM_BUILDERS
    from src.evaluation.calibration import StratifiedCalibrator
    from src.data.preprocessing import create_preprocessing_pipeline
    from src.inference.model_loader import build_arm_bundle

    dl = DataLoader()
    train_inst, feats = dl.process_raw_data(_synth_score_df(1500, 1), max_loans=2)
    val_inst, _        = dl.process_raw_data(_synth_score_df(600, 2), max_loans=2)

    scaler = create_preprocessing_pipeline(feats, config.BINARY_FEATURES)
    scaler.fit([i["features"] for i in train_inst])
    for split in (train_inst, val_inst):
        for inst, x in zip(split, scaler.transform([i["features"] for i in split])):
            inst["features"] = x

    Xtr, ytr = aggregate_features(train_inst)
    Xv, yv   = aggregate_features(val_inst)
    cattr = np.array([i["current_cat"] for i in train_inst])
    catv  = np.array([i["current_cat"] for i in val_inst])
    arm = ARM_BUILDERS["multiclass"]()
    arm.train(Xtr, ytr, cattr, Xv, yv, catv)
    cal = StratifiedCalibrator(min_stratum_n=100).fit(arm.predict_proba(Xv, catv), yv, catv)

    bundle_path = tmp_path / "model_bundle.pkl"
    joblib.dump(build_arm_bundle(scaler, arm, cal, 2, feats), bundle_path)
    return bundle_path


def test_score_dataframe_matches_predictor_shape(tmp_path):
    from src.inference.predictor import score_dataframe

    bundle_path = _fit_bundle_via_dataframe(tmp_path)
    pred_df = _synth_score_df(30, 9, with_target=False)
    q = score_dataframe(pred_df, bundle_path)

    assert len(q) == 30
    for col in ["RISK_RANK", "RISK_SCORE", "RULE_FLAG", "CURRENT_CAT",
               "P_NO_DELAY", "P_CURRENT", "P_PAST_DUE", "P_SEVERE_PAST_DUE"]:
        assert col in q.columns
    probs = q[["P_NO_DELAY", "P_CURRENT", "P_PAST_DUE", "P_SEVERE_PAST_DUE"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    # Ranked rows come first, sorted by RISK_SCORE descending
    ranked = q[q["RULE_FLAG"] == ""]
    assert (ranked["RISK_SCORE"].diff().dropna() <= 1e-9).all()


def test_score_dataframe_calibration_refresh(tmp_path):
    from src.inference.predictor import score_dataframe

    bundle_path = _fit_bundle_via_dataframe(tmp_path)
    pred_df = _synth_score_df(20, 11, with_target=False)
    cal_df  = _synth_score_df(500, 12, with_target=True)   # matured, labeled

    q = score_dataframe(pred_df, bundle_path, calibration_df=cal_df)
    assert len(q) == 20
    probs = q[["P_NO_DELAY", "P_CURRENT", "P_PAST_DUE", "P_SEVERE_PAST_DUE"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_build_scoring_package_runs_standalone(tmp_path):
    """
    The real proof: package the model, then score in a SEPARATE process with
    this repo removed from sys.path and torch/optuna/umap/pyodbc blocked at
    import time. If the package secretly depended on any of them, this fails.
    """
    import subprocess
    import sys
    import textwrap

    bundle_path = _fit_bundle_via_dataframe(tmp_path)
    pkg_out = tmp_path / "scoring_package"
    repo_root = Path(__file__).resolve().parent.parent

    r = subprocess.run(
        [sys.executable, str(repo_root / "build_scoring_package.py"),
         "--bundle", str(bundle_path), "--output", str(pkg_out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (pkg_out / "model_bundle.pkl").exists()
    assert (pkg_out / "requirements-scoring.txt").exists()
    assert (pkg_out / "README_SCORING.md").exists()

    pred_csv = tmp_path / "score_input.csv"
    _synth_score_df(15, 20, with_target=False).to_csv(pred_csv, index=False)

    script = textwrap.dedent(f"""
        import sys
        sys.path = [p for p in sys.path if {str(repo_root)!r} not in p]
        sys.path.insert(0, {str(pkg_out)!r})

        class _Blocked:
            def find_module(self, name, path=None):
                if name.split(".")[0] in ("torch", "optuna", "umap", "pyodbc"):
                    raise ImportError(f"BLOCKED: {{name}}")
        sys.meta_path.insert(0, _Blocked())

        import pandas as pd
        from src.inference.predictor import score_dataframe
        df = pd.read_csv({str(pred_csv)!r})
        q = score_dataframe(df, {str(pkg_out / "model_bundle.pkl")!r})
        assert len(q) == 15
        print("OK")
    """)
    r2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "OK" in r2.stdout


# ── explore_shap.py bundle mode (July 11 doc/tooling refresh) ──────────────────

def test_explore_shap_load_from_bundle_produces_named_features(tmp_path):
    import joblib
    from src.data.data_loader import DataLoader
    from src.baselines.aggregated_xgboost import aggregate_features, ARM_BUILDERS
    from src.data.preprocessing import create_preprocessing_pipeline
    from src.inference.model_loader import build_arm_bundle
    from explore_shap import load_from_bundle

    dl = DataLoader()
    train_inst, feats = dl.process_raw_data(_synth_score_df(800, 1), max_loans=2)
    scaler = create_preprocessing_pipeline(feats, config.BINARY_FEATURES)
    scaler.fit([i["features"] for i in train_inst])
    for inst, x in zip(train_inst, scaler.transform([i["features"] for i in train_inst])):
        inst["features"] = x
    Xtr, ytr = aggregate_features(train_inst)
    cattr = np.array([i["current_cat"] for i in train_inst])
    arm = ARM_BUILDERS["multiclass"]()
    arm.train(Xtr, ytr, cattr)

    bundle_path = tmp_path / "model_bundle.pkl"
    joblib.dump(build_arm_bundle(scaler, arm, None, 2, feats), bundle_path)
    data_path = tmp_path / "snap.csv"
    _synth_score_df(40, 9, with_target=False).to_csv(data_path, index=False)

    model, X, y, names = load_from_bundle(bundle_path, data_path)

    assert X.shape == (40, 4 * len(feats) + 1)
    assert len(names) == X.shape[1]
    assert names[0] == f"MIN_{feats[0]}" and names[-1] == "N_LOANS"
    # Reloaded (deserialized) model must predict identically to the original
    assert np.allclose(model.predict_proba(X), arm.model.predict_proba(X))


def test_explore_shap_rejects_multi_model_arms(tmp_path):
    import joblib
    from src.data.data_loader import DataLoader
    from src.baselines.aggregated_xgboost import aggregate_features, ARM_BUILDERS
    from src.data.preprocessing import create_preprocessing_pipeline
    from src.inference.model_loader import build_arm_bundle
    import explore_shap

    dl = DataLoader()
    train_inst, feats = dl.process_raw_data(_synth_score_df(500, 3), max_loans=2)
    scaler = create_preprocessing_pipeline(feats, config.BINARY_FEATURES)
    scaler.fit([i["features"] for i in train_inst])
    for inst, x in zip(train_inst, scaler.transform([i["features"] for i in train_inst])):
        inst["features"] = x
    Xtr, ytr = aggregate_features(train_inst)
    cattr = np.array([i["current_cat"] for i in train_inst])
    ordinal = ARM_BUILDERS["ordinal"]()
    ordinal.train(Xtr, ytr, cattr)   # has .models (list), no single .model

    bundle_path = tmp_path / "model_bundle.pkl"
    joblib.dump(build_arm_bundle(scaler, ordinal, None, 2, feats), bundle_path)
    data_path = tmp_path / "snap.csv"
    _synth_score_df(10, 8, with_target=False).to_csv(data_path, index=False)

    try:
        explore_shap.load_from_bundle(bundle_path, data_path)
        raised = False
    except SystemExit:
        raised = True
    assert raised, "load_from_bundle should reject multi-model arms (ordinal/per_cat)"
