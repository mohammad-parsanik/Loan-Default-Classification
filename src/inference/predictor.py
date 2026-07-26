"""
End-to-end scoring pipeline. The model itself (XGBoost arm, or legacy
DeepSets) is abstracted behind a Scorer loaded by ModelLoader; this module
only turns raw probabilities into the deliverable.

Output is the ranked API queue: customers not yet severe, ordered by
calibrated+masked P(entering the severe class), consumed at
config.API_RATE_PER_HOUR. Already-severe customers are rule-flagged
(ALREADY_SEVERE) and never compete for API budget; when several snapshots
are scored, only each customer's newest row enters the queue (older rows
flagged SUPERSEDED — the enrichment API returns present-time data only, so
acting on a stale score wastes budget).

With config.RECALIBRATE_ON_PREDICT = True, the calibrator is refreshed at
scoring time on the newest snapshot whose labels have matured, so
probability thresholds in the downstream rule system track base-rate drift.

Two entry points:
  - Predictor.predict()  — fetches snapshot(s) from TRAIN_TABLE via this
                           project's DB connector (the CLI / server path).
  - score_dataframe()    — scores a caller-supplied DataFrame directly, no
                           DB. This is the integration point for embedding
                           scoring in another system — see build_scoring_
                           package.py for a minimal standalone dependency
                           set (no torch, no DB driver required).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import project_config as config
from src.data.data_loader import DataLoader
from src.data.temporal_split import _get_usable_snapshots
from src.evaluation.calibration import StratifiedCalibrator
from src.evaluation.decision import (
    cost_decisions,
    mask_monotone,
    risk_scores,
    severity_scores,
)
from src.inference.model_loader import ModelLoader

logger = logging.getLogger(__name__)


def apply_queue_flags(
    results: pd.DataFrame,
    multi_snapshot: bool,
    called_log: Optional[pd.DataFrame] = None,
    *,
    carve_current_cat_ge: Optional[int] = None,
    certainty_act_threshold: Optional[float] = None,
    api_data_ttl_days: Optional[int] = None,
    pred_dedup_latest: Optional[bool] = None,
) -> np.ndarray:
    """
    RULE_FLAG per row. Priority (first match wins):
      ALREADY_SEVERE   — current_cat >= carve_current_cat_ge: known risk, no API
      SUPERSEDED       — an older snapshot row of a customer whose newer row
                         carries the decision (only when several snapshots
                         were scored and pred_dedup_latest is on)
      PREDICTED_SEVERE — RISK_SCORE >= certainty_act_threshold (when set):
                         certain enough to act on directly, API adds nothing
      RECENTLY_CALLED  — last API call fresher than api_data_ttl_days:
                         re-calling buys no new information yet
      ""               — competes for API budget in the ranked queue

    Keyword-only knobs default to project_config when omitted (the
    Predictor.predict()/score_dataframe() call sites); a caller that has
    already resolved a ScoringParams (run_scoring) passes them explicitly
    instead, so nothing here needs to re-read config at all in that path.
    """
    if carve_current_cat_ge is None:
        carve_current_cat_ge = config.CARVE_CURRENT_CAT_GE
    if pred_dedup_latest is None:
        pred_dedup_latest = getattr(config, "PRED_DEDUP_LATEST", True)
    if certainty_act_threshold is None:
        certainty_act_threshold = getattr(config, "CERTAINTY_ACT_THRESHOLD", None)
    if api_data_ttl_days is None:
        api_data_ttl_days = config.API_DATA_TTL_DAYS

    flags = np.full(len(results), "", dtype=object)

    strata = results["CURRENT_CAT"].to_numpy()
    flags[strata >= carve_current_cat_ge] = "ALREADY_SEVERE"

    if multi_snapshot and pred_dedup_latest:
        latest = results.groupby("NATIONAL_CODE")["SNAPSHOT_DATE"].transform("max")
        stale = (results["SNAPSHOT_DATE"] < latest).to_numpy()
        flags[(flags == "") & stale] = "SUPERSEDED"

    if certainty_act_threshold is not None:
        certain = results["RISK_SCORE"].to_numpy() >= certainty_act_threshold
        flags[(flags == "") & certain] = "PREDICTED_SEVERE"

    if called_log is not None and len(called_log):
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=api_data_ttl_days)
        fresh_calls = called_log[pd.to_datetime(called_log["CALLED_AT"]) >= cutoff]
        recently = results["NATIONAL_CODE"].isin(set(fresh_calls["NATIONAL_CODE"])).to_numpy()
        flags[(flags == "") & recently] = "RECENTLY_CALLED"

    return flags


def score_instances(
    instances: list[dict],
    scorer,
    calibrator=None,
    called_log: Optional[pd.DataFrame] = None,
    *,
    cost_matrix=None,
    carve_current_cat_ge: Optional[int] = None,
    certainty_act_threshold: Optional[float] = None,
    api_data_ttl_days: Optional[int] = None,
    pred_dedup_latest: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Core scoring transform shared by Predictor.predict(), score_dataframe(),
    and run_scoring(): instances -> calibrated+masked probabilities ->
    ranked queue DataFrame. Pure function of its arguments — no DB, no
    project-specific data access — so it's the safe reuse point for a
    standalone deployment.

    The keyword-only knobs default to project_config when omitted (as
    before); run_scoring() passes an already-resolved ScoringParams'
    values explicitly instead.
    """
    if not instances:
        return pd.DataFrame()

    strata    = np.array([i["current_cat"] for i in instances])
    probs_raw = scorer.raw_probs(instances)

    if isinstance(calibrator, StratifiedCalibrator):
        probs = calibrator.transform(probs_raw, strata)
    elif calibrator is not None:
        probs = calibrator.transform(probs_raw)
    else:
        probs = probs_raw
    # label >= current_cat always — zero the impossible mass
    probs = mask_monotone(probs, strata)

    results = pd.DataFrame({
        "SNAPSHOT_DATE":        [i["snapshot_date"]  for i in instances],
        "NATIONAL_CODE":        [i["national_code"]  for i in instances],
        "N_LOANS_IN_PORTFOLIO": [i["n_loans"]        for i in instances],
        "CURRENT_CAT":          strata,
        "P_NO_DELAY":           probs[:, 0],
        "P_CURRENT":            probs[:, 1],
        "P_PAST_DUE":           probs[:, 2],
        "P_SEVERE_PAST_DUE":    probs[:, 3],
        # Minimum-expected-cost class under the business cost matrix
        "PREDICTED_CLASS":      cost_decisions(probs, cost_matrix),
        # THE queue ranking score: calibrated P(entering severe)
        "RISK_SCORE":           severity_scores(probs),
        # Secondary/diagnostic: expected cost of taking no action
        "EXPECTED_COST":        risk_scores(probs, cost_matrix),
    })

    multi_snapshot = results["SNAPSHOT_DATE"].nunique() > 1
    results["RULE_FLAG"] = apply_queue_flags(
        results, multi_snapshot=multi_snapshot, called_log=called_log,
        carve_current_cat_ge=carve_current_cat_ge,
        certainty_act_threshold=certainty_act_threshold,
        api_data_ttl_days=api_data_ttl_days,
        pred_dedup_latest=pred_dedup_latest,
    )

    # Queue rows first (by risk), flagged rows after; rank only the queue
    results["_flagged"] = results["RULE_FLAG"] != ""
    results = (
        results.sort_values(["_flagged", "RISK_SCORE"], ascending=[True, False])
        .drop(columns="_flagged")
        .reset_index(drop=True)
    )
    n_queue = int((results["RULE_FLAG"] == "").sum())
    rank = np.full(len(results), np.nan)
    rank[:n_queue] = np.arange(1, n_queue + 1)
    results.insert(0, "RISK_RANK", rank)

    flag_counts = results.loc[results["RULE_FLAG"] != "", "RULE_FLAG"].value_counts()
    flag_str = " | ".join(f"{int(v):,} {k}" for k, v in flag_counts.items()) or "none flagged"
    logger.info(
        f"Queue: {n_queue:,} ranked | {flag_str}. "
        f"At {config.API_RATE_PER_HOUR}/h the full queue takes "
        f"{n_queue / config.API_RATE_PER_HOUR:.0f}h to call."
    )
    return results


def score_dataframe(
    df: pd.DataFrame,
    artifact_dir,
    calibration_df: Optional[pd.DataFrame] = None,
    called_log_path: Optional[str] = None,
    max_loans: Optional[int] = None,
) -> pd.DataFrame:
    """
    Score a caller-supplied DataFrame directly — no database access.

    `df` must have the same columns as TRAIN_TABLE (see column_changes.md),
    minus WORST_FUTURE_CAT/WORST_FUTURE_DPD (this is prediction time — if
    present they're ignored). Multiple snapshots may be mixed in one call;
    dedup/supersede logic applies exactly as in Predictor.predict().

    `calibration_df` (optional): a DataFrame of a single MATURED snapshot
    WITH WORST_FUTURE_CAT populated, used to refresh the calibrator before
    scoring — the DB-free equivalent of Predictor's RECALIBRATE_ON_PREDICT.
    Omit to use the calibrator shipped in the bundle unchanged.
    """
    from src.data.data_loader import DataLoader   # local: keeps this function's
                                                    # dependency footprint explicit
    loader = ModelLoader(Path(artifact_dir))
    scorer, calibrator, _ = loader.load_pipeline()
    dl = DataLoader()

    if calibration_df is not None:
        cal_inst, _ = dl.process_raw_data(calibration_df, scorer.truncate_loans)
        cal_inst = [i for i in cal_inst if i["label"] >= 0]
        if cal_inst:
            probs  = scorer.raw_probs(cal_inst)
            y      = np.array([i["label"] for i in cal_inst])
            strata = np.array([i["current_cat"] for i in cal_inst])
            calibrator = StratifiedCalibrator(
                min_stratum_n=config.CALIBRATION_MIN_STRATUM_N
            ).fit(probs, y, strata)
            logger.info(f"Calibrator refreshed on {len(cal_inst):,} supplied instances.")

    instances, _ = dl.process_raw_data(df, max_loans or scorer.truncate_loans)
    called_log = Predictor._load_called_log(called_log_path)
    return score_instances(instances, scorer, calibrator, called_log)


class Predictor:
    """Loads trained artifacts and scores a new snapshot."""

    def __init__(self, artifact_dir):
        self.loader = ModelLoader(Path(artifact_dir))
        self.scorer, self.calibrator, _ = self.loader.load_pipeline()
        self.truncate_loans = self.scorer.truncate_loans
        self.data_loader = DataLoader()

    # ── Internal scoring path (shared by predict & recalibration) ────────────

    def _predict_probs(self, instances: list[dict]) -> np.ndarray:
        """instances → raw (uncalibrated) class probabilities via the scorer."""
        return self.scorer.raw_probs(instances)

    # ── Calibration refresh ───────────────────────────────────────────────────

    def _refresh_calibrator(self) -> None:
        """Refit the calibrator on the newest matured-label snapshot."""
        try:
            instances, _ = self.data_loader.load_train_portfolios(use_cache=True)
        except Exception as e:
            logger.warning(f"Recalibration skipped (train data unavailable): {e}")
            return

        usable_raw, _ = _get_usable_snapshots(instances)
        if not usable_raw:
            logger.warning("Recalibration skipped: no matured snapshots available.")
            return

        latest = usable_raw[-1]
        subset = [i for i in instances
                  if i["snapshot_date"] == latest and i["label"] >= 0]
        if not subset:
            logger.warning(f"Recalibration skipped: no labelled rows in {latest}.")
            return

        logger.info(f"Refreshing calibrator on snapshot {latest} "
                    f"({len(subset):,} instances)…")
        probs  = self._predict_probs(subset)
        y      = np.array([i["label"] for i in subset])
        strata = np.array([i["current_cat"] for i in subset])
        self.calibrator = StratifiedCalibrator(
            min_stratum_n=config.CALIBRATION_MIN_STRATUM_N
        ).fit(probs, y, strata)
        logger.info("Calibrator refreshed.")

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_snapshot_arg(snapshot_date) -> Optional[list]:
        """None/int/list -> list, falling back to config.PRED_SNAPSHOT_DATES."""
        if snapshot_date is None:
            snapshot_date = getattr(config, "PRED_SNAPSHOT_DATES", None)
        if snapshot_date is None:
            return None
        if isinstance(snapshot_date, (list, tuple)):
            return [int(s) for s in snapshot_date] or None
        return [int(snapshot_date)]

    def _default_output_path(self, resolved: list) -> Path:
        tag = str(resolved[0]) if len(resolved) == 1 else f"{min(resolved)}_{max(resolved)}"
        artifact_path = Path(self.loader.artifact_dir)
        base_dir = artifact_path if artifact_path.is_dir() else artifact_path.parent
        return base_dir / "predictions" / f"predictions_{tag}.csv"

    @staticmethod
    def _load_called_log(called_log_path=None) -> Optional[pd.DataFrame]:
        """Load the API-call ledger CSV (NATIONAL_CODE, CALLED_AT) if any."""
        path = called_log_path or getattr(config, "API_CALL_LOG", None)
        if not path:
            return None
        path = Path(path)
        if not path.exists():
            logger.warning(f"Call ledger not found: {path} — RECENTLY_CALLED flags skipped.")
            return None
        log_df = pd.read_csv(path)
        missing = {"NATIONAL_CODE", "CALLED_AT"} - set(log_df.columns)
        if missing:
            logger.warning(f"Call ledger {path} lacks columns {missing} — ignored.")
            return None
        logger.info(f"Loaded call ledger: {len(log_df):,} entries from {path}.")
        return log_df

    def predict(self, snapshot_date=None, output_path=None,
                called_log_path=None) -> pd.DataFrame:
        requested = self._normalize_snapshot_arg(snapshot_date)
        resolved  = self.data_loader.resolve_pred_snapshots(requested)
        called_log = self._load_called_log(called_log_path)
        logger.info(f"Starting inference for snapshot(s) {resolved}")

        if getattr(config, "RECALIBRATE_ON_PREDICT", False):
            self._refresh_calibrator()

        instances: list[dict] = []
        for snap in resolved:
            snap_instances, _ = self.data_loader.load_pred_portfolios(snap, self.truncate_loans)
            instances.extend(snap_instances)

        if not instances:
            logger.warning("No data found to predict.")
            return pd.DataFrame()

        results = score_instances(instances, self.scorer, self.calibrator, called_log)

        if output_path is None:
            output_path = self._default_output_path(resolved)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results):,} predictions to {output_path}")

        return results
