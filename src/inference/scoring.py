"""
run_scoring() — the manager-code integration point.

For a codebase that owns its own data sourcing and just needs "give me a
dataframe, get back a ranked queue": import this module, build a
ScoringParams (or a plain dict of the fields it needs), call run_scoring().
No database access, no knowledge of this project's training pipeline.

    from src.inference.scoring import run_scoring
    from src.inference.scoring_params import ScoringParams

    params = ScoringParams(bundle_path="model_bundle.pkl", output_path="queue.csv")
    queue = run_scoring(df, params)

Any field left unset on ScoringParams is filled from project_config.py,
and a warning names exactly which ones — see scoring_params.py.

This is a thin wrapper over the same score_instances() transform used by
Predictor.predict() and score_dataframe(); all three stay in lock-step by
construction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.calibration import StratifiedCalibrator
from src.inference.model_loader import ModelLoader
from src.inference.predictor import score_instances
from src.inference.scoring_params import ScoringParams

logger = logging.getLogger(__name__)


def _load_called_log(path) -> pd.DataFrame | None:
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


def run_scoring(df: pd.DataFrame, params: ScoringParams) -> pd.DataFrame:
    """
    Score `df` (same columns as TRAIN_TABLE, minus WORST_FUTURE_CAT/DPD —
    prediction time) against the model in `params.bundle_path`. Returns the
    ranked queue DataFrame; also writes it to `params.output_path` if set.
    """
    from src.data.data_loader import DataLoader   # local: explicit footprint

    params = params.resolve()

    loader = ModelLoader(Path(params.bundle_path))
    # `features` = the feature list the bundle was fitted on. Both frames
    # below are projected to it by name, so the caller's column order is
    # irrelevant and a column the model has never seen is dropped with a
    # warning rather than shifting every other column's meaning.
    scorer, calibrator, features = loader.load_pipeline()
    dl = DataLoader()

    if params.calibration_df is not None:
        cal_inst, _ = dl.process_raw_data(params.calibration_df, scorer.truncate_loans,
                                          scorer.grain, feature_order=features)
        cal_inst = [i for i in cal_inst if i["label"] >= 0]
        if cal_inst:
            probs  = scorer.raw_probs(cal_inst)
            y      = np.array([i["label"] for i in cal_inst])
            strata = np.array([i["current_cat"] for i in cal_inst])
            calibrator = StratifiedCalibrator(
                min_stratum_n=params.calibration_min_stratum_n
            ).fit(probs, y, strata)
            logger.info(f"Calibrator refreshed on {len(cal_inst):,} supplied instances.")
        else:
            logger.warning("calibration_df had no matured/labelled rows — calibrator unchanged.")

    instances, _ = dl.process_raw_data(df, scorer.truncate_loans, scorer.grain,
                                       feature_order=features)
    called_log = _load_called_log(params.called_log_path)

    results = score_instances(
        instances, scorer, calibrator, called_log,
        cost_matrix=params.cost_matrix,
        carve_current_cat_ge=params.carve_current_cat_ge,
        certainty_act_threshold=params.certainty_act_threshold,
        api_data_ttl_days=params.api_data_ttl_days,
        pred_dedup_latest=params.pred_dedup_latest,
    )

    if params.output_path:
        out = Path(params.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        logger.info(f"Saved {len(results):,} rows to {out}")

    return results
