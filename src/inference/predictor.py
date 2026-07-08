"""
End-to-end scoring pipeline using the trained DeepSets + XGBoost pipeline.

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
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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


class Predictor:
    """Loads trained artifacts and scores a new snapshot."""

    def __init__(self, artifact_dir):
        self.loader = ModelLoader(Path(artifact_dir))
        (self.scaler, self.model, self.xgb_model,
         self.calibrator, self.max_loans, _) = self.loader.load_pipeline()
        self.device = self.loader.device
        self.data_loader = DataLoader()

    # ── Internal scoring path (shared by predict & recalibration) ────────────

    def _predict_probs(self, instances: list[dict]) -> np.ndarray:
        """instances → preprocess → DeepSets embeddings → raw XGBoost probs."""
        X_raw    = [inst["features"] for inst in instances]
        X_scaled = self.scaler.transform(X_raw)

        batch_size  = 512
        n_features  = X_scaled[0].shape[1]
        all_embeddings = []

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(X_scaled), batch_size), desc="Embedding batches"):
                batch_X = X_scaled[i : i + batch_size]
                n = len(batch_X)

                padded = np.zeros((n, self.max_loans, n_features), dtype=np.float32)
                mask   = np.ones((n, self.max_loans), dtype=bool)

                for j, arr in enumerate(batch_X):
                    seq_len = min(len(arr), self.max_loans)
                    padded[j, :seq_len] = arr[:seq_len]
                    mask[j, :seq_len]   = False

                t_feat = torch.from_numpy(padded).to(self.device)
                t_mask = torch.from_numpy(mask).to(self.device)

                emb = self.model.extract_embeddings(t_feat, t_mask)
                all_embeddings.append(emb.cpu().numpy())

        return self.xgb_model.predict_proba(np.vstack(all_embeddings))

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

    def predict(self, snapshot_date=None, output_path=None) -> pd.DataFrame:
        requested = self._normalize_snapshot_arg(snapshot_date)
        resolved  = self.data_loader.resolve_pred_snapshots(requested)
        logger.info(f"Starting inference for snapshot(s) {resolved}")

        if getattr(config, "RECALIBRATE_ON_PREDICT", False):
            self._refresh_calibrator()

        instances: list[dict] = []
        for snap in resolved:
            snap_instances, _ = self.data_loader.load_pred_portfolios(snap, self.max_loans)
            instances.extend(snap_instances)

        if not instances:
            logger.warning("No data found to predict.")
            return pd.DataFrame()

        strata    = np.array([i["current_cat"] for i in instances])
        probs_raw = self._predict_probs(instances)

        if isinstance(self.calibrator, StratifiedCalibrator):
            probs = self.calibrator.transform(probs_raw, strata)
        elif self.calibrator is not None:
            probs = self.calibrator.transform(probs_raw)
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
            "PREDICTED_CLASS":      cost_decisions(probs),
            # THE queue ranking score: calibrated P(entering severe)
            "RISK_SCORE":           severity_scores(probs),
            # Secondary/diagnostic: expected cost of taking no action
            "EXPECTED_COST":        risk_scores(probs),
        })

        # ── Rule flags: who does NOT compete for API budget ──────────────
        flags = np.where(
            strata >= config.CARVE_CURRENT_CAT_GE, "ALREADY_SEVERE", ""
        ).astype(object)
        if getattr(config, "PRED_DEDUP_LATEST", True) and len(resolved) > 1:
            latest_per_cust = results.groupby("NATIONAL_CODE")["SNAPSHOT_DATE"].transform("max")
            stale = (results["SNAPSHOT_DATE"] < latest_per_cust) & (flags == "")
            flags[stale.to_numpy()] = "SUPERSEDED"
        results["RULE_FLAG"] = flags

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

        logger.info(
            f"Queue: {n_queue:,} ranked | "
            f"{int((results['RULE_FLAG'] == 'ALREADY_SEVERE').sum()):,} already-severe (rule) | "
            f"{int((results['RULE_FLAG'] == 'SUPERSEDED').sum()):,} superseded. "
            f"At {config.API_RATE_PER_HOUR}/h the full queue takes "
            f"{n_queue / config.API_RATE_PER_HOUR:.0f}h to call."
        )

        if output_path is None:
            output_path = self._default_output_path(resolved)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results):,} predictions to {output_path}")

        return results
