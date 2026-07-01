"""
Data loading with vectorised portfolio grouping and cache-invalidation support.

Performance vs. original:
  - Replaced Python-loop groupby with numpy sort + np.unique indexing
  - Cache stored as compact NPZ (arrays) instead of joblib list-of-dicts
  - Cache manifest (JSON) tracks DATA_VERSION + schema hash for auto-invalidation
  - No full-DataFrame sort for truncation; secondary sort key handles it
"""

import json
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import project_config as config
from src.db.mssql_connection import MSSQLConnector

logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(extra: str = "") -> str:
    """
    Returns a short hex string that uniquely identifies the data schema.
    Changing DATA_VERSION, table name, META_COLS, or the DB name busts the cache.
    """
    payload = json.dumps(
        {
            "data_version": config.DATA_VERSION,
            "train_table": config.TRAIN_TABLE,
            "meta_cols": sorted(config.META_COLS),
            "database": config.MSSQL_DATABASE,
            "extra": extra,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".manifest.json")


def _cache_is_valid(cache_path: Path, key: str) -> bool:
    """Check that the cache file exists and its manifest key matches."""
    if not cache_path.exists():
        return False
    manifest = _manifest_path(cache_path)
    if not manifest.exists():
        return False
    try:
        with open(manifest) as f:
            data = json.load(f)
        return data.get("cache_key") == key
    except Exception:
        return False


def _write_manifest(cache_path: Path, key: str, meta: dict) -> None:
    manifest = _manifest_path(cache_path)
    with open(manifest, "w") as f:
        json.dump({"cache_key": key, **meta}, f, indent=2)


# ── Core data class ───────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads EDP_Feature_Train from MSSQL and groups records into customer
    portfolio instances.  Results are cached as NPZ for fast subsequent runs.
    """

    def __init__(self, mssql_connector: Optional[MSSQLConnector] = None):
        self.conn = mssql_connector

    # ── Label helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_capped_labels(y_arr: np.ndarray) -> int:
        """Worst-case capped label for a group of loan labels."""
        return int(min(int(y_arr.max()), config.NUM_CLASSES - 1))

    # ── Feature column discovery ──────────────────────────────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in config.META_COLS]

    # ── Vectorised portfolio grouping ─────────────────────────────────────────

    def process_raw_data(
        self,
        df: pd.DataFrame,
        max_loans: Optional[int] = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Convert flat DataFrame → list of customer portfolio dicts.

        Vectorised approach:
          1. Sort by (CUSTOMER, SNAPSHOT, DPD desc) — one sort, no per-group copies
          2. np.unique on composite key → group boundaries in O(N)
          3. Slice pre-extracted numpy arrays per group (no pandas overhead in loop)
        """
        logger.info(f"Vectorising {len(df):,} rows into customer portfolios…")

        feature_cols = self.get_feature_columns(df)

        # Sort: primary by group keys, secondary by DPD descending (for truncation)
        df = df.sort_values(
            [config.CUSTOMER_COL, config.SNAPSHOT_COL, "WORST_FUTURE_DPD"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

        # Pre-extract numpy arrays (avoids per-group pandas overhead)
        X_all        = df[feature_cols].values.astype(np.float32)   # (N, F)
        y_all        = df[config.TARGET_COL].values                  # (N,)
        customers    = df[config.CUSTOMER_COL].to_numpy(dtype=object)                # (N,)
        snapshots    = df[config.SNAPSHOT_COL].to_numpy(dtype=object)                # (N,)

        # Build composite group key and find group boundaries
        composite = np.char.add(
            customers.astype(str),
            np.char.add("_", snapshots.astype(str)),
        )
        _, start_idx, group_sizes = np.unique(
            composite, return_index=True, return_counts=True
        )

        # Re-sort by original order (np.unique returns lexicographic order)
        order = np.argsort(start_idx)
        start_idx  = start_idx[order]
        group_sizes = group_sizes[order]

        instances: list[dict] = []
        for start, size in tqdm(
            zip(start_idx, group_sizes),
            total=len(start_idx),
            desc="Building portfolios",
        ):
            end = start + size
            keep = min(size, max_loans) if max_loans else size

            feature_matrix = X_all[start : start + keep]          # (keep, F)
            label = int(min(int(y_all[start:end].max()), config.NUM_CLASSES - 1))

            instances.append(
                {
                    "national_code":  customers[start],
                    "snapshot_date":  snapshots[start],
                    "n_loans":        int(size),
                    "features":       feature_matrix,
                    "label":          label,
                }
            )

        logger.info(f"Created {len(instances):,} customer portfolio instances.")
        return instances, feature_cols

    # ── NPZ cache I/O ─────────────────────────────────────────────────────────

    def _save_cache(
        self,
        cache_path: Path,
        instances: list[dict],
        feature_cols: list[str],
        max_loans: int,
        key: str,
    ) -> None:
        """
        Save instances as a compact NPZ file.
        Structure:
          features_flat : (N_total_loans, N_features)  — all loan arrays stacked
          offsets       : (N_instances + 1,)            — cumsum for slicing
          labels        : (N_instances,)
          n_loans       : (N_instances,)
          national_codes: (N_instances,)  object array
          snapshot_dates: (N_instances,)  object array
          feature_cols  : serialised as JSON in separate manifest
        """
        logger.info(f"Saving portfolio cache to {cache_path}…")

        features_flat = np.vstack([inst["features"] for inst in instances])
        sizes = np.array([inst["features"].shape[0] for inst in instances], dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        labels  = np.array([inst["label"] for inst in instances], dtype=np.int32)
        n_loans = np.array([inst["n_loans"] for inst in instances], dtype=np.int32)
        national_codes = np.array([inst["national_code"] for inst in instances])
        snapshot_dates = np.array([inst["snapshot_date"] for inst in instances])

        np.savez_compressed(
            cache_path,
            features_flat=features_flat,
            offsets=offsets,
            labels=labels,
            n_loans=n_loans,
            national_codes=national_codes,
            snapshot_dates=snapshot_dates,
        )

        _write_manifest(
            cache_path,
            key,
            {"feature_cols": feature_cols, "max_loans": max_loans, "n_instances": len(instances)},
        )
        logger.info(f"Cache saved ({cache_path.stat().st_size / 1e6:.1f} MB).")

    def _load_cache(self, cache_path: Path) -> tuple[list[dict], list[str]]:
        """Restore instances from NPZ cache."""
        logger.info(f"Loading portfolio cache from {cache_path}…")
        with np.load(cache_path, allow_pickle=True) as npz:
            features_flat  = npz["features_flat"]
            offsets        = npz["offsets"]
            labels         = npz["labels"]
            n_loans        = npz["n_loans"]
            national_codes = npz["national_codes"]
            snapshot_dates = npz["snapshot_dates"]

        with open(_manifest_path(cache_path)) as f:
            manifest = json.load(f)
        feature_cols = manifest["feature_cols"]

        instances = [
            {
                "national_code": national_codes[i],
                "snapshot_date": snapshot_dates[i],
                "n_loans":       int(n_loans[i]),
                "features":      features_flat[offsets[i] : offsets[i + 1]],
                "label":         int(labels[i]),
            }
            for i in range(len(labels))
        ]
        logger.info(f"Loaded {len(instances):,} instances from cache.")
        return instances, feature_cols

    # ── Public API ────────────────────────────────────────────────────────────

    def load_train_portfolios(
        self,
        snapshot_dates: Optional[list] = None,
        max_loans: Optional[int] = None,
        use_cache: bool = True,
    ) -> tuple[list[dict], list[str]]:
        """
        Load training data from MSSQL → portfolio instances.
        Returns (instances, feature_names).

        Cache is invalidated when:
          - DATA_VERSION changes in project_config.py
          - Table name, META_COLS, or DB name changes
          - max_loans changes
        """
        extra = str(max_loans or "auto")
        key   = _cache_key(extra)
        cache_path = config.DATA_DIR / "train_portfolios_cache.npz"

        if use_cache and _cache_is_valid(cache_path, key):
            return self._load_cache(cache_path)

        if use_cache and cache_path.exists():
            logger.info("Cache key mismatch — regenerating portfolio cache.")

        conn, close_conn = self._get_conn()
        try:
            df = conn.load_training_data(snapshot_dates=snapshot_dates)
            instances, feature_cols = self.process_raw_data(df, max_loans)
            if use_cache:
                ml = max_loans or (
                    int(np.percentile([i["n_loans"] for i in instances], 99))
                )
                self._save_cache(cache_path, instances, feature_cols, ml, key)
            return instances, feature_cols
        finally:
            if close_conn:
                conn.close()

    def load_pred_portfolios(
        self,
        snapshot_date: int,
        max_loans: Optional[int] = None,
        use_cache: bool = True,
    ) -> tuple[list[dict], list[str]]:
        """Load prediction data for a single snapshot."""
        extra = f"pred_{snapshot_date}_{max_loans or 'auto'}"
        key   = _cache_key(extra)
        cache_path = config.DATA_DIR / f"pred_portfolios_cache_{snapshot_date}.npz"

        if use_cache and _cache_is_valid(cache_path, key):
            return self._load_cache(cache_path)

        conn, close_conn = self._get_conn()
        try:
            df = conn.load_prediction_data(snapshot_date=snapshot_date)
            instances, feature_cols = self.process_raw_data(df, max_loans)
            if use_cache:
                self._save_cache(cache_path, instances, feature_cols, max_loans or 99, key)
            return instances, feature_cols
        finally:
            if close_conn:
                conn.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_conn(self) -> tuple[MSSQLConnector, bool]:
        if self.conn is not None:
            return self.conn, False
        return MSSQLConnector(), True
