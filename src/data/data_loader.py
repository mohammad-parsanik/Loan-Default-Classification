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
from src.data.column_contract import CONTRACT_VERSION, FEATURE_ORDER, feature_ordinal
from src.data.feed_checks import assert_feed_invariants, label_horizons
from src.data import temporal_split
from src.data.temporal_split import filter_mature_snapshots, register_label_horizons

try:
    from src.db.mssql_connection import MSSQLConnector
except ImportError:          # pyodbc absent on dev machines without DB access
    MSSQLConnector = None    # cache-only workflows still function

logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(extra: str = "", grain: Optional[str] = None,
               feature_order: Optional[list] = None) -> str:
    """
    Returns a short hex string that uniquely identifies the data schema.
    Changing DATA_VERSION, the table, META_COLS, the DB name, the grain, OR
    the column contract (its version, or the feature list IN ORDER) busts the
    cache.

    The feature list is hashed as an ordered list, not a set, on purpose: a
    cache built when features meant one set of positions must not be reused
    by a model fitted against another. That was the gap — the key described
    the schema's shape but not its column identity, so a reorder survived it.
    """
    payload = json.dumps(
        {
            "data_version": config.DATA_VERSION,
            "train_table": config.TRAIN_TABLE,
            "meta_cols": sorted(config.META_COLS),
            "database": config.MSSQL_DATABASE,
            "contract_version": CONTRACT_VERSION,
            "feature_order": list(feature_order or FEATURE_ORDER),
            # An instance means something different per grain, so the two
            # must never share a cache file. Taken as an argument, not read
            # from config, so an explicit-grain caller keys its own cache.
            "grain": grain or config.PREDICTION_GRAIN,
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

    # ── Feature projection (by NAME, never by position) ───────────────────────

    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> list[str]:
        """
        The frame's feature columns in CONTRACT order — used when no explicit
        order is supplied. Columns the contract does not know (a synthetic
        fixture's, or an appended column 72) sort after the known ones, by
        name, so the result is a function of the column SET and never of the
        order `SELECT *` happened to return them in.
        """
        present = [c for c in df.columns if c not in config.META_COLS]
        return sorted(present, key=lambda c: (feature_ordinal(c), c))

    @staticmethod
    def project_features(df: pd.DataFrame, order: Optional[list[str]] = None) -> list[str]:
        """
        Decide the feature columns to read from `df`, by name.

        `order` is the authoritative list — the contract's FEATURE_ORDER when
        training, or a trained model's own saved feature list when scoring
        (which may be an older, shorter contract). Mismatch policy:

          missing feature  -> raise, naming it. The model cannot be fed a
                              column it was fitted on but did not receive;
                              at identical width every downstream statistic
                              would silently apply to the wrong column.
          unexpected column -> warn, naming it, and drop. This is the NORMAL
                              path once the feed appends a column 72: an
                              already-trained model must keep scoring on the
                              features it knows.

        `order=None` falls back to get_feature_columns (contract order over
        whatever the frame happens to hold).
        """
        if not order:
            # Empty as well as None: a legacy artifact that records no feature
            # list falls back to contract order rather than to zero features.
            return DataLoader.get_feature_columns(df)

        order = list(order)
        present = set(df.columns)
        missing = [c for c in order if c not in present]
        if missing:
            raise KeyError(
                f"Frame is missing {len(missing)} required feature column(s): "
                f"{missing}. Scoring or training against a partial feature set "
                "would misalign every column-position-keyed transform."
            )

        known = set(order) | set(config.META_COLS)
        extra = [c for c in df.columns if c not in known]
        if extra:
            logger.warning(
                f"Dropping {len(extra)} column(s) the model does not know: "
                f"{extra}. Expected when the feed appends a new feature — the "
                "model keeps scoring on the features it was fitted on."
            )
        return order

    # ── Vectorised portfolio grouping ─────────────────────────────────────────

    def process_raw_data(
        self,
        df: pd.DataFrame,
        max_loans: Optional[int] = None,
        grain: Optional[str] = None,
        feature_order: Optional[list[str]] = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Convert flat DataFrame → list of instance dicts.

        `grain` (default config.PREDICTION_GRAIN) decides what an instance is:
          "portfolio" — one per (CUSTOMER, SNAPSHOT); loans collapsed, label
                        = max over the customer's loans.
          "loan"      — one per input row; label and current_cat are that
                        loan's own, as the ETL computes them. `max_loans` is
                        irrelevant here (nothing to truncate) and is ignored.

        Vectorised approach:
          1. Sort by (CUSTOMER, SNAPSHOT, DPD desc) — one sort, no per-group copies
          2. np.unique on composite key → group boundaries in O(N)
          3. Slice pre-extracted numpy arrays per group (no pandas overhead in loop)

        `feature_order` names the features to read, in order — the contract's
        FEATURE_ORDER when training, a model's own saved list when scoring.
        See project_features. Omit it and the frame's own feature columns are
        used, sorted into contract order.

        Both grains carry `portfolio_n_loans` (the customer's true loan count
        at that snapshot) so downstream output keeps portfolio context even
        when scoring one loan at a time. It is NOT a model feature.

        The result is a function of the frame's CONTENT alone: columns are
        projected by name and rows are put in a canonical order below, so a
        shuffled or reordered copy of the same data yields identical
        instances.
        """
        grain = grain or config.PREDICTION_GRAIN
        if grain not in ("loan", "portfolio"):
            raise ValueError(f"unknown PREDICTION_GRAIN {grain!r}")
        logger.info(f"Vectorising {len(df):,} rows into {grain} instances…")

        # Guard against duplicate header rows embedded in the data (e.g. a
        # source CSV concatenated from multiple export batches/runs) — such
        # a row has the literal column name as its CUSTOMER_COL value, which
        # can never occur in real data, and otherwise blows up label casting.
        header_leak = df[config.CUSTOMER_COL].astype(str) == config.CUSTOMER_COL
        if header_leak.any():
            logger.warning(
                f"Dropping {int(header_leak.sum())} duplicate header row(s) "
                "embedded in the input data."
            )
            df = df.loc[~header_leak].reset_index(drop=True)

        feature_cols = self.project_features(df, feature_order)

        # Canonical row order, established once and inherited by everything
        # downstream. Primary by group keys, then *current* DPD descending so
        # truncation keeps the currently-worst loans. Must NOT use a label
        # column (WORST_FUTURE_DPD): that leaks the future into loan selection
        # and does not exist in the prediction table.
        #
        # LOAN_ID is the final tie-break and it is what makes the key UNIQUE:
        # the feed guarantees one row per (LOAN_ID, SNAPSHOT_DATE), so no tie
        # can survive it. Without it the sort is stable, which only means ties
        # keep their ARRIVAL order — and DPD_DAYS is 0 for most loans, so the
        # ties are enormous. Truncation, XGBoost's index-based subsample /
        # colsample draws, and every group scan below then depend on how the
        # source laid the rows out rather than on the data.
        sort_keys = [config.CUSTOMER_COL, config.SNAPSHOT_COL, "DPD_DAYS"]
        ascending = [True, True, False]
        if config.ID_COL in df.columns:
            sort_keys.append(config.ID_COL)
            ascending.append(True)
        else:
            logger.warning(
                f"{config.ID_COL} absent — row order can only be made canonical "
                "up to ties on (customer, snapshot, DPD_DAYS)."
            )
        df = df.sort_values(sort_keys, ascending=ascending).reset_index(drop=True)

        # Pre-extract numpy arrays (avoids per-group pandas overhead).
        # Coerce rather than astype: source exports occasionally contain
        # corrupted numeric strings (e.g. a decimal point mangled into a
        # '/' by an upstream Excel re-save). Bad cells become NaN, which
        # DomainAwareImputer already handles, instead of crashing the run.
        X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        bad_mask = X_df.isna() & df[feature_cols].notna()
        if bad_mask.to_numpy().any():
            bad_cols = bad_mask.any(axis=0)
            examples = {
                col: df.loc[bad_mask[col], col].iloc[0]
                for col in bad_cols[bad_cols].index
            }
            logger.warning(
                f"{int(bad_mask.to_numpy().sum())} unparseable numeric values "
                f"coerced to NaN across {len(examples)} column(s): {examples}"
            )
        X_all        = X_df.values.astype(np.float32)   # (N, F)
        # Prediction table has no label column — instances get label = -1
        has_target   = config.TARGET_COL in df.columns
        y_all        = df[config.TARGET_COL].values if has_target else None  # (N,)
        # Current worst category across the FULL portfolio (pre-truncation),
        # capped like the label; used for stratified evaluation.
        cat_all      = df["LOAN_CATEGORY"].values                    # (N,)
        customers    = df[config.CUSTOMER_COL].to_numpy(dtype=object)                # (N,)
        snapshots    = df[config.SNAPSHOT_COL].to_numpy(dtype=object)                # (N,)
        # Identifies the scored row at loan grain. A synthetic/legacy frame may
        # not carry it; downstream output falls back to None.
        loan_ids = (
            df[config.ID_COL].to_numpy(dtype=object)
            if config.ID_COL in df.columns else np.full(len(df), None, dtype=object)
        )

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
        cap = config.NUM_CLASSES - 1

        if grain == "loan":
            # Every row is its own instance; the group scan is only needed to
            # attach the customer's loan count to each of their rows.
            portfolio_sizes = np.repeat(group_sizes, group_sizes)
            for i in tqdm(range(len(df)), desc="Building loan instances"):
                instances.append(
                    {
                        "national_code":     customers[i],
                        "snapshot_date":     snapshots[i],
                        "loan_id":           loan_ids[i],
                        "n_loans":           1,
                        "portfolio_n_loans": int(portfolio_sizes[i]),
                        "features":          X_all[i : i + 1],       # (1, F)
                        "label":             int(min(int(y_all[i]), cap)) if has_target else -1,
                        "current_cat":       int(min(int(cat_all[i]), cap)),
                    }
                )
        else:
            for start, size in tqdm(
                zip(start_idx, group_sizes),
                total=len(start_idx),
                desc="Building portfolios",
            ):
                end = start + size
                keep = min(size, max_loans) if max_loans else size

                feature_matrix = X_all[start : start + keep]          # (keep, F)
                label = (
                    int(min(int(y_all[start:end].max()), cap))
                    if has_target else -1
                )
                current_cat = int(min(int(cat_all[start:end].max()), cap))

                instances.append(
                    {
                        "national_code":     customers[start],
                        "snapshot_date":     snapshots[start],
                        "loan_id":           None,   # a portfolio is not one loan
                        "n_loans":           int(size),
                        "portfolio_n_loans": int(size),
                        "features":          feature_matrix,
                        "label":             label,
                        "current_cat":       current_cat,
                    }
                )

        logger.info(f"Created {len(instances):,} {grain} instances.")
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
        current_cats = np.array([inst["current_cat"] for inst in instances], dtype=np.int32)
        n_loans = np.array([inst["n_loans"] for inst in instances], dtype=np.int32)
        national_codes = np.array([inst["national_code"] for inst in instances])
        snapshot_dates = np.array([inst["snapshot_date"] for inst in instances])
        loan_ids = np.array([inst.get("loan_id") for inst in instances], dtype=object)
        portfolio_n_loans = np.array(
            [inst.get("portfolio_n_loans", inst["n_loans"]) for inst in instances],
            dtype=np.int32,
        )

        np.savez_compressed(
            cache_path,
            features_flat=features_flat,
            offsets=offsets,
            labels=labels,
            current_cats=current_cats,
            n_loans=n_loans,
            national_codes=national_codes,
            snapshot_dates=snapshot_dates,
            loan_ids=loan_ids,
            portfolio_n_loans=portfolio_n_loans,
        )

        _write_manifest(
            cache_path,
            key,
            {"feature_cols": feature_cols, "max_loans": max_loans,
             "n_instances": len(instances),
             # snapshot -> LABEL_HORIZON_DATE, so a cache-only run (no DB on
             # this machine) can still tell a matured snapshot from an
             # immature one without re-deriving it from the wall clock.
             "label_horizons": {str(k): int(v)
                                for k, v in temporal_split.LABEL_HORIZONS.items()}},
        )
        logger.info(f"Cache saved ({cache_path.stat().st_size / 1e6:.1f} MB).")

    def _load_cache(self, cache_path: Path) -> tuple[list[dict], list[str]]:
        """Restore instances from NPZ cache."""
        logger.info(f"Loading portfolio cache from {cache_path}…")
        with np.load(cache_path, allow_pickle=True) as npz:
            features_flat  = npz["features_flat"]
            offsets        = npz["offsets"]
            labels         = npz["labels"]
            current_cats   = npz["current_cats"]
            n_loans        = npz["n_loans"]
            national_codes = npz["national_codes"]
            snapshot_dates = npz["snapshot_dates"]
            loan_ids          = npz["loan_ids"]
            portfolio_n_loans = npz["portfolio_n_loans"]

        with open(_manifest_path(cache_path)) as f:
            manifest = json.load(f)
        feature_cols = manifest["feature_cols"]
        register_label_horizons(manifest.get("label_horizons", {}))

        instances = [
            {
                "national_code":     national_codes[i],
                "snapshot_date":     snapshot_dates[i],
                "loan_id":           loan_ids[i],
                "n_loans":           int(n_loans[i]),
                "portfolio_n_loans": int(portfolio_n_loans[i]),
                "features":          features_flat[offsets[i] : offsets[i + 1]],
                "label":             int(labels[i]),
                "current_cat":       int(current_cats[i]),
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
        grain: Optional[str] = None,
        feature_order: Optional[list[str]] = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Load training data from MSSQL → instances at `grain`
        (default config.PREDICTION_GRAIN). Returns (instances, feature_names).

        Cache is invalidated when:
          - DATA_VERSION changes in project_config.py
          - Table name, META_COLS, or DB name changes
          - the column contract changes (version, or the feature list/order)
          - max_loans or the grain changes
        """
        extra = str(max_loans or "auto")
        key   = _cache_key(extra, grain, feature_order)
        cache_path = config.DATA_DIR / "train_portfolios_cache.npz"

        if use_cache and _cache_is_valid(cache_path, key):
            return self._load_cache(cache_path)

        if use_cache and cache_path.exists():
            logger.info("Cache key mismatch — regenerating portfolio cache.")

        conn, close_conn = self._get_conn()
        try:
            self._warn_on_failed_etl_runs(conn)
            df = conn.load_training_data(snapshot_dates=snapshot_dates)
            df = self._ingest_checks(df)
            instances, feature_cols = self.process_raw_data(
                df, max_loans, grain, feature_order=feature_order or FEATURE_ORDER
            )
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
        grain: Optional[str] = None,
        feature_order: Optional[list[str]] = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Load prediction data for a single snapshot at `grain`.

        `feature_order` is the trained model's own feature list — pass it so a
        model older than the current contract keeps being fed exactly the
        columns it was fitted on. Defaults to the contract's order.
        """
        extra = f"pred_{snapshot_date}_{max_loans or 'auto'}"
        key   = _cache_key(extra, grain, feature_order)
        cache_path = config.DATA_DIR / f"pred_portfolios_cache_{snapshot_date}.npz"

        if use_cache and _cache_is_valid(cache_path, key):
            return self._load_cache(cache_path)

        conn, close_conn = self._get_conn()
        try:
            df = conn.load_prediction_data(snapshot_date=snapshot_date)
            # TRAIN_TABLE carries WORST_FUTURE_CAT/DPD for every row, but on
            # an immature snapshot those values are degenerate ("worst
            # category observed so far", not the true future outcome) —
            # never real labels. Drop them so process_raw_data's normal
            # "column absent -> label = -1" path applies here too.
            df = self._ingest_checks(df)
            df = df.drop(columns=[config.TARGET_COL, "WORST_FUTURE_DPD"], errors="ignore")
            instances, feature_cols = self.process_raw_data(
                df, max_loans, grain, feature_order=feature_order or FEATURE_ORDER
            )
            if use_cache:
                self._save_cache(cache_path, instances, feature_cols, max_loans or 99, key)
            return instances, feature_cols
        finally:
            if close_conn:
                conn.close()

    def resolve_pred_snapshots(self, requested: Optional[list] = None) -> list:
        """
        Decide which snapshot(s) to score.

          - `requested` given: keep only dates present in TRAIN_TABLE (warn on
            misses). If any survive, return them.
          - Otherwise (nothing requested, or none of it found): auto-select
            every currently-immature snapshot (labels not yet matured).
          - If that's empty too: fall back to the single latest snapshot.
        """
        conn, close_conn = self._get_conn()
        try:
            self._warn_on_failed_etl_runs(conn)
            available = conn.get_available_snapshots()
            # Maturity comes from the feed's own horizon column, not the
            # calendar — read it before deciding what is still immature.
            self._register_horizons_from_db(conn)
        finally:
            if close_conn:
                conn.close()

        if not available:
            raise RuntimeError(f"No snapshots found in {config.TRAIN_TABLE}.")
        available = sorted(int(s) for s in available)

        if requested:
            requested = [int(s) for s in requested]
            found   = [s for s in requested if s in available]
            missing = [s for s in requested if s not in available]
            if missing:
                logger.warning(f"Requested snapshot(s) not found in {config.TRAIN_TABLE}: {missing}")
            if found:
                return sorted(found)
            logger.warning("None of the requested snapshots were found — falling back to auto-selection.")

        immature = sorted(set(available) - set(filter_mature_snapshots(available)))
        if immature:
            return immature

        logger.warning("No immature snapshot found — falling back to the single latest snapshot.")
        return [available[-1]]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _ingest_checks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Everything that must happen between "rows arrived" and "rows are used":
        record each snapshot's label horizon, then check the feed's row-level
        invariants. Violations are logged, not raised — a handful of bad rows
        in a 577k-row snapshot should not abort a multi-hour run, but they
        must not pass unremarked either.
        """
        register_label_horizons(label_horizons(df))
        assert_feed_invariants(df)
        return df

    @staticmethod
    def _register_horizons_from_db(conn) -> None:
        """Populate the snapshot -> LABEL_HORIZON_DATE map from the table."""
        try:
            register_label_horizons(conn.get_label_horizons())
        except Exception as e:
            logger.info(
                f"{config.HORIZON_COL} not read ({e}) — maturity falls back to "
                "the calendar rule."
            )

    @staticmethod
    def _warn_on_failed_etl_runs(conn) -> None:
        """
        A snapshot missing from the table is otherwise indistinguishable from
        one whose ETL run never happened or stopped halfway. The upstream job
        ledger is the only thing that can tell them apart, so read it before
        deciding that what is in the table is all there is. Advisory only:
        the ledger may not be readable from this account, and a run that reads
        the table successfully should not be blocked on bookkeeping.
        """
        try:
            runs = conn.get_etl_runs()
        except Exception as e:
            logger.info(f"ETL job ledger not read ({e}) — snapshot completeness unverified.")
            return
        if runs is None or not len(runs):
            return

        incomplete = runs[runs["status"].astype(str).str.upper() != "SUCCESS"]
        if len(incomplete):
            logger.warning(
                f"{len(incomplete)} upstream ETL run(s) did not report SUCCESS — "
                "the snapshots they would have produced are absent or partial:"
            )
            for row in incomplete.head(12).itertuples(index=False):
                logger.warning(
                    f"  {getattr(row, 'snapshot_date', '?')}: "
                    f"status={getattr(row, 'status', '?')} "
                    f"last_step={getattr(row, 'last_step', '?')}"
                )
        else:
            logger.info(f"ETL job ledger: {len(runs)} run(s), all SUCCESS.")

    def _get_conn(self):
        if self.conn is not None:
            return self.conn, False
        if MSSQLConnector is None:
            raise RuntimeError(
                "pyodbc/MSSQL is unavailable on this machine and no valid "
                "NPZ cache was found. Run on the training server, or copy "
                "data/train_portfolios_cache.npz (+ manifest) here."
            )
        return MSSQLConnector(), True
