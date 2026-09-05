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


def _cache_dir(stage: str, key: str) -> Path:
    """
    Directory holding one NPZ per snapshot for a given (stage, schema key).

    Snapshots are cached individually because that is how the upstream ETL
    treats them. Each monthly load recomputes the last 7 snapshots: the newest
    (T) is fresh, the oldest of the seven (T-7) has just matured, and the ones
    between are rewritten without changing anything the ML side reads. A
    matured snapshot is never touched again, so its NPZ is permanent; an
    immature one is provisional and gets rewritten until it matures. One
    monolithic cache file cannot express either half — a single new snapshot
    forced a full rebuild of all ~40M rows, and a rewritten immature snapshot
    was invisible.
    """
    return config.DATA_DIR / "snapshots" / f"{stage}_{key}"


def _snapshot_npz(cache_dir: Path, snapshot: int) -> Path:
    return cache_dir / f"{int(snapshot)}.npz"


def _cached_snapshots(cache_dir: Path, key: str) -> list[int]:
    """Snapshots present in `cache_dir` with a manifest matching `key`."""
    if not cache_dir.is_dir():
        return []
    found = []
    for npz in cache_dir.glob("*.npz"):
        if npz.stem.isdigit() and _cache_is_valid(npz, key):
            found.append(int(npz.stem))
    return sorted(found)


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
        meta: Optional[dict] = None,
    ) -> None:
        """
        Save one snapshot's instances as an NPZ file.
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
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Fill a preallocated block rather than np.vstack-ing the per-instance
        # arrays: at loan grain there are tens of millions of (1, F) slices,
        # and vstack builds an atleast_2d list over all of them and its own
        # concatenate temporary before returning the same block.
        sizes = np.fromiter(
            (inst["features"].shape[0] for inst in instances),
            dtype=np.int64, count=len(instances),
        )
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        features_flat = np.empty(
            (int(offsets[-1]), instances[0]["features"].shape[1]), dtype=np.float32
        )
        for inst, start, end in zip(instances, offsets[:-1], offsets[1:]):
            features_flat[start:end] = inst["features"]
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

        # Uncompressed on purpose: at the >=7B population features_flat is
        # several GB, and single-threaded zlib over it costs tens of minutes
        # on every rebuild to save a few GB of disk. np.load reads either.
        np.savez(
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
                                for k, v in temporal_split.LABEL_HORIZONS.items()},
             **(meta or {})},
        )
        logger.info(f"Cache saved ({cache_path.stat().st_size / 1e6:.1f} MB).")

    def _maybe_conn(self):
        """Like _get_conn, but returns (None, False) when there is no DB on this
        machine — a cached snapshot directory is then the only source."""
        try:
            return self._get_conn()
        except Exception as e:
            logger.info(f"No DB connection ({e}) — cache-only.")
            return None, False

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

        Snapshots are read and vectorised ONE AT A TIME, never as a single
        `SELECT *` over the whole table. At the ≤7B population the table is
        ~43M rows; one fetchall of that is tens of GB of pyodbc row objects
        before pandas even builds the frame, and the sort/coerce steps in
        process_raw_data each copy it again. Per snapshot the peak is ~1/20th
        of that, and the result is identical: instances group by
        (customer, snapshot), so no group can span a snapshot boundary.
        What per-snapshot processing does NOT preserve on its own is the
        canonical INSTANCE order — see _canonical_order.

        Only mature snapshots are read; the split discards immature ones
        anyway, so loading them was ~25% of the rows for nothing. Each mature
        snapshot is cached as its OWN NPZ (see _cache_dir) and is permanent —
        a matured snapshot never changes upstream — so next month's run reads
        one new snapshot from the DB and the rest from disk.

        Instance order is (SNAPSHOT_DATE, NATIONAL_CODE, DPD_DAYS desc,
        LOAN_ID): snapshots are processed in ascending order and
        process_raw_data sorts within each. That is a function of the data
        alone, which is what order independence requires — the previous
        customer-major order was one valid choice of canonical order, not the
        requirement itself, and snapshot-major is the order per-snapshot
        caching produces for free.

        Cache is invalidated when:
          - DATA_VERSION changes in project_config.py
          - Table name, META_COLS, or DB name changes
          - the column contract changes (version, or the feature list/order)
          - max_loans or the grain changes
        A snapshot that has since matured is simply a file that is not there.
        """
        extra = str(max_loans or "auto")
        key   = _cache_key(extra, grain, feature_order)
        cache_dir = _cache_dir("train", key)

        conn, close_conn = self._maybe_conn()
        try:
            snaps = self._training_snapshots(conn, cache_dir, key, snapshot_dates)

            instances: list[dict] = []
            feature_cols: list[str] = []
            for n, snap in enumerate(snaps, 1):
                npz = _snapshot_npz(cache_dir, snap)
                if use_cache and _cache_is_valid(npz, key):
                    part, feature_cols = self._load_cache(npz)
                else:
                    if conn is None:
                        raise RuntimeError(
                            f"Snapshot {snap} is not cached and no DB connection "
                            "is available to read it."
                        )
                    logger.info(f"Snapshot {snap} ({n}/{len(snaps)}) — reading from DB…")
                    df = conn.load_training_data(snapshot_dates=[snap])
                    df = self._ingest_checks(df)
                    part, feature_cols = self.process_raw_data(
                        df, max_loans, grain, feature_order=feature_order or FEATURE_ORDER
                    )
                    del df
                    if use_cache and part:
                        ml = max_loans or (
                            int(np.percentile([i["n_loans"] for i in part], 99))
                        )
                        self._save_cache(npz, part, feature_cols, ml, key,
                                         meta={"snapshot": int(snap), "mature": True})
                instances.extend(part)

            logger.info(f"{len(instances):,} instances from {len(snaps)} snapshot(s).")
            return instances, feature_cols
        finally:
            if close_conn and conn is not None:
                conn.close()

    def _training_snapshots(self, conn, cache_dir: Path, key: str,
                            requested: Optional[list] = None) -> list:
        """
        The snapshots to train on, ascending: `requested` if the caller named
        them, else every mature snapshot in the table. With no DB reachable
        (a dev machine with a copied cache directory) it is whatever the cache
        holds.
        """
        if requested:
            return sorted(int(s) for s in requested)

        if conn is None:
            snaps = _cached_snapshots(cache_dir, key)
            if not snaps:
                raise RuntimeError(
                    f"No DB connection and no cached snapshots in {cache_dir}."
                )
            logger.info(f"No DB — using {len(snaps)} cached snapshot(s).")
            return snaps

        self._warn_on_failed_etl_runs(conn)
        # Before anything else, so LABEL_HORIZONS knows about every snapshot in
        # the table — including the immature ones we are about to skip.
        self._register_horizons_from_db(conn)
        snaps = filter_mature_snapshots(conn.get_available_snapshots())
        if not snaps:
            raise RuntimeError(
                f"No mature snapshots in {config.TRAIN_TABLE} — nothing to train on."
            )
        logger.info(f"{len(snaps)} mature snapshot(s): {snaps[0]} … {snaps[-1]}")
        return snaps

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

        Scoring targets immature snapshots, and the ETL rewrites the newest 7
        of those on every monthly load — so unlike a mature snapshot's cache,
        this one is PROVISIONAL. It is reused only while the ETL run that
        produced it is still the newest successful one; after the next load it
        is rewritten. If the job ledger cannot be read the run tag is unknown,
        and an immature snapshot is then re-read from the DB every time rather
        than risk serving last month's rows.
        """
        extra = f"pred_{max_loans or 'auto'}"
        key   = _cache_key(extra, grain, feature_order)
        cache_dir = _cache_dir("pred", key)
        npz = _snapshot_npz(cache_dir, snapshot_date)

        conn, close_conn = self._get_conn()
        try:
            if use_cache and _cache_is_valid(npz, key) and self._pred_cache_is_fresh(npz, conn):
                return self._load_cache(npz)

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
            if use_cache and instances:
                mature = int(snapshot_date) in set(
                    filter_mature_snapshots([int(snapshot_date)])
                )
                self._save_cache(
                    npz, instances, feature_cols, max_loans or 99, key,
                    meta={"snapshot": int(snapshot_date), "mature": mature,
                          "etl_run": self._etl_run_tag(conn)},
                )
            return instances, feature_cols
        finally:
            if close_conn:
                conn.close()

    @staticmethod
    def _etl_run_tag(conn) -> Optional[str]:
        """
        The newest SUCCESSful upstream load, identifying the current ETL cycle;
        None when the ledger is unreadable (no grant, table absent).
        """
        try:
            runs = conn.get_etl_runs()
        except Exception:
            return None
        ok = runs[runs["status"].astype(str).str.upper() == "SUCCESS"]
        return None if ok.empty else str(ok.iloc[0]["snapshot_date"])

    @classmethod
    def _pred_cache_is_fresh(cls, npz: Path, conn) -> bool:
        """A mature snapshot's cache is final; an immature one is only good for
        the ETL cycle that built it."""
        try:
            with open(_manifest_path(npz)) as f:
                manifest = json.load(f)
        except Exception:
            return False
        if manifest.get("mature"):
            return True
        built = manifest.get("etl_run")
        current = cls._etl_run_tag(conn)
        if built is None or current is None or built != current:
            logger.info(
                f"{npz.name}: immature snapshot cached under ETL run {built!r}, "
                f"current is {current!r} — re-reading from the DB."
            )
            return False
        return True

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
                "data/snapshots/<stage>_<key>/ here."
            )
        return MSSQLConnector(), True


# ── Raw-array access for the diagnostic scripts ───────────────────────────────

def train_cache_dir(max_loans: Optional[int] = None, grain: Optional[str] = None,
                    feature_order: Optional[list[str]] = None) -> Path:
    """The directory load_train_portfolios caches into, for the given schema."""
    return _cache_dir("train", _cache_key(str(max_loans or "auto"), grain, feature_order))


def load_cached_arrays(cache_dir: Optional[Path] = None) -> tuple[dict, list[str]]:
    """
    The train cache as flat arrays — features_flat, offsets, labels,
    current_cats, n_loans, national_codes, snapshot_dates, loan_ids,
    portfolio_n_loans — concatenated across the per-snapshot NPZ files in
    snapshot order, with `offsets` rebased so it indexes the joined block.

    This is what the standalone diagnostics (explore_iv_woe, explore_umap,
    explore_clip_impact) want: the arrays, not tens of millions of instance
    dicts rebuilt from them. Label horizons are registered as a side effect,
    so filter_mature_snapshots works afterwards without a DB.
    """
    cache_dir = cache_dir or train_cache_dir()
    parts = sorted(
        (p for p in cache_dir.glob("*.npz") if p.stem.isdigit()),
        key=lambda p: int(p.stem),
    ) if cache_dir.is_dir() else []
    if not parts:
        raise FileNotFoundError(
            f"No cached snapshots in {cache_dir} — run `python run.py train` first."
        )

    stacked: dict[str, list] = {}
    feature_cols: list[str] = []
    row_base = 0
    for npz_path in parts:
        with np.load(npz_path, allow_pickle=True) as npz:
            for name in npz.files:
                arr = npz[name]
                if name == "offsets":
                    # Each file's offsets start at 0; drop that and shift.
                    arr = arr[1:] + row_base
                    row_base = int(arr[-1]) if len(arr) else row_base
                stacked.setdefault(name, []).append(arr)
        with open(_manifest_path(npz_path)) as f:
            manifest = json.load(f)
        feature_cols = manifest["feature_cols"]
        register_label_horizons(manifest.get("label_horizons", {}))

    arrays = {k: np.concatenate(v) for k, v in stacked.items()}
    arrays["offsets"] = np.concatenate([[0], arrays["offsets"]])
    logger.info(
        f"Loaded {len(arrays['labels']):,} instances / "
        f"{arrays['features_flat'].shape[0]:,} loans from {len(parts)} snapshot(s)."
    )
    return arrays, feature_cols
