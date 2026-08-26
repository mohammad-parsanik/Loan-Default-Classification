"""
Vectorised preprocessing pipeline.

Performance vs. original:
  - All three transformers now operate on a flat (N_total_loans, N_features) matrix
    rather than looping over individual customer instances.
  - fit()   : vstack once → compute stats
  - transform() : vstack → transform matrix in one pass → np.split back
  - Eliminates 3 redundant np.copy() passes and millions of Python iterations.

Every transformer keys on the INTEGER POSITION of a column (fill_values_,
bounds_, scale_indices_), because it receives bare arrays, not DataFrames.
That is only safe while the caller projects columns by name first
(DataLoader.project_features). To make the assumption checkable rather than
assumed, each transformer records the feature list it was fitted against as
`feature_names_in_`, and `assert_pipeline_features` compares a caller's list
against it — see the module docstring of src/data/column_contract.py.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data.column_contract import NO_CLIP, NO_SCALE

logger = logging.getLogger(__name__)


class FeatureContractError(ValueError):
    """Raised when a fitted transformer is handed a different feature list."""


def _check_width(flat: np.ndarray, names: Optional[list], who: str) -> None:
    """Cheapest possible positional guard: the matrix must be the fitted width."""
    if names is not None and flat.shape[1] != len(names):
        raise FeatureContractError(
            f"{who} was fitted on {len(names)} features but received "
            f"{flat.shape[1]}. Every statistic this transformer holds is keyed "
            f"by column POSITION, so a width change silently applies the wrong "
            f"median/bounds/scale to every column. Project the frame to the "
            f"model's own feature list before transforming."
        )


def assert_pipeline_features(pipeline, feature_names: list[str]) -> None:
    """
    Verify a fitted preprocessing pipeline against the feature list the caller
    is about to feed it. Raises FeatureContractError on any difference —
    including a pure REORDER, which is the dangerous case: same width, no
    error, every column silently transformed with its neighbour's statistics.

    Steps fitted before `feature_names_in_` existed (older scaler.pkl files)
    cannot be verified; those warn instead, since the artifact's own
    metadata.json feature list is checked separately at load time.
    """
    steps = getattr(pipeline, "steps", None) or [("pipeline", pipeline)]
    for name, step in steps:
        fitted = getattr(step, "feature_names_in_", None)
        if fitted is None:
            logger.warning(
                f"Preprocessing step '{name}' predates feature-name checking "
                "(no feature_names_in_) — cannot verify column identity."
            )
            continue
        if list(fitted) != list(feature_names):
            extra   = [c for c in feature_names if c not in set(fitted)]
            missing = [c for c in fitted if c not in set(feature_names)]
            detail  = (f"missing={missing} extra={extra}" if (missing or extra)
                       else "same columns, DIFFERENT ORDER")
            raise FeatureContractError(
                f"Preprocessing step '{name}' was fitted on a different feature "
                f"list than the one supplied ({detail}). This transformer keys "
                f"every statistic by column position, so proceeding would apply "
                f"each column's parameters to a different column at identical "
                f"width and with no error."
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vstack_split(X: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack list of 2-D arrays → flat matrix; also return sizes for re-splitting."""
    sizes = np.array([arr.shape[0] for arr in X], dtype=np.int32)
    flat = np.vstack(X)
    return flat, sizes


def _split_back(flat: np.ndarray, sizes: np.ndarray) -> list[np.ndarray]:
    """Reverse of _vstack_split — split flat matrix into per-instance arrays."""
    split_indices = np.cumsum(sizes[:-1])
    return np.split(flat, split_indices)


# ── Step 1: Domain-aware imputer ──────────────────────────────────────────────

class DomainAwareImputer(BaseEstimator, TransformerMixin):
    """
    Imputes NaN values by feature group:
      - DAYS_SINCE_*   → max value seen in training + 1  (event never happened)
        NOTE: unreachable on the current feed, which COALESCEs those columns
        to their own "never" sentinel upstream so NaN never arrives. Same
        intent, reached independently. Kept as a fallback; do not rely on it.
      - *AMNT* / *RATIO* → median (fitted on training flat matrix)
      - Everything else → 0  (trajectory lags, binary flags, counts)
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def fit(self, X: list[np.ndarray], y=None):
        flat, _ = _vstack_split(X)          # (N_total, F)

        self.fill_values_: dict[int, float] = {}

        for i, col in enumerate(self.feature_names):
            col_data = flat[:, i]
            valid    = col_data[~np.isnan(col_data)]

            if col.startswith("DAYS_SINCE"):
                self.fill_values_[i] = float(valid.max() + 1) if len(valid) else 9999.0
            elif "AMNT" in col or "RATIO" in col:
                self.fill_values_[i] = float(np.median(valid)) if len(valid) else 0.0
            else:
                self.fill_values_[i] = 0.0

        self.feature_names_in_ = list(self.feature_names)
        self.is_fitted_ = True
        return self

    def transform(self, X: list[np.ndarray], y=None) -> list[np.ndarray]:
        flat, sizes = _vstack_split(X)      # single allocation
        _check_width(flat, getattr(self, "feature_names_in_", None), "DomainAwareImputer")

        for i, fill in self.fill_values_.items():
            nan_mask = np.isnan(flat[:, i])
            if nan_mask.any():
                flat[nan_mask, i] = fill

        return _split_back(flat, sizes)


# ── Step 2: Outlier clipper ───────────────────────────────────────────────────

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips continuous features at [1st, 99th] percentile.
    Ratios (*RATIO*) are clipped to [0, 1].
    Binary features are skipped.

    So are the contract's `clip: false` columns (NO_CLIP). Those carry a
    sentinel that is a CODE rather than a quantity — clipping it to p99 turns
    one risk state into a different one. The damage is invisible in a column
    where the sentinel is the majority value (p99 IS the sentinel, so the clip
    is a no-op) and destructive in its sibling where it is a minority, so a
    spot-check of one column tells you nothing about the other.
    """

    def __init__(self, feature_names: list[str], binary_features: list[str],
                 no_clip: Optional[set] = None):
        self.feature_names  = feature_names
        self.binary_features = set(binary_features)
        self.no_clip = set(NO_CLIP if no_clip is None else no_clip)

    def fit(self, X: list[np.ndarray], y=None):
        flat, _ = _vstack_split(X)

        self.bounds_: dict[int, tuple[float, float]] = {}
        self.feature_names_in_ = list(self.feature_names)

        for i, col in enumerate(self.feature_names):
            if col in self.binary_features or col in self.no_clip:
                continue
            col_data = flat[:, i]
            if "RATIO" in col:
                self.bounds_[i] = (0.0, 1.0)
            else:
                p1  = float(np.percentile(col_data, 1))
                p99 = float(np.percentile(col_data, 99))
                self.bounds_[i] = (p1, p99)

        self.is_fitted_ = True
        return self

    def transform(self, X: list[np.ndarray], y=None) -> list[np.ndarray]:
        flat, sizes = _vstack_split(X)
        _check_width(flat, getattr(self, "feature_names_in_", None), "OutlierClipper")

        for i, (lo, hi) in self.bounds_.items():
            np.clip(flat[:, i], lo, hi, out=flat[:, i])

        return _split_back(flat, sizes)


# ── Step 3: RobustScaler (non-binary only) ────────────────────────────────────

class PortfolioRobustScaler(BaseEstimator, TransformerMixin):
    """
    Applies sklearn RobustScaler to all non-binary columns.
    Binary features pass through unchanged, and so do the contract's
    `scale: false` columns (NO_SCALE) — see OutlierClipper. XGBoost splits on
    raw values, so leaving a sentinel-bearing column unscaled costs nothing
    and keeps "never reached this band" at the far end of its own axis.
    """

    def __init__(self, feature_names: list[str], binary_features: list[str],
                 no_scale: Optional[set] = None):
        self.feature_names   = feature_names
        self.binary_features = set(binary_features)
        self.no_scale = set(NO_SCALE if no_scale is None else no_scale)

        skip = self.binary_features | self.no_scale
        self.scale_indices_: list[int] = [
            i for i, col in enumerate(feature_names) if col not in skip
        ]

    def fit(self, X: list[np.ndarray], y=None):
        self.feature_names_in_ = list(self.feature_names)
        self.is_fitted_ = True
        if not self.scale_indices_:
            return self

        flat, _ = _vstack_split(X)
        self.scaler_ = RobustScaler()
        self.scaler_.fit(flat[:, self.scale_indices_])
        return self

    def transform(self, X: list[np.ndarray], y=None) -> list[np.ndarray]:
        if not self.scale_indices_:
            return X

        flat, sizes = _vstack_split(X)
        _check_width(flat, getattr(self, "feature_names_in_", None), "PortfolioRobustScaler")
        flat[:, self.scale_indices_] = self.scaler_.transform(flat[:, self.scale_indices_])
        return _split_back(flat, sizes)


# ── Factory ───────────────────────────────────────────────────────────────────

def create_preprocessing_pipeline(
    feature_names: list[str],
    binary_features: list[str],
) -> Pipeline:
    """
    Creates the full sklearn Pipeline for preprocessing portfolio instances.
    Accepts and returns list[np.ndarray] (one per customer portfolio).

    `feature_names` must be the SAME list, in the same order, that the caller
    projects its frames to; it is recorded on every step at fit time and
    checked by assert_pipeline_features at scoring time.
    """
    return Pipeline(
        [
            ("imputer", DomainAwareImputer(feature_names)),
            ("clipper", OutlierClipper(feature_names, binary_features)),
            ("scaler",  PortfolioRobustScaler(feature_names, binary_features)),
        ]
    )
