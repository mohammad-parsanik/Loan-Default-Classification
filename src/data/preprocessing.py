"""
Vectorised preprocessing pipeline.

Performance vs. original:
  - All three transformers now operate on a flat (N_total_loans, N_features) matrix
    rather than looping over individual customer instances.
  - fit()   : vstack once → compute stats
  - transform() : vstack → transform matrix in one pass → np.split back
  - Eliminates 3 redundant np.copy() passes and millions of Python iterations.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


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

        self.is_fitted_ = True
        return self

    def transform(self, X: list[np.ndarray], y=None) -> list[np.ndarray]:
        flat, sizes = _vstack_split(X)      # single allocation

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
    """

    def __init__(self, feature_names: list[str], binary_features: list[str]):
        self.feature_names  = feature_names
        self.binary_features = set(binary_features)

    def fit(self, X: list[np.ndarray], y=None):
        flat, _ = _vstack_split(X)

        self.bounds_: dict[int, tuple[float, float]] = {}

        for i, col in enumerate(self.feature_names):
            if col in self.binary_features:
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

        for i, (lo, hi) in self.bounds_.items():
            np.clip(flat[:, i], lo, hi, out=flat[:, i])

        return _split_back(flat, sizes)


# ── Step 3: RobustScaler (non-binary only) ────────────────────────────────────

class PortfolioRobustScaler(BaseEstimator, TransformerMixin):
    """
    Applies sklearn RobustScaler to all non-binary columns.
    Binary features pass through unchanged.
    """

    def __init__(self, feature_names: list[str], binary_features: list[str]):
        self.feature_names   = feature_names
        self.binary_features = set(binary_features)

        self.scale_indices_: list[int] = [
            i for i, col in enumerate(feature_names) if col not in self.binary_features
        ]

    def fit(self, X: list[np.ndarray], y=None):
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
    """
    return Pipeline(
        [
            ("imputer", DomainAwareImputer(feature_names)),
            ("clipper", OutlierClipper(feature_names, binary_features)),
            ("scaler",  PortfolioRobustScaler(feature_names, binary_features)),
        ]
    )
