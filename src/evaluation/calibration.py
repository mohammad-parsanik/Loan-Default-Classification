"""
Per-class isotonic probability calibration.

Both the cost-sensitive focal loss (DeepSets) and inverse-frequency sample
weights (XGBoost) intentionally distort predicted probabilities.  The
expected-cost decision rule and any threshold in the downstream rule system
need honest frequencies, so a monotonic per-class correction is fitted on
held-out data (the customer-disjoint validation set) and applied before
any decision/ranking.  Monotonic ⇒ per-class ranking is preserved.
"""

import logging

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class PerClassIsotonicCalibrator:
    """One isotonic regressor per class (one-vs-rest), then row renormalise."""

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PerClassIsotonicCalibrator":
        probs = np.asarray(probs, dtype=np.float64)
        y_true = np.asarray(y_true)
        self.n_classes_ = probs.shape[1]
        self.calibrators_ = []
        for c in range(self.n_classes_):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(probs[:, c], (y_true == c).astype(np.float64))
            self.calibrators_.append(iso)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        out = np.column_stack(
            [self.calibrators_[c].predict(probs[:, c]) for c in range(self.n_classes_)]
        )
        # Guard against all-zero rows, then renormalise to a distribution
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)


class StratifiedCalibrator:
    """
    One PerClassIsotonicCalibrator per stratum (customer's current category).

    Why: P(severe) is a rare event for current-cat-0 customers but a common
    one for current-cat-2. A pooled isotonic fit miscalibrates the strata
    against each other — and the ranked API queue mixes strata, so
    cross-stratum comparability of P(severe) is exactly what the ranking
    depends on. Strata with fewer than `min_stratum_n` fitting samples fall
    back to a pooled calibrator.
    """

    def __init__(self, min_stratum_n: int = 5000):
        self.min_stratum_n = min_stratum_n

    def fit(self, probs: np.ndarray, y_true: np.ndarray,
            strata: np.ndarray) -> "StratifiedCalibrator":
        probs  = np.asarray(probs, dtype=np.float64)
        y_true = np.asarray(y_true)
        strata = np.asarray(strata)

        self.pooled_ = PerClassIsotonicCalibrator().fit(probs, y_true)
        self.per_stratum_ = {}
        for s in np.unique(strata):
            mask = strata == s
            if mask.sum() >= self.min_stratum_n:
                self.per_stratum_[int(s)] = PerClassIsotonicCalibrator().fit(
                    probs[mask], y_true[mask]
                )
            else:
                logger.info(
                    f"Calibration stratum {s}: only {int(mask.sum()):,} samples "
                    f"(< {self.min_stratum_n:,}) — using pooled calibrator."
                )
        return self

    def transform(self, probs: np.ndarray, strata: np.ndarray) -> np.ndarray:
        probs  = np.asarray(probs, dtype=np.float64)
        strata = np.asarray(strata)
        out = self.pooled_.transform(probs)
        for s, cal in self.per_stratum_.items():
            mask = strata == s
            if mask.any():
                out[mask] = cal.transform(probs[mask])
        return out


if __name__ == "__main__":
    # Self-check: overconfident probs get pulled toward observed frequencies.
    rng = np.random.default_rng(0)
    n = 20_000
    n_classes = 4
    y = rng.integers(0, n_classes, n)
    probs = np.full((n, n_classes), 0.2 / (n_classes - 1))
    probs[np.arange(n), y] = 0.8                       # model "knows" the class…
    flip = rng.random(n) < 0.3                          # …but is wrong 30% of the time
    y_obs = np.where(flip, (y + 1) % n_classes, y)

    cal = PerClassIsotonicCalibrator().fit(probs, y_obs)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0)
    top = out[np.arange(n), probs.argmax(axis=1)]
    assert abs(top.mean() - 0.7) < 0.05, top.mean()     # ≈ true 70% accuracy

    # Stratified: stratum 0 is wrong 50% of the time, stratum 1 only 10%.
    # A pooled calibrator averages them; the stratified one separates them.
    strata = (rng.random(n) < 0.5).astype(int)
    flip2 = rng.random(n) < np.where(strata == 0, 0.5, 0.1)
    y_obs2 = np.where(flip2, (y + 1) % n_classes, y)
    scal = StratifiedCalibrator(min_stratum_n=1000).fit(probs, y_obs2, strata)
    out2 = scal.transform(probs, strata)
    top2 = out2[np.arange(n), probs.argmax(axis=1)]
    assert abs(top2[strata == 0].mean() - 0.5) < 0.05
    assert abs(top2[strata == 1].mean() - 0.9) < 0.05
    assert np.allclose(out2.sum(axis=1), 1.0)
    print("calibration.py self-check OK")
