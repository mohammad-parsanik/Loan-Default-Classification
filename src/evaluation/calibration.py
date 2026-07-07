"""
Per-class isotonic probability calibration.

Both the cost-sensitive focal loss (DeepSets) and inverse-frequency sample
weights (XGBoost) intentionally distort predicted probabilities.  The
expected-cost decision rule and any threshold in the downstream rule system
need honest frequencies, so a monotonic per-class correction is fitted on
held-out data (the customer-disjoint validation set) and applied before
any decision/ranking.  Monotonic ⇒ per-class ranking is preserved.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


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


if __name__ == "__main__":
    # Self-check: overconfident probs get pulled toward observed frequencies.
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.integers(0, 3, n)
    probs = np.full((n, 3), 0.1)
    probs[np.arange(n), y] = 0.8                       # model "knows" the class…
    flip = rng.random(n) < 0.3                          # …but is wrong 30% of the time
    y_obs = np.where(flip, (y + 1) % 3, y)

    cal = PerClassIsotonicCalibrator().fit(probs, y_obs)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0)
    top = out[np.arange(n), probs.argmax(axis=1)]
    assert abs(top.mean() - 0.7) < 0.05, top.mean()     # ≈ true 70% accuracy
    print("calibration.py self-check OK")
