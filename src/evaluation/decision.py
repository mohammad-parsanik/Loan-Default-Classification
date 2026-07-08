"""
Expected-cost decision rule (Bayes decision under the business cost matrix).

Given class probabilities P and cost matrix C (C[true][pred]), the expected
cost of predicting class j is sum_i P_i * C[i][j].  Predicting the argmin
column is optimal when probabilities are calibrated — this replaces plain
argmax everywhere a decision or a ranking is needed.
"""

import numpy as np

import project_config as config

COST_MATRIX = np.asarray(config.COST_MATRIX, dtype=np.float64)


def expected_costs(probs: np.ndarray) -> np.ndarray:
    """(N, n_classes) probabilities → (N, n_classes) expected cost per action."""
    return np.asarray(probs) @ COST_MATRIX


def cost_decisions(probs: np.ndarray) -> np.ndarray:
    """Minimum-expected-cost class prediction. (N,) int array."""
    return expected_costs(probs).argmin(axis=1)


def risk_scores(probs: np.ndarray) -> np.ndarray:
    """
    Expected cost of *doing nothing* (predicting class 0) for each customer.
    Kept as a secondary/diagnostic score — the ranked API queue uses
    severity_scores() instead, because the cost matrix values are guessed
    while P(severe) uses only the observed label.
    """
    return expected_costs(probs)[:, 0]


def severity_scores(probs: np.ndarray) -> np.ndarray:
    """
    THE ranking score for the API queue: calibrated P(entering the severe
    class within the horizon). Cost-matrix-free by design.
    """
    return np.asarray(probs)[:, config.NUM_CLASSES - 1]


def mask_monotone(probs: np.ndarray, current_cat: np.ndarray) -> np.ndarray:
    """
    Enforce the label's monotonicity: WORST_FUTURE_CAT includes the current
    month, so label >= current_cat always — P(label < current_cat) is
    logically impossible. Zero that mass and renormalise rows. Sharpens
    P(severe) for already-delinquent customers and guarantees coherent output.
    """
    probs = np.asarray(probs, dtype=np.float64).copy()
    current_cat = np.asarray(current_cat)
    class_idx = np.arange(probs.shape[1])
    probs[class_idx[None, :] < current_cat[:, None]] = 0.0
    # Degenerate guard: if everything was masked away (can't happen with
    # capped current_cat, but cheap), fall back to the top allowed class.
    row_sums = probs.sum(axis=1, keepdims=True)
    dead = row_sums[:, 0] == 0.0
    if dead.any():
        probs[dead, -1] = 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
    return probs / row_sums


if __name__ == "__main__":
    # Self-check: cost rule flags a risky-but-not-modal customer; argmax doesn't.
    p = np.array([[0.45, 0.20, 0.30, 0.05], [0.90, 0.06, 0.03, 0.01]])
    assert p.argmax(axis=1).tolist() == [0, 0]
    assert cost_decisions(p).tolist() == [2, 0]
    assert np.isclose(risk_scores(p)[0], 0.20 * 1.5 + 0.30 * 4.0 + 0.05 * 7.5)
    assert np.allclose(severity_scores(p), [0.05, 0.01])

    # Masking: a cat-2 customer cannot end below 2.
    m = mask_monotone(p, np.array([2, 0]))
    assert m[0, 0] == 0.0 and m[0, 1] == 0.0
    assert np.isclose(m[0, 2], 0.30 / 0.35) and np.isclose(m[0, 3], 0.05 / 0.35)
    assert np.allclose(m[1], p[1])          # cat-0 row untouched
    assert np.allclose(m.sum(axis=1), 1.0)
    print("decision.py self-check OK")
