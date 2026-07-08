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
    Ranking score for the top-K API budget: the expected cost of *doing
    nothing* (predicting class 0) for each customer.  With the current
    matrix this is 1.5*P(cat1) + 4.0*P(cat2) — the principled replacement
    for the ad-hoc 2*P2 + P1.
    """
    return expected_costs(probs)[:, 0]


if __name__ == "__main__":
    # Self-check: cost rule flags a risky-but-not-modal customer; argmax doesn't.
    p = np.array([[0.45, 0.20, 0.30, 0.05], [0.90, 0.06, 0.03, 0.01]])
    assert p.argmax(axis=1).tolist() == [0, 0]
    assert cost_decisions(p).tolist() == [2, 0]
    assert np.isclose(risk_scores(p)[0], 0.20 * 1.5 + 0.30 * 4.0 + 0.05 * 7.5)
    print("decision.py self-check OK")
