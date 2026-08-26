"""
Ranking metrics for the API-budget deliverable.

The product is a ranked queue: customers not yet severe (current_cat <
CARVE_CURRENT_CAT_GE), ordered by calibrated P(entering the severe class
within the horizon), consumed at API_RATE_PER_HOUR. So the questions that
matter are "of everyone who actually went severe, how many sit within the
first H hours of calling?" (recall@K) and "how much better than calling at
random?" (lift@K).

These use only the observed label — no cost-matrix guesses, no unknowable
"was the API call right" ground truth.

Ties are resolved explicitly. Calibrated probabilities tie heavily (isotonic
regression is a step function), and a stable argsort resolves ties by INPUT
ORDER — so recall@K and lift@K would otherwise be functions of how the rows
happened to arrive. `tie_break` makes that choice a stated one; omitted, it
is the instance's position under the loader's canonical row order, which is
itself a function of the data. See _order().
"""

import numpy as np

import project_config as config

SEVERE_CLASS = config.NUM_CLASSES - 1


def _order(scores: np.ndarray, tie_break: np.ndarray = None) -> np.ndarray:
    """
    Indices sorting `scores` descending, ties broken by `tie_break` ascending
    (position when omitted). np.lexsort applies the LAST key first, so the
    score is passed last.
    """
    if tie_break is None:
        tie_break = np.arange(len(scores))
    return np.lexsort((tie_break, -scores))


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    strata: np.ndarray = None,
    tie_break: np.ndarray = None,
) -> dict:
    """
    Evaluate a severity ranking on the modelled (non-carved) population.

    Args:
        y_true : (N,) int class labels
        scores : (N,) ranking score, higher = riskier (calibrated P(severe))
        strata : (N,) current_cat per instance; when given, rows with
                 current_cat >= CARVE_CURRENT_CAT_GE are excluded (they are
                 rule-flagged, never ranked) and per-stratum breakdowns of
                 the reference-window recalls are added.
        tie_break : (N,) sortable key (e.g. LOAN_ID) deciding the order of
                 rows on identical scores. Defaults to input position.

    Returns dict:
        n_ranked, n_severe, base_rate, pr_auc,
        at_<window>: {k, recall, precision, lift}   per RANKING_REF_WINDOWS
        by_current_cat: {stratum: {n, n_severe, recall_at_<window>...}}
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    tie_break = np.arange(len(scores)) if tie_break is None else np.asarray(tie_break)

    if strata is not None:
        strata = np.asarray(strata)
        keep = strata < config.CARVE_CURRENT_CAT_GE
        y_true, scores, strata = y_true[keep], scores[keep], strata[keep]
        tie_break = tie_break[keep]

    y_bin = (y_true == SEVERE_CLASS).astype(np.int32)
    n = len(y_bin)
    n_severe = int(y_bin.sum())
    out = {
        "n_ranked": n,
        "n_severe": n_severe,
        "base_rate": float(n_severe / n) if n else float("nan"),
    }
    if n == 0 or n_severe == 0:
        return out

    order = _order(scores, tie_break)
    hits_cum = np.cumsum(y_bin[order])          # severe captured in top-i

    out["pr_auc"] = _average_precision(y_bin, scores)

    for name, hours in config.RANKING_REF_WINDOWS.items():
        k = min(int(config.API_RATE_PER_HOUR * hours), n)
        recall = float(hits_cum[k - 1] / n_severe)
        precision = float(hits_cum[k - 1] / k)
        out[f"at_{name}"] = {
            "k": k,
            "recall": recall,
            "precision": precision,
            "lift": float(precision / out["base_rate"]),
        }

    if strata is not None:
        by = {}
        for s in np.unique(strata):
            mask = strata == s
            sub = ranking_metrics(y_true[mask], scores[mask], strata=None,
                                  tie_break=tie_break[mask])
            by[f"current_cat_{int(s)}"] = sub
        out["by_current_cat"] = by

    return out


def capture_curve(y_true: np.ndarray, scores: np.ndarray, n_points: int = 200,
                  tie_break: np.ndarray = None) -> dict:
    """
    Cumulative-gains curve data for plotting / operating-point selection.
    Returns {"frac_called", "recall", "hours"} arrays (JSON-safe lists);
    hours = customers called / API_RATE_PER_HOUR.
    """
    y_bin = (np.asarray(y_true) == SEVERE_CLASS).astype(np.int32)
    order = _order(np.asarray(scores), tie_break)
    hits_cum = np.cumsum(y_bin[order])
    n, n_severe = len(y_bin), max(int(y_bin.sum()), 1)

    ks = np.unique(np.linspace(1, n, min(n_points, n)).astype(int))
    return {
        "frac_called": (ks / n).tolist(),
        "recall":      (hits_cum[ks - 1] / n_severe).tolist(),
        "hours":       (ks / config.API_RATE_PER_HOUR).tolist(),
    }


def _average_precision(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC (average precision), threshold-free summary of the ranking."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_bin, scores))


if __name__ == "__main__":
    # Self-check on a hand-built ranking: 10 customers, 3 severe, scores
    # place two severe in the top 3 and one at the bottom.
    y = np.array([3, 3, 0, 1, 2, 0, 0, 1, 0, 3])
    s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .15, .1])

    y_bin = (y == 3).astype(int)
    order = np.argsort(-s)
    assert np.cumsum(y_bin[order])[2] == 2          # top-3 holds 2 of 3 severe

    import project_config as _c
    _c.RANKING_REF_WINDOWS = {"test": 3 / _c.API_RATE_PER_HOUR}  # K=3
    m = ranking_metrics(y, s)
    assert m["n_severe"] == 3 and m["n_ranked"] == 10
    assert np.isclose(m["at_test"]["recall"], 2 / 3)
    assert np.isclose(m["at_test"]["precision"], 2 / 3)
    assert np.isclose(m["at_test"]["lift"], (2 / 3) / 0.3)

    # Carve-out: severe-current customers leave the ranked population.
    strata = np.array([3, 0, 0, 1, 2, 0, 0, 1, 0, 2])
    m2 = ranking_metrics(y, s, strata=strata)
    assert m2["n_ranked"] == 9 and m2["n_severe"] == 2

    cc = capture_curve(y, s)
    assert cc["recall"][-1] == 1.0

    # Ties are broken by the stated key, not by arrival order: two customers
    # on the same score, one severe, and the tie-break decides which is called
    # first — so recall@1 flips with it and nothing else does.
    y_t = np.array([0, 3])
    s_t = np.array([0.5, 0.5])
    _c.RANKING_REF_WINDOWS = {"one": 1 / _c.API_RATE_PER_HOUR}   # K=1
    assert ranking_metrics(y_t, s_t, tie_break=np.array([0, 1]))["at_one"]["recall"] == 0.0
    assert ranking_metrics(y_t, s_t, tie_break=np.array([1, 0]))["at_one"]["recall"] == 1.0
    print("ranking.py self-check OK")
