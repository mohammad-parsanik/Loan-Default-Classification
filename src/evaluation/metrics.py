import numpy as np
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    brier_score_loss,
    recall_score,
    accuracy_score,
    confusion_matrix
)
from sklearn.utils import resample

import project_config as config

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes all required metrics for the Loan Default Classification project.
    """
    metrics = {}
    labels = list(range(config.NUM_CLASSES))

    # Standard metrics. `labels=` matters on the current-cat slices: the
    # current_cat_3 stratum is a single class by the `label >= current_cat`
    # identity, and without it sklearn scores macro-F1 over the ONE observed
    # class (a free 1.0000, not comparable to the other slices) and returns a
    # NaN QWK with two warnings per arm.
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro',
                                         labels=labels, zero_division=0))
    # Kappa is undefined when nothing varies — chance agreement is already 1.
    # Say so directly instead of letting a 0/0 inside sklearn say it.
    metrics['qwk'] = (
        float('nan') if len(np.unique(np.concatenate([y_true, y_pred]))) < 2
        else float(cohen_kappa_score(y_true, y_pred, weights='quadratic',
                                     labels=labels))
    )
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Per-class recall (explicit labels: keeps indices right on strata where
    # a class is absent, e.g. current-cat slices)
    recalls = recall_score(y_true, y_pred, average=None,
                           labels=list(range(config.NUM_CLASSES)), zero_division=0)
    for i, r in enumerate(recalls):
        metrics[f'recall_class_{i}'] = float(r)
        
    # Brier Score (if probabilities are provided)
    if y_prob is not None:
        y_true_oh = np.zeros((len(y_true), config.NUM_CLASSES))
        y_true_oh[np.arange(len(y_true)), y_true] = 1
        brier = np.mean([brier_score_loss(y_true_oh[:, c], y_prob[:, c]) for c in range(config.NUM_CLASSES)])
        metrics['brier_score'] = float(brier)
        
    # Cost-weighted accuracy (single source of truth: project_config.COST_MATRIX)
    cost_matrix = np.asarray(config.COST_MATRIX)
    metrics['avg_cost'] = float(cost_matrix[np.asarray(y_true), np.asarray(y_pred)].mean())

    return metrics


def exposure_decile_ranking(y_true, scores, strata, exposure, tie_break=None,
                            n_bins: int = 10) -> dict:
    """
    The ranking block, cut by exposure (REMAINING_AMNT) decile.

    Answers whether the queue ranks equally well at every loan size, which is
    the precondition for two things the project keeps being asked about:

      * **Banding.** A separate model per amount band is only worth building if
        the feature->target relationship differs across bands. A different base
        rate is not that — one tree split handles it, at a threshold the data
        picks. Flat lift@K across deciles means banding buys nothing.
      * **Deletion bias (§24).** The upstream table is hard-deleted for
        post-NPL loans. If that deletion rate varies with loan size, the
        model learns a size-dependent distortion as if it were signal, and
        "large loans are safer" (clip_impact tail_lift 0.26-0.59) is an
        artifact rather than a fact.

    Deciles are cut on the RANKED population only — carved rows never reach the
    queue, so including them would move the boundaries for nothing. Each
    decile's block is a full `ranking_metrics` call, so `pr_auc` and `lift` are
    comparable across deciles; the raw `recall` is not, because K is the whole
    API budget spent inside one decile. Read `pr_auc` and `lift`.
    """
    from src.evaluation.ranking import ranking_metrics

    y_true, scores = np.asarray(y_true), np.asarray(scores)
    strata, exposure = np.asarray(strata), np.asarray(exposure, dtype=np.float64)
    idx = np.flatnonzero(strata < config.CARVE_CURRENT_CAT_GE)
    if len(idx) < n_bins:
        return {}

    # Rank-based cut, so a heavily tied or skewed amount distribution still
    # yields balanced bins. Ties land in the same bin by construction.
    order = np.argsort(exposure[idx], kind="stable")
    bin_of = np.empty(len(idx), dtype=np.int32)
    bin_of[order] = (np.arange(len(idx)) * n_bins) // len(idx)

    out = {}
    for b in range(n_bins):
        sel = idx[bin_of == b]
        if not len(sel):
            continue
        block = ranking_metrics(
            y_true[sel], scores[sel], strata=strata[sel],
            tie_break=None if tie_break is None else np.asarray(tie_break)[sel],
        )
        block["exposure_min"] = float(exposure[sel].min())
        block["exposure_max"] = float(exposure[sel].max())
        out[f"decile_{b + 1}"] = block
    return out


def full_evaluation(y_true, probs, strata=None, calibrator=None,
                    portfolio_sizes=None, tie_break=None, exposure=None) -> dict:
    """
    Single evaluation path shared by the baseline and the DeepSets+XGB model,
    so the two are compared under the SAME decision policy.

    Pipeline: raw probs → calibrate (stratified if the calibrator supports
    it) → monotone mask (label >= current_cat) → score.

    Top-level metrics: raw-prob argmax (backwards-compatible with earlier runs).
    "ranking":         THE headline — recall/lift at API-budget reference
                       windows on P(severe), carved population only.
    "argmax_cal":      argmax on calibrated+masked probs (separates the
                       effect of calibration from the cost rule).
    "cost_rule":       expected-cost decisions on calibrated+masked probs.
    "by_current_cat":  cost-rule metrics per current-category slice.
    "ranking_by_exposure": the ranking block per REMAINING_AMNT decile (needs
                       `exposure`). Flat lift@K across deciles means an
                       amount-banded model would buy nothing; a dip says
                       otherwise. See exposure_decile_ranking.
    "ranking_single_loan": the ranking block restricted to customers holding
                       exactly ONE loan (needs `portfolio_sizes`). For those
                       customers a loan-grain row and a portfolio-grain row
                       are identical by construction, so this slice — and
                       only this slice — is comparable across the two grains
                       and back to the portfolio-grain benchmark
                       (results_3..results_6). The headline "ranking" block
                       is NOT: loan grain changes the population (healthy
                       siblings of severe loans now enter the queue) and the
                       severe base rate with it, and PR-AUC moves with the
                       base rate whatever the model does.
    `tie_break` (e.g. per-instance LOAN_ID) decides the order of rows on
    identical scores in every ranking block. Calibrated probabilities tie in
    large blocks, so without it recall@K is a function of input order — see
    src/evaluation/ranking.py.

    Also returns "_cost_preds"/"_probs_cal" for plots/CIs.
    """
    from src.evaluation.calibration import StratifiedCalibrator
    from src.evaluation.decision import cost_decisions, mask_monotone, severity_scores
    from src.evaluation.ranking import ranking_metrics

    y_true = np.asarray(y_true)

    if calibrator is None:
        probs_cal = np.asarray(probs, dtype=np.float64)
    elif isinstance(calibrator, StratifiedCalibrator):
        probs_cal = calibrator.transform(probs, strata)
    else:
        probs_cal = calibrator.transform(probs)

    if strata is not None:
        probs_cal = mask_monotone(probs_cal, strata)

    cost_preds = cost_decisions(probs_cal)

    out = compute_metrics(y_true, probs.argmax(axis=1), probs)
    out["argmax_cal"] = compute_metrics(y_true, probs_cal.argmax(axis=1), probs_cal)
    out["cost_rule"] = compute_metrics(y_true, cost_preds, probs_cal)
    if strata is not None:
        sev = severity_scores(probs_cal)
        out["ranking"] = ranking_metrics(y_true, sev, strata, tie_break=tie_break)
        out["by_current_cat"] = stratified_metrics(y_true, cost_preds, strata, probs_cal)
        if portfolio_sizes is not None:
            one = np.asarray(portfolio_sizes) == 1
            out["ranking_single_loan"] = ranking_metrics(
                y_true[one], sev[one], np.asarray(strata)[one],
                tie_break=None if tie_break is None else np.asarray(tie_break)[one],
            )
        if exposure is not None:
            out["ranking_by_exposure"] = exposure_decile_ranking(
                y_true, sev, strata, exposure, tie_break=tie_break)
    out["_cost_preds"] = cost_preds
    out["_probs_cal"] = probs_cal
    return out


def stratified_metrics(y_true, y_pred, strata, y_prob=None) -> dict:
    """
    Metrics per stratum (e.g. customer's CURRENT worst category).

    The aggregate score blends the easy segment (already-delinquent customers,
    whose future label is largely mechanical DPD accrual) with the hard one
    (currently-clean customers — the actual early-warning task). Reporting per
    current-category slices keeps those separate.

    Returns {"current_cat_0": {...}, "current_cat_1": {...}, ...} with an
    "n" count added per slice.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    strata = np.asarray(strata)

    out = {}
    for s in np.unique(strata):
        mask = strata == s
        m = compute_metrics(
            y_true[mask], y_pred[mask],
            y_prob[mask] if y_prob is not None else None,
        )
        m["n"] = int(mask.sum())
        out[f"current_cat_{int(s)}"] = m
    return out

def bootstrap_confidence_intervals(y_true, y_pred, y_prob=None, n_iterations=1000, alpha=0.05):
    """Computes 95% CI for metrics using bootstrapping on the test set."""
    n_size = len(y_true)
    stats = {
        'macro_f1': [],
        'qwk': [],
        'recall_class_2': [],
        'recall_class_3': [],
    }

    for _ in range(n_iterations):
        # Prepare indices for sampling
        indices = resample(np.arange(n_size))
        y_t = y_true[indices]
        y_p = y_pred[indices]

        # Calculate and store metrics for this sample
        stats['macro_f1'].append(f1_score(y_t, y_p, average='macro'))
        stats['qwk'].append(cohen_kappa_score(y_t, y_p, weights='quadratic'))

        # Explicit labels: keeps indices aligned when a class is absent
        # from the bootstrap sample
        recalls = recall_score(y_t, y_p, average=None,
                               labels=list(range(config.NUM_CLASSES)), zero_division=0)
        stats['recall_class_2'].append(recalls[2])
        stats['recall_class_3'].append(recalls[3])
            
    # Calculate confidence intervals
    ci = {}
    lower_p = (alpha / 2.0) * 100
    upper_p = (1.0 - (alpha / 2.0)) * 100
    
    for metric_name, values in stats.items():
        ci[metric_name] = {
            'mean': float(np.mean(values)),
            'lower_ci': float(np.percentile(values, lower_p)),
            'upper_ci': float(np.percentile(values, upper_p))
        }
        
    return ci
