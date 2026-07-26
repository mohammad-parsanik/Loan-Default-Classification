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
    
    # Standard metrics
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    metrics['qwk'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
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


def full_evaluation(y_true, probs, strata=None, calibrator=None,
                    portfolio_sizes=None) -> dict:
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
        out["ranking"] = ranking_metrics(y_true, sev, strata)
        out["by_current_cat"] = stratified_metrics(y_true, cost_preds, strata, probs_cal)
        if portfolio_sizes is not None:
            one = np.asarray(portfolio_sizes) == 1
            out["ranking_single_loan"] = ranking_metrics(
                y_true[one], sev[one], np.asarray(strata)[one]
            )
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
