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

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Computes all required metrics for the Loan Default Classification project.
    """
    metrics = {}
    
    # Standard metrics
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    metrics['qwk'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Per-class recall
    recalls = recall_score(y_true, y_pred, average=None)
    for i, r in enumerate(recalls):
        metrics[f'recall_class_{i}'] = float(r)
        
    # Brier Score (if probabilities are provided)
    if y_prob is not None:
        y_true_oh = np.zeros((len(y_true), 3))
        y_true_oh[np.arange(len(y_true)), y_true] = 1
        brier = np.mean([brier_score_loss(y_true_oh[:, c], y_prob[:, c]) for c in range(3)])
        metrics['brier_score'] = float(brier)
        
    # Cost-weighted accuracy
    cost_matrix = np.array([
        [0.0, 0.5, 1.0],
        [1.5, 0.0, 0.5],
        [4.0, 2.0, 0.0]
    ])
    
    costs = np.array([cost_matrix[t, p] for t, p in zip(y_true, y_pred)])
    metrics['avg_cost'] = float(np.mean(costs))
    
    return metrics

def bootstrap_confidence_intervals(y_true, y_pred, y_prob=None, n_iterations=1000, alpha=0.05):
    """Computes 95% CI for metrics using bootstrapping on the test set."""
    n_size = len(y_true)
    stats = {
        'macro_f1': [],
        'qwk': [],
        'recall_class_2': []
    }
    
    for _ in range(n_iterations):
        # Prepare indices for sampling
        indices = resample(np.arange(n_size))
        y_t = y_true[indices]
        y_p = y_pred[indices]
        
        # Calculate and store metrics for this sample
        stats['macro_f1'].append(f1_score(y_t, y_p, average='macro'))
        stats['qwk'].append(cohen_kappa_score(y_t, y_p, weights='quadratic'))
        
        recalls = recall_score(y_t, y_p, average=None, zero_division=0)
        if len(recalls) > 2:
            stats['recall_class_2'].append(recalls[2])
        else:
            # Handle edge case where class 2 is not in the bootstrap sample
            stats['recall_class_2'].append(0.0)
            
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
