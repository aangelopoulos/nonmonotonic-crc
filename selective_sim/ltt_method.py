"""
Learn Then Test (LTT) Method for Selective Classification

This implements the binomial-based RCPS method from:
Angelopoulos et al. (2021). "Learn then Test: Calibrating Predictive Algorithms 
to Achieve Risk Control" (https://arxiv.org/abs/2110.01052)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom


def apply_ltt_to_imagenet(cal_smx, cal_labels, val_smx, val_labels, 
                          alpha=0.1, delta=0.1, verbose=True):
    """
    Apply Learn Then Test method to ImageNet data.
    
    Parameters
    ----------
    cal_smx : np.ndarray, shape (n_cal, n_classes)
        Softmax outputs for calibration set
    cal_labels : np.ndarray, shape (n_cal,)
        True labels for calibration set
    val_smx : np.ndarray, shape (n_val, n_classes)
        Softmax outputs for validation set
    val_labels : np.ndarray, shape (n_val,)
        True labels for validation set
    alpha : float
        Target risk level (1-alpha is desired selective accuracy)
    delta : float
        Failure rate for high-probability bound
    verbose : bool
        Print results
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - theta_hat: chosen threshold
        - predictions_kept: boolean array for validation set
        - empirical_selective_accuracy: accuracy on selected examples
        - prediction_rate: fraction of examples selected
        - selective_error: error rate on selected examples
    """
    # Get predictions and confidence scores
    cal_yhats = np.argmax(cal_smx, axis=1)
    cal_phats = cal_smx.max(axis=1)
    
    val_yhats = np.argmax(val_smx, axis=1)
    val_phats = val_smx.max(axis=1)
    
    # Create lambda grid
    lambdas = np.linspace(0, 1, 5000)
    
    # Define selective risk
    def selective_risk(lam): 
        selected = cal_phats >= lam
        if selected.sum() == 0:
            return 0
        return (cal_yhats[selected] != cal_labels[selected]).sum() / selected.sum()
    
    def nlambda(lam): 
        return (cal_phats > lam).sum()
        
    def invert_for_ub(r, lam): 
        return binom.cdf(selective_risk(lam) * nlambda(lam), nlambda(lam), r) - delta
    
    # Construct upper bound
    def selective_risk_ub(lam): 
        return brentq(invert_for_ub, 0, 0.9999, args=(lam,))
    
    # Scan to choose lambda hat
    lhat = 1.0
    for lam in np.flip(lambdas):
        if selective_risk_ub(lam - 1/lambdas.shape[0]) > alpha: 
            lhat = lam
            break
    
    # Apply to validation set
    predictions_kept = val_phats >= lhat
    
    # Compute metrics
    if predictions_kept.sum() > 0:
        empirical_selective_accuracy = (val_yhats[predictions_kept] == val_labels[predictions_kept]).mean()
        selective_error = 1 - empirical_selective_accuracy
        prediction_rate = predictions_kept.mean()
    else:
        empirical_selective_accuracy = 0.0
        selective_error = 1.0
        prediction_rate = 0.0
    
    if verbose:
        print(f"Threshold λ̂: {lhat:.4f}")
        print(f"Selective accuracy: {empirical_selective_accuracy:.4f}")
        print(f"Selective error rate: {selective_error:.4f}")
        print(f"Prediction rate: {prediction_rate:.4f}")
    
    results = {
        'theta_hat': lhat,
        'predictions_kept': predictions_kept,
        'empirical_selective_accuracy': empirical_selective_accuracy,
        'selective_error': selective_error,
        'prediction_rate': prediction_rate
    }
    
    return results

