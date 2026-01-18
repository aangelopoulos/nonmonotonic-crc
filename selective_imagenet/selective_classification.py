"""
Selective Classification for ImageNet Experiments

This module provides the selective classification algorithm for use in ImageNet experiments.
It imports from the shared selective module.
"""

import os
import sys
import numpy as np

# Add parent directory to path to import shared module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from selective import (
    LTTSelectiveClassifier,
    SelectiveClassifier, 
    SelectiveClassifierStabilityEstimator
)

__all__ = [
    'SelectiveClassifier', 
    'SelectiveClassifierStabilityEstimator'
]


def apply_selective_classification_to_imagenet(cal_smx, cal_labels, val_smx, val_labels, 
                                               alpha=0.1, LTT=False, stability_estimator=None,
                                               delta=0.1, min_n=50, grid_size=5000, verbose=True):
    """
    Apply selective classification to ImageNet data.
    
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
    LTT : bool
        True if we use LTT
    stability_estomator : ['K', 'df', 'definition', float, None]
        Picks the stability estimator. Definition is the most accurate, 
        K is an efficient upper-bound for selective classification, 
        df is a distribution-free bound, 
        a float is a fixed stability parameter,
        and None does no correction.
    delta : float
        Error rate for LTT (only used when LTT=True)
    min_n : int
        Minimum number of examples to use for LTT
    grid_size : int
        Number of grid points to use for LTT
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
        - false_flag_rate: fraction of abstained examples that were actually correct
    """
    # Get predictions and confidence scores
    cal_yhats = np.argmax(cal_smx, axis=1)
    cal_phats = cal_smx.max(axis=1)
    
    val_yhats = np.argmax(val_smx, axis=1)
    val_phats = val_smx.max(axis=1)
    
    # Compute error indicators
    cal_errors = (cal_yhats != cal_labels).astype(int)
    val_errors = (val_yhats != val_labels).astype(int)
    
    # Fit selective classifier
    estimator = SelectiveClassifierStabilityEstimator(alpha=alpha, n_bootstrap=1000)
    if stability_estimator == "df":
        beta_hat = estimator.estimate_beta_df(cal_phats, cal_errors)
    elif stability_estimator == 'K':
        beta_hat = estimator.estimate_beta_K(cal_phats, cal_errors)
    elif stability_estimator == 'def': 
        beta_hat = estimator.estimate_beta_def(cal_phats, cal_errors)
    elif isinstance(stability_estimator, float):
        beta_hat = stability_estimator
    else:
        beta_hat = 0

    # Compute conservative target
    alpha_adjusted = max(0.0, alpha - beta_hat)

    if LTT:
        classifier = LTTSelectiveClassifier(alpha=alpha, delta=delta, min_n=min_n, grid_size=grid_size)
    else:
        classifier = SelectiveClassifier(alpha=alpha_adjusted)
    theta_hat = classifier.fit(cal_phats, cal_errors)
    
    # Apply to validation set
    predictions_kept = classifier.predict(val_phats)
    
    # Compute metrics
    if predictions_kept.sum() > 0:
        empirical_selective_accuracy = (val_yhats[predictions_kept] == val_labels[predictions_kept]).mean()
        selective_error = val_errors[predictions_kept].mean()
    else:
        empirical_selective_accuracy = 0.0
        selective_error = 1.0
    
    if (~predictions_kept).sum() > 0:
        false_flag_rate = (val_yhats[~predictions_kept] == val_labels[~predictions_kept]).mean()
    else:
        false_flag_rate = 0.0
    
    prediction_rate = predictions_kept.mean()
    
    if verbose:
        print(f"Threshold θ̂: {theta_hat:.4f}")
        print(f"Empirical selective accuracy: {empirical_selective_accuracy:.4f}")
        print(f"Selective error rate: {selective_error:.4f}")
        print(f"Prediction rate (fraction kept): {prediction_rate:.4f}")
        print(f"False flag rate: {false_flag_rate:.4f}")
    
    results = {
        'theta_hat': theta_hat,
        'predictions_kept': predictions_kept,
        'empirical_selective_accuracy': empirical_selective_accuracy,
        'selective_error': selective_error,
        'prediction_rate': prediction_rate,
        'false_flag_rate': false_flag_rate,
        'classifier': classifier
    }
    
    return results

def estimate_beta_by_definition(cal_phats, cal_errors, alpha=0.1, n_bootstrap=1000):
    """
    Estimate beta by definition.
    """
    estimator = SelectiveClassifierStabilityEstimator(alpha=alpha, n_bootstrap=n_bootstrap)
    beta_hat = estimator.estimate_beta_def(cal_phats, cal_errors)
    return beta_hat