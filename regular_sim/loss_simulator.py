"""
Loss Simulator for Regular (Smooth/Continuous) Losses

This module simulates empirical risk for smooth, continuous loss functions
that are monotonic in the confidence score.
"""

import numpy as np


def compute_loss(y_true, y_pred, alpha):
    """
    Smooth loss function (e.g., squared error or cross-entropy style).
    
    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Predicted probabilities (continuous in [0,1])
    alpha : float
        Risk level parameter
        
    Returns
    -------
    loss : array-like
        Smooth loss values
    """
    # Squared error loss adjusted for selective prediction
    return (y_true - y_pred) ** 2 + alpha * (1 - y_pred)


def error_prob_model(p, alpha, base_strength=10, instability_parameter=0):
    """
    Error probability model (same as selective case for consistency).
    
    p(error | P̂) = (1-ε) * sigmoid(-base_strength * (P̂ - 0.5)) + ε * alpha
    """
    # Base monotone sigmoid component
    base_prob = 1 / (1 + np.exp(base_strength * (p - 0.5)))
    
    # Instability parameter component
    prob = (1 - instability_parameter) * base_prob + instability_parameter * alpha
    
    # Clip to valid probability range
    return np.clip(prob, 0, 1)


def simulate_empirical_risk(n, alpha, theta_grid, instability_parameter, base_strength=10):
    """
    Simulate empirical risk for regular (smooth) loss.
    
    Parameters
    ----------
    n : int
        Number of datapoints
    alpha : float
        Target risk level
    theta_grid : np.ndarray
        Grid of threshold values
    instability_parameter : float
        Instability parameter (0 = monotone, 1 = maximum instability)
    base_strength : float
        Base strength of sigmoid function
        
    Returns
    -------
    P_hat : np.ndarray
        Confidence scores
    y_true : np.ndarray
        True labels (continuous)
    empirical_risk : np.ndarray
        Empirical risk at each threshold (smooth loss)
    """
    # Generate confidence scores uniformly
    P_hat = np.random.uniform(0, 1, n)
    
    # Generate true probabilities based on model
    true_prob = 1 - error_prob_model(P_hat, alpha, base_strength, instability_parameter)
    
    # Generate continuous targets (for smooth loss)
    y_true = np.random.binomial(1, true_prob)
    
    # Compute smooth empirical risk for each threshold
    empirical_risk = []
    for theta in theta_grid:
        # Select based on confidence
        selected = P_hat > theta
        if selected.sum() > 0:
            # Use smooth loss on selected examples
            losses = compute_loss(y_true[selected], P_hat[selected], alpha)
            empirical_risk.append(losses.mean())
        else:
            empirical_risk.append(1.0)  # High penalty for no selection
    
    empirical_risk = np.array(empirical_risk)
    
    return P_hat, y_true, empirical_risk

