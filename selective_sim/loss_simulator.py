import numpy as np

def compute_loss(P_i, E_i, theta, alpha):
    """
    Selective prediction loss:
    ell(i; θ) = ind{E_i=1, P̂_i > θ} - alpha·ind{P̂_i > θ} + alpha
    """
    return (E_i * (P_i > theta).astype(float) - alpha * (P_i > theta).astype(float) + alpha).astype(float)

def error_prob_model(p, alpha, base_strength=10, instability_parameter=0):
    """
    Error probability model with parametrized deviation from monotonicity.
    
    Base model: sigmoid with base_strength
    Deviation: adds sinusoidal perturbation controlled by deviation_strength
    
    p(error | P̂) = (1-instability_parameter) * sigmoid(-base_strength * (P̂ - 0.5)) + instability_parameter * alpha
    """
    # Base monotone sigmoid component
    base_prob = 1 / (1 + np.exp(base_strength * (p - 0.5)))

    # Instability parameter component
    prob = (1-instability_parameter) * base_prob + instability_parameter * alpha

    # Clip to valid probability range
    return np.clip(prob, 0, 1)

def simulate_empirical_risk(n, alpha, theta_grid, instability_parameter, base_strength=10):
    """
    Simulate empirical risk for a given instability parameter.
    
    Parameters:
    -----------
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
        
    Returns:
    --------
    P_hat : np.ndarray
        Confidence scores
    E : np.ndarray
        Error indicators
    empirical_risk : np.ndarray
        Empirical risk at each threshold
    """
    # Generate confidence scores uniformly
    P_hat = np.random.uniform(0, 1, n)
    
    # Generate errors based on model with deviation
    error_prob = error_prob_model(P_hat, alpha, base_strength=base_strength, 
                                  instability_parameter=instability_parameter)
    E = np.random.binomial(1, error_prob)
    
    # Compute empirical risk
    empirical_risk = np.mean([compute_loss(P_hat[i], E[i], theta_grid, alpha) 
                              for i in range(n)], axis=0)
    
    return P_hat, E, empirical_risk