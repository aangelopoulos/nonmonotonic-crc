"""
Selective Classification
=================================================

Shared module implementing selective classification algorithms and stability estimation.

This module can be imported by both selective_sim and selective_imagenet experiments.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom
from typing import Callable, Optional


class SelectiveClassifier:
    """
    Selective classification with conformal risk control for non-monotonic losses.
    
    The algorithm finds a confidence threshold θ such that:
    P(Ŷ ≠ Y | P̂ > θ) ≤ α
    
    where Ŷ are predictions, Y are true labels, and P̂ are confidence scores.
    
    References
    ----------
    Angelopoulos, A. N. (2025). "Conformal Risk Control for Non-Monotonic Losses"
    Section 2.2.1 - Selective Classification Algorithm
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize the selective classifier.
        
        Parameters
        ----------
        alpha : float
            Target risk level (miscoverage rate on selected examples)
        """
        self.alpha = alpha
        self.theta_hat = None
        self.j_star = None
        
    def compute_loss(self, P_i: float, E_i: int, theta: float) -> float:
        """
        Selective prediction loss function.
        
        ℓ(i; θ) = 𝟙{E_i=1, P̂_i > θ} - α·𝟙{P̂_i > θ} + α
        
        Parameters
        ----------
        P_i : float
            Confidence score for example i
        E_i : int
            Error indicator (1 if prediction is wrong, 0 if correct)
        theta : float
            Confidence threshold
            
        Returns
        -------
        loss : float
            Loss value
        """
        return E_i * (P_i > theta) - self.alpha * (P_i > theta) + self.alpha
    
    def _compute_empirical_risk(self, P_hat: np.ndarray, E: np.ndarray, theta: float) -> float:
        """Compute empirical risk at threshold theta."""
        n = len(P_hat)
        return np.mean([self.compute_loss(P_hat[i], E[i], theta) for i in range(n)])
    
    def fit(self, P_hat: np.ndarray, E: np.ndarray) -> float:
        """
        Fit the selective classifier by finding the optimal threshold.
        
        Algorithm A(D) from equation (11):
        A(D) = inf{θ : (1/|D|)Σ ℓ(x,y;θ) ≤ α}
        
        Parameters
        ----------
        P_hat : np.ndarray, shape (n,)
            Confidence scores for calibration examples
        E : np.ndarray, shape (n,)
            Error indicators (1 if prediction wrong, 0 if correct)
            
        Returns
        -------
        theta_hat : float
            The chosen confidence threshold
        """
        n = len(P_hat)
        
        # Get order statistics (sorted indices by decreasing confidence)
        V = np.argsort(P_hat)[::-1]
        
        # Separate correct and incorrect predictions
        V_eq = V[E[V] == 0]  # Correct predictions
        V_neq = V[E[V] == 1]  # Incorrect predictions
        
        # Compute weights: w_i = α for correct, -(1-α) for incorrect
        w = np.zeros(n)
        w[V_eq] = self.alpha
        w[V_neq] = -(1 - self.alpha)
        
        # Find j* using the index-space algorithm (equation after line 199)
        # j* = max{j ∈ {0,...,n} : 1 + Σ_{i≤j} w_i ≥ 0}
        cumsum_w = np.cumsum(w[V])
        feasible = 1 + cumsum_w >= 0
        
        if np.any(feasible):
            self.j_star = np.where(feasible)[0][-1]
        else:
            self.j_star = -1
        
        # Convert back to threshold
        if self.j_star >= 0:
            self.theta_hat = P_hat[V[self.j_star]]
        else:
            self.theta_hat = 1.0  # Reject all if no threshold works
            
        return self.theta_hat
    
    def predict(self, P_hat_test: np.ndarray) -> np.ndarray:
        """
        Predict which examples to select (not abstain from).
        
        Parameters
        ----------
        P_hat_test : np.ndarray
            Confidence scores for test examples
            
        Returns
        -------
        selected : np.ndarray
            Boolean array indicating which examples are selected
        """
        if self.theta_hat is None:
            raise ValueError("Must call fit() before predict()")
        return P_hat_test > self.theta_hat
    
    def get_prediction_rate(self, P_hat_test: np.ndarray) -> float:
        """
        Compute prediction rate (fraction of examples selected).
        
        Parameters
        ----------
        P_hat_test : np.ndarray
            Confidence scores for test examples
            
        Returns
        -------
        prediction_rate : float
            Fraction of examples with P̂ > θ̂
        """
        return self.predict(P_hat_test).mean()
    
    def get_realized_risk(self, P_hat_test: np.ndarray, E_test: np.ndarray) -> float:
        """
        Compute realized risk on test set (error rate on selected examples).
        
        Parameters
        ----------
        P_hat_test : np.ndarray
            Confidence scores for test examples
        E_test : np.ndarray
            Error indicators for test examples
            
        Returns
        -------
        risk : float
            Error rate on selected examples, or 0 if no examples selected
        """
        selected = self.predict(P_hat_test)
        if selected.sum() > 0:
            return E_test[selected].mean()
        else:
            return 0.0

class LTTSelectiveClassifier:
    """
    Selective classification with LTT for non-monotonic losses.
    
    The algorithm finds a confidence threshold θ such that:
    P(P(Ŷ ≠ Y | P̂ > θ, D) ≤ α) > 1-\delta
    
    where Ŷ are predictions, Y are true labels, and P̂ are confidence scores.
    
    References
    ----------
    Angelopoulos, A. N. (2025). "Conformal Risk Control for Non-Monotonic Losses"
    Section 2.2.1 - Selective Classification Algorithm
    """
    
    def __init__(self, alpha: float = 0.1, delta: float = 0.1, min_n: int = 50, grid_size: int = 5000):
        """
        Initialize the selective classifier.
        
        Parameters
        ----------
        alpha : float
            Target risk level (miscoverage rate on selected examples)

        delta : float
            Target failure level (fraction of calibration sets for which guarantee can fail)
        """
        self.alpha = alpha
        self.delta = delta
        self.theta_hat = None
        self.j_star = -1
        self.min_n = min_n
        self.grid_size = grid_size
        
    def compute_loss(self, P_i: float, E_i: int, theta: float) -> float:
        """
        Selective prediction loss function.
        
        ℓ(i; θ) = 𝟙{E_i=1, P̂_i > θ} - α·𝟙{P̂_i > θ} + α
        
        Parameters
        ----------
        P_i : float
            Confidence score for example i
        E_i : int
            Error indicator (1 if prediction is wrong, 0 if correct)
        theta : float
            Confidence threshold
            
        Returns
        -------
        loss : float
            Loss value
        """
        return E_i * (P_i > theta) - self.alpha * (P_i > theta) + self.alpha
    
    def _compute_empirical_risk(self, P_hat: np.ndarray, E: np.ndarray, theta: float) -> float:
        """Compute empirical risk at threshold theta."""
        n = len(P_hat)
        return np.mean([self.compute_loss(P_hat[i], E[i], theta) for i in range(n)])
    
    def fit(self, P_hat: np.ndarray, E: np.ndarray) -> float:
        """
        Fit the selective classifier by using LTT
                
        Parameters
        ----------
        P_hat : np.ndarray, shape (n,)
            Confidence scores for calibration examples
        E : np.ndarray, shape (n,)
            Error indicators (1 if prediction wrong, 0 if correct)
            
        Returns
        -------
        theta_hat : float
            The chosen confidence threshold
        """
        n = len(P_hat)
        
        # Get order statistics (sorted indices by decreasing confidence)
        V = np.argsort(P_hat)[::-1]
        
        # Separate correct and incorrect predictions
        V_eq = V[E[V] == 0]  # Correct predictions
        V_neq = V[E[V] == 1]  # Incorrect predictions
        
        # Compute weights: w_i = α for correct, -(1-α) for incorrect
        w = np.zeros(n)
        w[V_eq] = self.alpha
        w[V_neq] = -(1 - self.alpha)
        
        # Find j* using LTT
        cumsum_w = np.cumsum(w[V])

        lb = self.get_lb_cumsum_w(cumsum_w, n)

        infeasible = 1 + lb < 0
        
        if np.all(infeasible[self.min_n:]):
            self.j_star = -1
        elif np.any(infeasible[self.min_n:]):
            self.j_star = np.where(infeasible[self.min_n:])[0][0] + self.min_n - 1
        else:
            print("Predict all")
            self.j_star = n-1
        
        # Convert back to threshold
        if self.j_star >= 0:
            self.theta_hat = P_hat[V[self.j_star]]
        else:
            self.theta_hat = 1.0  # Reject all if no threshold works
            
        return self.theta_hat

    def get_lb_cumsum_w(self, cumsum_w, n):
        lbs = -np.arange(1,n+1)*(1-self.alpha) # by default, none are feasible
        for j in range(self.min_n,n+1):
            if self.invert_for_lb(0,cumsum_w[j-1],j) == self.invert_for_lb(j,cumsum_w[j-1],j):
                lbs[j-1] = -j*(1-self.alpha)
            else:
                lbs[j-1] = brentq(self.invert_for_lb, 0, j-1e-8, args=(cumsum_w[j-1],j))-j*(1-self.alpha)
        return lbs

    def invert_for_lb(self, r, cumsum_w, num_samples):
        inversion = binom.cdf(cumsum_w+num_samples*(1-self.alpha), num_samples, r/num_samples) - (1-self.delta)
        return inversion
  
    def predict(self, P_hat_test: np.ndarray) -> np.ndarray:
        """
        Predict which examples to select (not abstain from).
        
        Parameters
        ----------
        P_hat_test : np.ndarray
            Confidence scores for test examples
            
        Returns
        -------
        selected : np.ndarray
            Boolean array indicating which examples are selected
        """
        if self.theta_hat is None:
            raise ValueError("Must call fit() before predict()")
        return P_hat_test > self.theta_hat
    
    def get_prediction_rate(self, P_hat_test: np.ndarray) -> float:
        """
        Compute prediction rate (fraction of examples selected).
        
        Parameters
        ----------
        P_hat_test : np.ndarray
            Confidence scores for test examples
            
        Returns
        -------
        prediction_rate : float
            Fraction of examples with P̂ > θ̂
        """
        return self.predict(P_hat_test).mean()
    
    def get_realized_risk(self, P_hat_test: np.ndarray, E_test: np.ndarray) -> float:
        """
        Compute realized risk on test set (error rate on selected examples).
        
        Parameters
        ----------
        P_hat_test : np.ndarray
            Confidence scores for test examples
        E_test : np.ndarray
            Error indicators for test examples
            
        Returns
        -------
        risk : float
            Error rate on selected examples, or 0 if no examples selected
        """
        selected = self.predict(P_hat_test)
        if selected.sum() > 0:
            return E_test[selected].mean()
        else:
            return 0.0

class SelectiveClassifierStabilityEstimator:
    """
    Estimate stability parameter β for selective classification.
    
    Implements the bootstrap methods from Section 2.3.
    
    References
    ----------
    Section 2.3 - Stability Estimation for Selective Classification
    """
    
    def __init__(self, alpha: float = 0.1, n_bootstrap: int = 1000):
        """
        Initialize stability estimator.
        
        Parameters
        ----------
        alpha : float
            Target risk level
        n_bootstrap : int
            Number of bootstrap replicates
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        
    def _compute_K(self, P_hat: np.ndarray, E: np.ndarray) -> int:
        """
        Compute K = max_i |ĵ_{-i} - j*|
        
        This measures the stability of the algorithm under leave-one-out.
        """
        n = len(P_hat)
        classifier = SelectiveClassifier(alpha=self.alpha)
        
        # Fit on full data
        classifier.fit(P_hat, E)
        j_star = classifier.j_star
        
        # Compute leave-one-out indices
        j_loo = np.zeros(n, dtype=int)
        for i in range(n):
            # Leave one out
            idx = np.concatenate([np.arange(i), np.arange(i+1, n)])
            P_hat_loo = P_hat[idx]
            E_loo = E[idx]
            
            classifier_loo = SelectiveClassifier(alpha=self.alpha)
            classifier_loo.fit(P_hat_loo, E_loo)
            j_loo[i] = classifier_loo.j_star
        
        # Compute K
        K = np.max(np.abs(j_loo - j_star))
        return K
    
    def estimate_beta_K(self, P_hat: np.ndarray, E: np.ndarray) -> float:
        """
        Estimate β using the K-based method (Section 2.3.2).
        
        β̂_K = (2·max{α, 1-α}/(n+1)) · E[K]
        
        Parameters
        ----------
        P_hat : np.ndarray
            Confidence scores
        E : np.ndarray
            Error indicators
            
        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(P_hat)
        K_values = []
        
        # Bootstrap
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            idx = np.random.choice(n, size=n+1, replace=True)
            P_boot = P_hat[idx]
            E_boot = E[idx]
            
            # Compute K for this bootstrap sample
            K_b = self._compute_K(P_boot, E_boot)
            K_values.append(K_b)
        
        # Average K across bootstrap samples
        K_bar = np.mean(K_values)
        
        # Compute beta estimate (Proposition 3)
        beta_hat = (2 * max(self.alpha, 1 - self.alpha) * K_bar) / (n + 1)
        
        return beta_hat
    
    def estimate_beta_df(self, P_hat: np.ndarray, E: np.ndarray) -> float:
        """
        Estimate β using distribution-free bound (Section 2.3.3).
        
        Uses Proposition 4: E[K] ≤ Σ_j P(Ē_j ∈ I_j)
        where I_j = (α + (1-α)/j, α + (2-α)/j]
        
        Parameters
        ----------
        P_hat : np.ndarray
            Confidence scores
        E : np.ndarray
            Error indicators
            
        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(P_hat)
        EK_estimates = []
        
        # Bootstrap
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            idx = np.random.choice(n, size=n+1, replace=True)
            P_boot = P_hat[idx]
            E_boot = E[idx]
            
            # Sort by decreasing confidence
            sort_idx = np.argsort(P_boot)[::-1]
            E_sorted = E_boot[sort_idx]
            
            # Compute indicators J_j
            J_sum = 0
            for j in range(1, n+2):
                # Compute Ē_j (average error up to position j)
                E_bar_j = np.mean(E_sorted[:j])
                
                # Check if Ē_j is in interval I_j
                lower = self.alpha + (1 - self.alpha) / j
                upper = self.alpha + (2 - self.alpha) / j
                
                if lower < E_bar_j <= upper:
                    J_sum += 1
            
            EK_estimates.append(J_sum)
        
        # Average across bootstrap samples
        EK_hat = np.mean(EK_estimates)
        
        # Compute beta estimate
        beta_hat = (2 * max(self.alpha, 1 - self.alpha) * EK_hat) / (n + 1)
        
        return beta_hat
    
    def estimate_beta_def(self, P_hat: np.ndarray, E: np.ndarray, 
                          A_star: Optional[Callable] = None) -> float:
        """
        Estimate β directly from definition (Section 2.3.1).
        
        Δ = (1/(n+1)) Σ_i [ℓ(Z_i; A(D_{-i})) - ℓ(Z_i; A*(D))]
        β̂_def = (Δ̄)_+
        
        Parameters
        ----------
        P_hat : np.ndarray
            Confidence scores
        E : np.ndarray
            Error indicators
        A_star : callable, optional
            Reference algorithm. If None, uses A* = A (same algorithm on full data)
            
        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(P_hat)
        Delta_values = []
        
        # Bootstrap
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            idx = np.random.choice(n, size=n+1, replace=True)
            P_boot = P_hat[idx]
            E_boot = E[idx]
            
            # Fit A* on full bootstrap data
            classifier_star = SelectiveClassifier(alpha=self.alpha)
            theta_star = classifier_star.fit(P_boot, E_boot)
            
            # Compute leave-one-out losses
            delta_sum = 0
            for i in range(n+1):
                # Leave one out
                loo_idx = np.concatenate([np.arange(i), np.arange(i+1, n+1)])
                P_loo = P_boot[loo_idx]
                E_loo = E_boot[loo_idx]
                
                # Fit A on leave-one-out data
                classifier_loo = SelectiveClassifier(alpha=self.alpha)
                theta_loo = classifier_loo.fit(P_loo, E_loo)
                
                # Compute loss difference
                loss_loo = classifier_loo.compute_loss(P_boot[i], E_boot[i], theta_loo)
                loss_star = classifier_star.compute_loss(P_boot[i], E_boot[i], theta_star)
                
                delta_sum += loss_loo - loss_star
            
            Delta_values.append(delta_sum / (n + 1))
        
        # Take positive part of average
        Delta_bar = np.mean(Delta_values)
        beta_hat = max(0, Delta_bar)
        
        return beta_hat