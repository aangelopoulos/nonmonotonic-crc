"""
Generic Conformal Risk Control
=================================================

Shared module implementing generic conformal risk control algorithms and stability estimation.

This module handles general losses (both monotonic and non-monotonic) using:
- Discretization for generic bounded losses
- Lipschitz-slope bounds for smooth losses
- Bootstrap-based stability estimation

All implementations use efficient numpy vectorized operations where possible.

This module can be imported by experiments requiring generic risk control.
"""

import numpy as np
from typing import Callable, Optional
from scipy.special import lambertw


class GenericConformalRiskControl:
    """
    Generic conformal risk control for non-monotonic losses.

    The algorithm finds the smallest parameter θ such that:
    (1/|D|) Σ ℓ(x,y;θ) ≤ α

    This works for arbitrary loss functions ℓ(x,y;θ) where θ ∈ Θ is a
    scalar or vector parameter.

    References
    ----------
    Angelopoulos, A. N. (2025). "Conformal Risk Control for Non-Monotonic Losses"
    Section 2.2 - Generic Algorithm
    Equation (11): A(D) = inf{θ : (1/|D|)Σ ℓ(x,y;θ) ≤ α}
    """

    def __init__(self, alpha: float = 0.1, theta_grid: Optional[np.ndarray] = None):
        """
        Initialize the generic risk controller.

        Parameters
        ----------
        alpha : float
            Target risk level
        theta_grid : np.ndarray, optional
            Grid of theta values to search over. If None, will be constructed
            based on the data during fit.
        """
        self.alpha = alpha
        self.theta_grid = theta_grid
        self.theta_hat = None

    def _compute_empirical_risk(self,
                                 X: np.ndarray,
                                 Y: np.ndarray,
                                 theta: float,
                                 loss_fn: Callable) -> float:
        """
        Compute empirical risk at parameter theta.

        Parameters
        ----------
        X : np.ndarray
            Features
        Y : np.ndarray
            Labels
        theta : float
            Parameter value
        loss_fn : callable
            Loss function ℓ(x, y; θ)

        Returns
        -------
        risk : float
            Empirical risk
        """
        # Try vectorized computation first
        try:
            losses = loss_fn(X, Y, theta)
            if np.isscalar(losses):
                # If loss_fn returns scalar, fall back to elementwise
                raise TypeError
            return np.mean(losses)
        except (TypeError, ValueError):
            # Fall back to element-wise computation
            n = len(X)
            losses = np.fromiter(
                (loss_fn(X[i], Y[i], theta) for i in range(n)),
                dtype=np.float64,
                count=n
            )
            return np.mean(losses)

    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            loss_fn: Callable,
            theta_grid: Optional[np.ndarray] = None) -> float:
        """
        Fit the risk controller by finding the optimal parameter.

        Algorithm from equation (11):
        A(D) = inf{θ : (1/|D|)Σ ℓ(x,y;θ) ≤ α}

        Parameters
        ----------
        X : np.ndarray, shape (n,)
            Features for calibration examples
        Y : np.ndarray, shape (n,)
            Labels for calibration examples
        loss_fn : callable
            Loss function ℓ(x, y; θ) that takes (x, y, theta) and returns loss
        theta_grid : np.ndarray, optional
            Grid of theta values to search. If None, uses self.theta_grid

        Returns
        -------
        theta_hat : float
            The chosen parameter value
        """
        if theta_grid is not None:
            self.theta_grid = theta_grid

        if self.theta_grid is None:
            raise ValueError("Must provide theta_grid either in __init__ or fit()")

        # Sort theta_grid to ensure we find the infimum (smallest valid theta)
        theta_sorted = np.sort(self.theta_grid)

        # Compute all empirical risks for efficiency
        m = len(theta_sorted)
        risks = np.zeros(m)
        for j, theta in enumerate(theta_sorted):
            risks[j] = self._compute_empirical_risk(X, Y, theta, loss_fn)

        # Find smallest theta where empirical risk ≤ alpha
        feasible = risks <= self.alpha
        if np.any(feasible):
            idx = np.argmax(feasible)  # First True value
            self.theta_hat = theta_sorted[idx]
        else:
            # If no theta satisfies the constraint, return the last (largest) value
            self.theta_hat = theta_sorted[-1]

        return self.theta_hat

    def predict_risk(self,
                     X_test: np.ndarray,
                     Y_test: np.ndarray,
                     loss_fn: Callable) -> float:
        """
        Compute realized risk on test set.

        Parameters
        ----------
        X_test : np.ndarray
            Test features
        Y_test : np.ndarray
            Test labels
        loss_fn : callable
            Loss function

        Returns
        -------
        risk : float
            Realized risk on test set
        """
        if self.theta_hat is None:
            raise ValueError("Must call fit() before predict_risk()")

        return self._compute_empirical_risk(X_test, Y_test, self.theta_hat, loss_fn)


class GenericStabilityEstimator:
    """
    Estimate stability parameter β for generic conformal risk control.

    Implements three methods from Section 2.3:
    1. Generic bootstrap bound (directly from definition)
    2. Discretization bound for generic losses
    3. Lipschitz-slope bound for smooth losses

    References
    ----------
    Section 2.3 - Stability Estimation
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

    def estimate_beta_def(self,
                         X: np.ndarray,
                         Y: np.ndarray,
                         loss_fn: Callable,
                         theta_grid: np.ndarray) -> float:
        """
        Estimate β directly from definition (Section 2.3.1).

        This is the generic bootstrap-based method that works for any loss.

        Δ = (1/(n+1)) Σ_i [ℓ(Z_i; A(D_{-i})) - ℓ(Z_i; A*(D))]
        β̂_def = (Δ̄)_+

        Parameters
        ----------
        X : np.ndarray
            Features
        Y : np.ndarray
            Labels
        loss_fn : callable
            Loss function ℓ(x, y; θ)
        theta_grid : np.ndarray
            Grid of theta values

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(X)
        Delta_values = np.zeros(self.n_bootstrap)

        # Bootstrap
        for b in range(self.n_bootstrap):
            # Sample with replacement (n+1 samples)
            idx = np.random.choice(n, size=n+1, replace=True)
            X_boot = X[idx]
            Y_boot = Y[idx]

            # Fit A* on full bootstrap data
            controller_star = GenericConformalRiskControl(alpha=self.alpha)
            theta_star = controller_star.fit(X_boot, Y_boot, loss_fn, theta_grid)

            # Compute leave-one-out losses using vectorized operations
            loss_diffs = np.zeros(n+1)

            for i in range(n+1):
                # Leave one out
                mask = np.ones(n+1, dtype=bool)
                mask[i] = False
                X_loo = X_boot[mask]
                Y_loo = Y_boot[mask]

                # Fit A on leave-one-out data
                controller_loo = GenericConformalRiskControl(alpha=self.alpha)
                theta_loo = controller_loo.fit(X_loo, Y_loo, loss_fn, theta_grid)

                # Compute loss difference
                loss_loo = loss_fn(X_boot[i], Y_boot[i], theta_loo)
                loss_star = loss_fn(X_boot[i], Y_boot[i], theta_star)

                loss_diffs[i] = loss_loo - loss_star

            Delta_values[b] = np.mean(loss_diffs)

        # Take positive part of average
        Delta_bar = np.mean(Delta_values)
        beta_hat = max(0, Delta_bar)

        return beta_hat

    def estimate_beta_discretized(self,
                                   n: int,
                                   m: int = 100) -> float:
        """
        Estimate β using discretization for generic bounded losses (Section 2.3.2).

        For general losses bounded in [0,1], uses Proposition 5 which gives:
        E[ℓ(Z_{n+1}; A(D_{1:n}))] ≤ α + ε* + 1/(4nε*)

        where ε* = √(-W_{-1}(-1/(4n(m+1)²))/(8n))
        and W_{-1} is the -1st branch of the Lambert W function.

        This assumes:
        - Loss is bounded in [0,1]
        - Theta space is discretized to Θ_m = {0, 1/m, 2/m, ..., 1}

        This is a distribution-free bound that only depends on the sample size n
        and the discretization level m, not on the actual data.

        Parameters
        ----------
        n : int
            Sample size
        m : int
            Number of discretization points (default 100)

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """

        # Compute ε* using Lambert W function
        # ε* = √(-W_{-1}(-1/(4n(m+1)²))/(8n))
        arg = -1.0 / (4 * n * (m + 1)**2)

        # The Lambert W_{-1} branch requires arg ∈ [-1/e, 0)
        if arg < -1/np.e or arg >= 0:
            # If outside valid range, use simplified approximation
            # β ≈ √(ln(n(m+1)) / n)
            epsilon_star = np.sqrt(np.log(n * (m + 1)) / n)
        else:
            w_minus1 = np.real(lambertw(arg, k=-1))
            epsilon_star = np.sqrt(-w_minus1 / (8 * n))

        # The bound is: β ≤ ε* + 1/(4nε*)
        beta_hat = epsilon_star + 1.0 / (4 * n * epsilon_star)

        return beta_hat

    def estimate_beta_smooth(self,
                            X: np.ndarray,
                            Y: np.ndarray,
                            loss_fn: Callable,
                            theta_grid: np.ndarray,
                            delta_min: float = 1e-4,
                            delta_max: float = 1e-1) -> float:
        """
        Estimate β using Lipschitz-slope bound for smooth losses (Section 2.3.3).

        Under Proposition 3 (stability for smooth losses):
        β ≤ L / (m(n+1))

        where:
        - L is the Lipschitz constant of the loss w.r.t. θ
        - m is the local slope of the empirical risk near the root

        We estimate both L and m from bootstrap replicates.

        Parameters
        ----------
        X : np.ndarray
            Features
        Y : np.ndarray
            Labels
        loss_fn : callable
            Loss function ℓ(x, y; θ), assumed continuous and Lipschitz in θ
        theta_grid : np.ndarray
            Grid of theta values for estimation
        delta_min : float
            Minimum delta for slope estimation
        delta_max : float
            Maximum delta for slope estimation

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(X)
        beta_values = np.zeros(self.n_bootstrap)

        # Bootstrap
        for b in range(self.n_bootstrap):
            # Sample with replacement (n+1 samples)
            idx = np.random.choice(n, size=n+1, replace=True)
            X_boot = X[idx]
            Y_boot = Y[idx]

            # Estimate Lipschitz constant L using vectorized operations
            m_theta = len(theta_grid)

            # Compute loss matrix: losses[i, j] = loss_fn(X_boot[i], Y_boot[i], theta_grid[j])
            losses = np.zeros((n+1, m_theta))
            for i in range(n+1):
                for j in range(m_theta):
                    losses[i, j] = loss_fn(X_boot[i], Y_boot[i], theta_grid[j])

            # Compute Lipschitz constants for all pairs of theta values
            # For each data point, compute max |loss(θ1) - loss(θ2)| / |θ1 - θ2|
            theta_diffs = np.abs(theta_grid[:, None] - theta_grid[None, :])
            # Avoid division by zero
            theta_diffs[theta_diffs == 0] = np.inf

            L_hat = 0.0
            for i in range(n+1):
                loss_diffs = np.abs(losses[i, :, None] - losses[i, None, :])
                L_estimates_i = loss_diffs / theta_diffs
                L_hat = max(L_hat, np.max(L_estimates_i[np.isfinite(L_estimates_i)]))

            # Fit algorithm on bootstrap data
            controller = GenericConformalRiskControl(alpha=self.alpha)
            theta_hat_boot = controller.fit(X_boot, Y_boot, loss_fn, theta_grid)

            # Estimate local slope m near theta_hat
            # m = inf_{δ ∈ [δ_min, δ_max]} |R(θ̂+δ) - R(θ̂)| / δ
            delta_range = np.linspace(delta_min, delta_max, 20)

            # Vectorize slope computation
            theta_perturbed = theta_hat_boot + delta_range
            valid_mask = (theta_perturbed >= theta_grid.min()) & (theta_perturbed <= theta_grid.max())

            if np.any(valid_mask):
                risk_hat = controller._compute_empirical_risk(
                    X_boot, Y_boot, theta_hat_boot, loss_fn
                )
                risks_perturbed = np.array([
                    controller._compute_empirical_risk(X_boot, Y_boot, theta_p, loss_fn)
                    for theta_p in theta_perturbed[valid_mask]
                ])
                slopes = np.abs(risks_perturbed - risk_hat) / delta_range[valid_mask]
                m_hat = max(np.min(slopes), 1e-6)
            else:
                m_hat = 1e-6

            # Compute beta estimate for this bootstrap sample
            beta_values[b] = L_hat / (m_hat * (n + 1))

        # Average across bootstrap samples
        beta_hat = np.mean(beta_values)

        return beta_hat

    def estimate_all(self,
                    X: np.ndarray,
                    Y: np.ndarray,
                    loss_fn: Callable,
                    theta_grid: np.ndarray,
                    m: int = 100,
                    delta_min: float = 1e-4,
                    delta_max: float = 1e-1) -> dict:
        """
        Estimate β using all three methods and return as a dictionary.

        Parameters
        ----------
        X : np.ndarray
            Features
        Y : np.ndarray
            Labels
        loss_fn : callable
            Loss function
        theta_grid : np.ndarray
            Grid of theta values
        m : int
            Discretization parameter for discretized method
        delta_min : float
            Min delta for smooth method
        delta_max : float
            Max delta for smooth method

        Returns
        -------
        estimates : dict
            Dictionary with keys 'definition', 'discretized', 'smooth'
        """
        n = len(X)

        print("Estimating β from definition (bootstrap)...")
        beta_def = self.estimate_beta_def(X, Y, loss_fn, theta_grid)

        print("Estimating β using discretization bound...")
        beta_disc = self.estimate_beta_discretized(n, m)

        print("Estimating β using Lipschitz-slope bound...")
        beta_smooth = self.estimate_beta_smooth(X, Y, loss_fn, theta_grid,
                                                delta_min, delta_max)

        return {
            'definition': beta_def,
            'discretized': beta_disc,
            'smooth': beta_smooth
        }
