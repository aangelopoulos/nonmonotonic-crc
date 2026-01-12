"""
Discretized Regularized Empirical Risk Minimization
====================================================

Implements regularized ERM with precomputed loss matrices and discrete theta grids,
following the structure of generic.py but for ERM instead of conformal risk control.

This module handles:
- Discretized regularized ERM: A(D) = argmin_{θ ∈ Θ_grid} R̂_D(θ) + (λ/2)||θ||²_2
- Stability estimation for discretized ERM
- Bootstrap-based stability bounds
- Direct stability estimation from definition

All implementations use efficient numpy vectorized operations where possible.

References
----------
Section 2.3 - Guarantees for empirical risk minimization
"""

import numpy as np
from typing import Optional


class DiscretizedRegularizedERM:
    """
    Discretized regularized empirical risk minimization.

    Given a precomputed loss matrix (n x m) for n data points and m theta values,
    finds the theta that minimizes: R̂_D(θ) + (λ/2)||θ||²_2

    This is analogous to GenericConformalRiskControl but for ERM instead of CRC.

    References
    ----------
    Section 2.3, Equation after line 416
    """

    def __init__(self, lam: float = 0.01, theta_grid: Optional[np.ndarray] = None):
        """
        Initialize discretized regularized ERM.

        Parameters
        ----------
        lam : float
            Regularization parameter λ
        theta_grid : np.ndarray, optional
            Grid of theta values to search over
        """
        self.lam = lam
        self.theta_grid = theta_grid
        self.theta_hat = None
        self.theta_hat_idx = None

    def _compute_regularized_risk(self,
                                  loss_matrix: np.ndarray,
                                  theta_idx: int) -> float:
        """
        Compute regularized empirical risk at theta index.

        Parameters
        ----------
        loss_matrix : np.ndarray
            n x m matrix of losses
        theta_idx : int
            Index into theta_grid

        Returns
        -------
        regularized_risk : float
            R̂_D(θ) + (λ/2)||θ||²_2
        """
        empirical_risk = np.mean(loss_matrix[:, theta_idx])
        theta = self.theta_grid[theta_idx]

        # Handle scalar and vector theta
        if np.isscalar(theta):
            regularization = 0.5 * self.lam * theta ** 2
        else:
            regularization = 0.5 * self.lam * np.sum(theta ** 2)

        return empirical_risk + regularization

    def fit(self,
            loss_matrix: np.ndarray,
            theta_grid: Optional[np.ndarray] = None) -> float:
        """
        Fit ERM by finding the theta that minimizes regularized risk.

        Algorithm: A(D) = argmin_{θ ∈ Θ_grid} (1/|D|)Σ ℓ(x,y;θ) + (λ/2)||θ||²

        Parameters
        ----------
        loss_matrix : np.ndarray, shape (n, m)
            n x m matrix of losses for n data points and m theta values
        theta_grid : np.ndarray, optional
            Grid of theta values. If None, uses self.theta_grid

        Returns
        -------
        theta_hat : float or np.ndarray
            The chosen parameter value
        """
        if theta_grid is not None:
            self.theta_grid = theta_grid

        if self.theta_grid is None:
            raise ValueError("Must provide theta_grid either in __init__ or fit()")

        # Compute regularized risk for all theta values
        m = loss_matrix.shape[1]
        regularized_risks = np.zeros(m)

        for j in range(m):
            regularized_risks[j] = self._compute_regularized_risk(loss_matrix, j)

        # Find minimum
        self.theta_hat_idx = np.argmin(regularized_risks)
        self.theta_hat = self.theta_grid[self.theta_hat_idx]

        return self.theta_hat

    def predict_risk(self, loss_matrix: np.ndarray) -> float:
        """
        Compute realized empirical risk on test set (without regularization).

        Parameters
        ----------
        loss_matrix : np.ndarray
            n x m matrix of losses for n data points and m theta values

        Returns
        -------
        risk : float
            Realized empirical risk on test set
        """
        if self.theta_hat is None:
            raise ValueError("Must call fit() before predict_risk()")

        return np.mean(loss_matrix[:, self.theta_hat_idx])


class DiscretizedERMStabilityEstimator:
    """
    Estimate stability parameter β for discretized regularized ERM.

    Implements stability estimation methods adapted for discretized ERM:
    1. Loss-based stability (Proposition: erm-stable-loss)
    2. Direct loss-based stability (from definition, with discrete selection)

    The discretized version works with precomputed loss matrices and discrete
    theta grids, similar to GenericStabilityEstimator.

    References
    ----------
    Section 2.3 - Stability estimation for ERM
    """

    def __init__(self, lam: float = 0.01, n_bootstrap: int = 100):
        """
        Initialize stability estimator.

        Parameters
        ----------
        lam : float
            Regularization parameter λ
        n_bootstrap : int
            Number of bootstrap replicates
        """
        self.lam = lam
        self.n_bootstrap = n_bootstrap

    def estimate_beta_loss(self,
                          loss_matrix: np.ndarray,
                          theta_grid: np.ndarray) -> float:
        """
        Estimate β using loss-based bound (line 674-682).

        For Proposition erm-stable-loss:
        β̂_ERMLoss = 2ρ̂²/(λ(n+1))

        where ρ̂² estimates the Lipschitz constant squared.

        Parameters
        ----------
        loss_matrix : np.ndarray
            n x m matrix of losses
        theta_grid : np.ndarray
            Grid of theta values (m values)

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n, m = loss_matrix.shape
        rho_squared_sum = 0.0

        # For each data point, estimate its Lipschitz constant
        for i in range(n):
            max_lipschitz_sq = 0.0

            # Compute Lipschitz constant over grid
            for j in range(m):
                for k in range(j + 1, m):
                    theta_j = theta_grid[j]
                    theta_k = theta_grid[k]

                    loss_diff = abs(loss_matrix[i, j] - loss_matrix[i, k])

                    # Handle scalar and vector theta
                    if np.isscalar(theta_j):
                        theta_diff = abs(theta_j - theta_k)
                    else:
                        theta_diff = np.linalg.norm(theta_j - theta_k)

                    if theta_diff > 0:
                        lipschitz_est = loss_diff / theta_diff
                        max_lipschitz_sq = max(max_lipschitz_sq, lipschitz_est ** 2)

            rho_squared_sum += max_lipschitz_sq

        rho_squared_hat = rho_squared_sum / n
        beta_hat = 2 * rho_squared_hat / (self.lam * (n + 1))

        return beta_hat

    def estimate_beta_def(self,
                         loss_matrix: np.ndarray,
                         theta_grid: np.ndarray) -> float:
        """
        Estimate β directly from definition (Section 2.3.1).

        This is the generic bootstrap-based method adapted for ERM.

        Computes:
        Δ = (1/(n+1)) Σ_i [ℓ(Z_i; A(D_{-i})) - ℓ(Z_i; A*(D_{1:n+1}))]
        β̂_def = (Δ̄)_+

        Uses bootstrap replicates and discrete optimization.

        Parameters
        ----------
        loss_matrix : np.ndarray
            n x m matrix of losses
        theta_grid : np.ndarray
            Grid of theta values

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(loss_matrix)
        Delta_values = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, size=n + 1, replace=True)
            loss_matrix_boot = loss_matrix[idx]

            # Fit A* on full bootstrap data
            erm_star = DiscretizedRegularizedERM(lam=self.lam)
            erm_star.fit(loss_matrix_boot, theta_grid)
            idx_star = erm_star.theta_hat_idx

            # Compute leave-one-out losses
            loss_diffs = np.zeros(n + 1)

            for i in range(n + 1):
                # Leave-one-out dataset
                mask = np.ones(n + 1, dtype=bool)
                mask[i] = False
                loss_matrix_loo = loss_matrix_boot[mask]

                # Fit A on leave-one-out data
                erm_loo = DiscretizedRegularizedERM(lam=self.lam)
                erm_loo.fit(loss_matrix_loo, theta_grid)
                idx_loo = erm_loo.theta_hat_idx

                # Compute loss difference
                loss_loo = loss_matrix_boot[i, idx_loo]
                loss_star = loss_matrix_boot[i, idx_star]

                loss_diffs[i] = loss_loo - loss_star

            Delta_values[b] = np.mean(loss_diffs)

        # Take positive part of average
        Delta_bar = np.mean(Delta_values)
        beta_hat = max(0.0, Delta_bar)

        return beta_hat

    def estimate_beta_discretized(self,
                                   n: int,
                                   m: int = 100) -> float:
        """
        Estimate β using discretization bound for generic bounded losses.

        This is the distribution-free bound from Section 2.3.2 that only
        depends on sample size n and discretization level m.

        For general losses bounded in [0,1], uses:
        β ≤ ε* + 1/(4nε*)

        where ε* = √(-W_{-1}(-1/(4n(m+1)²))/(8n))

        Parameters
        ----------
        n : int
            Sample size
        m : int
            Number of discretization points

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        from scipy.special import lambertw

        # Compute ε* using Lambert W function
        arg = -1.0 / (4 * n * (m + 1)**2)

        # The Lambert W_{-1} branch requires arg ∈ [-1/e, 0)
        if arg < -1/np.e or arg >= 0:
            # If outside valid range, use simplified approximation
            epsilon_star = np.sqrt(np.log(n * (m + 1)) / n)
        else:
            w_minus1 = np.real(lambertw(arg, k=-1))
            epsilon_star = np.sqrt(-w_minus1 / (8 * n))

        # The bound is: β ≤ ε* + 1/(4nε*)
        beta_hat = epsilon_star + 1.0 / (4 * n * epsilon_star)

        return beta_hat

    def estimate_beta_smooth(self,
                            loss_matrix: np.ndarray,
                            theta_grid: np.ndarray,
                            delta_min: float = 1e-4,
                            delta_max: float = 1e-1) -> float:
        """
        Estimate β using Lipschitz-slope bound for smooth losses (Section 2.3.3).

        Adapted from generic.py for ERM. Uses Lipschitz constant L and
        local slope m to estimate:
        β ≤ L / (m(n+1))

        where:
        - L is the Lipschitz constant of the loss w.r.t. θ
        - m is the local slope of the empirical risk near the optimum

        Parameters
        ----------
        loss_matrix : np.ndarray
            n x m matrix of losses
        theta_grid : np.ndarray
            Grid of theta values
        delta_min : float
            Minimum delta for slope estimation
        delta_max : float
            Maximum delta for slope estimation

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(loss_matrix)
        beta_values = np.zeros(self.n_bootstrap)

        # Bootstrap
        for b in range(self.n_bootstrap):
            # Sample with replacement (n+1 samples)
            idx = np.random.choice(n, size=n+1, replace=True)
            loss_matrix_boot = loss_matrix[idx]

            # Compute Lipschitz constants for all pairs of theta values
            # For each data point, compute max |loss(θ1) - loss(θ2)| / |θ1 - θ2|
            if len(theta_grid.shape) == 1:
                # Scalar theta
                theta_diffs = np.abs(theta_grid[:, None] - theta_grid[None, :])
            else:
                # Vector theta - compute pairwise norms
                theta_diffs = np.linalg.norm(
                    theta_grid[:, None, :] - theta_grid[None, :, :],
                    axis=2
                )

            # Avoid division by zero
            theta_diffs[theta_diffs == 0] = np.inf

            L_hat = 0.0
            for i in range(n+1):
                loss_diffs = np.abs(loss_matrix_boot[i, :, None] - loss_matrix_boot[i, None, :])
                L_estimates_i = loss_diffs / theta_diffs
                L_hat = max(L_hat, np.max(L_estimates_i[np.isfinite(L_estimates_i)]))

            # Fit ERM on bootstrap data
            erm = DiscretizedRegularizedERM(lam=self.lam)
            theta_hat_boot = erm.fit(loss_matrix_boot, theta_grid)
            idx_hat = erm.theta_hat_idx

            # Estimate local slope m near theta_hat
            # m = inf_{δ ∈ [δ_min, δ_max]} |R(θ̂+δ) - R(θ̂)| / δ
            delta_range = np.linspace(delta_min, delta_max, 20)

            # Compute empirical risk at theta_hat
            risk_hat = np.mean(loss_matrix_boot[:, idx_hat])

            # Vectorize slope computation
            if len(theta_grid.shape) == 1:
                # Scalar theta
                theta_perturbed = theta_hat_boot + delta_range
                valid_mask = (theta_perturbed >= theta_grid.min()) & (theta_perturbed <= theta_grid.max())

                if np.any(valid_mask):
                    risks_perturbed = np.zeros(valid_mask.sum())
                    for j, theta_p in enumerate(theta_perturbed[valid_mask]):
                        idx_p = np.argmin(np.abs(theta_grid - theta_p))
                        risks_perturbed[j] = np.mean(loss_matrix_boot[:, idx_p])
                    slopes = np.abs(risks_perturbed - risk_hat) / delta_range[valid_mask]
                    m_hat = max(np.min(slopes), 1e-6)
                else:
                    m_hat = 1e-6
            else:
                # Vector theta - perturb in random directions
                m_hat = 1e-6
                for delta in delta_range:
                    # Random unit direction
                    direction = np.random.randn(theta_grid.shape[1])
                    direction = direction / np.linalg.norm(direction)
                    theta_p = theta_hat_boot + delta * direction

                    # Find nearest in grid
                    dists = np.linalg.norm(theta_grid - theta_p[None, :], axis=1)
                    idx_p = np.argmin(dists)

                    risk_p = np.mean(loss_matrix_boot[:, idx_p])
                    slope = np.abs(risk_p - risk_hat) / delta
                    m_hat = max(m_hat, slope)

            # Compute beta estimate for this bootstrap sample
            beta_values[b] = L_hat / (m_hat * (n + 1))

        # Average across bootstrap samples
        beta_hat = np.mean(beta_values)

        return beta_hat

    def estimate_all(self,
                    loss_matrix: np.ndarray,
                    theta_grid: np.ndarray,
                    delta_min: float = 1e-4,
                    delta_max: float = 1e-1) -> dict:
        """
        Estimate β using all three methods and return as a dictionary.

        Parameters
        ----------
        loss_matrix : np.ndarray
            n x m matrix of losses
        theta_grid : np.ndarray
            Grid of theta values
        delta_min : float
            Min delta for smooth method
        delta_max : float
            Max delta for smooth method

        Returns
        -------
        estimates : dict
            Dictionary with keys 'definition', 'discretized', 'smooth'
        """
        n, m = loss_matrix.shape

        beta_def = self.estimate_beta_def(loss_matrix, theta_grid)
        beta_disc = self.estimate_beta_discretized(n, m)
        beta_smooth = self.estimate_beta_smooth(loss_matrix, theta_grid,
                                                delta_min, delta_max)

        return {
            'definition': beta_def,
            'discretized': beta_disc,
            'smooth': beta_smooth
        }
