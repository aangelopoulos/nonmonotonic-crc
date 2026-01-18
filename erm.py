"""
Regularized Empirical Risk Minimization
========================================

Implements regularized ERM and stability estimation from Section 2.3.

This module handles:
- Regularized ERM: A(D) = argmin_θ R̂_D(θ) + (λ/2)||θ||²_2
- Stability estimation for loss-based bounds (Proposition: erm-stable-loss)
- Stability estimation for gradient-based bounds (Proposition: erm-grad-stability)
- Direct stability estimation from definition (with warm-start optimization)

All implementations use efficient numpy vectorized operations and warm starts where possible.

References
----------
Section 2.3 - Guarantees for empirical risk minimization
"""

import numpy as np
from typing import Callable, Optional, Union
from scipy.optimize import minimize


class RegularizedERM:
    """
    Regularized empirical risk minimization.

    Solves: A(D) = argmin_{θ ∈ R^d} R̂_D(θ) + (λ/2)||θ||²_2

    where R̂_D(θ) = (1/|D|) Σ ℓ(z; θ) is the empirical risk.

    References
    ----------
    Section 2.3, Equation after line 416
    """

    def __init__(self,
                 lam: float = 0.01,
                 theta_init: Optional[np.ndarray] = None,
                 method: str = 'L-BFGS-B'):
        """
        Initialize regularized ERM.

        Parameters
        ----------
        lam : float
            Regularization parameter λ
        theta_init : np.ndarray, optional
            Initial parameter value for optimization
        method : str
            Optimization method for scipy.optimize.minimize
        """
        self.lam = lam
        self.theta_init = theta_init
        self.method = method
        self.theta_hat = None

    def _objective(self,
                   theta: np.ndarray,
                   loss_fn: Callable,
                   data: np.ndarray) -> float:
        """
        Compute regularized empirical risk.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
        loss_fn : callable
            Loss function ℓ(z; θ)
        data : np.ndarray
            Dataset (n x d)

        Returns
        -------
        objective : float
            R̂_D(θ) + (λ/2)||θ||²_2
        """
        empirical_risk = loss_fn(data, theta).mean()
        regularization = 0.5 * self.lam * np.sum(theta ** 2)
        return empirical_risk + regularization

    def _gradient(self,
                  theta: np.ndarray,
                  grad_fn: Callable,
                  data: np.ndarray) -> np.ndarray:
        """
        Compute gradient of regularized empirical risk.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector
        grad_fn : callable
            Gradient function ∇ℓ(z; θ)
        data : np.ndarray
            Dataset

        Returns
        -------
        gradient : np.ndarray
            ∇R̂_D(θ) + λθ
        """
        empirical_grad = grad_fn(data, theta).mean(axis=0)
        return empirical_grad + self.lam * theta

    def fit(self,
            data: np.ndarray,
            loss_fn: Callable,
            grad_fn: Callable, 
            theta_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit regularized ERM by minimizing the objective.

        Parameters
        ----------
        data : np.ndarray
            Training dataset
        loss_fn : callable
            Loss function ℓ(z; θ)
        grad_fn : callable
            Gradient function ∇ℓ(z; θ).
        theta_init : np.ndarray
            Initial parameter value for optimization

        Returns
        -------
        theta_hat : np.ndarray
            Optimal parameter
        """
        # Initialize theta if not provided
        if self.theta_init is None:
            self.theta_init = theta_init
        if self.theta_init is None:
            raise ValueError("theta_init must be provided in fit() or __init__()") # Throw error

        # Optimize
        result = minimize(
            fun=lambda theta: self._objective(theta, loss_fn, data),
            x0=self.theta_init,
            jac=lambda theta: self._gradient(theta, grad_fn, data),
            method=self.method
        )

        self.theta_hat = result.x
        return self.theta_hat

class ERMStabilityEstimator:
    """
    Estimate stability parameter β for regularized ERM.

    Implements four methods from Section 2.3:
    1. Loss-based stability (Proposition: erm-stable-loss, line 674-682)
    2. Gradient-based stability (Proposition: erm-grad-stability, line 684-712)
    3. Direct loss-based stability (from definition, with warm starts)
    4. Direct gradient-based stability (from definition, with warm starts)

    The direct methods estimate β by repeatedly solving ERM with leave-one-out
    datasets and computing the stability directly from the definition. They use
    warm starts (initializing at the full-data solution) for efficiency.

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
                          data: np.ndarray,
                          loss_fn: Callable,
                          theta_grid: np.ndarray) -> float:
        """
        Estimate β using loss-based bound (line 674-682).

        For Proposition erm-stable-loss:
        β̂_ERMLoss = 2ρ̂²/(λ(n+1))

        where ρ̂² = (1/n) Σ_i max_{θ,θ' ∈ Θ_G} |ℓ(Z_i;θ)-ℓ(Z_i;θ')|/||θ-θ'||

        Parameters
        ----------
        data : np.ndarray
            Calibration dataset
        loss_fn : callable
            Loss function ℓ(z; θ)
        theta_grid : np.ndarray
            Grid of theta values for Lipschitz estimation (shape: m x d)

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(data)
        rho_squared_sum = 0.0

        # For each data point, estimate its Lipschitz constant
        for i in range(n):
            z_i = data[i]
            max_lipschitz_sq = 0.0

            # Compute Lipschitz constant over grid
            for j in range(len(theta_grid)):
                for k in range(j + 1, len(theta_grid)):
                    theta_j = theta_grid[j]
                    theta_k = theta_grid[k]

                    loss_diff = abs(loss_fn(z_i, theta_j) - loss_fn(z_i, theta_k))
                    theta_diff = np.linalg.norm(theta_j - theta_k)

                    if theta_diff > 0:
                        lipschitz_est = loss_diff / theta_diff
                        max_lipschitz_sq = max(max_lipschitz_sq, lipschitz_est ** 2)

            rho_squared_sum += max_lipschitz_sq

        rho_squared_hat = rho_squared_sum / n
        beta_hat = 2 * rho_squared_hat / (self.lam * (n + 1))

        return beta_hat

    def estimate_beta_grad(self,
                          data: np.ndarray,
                          loss_fn: Callable,
                          grad_fn: Callable,
                          theta_grid: np.ndarray,
                          mu: float = 0.0) -> float:
        """
        Estimate β using gradient-based bound (line 684-712).

        For Proposition erm-grad-stability:
        β̂_ERMGrad = min{1, 2/(λ(n+1))(Ḡ_1 + Ḡ_2)}

        where:
        - G^(b)_1 = (1/(n+1)) Σ_i ρ̂^(b)_i ||∇ℓ(Z^(b)_i; θ̂^(b)_{-i})||_2
        - G^(b)_2 = (1/(n+1)) Σ_i ||(1/n) Σ_{j≠i} ∇ℓ(Z^(b)_j; θ̂^(b)_{-i})||_2
        - Ḡ_k = (1/B) Σ_b G^(b)_k

        Parameters
        ----------
        data : np.ndarray
            Calibration dataset
        loss_fn : callable
            Loss function ℓ(z; θ)
        grad_fn : callable
            Gradient function ∇ℓ(z; θ)
        theta_grid : np.ndarray
            Grid of theta values for Lipschitz estimation
        mu : float
            Strong convexity parameter (default 0.0)

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(data)
        G1_values = np.zeros(self.n_bootstrap)
        G2_values = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, size=n + 1, replace=True)
            data_boot = data[idx]

            G1_sum = 0.0
            G2_sum = 0.0

            # For each i, fit leave-one-out and compute terms
            for i in range(n + 1):
                # Leave-one-out dataset
                mask = np.ones(n + 1, dtype=bool)
                mask[i] = False
                data_loo = data_boot[mask]

                # Fit ERM on leave-one-out data
                erm_loo = RegularizedERM(lam=self.lam)
                theta_loo = erm_loo.fit(data_loo, loss_fn, grad_fn)

                # Estimate Lipschitz constant for data point i
                z_i = data_boot[i]
                rho_i = self._estimate_lipschitz(z_i, loss_fn, theta_grid)

                # Compute G1 term: ρ_i ||∇ℓ(z_i; θ̂_{-i})||_2
                grad_i = grad_fn(z_i, theta_loo)
                G1_sum += rho_i * np.linalg.norm(grad_i)

                # Compute G2 term: ||(1/n) Σ_{j≠i} ∇ℓ(z_j; θ̂_{-i})||_2
                grad_sum = np.zeros_like(theta_loo)
                for j in range(n + 1):
                    if j != i:
                        grad_sum += grad_fn(data_boot[j], theta_loo)
                grad_avg = grad_sum / n
                G2_sum += np.linalg.norm(grad_avg)

            G1_values[b] = G1_sum / (n + 1)
            G2_values[b] = G2_sum / (n + 1)

        # Average across bootstrap samples
        G1_bar = np.mean(G1_values)
        G2_bar = np.mean(G2_values)

        # Compute beta
        beta_hat = min(1.0, 2 * (G1_bar + G2_bar) / ((mu + self.lam) * (n + 1)))

        return beta_hat

    def _estimate_lipschitz(self,
                           z: Union[np.ndarray, tuple],
                           loss_fn: Callable,
                           theta_grid: np.ndarray) -> float:
        """
        Estimate Lipschitz constant for a single data point.

        ρ(z) = max_{θ,θ' ∈ Θ_G} |ℓ(z;θ)-ℓ(z;θ')|/||θ-θ'||

        Parameters
        ----------
        z : np.ndarray or tuple
            Data point
        loss_fn : callable
            Loss function
        theta_grid : np.ndarray
            Grid of theta values

        Returns
        -------
        rho : float
            Estimated Lipschitz constant
        """
        max_lipschitz = 0.0

        for j in range(len(theta_grid)):
            for k in range(j + 1, len(theta_grid)):
                theta_j = theta_grid[j]
                theta_k = theta_grid[k]

                loss_diff = abs(loss_fn(z, theta_j) - loss_fn(z, theta_k))
                theta_diff = np.linalg.norm(theta_j - theta_k)

                if theta_diff > 0:
                    lipschitz_est = loss_diff / theta_diff
                    max_lipschitz = max(max_lipschitz, lipschitz_est)

        return max_lipschitz

    def estimate_beta_loss_direct(self,
                                  data: np.ndarray,
                                  loss_fn: Callable,
                                  grad_fn: Callable,
                                  theta_init: np.ndarray) -> float:
        """
        Estimate β directly from definition using loss scale.

        Computes:
        Δ = (1/(n+1)) Σ_i [ℓ(Z_i; A(D_{-i})) - ℓ(Z_i; A*(D_{1:n+1}))]
        β̂ = E[(Δ)_+]

        Uses bootstrap replicates and warm starts for efficiency.

        Parameters
        ----------
        data : np.ndarray
            Calibration dataset
        loss_fn : callable
            Loss function ℓ(z; θ)
        grad_fn : callable
            Gradient function for erm
        theta_init : np.ndarray
            Initial parameter value for optimization

        Returns
        -------
        beta_hat : float
            Estimated stability parameter
        """
        n = len(data)
        Delta_values = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, size=n + 1, replace=True)
            data_boot = data[idx]

            # Fit A* on full bootstrap data
            erm_star = RegularizedERM(lam=self.lam, theta_init=theta_init)
            theta_star = erm_star.fit(data_boot, loss_fn, grad_fn)

            # Compute leave-one-out losses
            loss_diffs = np.zeros(n + 1)

            for i in range(n + 1):
                # Leave-one-out dataset
                mask = np.ones(n + 1, dtype=bool)
                mask[i] = False
                data_loo = data_boot[mask]

                # Fit A on leave-one-out data with warm start
                erm_loo = RegularizedERM(lam=self.lam, theta_init=theta_star)
                theta_loo = erm_loo.fit(data_loo, loss_fn, grad_fn)

                # Compute loss difference
                z_i = data_boot[i:i+1, :]
                loss_loo = loss_fn(z_i, theta_loo)
                loss_star = loss_fn(z_i, theta_star)

                loss_diffs[i] = loss_loo - loss_star

            Delta_values[b] = np.mean(loss_diffs)

        # Take positive part of average
        Delta_bar = np.mean(Delta_values)
        beta_hat = max(0.0, Delta_bar)

        return beta_hat

    def estimate_beta_grad_direct(self,
                                  data: np.ndarray,
                                  loss_fn: Callable,
                                  grad_fn: Callable,
                                  theta_init: np.ndarray) -> np.ndarray:
        """
        Estimate β directly from definition using gradient scale.

        Computes:
        Δ = (1/(n+1)) Σ_i [∇ℓ(Z_i; A(D_{-i})) - ∇ℓ(Z_i; A*(D_{1:n+1}))]
        β̂ = E[(Δ)_+]

        Uses bootstrap replicates and warm starts for efficiency.

        Parameters
        ----------
        data : np.ndarray
            Calibration dataset
        loss_fn : callable
            Loss function ℓ(z; θ)
        grad_fn : callable
            Gradient function ∇ℓ(z; θ)
        theta_init : np.ndarray
            Initial parameter value for optimization

        Returns
        -------
        beta_hat : np.ndarray
            Estimated stability parameter (vector)
        """
        n = len(data)
        d = len(theta_init)

        Delta_values = np.zeros((self.n_bootstrap, d))

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, size=n + 1, replace=True)
            data_boot = data[idx]

            # Fit A* on full bootstrap data
            erm_star = RegularizedERM(lam=self.lam, theta_init=theta_init)
            theta_star = erm_star.fit(data_boot, loss_fn, grad_fn)

            # Compute leave-one-out gradient differences
            grad_diffs = np.zeros((n + 1, d))

            for i in range(n + 1):
                # Leave-one-out dataset
                mask = np.ones(n + 1, dtype=bool)
                mask[i] = False
                data_loo = data_boot[mask]

                # Fit A on leave-one-out data with warm start
                erm_loo = RegularizedERM(lam=self.lam, theta_init=theta_star) # Use theta_star as warm start
                theta_loo = erm_loo.fit(data_loo, loss_fn, grad_fn)

                # Compute gradient difference
                z_i = data_boot[i]
                grad_loo = grad_fn(z_i, theta_loo)
                grad_star = grad_fn(z_i, theta_star)

                grad_diffs[i] = grad_loo - grad_star

            Delta_values[b] = np.mean(grad_diffs, axis=0)

        # Take positive part of average (element-wise)
        Delta_bar = np.mean(Delta_values, axis=0)
        beta_hat = np.maximum(0.0, Delta_bar)

        return beta_hat

    def estimate_all(self,
                    data: np.ndarray,
                    loss_fn: Callable,
                    grad_fn: Callable,
                    theta_grid: np.ndarray,
                    mu: float = 0.0,
                    include_direct: bool = False) -> dict:
        """
        Estimate β using all available methods.

        Parameters
        ----------
        data : np.ndarray
            Calibration dataset
        loss_fn : callable
            Loss function
        grad_fn : callable
            Gradient function
        theta_grid : np.ndarray
            Grid of theta values
        mu : float
            Strong convexity parameter
        include_direct : bool
            If True, also compute direct estimators (slower but more accurate)

        Returns
        -------
        estimates : dict
            Dictionary with keys 'loss', 'gradient', and optionally
            'loss_direct' and 'gradient_direct'
        """
        beta_loss = self.estimate_beta_loss(data, loss_fn, theta_grid)
        beta_grad = self.estimate_beta_grad(data, loss_fn, grad_fn, theta_grid, mu)

        results = {
            'loss': beta_loss,
            'gradient': beta_grad
        }

        if include_direct:
            beta_loss_direct = self.estimate_beta_loss_direct(data, loss_fn, grad_fn)
            beta_grad_direct = self.estimate_beta_grad_direct(data, loss_fn, grad_fn)
            results['loss_direct'] = beta_loss_direct
            results['gradient_direct'] = beta_grad_direct

        return results
