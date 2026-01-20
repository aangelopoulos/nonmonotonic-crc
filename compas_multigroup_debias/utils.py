"""
Utility functions for COMPAS multigroup debiasing experiments.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import os

# Add parent directory to system path
notebook_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(notebook_dir)
sys.path.append(parent_dir)

from erm import RegularizedERM, ERMStabilityEstimator


class OLS:
    """
    Ordinary Least Squares (Ridge Regression) using RegularizedERM with optional conservative guarantee.

    Solves: min_θ (1/n) Σ (y_i - x_i^T θ)² + (λ/2)||θ||² - γ 1_d^T θ

    The linear term -γ 1_d^T θ provides a conservative gradient guarantee.

    Provides vectorized fit and predict methods.
    """

    def __init__(self, lam=0.01, gamma=0.0):
        """
        Initialize OLS model.

        Parameters
        ----------
        lam : float
            Regularization parameter (ridge penalty)
        gamma : float
            Linear term coefficient for conservative guarantee.
        """
        self.lam = lam
        self.gamma = gamma
        self.theta = None
        self.erm = None

    def _loss_fn(self, data, theta):
        """
        Squared loss: (y - x^T θ)²

        Parameters
        ----------
        data : tuple or array
            If tuple: (X, y) where X is (n, d) and y is (n,)
            If array: assumes structure is data[:, :-1] = X, data[:, -1] = y
        theta : np.ndarray
            Parameter vector (d,)

        Returns
        -------
        loss : np.ndarray
            Loss for each sample (n,)
        """
        if isinstance(data, tuple):
            X, y = data
        else:
            X = data[:, :-1]
            y = data[:, -1]

        predictions = X @ theta
        residuals = y - predictions
        return residuals ** 2

    def _grad_fn(self, data, theta):
        """
        Gradient of squared loss: -2(y - x^T θ) x

        Parameters
        ----------
        data : tuple or array
            If tuple: (X, y) where X is (n, d) and y is (n,)
            If array: assumes structure is data[:, :-1] = X, data[:, -1] = y
        theta : np.ndarray
            Parameter vector (d,)

        Returns
        -------
        gradient : np.ndarray
            Gradient for each sample (n, d)
        """
        if isinstance(data, tuple):
            X, y = data
        else:
            X = data[:, :-1]
            y = data[:, -1]

        predictions = X @ theta
        residuals = y - predictions
        return -2 * residuals[:, np.newaxis] * X

    def _loss_fn_conservative(self, data, theta):
        """
        Regularized empirical risk with linear term: R̂_D(θ) + (λ/2)||θ||²_2 + γ 1_d^T θ
        """
        return self._loss_fn(data, theta) + self.gamma * np.sum(theta)

    def _grad_fn_conservative(self, data, theta):
        """
        Gradient with linear term: ∇R̂_D(θ) + λθ - γ 1_d
        """
        return self._grad_fn(data, theta) + self.gamma * np.ones_like(theta)

    def fit(self, X, y):
        """
        Fit OLS model to training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n, d)
        y : np.ndarray
            Target vector (n,)

        Returns
        -------
        self : OLS
            Fitted model
        """
        # Store data as tuple for loss/grad functions
        data = (X, y)

        # Initialize theta
        d = X.shape[1]
        theta_init = np.zeros(d)

        # Use standard RegularizedERM
        self.erm = RegularizedERM(lam=self.lam, theta_init=theta_init)
        self.theta = self.erm.fit(data, self._loss_fn, self._grad_fn)

        return self

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n, d)

        Returns
        -------
        predictions : np.ndarray
            Predicted values (n,)
        """
        if self.theta is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return X @ self.theta


def precompute_beta_ols(X, y, lam=0.01, n_bootstrap=100):
    """
    Precompute stability parameter β for OLS using direct estimation.

    Uses ERMStabilityEstimator.estimate_beta_loss_direct() which implements:
    Δ^(b) = (1/(n+1)) Σ_i [ℓ(Z_i^(b); A(D_{-i}^(b))) - ℓ(Z_i^(b); A*(D^(b)))]
    β̂ = E[Δ]_+

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n, d)
    y : np.ndarray
        Target vector (n,)
    lam : float
        Regularization parameter
    n_bootstrap : int
        Number of bootstrap replicates

    Returns
    -------
    beta_hat : float
        Estimated stability parameter
    """
    # Create OLS instance to use its loss and gradient functions
    ols = OLS(lam=lam, gamma=0.0)

    # Prepare data as concatenated array for ERMStabilityEstimator
    data = np.concatenate([X, y[:, np.newaxis]], axis=1)

    # Create stability estimator
    estimator = ERMStabilityEstimator(lam=lam, n_bootstrap=n_bootstrap)

    # Estimate beta using direct method with OLS loss and gradient functions
    beta_hat = estimator.estimate_beta_loss_direct(
        data=data,
        loss_fn=ols._loss_fn,
        grad_fn=ols._grad_fn,
        theta_init=np.zeros(X.shape[1])
    )

    return beta_hat


def compute_group_bias(y_true, y_pred, group_indicators, group_names):
    """
    Compute bias for each group: E[Y | G=1] - E[pred | G=1]

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predictions (probabilities)
    group_indicators : np.ndarray
        Binary indicators for group membership (n, k)
    group_names : list
        Names of groups

    Returns
    -------
    biases : dict
        Dictionary mapping group names to bias values
    """
    biases = {}
    for i, group_name in enumerate(group_names):
        group_mask = group_indicators[:, i] == 1
        if group_mask.sum() > 0:
            bias = y_true[group_mask].mean() - y_pred[group_mask].mean()
            biases[group_name] = bias
    return biases


def run_train_test_splits(X, y, f, races, sexes, order_races, order_sexes, beta_hat,
                          n_splits=100, test_size=0.3, lam=0.01, verbose=True):
    """
    Run multiple train-test splits and collect per-group bias results.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n, d)
    y : np.ndarray
        True labels (n,)
    f : np.ndarray
        Original predictions (n,)
    races : np.ndarray
        Race indicators (n, n_races)
    sexes : np.ndarray
        Sex indicators (n, n_sexes)
    order_races : list
        Names of race groups
    order_sexes : list
        Names of sex groups
    beta_hat : float
        Stability parameter for conservative method
    n_splits : int
        Number of train-test splits
    test_size : float
        Fraction of data to use for testing
    lam : float
        Regularization parameter
    verbose : bool
        Whether to print progress

    Returns
    -------
    results : dict
        Dictionary with keys 'original', 'standard', 'conservative', each containing
        per-group bias values across all splits
    """
    all_groups = order_races + order_sexes
    results = {
        'original': {group: [] for group in all_groups},
        'standard': {group: [] for group in all_groups},
        'conservative': {group: [] for group in all_groups}
    }

    if verbose:
        print(f"Running {n_splits} train-test splits...")
        print(f"Beta used for conservative method: {beta_hat:.6f}")
        print("Progress: ", end="", flush=True)

    for split_idx in range(n_splits):
        if verbose and (split_idx + 1) % 10 == 0:
            print(f"{split_idx + 1}...", end="", flush=True)

        # Random train-test split
        n = len(y)
        indices = np.random.permutation(n)
        split_point = int(n * (1 - test_size))
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        f_train, f_test = f[train_idx], f[test_idx]
        races_test = races[test_idx]
        sexes_test = sexes[test_idx]

        # Compute bias on training set
        bias_train = f_train - y_train

        # Fit standard OLS
        ols_standard = OLS(lam=lam, gamma=0.0)
        ols_standard.fit(X_train, bias_train)
        bias_pred_standard = ols_standard.predict(X_test)
        f_debiased_standard = f_test - bias_pred_standard

        # Fit conservative OLS
        ols_conservative = OLS(lam=lam, gamma=beta_hat)
        ols_conservative.fit(X_train, bias_train)
        bias_pred_conservative = ols_conservative.predict(X_test)
        f_debiased_conservative = f_test - bias_pred_conservative

        # Compute biases for races
        race_bias_orig = compute_group_bias(y_test, f_test, races_test, order_races)
        race_bias_std = compute_group_bias(y_test, f_debiased_standard, races_test, order_races)
        race_bias_cons = compute_group_bias(y_test, f_debiased_conservative, races_test, order_races)

        # Compute biases for sexes
        sex_bias_orig = compute_group_bias(y_test, f_test, sexes_test, order_sexes)
        sex_bias_std = compute_group_bias(y_test, f_debiased_standard, sexes_test, order_sexes)
        sex_bias_cons = compute_group_bias(y_test, f_debiased_conservative, sexes_test, order_sexes)

        # Store results
        for group in order_races:
            results['original'][group].append(race_bias_orig[group])
            results['standard'][group].append(race_bias_std[group])
            results['conservative'][group].append(race_bias_cons[group])
        for group in order_sexes:
            results['original'][group].append(sex_bias_orig[group])
            results['standard'][group].append(sex_bias_std[group])
            results['conservative'][group].append(sex_bias_cons[group])

    if verbose:
        print(" Done!")

    return results


def results_to_dataframe(results, order_races, order_sexes):
    """
    Convert results dictionary to long-format DataFrame for plotting.

    Parameters
    ----------
    results : dict
        Results from run_train_test_splits
    order_races : list
        Names of race groups
    order_sexes : list
        Names of sex groups

    Returns
    -------
    df : pd.DataFrame
        Long-format DataFrame with columns: Group, Method, Bias, Category
    """
    all_groups = order_races + order_sexes
    plot_data = []

    for group in all_groups:
        for method in ['Original', 'Standard OLS', 'Conservative OLS']:
            method_key = {'Original': 'original',
                         'Standard OLS': 'standard',
                         'Conservative OLS': 'conservative'}[method]
            for bias_value in results[method_key][group]:
                plot_data.append({
                    'Group': group,
                    'Method': method,
                    'Bias': bias_value,
                    'Category': 'Race' if group in order_races else 'Sex'
                })

    return pd.DataFrame(plot_data)


def print_summary_statistics(results, order_races, order_sexes):
    """
    Print summary statistics for bias distributions.

    Parameters
    ----------
    results : dict
        Results from run_train_test_splits
    order_races : list
        Names of race groups
    order_sexes : list
        Names of sex groups
    """
    all_groups = order_races + order_sexes

    print("\n" + "=" * 100)
    print("Summary Statistics: Bias Distribution Across Splits")
    print("=" * 100)
    print(f"{'Group':<20} {'Method':<20} {'Mean':<12} {'Std':<12} {'|Mean|':<12}")
    print("-" * 100)

    for group in all_groups:
        mean_orig = np.mean(results['original'][group])
        std_orig = np.std(results['original'][group])
        mean_std = np.mean(results['standard'][group])
        std_std = np.std(results['standard'][group])
        mean_cons = np.mean(results['conservative'][group])
        std_cons = np.std(results['conservative'][group])

        print(f"{group:<20} {'Original':<20} {mean_orig:+.6f}   {std_orig:.6f}   {abs(mean_orig):.6f}")
        print(f"{'':<20} {'Standard OLS':<20} {mean_std:+.6f}   {std_std:.6f}   {abs(mean_std):.6f}")
        print(f"{'':<20} {'Conservative OLS':<20} {mean_cons:+.6f}   {std_cons:.6f}   {abs(mean_cons):.6f}")
        reduction_std = abs(mean_orig) - abs(mean_std)
        reduction_cons = abs(mean_orig) - abs(mean_cons)
        print(f"{'':<20} {'Reduction (Std)':<20} {reduction_std:+.6f}")
        print(f"{'':<20} {'Reduction (Cons)':<20} {reduction_cons:+.6f}")
        print("-" * 100)
