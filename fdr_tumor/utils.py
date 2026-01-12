import numpy as np
import hashlib
import time
from pathlib import Path
from scipy.stats import norm
import sys
sys.path.insert(0, '..')
from generic import GenericConformalRiskControl, GenericStabilityEstimator

# Cache directory
CACHE_DIR = Path('.cache')

def fdr_loss(X_scores, Y_masks, theta):
    """Compute FDR loss (1 - precision) for tumor segmentation."""
    pred_masks = (X_scores >= theta).astype(float)
    tp = (pred_masks * Y_masks).sum(axis=-1).sum(axis=-1)
    predicted_positives = pred_masks.sum(axis=-1).sum(axis=-1)
    precision = np.where(
        predicted_positives > 0, 
        tp / np.maximum(predicted_positives, 
        1.0), 
    1.0)
    return 1 - precision


def get_cache_key(data_shape, theta_grid):
    """Generate cache key based on data shape and theta grid."""
    key_str = f"{data_shape}_{len(theta_grid)}_{theta_grid[0]:.6f}_{theta_grid[-1]:.6f}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_fdr_cache(cache_key):
    """Load cached FDR matrix if it exists."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"fdr_matrix_{cache_key}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return data['fdr_matrix'], data['theta_grid']
    return None, None


def save_fdr_cache(fdr_matrix, theta_grid, cache_key):
    """Save FDR matrix to cache."""
    cache_file = CACHE_DIR / f"fdr_matrix_{cache_key}.npz"
    np.savez_compressed(cache_file, fdr_matrix=fdr_matrix, theta_grid=theta_grid)


def compute_or_load_fdr_matrix(X_scores, Y_masks, theta_grid, verbose=True):
    """Compute FDR matrix for all images and thetas, with caching."""
    cache_key = get_cache_key(X_scores.shape, theta_grid)
    
    # Try to load from cache
    fdr_matrix, cached_theta_grid = load_fdr_cache(cache_key)
    
    if fdr_matrix is not None and np.allclose(cached_theta_grid, theta_grid):
        if verbose:
            print(f"Loaded FDR matrix from cache: {X_scores.shape[0]} images x {len(theta_grid)} thetas")
        return fdr_matrix
    
    # Compute FDR matrix
    if verbose:
        print(f"Computing FDR matrix: {X_scores.shape[0]} images x {len(theta_grid)} thetas...")
        start_time = time.time()
    
    n, m = len(X_scores), len(theta_grid)
    fdr_matrix = np.zeros((n, m))
    
    for j, theta in enumerate(theta_grid):
        fdr_matrix[:, j] = fdr_loss(X_scores, Y_masks, theta)
    
    # Save to cache
    save_fdr_cache(fdr_matrix, theta_grid, cache_key)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Computed and cached in {elapsed:.2f}s")
    
    return fdr_matrix

def compute_or_load_prediction_rate(X_scores, theta_grid):
    """Compute prediction rate for all images and thetas, with caching."""
    cache_key = get_cache_key(X_scores.shape, theta_grid)
    cache_file = CACHE_DIR / f"prediction_rate_{cache_key}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return data['prediction_rate']
    theta_grid = np.asarray(theta_grid).reshape(1, -1)
    prediction_rate = (X_scores[:, :, :, None] >= theta_grid).mean(axis=1).mean(axis=1)
    np.savez_compressed(cache_file, prediction_rate=prediction_rate, theta_grid=theta_grid.ravel())
    return prediction_rate


def apply_ltt_fdr_control(fdr_matrix_cal, fdr_matrix_val, prediction_rate_val, alpha, delta, theta_grid, verbose=False):
    """Apply LTT (Learn Then Test) method using precomputed FDR matrix."""
    n = len(fdr_matrix_cal)
    z = norm.ppf(1 - delta)
    
    # Compute bounds from calibration FDR matrix
    emp_fdr = fdr_matrix_cal.mean(axis=0)
    sigma_hat = np.maximum(fdr_matrix_cal.std(axis=0), 1e-6)
    ub = np.minimum(emp_fdr + z * sigma_hat / np.sqrt(n), 1.0)
    
    # Select threshold
    feasible = ub <= alpha
    idx_hat = np.argmax(feasible) if np.any(feasible) else len(theta_grid) - 1
    theta_hat = theta_grid[idx_hat]
    
    # Apply to validation
    empirical_fdr = fdr_matrix_val.mean(axis=0)[idx_hat]
    prediction_rate = prediction_rate_val.mean(axis=0)[idx_hat]
    
    if verbose:
        print(f"LTT: θ̂={theta_hat:.4f}, FDR={empirical_fdr:.4f}, pred_rate={prediction_rate:.4f}")
    
    return {
        'theta_hat': theta_hat,
        'empirical_fdr': empirical_fdr,
        'prediction_rate': prediction_rate
    }


def apply_crc_fdr_control(fdr_matrix_cal, fdr_matrix_val, prediction_rate_val,
                          alpha, theta_grid, beta_hat=0.0, verbose=False):
    """Apply CRC method using precomputed FDR matrix."""
    alpha_adjusted = max(0.0, alpha - beta_hat)
    
    controller = GenericConformalRiskControl(alpha=alpha_adjusted, theta_grid=theta_grid)
    theta_hat = controller.fit(fdr_matrix_cal, theta_grid)
    idx_hat = np.argmin(np.abs(theta_grid - theta_hat))
    
    # Apply to validation
    empirical_fdr = fdr_matrix_val.mean(axis=0)[idx_hat]
    prediction_rate = prediction_rate_val.mean(axis=0)[idx_hat]
    if verbose:
        method = "CRC-C" if beta_hat > 0 else "CRC"
        print(f"{method}: thetahat={theta_hat:.4f}, FDR={empirical_fdr:.4f}, pred_rate={prediction_rate:.4f}")
    
    return {
        'theta_hat': theta_hat,
        'beta_hat': beta_hat,
        'empirical_fdr': empirical_fdr,
        'prediction_rate': prediction_rate,
        'controller': controller
    }


def estimate_beta(fdr_matrix, alpha, theta_grid, n_bootstrap, method="def"):
    """Estimate stability parameter beta."""

    estimator = GenericStabilityEstimator(alpha=alpha, n_bootstrap=n_bootstrap)
    if method == "def":
        beta_hat = estimator.estimate_beta_def(fdr_matrix, theta_grid)
    elif method == "disc":
        beta_hat = estimator.estimate_beta_discretized(len(fdr_matrix), len(theta_grid))
    elif method == "smooth":
        beta_hat = estimator.estimate_beta_smooth(fdr_matrix, theta_grid)
    else:
        raise ValueError(f"Invalid method: {method}")
        
    return beta_hat