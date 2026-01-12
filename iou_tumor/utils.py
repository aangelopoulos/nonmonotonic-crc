import numpy as np
import hashlib
import time
from pathlib import Path
import sys
sys.path.insert(0, '..')
from discretized_erm import DiscretizedRegularizedERM, DiscretizedERMStabilityEstimator

# Cache directory
CACHE_DIR = Path('.cache')


def iou_loss(X_scores, Y_masks, theta):
    """
    Compute negative IOU loss for tumor segmentation.

    IOU = TP / (TP + FP + FN) = intersection / union
    Loss = 1 - IOU (so lower is better)

    Parameters
    ----------
    X_scores : np.ndarray
        Predicted scores (n_images, height, width)
    Y_masks : np.ndarray
        Ground truth masks (n_images, height, width)
    theta : float
        Threshold for converting scores to binary predictions

    Returns
    -------
    loss : np.ndarray
        1 - IOU for each image
    """
    pred_masks = (X_scores >= theta).astype(float)

    # Compute intersection and union
    intersection = (pred_masks * Y_masks).sum(axis=-1).sum(axis=-1)
    union = ((pred_masks + Y_masks) > 0).astype(float).sum(axis=-1).sum(axis=-1)

    # IOU, handling division by zero
    iou = np.where(union > 0, intersection / union, 0.0)

    # Return negative IOU as loss (1 - IOU)
    return 1.0 - iou


def get_cache_key(data_shape, theta_grid):
    """Generate cache key based on data shape and theta grid."""
    key_str = f"{data_shape}_{len(theta_grid)}_{theta_grid[0]:.6f}_{theta_grid[-1]:.6f}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_iou_cache(cache_key):
    """Load cached IOU matrix if it exists."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"iou_matrix_{cache_key}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return data['iou_matrix'], data['theta_grid']
    return None, None


def save_iou_cache(iou_matrix, theta_grid, cache_key):
    """Save IOU matrix to cache."""
    cache_file = CACHE_DIR / f"iou_matrix_{cache_key}.npz"
    np.savez_compressed(cache_file, iou_matrix=iou_matrix, theta_grid=theta_grid)


def compute_or_load_iou_matrix(X_scores, Y_masks, theta_grid, verbose=True):
    """Compute IOU loss matrix for all images and thetas, with caching."""
    cache_key = get_cache_key(X_scores.shape, theta_grid)

    # Try to load from cache
    iou_matrix, cached_theta_grid = load_iou_cache(cache_key)

    if iou_matrix is not None and np.allclose(cached_theta_grid, theta_grid):
        if verbose:
            print(f"Loaded IOU matrix from cache: {X_scores.shape[0]} images x {len(theta_grid)} thetas")
        return iou_matrix

    # Compute IOU matrix
    if verbose:
        print(f"Computing IOU matrix: {X_scores.shape[0]} images x {len(theta_grid)} thetas...")
        start_time = time.time()

    n, m = len(X_scores), len(theta_grid)
    iou_matrix = np.zeros((n, m))

    for j, theta in enumerate(theta_grid):
        iou_matrix[:, j] = iou_loss(X_scores, Y_masks, theta)

    # Save to cache
    save_iou_cache(iou_matrix, theta_grid, cache_key)

    if verbose:
        elapsed = time.time() - start_time
        print(f"Computed and cached in {elapsed:.2f}s")

    return iou_matrix


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


def apply_erm_iou_control(iou_matrix_cal, iou_matrix_val, prediction_rate_val,
                          theta_grid, lam=0.01, verbose=False):
    """
    Apply discretized regularized ERM for IOU control.

    Parameters
    ----------
    iou_matrix_cal : np.ndarray
        IOU loss matrix for calibration data (n x m)
    iou_matrix_val : np.ndarray
        IOU loss matrix for validation data (n_val x m)
    prediction_rate_val : np.ndarray
        Prediction rate matrix for validation data (n_val x m)
    theta_grid : np.ndarray
        Grid of theta values (m,)
    lam : float
        Regularization parameter
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Dictionary with theta_hat, empirical_iou_loss, prediction_rate
    """
    # Fit discretized ERM
    erm = DiscretizedRegularizedERM(lam=lam)
    theta_hat = erm.fit(iou_matrix_cal, theta_grid)
    idx_hat = erm.theta_hat_idx

    # Evaluate on validation set
    empirical_iou_loss = iou_matrix_val.mean(axis=0)[idx_hat]
    prediction_rate = prediction_rate_val.mean(axis=0)[idx_hat]

    if verbose:
        print(f"ERM: θ̂={theta_hat:.4f}, IOU Loss={empirical_iou_loss:.4f}, pred_rate={prediction_rate:.4f}")

    return {
        'theta_hat': theta_hat,
        'empirical_iou_loss': empirical_iou_loss,
        'empirical_iou': 1.0 - empirical_iou_loss,
        'prediction_rate': prediction_rate
    }


def estimate_beta_erm(iou_matrix_cal, theta_grid, lam=0.01, n_bootstrap=100,
                      method='definition'):
    """
    Estimate stability parameter beta for discretized ERM.

    Parameters
    ----------
    iou_matrix_cal : np.ndarray
        IOU loss matrix for calibration data (n x m)
    theta_grid : np.ndarray
        Grid of theta values (m,)
    lam : float
        Regularization parameter
    n_bootstrap : int
        Number of bootstrap samples
    method : str
        Estimation method: 'definition', 'discretized', 'smooth', or 'loss'

    Returns
    -------
    beta_hat : float
        Estimated stability parameter
    """
    estimator = DiscretizedERMStabilityEstimator(lam=lam, n_bootstrap=n_bootstrap)

    if method == 'definition':
        beta_hat = estimator.estimate_beta_def(iou_matrix_cal, theta_grid)
    elif method == 'discretized':
        n, m = iou_matrix_cal.shape
        beta_hat = estimator.estimate_beta_discretized(n, m)
    elif method == 'smooth':
        beta_hat = estimator.estimate_beta_smooth(iou_matrix_cal, theta_grid)
    elif method == 'loss':
        beta_hat = estimator.estimate_beta_loss(iou_matrix_cal, theta_grid)
    else:
        raise ValueError(f"Invalid method: {method}")

    return beta_hat
