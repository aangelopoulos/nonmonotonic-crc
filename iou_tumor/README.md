# IOU-based Tumor Segmentation with Discretized Regularized ERM

This folder contains an implementation of tumor segmentation using:
- **Loss**: IOU loss (1 - IOU, where IOU = Intersection over Union)
- **Algorithm**: Discretized Regularized Empirical Risk Minimization (ERM)
- **Dataset**: Same polyp segmentation data as `fdr_tumor`
- **Implementation**: Uses precomputed loss matrices like `generic.py`

## Files

- `tumor-segmentation-iou.ipynb`: Main notebook with step-by-step experiments
- `utils.py`: Utility functions for IOU computation, caching, and discretized ERM application
- `.cache/`: Cached IOU matrices for fast reloading
- `plots/`: Generated figures

## Key Differences from FDR Control

| Aspect | FDR Control | IOU-based ERM |
|--------|-------------|---------------|
| Loss | FDR (1 - precision) | 1 - IOU |
| Algorithm | Conformal Risk Control | Discretized Regularized ERM |
| Parameter | Threshold θ | Threshold θ |
| Optimization | Find smallest θ satisfying constraint | Minimize regularized empirical risk |
| Implementation | Uses `generic.py` | Uses `discretized_erm.py` |
| Input | Precomputed loss matrix (n × m) | Precomputed loss matrix (n × m) |

## Usage

Open the notebook in Jupyter and run cells sequentially:

```bash
cd iou_tumor
jupyter notebook tumor-segmentation-iou.ipynb
```

## Notebook Structure

1. **Setup**: Load dataset and imports
2. **Precomputation**: Compute/cache IOU matrix for all thresholds
3. **Single Split**: Run discretized ERM on one calibration/validation split
4. **Stability**: Estimate β parameter using bootstrap
5. **Multi-replicate**: Assess performance across many random splits
6. **Visualization**: Plot IOU curves, distributions, and example segmentations

## Parameters

- `n`: Calibration set size (default: 500)
- `lam`: Regularization parameter λ (default: 0.01)
- `n_replicates`: Number of random splits (default: 100)
- `n_bootstrap_beta`: Bootstrap samples for β estimation (default: 100)
- `theta_grid`: Discrete grid of theta values (default: 1000 points)
- `DEBUG_MODE`: Set to `True` for faster debugging with fewer samples

## Implementation Details

The discretized ERM algorithm minimizes over a discrete grid:

$$\hat{\theta} = \arg\min_{\theta \in \Theta_{grid}} \frac{1}{n}\sum_{i=1}^n (1 - \text{IOU}(x_i, y_i; \theta)) + \frac{\lambda}{2}\theta^2$$

where:
- IOU(x, y; θ) = TP / (TP + FP + FN) with predictions = (scores ≥ θ)
- λ is the regularization parameter
- Θ_grid is a discrete set of candidate thresholds

### Key Advantages

1. **Speed**: Operates on precomputed loss matrices - no need to recompute losses
2. **Efficiency**: All theta values are evaluated once upfront, then cached
3. **Simplicity**: Simple argmin over discrete grid, no continuous optimization needed
4. **Consistency**: Same API as `generic.py` for conformal risk control

### Differences from Continuous ERM

Unlike `erm.py` which uses continuous optimization:
- **No scipy.optimize**: Simple discrete search instead
- **No gradients**: Works directly with loss matrix
- **Faster**: No iterative optimization, just one pass through the grid
- **Identical to generic.py API**: Same function signatures

## Stability Estimation

Four methods available:
1. **definition**: Direct bootstrap-based estimation (default)
2. **discretized**: Distribution-free bound using Lambert W function
3. **smooth**: Lipschitz-slope bound for smooth losses
4. **loss**: Loss-based bound using Lipschitz constants

## Caching

IOU matrices are cached in `.cache/` for efficiency. Delete the cache folder to force recomputation.

## Example Usage

```python
from utils import compute_or_load_iou_matrix, apply_erm_iou_control, estimate_beta_erm

# Precompute IOU matrix
theta_grid = np.linspace(0, 1, 1000)
iou_matrix = compute_or_load_iou_matrix(X_scores, Y_masks, theta_grid)

# Split data
iou_matrix_cal = iou_matrix[cal_idx, :]
iou_matrix_val = iou_matrix[val_idx, :]

# Apply discretized ERM
results = apply_erm_iou_control(
    iou_matrix_cal,
    iou_matrix_val,
    prediction_rate_val,
    theta_grid,
    lam=0.01
)

# Estimate stability
beta = estimate_beta_erm(iou_matrix_cal, theta_grid, lam=0.01, method='definition')
```
