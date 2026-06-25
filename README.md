# Conformal Risk Control for Non-Monotonic Losses

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.20151-b31b1b.svg)](https://arxiv.org/abs/2602.20151)

Code for conformal risk control with non-monotonic loss functions.

## Install

```bash
pip install numpy scipy
```

## Quickstart

### Any loss (generic)

```python
from generic import GenericConformalRiskControl
import numpy as np

controller = GenericConformalRiskControl(alpha=0.1)
theta_hat = controller.fit(loss_matrix, theta_grid)
```

### Selective classification

```python
from selective import SelectiveClassifier

clf = SelectiveClassifier(alpha=0.1)
clf.fit(confidences, errors)
keep = clf.predict(test_confidences)
```

### Regularized ERM

```python
from erm import RegularizedERM

erm = RegularizedERM(lam=0.01)
theta = erm.fit(data, loss_fn, grad_fn)
```

## Modules

| File                  | Purpose                              |
|-----------------------|--------------------------------------|
| `generic.py`          | Generic risk control                 |
| `selective.py`        | Selective classification             |
| `erm.py`              | Regularized ERM                      |
| `discretized_erm.py`  | Grid-based ERM (fast for IOU etc.)   |

Each module also has stability estimators (beta).

## Examples

Jupyter notebooks with complete pipelines:

- `selective_imagenet/` — Selective classification on ImageNet
- `fdr_tumor/` — FDR control for tumor segmentation
- `iou_tumor/` — IOU control with discretized ERM
- `compas_multigroup_debias/` — Multigroup debiasing

## Stability

All modules include bootstrap and closed-form estimators for the stability parameter (beta) used to get valid guarantees.

## Citation

```bibtex
@article{angelopoulos2026nonmonotonic,
  title={Conformal Risk Control for Non-Monotonic Losses},
  author={Angelopoulos, Anastasios N.},
  journal={arXiv preprint arXiv:2602.20151},
  year={2026}
}
```

## License

MIT