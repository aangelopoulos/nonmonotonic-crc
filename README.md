# Conformal Risk Control for Non-Monotonic Losses

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2602.20151-b31b1b.svg)]([https://arxiv.org/](https://arxiv.org/abs/2602.20151))

Code for conformal risk control with non-monotonic loss functions.

---

## Core Methods

| Module | Description |
|--------|-------------|
| `generic.py` | Generic conformal risk control for 1-dimensional problems |
| `selective.py` | Selective classification with conformal guarantees |
| `erm.py` | Regularized empirical risk minimization with guarantees on the risk or the gradient |
| `discretized_erm.py` | Discretized ERM for 1-dimensional problems |

---

## Experiments

Each experiment folder contains a Jupyter notebook demonstrating the method on real data.

```
selective_imagenet/     Selective classification on ImageNet
fdr_tumor/              FDR control for tumor segmentations
iou_tumor/              IOU control for tumor segmentations
compas_multigroup_debias/   Multigroup debiasing of COMPAS recidivism predictions
```
