# Fix #005: `is_image` metadata never set in generate_dataset()

**Date**: 2026-05-12
**Severity**: Critical — Conv-AE/SelfLabel image pipeline is dead code in Web UI
**Status**: ✅ Fixed

## Error

DimensionExpert's Conv-AE + SelfLabel(Conv) pipelines never activate for ANY image dataset loaded through the Web UI. CIFAR-10 raw/gap/resnet and MNIST all default to MLP-AE (tabular) pipelines.

## Root Cause

`data_factory.py` `generate_dataset()` never sets `is_image` or `original_shape` in DatasetBundle metadata. The only place that sets `is_image: True` is `benchmark/dataloader.py:111`, which is used by `mnist_full` and `fashion_mnist` but NOT by the web demo's `generate_dataset()` path.

DimensionExpert line 679:
```python
is_image = bool(dataset.metadata.get("is_image")) if dataset.metadata else False
# → Always False for ALL web UI datasets
```

Consequences:
- `scaler_class = "StandardScaler"` instead of `"MinMaxScaler"` (pixel values are 0-255)
- Conv-AE not generated → MLP-AE used instead (worse for image data)
- SelfLabel(Conv) not generated → no teacher-student distillation for images
- Res-AE activated for image data (wrong — it's designed for tabular GAP features)

## Affected Datasets

| Dataset | n_features | Should be is_image | Currently is_image |
|---|---|---|---|
| cifar10_raw | 3072 | True (32×32×3) | False |
| cifar10_gap | 64 | False (semantic vectors) | False ✅ |
| cifar10_resnet | 512 | False (semantic vectors) | False ✅ |
| mnist | 784 | True (28×28) | False |
| fashion_mnist | 784 | True (28×28) | False |
| coil20 | 1024 | True (32×32) | False |

## Fix Plan

Add `is_image` and `original_shape` metadata to:
- `_load_cifar10()` — only for `feature_mode="raw"`
- MNIST dataset loader in `generate_dataset()`
- `_load_coil20()`

For GAP and ResNet modes, `is_image` should stay False (they are semantic feature vectors, not raw images).
