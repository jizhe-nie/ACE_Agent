# Fix #003: CIFAR-10 resnet feature mode mismatch

**Date**: 2026-05-12
**Severity**: Medium — prevents loading cifar10_resnet dataset
**Status**: ✅ Fixed

## Error

```
ValueError: Unknown CIFAR-10 feature_mode: resnet
```

## Root Cause

`data_factory.py:314` does `mode = dataset_name.replace("cifar10_", "")` which produces `"resnet"` from `"cifar10_resnet"`. But `_load_cifar10()` internally only accepts `"resnet18"` as the valid mode name (line 733).

## Fix

Added mapping in `generate_dataset()`:
```python
if mode == "resnet":
    mode = "resnet18"
```

## Prevention

When adding new dataset name aliases, verify the internal mode strings match across all layers. The dataset name prefix (`cifar10_`) and internal mode name (`resnet18`) use different naming conventions — this inconsistency caused the bug.
