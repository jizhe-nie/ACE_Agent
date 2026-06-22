# Fix #012: CIFAR-10 ResNet feature extraction transpose bug

**Date**: 2026-05-12
**Severity**: Critical — crashes CIFAR-10 resnet18/gap feature modes
**Status**: ✅ Fixed

## Error
```
ValueError: pic should not have > 4 channels. Got 32 channels.
```
Occurs in `_load_cifar10(feature_mode="resnet18")` when converting batch images to PIL.

## Root Cause
`torchvision.datasets.CIFAR10.data` returns images in `(N, 32, 32, 3)` NHWC format.
The code called `img.transpose(1, 2, 0)` on each `(32, 32, 3)` image, which reorders
axes from (0,1,2) → (1,2,0), producing `(32, 3, 32)` — 32 in the channel position.
`ToPILImage()` then rejected it for having >4 channels.

The image was already in (H, W, C) format; no transpose was needed.

## Fix
`tools/data_factory.py:867` — removed the erroneous `.transpose(1, 2, 0)`:
```python
# Before (wrong):
T.ToPILImage()(img.transpose(1, 2, 0).astype(np.uint8))
# After (correct):
T.ToPILImage()(img.astype(np.uint8))
```

## Files Changed
| File | Change |
|---|---|
| `tools/data_factory.py` | Line 867: removed `.transpose(1, 2, 0)` from PIL conversion |
