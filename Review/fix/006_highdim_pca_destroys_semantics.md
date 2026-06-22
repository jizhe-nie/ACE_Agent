# Fix #006: High-dim PCA gatekeeper destroys semantic information

**Date**: 2026-05-12
**Severity**: Critical — renders CIFAR-10 raw completely unclusterable
**Status**: ✅ Fixed

## Error

CIFAR-10 raw (3072D) passes through `_apply_highdim_reduction()` which runs PCA retaining 95% variance. The resulting ~50D space preserves brightness/background variance but destroys ALL class-discriminative information. Every clustering algorithm then produces ARI ≈ 0.

## Root Cause

`supervisor.py:201-204`:
```python
if _n_features_raw > 100:
    _highdim_result = self._apply_highdim_reduction(dataset, trace)
```

This is a blanket rule: "if >100D, PCA to 95% variance." It assumes dimensionality is noise, but for image data, the raw pixel dimensions ARE the signal carriers. PCA on pixels preserves the wrong kind of variance.

There is no check for whether the data is image-shaped before applying this reduction.

## Consequence Chain

```
CIFAR-10 raw 3072D
→ PCA(100 components, 95% var) → ~50D
→ All experts cluster on 50D PCA space
→ ARI ≈ 0.04 for all algorithms
→ FAILED verdict triggered
→ Geodesic Pipeline rescue (UMAP of PCA space → still noise)
→ HONEST FAILURE report
```

Every step is "correct" given the initial wrong decision to PCA raw pixels.

## Fix Plan

Option A: For image-shaped data (>100D that looks like w×h×c), skip PCA reduction entirely and let DimensionExpert handle it with Conv-AE.

Option B: For image-shaped data, auto-switch to GAP features (8×8 pooling per channel) instead of PCA.

Option C: Add `is_image` metadata check BEFORE the high-dim gatekeeper. If `is_image=True`, skip PCA and route to Conv-AE pipeline.
