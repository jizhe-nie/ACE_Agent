# Fix #009: Elimination of hardcoded patterns across the project

**Date**: 2026-05-12
**Severity**: High — hardcoded assumptions silently fail for custom/novel datasets
**Status**: ✅ Fixed

## Problem

The project had systemic hardcoded assumptions that made it brittle for datasets
outside the predefined benchmark set:

1. **`FIXED_SIZE_DATASETS` hardcoded set** (web_demo.py): 19 dataset names
   manually listed. Adding a new dataset required updating two places
   (generate_dataset + FIXED_SIZE_DATASETS). Forgot one → broken UI.

2. **`_detect_image_data()` hardcoded dict** (supervisor.py): Only 5 entries
   {3072, 1024, 784, 4096, 16384}. A 224×224×3=150528D image or 64×64×3=12288D
   image would NOT be detected as image data.

3. **`_classify_data_structure()` n_features ≤ 5 cap** (supervisor.py): Geodesic
   distortion only computed for ≤5D data. Any 6D+ non-convex dataset (e.g.
   Pendigits 16D, Letter 16D, USPS 256D) got shape_family classification but
   ZERO geodesic insight. Graph expert never activated for these datasets.

4. **`_connectivity_pre_check()` n_features > 10 skip** (supervisor.py):
   Pre-check returned immediately for all datasets >10D. Centroid ban never
   triggered for e.g. S-curve 3D embedded in higher dims.

5. **Custom uploads: no is_image, shape_family="unknown"** (data_factory.py):
   Every uploaded CSV got bare metadata with no image detection, no structure
   inference, no cluster count estimation.

## Root Cause

All these patterns share the same root: **configuration by enumeration instead
of detection by computation**. The project grew organically by adding datasets
one at a time, each with manually configured metadata, but never developed the
auto-detection layer needed for truly automatic clustering.

## Fix

### 1. Auto image shape detection (data_factory.py + supervisor.py)

New `_decompose_image_shape(n_features)` in data_factory.py factorizes feature
count into H×W or H×W×C, verifying aspect ratio sanity (0.25–4.0).

New `_decompose_dim_to_image_hint(n_features)` in supervisor.py does the same
but returns string hints for trace messages.

Both replace hardcoded dict lookups with algorithmic factorization.

### 2. Auto shape_family inference (data_factory.py)

New `_infer_shape_family(X, n_features)` uses PCA variance concentration:
- 1st component >75% → `manifold`
- 2 components <30% cumulative → `sparse`
- 5 components >90% cumulative → `spherical`
- Otherwise → `manifold`

Called automatically in `load_custom_dataset()`.

### 3. Auto fixed-size detection (web_demo.py)

Replaced hardcoded `FIXED_SIZE_DATASETS` set with `_is_fixed_size_dataset()`
that generates a 10-sample test dataset and checks if the result size differs.

### 4. Increased geodesic distortion caps (supervisor.py)

- `_classify_data_structure()`: n_features ≤ 5 → ≤ 50
- `_connectivity_pre_check()`: n_features > 10 → > 50

Both already had anchor sampling for N > 2000, so the increased dim cap adds
minimal cost.

### 5. UI manual image override (web_demo.py)

Added "图像数据" checkbox in upload tab for cases where auto-detection fails.
Wired through to `_handle_prompt()`.

## Prevention

- Never hardcode per-dataset configuration that can be computed from the data.
- Dimension caps should be generous enough to cover real-world data shapes.
- Always provide manual override for auto-detection that may fail.

## Files Changed

| File | Change |
|---|---|
| `tools/data_factory.py` | +70: `_decompose_image_shape()`, `_infer_shape_family()`, `load_custom_dataset()` calls both |
| `web_demo.py` | +30/-10: auto fixed-size detection, image checkbox, flag wiring |
| `agent_core/supervisor.py` | +20/-10: `_decompose_dim_to_image_hint()`, increased caps in classify + pre-check |
