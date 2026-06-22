# Fix #002: `n_features` not defined in critic_expert._generate_code()

**Date**: 2026-05-11
**Severity**: High — causes Critic audit crash on all datasets
**Status**: ✅ Fixed

## Error

```
【审计】Critic 审计过程异常: name 'n_features' is not defined
```

## Root Cause

`critic_expert.py:174` uses `{n_features}` in an f-string in `_generate_code()`, but `n_features` was only defined in `execute_audit()` (line 56). The variable was not extracted from the `dataset` parameter within `_generate_code()` scope.

## Fix

Added at line 136 in `_generate_code()`:
```python
n_features = dataset.X.shape[1] if dataset.X.ndim == 2 else 1
```

## Prevention

When adding new prompt template variables, always check they are extracted from parameters in the same method scope. Different methods on the same class do not share local variables.
