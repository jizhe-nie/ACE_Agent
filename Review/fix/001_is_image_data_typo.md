# Fix #001: `is_image_data` typo in dimension_expert.py

**Date**: 2026-05-11
**Severity**: High — causes DimensionExpert crash on all image datasets
**Status**: ✅ Fixed

## Error

```
【主控】专家 'dimension' 执行时发生未捕获异常: name 'is_image_data' is not defined
```

## Root Cause

`dimension_expert.py:783` referenced variable `is_image_data` which does not exist. The correct variable name is `is_image` (defined at line 679).

## Fix

```diff
- .replace("{RES_AE_IMPORT}", res_ae_import if not is_image_data else "")
+ .replace("{RES_AE_IMPORT}", res_ae_import if not is_image else "")
```

## Prevention

Always verify variable names match across the scope. `is_image` was defined at L679 but `is_image_data` was used at L783. IDE linter would have caught this.
