# Fix #004: Fixed-size datasets still showing random seed in UI

**Date**: 2026-05-12
**Severity**: Low — UX issue, no functional impact
**Status**: ✅ Fixed

## Error

For fixed-size datasets (iris, wine, mnist, cifar10_*, etc.), the `sc` (sample count) and `noise` sliders are correctly hidden, but the `seed` number input was always visible. These datasets ignore `random_state` — the seed has no effect.

## Root Cause

`web_demo.py:368` unconditionally rendered `seed = c4.number_input(...)` outside the `if is_fixed` / `else` branch.

## Fix

```python
if is_fixed:
    noise = 0.06
    seed = 42
    c2.caption("(样本量/噪声/种子 由数据集固定)")
else:
    noise = c3.slider("噪声", 0.01, 0.18, 0.06, 0.01)
    seed = c4.number_input("随机种子", 0, 9999, 42)
```

## Prevention

When adding conditional UI logic, check that ALL related controls are inside the same conditional block. The seed was left dangling outside the if/else.
