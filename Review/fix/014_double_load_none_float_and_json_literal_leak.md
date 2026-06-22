# Fix #014: Double dataset loading + None float crash + JSON literal leak (deep fix)

**Date**: 2026-05-13
**Severity**: High — wasted 140s on CIFAR-10 reload, TypeErrors on audit, persistent `true` errors
**Status**: ✅ Fixed

## Bug A: Double dataset loading (CIFAR-10 loads twice)

### Root Cause
`web_demo.py:101` `_is_fixed_size_dataset()` called `generate_dataset(ds_name, n_samples=10, ...)` which for CIFAR-10 variants loads ALL 60K images through ResNet-18 (~70s) just to check `X.shape[0] != 10`. Later, `_cached_preview_data(ds_name, sc, ...)` triggers another full load (~70s) with a different Streamlit cache key.

### Fix
Replaced the trial-load approach with a static name-based check. Only 6 datasets (blobs, moons, s_curve, smile, high_dim, multi_view) are synthetically generated and respect n_samples. All others are fixed-size.
- `_DYNAMIC_SIZES = frozenset({"blobs", "moons", "s_curve", "smile", "high_dim", "multi_view"})`
- `_is_fixed_size_dataset` now returns `ds_name not in _DYNAMIC_SIZES` (no data loading)
- Removed `@st.cache_data` decorator and `.clear()` call (no longer needed)

### Affected
- `web_demo.py`: `_is_fixed_size_dataset` (rewritten), cache clearing (line removed)

---

## Bug B: TypeError when stability_score key exists but value is None

### Error
```
TypeError: float() argument must be a string or a real number, not 'NoneType'
```

### Root Cause
`supervisor.py:552` — `audit_report.get("stability_score", -1)` returns `None` when the key **exists** but has value `None`. Python's `dict.get(key, default)` only applies the default when the key is **missing**, not when it's None.

### Fix
Explicit None-check before `float()`:
```python
_sv = audit_report.get("stability_score")
_stab = float(_sv if _sv is not None else -1)
```
Same pattern applied to hopkins, confidence_level, and the audit collapse detector at line 583.

### Affected
- `agent_core/supervisor.py`: lines 552-554 (rewritten), line 583 (fixed), lines 1844-1845 (hardened)

---

## Bug C: JSON literal leak — sandbox last-resort defense

### Root Cause
`_sanitize_python_literals()` in `base.py` was correctly wired at expert-layer code generation (lines 96, 350), but some code paths (critic internal generation, ensemble code runs, dimension/graph LLM output with partial JSON parse failure) could still deliver `true`/`false`/`null` to the sandbox.

### Fix
Added identical regex sanitization directly in `CoderSandbox.execute()` (tools/coder_sandbox.py:571-578) as a last-resort defense. Every code string entering the sandbox is now sanitized regardless of which code path generated it.

```python
import re as _re_sbx
code = _re_sbx.sub(r'\btrue\b', 'True', code)
code = _re_sbx.sub(r'\bfalse\b', 'False', code)
code = _re_sbx.sub(r'\bnull\b', 'None', code)
```

### Affected
- `tools/coder_sandbox.py`: +7 lines in `execute()` method
