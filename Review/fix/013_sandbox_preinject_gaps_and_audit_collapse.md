# Fix #013: Sandbox pre-injection gaps + audit timeout collapse + JSON literal leak

**Date**: 2026-05-12
**Severity**: Critical — deep pipeline crash, audit collapse, all high-dim data affected
**Status**: ✅ Fixed

## Bug A: `MinMaxScaler` not pre-injected

### Error
```
NameError: name 'MinMaxScaler' is not defined
```
Occurs in DimensionExpert generated code when processing image data (CIFAR-10).

### Root Cause
`dimension_expert.py:731` — when `is_image=True`, the skeleton sets `SCALER_CLASS = "MinMaxScaler"`, which renders as `_scaler = MinMaxScaler()` in generated code.
`tools/coder_sandbox.py` `_build_core_pre_inject()` (line ~199) — only `StandardScaler` is pre-injected. `MinMaxScaler` is missing.

Every deep pipeline on image data fails at line 1 of execution.

### Fix
Add `MinMaxScaler` import to `_build_core_pre_inject()`:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

### Affected
- All image datasets (CIFAR-10, MNIST, custom uploads with is_image=True)
- DimensionExpert's 7 deep pipelines all crash before any computation

---

## Bug B: JSON `true`/`false` leaked into Python code

### Error
```
NameError: name 'true' is not defined
```
Occurs in ZooExpert and potentially other LLM-generated code.

### Root Cause
Two contributing factors:

1. **`zoo_expert.py:104`** — `json.dumps()` serializes Python bools to `true`/`false`.
   Only `null → None` replacement exists. Missing `true → True` and `false → False`.
   Compare `dimension_expert.py:777` and `graph_expert.py:782` which correctly do all three.

2. **No centralized sanitization** — `base.py:_strip_code_fences()` only removes markdown
   fences. There is no post-processing that catches JavaScript/Python literal mismatches
   in LLM output. Any expert that generates code via a single LLM call (centroid,
   topology, etc.) is vulnerable.

### Fix
(A) Add `true→True` and `false→False` replacements to `zoo_expert.py:104`.
(B) Add `_sanitize_python_literals(code: str) -> str` to `base.py` that all experts
    can reuse, converting `true`/`false`/`null` → `True`/`False`/`None`.

---

## Bug C: Audit sandbox timeout collapse (most critical)

### Error
```
审计引擎坍缩：bootstrap_stability=-1.00, hopkins=-1.00
审计超时，无法完成全量分析。
建议: 审计沙箱超时（timeout=45s）
```

### Root Cause (3-layer cascade)

**Layer 1 — Env var approach is a no-op**
`base.py:56` — `CoderSandbox()` reads `timeout_sec` from constructor arg or module-level
`_DEFAULT_TIMEOUT_SEC` (frozen at import time from env var). Once constructed, the
sandbox's `self.timeout_sec` is a plain instance attribute.
`supervisor.py:917` — `os.environ["ACE_SANDBOX_TIMEOUT_SEC"] = _audit_timeout` sets env
var AFTER sandbox already exists. This has ZERO effect — the sandbox doesn't re-read
the env var on each execution.

**Layer 2 — Fix #011 adaptive timeout never reaches critic**
`supervisor.py:267-276` — the adaptive timeout loop iterates over `active_experts`
(`["centroid", "topology", "zoo"]`). The critic is never in this list. Its sandbox
timeout stays at whatever the module-level default was (60s).

**Layer 3 — 3-tier retry bypassed**
`supervisor.py:934-999` — fast_audit and auto-relaxed retry chain gated by
`elif audit and audit.get("degraded")`. When sandbox times out, `critic.execute_audit()`
returns `None`, hitting the `else` at line ~1000 directly. The 3-tier logic is never
entered. The hardcoded fallback generates the "timeout=45s" message regardless of
actual data dimensions.

The formula at line ~909 (`45 + max(0, (n_features - 100) // 100 * 5)`) would compute
~190s for 3072D CIFAR-10, but it's irrelevant because the sandbox ignores it.

**Sentinel -1.00**: When sandbox times out mid-execution, the audit report dict may
contain partial/default values (-1.00 for bootstrap_stability, -1 for hopkins). The
audit_collapse detector at line ~539 triggers on `_stab <= 0.01 and _hop <= 0.01`,
which matches these unreplaced defaults.

### Fix
(A) At `supervisor.py` audit execution point: directly set `critic.sandbox.timeout_sec = _audit_timeout_sec`
    before calling `critic.execute_audit()`, mirroring what the clustering loop does for experts.
(B) When `critic.execute_audit()` returns `None`, attempt fast_audit before falling
    back to degraded response (fix the else-branch bypass).
(C) Ensure `_audit_timeout_sec` is computed based on n_features/n_samples BEFORE
    the critic call, not derived from ineffectual env vars.

### Affected
- ALL datasets with n_features > 100: audit collapses to unusable defaults
- CIFAR-10 raw (3072D), CIFAR-10 gap (64D), any high-dim custom upload
- The "NO VALID CLUSTERS" verdict is partially caused by this — the system can't
  properly assess results because the auditor is broken

---

## Phase B: Architecture enhancements (applied simultaneously)

### B1: CIFAR-10 smart mode
Added `cifar10_auto` dataset entry — auto-routes to ResNet-18 features (512D)
for semantic embeddings instead of raw pixels.  ResNet-18 transpose bug (Fix #012)
was also resolved, enabling the feature mode to work.

### B2: Image-aware expert routing
`supervisor.py:_execute_full_analysis()` — when `is_image=True` and n_features > 500,
DimensionExpert is force-activated and CentroidExpert is suppressed.  Euclidean
distance in raw pixel space has no semantic discriminability.

### B3: Auto PCA pre-reduction
Already implemented via `_apply_highdim_reduction()` (Phase 5.4 gatekeeper).
Non-image data with n_features > 100 gets PCA→95% variance.  Image data
correctly skips PCA and relies on Conv-AE pipelines.

### B4: Audit sentinel value disambiguation
`supervisor.py` audit collapse detector now distinguishes:
- Negative values (e.g. -1.00) = "not computed" (sandbox timeout)
- Values in [0, 0.01] = genuine near-zero metric
Each case produces a distinct diagnostic message.

## Files Changed

| File | Change |
|---|---|
| `tools/coder_sandbox.py` | +1: `MinMaxScaler` added to `CORE_PRE_INJECT` |
| `expert_sub_agents/base.py` | +18: `_sanitize_python_literals()` function; wired into `execute_with_self_correction()` and `_fix_code()` |
| `expert_sub_agents/zoo_expert.py` | +1: `true→True`, `false→False` replacements in JSON serialization |
| `agent_core/supervisor.py` | +30/-35: direct sandbox timeout setting (no env vars); else-branch fast_audit retry; image-aware routing; sentinel-aware collapse detection |
| `tools/data_factory.py` | +4: `cifar10_auto` mode; ResNet label key fix |
