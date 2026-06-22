# Fix #015: Router exception-fallback dead ternary branch

**Date**: 2026-06-22
**Severity**: Low (logic smell; no incorrect runtime behavior, but misleading)
**Status**: ✅ Fixed
**Discovered by**: PM 接手审计 (Engineering Log #0001, defect D1)

## Error / Symptom
No runtime error. `MasterRouter.analyze_intent()` exception-fallback contained a
ternary whose two branches were identical:

```python
"intent": "FOLLOW_UP" if ("?" in prompt or "？" in prompt) else "FOLLOW_UP",
```

Both branches return `"FOLLOW_UP"`, so the condition `("?" in prompt or "？" in prompt)`
was evaluated and discarded — dead logic.

## Root Cause
Vestigial code: the class docstring and the inline comment both state the intended
behavior is "router 异常时 fallback 到 FOLLOW_UP（而非 NEW_TASK），避免误触发昂贵的
新任务流程". The ternary was likely a half-finished edit (one branch presumably meant
to be `NEW_TASK` or `QUESTION`) that was never completed, leaving a no-op condition.

## Fix Applied
Simplified to an unconditional `"FOLLOW_UP"`, matching the documented intent:

```python
return {
    "intent": "FOLLOW_UP",
    "reasoning": "语义解析异常，安全兜底为 FOLLOW_UP",
}
```

## Affected Files
- `agent_core/router.py` (exception handler at the end of `analyze_intent`)

## Verification
- Import + smoke: unconfigured-LLM path still returns `NEW_TASK` (untouched);
  edit only affects the JSON-parse exception fallback. `X if cond else X` → `X` is
  value-equivalent.
- `pytest tests/test_p0.py tests/test_core.py tests/test_fence_stripping.py`
  → **54 passed (exit 0)**. (test_follow_up excluded — it hangs on ACESupervisor
  cold start, unrelated to this change; tracked as Engineering Log T-1.)

## Prevention
- Watch for `A if cond else A` patterns — ruff `SIM` rules can be extended to catch
  some, but identical-branch ternaries need review. Avoid leaving half-finished
  conditionals; if a branch is undecided, leave a `# TODO` not a no-op condition.
