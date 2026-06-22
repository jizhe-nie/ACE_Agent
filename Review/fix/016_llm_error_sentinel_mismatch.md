# Fix #016: LLM error sentinel prefix mismatch

**Date**: 2026-06-22
**Severity**: Medium (LLM errors silently leak downstream as "valid" content)
**Status**: ✅ Fixed
**Discovered by**: PM 接手审计 (Engineering Log #0001, defect D2)

## Error / Symptom
`UniversalLLMClient.chat_completion()` documents its contract (llm_client.py:313):
> "Assistant reply string, or an error string starting with `Error:`."

But the non-fallback error path returned a string that did **not** match that prefix:

```python
reply = f"Error during LLM call: {error_msg}"   # starts with "Error d", not "Error:"
```

Meanwhile the cost-accounting guard and the documented contract key off `"Error:"`:

```python
completion_tokens = count_tokens(reply) if reply and not reply.startswith("Error:") else 0
```

## Impact
- Cost accounting counted completion tokens for failed calls.
- Any caller relying on the documented `"Error:"` sentinel would treat the error
  message as a real LLM reply. In the self-heal loop (`base.py`), an error string
  flows into `reply or old_code` → fence-strip → sandbox execution → guaranteed
  failure + wasted retries. Router/summary paths would surface error text as answers.

## Root Cause
Two error-return sites in the same method used inconsistent prefixes:
- fallback-failure path: `f"Error: {error_msg}"` ✅ (matches contract)
- no-fallback path: `f"Error during LLM call: {error_msg}"` ❌ (violates contract)

## Fix Applied
Aligned the no-fallback path to the documented `"Error:"` prefix:

```python
reply = f"Error: LLM call failed: {error_msg}"
```

## Affected Files
- `tools/llm_client.py` (`chat_completion`, no-fallback error branch)

## Follow-up (not in this fix, logged for later)
Downstream callers (`base.py._fix_code`, `_generate_code`, `router`, `summarize_report`)
do not all check the `"Error:"` sentinel before using the reply as code/answer.
Aligning the producer is the P0 correctness step; making consumers honor the sentinel
is a P1 robustness improvement → tracked in Engineering Log.

## Verification
- Direct stub test: a provider whose `chat()` raises now drives the no-fallback
  branch to return `'Error: LLM call failed: <msg>'`, and `.startswith("Error:")`
  == **True** (was False before the fix).
- `pytest tests/test_p0.py tests/test_core.py tests/test_fence_stripping.py`
  → **54 passed (exit 0)**, no regression.

## Prevention
- Single source of truth for the error sentinel: define `_ERR_PREFIX = "Error:"` and
  build/check error strings from it, so producer and consumer cannot drift.
