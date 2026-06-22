# Fix #008: Audit sandbox collapse on high-dim data

**Date**: 2026-05-12
**Severity**: Medium — audit produces useless reports on >32D data
**Status**: ⚠️ Partially Mitigated (root cause requires architectural change)

## Error Pattern

```
【审计坍缩警告】bootstrap_stability 和 hopkins 均为 0，审计引擎疑似在高维数据上超时崩溃
【审计】Critic 审计过程异常: name 'n_features' is not defined  (fixed in Fix #002)
Bootstrap 稳定性: 0.00 (or -1.00)
Hopkins 趋势: 0.00 (or -1.00)
过拟合风险: unknown
```

## Root Cause

1. **Sandbox timeout (45-120s) insufficient** for audit tasks on >32D data:
   - Bootstrap stability requires 5-15 rounds of clustering with distance matrix computation
   - High-dim distance matrices are O(N² × D), causing timeout before completion
   - The multi-tier retry (fast_audit → auto-relax) helps but can't fix fundamental timeout

2. **Default values leak through on timeout**: When sandbox times out, artifacts may contain partially written default values (0.0, -1.0, empty lists), which are indistinguishable from real results.

3. **Audit dimension reduction (PCA→16D) only in Critic**, not in other experts. The audit analyzes data in a different space than what experts clustered in.

## Mitigations Already in Place

- Three-tier audit timeout: normal → fast_audit (skip bootstrap) → auto-relaxed (500 samples + 2x timeout)
- `audit_collapse` detection flag (supervisor.py:479-501): detects when bootstrap + hopkins are both ≈0
- Section 0.1 PCA→16D in Critic audit prompt for n_features > 32

## Remaining Issues

- The "degraded audit" fallback produces a generic `confidence_level=0.3` report that adds noise, not signal
- Audit sandbox and expert sandbox share the same timeout mechanism — deep learning pipelines (AE) need different timeout rules than distance matrix computation
- No automatic correlation between "audit collapsed" and "all ARI < 0.2" to produce a unified "this data is unclusterable" verdict

## Recommended Architectural Fix (deferred)

1. Separate sandbox pools: "fast" (sklearn, 60s timeout) and "slow" (deep learning, 300s timeout)
2. Pre-compute audit-relevant metrics (Hopkins, PCA variance) BEFORE expert dispatch
3. When audit collapses AND all ARI < 0.2, skip the degraded audit and display a clean "数据不可聚类" verdict
