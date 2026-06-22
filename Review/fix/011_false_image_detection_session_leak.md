# Fix #011: False image detection + session context leak + sandbox timeout

**Date**: 2026-05-12
**Severity**: Critical — wrong routing, wrong data classification, expert timeout
**Status**: ✅ Fixed

## Issue 1: False positive image detection

### Error
Letter dataset (16D statistical features) was classified as image data:
```
【图像语义感知】检测到图像形数据（16D = 4×4）
```

### Root Cause
`_decompose_dim_to_image_hint(16)` factored 16 as 4×4 and returned "4×4".
The factorization function had a minimum threshold of n_features ≥ 16, which is
far too low — 4×4 is not a meaningful image dimension. 16 features can represent
anything (statistical features, sensor readings, etc.).

### Fix
- Raised minimum threshold from n_features ≥ 16 to n_features ≥ 256 (≈16×16 min)
- Raised minimum spatial dimension from 4 to 8 per side
- Widened aspect ratio tolerance from 0.25–4.0 to 0.2–5.0
- Simplified `_decompose_image_shape()` in data_factory.py (unified channel loop)

### Verdict
16D → None, 64D → None, 100D → None, 256D → "8×32", 784D → "28×28" ✓

## Issue 2: Session context leak across conversations

### Error
User started a new conversation (clicked "新建对话"), selected blob dataset,
entered "请分析该数据", but the system routed as FOLLOW_UP and analyzed the
previous session's Letter dataset.

### Root Cause
`ACESupervisor` is cached via `@st.cache_resource` as a singleton. Its
`self.memory` and `self.last_report` accumulate across sessions. When the
user creates a new session (clears messages, new session_id), the supervisor's
memory still contains the old conversation context. `MasterRouter.analyze_intent()`
sees the old context and routes "分析该数据" as FOLLOW_UP.

### Fix
1. Added `ACESupervisor.reset_state()` method that clears `self.memory` and
   `self.last_report`.
2. In `web_demo.py` `_handle_prompt()`, detected fresh session (first user
   message) and called `supervisor.reset_state()` before processing.

## Issue 3: Sandbox timeout on large datasets

### Error
CentroidExpert and TopologyExpert both timed out (60s wall-clock limit) on
Letter dataset (20000 samples × 16D):
```
[质心专家] 沙箱资源超限 (timeout): Execution exceeded 60s wall-clock limit.
[拓扑专家] 沙箱资源超限 (timeout): Execution exceeded 60s wall-clock limit.
```

### Root Cause
Default sandbox timeout is 60s for all datasets. With 20000 samples, DBSCAN
computes a full 20000² distance matrix (400M pairs), and GMM full covariance
runs O(N×D²) iterations — both exceed 60s.

### Fix
Added adaptive sandbox timeout scaling in `_execute_full_analysis()`:
- n ≤ 2000: 60s (default)
- n ≤ 5000: 90s
- n ≤ 10000: 120s
- n ≤ 20000: 180s
- n > 20000: 240s

Timeout is applied to each expert's sandbox before dispatch.

## Files Changed

| File | Change |
|---|---|
| `agent_core/supervisor.py` | +15: `reset_state()`, +15: adaptive sandbox timeout, fixed `_decompose_dim_to_image_hint()` thresholds |
| `web_demo.py` | +5: fresh session detection + `reset_state()` call |
| `tools/data_factory.py` | Simplified `_decompose_image_shape()` — unified loop, higher thresholds |
