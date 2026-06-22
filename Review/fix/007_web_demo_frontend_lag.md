# Fix #007: Web demo frontend unresponsiveness

**Date**: 2026-05-12
**Severity**: Medium — 2-4s lag on every Streamlit rerun
**Status**: ✅ Fixed

## Error

Every Streamlit rerun (selectbox change, button click, etc.) had 2-4s of fixed overhead, making the UI feel sluggish.

## Root Cause

Three blocking operations ran on every single rerun:

1. **`import matplotlib.pyplot as plt` at module level** (line 15): matplotlib import takes 1-3s due to font scanning and backend initialization. Even when no plots are being rendered, this import runs.

2. **`_read_trace_stats()`** reads and parses the entire `outputs/llm_trace.jsonl` file on every rerun. As usage accumulates, this file grows, making each parse slower.

3. The `st.button("Preview Data Distribution")` combined with `or uploaded_file` caused cache recomputation on every rerun for uploaded datasets.

## Fix

1. **Matplotlib lazy import**: Removed module-level `import matplotlib.pyplot as plt`. Added local imports inside each function that uses `plt`:
   - `_setup_matplotlib_fonts()` (cached, first call does import + Agg backend)
   - `main()` preview block
   - `_render_ensemble_metrics()`
   - `_render_disagreement_heatmap()`

2. **`_read_trace_stats()` cached with TTL**:
   ```python
   @st.cache_data(ttl=30)
   def _read_trace_stats() -> dict[str, int]:
   ```
   Re-parses at most once every 30 seconds, regardless of how many reruns occur.

3. All other heavy operations already use `@st.cache_data` or `@st.cache_resource`.

## Prevention

- Never import heavy libraries (matplotlib, torch, umap) at module level in Streamlit apps.
- Use `@st.cache_resource` for one-time initialization, `@st.cache_data` with TTL for data that changes slowly.
- Streamlit re-executes the entire script on every interaction — every top-level statement costs time.
