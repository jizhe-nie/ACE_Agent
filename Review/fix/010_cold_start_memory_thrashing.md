# Fix #010: Cold-start memory thrashing and performance regression

**Date**: 2026-05-12
**Severity**: Critical — memory/disk 100%, 17s+ cold start, UI unresponsive
**Status**: ✅ Fixed

## Symptom

Starting from Phase 3 (Topology-Aware upgrade), the project experienced severe
performance regression:
- Cold start: 17s+ black screen before UI appears
- Every interaction: 2-5s lag
- Memory and disk utilization near 100% (thrashing)
- Previously the project had no such issues

## Root Cause

**Import cascade at cold start.** Four modules loaded heavy libraries at module
level, and they all triggered simultaneously when `ACESupervisor()` was created:

```
web_demo.py
  └─ _get_supervisor() → ACESupervisor()
       ├─ KnowledgeEngine.__init__()
       │    └─ SentenceTransformer("all-MiniLM-L6-v2")  ← PyTorch + transformers (~2GB)
       │
       ├─ build_expert_registry()
       │    └─ imports all 7 expert modules
       │         ├─ ensemble_expert.py: module-level matplotlib import (~100MB)
       │         └─ (others)
       │
       └─ supervisor.py module body
            └─ from ACE_Agent.tools.graph_builder import GraphBuilder  ← NEW in Phase 3
                 └─ graph_builder.py module-level imports:
                      ├─ matplotlib + matplotlib.pyplot (~100MB)
                      ├─ scipy.sparse.csgraph (~50MB)
                      └─ sklearn.neighbors (~200MB)
```

**Total simultaneous memory pressure**: PyTorch (~2GB) + transformers (~500MB) +
scipy (~200MB) + sklearn (~200MB) + matplotlib (~100MB) + chromadb (~50MB) + numpy (~100MB)
≈ **3+ GB** loaded in one shot.

On a machine with limited physical RAM, this caused Windows to start paging to
disk (swap), resulting in the 100% disk utilization observed.

**Why it was fast before Phase 3**: `graph_builder.py` was added in Phase 3
(Topology-Aware). Its module-level import in supervisor.py triggered the cascade
that pushed memory over the edge.

## Fix

### 1. supervisor.py — lazy GraphBuilder import

Moved `from ACE_Agent.tools.graph_builder import GraphBuilder` from module level
into each of the 3 methods that actually use it:
- `_classify_data_structure()`
- `_connectivity_pre_check()`
- `_check_topology_failure()`

GraphBuilder is only needed when data is classified as graph-connected (rare);
it was being loaded on every single cold start.

### 2. knowledge_engine.py — lazy SentenceTransformer loading

SentenceTransformer model loading deferred from `__init__()` to first
`query()` or `ingest_docs()` call via `_ensure_embed_fn()` lazy init pattern.

`KnowledgeEngine.__init__()` now only does:
- Lightweight ChromaDB PersistentClient (no model needed)
- Manifest JSON read

The ~2GB PyTorch + transformers + model load is deferred until the first
actual RAG query, which may never happen in a session.

### 3. graph_builder.py — lazy matplotlib import

Moved `import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt`
from module level into `save_adjacency_image()` — the only method that uses it.

### 4. ensemble_expert.py — lazy matplotlib import

Moved matplotlib imports from module level into `_generate_consensus_plot()` —
the only function that uses it.

## Expected Improvement

| Metric | Before | After |
|---|---|---|
| Cold start import time | 17s+ | ~3-5s |
| Memory at idle | ~3GB+ | ~500MB |
| Disk I/O at startup | 100% (thrashing) | minimal |
| UI interaction lag | 2-5s | <1s |

## GPU Acceleration

CUDA is not available on this machine (`torch.cuda.is_available() = False`).
When a GPU is available, the AE/DEC pipelines in DimensionExpert already include
device detection in their generated code. No additional changes needed.

## Co-association Matrix

The co-association matrix is central to EnsembleConsensus — it's how the system
fuses multiple expert opinions. It should NOT be deleted. The earlier concern
about `.ace_sessions.json` bloat was incorrect (the file is 0 bytes). The actual
memory pressure was from the import cascade documented above.

## Files Changed

| File | Change |
|---|---|
| `agent_core/supervisor.py` | +4 lazy imports/-1 module-level: GraphBuilder imported inside 3 methods |
| `agent_brain/knowledge_engine.py` | +35: `_ensure_embed_fn()` lazy init, defer SentenceTransformer |
| `tools/graph_builder.py` | +3/-3: matplotlib import moved into `save_adjacency_image()` |
| `expert_sub_agents/ensemble_expert.py` | +3/-3: matplotlib import moved into `_generate_consensus_plot()` |
