# CLAUDE.md

## Project: ACE Agent (Automated Clustering Expert)

ACE Agent is a multi-expert LLM-driven clustering system. The supervisor orchestrates
a pool of 7 specialist agents (centroid, topology, zoo, dimension, critic, ensemble,
graph) that each generate Python code, execute it in a sandbox, and self-correct on
failure. Results are ranked via ARI (when ground-truth labels exist) or internal
metrics, audited post-hoc, and fused into a consensus.

## Commands

```bash
# Run the Streamlit web demo
streamlit run web_demo.py

# Run unit tests (183+ tests)
pytest tests/test_benchmark.py -q

# Run quick benchmark on a specific dataset
python _bench_70k_v4.py

# Run system integration tests
pytest tests/test_system_integration.py -q
```

## Architecture

```
user prompt → MasterRouter (intent)
           → ACESupervisor
               ├── RAG (knowledge_engine)
               ├── DataStructureClassifier (graph_connected/spherical/...)
               ├── Expert dispatch (parallel)
               │     ├── CentroidExpert (KMeans/GMM/Birch)
               │     ├── TopologyExpert (DBSCAN/HDBSCAN/OPTICS/Spectral)
               │     ├── ZooExpert (dynamic algorithm selection)
               │     ├── DimensionExpert (7 deep pipelines)
               │     └── GraphExpert (community discovery, conditional)
               ├── Informed Ranking (ARI veto when labels exist)
               ├── CriticAudit (post-hoc, 3-tier timeout)
               ├── Critic 2.0 Feedback Loop (RETRY with constraints)
               ├── Ensemble Consensus (co-association matrix)
               └── LLM Summary
```

## Key Design Decisions

- **ARI is the sole ranking criterion when ground-truth labels exist.** Internal
  metrics (Silhouette, Edge Cut, modularity) are excluded from ranking — they are
  for reference only. This prevents Silhouette bias toward centroid algorithms on
  non-convex data.
- **Consensus Trap Detection**: When EnsembleConsensus has high self-reported
  agreement (≥0.7) but lower ARI than the best individual expert, the ensemble is
  flagged as overfitting/consensus-bias and ranked below the individual winner.
- **Connectivity Pre-check**: k-NN geodesic distortion is computed BEFORE expert
  dispatch. When distortion > 0.35, centroid algorithms (KMeans/GMM/Birch) are
  banned from winning, regardless of ARI.
- **Multi-tier Audit Timeout**: Normal → fast_audit (skip bootstrap) → auto-relaxed
  (500 samples + 2x timeout). Designed for high-dim data where distance matrices
  cause 120s+ sandbox timeouts.

## Development Guidelines

### When implementing new algorithms or improvements:
- **Use web search and retrieval** to research SOTA clustering algorithms before
  implementation. Check recent papers (2023-2026) on arxiv, paperswithcode, and
  sklearn-contrib for established best practices. Specifically:
  - Search for "[dataset name] clustering benchmark" to see what algorithms perform
    well on each dataset in the project's benchmark suite.
  - Search for "[algorithm] sklearn implementation best practices" to verify
    parameter settings and preprocessing requirements.
  - Compare ACE Agent's results against published benchmarks to identify when the
    project's implementation is underperforming relative to SOTA.

### Code style:
- Python 3.11+, `from __future__ import annotations`
- Type hints on all public methods
- No comments that narrate WHAT the code does — names should do that
- Short inline comments only for non-obvious WHY

### Expert implementation:
- All experts subclass `BaseExpert` and implement `_generate_code()`
- Experts that don't need LLM set `REQUIRES_LLM = False`
- Generated code writes to the `artifacts` dict injected by the sandbox
- Code MUST NOT: use `if __name__ == "__main__"`, define uncalled `main()`/`run()`,
  or reassign the `artifacts` variable

### Audit reports:
- Written to `Review/YYYY-MM-DD-<phase>-Audit/assessment.md`
- Follow the structure: 综述 → 核心成果 → 技术细节 → 结论与建议

### Bug fix tracking (MANDATORY):
- **Every discovered bug MUST be documented** in `Review/fix/NNN_descriptive_name.md`
  with sequential numbering (001, 002, ...).
- Each fix file must contain: Date, Severity, Error message, Root cause, Fix applied,
  Affected files, and Prevention notes.
- **Before modifying any file**, check `Review/fix/` to see if the same bug or
  related code area has been fixed before — avoid regressions and repeated mistakes.
- When you produce wrong cluster plots, wrong data, or wrong outputs during
  development, save them alongside the fix file for future reference.
- Common bug categories to watch for:
  - Variable scope errors (name not defined in method scope)
  - Metadata flags not propagated through the pipeline (is_image, shape_family)
  - String mismatch between dataset names and internal mode strings
  - Sandbox timeout on high-dim data (audit collapse pattern)
  - UI controls outside conditional blocks (shown when they should be hidden)

## Memory

Claude Code memory is at `.claude/projects/.../memory/`. Check MEMORY.md before
starting work for context on user preferences and ongoing tasks.
