---
name: Phase 0.5 Engineering Decisions
description: Key design decisions made during P0.5 emergency patch (2026-04-20)
type: project
---

**P0.5-A ZooExpert interface**: ZooExpert now inherits BaseExpert and implements `_generate_code` deterministically (no LLM call). Code is built as a list of unindented strings joined with `\n` — NOT via `textwrap.dedent(f"""...""")` because f-string interpolation of blocks with different indentation breaks dedent's common-prefix detection. The `run()` method retained as deprecated alias emitting `DeprecationWarning`.

**DBSCAN eps correction**: AlgorithmZoo defaults `eps=0.5` (tuned for unscaled data). ZooExpert's `_generate_code` post-processes DBSCAN params to use `eps=0.3` for StandardScaler-normalized data. With eps=0.5 on scaled moons data, all points merge into one cluster (score=0). This is a ZooExpert-level override, not a change to AlgorithmZoo.

**P0.5-B Expert registry**: `build_expert_registry()` in `expert_sub_agents/__init__.py` wraps each instantiation in `try/except TypeError` to gracefully skip WIP experts (DimensionExpert, DeepRepresentationExpert, MultiViewExpert) that don't implement `_generate_code` yet. Default active experts in supervisor: `["centroid", "topology", "zoo"]`. Phase 1 Critic/smart routing will take over selection.

**P0.5-C Router CODE_EXAMPLE**: Three intents now: NEW_TASK, FOLLOW_UP, CODE_EXAMPLE. Fallback on router error changed from NEW_TASK to FOLLOW_UP (safer). `_handle_code_example` in supervisor generates Markdown code block via LLM, does NOT run sandbox, does NOT update `self.last_report`. `LatexReportGenerator.generate()` raises ValueError early for `response_type=="CODE_EXAMPLE"` so the `try/except pass` in supervisor silently skips it.

**P0.5-D Soft failure detection**: `base.py` execute loop now checks `success AND artifacts` (not just `success`). When success=True but artifacts={}, a soft-failure message is constructed mentioning artifacts convention, and the LLM fix loop is triggered with that hint. `_error_report` accepts `expert_logs` dict and embeds last-3-lines from each expert into the summary.

**mypy baseline**: Before P0.5 = 27 errors. After P0.5 = 15 errors. No new errors introduced; 12 errors eliminated.

**Why**: P0.5 sprint 2026-04-20, fixing 3 production bugs (moon data DBSCAN miss, CODE_EXAMPLE misrouting, success-but-empty artifacts swallowed).
**How to apply**: When extending ZooExpert or adding new experts, follow the `_generate_code`-as-list pattern. When adding new intent types, update both router system prompt and supervisor's `run()` dispatch.
