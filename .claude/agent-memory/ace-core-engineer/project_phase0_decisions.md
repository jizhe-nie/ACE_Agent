---
name: Phase 0 Engineering Decisions
description: Key design decisions made during Phase 0 implementation (2026-04-20)
type: project
---

**P0-4 Sandbox memory monitoring**: Uses delta-RSS (current - baseline) not absolute RSS. The Python process with torch/numpy loads at ~750 MiB, so absolute-RSS monitoring would fire immediately. Delta approach measures only what the sandbox CODE allocates. Default limit = 2 GiB delta. Windows-safe via psutil + daemon thread.

**P0-5 Provider fallback**: Fallback triggers on HTTP 429/500/502/503/504 or Timeout. Max 1 fallback attempt. Fallback event written to trace as `event: provider_fallback`. Default providers: DeepSeek (primary, cheap), DashScope (fallback candidate). Ollama intentionally excluded per Phase 0 spec.

**P0-3 Trace format**: JSONL at `outputs/llm_trace.jsonl`. Fields: timestamp, model, provider, prompt_tokens, completion_tokens, latency_ms, caller, is_retry, attempt, fallback_triggered, cost_usd. Caller naming convention: `{expert_key}:generate`, `{expert_key}:fix:{N}`, `router`, `summarize_report`.

**P0-2 Coverage baseline**: Phase 0 achieved ~62% coverage. CI `--cov-fail-under=30` (not 60) to avoid blocking CI on pre-existing uncovered code. TODO raise to 60 in Phase 1.

**mypy.ini encoding**: File must be ASCII (no UTF-8 em dashes). Windows GBK codec chokes on UTF-8 multi-byte chars in .ini files. Use `--` for em dash in comments.

**Why**: Decisions made during P0 sprint 2026-04-20.
**How to apply**: Reference when extending sandbox, adding providers, or tightening CI gates.
