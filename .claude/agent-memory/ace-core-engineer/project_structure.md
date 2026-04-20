---
name: ACE Agent Project Structure
description: Key file locations, module roles, and Python environment for ACE Agent
type: project
---

**Project root**: `D:/PycharmProject/ACE_Agent`
**Python env**: conda `Tumor_Subtype_Agent` at `C:/Users/Administrator/anaconda3/envs/Tumor_Subtype_Agent/`
**Run commands**: prefix all python/pytest/ruff/mypy with `C:/Users/Administrator/anaconda3/envs/Tumor_Subtype_Agent/python.exe -m ...`
**Test root**: run from `cd /d/PycharmProject` then `python.exe -m pytest ACE_Agent/tests/`

Key modules:
- `tools/llm_client.py`: LLMProvider ABC + DeepSeekProvider/DashScopeProvider/OpenAIProvider; UniversalLLMClient (tracing + fallback); trace written to `outputs/llm_trace.jsonl`
- `tools/coder_sandbox.py`: CoderSandbox with wall-clock timeout (threading) + delta-RSS memory watchdog (psutil); SandboxResourceExceeded(reason) exception; Windows-safe (no resource module)
- `expert_sub_agents/base.py`: BaseExpert with Think-Act-Fix loop (MAX_RETRIES=3); passes caller/attempt to LLM client
- `agent_core/supervisor.py`: ACESupervisor orchestrator; LLM-driven intent routing; calls centroid + topology experts
- `tools/settings_store.py`: SettingsStore (JSON config), SessionManager, DEFAULT_PROVIDERS dict
- `web_demo.py`: Streamlit entry point; sidebar has Model Config popover + LLM Call Monitor expander

**Why**: Phase 0 completed 2026-04-20. All 5 tasks (P0-1 through P0-5) delivered.
**How to apply**: Always read these files before editing — the code evolves fast.
