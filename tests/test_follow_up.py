"""
tests/test_follow_up.py
=======================
Integration test for the FOLLOW_UP intent path.

Requires a live LLM configuration. Set environment variables or use .ace_demo_config.json.
When LLM is not configured, the test is skipped automatically.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root's parent is on sys.path for ACE_Agent.* imports
_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

import pytest
from dotenv import load_dotenv

from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.tools.data_factory import generate_dataset
from ACE_Agent.tools.llm_client import LLMSettings
from ACE_Agent.tools.settings_store import DEFAULT_PROVIDERS, SettingsStore


def _build_settings() -> LLMSettings:
    """Attempt to build a configured LLMSettings from env / config store."""
    load_dotenv()
    store = SettingsStore()
    active = store.get("active_provider", "DeepSeek")
    api_keys = store.get("api_keys", {})
    api_key = api_keys.get(active, "")

    # Also check environment variable fallback
    if not api_key:
        api_key = os.environ.get(f"ACE_{active.upper()}_API_KEY", "")

    provider_cfg = DEFAULT_PROVIDERS.get(active, {})
    return LLMSettings(
        provider=active,
        base_url=provider_cfg.get("base_url", ""),
        api_key=api_key,
        model=store.get("model") or (provider_cfg.get("models", [""])[0]),
        enabled=bool(api_key),
    )


@pytest.mark.skipif(
    not _build_settings().is_configured,
    reason="LLM not configured — set ACE_DEEPSEEK_API_KEY or configure via UI",
)
def test_follow_up() -> None:
    """Verify that FOLLOW_UP intent reuses previous analysis context without rerunning algorithms."""
    settings = _build_settings()
    supervisor = ACESupervisor()

    # Task 1: initial clustering analysis
    dataset = generate_dataset("smile")
    print("\n--- Task 1: analyse smile dataset ---")
    report1 = supervisor.run(dataset, "帮我分析这个笑脸数据", llm_settings=settings)
    print(f"intent: {report1.response_type}")
    if report1.llm_summary:
        print(f"summary: {report1.llm_summary[:100]}...")

    # Task 2: follow-up question — should NOT re-run algorithms
    print("\n--- Task 2: follow-up question ---")
    report2 = supervisor.run(
        dataset,
        "具体解释一下为什么选择这个算法？它的轮廓系数是多少？",
        llm_settings=settings,
    )
    print(f"intent: {report2.response_type}")
    print(f"answer: {report2.llm_summary}")

    assert report2.response_type == "FOLLOW_UP", (
        "Agent should recognise follow-up and skip algorithm re-run"
    )


if __name__ == "__main__":
    test_follow_up()
