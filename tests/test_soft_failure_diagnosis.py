"""
tests/test_soft_failure_diagnosis.py
====================================
Verify that the soft-failure branch in
``expert_sub_agents/base.BaseExpert.execute_with_self_correction``
injects a *pointed* diagnosis into the fix prompt based on what the
old code contained:

- Mentions ``__main__`` guard when the old code contains
  ``if __name__``.
- Mentions ``def main`` / ``def run`` when the old code defines one
  of those functions without calling it.
- Falls back to a generic hint when neither pattern is present.

We patch ``UniversalLLMClient.chat_completion`` to capture the
messages passed to the fix call, and supply a tiny concrete
subclass of ``BaseExpert`` that returns a canned initial code
string (each scenario triggers the soft-failure path because the
canned code either never executes, or doesn't write to
``artifacts``).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

from ACE_Agent.agent_core.schemas import DatasetBundle  # noqa: E402
from ACE_Agent.expert_sub_agents.base import BaseExpert  # noqa: E402
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient  # noqa: E402


class _StubExpert(BaseExpert):
    """Minimal expert used only for exercising the self-correction loop.

    The initial code is injected via ``__init__`` so each test can
    control which soft-failure scenario the loop sees.
    """

    def __init__(self, initial_code: str) -> None:
        super().__init__(key="stub", label="StubExpert")
        self._initial_code = initial_code

    def _generate_code(self, client, dataset, prompt):  # type: ignore[override]
        return self._initial_code


def _dataset() -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 2))
    return DatasetBundle(name="stub", X=X, y=None, description="stub dataset")


def _settings() -> LLMSettings:
    # Not actually dialed out — we patch chat_completion.
    return LLMSettings(
        provider="DeepSeek",
        base_url="http://unused",
        api_key="unused",
        model="unused",
        enabled=True,
    )


def _run_and_capture(initial_code: str) -> list[dict]:
    """Execute the self-correction loop with ``chat_completion`` patched.

    Returns the list of messages passed to every fix call so tests
    can assert on their contents.
    """
    captured: list[dict] = []

    def fake_chat_completion(self, messages, system_prompt=None, *, caller=None, attempt=1):
        # Record every call; return the old code unchanged so the loop
        # keeps soft-failing on every retry (giving us up to MAX_RETRIES-1
        # fix calls per run).
        captured.append(
            {
                "messages": messages,
                "system_prompt": system_prompt,
                "caller": caller,
                "attempt": attempt,
            }
        )
        # Return the original code so artifacts stay empty and the loop
        # enters the soft-failure branch again on the next attempt.
        # That's fine — we only need the FIRST fix call for assertions.
        return initial_code

    expert = _StubExpert(initial_code)
    with patch.object(
        UniversalLLMClient, "chat_completion", autospec=True, side_effect=fake_chat_completion
    ):
        expert.execute_with_self_correction(_dataset(), "stub prompt", _settings())
    return captured


def _fix_user_content(call_records: list[dict]) -> str:
    """Return the user-message content of the FIRST fix call."""
    # The first recorded call may be the initial generation call in
    # some subclasses, but our _StubExpert doesn't use the client for
    # generation — it returns the canned string directly. So every
    # captured call is a fix call.
    assert call_records, "expected at least one fix call"
    msgs = call_records[0]["messages"]
    # _fix_code sends a single user message
    user_msg = next(m for m in msgs if m.get("role") == "user")
    return user_msg["content"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSoftFailureDiagnosis:
    def test_main_guard_triggers_main_hint(self) -> None:
        code = (
            "if __name__ == '__main__':\n"
            "    artifacts['KMeans'] = {'labels': [0], 'metrics': {}, 'plot_path': ''}\n"
        )
        calls = _run_and_capture(code)
        content = _fix_user_content(calls)
        assert "__main__" in content
        assert "守卫" in content or "顶层" in content

    def test_def_main_triggers_function_hint(self) -> None:
        code = (
            "def main():\n"
            "    artifacts['KMeans'] = {'labels': [0], 'metrics': {}, 'plot_path': ''}\n"
        )
        calls = _run_and_capture(code)
        content = _fix_user_content(calls)
        assert "def main" in content or "def run" in content
        assert "顶层" in content

    def test_def_run_triggers_function_hint(self) -> None:
        code = (
            "def run():\n"
            "    artifacts['KMeans'] = {'labels': [0], 'metrics': {}, 'plot_path': ''}\n"
        )
        calls = _run_and_capture(code)
        content = _fix_user_content(calls)
        assert "def main" in content or "def run" in content

    def test_generic_hint_when_no_known_pattern(self) -> None:
        # Empty-ish code: runs cleanly, writes nothing, contains neither
        # `__main__` nor `def main`/`def run`.
        code = "x = 1 + 1\n"
        calls = _run_and_capture(code)
        content = _fix_user_content(calls)
        # The specific bullet-point diagnoses should NOT fire; the generic
        # fallback should. The boilerplate tail of the hint always mentions
        # __main__/函数 to forbid them, so we scan only the diagnosis
        # bullet line (the one starting with "- ").
        diagnosis_lines = [ln for ln in content.splitlines() if ln.startswith("- ")]
        assert diagnosis_lines, "expected at least one diagnosis bullet"
        combined = "\n".join(diagnosis_lines)
        assert "__main__" not in combined
        assert "def main" not in combined and "def run" not in combined
        assert "顶层直接调用" in combined or "请确保" in combined

    def test_both_patterns_both_hints(self) -> None:
        code = (
            "def main():\n"
            "    pass\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        calls = _run_and_capture(code)
        content = _fix_user_content(calls)
        assert "__main__" in content
        assert "def main" in content or "def run" in content
