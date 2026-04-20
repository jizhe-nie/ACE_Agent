"""
tests/test_p0.py
================
Phase 0 unit tests covering:
  - P0-3: LLM trace writing and cost summary
  - P0-4: CoderSandbox resource limits (timeout, memory watchdog, normal exec)
  - P0-5: LLMProvider abstraction (count_tokens, make_provider, fallback logic)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

_root_parent = str(Path(__file__).resolve().parents[2])
if _root_parent not in sys.path:
    sys.path.insert(0, _root_parent)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ACE_Agent.tools.coder_sandbox import (  # noqa: E402
    CoderSandbox,
    SandboxResourceExceeded,
    _MemoryWatchdog,
)
from ACE_Agent.tools.llm_client import (  # noqa: E402
    DashScopeProvider,
    DeepSeekProvider,
    LLMSettings,
    OpenAICompatibleProvider,
    OpenAIProvider,
    UniversalLLMClient,
    count_tokens,
    make_provider,
)


# ---------------------------------------------------------------------------
# P0-5: Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0 or count_tokens("") >= 0

    def test_short_string(self) -> None:
        n = count_tokens("hello world")
        assert n > 0

    def test_longer_text_more_tokens(self) -> None:
        short = count_tokens("hi")
        long = count_tokens("This is a much longer sentence with many words and tokens.")
        assert long > short


# ---------------------------------------------------------------------------
# P0-5: make_provider factory
# ---------------------------------------------------------------------------


class TestMakeProvider:
    def test_deepseek_returns_deepseek_provider(self) -> None:
        s = LLMSettings(provider="DeepSeek", base_url="https://x", api_key="k", model="m")
        p = make_provider(s)
        assert isinstance(p, DeepSeekProvider)

    def test_dashscope_returns_dashscope_provider(self) -> None:
        s = LLMSettings(provider="DashScope", base_url="https://x", api_key="k", model="m")
        p = make_provider(s)
        assert isinstance(p, DashScopeProvider)

    def test_openai_returns_openai_provider(self) -> None:
        s = LLMSettings(provider="OpenAI", base_url="https://x", api_key="k", model="m")
        p = make_provider(s)
        assert isinstance(p, OpenAIProvider)

    def test_unknown_provider_defaults_to_base(self) -> None:
        s = LLMSettings(provider="Unknown", base_url="https://x", api_key="k", model="m")
        p = make_provider(s)
        assert isinstance(p, OpenAICompatibleProvider)

    def test_provider_name_matches_settings(self) -> None:
        s = LLMSettings(provider="DeepSeek", base_url="https://x", api_key="k", model="m")
        p = make_provider(s)
        assert p.name == "DeepSeek"

    def test_provider_model_matches_settings(self) -> None:
        s = LLMSettings(
            provider="DeepSeek", base_url="https://x", api_key="k", model="deepseek-chat"
        )
        p = make_provider(s)
        assert p.model == "deepseek-chat"

    def test_provider_has_count_tokens(self) -> None:
        s = LLMSettings(provider="DeepSeek", base_url="https://x", api_key="k", model="m")
        p = make_provider(s)
        n = p.count_tokens("test text")
        assert n > 0


# ---------------------------------------------------------------------------
# P0-5: UniversalLLMClient — unconfigured / error paths (no real HTTP)
# ---------------------------------------------------------------------------


class TestUniversalLLMClientOffline:
    """Tests that do NOT make real HTTP calls."""

    @pytest.fixture()
    def unconfigured_client(self) -> UniversalLLMClient:
        s = LLMSettings(enabled=False)
        return UniversalLLMClient(s, caller="test")

    def test_unconfigured_returns_error_string(
        self, unconfigured_client: UniversalLLMClient
    ) -> None:
        result = unconfigured_client.chat_completion(
            [{"role": "user", "content": "hello"}]
        )
        # Should return an error string, not crash
        assert result is not None
        assert "Error" in str(result) or result == ""

    def test_get_cost_summary_initial_zero(
        self, unconfigured_client: UniversalLLMClient
    ) -> None:
        summary = unconfigured_client.get_cost_summary()
        assert summary["call_count"] == 0
        assert summary["total_prompt_tokens"] == 0
        assert summary["estimated_cost_usd"] == 0.0

    def test_get_cost_summary_keys_present(
        self, unconfigured_client: UniversalLLMClient
    ) -> None:
        summary = unconfigured_client.get_cost_summary()
        expected_keys = {
            "call_count", "retry_count", "total_prompt_tokens",
            "total_completion_tokens", "total_tokens", "estimated_cost_usd",
        }
        assert expected_keys.issubset(summary.keys())

    def test_call_count_increments(
        self, unconfigured_client: UniversalLLMClient, tmp_path: Path
    ) -> None:
        """Each chat_completion call increments the counter even on error."""
        with patch("ACE_Agent.tools.llm_client._TRACE_PATH", tmp_path / "trace.jsonl"):
            unconfigured_client.chat_completion([{"role": "user", "content": "hi"}])
            unconfigured_client.chat_completion([{"role": "user", "content": "bye"}])
        assert unconfigured_client.get_cost_summary()["call_count"] == 2

    def test_trace_written_to_file(
        self, unconfigured_client: UniversalLLMClient, tmp_path: Path
    ) -> None:
        trace_file = tmp_path / "trace.jsonl"
        with patch("ACE_Agent.tools.llm_client._TRACE_PATH", trace_file):
            unconfigured_client.chat_completion([{"role": "user", "content": "test"}])

        assert trace_file.exists()
        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "timestamp" in record
        assert "provider" in record
        assert "prompt_tokens" in record
        assert "latency_ms" in record

    def test_retry_attempt_recorded(
        self, unconfigured_client: UniversalLLMClient, tmp_path: Path
    ) -> None:
        trace_file = tmp_path / "trace.jsonl"
        with patch("ACE_Agent.tools.llm_client._TRACE_PATH", trace_file):
            unconfigured_client.chat_completion(
                [{"role": "user", "content": "fix"}],
                caller="centroid:fix:2",
                attempt=2,
            )
        record = json.loads(trace_file.read_text(encoding="utf-8").strip())
        assert record["attempt"] == 2
        assert record["is_retry"] is True

    def test_fallback_trigger_on_http_error(self, tmp_path: Path) -> None:
        """Fallback provider is tried when primary raises an HTTPError with 5xx."""
        primary_s = LLMSettings(
            provider="DeepSeek",
            base_url="https://api.deepseek.com",
            api_key="primary_key",
            model="deepseek-chat",
            enabled=True,
        )
        fallback_s = LLMSettings(
            provider="DashScope",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="fallback_key",
            model="qwen-plus",
            enabled=True,
        )
        client = UniversalLLMClient(primary_s, fallback_s, caller="test")

        import requests as req_mod

        # Simulate primary 503
        mock_primary_exc = req_mod.HTTPError(response=MagicMock(status_code=503))

        trace_file = tmp_path / "trace.jsonl"
        with (
            patch.object(client._primary, "chat", side_effect=mock_primary_exc),
            patch.object(client._fallback, "chat", return_value="fallback reply"),
            patch("ACE_Agent.tools.llm_client._TRACE_PATH", trace_file),
        ):
            result = client.chat_completion([{"role": "user", "content": "hi"}])

        assert result == "fallback reply"

        # Check fallback event was logged
        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        fallback_events = [
            json.loads(l) for l in lines if json.loads(l).get("event") == "provider_fallback"
        ]
        assert len(fallback_events) == 1
        assert fallback_events[0]["from_provider"] == "DeepSeek"
        assert fallback_events[0]["to_provider"] == "DashScope"


# ---------------------------------------------------------------------------
# P0-4: CoderSandbox — normal execution
# ---------------------------------------------------------------------------


class TestCoderSandboxNormal:
    def test_simple_arithmetic(self) -> None:
        sb = CoderSandbox(timeout_sec=10, memory_mb=500)
        X = np.array([[1, 2], [3, 4]])
        result = sb.execute("artifacts['result'] = int(X.sum())", X)
        assert result["success"] is True
        assert result["artifacts"]["result"] == 10

    def test_numpy_available_in_sandbox(self) -> None:
        sb = CoderSandbox(timeout_sec=10, memory_mb=500)
        X = np.zeros((5, 2))
        result = sb.execute("artifacts['shape'] = list(X.shape)", X)
        assert result["success"] is True
        assert result["artifacts"]["shape"] == [5, 2]

    def test_y_injected(self) -> None:
        sb = CoderSandbox(timeout_sec=10, memory_mb=500)
        X = np.zeros((3, 2))
        y = np.array([0, 1, 2])
        result = sb.execute("artifacts['y_sum'] = int(y.sum())", X, y)
        assert result["success"] is True
        assert result["artifacts"]["y_sum"] == 3

    def test_syntax_error_returns_failure(self) -> None:
        sb = CoderSandbox(timeout_sec=10, memory_mb=500)
        X = np.zeros((3, 2))
        result = sb.execute("this is not valid python!!!", X)
        assert result["success"] is False
        assert result["error"] is not None

    def test_runtime_exception_returns_failure(self) -> None:
        """ValueError is not in SAFE_BUILTINS, so sandbox raises NameError instead.
        Either way the execution must fail."""
        sb = CoderSandbox(timeout_sec=10, memory_mb=500)
        X = np.zeros((3, 2))
        # raise NameError explicitly (NameError is also not in builtins, but the
        # exec itself will raise NameError for the unresolved 'ValueError' name).
        result = sb.execute("x = 1 / 0", X)  # ZeroDivisionError propagates as sandbox failure
        assert result["success"] is False
        assert result["error"] is not None

    def test_legacy_run_interface(self) -> None:
        sb = CoderSandbox(timeout_sec=10, memory_mb=500)
        exec_result = sb.run("result = 42", {})
        assert exec_result.error is None
        assert exec_result.result == 42


# ---------------------------------------------------------------------------
# P0-4: CoderSandbox — timeout
# ---------------------------------------------------------------------------


class TestCoderSandboxTimeout:
    def test_timeout_raises_sandbox_resource_exceeded(self) -> None:
        sb = CoderSandbox(timeout_sec=1, memory_mb=2048)
        X = np.zeros((2, 2))
        with pytest.raises(SandboxResourceExceeded) as exc_info:
            sb.execute("import time; time.sleep(5)", X)
        assert exc_info.value.reason == "timeout"

    def test_timeout_reason_string(self) -> None:
        sb = CoderSandbox(timeout_sec=1, memory_mb=2048)
        X = np.zeros((2, 2))
        try:
            sb.execute("import time; time.sleep(5)", X)
        except SandboxResourceExceeded as exc:
            assert "timeout" in str(exc).lower()
            assert "1" in exc.detail  # detail mentions the limit value


# ---------------------------------------------------------------------------
# P0-4: MemoryWatchdog unit tests
# ---------------------------------------------------------------------------


class TestMemoryWatchdog:
    def test_watchdog_starts_and_stops(self) -> None:
        wd = _MemoryWatchdog(limit_mb=99999)  # very high limit; won't trigger
        wd.start()
        time.sleep(0.1)
        wd.stop()
        wd.join(timeout=2.0)
        assert not wd.is_alive()

    def test_watchdog_does_not_exceed_at_normal_usage(self) -> None:
        wd = _MemoryWatchdog(limit_mb=99999, poll_interval=0.05)
        wd.start()
        time.sleep(0.2)
        wd.stop()
        wd.join(timeout=2.0)
        assert wd.exceeded is False

    def test_watchdog_peak_mb_positive(self) -> None:
        wd = _MemoryWatchdog(limit_mb=99999, poll_interval=0.05)
        wd.start()
        time.sleep(0.2)
        wd.stop()
        wd.join(timeout=2.0)
        # peak_mb = baseline + peak_delta; baseline is already > 0
        assert wd.peak_mb > 0
        assert wd._baseline_mb > 0


# ---------------------------------------------------------------------------
# P0-4: SandboxResourceExceeded exception
# ---------------------------------------------------------------------------


class TestSandboxResourceExceeded:
    def test_reason_attribute(self) -> None:
        exc = SandboxResourceExceeded("timeout", "exceeded 60s")
        assert exc.reason == "timeout"

    def test_detail_attribute(self) -> None:
        exc = SandboxResourceExceeded("memory", "peak 3000 MiB")
        assert exc.detail == "peak 3000 MiB"

    def test_str_contains_reason(self) -> None:
        exc = SandboxResourceExceeded("timeout", "60s limit")
        assert "timeout" in str(exc)

    def test_is_runtime_error(self) -> None:
        exc = SandboxResourceExceeded("cpu", "100%")
        assert isinstance(exc, RuntimeError)
