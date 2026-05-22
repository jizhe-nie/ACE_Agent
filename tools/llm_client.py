"""
tools/llm_client.py
===================
LLM client abstraction for ACE Agent (Phase 0).

Supports multiple cloud providers via a common LLMProvider ABC.
Implements automatic fallback: primary -> secondary on timeout / 429 / 5xx.
Token counting is approximate (tiktoken cl100k_base encoding).

Providers implemented:
  - OpenAIProvider  : OpenAI and any OpenAI-compatible endpoint (incl. DeepSeek)
  - DeepSeekProvider: DeepSeek-specific defaults layered on top of OpenAIProvider
  - DashScopeProvider: Alibaba DashScope (OpenAI-compat mode)

Fallback events are logged to the LLM trace file (P0-3).
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import tiktoken

# ---------------------------------------------------------------------------
# Trace writer (P0-3 integration)
# ---------------------------------------------------------------------------
_TRACE_PATH = Path(__file__).resolve().parents[1] / "outputs" / "llm_trace.jsonl"
_trace_lock = threading.Lock()


def _write_trace(record: dict) -> None:
    """Append one JSON-Lines record to outputs/llm_trace.jsonl (thread-safe)."""
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _trace_lock, open(_TRACE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Settings dataclass (backward compatible)
# ---------------------------------------------------------------------------
@dataclass
class LLMSettings:
    provider: str = "DeepSeek"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    enabled: bool = True
    fast_audit: bool = False  # Phase 5.3: skip bootstrap, Hopkins + CVI only
    deep_mode: bool = False  # Phase 8: allow heavy AE/DEC pipelines on large data
    extra_body: dict[str, Any] | None = None  # e.g. {"thinking": {"type": "disabled"}}
    reasoning_effort: str | None = None  # e.g. "high", "max" (DeepSeek V4)

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.base_url.strip()) and bool(self.api_key.strip()) and bool(self.model.strip())


@dataclass
class MultiLLMConfig:
    """Three-LLM configuration for ACE Agent pipeline stages.

    Each field holds the LLMSettings for a specific role.  If a role is
    ``None``, the caller should use the *worker* settings as fallback.
    """

    router: LLMSettings | None = None   # LLM-1: intent routing (cheap/fast)
    worker: LLMSettings | None = None   # LLM-2: code generation (parallel)
    reflection: LLMSettings | None = None  # LLM-3: audit + summary (reasoning-heavy)

    def get_router(self) -> LLMSettings:
        return self.router if self.router else self.get_worker()

    def get_worker(self) -> LLMSettings:
        if self.worker:
            return self.worker
        if self.reflection:
            return self.reflection
        return LLMSettings()

    def get_reflection(self) -> LLMSettings:
        return self.reflection if self.reflection else self.get_worker()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base (works for GPT-4 / DeepSeek)."""
    try:
        return len(_ENCODING.encode(text))
    except Exception:
        # Fallback: rough character-based estimate
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Abstract base provider
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'DeepSeek', 'DashScope')."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Active model identifier."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """
        Send chat messages and return the assistant reply.

        Args:
            messages: List of {'role': ..., 'content': ...} dicts.
            **kwargs: Optional overrides (temperature, max_tokens, etc.).

        Returns:
            Assistant reply string.

        Raises:
            requests.HTTPError: on 4xx/5xx responses.
            requests.Timeout: on network timeout.
        """

    def count_tokens(self, text: str) -> int:
        """Approximate token count (cl100k_base)."""
        return count_tokens(text)


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (covers OpenAI, DeepSeek, DashScope, etc.)
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider(LLMProvider):
    """
    Generic OpenAI-compatible chat/completions provider.

    Works for any endpoint following the OpenAI REST protocol.
    """

    # Cost per 1K tokens in USD — override in subclasses.
    # Zero means "not tracked / free tier"
    _INPUT_COST_PER_1K: float = 0.0
    _OUTPUT_COST_PER_1K: float = 0.0

    def __init__(self, settings: LLMSettings):
        self._settings = settings

    @property
    def name(self) -> str:
        return self._settings.provider

    @property
    def model(self) -> str:
        return self._settings.model

    @property
    def input_cost_per_1k(self) -> float:
        return self._INPUT_COST_PER_1K

    @property
    def output_cost_per_1k(self) -> float:
        return self._OUTPUT_COST_PER_1K

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        if not self._settings.is_configured:
            raise RuntimeError(f"Provider {self.name} is not configured (missing url/key/model).")

        url = f"{self._settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "temperature": kwargs.get("temperature", self._settings.temperature),
            "messages": messages,
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        # Merge extra_body from settings (e.g. DeepSeek thinking control)
        _eb = self._settings.extra_body
        if _eb:
            payload.update(_eb)

        # DeepSeek V4 reasoning_effort control
        if self._settings.reasoning_effort:
            payload["reasoning_effort"] = self._settings.reasoning_effort

        response = requests.post(url, headers=headers, json=payload, timeout=kwargs.get("timeout", 90))
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return str(content).strip() if content else ""


# ---------------------------------------------------------------------------
# Named provider subclasses with cost tables
# ---------------------------------------------------------------------------
class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek cloud provider (OpenAI-compatible, CN-available)."""

    # deepseek-chat pricing (2025-04 public): $0.14/1M input, $0.28/1M output
    _INPUT_COST_PER_1K: float = 0.00014
    _OUTPUT_COST_PER_1K: float = 0.00028


class DashScopeProvider(OpenAICompatibleProvider):
    """Alibaba DashScope (qwen-* family) via OpenAI-compat endpoint."""

    # qwen-plus pricing: roughly $0.4/1M tokens (rough approx)
    _INPUT_COST_PER_1K: float = 0.0004
    _OUTPUT_COST_PER_1K: float = 0.0012


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI cloud provider."""

    # gpt-4o pricing: $5/1M input, $15/1M output
    _INPUT_COST_PER_1K: float = 0.005
    _OUTPUT_COST_PER_1K: float = 0.015


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------
_PROVIDER_CLASSES: dict[str, type[OpenAICompatibleProvider]] = {
    "DeepSeek": DeepSeekProvider,
    "DashScope": DashScopeProvider,
    "OpenAI": OpenAIProvider,
    "Moonshot": OpenAICompatibleProvider,
    "Gemini": OpenAICompatibleProvider,
}


def make_provider(settings: LLMSettings) -> LLMProvider:
    """Instantiate the correct provider subclass from an LLMSettings object."""
    cls = _PROVIDER_CLASSES.get(settings.provider, OpenAICompatibleProvider)
    return cls(settings)


# ---------------------------------------------------------------------------
# UniversalLLMClient — facade with tracing + fallback (P0-3 / P0-5)
# ---------------------------------------------------------------------------
class UniversalLLMClient:
    """
    Stateful LLM client facade.

    Features:
    - Calls primary provider; falls back to secondary on timeout/429/5xx
    - Emits structured trace records to outputs/llm_trace.jsonl
    - Exposes get_cost_summary() for the Streamlit sidebar
    """

    # Retryable HTTP status codes that trigger provider fallback
    _FALLBACK_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        settings: LLMSettings,
        fallback_settings: LLMSettings | None = None,
        *,
        caller: str = "unknown",
    ):
        self._primary = make_provider(settings)
        self._fallback = make_provider(fallback_settings) if fallback_settings else None
        self._caller = caller

        # Session-level cost accumulators
        self._call_count: int = 0
        self._retry_count: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        *,
        caller: str | None = None,
        attempt: int = 1,
    ) -> str | None:
        """
        Send a chat request with automatic fallback and cost tracing.

        Args:
            messages: Chat history (role/content dicts).
            system_prompt: Optional system message prepended.
            caller: Override the caller label for this invocation.
            attempt: Retry attempt number (1 = first try; used by self-healing loop).

        Returns:
            Assistant reply string, or an error string starting with "Error:".
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        effective_caller = caller or self._caller
        is_retry = attempt > 1

        prompt_tokens = sum(count_tokens(m.get("content", "")) for m in full_messages)

        t0 = time.monotonic()
        provider_used = self._primary.name
        fallback_triggered = False
        error_msg: str | None = None
        reply = ""

        try:
            reply = self._primary.chat(full_messages)
        except Exception as exc:
            error_msg = str(exc)
            # Decide whether to fallback
            status_code = self._http_status(exc)
            should_fallback = (
                self._fallback is not None
                and self._fallback._settings.is_configured  # type: ignore[attr-defined]
                and (
                    status_code in self._FALLBACK_STATUS_CODES
                    or "timeout" in error_msg.lower()
                    or "Timeout" in error_msg
                )
            )
            if should_fallback:
                fallback_triggered = True
                provider_used = self._fallback.name  # type: ignore[union-attr]
                fallback_record = {
                    "event": "provider_fallback",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "from_provider": self._primary.name,
                    "to_provider": self._fallback.name,  # type: ignore[union-attr]
                    "reason": error_msg,
                    "caller": effective_caller,
                }
                _write_trace(fallback_record)
                try:
                    reply = self._fallback.chat(full_messages)  # type: ignore[union-attr]
                    error_msg = None
                except Exception as exc2:
                    error_msg = f"Primary: {error_msg} | Fallback: {exc2}"
                    reply = f"Error: {error_msg}"
            else:
                reply = f"Error during LLM call: {error_msg}"

        latency_ms = int((time.monotonic() - t0) * 1000)
        completion_tokens = count_tokens(reply) if reply and not reply.startswith("Error:") else 0

        # Update accumulators
        self._call_count += 1
        if is_retry:
            self._retry_count += 1
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Estimate cost based on the provider actually used
        active_provider: LLMProvider = self._fallback if fallback_triggered else self._primary  # type: ignore[assignment]
        cost_usd = 0.0
        if isinstance(active_provider, OpenAICompatibleProvider):
            cost_usd = (
                prompt_tokens / 1000.0 * active_provider.input_cost_per_1k
                + completion_tokens / 1000.0 * active_provider.output_cost_per_1k
            )
        self._total_cost_usd += cost_usd

        # Emit trace record
        trace_record: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self._primary.model,
            "provider": provider_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "caller": effective_caller,
            "is_retry": is_retry,
            "attempt": attempt,
            "fallback_triggered": fallback_triggered,
            "cost_usd": round(cost_usd, 6),
        }
        if error_msg:
            trace_record["error"] = error_msg
        _write_trace(trace_record)

        return reply if reply else None

    def summarize_report(self, summary_payload: dict[str, Any]) -> str | None:
        """Legacy API: generate a Chinese summary from a clustering result payload."""
        system_prompt = (
            "你是一个 ACE 聚类分析助手。你的任务是根据聚类实验结果生成专业的中文总结。\n"
            "关键要求：\n"
            "1. 必须优先响应用户意图 (user_intent)。如果用户要求使用特定算法，你的总结应重点围绕该算法的表现展开。\n"
            "2. 如果用户指定的算法表现不如其他算法（如 KMeans），应客观指出，并简要对比差异，但不能忽略用户的要求。\n"
            "3. 提到优胜算法 (best_algo) 及其评分，并给出后续行动建议。\n"
            "4. 保持专业、简洁、有说服力。不要编造数据。\n"
            "5. 关于排名指标 (score_source)：\n"
            "   - 当 score_source == 'ari' 时，排名反映的是与真实标签的一致性（Adjusted Rand Index），\n"
            "     这是带标签场景下的正确度量，对非凸簇形（如半月形）也公正。\n"
            "   - 当 score_source == 'silhouette' 时，排名反映的是内部凝聚度（轮廓系数），\n"
            "     注意它对非凸簇结构存在结构性偏差（会偏爱 KMeans/GMM 这类球形聚类）。\n"
            "   在总结里点明本次排名使用了哪种度量，以便用户理解为什么冠军算法胜出。\n"
            "   若 score_source 缺失则按原有行为处理（不要强行提及）。\n"
            "6. **关键: 若 all_algorithms_failed == true, 你必须以红色预警开头:**\n"
            "   '### ⛔ FAILURE / NO VALID CLUSTERS — 聚类全面失败 (ARI < 0.2)'\n"
            "   并在总结中明确指出: 所有算法在欧氏空间中均无法有效捕捉数据结构，\n"
            "   该数据需要非欧氏距离度量或深度学习方法。"
        )
        messages = [{"role": "user", "content": json.dumps(summary_payload, ensure_ascii=False)}]
        return self.chat_completion(messages, system_prompt, caller="summarize_report")

    def get_cost_summary(self) -> dict[str, Any]:
        """Return accumulated cost metrics for this client instance."""
        return {
            "call_count": self._call_count,
            "retry_count": self._retry_count,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "estimated_cost_usd": round(self._total_cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _http_status(exc: Exception) -> int:
        """Extract HTTP status code from a requests.HTTPError if available."""
        if hasattr(exc, "response") and exc.response is not None:
            return int(exc.response.status_code)
        return 0


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
class OpenAICompatibleClient(UniversalLLMClient):
    """Deprecated alias kept for backward compatibility."""
