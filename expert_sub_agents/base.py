"""
expert_sub_agents/base.py
=========================
Base class for all ACE expert agents.

Implements the Think-Act-Fix self-healing loop (up to 3 attempts).
Passes `caller` and `attempt` metadata to the LLM client so that
P0-3 trace records can distinguish initial calls from retries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.tools.coder_sandbox import CoderSandbox, SandboxResourceExceeded
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```python / ```py / ```) that LLMs often wrap code in.

    Handles leading ```python, ```py, or bare ``` at the start, and a trailing ```
    at the end. Whitespace is trimmed on both sides before and after fence removal.
    """
    if text is None:
        return ""
    stripped = text.strip()
    # Remove leading fence
    if stripped.startswith("```python"):
        stripped = stripped[len("```python") :]
    elif stripped.startswith("```py"):
        stripped = stripped[len("```py") :]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    # Remove trailing fence
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


class BaseExpert(ABC):
    """Expert agent base class: supports Think -> Act -> Fix loop."""

    MAX_RETRIES: int = 3
    # Pre-injected names exposed in the sandbox namespace alongside the core
    # sklearn set.  Subclasses override this to make additional packages
    # (umap, torch, …) available without import statements.
    PRE_INJECT: dict[str, Any] = {}

    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label
        self.sandbox = CoderSandbox()
        self.REQUIRES_LLM: bool = True

    def execute_with_self_correction(
        self,
        dataset: DatasetBundle,
        prompt: str,
        settings: LLMSettings,
        constraints: dict | None = None,
    ) -> list[AlgorithmRunResult]:
        """
        Core method: run the expert with automatic code self-correction.

        The LLM is called with caller='{key}:generate' on the first attempt,
        and caller='{key}:fix:N' on the Nth retry so that P0-3 trace records
        can distinguish generation from repair calls.
        """
        client = UniversalLLMClient(settings, caller=f"{self.key}:generate")
        results: list[AlgorithmRunResult] = []
        logs = [f"[{self.label}] 开始分析数据特征并生成代码..."]

        # 1. Initial code generation
        raw_code = self._generate_code(client, dataset, prompt, constraints=constraints)
        code = _strip_code_fences(raw_code)

        # 2. Attempt to run + fix (up to MAX_RETRIES times)
        for attempt in range(1, self.MAX_RETRIES + 1):
            logs.append(f"[{self.label}] 第 {attempt} 次尝试运行代码...")
            try:
                run_result = self.sandbox.execute(
                    code,
                    dataset.X,
                    dataset.y,
                    pre_inject=self.PRE_INJECT or None,
                    display_name=dataset.display_name,
                    expected_clusters=dataset.metadata.get("expected_clusters", 3) if dataset.metadata else 3,
                    metadata=dataset.metadata,
                )
            except SandboxResourceExceeded as exc:
                logs.append(f"[{self.label}] 沙箱资源超限 ({exc.reason}): {exc.detail}")
                # Surface resource limit to caller; do not retry
                break

            if run_result["success"] and run_result["artifacts"]:
                # 只有代码运行成功 AND artifacts 非空才视为真正成功
                logs.append(f"[{self.label}] 运行成功！")
                for algo, info in run_result["artifacts"].items():
                    # 跳过错误占位条目（key 以 _error 结尾）
                    if algo.endswith("_error"):
                        continue
                    plot_raw = info.get("plot_path", "")
                    # 若 plot_path 为空或文件不存在，使用占位说明，不崩溃
                    plot_path_obj: Path
                    if plot_raw and Path(plot_raw).exists():
                        plot_path_obj = Path(plot_raw)
                    else:
                        plot_path_obj = Path(plot_raw) if plot_raw else Path("")
                    results.append(
                        AlgorithmRunResult(
                            algorithm_name=algo,
                            expert_key=self.key,
                            expert_label=self.label,
                            labels=info.get("labels"),
                            metrics=info.get("metrics", {}),
                            plot_path=plot_path_obj,
                            code=code,
                        )
                    )
                break
            elif run_result["success"] and not run_result["artifacts"]:
                # 软失败：代码运行成功但未产出 artifacts
                soft_fail_msg = (
                    f"[{self.label}] 软失败：代码运行成功但未产出 artifacts[algo_name]。"
                    f" 请确保代码将结果写入 artifacts 字典（而非其他变量名）。"
                )
                logs.append(soft_fail_msg)
                if attempt < self.MAX_RETRIES:
                    logs.append(f"[{self.label}] 正在注入 artifacts 约定并重写代码 (第{attempt}次重试)...")
                    fix_client = UniversalLLMClient(
                        settings,
                        caller=f"{self.key}:fix:{attempt}",
                    )
                    diagnosis_hints = []
                    if "artifacts =" in code or "artifacts=" in code:
                        diagnosis_hints.append(
                            "- 你的代码中包含 `artifacts = {}` 之类的赋值语句，这会**覆盖**沙箱注入的 artifacts 字典，导致结果丢失。请删除这行，直接对 artifacts 字典写入即可（如 `artifacts['algo'] = {...}`）。"
                        )
                    if "if __name__" in code:
                        diagnosis_hints.append(
                            "- 你在代码里使用了 `if __name__ == \"__main__\":` 守卫，但沙箱环境下 __name__ != '__main__'，导致主逻辑从未执行。请删除该守卫，让代码在顶层直接运行。"
                        )
                    if "def main" in code or "def run" in code:
                        diagnosis_hints.append(
                            "- 你定义了 `def main()` 或 `def run()` 函数但没有在代码末尾调用它。请删除函数包装，让逻辑直接在顶层执行。"
                        )
                    diagnosis = (
                        "\n".join(diagnosis_hints)
                        if diagnosis_hints
                        else "- 请确保在顶层直接调用算法并写入 artifacts。"
                    )
                    artifacts_hint = (
                        "代码运行成功但没有向 artifacts 字典写入任何内容。根因分析：\n"
                        + diagnosis
                        + "\n\n必须将结果写入 artifacts[algo_name] = "
                        + '{"labels": <list>, "metrics": {"score": float, ...}, "plot_path": str}；'
                        + "禁止用其他变量名，禁止把逻辑藏在 __main__ 守卫或未调用的函数里。"
                    )
                    code = self._fix_code(fix_client, code, artifacts_hint, attempt=attempt)
                else:
                    logs.append(f"[{self.label}] 重试次数耗尽，artifacts 始终为空，任务失败。")
            else:
                logs.append(f"[{self.label}] 运行失败，错误信息: {run_result['error'][:200]}")
                if attempt < self.MAX_RETRIES:
                    logs.append(f"[{self.label}] 正在分析错误并重写代码 (第{attempt}次重试)...")
                    fix_client = UniversalLLMClient(
                        settings,
                        caller=f"{self.key}:fix:{attempt}",
                    )
                    code = self._fix_code(fix_client, code, run_result["error"], attempt=attempt)
                else:
                    logs.append(f"[{self.label}] 重试次数耗尽，任务失败。")

        self.last_logs = logs
        return results

    @abstractmethod
    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
        constraints: dict | None = None,
    ) -> str:
        """Each expert implements its own code generation logic.

        **Artifacts contract (all experts must follow):**
        Generated code MUST write results into the ``artifacts`` dict injected
        by the sandbox.  Required format::

            artifacts[algo_name] = {
                "labels": <list or np.ndarray>,
                "metrics": {"score": float, ...},
                "plot_path": str,
            }

        Forbidden: writing to any other variable name.
        """

    @staticmethod
    def _inject_constraints_prompt(constraints: dict | None) -> str:
        """Build a constraint instruction block for the LLM system prompt.

        Used by Critic 2.0 to inject retry_constraints into code generation.
        """
        if not constraints:
            return ""
        lines = ["\n## 约束指令（必须严格遵守）"]
        if constraints.get("force_k"):
            lines.append(f"- 聚类数 k 必须为 {constraints['force_k']}")
        if constraints.get("blocked_algorithms"):
            blocked = ", ".join(constraints["blocked_algorithms"])
            lines.append(f"- 禁止使用以下算法：{blocked}")
        if constraints.get("force_preprocessing"):
            lines.append(f"- 必须对数据应用 {constraints['force_preprocessing']} 预处理")
        return "\n".join(lines) + "\n"

    def _fix_code(
        self,
        client: UniversalLLMClient,
        old_code: str,
        error: str,
        *,
        attempt: int = 1,
    ) -> str:
        """Generic code-fix logic using the LLM."""
        system_prompt = (
            "你是一个 Python 调试专家。修正以下代码以解决给出的错误信息。只返回修正后的 Python 代码，不要解释。"
        )
        user_input = f"原始代码：\n{old_code}\n\n错误信息：\n{error}"
        reply = client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
            caller=f"{self.key}:fix:{attempt}",
            attempt=attempt,
        )
        reply_text = reply or old_code
        # Strip markdown code fences that LLMs sometimes wrap code in
        return _strip_code_fences(reply_text)
