from __future__ import annotations

import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_core.schemas import (
    AlgorithmRunResult,
    DatasetBundle,
    ProfileReport,
    RoutingDecision,
    SupervisorReport,
)
from ACE_Agent.expert_sub_agents import build_expert_registry
from ACE_Agent.tools.latex_generator import LatexReportGenerator
from ACE_Agent.tools.llm_client import LLMSettings, UniversalLLMClient


class ACESupervisor:
    """主控编排器 (Orchestrator)：协调多代理完成复杂任务。

    P0.5 变更（2026-04-20）：
    - 专家注册表改用 build_expert_registry()，包含 zoo 专家（含 DBSCAN/HDBSCAN）。
    - 默认激活策略：centroid + topology + zoo（三家并行）。
    - dimension / deep_representation / multi_view 已注册但默认不激活；
    - Critic 重构为后验审计方（2026-04-29）：从并行池移除，在选出最优结果后
      独立运行 _execute_audit()，输出结构化 audit_report，不参与排名。
    - 新增 CODE_EXAMPLE 分流路径（只返回代码 Markdown，不走沙箱）。
    - 每个专家执行用 try/except 包裹，异常转为日志不中断编排。
    - _error_report 增强：汇总每个专家最后 3 行日志作为排错依据。
    """

    # 默认激活的专家 key（ensemble 在第 2 步由 _execute_ensemble 独立调用）
    _DEFAULT_ACTIVE_EXPERTS: list[str] = ["centroid", "topology", "zoo"]

    def __init__(self) -> None:
        self.router = MasterRouter()
        self.knowledge_engine = KnowledgeEngine()
        try:
            self.knowledge_engine.ingest_docs()
        except Exception as e:
            print(f"知识库初始化警告: {e}")

        self.memory: list[dict[str, Any]] = []
        self.last_report: SupervisorReport | None = None

        # 完整专家注册表（含 zoo, critic, dimension, deep）
        self.experts: dict[str, Any] = build_expert_registry()

    def run(
        self,
        dataset: DatasetBundle | None,
        user_prompt: str,
        llm_settings: LLMSettings,
        intent_data: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> SupervisorReport:
        """核心编排流程。"""
        # 1. 语义路由
        if not intent_data:
            intent_data = self.router.analyze_intent(user_prompt, self.memory, llm_settings)

        intent = str(intent_data.get("intent", "NEW_TASK")).upper()
        reasoning = intent_data.get("reasoning", "接收到指令")

        # RAG 增强：检索背景知识
        knowledge_context = self.knowledge_engine.query(user_prompt)
        if knowledge_context:
            trace_msg = "【RAG】成功检索到相关学术理论片段，已注入上下文。"
            user_prompt = f"{knowledge_context}\n用户指令: {user_prompt}"
        else:
            trace_msg = "【RAG】未在知识库中匹配到强相关的理论支持。"

        trace = [f"【主控】确认意图: {intent}", f"【逻辑】{reasoning}", trace_msg]

        # HITL约束检测
        if constraints and constraints.get("reference_labels"):
            trace.append(
                f"【HITL】检测到人工标注参考标签（{len(constraints['reference_labels'])} 个数据点），"
                "将作为约束传递给各专家。"
            )

        # 2. 意图分流
        if intent == "FOLLOW_UP":
            return self._handle_follow_up(user_prompt, llm_settings, trace)

        if intent == "CODE_EXAMPLE":
            return self._handle_code_example(user_prompt, llm_settings, trace)

        # 3. NEW_TASK：需要数据集
        if not dataset:
            return self._error_report(
                "意图识别为新任务，但未识别到数据。请先配置/预览数据，或直接进行学术提问。",
                trace,
                expert_logs={},
            )

        # Phase 1 增强：高维数据自动感知
        n_features = dataset.X.shape[1]
        active_experts = list(self._DEFAULT_ACTIVE_EXPERTS)
        if n_features > 2 and "dimension" not in active_experts:
            trace.append(f"【主控】检测到数据维度为 {n_features}，已自动激活维度专家。")
            active_experts.append("dimension")

        return self._execute_full_analysis(dataset, user_prompt, llm_settings, trace, active_experts, constraints=constraints)

    # ------------------------------------------------------------------
    # 完整分析流
    # ------------------------------------------------------------------

    def _execute_full_analysis(
        self,
        dataset: DatasetBundle,
        prompt: str,
        settings: LLMSettings,
        trace: list[str],
        active_experts: list[str],
        constraints: dict[str, Any] | None = None,
    ) -> SupervisorReport:
        """执行完整的自动化聚类实验流。"""
        output_dir = self._prepare_output_dir(dataset.name)
        all_results: list[AlgorithmRunResult] = []
        expert_logs: dict[str, list[str]] = {}

        for key in active_experts:
            expert = self.experts.get(key)
            if expert is None:
                trace.append(f"【主控】警告：专家 '{key}' 未在注册表中找到，跳过。")
                continue
            try:
                expert_results = expert.execute_with_self_correction(dataset, prompt, settings, constraints=constraints)
                all_results.extend(expert_results)
                trace.extend(expert.last_logs)
                expert_logs[key] = list(expert.last_logs)
            except Exception as exc:
                err_msg = f"【主控】专家 '{key}' 执行时发生未捕获异常: {exc}"
                trace.append(err_msg)
                expert_logs[key] = [err_msg]

        if not all_results:
            return self._error_report(
                "所有专家执行均失败。",
                trace,
                expert_logs=expert_logs,
            )

        # 排序与摘要
        ranking = sorted(all_results, key=lambda x: x.metrics.get("score", 0.0), reverse=True)
        best = ranking[0]

        # 后验审计：Critic 独立审查最优结果
        audit_report = self._execute_audit(best, dataset, settings, trace)

        # Critic 2.0 反馈闭环：审计→约束重试→复验
        retry_results = self._handle_audit_feedback(
            audit_report, dataset, prompt, settings, trace, active_experts
        )
        if retry_results:
            all_results.extend(retry_results)
            ranking = sorted(all_results, key=lambda x: x.metrics.get("score", 0.0), reverse=True)
            best = ranking[0]
            trace.append("【Critic 2.0】约束重试完成，已重新排名。")
            # Re-audit the new best result
            audit_report = self._execute_audit(best, dataset, settings, trace)

        # 集成共识：仅在 Critic 对单一最优结果有保留时触发
        _should_ensemble = True
        if audit_report is None:
            trace.append("【集成】无审计报告，保守触发集成共识。")
        elif (
            audit_report.get("endorsement") == "endorsed"
            and audit_report.get("confidence_level", 0.0) >= 0.75
        ):
            _should_ensemble = False
            trace.append(
                "【集成】Critic 已 endorsement 单一最优结果"
                f"（置信度 {audit_report.get('confidence_level', 0):.0%}），跳过集成共识。"
            )
        else:
            trace.append(
                f"【集成】Critic 裁决为 '{audit_report.get('endorsement')}'"
                f"（置信度 {audit_report.get('confidence_level', 0):.0%}），触发集成融合拯救。"
            )

        if _should_ensemble:
            consensus_result = self._execute_ensemble(all_results, dataset, trace)
            if consensus_result is not None:
                all_results.append(consensus_result)
                # Re-rank with consensus result included
                ranking = sorted(all_results, key=lambda x: x.metrics.get("score", 0.0), reverse=True)
                best = ranking[0]

        client = UniversalLLMClient(settings)
        summary = client.summarize_report(
            {
                "user_intent": prompt,
                "dataset": dataset.display_name,
                "best_algo": best.algorithm_name,
                "metrics": best.metrics,
                "score_source": best.metrics.get("score_source", "silhouette"),
                "all_results": [
                    {
                        "algo": r.algorithm_name,
                        "score": r.metrics.get("score", 0.0),
                        "score_source": r.metrics.get("score_source", "silhouette"),
                    }
                    for r in all_results
                ],
            }
        )

        report = SupervisorReport(
            dataset=dataset,
            routing=RoutingDecision(
                profile=ProfileReport(dataset.X.shape[0], dataset.X.shape[1], 0, 0, 0, False, False, False),
                selected_experts=[],
                trace=trace,
            ),
            dataset_plot_path=self._save_raw_plot(dataset, output_dir),
            output_dir=output_dir,
            results=all_results,
            ranking=ranking,
            executive_summary=summary or "聚类分析完成。",
            decision_trace=trace,
            audit_report=audit_report,
            response_type="CLUSTER_TASK",
        )

        # 自动生成 LaTeX（静默失败，不阻塞 UI；CODE_EXAMPLE 类型会在生成器内跳过）
        with contextlib.suppress(Exception):
            report.latex_path = LatexReportGenerator().generate(report)

        self.last_report = report
        self.memory.append({"role": "user", "content": prompt})
        self.memory.append({"role": "assistant", "content": report.executive_summary})
        return report

    # ------------------------------------------------------------------
    # 后验审计
    # ------------------------------------------------------------------

    def _execute_audit(
        self,
        winner: AlgorithmRunResult,
        dataset: DatasetBundle,
        settings: LLMSettings,
        trace: list[str],
    ) -> dict[str, Any] | None:
        """Run CriticExpert as a post-hoc auditor on the winning result.

        Returns an ``audit_report`` dict or ``None`` if the audit is unavailable.
        """
        critic = self.experts.get("critic")
        if critic is None:
            trace.append("【审计】Critic 专家未注册，跳过审计。")
            return None

        trace.append(f"【审计】对最优结果 '{winner.algorithm_name}' 启动独立后验审计...")
        try:
            audit = critic.execute_audit(winner, dataset, settings)
            if audit:
                endorsement = audit.get("endorsement", "?")
                confidence = audit.get("confidence_level", "?")
                trace.append(
                    f"【审计】完成 — 裁决: {endorsement}, 置信度: {confidence}"
                )
            else:
                trace.append("【审计】审计未产出有效报告。")
            return audit
        except Exception as exc:
            trace.append(f"【审计】Critic 审计过程异常: {exc}")
            return None

    # ------------------------------------------------------------------
    # Critic 2.0 反馈闭环
    # ------------------------------------------------------------------

    def _handle_audit_feedback(
        self,
        audit_report: dict | None,
        dataset: DatasetBundle,
        prompt: str,
        settings: LLMSettings,
        trace: list[str],
        active_experts: list[str],
    ) -> list[AlgorithmRunResult]:
        """Critic 2.0 closed-loop: RETRY with constraints, max 2 attempts.

        When the audit finds the best result untrustworthy (action=RETRY),
        re-dispatch the expert pool with constraint instructions derived
        from the audit findings.
        """
        if audit_report is None:
            return []

        action = audit_report.get("action", "CLEAR")
        if action == "CLEAR":
            return []
        if action == "WARN":
            trace.append("【Critic 2.0】审计裁决为 WARN，接受当前结果（不重试）。")
            return []

        # action == "RETRY"
        constraints = audit_report.get("retry_constraints", {})
        if not constraints:
            trace.append("【Critic 2.0】RETRY 但无有效约束，跳过重试。")
            return []

        trace.append(
            f"【Critic 2.0】审计裁决为 RETRY，启动约束重试..."
            f" force_k={constraints.get('force_k')},"
            f" blocked={constraints.get('blocked_algorithms')}"
        )

        for attempt in range(1, 3):  # max_retries=2
            trace.append(f"【Critic 2.0】第 {attempt}/2 次约束重试...")
            retry_results: list[AlgorithmRunResult] = []

            for key in active_experts:
                expert = self.experts.get(key)
                if expert is None:
                    continue
                try:
                    if attempt > 1:
                        import os
                        os.environ.setdefault("ACE_SANDBOX_TIMEOUT_SEC", "120")
                    results = expert.execute_with_self_correction(
                        dataset, prompt, settings, constraints=constraints,
                    )
                    retry_results.extend(results)
                    trace.extend(expert.last_logs)
                except Exception as exc:
                    trace.append(
                        f"【Critic 2.0】专家 '{key}' 约束重试异常: {exc}"
                    )

            if retry_results:
                trace.append(
                    f"【Critic 2.0】第 {attempt} 次重试产出 {len(retry_results)} 个结果。"
                )
                return retry_results

            trace.append(f"【Critic 2.0】第 {attempt} 次重试未产出有效结果。")

        return []

    # ------------------------------------------------------------------
    # 集成共识
    # ------------------------------------------------------------------

    def _execute_ensemble(
        self,
        all_results: list[AlgorithmRunResult],
        dataset: DatasetBundle,
        trace: list[str],
    ) -> AlgorithmRunResult | None:
        """Run EnsembleConsensusExpert to fuse all expert labels.

        Builds a co-association matrix from all valid result labels and
        produces consensus labels via hierarchical clustering.

        Returns ``None`` if the ensemble expert is unavailable or fewer
        than 2 valid label sets exist.
        """
        ensemble = self.experts.get("ensemble")
        if ensemble is None:
            trace.append("【集成】Ensemble 专家未注册，跳过共识融合。")
            return None

        valid_count = sum(
            1 for r in all_results
            if getattr(r, "labels", None) is not None
            and len(getattr(r, "labels", [])) > 0
        )
        if valid_count < 2:
            trace.append(f"【集成】有效标签集不足 ({valid_count} < 2)，跳过共识融合。")
            return None

        trace.append(f"【集成】对 {valid_count} 套专家标签启动 Co-association 共识融合...")
        try:
            result = ensemble.execute_ensemble(all_results, dataset)
            if result is not None:
                trace.append(
                    f"【集成】完成 — 融合 {result.metrics.get('n_experts_fused', '?')} 位专家, "
                    f"一致性={result.metrics.get('agreement', 0):.3f}"
                )
            else:
                trace.append("【集成】共识融合未产出有效结果。")
            return result
        except Exception as exc:
            trace.append(f"【集成】共识融合异常: {exc}")
            return None

    # ------------------------------------------------------------------
    # FOLLOW_UP 路径
    # ------------------------------------------------------------------

    def _handle_follow_up(self, prompt: str, settings: LLMSettings, trace: list[str]) -> SupervisorReport:
        """纯 LLM 驱动的追问或学术咨询处理。"""
        client = UniversalLLMClient(settings)

        if self.last_report:
            context = {
                "last_summary": self.last_report.executive_summary,
                "ranking": [
                    {"algo": r.algorithm_name, "score": r.metrics.get("score")} for r in self.last_report.ranking
                ],
            }
            system_msg = f"你是一个数据科学专家。请基于以下聚类背景及检索到的知识回答用户问题：\n{context}"
        else:
            system_msg = "你是一个数据科学专家。请基于检索到的学术背景知识回答用户的理论咨询。"

        res = client.chat_completion([{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}])
        trace.append("【主控】正在基于知识库与会话上下文进行深度解析...")

        report = SupervisorReport(
            dataset=(
                self.last_report.dataset
                if self.last_report
                else DatasetBundle("Consultation", np.array([[0, 0]]), None)
            ),
            routing=(self.last_report.routing if self.last_report else RoutingDecision(None, [], trace)),
            dataset_plot_path=(self.last_report.dataset_plot_path if self.last_report else Path("")),
            output_dir=self.last_report.output_dir if self.last_report else Path(""),
            results=[],
            ranking=self.last_report.ranking if self.last_report else [],
            executive_summary=res or "无法生成回答。",
            decision_trace=trace,
            response_type="FOLLOW_UP",
        )
        return report

    # ------------------------------------------------------------------
    # CODE_EXAMPLE 路径（P0.5-C 新增）
    # ------------------------------------------------------------------

    def _handle_code_example(self, prompt: str, settings: LLMSettings, trace: list[str]) -> SupervisorReport:
        """处理 CODE_EXAMPLE 意图：用 LLM 生成自包含代码，不走沙箱，不生成图。

        返回 SupervisorReport(response_type="CODE_EXAMPLE")，
        executive_summary 为 Markdown 代码块字符串。
        """
        trace.append("【主控】识别为 CODE_EXAMPLE 意图，生成代码示例（不执行实验）。")
        client = UniversalLLMClient(settings)

        code_system = (
            "你是一个 Python 数据科学专家。用户要求你提供一段**完整可运行的** Python 代码示例。\n"
            "要求：\n"
            "1. 代码必须自包含（包含所有 import）。\n"
            "2. 数据使用 sklearn.datasets 内置数据集或 make_* 函数构造，不读取外部文件。\n"
            "3. 必须包含：算法调用、结果可视化（matplotlib）、指标计算（轮廓系数等）。\n"
            "4. 用中文注释解释关键步骤。\n"
            "5. 只返回 Markdown 代码块，格式为 ```python ... ```，不要额外解释。"
        )
        raw = client.chat_completion([{"role": "user", "content": prompt}], code_system)
        # 确保结果包裹在 markdown 代码块里
        if raw and "```" not in raw:
            code_md = f"```python\n{raw.strip()}\n```"
        else:
            code_md = raw or "```python\n# 无法生成代码示例，请检查 LLM 配置。\n```"

        trace.append("【主控】代码示例生成完毕。")

        # 复用上次报告的 dataset/routing/output_dir 字段，保持 UI 不崩溃
        placeholder_ds = (
            self.last_report.dataset if self.last_report else DatasetBundle("code_example", np.array([[0, 0]]), None)
        )
        report = SupervisorReport(
            dataset=placeholder_ds,
            routing=(self.last_report.routing if self.last_report else RoutingDecision(None, [], trace)),
            dataset_plot_path=(self.last_report.dataset_plot_path if self.last_report else Path("")),
            output_dir=self.last_report.output_dir if self.last_report else Path(""),
            results=[],
            ranking=self.last_report.ranking if self.last_report else [],
            executive_summary=code_md,
            decision_trace=trace,
            response_type="CODE_EXAMPLE",
        )
        # 不更新 self.last_report（不覆盖上次真实分析结果）
        return report

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _prepare_output_dir(self, name: str) -> Path:
        path = Path(f"outputs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_raw_plot(self, dataset: DatasetBundle, out_dir: Path) -> Path:
        import matplotlib.pyplot as plt

        path = out_dir / "raw_data.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c="gray", alpha=0.5, s=10)
        plt.title(f"Dataset: {dataset.display_name}")
        plt.savefig(path)
        plt.close()
        return path

    def _error_report(
        self,
        msg: str,
        trace: list[str],
        expert_logs: dict[str, list[str]] | None = None,
    ) -> SupervisorReport:
        """构造错误报告。

        P0.5-D：当 expert_logs 非空时，汇总每个专家最后 3 行日志
        作为排错依据，而不是只输出一句通用错误。
        """
        debug_lines: list[str] = [msg]
        if expert_logs:
            debug_lines.append("\n排错信息（各专家最后日志）：")
            for key, logs in expert_logs.items():
                last_3 = logs[-3:] if logs else ["（无日志）"]
                debug_lines.append(f"  [{key}] " + " | ".join(last_3))
        full_msg = "\n".join(debug_lines)

        return SupervisorReport(
            dataset=DatasetBundle("error", np.array([[0, 0]]), None),
            routing=RoutingDecision(None, [], trace),
            dataset_plot_path=Path(""),
            output_dir=Path(""),
            results=[],
            ranking=[],
            executive_summary=full_msg,
            decision_trace=trace,
            response_type="FOLLOW_UP",
        )
