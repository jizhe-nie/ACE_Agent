from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_core.schemas import AlgorithmRunResult, ChatMessage, DatasetBundle, SupervisorReport
from ACE_Agent.expert_sub_agents import build_expert_registry
from ACE_Agent.expert_sub_agents.base import save_dataset_preview
from ACE_Agent.tools.latex_generator import LatexReportGenerator
from ACE_Agent.tools.llm_client import LLMSettings, OpenAICompatibleClient


from concurrent.futures import ThreadPoolExecutor, as_completed

class ACESupervisor:
    def __init__(self) -> None:
        self.router = MasterRouter()
        self.experts = build_expert_registry()
        self.latex = LatexReportGenerator()
        self.memory: list[ChatMessage] = []
        self.last_report: SupervisorReport | None = None

    def reset(self):
        """重置会话记忆。"""
        self.memory = []
        self.last_report = None

    def run(
        self,
        dataset: DatasetBundle,
        user_prompt: str = "",
        llm_settings: LLMSettings | None = None,
        output_root: str | Path | None = None,
    ) -> SupervisorReport:
        # 1. 意图识别
        intent = self.router.analyze_intent(user_prompt, self.memory, llm_settings)
        logger.info(f"意图识别结果: {intent}")

        # 2. 如果是追问且有历史报告，直接回答
        if intent == "FOLLOW_UP" and self.last_report:
            logger.info("检测到追问，从上一次报告中提取信息...")
            report = self._handle_follow_up(user_prompt, llm_settings)
            self.memory.append(ChatMessage(role="user", content=user_prompt))
            self.memory.append(ChatMessage(role="assistant", content=report.llm_summary or ""))
            return report

        # 3. 正常执行新任务逻辑
        logger.info(f"开始执行新聚类任务: 数据集={dataset.display_name}")
        root = Path(output_root or Path(__file__).resolve().parents[1] / "outputs")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / f"{dataset.name}_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录已创建: {output_dir}")

        dataset_plot = save_dataset_preview(dataset, output_dir)
        routing = self.router.route(dataset, user_prompt, llm_settings=llm_settings)
        logger.info(f"路由决策完成，激活专家: {[e.expert_label for e in routing.selected_experts]}")

        results: list[AlgorithmRunResult] = []
        decision_trace = list(routing.trace)

        # 并行执行专家任务
        with ThreadPoolExecutor(max_workers=min(len(routing.selected_experts), 4)) as executor:
            future_to_expert = {
                executor.submit(self.experts[assignment.expert_key].run, dataset, output_dir): assignment
                for assignment in routing.selected_experts
            }

            for future in as_completed(future_to_expert):
                assignment = future_to_expert[future]
                try:
                    expert_results = future.result()
                    results.extend(expert_results)
                    if expert_results:
                        best_expert_result = max(expert_results, key=lambda item: float(item.metrics.get("score", 0.0)))
                        logger.success(f"{assignment.expert_label} 完成任务")
                        decision_trace.append(
                            f"{assignment.expert_label} 完成了 {len(expert_results)} 次运行；最佳得分 {float(best_expert_result.metrics.get('score', 0.0)):.3f}，采用算法 {best_expert_result.algorithm_name}。"
                        )
                except Exception as e:
                    logger.error(f"{assignment.expert_label} 执行异常: {str(e)}")
                    decision_trace.append(f"{assignment.expert_label} 运行失败: {str(e)}")

        if not results:
            raise RuntimeError("所有专家均未能生成有效的聚类结果。")

        ranking = sorted(results, key=lambda item: float(item.metrics.get("score", 0.0)), reverse=True)
        executive_summary = self._build_executive_summary(dataset, ranking)
        
        report = SupervisorReport(
            dataset=dataset,
            routing=routing,
            dataset_plot_path=dataset_plot,
            output_dir=output_dir,
            results=results,
            ranking=ranking,
            executive_summary=executive_summary,
            decision_trace=decision_trace,
            latex_path=output_dir / "ace_report.tex",
            response_type="CLUSTER_TASK"
        )
        
        # 生成 LaTeX 报告
        try:
            report.latex_path = self.latex.generate(report)
        except Exception as e:
            logger.warning(f"Latex 报告生成失败: {e}")

        # 生成 LLM 总结
        if llm_settings and llm_settings.is_configured:
            report.llm_summary = OpenAICompatibleClient(llm_settings).summarize_report(self._llm_payload(report))
        
        # 更新状态
        self.last_report = report
        self.memory.append(ChatMessage(role="user", content=user_prompt))
        self.memory.append(ChatMessage(role="assistant", content=report.llm_summary or report.executive_summary))
        
        return report

    def _handle_follow_up(self, user_prompt: str, llm_settings: LLMSettings | None) -> SupervisorReport:
        """
        处理追问逻辑：不重新运行算法，直接基于 last_report 的内容由 LLM 回答。
        """
        if not self.last_report:
            raise RuntimeError("没有找到上一次任务的报告，无法进行追问。")

        report = self.last_report
        if llm_settings and llm_settings.is_configured:
            client = OpenAICompatibleClient(llm_settings)
            context_data = self._llm_payload(report)
            history_text = "\n".join([f"{m.role}: {m.content}" for m in self.memory[-5:]])
            
            prompt = (
                f"你是一个资深数据分析专家。用户正在针对以下聚类分析结果进行追问。\n"
                f"聚类分析摘要: {report.executive_summary}\n"
                f"详细指标: {str(context_data.get('winning_metrics'))}\n"
                f"决策过程: {str(report.decision_trace)}\n"
                f"对话历史:\n{history_text}\n"
                f"用户追问: {user_prompt}\n\n"
                "请根据聚类结果和对话历史，专业地回答用户的追问。如果追问涉及具体算法参数或指标对比，请详细说明。"
            )
            report.llm_summary = client.summarize_report({"custom_request": prompt})
        else:
            report.llm_summary = f"基于上次任务的结果（{report.dataset.display_name}），最优算法是 {report.ranking[0].algorithm_name}。目前 LLM 未配置，无法提供更深入的追问分析。"

        report.response_type = "FOLLOW_UP"
        return report

    def _build_executive_summary(self, dataset: DatasetBundle, ranking: list[AlgorithmRunResult]) -> str:
        best = ranking[0]
        runner_up = ranking[1] if len(ranking) > 1 else None
        lines = [
            f"ACE 智能体分析了 {dataset.display_name} 并从 {best.expert_label} 中选择了 {best.algorithm_name} 作为当前的优胜方案。",
            f"该方案的综合得分为 {float(best.metrics.get('score', 0.0)):.3f}，其中 AMI 为 {self._fmt(best.metrics.get('ami'))}，轮廓系数（Silhouette）为 {self._fmt(best.metrics.get('silhouette'))}。",
        ]
        if runner_up is not None:
            lines.append(
                f"最强候选方案是来自 {runner_up.expert_label} 的 {runner_up.algorithm_name}，得分为 {float(runner_up.metrics.get('score', 0.0)):.3f}。"
            )
        return " ".join(lines)

    def _llm_payload(self, report: SupervisorReport) -> dict[str, Any]:
        top = report.ranking[0]
        return {
            "dataset": report.dataset.display_name,
            "dataset_description": report.dataset.description,
            "winning_algorithm": top.algorithm_name,
            "winning_expert": top.expert_label,
            "winning_metrics": top.metrics,
            "executive_summary": report.executive_summary,
            "routing_trace": report.routing.trace,
        }

    @staticmethod
    def _fmt(value: Any) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.3f}"
        except:
            return str(value)
