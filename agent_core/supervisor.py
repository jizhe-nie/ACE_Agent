from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle, SupervisorReport
from ACE_Agent.expert_sub_agents import build_expert_registry
from ACE_Agent.expert_sub_agents.base import save_dataset_preview
from ACE_Agent.tools.latex_generator import LatexReportGenerator
from ACE_Agent.tools.llm_client import LLMSettings, OpenAICompatibleClient


class ACESupervisor:
    def __init__(self) -> None:
        self.router = MasterRouter()
        self.experts = build_expert_registry()
        self.latex = LatexReportGenerator()

    def run(
        self,
        dataset: DatasetBundle,
        user_prompt: str = "",
        llm_settings: LLMSettings | None = None,
        output_root: str | Path | None = None,
    ) -> SupervisorReport:
        root = Path(output_root or Path(__file__).resolve().parents[1] / "outputs")
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root / f"{dataset.name}_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_plot = save_dataset_preview(dataset, output_dir)
        routing = self.router.route(dataset, user_prompt)

        results: list[AlgorithmRunResult] = []
        decision_trace = list(routing.trace)
        for assignment in routing.selected_experts:
            expert = self.experts[assignment.expert_key]
            expert_results = expert.run(dataset, output_dir)
            results.extend(expert_results)
            best_expert_result = max(expert_results, key=lambda item: float(item.metrics.get("score", 0.0)))
            decision_trace.append(
                f"{assignment.expert_label} 完成了 {len(expert_results)} 次运行；最佳得分 {float(best_expert_result.metrics.get('score', 0.0)):.3f}，采用算法 {best_expert_result.algorithm_name}。"
            )

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
        )
        report.latex_path = self.latex.generate(report)
        if llm_settings and llm_settings.is_configured:
            report.llm_summary = OpenAICompatibleClient(llm_settings).summarize_report(self._llm_payload(report))
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
        if dataset.shape_family in {"non_convex", "manifold"}:
            lines.append("由于数据集几何结构非纯球形，路由策略倾向于采用拓扑、嵌入（Embedding）或多视图共识方案。")
        else:
            lines.append("由于数据看起来分布紧凑且呈簇状，路由策略将质心算法作为首选。")
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
        return f"{float(value):.3f}"

