from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from ACE_Agent.agent_core.schemas import (
    SupervisorReport, DatasetBundle, RoutingDecision, 
    AlgorithmRunResult, ProfileReport, ExpertRecommendation
)
from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_brain.knowledge_engine import KnowledgeEngine
from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
from ACE_Agent.expert_sub_agents.topology_expert import TopologyExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient, LLMSettings
from ACE_Agent.tools.latex_generator import LatexReportGenerator

class ACESupervisor:
    """主控编排器 (Orchestrator)：协调多代理完成复杂任务"""
    
    def __init__(self):
        self.router = MasterRouter()
        self.knowledge_engine = KnowledgeEngine()
        # 初始化扫描 docs 目录
        try:
            self.knowledge_engine.ingest_docs()
        except Exception as e:
            print(f"知识库初始化警告: {e}")
            
        self.memory = []
        self.last_report: Optional[SupervisorReport] = None
        self.experts = {
            "centroid": CentroidExpert(),
            "topology": TopologyExpert()
        }

    def run(self, dataset: Optional[DatasetBundle], user_prompt: str, llm_settings: LLMSettings, intent_data: Optional[Dict] = None) -> SupervisorReport:
        """核心编排流程"""
        # 1. 语义路由
        if not intent_data:
            intent_data = self.router.analyze_intent(user_prompt, self.memory, llm_settings)
        
        intent = str(intent_data.get("intent", "NEW_TASK")).upper()
        reasoning = intent_data.get("reasoning", "接收到指令")
        
        # RAG 增强：检索背景知识
        knowledge_context = self.knowledge_engine.query(user_prompt)
        if knowledge_context:
            # 记录具体的知识来源信息到思考流
            trace_msg = "【RAG】成功检索到相关学术理论片段，已注入上下文。"
            user_prompt = f"{knowledge_context}\n用户指令: {user_prompt}"
        else:
            trace_msg = "【RAG】未在知识库中匹配到强相关的理论支持。"
            
        trace = [f"【主控】确认意图: {intent}", f"【逻辑】{reasoning}", trace_msg]
        
        # 2. 意图分流：FOLLOW_UP 模式 (支持纯咨询)
        if intent == "FOLLOW_UP":
            return self._handle_follow_up(user_prompt, llm_settings, trace)
        
        # 3. 新任务执行 (NEW_TASK)
        if not dataset:
            # 只有在 NEW_TASK 意图下缺失数据集才是错误
            return self._error_report("意图识别为新任务，但未识别到数据。请先配置/预览数据，或直接进行学术提问。", trace)

        return self._execute_full_analysis(dataset, user_prompt, llm_settings, trace)

    def _execute_full_analysis(self, dataset: DatasetBundle, prompt: str, settings: LLMSettings, trace: List[str]) -> SupervisorReport:
        """执行完整的自动化聚类实验流"""
        output_dir = self._prepare_output_dir(dataset.name)
        all_results = []
        
        # 模拟路由选择专家 (在 M1 重构中我们简化为激活所有相关专家)
        selected_keys = ["centroid", "topology"] 
        
        for key in selected_keys:
            expert = self.experts.get(key)
            if expert:
                # 调用具备自愈能力的执行逻辑
                expert_results = expert.execute_with_self_correction(dataset, prompt, settings)
                all_results.extend(expert_results)
                # 聚合思考日志
                trace.extend(expert.last_logs)

        if not all_results:
            return self._error_report("所有专家执行均失败，请检查模型配置或数据集。", trace)

        # 排序与摘要
        ranking = sorted(all_results, key=lambda x: x.metrics.get("score", 0.0), reverse=True)
        client = UniversalLLMClient(settings)
        summary = client.summarize_report({
            "user_intent": prompt,
            "dataset": dataset.display_name,
            "best_algo": ranking[0].algorithm_name,
            "metrics": ranking[0].metrics,
            "all_results": [{"algo": r.algorithm_name, "score": r.metrics.get("score", 0.0)} for r in all_results]
        })

        # 生成报告
        report = SupervisorReport(
            dataset=dataset,
            routing=RoutingDecision(
                profile=ProfileReport(dataset.X.shape[0], dataset.X.shape[1], 0, 0, 0, False, False, False),
                selected_experts=[], trace=trace
            ),
            dataset_plot_path=self._save_raw_plot(dataset, output_dir),
            output_dir=output_dir,
            results=all_results,
            ranking=ranking,
            executive_summary=summary or "聚类分析完成。",
            decision_trace=trace,
            response_type="CLUSTER_TASK"
        )
        
        # 自动生成 LaTeX (静默失败，不阻塞 UI)
        try:
            report.latex_path = LatexReportGenerator().generate(report)
        except: pass
        
        self.last_report = report
        self.memory.append({"role": "user", "content": prompt})
        self.memory.append({"role": "assistant", "content": report.executive_summary})
        return report

    def _handle_follow_up(self, prompt: str, settings: LLMSettings, trace: List[str]) -> SupervisorReport:
        """纯 LLM 驱动的追问或学术咨询处理"""
        client = UniversalLLMClient(settings)
        
        # 构造上下文：如果有历史报告则带上，没有则是纯咨询
        if self.last_report:
            context = {
                "last_summary": self.last_report.executive_summary,
                "ranking": [{"algo": r.algorithm_name, "score": r.metrics.get("score")} for r in self.last_report.ranking]
            }
            system_msg = f"你是一个数据科学专家。请基于以下聚类背景及检索到的知识回答用户问题：\n{context}"
        else:
            system_msg = "你是一个数据科学专家。请基于检索到的学术背景知识回答用户的理论咨询。"

        res = client.chat_completion([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ])
        
        trace.append("【主控】正在基于知识库与会话上下文进行深度解析...")
        
        import numpy as np
        # 构造一个咨询类报告
        report = SupervisorReport(
            dataset=self.last_report.dataset if self.last_report else DatasetBundle("Consultation", np.array([[0,0]]), None),
            routing=self.last_report.routing if self.last_report else RoutingDecision(None, [], trace),
            dataset_plot_path=self.last_report.dataset_plot_path if self.last_report else Path(""),
            output_dir=self.last_report.output_dir if self.last_report else Path(""),
            results=[], 
            ranking=self.last_report.ranking if self.last_report else [],
            executive_summary=res or "无法生成回答。",
            decision_trace=trace,
            response_type="FOLLOW_UP"
        )
        return report

    def _prepare_output_dir(self, name: str) -> Path:
        path = Path(f"outputs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_raw_plot(self, dataset: DatasetBundle, out_dir: Path) -> Path:
        import matplotlib.pyplot as plt
        path = out_dir / "raw_data.png"
        plt.figure(figsize=(6, 4))
        plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c='gray', alpha=0.5, s=10)
        plt.title(f"Dataset: {dataset.display_name}")
        plt.savefig(path); plt.close()
        return path

    def _error_report(self, msg: str, trace: List[str]) -> SupervisorReport:
        # 构造一个空的错误报告，确保 UI 不崩溃
        return SupervisorReport(
            dataset=DatasetBundle("error", np.array([[0,0]]), None),
            routing=RoutingDecision(None, [], trace),
            dataset_plot_path=Path(""), output_dir=Path(""),
            results=[], ranking=[], executive_summary=msg,
            decision_trace=trace, response_type="FOLLOW_UP"
        )

import numpy as np
