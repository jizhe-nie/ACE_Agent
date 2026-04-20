"""
expert_sub_agents/critic_expert.py
==================================
Critic Expert: The independent auditor for clustering quality.
Calculates cluster tendency (Hopkins) and internal/external validation scores
beyond just silhouette.

Phase 1 Milestone.
"""
from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient


class CriticExpert(BaseExpert):
    """专家：聚类质量的独立审计方。"""

    def __init__(self) -> None:
        super().__init__("critic", "评价专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
    ) -> str:
        system_prompt = (
            "你是一个 Python 统计学与聚类评估专家。\n"
            "你的职责：评估数据集的聚类趋势和现有聚类结果的数学合理性。\n"
            "任务要求：\n"
            "1. 生成代码计算 Hopkins Statistic（测量聚类趋势，0.5 左右为随机分布，趋近 1 为强聚类趋势）。\n"
            "2. 生成代码计算不同 K 值的评估曲线（如 Elbow Method / Gap Statistic 逻辑）。\n"
            "3. 如果用户提供了具体聚类结果（X 和 y），评估其稳定性（如多次扰动后的标签一致性）。\n"
            "4. 绘图：绘制 Hopkins 指标图或 K-评分曲线。\n"
            "5. 输入变量：直接使用 X (numpy.ndarray)。\n"
            "6. 输出：将评估指标和建议存入 artifacts['Critic_Audit'] = {labels: [], metrics: {hopkins: ..., suggestions: ...}, plot_path: ...}。\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"对该 {dataset.X.shape[0]} 样本数据集进行聚类趋势审计。"
        return client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
        ).strip()
