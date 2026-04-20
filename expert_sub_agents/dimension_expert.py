"""
expert_sub_agents/dimension_expert.py
====================================
Dimension Expert: Specialized in handling high-dimensional or sparse data
using dimensionality reduction (PCA, UMAP, t-SNE) before clustering.

Integrated with the self-healing Think-Act-Fix loop (Phase 1).
"""
from __future__ import annotations

from ACE_Agent.agent_core.schemas import DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert
from ACE_Agent.tools.llm_client import UniversalLLMClient


class DimensionExpert(BaseExpert):
    """专家：专注于降维与流形学习预处理。"""

    def __init__(self) -> None:
        super().__init__("dimension", "维度专家")

    def _generate_code(
        self,
        client: UniversalLLMClient,
        dataset: DatasetBundle,
        prompt: str,
    ) -> str:
        system_prompt = (
            "你是一个 Python 数据科学专家（降维与聚类分支）。\n"
            "你的职责：处理高维数据，在聚类前先执行降维预处理。\n"
            "任务要求：\n"
            f"1. 分析用户意图：'{prompt}'\n"
            f"2. 数据画像：{dataset.description}，样本量：{dataset.X.shape[0]}，特征数：{dataset.X.shape[1]}。\n"
            "3. 策略选择：\n"
            "   - 若特征数较多，优先考虑 PCA 提取主成分。\n"
            "   - 若指令暗示流形结构或可视化需求，考虑使用 t-SNE 或 SpectralEmbedding。\n"
            "   - 必须在降维后的特征空间执行聚类（如 KMeans 或 GMM）。\n"
            "4. 输入变量：直接使用内存中的 X (numpy.ndarray)。\n"
            "5. 输出要求：结果必须写入 artifacts[algo_name]，包含 labels, metrics, plot_path。\n"
            "   注意：metrics['score'] 必须是降维后空间的评估指标。\n"
            "只返回 Python 代码，不要有任何解释。"
        )
        user_input = f"为该 {dataset.X.shape[1]} 维数据集生成降维聚类管线。"
        return client.chat_completion(
            [{"role": "user", "content": user_input}],
            system_prompt,
        ).strip()
